"""Re-key content under a master key on a reader's own call (review 26, open item a; review 27).

Content written before collections had keys is under its principal's master key, which Origin
releases to that principal alone. Under threshold custody, the first time Mantle holds a person's
own credential for their own master key, or for a root's whose content they reach (a top-level
collection's, review 27; `ThresholdMasterKeyStore.get_scoped`, the call that also splits a whole
key), it schedules this: the rekey pass's own `rekey_one`, on that person's
credential (`as_caller`), over the rows keyed under their master key.

  * In the background of the request (`_start`, a daemon thread): the call that triggered it
    returns at once, and the work runs on the credential the call carried, for its life.
  * Bounded: at most `BATCH` rows a call. What is left is done on a later call.
  * Idempotent and resumable: a row already under its collection key is skipped by `rekey_one`,
    a row someone rewrote meanwhile is left for the next call, and once nothing is left a per-box
    record (`custody:owner-rekeyed:<principal>`) says so and nothing runs again. One run per
    principal at a time in this process.
  * A collection key it mints is minted by the threshold store's `put`, so it is split from the
    start, never stored whole.
  * The rewrite replaces each row in place; `secure_delete` zeroes the old bytes and a settle
    (`db/scrub.settle_replaced`) moves them out of the store file, as for a split key row.

What is left under master keys is `split_master_keys.exposure_counts` (`content_under_master_keys`,
`people_with_content_under_master_keys`).
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: The most rows re-keyed on one call; the rest wait for the next.
BATCH = 200

DONE_CT = "application/vnd.agience.owner-rekeyed+json"
_DONE_PREFIX = "custody:owner-rekeyed:"

_running: set = set()
_running_lock = threading.Lock()


def _start(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="owner-rekey", daemon=True).start()


def _cas():
    from mantle.system.rekey_collections import Cas
    return Cas.default()


def _done_id(principal_id: str) -> str:
    return _DONE_PREFIX + str(principal_id)


def is_done(db, principal_id: str) -> bool:
    """Does this box's own done record say `principal_id`'s content is re-keyed? Only a row of
    the record's own kind, pinned at this box's operational origin (review 27, L2): a row of any
    other kind or origin under the id (a peer's, a planted one) says nothing."""
    try:
        doc = db.artifacts.get_artifact(_done_id(principal_id))
    except Exception:  # noqa: BLE001 — unreadable: not known to be done
        return False
    if not doc or doc.get("content_type") != DONE_CT:
        return False
    if hasattr(db.artifacts, "page_by_origin"):
        return str(doc.get("_origin") or "") == "_local:" + _done_id(principal_id)
    return True


def row_principal(doc: dict) -> Optional[str]:
    """The principal whose master key a row's content is under, as the rekey pass reads it."""
    from mantle.db import doc_boundary as b
    p = doc.get(b.CONTENT_KEY_PRINCIPAL)
    if not p:
        try:
            p = b.content_key_principal(doc)
        except Exception:  # noqa: BLE001 — a row with no key principal names no one
            p = None
    p = p or doc.get("created_by")
    return str(p) if p else None


def _rows_of(db, principal_id: str):
    """The rows whose content key may root at `principal_id`, found by the store's indexes, not
    by a scan of every row: by `content_key_principal` (a row keeps the principal stamped when it
    was written, whatever root it sits in now; review 28, L2), by `origin_root`, and, for a row
    with neither, by its creator. `rows_under_master_key` then keeps only those whose key
    principal it is (`row_principal`). A store with no SQL behind it is listed."""
    from mantle.system import rekey_collections as rk

    lattice = getattr(db.artifacts, "db", None)
    if lattice is None or not hasattr(lattice, "read"):
        yield from rk._rows(db)
        return
    pid = str(principal_id)
    ids = [r[0] for r in lattice.read().execute(
        "SELECT id FROM vertex WHERE json_extract(doc, '$.content_key_principal') = ? "
        "UNION SELECT id FROM vertex WHERE origin_root = ? "
        "UNION SELECT id FROM vertex WHERE origin_root IS NULL AND created_by = ?",
        (pid, pid, pid)).fetchall()]
    for rid in ids:
        doc = db.artifacts.get_artifact(rid)
        if not doc or doc.get("content_type") in rk._SKIP_TYPES \
                or str(doc.get("id", "")).startswith(("master-key:", "custody:")):
            continue
        yield doc


#: How long a refusal is remembered (seconds): the life of a person's token for Mantle, since a
#: refusal is Origin's answer for that caller's reach. Within it the rows are not walked again.
REFUSAL_MEMORY_S = 600.0

_refused: dict = {}
_refused_lock = threading.Lock()


def forget_refusals() -> None:
    with _refused_lock:
        _refused.clear()


def _caller() -> str:
    from mantle.services.acting_principal import current_acting_principal
    acting = current_acting_principal()
    return str(acting.principal_id) if acting is not None else ""


def _remembered(principal_id: str, caller: str) -> set:
    import time as _time
    now = _time.monotonic()
    with _refused_lock:
        held = _refused.get((principal_id, caller), {})
        live = {rid for rid, until in held.items() if until > now}
        _refused[(principal_id, caller)] = {rid: held[rid] for rid in live}
        return live


def _remember(principal_id: str, caller: str, row_id: str) -> None:
    import time as _time
    with _refused_lock:
        _refused.setdefault((principal_id, caller), {})[row_id] = (
            _time.monotonic() + REFUSAL_MEMORY_S)


def rows_under_master_key(db, principal_id: Optional[str] = None, *, cas=None):
    """Rows still under a master key (`principal_id`'s, or anyone's), with their principal."""
    from mantle.system import rekey_collections as rk

    marked = cas or rk.Cas(raw=lambda doc: None)
    rows = rk._rows(db) if principal_id is None else _rows_of(db, principal_id)
    for doc in rows:
        try:
            todo = rk._plan(doc, marked, drops=False)
        except Exception:  # noqa: BLE001 — an unreadable row counts as still to do
            todo = {"unreadable": True}
        if not todo:
            continue
        who = row_principal(doc)
        if principal_id is None or who == str(principal_id):
            yield doc, who


def rekey_owner(db, principal_id: str, *, cas=None, limit: Optional[int] = None) -> dict:
    """Re-key up to `limit` (`BATCH`) of `principal_id`'s rows on the acting credential, which is
    theirs. Marks the principal done when nothing re-keyable is left."""
    from mantle.db.scrub import settle_replaced
    from mantle.system import rekey_collections as rk

    from mantle.search.mantle.custody import MasterKeyUnavailable
    from mantle.services.acting_principal import KeyCustodyDenied

    cas = cas or _cas()
    limit = BATCH if limit is None else limit
    counts = {"rekeyed": 0, "already": 0, "unscoped": 0, "changed": 0, "failed": 0, "left": 0}
    more = False
    refs = None
    caller = _caller()
    skip = _remembered(str(principal_id), caller)
    for doc, _who in rows_under_master_key(db, principal_id, cas=cas):
        if str(doc.get("id")) in skip:
            continue                          # refused this caller lately: not walked again
        # A refusal counts toward the batch as a re-key does (review 28, L5): each call does a
        # bounded amount of work, whatever the answer.
        if counts["rekeyed"] + counts["failed"] + counts["changed"] + counts["left"] >= limit:
            more = True
            break
        if refs is None:
            refs = rk.key_refs(db)            # once a run, kept current by `rekey_one`
        try:
            outcome = rk.rekey_one(db, doc, cas, as_caller=True, refs=refs)
        except (KeyCustodyDenied, PermissionError, MasterKeyUnavailable) as exc:
            # This caller may not open it (a refusal, not an outage): left for someone who may,
            # and remembered, so this caller's next calls do not walk it again.
            counts["left"] += 1
            _remember(str(principal_id), caller, str(doc.get("id")))
            logger.info("owner re-key: %s left for another reader: %s", doc.get("id"), exc)
            continue
        except Exception as exc:  # noqa: BLE001 — one row's failure does not stop the rest
            counts["failed"] += 1
            logger.warning("owner re-key: %s left as it was: %s", doc.get("id"), exc)
            continue
        counts[outcome] = counts.get(outcome, 0) + 1
    if counts["rekeyed"]:
        settle_replaced(db)
    # Done only when no row under this principal's master key is left anywhere (review 28, L2):
    # not one refused to this caller, not one unscoped, not one in another root.
    if next(iter(rows_under_master_key(db, principal_id, cas=cas)), None) is None:
        done = {"id": _done_id(principal_id), "content_type": DONE_CT,
                "principal_id": str(principal_id)}
        if hasattr(db.artifacts, "page_by_origin"):
            # A per-box row, pinned outside every observer's sequence (`mesh.sync._put_op`).
            done.update({"_origin": "_local:" + done["id"], "_seq": 1})
        db.artifacts.put_artifact(done, stamp_rev=False)
    logger.info("owner re-key for %s: %s%s", principal_id, counts, " (more left)" if more else "")
    return counts


def schedule(db_factory, principal_id: str, credential: str) -> bool:
    """Start the owner's re-key in the background on `credential` (theirs), unless it is done or
    already running. Returns whether it was started. Never raises into the caller."""
    try:
        db = db_factory()
        if is_done(db, principal_id):
            return False
        with _running_lock:
            if principal_id in _running:
                return False
            _running.add(principal_id)
    except Exception:  # noqa: BLE001
        logger.warning("owner re-key not scheduled for %s", principal_id, exc_info=True)
        return False

    # The work acts as the credential's own subject (review 29, L3): the person whose call this
    # is, who may be a reader of a root rather than the key's principal. Their refusals are
    # remembered under their name.
    subject = _caller() or str(principal_id)

    def run() -> None:
        from mantle.services.acting_principal import ForwardedCredential, acting_as
        try:
            with acting_as(subject, principal_type="user", source="request",
                           credential=ForwardedCredential(credential)):
                rekey_owner(db_factory(), principal_id)
        except Exception:  # noqa: BLE001 — the next call tries again
            logger.warning("owner re-key for %s stopped; the next call resumes it",
                           principal_id, exc_info=True)
        finally:
            with _running_lock:
                _running.discard(principal_id)

    try:
        _start(run)
    except Exception:  # noqa: BLE001
        with _running_lock:
            _running.discard(principal_id)
        logger.warning("owner re-key could not start for %s", principal_id, exc_info=True)
        return False
    return True
