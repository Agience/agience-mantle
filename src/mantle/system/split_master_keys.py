"""Split every whole master key into threshold shares, or combine them back.

WHY. Under ``MANTLE_MASTER_KEY_CUSTODY=threshold`` each principal's master key is held as 2-of-3
Shamir shares: share 1 in this node's lattice under the KEK, share 2 at Origin, share 3 sealed to
the operator's offline age recipient in the durable object store
(`search/mantle/threshold_custody.py`). Keys minted before that are stored whole, wrapped by the
KEK (`oracle.LatticeMasterKeyStore`). This pass splits each one. The key itself does not change,
so no content, cell or index entry is re-encrypted.

For each ``master-key:<principal>`` row stored whole:

  1. unwrap it, split it, and check that every pair of shares recombines to it;
  2. (after all keys are split) seal every escrow share into ONE export file for the operator;
  3. under the custody lock, re-read the row and go on only if its stored form is the one read in
     step 1; write the sealed escrow share to the durable store; store share 2 at Origin, fetch it
     back, and check that it recombines with share 1 to the key;
  4. only then replace the row with the share form, and read the key back through the threshold
     store; a key that does not read back is restored to its whole form and reported.

The custody lock (`threshold_custody.CustodyLock`, a file beside the lattice) is the one the server
takes around every key-row write, so this pass and a live server cannot interleave on a row.
Idempotent: a row already in share form is left as it is, and a re-run after a partial one splits
only what is still whole. A key whose Origin share is already stored (a re-split after a rollback)
replaces it, since Origin accepts a share of the same key.

`--combine` is the reverse, for a rollback to a release that reads only whole keys: each split key
is re-read under the lock, combined (shares 1 and 2, or 1 and 3 in recovery mode) and written back
whole under the KEK, then read back. A ``pending`` row (a key minted under threshold custody whose
write never completed, so never served) is dropped, if it is still the row first read. Origin's
shares are left in place, and a later split replaces them; escrow objects are never replaced, so
a later split writes its own beside them.

`--export-escrow PATH` copies every split key's sealed escrow share from the durable store into
one file, the operator's offline copy. Each entry opens with the escrow identity.

    python -m mantle.system.split_master_keys --dry-run
    python -m mantle.system.split_master_keys --escrow-out /path/master-key-escrow.age
    python -m mantle.system.split_master_keys --combine
    python -m mantle.system.split_master_keys --export-escrow /path/master-key-escrow-sealed.json

Both passes read Origin's share of each key, so they run as the platform system principal and show
Origin a delegation it mints for each key's principal, which it does only while the operator's
background authority is in force (``POST /oracle/custody-background`` at Origin). Origin releases a
share only for a key its replica of the authority ledger knows, so ``mantle.system.cosign_authority``
runs first.

The escrow recipient is read from ``MANTLE_ESCROW_RECIPIENT_FILE`` (or ``--recipient-file``): a
public key. This pass never reads the identity, except as the recovery mode of the threshold store
when ``--combine`` runs with ``MANTLE_MASTER_KEY_RECOVERY=escrow``.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("split_master_keys")

_PREFIX = "master-key:"


def _principal(doc: dict) -> str:
    return str(doc.get("id", ""))[len(_PREFIX):]


def _rows(db, content_type: str) -> list:
    return [d for d in db.artifacts.list_artifacts(content_type=content_type)
            if isinstance(d, dict) and str(d.get("id", "")).startswith(_PREFIX)]


def _unchanged(db, doc: dict) -> bool:
    from mantle.search.mantle.threshold_custody import stored_form
    return stored_form(db.artifacts.get_artifact(doc["id"])) == stored_form(doc)


def classify(principal_ids, custodian=None) -> dict:
    """Origin's replica's answer to what each key principal is (`POST
    /internal/authority/principal-kinds`, review 27): `person`, `collection_key`, `platform`,
    `root` or `unknown`. An id's shape says nothing: a top-level collection's id is a random UUID
    like a person's. Asked in pages; `{}` when there is no Origin to ask, or it does not answer,
    and every id then reads as non-person (`principal_kind`), so a gate on the counts fails
    closed."""
    ids = sorted({str(i) for i in principal_ids})
    if not ids:
        return {}
    if custodian is None:
        try:
            from mantle import config
            if not (getattr(config, "ORIGIN_URI", "") or "").strip():
                return {}
            from mantle.clients.origin_key_share_client import OriginShareCustodian
            custodian = OriginShareCustodian()
        except Exception:  # noqa: BLE001 — no Origin: nothing known
            return {}
    ask = getattr(custodian, "principal_kinds", None)
    if ask is None:
        return {}
    out: dict = {}
    try:
        for i in range(0, len(ids), KINDS_PAGE):
            out.update(ask(ids[i:i + KINDS_PAGE]) or {})
    except Exception as exc:  # noqa: BLE001 — unanswered: non-person, failing closed
        logger.warning("Origin did not say who is a person (%s); every key principal counts as "
                       "non-person", exc)
        return {}
    return out


#: How many principals one `principal-kinds` question names (Origin's `MAX_KINDS`).
KINDS_PAGE = 1000


def principal_kind(principal_id: str, kinds: Optional[dict] = None) -> str:
    """`collection`, `person` or `non_person`, for a key row's principal (reviews 26 and 27).

    A collection key is `ck:<collection>`. A person is whoever Origin's replica says is one
    (`classify`). Everything else is non-person: the platform user, the system principal, a
    persona's server principal, and every root (a top-level collection or artifact) whose own
    master key keys its legacy content. No one acts as those on their own credential. A root's
    key is split on the call of a person Origin admits for it, or by this pass under a job grant
    its Admin made; the platform's own under the operator's."""
    from prism.trust.ledger import COLLECTION_KEY_PREFIX

    pid = str(principal_id)
    if pid.startswith(COLLECTION_KEY_PREFIX):
        return "collection"
    return "person" if (kinds or {}).get(pid) == "person" else "non_person"


def _key_rows(db) -> list:
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore
    return _rows(db, LatticeMasterKeyStore.CONTENT_TYPE) + _rows(db, SHARE_CONTENT_TYPE)


def exposure_counts(db, *, custodian=None, kinds: Optional[dict] = None) -> dict:
    """What a Mantle that is taken over could still open alone (reviews 25 to 27), as counts
    only, by kind, each principal classified by Origin's replica (`classify`):

      whole_person_master_keys      split at the person's next call, or by this pass
      whole_non_person_master_keys  split by this pass under a job grant, or (a root's) on the
                                    call of a person Origin admits for it
      whole_collection_keys         split at a reader's next call, or by this pass
      non_person_master_keys_holding_content
                                    non-person master keys (whole or split) that still key
                                    content; one that keys none is removed
                                    (`retire_empty_master_keys`)
      content_under_master_keys     rows still keyed under a master key rather than their
                                    collection's; re-keyed on a reader's own next call
                                    (`services/owner_rekey.py`) or by `rekey_collections`
      people_with_content_under_master_keys   how many people (Origin's persons) own those rows
      unsettled_key_replacements    replaced key rows whose old bytes a checkpoint has not yet
                                    removed from the store file (`db/scrub.settle_replaced`)

    `whole_master_keys` is the sum of the three whole counts."""
    from mantle.db.scrub import due
    from mantle.search.mantle.oracle import LatticeMasterKeyStore
    from mantle.services.owner_rekey import rows_under_master_key

    keyrows = _key_rows(db)
    under_rows, owners = 0, set()
    for _doc, who in rows_under_master_key(db):
        under_rows += 1
        owners.add(who)
    if kinds is None:
        kinds = classify({_principal(d) for d in keyrows} | {o for o in owners if o},
                         custodian)
    whole = {"person": 0, "non_person": 0, "collection": 0}
    holding = set()
    for d in keyrows:
        pid = _principal(d)
        kind = principal_kind(pid, kinds)
        if d.get("content_type") == LatticeMasterKeyStore.CONTENT_TYPE and d.get("token"):
            whole[kind] += 1
        if kind == "non_person" and pid in owners:
            holding.add(pid)
    off_root = 0
    for doc, who in rows_under_master_key(db):
        try:
            from mantle.db.lattice_api import get_origin_root
            root = get_origin_root(db, str(doc.get("root_id") or doc.get("id")))
        except Exception:  # noqa: BLE001 — a store with no graph behind it names no root
            continue
        if who and root and str(who) != str(root):
            off_root += 1
    return {"whole_person_master_keys": whole["person"],
            # Legacy rows under a master key that is not their root's (review 29, I3): a person
            # wrote them under their own key into someone else's root. Re-keyed on that person's
            # call, or by step 21.
            "rows_keyed_off_their_root": off_root,
            "whole_non_person_master_keys": whole["non_person"],
            "whole_collection_keys": whole["collection"],
            "whole_master_keys": sum(whole.values()),
            "non_person_master_keys_holding_content": len(holding),
            "content_under_master_keys": under_rows,
            "people_with_content_under_master_keys": sum(
                1 for o in owners if o and principal_kind(o, kinds) == "person"),
            "unsettled_key_replacements": due(db)}


#: Origin's answers for which an emptied whole key is removed (review 29, L4): a root's, or one
#: of the platform's own. `member` and `unknown` are removed only with the explicit remedy
#: (`member_keys`, step 21's `--retire-member-keys`, L5).
RETIRED_KINDS = frozenset({"root", "platform"})
REMEDY_KINDS = frozenset({"member", "unknown"})


def retire_empty_master_keys(db, custodian=None, *, kinds: Optional[dict] = None,
                             member_keys: bool = False) -> dict:
    """Remove each whole non-person master key that keys no content any more (reviews 27 and
    28): its legacy content is all on collection keys, so nothing needs it, and a key row kept
    whole is one a Mantle taken over opens alone. A person's key is never removed. Each removal
    is a delete under the custody lock's store (`secure_delete` zeroes the row), and the store
    file is settled after. Run after `rekey_collections` has moved the content and the index
    (its `main` does). Returns `{"removed": [...], "holding_content": [...]}`."""
    from mantle.db.scrub import settle_replaced
    from mantle.services.owner_rekey import rows_under_master_key

    from mantle.search.mantle.oracle import LatticeMasterKeyStore

    # Whole keys only (review 28, I1): a split key's shares are no key a Mantle taken over can
    # open, and once custody is threshold a split key is what every principal's content opens
    # with, so a re-run after step 24 removes nothing that is in use.
    keyrows = [d for d in _rows(db, LatticeMasterKeyStore.CONTENT_TYPE)
               if d.get("token") and not _principal(d).startswith("ck:")]
    if kinds is None:
        kinds = classify({_principal(d) for d in keyrows}, custodian)
    allowed = RETIRED_KINDS | (REMEDY_KINDS if member_keys else frozenset())
    removed, holding = [], []
    for d in keyrows:
        pid = _principal(d)
        if (kinds or {}).get(pid) not in allowed:
            continue                       # a person's, or one Origin did not answer for: kept
        if next(iter(rows_under_master_key(db, pid)), None) is not None:
            holding.append(pid)
            continue
        db.artifacts.delete_artifact(d["id"])
        removed.append(pid)
    if removed:
        settle_replaced(db)
    logger.info("non-person master keys: %d removed, %d still keying content", len(removed),
                len(holding))
    return {"removed": sorted(removed), "holding_content": sorted(holding)}


def gate_exit(left: dict, counts: dict) -> int:
    """The split pass's exit code: the cutover's non-person gate (review 28, I3). Non-zero while
    any non-person master key is whole, by Origin's count; whatever else is left (a person's key,
    a collection key, a key whose split failed or changed underfoot) is reported, not gated."""
    # A split whose activation Origin never heard is not done (review 32, F2): its share stays
    # pending at Origin, and step 23's check lists it.
    return 1 if left.get("whole_non_person_master_keys") or counts.get("activation_failed") \
        else 0


def split_master_keys(db, kek, custodian, recipient: str, escrow_store, lock, *,
                      dry_run: bool = False, write_escrow=None) -> dict:
    """Split every whole key. `write_escrow(entries)` receives the escrow shares, in the clear,
    before any row is replaced; it must seal and persist them. Returns the counts."""
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore
    from mantle.search.mantle.threshold_custody import (
        MANTLE_X, ThresholdMasterKeyStore, _unframe_share, escrow_name, share_credential,
        split_key,
    )
    from prism.trust import shamir

    whole_rows = [d for d in _rows(db, LatticeMasterKeyStore.CONTENT_TYPE) if d.get("token")]
    counts = {"whole": len(whole_rows), "already_split": len(_rows(db, SHARE_CONTENT_TYPE)),
              "split": 0, "changed": 0, "failed": 0, "refused_unrotated": 0,
              "activation_failed": 0}
    if dry_run or not whole_rows:
        return counts

    whole = LatticeMasterKeyStore(kek, lambda: db)
    threshold = ThresholdMasterKeyStore(kek, lambda: db, custodian, recipient, escrow_store, lock)
    charged = _charged(custodian, [_principal(d) for d in whole_rows])
    if charged is not None:
        # Roots Mantle's own off-root test finds a removed part under (review 32, F4) are held
        # to the mark too, though Origin charges none.
        own = _off_root_roots(db, custodian, [_principal(d) for d in whole_rows])
        charged = None if own is None else charged | own

    planned = []
    for doc in whole_rows:
        pid = _principal(doc)
        if not pid.startswith("ck:") and (charged is None or pid in charged):
            # A charged root's key is split only once step 21 rotated it (review 31): its row
            # carries the rotation mark for this very key. Origin unanswered: no master key.
            from mantle.system.rekey_collections import rotation_mark
            try:
                marked = charged is not None and rotation_mark(doc, pid, whole.get(pid))
            except Exception:  # noqa: BLE001 — an unreadable key is no marked one
                marked = False
            if not marked:
                counts["refused_unrotated"] += 1
                logger.error("  %s not split: %s", pid,
                             "Origin did not say which roots are charged" if charged is None
                             else "a charged root whose key step 21 has not rotated")
                continue
        try:
            key = whole.get(pid)
            row, origin_share, escrow, sealed = split_key(pid, key, kek, recipient)
            x, y = _unframe_share(pid, kek.unwrap(row["share"]))
            if x != MANTLE_X or shamir.combine(
                    [(x, y), (escrow["x"], base64.b64decode(escrow["y"]))]) != key:
                raise RuntimeError("the wrapped share does not recombine with the escrow share")
        except Exception as exc:  # noqa: BLE001 — per key; the next key
            counts["failed"] += 1
            logger.error("  cannot split %s: %s", pid, exc)
            continue
        planned.append((doc, pid, key, row, origin_share, escrow, sealed))

    if planned and write_escrow is not None:
        write_escrow([p[5] for p in planned])

    for doc, pid, key, row, origin_share, _escrow, sealed in planned:
        with lock():
            if not _unchanged(db, doc):
                counts["changed"] += 1
                logger.warning("  %s changed since it was read; left for a re-run", pid)
                continue
            try:
                escrow_store.put_if_absent(escrow_name(row), sealed)
                custodian.put_share(pid, origin_share,
                                    credential=share_credential(pid, custodian))
                theirs = custodian.get_share(pid, credential=share_credential(pid, custodian))
                mine = _unframe_share(pid, kek.unwrap(row["share"]))[1]
                if theirs is None or not hmac.compare_digest(
                        shamir.combine([(MANTLE_X, mine),
                                        (int(theirs["x"]), base64.b64decode(theirs["y"]))]), key):
                    raise RuntimeError("Origin's stored share does not recombine to the key")
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                logger.error("  cannot store the other shares of %s: %s (left whole)", pid, exc)
                continue
            db.artifacts.put_artifact(row)
            try:
                ok = threshold.get(pid) == key
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                db.artifacts.put_artifact(doc)
                counts["failed"] += 1
                logger.error("  %s did not read back through the threshold store; restored whole",
                             pid)
                continue
            # The split key is active: Origin's share of it stops being replaceable. The pass
            # replaces no share of another key (a re-run restates the same key's), so it never
            # marks one for re-key.
            activate = getattr(custodian, "activate_share", None)
            if activate is not None:
                try:
                    activate(pid, credential=share_credential(pid, custodian))
                except Exception:  # noqa: BLE001 — counted: the run exits non-zero (review 32)
                    counts["failed"] += 1
                    counts["activation_failed"] += 1
                    logger.error("  %s is split, but Origin was not told it is active: its share "
                                 "stays pending; see step 23's check", pid, exc_info=True)
                    continue
        counts["split"] += 1
    if counts["split"]:
        _scrub(db)
    return counts


def _charged(custodian, pids) -> Optional[set]:
    """The charged roots among `pids`, by Origin (review 31), or None when Origin does not
    answer. A custodian with no such question (a test's stand-in for the share routes alone)
    names none."""
    ask = getattr(custodian, "charged_roots", None)
    if ask is None:
        return set()
    ids = sorted({str(p) for p in pids if not str(p).startswith("ck:")})
    out: set = set()
    try:
        for i in range(0, len(ids), KINDS_PAGE):
            out.update(str(r) for r in (ask(ids[i:i + KINDS_PAGE]) or []))
    except Exception as exc:  # noqa: BLE001 — unanswered: no master key is split
        logger.error("Origin did not say which roots are charged (%s); no master key is split",
                     exc)
        return None
    return out


def _off_root_roots(db, custodian, pids) -> Optional[set]:
    """The roots (by Origin's `principal-kinds`) among `pids` whose key has a removed part's rows
    under it, by this store's own creation edges (`rekey_collections.off_root_rows`). None when
    Origin was asked and did not answer, or the test could not run: no master key is split. A
    custodian with no such question (a test's stand-in) names none."""
    if getattr(custodian, "principal_kinds", None) is None:
        return set()
    ids = sorted({str(p) for p in pids if not str(p).startswith("ck:")})
    if not ids:
        return set()
    kinds = classify(ids, custodian)
    if not kinds:
        logger.error("Origin did not say which key principals are roots; no master key is split")
        return None
    from mantle.system.rekey_collections import off_root_rows
    out = set()
    try:
        for pid in ids:
            if kinds.get(pid) == "root" and off_root_rows(db, pid):
                out.add(pid)
    except Exception as exc:  # noqa: BLE001 — the test did not run: nothing split, failing closed
        logger.error("the off-root test did not run (%s); no master key is split", exc)
        return None
    return out


def _scrub(db) -> None:
    """The whole keys the split replaced leave no bytes in the store file (`db.scrub`)."""
    from mantle.db.scrub import scrub_freed
    try:
        scrub_freed(db)
    except AttributeError:                 # a store with no SQLite file behind it: nothing to scrub
        return


def combine_master_keys(db, kek, threshold_store, lock, *, dry_run: bool = False) -> dict:
    """Write every split key back whole under the KEK. Returns the counts."""
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore

    rows = _rows(db, SHARE_CONTENT_TYPE)
    pending = [d for d in rows if d.get("state") == "pending"]
    counts = {"split": len(rows) - len(pending), "pending": len(pending), "combined": 0,
              "pending_dropped": 0, "changed": 0, "failed": 0}
    if dry_run:
        return counts
    whole = LatticeMasterKeyStore(kek, lambda: db)
    for doc in rows:
        pid = _principal(doc)
        with lock():
            if not _unchanged(db, doc):
                counts["changed"] += 1
                logger.warning("  %s changed since it was read; left for a re-run", pid)
                continue
            if doc.get("state") == "pending":
                db.artifacts.delete_artifact(doc["id"])
                counts["pending_dropped"] += 1
                continue
            try:
                key = threshold_store.get(pid)
                if key is None:
                    raise RuntimeError("no key combined")
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                logger.error("  cannot combine %s: %s (left split)", pid, exc)
                continue
            whole.put(pid, key)
            try:
                ok = whole.get(pid) == key
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                db.artifacts.put_artifact(doc)
                counts["failed"] += 1
                logger.error("  %s did not read back whole; restored its shares", pid)
                continue
        counts["combined"] += 1
    return counts


def sealed_escrow_rows(db, escrow_store) -> list:
    """Every active split key's sealed escrow share, as `{principal_id, kcv, escrow}` (escrow is
    the base64 of an age file, or None when the durable store holds none for it)."""
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE
    from mantle.search.mantle.threshold_custody import escrow_name

    out = []
    for d in _rows(db, SHARE_CONTENT_TYPE):
        if d.get("state") == "pending":
            continue
        sealed = escrow_store.get(escrow_name(d))
        out.append({"principal_id": _principal(d), "kcv": d.get("kcv"),
                    "escrow": base64.b64encode(sealed).decode() if sealed else None})
    return out


def escrow_writer(path: str, recipient: str):
    """A `write_escrow` that seals all entries into one age file at `path`, refusing to replace a
    file that exists (an earlier export is the only copy of its shares)."""
    from mantle.search.mantle.threshold_custody import seal

    def write(entries: list) -> None:
        doc = {"format": "agience-mantle-escrow-v1",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "shares": entries}
        sealed = seal(json.dumps(doc, sort_keys=True).encode(), recipient)
        with open(path, "xb") as fh:
            fh.write(sealed)
        logger.info("escrow shares for %d key(s) sealed to %s", len(entries), path)

    return write


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--combine", action="store_true",
                    help="the reverse: write split keys back whole under the KEK, before a rollback")
    ap.add_argument("--escrow-out", help="the sealed export file for this run's escrow shares")
    ap.add_argument("--export-escrow", help="copy every sealed escrow share into this file")
    ap.add_argument("--recipient-file", default=os.getenv("MANTLE_ESCROW_RECIPIENT_FILE", ""),
                    help="the escrow age recipient (public key); default MANTLE_ESCROW_RECIPIENT_FILE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    # Arguments are settled before any key is loaded, so a mistyped run touches nothing.
    from mantle.search.mantle import threshold_custody as tc
    recipient = None
    if not (args.combine or args.export_escrow):
        if not args.recipient_file:
            ap.error("the escrow recipient is required: MANTLE_ESCROW_RECIPIENT_FILE or "
                     "--recipient-file")
        recipient = tc.recipient_from_file(args.recipient_file)
        if not args.dry_run and not args.escrow_out:
            ap.error("--escrow-out is required: it is the one export of this run's escrow shares")
        if args.escrow_out and os.path.exists(args.escrow_out):
            ap.error(f"{args.escrow_out} exists; an earlier export is the only copy of its shares")
    if args.export_escrow and os.path.exists(args.export_escrow):
        ap.error(f"{args.export_escrow} exists")

    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()
    from mantle.search.mantle.key_provider import build_key_provider
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    kek = build_key_provider()
    lock = tc.CustodyLock(tc.default_lock_path())

    def escrow_store():
        try:
            return tc.durable_escrow_store()
        except ValueError as exc:
            ap.error(str(exc))

    from mantle.services.system_identity import system_acting_context

    with system_acting_context(scope="platform.custody"):
        return _run(args, ap, db, kek, lock, recipient, escrow_store)


def _run(args, ap, db, kek, lock, recipient, escrow_store) -> int:
    from mantle.search.mantle import threshold_custody as tc

    if args.export_escrow:
        entries = sealed_escrow_rows(db, escrow_store())
        with open(args.export_escrow, "x", encoding="utf-8") as fh:
            json.dump({"format": "agience-mantle-escrow-sealed-v1", "entries": entries}, fh)
        missing = sum(1 for e in entries if not e["escrow"])
        logger.info("sealed escrow shares of %d key(s) written to %s (%d with none stored)",
                    len(entries) - missing, args.export_escrow, missing)
        return 1 if missing else 0

    if args.combine:
        if not os.getenv("MANTLE_ESCROW_RECIPIENT_FILE") and args.recipient_file:
            os.environ["MANTLE_ESCROW_RECIPIENT_FILE"] = args.recipient_file
        counts = combine_master_keys(db, kek, tc.build_threshold_store(kek, lambda: db, offline=True), lock,
                                     dry_run=args.dry_run)
        logger.info("%s: %s", "would combine" if args.dry_run else "combined", counts)
        return 1 if counts["failed"] or counts["changed"] else 0

    from mantle.clients.origin_key_share_client import OriginShareCustodian
    counts = split_master_keys(
        db, kek, None if args.dry_run else OriginShareCustodian(), recipient,
        None if args.dry_run else escrow_store(), lock, dry_run=args.dry_run,
        write_escrow=escrow_writer(args.escrow_out, recipient) if args.escrow_out else None)
    logger.info("%s: %s", "would split" if args.dry_run else "split", counts)
    # The cutover's gate (step 23): no non-person principal's key is left whole. Those are split
    # only here, under a `platform.custody` job grant covering each one; no person's call splits
    # them.
    left = exposure_counts(db, custodian=None if args.dry_run else OriginShareCustodian())
    # Read `whole_non_person_master_keys` (step 23): the exit code is that gate alone. The rest is
    # reported: people's and collections' whole keys split on their next reader's call, and a key
    # whose split failed or changed underfoot is split by a re-run.
    logger.info("left: %s", left)
    if counts["failed"] or counts["changed"]:
        logger.warning("%d split(s) failed and %d changed underfoot; a re-run takes them",
                       counts["failed"], counts["changed"])
    if counts.get("refused_unrotated"):
        logger.error("%d key(s) not split: a charged root whose key step 21 has not rotated (run "
                     "`rekey_collections --mantle-stopped` again), or Origin did not say which "
                     "roots are charged", counts["refused_unrotated"])
    code = 0 if args.dry_run else gate_exit(left, counts)
    if code:
        logger.error("whole_non_person_master_keys = %d (Origin's count; %d still key content): "
                     "a full reader-Admin of each root grants `platform.custody` over its key at "
                     "Origin's page, the operator grants it over the platform's own (`POST "
                     "/oracle/job-grants` with `principal`); run again",
                     left["whole_non_person_master_keys"],
                     left["non_person_master_keys_holding_content"])
    return code

if __name__ == "__main__":
    sys.exit(main())
