"""Mantle's side of the authority ledger: each change to reach goes to Origin to be co-signed.

The store journals every grant and containment change, and each key principal's origin root, before
it writes (`db.reach_journal`). Installed here, each entry is sent to Origin
(`POST /internal/authority/entries`) on the acting person's own credential. Origin verifies the
credential, checks against its own replica that this person may make the change, co-signs and
appends it, and Origin's receipt (`seq`, `entry_hash`, `signature`, `kid`) is kept beside the row:
in the store's `authority_receipt` table, written in the same transaction as the row.

The credential is the one a key release would show (`threshold_custody.acting_credential`): a
person's own bearer, or for background work and the platform system principal, a delegation Origin
mints for the system principal. A grant key is not a credential Origin can verify.

What happens when Origin refuses, or cannot be asked, depends on the custody mode:

  enforce   `MANTLE_MASTER_KEY_CUSTODY=threshold`. Fail closed: the change is not written
            (`ReachChangeRefused`). Under threshold custody Origin releases keys by its replica,
            so a change it has not co-signed would be a grant that does not work, and a change made
            around it would be a grant Origin never checked.
  record    lattice custody with an Origin configured (`ORIGIN_URI`), the default. The ledger is
            written and not enforced: the change is written either way, and a refusal or an
            unreachable Origin is logged (the row then carries no receipt). Origin's replica
            converges at the cutover, when `mantle.system.cosign_authority` co-signs what is here.
  off       `MANTLE_AUTHORITY_LEDGER=off`, or no Origin: nothing is sent. Not accepted under
            threshold custody.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from mantle.db import reach_journal

logger = logging.getLogger(__name__)

MODE_SETTING = "MANTLE_AUTHORITY_LEDGER"
_RECEIPT_KEYS = ("seq", "entry_hash", "signature", "kid", "head_seq", "outcome", "batch")

_warned: set = set()
_warn_lock = threading.Lock()


def mode() -> str:
    """`enforce`, `record` or `off`, from the custody mode and `MANTLE_AUTHORITY_LEDGER`."""
    from mantle.search.mantle.threshold_custody import custody_kind

    setting = (os.getenv(MODE_SETTING) or "").strip().lower()
    if setting not in ("", "record", "off"):
        raise ValueError(f"Unknown {MODE_SETTING}: {setting!r} (expected record | off; threshold "
                         f"custody enforces the ledger itself)")
    if custody_kind() == "threshold":
        if setting == "off":
            raise ValueError(f"{MODE_SETTING}=off is refused under threshold custody: Origin "
                             f"releases keys by the ledger, so every change to reach goes to it")
        return "enforce"
    if setting == "off":
        return "off"
    from mantle import config
    return "record" if (getattr(config, "ORIGIN_URI", "") or "").strip() else "off"


def _takes_batch(custodian) -> bool:
    import inspect

    try:
        params = inspect.signature(custodian.submit_entry).parameters
    except (TypeError, ValueError):
        return False
    return "batch" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def receipt_of(answer: dict) -> Optional[dict]:
    """The part of Origin's answer a row keeps: None for an unchanged removal with no entry."""
    if not answer or answer.get("seq") is None:
        return None
    return {k: answer.get(k) for k in _RECEIPT_KEYS}


def _system_id() -> str:
    try:
        from mantle.services.peer_signing import get_system_principal_id
        return get_system_principal_id() or ""
    except Exception:  # noqa: BLE001
        return ""


def _warn_once(reason: str, entry: dict, exc: BaseException) -> None:
    key = (entry.get("kind"), reason)
    with _warn_lock:
        first = key not in _warned
        _warned.add(key)
    log = logger.warning if first else logger.debug
    log("authority ledger (record mode): %s entry not co-signed (%s): %s; the change is written "
        "without a receipt%s", entry.get("kind"), reason, exc,
        "" if first else " (repeat)")


class Submitter:
    """The function `reach_journal` calls, bound to a mode and an Origin client."""

    def __init__(self, custodian, ledger_mode: str) -> None:
        self.custodian = custodian
        self.mode = ledger_mode
        #: The highest ledger head Origin has named to this process; a lower one is refused.
        self.head: Optional[int] = None

    def _head(self, answer: Optional[dict]) -> None:
        head = (answer or {}).get("head_seq")
        if head is None:
            return
        if self.head is not None and int(head) < self.head:
            raise reach_journal.ReachChangeRefused(
                f"Origin's ledger head went backwards, from {self.head} to {head}: its ledger may "
                f"have been truncated or restored from an older copy")
        self.head = max(self.head or 0, int(head))

    def confirm(self, seqs: list) -> None:
        """Tell Origin the writes for these entries committed (`/internal/authority/confirm`)."""
        tell = getattr(self.custodian, "confirm_entries", None)
        if tell is not None:
            tell(list(seqs))

    def retry(self, entry: dict, seq: int) -> Optional[dict]:
        """An outstanding withdrawal of entry `seq`, submitted again by the system principal.
        Origin takes it past its compensation window, and only while that entry is still the
        latest change to its row."""
        from mantle.services.acting_principal import acting_as

        system = _system_id()
        if not system:
            raise reach_journal.ReachChangeRefused("no system principal to withdraw as")
        with acting_as(system, principal_type="user", source="system", scope="ledger.retry"):
            return self(entry, compensates=seq)

    def __call__(self, entry: dict, *, proof: Optional[dict] = None,
                 compensates: Optional[int] = None, batch: Optional[str] = None
                 ) -> Optional[dict]:
        from mantle.clients.origin_key_share_client import AuthorityRefused
        from mantle.search.mantle.threshold_custody import acting_credential

        try:
            credential = acting_credential(_system_id(), self.custodian)
            extra = {"batch": batch} if batch and _takes_batch(self.custodian) else {}
            answer = self.custodian.submit_entry(entry, credential=credential, proof=proof,
                                                 compensates=compensates, **extra)
        except AuthorityRefused as exc:
            if self.mode == "enforce":
                raise reach_journal.ReachChangeRefused(
                    f"Origin did not co-sign this {entry.get('kind')} change: {exc.detail}"
                ) from exc
            _warn_once(f"refused {exc.status}", entry, exc)
            return None
        except reach_journal.ReachChangeRefused:
            raise
        except Exception as exc:  # noqa: BLE001 — no credential, or Origin unreachable
            if self.mode == "enforce":
                raise reach_journal.ReachChangeRefused(
                    f"the {entry.get('kind')} change could not be co-signed by Origin, and under "
                    f"threshold custody it is not written without: {exc}") from exc
            _warn_once(type(exc).__name__, entry, exc)
            return None
        receipt = receipt_of(answer)
        try:
            self._head(answer)
        except reach_journal.ReachChangeRefused:
            # Co-signed, then found wrong: withdraw it rather than leave it at Origin.
            if receipt and receipt.get("seq") is not None and compensates is None:
                reach_journal.compensate([(entry, receipt)])
            raise
        return receipt


def install_for_cli() -> str:
    """For a command-line pass that writes grants or edges: install the ledger as the server
    does, so its changes are journaled. Under threshold custody a pass that cannot install it
    would be refused at the store anyway; this says so first."""
    try:
        return install()
    except Exception:
        from mantle.search.mantle.threshold_custody import custody_kind
        if custody_kind() == "threshold":
            raise
        logger.warning("authority ledger not installed for this pass; its changes to reach are "
                       "not co-signed", exc_info=True)
        return "off"


def install(custodian=None) -> str:
    """Install the submitter for this process's mode; returns the mode. `off` uninstalls."""
    m = mode()
    if m == "off":
        reach_journal.uninstall()
        return m
    if custodian is None:
        from mantle.clients.origin_key_share_client import OriginShareCustodian
        custodian = OriginShareCustodian()
    reach_journal.install(Submitter(custodian, m))
    logger.info("authority ledger: %s (changes to reach go to Origin to be co-signed)", m)
    return m


def retry_outstanding(db) -> int:
    """At startup: submit again every withdrawal Origin did not take (`reach_journal.retry_due`).
    Returns how many are still outstanding; under threshold custody, changes to reach are refused
    while any is."""
    conn = db.graph.db
    with conn.write() as cur:
        reach_journal.retry_due(cur)
        left = len(reach_journal.due(cur))
    if left:
        logger.error("authority ledger: %d withdrawal(s) Origin has not taken are outstanding; "
                     "changes to reach are refused under threshold custody until they clear", left)
    return left


# -- what Mantle did not submit: people's changes at Origin, and the operator's -------------------

#: How often the ledger is read for entries this node did not submit: within the canon's 30 s, and
#: short enough that a person who accepts an invite at Origin opens what it shares within seconds
#: (2026-11-13, 71/test: 15 s left the invitee refused for the first quarter-minute). One small
#: read of Origin's feed per interval.
FEED_INTERVAL_S = 5.0
_APPLIED = "ledger:applied"


#: Origin's genesis hash, entry envelope version, and the credential kinds its feed carries that
#: this store applies (review 17): one list both repos read, in `prism.trust.ledger`.
from prism.trust.ledger import GENESIS, LEDGER_VERSION, MANTLE_APPLIES  # noqa: E402
#: The catch-up's marker. Review 17's run read one page; review 18's stopped at the cursor and
#: could undo a change Mantle made itself after it. A store either marked is caught up again under
#: this name (review 19).
_CATCHUP = "ledger:catchup-review19"


class FeedEntryRefused(Exception):
    """A feed entry that does not verify: never applied, and the pass stops before it."""


def _canonical(obj) -> str:
    import json as _json
    return _json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entry_hash(e: dict) -> str:
    """Origin's hash of a feed entry, recomputed from its own fields (`entry_hash` at Origin)."""
    import hashlib
    env = {"v": LEDGER_VERSION, "seq": int(e["seq"]), "prev": e.get("prev_hash"),
           "body": e.get("body"), "actor": e.get("actor"),
           "credential_kind": e.get("credential_kind"),
           "credential_ref": e.get("credential_ref"), "bootstrap": bool(e.get("bootstrap")),
           "compensates": e.get("compensates"), "kid": e.get("kid"),
           "claimed_from": e.get("claimed_from")}
    return hashlib.sha256(_canonical(env).encode("utf-8")).hexdigest()


def _origin_keys() -> dict:
    """kid -> public key, from the authority manifest's Origin anchor alone (never from the feed's
    own key list, which Origin's answer could state for itself)."""
    from jose import jwk as _jwk

    from mantle.services.oidc import _read_authority_manifest
    anchors = ((_read_authority_manifest() or {}).get("trust_anchors") or {})
    keys = {}
    for k in ((anchors.get("origin") or {}).get("jwks") or {}).get("keys") or []:
        if isinstance(k, dict) and k.get("kid"):
            try:
                keys[str(k["kid"])] = _jwk.construct(k, "RS256")
            except Exception:  # noqa: BLE001 — an anchor that does not parse verifies nothing
                continue
    return keys


def verify_entry(e: dict, prev_hash: Optional[str], keys: Optional[dict] = None) -> str:
    """Check one feed entry (review 12, L13): its hash recomputed from its body, Origin's
    co-signature over that hash by the kid's key from the authority manifest, and the chain from
    `prev_hash` (the last entry this store applied; None before the first, when only the genesis
    is accepted for entry 1). Returns its hash, or raises `FeedEntryRefused`."""
    import base64
    digest = entry_hash(e)
    if digest != e.get("entry_hash"):
        raise FeedEntryRefused(f"entry {e.get('seq')}: its body does not match its hash")
    expected = prev_hash if prev_hash is not None else (GENESIS if int(e["seq"]) == 1 else None)
    if expected is not None and e.get("prev_hash") != expected:
        raise FeedEntryRefused(f"entry {e.get('seq')}: the chain breaks before it")
    key = (keys if keys is not None else _origin_keys()).get(str(e.get("kid") or ""))
    if key is None:
        raise FeedEntryRefused(f"entry {e.get('seq')}: signed by a key the authority manifest "
                               f"does not anchor for Origin")
    sig = str(e.get("signature") or "")
    try:
        raw = base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4))
    except (ValueError, TypeError) as exc:
        raise FeedEntryRefused(f"entry {e.get('seq')}: its co-signature does not parse") from exc
    if not key.verify(digest.encode("ascii"), raw):
        raise FeedEntryRefused(f"entry {e.get('seq')}: Origin's co-signature does not verify")
    return digest


def _grant_from_body(b: dict, actor: str):
    from mantle.attenuation import ACTIONS
    from mantle.entities.grant import Grant
    t = b.get("target") or {}
    flags = {f"can_{a}": a in set(b.get("actions") or ()) for a in ACTIONS}
    g = Grant(resource_id=b["resource_id"], grantee_type=b["grantee_type"],
              grantee_id=b["grantee_id"], granted_by=str(actor or ""),
              effect=b.get("effect") or "allow", state=b.get("state") or "active",
              id=b["grant_id"], expires_at=b.get("expires_at"),
              max_claims=b.get("max_claims"), **flags)
    if t.get("kind"):
        g.target_entity_type = t["kind"]
        g.target_digest = t.get("digest")
    return g


def _apply_session(db, e: dict) -> int:
    """An authority change a person made in their Origin session (review 12, decision 1): written
    into this store as Origin decided it."""
    from mantle.db import lattice_api as la
    b = e.get("body") or {}
    kind = b.get("kind")
    if kind == "grant":
        g = _grant_from_body(b, e.get("actor"))
        if la.get_grant_by_id(db, g.id) is None:
            la.create_grant(db, g)
        else:
            la.update_grant(db, g)
        return 1
    if kind == "edge.add":
        la.add_artifact_to_collection(db, b["src"], b["dst"], origin=bool(b.get("origin")),
                                      propagate=b.get("propagate"))
        return 1
    if kind == "edge.remove":
        return 1 if db.graph.delete_edge(b["src"], b["dst"], "contains") else 0
    return 0                             # roots, job grants: Origin's alone


def _retire_add_seal(db, principal_id: str) -> int:
    """An id the operator retired keeps nothing here that an id created again could inherit: its
    cached Add seal goes (review 17)."""
    gone = 0
    for rid in (f"add-seal:{principal_id}", f"drop-key:{principal_id}"):
        if db.artifacts.get_artifact(rid) is not None:
            db.artifacts.delete_artifact(rid)
            gone = 1
    return gone


def _apply_one(db, e: dict) -> int:
    """Apply one feed entry; returns 1 when it changed this store. Only the kinds
    `MANTLE_APPLIES` names: a person's own change at Origin (session or page) and the
    operator's. The rest are this store's own submissions, already in it."""
    kind = e.get("credential_kind")
    if kind not in MANTLE_APPLIES:
        return 0
    if kind != "operator":
        with reach_journal.suspended():
            return _apply_session(db, e)
    b = e.get("body") or {}
    with reach_journal.suspended():
        if b.get("kind") == "root.retire":
            return _retire_add_seal(db, b["principal_id"])
        if b.get("kind") == "grant":
            doc = db.artifacts.get_artifact(b["grant_id"])
            if doc and doc.get("state") == "active" and b.get("state") != "active":
                db.artifacts.put_artifact({**{k: v for k, v in doc.items()
                                              if not k.startswith("_")},
                                           "state": b["state"]})
                return 1
        elif b.get("kind") == "edge.remove":
            if db.graph.delete_edge(b["src"], b["dst"], "contains"):
                return 1
        elif b.get("kind") == "edge.add" and b.get("origin"):
            # The operator restores a creation edge a removal took from a member still held
            # elsewhere (2026-11-14): back as it was, mask included.
            from mantle.db import lattice_api as la
            la.add_artifact_to_collection(db, b["src"], b["dst"], origin=True,
                                          propagate=b.get("propagate"))
            return 1
    return 0


def apply_remote(db, custodian=None) -> int:
    """Apply to this store the ledger entries it did not submit itself (`MANTLE_APPLIES`): the
    changes people made in their Origin session or at Origin's page, and the operator's
    revocations, edge removals, retirements and resolved withdrawals. Journaling is suspended
    while they are written (they are already in the ledger). Returns how many were applied.

    Entries are applied in ledger order, and the cursor moves only past those applied: one that
    fails stops the pass there, and the next pass starts from it."""
    custodian = custodian or getattr(reach_journal._authority, "custodian", None)
    feed = getattr(custodian, "ledger_feed", None)
    if feed is None:
        return 0
    try:
        caught = catch_up_skipped(db, custodian)
    except Exception:  # noqa: BLE001 — the catch-up stays undone; the feed's own pass goes on
        logger.error("ALARM authority ledger catch-up failed; it stays undone and runs again on "
                     "the next pass, and the feed is applied meanwhile", exc_info=True)
        caught = 0
    conn = db.graph.db
    row = conn.read().execute("SELECT seq, entry_hash FROM authority_receipt WHERE subject = ?",
                              (_APPLIED,)).fetchone()
    after = int(row[0]) if row else 0
    # The chain continues from the last entry applied here. A cursor kept by an earlier Mantle
    # recorded no hash: its next entry starts the chain this store checks from.
    prev = (row[1] if row and row[1] else None) if after else None
    chain_known = not after or bool(row and row[1])
    answer = feed(after)
    keys = _origin_keys()
    applied = 0
    last = after
    for e in sorted(answer.get("entries") or [], key=lambda x: int(x["seq"])):
        seq = int(e["seq"])
        if seq <= last:
            continue
        try:
            digest = verify_entry(e, prev if chain_known else None, keys)
        except FeedEntryRefused as exc:
            logger.error("ALARM authority ledger: %s; nothing from it on is applied", exc)
            break
        try:
            applied += _apply_one(db, e)
        except Exception:  # noqa: BLE001 — not applied: the cursor stays before it
            logger.error("authority ledger: entry %s from Origin could not be applied; the feed "
                         "resumes from it", seq, exc_info=True)
            break
        last, prev, chain_known = seq, digest, True
    if last != after:
        with conn.write() as cur:
            cur.execute("INSERT OR REPLACE INTO authority_receipt(subject, kind, seq, entry_hash) "
                        "VALUES(?, 'applied', ?, ?)", (_APPLIED, last, prev))
    if applied or caught:
        logger.warning("authority ledger: applied %d change(s) from Origin", applied + caught)
        # A grant made at Origin (an accepted invite, a revocation) holds from the moment it is
        # applied here, never after the memo's TTL. 2026-11-13, 71/test: an invitee who accepted
        # could not open the workspace until a later sign-in.
        try:
            from mantle.search.mantle.wiring import invalidate_grant_cache
            invalidate_grant_cache()
        except Exception:  # noqa: BLE001 — the memo then lapses at its TTL, as before
            logger.warning("authority ledger: the grant memo could not be dropped", exc_info=True)
    return applied + caught


def feed_at_head(db, custodian) -> tuple:
    """Is this store's ledger feed cursor at Origin's head (review 31)? `(at_head, cursor,
    head)`. Step 21's rotation decides what a charged root keeps by this store's creation edges,
    so it runs only once every entry Origin holds has been applied here. No answer from Origin,
    or none naming its head, is not at the head."""
    try:
        row = db.graph.db.read().execute(
            "SELECT seq FROM authority_receipt WHERE subject = ?", (_APPLIED,)).fetchone()
    except Exception:  # noqa: BLE001 — no receipt table: nothing applied yet
        row = None
    cursor = int(row[0]) if row else 0
    feed = getattr(custodian, "ledger_feed", None)
    if feed is None:
        return False, cursor, None
    try:
        head = (feed(cursor) or {}).get("head")
    except Exception:  # noqa: BLE001 — unanswered: not at the head
        logger.warning("Origin's ledger feed did not answer; the cursor is not known to be at "
                       "the head", exc_info=True)
        return False, cursor, None
    if head is None:
        return False, cursor, None
    return cursor >= int(head), cursor, int(head)


def _row_key(b: dict):
    if b.get("kind") == "grant":
        return ("grant", b.get("grant_id"))
    if b.get("kind") in ("edge.add", "edge.remove"):
        return ("edge", b.get("src"), b.get("dst"))
    return None


class CatchUpIncomplete(Exception):
    """The catch-up could not read and verify the whole ledger up to the cursor: nothing is
    applied, and it is not marked done, so the next pass tries again."""


def _read_ledger(feed, upto: int = 0) -> list:
    """The whole ledger, to its head (and at least to `upto`), page after page (Origin answers a
    few hundred at a time), each entry verified in one chain from the genesis. Raises
    `CatchUpIncomplete` when a page comes back empty short of the head (or of `upto`), an entry is
    skipped, or the chain breaks across pages."""
    keys, prev, after, out, head = _origin_keys(), None, 0, [], None
    while True:
        page = feed(after) or {}
        if page.get("head") is not None:
            head = int(page["head"])
        want = max(upto, head or 0)
        rows = sorted((e for e in (page.get("entries") or []) if int(e["seq"]) > after),
                      key=lambda x: int(x["seq"]))
        if not rows:
            if after < want:
                raise CatchUpIncomplete(f"the feed answered nothing after entry {after}, short "
                                        f"of {want} (head {head})")
            return out
        for e in rows:
            seq = int(e["seq"])
            if seq != after + 1:
                raise CatchUpIncomplete(f"the feed skipped from entry {after} to {seq}")
            try:
                prev = verify_entry(e, prev, keys)
            except FeedEntryRefused as exc:
                raise CatchUpIncomplete(str(exc)) from exc
            out.append(e)
            after = seq
        if head is not None and after >= head:
            return out


def catch_up_skipped(db, custodian=None) -> int:
    """Once per store (review 17, redone in review 18): an older Mantle applied only
    `origin_page` entries and moved its cursor past every `origin_session` one, so a person's
    changes in Facet never landed here; and the first catch-up read only the feed's first page,
    so it could put an early session state back over a later change.

    It reads the whole ledger to its head, page after page, verified in one chain from the
    genesis, and only then works out the latest change to each row over all of it (review 19),
    so a change this store made itself after the cursor is the latest and is never undone. For every row a session entry
    touched it writes that latest change, whatever its kind (a person's, the operator's, or this
    store's own), so the row holds what Origin last recorded. Recorded when done, under a marker
    of its own (an earlier, partial run's marker does not count); a short page or a broken chain
    leaves it undone for the next pass. A store with no cursor yet has nothing to catch up.
    Returns how many rows it wrote."""
    conn = db.graph.db
    done = conn.read().execute("SELECT seq FROM authority_receipt WHERE subject = ?",
                               (_CATCHUP,)).fetchone()
    if done is not None:
        return 0
    row = conn.read().execute("SELECT seq FROM authority_receipt WHERE subject = ?",
                              (_APPLIED,)).fetchone()
    upto = int(row[0]) if row else 0
    feed = getattr(custodian or getattr(reach_journal._authority, "custodian", None),
                   "ledger_feed", None)
    if feed is None:
        return 0
    applied = 0
    if upto:
        try:
            entries = _read_ledger(feed, upto)
        except CatchUpIncomplete as exc:
            logger.error("ALARM authority ledger catch-up: %s; nothing is replayed, and it runs "
                         "again on the next pass", exc)
            return 0
        latest, touched = {}, set()
        for e in entries:
            k = _row_key(e.get("body") or {})
            if k is None:
                continue
            latest[k] = e
            if e.get("credential_kind") == "origin_session":
                touched.add(k)
        for k in sorted(touched, key=lambda x: tuple(map(str, x))):
            with reach_journal.suspended():
                applied += _apply_session(db, latest[k])
        if applied:
            logger.warning("authority ledger: caught up %d row(s) a person's session changed",
                           applied)
    with conn.write() as cur:
        cur.execute("INSERT OR REPLACE INTO authority_receipt(subject, kind, seq, entry_hash) "
                    "VALUES(?, 'catchup', ?, NULL)", (_CATCHUP, upto))
    return applied


# -- reconciliation: this store's reach against Origin's replica -----------------------------------

#: How often this store's digest of reach is compared with Origin's replica.
VERIFY_INTERVAL_S = 300.0


def canonical_reach(grants: list, edges: list) -> str:
    """`prism.trust.ledger.canonical_reach`, the one form both sides digest (review 19): every
    active grant with its canonical UTC expiry, an invite's target and claim limit, and every
    containment edge."""
    from prism.trust.ledger import canonical_reach as _shared
    return _shared(grants, edges)


def _gone_nodes(db, edges) -> set:
    """Every node deleted at Origin as this store knows it: one whose creation edge this store
    removed (`gone:<id>`), and every member under one, by the creation edges that remain."""
    marked = {r[0][len("gone:"):] for r in db.graph.db.read().execute(
        "SELECT subject FROM authority_receipt WHERE kind = 'gone'").fetchall()}
    if not marked:
        return set()
    parent = {dst: src for src, dst, origin, _p in edges if origin}
    gone = set()
    for node in set(parent) | marked:
        cur, seen = node, set()
        while cur and cur not in seen:
            if cur in marked:
                gone.add(node)
                break
            seen.add(cur)
            cur = parent.get(cur)
    return gone


def reach_digest(db) -> dict:
    """`{"digest", "grants", "edges"}`: the SHA-256 of `canonical_reach` over this store's active
    grants and its containment edges."""
    import hashlib

    from mantle.db import lattice_api as la

    grants = [reach_journal.grant_entry(d) for d in la._grant_docs(db, state="active")]
    grants = [g for g in grants if g["state"] == "active"]
    rows = db.graph.db.read().execute(
        "SELECT src, dst, is_origin, propagate FROM edge WHERE label = ?",
        (reach_journal.CONTAINS,)).fetchall()
    edges = [(r[0], r[1], bool(r[2]), reach_journal.propagate_of(r[3])) for r in rows]
    # A deny on a node deleted at Origin is left out, as Origin's digest leaves it out (review 15
    # and 19): it holds nothing open, since the node opens for no one.
    gone = _gone_nodes(db, edges)
    grants = [g for g in grants if not (g["effect"] == "deny" and g["resource_id"] in gone)]
    text = canonical_reach(grants, edges)
    return {"digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "grants": len(grants), "edges": len(edges)}


def verify_with_origin(db, custodian=None) -> Optional[bool]:
    """Compare this store's digest of reach with Origin's replica. A divergence is an alarm
    (logged at ERROR): reach here is no longer what Origin co-signed. None when no Origin
    answers the question."""
    custodian = custodian or getattr(reach_journal._authority, "custodian", None)
    verify = getattr(custodian, "verify_reach", None)
    if verify is None:
        return None
    mine = reach_digest(db)
    answer = verify(mine) or {}
    if answer.get("match"):
        return True
    theirs = answer.get("replica") or {}
    logger.error("ALARM authority ledger: this store's reach diverges from Origin's replica "
                 "(here %s grants, %s edges, digest %s; Origin %s grants, %s edges, digest %s)",
                 mine["grants"], mine["edges"], mine["digest"][:12], theirs.get("grants"),
                 theirs.get("edges"), str(theirs.get("digest") or "")[:12])
    return False


def start_feed(db_factory, interval_s: float = FEED_INTERVAL_S):
    """Read the ledger every `interval_s` seconds and apply what this node did not submit, and
    every `VERIFY_INTERVAL_S` compare this store's reach with Origin's replica."""
    stop = threading.Event()

    def run():
        import time as _time

        verified = _time.monotonic()
        while not stop.wait(interval_s):
            try:
                apply_remote(db_factory())
            except Exception:  # noqa: BLE001 — retried on the next tick
                logger.warning("authority ledger feed not applied", exc_info=True)
            if _time.monotonic() - verified >= VERIFY_INTERVAL_S:
                verified = _time.monotonic()
                try:
                    verify_with_origin(db_factory())
                except Exception:  # noqa: BLE001 — asked again on the next interval
                    logger.warning("authority ledger: reach not verified with Origin",
                                   exc_info=True)

    t = threading.Thread(target=run, name="authority-ledger-feed", daemon=True)
    t.start()
    return stop


# -- Origin's verdict on every action, under threshold custody (M3) --------------------------------

#: An allow from Origin is reused at most this long, per requester, resource and action
#: (architecture §51's verdict memo, as for a key release).
VERDICT_TTL_S = 30.0
_verdicts: dict = {}
_verdicts_lock = threading.Lock()


def forget_verdicts() -> None:
    with _verdicts_lock:
        _verdicts.clear()


def forget_verdicts_of(principal_id: Optional[str]) -> None:
    """Forget Origin's verdicts for one person (`None`: everyone's), when their sign-in bound
    moves (review 11, M2): a verdict their ended token earned is not reused."""
    with _verdicts_lock:
        for key in [k for k in _verdicts if principal_id is None or k[0][0] == principal_id]:
            _verdicts.pop(key, None)
    # The key-release verdicts and the key cache too (review 12, M7): a cached key or a release
    # verdict the ended token earned is never served again.
    try:
        from mantle.search.mantle import wiring
        oracle = getattr(wiring, "_oracle_singleton", None)
    except Exception:  # noqa: BLE001 — no oracle built in this process: nothing cached
        oracle = None
    if oracle is not None:
        store = getattr(oracle, "_store", None)
        forget = getattr(store, "forget_verdicts_of", None)
        if forget is not None:
            forget(principal_id)
        oracle.forget_cache()


_listening_to = None


def listen_for_sign_in_bounds() -> None:
    """Forget a person's verdicts whenever Origin's published bound for them moves. Idempotent
    per poller."""
    global _listening_to
    from prism.trust import not_before
    nb = not_before.shared()
    if _listening_to is nb:
        return
    nb.on_change(forget_verdicts_of)
    _listening_to = nb


def origin_allows(resource_id: str, action: str) -> bool:
    """Under threshold custody: does Origin's own walk let the acting caller do `action` on
    `resource_id`? True outside threshold custody (this store decides alone there). An allow is
    remembered for `VERDICT_TTL_S` per requester, resource and action; a refusal is not, and an
    Origin that cannot answer, or no credential to show it, refuses."""
    import time as _time

    if not reach_journal.required():
        return True
    from mantle.services.acting_principal import current_acting_principal

    actor = current_acting_principal()
    who = (actor.principal_id, actor.principal_type) if actor else ("", "")
    # A verdict is the one Origin gave this credential: a narrowed delegation (an Ember one) never
    # rides on a verdict its subject's own token earned, nor the other way round.
    held_cred = getattr(getattr(actor, "credential", None), "token", "") if actor else ""
    import hashlib as _hashlib
    cred_ref = _hashlib.sha256(held_cred.encode()).hexdigest()[:16] if held_cred else ""
    key = (who, cred_ref, str(resource_id), str(action))
    with _verdicts_lock:
        held = _verdicts.get(key)
    if held is not None and held > _time.monotonic():
        return True
    custodian = getattr(reach_journal._authority, "custodian", None)
    ask = getattr(custodian, "reach_verdict", None)
    if ask is None:
        logger.warning("under threshold custody %s on %s needs Origin's verdict, and no Origin "
                       "answers reach; refused", action, resource_id)
        return False
    try:
        from mantle.search.mantle.threshold_custody import acting_credential
        credential = acting_credential(_system_id(), custodian)
        allowed = bool(ask(str(resource_id), str(action), credential=credential))
    except Exception as exc:  # noqa: BLE001 — no verdict: refused
        logger.warning("Origin gave no verdict for %s on %s (%s); refused", action, resource_id,
                       exc)
        return False
    if allowed:
        with _verdicts_lock:
            _verdicts[key] = _time.monotonic() + VERDICT_TTL_S
    return allowed
