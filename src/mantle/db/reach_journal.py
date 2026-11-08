"""Changes to reach, journaled for an authority to co-sign. Stdlib only: this is on the embeddable
surface, and with no authority installed it does nothing (except under threshold custody, below).

The store records every change that affects who reaches what, at the layer every writer goes
through, inside the write transaction that makes it:

    grant        a grant document written (created, updated, revoked) or deleted
                 (`vertex._write_row`, `vertex.delete_artifact`)
    edge.add     a `contains` edge added, or its origin flag or `propagate` mask changed
                 (`edge._add_one`: every edge writer, including the creation edge the vertex
                 store writes for an artifact placed in a collection, and mesh consume)
    edge.remove  a `contains` edge retired (`edge._retire_edge_row`: every edge delete, and the
                 edges a vertex delete takes with it)

and the oracle submits `root` (a key principal and its origin root) before it stores a new key.

Recorded changes are submitted when the outermost write transaction is about to commit
(`seq.LatticeConn.write` calls `flush`), in the order they were made. A refusal raises
`ReachChangeRefused` there, and the transaction rolls back: nothing the authority refused is
written. If an entry was co-signed and the transaction then fails (a later entry refused, or the
commit itself failing), each co-signed widening entry is withdrawn by a compensating narrowing
entry (`compensate`), so the authority never holds reach the store does not.

Each receipt (the authority's co-signature) is written in the same transaction to the
`authority_receipt` table, keyed by the row it belongs to (`grant:<id>`, `edge:<src> <dst>`).
The highest ledger head a receipt has named is kept there too (`ledger:head`): a receipt naming a
lower head means the authority's ledger went backwards (truncated, or restored from an older
copy), and the change is refused.

A withdrawal the authority did not take leaves it holding reach the store does not. That is not
only logged: it is recorded (`authority_compensation_due`), retried before the next change and at
startup (the inverse is a narrowing, which the system principal may make), and under threshold
custody no further change to reach is written until it clears.

Under threshold custody (`MANTLE_MASTER_KEY_CUSTODY=threshold`) a change to reach with no authority
installed is refused here, at the store, whatever wrote it. Elsewhere, with none installed, nothing
is sent.

Entries hold structure only: ids, the CRUDEASIO actions a grant's flags carry, effect, state,
expiry, an invite's target digest and claim limit, and the edge's mask. Never a name, note, label,
address or content.
"""
from __future__ import annotations

import contextvars
import re
import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, List, Optional

from mantle.attenuation import ACTIONS, TOP, Mask

__all__ = [
    "ReachChangeRefused", "install", "uninstall", "installed", "submit", "suspended", "proof",
    "record", "begin", "flush", "end", "abort", "compensate", "receipt_of_row", "due",
    "retry_due",
    "grant_entry", "edge_entry", "unedge_entry", "root_entry", "propagate_of", "target_digest",
    "GRANT_CT", "CONTAINS",
]

_log = logging.getLogger(__name__)

GRANT_CT = "application/vnd.agience.grant+json"
CONTAINS = "contains"


class ReachChangeRefused(PermissionError):
    """The authority refused this change to reach (or could not be asked, where that refuses);
    the store has not written it."""


#: `fn(entry, *, proof=None, compensates=None) -> receipt | None`
_authority: Optional[Callable[..., Optional[dict]]] = None
_suspended: contextvars.ContextVar[bool] = contextvars.ContextVar("reach_journal_suspended",
                                                                  default=False)
_proof: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("reach_journal_proof",
                                                                        default=None)
_txn = threading.local()
#: The submission batch: one operation's changes, co-signed together. The authority issues the
#: batch id with the first receipt and the operation carries it after (a caller-chosen name means
#: nothing to it); a creation's own grant is accepted only in the batch that creates it.
_batch: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar("reach_journal_batch",
                                                                        default=None)


@contextmanager
def batch() -> Iterator[dict]:
    """One operation's changes to reach go to the authority as one batch (a creation and its
    creation grant). Nested calls join the outer batch. Yields the holder of the id the
    authority issues."""
    current = _batch.get()
    if current is not None:
        yield current
        return
    token = _batch.set({"id": None})
    try:
        yield _batch.get()
    finally:
        _batch.reset(token)


def install(fn: Callable[..., Optional[dict]]) -> None:
    global _authority
    _authority = fn


def uninstall() -> None:
    global _authority
    _authority = None


def installed() -> bool:
    return _authority is not None


def required() -> bool:
    """Under threshold custody every change to reach must be co-signed."""
    return (os.getenv("MANTLE_MASTER_KEY_CUSTODY") or "").strip().lower() == "threshold"


@contextmanager
def suspended() -> Iterator[None]:
    """Write without journaling: only for a pass that writes back what the authority already
    holds (the bootstrap's receipts)."""
    token = _suspended.set(True)
    try:
        yield
    finally:
        _suspended.reset(token)


@contextmanager
def proof(**evidence) -> Iterator[None]:
    """Evidence the authority checks for the next entries and does not store: an invite claim's
    token (`invite_id`, `token`)."""
    token = _proof.set(dict(evidence))
    try:
        yield
    finally:
        _proof.reset(token)


def _call(entry: dict, **kw) -> Optional[dict]:
    if _authority is None:
        if required():
            raise ReachChangeRefused(
                f"a {entry.get('kind')} change to reach under threshold custody needs the "
                f"authority ledger, and none is installed in this process; nothing is written")
        return None
    holder = _batch.get()
    if "batch" not in kw:
        kw["batch"] = holder["id"] if holder else None
    answer = _authority(entry, proof=_proof.get(), **kw)
    if holder is not None and answer and answer.get("batch") and holder["id"] is None \
            and kw.get("compensates") is None:
        holder["id"] = answer["batch"]          # the batch the authority issued
    return answer


def submit(entry: dict) -> Optional[dict]:
    """Submit one entry now, outside any store transaction (the oracle's `root`). The caller
    confirms it (`confirm_submitted`) once the write it was made for has landed."""
    if _suspended.get():
        return None
    return _call(entry)


def confirm_submitted(entry: dict, receipt: Optional[dict], db=None) -> None:
    """Tell the authority the write a `submit`ted entry was made for landed. No store transaction
    carries it, so no commit confirms it: without this it stays withdrawable and on the
    operator's unconfirmed list. One the authority does not hear is recorded in `db`'s
    `authority_confirm_due` and told again; with no store at hand it is held in this process
    (`_owed`) until the next retry writes it there."""
    if not receipt:
        return
    confirm([(entry, receipt)])              # Origin is asked with no store lock held
    lattice = getattr(getattr(db, "artifacts", None), "db", None)
    if lattice is None:
        return
    seq = receipt.get("seq")
    with _owed_lock:
        if seq is None or int(seq) not in _owed:
            return
        _owed.discard(int(seq))
    with lattice.write() as cur:
        cur.execute("INSERT OR IGNORE INTO authority_confirm_due(seq) VALUES(?)", (int(seq),))


#: Seqs whose confirmation failed with no store connection at hand; written to
#: `authority_confirm_due` by the next `retry_due`.
_owed: set = set()
_owed_lock = threading.Lock()


# -- the transaction --------------------------------------------------------------------------

def _stack() -> list:
    st = getattr(_txn, "stack", None)
    if st is None:
        st = _txn.stack = []
    return st


def begin() -> None:
    """A write transaction opens (one per connection; two connections can nest on a thread)."""
    _stack().append({"pending": [], "done": []})


def end() -> None:
    st = _stack()
    if st:
        st.pop()


def abort(conn=None) -> None:
    """The transaction failed: withdraw whatever it had co-signed. `conn` (autocommit, outside the
    rolled-back transaction) records any withdrawal the authority did not take."""
    st = _stack()
    if st:
        done, st[-1]["done"] = st[-1]["done"], []
        compensate(done, conn)


def record(entry: dict, receipt_key: str) -> None:
    """Record a change made in the open write transaction; submitted at `flush`."""
    if _suspended.get():
        return
    st = _stack()
    if not st:                                # no transaction is collecting: submit at once
        _call(entry)
        return
    st[-1]["pending"].append((entry, receipt_key))


def is_suspended() -> bool:
    return bool(_suspended.get())


def take_pending() -> List[tuple]:
    """The changes this transaction recorded, `[(entry, receipt_key)]`, taken from it."""
    st = _stack()
    if not st:
        return []
    pending, st[-1]["pending"] = st[-1]["pending"], []
    return pending


def cosign(pending: List[tuple], conn=None) -> List[tuple]:
    """Submit recorded changes with no lock held; `[(entry, receipt, receipt_key)]`. A refusal
    withdraws what was already co-signed and raises."""
    done: List[tuple] = []
    try:
        with batch():                       # a transaction outside an operation is its own batch
            for entry, key in pending:
                receipt = _call(entry)
                done.append((entry, receipt, key))
    except BaseException:
        compensate([(e, r) for e, r, _k in done], conn)
        raise
    return done


def write_receipts(cur, done: List[tuple]) -> None:
    """Write the receipts of co-signed changes with `cur`, in the transaction that writes them.
    Raises `ReachChangeRefused` when a receipt names a ledger head lower than one seen."""
    for entry, receipt, key in done:
        _check_head(cur, receipt)
        gone = entry["kind"] == "edge.remove" or (entry["kind"] == "grant"
                                                  and entry["state"] == "deleted")
        if gone:
            cur.execute("DELETE FROM authority_receipt WHERE subject = ?", (key,))
        elif receipt:
            cur.execute(
                "INSERT OR REPLACE INTO authority_receipt(subject, kind, seq, entry_hash, "
                "signature, kid) VALUES(?,?,?,?,?,?)",
                (key, entry["kind"], receipt.get("seq"), receipt.get("entry_hash"),
                 receipt.get("signature"), receipt.get("kid")))


def flush(cur) -> List[tuple]:
    """Submit what this transaction recorded, write the receipts with `cur`, and return
    `[(entry, receipt)]`. Raises `ReachChangeRefused` (after compensating anything already
    co-signed) when one is refused."""
    st = _stack()
    frame = st[-1] if st else {"pending": [], "done": []}
    pending, frame["pending"] = frame["pending"], []
    done: List[tuple] = frame["done"]
    if pending and _authority is not None:
        retry_due(cur)
        if required() and due(cur):
            raise ReachChangeRefused(
                "a withdrawal the authority has not taken is outstanding; under threshold "
                "custody no change to reach is written until it clears")
    try:
        for entry, key in pending:
            receipt = _call(entry)
            done.append((entry, receipt))   # co-signed: withdrawn if anything below refuses
            _check_head(cur, receipt)
            gone = entry["kind"] == "edge.remove" or (entry["kind"] == "grant"
                                                      and entry["state"] == "deleted")
            if gone:
                cur.execute("DELETE FROM authority_receipt WHERE subject = ?", (key,))
            elif receipt:
                cur.execute(
                    "INSERT OR REPLACE INTO authority_receipt(subject, kind, seq, entry_hash, "
                    "signature, kid) VALUES(?,?,?,?,?,?)",
                    (key, entry["kind"], receipt.get("seq"), receipt.get("entry_hash"),
                     receipt.get("signature"), receipt.get("kid")))
    except BaseException:
        # The transaction rolls back; `abort(conn)` then withdraws what was co-signed, and records
        # any withdrawal Origin does not take.
        if not st:
            compensate(done)
        raise
    return list(done)


_HEAD = "ledger:head"


def _check_head(cur, receipt: Optional[dict]) -> None:
    """Keep the highest ledger head a receipt named; refuse one that names a lower head."""
    head = (receipt or {}).get("head_seq")
    if head is None:
        return
    r = cur.execute("SELECT seq FROM authority_receipt WHERE subject = ?", (_HEAD,)).fetchone()
    seen = r[0] if r is not None else None
    if seen is not None and int(head) < int(seen):
        raise ReachChangeRefused(
            f"the authority's ledger head went backwards, from {seen} to {head}: its ledger may "
            f"have been truncated or restored from an older copy; nothing is written")
    if seen is None or int(head) > int(seen):
        cur.execute("INSERT OR REPLACE INTO authority_receipt(subject, kind, seq) "
                    "VALUES(?, 'head', ?)", (_HEAD, int(head)))


def _inverse(entry: dict) -> Optional[dict]:
    if entry["kind"] == "edge.add":
        return unedge_entry(entry["src"], entry["dst"])
    if entry["kind"] == "grant" and entry["state"] not in ("deleted", "retired"):
        return {**entry, "state": "retired"}
    return None


def compensate(done: List[tuple], conn=None) -> None:
    """Withdraw co-signed widening entries whose write did not commit: an added edge is removed,
    a put grant is retired. Narrowing entries need nothing: the authority is then narrower than
    the store, which refuses rather than grants. A withdrawal the authority does not take is
    recorded with `conn` (`authority_compensation_due`) to be retried."""
    if _authority is None:
        return
    for entry, receipt in reversed(done):
        if not receipt or receipt.get("seq") is None:
            continue
        if receipt.get("outcome") == "unchanged":
            continue                 # it names an older entry, not one this write made
        inverse = _inverse(entry)
        if inverse is None:
            continue
        try:
            _authority(inverse, proof=None, compensates=receipt["seq"])
        except Exception as exc:  # noqa: BLE001 — recorded, and retried before the next change
            if _superseded(exc, inverse, conn):
                continue                 # our own later change stands; nothing to withdraw
            _log.error("could not withdraw authority entry %s for a write that did not commit; "
                       "Origin holds reach this store does not (%s); recorded to retry",
                       receipt.get("seq"), entry.get("kind"), exc_info=True)
            if conn is not None:
                try:
                    conn.execute("INSERT OR REPLACE INTO authority_compensation_due(seq, entry) "
                                 "VALUES(?, ?)", (int(receipt["seq"]), json.dumps(inverse)))
                except Exception:  # noqa: BLE001 — the log above is then the only record
                    _log.critical("could not record the outstanding withdrawal of authority "
                                  "entry %s", receipt.get("seq"), exc_info=True)


_LATEST = re.compile(r"latest entry (\d+)")


def _subject_of(entry: dict) -> Optional[str]:
    if entry.get("kind") in ("edge.add", "edge.remove"):
        return f"edge:{entry['src']} {entry['dst']}"
    if entry.get("kind") == "grant":
        return f"grant:{entry['grant_id']}"
    return None


def _superseded(exc: BaseException, inverse: dict, conn) -> bool:
    """The authority answered 409 to a withdrawal: the row has changed since. Nothing is left to
    withdraw only if that later change is one this store committed, its receipt naming the entry
    the authority calls latest. Otherwise the authority may hold a change this store never wrote,
    and the withdrawal stays due."""
    for e in (exc, getattr(exc, "__cause__", None)):
        if getattr(e, "status", None) == 410:
            return True              # the operator resolved it at Origin
        if getattr(e, "status", None) == 409:
            m = _LATEST.search(str(getattr(e, "detail", "") or e))
            key = _subject_of(inverse)
            if not m or not key or conn is None:
                return False
            r = conn.execute("SELECT seq FROM authority_receipt WHERE subject = ?",
                             (key,)).fetchone()
            return r is not None and r[0] == int(m.group(1))
    return False


def confirm(done: List[tuple], conn=None) -> None:
    """Tell the authority these co-signed entries' writes committed, so none can be withdrawn
    later. One it does not hear is recorded (`authority_confirm_due`) and told again."""
    seqs = [int(r["seq"]) for _e, r, *_k in done
            if r and r.get("seq") is not None and r.get("outcome") != "unchanged"]
    tell = getattr(_authority, "confirm", None)
    if not seqs or tell is None:
        return
    try:
        tell(seqs)
    except Exception:  # noqa: BLE001 — recorded, and told again
        _log.warning("authority entries %s committed and not yet confirmed; recorded", seqs,
                     exc_info=True)
        if conn is not None:
            for s in seqs:
                conn.execute("INSERT OR IGNORE INTO authority_confirm_due(seq) VALUES(?)", (s,))
        else:
            with _owed_lock:
                _owed.update(seqs)


def due(conn) -> List[tuple]:
    """Withdrawals the authority has not taken: `[(seq, inverse entry)]`."""
    return [(r[0], json.loads(r[1])) for r in conn.execute(
        "SELECT seq, entry FROM authority_compensation_due ORDER BY seq").fetchall()]


def retry_due(conn) -> int:
    """Submit each outstanding withdrawal again; returns how many cleared. It goes as the system
    principal (`retry` on the installed authority), still naming the entry it withdraws, so the
    authority takes it only while that entry is the latest change to its row."""
    if _authority is None:
        return 0
    retry = getattr(_authority, "retry", None)
    cleared = 0
    for seq, inverse in due(conn):
        try:
            if retry is not None:
                retry(inverse, seq)
            else:
                _authority(inverse, proof=None, compensates=seq)
        except Exception as exc:  # noqa: BLE001 — still outstanding
            if _superseded(exc, inverse, conn):
                conn.execute("DELETE FROM authority_compensation_due WHERE seq = ?", (seq,))
                cleared += 1
                continue
            _log.warning("outstanding withdrawal of authority entry %s not taken yet: %s",
                         seq, exc)
            continue
        conn.execute("DELETE FROM authority_compensation_due WHERE seq = ?", (seq,))
        cleared += 1
    tell = getattr(_authority, "confirm", None)
    with _owed_lock:
        held = sorted(_owed)
        _owed.clear()
    for s in held:
        conn.execute("INSERT OR IGNORE INTO authority_confirm_due(seq) VALUES(?)", (s,))
    owed = [r[0] for r in conn.execute("SELECT seq FROM authority_confirm_due").fetchall()]
    if owed and tell is not None:
        try:
            tell(owed)
            for s in owed:
                conn.execute("DELETE FROM authority_confirm_due WHERE seq = ?", (s,))
        except Exception as exc:  # noqa: BLE001 — still owed
            _log.warning("authority entries %s still not confirmed: %s", owed, exc)
    return cleared


def receipt_of_row(conn, key: str) -> Optional[dict]:
    """The receipt kept for a row, or None."""
    r = conn.execute("SELECT seq, entry_hash, signature, kid FROM authority_receipt "
                     "WHERE subject = ?", (key,)).fetchone()
    if r is None:
        return None
    return {"seq": r[0], "entry_hash": r[1], "signature": r[2], "kid": r[3]}


# -- entries -------------------------------------------------------------------------------------

def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def target_digest(kind: Optional[str], value: Optional[str]) -> Optional[str]:
    """An invite target as the ledger carries it: a digest of the normalised value, never the
    value. Origin compares it with the claimant's own verified identity. `mantle.db` is
    embeddable and imports no sibling package, so this is its own copy of
    `prism.trust.ledger.target_digest`; Origin's cross-repo test holds the two to one answer."""
    import hashlib

    if not kind or not value:
        return None
    v = str(value).strip()
    if kind in ("email", "domain"):
        v = v.lower()
    return hashlib.sha256(f"agience-invite-target:{kind}:{v}".encode("utf-8")).hexdigest()


def grant_entry(grant: Any, *, state: Optional[str] = None) -> dict:
    """A grant (entity or stored document) as a `grant` entry. The actions are the CRUDEASIO
    flags set, effect-blind, as `Mask.to_flags` reads them. The effect is `allow` only for exactly
    that (`grant_is_allow`); anything else is `deny`, which is how every reach rule reads it.

    An invite also carries its target, as a kind and a digest, and its claim limit."""
    actions = [a for a in ACTIONS if _get(grant, "can_" + a, False)]
    effect = str(_get(grant, "effect") or "").strip().lower()
    gtype = str(_get(grant, "grantee_type") or "")
    invite = gtype == "invite"
    tkind = _get(grant, "target_entity_type") if invite else None
    digest = None
    if invite and tkind:
        # The stored document keeps the digest (`lattice_api.to_lattice_doc`); an entity has the
        # address itself.
        digest = _get(grant, "target_digest") or target_digest(tkind, _get(grant, "target_entity"))
    max_claims = _get(grant, "max_claims") if invite else None
    return {
        "kind": "grant",
        "grant_id": str(_get(grant, "id") or ""),
        "grantee_id": str(_get(grant, "grantee_id") or ""),
        "grantee_type": gtype,
        "resource_id": str(_get(grant, "resource_id") or ""),
        "actions": actions,
        "effect": "allow" if effect == "allow" else "deny",
        "state": state or str(_get(grant, "state") or "active"),
        "expires_at": _get(grant, "expires_at") or None,
        "target": {"kind": str(tkind), "digest": digest} if digest else None,
        "max_claims": int(max_claims) if invite and max_claims is not None else None,
    }


def _confers(entry: Optional[dict]) -> set:
    if not entry or entry.get("state") != "active" or entry.get("effect") != "allow":
        return set()
    return set(entry.get("actions") or ())


def _time(value: Any):
    """An expiry as an aware datetime; one that does not parse is the earliest time there is, so
    it never reads as a longer grant."""
    from datetime import datetime, timezone

    try:
        t = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def grant_widens(prior: Optional[dict], new: dict) -> bool:
    """Does `new` authorize anything `prior` did not, or for longer? Origin's own test
    (`authority_ledger._widens`), on the entries."""
    if prior and prior.get("effect") == "deny" and prior.get("state") == "active" and (
            new.get("effect") != "deny" or new.get("state") != "active"
            or not set(prior.get("actions") or ()) <= set(new.get("actions") or ())):
        return True
    if _confers(new) - _confers(prior):
        return True
    if prior and prior.get("effect") == "deny" and prior.get("state") == "active":
        a, b = prior.get("expires_at"), new.get("expires_at")
        if a != b and (a is None or (b is not None and _time(a) > _time(b))):
            return True                      # a deny that ends sooner lifts it sooner
    if _confers(new) and prior is not None:
        a, b = new.get("expires_at"), prior.get("expires_at")
        return a != b and (a is None or (b is not None and _time(a) > _time(b)))
    return False


def edge_widens(prior: Optional[tuple], origin: bool, propagate: Any) -> bool:
    """Does writing this `contains` edge widen reach over `prior` `(is_origin, propagate)`?"""
    if prior is None:
        return True
    if origin and not prior[0]:
        return True
    return not Mask.from_propagate(propagate).actions <= Mask.from_propagate(prior[1]).actions


def hold(cur, kind: str, subject: str, payload: dict, received_from: str) -> None:
    """Hold a widening that arrived from a peer (mesh consume) until a person here accepts it
    (`services.held_reach.accept`): it is not written, and the acceptance is the co-signed
    change. Only under threshold custody, where Origin decides reach by what it co-signed."""
    resource = payload.get("resource_id") if kind == "grant" else payload.get("src")
    cur.execute("INSERT INTO held_reach_change(kind, subject, payload, received_from, "
                "resource_id) VALUES(?,?,?,?,?)",
                (kind, subject, json.dumps(payload), received_from, resource))


def propagate_of(value: Any) -> Optional[list]:
    """An edge's stored `propagate` as the entry carries it: None when it conducts every action,
    else the actions it conducts, decoded exactly as the lattice reads the column."""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    mask = Mask.from_propagate(value)
    if mask == TOP:
        return None
    return [a for a in ACTIONS if mask.allows(a)]


def edge_entry(src: str, dst: str, *, origin: bool, propagate: Any) -> dict:
    return {"kind": "edge.add", "src": str(src), "dst": str(dst), "origin": bool(origin),
            "propagate": propagate_of(propagate)}


def unedge_entry(src: str, dst: str) -> dict:
    return {"kind": "edge.remove", "src": str(src), "dst": str(dst)}


def root_entry(principal_id: str, root_id: Optional[str] = None) -> dict:
    return {"kind": "root", "principal_id": str(principal_id),
            "root_id": str(root_id or principal_id)}
