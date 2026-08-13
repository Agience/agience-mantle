"""Access-audit accessor over the lattice's `access_event` table.

The ONE module that touches that table — services never issue SQL against the lattice
themselves (`services/audit_service.py` calls these in lattice mode). Append-only writes,
one indexed read shape (an artifact's history, newest first). See the DDL note in
`schema.py` for why this is a sidecar table and neither edges nor vertices.

Retention
---------
The default is **retain indefinitely**, and that default is a decision, not an omission.
An access log is the record that an authorization decision was made; a node that discards
it after a fixed window cannot answer "who read this, and when" beyond that window, which
is the one question the log exists to answer. Deployments where the log is itself the
regulated data want the opposite, so the age bound is a setting rather than a constant:

    MANTLE_AUDIT_RETENTION_DAYS   unset or 0  → unlimited (the default)
                                  N > 0       → `prune_access_events` drops rows older than N days

Nothing prunes on its own at the default. `services/audit_service` runs the sweep on a slow
cadence only while the setting is set, so an unconfigured node executes no delete at all.

An event names an artifact id forever, including after that artifact is erased. That is
deliberate for the same reason the rest of the log is append-only — but it means erasure
(`shard/erasure.py`) removes the artifact, not the record that it was read, and a
deployment that needs both needs the age bound set.

`ts` is an ISO-8601 UTC string, so `<` on it is chronological as long as every writer uses
the one format `audit_service.record_access` writes. It is compared, never parsed, here.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Rows deleted per prune pass. `access_event` is indexed on `(artifact_id, ts)`, not on `ts`
#: alone, so the age predicate is a scan; a bounded pass keeps one sweep from holding the
#: write lock across a large backlog. The caller loops until a pass comes back short, so the
#: value sets the granularity of the sweep and never its outcome.
_PRUNE_BATCH = 5000


def append_access_events(db, batch: List[Dict[str, Any]]) -> int:
    """Append a drained batch of access events. Returns rows written.

    `db` is the low-level `LatticeConn` (the handle's `.conn`). Event dicts use the
    audit service's shape; lattice-era `_from`/`_to` keys are ignored."""
    if not batch:
        return 0
    rows = [(e.get("principal_id"), e.get("artifact_id"), e.get("action"),
             e.get("result"), e.get("ts"), json.dumps(e.get("ctx") or {}))
            for e in batch if e.get("artifact_id")]
    if not rows:
        return 0
    with db.write() as cur:
        cur.executemany(
            "INSERT INTO access_event(principal_id, artifact_id, action, result, ts, ctx)"
            " VALUES(?,?,?,?,?,?)", rows)
    return len(rows)


def access_log_of(db, artifact_id: str, *, limit: int = 100, offset: int = 0,
                  result: Optional[str] = None) -> List[Dict[str, Any]]:
    """An artifact's access history, newest first — a bounded walk of `ix_ae_artifact`.

    OFFSET is acceptable HERE (unlike artifact paging): the log is viewed newest-first in
    small admin pages, and the index already orders the walk — no deep-skip workload exists."""
    q = ("SELECT principal_id, action, result, ts, ctx FROM access_event"
         " WHERE artifact_id = ?")
    params: list = [artifact_id]
    if result in ("allowed", "denied"):
        q += " AND result = ?"
        params.append(result)
    q += " ORDER BY ts DESC LIMIT ? OFFSET ?"
    params += [int(limit), int(offset)]
    out = []
    for r in db.read().execute(q, params):
        ctx = r["ctx"]
        try:
            ctx = json.loads(ctx) if ctx else {}
        except ValueError:
            ctx = {}
        out.append({"principal_id": r["principal_id"], "action": r["action"],
                    "result": r["result"], "ts": r["ts"], "ctx": ctx})
    return out


def retention_days() -> int:
    """The configured age bound in days; 0 means unlimited (the default).

    A malformed or negative value reads as unlimited and says so, because the failure mode
    of guessing here is deleting an audit trail nobody asked to have deleted."""
    raw = (os.getenv("MANTLE_AUDIT_RETENTION_DAYS") or "").strip()
    if not raw:
        return 0
    try:
        days = int(raw)
    except ValueError:
        logger.warning("MANTLE_AUDIT_RETENTION_DAYS=%r is not an integer — retaining indefinitely", raw)
        return 0
    if days <= 0:
        return 0
    return days


def retention_cutoff(days: Optional[int] = None) -> Optional[str]:
    """The ISO timestamp rows must be newer than, or None when retention is unlimited."""
    days = retention_days() if days is None else days
    if days <= 0:
        return None
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def prune_access_events(db, *, before_ts: Optional[str] = None,
                        days: Optional[int] = None) -> int:
    """Drop access events older than the cutoff. Returns rows deleted.

    Unlimited retention is a real answer, not a missing one: with no cutoff resolvable this
    returns 0 having issued no statement, so calling it unconditionally is safe.

    `db` is the low-level `LatticeConn` (the handle's `.conn`), matching `append_access_events`."""
    cutoff = before_ts if before_ts is not None else retention_cutoff(days)
    if not cutoff:
        return 0
    deleted = 0
    while True:
        with db.write() as cur:
            cur.execute(
                "DELETE FROM access_event WHERE rowid IN ("
                "  SELECT rowid FROM access_event WHERE ts < ? LIMIT ?)",
                (cutoff, _PRUNE_BATCH))
            n = cur.rowcount or 0
        deleted += n
        # A short pass means the predicate is exhausted. Looping on the count rather than on a
        # second query keeps the sweep to one statement shape.
        if n < _PRUNE_BATCH:
            break
    if deleted:
        logger.info("audit retention: pruned %d access event(s) older than %s", deleted, cutoff)
    return deleted
