"""Access-audit accessor over the lattice's `access_event` table.

The ONE module that touches that table — services never issue SQL against the lattice
themselves (`services/audit_service.py` calls these in lattice mode). Append-only writes,
one indexed read shape (an artifact's history, newest first). See the DDL note in
`schema.py` for why this is a sidecar table and neither edges nor vertices.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
