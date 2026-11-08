"""Widenings of reach that arrived from a peer, held until a person here accepts them.

Under threshold custody Origin decides key release by the changes it co-signed, and a peer's
change was made on the peer's authority, which Origin here has not seen. So a mesh-consumed
`contains` edge or grant that widens reach is not written: the store holds it
(`reach_journal.hold`, table `held_reach_change`). A narrowing is written and journaled as usual.

A person on this side accepts a held change by writing it, on their own credential: the write is
journaled like any other, Origin checks that this person may make it, and that co-signed write is
the change. Refused, it stays held. Rejecting it drops it.
"""
from __future__ import annotations

import json
from typing import List, Optional


def held_resources(db) -> List[str]:
    """The distinct resources held changes would widen (indexed)."""
    return [r[0] for r in db.graph.db.read().execute(
        "SELECT DISTINCT resource_id FROM held_reach_change WHERE resource_id IS NOT NULL")]


def list_held(db, resources: Optional[List[str]] = None) -> List[dict]:
    """Held changes, oldest first; only those on `resources` when given."""
    sql = ("SELECT id, kind, subject, payload, received_from, held_at FROM held_reach_change")
    params: tuple = ()
    if resources is not None:
        if not resources:
            return []
        sql += " WHERE resource_id IN (%s)" % ",".join("?" * len(resources))
        params = tuple(resources)
    rows = db.graph.db.read().execute(sql + " ORDER BY id", params).fetchall()
    return [{"id": r["id"], "kind": r["kind"], "subject": r["subject"],
             "payload": json.loads(r["payload"]), "received_from": r["received_from"],
             "held_at": r["held_at"]} for r in rows]


def _row(db, held_id: int) -> Optional[dict]:
    return next((h for h in list_held(db) if h["id"] == int(held_id)), None)


def accept(db, held_id: int) -> dict:
    """Write a held change as the acting person: journaled, co-signed by Origin or refused. On
    success the hold is dropped; a refusal raises (`ReachChangeRefused`) and the hold stays."""
    held = _row(db, held_id)
    if held is None:
        raise KeyError(f"no held change {held_id}")
    p = held["payload"]
    with db.graph.db.write() as cur:
        if held["kind"] == "edge":
            db.graph.add_edges([(p["src"], p["dst"], p["label"], p["props"])])
        else:
            db.artifacts.put_artifact(p)
        cur.execute("DELETE FROM held_reach_change WHERE id = ?", (int(held_id),))
    return held


def reject(db, held_id: int) -> bool:
    with db.graph.db.write() as cur:
        cur.execute("DELETE FROM held_reach_change WHERE id = ?", (int(held_id),))
        return cur.rowcount > 0
