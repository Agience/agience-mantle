"""Trash: Delete moves an artifact here, and it is deleted for good after 30 days.

An artifact in Trash is an archived artifact that carries ``trashed_by`` and ``trashed_time`` on
the newest version of its lineage. Archived, so every default read passes it by (`spine.md` §2);
marked, so a person's Trash lists it and a container's Archived view does not. Restoring it is
the ordinary restore (`state: live`), which clears both marks.

Old versions "are deleted only by policy, and the policy itself is the audit trail"
(`architecture.md`). This is that policy for Trash: :func:`purge_expired` deletes what has been
in Trash longer than :data:`RETENTION_DAYS`, through the same delete a person runs, as the person
who trashed it. If their right to delete it is gone, it stays, and the refusal is logged.

A person's Trash is listed from the ``listkey`` index on ``trashed_by``, so it costs what the
Trash holds rather than a scan of every archived version. A store whose index predates
``listkey`` is scanned instead, and says so.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional

from mantle.db.constants import state_of

logger = logging.getLogger(__name__)

#: How long an artifact stays in Trash before it is deleted.
RETENTION_DAYS = 30

#: The most items one Trash listing returns.
LIST_LIMIT = 1000

_ARCHIVED = "archived"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse(stamp: Any) -> Optional[datetime]:
    try:
        when = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def purge_after(trashed_time: Any) -> Optional[str]:
    """When an artifact trashed at ``trashed_time`` is deleted, or ``None`` if unreadable."""
    when = _parse(trashed_time)
    return (when + timedelta(days=RETENTION_DAYS)).isoformat() if when else None


def is_trashed(doc: Optional[Dict[str, Any]]) -> bool:
    return bool(doc and doc.get("trashed_by") and state_of(doc) == _ARCHIVED)


def _head(db: Any, artifact_id: str) -> Optional[Dict[str, Any]]:
    """The newest version of ``artifact_id``'s lineage, raw."""
    raw = db.artifacts.get_artifact(artifact_id)
    root = (raw or {}).get("root_id") or artifact_id
    versions = db.artifacts.versions_of(root) or []
    return dict(versions[-1]) if versions else (dict(raw) if raw else None)


def mark(db: Any, user_id: str, artifact_id: str) -> Optional[Dict[str, Any]]:
    """Mark the archived head of ``artifact_id`` as trashed by ``user_id``, now.

    Returns the marked document, or ``None`` when the head is not archived (archive it first).
    Written through the store's artifact update, which announces it on the change feed."""
    from mantle.db import backend as db_store
    from mantle.db.lattice_api import from_lattice_doc
    from mantle.entities.artifact import Artifact

    head = _head(db, artifact_id)
    if not head or state_of(head) != _ARCHIVED:
        return None
    entity = from_lattice_doc(head, Artifact)
    entity.trashed_by = user_id
    entity.trashed_time = _now().isoformat()
    entity.modified_by = user_id
    db_store.update_artifact(db, entity)
    return entity.to_dict()


def _mine(doc: Dict[str, Any], user_id: str) -> bool:
    return is_trashed(doc) and str(doc.get("trashed_by")).lower() == str(user_id).lower()


def list_for(db: Any, user_id: str) -> List[Dict[str, Any]]:
    """``user_id``'s Trash, most recently trashed first."""
    from mantle.db.vertex import ListIndexUnbuilt
    try:
        docs = list(db.artifacts.lookup_by_list_field("trashed_by", user_id, limit=LIST_LIMIT))
    except ListIndexUnbuilt:
        logger.warning("trash: the listkey index is not built on this store, so Trash is listed "
                       "by scanning archived artifacts. Run rebuild_list_index() once.")
        docs = [d for d in db.artifacts.list_artifacts(state=_ARCHIVED) if d.get("trashed_by")]
    mine = [d for d in docs if _mine(d, user_id)]
    return sorted(mine, key=lambda d: str(d.get("trashed_time") or ""), reverse=True)


def _all_trashed(db: Any) -> Iterator[Dict[str, Any]]:
    from mantle.db.vertex import ListIndexUnbuilt
    try:
        after = ""
        while True:
            page = db.artifacts.lookup_with_list_field("trashed_by", after=after)
            yield from page
            if not getattr(page, "truncated", False) or not page:
                return
            after = page[-1]["id"]
    except ListIndexUnbuilt:
        yield from (d for d in db.artifacts.list_artifacts(state=_ARCHIVED) if d.get("trashed_by"))


def expired(db: Any, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Every trashed artifact whose retention has run out at ``now``."""
    cutoff = (now or _now()) - timedelta(days=RETENTION_DAYS)
    out = []
    for doc in _all_trashed(db):
        when = _parse(doc.get("trashed_time"))
        if is_trashed(doc) and when is not None and when <= cutoff:
            out.append(doc)
    return out


def purge_expired(db: Any, now: Optional[datetime] = None) -> Dict[str, int]:
    """Delete every artifact that has been in Trash longer than :data:`RETENTION_DAYS`.

    Each is deleted through the delete a person runs, as the person who trashed it, so the
    policy can do nothing they could not. Returns ``{"deleted", "refused", "failed"}``."""
    from fastapi import HTTPException
    from mantle.services import workspace_service
    from mantle.services.system_identity import system_acting_context

    counts = {"deleted": 0, "refused": 0, "failed": 0}
    for doc in expired(db, now):
        aid, who = doc["id"], str(doc["trashed_by"])
        container_id = doc.get("collection_id")
        try:
            with system_acting_context(scope="platform.trash-purge"):
                if container_id:
                    workspace_service.delete_artifact(db, who, container_id, aid)
                else:
                    workspace_service.delete_workspace(db, who, aid)
            counts["deleted"] += 1
            logger.info("trash purge: deleted %s, trashed by %s at %s", aid, who,
                        doc.get("trashed_time"))
        except HTTPException as exc:
            counts["refused"] += 1
            logger.warning("trash purge: %s kept — %s may no longer delete it (%s)",
                           aid, who, exc.detail)
        except Exception:
            counts["failed"] += 1
            logger.error("trash purge: deleting %s failed", aid, exc_info=True)
    if any(counts.values()):
        logger.info("trash purge: %s", counts)
    return counts


#: How often the purge runs. Retention is counted in days, so an hour late is immaterial.
PURGE_INTERVAL_S = 3600


async def run_purge_loop(interval_s: float = PURGE_INTERVAL_S) -> None:
    """Run :func:`purge_expired` now and then every ``interval_s``, until cancelled."""
    import asyncio
    from mantle.services.dependencies import get_store_db

    def _once() -> None:
        gen = get_store_db()
        db = next(gen)
        try:
            purge_expired(db)
        finally:
            gen.close()

    while True:
        try:
            await asyncio.to_thread(_once)
        except Exception:
            logger.error("trash purge pass failed", exc_info=True)
        await asyncio.sleep(interval_s)


__all__ = ["LIST_LIMIT", "PURGE_INTERVAL_S", "RETENTION_DAYS", "expired", "is_trashed",
           "list_for", "mark", "purge_after", "purge_expired", "run_purge_loop"]
