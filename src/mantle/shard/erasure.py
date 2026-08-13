"""ERASURE — wipe everything attached to one higgs, and nothing else.

For a system whose whole ontology is grounding / grants / conscious observation (§13.11.7), being
able to remove your own portion is the same right as being able to add to it.

Two categories distinguish what erasure touches from what it leaves alone:

  * Grounded at you — you authored it, or it lives in your private collection. It is yours, and
    erasure removes it.
  * Reached by you — the commons you looked at. WordNet, Wikipedia, anything CC/OA. Observing a
    thing does not attach it to you, and deleting it would delete the training set.

Erasure is defined positively: it collects what is provably grounded at this person and removes
that. It never works by exclusion ("everything except the corpus"), because an exclusion list
misses whatever class nobody thought to name — that artifact would be destroyed silently.

`erase()` reports and changes nothing unless `apply=True` is passed explicitly. The inventory is
the product; the deletion is a separate decision made by a human who has read it.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


def _person_ids(store, person: str) -> Set[str]:
    """Every id this principal is known by — the raw claim AND its resolved person artifact.

    Both are needed: rows are stamped with the resolved uuid, while collections and grants are keyed
    on the raw principal string. Erasing on one alone leaves the other half attached."""
    ids = {str(person)}
    from mantle.system import runner_hooks     # author resolution lives here (mantle -> ember is a one-way street)
    try:
        ref = runner_hooks.author_ref(store, person)
    except Exception:
        ref = None
    if ref is not None:
        ids.add(str(ref))
    return {i for i in ids if i}


# The classes of thing a higgs is made of. Named explicitly so the inventory is readable — an
# erasure report that says "417 artifacts" tells its reader nothing about what they are losing.
CLASSES = (
    ("private", "artifacts in your private collection"),
    ("authored", "artifacts you authored"),
    ("conversation", "messages and conversations"),
    ("identity", "your person artifact itself"),
)

#
# This is the grounding question framed in §13.11.7: the operators are grounded at the foundation,
# the commons at ground zero, and only the observer's own work at the observer. So these are
# reported as `not_yours` rather than silently skipped — "I found these and they are not yours to
# erase" is a different and more useful statement than not mentioning them.
_REGISTRY_TYPES = ("operator+json", "vtype+json", "etype+json", "rung+json", "collection+json",
                   "content-type+json", "x-citation", "shard-done+json", "form-mesh+json")


def _is_registry(ct: str) -> bool:
    return any(ct.endswith(t) or ct == t for t in _REGISTRY_TYPES)


def attached(store, person: str, *, include_identity: bool = False) -> Dict[str, Any]:
    """INVENTORY — what is grounded at this higgs. Reads only; changes nothing.

    `include_identity` decides between a RESET (keep the person, drop everything they made) and a
    full erasure (the person goes too). They are genuinely different acts and the caller must say
    which, rather than one being a silent default."""
    ids = _person_ids(store, person)   # author resolution lives in runner_hooks (mantle ↛ ember)
    private_ids = {"private.%s" % p for p in ids}
    found: Dict[str, List[str]] = {k: [] for k, _d in CLASSES}
    not_yours: List[str] = []
    unresolved: List[str] = []
    scanned = 0

    conn = store.artifacts.db.read()
    import json
    for (doc,) in conn.execute("SELECT doc FROM vertex"):
        scanned += 1
        try:
            d = json.loads(doc)
        except Exception:
            continue
        aid = str(d.get("id") or "")
        ct = str(d.get("content_type") or "")
        coll = str(d.get("collection_id") or "")
        made_by = str(d.get("created_by") or "")

        if aid in ids:
            if include_identity and ct.endswith("person+json"):
                found["identity"].append(aid)
            continue
        if coll in private_ids:
            found["private"].append(aid)
            continue
        if made_by in ids:
            if _is_registry(ct):
                not_yours.append(aid)          # the ontology: grounded at the foundation
                continue
            if d.get("cited_from") or str(coll).startswith("stage."):
                continue
            (found["conversation"] if "message" in ct or aid.startswith("convo.")
             else found["authored"]).append(aid)
    return {"person": person, "ids": sorted(ids), "scanned": scanned,
            "found": found,
            "counts": {k: len(v) for k, v in found.items()},
            "total": sum(len(v) for v in found.values()),
            "not_yours": not_yours,
            "unresolved": unresolved}


def _container_of(store, aid: str) -> str:
    """The collection an artifact is in, read BEFORE it is erased.

    Only the container is taken, and only so the event can be addressed. An event addressed to
    the artifact's own id reaches nobody watching the collection it lived in, which is precisely
    the subscriber that still holds state keyed on it."""
    try:
        return str((store.artifacts.get_artifact(aid) or {}).get("collection_id") or "")
    except Exception:
        return ""


def _announce_erased(aid: str, container_id: str) -> None:
    """Tell the change feed one artifact is gone.

    **The event carries an id and a delete verb and nothing else.** No field is read off the
    erased doc — not a title, not a content type, not an author. An erasure that announced what
    it erased would put the erased data on a feed every wildcard subscriber can read, and the
    announcement would outlive the thing it was announcing in every subscriber's log. What a
    subscriber needs in order to comply is exactly the id it must drop.

    Best-effort, like every emit on a write path: a feed that cannot be written to must never
    stop an erasure from completing. The deletion is the right being exercised; the event is
    only how derived state learns to follow.
    """
    try:
        from mantle.db.doc_boundary import DELETED, emit_artifact_change
        emit_artifact_change({"id": aid, "collection_id": container_id}, DELETED)
    except Exception:
        logger.debug("erasure event emit failed for %s", aid, exc_info=True)


def erase(store, person: str, *, apply: bool = False,
          include_identity: bool = False) -> Dict[str, Any]:
    """Remove everything grounded at this higgs. **DRY RUN unless `apply=True`.**

    Returns the inventory plus what was actually removed, so the report is the same shape either
    way and a caller can diff a dry run against the real one.

    Each removal is announced on the change feed, after the delete succeeds — a subscriber
    holding derived state keyed on these ids is the one place erased data can survive the
    erasure, so it is the one deletion where being told matters most. A row that failed to
    delete is not announced: an event for an artifact still present is a subscriber dropping
    state the store still has."""
    inv = attached(store, person, include_identity=include_identity)
    if not apply:
        inv["applied"] = False
        return inv
    removed, failed = [], []
    for _cls, ids in inv["found"].items():
        for aid in ids:
            container = _container_of(store, aid)
            try:
                store.artifacts.delete_artifact(aid)
                removed.append(aid)
                _announce_erased(aid, container)
            except Exception as e:
                failed.append("%s: %s" % (aid, type(e).__name__))
    inv["applied"] = True
    inv["removed"] = len(removed)
    inv["failed"] = failed
    # A partial erasure that is reported is recoverable; one reported as complete is not.
    inv["complete"] = not failed and len(removed) == inv["total"]
    return inv


__all__ = ["CLASSES", "attached", "erase"]
