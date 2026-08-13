"""Curation — collections, who owns what, and moving things in and out.

No new surface, no new store: collections are already artifacts and membership is already a
field, so curation is reading and writing what is there rather than a parallel system that can
disagree with it.

    the commons       grounded at ground zero — CC/OA material anyone may read and nobody curates
                      into their own collections. Observing does not confer disposal.
    the foundation    operators and the type registries. Not one person's to move.
    the observer      what you authored, and what lives in your private collection. Yours.

"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# The change-feed vocabulary, taken from the module that defines it rather than spelled out here:
# a subscriber filters on these exact names, so a second spelling of them is a filter that silently
# selects nothing.
from mantle.db.doc_boundary import CREATED as _CREATED, DELETED as _DELETED

logger = logging.getLogger(__name__)

COLLECTION_CT = "application/vnd.agience.collection+json"

# The system's own provenance anchor. `cite.genesis` means the GENESIS system authored this
# deliberately — it is not a source the row was ingested from. Treating it as one would
# misclassify a person's private conversation as commons; see `owner_of`.
_SYSTEM_CITE = "cite.genesis"

COMMONS_PRINCIPAL = "common@ground"
LEGACY_COMMONS = ("anonymous@public",)


def is_commons_principal(who: str) -> bool:
    return str(who) == COMMONS_PRINCIPAL or str(who) in LEGACY_COMMONS

# Grounded at the foundation, never at a person — see the header and `erasure._REGISTRY_TYPES`.
_REGISTRY = ("operator+json", "vtype+json", "etype+json", "rung+json", "collection+json",
             "content-type+json", "x-citation", "shard-done+json", "form-mesh+json")


def _is_registry(ct: str) -> bool:
    return any(str(ct).endswith(t) for t in _REGISTRY)


def _memberships(a: Dict[str, Any]) -> List[str]:
    out = list(a.get("collections") or [])
    cid = a.get("collection_id")
    if cid and cid not in out:
        out.insert(0, str(cid))
    return [str(x) for x in out]


def owner_of(a: Dict[str, Any], gated_owners: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """`(layer, who)` — which grounding layer this artifact sits in, and who holds it. Reported rather
    than decided: a caller can always see WHY something is or is not theirs.

    grant-gating outranks any citation, so a private memory stays curatable by its owner."""
    owners = gated_owners or {}
    g = str(a.get("collection_id") or a.get("origin_root") or "")
    if g and g in owners:
        return "observer", str(owners[g])            # grant-gated -> owner = the grant's grantee
    ct = str(a.get("content_type") or "")
    if _is_registry(ct):
        return "foundation", "the foundation"
    cited = str(a.get("cited_from") or "")
    if (cited and cited != _SYSTEM_CITE) or g.startswith("stage."):
        return "commons", cited or "ground zero"
    return "observer", str(a.get("created_by") or "unknown")


def may_curate(store, a: Dict[str, Any], person: str) -> Tuple[bool, str]:
    """May `person` file or unfile this artifact? Returns `(ok, why)` — the why is always populated,
    because a decision a caller cannot see the reason for is one they will route around. Ownership
    is grant-derived (`access.gated_owner_map`), never a flag or the `private.` name."""
    from mantle.db import access
    layer, who = owner_of(a, access.gated_owner_map(store))
    if layer == "commons":
        return False, ("this is commons material (cited from %s): readable by everyone, "
                       "curated by no one" % who)
    if layer == "foundation":
        return False, "this is part of the ontology (%s), grounded at the foundation" % (
            a.get("content_type"))
    if str(who) != str(person):
        return False, "grounded at %s, not at you" % who
    return True, "yours"


def collections(store, *, person: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every collection, with what it holds and who holds it.

    Counts are computed by scanning membership rather than read off a counter, because a collection
    counter does not exist yet and inventing one that could silently drift is worse than a scan a
    human triggers."""
    from mantle.db import access
    gated_owners = access.gated_owner_map(store)          # grants scanned ONCE for the whole loop
    counts: Dict[str, int] = {}
    owners: Dict[str, Set[str]] = {}
    import json
    conn = store.artifacts.db.read()
    for (doc,) in conn.execute("SELECT doc FROM vertex"):
        try:
            d = json.loads(doc)
        except Exception:
            continue
        _layer, who = owner_of(d, gated_owners)
        for cid in _memberships(d):
            counts[cid] = counts.get(cid, 0) + 1
            owners.setdefault(cid, set()).add(str(who))
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for a in store.artifacts.list_artifacts(content_type=COLLECTION_CT):
        cid = str(a.get("id"))
        seen.add(cid)
        out.append({"id": cid, "name": a.get("title") or a.get("name") or cid,
                    "declared": True, "count": counts.get(cid, 0),
                    "owners": sorted(owners.get(cid, ()))[:6]})
    for cid, n in sorted(counts.items()):
        if cid not in seen:
            out.append({"id": cid, "name": cid, "declared": False, "count": n,
                        "owners": sorted(owners.get(cid, ()))[:6]})
    if person is not None:
        mine = "private.%s" % person
        out.sort(key=lambda c: (c["id"] != mine, c["id"]))
    return out


def members(store, collection_id: str, *, skip: int = 0, limit: int = 40) -> Dict[str, Any]:
    """What is in a collection, with each member's layer and holder."""
    from mantle.db import access
    gated_owners = access.gated_owner_map(store)
    import json
    conn = store.artifacts.db.read()
    hits, n = [], 0
    for (doc,) in conn.execute("SELECT doc FROM vertex"):
        try:
            d = json.loads(doc)
        except Exception:
            continue
        if collection_id not in _memberships(d):
            continue
        n += 1
        if n <= skip or len(hits) >= limit:
            continue
        layer, who = owner_of(d, gated_owners)
        hits.append({"id": d.get("id"), "content_type": d.get("content_type"),
                     "label": d.get("title") or (d.get("lemmas") or [None])[0] or d.get("id"),
                     "layer": layer, "owner": who})
    return {"collection": collection_id, "total": n, "skip": skip, "limit": limit, "items": hits}


def _announce(doc: Dict[str, Any], container_id: str, event_name: str) -> None:
    """Put a curation write on the artifact change feed.

    Curation is operator tooling with no caller inside the service layer, but the rows it writes
    are ordinary artifacts in the live store — a subscriber cannot tell which layer moved one, and
    a placement it is never told about is a view that stays wrong until something else refreshes
    it. So these announce themselves for the same reason the API writes do.

    The descriptor carries the container the event is addressed to rather than the doc's own
    `collection_id`, because a move has to reach a watcher of the collection it left as well as
    the one it arrived in. `content` is dropped: a stored doc holds it as ciphertext, and a
    placement change is not a content change.

    Best-effort, like the boundary emitter: curation must not fail because the feed did.
    """
    try:
        from mantle.db.doc_boundary import emit_artifact_change
        descriptor = {k: v for k, v in doc.items() if k not in ("content", "content_encrypted")}
        descriptor["collection_id"] = container_id
        emit_artifact_change(descriptor, event_name)
    except Exception:
        logger.debug("curation event emit failed (%s)", event_name, exc_info=True)


def file_into(store, artifact_id: str, collection_id: str, *, person: str,
              apply: bool = False) -> Dict[str, Any]:
    """Add an artifact to a collection. DRY RUN unless `apply=True`."""
    a = store.artifacts.get_artifact(artifact_id)
    if not a:
        return {"ok": False, "why": "no such artifact"}
    ok, why = may_curate(store, a, person)
    if not ok:
        return {"ok": False, "why": why}
    before = _memberships(a)
    if collection_id in before:
        return {"ok": True, "why": "already there", "memberships": before, "applied": False}
    after = before + [collection_id]
    if apply:
        d = dict(a)
        d["collections"] = after
        store.artifacts.put_artifact(d)
        # Addressed to the collection it arrived in, and named `created` there for the same
        # reason `workspace_service.move_artifact` does: to a watcher of that container this
        # artifact is new. A dry run announces nothing, because nothing happened.
        _announce(d, collection_id, _CREATED)
    return {"ok": True, "why": why, "memberships": after, "applied": bool(apply)}


def unfile(store, artifact_id: str, collection_id: str, *, person: str,
           apply: bool = False) -> Dict[str, Any]:
    """Remove an artifact FROM a collection. It is un-filed, never deleted.
    a tidy-up. Move it (`file_into` the new home first) or erase it — say which."""
    a = store.artifacts.get_artifact(artifact_id)
    if not a:
        return {"ok": False, "why": "no such artifact"}
    ok, why = may_curate(store, a, person)
    if not ok:
        return {"ok": False, "why": why}
    if str(a.get("collection_id") or "") == collection_id:
        return {"ok": False, "why": "that is this artifact's HOME collection — move it or erase it, "
                                    "do not leave it homeless"}
    after = [c for c in list(a.get("collections") or []) if c != collection_id]
    if apply:
        d = dict(a)
        d["collections"] = after
        store.artifacts.put_artifact(d)
        # The artifact is not deleted, but it has left this container, which is the only thing a
        # watcher of this container models — same verb the detach path in the services uses.
        _announce(d, collection_id, _DELETED)
    return {"ok": True, "why": why, "memberships": _memberships({**a, "collections": after}),
            "applied": bool(apply)}


def move(store, artifact_id: str, to_collection: str, *, person: str,
         apply: bool = False) -> Dict[str, Any]:
    """Change an artifact's HOME collection. One act, so it can never land homeless in between."""
    a = store.artifacts.get_artifact(artifact_id)
    if not a:
        return {"ok": False, "why": "no such artifact"}
    ok, why = may_curate(store, a, person)
    if not ok:
        return {"ok": False, "why": why}
    frm = str(a.get("collection_id") or "")
    after = [c for c in list(a.get("collections") or []) if c != frm]
    if to_collection not in after:
        after.append(to_collection)
    if apply:
        d = dict(a)
        d["collection_id"] = to_collection
        d["collections"] = after
        store.artifacts.put_artifact(d)
        # A move is a leave and an arrive, so it takes two events: one write, but two containers
        # whose watchers each learn only about their own side. The old container is announced
        # first — the order a subscriber holding both reads as a move rather than a duplicate.
        if frm:
            _announce(d, frm, _DELETED)
        _announce(d, to_collection, _CREATED)
    return {"ok": True, "why": why, "from": frm, "to": to_collection,
            "memberships": [to_collection] + [c for c in after if c != to_collection],
            "applied": bool(apply)}


__all__ = ["COLLECTION_CT", "owner_of", "may_curate", "collections", "members",
           "file_into", "unfile", "move"]
