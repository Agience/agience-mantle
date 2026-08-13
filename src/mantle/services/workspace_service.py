# services/workspace_service.py
#
# Unified Artifact Store service layer.
#
# In the unified model a "workspace" is just a Collection with
# `content_type == "application/vnd.agience.workspace+json"`. Artifacts live
# in a single `artifacts` table and carry a `state` in
# {draft, committed, archived}. Commit is a state flip; no data copies.
#
# Consumers still call `workspace_service.*` with `workspace_id` — that id
# is the collection_id. `db` and `store_db` / `workspace_db` / `collection_db`
# parameters are all the same the lattice handle.

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from mantle.db.store import Database
from fastapi import HTTPException, status

from mantle.entities.grant import Grant as GrantEntity, grant_is_allow, grant_is_deny
from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.entities.collection import (
    Collection as CollectionEntity,
    WORKSPACE_CONTENT_TYPE,
    COLLECTION_CONTENT_TYPE,
)

import mantle.db.backend as store
from mantle.db.backend import after_key, mid_key
import mantle.events.event_bus as event_bus

logger = logging.getLogger(__name__)


# Workspaces are collections with this content type.
WORKSPACE_MIME = WORKSPACE_CONTENT_TYPE

#: Largest UTF-8 payload kept as a duplicate INLINE copy alongside the object-storage write.
#: A storage/transfer bound on the row, not a judgement about the content: the authoritative copy
#: is in object storage at every size, so no value here can lose anything or change an outcome.
#: A different value would be right if the row-size budget it is really about were measured
#: rather than assumed — the store's per-row page/overflow cost is the thing that should set it.
_MAX_INLINE_BYTES = 128 * 1024


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_json_str(d: Optional[dict]) -> str:
    return json.dumps(d or {}, separators=(",", ":"), ensure_ascii=False)


def _safe_parse_context(context_json: Optional[str]) -> Dict[str, Any]:
    if not context_json:
        return {}
    try:
        parsed = json.loads(context_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _emit_event(collection_id: str, name: str, payload: dict, *, actor_id: Optional[str] = None) -> None:
    # artifact.created / artifact.updated are emitted at the write chokepoint
    # (`db.doc_boundary.emit_artifact_change`), so every write — raw or service —
    # is covered exactly once. This helper only forwards events the db path
    # doesn't (deletes carry the container context the db delete path lacks).
    if name in ("artifact.created", "artifact.updated"):
        return
    try:
        event_bus.emit_artifact_event_sync(collection_id, name, payload, actor_id=actor_id)
    except Exception:
        logger.debug("event bus emit failed", exc_info=True)


# ---------------------------------------------------------------------------
# Workspace (= collection) CRUD
# ---------------------------------------------------------------------------

def create_container(
    db: Database,
    user_id: str,
    content_type: Optional[str] = None,
    name: Optional[str] = None,
    context: str = "",
    content: str = "",
    description: Optional[str] = None,
    vector=None,
    artifact_id: Optional[str] = None,
) -> CollectionEntity:
    """Create a TOP-LEVEL container artifact (no parent) and grant the creator
    full CRUDEASIO.

    This is Mantle's direct *container primitive* — the one create path that does
    NOT require an existing ``container_id`` (unlike ``POST /artifacts``, which
    inserts a child into a container). A container is just an artifact other
    artifacts can be added to; ``content_type`` is the opaque label that an
    application uses to distinguish workspace / collection / domain containers.
    The gateway routes ``native`` container-create operations here (Phase 5).

    ``vector`` is a validated :class:`api.vectors.SuppliedVector` when the writer sent
    one — it rides this write into the semantic arm. ``None`` is the ordinary case and
    changes nothing.

    ``artifact_id`` pins the new artifact's id instead of minting a ``uuid4``. The only
    supplier is the router's ``identity`` path (`services/artifact_identity`), which derives it
    from the acting principal and a caller-chosen natural key — so it is already bound to this
    caller and cannot name another principal's artifact. It is NOT a general "let the client
    choose an id" seam: a raw client-chosen id would be exactly the collision the derivation
    exists to make unconstructable.
    """
    container_id = artifact_id or str(uuid.uuid4())
    entity = CollectionEntity(
        id=container_id,
        name=name,
        description=description,
        created_by=user_id,
        content_type=content_type or COLLECTION_CONTENT_TYPE,
        state=CollectionEntity.STATE_COMMITTED,
        context=context or "",
        content=content or "",
        created_time=_now_iso(),
        modified_time=_now_iso(),
    )
    store.create_collection(db, entity)

    # Issue explicit full-CRUDEASIO grant to the creator.
    store.upsert_user_collection_grant(
        db,
        user_id=user_id,
        collection_id=container_id,
        granted_by=user_id,
        can_create=True,
        can_read=True,
        can_update=True,
        can_delete=True,
        can_evict=True,
        can_invoke=True,
        can_add=True,
        can_share=True,
        can_admin=True,
    )

    # The oracle memoizes light-cone decisions; without this, the creator keeps being
    # DENIED content keys on their brand-new container until the TTL lapses.
    try:
        from mantle.search.mantle.wiring import invalidate_grant_cache
        invalidate_grant_cache(user_id)
    except Exception:
        logger.debug("grant-cache invalidation failed", exc_info=True)

    # Index AFTER grant + cache-invalidation so the descriptor's cell encryption can mint
    # the container's DEK on the first attempt (see the ordering note above).
    _index_container(db, entity, vector=vector)

    return entity


def _index_container(db: Database, entity: CollectionEntity, *, vector=None) -> None:
    """Index a top-level container artifact, carrying a writer-supplied vector if there is one.

    ``ensure_collection_descriptor`` is the ordinary path and stays the ordinary path. It
    takes no vector, so a container written WITH one calls the same pipeline entry point
    directly rather than through it — one extra argument, not a second indexing path.
    """
    if vector is None:
        from mantle.services.collection_service import ensure_collection_descriptor
        ensure_collection_descriptor(db, entity)
        return
    try:
        from mantle.search.ingest.pipeline_unified import index_artifact
        index_artifact(entity, entity.id, is_head=True, vector=vector)
    except Exception:
        logger.warning("Failed to index container artifact %s", entity.id, exc_info=True)


def create_workspace(
    db: Database,
    user_id: str,
    name: str,
) -> CollectionEntity:
    """Create a new workspace (a top-level container with content_type=workspace).

    Thin convenience wrapper over :func:`create_container` — workspaces are
    regular container artifacts, each with a fresh UUID and no ID-pinning.
    """
    return create_container(
        db, user_id, content_type=WORKSPACE_CONTENT_TYPE, name=name,
    )


def list_workspaces(db: Database, user_id: str) -> List[CollectionEntity]:
    """Every workspace inside the user's read light-cone — the containers carrying the workspace
    content type that any active grant reaches.

    Reachability, not authorship: a workspace shared with the user appears here, and one they
    created but no longer hold a grant on does not.
    """
    return store.get_collections_by_owner_and_type(db, user_id, WORKSPACE_CONTENT_TYPE)


def get_workspace(
    db: Database,
    user_id: str,
    workspace_id: str,
    required: str = "read",
) -> CollectionEntity:
    """Load a workspace, authorizing the caller for *required* (a CRUDEASIO verb).

    Deny grants are honoured: an explicit deny on the required verb wins over any allow.

    Always a 404 (never 403) on refusal — not leaking existence is deliberate.
    """
    entity = store.get_collection_by_id(db, workspace_id)
    if not entity:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workspace not found")

    flag = f"can_{required}"
    grants = store.get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=workspace_id
    )

    # An explicit deny on this verb wins over any allow.
    for g in grants:
        if grant_is_deny(g) and getattr(g, flag, False):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Workspace not found")

    if not any(grant_is_allow(g) and getattr(g, flag, False) for g in grants):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Workspace not found")
    return entity


def get_workspace_unsafe(db: Database, workspace_id: str) -> Optional[CollectionEntity]:
    """The container entity for ``workspace_id`` with NO access check, or ``None`` if absent.

    ``unsafe`` is the whole contract: nothing here consults grants, so this belongs only on
    paths that have already authorized the caller. :func:`get_workspace` is the checked read.
    """
    return store.get_collection_by_id(db, workspace_id)


def update_workspace(
    db: Database,
    user_id: str,
    workspace_id: str,
    name: Optional[str],
    description: Optional[str],
    context: Optional[str] = None,
    vector=None,
    content: Optional[str] = None,
    content_type: Optional[str] = None,
    state: Optional[str] = None,
) -> CollectionEntity:
    """Update a top-level artifact's name, description, context, content and/or content_type;
    return the current entity.

    ``None`` fields are left alone, and the store write plus reindex happen only if something
    actually moved — except that supplying ``vector`` is itself a change, since the writer is
    restating what the container means even when no scalar field differs.

    ``context`` replaces the stored context wholesale (see :func:`update_workspace_context`), and
    a string is parsed as JSON with an unparseable value read as ``{}``. Requires ``update``;
    raises ``HTTPException(404)`` for a missing workspace or a caller without that verb.

    ``content`` and ``content_type`` are here because a TOP-LEVEL ARTIFACT IS NOT ALWAYS A
    CONTAINER. :func:`create_container` is the only create path that takes no parent, so every
    artifact written without a ``container_id`` — a note, a transcript, a file a hook captured —
    is created through it WITH content (see its ``content`` parameter), and lands here on the way
    back out. Without these two, `PATCH /artifacts/{id}` on such an artifact accepted a new body,
    returned 200, and silently discarded it: the router's top-level branch had nothing to pass
    the content to. That made every rewrite a fresh artifact instead of a new version of one, so
    the store accumulated duplicate copies of the same thing with no way to tell which was
    current — the failure `tests/test_workspace_service.py::test_update_workspace_replaces_content`
    now pins. Stored on the entity directly, exactly as ``create_container`` stores it, and
    reindexed through the same ``_index_container`` call the other fields use.

    ``state`` archives and unarchives, and is here for the same reason ``content`` is: without
    it the router's top-level branch had nothing to pass the caller's ``state`` to, so
    ``PATCH /artifacts/{id}`` with ``state: "archived"`` returned 200 and changed nothing. That
    left the archive mechanism unreachable for every artifact created without a ``container_id``
    — which is every note, transcript and captured file — so superseded copies could only be
    deleted, never retired. Pinned by
    ``tests/test_workspace_service.py::test_update_workspace_archives_a_top_level_artifact``.

    **Unarchiving returns a top-level artifact to COMMITTED, not to draft**, which is where it
    differs from :func:`update_artifact` deliberately. A collection member unarchives to draft
    because there is a commit path waiting for it — that same function's ``state == committed``
    branch. A top-level artifact has none: :func:`create_container` writes ``committed``
    directly and no path here ever promotes a draft, so unarchiving one into draft would move it
    into a segment the default recall does not search and nothing could ever move it out of.
    """
    ws = get_workspace(db, user_id, workspace_id, required="update")

    # State transitions settle the artifact's segment and return; they never combine with a
    # content edit in one call, because the two answer different questions ("where does this
    # live" vs. "what does it say") and the index move has to be the whole of the write.
    if state == CollectionEntity.STATE_ARCHIVED and ws.state != CollectionEntity.STATE_ARCHIVED:
        ws.state = CollectionEntity.STATE_ARCHIVED
        ws.modified_by = user_id
        ws.modified_time = _now_iso()
        store.update_collection(db, ws)
        _reindex_after_state_change(ws, user_id, vacate=["committed", "draft"])
        return ws

    if ws.state == CollectionEntity.STATE_ARCHIVED:
        if state and state != CollectionEntity.STATE_ARCHIVED:
            ws.state = CollectionEntity.STATE_COMMITTED
            ws.modified_by = user_id
            ws.modified_time = _now_iso()
            store.update_collection(db, ws)
            _reindex_after_state_change(ws, user_id, vacate=["archived"])
            return ws
        # Editing an archived artifact is refused rather than silently reviving it — the same
        # 409 `update_artifact` raises, for the same reason: an archived artifact has been
        # retired, and a write that quietly un-retired it would defeat the retirement.
        raise HTTPException(status.HTTP_409_CONFLICT, "Cannot edit an archived artifact")

    changed = False
    if name is not None and name != ws.name:
        ws.name = name
        changed = True
    if description is not None and description != ws.description:
        ws.description = description
        changed = True
    if content is not None and content != ws.content:
        ws.content = content
        changed = True
    if content_type is not None and content_type != ws.content_type:
        ws.content_type = content_type
        changed = True
    # A re-supplied vector is itself a change worth reindexing for: the writer is saying
    # the container's meaning moved, even when none of its scalar fields did.
    if changed or vector is not None:
        ws.modified_time = _now_iso()
        store.update_collection(db, ws)
        _index_container(db, ws, vector=vector)

    if context is not None:
        parsed = _safe_parse_context(context) if isinstance(context, str) else context
        update_workspace_context(db, user_id, workspace_id, parsed)
        ws = get_workspace(db, user_id, workspace_id, required="update")

    return ws


def _delete_or_detach_members(
    db: Database, container_id: str, user_id: str, rows: list, *, cascade: bool,
) -> None:
    """The shared body of "what happens to a container's members when the container goes."

    ``cascade=False`` (the default everywhere this is called) DETACHES: each member is evicted
    from ``container_id`` — its edge to this container is dropped — and is otherwise untouched,
    exactly as :func:`remove_artifact_from_container` leaves it. Nothing is destroyed, so a
    member that was reachable only through this container becomes unreachable through it but
    still exists, findable by anyone still holding its id or a direct grant on it — recoverable,
    the way `rmdir` refusing a non-empty directory is recoverable and `rm -r` is not.

    ``cascade=True`` is the old unconditional behaviour, preserved verbatim: a member still
    linked into another container is evicted from this one only (the destroy would reach past
    what the caller has rights over); a member with nowhere else to be reached from is destroyed
    outright — its search index and object-storage content dropped along with the row. This is
    the only branch that can lose data, and only because the caller asked for exactly that.

    One event per root either way, because a watcher of this container sees the same thing in
    both cases: the artifact is no longer in it. Which of "evicted" and "destroyed" happened is
    not this event's business — `tests/test_scoped_deletion_and_urls.py` is where that is
    decided and proven, not announced.
    """
    for row in rows:
        art_id = row.get("id")
        root_id = row.get("root_id") or art_id
        if not art_id:
            continue

        if cascade:
            try:
                from mantle.search.ingest.pipeline_unified import delete_artifact_from_index
                delete_artifact_from_index(art_id, root_id, collection_id=container_id)
            except Exception:
                logger.debug("index delete failed", exc_info=True)

            shared_elsewhere = store.count_other_containers_for_root(
                db, root_id, container_id
            ) > 0
            if shared_elsewhere:
                # Evict from THIS container only; the artifact survives elsewhere.
                store.remove_artifact_from_collection(db, container_id, root_id)
            else:
                store.delete_artifacts_by_root(db, root_id)
                store.remove_all_edges_for_root(db, root_id)
        else:
            store.remove_artifact_from_collection(db, container_id, root_id)

        _emit_event(container_id, "artifact.deleted", {"artifact_id": root_id},
                    actor_id=user_id)


def delete_workspace(
    db: Database, user_id: str, workspace_id: str, *, cascade: bool = False,
) -> None:
    """Delete a workspace. Its members are DETACHED, not destroyed, unless ``cascade=True``.

    ``cascade=False`` (the default): each member is evicted from this workspace — its edge
    dropped — and otherwise left exactly as it was; nothing under it is deleted. ``cascade=True``
    recurses the old way: a member linked into no other container is destroyed outright (index,
    content, row); one still reachable elsewhere is evicted from this container only. See
    :func:`_delete_or_detach_members` for the shared logic either branch runs.

    Announced per root and then once for the workspace itself. A single event for the container
    would not be enough: a subscriber holds derived state keyed on artifact ids, and told only
    that the container went it has no way to name the ids it must drop. Emitted here rather than
    at the write boundary for the reason every delete is — by the time the db layer has an id the
    doc naming the container is gone, and the container is what the event is addressed to.

    Each emit follows the write it announces, so nothing is announced that did not happen, and
    `_emit_event` swallows its own failures — a feed that cannot be written to must not leave a
    workspace half-deleted."""
    get_workspace(db, user_id, workspace_id, required="delete")

    rows = store.list_collection_artifacts(db, workspace_id, include_archived=True)
    _delete_or_detach_members(db, workspace_id, user_id, rows, cascade=cascade)

    store.delete_collection(db, workspace_id)
    # The container itself. A subscriber watching it learns the subscription's subject is gone,
    # rather than going quiet and looking merely idle.
    _emit_event(workspace_id, "artifact.deleted", {"artifact_id": workspace_id},
                actor_id=user_id)


def get_workspace_context(db: Database, user_id: str, workspace_id: str) -> dict:
    """The workspace's parsed context, always carrying a ``collections`` list (empty if unset).

    A context that is absent or not valid JSON reads as ``{}`` rather than raising — context is
    caller-authored and a malformed one must not make the workspace unreadable. Mutating the
    result persists nothing; :func:`update_workspace_context` is what writes it back. Requires
    ``read``; raises ``HTTPException(404)`` otherwise.
    """
    ws = get_workspace(db, user_id, workspace_id, required="read")
    parsed = _safe_parse_context(ws.context) if isinstance(ws.context, str) else (ws.context or {})
    parsed.setdefault("collections", [])
    return parsed


def update_workspace_context(
    db: Database, user_id: str, workspace_id: str, context: dict
) -> dict:
    """Replace the workspace's context with ``context`` and return what was stored.

    Whole-document replacement, NOT a merge: any key absent from ``context`` is dropped, so a
    caller doing a partial edit must read the current context first and pass it back complete.
    A non-dict argument is coerced to ``{}``, and a ``collections`` list is always present in
    what is stored. Requires ``update``; raises ``HTTPException(404)`` otherwise.
    """
    ws = get_workspace(db, user_id, workspace_id, required="update")
    if not isinstance(context, dict):
        context = {}
    context.setdefault("collections", [])
    ws.context = json.dumps(context)
    ws.modified_time = _now_iso()
    store.update_collection(db, ws)
    return context


def apply_workspace_card_actions(
    db: Database, user_id: str, workspace_id: str, actions: List[dict]
) -> dict:
    """Apply ``attach_collection`` actions to the workspace context and return the stored context.

    Attaching a collection already listed updates its ``mode`` (``own`` / ``shared``) in place
    rather than appending a duplicate, which makes the call idempotent. Entries that are not
    dicts, carry an unrecognised ``type``, or omit ``collection_id`` are skipped in silence — a
    single bad action does not reject the batch. An unrecognised ``mode`` falls back to ``own``.

    Requires ``read`` and ``update``; raises ``HTTPException(404)`` otherwise.
    """
    current = get_workspace_context(db, user_id, workspace_id)
    coll_by_id = {c.get("collection_id"): c for c in current.get("collections", []) if isinstance(c, dict)}
    for act in actions or []:
        if not isinstance(act, dict):
            continue
        if act.get("type") == "attach_collection":
            cid = act.get("collection_id")
            if not cid:
                continue
            existing = coll_by_id.get(cid)
            if existing:
                mode = act.get("mode")
                if mode in ("own", "shared"):
                    existing["mode"] = mode
            else:
                item = {
                    "collection_id": cid,
                    "mode": act.get("mode") if act.get("mode") in ("own", "shared") else "own",
                }
                current["collections"].append(item)
                coll_by_id[cid] = item
    return update_workspace_context(db, user_id, workspace_id, current)


# ---------------------------------------------------------------------------
# Workspace Bindings — Cascade Resolution
# ---------------------------------------------------------------------------

def _extract_bindings(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Read the ``bindings`` dict from a parsed context, returning ``{}`` if absent or invalid."""
    if not context or not isinstance(context, dict):
        return {}
    bindings = context.get("bindings")
    if not isinstance(bindings, dict):
        return {}
    return bindings


def _user_can_read_collection(
    db: Database, user_id: str, collection_id: str
) -> bool:
    """Return ``True`` if *user_id* has at least read access to *collection_id*."""
    col = store.get_collection_by_id(db, collection_id)
    if not col:
        return False
    grants = store.get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=collection_id,
    )
    return any(getattr(g, "can_read", False) for g in grants)


def _resolve_binding_from(
    bindings: Dict[str, Any], role: str,
) -> Optional[str]:
    """Extract the bound artifact id for *role* from a bindings dict, or ``None``."""
    entry = bindings.get(role)
    if isinstance(entry, dict):
        aid = entry.get("artifact_id")
        if isinstance(aid, str) and aid:
            return aid
    return None


def _resolve_binding_multi_from(
    bindings: Dict[str, Any], role: str,
) -> List[str]:
    """Extract a list of bound artifact ids for a multi-valued *role*."""
    entry = bindings.get(role)
    if isinstance(entry, dict):
        ids = entry.get("artifact_ids")
        if isinstance(ids, list):
            return [i for i in ids if isinstance(i, str) and i]
    return []


def resolve_binding(
    db: Database,
    user_id: str,
    workspace_id: str,
    role: str,
    *,
    transform_context: Optional[Dict[str, Any]] = None,
    step_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Resolve a binding role through the cascade: step → transform → workspace.

    Returns the artifact id of the first accessible binding found, or
    ``None`` if the role is unbound at every level.
    """
    # 1. Step-level
    cid = _resolve_binding_from(_extract_bindings(step_context), role)
    if cid and _user_can_read_collection(db, user_id, cid):
        return cid

    # 2. Transform-level
    cid = _resolve_binding_from(_extract_bindings(transform_context), role)
    if cid and _user_can_read_collection(db, user_id, cid):
        return cid

    # 3. Workspace-level
    ws_context = get_workspace_context(db, user_id, workspace_id)
    cid = _resolve_binding_from(_extract_bindings(ws_context), role)
    if cid and _user_can_read_collection(db, user_id, cid):
        return cid

    # 4. Platform defaults — not implemented in Phase 1
    return None


def resolve_all_bindings(
    db: Database,
    user_id: str,
    workspace_id: str,
    *,
    transform_context: Optional[Dict[str, Any]] = None,
    step_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Resolve all binding roles through the cascade.

    Returns ``{role: artifact_id}`` for every role that resolves to an
    accessible artifact.  Roles where the user lacks access are omitted.
    """
    ws_context = get_workspace_context(db, user_id, workspace_id)

    # Collect all role names across cascade levels.
    all_roles: set[str] = set()
    all_roles.update(_extract_bindings(ws_context).keys())
    all_roles.update(_extract_bindings(transform_context).keys())
    all_roles.update(_extract_bindings(step_context).keys())

    result: Dict[str, str] = {}
    for role in all_roles:
        # Inline cascade to avoid re-reading workspace context per role.
        cid = _resolve_binding_from(_extract_bindings(step_context), role)
        if cid and _user_can_read_collection(db, user_id, cid):
            result[role] = cid
            continue

        cid = _resolve_binding_from(_extract_bindings(transform_context), role)
        if cid and _user_can_read_collection(db, user_id, cid):
            result[role] = cid
            continue

        cid = _resolve_binding_from(_extract_bindings(ws_context), role)
        if cid and _user_can_read_collection(db, user_id, cid):
            result[role] = cid

    return result


# ---------------------------------------------------------------------------
# Workspace Bindings — Multi-valued Resolution
# ---------------------------------------------------------------------------

# Known binding roles and their cardinality.
SINGLE_BINDING_ROLES = {"memory", "tools", "resources", "ask_prompt", "engagement_channels"}
MULTI_BINDING_ROLES = {"target_collections"}
KNOWN_BINDING_ROLES = SINGLE_BINDING_ROLES | MULTI_BINDING_ROLES


def resolve_binding_multi(
    db: Database,
    user_id: str,
    workspace_id: str,
    role: str,
    *,
    transform_context: Optional[Dict[str, Any]] = None,
    step_context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Resolve a multi-valued binding role through the cascade.

    Returns a list of artifact ids for which the user has read access.
    """
    for ctx in (step_context, transform_context):
        ids = _resolve_binding_multi_from(_extract_bindings(ctx), role)
        accessible = [i for i in ids if _user_can_read_collection(db, user_id, i)]
        if accessible:
            return accessible

    ws_context = get_workspace_context(db, user_id, workspace_id)
    ids = _resolve_binding_multi_from(_extract_bindings(ws_context), role)
    return [i for i in ids if _user_can_read_collection(db, user_id, i)]


# ---------------------------------------------------------------------------
# Workspace Bindings — Set / Clear
# ---------------------------------------------------------------------------

def set_binding(
    db: Database,
    user_id: str,
    workspace_id: str,
    role: str,
    *,
    artifact_id: Optional[str] = None,
    artifact_ids: Optional[List[str]] = None,
) -> dict:
    """Set a workspace binding for *role*.

    Single-valued roles accept ``artifact_id``; multi-valued roles accept
    ``artifact_ids``.  Raises ``ValueError`` on unknown role or cardinality
    mismatch.  Raises ``HTTPException(403)`` if the caller cannot update.
    """
    if role not in KNOWN_BINDING_ROLES:
        raise ValueError(f"Unknown binding role: {role}")

    if role in MULTI_BINDING_ROLES:
        if artifact_id is not None or artifact_ids is None:
            raise ValueError(f"Multi-valued role '{role}' requires artifact_ids, not artifact_id")
        value: Dict[str, Any] = {"artifact_ids": artifact_ids}
    else:
        if artifact_ids is not None or artifact_id is None:
            raise ValueError(f"Single-valued role '{role}' requires artifact_id, not artifact_ids")
        value = {"artifact_id": artifact_id}

    ctx = get_workspace_context(db, user_id, workspace_id)
    bindings = ctx.setdefault("bindings", {})
    bindings[role] = value
    update_workspace_context(db, user_id, workspace_id, ctx)

    event_bus.emit_artifact_event_sync(
        workspace_id,
        "workspace.binding.set",
        {"workspace_id": workspace_id, "role": role, "binding": value},
        actor_id=user_id,
    )
    return value


def clear_binding(
    db: Database,
    user_id: str,
    workspace_id: str,
    role: str,
) -> None:
    """Remove the binding for *role* from the workspace context."""
    ctx = get_workspace_context(db, user_id, workspace_id)
    bindings = ctx.get("bindings")
    if isinstance(bindings, dict) and role in bindings:
        del bindings[role]
        update_workspace_context(db, user_id, workspace_id, ctx)

    event_bus.emit_artifact_event_sync(
        workspace_id,
        "workspace.binding.cleared",
        {"workspace_id": workspace_id, "role": role},
        actor_id=user_id,
    )


# ---------------------------------------------------------------------------
# Artifact CRUD (collection-scoped)
# ---------------------------------------------------------------------------

def list_workspace_artifacts(
    db: Database,
    user_id: str,
    workspace_id: str,
) -> List[ArtifactEntity]:
    """The workspace's current members, ordered by ``order_key``, archived ones omitted.

    Edge-resolved: each membership edge contributes the current version of its root, which for a
    root linked in from elsewhere is a version whose own ``collection_id`` is another container.
    Requires ``read``; raises ``HTTPException(404)`` otherwise.
    """
    get_workspace(db, user_id, workspace_id, required="read")
    rows = store.list_collection_artifacts(db, workspace_id)
    return [ArtifactEntity.from_dict(r) for r in rows]


def get_workspace_artifact(
    db: Database,
    user_id: str,
    workspace_id: str,
    artifact_id: str,
) -> ArtifactEntity:
    """One artifact VERSION by id, confirmed to be homed in this workspace.

    ``artifact_id`` is a version id, not a root id, and the version's own ``collection_id`` must
    equal ``workspace_id`` — a root merely linked into the workspace by a membership edge does
    NOT resolve here, because this is the read the write paths authorize against.

    Requires ``read``. Raises ``HTTPException(404)`` for a missing workspace, a caller without
    read, a missing artifact, or one homed in a different container — all the same answer, so a
    refusal does not disclose which.
    """
    get_workspace(db, user_id, workspace_id, required="read")
    artifact = store.get_artifact(db, artifact_id)
    if not artifact or artifact.collection_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found")
    return artifact


def get_artifact_unsafe_by_id(db: Database, artifact_id: str) -> Optional[ArtifactEntity]:
    """One artifact version by id with NO access check and no container check, or ``None``.

    ``unsafe`` is the whole contract: use :func:`get_workspace_artifact` on any path a caller
    can reach.
    """
    return store.get_artifact(db, artifact_id)


def get_workspace_artifacts_by_ids(
    db: Database,
    user_id: str,
    workspace_id: str,
    artifact_ids: List[str],
) -> List[ArtifactEntity]:
    """The named artifact VERSIONS that are homed in ``workspace_id``.

    Ids naming a missing artifact, or one homed in another container, are skipped, so the result
    may be shorter than the input. Repeated ids are read once and yielded once. Archived versions
    ARE included — the filter is containment, not state.

    Deliberately not named ``*_batch``: the store publishes no multi-get, so this is one point
    read per distinct id and there is no shared work to collapse. A batch name would invite
    callers to hand it ids by the thousand, which is the one thing this shape cannot absorb.
    Access is checked once for the workspace, not per artifact — every id that survives the
    containment filter is in the workspace the caller was just authorized for.

    Requires ``read``; raises ``HTTPException(404)`` otherwise.
    """
    get_workspace(db, user_id, workspace_id, required="read")
    out: List[ArtifactEntity] = []
    for aid in dict.fromkeys(artifact_ids or []):
        a = store.get_artifact(db, aid)
        if a and a.collection_id == workspace_id:
            out.append(a)
    return out


def get_workspace_artifacts_by_ids_global(
    db: Database,
    user_id: str,
    artifact_ids: List[str],
) -> List[ArtifactEntity]:
    """The named artifact VERSIONS homed in ANY workspace the user's grants reach.

    The reachable set is resolved once (see :func:`list_workspaces`) and each artifact's home is
    matched against it in memory, so the cost is one point read per distinct id rather than one
    per (id, workspace) pair. Ids naming a missing artifact, or one homed outside that set, are
    skipped; repeated ids are read once and yielded once.

    Not named ``*_batch`` for the same reason as :func:`get_workspace_artifacts_by_ids`: the
    per-id read is the store's floor here and the name should not promise otherwise.
    """
    workspaces = {w.id for w in list_workspaces(db, user_id)}
    out: List[ArtifactEntity] = []
    for aid in dict.fromkeys(artifact_ids or []):
        a = store.get_artifact(db, aid)
        if a and a.collection_id in workspaces:
            out.append(a)
    return out


def _safe_content_key(ctx: dict, artifact_id: str, *, default_prefix: str = "artifacts") -> str:
    """The object-store key for `artifact_id`, refusing any caller-supplied key that names a
    DIFFERENT artifact.

      • WRITE — create an artifact in YOUR OWN container with
        `context = {"content_key": "<victim>/<their-artifact>.content"}`; `_store_content_in_s3`
        then calls `put_text_direct` on the victim's key, overwriting their object with bytes
        encrypted under YOUR master key. Their next read fails to decrypt. Unrecoverable.
      • DELETE — the same trick with a throwaway artifact: `delete_object(content_key)` removes
        the victim's object from BOTH the edge and durable buckets.
      • READ — a signed URL, or the content endpoint, minted for someone else's key.

    The binding: a legitimate key always ends `/{artifact_id}.content` — the server derives either
    `artifacts/{id}.content` (here) or `{tenant}/{id}.content` (`initiate_upload_and_create_artifact`).
    Tying the key to ITS OWN artifact id therefore accepts every legitimately-derived key, including
    every one already stored, and rejects exactly the cross-artifact case. A stored key that fails
    the check is discarded in favour of the derived one rather than honoured."""
    derived = f"{default_prefix}/{artifact_id}.content"
    claimed = ctx.get("content_key")
    if not claimed or not isinstance(claimed, str):
        return derived
    if claimed == derived or claimed.endswith(f"/{artifact_id}.content"):
        return claimed
    logger.warning(
        "refusing caller-supplied content_key %r on artifact %s — it names a different artifact; "
        "using the derived key instead", claimed, artifact_id,
    )
    return derived


def _store_content_in_s3(
    artifact_id: str,
    content: str,
    context_str: str,
    owner_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Upload inline content to S3 and return (content_key, inline_content).

    For small text (< 128 KB), the content is also kept inline in the lattice as a
    fallback so the artifact remains readable even if S3 is temporarily
    unreachable.  Large content (>= 128 KB) is cleared from inline to keep the
    document store lean.

    Derives the content_type from the artifact context if present.
    Falls back to text/plain. Idempotent — safe to call on every create/update.
    """
    from mantle.services.content_service import put_text_direct

    try:
        ctx = json.loads(context_str) if context_str else {}
    except (json.JSONDecodeError, TypeError):
        ctx = {}

    content_type = ctx.get("content_type") or "text/plain"
    content_key = _safe_content_key(ctx, artifact_id)

    try:
        put_text_direct(content_key, content, content_type, owner_id=owner_id)
    except Exception:
        logger.warning("Failed to upload artifact content to S3 for %s — keeping inline", artifact_id, exc_info=True)
        return content_key, content  # degrade gracefully: keep inline

    # Keep small text inline as a fallback; clear inline for large content. The content is ALREADY
    # in object storage at this point either way — this only decides whether a duplicate copy also
    # rides along in the row, so nothing is lost at any value and no measurement is being judged.
    if len(content.encode("utf-8")) <= _MAX_INLINE_BYTES:
        return content_key, content
    return content_key, ""


def _link_to_target_collections(
    db: Database,
    user_id: str,
    workspace_id: str,
    artifact: ArtifactEntity,
) -> None:
    """Create ``collection_artifacts`` edges to each target collection bound to the workspace.

    Skips silently (logs at INFO) for any target the caller cannot add to.
    """
    target_ids = resolve_binding_multi(db, user_id, workspace_id, "target_collections")
    for target_id in target_ids:
        try:
            # Check the user has at least read (access proxy for "can_add") on the target.
            if not _user_can_read_collection(db, user_id, target_id):
                logger.info(
                    "Skipping target_collection link: user %s lacks access to %s",
                    user_id, target_id,
                )
                continue
            store.add_artifact_to_collection(db, target_id, artifact.root_id)
            # Addressed to the target collection, not the workspace: a subscriber watching the
            # collection the artifact just entered is the one that needs to hear about it.
            event_bus.emit_artifact_event_sync(
                target_id,
                "workspace.target_collection.linked",
                {
                    "workspace_id": workspace_id,
                    "target_collection_id": target_id,
                    "artifact_id": artifact.id,
                },
                actor_id=user_id,
            )
        except Exception:
            logger.info(
                "Failed to link artifact %s to target_collection %s",
                artifact.id, target_id, exc_info=True,
            )


def create_workspace_artifact(
    db: Database,
    user_id: str,
    workspace_id: str,
    context: str,
    content: str,
    root_id: Optional[str] = None,
    order_key: Optional[str] = None,
    enqueue_index: bool = True,
    name: Optional[str] = None,
    content_type: Optional[str] = None,
    index: Optional[str] = None,
    vector=None,
) -> ArtifactEntity:
    """Create an artifact inside a container.

    ``vector`` is a validated :class:`api.vectors.SuppliedVector` when the writer sent one;
    it rides the same index job as the rest of the write. Mantle produces no vector of its
    own, so ``None`` means the vector arm receives nothing for this artifact.
    """
    get_workspace(db, user_id, workspace_id, required="create")

    now = _now_iso()
    artifact_id = str(uuid.uuid4())

    # Store content in S3; update context with content_key.
    resolved_content = content
    if content:
        content_key, resolved_content = _store_content_in_s3(artifact_id, content, context, owner_id=user_id)
        # Inject content_key into context JSON if not already present.
        try:
            ctx_obj = json.loads(context) if context else {}
        except (json.JSONDecodeError, TypeError):
            ctx_obj = {}
        if "content_key" not in ctx_obj:
            ctx_obj["content_key"] = content_key
            context = json.dumps(ctx_obj)

    artifact = ArtifactEntity(
        id=artifact_id,
        root_id=root_id or artifact_id,
        collection_id=workspace_id,
        context=context,
        content=resolved_content,
        state=ArtifactEntity.STATE_DRAFT,
        created_by=user_id,
        modified_by=user_id,
        created_time=now,
        modified_time=now,
        name=name,
        content_type=content_type,
    )
    store.create_artifact(db, artifact)

    # Insert the stable collection_artifacts edge pointing at root_id.
    if order_key is None:
        order_key = after_key(store.get_last_order_key(db, workspace_id))
    store.add_artifact_to_collection(db, workspace_id, artifact.root_id, order_key)

    # Wire target_collections binding: create collection_artifacts edges to
    # each target collection so the draft artifact is associated immediately.
    _link_to_target_collections(db, user_id, workspace_id, artifact)

    # Direct owner grant: the creator owns the artifact explicitly (full CRUDEASIO),
    # not by virtue of edge propagation from its container. Access is a grant on the
    # artifact, so it survives the artifact being evicted from the collection — a
    # collection is just an artifact with child edges. (The container edge still
    # propagates the collection's grants to OTHER members for sharing; see `origin`.)
    try:
        store.upsert_user_collection_grant(
            db, user_id=user_id, collection_id=artifact.root_id, granted_by=user_id,
            can_create=True, can_read=True, can_update=True, can_delete=True,
            can_evict=True, can_invoke=True, can_add=True, can_share=True, can_admin=True,
        )
    except Exception:
        logger.debug("owner grant on create failed", exc_info=True)

    # WHERE index: eager (default) or deferred to first access (lazy). A latent
    # write skips indexing and carries no materialization marker (WHO+WHEN only).
    from mantle.search.lazy import resolve_lazy
    if enqueue_index and not resolve_lazy(index):
        try:
            from mantle.search.ingest.pipeline_unified import enqueue_index_artifact
            enqueue_index_artifact(
                artifact, artifact.collection_id, tenant_id=user_id, vector=vector,
            )
            store.mark_materialized(db, artifact.id)
        except Exception:
            logger.debug("index enqueue failed", exc_info=True)

    _emit_event(workspace_id, "artifact.created", {"artifact": artifact.to_dict()}, actor_id=user_id)
    return artifact


def materialize_on_access(db, *, artifact_id, collection_id, tenant_id=None, artifact=None) -> None:
    """Lazy indexing: materialize a latent vertex on first authorized access —
    enqueue its WHERE index and mark it. No-op when lazy indexing is off or the
    vertex is already materialized. Best-effort; never raises into the read path."""
    from mantle.search.lazy import lazy_index_default
    if not lazy_index_default():
        return
    try:
        if not artifact_id or not collection_id or store.is_materialized(db, artifact_id):
            return
        if artifact is None:
            raw = store.get_artifact(db, artifact_id)
            if not raw:
                return
            artifact = ArtifactEntity.from_dict(raw) if isinstance(raw, dict) else raw
        from mantle.search.ingest.pipeline_unified import enqueue_index_artifact
        enqueue_index_artifact(artifact, collection_id, is_head=True, tenant_id=tenant_id)
        store.mark_materialized(db, artifact_id)
    except Exception:
        logger.warning("materialize-on-access failed for %s", artifact_id, exc_info=True)


def warm_collection(db, collection_id, *, tenant_id=None) -> int:
    """Warm-sweep guardrail: materialize every latent artifact in a collection so
    it is searchable up front (for corpora that must not wait for first access).
    Returns the count newly materialized. Idempotent — safe to re-run."""
    n = 0
    seen: set = set()
    try:
        from mantle.search.ingest.pipeline_unified import enqueue_index_artifact
        # Members regardless of state: committed (dicts, drafts of this workspace
        # made visible via draft_workspace_id) plus the workspace's own drafts.
        items = []
        for raw in (store.list_collection_artifacts(db, collection_id, draft_workspace_id=collection_id) or []):
            items.append(ArtifactEntity.from_dict(raw) if isinstance(raw, dict) else raw)
        items.extend(store.list_draft_artifacts(db, collection_id) or [])
        for art in items:
            aid = getattr(art, "id", None)
            if not aid or aid in seen:
                continue
            seen.add(aid)
            if store.is_materialized(db, aid):
                continue
            try:
                enqueue_index_artifact(art, collection_id, is_head=True, tenant_id=tenant_id)
                store.mark_materialized(db, aid)
                n += 1
            except Exception:
                logger.warning("warm_collection: materialize %s failed", aid, exc_info=True)
    except Exception:
        logger.warning("warm_collection(%s) failed", collection_id, exc_info=True)
    return n


def create_workspace_artifacts_bulk(
    db: Database,
    user_id: str,
    workspace_id: str,
    items: Sequence[Union[Tuple[str, str], Tuple[str, str, Optional[Sequence[str]]], Dict[str, Any]]],
) -> List[ArtifactEntity]:
    """Create several artifacts in one workspace, returned in input order.

    Each item is a mapping with ``context``/``content`` keys or a sequence of the two; non-string
    values are JSON-encoded first. Access is checked once up front, then every item goes through
    :func:`create_workspace_artifact` — so this is a convenience loop over the ordinary create
    path, not a bulk store write, and it is NOT atomic: a failure part-way leaves the artifacts
    already created in place.

    Raises ``ValueError`` for an item that is neither a mapping nor a sequence, and
    ``HTTPException(404)`` if the caller lacks ``create`` on the workspace.
    """
    get_workspace(db, user_id, workspace_id, required="create")

    out: List[ArtifactEntity] = []
    for raw in items:
        if isinstance(raw, dict):
            context_val = raw.get("context", "")
            content_val = raw.get("content", "")
        elif isinstance(raw, (list, tuple)):
            context_val = raw[0] if len(raw) >= 1 else ""
            content_val = raw[1] if len(raw) >= 2 else ""
        else:
            raise ValueError("Bulk item must be mapping or tuple")

        if not isinstance(context_val, str):
            context_val = json.dumps(context_val or {})
        if not isinstance(content_val, str):
            content_val = json.dumps(content_val)

        out.append(
            create_workspace_artifact(
                db,
                user_id,
                workspace_id,
                context=context_val,
                content=content_val,
                enqueue_index=True,
            )
        )
    return out


def _ensure_draft(
    db: Database, user_id: str, committed: ArtifactEntity
) -> ArtifactEntity:
    """
    Edit-after-commit: create a new draft record with the same root_id
    containing a copy of the committed content, leaving the committed
    version untouched.
    """
    existing_draft = store.get_draft_artifact(db, committed.root_id, committed.collection_id)
    if existing_draft:
        return existing_draft

    now = _now_iso()
    new_id = str(uuid.uuid4())

    draft = ArtifactEntity(
        id=new_id,
        root_id=committed.root_id,
        collection_id=committed.collection_id,
        context=committed.context,
        content=committed.content,
        state=ArtifactEntity.STATE_DRAFT,
        created_by=user_id,
        modified_by=user_id,
        created_time=now,
        modified_time=now,
    )
    store.create_artifact(db, draft)
    return draft


def _reindex_after_state_change(
    artifact: ArtifactEntity, user_id: str, *, vacate: list[str],
) -> None:
    """After a state transition: (re)index the artifact into its new state's
    segment and remove its root from the segment(s) it left (``vacate``).

    The index is per-state (committed/draft/archived live in separate trees);
    ``index_artifact`` routes by ``artifact.state`` and the same enqueued job
    vacates exactly the old segment(s) — never a blanket purge, since a root may
    legitimately keep a committed version while a draft is also indexed.

    The container id is ``collection_id or id`` because **a top-level artifact is its own
    container** — the same identity ``_index_container`` passes when it indexes one at create
    time. Reading ``collection_id`` alone would hand the empty string to a segment move for
    every artifact written without a ``container_id``, which is the whole top-level population:
    the state on the row would change and the index would not follow it.
    """
    try:
        from mantle.search.ingest.pipeline_unified import enqueue_index_artifact
        enqueue_index_artifact(
            artifact, artifact.collection_id or artifact.id, tenant_id=user_id, vacate=vacate,
        )
    except Exception:
        logger.debug("reindex after state change failed", exc_info=True)


def update_artifact(
    db: Database,
    user_id: str,
    workspace_id: str,
    artifact_id: str,
    context: Optional[str] = None,
    content: Optional[str] = None,
    state: Optional[str] = None,
    content_type: Optional[str] = None,
    reindex: bool = True,
    vector=None,
) -> ArtifactEntity:
    """Update an artifact in a container.

    ``vector`` is a validated :class:`api.vectors.SuppliedVector`. Supplying one is itself a
    change: the writer is restating what this artifact means, so the update proceeds and
    reindexes even when no scalar field moved.
    """
    get_workspace(db, user_id, workspace_id, required="update")

    target = store.get_artifact(db, artifact_id)
    if target is not None and target.collection_id != workspace_id:
        target = None
    if target is None:
        target = store.get_draft_artifact(db, artifact_id, workspace_id)
    if target is None:
        target = store.get_latest_committed_artifact(db, artifact_id, workspace_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found")

    # Archive toggle.
    if state == ArtifactEntity.STATE_ARCHIVED and target.state != ArtifactEntity.STATE_ARCHIVED:
        target.state = ArtifactEntity.STATE_ARCHIVED
        target.modified_by = user_id
        target.modified_time = _now_iso()
        if store.update_artifact(db, target) is None:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist artifact update")
        # Archive: index into the archived segment and vacate committed + draft
        # (an archived root has no live committed/draft entry).
        if reindex:
            _reindex_after_state_change(target, user_id, vacate=["committed", "draft"])
        _emit_event(workspace_id, "artifact.updated", {"artifact": target.to_dict()}, actor_id=user_id)
        return target

    if target.state == ArtifactEntity.STATE_ARCHIVED and state and state != ArtifactEntity.STATE_ARCHIVED:
        target.state = ArtifactEntity.STATE_DRAFT
        target.modified_by = user_id
        target.modified_time = _now_iso()
        if store.update_artifact(db, target) is None:
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist artifact update")
        # Unarchive: index into draft and vacate the archived segment.
        if reindex:
            _reindex_after_state_change(target, user_id, vacate=["archived"])
        _emit_event(workspace_id, "artifact.updated", {"artifact": target.to_dict()}, actor_id=user_id)
        return target

    if target.state == ArtifactEntity.STATE_ARCHIVED:
        raise HTTPException(status.HTTP_409_CONFLICT, "Cannot edit an archived artifact")

    # Commit: flip a draft to committed in-place — triggers search indexing.
    if state == ArtifactEntity.STATE_COMMITTED:
        if target.state == ArtifactEntity.STATE_DRAFT:
            target.state = ArtifactEntity.STATE_COMMITTED
            target.modified_by = user_id
            target.modified_time = _now_iso()
            if store.update_artifact(db, target) is None:
                raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist artifact update")
            # Index into the committed segment and vacate the draft segment (the
            # draft just became the committed version).
            if reindex:
                _reindex_after_state_change(target, user_id, vacate=["draft"])
            _emit_event(workspace_id, "artifact.updated", {"artifact": target.to_dict()}, actor_id=user_id)
        return target  # already committed — no-op

    # If we're editing a committed version, promote to a new draft with same root_id.
    if target.state == ArtifactEntity.STATE_COMMITTED:
        target = _ensure_draft(db, user_id, target)

    dirty = False
    if context is not None and context != target.context:
        target.context = context
        dirty = True
    if content_type is not None and content_type != target.content_type:
        target.content_type = content_type
        dirty = True
    if content is not None and content != target.content:
        # Store new content in S3; update target.context with content_key.
        if content:
            content_key, stored_content = _store_content_in_s3(target.id, content, target.context, owner_id=target.created_by)
            try:
                ctx_obj = json.loads(target.context) if target.context else {}
            except (json.JSONDecodeError, TypeError):
                ctx_obj = {}
            ctx_obj["content_key"] = content_key
            target.context = json.dumps(ctx_obj)
            target.content = stored_content
        else:
            target.content = content
        dirty = True
    if not dirty and vector is None:
        return target

    target.modified_by = user_id
    target.modified_time = _now_iso()
    result = store.update_artifact(db, target)
    if result is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to persist artifact update")

    if reindex:
        try:
            from mantle.search.ingest.pipeline_unified import enqueue_index_artifact
            enqueue_index_artifact(
                target, target.collection_id, tenant_id=user_id, vector=vector,
            )
        except Exception:
            logger.debug("reindex failed", exc_info=True)

    _emit_event(workspace_id, "artifact.updated", {"artifact": target.to_dict()}, actor_id=user_id)
    return target


def delete_artifact(
    db: Database, user_id: str, workspace_id: str, artifact_id: str, *, cascade: bool = False,
) -> None:
    """Delete one artifact version plus the copies of it that live outside the vertex —
    the object-storage content and the search index.

    ``workspace_id`` is the containing collection and is REQUIRED, not optional. Both the
    S3 arm and the index arm are keyed on it, so a blank one deletes the row and leaves the
    plaintext chunks searchable — a delete that reports success and erases nothing that
    matters. Refuse the call instead: a caller that cannot name the container has not
    established which artifact it is deleting.

    This artifact may itself be a container — a sub-collection filed inside another one, not a
    top-level workspace — in which case it can have members of its own. Those are handled exactly
    as :func:`delete_workspace` handles a top-level container's members: detached (evicted, not
    destroyed) unless ``cascade=True``. Skipping this check left the previous version of this
    function silently unlinking a sub-collection while leaving its own members' edges pointing at
    a container id that no longer resolves to anything.
    """
    if not workspace_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "delete_artifact requires the containing collection id",
        )
    artifact = get_workspace_artifact(db, user_id, workspace_id, artifact_id)

    children = store.list_collection_artifacts(db, artifact_id, include_archived=True)
    if children:
        _delete_or_detach_members(db, artifact_id, user_id, children, cascade=cascade)

    # S3 cleanup — read the content_key straight from context.
    try:
        ctx = _safe_parse_context(artifact.context)
        # Validated, not trusted: an unchecked key here deletes ANOTHER tenant's object from both
        # the edge and durable buckets, and needs no read access to that artifact whatsoever.
        content_key = _safe_content_key(ctx, artifact_id)
        if content_key:
            from mantle.services.content_service import delete_object
            delete_object(content_key)
    except Exception:
        logger.debug("S3 cleanup failed", exc_info=True)

    # If this was the only version for the root, drop all edges too.
    other_versions = [
        v for v in store.list_version_history(db, artifact.root_id) if v.id != artifact_id
    ]
    draft = store.get_draft_artifact(db, artifact.root_id, workspace_id)

    store.delete_artifact(db, artifact_id)

    if not other_versions and (draft is None or draft.id == artifact_id):
        store.remove_all_edges_for_root(db, artifact.root_id)

    try:
        from mantle.search.ingest.pipeline_unified import delete_artifact_from_index
        # collection_id is REQUIRED for the removal to happen at all — without it both arms
        # no-op and the deleted artifact's chunks (with their plaintext text) stay searchable.
        delete_artifact_from_index(artifact_id, artifact.root_id, collection_id=workspace_id)
    except Exception:
        logger.debug("search delete failed", exc_info=True)

    _emit_event(workspace_id, "artifact.deleted", {"artifact_id": artifact_id}, actor_id=user_id)


def remove_artifact_from_container(
    db: Database,
    user_id: str,
    container_id: str,
    artifact_id: str,
) -> ArtifactEntity:
    """Remove an artifact from a container by detaching its edge.

    P2 — works on any container, not just workspaces. Access is gated by the
    caller's `check_access`, with the `evict` action. If a draft
    version of the artifact is owned by this container, the draft row is
    cleaned up as part of the removal so it does not linger.
    """
    # Resolve the root_id: the caller may pass a version id or a root_id.
    root_id = artifact_id
    artifact = store.get_artifact(db, artifact_id)
    if artifact:
        root_id = artifact.root_id or artifact.id

    # The edge is the canonical link. If there's no edge, the artifact
    # is not in this container.
    edge = store.get_edge(db, container_id, root_id)
    if not edge:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found")

    store.remove_artifact_from_collection(db, container_id, root_id)

    # If a draft version is owned by this container, clean it up.
    local = store.get_current_in_collection(db, container_id, root_id)
    if local and local.state == ArtifactEntity.STATE_DRAFT and local.collection_id == container_id:
        store.delete_artifact(db, local.id)
        try:
            from mantle.search.ingest.pipeline_unified import delete_artifact_from_index
            delete_artifact_from_index(local.id, local.root_id, collection_id=container_id)
        except Exception:
            logger.debug("search delete failed", exc_info=True)

    _emit_event(container_id, "artifact.deleted", {"artifact_id": artifact_id}, actor_id=user_id)
    return artifact or store.get_artifact(db, root_id) or ArtifactEntity(id=root_id, root_id=root_id)


def revert_artifact(
    workspace_db: Database,
    collection_db: Database,
    user_id: str,
    workspace_id: str,
    artifact_id: str,
) -> Optional[ArtifactEntity]:
    """
    Revert: drop the current draft, leaving the committed version in place.
    Returns the surviving committed version (or None if none exists).
    """
    get_workspace(workspace_db, user_id, workspace_id, required="update")
    target = store.get_artifact(workspace_db, artifact_id)
    if not target or target.collection_id != workspace_id:
        return None

    if target.state != ArtifactEntity.STATE_DRAFT:
        return target

    committed = store.get_latest_committed_artifact(
        workspace_db, target.root_id, workspace_id
    )
    if not committed:
        return None

    store.delete_artifact(workspace_db, target.id)

    try:
        from mantle.search.ingest.pipeline_unified import enqueue_index_artifact
        enqueue_index_artifact(committed, committed.collection_id, tenant_id=user_id)
    except Exception:
        logger.debug("reindex failed", exc_info=True)

    _emit_event(workspace_id, "artifact.updated", {"artifact": committed.to_dict()}, actor_id=user_id)
    return committed


def add_artifact_to_workspace(
    workspace_db: Database,
    collection_db: Database,
    user_id: str,
    workspace_id: str,
    artifact_root_id: str,
    grant_key: Optional[GrantEntity] = None,
) -> Optional[ArtifactEntity]:
    """
    Link an existing artifact (by root_id) into another collection via an edge.
    This is the "publish" action from the spec — it does not create a new
    artifact record.
    """
    get_workspace(workspace_db, user_id, workspace_id, required="add")

    # Must resolve to a committed version somewhere.
    committed = store.get_latest_committed_artifact(workspace_db, artifact_root_id)
    if not committed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not readable")

    order_key = after_key(store.get_last_order_key(workspace_db, workspace_id))
    store.add_artifact_to_collection(workspace_db, workspace_id, artifact_root_id, order_key)
    return committed


def move_artifact_between_containers(
    db: Database,
    user_id: str,
    source_container_id: str,
    target_container_id: str,
    artifact_id: str,
) -> ArtifactEntity:
    """Move an artifact from one container to another. P2 — type-blind."""
    artifact = store.get_artifact(db, artifact_id)
    if not artifact or artifact.collection_id != source_container_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found")

    artifact.collection_id = target_container_id
    artifact.modified_by = user_id
    artifact.modified_time = _now_iso()
    store.update_artifact(db, artifact)

    store.remove_artifact_from_collection(db, source_container_id, artifact.root_id)
    order_key = after_key(store.get_last_order_key(db, target_container_id))
    store.add_artifact_to_collection(db, target_container_id, artifact.root_id, order_key)

    _emit_event(source_container_id, "artifact.deleted", {"artifact_id": artifact_id}, actor_id=user_id)
    _emit_event(target_container_id, "artifact.created", {"artifact": artifact.to_dict()}, actor_id=user_id)
    return artifact


def move_workspace_artifact(
    db: Database,
    user_id: str,
    workspace_id: str,
    artifact_id: str,
    before_id: Optional[str],
    after_id: Optional[str],
    expected_version: Optional[int] = None,
) -> int:
    """Reposition an artifact within its workspace by rewriting its membership edge's order key.

    ``before_id`` and ``after_id`` name the neighbours to land between; a fractional key between
    their two order keys is assigned, so no other edge is touched and concurrent moves elsewhere
    in the list do not collide. Either may be ``None`` for "at the start"/"at the end".

    Always returns ``0``: ordering is not versioned, edges are authoritative, and
    ``expected_version`` is accepted for call-site compatibility WITHOUT being enforced — this
    call has no optimistic-concurrency check and a caller must not read one into it.

    Requires ``update``; raises ``HTTPException(404)`` for a missing workspace or artifact.
    """
    get_workspace(db, user_id, workspace_id, required="update")
    artifact = store.get_artifact(db, artifact_id)
    if not artifact:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Artifact not found")
    root_id = artifact.root_id

    before_key = None
    after_key_str = None
    if before_id:
        before = store.get_edge(db, workspace_id, before_id) or {}
        before_key = before.get("order_key")
    if after_id:
        after_rec = store.get_edge(db, workspace_id, after_id) or {}
        after_key_str = after_rec.get("order_key")

    new_key = mid_key(before_key, after_key_str)
    store.set_edge_order_key(db, workspace_id, root_id, new_key)
    return 0


def get_artifacts_order_version(db: Database, user_id: str, workspace_id: str) -> int:
    """Always ``0``. Membership edges carry ordering directly, so there is no order counter to
    report; the call survives to give ordering clients a stable value to echo back, and the
    access check it performs is the part that still has an effect.

    Requires ``read``; raises ``HTTPException(404)`` otherwise.
    """
    get_workspace(db, user_id, workspace_id, required="read")
    return 0  # order is not versioned — edges are authoritative for ordering


# ---------------------------------------------------------------------------
# File upload helpers
# ---------------------------------------------------------------------------

def _tenant_prefix_for_workspace(db: Database, user_id: str, workspace_id: str) -> str:
    ws = get_workspace(db, user_id, workspace_id, required="read")
    return f"{ws.created_by}"


def initiate_upload_and_create_artifact(
    db: Database,
    user_id: str,
    workspace_id: str,
    filename: str,
    content_type: str,
    size: int,
    order_key: Optional[str] = None,
    context: Optional[dict] = None,
):
    """Create a draft artifact to receive a file upload; return ``(descriptor, artifact)``.

    The descriptor is ``{"upload_id", "mode", "url", "method", "key"}`` and always describes a
    PROXIED upload — the client PUTs bytes to Mantle, which envelope-encrypts them before they
    reach object storage. No presigned storage URL is ever handed out, so storage stays invisible
    to callers and never receives plaintext.

    ``size`` and ``content_type`` are the client's CLAIM, recorded in context so the UI has
    something to show; the true values are read back from storage on completion. The artifact is
    created unindexed because it has no content yet — :func:`update_upload_status` indexes it
    once the bytes land.

    Requires ``create`` on the workspace; raises ``HTTPException(404)`` otherwise.
    """
    from mantle.services.content_service import presign_put_or_multipart, get_content_storage_mode
    from mantle.services.ingest_runner_service import describe_content_processing

    tenant = _tenant_prefix_for_workspace(db, user_id, workspace_id)

    base_ctx: Dict[str, Any] = {
        "content_source": "agience-content",
        "access": "private",
        "filename": filename,
        "content_type": content_type,
        "size": size,
        "storage": {"mode": get_content_storage_mode()},
        "processing": describe_content_processing(content_type, upload_complete=False),
        "upload": {"status": "initiated", "progress": 0.0},
    }
    if context:
        base_ctx.update(context)

    artifact = create_workspace_artifact(
        db,
        user_id,
        workspace_id,
        context=_ensure_json_str(base_ctx),
        content="",
        order_key=order_key,
        enqueue_index=False,
    )

    key = f"{tenant}/{artifact.id}.content"

    patched_ctx = _safe_parse_context(artifact.context)
    patched_ctx["content_key"] = key
    up = patched_ctx.setdefault("upload", {})
    # Proxied upload: the client PUTs bytes to Mantle, which envelope-encrypts them on
    # the byte path before writing to the object store. No presigned S3 URL — storage
    # is invisible to callers and never receives plaintext.
    up["mode"] = "proxied"
    up["s3_key"] = key

    updated = update_artifact(
        db, user_id, workspace_id, artifact.id,
        context=_ensure_json_str(patched_ctx),
        reindex=False,
    )

    return (
        {
            "upload_id": updated.id,
            "mode": "proxied",
            "url": f"/artifacts/{updated.id}/content",
            "method": "PUT",
            "key": key,
        },
        updated,
    )


def update_upload_status(
    db: Database,
    user_id: str,
    workspace_id: str,
    upload_id: str,
    status_value: str,
    progress: Optional[float] = None,
    parts: Optional[List[Dict]] = None,
    context_patch: Optional[Dict] = None,
):
    """Advance an in-progress upload and return the updated artifact.

    ``uploading`` and ``failed`` only record state in the artifact's context and return without
    touching storage; ``failed`` additionally marks every processing stage failed. ``complete``
    finalizes a multipart write if there is one, reads the object's TRUE size and content type
    back from storage (the values recorded at initiation were the client's claim), mirrors the
    object to durable storage, and drops the ``upload`` section — after which the artifact
    indexes on the ordinary update path.

    Raises ``HTTPException(400)`` for an unrecognised ``status_value``, for a ``complete`` whose
    context carries no key or mode, for a multipart completion missing its id or parts, and when
    the object is not in storage. Raises ``HTTPException(502)`` if the durable copy fails, which
    leaves the upload unfinished rather than reporting a success only the edge bucket can serve.
    """
    from mantle.services.content_service import (
        complete_multipart,
        head_object,
        persist_object_to_durable,
    )
    from mantle.services.ingest_runner_service import describe_content_processing

    artifact = get_workspace_artifact(db, user_id, workspace_id, upload_id)
    ctx = _safe_parse_context(artifact.context)

    if context_patch:
        for k, v in context_patch.items():
            ctx[k] = v

    up = dict(ctx.get("upload") or {})
    if progress is not None:
        up["progress"] = max(0.0, min(1.0, progress))

    key = up.get("s3_key")
    mode = up.get("mode")

    if status_value in ("uploading", "failed"):
        up["status"] = status_value
        ctx["upload"] = up
        if status_value == "failed":
            processing = dict(ctx.get("processing") or {})
            for k in ("asset_status", "content_status", "index_status", "status"):
                processing[k] = "failed"
            ctx["processing"] = processing
        return update_artifact(
            db, user_id, workspace_id, upload_id,
            context=_ensure_json_str(ctx),
            reindex=False,
        )

    if status_value == "complete":
        if not key or not mode:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Upload context missing key or mode")

        if mode == "multipart":
            multipart_id = up.get("multipart_id")
            if not multipart_id:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Multipart requires multipart_id")
            effective_parts = parts
            if not effective_parts:
                head = head_object(key)
                if not head:
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Multipart requires parts list")
                effective_parts = []
            normalized: List[Dict[str, Any]] = []
            for p in effective_parts or []:
                if not isinstance(p, dict):
                    continue
                part_num = p.get("PartNumber") or p.get("part_number") or p.get("partNumber") or p.get("part")
                etag = p.get("ETag") or p.get("etag") or p.get("e_tag")
                if not part_num or not etag:
                    continue
                try:
                    part_num_int = int(part_num)
                except Exception:
                    continue
                if isinstance(etag, str) and etag.startswith('"') and etag.endswith('"'):
                    etag = etag[1:-1]
                normalized.append({"PartNumber": part_num_int, "ETag": etag})
            if normalized:
                normalized.sort(key=lambda x: x["PartNumber"])
                complete_multipart(key, multipart_id, normalized)

        head = head_object(key)
        if not head:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Object not found in S3")

        ctx["size"] = head.get("ContentLength", ctx.get("size"))
        ctx["content_type"] = head.get("ContentType", ctx.get("content_type"))
        ctx["processing"] = describe_content_processing(ctx.get("content_type") or "", upload_complete=True)

        try:
            if persist_object_to_durable(key):
                storage = dict(ctx.get("storage") or {})
                storage["durable_synced"] = True
                storage["durable_key"] = key
                ctx["storage"] = storage
        except Exception:
            logger.warning("Durable content sync failed for key=%s", key)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Failed to persist upload to durable storage")

        ctx.pop("upload", None)

        result = update_artifact(
            db, user_id, workspace_id, upload_id,
            context=_ensure_json_str(ctx),
        )
        _emit_event(workspace_id, "upload.complete", {"artifact": result.to_dict()}, actor_id=user_id)
        return result

    raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid status")


# ---------------------------------------------------------------------------
# Artifact-scoped grant keys (inbound / stream)
# ---------------------------------------------------------------------------
#
# A card key is an ordinary grant key (services/grant_key_service) scoped to the one
# workspace the card lives in, so what the key may do is legible in the same CRUDEASIO
# bits as any other grant.

_KEY_CONTEXT_MAP: Dict[str, Tuple[str, str, str]] = {
    "stream":  ("stream",  "obs_grant_key_id", "Stream Source"),
    "inbound": ("inbound", "grant_key_id",     "Inbound Source"),
}

#: What a card key may do in its workspace: read the card and post messages back.
#: Deliberately not `can_delete`/`can_admin` — an inbound webhook URL is the most
#: exposed credential in the system and should not be able to destroy the workspace.
_CARD_KEY_FLAGS = {"can_read": True, "can_create": True, "can_update": True, "can_add": True}


def rotate_artifact_key(
    db: Database,
    user_id: str,
    workspace_id: str,
    artifact_id: str,
    store_db: Database,
    key_context: str,
) -> Dict[str, str]:
    """Issue a fresh card key for an artifact's ``stream`` or ``inbound`` endpoint, revoking the
    one it replaces, and record the new key's id in the artifact's context.

    Returns ``{"workspace_id", "artifact_id", "key_id", "key"}``. ``key`` is shaped
    ``"<artifact_id>:<secret>"`` because :func:`resolve_card_key` authenticates the pair, and
    this return is the ONLY time the secret is available — it is hashed at rest and cannot be
    recovered afterwards, so a caller that drops it must rotate again.

    The grant is scoped to the WORKSPACE, not to the artifact, carrying read/create/update/add
    and deliberately not delete or admin: an inbound webhook URL is the most exposed credential
    in the system and must not be able to destroy the workspace it posts into. Binding to the
    issuing artifact is enforced separately, by the ``key_id`` written into context here.

    The prior key is revoked rather than deleted so it stays visible in the grant ledger. A
    revocation that fails is logged and rotation proceeds — the replaced key can therefore
    outlive the rotation, and a caller rotating in response to a leak must confirm the old key
    is revoked rather than assume it.

    The only access this checks is ``read`` on the workspace, by way of the artifact lookup —
    while the key it mints carries create/update/add. Callers that expose this must gate it on
    the authority they intend to require; it does not gate itself.

    Raises ``HTTPException(400)`` for a ``key_context`` outside ``{"stream", "inbound"}`` and
    ``HTTPException(404)`` if the artifact is not readable in this workspace.
    """
    from mantle.services import grant_key_service
    binding = _KEY_CONTEXT_MAP.get(key_context)
    if not binding:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown key_context: {key_context}")
    context_section, key_id_field, key_label_prefix = binding

    artifact = get_workspace_artifact(db, user_id, workspace_id, artifact_id)
    ctx = _safe_parse_context(artifact.context)
    section_cfg = dict(ctx.get(context_section) or {})

    # Rotation must invalidate the old credential. Revoked rather than deleted so the
    # replaced key stays visible in the grant ledger.
    old_key_id = section_cfg.get(key_id_field)
    if isinstance(old_key_id, str) and old_key_id.strip():
        try:
            old = store.get_grant_by_id(store_db, old_key_id.strip())
            if old is not None:
                grant_key_service.revoke(store_db, old, user_id)
        except Exception:
            logger.warning("card-key rotation: revoking prior key failed", exc_info=True)

    created, raw_key = grant_key_service.mint(
        store_db,
        user_id=user_id,
        name=f"{key_label_prefix} - {artifact_id}",
        resource_id=workspace_id,
        flags=_CARD_KEY_FLAGS,
        notes=f"card:{key_context}:{artifact_id}",
    )

    section_cfg[key_id_field] = created.id
    ctx[context_section] = section_cfg
    update_artifact(db, user_id, workspace_id, artifact_id, context=_ensure_json_str(ctx))

    return {
        "workspace_id": workspace_id,
        "artifact_id": artifact_id,
        "key_id": created.id,
        "key": f"{artifact_id}:{raw_key}",
    }


def resolve_card_key(
    db: Database,
    artifact_id: str,
    token: str,
    store_db: Database,
    key_context: Optional[str] = None,
) -> Tuple[ArtifactEntity, CollectionEntity, str]:
    """Authenticate a card key and return the artifact, its workspace, and the owner.

    Every failure is 404 regardless of cause, so a caller holding a wrong token cannot
    use the response to learn which artifact ids exist.
    """
    from mantle.services import grant_key_service

    artifact = store.get_artifact(db, artifact_id)
    if not artifact:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    workspace_id = artifact.collection_id
    root = grant_key_service.authenticate(store_db, token)
    if root is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials")

    ws = store.get_collection_by_id(db, workspace_id)
    if not ws:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    # The key must actually reach THIS workspace and be able to write to it. Resolving
    # the bundle means a key that carries the workspace as a member works too, without
    # this path needing to know whether it was a bundle.
    effective = grant_key_service.resolve(store_db, root)
    if not any(
        g.resource_id == workspace_id and getattr(g, "can_create", False)
        for g in effective
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    # Bind the key to the specific card that issued it: a stream key for one card must
    # not post as another card in the same workspace.
    ctx = _safe_parse_context(artifact.context)
    binding = _KEY_CONTEXT_MAP.get(key_context) if key_context else None
    if binding:
        section_cfg = ctx.get(binding[0]) or {}
        if isinstance(section_cfg, dict):
            expected = section_cfg.get(binding[1])
            if isinstance(expected, str) and expected.strip():
                if str(root.id or "") != expected.strip():
                    raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")

    return artifact, ws, ws.created_by


def receive_card_inbound_message(
    db: Database,
    artifact_id: str,
    token: str,
    text: str,
    channel: Optional[str] = None,
    context: Optional[dict] = None,
    metadata: Optional[dict] = None,
    store_db: Optional[Database] = None,
) -> Tuple[str, str]:
    """Record an inbound webhook message as a new artifact in the card's workspace.

    Authenticates ``token`` as that card's ``inbound`` key, then creates the message artifact as
    the WORKSPACE OWNER rather than as the credential: a webhook has no user behind it, and the
    owner is the principal accountable for what lands in their workspace. The originating card is
    kept in the new artifact's context as ``source_artifact_id``, so provenance survives even
    though the writer is recorded as the owner.

    Returns ``(message_artifact_id, workspace_id)``. Raises ``HTTPException(401)`` for a token
    that does not authenticate and ``HTTPException(404)`` for every other failure — including a
    valid token pointed at the wrong card — so a caller cannot use the response to probe which
    artifact ids exist.
    """
    source_artifact, ws, owner_id = resolve_card_key(
        db, artifact_id, token, store_db or db, key_context="inbound"
    )
    card_ctx: Dict[str, Any] = {
        "source_artifact_id": artifact_id,
        "inbound": {"channel": channel or "unknown", "via": "webhook"},
    }
    if isinstance(context, dict):
        card_ctx.update(context)
    if isinstance(metadata, dict):
        card_ctx.setdefault("metadata", metadata)

    msg = create_workspace_artifact(
        db=db, user_id=owner_id, workspace_id=ws.id,
        context=_ensure_json_str(card_ctx), content=text or "",
    )
    return msg.id or "", ws.id


# ---------------------------------------------------------------------------
# Native dispatch handlers — called by operation_dispatcher for type.json
# ``dispatch: { kind: "native", target: "workspace_service.<fn>" }``
# ---------------------------------------------------------------------------

async def dispatch_create_workspace(artifact: dict, body: dict, ctx: Any) -> dict:
    """Create a workspace via the ``create`` operation on workspace type."""
    name = (body or {}).get("name", "New Workspace")
    ws = create_workspace(ctx.store_db, ctx.user_id, name)
    return ws.to_dict()
