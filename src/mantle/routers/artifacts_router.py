# routers/artifacts_router.py
#
# Unified Artifact API — single REST surface for all artifact operations.
#
# Replaces per-container endpoints (workspaces, collections, agents, inbound,
# search) with a container-agnostic set of verbs:
#
#   POST   /artifacts                → Create
#   GET    /artifacts/{id}           → Read
#   PATCH  /artifacts/{id}           → Update
#   DELETE /artifacts/{id}           → Delete
#   PUT    /artifacts/{container_id} → Add item to a container
#   POST   /artifacts/search         → Search
#
# Specialized endpoints:
#   POST   /artifacts/{id}/upload-initiate     → Initiate S3 upload
#   PATCH  /artifacts/{id}/upload-status       → Update upload progress
#   GET    /artifacts/{id}/multipart-part-url  → Presigned URL for upload part
#   GET    /artifacts/{id}/content-url         → Signed content URL
#   PATCH  /artifacts/{container_id}/order      → Reorder workspace artifacts
#   POST   /artifacts/{id}/move                → Move artifact between workspaces
#   POST   /artifacts/batch                    → Batch fetch by IDs
#   GET    /artifacts/{container_id}/commits    → List commits for collection
#
# Mantle stores artifacts as (content_type, context, content); content_type is a
# LABEL (not resolved to a type.json). Operation dispatch (/op) was removed in
# Phase 2b — that runtime relocates to the Chorus `core` gateway. (One create-time
# dispatch remains for top-level container types pending the gateway.)
#
# Real-time event subscription is handled by the unified /events WebSocket
# (see routers/events_router.py), not a per-container SSE endpoint.

import json
import logging
import uuid
from typing import Any, Dict, List, Literal, Optional, Set

from mantle.db.store import Database
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from mantle.services.dependencies import get_store_db
import mantle.db.backend as store
from mantle.db.backend import has_children as db_has_children, count_children as db_count_children
from mantle.services.dependencies import (
    get_auth,
    AuthContext,
    check_access,
    check_inbound_nonce,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Request / Response Models
# =============================================================================

class CreateArtifactRequest(BaseModel):
    """Create an artifact.

    A collection is just an artifact with child edges, so there's one create path:
    with ``container_id`` the new artifact is also edged into that collection (the
    CRUDEASIO *Add*); without it, it's a top-level artifact. Either way the creator
    gets a direct owner grant — ownership is a grant on the artifact, not a function
    of where it lives.
    """
    container_id: Optional[str] = None   # optional: also add a membership edge into this collection
    source_artifact_id: Optional[str] = None  # link an existing artifact instead of creating one
    name: Optional[str] = None
    context: Optional[str] = None       # JSON string
    content: Optional[str] = None
    content_type: Optional[str] = None
    description: Optional[str] = None
    # WHERE-indexing hint (Information Gauge DB, Phase 1): "eager" indexes now,
    # "lazy" leaves the artifact latent (indexed on first access). None uses the
    # deployment default (MANTLE_LAZY_INDEX). Applies to content in a container;
    # top-level containers are always indexed eagerly (the navigable frame).
    index: Optional[str] = None


class UpdateArtifactRequest(BaseModel):
    """Partial update to an artifact or container."""
    context: Optional[str] = None
    content: Optional[str] = None
    state: Optional[str] = None
    content_type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class InvokeArtifactRequest(BaseModel):
    """Invoke an operator artifact."""
    name: Optional[str] = None              # tool name (for mcp_tool dispatch via $.body.name)
    arguments: Optional[Dict[str, Any]] = None  # tool arguments (for mcp_tool dispatch)
    workspace_id: Optional[str] = None
    artifacts: Optional[List[str]] = None   # context artifact IDs
    input: Optional[str] = None
    params: Optional[Dict[str, Any]] = None



class RemoveItemRequest(BaseModel):
    """Remove an item (artifact root/current version) from a workspace container."""
    container_id: str


class ArtifactSearchRequest(BaseModel):
    """Search across accessible artifacts. ``query_text`` is required.

    Unknown fields are IGNORED (pydantic default), not rejected — a client still
    sending the removed ``embedding`` or ``aperture`` gets a normal lexical search,
    not a 422. Neither field has any effect.
    """
    model_config = ConfigDict(populate_by_name=True)

    query_text: Optional[str] = None
    # No `embedding`: a caller-supplied query vector is trained-model output — BYOK by
    # another name — and accepting one lit the vector arm with no provider configured.
    # Removed 2026-07-30, no-models rule (universal, incl. BYOK).
    scope: Optional[List[str]] = None           # container IDs to restrict
    state: str = "committed"                     # index segment: committed (default) | draft | archived
    content_types: Optional[List[str]] = None
    use_hybrid: Optional[bool] = None
    # `aperture` removed 2026-07-30 — it was never read; see mantle/search/types.py.
    from_: int = 0
    size: int = 20
    sort: Optional[Literal["relevance", "recency"]] = None
    highlight: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_from_alias(cls, data):
        if isinstance(data, dict) and "from" in data and "from_" not in data:
            data = dict(data)
            data["from_"] = data.pop("from")
        return data


class SearchHitResponse(BaseModel):
    """Search hit with content fields for downstream consumers."""
    id: str
    score: float
    root_id: str
    version_id: str
    collection_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    highlights: Optional[Dict[str, List[str]]] = None


class ArtifactSearchResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hits: List[SearchHitResponse]
    total: int
    query_text: str
    parsed_query: Optional[str] = None
    corrections: List[str] = Field(default_factory=list)
    used_hybrid: bool
    from_: int = 0
    size: int

    @model_validator(mode="before")
    @classmethod
    def _accept_from_alias(cls, data):
        if isinstance(data, dict) and "from" in data and "from_" not in data:
            data = dict(data)
            data["from_"] = data.pop("from")
        return data

    @model_serializer(mode="wrap")
    def _emit_from_alias(self, handler: SerializerFunctionWrapHandler):
        data = handler(self)
        if isinstance(data, dict) and "from_" in data and "from" not in data:
            data["from"] = data.pop("from_")
        return data


# =============================================================================
# Helpers
# =============================================================================

# Unified artifact store: containers and artifacts both live in `artifacts`.
_COLL_ARTIFACTS = "artifacts"

def _artifact_exists(db: Database, artifact_id: str) -> bool:
    """Return True if artifact_id refers to an existing artifact document."""
    try:
        from mantle.db.backend import get_raw_artifact
        return get_raw_artifact(db, artifact_id) is not None
    except Exception:
        return False



def _find_artifact(db: Database, artifact_id: str) -> Optional[dict]:
    """Locate an artifact in the unified store.

    First resolves builtin server short-names ("astra", "verso", etc.) to their
    stable UUID via the server registry so callers can use human-readable names.
    Then tries exact _key lookup; if not found, resolves by root_id (operation
    routes commonly receive root_id values for built-in server artifacts).
    Archived artifacts return None.
    """
    # Resolve builtin server names to their stable bootstrap UUID.
    from mantle.services import server_registry as _server_registry
    from mantle.db.backend import _decrypt_artifact_content as _decrypt_doc
    resolved_id = _server_registry.get_id(artifact_id)
    if resolved_id:
        artifact_id = resolved_id

    from mantle.db.backend import get_raw_artifact, find_newest_by_root
    try:
        doc = get_raw_artifact(db, artifact_id)
        if doc and doc.get("state") != "archived":
            _decrypt_doc(doc)  # this raw-doc path bypasses the entity converters — decrypt here
            return doc
    except Exception:
        logger.warning("_find_artifact: key lookup failed for %r", artifact_id, exc_info=True)

    # Resolve stable root IDs to the newest non-archived version row.
    try:
        doc = find_newest_by_root(db, artifact_id)
        if doc:
            _decrypt_doc(doc)
            return doc
    except Exception:
        logger.warning("_find_artifact: root_id scan failed for %r", artifact_id, exc_info=True)

    return None


def _normalize_artifact_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an artifact document for API responses.

    Sets defaults for missing fields and strips the lattice internal keys.
    """
    normalized = dict(doc)

    # Defense-in-depth: decrypt inline content for any raw-doc path that reaches an
    # API response through here. Idempotent (flag-gated) — a no-op on docs already
    # decrypted by from_store_doc / _find_artifact / list_collection_artifacts.
    from mantle.db.backend import _decrypt_artifact_content as _decrypt_doc
    _decrypt_doc(normalized)

    artifact_id = normalized.get("id") or normalized.get("_key")
    if artifact_id and not normalized.get("root_id"):
        normalized["root_id"] = artifact_id

    if normalized.get("context") is None:
        normalized["context"] = ""

    if normalized.get("content") is None:
        normalized["content"] = ""

    if "_key" in normalized:
        normalized.setdefault("id", normalized.pop("_key"))

    normalized.pop("_id", None)
    normalized.pop("_rev", None)

    return normalized


def _strip_immutable_context_fields(
    doc: Dict[str, Any],
    context: Optional[str],
) -> Optional[str]:
    """No-op since Phase 2b. Mantle treats ``content_type`` as a label and does
    not resolve type definitions, so it does not enforce a type's
    ``context_schema`` mutability rules — field-mutability is an application
    concern (enforced at the Chorus gateway). Returns the context unchanged."""
    return context


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


# ---------- GET /artifacts/visible — list artifacts the caller can read ----------
#
# Browser UX needs "show me every workspace / collection I can see" without
# having to know a parent ID. /search requires query_text and is for relevance-
# ranked queries; this is the flat-list affordance, scoped through the canonical
# LightConeResolver (same ACL path /search uses internally).
@router.get("/visible")
async def list_visible(
    content_type: Optional[str] = Query(
        None,
        description="Filter by exact content_type (MIME). Omit to list every accessible artifact.",
    ),
    action: str = Query(
        "read",
        description=(
            "CRUDEASIO action to filter by (read, create, add, update, ...). Returns "
            "only artifacts the caller may perform this action on. Defaults to 'read' "
            "(everything visible). Use 'create' to list collections an artifact may be "
            "assigned into — read-only platform collections are excluded."
        ),
    ),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    from mantle.services.dependencies import _ACTION_FLAG_MAP

    flag_attr = _ACTION_FLAG_MAP.get(action)
    if flag_attr is None:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    from mantle.search.mantle.lightcone import LightConeResolver

    resolver = LightConeResolver(store_db)

    # First-login provisioning is keyed on READ access (the baseline seed grant
    # set). A user with nothing readable has not yet been granted the platform
    # seed collections — provision them now (idempotent, safe on every startup
    # after a factory reset). Always resolved against "read", never the requested
    # action, so e.g. ?action=create does not retrigger provisioning for users
    # who legitimately have no create grants.
    read_authorized: Set[str] = resolver.resolve(auth.user_id, "read") if auth.user_id else set()
    if auth.user_id and not read_authorized:
        try:
            from mantle.services.seed_provisioning import provision_user
            # Capture profile + tenant from the token (external-IdP logins carry
            # email/name/issuer); platform users pass None and are unaffected.
            provision_user(
                store_db,
                user_id=auth.user_id,
                email=getattr(auth, "email", None),
                name=getattr(auth, "name", None),
                tenant=getattr(auth, "authority", None),
            )
            read_authorized = resolver.resolve(auth.user_id, "read")
            logger.info("First-login provisioning completed for user %s", auth.user_id)
        except Exception:
            logger.warning(
                "First-login provisioning failed for user %s (non-fatal)", auth.user_id, exc_info=True
            )

    if action == "read":
        authorized: Set[str] = set(read_authorized)
    else:
        authorized = resolver.resolve(auth.user_id, action) if auth.user_id else set()
    if auth.bearer_grant and getattr(auth.bearer_grant, flag_attr, False) and auth.bearer_grant.resource_id:
        authorized.add(auth.bearer_grant.resource_id)

    results: list = []
    for aid in authorized:
        doc = _find_artifact(store_db, aid)
        if not doc:
            continue
        if content_type and doc.get("content_type") != content_type:
            continue
        results.append(_normalize_artifact_doc(doc))
    return results


# ---------- POST /artifacts — Create ----------

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_artifact(
    request: Request,
    body: Dict[str, Any] = Body(...),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Create a new artifact.

    If the resolved content type declares a ``create`` operation in its
    ``type.json``, dispatches through the operation dispatcher. Otherwise
    falls back to default artifact creation via ``workspace_service``.
    """
    # Bot protection: keys flagged `requires_nonce` (inbound website keys) must
    # present a valid X-Agience-Challenge. No-op for users / non-nonce keys.
    check_inbound_nonce(request, auth)

    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    # Mantle is the database layer — `create` is a plain insert; no operation
    # dispatch, no type resolution. content_type is an opaque label.
    return await _default_create_artifact(body, auth, store_db)


async def _default_create_artifact(
    body: Dict[str, Any],
    auth: AuthContext,
    store_db: Any,
) -> Dict[str, Any]:
    """Create an artifact. One path for everything — a collection is just an
    artifact with child edges, so there's no separate container-create:

    - no ``container_id`` -> a TOP-LEVEL artifact (any authenticated user may create
      one they own). This subsumes the old POST /artifacts/containers.
    - with ``container_id`` -> also edge the new artifact into that collection (the
      caller must be able to Add to it). 404 if the collection doesn't exist.

    Either way the creator gets a direct owner grant (see workspace_service) — access
    is a grant on the artifact, independent of where it sits.
    """
    parsed = CreateArtifactRequest(**body)
    context_str = _merge_content_type_into_context(parsed.context, parsed.content_type)

    from mantle.services import workspace_service

    # Top-level: no parent to authorize against; the creator owns it.
    if not parsed.container_id:
        entity = workspace_service.create_container(
            db=store_db,
            user_id=auth.user_id,
            content_type=parsed.content_type,
            name=parsed.name,
            context=context_str or "",
            content=parsed.content or "",
            description=parsed.description,
        )
        return entity.to_dict()

    # Into a collection — the caller needs create/Add permission on it.
    check_access(auth, parsed.container_id, "create", store_db)
    if not _artifact_exists(store_db, parsed.container_id):
        raise HTTPException(status_code=404, detail="Container not found")

    # source_artifact_id -> LINK an existing artifact in (edge only), no new artifact.
    if parsed.source_artifact_id:
        return _link_source_artifact(store_db, parsed, auth)

    entity = workspace_service.create_workspace_artifact(
        db=store_db,
        user_id=auth.user_id,
        workspace_id=parsed.container_id,
        context=context_str or "",
        content=parsed.content or "",
        content_type=parsed.content_type,
        name=parsed.name,
        index=parsed.index,
    )
    return entity.to_dict()


def _link_source_artifact(
    store_db: Any,
    parsed: CreateArtifactRequest,
    auth: Any,
) -> Dict[str, Any]:
    """Link an existing artifact into a container instead of creating a duplicate.

    ⛔ THIS RAN WITH NO CHECK ON THE SOURCE AT ALL.
    `check_access` had been applied to the *container* only — which the attacker
    owns — so `POST /artifacts {container_id: <mine>, source_artifact_id: <yours>}`
    did two things:

      (a) returned `source.to_dict()`, i.e. the victim's CONTENT, decrypted. The
          read choke point in `db.store.from_store_doc` selects the decryption
          key from the stored document's `created_by`, so the attacker's identity
          never enters the decryption path — it decrypts with the VICTIM's key.

      (b) wrote a **creation-lineage** edge (`origin=True`, `propagate=None` — the
          defaults) from the attacker's container to the victim's root. Grants
          propagate parent -> child and `check_access` walks UP from a target via
          `get_origin_parent`, so the attacker's container could be returned as
          the victim's origin parent and confer grants over the whole subtree.
          With `propagate=None` no action is masked out: all of CRUDEASIO.

    Two fixes, both required:
      1. Authorize the source for `read` before touching it.
      2. Link with `origin=False, propagate=[]` so a link edge can never be a
         grant-inheritance path, regardless of who is allowed to create it.
    """
    from mantle.db.backend import (
        get_artifact as _get_artifact,
        get_latest_committed_artifact,
        add_artifact_to_collection,
    )

    # Authorize BEFORE loading: the load decrypts, and the 404-vs-403 distinction
    # would otherwise confirm the artifact's existence to a caller with no read.
    check_access(auth, parsed.source_artifact_id, "read", store_db)

    source = _get_artifact(store_db, parsed.source_artifact_id)
    if not source:
        source = get_latest_committed_artifact(store_db, parsed.source_artifact_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source artifact not found")
    root_id = source.root_id or source.id
    add_artifact_to_collection(
        store_db,
        parsed.container_id,
        root_id,
        origin=False,
        propagate=[],
    )
    return source.to_dict()


def _merge_content_type_into_context(
    context_str: Optional[str],
    content_type: Optional[str],
) -> Optional[str]:
    """Merge content_type into a context JSON string if provided."""
    if not content_type:
        return context_str
    if context_str:
        try:
            ctx = json.loads(context_str)
            ctx.setdefault("content_type", content_type)
            return json.dumps(ctx)
        except (json.JSONDecodeError, TypeError):
            return context_str
    return json.dumps({"content_type": content_type})


# ---------- GET /artifacts/{artifact_id} — Read ----------

@router.get("/{artifact_id}")
async def read_artifact(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Read a single artifact by ID."""
    check_access(auth, artifact_id, "read", store_db)

    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # Normalize the lattice internal keys.
    doc.pop("_id", None)
    doc.pop("_rev", None)
    if "_key" in doc:
        doc.setdefault("id", doc.pop("_key"))

    # Inject computed child-containment fields.
    root_id = doc.get("root_id") or doc.get("id") or artifact_id
    doc["has_children"] = db_has_children(store_db, root_id)
    doc["child_count"] = db_count_children(store_db, root_id) if doc["has_children"] else 0

    # Lazy indexing: first genuine (authorized) access materializes a latent vertex.
    # No-op unless MANTLE_LAZY_INDEX is on and the vertex isn't already materialized.
    try:
        from mantle.services import workspace_service as _ws
        from mantle.entities.artifact import Artifact as _Artifact
        _ws.materialize_on_access(
            store_db,
            artifact_id=doc.get("id") or artifact_id,
            collection_id=doc.get("collection_id"),
            tenant_id=auth.user_id,
            artifact=_Artifact.from_dict(doc),
        )
    except Exception:
        pass

    return doc


# ---------- POST /artifacts/{container_id}/warm — Warm-sweep (lazy indexing) ----------

@router.post("/{container_id}/warm")
async def warm_collection_endpoint(
    container_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Warm-sweep guardrail: materialize every latent artifact in a collection so
    it is searchable up front, rather than waiting for each to be accessed. For
    corpora that must be searchable immediately. Requires read access."""
    check_access(auth, container_id, "read", store_db)
    from mantle.services import workspace_service as _ws
    n = _ws.warm_collection(store_db, container_id, tenant_id=auth.user_id)
    return {"collection_id": container_id, "materialized": n}


# ── GET /artifacts/{id}/embedding: REMOVED 2026-07-30 ────────────────────────
# "Raw vectors out, no text" — an embedding-serving endpoint, and the stored
# vectors are bge-m3 output (trained weights on someone else's disk).
#
# Removed ENTIRELY rather than 404/501, per the standing ruling on `/coherence`
# and `/embed`: an observer does not offer "embed this" or "score this" as a
# service. No caller existed in any repo. No-models rule.



# ---------- GET /artifacts/{artifact_id}/children — List children ----------

@router.get("/{artifact_id}/children")
async def list_children(
    artifact_id: str,
    request: Request,
    content_type: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    store_db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """List children of any artifact (universal container model).

    Optional filters:
    - content_type: filter children by their content_type
    - workspace_id: include draft children from this workspace

    Each child is enriched with `committed_collection_ids` — the set of committed
    containers it currently appears in.
    """
    check_access(auth, artifact_id, "read", store_db)

    # A draft is workspace-private. Only surface drafts linked into this container
    # when the caller passes a workspace_id they can READ — then drafts homed in
    # THAT workspace (the caller's own) show here, never anyone else's.
    draft_workspace_id: Optional[str] = None
    if workspace_id:
        try:
            check_access(auth, workspace_id, "read", store_db)
            draft_workspace_id = workspace_id
        except HTTPException:
            draft_workspace_id = None  # no access → don't include its drafts

    children = store.list_collection_artifacts(
        store_db, artifact_id, draft_workspace_id=draft_workspace_id
    )

    # Filter out operator edges (relationship != null means non-containment)
    children = [c for c in children if not c.get("relationship")]

    # Optional content_type filter
    if content_type:
        children = [c for c in children if c.get("content_type") == content_type]

    # Enrich with committed_collection_ids (structural — pure edge traversal)
    from mantle.entities.artifact import Artifact as ArtifactEntity
    from mantle.services.collection_service import attach_committed_collection_ids
    entities = [ArtifactEntity.from_dict(c) for c in children]
    attach_committed_collection_ids(store_db, entities)
    for raw, entity in zip(children, entities):
        raw["committed_collection_ids"] = getattr(entity, "committed_collection_ids", [])

    # Normalize each child
    for child in children:
        _normalize_artifact_doc(child)

    return children


# ---------- PATCH /artifacts/{artifact_id} — Update ----------

@router.patch("/{artifact_id}")
async def update_artifact(
    artifact_id: str,
    body: UpdateArtifactRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Partially update an artifact or container."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", store_db)

    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # Strip immutable fields from context updates (schema-driven mutability)
    context = _strip_immutable_context_fields(doc, body.context)

    from mantle.services import workspace_service

    container_id = doc.get("collection_id")
    if not container_id:
        # Top-level container artifact (workspace/collection) — no parent collection_id.
        updated = workspace_service.update_workspace(
            store_db,
            auth.user_id,
            artifact_id,
            name=body.name,
            description=body.description,
            context=context,
        )
        return updated.to_dict()

    updated = workspace_service.update_artifact(
        store_db,
        auth.user_id,
        container_id,
        artifact_id,
        context=context,
        content=body.content,
        state=body.state,
        content_type=body.content_type,
    )
    return updated.to_dict()


# ---------- DELETE /artifacts/{artifact_id} ----------

@router.delete("/{artifact_id}", status_code=status.HTTP_200_OK)
async def delete_artifact(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Delete or archive an artifact."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "delete", store_db)

    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    from mantle.services import workspace_service
    container_id = doc.get("collection_id")
    if not container_id:
        raise HTTPException(status_code=500, detail="Artifact missing collection_id")

    workspace_service.delete_artifact(store_db, auth.user_id, container_id, artifact_id)
    return {"id": artifact_id, "deleted": True}


@router.post("/{artifact_id}/remove", status_code=status.HTTP_200_OK)
async def remove_artifact_from_container_endpoint(
    artifact_id: str,
    body: RemoveItemRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Detach an artifact from a container without hard-deleting the root."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, body.container_id, "evict", store_db)

    from mantle.services import workspace_service

    artifact = workspace_service.remove_artifact_from_container(
        store_db,
        auth.user_id,
        body.container_id,
        artifact_id,
    )
    return {"id": artifact.id, "removed": True, "container_id": body.container_id}


# ---------- POST /artifacts/search — Search ----------

@router.post("/search", response_model=ArtifactSearchResponse)
async def search_artifacts(
    body: ArtifactSearchRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Search across accessible artifacts.

    Supports the same query syntax as the legacy ``POST /search`` endpoint:
    +term (AND), !term (exclude), ~term (semantic), ="phrase" (exact),
    field:value filters, and @hybrid:on/off control.

    Scope can be narrowed with ``scope`` (list of container IDs).
    """
    user_id = auth.user_id
    bearer_grant = auth.bearer_grant
    api_key_grants = auth.grants if auth.principal_type == "api_key" else []

    if not user_id and not bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    if not (body.query_text and body.query_text.strip()):
        raise HTTPException(
            status_code=400,
            detail="query_text is required",
        )

    # Resolve explicit container scope when body.scope is provided.
    # A workspace IS a collection — no distinction needed.
    #
    # Scope precedence:
    # 1. Explicit body.scope — user chose specific containers to search.
    # 2. API-key principal — restrict to the key's authorized resources
    #    (the key has narrower access than the owning user's full lightcone).
    # 3. Bearer-only access (no user_id) — restrict to the bearer's resource.
    # 4. None — accessor runs the full light-cone for the authenticated user.
    scope: Optional[List[str]] = None

    if body.scope:
        col_ids = [cid for cid in body.scope if _artifact_exists(store_db, cid)]
        scope = col_ids or None
    elif auth.principal_type == "api_key" and api_key_grants:
        api_scope = [
            g.resource_id for g in api_key_grants
            if getattr(g, "can_read", False) and g.resource_id
        ]
        scope = api_scope or None
    elif not user_id and bearer_grant and getattr(bearer_grant, "can_read", False):
        scope = [bearer_grant.resource_id] if bearer_grant.resource_id else None

    # Build and execute search query.
    from mantle.search.types import SearchQuery

    query = SearchQuery(
        query_text=body.query_text or "",
        query_embedding=None,
        user_id=user_id or "",
        scope=scope,
        use_hybrid=body.use_hybrid,
        from_=body.from_,
        size=body.size,
        sort=body.sort or "relevance",
        highlight=body.highlight,
    )

    # MANTLE-SSE is the canonical search backend after Step 2.6.9.
    # the legacy lexical index is retired; the legacy SearchAccessor / MantleSearchAccessor
    # are gone. If SSE prerequisites (Oracle, S3, the lattice) aren't satisfied,
    # search returns 503 — there's no plaintext fallback by design.
    from mantle.search.mantle.wiring import VALID_SEGMENTS, build_sse_search_accessor
    segment = (body.state or "committed").lower()
    if segment not in VALID_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"state must be one of {', '.join(VALID_SEGMENTS)}",
        )
    accessor = build_sse_search_accessor(store_db, segment=segment)
    if accessor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Encrypted search is not available — Oracle, S3, or the lattice "
                "prerequisite missing. Check platform/key_manager + "
                "content_service initialization."
            ),
        )

    try:
        result = accessor.search(query)
    except Exception as e:
        logger.error("Artifact search error: %s", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return ArtifactSearchResponse(
        hits=[
            SearchHitResponse(
                id=hit.doc_id,
                score=hit.score,
                root_id=hit.root_id,
                version_id=hit.version_id,
                collection_id=hit.collection_id,
                title=hit.title or None,
                description=hit.description or None,
                content=(hit.content or "")[:500] or None,
                tags=hit.tags or None,
                highlights=hit.highlights,
            )
            for hit in result.hits
        ],
        total=result.total,
        query_text=query.query_text,
        parsed_query=str(result.parsed_query),
        corrections=result.corrections,
        used_hybrid=result.used_hybrid,
        **{"from": body.from_},
        size=body.size,
    )


# ── /activate: REMOVED 2026-07-30 ────────────────────────────────────────────
# "Native embedding interaction" — it accepted a caller-supplied `embedding` +
# `model_id`, and echoed the raw carrier vector back. That is an embed/score
# service, and it is BYOK by another name: a caller POSTing a vector lit the
# whole vector arm with no provider configured.
#
# Removed ENTIRELY rather than 501/503, per the standing ruling on `/coherence`
# and `/embed`: an observer does not offer "embed this" or "score this" as a
# service. No caller existed in any repo. No-models rule, universal incl. BYOK.




# =============================================================================
# Specialized Request Models (defined early so static-path endpoints can use them)
# =============================================================================

class UploadInitiateRequest(BaseModel):
    """Initiate an S3 upload for an artifact."""
    filename: str
    content_type: str
    size: int
    order_key: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class UploadStatusRequest(BaseModel):
    """Update upload progress/completion."""
    status: Optional[str] = None
    progress: Optional[float] = None
    parts: Optional[List[Dict[str, Any]]] = None
    context_patch: Optional[Dict[str, Any]] = None


class ReorderRequest(BaseModel):
    """Reorder artifacts in a workspace."""
    ordered_ids: List[str]
    order_version: Optional[int] = None


class MoveArtifactRequest(BaseModel):
    """Move an artifact to a different workspace."""
    target_container_id: str


class BatchFetchRequest(BaseModel):
    """Batch fetch artifacts by IDs."""
    artifact_ids: List[str]


# =============================================================================
# Batch Operations (static path — registered before /{id} sub-paths)
# =============================================================================

# ---------- POST /artifacts/batch ----------

@router.post("/batch")
async def batch_fetch_artifacts(
    body: BatchFetchRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Batch fetch artifacts by IDs across all containers."""
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    results = []
    for aid in body.artifact_ids:
        doc = _find_artifact(store_db, aid)
        if not doc:
            continue

        # Verify read access silently — skip inaccessible artifacts.
        try:
            check_access(auth, aid, "read", store_db)
        except HTTPException:
            continue

        results.append(_normalize_artifact_doc(doc))

    return {"artifacts": results}


# =============================================================================
# Upload Endpoints
# =============================================================================

# ---------- POST /artifacts/{artifact_id}/upload-initiate ----------

@router.post("/{artifact_id}/upload-initiate")
async def upload_initiate(
    artifact_id: str,
    body: UploadInitiateRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Initiate an S3 upload for an artifact.

    The artifact_id here is the *container* (workspace) the upload belongs to.
    Delegates to workspace_service.initiate_upload_and_create_artifact().
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "create", store_db)

    from mantle.services.workspace_service import initiate_upload_and_create_artifact

    try:
        out, artifact = initiate_upload_and_create_artifact(
            db=store_db,
            user_id=auth.user_id,
            workspace_id=artifact_id,
            filename=body.filename,
            content_type=body.content_type,
            size=body.size,
            order_key=body.order_key,
            context=body.context,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload initiate failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload initiation failed: {exc}")

    return {
        **out,
        "artifact": artifact.to_dict() if artifact is not None else None,
    }


# ---------- PATCH /artifacts/{artifact_id}/upload-status ----------

@router.patch("/{artifact_id}/upload-status")
async def upload_status(
    artifact_id: str,
    body: UploadStatusRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Update upload progress or mark complete/failed."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", store_db)

    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    workspace_id = doc.get("collection_id")
    if not workspace_id:
        raise HTTPException(status_code=500, detail="Artifact missing collection_id")


    from mantle.services.workspace_service import update_upload_status as svc_update_upload

    try:
        result = svc_update_upload(
            db=store_db,
            user_id=auth.user_id,
            workspace_id=workspace_id,
            upload_id=artifact_id,
            status_value=body.status or "uploading",
            progress=body.progress,
            parts=body.parts,
            context_patch=body.context_patch,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload status update failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload status update failed: {exc}")

    return result.to_dict() if hasattr(result, "to_dict") else result


# ---------- GET /artifacts/{artifact_id}/multipart-part-url ----------

@router.get("/{artifact_id}/multipart-part-url")
async def multipart_part_url(
    artifact_id: str,
    part_number: int = Query(..., ge=1),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Superseded. Presigned multipart parts upload directly to the object store,
    bypassing Mantle — which cannot envelope-encrypt them. Content is now encrypted
    on Mantle's byte path via the proxied `PUT /artifacts/{id}/content` upload
    (see `upload-initiate`, which returns that URL). Chunked proxied upload is a
    post-MVP enhancement."""
    raise HTTPException(
        status_code=409,
        detail="Presigned multipart upload is disabled (content must be encrypted on "
               "Mantle's byte path). Use the proxied PUT /artifacts/{id}/content upload.",
    )


# ---------- GET /artifacts/{artifact_id}/content-url ----------

@router.get("/{artifact_id}/content-url")
async def content_url(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Get a signed content URL for an artifact's stored content."""
    check_access(auth, artifact_id, "read", store_db)

    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    context_raw = doc.get("context")
    ctx: Dict[str, Any] = {}
    if context_raw:
        try:
            ctx = json.loads(context_raw) if isinstance(context_raw, str) else context_raw
        except (json.JSONDecodeError, TypeError):
            pass

    content_key = ctx.get("content_key")
    if not content_key:
        raise HTTPException(status_code=404, detail="No downloadable content for this artifact")

    # S3 is invisible to callers and holds only ciphertext, so we no longer hand out
    # a presigned S3 URL (a direct S3 fetch would return undecryptable ciphertext).
    # Point callers at Mantle's proxied content endpoint, which decrypts on its byte path.
    return {"url": f"/artifacts/{artifact_id}/content"}


@router.get("/{artifact_id}/content")
async def get_artifact_content(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Proxied content download — Mantle streams the artifact's stored bytes,
    **decrypting on the byte path**. The object store never yields plaintext and is
    invisible to callers (no presigned S3 URL). Requires read access."""
    check_access(auth, artifact_id, "read", store_db)
    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    ctx: Dict[str, Any] = {}
    craw = doc.get("context")
    if craw:
        try:
            ctx = json.loads(craw) if isinstance(craw, str) else craw
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    content_key = ctx.get("content_key")
    if not content_key:
        raise HTTPException(status_code=404, detail="No downloadable content for this artifact")

    owner_id = doc.get("created_by")
    from mantle.services.content_service import get_bytes_decrypted
    from fastapi import Response
    try:
        data = get_bytes_decrypted(content_key, owner_id)
    except Exception as exc:
        logger.error("content download failed for %s: %s", artifact_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read content")

    headers = {}
    fn = ctx.get("filename")
    if fn:
        headers["Content-Disposition"] = f'attachment; filename="{fn}"'
    return Response(content=data, media_type=ctx.get("content_type") or "application/octet-stream", headers=headers)


@router.put("/{artifact_id}/content")
async def put_artifact_content(
    artifact_id: str,
    request: Request,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Proxied content upload — Mantle receives the bytes and **encrypts them on the
    byte path** before writing to the object store (no presigned S3 PUT; storage never
    receives plaintext). Requires update access. This is the target of `upload-initiate`."""
    check_access(auth, artifact_id, "update", store_db)
    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    ctx: Dict[str, Any] = {}
    craw = doc.get("context")
    if craw:
        try:
            ctx = json.loads(craw) if isinstance(craw, str) else craw
        except (json.JSONDecodeError, TypeError):
            ctx = {}
    content_key = ctx.get("content_key") or f"artifacts/{artifact_id}.content"
    owner_id = doc.get("created_by")
    ctype = request.headers.get("content-type") or ctx.get("content_type") or "application/octet-stream"

    body = await request.body()
    from mantle.services.content_service import put_bytes_encrypted
    try:
        put_bytes_encrypted(content_key, body, ctype, owner_id)
    except Exception as exc:
        logger.error("content upload failed for %s: %s", artifact_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to store content")
    return {"stored": True, "size": len(body), "content_key": content_key}


# =============================================================================
# Ordering / Move Endpoints
# =============================================================================

# ---------- PATCH /artifacts/{artifact_id}/children/order ----------

@router.patch("/{artifact_id}/children/order")
async def reorder_children(
    artifact_id: str,
    body: ReorderRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Reorder children of any artifact (any container)."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", store_db)

    if not _artifact_exists(store_db, artifact_id):
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Resolve version-ids or root-ids → root-ids; reorder edges.
    ordered_roots: List[str] = []
    for aid in body.ordered_ids:
        a = store.get_artifact(store_db, aid)
        if a:
            ordered_roots.append(a.root_id)
    store.reorder_collection_artifacts(store_db, artifact_id, ordered_roots)

    return {"order_version": 0}


# ---------- POST /artifacts/{artifact_id}/revert — Phase D.1 dedicated route ----------

@router.post("/{artifact_id}/revert")
async def revert_artifact_endpoint(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Restore the artifact's last committed version, discarding the draft delta.

    Phase D.1 carves this out as a distinct endpoint instead of routing through
    `op/revert`. Revert touches version history (it doesn't just flip a state
    field) so it warrants its own verb. If the artifact has no committed
    version yet, returns `204 No Content` per the design doc's "no-op" rule.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", store_db)

    doc = _find_artifact(store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    workspace_id = doc.get("collection_id") or doc.get("_key") or ""

    from mantle.services.workspace_service import revert_artifact

    try:
        result = revert_artifact(
            workspace_db=store_db,
            collection_db=store_db,
            user_id=auth.user_id,
            workspace_id=workspace_id,
            artifact_id=artifact_id,
        )
    except HTTPException:
        raise

    if result is None:
        # No committed version exists — revert is a no-op per the design.
        from fastapi import Response
        return Response(status_code=204)
    return result.to_dict()


# =============================================================================
# Container Metadata
# =============================================================================

# ---------- GET /artifacts/{container_id}/commits ----------

@router.get("/{container_id}/commits")
async def list_commits(
    container_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List commits for a collection container."""
    check_access(auth, container_id, "read", store_db)

    if not _artifact_exists(store_db, container_id):
        raise HTTPException(status_code=400, detail="Commits only available for collections")

    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    from mantle.services.collection_service import get_commits_for_collection

    try:
        commits = get_commits_for_collection(store_db, auth.user_id, container_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list commits: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list commits: {exc}")

    return {
        "commits": [
            {
                "id": getattr(c, "id", None),
                "collection_id": getattr(c, "collection_id", container_id),
                "message": getattr(c, "message", None),
                "author_id": getattr(c, "author_id", None),
                "created_time": getattr(c, "created_time", None),
                "adds": getattr(c, "adds", []),
                "removes": getattr(c, "removes", []),
            }
            for c in commits
        ],
    }


@router.get("/{artifact_id}/access-log")
async def get_artifact_access_log(
    artifact_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    result: Optional[str] = Query(None, description="filter: allowed | denied"),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """The artifact's own access history — who touched it, when, allowed or denied
    (the access-audit "force"). Requires ``admin`` on the artifact; reading it is
    itself witnessed (recursively) via the same check_access gate."""
    check_access(auth, artifact_id, "admin", store_db)
    from mantle.services import audit_service
    events = audit_service.get_artifact_access_log(
        store_db, artifact_id, limit=limit, offset=offset, result=result
    )
    return {"artifact_id": artifact_id, "count": len(events), "events": events}

