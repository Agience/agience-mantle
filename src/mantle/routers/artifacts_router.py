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

from arango.database import StandardDatabase
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic.functional_serializers import SerializerFunctionWrapHandler

from services.dependencies import get_arango_db
import db.arango as arango
from db.arango import has_children as db_has_children, count_children as db_count_children
from services.dependencies import (
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
    """Search across accessible artifacts.

    Provide exactly one of ``query_text`` (text → embedded + hybrid BM25/kNN) or
    ``embedding`` (a raw query vector → kNN directly, skipping the text-embed
    step — "embedding activation" for callers that already hold a vector).
    """
    model_config = ConfigDict(populate_by_name=True)

    query_text: Optional[str] = None
    embedding: Optional[List[float]] = None      # raw query vector (XOR query_text)
    scope: Optional[List[str]] = None           # container IDs to restrict
    state: str = "committed"                     # index segment: committed (default) | draft | archived
    content_types: Optional[List[str]] = None
    use_hybrid: Optional[bool] = None
    aperture: Optional[float] = None
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

def _artifact_exists(db: StandardDatabase, artifact_id: str) -> bool:
    """Return True if artifact_id refers to an existing artifact document."""
    try:
        coll = db.collection(_COLL_ARTIFACTS)
        doc = coll.get(artifact_id)
        return doc is not None
    except Exception:
        return False



def _find_artifact(db: StandardDatabase, artifact_id: str) -> Optional[dict]:
    """Locate an artifact in the unified store.

    First resolves builtin server short-names ("astra", "verso", etc.) to their
    stable UUID via the server registry so callers can use human-readable names.
    Then tries exact _key lookup; if not found, resolves by root_id (operation
    routes commonly receive root_id values for built-in server artifacts).
    Archived artifacts return None.
    """
    # Resolve builtin server names to their stable bootstrap UUID.
    from services import server_registry as _server_registry
    resolved_id = _server_registry.get_id(artifact_id)
    if resolved_id:
        artifact_id = resolved_id

    try:
        coll = db.collection(_COLL_ARTIFACTS)
        doc = coll.get(artifact_id)
        if doc and doc.get("state") != "archived":
            return doc
    except Exception:
        logger.warning("_find_artifact: key lookup failed for %r", artifact_id, exc_info=True)

    # Resolve stable root IDs to the newest non-archived version row.
    try:
        cursor = db.aql.execute(
            """
            FOR a IN @@col
              FILTER a.root_id == @root_id
                AND a.state != "archived"
              SORT a.modified_time DESC
              LIMIT 1
              RETURN a
            """,
            bind_vars={"@col": _COLL_ARTIFACTS, "root_id": artifact_id},
        )
        doc = next(iter(cursor), None)
        if doc:
            return doc
    except Exception:
        logger.warning("_find_artifact: root_id scan failed for %r", artifact_id, exc_info=True)

    return None


def _normalize_artifact_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an artifact document for API responses.

    Sets defaults for missing fields and strips ArangoDB internal keys.
    """
    normalized = dict(doc)

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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    from services.dependencies import _ACTION_FLAG_MAP

    flag_attr = _ACTION_FLAG_MAP.get(action)
    if flag_attr is None:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    from search.mantle.lightcone import LightConeResolver

    resolver = LightConeResolver(arango_db)

    # First-login provisioning is keyed on READ access (the baseline seed grant
    # set). A user with nothing readable has not yet been granted the platform
    # seed collections — provision them now (idempotent, safe on every startup
    # after a factory reset). Always resolved against "read", never the requested
    # action, so e.g. ?action=create does not retrigger provisioning for users
    # who legitimately have no create grants.
    read_authorized: Set[str] = resolver.resolve(auth.user_id, "read") if auth.user_id else set()
    if auth.user_id and not read_authorized:
        try:
            from services.seed_provisioning import provision_user
            # Capture profile + tenant from the token (external-IdP logins carry
            # email/name/issuer); platform users pass None and are unaffected.
            provision_user(
                arango_db,
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
        doc = _find_artifact(arango_db, aid)
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
    arango_db: StandardDatabase = Depends(get_arango_db),
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
    return await _default_create_artifact(body, auth, arango_db)


async def _default_create_artifact(
    body: Dict[str, Any],
    auth: AuthContext,
    arango_db: Any,
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

    from services import workspace_service

    # Top-level: no parent to authorize against; the creator owns it.
    if not parsed.container_id:
        entity = workspace_service.create_container(
            db=arango_db,
            user_id=auth.user_id,
            content_type=parsed.content_type,
            name=parsed.name,
            context=context_str or "",
            content=parsed.content or "",
            description=parsed.description,
        )
        return entity.to_dict()

    # Into a collection — the caller needs create/Add permission on it.
    check_access(auth, parsed.container_id, "create", arango_db)
    if not _artifact_exists(arango_db, parsed.container_id):
        raise HTTPException(status_code=404, detail="Container not found")

    # source_artifact_id -> LINK an existing artifact in (edge only), no new artifact.
    if parsed.source_artifact_id:
        return _link_source_artifact(arango_db, parsed)

    entity = workspace_service.create_workspace_artifact(
        db=arango_db,
        user_id=auth.user_id,
        workspace_id=parsed.container_id,
        context=context_str or "",
        content=parsed.content or "",
        content_type=parsed.content_type,
        name=parsed.name,
    )
    return entity.to_dict()


def _link_source_artifact(
    arango_db: Any,
    parsed: CreateArtifactRequest,
) -> Dict[str, Any]:
    """Link an existing artifact into a container instead of creating a duplicate."""
    from db.arango import (
        get_artifact as _get_artifact,
        get_latest_committed_artifact,
        add_artifact_to_collection,
    )

    source = _get_artifact(arango_db, parsed.source_artifact_id)
    if not source:
        source = get_latest_committed_artifact(arango_db, parsed.source_artifact_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source artifact not found")
    root_id = source.root_id or source.id
    add_artifact_to_collection(arango_db, parsed.container_id, root_id)
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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Read a single artifact by ID."""
    check_access(auth, artifact_id, "read", arango_db)

    doc = _find_artifact(arango_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # Normalize ArangoDB internal keys.
    doc.pop("_id", None)
    doc.pop("_rev", None)
    if "_key" in doc:
        doc.setdefault("id", doc.pop("_key"))

    # Inject computed child-containment fields.
    root_id = doc.get("root_id") or doc.get("id") or artifact_id
    doc["has_children"] = db_has_children(arango_db, root_id)
    doc["child_count"] = db_count_children(arango_db, root_id) if doc["has_children"] else 0

    return doc


# ---------- GET /artifacts/{artifact_id}/embedding — Raw stored vector(s) ----------

@router.get("/{artifact_id}/embedding")
async def get_artifact_embedding(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Return the artifact's stored MANTLE embedding(s) — raw vectors out, no text.

    The inverse of the `embedding` inputs on /search and /activate. `embedding` is
    the primary (first) chunk vector; `chunks` carries all of them (content that was
    chunked yields several). Caller must have read on the artifact.
    """
    check_access(auth, artifact_id, "read", arango_db)
    doc = _find_artifact(arango_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    from entities.artifact import Artifact as _Artifact
    from search.ingest.pipeline_unified import get_artifact_embeddings

    art = _Artifact.from_dict(doc)
    chunks = get_artifact_embeddings(art, art.collection_id)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No stored embedding (vector arm off, or nothing indexed for this artifact)",
        )
    return {
        "artifact_id": art.root_id or art.id,
        "model_id": chunks[0].get("model_id"),
        "embedding": chunks[0].get("embedding"),
        "chunks": [
            {"chunk_id": c.get("chunk_id"), "embedding": c.get("embedding")}
            for c in chunks
        ],
    }


# ---------- GET /artifacts/{artifact_id}/children — List children ----------

@router.get("/{artifact_id}/children")
async def list_children(
    artifact_id: str,
    request: Request,
    content_type: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    arango_db: StandardDatabase = Depends(get_arango_db),
    auth: AuthContext = Depends(get_auth),
):
    """List children of any artifact (universal container model).

    Optional filters:
    - content_type: filter children by their content_type
    - workspace_id: include draft children from this workspace

    Each child is enriched with `committed_collection_ids` — the set of committed
    containers it currently appears in.
    """
    check_access(auth, artifact_id, "read", arango_db)

    # A draft is workspace-private. Only surface drafts linked into this container
    # when the caller passes a workspace_id they can READ — then drafts homed in
    # THAT workspace (the caller's own) show here, never anyone else's.
    draft_workspace_id: Optional[str] = None
    if workspace_id:
        try:
            check_access(auth, workspace_id, "read", arango_db)
            draft_workspace_id = workspace_id
        except HTTPException:
            draft_workspace_id = None  # no access → don't include its drafts

    children = arango.list_collection_artifacts(
        arango_db, artifact_id, draft_workspace_id=draft_workspace_id
    )

    # Filter out operator edges (relationship != null means non-containment)
    children = [c for c in children if not c.get("relationship")]

    # Optional content_type filter
    if content_type:
        children = [c for c in children if c.get("content_type") == content_type]

    # Enrich with committed_collection_ids (structural — pure edge traversal)
    from entities.artifact import Artifact as ArtifactEntity
    from services.collection_service import attach_committed_collection_ids
    entities = [ArtifactEntity.from_dict(c) for c in children]
    attach_committed_collection_ids(arango_db, entities)
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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Partially update an artifact or container."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", arango_db)

    doc = _find_artifact(arango_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # Strip immutable fields from context updates (schema-driven mutability)
    context = _strip_immutable_context_fields(doc, body.context)

    from services import workspace_service

    container_id = doc.get("collection_id")
    if not container_id:
        # Top-level container artifact (workspace/collection) — no parent collection_id.
        updated = workspace_service.update_workspace(
            arango_db,
            auth.user_id,
            artifact_id,
            name=body.name,
            description=body.description,
            context=context,
        )
        return updated.to_dict()

    updated = workspace_service.update_artifact(
        arango_db,
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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Delete or archive an artifact."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "delete", arango_db)

    doc = _find_artifact(arango_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    from services import workspace_service
    container_id = doc.get("collection_id")
    if not container_id:
        raise HTTPException(status_code=500, detail="Artifact missing collection_id")

    workspace_service.delete_artifact(arango_db, auth.user_id, container_id, artifact_id)
    return {"id": artifact_id, "deleted": True}


@router.post("/{artifact_id}/remove", status_code=status.HTTP_200_OK)
async def remove_artifact_from_container_endpoint(
    artifact_id: str,
    body: RemoveItemRequest,
    auth: AuthContext = Depends(get_auth),
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Detach an artifact from a container without hard-deleting the root."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, body.container_id, "evict", arango_db)

    from services import workspace_service

    artifact = workspace_service.remove_artifact_from_container(
        arango_db,
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
    arango_db: StandardDatabase = Depends(get_arango_db),
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

    has_text = bool(body.query_text and body.query_text.strip())
    has_embedding = bool(body.embedding)
    if has_text == has_embedding:
        raise HTTPException(
            status_code=400,
            detail="provide exactly one of 'query_text' or 'embedding'",
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
        col_ids = [cid for cid in body.scope if _artifact_exists(arango_db, cid)]
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
    from search.types import SearchQuery

    query = SearchQuery(
        query_text=body.query_text or "",
        query_embedding=body.embedding,
        user_id=user_id or "",
        scope=scope,
        use_hybrid=body.use_hybrid,
        aperture=body.aperture if body.aperture is not None else 0.75,
        from_=body.from_,
        size=body.size,
        sort=body.sort or "relevance",
        highlight=body.highlight,
    )

    # MANTLE-SSE is the canonical search backend after Step 2.6.9.
    # OpenSearch is retired; the legacy SearchAccessor / MantleSearchAccessor
    # are gone. If SSE prerequisites (Oracle, S3, Arango) aren't satisfied,
    # search returns 503 — there's no plaintext fallback by design.
    from search.mantle.wiring import VALID_SEGMENTS, build_sse_search_accessor
    segment = (body.state or "committed").lower()
    if segment not in VALID_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"state must be one of {', '.join(VALID_SEGMENTS)}",
        )
    accessor = build_sse_search_accessor(arango_db, segment=segment)
    if accessor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Encrypted search is not available — Oracle, S3, or Arango "
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


class ActivateRequest(BaseModel):
    """Native embedding interaction — present a signal, see its grounded meaning."""
    text: Optional[str] = None
    embedding: Optional[List[float]] = None
    model_id: Optional[str] = None
    scope: Optional[List[str]] = None
    top_k: int = 10            # nearest neighbours to return when act=True
    top_anchors: int = 8       # anchors reported in the native activation
    act: bool = True           # also return nearest authorized neighbours


@router.post("/activate")
async def activate(
    body: ActivateRequest,
    auth: AuthContext = Depends(get_auth),
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Activate a signal in the meaning manifold (canonical plan §7).

    Feed-forward, one shot: present text *or* an embedding → reconcile to the
    native language (the grounded anchors it activates) + its density-zoom layer
    → optionally return the nearest authorized artifacts. The embedding is the
    carrier; classic ``/search`` and this share the same manifold.
    """
    user_id = auth.user_id
    if not user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    has_text = bool(body.text and body.text.strip())
    has_embedding = bool(body.embedding)
    if has_text == has_embedding:
        raise HTTPException(
            status_code=400, detail="provide exactly one of 'text' or 'embedding'",
        )

    # Resolve the carrier embedding.
    if has_embedding:
        vec = body.embedding
        model_id = body.model_id
    else:
        from embeddings import Embeddings, model_id as emb_model_id
        vectors = Embeddings()([body.text])
        if not vectors or not vectors[0]:
            raise HTTPException(
                status_code=503,
                detail="Embeddings provider unavailable — cannot activate text.",
            )
        vec = vectors[0]
        model_id = body.model_id or emb_model_id()

    # Native activation (geometry; non-authorizing — canonical plan §1).
    from search.anchors.activate import activate_vector
    try:
        activation = activate_vector(vec, model_id=model_id, top_anchors=body.top_anchors)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Act: nearest authorized artifacts via the canonical encrypted search.
    neighbors: List[dict] = []
    if body.act:
        from search.mantle.wiring import build_sse_search_accessor
        from search.types import SearchQuery
        accessor = build_sse_search_accessor(arango_db)
        if accessor is not None:
            query = SearchQuery(
                query_text="",
                query_embedding=vec,
                user_id=user_id or "",
                scope=body.scope or None,
                use_hybrid=False,
                size=body.top_k,
            )
            try:
                result = accessor.search(query)
                neighbors = [
                    {
                        "id": h.doc_id,
                        "root_id": h.root_id,
                        "collection_id": h.collection_id,
                        "score": h.score,
                        "title": h.title or None,
                    }
                    for h in result.hits
                ]
            except Exception as exc:
                logger.warning("activate: neighbour search failed: %s", exc)

    # Echo the raw carrier vector so text->embedding (or embedding->embedding) is a
    # single call — "receive an embedding directly".
    return {
        "activation": activation,
        "neighbors": neighbors,
        "model_id": model_id,
        "vector": [float(x) for x in vec],
    }


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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Batch fetch artifacts by IDs across all containers."""
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    results = []
    for aid in body.artifact_ids:
        doc = _find_artifact(arango_db, aid)
        if not doc:
            continue

        # Verify read access silently — skip inaccessible artifacts.
        try:
            check_access(auth, aid, "read", arango_db)
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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Initiate an S3 upload for an artifact.

    The artifact_id here is the *container* (workspace) the upload belongs to.
    Delegates to workspace_service.initiate_upload_and_create_artifact().
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "create", arango_db)

    from services.workspace_service import initiate_upload_and_create_artifact

    try:
        out, artifact = initiate_upload_and_create_artifact(
            db=arango_db,
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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Update upload progress or mark complete/failed."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", arango_db)

    doc = _find_artifact(arango_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    workspace_id = doc.get("collection_id")
    if not workspace_id:
        raise HTTPException(status_code=500, detail="Artifact missing collection_id")


    from services.workspace_service import update_upload_status as svc_update_upload

    try:
        result = svc_update_upload(
            db=arango_db,
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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Get a presigned URL for uploading a specific multipart part."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", arango_db)

    doc = _find_artifact(arango_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # Read upload context from artifact to get s3_key and multipart_id.
    context_raw = doc.get("context")
    ctx: Dict[str, Any] = {}
    if context_raw:
        try:
            ctx = json.loads(context_raw) if isinstance(context_raw, str) else context_raw
        except (json.JSONDecodeError, TypeError):
            pass

    upload = ctx.get("upload", {})
    s3_key = upload.get("s3_key") or ctx.get("content_key")
    multipart_id = upload.get("multipart_id")

    if not s3_key or not multipart_id:
        raise HTTPException(status_code=400, detail="No active multipart upload for this artifact")

    from services.content_service import generate_multipart_part_url as gen_part_url

    try:
        url = gen_part_url(s3_key, multipart_id, part_number)
    except Exception as exc:
        logger.error("Multipart part URL generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate part URL: {exc}")

    return {"url": url, "part_number": part_number}


# ---------- GET /artifacts/{artifact_id}/content-url ----------

@router.get("/{artifact_id}/content-url")
async def content_url(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Get a signed content URL for an artifact's stored content."""
    check_access(auth, artifact_id, "read", arango_db)

    doc = _find_artifact(arango_db, artifact_id)
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

    filename = ctx.get("filename")
    content_type = ctx.get("content_type")

    from services.content_service import generate_signed_url

    # When the caller is a server acting via delegation (actor field set),
    # use the server-facing endpoint so the URL is reachable from the
    # server's network context.
    use_server_facing = bool(getattr(auth, "actor", None))

    try:
        url = generate_signed_url(content_key, filename=filename, content_type=content_type, server_facing=use_server_facing)
    except Exception as exc:
        logger.error("Content URL generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate content URL: {exc}")

    return {"url": url}


# =============================================================================
# Ordering / Move Endpoints
# =============================================================================

# ---------- PATCH /artifacts/{artifact_id}/children/order ----------

@router.patch("/{artifact_id}/children/order")
async def reorder_children(
    artifact_id: str,
    body: ReorderRequest,
    auth: AuthContext = Depends(get_auth),
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Reorder children of any artifact (any container)."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", arango_db)

    if not _artifact_exists(arango_db, artifact_id):
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Resolve version-ids or root-ids → root-ids; reorder edges.
    ordered_roots: List[str] = []
    for aid in body.ordered_ids:
        a = arango.get_artifact(arango_db, aid)
        if a:
            ordered_roots.append(a.root_id)
    arango.reorder_collection_artifacts(arango_db, artifact_id, ordered_roots)

    return {"order_version": 0}


# ---------- POST /artifacts/{artifact_id}/revert — Phase D.1 dedicated route ----------

@router.post("/{artifact_id}/revert")
async def revert_artifact_endpoint(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """Restore the artifact's last committed version, discarding the draft delta.

    Phase D.1 carves this out as a distinct endpoint instead of routing through
    `op/revert`. Revert touches version history (it doesn't just flip a state
    field) so it warrants its own verb. If the artifact has no committed
    version yet, returns `204 No Content` per the design doc's "no-op" rule.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    check_access(auth, artifact_id, "update", arango_db)

    doc = _find_artifact(arango_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    workspace_id = doc.get("collection_id") or doc.get("_key") or ""

    from services.workspace_service import revert_artifact

    try:
        result = revert_artifact(
            workspace_db=arango_db,
            collection_db=arango_db,
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
    arango_db: StandardDatabase = Depends(get_arango_db),
):
    """List commits for a collection container."""
    check_access(auth, container_id, "read", arango_db)

    if not _artifact_exists(arango_db, container_id):
        raise HTTPException(status_code=400, detail="Commits only available for collections")

    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    from services.collection_service import get_commits_for_collection

    try:
        commits = get_commits_for_collection(arango_db, auth.user_id, container_id)
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

