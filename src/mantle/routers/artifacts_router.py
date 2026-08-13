# routers/artifacts_router.py
#
# Unified Artifact API — single REST surface for all artifact operations.
#
# Replaces per-container endpoints (workspaces, collections, agents, inbound,
# search) with a container-agnostic set of verbs:
#
#   POST   /artifacts                → Create (with `container_id` in the body, this
#                                      IS the container Add — there is no separate
#                                      add-to-container route)
#   GET    /artifacts/visible        → List what the caller's light cone reaches
#   GET    /artifacts/{id}           → Read
#   GET    /artifacts/{id}/children  → List a container's members
#   PATCH  /artifacts/{id}           → Update
#   DELETE /artifacts/{id}           → Delete
#   POST   /artifacts/{id}/remove    → Detach from one container (the root survives)
#   POST   /artifacts/recall         → Recall (ordered + hydrated; `candidates: true` for the
#                                      same narrowed set unordered — one handler, so both
#                                      modes resolve authorization identically)
#   POST   /artifacts/batch          → Batch fetch by IDs
#
# Specialized endpoints:
#   POST   /artifacts/{id}/upload-initiate      → Initiate an upload
#   PATCH  /artifacts/{id}/upload-status        → Update upload progress
#   GET    /artifacts/{id}/multipart-part-url   → Disabled (409); presigned parts bypass
#                                                 Mantle's encrypting byte path
#   GET    /artifacts/{id}/content-url          → Points at the proxied content route below
#   GET    /artifacts/{id}/content              → Proxied download (decrypts on the byte path)
#   PUT    /artifacts/{id}/content              → Proxied upload (encrypts on the byte path)
#   PATCH  /artifacts/{id}/children/order       → Reorder a container's children
#   POST   /artifacts/{id}/revert               → Restore the last committed version
#   POST   /artifacts/{id}/warm                 → Materialize a collection's latent artifacts
#   GET    /artifacts/{container_id}/commits    → List commits for collection
#   GET    /artifacts/{id}/access-log           → The artifact's own access history
#
# There is no move verb: an artifact's home is its membership edges, so moving one is
# `POST /artifacts` (link into the target) plus `POST /artifacts/{id}/remove`.
#
# Mantle stores artifacts as (content_type, context, content); content_type is a
# LABEL (not resolved to a type.json). Operation dispatch (/op) is an application
# concern and runs above this layer; one create-time dispatch remains here, for
# top-level container types.
#
# Real-time event subscription is handled by the unified /events WebSocket
# (see routers/events_router.py), not a per-container SSE endpoint.

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Literal, Optional, Set, Union

from mantle.db.store import Database
from mantle.search.beacon.density import dense_excerpt, dense_windows
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
    offload_sync,
)

logger = logging.getLogger(__name__)

# Every handler below is `async def`, and nothing in the store is awaitable — so a store call
# made directly from one holds the event loop for its whole duration. The list / read / search /
# create handlers and the byte path therefore run their store work through `offload_sync`, which
# hands a WHOLE operation to a worker thread (never half of one: a transaction may not span
# threads — see `db/seq.py`). Single indexed seeks are left on the loop; the hop costs
# more than the seek.


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

    #: A caller-chosen natural key for the thing this artifact is OF — `file:/repo/README.md`,
    #: `session:7c7bcb7b`. Supplying it makes the write **idempotent**: the id is derived from
    #: it (`services/artifact_identity`), so storing the same thing again UPDATES the artifact
    #: rather than creating a second one, and the caller keeps no id map to lose.
    #:
    #: Omit it and nothing changes — a fresh `uuid4`, exactly as before.
    identity: Optional[str] = None
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

    #: The semantic arm's ingress. Mantle never embeds, so the only way a vector
    #: reaches the vector arm is a writer handing one over on the write that produced
    #: the content it describes. ``space_id`` is mandatory alongside it — see
    #: `api/vectors.py`. Omit both and the write behaves exactly as before: the vector
    #: arm receives nothing and the artifact is lexical-only.
    vector: Optional[List[float]] = None
    space_id: Optional[str] = None


class UpdateArtifactRequest(BaseModel):
    """Partial update to an artifact or container."""
    context: Optional[str] = None
    content: Optional[str] = None
    state: Optional[str] = None
    content_type: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None

    #: Re-supply the vector when the content it describes changes. Same contract as on
    #: create: optional, but `space_id` comes with it when present.
    vector: Optional[List[float]] = None
    space_id: Optional[str] = None


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


class ArtifactRecallRequest(BaseModel):
    """Recall across accessible artifacts. ``query_text`` or ``vector`` is required.

    Unknown fields are IGNORED (pydantic default), not rejected — a client still
    sending the removed ``embedding``, ``aperture`` or ``use_hybrid`` gets a normal
    search, not a 422. None of them has any effect.
    """
    model_config = ConfigDict(populate_by_name=True)

    query_text: Optional[str] = None

    #: The semantic arm's query-side ingress, and the exact counterpart of ``vector``
    #: on :class:`CreateArtifactRequest`: a writer supplies the vector of what it
    #: stores, a reader supplies the vector of what it is looking for. Both are numbers
    #: computed elsewhere — Mantle stores, compares and ranks, and embeds neither of
    #: them — so both are validated by ``api/vectors.py`` and both require ``space_id``.
    #:
    #: It ACCOMPANIES ``query_text`` rather than replacing it: the lexical arm reads the
    #: text, the semantic arm reads the vector, and RRF fuses whatever each returns.
    #: Either may be sent alone — text alone is lexical recall, a vector alone is kNN.
    vector: Optional[List[float]] = None
    #: Names the embedding space ``vector``'s components live in. Required alongside it
    #: for the same reason it is required on a write: it is what makes two vectors
    #: comparable, and Mantle cannot infer it from the numbers.
    space_id: Optional[str] = None

    scope: Optional[List[str]] = None           # container IDs to restrict
    #: Which index SEGMENT to recall from — `committed` (default), `draft` or `archived`.
    #:
    #: Each is a separately keyed encrypted tree under its own object-storage prefix
    #: (`search/mantle/wiring._segment_prefixes`), selected when the accessor is built and
    #: before any query runs. That is why there is no `state:` query filter: a filter narrows
    #: a set of retrieved artifacts, and no draft is in the committed tree to be narrowed out
    #: of. `state:draft` in `query_text` is a 400 pointing back at this field.
    state: str = "committed"
    # No `content_types` field. It was declared here and read by nothing — the handler never
    # passed it on and `SearchQuery` had nowhere to put it — so it narrowed no recall, ever.
    # The one way to say it is the `content_type:` (alias `type:`) query filter, which does
    # narrow. Removing it costs a client nothing: unknown fields are ignored, so a request
    # still sending `content_types` gets exactly the search it got before.
    # No `aperture` field: it is never read; see mantle/search/types.py.
    # No `use_hybrid` field either, and for a sharper reason than "unread": there is nothing
    # left for it to switch. The lexical index NARROWS every recall and the cells RANK what
    # survives — neither is optional, so neither is selectable. What varies is whether a query
    # vector exists for the ranker, and ``vector`` says that already. ``ordering`` on the
    # response reports what actually happened.
    from_: int = 0
    size: int = 20
    #: Which ordering you want back.
    #:
    #: `recency` asks for most-recently-updated first and gets it, vector or no vector — and
    #: gets it without decrypting a cell, since the ordering is decided before the ranker runs.
    #: `relevance` (the default when unset) asks for the best ordering this recall can produce:
    #: a cosine when a query vector reaches the ranker, otherwise the query's own coverage —
    #: how many of its stems each hit carries. This is a REQUEST and cannot promise an outcome;
    #: read `ordering` on the response for what you got.
    sort: Optional[Literal["relevance", "recency"]] = None
    highlight: bool = True

    #: Return the NARROWED CANDIDATE SET instead of ordered, hydrated hits.
    #:
    #: This is the primitive external search flavors build on: they rank within the
    #: returned set and therefore can never widen access (MANTLE §1 holds by
    #: construction). It shares this endpoint rather than living at its own path
    #: because everything before the final call — light-cone resolution, grant-key
    #: scoping, segment validation, the `field:value` filter and the query's own terms —
    #: is identical for both. Same universe, no order and no hydration.
    #:
    #: Response shape is `{candidates, model_id}`, each candidate carrying `artifact_id`,
    #: `collection_id` and `principal_id`. THERE ARE NO SCORES ON IT. The former `sse_score` /
    #: `rrf_score` / `source` were a BM25 score, a rank-fusion constant's output and a
    #: which-arm-found-it flag; none of those quantities exists any more, so the keys are gone
    #: rather than null. `model_id` is `null`: nothing here retrieves by embedding.
    candidates: bool = False
    #: Candidates only: how many to return. The cut is by recency — the query-independent
    #: order — because choosing which candidates a budget keeps by anything the query said
    #: would be the ranking decision this mode declines to make. Ignored for ordered recall,
    #: which paginates with `from`/`size`.
    candidate_budget: int = 200

    @model_validator(mode="before")
    @classmethod
    def _accept_from_alias(cls, data):
        if isinstance(data, dict) and "from" in data and "from_" not in data:
            data = dict(data)
            data["from_"] = data.pop("from")
        return data


class RecallHitResponse(BaseModel):
    """One recalled artifact, with content fields for downstream consumers."""
    id: str
    #: What put this hit where it is, as a number. READ `ordering` TO KNOW WHICH KIND.
    #:
    #: `ordering: "semantic"` — the cosine that ranked it.
    #: `ordering: "coverage"` — the INTEGER COUNT of distinct query stems this artifact
    #: carries. It is not a relevance score: nothing weights it by how rare a term is, how
    #: often it occurs, or how long the field is, and a 2 means "two of these five stems" on
    #: one query and "two of these two" on another. Compare it across the hits of one response,
    #: never across responses, and never against a fixed threshold.
    #: `ordering: "recency"` — `null`. Those results are ordered by when they were last
    #: updated, so no number measured them against the query, and a zero or a rank would be
    #: something a client could threshold or re-sort on that would mean nothing.
    score: Optional[float] = None
    root_id: str
    version_id: str
    collection_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    highlights: Optional[Dict[str, List[str]]] = None


class ArtifactRecallResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hits: List[RecallHitResponse]
    total: int
    query_text: str
    parsed_query: Optional[str] = None
    #: The `field:value` filters that narrowed this result, canonically spelled.
    #:
    #: `parsed_query` is the whole parse and still includes the inert `@name:value` controls;
    #: this is the narrower, load-bearing statement — every entry was compiled into the
    #: predicate that cut the authorized artifact set before either retrieval arm ran. Nothing
    #: parses into a filter and then quietly fails to appear here: a filter this node cannot
    #: apply is a 400, so an empty list on a query containing `field:value` is impossible.
    applied_filters: List[str] = Field(default_factory=list)
    corrections: List[str] = Field(default_factory=list)
    ordering: Literal["semantic", "coverage", "recency"] = Field(
        description=(
            "What ordered these hits — what happened, not what was asked for. `semantic`: a "
            "query vector reached the ranker, it ranked the narrowed set by cosine, and each "
            "hit's `score` is that cosine. `coverage`: no cosine could, so the hits are "
            "ordered by how much of the query each one matched, and each hit's `score` is the "
            "INTEGER COUNT of distinct query stems it carries — not a relevance score, not "
            "normalised, and not comparable across queries. `recency`: neither applied, so the "
            "hits are most-recently-updated first and every `score` is `null`; two causes reach "
            "it and this does not separate them — `sort: \"recency\"` was requested, or the "
            "request carried no query terms to cover. A single-term query orders by `coverage` "
            "and comes back in recency order, because every hit matched the one term there was."
        ),
    )
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
    """Locate an artifact in the unified store by id.

    ``artifact_id`` is an id, never a name: an exact ``_key`` lookup first, then a
    ``root_id`` resolution to the newest non-archived version (operation routes commonly
    receive root ids). Nothing here maps a human-readable name onto an id — a caller that
    passes one matches neither read and gets ``None``, which every route turns into a 404
    rather than into some other artifact.

    Archived artifacts return None.
    """
    from mantle.db.backend import _decrypt_artifact_content as _decrypt_doc
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
    """A no-op: Mantle treats ``content_type`` as a label and does
    not resolve type definitions, so it does not enforce a type's
    ``context_schema`` mutability rules — field-mutability is an application
    concern, enforced above this layer. Returns the context unchanged."""
    return context


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _hydrate_batch(db: Database, artifact_ids: List[str]) -> Dict[str, dict]:
    """Load a page of artifacts by id — ``{id: normalized doc}``.

    Two passes, not one query. Pass one takes the cheap indexed ``_key`` read for every
    DISTINCT id; pass two resolves only what pass one missed, through the root-id lineage
    read — the half of :func:`_find_artifact` that costs a query per miss. So a page of
    ``n`` ids costs ``n`` seeks plus one lineage read per miss, and a duplicated id is
    read once.

    That is the shape every batch in this codebase has, and it is the store's shape, not
    a shortcut: ``collection_service.get_collection_artifacts_batch`` states it outright
    ("one lineage read per root ... there is no cross-root query that answers it in one
    pass"), and ``lattice_api.batch_get_collection_ids_for_roots`` — the keyed batch this
    mirrors — is likewise one edge read per key with the *derived* work collapsed across
    the set. There is no batched-by-id read to call: the ``db.store.ArtifactStore`` seam
    publishes ``get_artifact(id)`` and nothing plural, so a ``WHERE id IN (...)`` would
    have to be hand-rolled here against the vertex table. That is the probe-hand-rolling
    the store layer forbids; the fix is a plural read ON the seam
    (``ArtifactStore.get_many(ids)`` backed by a chunked ``SELECT id, doc FROM vertex
    WHERE id IN (...)``), and this function collapses to one call the day it exists.

    What bounds the cost meanwhile is the CALLER: ``list_visible`` pages the authorized
    set before calling here, so ``n`` is the page size (≤1000), not the caller's whole
    grant reach.

    Ids absent from the result did not resolve (or are archived); callers skip them.
    """
    out: Dict[str, dict] = {}
    if not artifact_ids:
        return out

    from mantle.db.backend import get_raw_artifact, find_newest_by_root
    from mantle.db.backend import _decrypt_artifact_content as _decrypt_doc

    # Ids only, exactly as `_find_artifact` reads them: the keyed read, then the lineage
    # read for whatever it missed. Deduplicated so a repeated id costs one seek.
    resolved: Dict[str, str] = {aid: aid for aid in dict.fromkeys(artifact_ids)}

    misses: List[str] = []
    for aid, key in resolved.items():
        try:
            doc = get_raw_artifact(db, key)
        except Exception:
            logger.warning("_hydrate_batch: key lookup failed for %r", key, exc_info=True)
            doc = None
        if doc and doc.get("state") != "archived":
            _decrypt_doc(doc)
            out[aid] = _normalize_artifact_doc(doc)
        else:
            misses.append(aid)

    for aid in misses:
        try:
            doc = find_newest_by_root(db, resolved[aid])
        except Exception:
            logger.warning("_hydrate_batch: root_id scan failed for %r", aid, exc_info=True)
            continue
        if doc:
            _decrypt_doc(doc)
            out[aid] = _normalize_artifact_doc(doc)
    return out


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/artifacts", tags=["Artifacts"])


# ---------- GET /artifacts/visible — list artifacts the caller can read ----------
#
# Browser UX needs "show me every workspace / collection I can see" without
# having to know a parent ID. /recall requires something to rank on — query text, a
# query vector, or both — and is for relevance-ranked queries; this is the flat-list
# affordance, scoped through the canonical
# LightConeResolver (same ACL path /recall uses internally).
@router.get(
    "/visible",
    summary="List artifacts the caller may act on",
    description=(
        "The flat-list affordance: everything reachable through the caller's light cone, "
        "without needing a parent id. Paginated over the authorized set in id order, and "
        "hydrated a page at a time — an unbounded list here grows with the caller's whole "
        "grant reach. An absent artifact may exist and simply not be visible."
    ),
)
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
    limit: int = Query(100, ge=1, le=1000, description="Page size."),
    offset: int = Query(0, ge=0, description="How many authorized artifacts to skip."),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    from mantle.attenuation import ACTIONS
    from mantle.entities.grant import mask_of

    if action not in ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    from mantle.search.mantle.lightcone import LightConeResolver

    resolver = LightConeResolver(store_db)

    # First-login provisioning is keyed on READ access (the baseline seed grant
    # set). A user with nothing readable has not yet been granted the platform
    # seed collections — provision them now (idempotent, safe on every startup
    # after a factory reset). Always resolved against "read", never the requested
    # action, so e.g. ?action=create does not retrigger provisioning for users
    # who legitimately have no create grants.
    # Each `resolve` is a light-cone BFS over the grant graph — the most expensive read this
    # router issues, and unbounded in the caller's reach — so it goes to a worker thread whole.
    read_authorized: Set[str] = (
        await offload_sync(resolver.resolve, auth.user_id, "read") if auth.user_id else set()
    )
    if auth.user_id and not read_authorized:
        try:
            from mantle.services.seed_provisioning import provision_user
            # Capture profile + tenant from the token (external-IdP logins carry
            # email/name/issuer); platform users pass None and are unaffected.
            # Offloaded as ONE call: it is a multi-write provisioning transaction, and
            # splitting it across threads would break gap-free `_seq` allocation.
            await offload_sync(
                provision_user,
                store_db,
                user_id=auth.user_id,
                email=getattr(auth, "email", None),
                name=getattr(auth, "name", None),
                tenant=getattr(auth, "authority", None),
            )
            read_authorized = await offload_sync(resolver.resolve, auth.user_id, "read")
            logger.info("First-login provisioning completed for user %s", auth.user_id)
        except Exception:
            logger.warning(
                "First-login provisioning failed for user %s (non-fatal)", auth.user_id, exc_info=True
            )

    if action == "read":
        authorized: Set[str] = set(read_authorized)
    else:
        authorized = (
            await offload_sync(resolver.resolve, auth.user_id, action) if auth.user_id else set()
        )
    # The bearer grant's own resource, if the grant actually authorizes this action.
    #
    # `mask_of(g).allows(action)` and not `getattr(g, flag_attr, False)`: the bare flag read
    # answers only "is the bit set", which is True for a DENY grant too — every deny grant
    # carries the bits naming what it denies. A deny-effect bearer key would therefore have
    # added its resource to the caller's visible set, which is audit S1 in the listing path.
    # One call now asks the whole question, through the same operator the light cone above
    # resolved with, so the two cannot drift apart.
    if (auth.bearer_grant and auth.bearer_grant.resource_id
            and mask_of(auth.bearer_grant).allows(action)):
        authorized.add(auth.bearer_grant.resource_id)

    # Page over the authorized ids BEFORE hydrating them. The set is the caller's whole
    # grant reach, so hydrating it entirely is work proportional to how much someone can
    # see rather than to how much they asked for. A stable sort makes the page well-defined:
    # a set has no order, so offset would otherwise index into a different list each call.
    page = sorted(authorized)[offset:offset + limit]

    # One hop for the whole page: `_hydrate_batch` is up to `limit` store reads, which is the
    # largest single block of blocking work in this router.
    hydrated = await offload_sync(_hydrate_batch, store_db, page)

    # The content_type filter applies WITHIN the page: it is a property of the document,
    # not of the grant, so it cannot be resolved before hydration and a filter-then-page
    # order would put the whole authorized set back through the loader. A filtered page may
    # therefore be shorter than `limit`; a short page does not mean the last page.
    results: list = []
    for aid in page:
        doc = hydrated.get(aid)
        if not doc:
            continue
        if content_type and doc.get("content_type") != content_type:
            continue
        results.append(doc)
    return results


# ---------- POST /artifacts — Create ----------

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Create an artifact",
    description=(
        "One create path for everything, because a collection is just an artifact with child "
        "edges. Omit `container_id` for a top-level artifact; supply it to also edge the new "
        "artifact into that collection. A `vector` + `space_id` pair rides the write into the "
        "semantic arm."
    ),
)
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
    vector = _parse_supplied_vector(parsed.vector, parsed.space_id)

    from mantle.services import workspace_service

    # Each service call below is offloaded WHOLE. A create is a multi-statement transaction
    # (artifact row, edges, owner grant, index) and the lattice's re-entrant write depth is
    # thread-local, so it must run start to finish in one thread — which is what handing the
    # top-level call across gives, and what wrapping anything inside it would take away.

    # Top-level: no parent to authorize against; the creator owns it.
    if not parsed.container_id:
        derived_id = _derived_identity_id(auth.user_id, parsed.identity)

        # An `identity` that already names an artifact makes this write an UPDATE. That is the
        # whole point of supplying one — the caller is saying "this is the same thing I stored
        # before", and the alternative to honouring it is the duplicate root this parameter
        # exists to prevent. The update runs through the ordinary top-level path, so it takes
        # the same `update` grant check every other rewrite takes: a caller whose grant was
        # revoked gets that path's 404, not a silent second copy.
        if derived_id and await offload_sync(_artifact_exists, store_db, derived_id):
            updated = await offload_sync(
                workspace_service.update_workspace,
                store_db,
                auth.user_id,
                derived_id,
                name=parsed.name,
                description=parsed.description,
                context=context_str,
                vector=vector,
                content=parsed.content,
                content_type=parsed.content_type,
            )
            return updated.to_dict()

        entity = await offload_sync(
            workspace_service.create_container,
            db=store_db,
            user_id=auth.user_id,
            content_type=parsed.content_type,
            name=parsed.name,
            context=context_str or "",
            content=parsed.content or "",
            description=parsed.description,
            vector=vector,
            artifact_id=derived_id,
        )
        return entity.to_dict()

    if parsed.identity:
        # Refused rather than ignored. A collection member is born a DRAFT and acquires a
        # second live version the first time it is edited after commit (`_ensure_draft`), so
        # "the artifact for this identity" stops being one row and the upsert above has no
        # single target to aim at. Accepting the parameter and quietly dropping it would tell
        # the caller their write was idempotent when it was not, which is worse than the
        # duplicate it was meant to prevent.
        raise HTTPException(
            status_code=400,
            detail="identity is supported on top-level artifacts only — it derives one id per "
                   "thing, and an artifact inside a collection has a draft/committed lifecycle "
                   "with more than one live version. Omit container_id, or omit identity.",
        )

    # Into a collection — the caller needs create/Add permission on it.
    await offload_sync(check_access, auth, parsed.container_id, "create", store_db)
    if not _artifact_exists(store_db, parsed.container_id):
        raise HTTPException(status_code=404, detail="Container not found")

    # source_artifact_id -> LINK an existing artifact in (edge only), no new artifact.
    if parsed.source_artifact_id:
        return await offload_sync(_link_source_artifact, store_db, parsed, auth)

    entity = await offload_sync(
        workspace_service.create_workspace_artifact,
        db=store_db,
        user_id=auth.user_id,
        workspace_id=parsed.container_id,
        context=context_str or "",
        content=parsed.content or "",
        content_type=parsed.content_type,
        name=parsed.name,
        index=parsed.index,
        vector=vector,
    )
    return entity.to_dict()


def _derived_identity_id(user_id: str, identity: Optional[str]) -> Optional[str]:
    """The derived artifact id for ``identity``, or ``None`` when the caller sent none.

    A malformed identity is a 400 for the same reason a malformed vector is: it is a statement
    about the request that the caller can fix, and the alternative — deriving from ``""`` —
    would give every identity-less write from this principal one id.
    """
    from mantle.services.artifact_identity import derived_id_for
    try:
        return derived_id_for(user_id, identity)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _parse_supplied_vector(values: Optional[List[float]], space_id: Optional[str]):
    """Validate a writer-supplied vector, or return ``None`` when none was sent.

    Malformed input is a 400 because the caller can fix it: an empty vector, a NaN, a
    zero norm, or a width the anchors cannot place are all statements about the request,
    not about this node's state. What is deliberately NOT checked is whether the vector
    is any *good* — that is a claim about someone else's model and Mantle does not make it.
    """
    if values is None:
        if space_id:
            raise HTTPException(
                status_code=400,
                detail="space_id was supplied without a vector — it names the space of a "
                       "vector that is not here.",
            )
        return None
    from mantle.api.vectors import VectorIngressError, validate_vector
    try:
        return validate_vector(values, space_id)
    except VectorIngressError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def _link_source_artifact(
    store_db: Any,
    parsed: CreateArtifactRequest,
    auth: Any,
) -> Dict[str, Any]:
    """Link an existing artifact into a container instead of creating a duplicate.

    Linking a source artifact requires:

      1. The source must be authorized for `read` before anything about it is
         returned or referenced. The read choke point in `db.store.from_store_doc`
         selects the decryption key from the stored document's `created_by`, so an
         unauthorized caller must never learn the source's content this way — the
         authorization check has to happen before the source is touched at all.

      2. The link must be written with `origin=False, propagate=[]`, so it never
         becomes a **creation-lineage** edge. Grants propagate parent -> child and
         `check_access` walks UP from a target via `get_origin_parent`; an
         `origin=True` link would let the linking container be returned as the
         source's origin parent and confer grants over the whole subtree.
         `propagate=[]` means no action propagates — belt-and-suspenders should
         `origin` alone ever stop being sufficient — so the link can never become
         a grant-inheritance path, regardless of who is allowed to create it.
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
    # `check_access` walks origin edges upward and witnesses the decision to the audit log —
    # several queries plus a write, on the front of every read.
    await offload_sync(check_access, auth, artifact_id, "read", store_db)

    doc = await offload_sync(_find_artifact, store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # Normalize the lattice internal keys.
    doc.pop("_id", None)
    doc.pop("_rev", None)
    if "_key" in doc:
        doc.setdefault("id", doc.pop("_key"))

    # Inject computed child-containment fields. Left on the loop: both are single index seeks on
    # `ix_e_src` (the first with `LIMIT 1`), which is cheaper than the thread hop that would
    # wrap them.
    root_id = doc.get("root_id") or doc.get("id") or artifact_id
    doc["has_children"] = db_has_children(store_db, root_id)
    doc["child_count"] = db_count_children(store_db, root_id) if doc["has_children"] else 0

    # Lazy indexing: first genuine (authorized) access materializes a latent vertex.
    # No-op unless MANTLE_LAZY_INDEX is on and the vertex isn't already materialized —
    # but when it is not a no-op it indexes the artifact, so it goes off the loop whole.
    try:
        from mantle.services import workspace_service as _ws
        from mantle.entities.artifact import Artifact as _Artifact
        await offload_sync(
            _ws.materialize_on_access,
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
    await offload_sync(check_access, auth, container_id, "read", store_db)
    from mantle.services import workspace_service as _ws
    # A sweep over every latent member — the longest-running read in this router.
    n = await offload_sync(_ws.warm_collection, store_db, container_id, tenant_id=auth.user_id)
    return {"collection_id": container_id, "materialized": n}


# ── No GET /artifacts/{id}/embedding ──────────────────────────────────────────
# An observer does not offer "embed this" or "score this" as a service — serving
# raw vectors would hand out bge-m3 output (trained weights) with no text
# attached. Absent entirely rather than answering 404/501, the same standing
# ruling as `/coherence` and `/embed`: the no-models rule.



# ---------- GET /artifacts/{artifact_id}/children — List children ----------

@router.get(
    "/{artifact_id}/children",
    summary="List a container's members",
    description=(
        "Universal container model: any artifact may have children. Paginated — a collection "
        "has no bounded size, and the per-child membership enrichment below costs a query each."
    ),
)
async def list_children(
    artifact_id: str,
    request: Request,
    content_type: Optional[str] = Query(None),
    workspace_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000, description="Page size."),
    offset: int = Query(0, ge=0, description="How many children to skip."),
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
    await offload_sync(check_access, auth, artifact_id, "read", store_db)

    # A draft is workspace-private. Only surface drafts linked into this container
    # when the caller passes a workspace_id they can READ — then drafts homed in
    # THAT workspace (the caller's own) show here, never anyone else's.
    draft_workspace_id: Optional[str] = None
    if workspace_id:
        try:
            await offload_sync(check_access, auth, workspace_id, "read", store_db)
            draft_workspace_id = workspace_id
        except HTTPException:
            draft_workspace_id = None  # no access → don't include its drafts

    # An edge scan plus a lineage resolution per member — unbounded in the container's size,
    # and it runs before the page is taken.
    children = await offload_sync(
        store.list_collection_artifacts,
        store_db, artifact_id, draft_workspace_id=draft_workspace_id,
    )

    # Filter out operator edges (relationship != null means non-containment)
    children = [c for c in children if not c.get("relationship")]

    # Optional content_type filter
    if content_type:
        children = [c for c in children if c.get("content_type") == content_type]

    # Page before enriching. `attach_committed_collection_ids` costs a membership query
    # per child, so a page taken afterwards would pay for the whole collection to return
    # twenty rows. The store returns children in edge order, so the page is stable.
    children = children[offset:offset + limit]

    # Enrich with committed_collection_ids (structural — pure edge traversal)
    from mantle.entities.artifact import Artifact as ArtifactEntity
    from mantle.services.collection_service import attach_committed_collection_ids
    entities = [ArtifactEntity.from_dict(c) for c in children]
    # A membership query per child on the page — the keyed batch, offloaded whole.
    await offload_sync(attach_committed_collection_ids, store_db, entities)
    for raw, entity in zip(children, entities):
        raw["committed_collection_ids"] = getattr(entity, "committed_collection_ids", [])

    # Normalize each child
    for child in children:
        _normalize_artifact_doc(child)

    return children


# ---------- PATCH /artifacts/{artifact_id} — Update ----------

@router.patch(
    "/{artifact_id}",
    summary="Update an artifact",
    description=(
        "Partial update. Re-supply `vector` + `space_id` when the content the vector "
        "describes has changed — the vector arm reindexes with the rest of the write."
    ),
)
async def update_artifact(
    artifact_id: str,
    body: UpdateArtifactRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Partially update an artifact or container."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    await offload_sync(check_access, auth, artifact_id, "update", store_db)

    doc = await offload_sync(_find_artifact, store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # Strip immutable fields from context updates (schema-driven mutability)
    context = _strip_immutable_context_fields(doc, body.context)
    vector = _parse_supplied_vector(body.vector, body.space_id)

    from mantle.services import workspace_service

    container_id = doc.get("collection_id")
    if not container_id:
        # Top-level artifact — no parent collection_id. NOT necessarily a container: every
        # artifact created without a `container_id` (a note, a transcript, a captured file)
        # goes through `create_container` WITH content, so `content`/`content_type` have to be
        # passed here too. Omitting them is what made a rewrite silently return 200 and change
        # nothing, which in turn forced writers to create a second artifact instead of a new
        # version of the same one.
        #
        # `state` is here for exactly that reason one step on: dropping it made archiving a
        # top-level artifact a 200 that did nothing, so a superseded copy could only be deleted
        # — and deleting is the one remediation that destroys the record it was retiring.
        updated = await offload_sync(
            workspace_service.update_workspace,
            store_db,
            auth.user_id,
            artifact_id,
            name=body.name,
            description=body.description,
            context=context,
            vector=vector,
            content=body.content,
            content_type=body.content_type,
            state=body.state,
        )
        return updated.to_dict()

    updated = await offload_sync(
        workspace_service.update_artifact,
        store_db,
        auth.user_id,
        container_id,
        artifact_id,
        context=context,
        content=body.content,
        state=body.state,
        content_type=body.content_type,
        vector=vector,
    )
    return updated.to_dict()


# ---------- DELETE /artifacts/{artifact_id} ----------

@router.delete("/{artifact_id}", status_code=status.HTTP_200_OK)
async def delete_artifact(
    artifact_id: str,
    cascade: bool = False,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Delete or archive an artifact.

    If this artifact is a collection (top-level or nested) with members, those members are
    DETACHED — evicted, not destroyed — unless ``cascade=true`` is passed, in which case a
    member reachable only through this collection is destroyed outright and one still reachable
    through another collection is evicted from this one only. Same rule either way: `rmdir`
    refuses a non-empty directory by default, `rm -r` does not.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    await offload_sync(check_access, auth, artifact_id, "delete", store_db)

    doc = await offload_sync(_find_artifact, store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    from mantle.services import workspace_service

    container_id = doc.get("collection_id")
    if not container_id:
        # Top-level container artifact (workspace/collection) — no parent collection_id.
        # `delete_artifact` is keyed on the containing collection for both its S3 arm and
        # its index arm and refuses a blank one, so the top-level case takes the container
        # primitive instead: `delete_workspace` walks the members, drops each one's index
        # docs and edges, and then removes the container itself. Same split as PATCH.
        await offload_sync(
            workspace_service.delete_workspace, store_db, auth.user_id, artifact_id,
            cascade=cascade,
        )
        return {"id": artifact_id, "deleted": True}

    await offload_sync(
        workspace_service.delete_artifact, store_db, auth.user_id, container_id, artifact_id,
        cascade=cascade,
    )
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

    await offload_sync(check_access, auth, body.container_id, "evict", store_db)

    from mantle.services import workspace_service

    artifact = await offload_sync(
        workspace_service.remove_artifact_from_container,
        store_db,
        auth.user_id,
        body.container_id,
        artifact_id,
    )
    return {"id": artifact.id, "removed": True, "container_id": body.container_id}


# ---------- POST /artifacts/recall — Recall ----------

# `response_model` is declared per-branch rather than on the decorator: the two modes
# return different shapes, and a single declared model would coerce one into the other
# — silently dropping every candidate field. The ranked branch still builds an
# `ArtifactRecallResponse`, so it stays validated at construction; the candidates branch
# returns the accessor's `{candidates, model_id}` verbatim.
@router.post(
    "/recall",
    response_model=None,
    summary="Recall artifacts",
    description=(
        "The one retrieval surface. Ordered and hydrated by default; `candidates: true` returns "
        "the same narrowed candidate set instead, unordered and unhydrated, for an external "
        "flavor to rank within. Both modes share this handler so authorization AND narrowing "
        "resolve identically. `query_text` NARROWS — its terms select which artifacts carry "
        "them — and `vector` + `space_id` RANKS what survives, by cosine. Either alone is a "
        "complete request. Text with no vector returns the narrowed set ordered by how much of "
        "the query each hit matched, with `ordering: \"coverage\"` and an integer stem count as "
        "each `score`; a cosine also needs a provisioned AnchorSet, so a node without one "
        "answers the same way. `field:value` in `query_text` narrows the result before "
        "retrieval, for a known filterable field only — any other word searches as an ordinary "
        "term. A known field carrying an operator it cannot take is a 400 naming it."
    ),
)
async def recall_artifacts(
    body: ArtifactRecallRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> Union[ArtifactRecallResponse, dict]:
    """Recall across accessible artifacts — the one retrieval surface.

    Query syntax: +term (AND), !term (exclude), ~term (semantic), ="phrase" (exact).

    ``field:value`` NARROWS the recall, and ``scope`` (container IDs) narrows it too — the
    first by artifact metadata, the second by container. A filter is resolved to a set of
    artifact ids and intersected with the light cone BEFORE retrieval, alongside the query's
    own terms, so ``total`` and pagination count filtered matches. It can only ever narrow:
    the predicate is shown docs of authorized artifacts only, so no filter can surface, or
    hint at, an artifact you could not already read.

    Filterable: ``id``, ``root_id``, ``collection_id``, ``content_type`` (alias ``type``),
    ``owner_id``, ``title``, ``description``, ``tags`` (alias ``tag``), ``created_at``,
    ``updated_at``. Operators: ``field:value`` (case-insensitive; ``a,b`` is any-of),
    ``field:="Exact Value"`` (case-sensitive, taken whole), ``!field:value`` (negated), and
    ``field:>value`` / ``field:<value`` on ``created_at`` / ``updated_at`` only. Filters
    conjoin. ``field:~value`` is not supported.

    ``word:value`` IS A FILTER ONLY WHEN ``word`` IS ONE OF THOSE FIELDS. Any other word keeps
    its colon and searches as an ordinary term, so ``https://example.com``, ``meeting at 3:30``,
    ``C:\\Users\\john`` and ``ratio 16:9`` are ordinary queries rather than 400s. The cost is
    deliberate and worth stating to a caller: a MISSPELLED field is a search term, not an error
    — ``titel:foo`` searches for that text and finds nothing rather than naming ``titel``. Check
    the field list above when a ``field:value`` query comes back empty.

    A word that IS a field and cannot be honoured is still a 400 naming it, never a silent drop:
    an unsupported operator (``field:~value``, a range on an unordered field), and ``state:`` and
    ``content:`` specifically — ``state`` selects the index segment and is the ``state`` request
    field's job, ``content`` is encrypted at rest. A query of nothing but filters is also a 400 —
    a filter narrows a recall, it does not constitute one.

    Filter tokens do not reach the index. Retrieval sees the query's TERMS only, so
    ``budget type:pdf`` searches for ``budget`` and filters on the type, rather than
    scoring documents that happen to contain the word "type". ``applied_filters`` on the
    response lists what actually narrowed the result.

    TEXT NARROWS, A VECTOR RANKS. The query's terms decide WHICH authorized artifacts come
    back — membership, read off the blind-token index — and ``vector`` + ``space_id`` decide
    what ORDER they come back in, by cosine, cut where the ranking's own spectrum stops. They
    are answers to two different questions, so they compose rather than compete, and either
    alone is a complete request.

    NO QUERY SYNTAX TURNS RANKING ON OR OFF. ``~term`` selects which text is sent for
    embedding, and that is the whole of its effect. Ranking happens when a query vector exists
    and this node has a provisioned AnchorSet — a request fact and a node fact, neither of them
    spellable in the query string.

    A TEXT QUERY WITH NO VECTOR IS NOT AN ERROR. It narrowed to a real set, and the narrowing
    knows how much of the query each member of it matched — so the set comes back with the
    fullest matches first: ``ordering`` is ``"coverage"`` and each hit's ``score`` is the
    integer count of distinct query stems it carries, ties broken most-recently-updated first.
    A caller that cannot embed — a shell script, a webhook — searches this way and gets an
    answer. A SINGLE-TERM QUERY IS EXACTLY RECENCY ORDER, since every hit matched the one term
    there was; the count says so, and does not pretend otherwise.

    THAT COUNT IS NOT A RELEVANCE SCORE. Nothing weights it by term rarity, term frequency or
    field length; it is a count of which of your stems were found. Do not threshold on it and
    do not compare it between queries.

    A VECTOR QUERY WITH NO ANCHORSET IS A 400, and it is the same 400 a foreign ``space_id``
    gets. Ranking answers only on a node with a provisioned AnchorSet; a node nobody has
    provisioned ranks in no space, so a supplied ``vector`` names a space that does not exist
    here and cannot be placed. Answering it anyway returned 200 over everything the caller can
    read, in an order nothing in the body distinguishes from the order a query carrying no
    vector at all comes back in. The message names both ways out: seed the set, or send the
    same recall without ``vector``.

    Refusing covers a hybrid text+vector recall as well, because this door already refuses one
    whose vector names a foreign space. What is NOT refused is a write: ``POST /artifacts``
    still accepts a writer's vector on an unseeded node, where it is provenance for data at
    rest. See README.md's "Semantic recall is inert until you seed an AnchorSet" for the
    operator step, and ``search/types.py`` on ``ordering``.

    Set ``candidates: true`` for the raw retrieval primitive — the SAME narrowed candidate set
    this mode returns, unordered and unhydrated, carrying ``artifact_id`` / ``collection_id`` /
    ``principal_id`` and no score of any kind — which is what an external recall flavor ranks
    within. Both modes resolve authorization and narrowing identically because they run the
    same handler up to the final call; that is the point of them sharing it. What the candidate
    mode declines to state is the ORDER, which is the flavor's to decide.
    """
    user_id = auth.user_id
    bearer_grant = auth.bearer_grant
    key_grants = auth.grants if auth.principal_type == "grant_key" else []

    if not user_id and not bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    # A query vector is a query. Text and vector are the two arms' inputs, so either
    # one alone is a complete request and both together is hybrid recall; what is
    # refused is neither, which asks for a ranking with nothing to rank on.
    query_vector = _parse_supplied_vector(body.vector, body.space_id)
    if not (body.query_text and body.query_text.strip()) and query_vector is None:
        raise HTTPException(
            status_code=400,
            detail="query_text or vector is required",
        )
    query_embedding = None
    if query_vector is not None:
        from mantle.api.vectors import VectorIngressError, project_to_anchor_space
        try:
            query_embedding = project_to_anchor_space(query_vector)
        except VectorIngressError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # Resolve explicit container scope when body.scope is provided.
    # A workspace IS a collection — no distinction needed.
    #
    # Scope precedence:
    # 1. Explicit body.scope — user chose specific containers to search.
    # 2. Grant-key principal — restrict to the resources the key actually carries.
    #    For a bundle that is every readable member, already narrowed by the bundle's
    #    ceiling at authentication, so this needs no bundle-awareness of its own.
    # 3. Bearer grant naming a single resource — restrict to it.
    # 4. None — accessor runs the full light-cone for the authenticated user.
    scope: Optional[List[str]] = None

    if body.scope:
        col_ids = [cid for cid in body.scope if _artifact_exists(store_db, cid)]
        scope = col_ids or None
    elif key_grants:
        readable = [
            g.resource_id for g in key_grants
            if getattr(g, "can_read", False) and g.resource_id
        ]
        # A key that carries nothing readable must search nothing. Leaving scope None
        # would fall through to the unscoped light cone.
        if not readable:
            raise HTTPException(status_code=403, detail="Grant key cannot read any resource")
        scope = readable
    elif not user_id and bearer_grant and getattr(bearer_grant, "can_read", False):
        scope = [bearer_grant.resource_id] if bearer_grant.resource_id else None

    # Build and execute search query. Imported here, not at module scope, for the same reason
    # `SearchQuery` is: the router must not drag the search package in at import time.
    from mantle.search.field_filters import QueryFilterError
    from mantle.search.types import SearchQuery

    query = SearchQuery(
        query_text=body.query_text or "",
        query_embedding=query_embedding,
        user_id=user_id or "",
        scope=scope,
        from_=body.from_,
        # Candidate retrieval is budget-bounded, not paginated — the caller ranks the
        # whole set, so a page-sized cap would silently truncate what it ranks within.
        size=body.candidate_budget if body.candidates else body.size,
        sort=body.sort or "relevance",
        highlight=False if body.candidates else body.highlight,
    )

    # MANTLE-SSE is the canonical search backend. If SSE prerequisites (Oracle, S3,
    # the lattice) aren't satisfied, search returns 503 — there's no plaintext
    # fallback by design.
    from mantle.search.mantle.wiring import VALID_SEGMENTS, build_sse_search_accessor
    segment = (body.state or "committed").lower()
    if segment not in VALID_SEGMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"state must be one of {', '.join(VALID_SEGMENTS)}",
        )
    # Wiring the accessor opens the lattice segment and the key oracle — synchronous, and on
    # the first request of a process it is the whole cost of the call.
    accessor = await offload_sync(build_sse_search_accessor, store_db, segment=segment)
    if accessor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Encrypted search is not available — Oracle, S3, or the lattice "
                "prerequisite missing. Check platform/key_manager + "
                "content_service initialization."
            ),
        )

    # Raw candidate set: the retrieval primitive, returned unranked and unhydrated for
    # a flavor to rank within. Everything above this line — authorization included —
    # has already run, which is the reason it shares this handler.
    if body.candidates:
        try:
            # THE blocking call of this handler: lattice reads, key derivation and decryption,
            # none of it awaitable. Offloaded whole — the accessor owns its own transactions.
            return await offload_sync(
                accessor.candidates,
                query, candidate_budget=body.candidate_budget, include_vectors=False,
            )
        except ValueError as e:
            # `candidates` rejects an empty parse as malformed rather than as a search
            # that matched nothing. That is a 400, not a 500. `QueryFilterError` is a
            # `ValueError` and lands here for the same reason: an unusable filter is a
            # request the caller can fix.
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Artifact candidate retrieval error: %s", e)
            raise HTTPException(status_code=500, detail=f"Recall failed: {str(e)}")

    try:
        result = await offload_sync(accessor.search, query)
    except QueryFilterError as e:
        # A filter naming a field this node refuses, an operator it does not support, or a
        # query that is nothing but filters. Caught BEFORE the blanket 500 below and ahead of
        # it in the except chain: a caller who wrote `state:draft` needs to be told to send the
        # `state` request field, and a 500 would read as "the server broke" for a fixable
        # request. A MISTYPED field (`typ:pdf`) does not arrive here at all — it is not a field,
        # so it is a search term, which is the trade that keeps `https://…` searchable. It is also
        # what stops the old bug — a filter that could not be honoured is now impossible to
        # ignore, on this path as on the candidates one.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Artifact recall error: %s", e)
        raise HTTPException(status_code=500, detail=f"Recall failed: {str(e)}")

    return ArtifactRecallResponse(
        hits=[
            RecallHitResponse(
                id=hit.doc_id,
                score=hit.score,
                root_id=hit.root_id,
                version_id=hit.version_id,
                collection_id=hit.collection_id,
                title=hit.title or None,
                description=hit.description or None,
                # The densest spans of `hit.content`, not a blind prefix — see
                # `beacon.density`. Length is never capped: it is exactly as long as
                # `beacon.cut.top_break` found signal for, which can be shorter or
                # longer than the old 500-character slice depending on the document.
                content=dense_excerpt(hit.content or "") or None,
                tags=hit.tags or None,
                # `hit.highlights` is always None (the search backend does not compute
                # query-term highlighting — see `search/mantle/sse/router_accessor.py`).
                # The same dense spans that built `content` above, individually, so a
                # client that wants to render them separately (rather than as one
                # reassembled string) can.
                highlights=(
                    {"content": dense_windows(hit.content or "")} if hit.content else None
                ),
            )
            for hit in result.hits
        ],
        total=result.total,
        query_text=query.query_text,
        parsed_query=str(result.parsed_query),
        applied_filters=list(result.applied_filters),
        corrections=result.corrections,
        ordering=result.ordering,
        **{"from": body.from_},
        size=body.size,
    )


# ── No /activate ───────────────────────────────────────────────────────────────
# Accepting a caller-supplied `embedding` + `model_id` and echoing the raw carrier
# vector back would be an embed/score service. Absent entirely rather than answering
# 501/503, the same standing ruling as `/coherence` and `/embed`: the no-models rule.
#
# What that rule refuses is producing vectors, not receiving them. A vector a caller
# already holds reaches the semantic arm through the two ingress points that exist —
# `vector` + `space_id` on the write, and the same pair on `/recall` — and Mantle
# stores, compares and ranks it without ever having a model. See `api/vectors.py`.




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


class BatchFetchRequest(BaseModel):
    """Batch fetch artifacts by IDs."""
    artifact_ids: List[str]


# =============================================================================
# Batch Operations (static path — registered before /{id} sub-paths)
# =============================================================================

# ---------- POST /artifacts/batch ----------

def _fetch_authorized_docs(store_db: Database, auth: AuthContext, artifact_ids: List[str]) -> list:
    """Resolve each id and keep only what the caller may read.

    One sync pass so the handler pays a single thread hop for the whole request: the body is two
    store operations per id, and awaiting each separately would spend more time hopping than
    reading. Unlike :func:`_hydrate_batch` this authorizes per id, so the two cannot be shared —
    the batch endpoint takes caller-supplied ids and must not become an existence oracle.
    """
    results: list = []
    for aid in artifact_ids:
        doc = _find_artifact(store_db, aid)
        if not doc:
            continue

        # Verify read access silently — skip inaccessible artifacts.
        try:
            check_access(auth, aid, "read", store_db)
        except HTTPException:
            continue

        results.append(_normalize_artifact_doc(doc))
    return results


@router.post("/batch")
async def batch_fetch_artifacts(
    body: BatchFetchRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Batch fetch artifacts by IDs across all containers."""
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    results = await offload_sync(_fetch_authorized_docs, store_db, auth, body.artifact_ids)
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

    await offload_sync(check_access, auth, artifact_id, "create", store_db)

    from mantle.services.workspace_service import initiate_upload_and_create_artifact

    try:
        # Artifact create plus an object-store handshake, in one transaction each.
        out, artifact = await offload_sync(
            initiate_upload_and_create_artifact,
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

    await offload_sync(check_access, auth, artifact_id, "update", store_db)

    doc = await offload_sync(_find_artifact, store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    workspace_id = doc.get("collection_id")
    if not workspace_id:
        # A top-level container artifact has no parent collection, and an upload is always
        # created INSIDE one (`POST /artifacts/{container_id}/upload`), so this artifact is
        # not an upload and there is no progress to advance. The caller named the wrong
        # artifact — a 400, not a server fault.
        raise HTTPException(
            status_code=400,
            detail="Not an upload artifact — a top-level artifact has no upload in progress",
        )

    from mantle.services.workspace_service import update_upload_status as svc_update_upload

    try:
        result = await offload_sync(
            svc_update_upload,
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
    """Disabled. Presigned multipart parts upload directly to the object store,
    bypassing Mantle, which cannot envelope-encrypt them on that path. Content is
    encrypted on Mantle's byte path via the proxied `PUT /artifacts/{id}/content`
    upload (see `upload-initiate`, which returns that URL); chunked proxied upload
    is not implemented."""
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
    await offload_sync(check_access, auth, artifact_id, "read", store_db)

    doc = await offload_sync(_find_artifact, store_db, artifact_id)
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

    # S3 is invisible to callers and holds only ciphertext, so no presigned S3 URL
    # is handed out (a direct S3 fetch would return undecryptable ciphertext).
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
    invisible to callers (no presigned S3 URL). Requires read access.

    Local CAS first, object store behind it — the one tiered path, so this finds what a write on
    a node with no object store stored, and still pulls through and sha256-verifies against the
    mirror on a node that has one."""
    await offload_sync(check_access, auth, artifact_id, "read", store_db)
    doc = await offload_sync(_find_artifact, store_db, artifact_id)
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
    # The local address of the same bytes, recorded by the PUT below. Validated, never trusted:
    # `is_cas_ref` is the shape gate and `get_bytes_decrypted` demands an envelope this route's
    # own writes carry, so a context edited to name another object yields an error, not bytes.
    cas_ref = ctx.get("content_cas_ref")
    if not content_key and not cas_ref:
        raise HTTPException(status_code=404, detail="No downloadable content for this artifact")

    owner_id = doc.get("created_by")
    from mantle.services.content_service import ContentStoreUnavailable, get_bytes_decrypted
    from fastapi import Response
    try:
        # Tier read plus decryption of the whole blob: the most obviously blocking
        # call in the router, and the one whose duration scales with the artifact's size.
        data = await offload_sync(
            get_bytes_decrypted, content_key, owner_id,
            cas_ref=cas_ref, collection_id=doc.get("collection_id"),
        )
    except ContentStoreUnavailable as exc:
        # Not a server fault and not a lost object: this node holds no copy and has no object
        # store to look in. 404 is the honest answer, and it names where the bytes may still be.
        logger.info("content download unavailable for %s: %s", artifact_id, exc)
        raise HTTPException(
            status_code=404,
            detail="No downloadable content on this node for this artifact. Its bytes are not in "
                   "the local content tier and no object store is configured to look in; an "
                   "artifact whose body was sent as the inline `content` field carries it on "
                   "GET /artifacts/{id} instead.")
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
    byte path** before storing them (no presigned S3 PUT; storage never receives plaintext).
    Requires update access. This is the target of `upload-initiate`.

    **Local first, mirror second.** The bytes land in this node's own encrypted CAS — the same
    `TieredContentStore` the ingest path writes and `shard/content_tier.py` drains — and then, on a
    node that has one, in the object store as well. A node with no object store is a complete
    configuration: the write succeeds, nothing warns, and nothing queues waiting for a mirror that
    does not exist.

    **A mirror this node has and cannot reach is the middle case, and it is recorded.** The
    response is unchanged — the bytes are local, verified and readable, and the request has
    genuinely succeeded — but the mirror leg is still owed, and without a durable note of that the
    content stays reachable here and nowhere else, forever. `_record_mirror_pending` puts that note
    on the store's own work pool. A permanent refusal (the store answered, and said no) is logged
    rather than queued: a retry of a byte-identical request gets a byte-identical answer.

    **Idempotent by content address.** A re-upload of identical bytes is decided against
    `sha256(body)`, so it re-encrypts nothing, writes nothing and creates no duplicate.
    """
    await offload_sync(check_access, auth, artifact_id, "update", store_db)
    doc = await offload_sync(_find_artifact, store_db, artifact_id)
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
    collection_id = doc.get("collection_id")
    ctype = request.headers.get("content-type") or ctx.get("content_type") or "application/octet-stream"

    body = await request.body()
    from mantle.services import content_service
    from mantle.services.content_service import ContentStoreUnavailable, put_bytes_encrypted

    # Content-addressed no-op. The decision is over the CONTENT, not a timestamp or an etag, so
    # re-uploading the same bytes cannot produce a second object — and it is made before the
    # envelope, because a fresh nonce would make identical plaintext look like new content.
    digest = hashlib.sha256(body).hexdigest()
    prior_ref = ctx.get("content_cas_ref")
    if (ctx.get("content_sha256") == digest
            and content_service.is_cas_ref(prior_ref)
            and await offload_sync(content_service.local_content_has, prior_ref)):
        return {"stored": True, "size": len(body), "content_key": content_key,
                "content_ref": prior_ref, "deduplicated": True}

    try:
        # Encrypt-then-store, both proportional to the upload's size.
        #
        # `on_mirror_deferred` fires only in the middle case — a mirror this node HAS and could
        # not reach — and records the work still owed. It runs inside the same worker thread as
        # the write, so the store call it makes is a whole synchronous operation on one thread,
        # which is the rule `offload_sync` exists to keep.
        ref = await offload_sync(put_bytes_encrypted, content_key, body, ctype, owner_id,
                                 collection_id=collection_id,
                                 on_mirror_deferred=lambda cas_ref, exc: _record_mirror_pending(
                                     store_db, artifact_id, content_key, cas_ref, exc,
                                     owner_id=owner_id))
    except OverflowError as exc:
        # The cipher's own bound, surfaced rather than replaced by a tuned one: AES-GCM refuses a
        # plaintext of 2**31 bytes or more, and every tier here takes the whole body in memory, so
        # this is the size at which the write stops being possible rather than merely large.
        logger.info("content upload for %s exceeds the cipher's size bound: %s", artifact_id, exc)
        raise HTTPException(
            status_code=413,
            detail=f"Content too large to encrypt in one envelope ({len(body)} bytes). AES-GCM "
                   f"accepts at most 2**31 - 1 bytes per message and this route encrypts the "
                   f"body whole; split the upload across artifacts.")
    except ContentStoreUnavailable as exc:
        # Both tiers absent. NAME THE REMEDY: a node reaches this only when it has neither a keys
        # directory (so no content key and no local CAS) nor object-store credentials, which is a
        # provisioning fault and not a transient one.
        logger.error("content upload for %s has nowhere to land: %s", artifact_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to store content: this node has neither a local content tier nor an "
                   "object store. Provision the keys directory (KEYS_DIR — `mantle-init-keys`) so "
                   "the local encrypted CAS can open, or configure the object store. The body can "
                   "also be sent as the `content` field on POST or PATCH /artifacts, which stores "
                   "it encrypted inside the artifact itself.")
    except Exception as exc:
        logger.error("content upload failed for %s: %s", artifact_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to store content. The server log carries the underlying error.")

    # Record where the bytes went. Without this the write is unfindable: the object store is
    # addressed by `content_key`, the local CAS by the content's own address, and only the
    # artifact knows both. `content_sha256` is what makes the next identical upload a no-op.
    #
    # `content_key` is recorded even when it was DERIVED here rather than read from the context.
    # It is the object store's whole address, so an upload that did not write it back left the
    # download route reading `ctx["content_key"] is None` — a 404 on content that was stored.
    await offload_sync(_record_content_ref, store_db, doc, content_key, ref, digest, ctype)

    return {"stored": True, "size": len(body), "content_key": content_key,
            "content_ref": ref, "deduplicated": False}


# ── the mirror leg this node still owes ──────────────────────────────────────────────────────
#
# Written to the store's OWN work pool (`db/schema.py`'s `task` sidecar), in the row SHAPE
# `ember/runtime/pool.py::enqueue` already uses — `operator` + `arguments` + `status` + `task_key`,
# whose `_sync_task` mirror in the `task` table is what makes `status` an indexed, selective
# predicate. One row shape, one set of typed accessors (`pending_window` / `try_claim` / `release`
# / `count_by_status`) — not a second convention that a drain would have to learn separately.
#
# Under its own CONTENT TYPE, though, and that distinction is the load-bearing one: the shape is
# what a drain reads, the content type is what a worker SELECTS on. `ember`'s claim predicate is
# `(ct, status)` with no operator, so sharing its type is what made these rows claimable — and
# dead-letterable — by a worker that implements no mirror. `mirror_drain.MIRROR_TASK_CT` is
# mantle's alone.
#
# The drain for this operator is `services/mirror_drain.py`, which claims these rows, redoes the
# mirror leg through the same `put_bytes_encrypted` path that failed, and settles them. The two
# halves share ONE definition of what the work is called and how it is addressed — imported below
# rather than restated here, because a task written under one spelling and claimed under another
# is a queue that accumulates forever while reporting itself drained.
from mantle.services import mirror_drain                                               # noqa: E402
from mantle.services.mirror_drain import (                                             # noqa: E402
    MIRROR_OPERATOR as _MIRROR_OPERATOR,
    MIRROR_TASK_CT as _MIRROR_TASK_CT,
    mirror_task_id as _mirror_task_id,
    mirror_task_key as _mirror_task_key,
)


def _record_mirror_pending(store_db: Database, artifact_id: str, content_key: str,
                           ref: str, exc: BaseException, *,
                           owner_id: Optional[str] = None) -> None:
    """Record that this node owes the object store a copy of `ref`, at `content_key`.

    The payload is everything the mirror leg needs to be redone and nothing that outlives it:
    which artifact, which object-store key, which local CAS ref. No presigned URL, no bucket name,
    no endpoint, no size — all of those are read from configuration at the time the retry runs, and
    a copy of them frozen here would be the stale half of a disagreement.

    `next_retry_at` is left NULL, which `pending_window` reads as "eligible now". Any other value
    would be a backoff interval invented here, with nothing to derive it from: backoff is a
    property of attempts, so it belongs to the code that makes them — `mirror_drain` computes it
    from the attempt it just watched fail and writes it through `tasks.settle(...)`. Writing a
    guess now would only mean the drain's first act is to overwrite it.

    No `attempts` is written either, and that is load-bearing rather than an omission: this row is
    being refreshed because the artifact's content CHANGED, so the failures the previous ref
    accumulated say nothing about the new one. Clearing the count is what makes new bytes start
    at one round of backoff instead of inheriting the old outage's.

    Never versioned and never replicated: `_put_op` pins the row outside every observer's proper
    time, so this per-box operational fact costs no `_seq` and cannot enter the publish feed — the
    same rule every cursor in `mesh/sync.py` is written under, and `MIRROR_TASK_CT` is registered
    in that module's `_OP_EXCLUDE` (`_put_op` REFUSES a replicated type outright, so the two
    cannot drift apart silently).

    Raises freely. `put_bytes_encrypted` holds the guard, because the rule being kept is its rule:
    a completed, durable, verified write is never turned into a failure by bookkeeping behind it.
    """
    from prism import grounding as genesis     # contract, not the runner: the provenance rung only
    from mantle.mesh.sync import _put_op

    tid = _mirror_task_id(content_key)
    key = _mirror_task_key(content_key)
    # Written unconditionally, because the natural key is what enforces "one row" and the LATEST
    # failure is the one that is true: it names the ref the artifact's context now points at. A
    # read-then-skip on top of that would only ever preserve a superseded ref.
    #
    # `content_ref` stays INSIDE `arguments`. At the top level it is a projected vertex column and
    # `shard/content_tier.promote_local_content` walks exactly that column — a task row would then
    # be picked up as content to promote to a different mirror entirely.
    #
    # `created_by` is the CONTENT'S owner — authorship, recorded because the row is about that
    # person's content and an unauthored row attributes it to nobody.
    #
    # It is NOT an access grant and does not make the row readable through `GET /artifacts/{id}`:
    # `check_access` decides on GRANTS, and `_put_op` writes a vertex without minting one, so a
    # read of this id answers 404 for every principal on the node. That is a property of how
    # operational rows are written here (every `_put_op` cursor and watermark has it), not of this
    # task, and closing it means deciding whether an operational row belongs in the grants
    # collection at all — a separate decision from retrying a mirror. Until it is made, this row
    # is reached through `/status`'s work-pool counts and through the `task` sidecar it projects
    # into, both of which name it, its operator, its error and its state.
    _put_op(store_db, {
        "id": tid, "content_type": _MIRROR_TASK_CT,
        "created_by": owner_id,
        "operator": _MIRROR_OPERATOR,
        "arguments": {"artifact_id": artifact_id, "content_key": content_key,
                      "content_ref": ref},
        "status": "pending", "priority": 0, "task_key": key,
        # Same construction as `shard/content_tier.py::_one` uses for its failed refs: the type and
        # a bounded prefix of the message, so "why is this still pending" is answerable from the
        # row instead of from a log someone has to still have.
        "last_error": "%s: %s" % (type(exc).__name__, str(exc)[:160]),
        "content": "task %s [%s]" % (_MIRROR_OPERATOR, key),
        "provenance": genesis.P_HUMAN, "cited_from": genesis.CITE_GENESIS})
    logger.warning("content for %s is pending mirror to %s; enqueued %s", artifact_id,
                   content_key, tid)
    # Wake this process's drain if it has one. The row is already durable, so this is latency and
    # never correctness — and it is what lets the drain have no poll interval at all: it sleeps on
    # the queue's own schedule and is woken by the only thing that can add to the queue.
    mirror_drain.notify_pending()


def _record_content_ref(store_db: Database, doc: Dict[str, Any], content_key: str,
                        ref: Optional[str], digest: str, content_type: str) -> None:
    """Write both of the content's addresses onto the artifact's context, through the entity
    boundary.

    Through `store.get_artifact` / `store.update_artifact` rather than a raw doc write, so the
    inline-content envelope and the change announcement both stay on their one chokepoint
    (`db/lattice_api.py`). It patches only the content fields and leaves the rest of the context
    as the caller wrote it.

    Never raises into the request: the bytes are already stored and verified. A failure here costs
    the artifact its pointer, which the next upload rewrites — losing the write to report it would
    be strictly worse.
    """
    try:
        version_id = doc.get("_key") or doc.get("id")
        entity = store.get_artifact(store_db, version_id)
        if entity is None:
            return
        ctx = {}
        if entity.context:
            try:
                parsed = json.loads(entity.context)
                ctx = parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError):
                ctx = {}
        ctx["content_key"] = content_key
        ctx["content_sha256"] = digest
        if ref:
            ctx["content_cas_ref"] = ref
        else:
            # No local copy this time (no keys dir, so no local CAS). A stale ref left behind would
            # send the next read at bytes that are no longer this artifact's content.
            ctx.pop("content_cas_ref", None)
        ctx.setdefault("content_type", content_type)
        entity.context = json.dumps(ctx)
        store.update_artifact(store_db, entity)
    except Exception:
        logger.warning("could not record the content address on artifact %s — the bytes are "
                       "stored at %s and the next upload rewrites the pointer",
                       doc.get("_key") or doc.get("id"), ref, exc_info=True)


# =============================================================================
# Ordering / Move Endpoints
# =============================================================================

# ---------- PATCH /artifacts/{artifact_id}/children/order ----------

def _reorder_children(store_db: Database, artifact_id: str, ordered_ids: List[str]) -> None:
    """Resolve version-ids or root-ids → root-ids, then rewrite the edge order.

    One unit because it is one operation: the resolution loop only exists to feed the reorder,
    and the reorder is a transaction the resolution must not be spliced into.
    """
    ordered_roots: List[str] = []
    for aid in ordered_ids:
        a = store.get_artifact(store_db, aid)
        if a:
            ordered_roots.append(a.root_id)
    store.reorder_collection_artifacts(store_db, artifact_id, ordered_roots)


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

    await offload_sync(check_access, auth, artifact_id, "update", store_db)

    if not _artifact_exists(store_db, artifact_id):
        raise HTTPException(status_code=404, detail="Artifact not found")

    await offload_sync(_reorder_children, store_db, artifact_id, body.ordered_ids)

    return {"order_version": 0}


# ---------- POST /artifacts/{artifact_id}/revert — Phase D.1 dedicated route ----------

@router.post("/{artifact_id}/revert")
async def revert_artifact_endpoint(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Restore the artifact's last committed version, discarding the draft delta.

    This is a distinct endpoint rather than routing through `op/revert`: revert
    touches version history (it doesn't just flip a state field) so it warrants
    its own verb. If the artifact has no committed version yet, returns
    `204 No Content` per the design doc's "no-op" rule.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    await offload_sync(check_access, auth, artifact_id, "update", store_db)

    doc = await offload_sync(_find_artifact, store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    workspace_id = doc.get("collection_id") or doc.get("_key") or ""

    from mantle.services.workspace_service import revert_artifact

    try:
        result = await offload_sync(
            revert_artifact,
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
    await offload_sync(check_access, auth, container_id, "read", store_db)

    if not _artifact_exists(store_db, container_id):
        raise HTTPException(status_code=400, detail="Commits only available for collections")

    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    from mantle.services.collection_service import get_commits_for_collection

    try:
        # A commit row plus its item rows for each commit in the collection's history.
        commits = await offload_sync(
            get_commits_for_collection, store_db, auth.user_id, container_id
        )
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
    await offload_sync(check_access, auth, artifact_id, "admin", store_db)
    from mantle.services import audit_service
    events = await offload_sync(
        audit_service.get_artifact_access_log,
        store_db, artifact_id, limit=limit, offset=offset, result=result,
    )
    return {"artifact_id": artifact_id, "count": len(events), "events": events}

