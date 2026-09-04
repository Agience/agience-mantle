# routers/artifacts_router.py
#
# Unified Artifact API — single REST surface for all artifact operations.
#
# Replaces per-container endpoints (workspaces, collections, agents, inbound,
# search) with a container-agnostic set of verbs:
#
# POST /artifacts → Create (with `container_id` in the body, this
# IS the container Add — there is no separate
# add-to-container route)
# GET /artifacts/visible → List what the caller's light cone reaches
# GET /artifacts/{id} → Read
# GET /artifacts/{id}/children → List a container's members
# PATCH /artifacts/{id} → Update
# DELETE /artifacts/{id} → Delete
# DELETE /artifacts/{id}/children/{child_id}
# → Detach one member from a container (the root survives)
# POST /artifacts/recall → Recall (ordered + hydrated; `candidates: true` for the
# same narrowed set unordered — one handler, so both
# modes resolve authorization identically)
# POST /artifacts/batch → Batch fetch by IDs
#
# Specialized endpoints:
#   POST   /artifacts/{id}/upload-initiate      → Initiate an upload
#   PATCH  /artifacts/{id}/upload-status        → Update upload progress
#   GET    /artifacts/{id}/content-url          → Points at the proxied content route below
#   GET    /artifacts/{id}/content              → Proxied download (decrypts on the byte path)
#   PUT    /artifacts/{id}/content              → Proxied upload (encrypts on the byte path)
#   PATCH  /artifacts/{id}/children/order       → Reorder a container's children
#   POST   /artifacts/{id}/revert               → Restore the last committed version
#   POST   /artifacts/{id}/warm                 → Materialize a collection's latent artifacts
#   GET    /artifacts/{artifact_id}/commits     → List commits for collection (id must be one)
#   GET    /artifacts/{id}/access-log           → The artifact's own access history
#
# There is no move verb: an artifact's home is its membership edges, so moving one is
# `POST /artifacts` (link into the target) plus `DELETE /artifacts/{artifact_id}/children/{child_id}`.
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
#: Imported at module scope: the helpers below read the environment at call time, and a
#: function-local import would make every call pay for it while leaving a missing name to
#: surface as a NameError inside whatever exception handler happened to surround the call.
import os
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Union

#: `ACTIONS` is imported at MODULE scope so the `action` parameter can publish its own enum.
#: `attenuation` imports nothing but stdlib, so there is no cycle to defer around.
from mantle.attenuation import ACTIONS
from mantle.search.lazy import INDEX_HINTS
#: The artifact-state vocabulary, imported so `PATCH` can publish and enforce it. The entity is
#: the authority on what state an artifact is in. `search.mantle.wiring`'s `VALID_SEGMENTS` is the
#: index's own mirror of the same three states; reconciling the two reaches into the search arm and
#: is not this route's call to take.
from mantle.entities.artifact import Artifact as _ArtifactStates

from mantle.db.store import Database
from mantle.search.beacon.density import dense_excerpt, dense_windows
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, status
from pydantic import (BaseModel, ConfigDict, Field, field_validator, model_serializer,
                      model_validator)
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
from mantle.search.field_filters import filterable_field_names as _filterable_field_names
from mantle.api.errors import ERROR_DESCRIPTIONS as _ERROR_DESCRIPTIONS

#: Derived from the entity rather than restated — see the note above the import. Kept below the
#: import block: a statement between imports makes every later module-level import an `E402`.
_ARTIFACT_STATES = sorted(_ArtifactStates.VALID_STATES)


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

#: The ceiling on any caller-supplied count. Ten times this router's own `le=1000` query idiom,
#: because the job is to remove the unbounded case rather than to tune a page size: a generous
#: ceiling refuses no plausible caller.
#:
#: It bounds what a single call can cost. `POST /artifacts/batch` costs two store operations per id
#: (see `_fetch_authorized_docs`), so an unbounded `artifact_ids` buys an unbounded number of store
#: reads inside one request — and the per-client rate limiter in `main.py` counts that request once
#: against its 600/min, so it bounds arrival rate and not per-call cost. `recall.size` and
#: `candidate_budget` reach the search path through the same clamp.
_MAX_PAGE = 10_000

#: Upload sizes are bounded by the CIPHER, not by a number chosen here. `PUT /artifacts/{id}/content`
#: already answers 413 above this: AES-GCM accepts at most 2**31 - 1 bytes per message and that
#: route encrypts the body whole, so it is the size at which the write stops being POSSIBLE rather
#: than merely large. `upload-initiate` declares the same bound instead of inventing a second one.
_MAX_CONTENT_BYTES = 2 ** 31 - 1


def _refuse_oversize_inline(content, where: str) -> None:
    """Refuse an inline body above this node's ceiling, raising 413.

    Called from the create path so the inline `content` field carries the same ceiling as
    `PUT /artifacts/{id}/content`: the same bytes cost the same whichever route they arrive on.

    Cheap first, exact second. UTF-8 is at least one byte per character, so a string longer than
    the limit in characters is over it in bytes and can be refused without encoding anything. Only
    a string that might still fit is encoded to get its true size — encoding a body to measure it
    is the cost the ceiling exists to avoid.
    """
    if not content:
        return
    limit = max_content_bytes()
    n = len(content) if len(content) > limit else len(content.encode("utf-8"))
    if n <= limit:
        return
    logger.info("refusing %d-byte inline content on %s (limit %d)", n, where, limit)
    raise HTTPException(
        status_code=413,
        detail=("Inline `content` of %d bytes exceeds this node's configured limit of %d "
                "(MANTLE_MAX_CONTENT_BYTES). Send the body to PUT /artifacts/{artifact_id}/content "
                "instead, which streams it rather than carrying it in the request." % (n, limit))
        if limit < _MAX_CONTENT_BYTES else
        ("Inline `content` of %d bytes exceeds what this node can encrypt in one envelope "
         "(AES-GCM accepts at most 2**31 - 1). Send the body to "
         "PUT /artifacts/{artifact_id}/content instead." % n))


def max_content_bytes() -> int:
    """The largest body this node accepts, in bytes.

    Reads `MANTLE_MAX_CONTENT_BYTES` and defaults to the cipher bound, so a node with nothing
    configured accepts whatever AES-GCM can encrypt in one envelope. The operator-settable value
    is what puts a smaller ceiling between "fine" and two gigabytes buffered in RAM.

    The configured value is clamped, not trusted. Above the cipher bound it cannot be honoured —
    AES-GCM will not encrypt it — so it is clamped down, and a non-integer or non-positive value
    falls back to the cipher bound with a warning rather than failing the node on a config typo.
    A ceiling that refuses to start is a ceiling nobody sets.

    Read at call time rather than at import, matching `lazy_index_default`.
    """
    raw = (os.getenv("MANTLE_MAX_CONTENT_BYTES", "") or "").strip()
    if not raw:
        return _MAX_CONTENT_BYTES
    try:
        want = int(raw)
    except ValueError:
        logger.warning(
            "MANTLE_MAX_CONTENT_BYTES=%r is not an integer — using the cipher bound %d", raw,
            _MAX_CONTENT_BYTES)
        return _MAX_CONTENT_BYTES
    if want <= 0:
        logger.warning(
            "MANTLE_MAX_CONTENT_BYTES=%d is not positive — using the cipher bound %d", want,
            _MAX_CONTENT_BYTES)
        return _MAX_CONTENT_BYTES
    return min(want, _MAX_CONTENT_BYTES)


class CreateArtifactRequest(BaseModel):
    """Create an artifact.

    A collection is just an artifact with child edges, so there's one create path:
    with ``container_id`` the new artifact is also edged into that collection (the
    CRUDEASIO *Add*); without it, it's a top-level artifact. Either way the creator
    gets a direct owner grant — ownership is a grant on the artifact, not a function
    of where it lives.
    """
    #: Unknown fields are accepted, and logged by name.
    #:
    #: pydantic v2 defaults to `extra="ignore"`, so a typo'd `contentType` is dropped and the write
    #: answers `201`. `extra="forbid"` is the end state this is measuring towards, and it is not
    #: taken here because the population is unknown: `crystal/dispatcher.py` posts a caller-supplied
    #: body verbatim to this route (`self.mantle.post("/artifacts", body or {})`), so the field
    #: population is whatever arbitrary content-type operations send, and forbidding would convert
    #: an unknown number of `201`s into `422`s with no way to know whose. The log below is what
    #: turns that unknown into a count.
    #:
    #: The two surfaces onto this handler disagree: all seven MCP tools declare
    #: `"additionalProperties": False`, so the same typo is refused over MCP and dropped over REST.
    @model_validator(mode="before")
    @classmethod
    def _warn_about_unknown_fields(cls, data):
        if isinstance(data, dict):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                #: WARNING, not debug: a field the caller believed it sent and this API dropped is
                #: exactly the class of silence that goes unnoticed at debug level.
                logger.warning(
                    "create_artifact ignored %d unknown field(s): %s — accepted with 201 and "
                    "dropped (audit C2)", len(unknown), ", ".join(unknown))
        return data

    container_id: Optional[str] = None   # optional: also add a membership edge into this collection
    source_artifact_id: Optional[str] = None  # link an existing artifact instead of creating one

    identity: Optional[str] = Field(
        None,
        description=(
            "A caller-chosen natural key for the thing this artifact is OF — `file:/repo/README.md`,"
            "`session:7c7bcb7b`. Supplying it makes the write **idempotent**: the id is derived from it"
            "(`services/artifact_identity`), so storing the same thing again UPDATES the artifact rather"
            "than creating a second one, and the caller keeps no id map to lose.\n\n"
            "Omit it and nothing changes — a fresh `uuid4`, exactly as before."
        ),
    )
    name: Optional[str] = None
    context: Optional[Union[Dict[str, Any], str]] = Field(
        None,
        description=(
            "Caller metadata about this artifact, as an OBJECT. A JSON STRING is still accepted "
            "and is deprecated: it exists because this field predates the object form, and 39 "
            "call sites across seven repos still send one. Both are normalised to the same "
            "stored shape, so the two forms cannot disagree."
        ),
    )
    content: Optional[str] = None
    content_type: Optional[str] = None
    description: Optional[str] = None
    # WHERE-indexing hint (Information Gauge DB, Phase 1): "eager" indexes now,
    # "lazy" leaves the artifact latent (indexed on first access). None uses the
    # deployment default (MANTLE_LAZY_INDEX). Applies to content in a container;
    # top-level containers are always indexed eagerly (the navigable frame).
    index: Optional[str] = Field(
        None,
        #: The enum is derived from `search.lazy.INDEX_HINTS`, so the vocabulary is published in
        #: one place and a generated client cannot produce the typo the description warns about.
        #:
        #: Runtime acceptance is deliberately looser than the published enum: `resolve_lazy` falls
        #: an unrecognised value through to the deployment default rather than refusing it.
        #: Tightening that is one `field_validator` and is John's call to make.
        #:
        #: The hint is read at exactly one call site. A top-level artifact is the navigable frame
        #: and is always eager; an `identity` member is always eager too, because
        #: `upsert_identity_member` calls `create_workspace_artifact(enqueue_index=False)` and then
        #: indexes synchronously — an identity artifact is a mirror whose purpose is to be found.
        #: On both of those branches the hint is ignored.
        description=(
            "When to index: `" + "` immediately, `".join(INDEX_HINTS)
            + "` deferred to first access. Unset uses the deployment default. "
            "Applies to a plain member of a collection only. A top-level artifact (the navigable "
            "frame) and an `identity` member (a mirror, whose purpose is to be found) are always "
            "indexed eagerly, and this hint is ignored on both. "
            "An unrecognised value is not refused: `search/lazy.py::resolve_lazy` returns the "
            "deployment default for anything that is neither, so a misspelling resolves to the "
            "default rather than raising."
        ),
        json_schema_extra={"enum": list(INDEX_HINTS)},
    )

    vector: Optional[List[float]] = Field(
        None,
        description=(
            "The semantic arm's ingress. Mantle never embeds, so the only way a vector reaches the vector"
            "arm is a writer handing one over on the write that produced the content it describes."
            "`space_id` is mandatory alongside it — see `api/vectors.py`. Omit both and the write behaves"
            "exactly as before: the vector arm receives nothing and the artifact is lexical-only."
        ),
    )
    space_id: Optional[str] = None


#: Every settable PATCH field repeats this, because a caller reads one field's description and not
#: the class docstring. `null` is not a clear: `update_workspace` leaves `None` fields alone, so an
#: absent field and an explicit null are the same request, and a field can be set but never unset.
_PATCH_LEAVE = (" Omitted — or `null`, which is the same request — leaves it unchanged; there is "
                "no way to CLEAR a field.")


class UpdateArtifactRequest(BaseModel):
    """Fields to change on an artifact. Every field is optional; omitted means "leave alone".

    There is no concurrency control on this request: no `If-Match`, no version and no `updated_at`
    precondition. Two callers updating the same artifact both succeed and the later write wins, so
    a client that read, edited and wrote back has no way to learn that someone else's change was
    overwritten in between.

    `PATCH /artifacts/{id}/children/order` is not the same guarantee. It takes an optional
    `order_version` and refuses a stale one with 409, but that token is over the child order rather
    than over the artifact's fields, so it does not help here.
    """

    context: Optional[Union[Dict[str, Any], str]] = Field(
        None,
        description=(
            "Caller metadata about this artifact, as an OBJECT. A JSON STRING is still accepted "
            "and is deprecated: it exists because this field predates the object form, and 39 "
            "call sites across seven repos still send one. Both are normalised to the same "
            "stored shape, so the two forms cannot disagree."
        ),
    )
    content: Optional[str] = Field(
        None,
        description="Replace the artifact's stored content. A change re-stores the body and "
                    "re-indexes it, so this is the most expensive field to touch." + _PATCH_LEAVE,
    )
    state: Optional[str] = Field(
        None,
        #: Published and enforced, both derived from `Artifact.VALID_STATES`, matching the `400
        #: state must be one of …` that `recall` answers for the same mistake.
        #:
        #: Enforcement is load-bearing rather than cosmetic. `update_workspace` assigns `ws.state`
        #: directly and its only test is whether the artifact is archived, so on an archived
        #: artifact an accepted misspelling such as `"drft"` takes the un-archive branch and moves
        #: it to committed — a state the caller never asked for, answered `200`. The validator
        #: below is what makes that unreachable.
        description="Move the artifact between index segments: "
                    + ", ".join("`%s`" % v for v in _ARTIFACT_STATES)
                    + ". Each is a separately keyed encrypted tree, so a state change re-indexes "
                      "rather than re-labels.",
        json_schema_extra={"enum": _ARTIFACT_STATES},
    )

    @field_validator("state")
    @classmethod
    def _state_must_be_known(cls, v):
        if v is not None and v not in _ARTIFACT_STATES:
            raise ValueError("must be one of %s" % ", ".join(_ARTIFACT_STATES))
        return v
    content_type: Optional[str] = Field(
        None,
        description="The MIME-ish label carried with the artifact. Mantle treats it as an opaque "
                    "label and does not interpret it, so changing it re-labels the artifact and "
                    "converts nothing." + _PATCH_LEAVE,
    )
    name: Optional[str] = Field(
        None, description="The artifact's display name." + _PATCH_LEAVE,
    )
    description: Optional[str] = Field(
        None, description="Prose about the artifact, for a reader rather than the index."
                          + _PATCH_LEAVE,
    )

    vector: Optional[List[float]] = Field(
        None,
        description=(
            "Re-supply the vector when the content it describes changes. Same contract as on create:"
            "optional, but `space_id` comes with it when present."
        ),
    )
    space_id: Optional[str] = Field(
        None,
        description="Names the vector space `vector` belongs to. Supplying it without a `vector` "
                    "is a `400` — *“it names the space of a vector that is not here”* — rather "
                    "than being ignored.",
    )


class InvokeArtifactRequest(BaseModel):
    """Invoke an operator artifact."""
    name: Optional[str] = None              # tool name (for mcp_tool dispatch via $.body.name)
    arguments: Optional[Dict[str, Any]] = None  # tool arguments (for mcp_tool dispatch)
    workspace_id: Optional[str] = None
    artifacts: Optional[List[str]] = None   # context artifact IDs
    input: Optional[str] = None
    params: Optional[Dict[str, Any]] = None



#: What `candidates: true` overrides, said on each field it overrides. The exclusion is two-way and
#: is stated in both directions: `candidate_budget` says it is ignored for ordered recall, and every
#: field this string is appended to says it is ignored in candidates mode.
_IGNORED_IN_CANDIDATES = (
    " Ignored when `candidates: true`: that mode is budget-bounded and unordered, so this is"
    " replaced rather than applied. No error is raised; see `candidates`."
)

#: `query_text`'s own description, carrying the `field:value` grammar. It sits on the field rather
#: than in the handler docstring because a client building an input for `query_text` reads the
#: field, and the grammar is the one part of this API a caller cannot guess.
#:
#: The filterable field list comes from `filterable_field_names`, the same function the parser's
#: own error text uses, so the spec cannot teach a filter the parser would refuse.
_QUERY_TEXT_DESCRIPTION = (
    "What to search for. Required unless `vector` is supplied.\n\n"
    "**Query syntax:** `+term` (must appear), `!term` (exclude), `~term` (semantic),"
    " `=\"phrase\"` (exact).\n\n"
    "**Filters.** `field:value` narrows the recall by artifact metadata before either"
    " retrieval arm runs, so `total` and pagination count FILTERED matches. It can only"
    " ever narrow: the predicate sees authorized artifacts only, so no filter can"
    " surface, or hint at, an artifact you could not already read.\n\n"
    "Filterable: %s.\n\n"
    "Operators: `field:value` (case-insensitive; `a,b` is any-of),"
    " `field:=\"Exact Value\"` (case-sensitive, whole), `!field:value` (negated), and"
    " `field:>value` / `field:<value` on date fields only. Filters conjoin."
    " `field:~value` is not supported.\n\n"
    "`word:value` is a filter only when `word` is one of those fields. Any other word"
    " keeps its colon and searches as an ordinary term, so a URL, `meeting at 3:30` or"
    " `ratio 16:9` are ordinary queries rather than errors. A misspelled field therefore"
    " searches as literal text: `titel:foo` looks for that string."
) % ", ".join("`%s`" % f for f in _filterable_field_names())


class ArtifactRecallRequest(BaseModel):
    """Recall across accessible artifacts. ``query_text`` or ``vector`` is required.

    Unknown fields are ignored (the pydantic default) rather than rejected, so a client sending a
    retired field gets a normal search rather than a 422. The retired set is ``embedding``,
    ``aperture``, ``use_hybrid`` and ``content_types``; none is read on any path.

    This compatibility window has no closing date. Closing it makes an old client that works start
    getting 422s, which is a call about callers in the field rather than about this file. The
    validator below logs every unknown key by name at WARNING, so whether anyone is still sending
    ``use_hybrid`` is answered from the log — the same instrument `CreateArtifactRequest` uses, and
    for the same reason: characterise the population before migrating it.
    """
    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _warn_about_unknown_fields(cls, data):
        if isinstance(data, dict):
            #: `populate_by_name=True` means an alias is also a legal key, so both spellings count
            #: as known — `from_`/`from` must not be reported as a typo.
            known = set(cls.model_fields) | {
                f.alias for f in cls.model_fields.values() if f.alias}
            unknown = sorted(set(data) - known)
            if unknown:
                logger.warning(
                    "recall ignored %d unknown field(s): %s — accepted and dropped. The retired "
                    "set is embedding, aperture, use_hybrid, content_types (audit Q6)",
                    len(unknown), ", ".join(unknown))
        return data

    query_text: Optional[str] = Field(None, description=_QUERY_TEXT_DESCRIPTION)

    vector: Optional[List[float]] = Field(
        None,
        description=(
            "The semantic arm's query-side ingress, and the exact counterpart of `vector` on"
            "`CreateArtifactRequest`: a writer supplies the vector of what it stores, a reader supplies"
            "the vector of what it is looking for. Both are numbers computed elsewhere — Mantle stores,"
            "compares and ranks, and embeds neither of them — so both are validated by `api/vectors.py`"
            "and both require `space_id`.\n\n"
            "It ACCOMPANIES `query_text` rather than replacing it: the lexical arm reads the text, the"
            "semantic arm reads the vector, and RRF fuses whatever each returns. Either may be sent alone"
            "— text alone is lexical recall, a vector alone is kNN."
        ),
    )
    space_id: Optional[str] = Field(
        None,
        description=(
            "Names the embedding space `vector`'s components live in. Required alongside it for the same"
            "reason it is required on a write: it is what makes two vectors comparable, and Mantle cannot"
            "infer it from the numbers."
        ),
    )

    scope: Optional[List[str]] = Field(
        None,
        description=(
            "Restrict the recall to these containers, by id. Narrows by container where"
            " `field:value` narrows by metadata; both intersect the light cone before"
            " retrieval. Naming a container you cannot read narrows the search rather than"
            " widening it, since it reaches nothing you could not already read."
        ),
    )
    state: str = Field(
        "committed",
        description=(
            "Which index SEGMENT to recall from — `committed` (default), `draft` or `archived`.\n\n"
            "Each is a separately keyed encrypted tree under its own object-storage prefix"
            "(`search/mantle/wiring._segment_prefixes`), selected when the accessor is built and before"
            "any query runs. That is why there is no `state:` query filter: a filter narrows a set of"
            "retrieved artifacts, and no draft is in the committed tree to be narrowed out of."
            "`state:draft` in `query_text` is a 400 pointing back at this field."
        ),
    )
    # No `content_types` field: it would do nothing if present, since the handler never passes
    # it on and `SearchQuery` has nowhere to put it, so it would narrow no recall. The one way to
    # say it is the `content_type:` (alias `type:`) query filter, which does narrow. Omitting it
    # costs a client nothing: unknown fields are ignored, so a request sending `content_types`
    # gets exactly the same search either way.
    # No `aperture` field: it is never read; see mantle/search/types.py.
    # No `use_hybrid` field either, and for a sharper reason than "unread": there is nothing
    # left for it to switch. The lexical index NARROWS every recall and the cells RANK what
    # survives — neither is optional, so neither is selectable. What varies is whether a query
    # vector exists for the ranker, and ``vector`` says that already. ``ordering`` on the
    # response reports what actually happened.
    from_: int = Field(
        0,
        ge=0,
        description="How many hits to skip. Floored at 0 and left uncapped, matching the "
                    "`offset` query parameter on the sibling list endpoints.",
    )
    size: int = Field(
        20,
        ge=1,
        le=_MAX_PAGE,
        description="How many hits to return." + _IGNORED_IN_CANDIDATES
                    + " `candidate_budget` sets the size there.",
    )
    sort: Optional[Literal["relevance", "recency"]] = Field(
        None,
        description=(
            "Which ordering you want back.\n\n"
            "`recency` asks for most-recently-updated first and gets it, vector or no vector — and gets"
            "it without decrypting a cell, since the ordering is decided before the ranker runs."
            "`relevance` (the default when unset) asks for the best ordering this recall can produce: a"
            "cosine when a query vector reaches the ranker, otherwise the query's own coverage — how many"
            "of its stems each hit carries. This is a REQUEST and cannot promise an outcome; read"
            "`ordering` on the response for what you got."
        ),
    )
    highlight: bool = Field(
        True,
        description="Return matched-term highlights on each hit."
                    + _IGNORED_IN_CANDIDATES
                    + " Candidates are not hydrated, so there is no text to highlight.",
    )

    candidates: bool = Field(
        False,
        description=(
            "Return the NARROWED CANDIDATE SET instead of ordered, hydrated hits.\n\n"
            "This is the primitive external search flavors build on: they rank within the returned set"
            "and therefore can never widen access (MANTLE §1 holds by construction). It shares this"
            "endpoint rather than living at its own path because everything before the final call —"
            "light-cone resolution, grant-key scoping, segment validation, the `field:value` filter and"
            "the query's own terms — is identical for both. Same universe, no order and no hydration.\n\n"
            "Response shape is `{candidates, model_id}`, each candidate carrying `artifact_id`,"
            "`collection_id` and `principal_id`. THERE ARE NO SCORES ON IT. The former `sse_score` /"
            "`rrf_score` / `source` were a BM25 score, a rank-fusion constant's output and a which-arm-"
            "found-it flag; none of those quantities exists any more, so the keys are gone rather than"
            "null. `model_id` is `null`: nothing here retrieves by embedding."
        ),
    )
    candidate_budget: int = Field(
        200,
        ge=1,
        le=_MAX_PAGE,
        description=(
            "Candidates only: how many to return. The cut is by recency — the query-independent order —"
            "because choosing which candidates a budget keeps by anything the query said would be the"
            "ranking decision this mode declines to make. Ignored for ordered recall, which paginates"
            "with `from`/`size`."
        ),
    )

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
    score: Optional[float] = Field(
        None,
        description=(
            "What put this hit where it is, as a number. READ `ordering` TO KNOW WHICH KIND.\n\n"
            "`ordering: \"semantic\"` — the cosine that ranked it. `ordering: \"reach\"` — how far this"
            "artifact's own position reached toward what the query is about, in spreads above what a"
            "candidate of its size would reach by nothing. Bigger is better and it can be negative, which"
            "is a reading and not an absence. Like the coverage count it is comparable across the hits of"
            "ONE response and across nothing else, because the spread it is measured in is this query's"
            "own. `ordering: \"coverage\"` — the INTEGER COUNT of distinct query stems this artifact"
            "carries. It is not a relevance score: nothing weights it by how rare a term is, how often it"
            "occurs, or how long the field is, and a 2 means \"two of these five stems\" on one query and"
            "\"two of these two\" on another. Compare it across the hits of one response, never across"
            "responses, and never against a fixed threshold. `ordering: \"recency\"` — `null`. Those"
            "results are ordered by when they were last updated, so no number measured them against the"
            "query, and a zero or a rank would be something a client could threshold or re-sort on that"
            "would mean nothing."
        ),
    )
    root_id: str
    version_id: str
    collection_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    highlights: Optional[Dict[str, List[str]]] = None


class RecallCandidate(BaseModel):
    """One member of the narrowed candidate set."""

    artifact_id: str = Field(description="The artifact this candidate names.")
    collection_id: Optional[str] = Field(
        None, description="The collection it was found in.")
    principal_id: Optional[str] = Field(
        None,
        description=("The CELL principal — the collection's origin root, which is what a content"
                     " key is derived per. A candidate names the same owner the index was"
                     " written under."))


class RecallCandidatesResponse(BaseModel):
    """What `candidates: true` returns — a different resource in everything but the URL.

    The operation has two 200 shapes and publishes both as a `oneOf`, because a caller chooses
    between them with `candidates` and needs to know what each looks like before sending.

    A candidate carries no score. This mode narrows and does not rank, so there is no ranking
    quantity to report and the keys are absent rather than null."""

    candidates: List[RecallCandidate] = Field(
        default_factory=list,
        description="The narrowed set, unranked and unhydrated. Rank within it; it cannot be"
                    " widened.")
    model_id: Optional[str] = Field(
        None,
        description="Always `null`: nothing on this path retrieves by embedding.")


class ArtifactRecallResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hits: List[RecallHitResponse]
    total: int
    query_text: str
    parsed_query: Optional[str] = None
    applied_filters: List[str] = Field(
        default_factory=list,
        description=(
            "The `field:value` filters that narrowed this result, canonically spelled.\n\n"
            "`parsed_query` is the whole parse and still includes the inert `@name:value` controls; this"
            "is the narrower, load-bearing statement — every entry was compiled into the predicate that"
            "cut the authorized artifact set before either retrieval arm ran. Nothing parses into a"
            "filter and then quietly fails to appear here: a filter this node cannot apply is a 400, so"
            "an empty list on a query containing `field:value` is impossible."
        ),
    )
    corrections: List[str] = Field(default_factory=list)
    ordering: Literal["semantic", "reach", "coverage", "recency"] = Field(
        description=(
            "What ordered these hits — what happened, not what was asked for. `semantic`: a "
            "query vector reached the ranker, it ranked the narrowed set by cosine, and each "
            "hit's `score` is that cosine. `reach`: no cosine could, but this node has an "
            "ontology bound, so the hits were re-ordered by how far each one reaches toward "
            "what the query is about and CUT where that reach stops — the only text ordering "
            "that returns fewer hits than matched, so `total` counts what survived the cut "
            "rather than every narrowed match. `coverage`: no cosine could, so the hits are "
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


def _context_as_stored(context: Optional[Union[Dict[str, Any], str]]) -> Optional[str]:
    """Normalise either accepted `context` form to the one shape the write path stores.

    Context is an object. Both forms are accepted rather than the string being refused, because
    39 call sites across 31 files in 7 repos send `json.dumps(...)` — mantle's own
    `seed_provisioning` and `issuers`, every chorus persona, prism's CLI, crystal's push, cloud's
    `ci_runner`, and the Claude Code memory hooks. The object is preferred; the string is
    deprecated and retires when those callers move.

    Both forms normalise to the same stored shape, so the two cannot disagree.
    """
    if context is None:
        return None
    if isinstance(context, dict):
        return json.dumps(context)
    return context


#: Published to callers on the two parameter descriptions whose routes filter before paging:
#: `/visible` narrows on `ix_v_ct` and intersects id sets before taking the page, and
#: `list_children` filters the full child list, counts it, then slices. On both, a short page means
#: the filtered set really is that small.
#:
#: One case has no `total`: `/visible` on a truncated light cone never builds the authorized set,
#: so it reports `total: null` and page length is the only signal. That is why `has_more` is the
#: thing to read rather than arithmetic over `total`.
_FILTER_BEFORE_PAGE = (
    " The filter is applied before the page is taken, so `items` is a page of the filtered set "
    "and `total` is the filtered count. Decide whether to continue from `has_more` rather than "
    "from the number of items, since `total` is `null` where the count is genuinely unknown."
)

def _page(items: list, *, total: Optional[int], has_more: bool) -> Dict[str, Any]:
    """The one list-response shape: `{items, total, has_more}`, on every list route.

    `total` is `None` where the count is genuinely unknown rather than zero. On `/visible` with a
    truncated light cone the authorized set is never built — that is what the truncation is for —
    so reporting `0` would be a measurement nobody took. A reader can tell "none" from "not
    counted" only while the two have different values.

    `has_more` is not `len(items)`. Both list routes filter before paging, so where `total` is
    known the caller's `has_more` is `offset + limit < total`; where `total` is `None` — `/visible`
    on a truncated cone — page length is the only signal left and is used instead.
    """
    return {"items": items, "total": total, "has_more": has_more}


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
    read — the half of:func:`_find_artifact` that costs a query per miss. So a page of
    ``n`` ids costs ``n`` seeks plus one lineage read per miss, and a duplicated id is
    read once.

    That is the shape every batch in this codebase has, and it is the store's shape, not
    a shortcut: ``collection_service.get_collection_artifacts_batch`` states it outright
    ("one lineage read per root... there is no cross-root query that answers it in one
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


#: The error contract, declared once and referenced by every route below. `responses=` is
#: spec-only: it changes no runtime behaviour, only what the schema says.
#:
#: Each route's set is derived rather than typed — the transitive closure of `HTTPException` raises
#: reachable from its handler, through the helpers it calls (`get_auth` -> `resolve_auth` -> 401,
#: `check_access` -> 404/500). One code is subtracted from that closure: `check_access` also raises
#: `400 Unknown action`, and all 20 call sites pass a string literal ("read", "create", "update",
#: "delete", "evict"), so that 400 is a programming error in this repo rather than something a
#: client can provoke. Declaring it would tell every client to handle a response it cannot receive.
#:
#: The table itself lives in `mantle/api/errors.py`, because `/grants` needs the same codes and two
#: statements of what a `404` means is the thing that must not happen. The builder below stays here
#: because each surface assembles differently, and this one carries four success-shape keywords.


# ── response envelopes ───────────────────────────────────────────────────────────────────────
#
# Nine routes return a fixed envelope, and each declares it, so a generated client can name the
# keys it will be handed.
#
# Documented with `responses={200: {"model":...}}` rather than with `response_model=`, and the
# difference is deliberate. `response_model=` filters: FastAPI drops any key the model does not
# declare, so the day a handler adds a field and the model is not updated, that field silently
# stops reaching clients — a data-loss bug whose symptom is a missing key nobody can trace.
# `responses=` only describes: a handler returning an undeclared extra key still returns it, while
# the schema still resolves to a `$ref`.
#
# The drift that trade-off accepts — a schema that says one thing while the handler returns another
# — is caught by `tests/test_response_envelopes_match_the_handlers.py`, which compares each model's
# fields against the literal keys of its route's `return` and fails on either kind of mismatch.
#
# Nested payloads are `Any` on purpose. `artifacts`, `commits` and `events` carry artifact and
# commit documents, and an artifact document is open — a content type may add fields, measured on a
# real lattice. Typing them would be the same drop-what-you-did-not-anticipate bug one level down.
#
# `upload-initiate` is declared here too, as `UploadInitiateResponse`. Its key set is knowable
# because `initiate_upload_and_create_artifact` has exactly one return and it is a literal,
# `{upload_id, mode, url, method, key}`.


class WarmResponse(BaseModel):
    """What one warm sweep did, and what it could not do.

    `materialized: 0` alone is ambiguous — nothing needed warming, the sweep raised and was
    swallowed to a `logger.warning`, or it stopped early all produce it, and all three answer
    `200`. `examined`, `failed`, `truncated` and `complete` are what let a caller tell them apart,
    for the same reason `_page` reports `total: None` rather than `0`."""

    collection_id: str = Field(description="The container that was swept.")
    materialized: int = Field(
        description="Members newly enqueued for indexing by this call. Already-materialized "
                    "members are skipped, so a re-run reports only what it added.")
    examined: int = Field(
        description="Distinct members looked at, whether or not they needed warming.")
    failed: int = Field(
        description="Members whose enqueue raised. Non-zero with `complete: true` means the sweep "
                    "finished but did not warm everything it examined.")
    truncated: bool = Field(
        description="The sweep stopped at its bound and more members remain. Re-run once the "
                    "enqueued work has been indexed: a member counts as materialized when its "
                    "index job completes rather than when it is enqueued, so a sweep re-run "
                    "immediately examines the same members again and reaches no further.")
    complete: bool = Field(
        description="The sweep ran to the end without the member listing itself failing. `false` "
                    "means every other number here is partial, which is what separates 'nothing "
                    "to do' from 'could not tell'.")


def _delete_counts(counts: Any) -> "tuple[int, int, int]":
    """`(detached, destroyed, refused)` from whatever the delete service handed back.

    Three named integers, never the service's dict. One delete call path returns an artifact rather
    than counts, so spreading the return into the response body would put the whole artifact
    document in a delete response: the shape becomes whatever the callee returns that day, and
    nothing declares it.

    This sanitises and returns a tuple; the response literal stays at the return site. That split
    is what `test_response_envelopes_match_the_handlers` needs — it reads `return {...}` literals
    out of each handler and cannot check an envelope built somewhere else.
    """
    c = counts if isinstance(counts, dict) else {}

    def _n(key: str) -> int:
        v = c.get(key, 0)
        return v if isinstance(v, int) else 0

    return _n("detached"), _n("destroyed"), _n("refused")


class DeleteArtifactResponse(BaseModel):
    """What the delete did, not merely that it happened.

    `deleted: true` alone cannot distinguish a `cascade=true` over a 1.85M-member collection from
    deleting a note, and the difference a caller most needs is the one those counts carry:
    detached is recoverable, destroyed is not.
    """
    id: str
    deleted: bool
    detached: int = 0
    destroyed: int = 0
    refused: int = Field(
        0,
        description=(
            "Members this cascade asked to destroy and could not, so they were evicted from this "
            "container instead and still exist. Non-zero means the call did less than it was "
            "asked to: the caller holds no delete right on those artifacts and this container is "
            "not their origin parent, so destroying them would reach past what the container "
            "grant covers. Always present; `0` on an ordinary delete."
        ),
    )


class RemoveItemResponse(BaseModel):
    id: str
    removed: bool
    container_id: Optional[str] = None


class PageResponse(BaseModel):
    """The one list-response shape — `{items, total, has_more}` — for every list route.

    One model, not one per route. `_page` is the single producer, so a second model would be a
    second place for the shape to be stated and a second place for it to drift.

    `total` is `Optional` because `_page` sends `None` where the count is genuinely unknown rather
    than zero: on a truncated light cone the authorized set is never built, so `0` would be a
    measurement nobody took. Typing it `int` would erase that distinction.
    """

    items: List[Any]
    total: Optional[int] = None
    has_more: bool


class ArtifactResponse(BaseModel):
    """An artifact document, with a fixed key set: every declared key is always present.

    `Artifact.to_dict` omits unset fields, so a raw entity dict varies its key set with the data
    rather than with the branch — `content_type`, `description`, `name` and `origin_root` come and
    go, and `origin_root` appears only when `collection_id` does not. `_artifact_body` fills the
    declared keys so a caller can rely on them: absent means "this API might not send it", `null`
    means "there is none", and only the second can be programmed against.

    Declared here, not enforced with `response_model=`. Enforcing would filter, and a model falling
    behind its handler would then silently stop returning a field. The handler builds the body;
    this documents it.

    The schema is open — `extra="allow"`, publishing `additionalProperties: true`. An artifact
    document is open by design: `to_dict` emits up to eighteen keys, a task artifact carries
    seventeen store-level fields, and a content type may add more. A closed schema would be a
    documented promise that the API sends nothing else, which is false. Open tells the generated
    client both halves of the truth: these keys are always present, and more may arrive.
    """
    model_config = ConfigDict(extra="allow")

    id: str
    root_id: Optional[str] = None
    collection_id: Optional[str] = None
    state: Optional[str] = None
    content: Optional[str] = None
    context: Optional[str] = None
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    #: Optional in the data and always present in the response, `null` when unset.
    name: Optional[str] = Field(None, description="`null` when the artifact has no name.")
    description: Optional[str] = Field(None, description="`null` when there is none.")
    content_type: Optional[str] = Field(None, description="`null` when none was supplied.")
    origin_root: Optional[str] = Field(
        None,
        description=("The root this artifact's grants descend from. Present only for a top-level "
                     "artifact — one with no `collection_id` is its own origin root — and `null` "
                     "for a member, whose origin is reached through its container."))


#: The key set `ArtifactResponse` promises, derived from the model so the two cannot drift.
_ARTIFACT_KEYS: tuple = tuple(ArtifactResponse.model_fields)


def _artifact_body(doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Fill the declared keys with `None` where absent, and drop nothing the document carries.

    Additive, never filtering. Selecting `_ARTIFACT_KEYS` out of the document instead would be a
    hand-written `response_model`, and it would drop every key the document carries that the model
    does not declare — `created_by`, `modified_by`, `content_ref`, `content_encrypted`, `lemmas`,
    `colimit_of`, and on a task artifact seventeen store-level fields at once.

    `to_dict` is what decides what a response carries, so the declared set is measured against the
    serialiser rather than against `Artifact(...)` constructor arguments: `created_by` is assigned
    as an attribute and never appears in a constructor sweep.

    The result is that every key the document carries survives, and the declared ones are
    guaranteed present so a caller can rely on them without testing for existence.
    """
    d = dict(doc or {})
    for k in _ARTIFACT_KEYS:
        d.setdefault(k, None)
    return d


class ArtifactDetailResponse(ArtifactResponse):
    """`GET /artifacts/{artifact_id}` — the artifact document plus what a read computes.

    The same fixed key set as every write, and two more. Neither is stored: both are resolved per
    read from the containment index, which is why they appear here and not on the model the writes
    return. The document is one shape whether it is written or read.
    """
    has_children: bool = Field(
        False, description="Whether this artifact holds any members. Computed per read.")
    child_count: int = Field(
        0, description="How many members it holds — `0` when it holds none, and `0` without a "
                       "count being taken when `has_children` is false.")


#: The read side's key set, adding the two fields a read computes. Derived from the subclass the
#: same way, so adding a field to the model cannot leave the guarantee behind.
_ARTIFACT_DETAIL_KEYS: tuple = tuple(ArtifactDetailResponse.model_fields)


def _artifact_detail_body(doc: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """`_artifact_body`, plus the two fields a read computes. Drops nothing either.

    The guarantee is a function rather than a convention so that a static check can read it.
    `read_artifact` assigning `has_children` and `child_count` onto the document is correct at
    runtime but invisible to `test_response_envelopes_match_the_handlers`, which would then see a
    model promising two fields it cannot find. Deriving `_ARTIFACT_DETAIL_KEYS` from
    `ArtifactDetailResponse` guarantees a field added to the model on the same commit.
    """
    d = dict(doc or {})
    for k in _ARTIFACT_DETAIL_KEYS:
        d.setdefault(k, None)
    return d


class UploadInitiateResponse(BaseModel):
    """The upload handshake.

    The key set is fixed here rather than spread from the service's return.
    `initiate_upload_and_create_artifact` has exactly one return and it is the literal
    `{upload_id, mode, url, method, key}`, so the shape is knowable and is declared. Spreading an
    internal return value into an API response makes the published shape whatever the callee
    happens to return that day, with nothing declaring it.
    """
    upload_id: str = Field(description="The artifact the bytes will be written to.")
    mode: str = Field(description="How to upload. `proxied` means PUT the bytes to `url` on this "
                                  "node; the decrypt and encrypt happen here, so there is no "
                                  "direct-to-object-store mode while content is encrypted at rest.")
    url: str = Field(description="Where to PUT the bytes — a path on THIS node, not a signed URL.")
    method: str = Field(description="The HTTP method to use against `url`.")
    key: Optional[str] = Field(None, description="The content key the bytes will be stored under.")
    artifact: Optional[Dict[str, Any]] = Field(
        None, description="The artifact created to receive the upload, or `null` if none was.")


class ContentUrlResponse(BaseModel):
    url: str


class PutContentResponse(BaseModel):
    stored: bool = Field(description="The bytes are durable on this node.")
    size: int = Field(description="Bytes stored, as received.")
    content_key: Any = Field(None, description="The envelope key this body was sealed under.")
    content_ref: Any = Field(None, description="Content-addressed ref for the stored object.")
    deduplicated: bool = Field(
        description="The identical bytes were already stored, so this call wrote nothing. "
                    "Not an error and not a different outcome — the artifact ends up in the "
                    "same state either way.")
    mirror_pending: bool = Field(
        False,
        description=(
            "A mirror this node is configured for could not be reached, so a mirror leg is"
            " still owed and has been queued. The write itself succeeded and the bytes are"
            " durable locally; until the queued task runs, this node is the only copy. A"
            " caller that cares about off-node durability reads `true` as 'not yet"
            " replicated'."
        ),
    )


class ReorderResponse(BaseModel):
    order_version: int


class AccessLogResponse(PageResponse):
    """A page, plus the artifact it is about. `get_artifact_access_log` returns
    `{"artifact_id":..., **_page(...)}`, and inheriting says that rather than restating it."""

    artifact_id: str


#: The two roles `{artifact_id}` plays. Every path parameter is spelled `{artifact_id}` and never
#: `{container_id}`, because the universal-container model says a container is an artifact and a
#: code generator seeing two templates emits two unrelated resources. The spelling therefore
#: carries no hint about which role the id plays on a given route, so the description carries it.
_ARTIFACT_PARAM = (
    "The artifact this operation acts on. Accepts a version id or a root id — a root id resolves "
    "to the current version."
)

#: On these routes the id names the container and the operation acts on what it holds. Same
#: parameter name, different subject: `POST /{artifact_id}/warm` materializes the container's
#: members and `PATCH /{artifact_id}/children/order` reorders its children, and neither touches
#: the artifact named in the path.
_CONTAINER_PARAM = (
    "The CONTAINER this operation acts on — every artifact can hold others, so this is an ordinary "
    "artifact id. The operation applies to what it holds, not to the artifact itself. Accepts a "
    "version id or a root id."
)


#: The member role, on `DELETE /{artifact_id}/children/{child_id}`. The container is the first
#: path segment and the member is the second, so the two ids read container-first like every other
#: two-id operation here, and the request carries no body.
_MEMBER_PARAM = (
    "The artifact to DETACH — the member. The container it is detached FROM is the first path"
    " segment, and `evict` is checked against that container rather than against this id:"
    " detaching is a container operation and the member survives it. Accepts a version id or a"
    " root id."
)

def _errors(*codes: int, ok: Optional[type] = None, created: Optional[type] = None,
            no_content: Optional[str] = None,
            updated: Optional[str] = None, binary: Optional[str] = None) -> dict:
    """`responses=` for one route. Declaration only — no runtime effect.

    `ok=` documents the 200 envelope. It is deliberately NOT `response_model=`: that filters, and a
    model which falls behind its handler would silently stop returning a field.

    `no_content=` declares a 204, which is a success rather than an error. It takes a sentence
    rather than a flag because the interesting part is never the code, it is which no-op the route
    is reporting, and it stays off the `*codes` list because those index `_ERROR_DESCRIPTIONS`."""
    out: dict = {c: {"description": _ERROR_DESCRIPTIONS[c]} for c in codes}
    if ok is not None:
        out[200] = {"model": ok}
    #: `updated=` declares a 200 on a route whose default is 201. It takes a sentence for the
    #: same reason `no_content=` does: the interesting part is which write turned out not to be a
    #: create.
    if updated is not None:
        #: A 200 carries both a description and a model where a route needs them: `POST
        #: /artifacts` says "nothing was created" and returns the artifact document.
        out[200] = dict(out.get(200) or {}, description=updated)
    #: `created=` documents the 201 body. `ok=` reaches only the 200, so a route whose success is
    #: a create needs this to declare what it returns.
    if created is not None:
        out[201] = {"model": created}
    if no_content is not None:
        out[204] = {"description": no_content}
    #: `binary=` declares a bytes 200. FastAPI infers `application/json` for every 200 it is not
    #: told otherwise about, and the content route returns `Response(content=…)`, so without this
    #: a generated client builds a JSON parser for a body that is never JSON.
    #:
    #: Published as `application/octet-stream` rather than the artifact's own media type, which is
    #: per-artifact and known only at request time. The response's `Content-Type` header carries
    #: the specific value, and that is what a client reads.
    if binary is not None:
        assert ok is None and updated is None, "the 200 is described once"
        out[200] = {
            "description": binary,
            "content": {"application/octet-stream":
                        {"schema": {"type": "string", "format": "binary"}}},
        }
    return out


# ---------- GET /artifacts/visible — list artifacts the caller can read ----------
#
# Browser UX needs "show me every workspace / collection I can see" without
# having to know a parent ID. /recall requires something to rank on — query text, a
# query vector, or both — and is for relevance-ranked queries; this is the flat-list
# affordance, scoped through the canonical
# LightConeResolver (same ACL path /recall uses internally).
def _ids_of_content_type(store_db, content_type: str) -> set:
    """Every artifact id with this content type, from the `ix_v_ct` index.

    This is what makes filter-before-page affordable. The cost is proportional to how many
    artifacts carry the type rather than to how much the caller can see, so intersecting it with
    the authorized set replaces "hydrate everything the caller can read" with two set operations.

    Archived versions are excluded. A versioned artifact has one committed row and N archived
    snapshots sharing its root, so including them would make a listing report the same thing once
    per past version. The sensor artifacts are exactly this shape: one per property, re-versioned
    on every real change.
    """
    ids = set()
    for doc in store_db.artifacts.list_artifacts(content_type=content_type):
        if doc.get("state") == "archived":
            continue
        aid = doc.get("id")
        if aid:
            ids.add(aid)
    return ids


@router.get(
    "/visible",
    summary="List artifacts the caller may act on",
    description=(
        "The flat-list affordance: everything reachable through the caller's light cone, "
        "without needing a parent id. Paginated over the authorized set in id order, and "
        "hydrated a page at a time — an unbounded list here grows with the caller's whole "
        "grant reach. An absent artifact may exist and simply not be visible."
    ),

    responses=_errors(400, 401, ok=PageResponse),
)
async def list_visible(
    content_type: Optional[str] = Query(
        None,
        description="Filter by exact content_type (MIME). Omit to list every accessible artifact."
                    + _FILTER_BEFORE_PAGE,
    ),
    action: str = Query(
        "read",
        #: The enum is derived from `ACTIONS`, so a caller learns the whole vocabulary from the
        #: spec rather than discovering it through a 400. A `Literal[...]` cannot be used here:
        #: `ACTIONS` is a runtime tuple, and restating it as literals would be a second home for
        #: the vocabulary that drifts silently.
        description=(
            "CRUDEASIO action to filter by. Returns only artifacts the caller may perform "
            "this action on. Defaults to 'read' (everything visible). Use 'create' to list "
            "collections an artifact may be assigned into — read-only platform collections "
            "are excluded. Permitted values, published as this parameter's enum: "
            + ", ".join(ACTIONS) + "."
        ),
        json_schema_extra={"enum": list(ACTIONS)},
    ),
    limit: int = Query(100, ge=1, le=1000, description="Page size."),
    offset: int = Query(0, ge=0, description="How many authorized artifacts to skip."),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    from mantle.entities.grant import mask_of

    if action not in ACTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    from mantle.db.edge import EdgesTruncated
    from mantle.search.mantle.lightcone import LightConeResolver, authorized_page

    resolver = LightConeResolver(store_db)

    # First-login provisioning is keyed on READ access (the baseline seed grant
    # set). A user with nothing readable has not yet been granted the platform
    # seed collections — provision them now (idempotent, safe on every startup
    # after a factory reset). Always resolved against "read", never the requested
    # action, so e.g. ?action=create does not retrigger provisioning for users
    # who legitimately have no create grants.
    # Each `resolve` is a light-cone BFS over the grant graph — the most expensive read this
    # router issues, and unbounded in the caller's reach — so it goes to a worker thread whole.
    # ── the cone may be too large to hold, and that must not be a 500 ──────────────────────
    # `resolve` walks down and materialises the whole cone. Measured on a principal granted on
    # `stage.0.lexicon` (1.85M members), it raises `EdgesTruncated` after 34-51s, and the raise
    # costs that principal this endpoint for every other collection it holds too.
    #
    # `authorized_page` answers the same question by walking UP per candidate instead — the same
    # check, never the whole set. It is the fallback rather than the default because it scans, so
    # it is fast exactly when `resolve` is slow (a dense cone) and slow when `resolve` is fast.
    truncated_cone = False
    try:
        read_authorized: Set[str] = (
            await offload_sync(resolver.resolve, auth.user_id, "read") if auth.user_id else set()
        )
    except EdgesTruncated:
        logger.info("light cone too large to materialise for %s; paging by candidate instead",
                    auth.user_id)
        truncated_cone = True
        read_authorized = set()
    #: `and not truncated_cone` is load-bearing. The `except` above sets `read_authorized` to an
    #: empty set so the scanning pager can answer, but "too many edges to materialise" and "this
    #: principal has no grants at all" are opposite states producing the identical empty set, and
    #: only `truncated_cone` separates them. Without this clause a principal with a dense cone
    #: takes the first-login path on every request: a multi-write provisioning transaction on a
    #: GET, then a re-resolve that truncates again — and it falls on exactly the accounts the
    #: fallback exists to serve. `test_a_truncated_cone_is_not_a_new_user.py` pins one
    #: provisioning call per GET to zero.
    if auth.user_id and not read_authorized and not truncated_cone:
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
    # added its resource to the caller's visible set, which is in the listing path.
    # One call now asks the whole question, through the same operator the light cone above
    # resolved with, so the two cannot drift apart.
    if (auth.bearer_grant and auth.bearer_grant.resource_id
            and mask_of(auth.bearer_grant).allows(action)):
        authorized.add(auth.bearer_grant.resource_id)

    # Page over the authorized ids BEFORE hydrating them. The set is the caller's whole
    # grant reach, so hydrating it entirely is work proportional to how much someone can
    # see rather than to how much they asked for. A stable sort makes the page well-defined:
    # a set has no order, so offset would otherwise index into a different list each call.
    # The `content_type` filter is applied here, before paging, and the order is load-bearing.
    # Filtering within the hydrated page instead returns whichever artifacts of that type happen to
    # fall inside the first `limit` ids of a set sorted by id, which at scale is almost none of
    # them. Measured on node 71/home over 10,393 artifacts: `sensor+json` returned 0 of 4 matches
    # at `limit=3` and 1 of 4 at `limit=1000`, and `host.71` sits at offset ~10,300 because `h`
    # orders after every hex-leading UUID, so no reasonable page size reached it.
    #
    # Filtering first does not mean hydrating first. `vertex.list_artifacts(content_type=)` narrows
    # on the `ix_v_ct` index, so the candidate set is proportional to how many artifacts carry that
    # type rather than to how much the caller can see, and intersecting two id sets costs nothing.
    if content_type:
        typed = await offload_sync(_ids_of_content_type, store_db, content_type)
        authorized = authorized & typed
        if truncated_cone:
            # The cone was truncated, so `authorized` is empty and so is the intersection above.
            # Fall back to the scanning pager and keep filtering per document: correct but slow,
            # the same trade the truncation path makes everywhere else.
            authorized = set()

    if truncated_cone and not content_type:
        # The set was never built, so there is nothing to sort: the page comes back already in id
        # order, which is the order `sorted(authorized)` produced.
        page = await offload_sync(
            authorized_page, store_db, resolver, auth.user_id,
            action=action, offset=offset, limit=limit,
        )
        #: Unknown, and reported as unknown. Counting it means building the set the truncation
        #: exists to avoid, so the price of a `total` here is the price of not truncating.
        total: Optional[int] = None
    elif truncated_cone:
        page = await offload_sync(
            authorized_page, store_db, resolver, auth.user_id,
            action=action, offset=offset, limit=limit,
        )
        total = None
    else:
        page = sorted(authorized)[offset:offset + limit]
        #: The filtered count. The intersection above has already narrowed `authorized`, so this
        #: counts what matches rather than everything the caller can reach.
        total = len(authorized)

    # One hop for the whole page: `_hydrate_batch` is up to `limit` store reads, which is the
    # largest single block of blocking work in this router.
    hydrated = await offload_sync(_hydrate_batch, store_db, page)

    # A safety net rather than the filter: the narrowing happened before paging, above. It is kept
    # because the two answers come from different places — an id set from the `ix_v_ct` index, and
    # the hydrated document — and the document is what the caller is handed. A row whose indexed
    # `ct` and stored `content_type` differ is a writer bug and is excluded rather than served.
    results: list = []
    for aid in page:
        doc = hydrated.get(aid)
        if not doc:
            continue
        if content_type and doc.get("content_type") != content_type:
            continue
        results.append(doc)
    #: Computed from `total`, which is the filtered count: `page` is a slice of the filtered set,
    #: so `offset + limit < total` is exact. Page length is not — it reports `has_more: true` for
    #: a final page that happens to be exactly `limit` long, sending a caller after a page that
    #: does not exist. `list_children` uses the same arithmetic.
    #:
    #: The truncated-cone path has no `total`, because the set it would count is the one the
    #: truncation exists to avoid building, so page length is the only signal available there.
    more = (offset + limit < total) if total is not None else (len(page) == limit)
    return _page(results, total=total, has_more=more)


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

    responses=_errors(
        400, 401, 403, 404, 413, 500,
        updated=("NOTHING WAS CREATED — the write resolved to an existing artifact. Two "
                 "requests produce this. (1) An `identity` that already names an artifact makes "
                 "the write an UPDATE, which is the whole reason to supply one. (2) A "
                 "`source_artifact_id` LINKS an artifact that already exists into the collection: "
                 "a membership edge is created, no artifact is. In both cases the body is the "
                 "artifact document, and only this status distinguishes them from a `201`."),
        ok=ArtifactResponse, created=ArtifactResponse,
    ),
)
async def create_artifact(
    request: Request,
    response: Response,
    body: CreateArtifactRequest,
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
    #: The route's default is `201 Created`, and two paths through the create do not create
    #: anything: an `identity` that already names an artifact is an update, top-level or inside a
    #: collection. `_default_create_artifact` reports which it did through `report=`, because its
    #: return value is the artifact document either way and cannot carry the distinction.
    outcome: Dict[str, Any] = {}
    result = await _default_create_artifact(body, auth, store_db, report=outcome)
    if outcome.get("created") is False:
        response.status_code = status.HTTP_200_OK

    #: The whole document, with the declared keys guaranteed present. `_artifact_body` fills every
    #: declared key from the document, `null` where there is none, and drops nothing else the
    #: document carries — `created_by`, `modified_by`, `content_ref`, and on a task artifact
    #: seventeen store-level fields.
    doc = _artifact_body(result)
    return _artifact_body(result)


async def _default_create_artifact(
    body: CreateArtifactRequest,
    auth: AuthContext,
    store_db: Any,
    report: Optional[Dict[str, Any]] = None,
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
    # The body is declared as `CreateArtifactRequest` on the route signature rather than taken as
    # a raw dict and parsed here. Three properties follow from that:
    #
    # 1. A validation failure is FastAPI's `RequestValidationError` and answers 422 naming the
    # field. Constructing the model inside the handler raises `ValidationError`, which
    # nothing converts, so it reaches `main.py`'s catch-all handler and answers 500 —
    # telling a caller who omitted a field that the server broke.
    # 2. The body is advertised with its field names, so a generated client can name them.
    # 3. `CreateArtifactRequest` reaches `components.schemas`, so the field documentation on it
    # is published rather than staying in the source.
    #
    # Acceptance is unchanged by the declaration: `CreateArtifactRequest(**body)` and
    # `CreateArtifactRequest.model_validate(body)`, which is what FastAPI calls, produce identical
    # results and both ignore unknown keys (`model_config` is empty and pydantic v2 defaults to
    # `extra="ignore"`). Neither this function nor `create_artifact` reads a raw key off the dict.
    parsed = body

    #: Checked before anything is minted, hashed or encrypted, all of which scale with size.
    _refuse_oversize_inline(parsed.content, "POST /artifacts")

    #: Refuse the combinations that would drop `source_artifact_id` instead of honouring it.
    #: Two request shapes reach a branch that never reads the parameter:
    #:
    #: * without `container_id`, the top-level branch above returns first, so the caller would
    #: get a new empty artifact where it asked for a link to an existing one;
    #: * with `identity`, the identity branch inside the collection path returns first, so an
    #: identity-derived member would be written instead of the link.
    #:
    #: Both would be silent data loss on a write: the request succeeds and does something the
    #: caller did not ask for. A `400` naming the conflict is the only answer that cannot be
    #: mistaken for what the caller wanted.
    if parsed.source_artifact_id:
        if not parsed.container_id:
            raise HTTPException(
                status_code=400,
                detail="source_artifact_id links an existing artifact INTO a collection; "
                       "supply container_id. Without it this would have created a new empty "
                       "artifact and ignored the link.")
        if parsed.identity:
            raise HTTPException(
                status_code=400,
                detail="identity and source_artifact_id are mutually exclusive: identity derives "
                       "the id of an artifact this call writes, source_artifact_id links one that "
                       "already exists. Supply one.")

    context_str = _merge_content_type_into_context(
        _context_as_stored(parsed.context), parsed.content_type)
    # Every create passes through here, top-level and child alike, so this is the one place the
    # store can guarantee an artifact is minted with the context it was minted in. See `_mint_context`.
    context_str = _mint_context(
        context_str,
        db=store_db,
        artifact_id=_derived_identity_id(auth.user_id, parsed.identity),
        content_type=parsed.content_type,
        content=parsed.content,
        container_id=parsed.container_id,
    )
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
            #: 200 rather than the route's default 201. An `identity` naming an existing artifact
            #: makes this an update, which is the point of supplying one, and `201 Created` on an
            #: overwrite is a claim the response cannot support.
            if report is not None:
                report["created"] = False
            #: The identity-update path: nothing was created, and the body is the artifact
            #: document, so it owes the same declared keys every other write returns.
            #: `_artifact_body` guarantees those are present and drops nothing else the document
            #: carries. Returning it rather than a literal means this branch carries no
            #: statically documented envelope, which is the trade an open document requires.
            body = _artifact_body(updated.to_dict())
            return body

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

    # Into a collection — the caller needs create/Add permission on it.
    await offload_sync(check_access, auth, parsed.container_id, "create", store_db)
    #: Offloaded because `_artifact_exists` is a synchronous SQLite read, and this is on the create
    #: path, so every write to a collection pays it. Called directly from an `async def` handler it
    #: would hold the event loop for the whole read while every other in-flight request waits.
    if not await offload_sync(_artifact_exists, store_db, parsed.container_id):
        raise HTTPException(status_code=404, detail="Container not found")

    if parsed.identity:
        # Identity works inside a collection, and it names the root.
        #
        # A member born as a draft acquires a second live version on edit-after-commit, so "the
        # artifact for this identity" would not be one row. That holds for a member with a draft
        # lifecycle and not for what `identity` describes:
        # a mirror of something that lives outside the store — a file, a session, a commit —
        # which has exactly one true version because the writer is restating it.
        #
        # The version ambiguity was never in the identity, it was in the lifecycle. Naming the
        # ROOT removes it: the root is what edges, grants and both search arms are already keyed
        # on (`pipeline_unified` indexes by `root_id`), and `upsert_identity_member` writes the
        # committed version in place instead of forking a draft. Ordinary members are untouched.
        #
        # Why this is worth supporting: without it every hook artifact is top-level, and a
        # top-level artifact is its own origin root — which is the SSE principal
        # (`search/mantle/principal.resolve_cell_principal`). 115 self-rooted artifacts are 115
        # separately keyed owners, and a recall must read every one of them. Containment collapses
        # that to one, so this parameter combination is the difference between recall
        # cost tracking the corpus and not.
        derived_root = _derived_identity_id(auth.user_id, parsed.identity)
        #: The upsert cannot say which half it did through its return value — both branches hand
        #: back an indistinguishable `ArtifactEntity` — so it reports into this dict instead. A
        #: member upsert that overwrites must not answer `201 Created`.
        upsert: Dict[str, Any] = {}
        entity = await offload_sync(
            workspace_service.upsert_identity_member,
            store_db,
            auth.user_id,
            parsed.container_id,
            derived_root,
            context=context_str or "",
            content=parsed.content or "",
            content_type=parsed.content_type,
            name=parsed.name,
            description=parsed.description,
            vector=vector,
            report=upsert,
        )
        if report is not None and upsert.get("created") is False:
            report["created"] = False
        return entity.to_dict()

    # source_artifact_id -> LINK an existing artifact in (edge only), no new artifact.
    if parsed.source_artifact_id:
        #: A link mints a membership edge and no artifact, and the body it returns is the source
        #: document — which existed before this call and is byte-identical after it. The 200 is
        #: what distinguishes a link from a create; `201 Created` with that body would tell a
        #: client "here is the artifact you just created".
        #:
        #: The body is the artifact rather than the edge, because the caller wants the artifact
        #: it just filed and no edge resource is addressable anywhere else in this API.
        if report is not None:
            report["created"] = False
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
        description=parsed.description,
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


def _mint_context(
    context_str: Optional[str],
    *,
    db: Any,
    artifact_id: Optional[str],
    content_type: Optional[str],
    content: Optional[str],
    container_id: Optional[str],
) -> Optional[str]:
    """Stamp the db-level context onto every create. The thin arm; see `services/mint_context`.

    On the write path rather than in an ingest service, because a caller can always write around
    an ingest service and context is not derivable from a row afterwards: a mint that does not
    record it loses it permanently. `pipeline_unified` treats `doc['context']` as the offer, so a
    namespace ingested without one has no offer and recall answers it from a fallback heading.
    Bulk-ingesting straight into the lattice is how that happens. The store's own create path is
    the only place the store can guarantee it.

    In this repo rather than in a tekton, because mantle ships Apache so a store can be taken,
    built on and shipped by anyone, and chorus is AGPL: a mint that needed a tekton would put an
    AGPL service on the critical path of an Apache store.

    The caller's context is preserved verbatim under `caller` and never merged into the facets
    the store observed. A caller may assert any `sha256` it likes, and the store's own must stay
    evidence about the bytes it actually holds.
    """
    from mantle.services import mint_context as _mc

    caller_ctx: Dict[str, Any] = {}
    if context_str:
        try:
            parsed_ctx = json.loads(context_str)
            if isinstance(parsed_ctx, dict):
                caller_ctx = parsed_ctx
        except (json.JSONDecodeError, TypeError):
            # Opaque, non-JSON context. Kept whole rather than dropped: it is the caller's
            # statement about its own artifact and this function is not entitled to discard it.
            caller_ctx = {"_opaque": context_str}
    try:
        minted = _mc.mint(
            db,
            artifact_id=artifact_id or "",
            content_type=content_type,
            content=content,
            collection_id=container_id,
            caller=caller_ctx or None,
        )
    except Exception:  # noqa: BLE001
        # A description must never fail a write. The caller's context stands unchanged, and the
        # coverage gate reports the miss — which is the honest outcome, and distinguishable from
        # a mint that recorded an empty screen.
        logger.warning("mint_context failed; the write proceeds with the caller's context only",
                       exc_info=True)
        return context_str
    return json.dumps(minted)


# ---------- GET /artifacts/{artifact_id} — Read ----------

@router.get("/{artifact_id}",
    responses=_errors(401, 404, 500,
        ok=ArtifactDetailResponse,
    ),
)
async def read_artifact(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Read a single artifact by ID.

    This GET writes. Two side effects, both deliberate, neither guessable from the verb:

    1. **Every read is witnessed.** `check_access` walks origin edges upward and records the
       authorization decision — allow or deny — to the access-audit log. That is several queries
       plus a write on the front of every read, and it is by design: auditability is a property of
       access itself here, not an option.
    2. **The first authorized read can INDEX the artifact.** Under `MANTLE_LAZY_INDEX` a latent
       vertex is materialized on first genuine access, so the read that finds it is the write that
       makes it searchable. A no-op once materialized, and a no-op with lazy indexing off.

    So a client polling this endpoint is generating audit records, and a client reading a latent
    artifact is triggering an index write. Neither is an error; both are worth knowing before
    building a poll loop.

    The response also carries two computed fields that are not stored: `has_children` and
    `child_count`, resolved per read from the containment index."""
    # `check_access` walks origin edges upward and witnesses the decision to the audit log —
    # several queries plus a write, on the front of every read.
    await offload_sync(check_access, auth, artifact_id, "read", store_db)

    doc = await offload_sync(_find_artifact, store_db, artifact_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    # The shared normaliser, so this route answers with the same shape as every other route that
    # returns an artifact document (`_hydrate_batch` ×2, which is what `/visible` pages through;
    # `list_children`; `_fetch_authorized_docs`, which is `/batch`).
    #
    # Inlining the `_id`/`_rev`/`_key` half of it here instead omits the second half: an artifact
    # stored without `context` or `content` then comes back from `/batch` as `context: ""` and
    # `content: ""` and from here with those keys absent, so a client reading `doc["content"]`
    # works against one endpoint and raises KeyError against the other.
    doc = _normalize_artifact_doc(doc)

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
        # Non-fatal and not silent. A materialization that fails leaves the artifact latent, and
        # nothing on any schedule looks for one, so a bare `pass` here would mean the vertex
        # never becomes searchable with nothing recorded anywhere.
        #
        # Swallowed on purpose: the read is the fact and indexing is the announcement of it, so a
        # failed announcement must not fail the read. Logged at WARNING so it is not also silent.
        logger.warning("lazy materialization failed for %s; it stays latent",
                       doc.get("id") or artifact_id, exc_info=True)

    #: The fixed key set, plus the two fields a read computes, so the same artifact comes back in
    #: the same shape from a `GET` as from a `POST`.
    body = _artifact_detail_body(doc)
    #: The whole document. `_artifact_body` guarantees the declared keys are present and drops
    #: nothing else the document carries. Returning it rather than a key literal means this
    #: route carries no statically documented envelope, which is the trade an open document
    #: requires: it cannot be declared as a closed one without becoming one.
    return body


# ---------- POST /artifacts/{artifact_id}/warm — Warm-sweep (lazy indexing) ----------

@router.post("/{artifact_id}/warm",
    responses=_errors(401, 404, 500, ok=WarmResponse),
)
async def warm_collection_endpoint(
    artifact_id: Annotated[str, Path(description=_CONTAINER_PARAM)],
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Warm-sweep guardrail: materialize every latent artifact in a collection so
    it is searchable up front, rather than waiting for each to be accessed. For
    corpora that must be searchable immediately.

    Requires `update` on the container rather than `read`. This is a write sweep: it materializes
    every latent member, enqueueing an index job for each, and the number of them is the
    container's size, chosen by the caller when they choose the container. Gating it on `read`
    would let a read grant buy an unbounded number of writes.

    Synchronous, and bounded rather than backgrounded. The sweep only enqueues — the indexing is
    already asynchronous — so what the request holds a worker for is one `is_materialized` read
    plus one enqueue per member, capped at `WARM_SWEEP_LIMIT`. A sweep that stops there reports
    `truncated: true`. Backgrounding the route itself would need a job handle this service does
    not have, and would trade a bounded wait for an unfinishable one.

    Re-running continues the sweep only once the enqueued work has been indexed. A member counts
    as materialized when its index job completes rather than when it is enqueued, so an immediate
    re-run examines the same members and reaches no further."""
    await offload_sync(check_access, auth, artifact_id, "update", store_db)
    from mantle.services import workspace_service as _ws
    # A sweep over every latent member — the longest-running WRITE in this router. Bounded
    # at `WARM_SWEEP_LIMIT`, and it reports `truncated` when it stops there.
    out = await offload_sync(_ws.warm_collection, store_db, artifact_id, tenant_id=auth.user_id)
    # Spelled out rather than `**out`. `test_response_envelopes_match_the_handlers` reads this
    # return statically to check the declared model against what is sent, and a spread is opaque
    # to it: a documented envelope nothing can check is worse than an undocumented one.
    return {
        "collection_id": artifact_id,
        "materialized": out["materialized"],
        "examined": out["examined"],
        "failed": out["failed"],
        "truncated": out["truncated"],
        "complete": out["complete"],
    }


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
        "has no bounded size, and the per-child membership enrichment below costs a query each. "
        "`total` is always `null` here. The authorized count is knowable only by authorizing "
        "every member, and each authorization is several queries plus an access-audit write, "
        "so reporting a number would make one page of an N-member collection cost N of both "
        "whatever `limit` said. Page with `has_more` rather than by arithmetic on `total`. "
        "Cost scales with how much of this container you may read rather than with its size: "
        "members are authorized a page at a time until the page is full. `has_more` means "
        "another member you may read exists, rather than merely that more members exist, so a "
        "page is short only when it is genuinely the last one. Both follow from the same rule: "
        "a short page would let you count, by arithmetic, the members you are not permitted to "
        "see. "
        "Order is the container's own child order — the ascending `order_key` on each "
        "membership edge, which is what `PATCH /artifacts/{artifact_id}/children/order` writes. "
        "It is stable across calls while nobody reorders. There is no `sort` parameter because "
        "the order is a property of the container rather than of the query."
    ),

    responses=_errors(401, 404, 500, ok=PageResponse),
)
async def list_children(
    artifact_id: Annotated[str, Path(description=_CONTAINER_PARAM)],
    request: Request,
    content_type: Optional[str] = Query(
        None,
        description="Filter the children by exact content_type (MIME). Omit to list "
                    "every child." + _FILTER_BEFORE_PAGE,
    ),
    workspace_id: Optional[str] = Query(
        None,
        description="Include draft children homed in this workspace. A workspace you cannot read "
                    "is ignored rather than refused: you get the committed children and no error, "
                    "because answering 403 would confirm the workspace exists to someone with no "
                    "access to it. An empty draft list is therefore indistinguishable from no "
                    "access; read the workspace directly when you need that distinction.",
    ),
    limit: int = Query(100, ge=1, le=1000, description="Page size."),
    offset: int = Query(0, ge=0, description="How many children to skip."),
    store_db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """List children of any artifact (universal container model).

    The cost and ordering statements a caller needs live in the decorator's `description=` rather
    than here, because an explicit `description=` overrides the docstring and this route sets
    one: text added here would reach no caller.

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

    # Every member is authorized separately: reading a container is not reading its members.
    #
    # `check_access` above answers for `artifact_id` — the container — and nothing else. But
    # `list_collection_artifacts` returns each member's WHOLE document and decrypts the body on the
    # way out (`lattice_api.py:565-601`, decrypt at:593), and that decryption succeeds for anyone
    # holding the collection, because `content_key_scope` keys a member by its COLLECTION
    # (`doc_boundary.py:111-113`) while the read gate is per artifact. Without this filter the
    # route hands back plaintext for members whose own gate refuses the same caller one call later:
    #
    #   * a member carrying an explicit `deny{read}` naming the caller — never consulted here;
    #   * a member linked in by `_link_source_artifact` with `origin=False, propagate=[]`, so no
    #     grant reaches it at all, which is exactly what that edge shape is FOR.
    #
    # So this is the same per-id authorization `_fetch_authorized_docs` performs for the batch
    # route, and for the same stated reason: silent skip, never a 403, so the endpoint reports
    # "not in this container, for you" rather than becoming an existence oracle over the members.
    #
    # The authorization runs on the page rather than on the whole container. `check_access` is
    # several queries plus an access-audit write per decision, so authorizing every member first
    # costs a 10,000-child container 10,000 of each to return 100 rows. Here the cost is
    # proportional to how densely readable the container is for this caller. The worst case is a
    # caller who may read almost nothing: that walks the members and returns empty.
    #
    # The page is filled, not sliced, and the loop is the security property rather than an
    # optimisation. Authorizing a single slice returns a short page whose shortfall counts the
    # members the caller may not read, and a caller can sum that page by page into an existence
    # oracle over the members. Filling means a short page only ever means a genuinely last one.
    # `tests/test_children_authorize_every_member.py` pins this against the sliced version.
    #
    # Filled to `limit + 1` readable rows rather than `limit`, so `has_more` means "another
    # member you may read exists" rather than "more members exist". Stopping at `limit` leaks the
    # same way: with two members, one readable, and `limit=1` it answers `has_more: true` and then
    # serves an empty page, the shortfall counting the member the caller may not see.
    #
    # What keeps the remaining disclosure bounded is `total: null`. Page length is the only signal
    # and it is bounded by `limit` rather than by the container, so a caller learns at most "some
    # of these 100 were not yours", never "you cannot see 9,900 things".
    members, page, cursor = children, [], offset
    while len(page) <= limit and cursor < len(members):
        chunk = members[cursor:cursor + limit]
        cursor += len(chunk)
        page.extend(await offload_sync(_readable_members, chunk, auth, store_db))
    has_more = len(page) > limit
    children = page[:limit]

    #: `total` is `null` here by design. The authorized count is only knowable by authorizing
    #: every member, which is the cost the per-page authorization above avoids, so reporting a
    #: number would put it straight back. It is passed inline at the `_page` call below so the
    #: value is visible where the promise is made.

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

    return _page(children, total=None, has_more=has_more)


# ---------- PATCH /artifacts/{artifact_id} — Update ----------

@router.patch(
    "/{artifact_id}",
    summary="Update an artifact",
    description=(
        "Partial update. Re-supply `vector` + `space_id` when the content the vector "
        "describes has changed — the vector arm reindexes with the rest of the write."
    ),

    responses=_errors(400, 401, 404, 413, 500,
        ok=ArtifactResponse,
    ),
)
async def update_artifact(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
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
    #: — the same ceiling as create. A PATCH can replace the whole body, so an
    #: unbounded update is the same gap by another verb.
    _refuse_oversize_inline(body.content, "PATCH /artifacts/{artifact_id}")

    context = _strip_immutable_context_fields(doc, _context_as_stored(body.context))
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
        # `state` is passed for the same reason: without it, archiving a top-level artifact
        # answers 200 and does nothing, leaving deletion as the only way to retire a superseded
        # copy — and deleting destroys the record it was retiring.
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
        body = _artifact_body(updated.to_dict())
        #: The whole document. `_artifact_body` guarantees the declared keys are present and drops
        #: nothing else the document carries. Returning it rather than a key literal means this
        #: route carries no statically documented envelope, which is the trade an open document
        #: requires: it cannot be declared as a closed one without becoming one.
        return body

    # `name` and `description` are passed here for the same reason the top-level branch above
    # passes them, and their absence was the same bug one branch over: a member update returned 200
    # and changed neither, while the identical call on a top-level artifact worked. Both are indexed
    # (`title` in the lexical arm IS `name`), so a renamed member also stayed findable only under
    # its old name.
    updated = await offload_sync(
        workspace_service.update_artifact,
        store_db,
        auth.user_id,
        container_id,
        artifact_id,
        name=body.name,
        description=body.description,
        context=context,
        content=body.content,
        state=body.state,
        content_type=body.content_type,
        vector=vector,
    )
    #: Through `_artifact_body`, because a raw `to_dict` omits unset fields and this route
    #: declares `ok=`: without it the response shape would vary with the data while the spec
    #: promises the declared keys are always present.
    return _artifact_body(updated.to_dict())


# ---------- DELETE /artifacts/{artifact_id} ----------

@router.delete("/{artifact_id}", status_code=status.HTTP_200_OK,
    responses=_errors(401, 404, 500, ok=DeleteArtifactResponse),
)
async def delete_artifact(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
    cascade: bool = Query(
        False,
        description=(
            "Destroy members instead of detaching them. Defaults to `false`, which evicts the "
            "members of a collection — they survive, detached. With `true`, a member reachable "
            "only through this collection is destroyed outright, while one also reachable "
            "through another collection is evicted from this one. `rmdir` refuses a non-empty "
            "directory by default; `rm -r` does not."
        ),
    ),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Delete an artifact — destroying or detaching what it contains.

    This route does not archive. `STATE_ARCHIVED` is set in exactly one place,
    `update_workspace`, so archiving an artifact is `PATCH {"state": "archived"}`.

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
        counts = await offload_sync(
            workspace_service.delete_workspace, store_db, auth.user_id, artifact_id,
            cascade=cascade, auth=auth,
        )
        detached, destroyed, refused = _delete_counts(counts)
        return {"id": artifact_id, "deleted": True,
                "detached": detached, "destroyed": destroyed, "refused": refused}

    counts = await offload_sync(
        workspace_service.delete_artifact, store_db, auth.user_id, container_id, artifact_id,
        cascade=cascade, auth=auth,
    )
    detached, destroyed, refused = _delete_counts(counts)
    return {"id": artifact_id, "deleted": True,
            "detached": detached, "destroyed": destroyed, "refused": refused}


@router.delete("/{artifact_id}/children/{child_id}", status_code=status.HTTP_200_OK,
    summary="Remove a member from a container",
    responses=_errors(401, 500, ok=RemoveItemResponse) | {
        # Overridden because this route's 404 also covers "not a member of this container", which
        # is an answer about the membership edge rather than about the artifact.
        404: {"description":
              "Either the container does not exist — or exists and you are not permitted to "
              "evict from it; those two are deliberately indistinguishable, so a stranger "
              "cannot enumerate ids. OR the artifact is simply NOT A MEMBER of this container, "
              "in which case nothing was detached and `detail` says so. A caller must not read "
              "this as \"the artifact is gone\"."},
    },
)
async def remove_child_from_container(
    #: `artifact_id` names the container here, by the surface's convention: the first path
    #: parameter carries one name everywhere, and `GET /artifacts/{artifact_id}/children` uses
    #: it the same way in the same position. The second segment is named for its role.
    artifact_id: Annotated[str, Path(description=_CONTAINER_PARAM)],
    child_id: Annotated[str, Path(description=_MEMBER_PARAM)],
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Detach a member from a container without hard-deleting the root.

    Both ids are in the path, container first, so the request carries no body. HTTP gives a
    `DELETE` body no defined semantics and intermediaries are permitted to drop it, so naming the
    container in the body would put a required field on a carrier that may not arrive.

    Authorizes `evict` on the container rather than `delete` on the member: detaching is a
    container operation and the member survives it. A cascade delete asks the other question,
    because it destroys.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    await offload_sync(check_access, auth, artifact_id, "evict", store_db)

    from mantle.services import workspace_service

    artifact = await offload_sync(
        workspace_service.remove_artifact_from_container,
        store_db,
        auth.user_id,
        artifact_id,
        child_id,
        auth,
    )
    #: The RESPONSE still says `container_id`, because there the name describes the value rather
    #: than a path position, and a caller reading the body should not have to know the URL shape.
    return {"id": artifact.id, "removed": True, "container_id": artifact_id}


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

    # Both 200 shapes are declared. A `Union` rather than one model, because the shape switches on
    # `candidates` and a caller picks between them before sending. Both are known: the ordered
    # branch constructs `ArtifactRecallResponse` itself, and every one of the accessor's three
    # returns is a `{candidates, model_id}` literal, which a test checks rather than assumes.
    responses=_errors(400, 401, 403, 500, 503,
                      ok=Union[ArtifactRecallResponse, RecallCandidatesResponse]),
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

    ``word:value`` is a filter only when ``word`` is one of those fields. Any other word keeps
    its colon and searches as an ordinary term, so ``https://example.com``, ``meeting at 3:30``,
    ``C:\\Users\\example`` and ``ratio 16:9`` are ordinary queries rather than 400s. The cost is
    deliberate and worth stating to a caller: a misspelled field is a search term, not an error
    — ``titel:foo`` searches for that text and finds nothing rather than naming ``titel``. Check
    the field list above when a ``field:value`` query comes back empty.

    A word that IS a field and cannot be honoured is still a 400 naming it, never a silent drop:
    an unsupported operator (``field:~value``, a range on an unordered field), and ``state:`` and
    ``content:`` specifically — ``state`` selects the index segment and is the ``state`` request
    field's job, ``content`` is encrypted at rest. A query of nothing but filters is also a 400 —
    a filter narrows a recall, it does not constitute one.

    Filter tokens do not reach the index. Retrieval sees the query's terms only, so
    ``budget type:pdf`` searches for ``budget`` and filters on the type, rather than
    scoring documents that happen to contain the word "type". ``applied_filters`` on the
    response lists what actually narrowed the result.

    Text narrows, a vector ranks. The query's terms decide which authorized artifacts come
    back — membership, read off the blind-token index — and ``vector`` + ``space_id`` decide
    what order they come back in, by cosine, cut where the ranking's own spectrum stops. They
    are answers to two different questions, so they compose rather than compete, and either
    alone is a complete request.

    No query syntax turns ranking on or off. ``~term`` selects which text is sent for
    embedding, and that is the whole of its effect. Ranking happens when a query vector exists
    and this node has a provisioned AnchorSet — a request fact and a node fact, neither of them
    spellable in the query string.

    A text query with no vector is not an error. It narrows to a real set, and the narrowing
    knows how much of the query each member of it matched — so the set comes back with the
    fullest matches first: ``ordering`` is ``"coverage"`` and each hit's ``score`` is the
    integer count of distinct query stems it carries, ties broken most-recently-updated first.
    A caller that cannot embed — a shell script, a webhook — searches this way and gets an
    answer. A single-term query is exactly recency order, since every hit matched the one term
    there was; the count says so, and does not pretend otherwise.

    That count is not a relevance score. Nothing weights it by term rarity, term frequency or
    field length; it is a count of which of your stems were found. Do not threshold on it and
    do not compare it between queries.

    A node with an ontology bound answers the same request with ``ordering: "reach"``. The
    coverage order is what it re-ranks, so nothing about narrowing or authorization changes; what
    changes is that the hits come back ordered by how far each one reaches toward what the query
    is about, and cut where that reach stops. It is the one text ordering that returns fewer hits
    than matched — ``total`` then counts what survived the cut — so a caller that needs every
    match rather than the answer should read ``ordering`` and page a coverage-ordered node, or
    ask for ``sort: "recency"``, which is honoured before any of this runs. Binding the ontology
    is the host's business and not the query's: no syntax reaches this arm, exactly as none
    reaches the cosine.

    A vector query with no AnchorSet is a 400, and it is the same 400 a foreign ``space_id``
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
        # No existence probe here, and no empty-to-None collapse. Filtering the caller's scope
        # through `_artifact_exists` and falling back to an unscoped search when that filter empties
        # the list carries two separate defects.
        #
        # The oracle. `_artifact_exists` is a raw store read with no authorization, and the
        # `or None` collapse made its answer observable: probe a real id and the contexts filter
        # to empty so recall returns 0 hits; probe an id that does not exist and the scope falls
        # away to None, so the SAME query runs unscoped over the caller's whole light cone and
        # returns its usual hits. Binary, deterministic, timing-free, over the entire id space —
        # against a system whose stated invariant is that denial and nonexistence are
        # indistinguishable.
        #
        # And it widens the request. Naming only containers you cannot read is a request to search
        # nothing, which `or None` turns into a request to search everything you hold.
        #
        # The probe buys no correctness either. `router_accessor` applies scope as
        # an intersection over the authorized contexts (`col in allowed`), so an id that does not
        # exist and an id you may not read both contribute nothing on their own — which is the
        # behaviour the probe was hand-computing, less safely, one layer up.
        scope = list(body.scope)
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
            logger.error("Artifact candidate retrieval error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Recall failed")

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
        logger.error("Artifact recall error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Recall failed")

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
                # Where this hit matched, not a blind prefix and not the document's
                # densest spans regardless of the question — see `beacon.density`.
                # Length is never capped: it is exactly as long as
                # `beacon.cut.top_break` found signal for. Passing the query's stems is
                # what makes two different questions about one artifact return two
                # different excerpts; without them every question got the same bytes.
                content=dense_excerpt(hit.content or "") or None,
                tags=hit.tags or None,
                # The same spans that built `content` above, individually, so a client
                # that wants to render them separately (rather than as one reassembled
                # string) can. `hit.highlights` from the backend stays None — the arms
                # report WHICH artifacts carry the terms, not where; this is the answer
                # to "where", computed once here from the text already hydrated.
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
    size: int = Field(
        ...,
        ge=0,
        le=_MAX_CONTENT_BYTES,
        description="Size of the content to be uploaded, in bytes. The ceiling is the "
                    "cipher's: this node encrypts a body whole and AES-GCM accepts at "
                    "most 2**31 - 1 bytes per message.",
    )
    order_key: Optional[str] = Field(
        None,
        description=(
            "Where this artifact sits among its container's members. A fractional index "
            "rather than a position: an opaque string compared lexicographically, so a new "
            "member can always be placed between two existing ones without renumbering "
            "anything. Omit it — the ordinary case — and the server appends after the "
            "current last key. Sending `1`, `2`, `3` does not extend the way it looks: once "
            "the sequence reaches `10`, that key sorts before `2`."
        ),
    )
    #: Object only, which is the canonical form. `create` and `update` also accept a deprecated
    #: JSON string; this route does not. The string survives on those two routes because 39 call
    #: sites across 31 files in 7 repos still send `json.dumps(...)`, and it retires when they
    #: move.
    context: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Caller metadata about this artifact, as an object. Unlike `POST /artifacts` and "
            "`PATCH /artifacts/{artifact_id}`, this route does not accept a JSON string: the "
            "string form is deprecated, so sending one here is a 422 rather than a quietly "
            "accepted legacy shape."
        ),
    )


class UploadPart(BaseModel):
    """One completed part of a multipart upload.

    The shape is the object store's: the `{PartNumber, ETag}` pair it hands back per part, and
    completion fails without both.

    Extra keys are allowed. Providers return more than these two, and refusing the rest would
    make a caller strip fields it was handed — a validation that creates work rather than
    preventing a mistake."""

    model_config = ConfigDict(extra="allow")

    PartNumber: int = Field(ge=1, description="1-based index of this part.")
    ETag: str = Field(description="The entity tag the store returned for this part.")


class UploadStatusRequest(BaseModel):
    """Update upload progress/completion."""
    status: Optional[str] = Field(
        None,
        description="`uploading` and `failed` record state only; `complete` finalises the "
                    "write, reads the object's true size and content type back from "
                    "storage, and mirrors it to durable storage. Defaults to `uploading`. "
                    "Any other value is refused with 400.",
    )
    progress: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "How far the upload has got, as a fraction between 0 and 1 — `0.5` is half."
            " Bounded at the route. The service clamps with `max(0.0, min(1.0, progress))`, so"
            " a caller working in percent would send `50` and record `1.0`: a"
            " completed-looking upload on its first progress report, with no error. A `422`"
            " naming the range is the answer a clamp cannot give."
        ),
    )
    parts: Optional[List[UploadPart]] = Field(
        None,
        description=(
            "Completed multipart parts, required to finalise a multipart upload. Each part is"
            " the `{PartNumber, ETag}` pair the object store returned when that part was"
            " written. Completing a multipart upload needs the full part list."
        ),
    )
    context_patch: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Top-level keys to merge into the artifact's context. A shallow merge: each key"
            " replaces its whole value, so a nested object is overwritten rather than merged"
            " into. Not part of the upload's own state, which is `status`, `progress` and"
            " `parts`."
        ),
    )


class ReorderRequest(BaseModel):
    """Reorder artifacts in a workspace."""
    ordered_ids: List[str]
    order_version: Optional[int] = None


class BatchFetchRequest(BaseModel):
    """Batch fetch artifacts by IDs."""
    artifact_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=_MAX_PAGE,
        description="The artifacts to fetch. Each id costs two store operations, so the list is "
                    "bounded. Ids you cannot read, and ids that do not exist, are both skipped: "
                    "the response carries only what was returnable, so asking for 50 may yield 48 "
                    "with no indication of which two. The two cases are indistinguishable because "
                    "reporting them separately would make this endpoint an existence oracle for "
                    "artifacts you cannot see. Match the response back by `id`.",
    )


# =============================================================================
# Batch Operations (static path — registered before /{id} sub-paths)
# =============================================================================

# ---------- POST /artifacts/batch ----------

def _readable_members(children: list, auth: AuthContext, store_db: Database) -> list:
    """Keep only the members *auth* may read. One sync pass, handed across whole.

    Module level rather than nested inside the handler on purpose: `check_access` is several
    queries plus an audit write, so it must never run on the event loop, and
    `tests/test_auth_dependencies.py` walks the routers' ASTs to enforce that. A helper defined
    inside the `async def` reads as a call on the loop whether or not it is offloaded — this is
    the plain `def` that guard describes as the correct shape.

    Silent skip, never a 403 — same contract as `_fetch_authorized_docs`: "not in this container,
    for you" and "not in this container" are one answer, so the route cannot be used to test
    whether a member id exists.

    One grant lookup per resource rather than per member. Every member's origin chain passes
    through the same container and the same ancestors, so without a memo the container's grant
    row is fetched once per member for a single page. `grant_memo` caches the per-resource grant
    list for the length of this call.

    The memo holds the lookup and never the verdict: deny-first, the flag test and the walk order
    still run per artifact, so a member carrying its own deny is refused under a container the
    caller may read. It is scoped to this one call, because a memo that outlived the request
    would serve a revoked grant to the next one.
    """
    keep = []
    grant_memo: dict = {}
    for child in children:
        child_id = child.get("id")
        if not child_id:
            continue
        try:
            check_access(auth, child_id, "read", store_db, grant_memo=grant_memo)
        except HTTPException:
            continue
        keep.append(child)
    return keep


def _fetch_authorized_docs(store_db: Database, auth: AuthContext, artifact_ids: List[str]) -> list:
    """Resolve each id and keep only what the caller may read.

    One sync pass so the handler pays a single thread hop for the whole request: the body is two
    store operations per id, and awaiting each separately would spend more time hopping than
    reading. Unlike:func:`_hydrate_batch` this authorizes per id, so the two cannot be shared —
    the batch endpoint takes caller-supplied ids and must not become an existence oracle.
    """
    #: Deduplicated with first-seen order preserved. The response is a page whose `total` the
    #: caller reads, so one document per distinct id is what makes that count artifacts rather
    #: than mentions: the same id three times would otherwise cost six store operations and come
    #: back as three identical documents under `total: 3`. The contract already permits fewer
    #: results than ids, so this needs no further caveat. `max_length` bounds a hostile list and
    #: says nothing about a merely repetitive one.
    results: list = []
    for aid in dict.fromkeys(artifact_ids):
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


@router.post("/batch",
    responses=_errors(401, 404, 500, ok=PageResponse),
)
async def batch_fetch_artifacts(
    body: BatchFetchRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Batch fetch artifacts by IDs across all containers."""
    if not auth.user_id and not auth.bearer_grant:
        raise HTTPException(status_code=401, detail="Missing authorization")

    results = await offload_sync(_fetch_authorized_docs, store_db, auth, body.artifact_ids)
    #: `artifacts` retired for the one shape. A batch fetch is not paged — it answers exactly
    #: the ids asked for — so `has_more` is False by construction rather than by measurement.
    return _page(results, total=len(results), has_more=False)


# =============================================================================
# Upload Endpoints
# =============================================================================

# ---------- POST /artifacts/{artifact_id}/upload-initiate ----------

@router.post("/{artifact_id}/upload-initiate",
    responses=_errors(401, 404, 500, ok=UploadInitiateResponse),
)
async def upload_initiate(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
    body: UploadInitiateRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Initiate an S3 upload for an artifact.

    The artifact_id here is the *container* (workspace) the upload belongs to.
    Delegates to workspace_service.initiate_upload_and_create_artifact.
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
            context=_context_as_stored(body.context),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload initiate failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Upload initiation failed")

    #: Named keys rather than `{**out}`, so the shape is declared here rather than being whatever
    #: the service returned. The literal is also what lets
    #: `test_response_envelopes_match_the_handlers` check the declared model: it reads
    #: `return {...}` literals and refuses an envelope it cannot see.
    return {
        "upload_id": out.get("upload_id"),
        "mode": out.get("mode"),
        "url": out.get("url"),
        "method": out.get("method"),
        "key": out.get("key"),
        "artifact": artifact.to_dict() if artifact is not None else None,
    }


# ---------- PATCH /artifacts/{artifact_id}/upload-status ----------

@router.patch("/{artifact_id}/upload-status",
    responses=_errors(400, 401, 404, 500,
        ok=ArtifactResponse,
    ),
)
async def upload_status(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
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
            # Not `or "uploading"`: an omitted status leaves the status alone.
            status_value=body.status,
            progress=body.progress,
            parts=body.parts,
            context_patch=body.context_patch,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Upload status update failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Upload status update failed")

    #: Through `_artifact_body`, because a raw `to_dict` omits unset fields and this route
    #: declares `ok=`: without it the response shape would vary with the data while the spec
    #: promises the declared keys are always present.
    return _artifact_body(result.to_dict() if hasattr(result, "to_dict") else result)
# There is no presigned multipart-part route on this surface. Presigned parts upload straight to
# the object store, bypassing Mantle, which cannot envelope-encrypt them on that path.
#
# Chunked upload that Mantle can encrypt is the proxied `PUT /artifacts/{id}/content`, reached
# through `upload-initiate`, which returns that URL.

# ---------- GET /artifacts/{artifact_id}/content-url ----------

@router.get("/{artifact_id}/content-url",
    #: An explicit summary, because FastAPI's auto-generated one is the route name and tells a
    #: caller nothing. It leads with the only fact this endpoint carries that a caller could not
    #: construct for itself: whether there are bytes at all.
    summary="Whether this artifact has content, and the path that serves it",
    responses=_errors(401, 404, 500, ok=ContentUrlResponse),
)
async def content_url(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Where this artifact's bytes can be fetched from — a path on THIS node, not a signed URL.

    Nothing here is signed and nothing here is an absolute URL: the value returned is the
    caller's own `GET /artifacts/{id}/content` path.

    Signing is not possible on this design. Object storage holds ciphertext only, so a presigned
    link would hand out bytes no client can read — the decrypt happens on the byte path inside
    this node, which is why the download is proxied rather than redirected.

    The one fact this endpoint carries that a caller could not construct for itself is whether
    content exists at all, reported as `200` versus `404`."""
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
            # A malformed context is a defect in the data rather than an absence of content, so
            # it is logged. Treating it as silently empty makes this route answer as though the
            # artifact has no content, which is a plausible wrong answer.
            #
            # Non-fatal: refusing would take a working route down over a field it does not need.
            logger.warning("artifact %s has a context that is not JSON; treating it as empty",
                           artifact_id, exc_info=True)

    # The same existence test `GET /content` uses: `content_key` or `content_cas_ref`. The value
    # this route returns is the path to that route, so a pointer that refused on a condition its
    # target does not use would be a broken pointer rather than a stricter one — it would answer
    # "no content" for every artifact whose bytes are in the local CAS without a `content_key`.
    content_key = ctx.get("content_key")
    cas_ref = ctx.get("content_cas_ref")
    if not content_key and not cas_ref:
        raise HTTPException(status_code=404, detail="No downloadable content for this artifact")

    # S3 is invisible to callers and holds only ciphertext, so no presigned S3 URL
    # is handed out (a direct S3 fetch would return undecryptable ciphertext).
    # Point callers at Mantle's proxied content endpoint, which decrypts on its byte path.
    return {"url": f"/artifacts/{artifact_id}/content"}


@router.get("/{artifact_id}/content",
    response_class=Response,
    responses=_errors(
        401, 404, 500,
        binary="The artifact's stored bytes, decrypted. The `Content-Type` header carries the "
               "artifact's own media type; `application/octet-stream` is the published floor, "
               "not what every response will say.",
    ) | {
        304: {"description": "Your `If-None-Match` matches the stored `content_sha256`, so the "
                             "bytes are unchanged and none are sent. The `ETag` header is "
                             "returned again."},
        #: This route raises three semantically different 404s, so it overrides the generic
        #: description to name all three. The third is a statement about this node rather than
        #: about the artifact, which is why it is named rather than collapsed into the others.
        #:
        #: The third is a 404 and not a 503: `ContentStoreUnavailable` here means no object store
        #: is configured to look in, which is a standing property of the node rather than a
        #: transient outage, and `503` would invite a retry that cannot succeed.
        #:
        #: Case (1) must keep the words "not permitted" and "indistinguishable".
        #: `test_the_404_description_states_that_denial_and_absence_are_the_same` requires both in
        #: every 404 description on this surface, because a client that reads 404 as "gone" will
        #: delete its local copy of something it merely lost access to.
        404: {"description":
              "One of three, told apart by `detail`. "
              "(1) The artifact does not exist — or exists and you are not permitted to read it. "
              "Those two are "
              "deliberately indistinguishable, so a stranger cannot enumerate ids. "
              "(2) The artifact exists and has no downloadable body; an artifact whose body was "
              "sent as the inline `content` field carries it on `GET /artifacts/{id}` instead. "
              "(3) The artifact has a body but THIS NODE holds no copy and has no object store "
              "configured to look in. That is a topology answer rather than a not-found: the "
              "bytes may exist elsewhere, and `detail` says where to look. Retrying here will "
              "not help."},
    },
)
async def get_artifact_content(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
    request: Request,
    disposition: Literal["attachment", "inline"] = Query(
        "attachment",
        description=(
            "How a browser should treat the body when the artifact carries a filename."
            " `attachment` downloads it; `inline` renders it in place. Ignored when there"
            " is no filename, because `Content-Disposition` is only sent with one."
        ),
    ),
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
            # A malformed context is a defect in the data rather than an absence of content, so
            # it is logged. Treating it as silently empty makes this route answer as though the
            # artifact has no content, which is a plausible wrong answer.
            #
            # Non-fatal: refusing would take a working route down over a field it does not need.
            logger.warning("artifact %s has a context that is not JSON; treating it as empty",
                           artifact_id, exc_info=True)
            ctx = {}
    content_key = ctx.get("content_key")
    # The local address of the same bytes, recorded by the PUT below. Validated, never trusted:
    # `is_cas_ref` is the shape gate and `get_bytes_decrypted` demands an envelope this route's
    # own writes carry, so a context edited to name another object yields an error, not bytes.
    cas_ref = ctx.get("content_cas_ref")
    if not content_key and not cas_ref:
        raise HTTPException(status_code=404, detail="No downloadable content for this artifact")

    # Conditional GET, keyed on the `content_sha256` the PUT side already stores to make an
    # identical re-upload a no-op.
    #
    # Checked before the tier read, so a matching ETag skips the decrypt entirely — the most
    # obviously blocking call in this router. Answering 304 after paying for it would save only
    # the transfer.
    #
    # Checked after authorization, never before: a 304 is an answer about the bytes, so handing
    # one to a caller who may not read them would make `If-None-Match` a content oracle.
    etag = None
    sha = ctx.get("content_sha256")
    if sha:
        etag = '"%s"' % sha
        inm = request.headers.get("if-none-match") or ""
        # A list is legal, and `*` means "any current representation".
        seen = [t.strip() for t in inm.split(",") if t.strip()]
        if "*" in seen or etag in seen or sha in [t.strip('"') for t in seen]:
            from fastapi import Response as _Resp
            return _Resp(status_code=304, headers={"ETag": etag})

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

    # The stored `Content-Type` is the caller's own string, trusted verbatim on upload and echoed
    # back as this response's media type. This is a general artifact store, so an allowlist would
    # refuse legitimate uploads to guard against a rendering decision that is the server's to
    # make. The type is neutralised at serve time instead.
    #
    # `nosniff`, set globally in `main.py`, does not cover this: it stops a browser guessing a
    # type, and the exposure here is a type honestly declared `text/html`. `sandbox` gives the
    # response a unique origin with scripting off, so stored markup cannot run against a viewer's
    # session on this node, while images and PDFs still render. It applies to every response,
    # because an artifact with no filename is sent with no `Content-Disposition` and a browser
    # renders it inline regardless of the `disposition` parameter.
    headers = {"Content-Security-Policy": "sandbox"}
    if etag:
        headers["ETag"] = etag
    fn = ctx.get("filename")
    if fn:
        # `attachment` is the default, because a download is the safe assumption for a store
        # holding arbitrary uploaded bytes. `inline` lets a viewer render an image or a PDF in
        # place without re-hosting the bytes.
        headers["Content-Disposition"] = '%s; filename="%s"' % (disposition, fn)
    return Response(content=data, media_type=ctx.get("content_type") or "application/octet-stream", headers=headers)


def _refuse_oversize(declared, artifact_id: str) -> None:
    """Refuse a body already known to exceed the ceiling, before reading it.

    Reads `Content-Length`, so the verdict costs nothing: without this the refusal arrives as an
    `OverflowError` inside the encrypt, after a declared 4 GB upload has been accepted, buffered
    whole and hashed.

    It catches a declared size only. A chunked upload sends no `Content-Length`, and its body is
    still buffered whole before the second check can look at it; avoiding that needs a streaming
    byte path. What this removes is the encrypt, and the wait, when the caller says up front.
    """
    try:
        n = int(declared)
    except (TypeError, ValueError):
        return
    limit = max_content_bytes()
    if n > limit:
        logger.info("refusing declared %s-byte upload for %s before reading it (limit %s)",
                    n, artifact_id, limit)
        #: The message names which ceiling was hit. An operator ceiling and the cipher bound are
        #: different facts and a caller acts on them differently: the first means "ask this
        #: node's operator", the second means "split the upload".
        raise HTTPException(
            status_code=413,
            detail=("Declared body of %d bytes exceeds this node's configured limit of %d "
                    "(MANTLE_MAX_CONTENT_BYTES). Refused before reading it." % (n, limit))
            if limit < _MAX_CONTENT_BYTES else
            ("Declared body of %d bytes exceeds what this node can encrypt in one envelope "
             "(AES-GCM accepts at most 2**31 - 1). Refused before reading it; split the upload "
             "across artifacts." % n))


@router.put("/{artifact_id}/content",
    # The body is declared here rather than inferred. The handler reads `await request.body`,
    # which FastAPI cannot see, so without this the operation publishes no requestBody and a
    # generated client has no method to upload with. Spelled through `openapi_extra` because
    # there is no model to hang it on: the body is raw bytes, and inventing one would put a
    # parse in front of a byte path that must not have one.
    openapi_extra={
        "requestBody": {
            "required": True,
            "description": (
                "The raw bytes to store — sent as the body, not as a form field or a"
                " JSON string. Mantle encrypts them on the byte path before storing,"
                " and records the request's `Content-Type` as the artifact's content type."
            ),
            "content": {"application/octet-stream":
                        {"schema": {"type": "string", "format": "binary"}}},
        }
    },
    responses=_errors(400, 401, 404, 413, 500, ok=PutContentResponse),
)
async def put_artifact_content(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
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
            # A malformed context is a defect in the data rather than an absence of content, so
            # it is logged. Treating it as silently empty makes this route answer as though the
            # artifact has no content, which is a plausible wrong answer.
            #
            # Non-fatal: refusing would take a working route down over a field it does not need.
            logger.warning("artifact %s has a context that is not JSON; treating it as empty",
                           artifact_id, exc_info=True)
            ctx = {}
    content_key = ctx.get("content_key") or f"artifacts/{artifact_id}.content"
    owner_id = doc.get("created_by")
    collection_id = doc.get("collection_id")
    ctype = request.headers.get("content-type") or ctx.get("content_type") or "application/octet-stream"

    # The cheap refusal first: if the caller declared a length this node cannot encrypt, say so
    # before buffering it.
    _refuse_oversize(request.headers.get("content-length"), artifact_id)
    body = await request.body()
    # And again on what actually arrived — a chunked upload declares nothing, so this is the
    # only check it gets. Still before the hash and the encrypt, both proportional to size.
    if len(body) > max_content_bytes():
        _refuse_oversize(len(body), artifact_id)
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
        # A dedup wrote nothing, so no mirror leg was created and none can be owed.
        return {"stored": True, "size": len(body), "content_key": content_key,
                "content_ref": prior_ref, "deduplicated": True,
                "mirror_pending": False}

    try:
        # Encrypt-then-store, both proportional to the upload's size.
        #
        # `on_mirror_deferred` fires only in the middle case — a mirror this node HAS and could
        # not reach — and records the work still owed. It runs inside the same worker thread as
        # the write, so the store call it makes is a whole synchronous operation on one thread,
        # which is the rule `offload_sync` exists to keep.
        # The deferred-mirror case reaches the caller as `mirror_pending`, not just the server
        # log. The write does not fail, because the bytes are durable on this node, but
        # "durable on one node" and "replicated" are different answers and a caller that cares
        # about off-node durability needs to be told which one it got.
        deferred = {"pending": False}

        def _on_mirror_deferred(cas_ref, exc):
            deferred["pending"] = True
            _record_mirror_pending(store_db, artifact_id, content_key, cas_ref, exc,
                                   owner_id=owner_id)

        ref = await offload_sync(put_bytes_encrypted, content_key, body, ctype, owner_id,
                                 collection_id=collection_id,
                                 on_mirror_deferred=_on_mirror_deferred)
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
        # Both tiers absent. Naming the remedy: a node reaches this only when it has neither a keys
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
    # `content_key` is recorded even when it is derived here rather than read from the context.
    # It is the object store's whole address, so an upload that did not write it back left the
    # download route reading `ctx["content_key"] is None` — a 404 on content that was stored.
    await offload_sync(_record_content_ref, store_db, doc, content_key, ref, digest, ctype)

    return {"stored": True, "size": len(body), "content_key": content_key,
            "content_ref": ref, "deduplicated": False,
            "mirror_pending": deferred["pending"]}


# ── the mirror leg this node still owes ──────────────────────────────────────────────────────
#
# Written to the store's own work pool (`db/schema.py`'s `task` sidecar), in the row shape
# `ember/runtime/pool.py::enqueue` already uses — `operator` + `arguments` + `status` + `task_key`,
# whose `_sync_task` mirror in the `task` table is what makes `status` an indexed, selective
# predicate. One row shape, one set of typed accessors (`pending_window` / `try_claim` / `release`
# / `count_by_status`) — not a second convention that a drain would have to learn separately.
#
# Under its own content type, though, and that distinction is the load-bearing one: the shape is
# what a drain reads, the content type is what a worker selects on. `ember`'s claim predicate is
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
            # No local copy this time (no keys dir, so no local CAS). A stale ref left behind
            # would point the next read at bytes that are not this artifact's content.
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

def _reorder_children(store_db: Database, artifact_id: str, ordered_ids: List[str]) -> int:
    """Resolve version-ids or root-ids → root-ids, rewrite the edge order, return edges updated.

    One unit because it is one operation: the resolution loop only exists to feed the reorder,
    and the reorder is a transaction the resolution must not be spliced into.

    Returns the number of edges actually updated, which is what lets the route refuse a reorder
    it could not apply whole. Two kinds of id apply to nothing and would otherwise be dropped
    silently under a 200:

      * an id resolving to no artifact — the `if a:` below;
      * an id resolving to an artifact that is not a member of this container —
        `set_edge_order_key` returns False when there is no membership edge.
    """
    ordered_roots: List[str] = []
    for aid in ordered_ids:
        a = store.get_artifact(store_db, aid)
        if a:
            ordered_roots.append(a.root_id)
    return store.reorder_collection_artifacts(store_db, artifact_id, ordered_roots)


@router.patch("/{artifact_id}/children/order",
    responses=_errors(401, 404, 409, 500, ok=ReorderResponse),
)
async def reorder_children(
    artifact_id: Annotated[str, Path(description=_CONTAINER_PARAM)],
    body: ReorderRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Reorder children of any artifact (any container).

    Send the complete membership. `ordered_ids` is applied as a sequence rather than as a
    specification of the whole order: `reorder_collection_artifacts` assigns monotonically
    increasing `order_key`s along the ids you send, starting at `after_key(None)` (`"U"`), and a
    member you do not name keeps the key it already has. The listing sorts by `order_key`, so a
    partial list gives the named ids a fresh ascending run and lets the result interleave with
    the rest. That is deterministic, and it is not predictable without reading the key
    arithmetic — a partial list means neither "move these and leave the rest alone" nor "these
    first, the rest after".

    Authorization is container-level: `update` on the container, with no per-child check,
    because reordering touches only the membership edges' sort keys and reads no member content.
    A cascade delete asks the stronger question, because the same grant would destroy members.

    The route does not become an existence oracle: its refusal says ids "do not exist or are not
    members of this container" without saying which."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    await offload_sync(check_access, auth, artifact_id, "update", store_db)

    #: Offloaded: `_artifact_exists` is a synchronous read, and calling it directly from an
    #: `async def` handler would hold the event loop for its duration.
    if not await offload_sync(_artifact_exists, store_db, artifact_id):
        raise HTTPException(status_code=404, detail="Artifact not found")

    # Optimistic concurrency over the child order. `store.order_fingerprint` derives the version
    # from the membership edges, which are already authoritative for ordering — see its docstring
    # for why nothing is stored.
    #
    # Sending `order_version` is optional and omitting it is unconditional. Only a caller that
    # supplies one can be refused, and only when the order it was holding has moved.
    if body.order_version is not None:
        current = await offload_sync(store.order_fingerprint, store_db, artifact_id)
        if body.order_version != current:
            raise HTTPException(
                status_code=409,
                detail="The child order has changed since %s was issued (it is now %s). Re-read "
                       "the order and reapply." % (body.order_version, current))

    applied = await offload_sync(_reorder_children, store_db, artifact_id, body.ordered_ids)
    asked = len(body.ordered_ids)
    if applied != asked:
        # All or nothing. A reorder is one intent, so applying 7 of 10 produces an arrangement
        # nobody asked for — and the response would hand back an `order_version` certifying it.
        #
        # The write has already happened for the ids that did resolve, so this refusal reports
        # the mismatch rather than undoing it: undoing needs the reorder to be transactional,
        # which it is not. A 400 here does not mean nothing changed.
        raise HTTPException(
            status_code=400,
            detail="Reorder applied to %d of %d ids. The rest name artifacts that do not exist or "
                   "are not members of this container; re-read the children and send ids from that "
                   "list." % (applied, asked))


    # The version AFTER the write, so a client chaining reorders can pass it straight back.
    return {"order_version": await offload_sync(store.order_fingerprint, store_db, artifact_id)}


# ---------- POST /artifacts/{artifact_id}/revert — Phase D.1 dedicated route ----------

@router.post("/{artifact_id}/revert",
    # The 204 is declared, so a generated client has a branch for the designed no-op.
    responses=_errors(
        400, 401, 404, 500,
        no_content="Nothing to revert: this artifact has no committed version yet, so there is "
                   "no state to restore. This is the designed no-op, NOT a failure — the draft "
                   "is left exactly as it was.",
        ok=ArtifactResponse,
    ),
)
async def revert_artifact_endpoint(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
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

    # `collection_id` only, with no `_key` fallback: `_key` is an alias for the artifact's own id
    # rather than for its container, so falling back to it would pass the artifact's own id as
    # the workspace, which `revert_artifact` rejects by construction
    # (`target.collection_id != workspace_id`).
    #
    # A top-level artifact has no workspace, and the branch below says that rather than letting
    # an empty `workspace_id` surface as `404 Workspace not found`.
    workspace_id = doc.get("collection_id") or ""
    if not workspace_id:
        raise HTTPException(
            status_code=400,
            detail="This artifact is top-level and has no containing workspace, so there is no "
                   "workspace-scoped draft to revert. Revert applies to an artifact inside a "
                   "container.",
        )

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
    except Exception as exc:
        # Paired with the `except HTTPException: raise` above, which on its own is a no-op
        # wrapper. Without this clause a store-level failure reaches FastAPI's default handler
        # with no log line and no message a caller could act on.
        logger.error("Failed to revert artifact %s: %s", artifact_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to revert artifact")

    if result is None:
        # No committed version exists — revert is a no-op per the design.
        from fastapi import Response
        return Response(status_code=204)
    #: Through `_artifact_body`, because a raw `to_dict` omits unset fields and this route
    #: declares `ok=`: without it the response shape would vary with the data while the spec
    #: promises the declared keys are always present.
    return _artifact_body(result.to_dict())


# =============================================================================
# Container Metadata
# =============================================================================

# ---------- GET /artifacts/{artifact_id}/commits ----------

@router.get("/{artifact_id}/commits",
    # No 400: nothing on this path refuses a request it has authorized.
    responses=_errors(401, 404, 500, ok=PageResponse),
)
async def list_commits(
    artifact_id: Annotated[str, Path(description=_CONTAINER_PARAM)],
    limit: int = Query(100, ge=1, le=1000, description="Page size."),
    offset: int = Query(0, ge=0, description="How many commits to skip, newest first."),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List commits for a collection container."""
    #: Authenticate before touching the store. Ordered after `check_access` instead, an
    #: unauthenticated caller gets a different status for an id that exists than for one that
    #: does not — an existence oracle built out of the difference between two error codes.
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    await offload_sync(check_access, auth, artifact_id, "read", store_db)

    # Any artifact may be asked for its commits, and one that has none gets an empty page. Nothing
    # on this path tests whether the artifact is a collection, and a read returning nothing is a
    # correct answer rather than an error.

    from mantle.services.collection_service import get_commits_for_collection

    try:
        # A commit row plus its item rows for each commit in the collection's history.
        commits = await offload_sync(
            get_commits_for_collection, store_db, auth.user_id, artifact_id
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list commits: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list commits")

    # The response uses the one page shape, so `total` is the full history length and `has_more`
    # is derived from the window returned.
    #
    # Direct attribute access, never `getattr(c,..., default)`. `adds` and `removes` are not
    # fields of `Commit` — they live in the `CommitItem`s that `item_ids` points at, and
    # `collection_service` sets both attributes from the store's own resolution. Reading them
    # with an empty-list default would publish a changeset in which nothing ever changed; a
    # direct read raises `AttributeError` on a rename instead of quietly emptying the response.
    #
    # `collection_id` is the route's own `artifact_id` rather than a lookup, because `Commit`
    # carries no collection.
    rows = [
        {
            "id": c.id,
            "collection_id": artifact_id,
            "message": c.message,
            "author_id": c.author_id,
            "created_time": c.created_time,
            "adds": c.adds,
            "removes": c.removes,
        }
        for c in commits
    ]
    # Paged, like every other list on this surface.
    #
    # This bounds the response, not the work. `get_commits_for_collection` scans every
    # `CommitItem` and every `Commit` doc in the store to decide which touch this container, and
    # paging cannot narrow a filter that is not indexed by container. Narrowing the scan is a
    # store-layer change.
    total = len(rows)
    window = rows[offset:offset + limit]
    return _page(window, total=total, has_more=(offset + len(window)) < total)


@router.get("/{artifact_id}/access-log",
    responses=_errors(401, 404, 500, ok=AccessLogResponse),
)
async def get_artifact_access_log(
    artifact_id: Annotated[str, Path(description=_ARTIFACT_PARAM)],
    limit: int = Query(100, ge=1, le=1000, description="Page size."),
    offset: int = Query(0, ge=0, description="How many log entries to skip."),
    since: Optional[str] = Query(
        None,
        description=(
            "Only events at or after this ISO-8601 timestamp. The log is append-only and grows"
            " with use, so a time bound is how an old event is reached without paging through"
            " everything newer than it."
        ),
    ),
    until: Optional[str] = Query(
        None,
        description="Only events at or before this ISO-8601 timestamp. Inclusive, like `since`.",
    ),
    result: Optional[Literal["allowed", "denied"]] = Query(
        None,
        description=(
            "Filter to one outcome. Omit for both. The filter is applied in the query,"
            " BEFORE the page is taken, so `items` is a page of the filtered set."
        ),
    ),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """The artifact's own access history — who touched it, when, allowed or denied
    (the access-audit "force"). Requires ``admin`` on the artifact; reading it is
    itself witnessed (recursively) via the same check_access gate."""
    await offload_sync(check_access, auth, artifact_id, "admin", store_db)
    from mantle.services import audit_service
    rows = await offload_sync(
        audit_service.get_artifact_access_log,
        store_db, artifact_id, limit=limit + 1, offset=offset, result=result,
        since=since, until=until,
    )
    #: `events`/`count` retired for the one shape. `artifact_id` stays: it names WHAT was listed,
    #: which the envelope does not carry and the caller cannot always reconstruct.
    # One row over the page is fetched so `has_more` is exact rather than guessed from a full
    # page. `total` is `None`, which is what `_page` documents `None` for: the count is genuinely
    # unknown without a second COUNT query over an append-only log, and a reader can tell "not
    # counted" from "none" only while the two have different values.
    has_more = len(rows) > limit
    events = rows[:limit]
    return {"artifact_id": artifact_id,
            **_page(events, total=None, has_more=has_more)}

