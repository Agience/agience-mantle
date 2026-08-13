"""Unified artifact indexing pipeline.

Every artifact — regardless of content type — goes through the same path:

    artifact (any type, any collection)
      → SSE: tokenize title/description/tags/content → encrypted posting lists
      → MANTLE: chunk content text → embed → encrypted IVF cells

Both arms are unconditional once the wiring prerequisites (Oracle, S3,
the lattice) are met. No feature flags. The router converts missing
prerequisites to 503 (no plaintext fallback by design).

The vector arm has one prerequisite the wiring cannot supply: a provisioned **AnchorSet**, the
coordinate system every chunk is routed against. Nothing in Mantle derives one, so a node
nobody has provisioned has none, and on that node :func:`_mantle_index_artifact` returns
``ARM_SKIPPED`` for every write while the lexical arm indexes normally. That is the state of a
fresh install, not an error: the write succeeds, the artifact is still narrowed to by its
terms, a WARNING names the gap, and recall answers ordered by how much of the query each hit
matched (``ordering: "coverage"``, an integer stem count as each score) until an operator
provisions anchors. See
:mod:`mantle.search.anchors.store`.

Vector ingress
--------------
The vector arm's embed step has no model behind it — Mantle never embeds. A write may
carry a vector the writer already computed (``vector=`` here, validated at the router
by ``api/vectors.py``), and it enters through the ordinary provider seam as a
:class:`embeddings.WriterSuppliedEmbeddings`. A vector-bearing write indexes as ONE
chunk: the supplied vector describes the artifact, and distributing it over several
chunks would attribute to each of them a claim the writer made about the whole.

Without a vector nothing changes: the provider resolves against the long-term cache
and returns empty for anything it has never seen, and the arm skips.

See `.dev/features/mantle-mvp.md` and
`.dev/features/mantle-sse-lexical-index.md`.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from mantle.search.embeddings import (
    Embeddings,
    WriterSuppliedEmbeddings,
    model_id as emb_model_id,
)
from mantle.entities.artifact import Artifact

from mantle.search.ingest.chunking import (
    chunk_text,
    extract_text_from_context,
    should_chunk_content,
)
from mantle.search.ingest.tags import (
    normalize_tags,
    parse_tags_from_context,
)
from mantle.search.mantle.oracle import MasterKeyMissing
from mantle.services.acting_principal import KeyCustodyDenied
from mantle.services.bootstrap_types import CREDENTIAL_CONTENT_TYPE
from mantle.services.ingest_runner_service import extract_text_from_artifact
from mantle.services.issuers import ISSUER_CONTENT_TYPE

logger = logging.getLogger(__name__)

_embeddings = Embeddings()


# Each artifact state indexes into its own physically separate index segment
# (separate S3 prefixes — see search.mantle.wiring._segment_prefixes). The
# segment name equals the state name. States are mutually exclusive, so an
# artifact's entry lives in exactly one segment; a transition (draft→committed,
# →archived, unarchive) moves it (index into the new segment, purge the others).
_SEGMENTS = ("committed", "draft", "archived")


def _segment_for_state(state: str) -> str:
    """Map an artifact state to its index segment (1:1).

    An absent or unrecognised state resolves to `Artifact.STATE_WHEN_ABSENT` — the single
    definition of what a doc with no `state` is in (`db/constants`), which every other reader
    derives from too. A second answer here would file a stateless doc into one tree and the entity
    layer would move it to another on the next write."""
    return state if state in _SEGMENTS else Artifact.STATE_WHEN_ABSENT


#: Content types that are never indexed. Two criteria admit an entry, and an artifact
#: qualifies under one of them or it is indexed like everything else.
#:
#: **1 — platform trust configuration, not user content.** All three must hold:
#:
#:   a. The record is owned by the system principal and read back by a
#:      ``created_by`` + content-type query, not through the grant ledger.
#:   b. It is deliberately grantless. There is no principal who should be able to
#:      find it by searching, because a grant that made it indexable would also make
#:      the platform's own trust config a search result for whoever held that grant.
#:   c. It carries no collection. ``create_issuer_artifact`` sets ``collection_id=""``.
#:
#: **2 — the artifact's content IS a secret.** A credential's value is its content: an
#: API key, an OAuth client secret, a refresh token. Indexing it tokenizes that value
#: into the SSE posting lists, and a recall hit hydrates the decrypted value into
#: ``SearchHit.content`` — so the secret becomes reachable by SEARCH as well as by
#: direct read.
#:
#: That is not a leak. Cells stay encrypted at rest, and every posting and every
#: hydration is cut by the same light cone that guards the read path, so no principal
#: learns anything it could not already fetch by id. It is a WIDER SURFACE, chosen
#: rather than inherited: recall answers a question nobody needs answered about a
#: secret ("which of these contains this string?"), and a full-text index is a second
#: representation of the plaintext, in a second store, with a second decrypt path and
#: a second set of ways to be wrong. A secret is not something you full-text search,
#: so the index does not carry one. Reaching a credential stays a matter of naming it
#: — through the artifact it hangs off, which is indexed and searchable as usual.
#:
#: This is a different exclusion from the storage-plane one (``lattice_api._SIDE_PLANE_CTS``):
#: everything named here is still an artifact — governable, audited, versioned — and stays
#: fully visible to the artifact API, content included. It is excluded only from search.
NON_INDEXABLE_CONTENT_TYPES = frozenset({
    ISSUER_CONTENT_TYPE,                       # criterion 1
    CREDENTIAL_CONTENT_TYPE,                   # criterion 2
})


def is_indexable(artifact: Artifact) -> bool:
    """False for trust config and for secret material — see
    :data:`NON_INDEXABLE_CONTENT_TYPES`."""
    return getattr(artifact, "content_type", None) not in NON_INDEXABLE_CONTENT_TYPES


# ---- Per-arm outcome -------------------------------------------------------
#
#
# SKIPPED is a distinct third answer: "this arm had nothing to do here" (no content,
# prerequisites absent, no AnchorSet provisioned) is not a failure, and counting it
# as one drains the failure count of meaning.
ARM_WRITTEN = "written"
ARM_SKIPPED = "skipped"
ARM_FAILED = "failed"


class IndexOutcome:
    """What each arm did for one artifact.

    Truthy iff no arm failed, so ``if index_artifact(...)`` callers keep their
    meaning; the per-arm detail is available to callers that report counts.
    """

    __slots__ = ("sse", "vector", "reason")

    def __init__(self, *, sse: str, vector: str, reason: str = "") -> None:
        self.sse = sse
        self.vector = vector
        self.reason = reason

    @property
    def failed(self) -> bool:
        return ARM_FAILED in (self.sse, self.vector)

    @property
    def wrote_nothing(self) -> bool:
        return ARM_WRITTEN not in (self.sse, self.vector)

    def __bool__(self) -> bool:
        return not self.failed

    def __repr__(self) -> str:
        detail = f" ({self.reason})" if self.reason else ""
        return f"<IndexOutcome sse={self.sse} vector={self.vector}{detail}>"


# Optional async queue
try:
    from mantle.search.ingest import index_queue
except Exception as exc:  # pragma: no cover — queue optional during static analysis
    logger.error("Failed to import index_queue: %s", exc, exc_info=True)
    index_queue = None  # type: ignore[assignment]


# ============================================================
#  Field extraction (shared by SSE + MANTLE)
# ============================================================


def _extract_artifact_fields(artifact: Artifact) -> dict[str, str]:
    """Build the long-form per-field text dict the SSE indexer wants.

    Returns ``{"title": ..., "description": ..., "tags": ..., "content": ...}``,
    omitting empty fields so the indexer skips them. ``content`` here is
    the full analyzable text (artifact.content + extracted text fields)
    — the same corpus the MANTLE chunker walks for embedding.
    """
    text_fields = extract_text_from_context(artifact.context)

    title = (
        text_fields.get("title", "").strip()
        or (getattr(artifact, "name", "") or "").strip()
    )
    description = (
        text_fields.get("description", "").strip()
        or (getattr(artifact, "description", "") or "").strip()
    )

    raw_tags = parse_tags_from_context(artifact.context)
    tags_canonical = normalize_tags(raw_tags)
    tags_text = " ".join(t for t in tags_canonical if t)

    content_text = (extract_text_from_artifact(artifact) or "").strip()
    if not content_text:
        content_text = (artifact.content or "").strip()

    fields: dict[str, str] = {}
    if title:
        fields["title"] = title
    if description:
        fields["description"] = description
    if tags_text:
        fields["tags"] = tags_text
    if content_text:
        fields["content"] = content_text
    return fields


def _build_chunk_id(root_id: str, chunk_id: int) -> str:
    return f"{root_id}:chunk:{chunk_id}"


def _content_chunks(content: str) -> list[dict]:
    """Chunk content for the MANTLE arm.

    Single source of truth so the bulk-reindex prewarm and the per-artifact
    index produce identical chunk texts — identical cache keys, so the warm
    pass actually hits the cache the prewarm populated.
    """
    if not content:
        return []
    if should_chunk_content(content):
        return list(chunk_text(content))
    return [{"chunk_id": 0, "text": content}]


# ============================================================
#  MANTLE vector hook — encrypted IVF, chunks + embeddings
# ============================================================


def _mantle_index_artifact(
    artifact: Artifact,
    collection_id: str,
    fields: dict[str, str],
    *,
    segment: str = "committed",
    vector=None,
) -> str:
    """Chunk + embed the artifact's content, write to MANTLE cells.

    ``vector`` is a writer-supplied :class:`api.vectors.SuppliedVector` when the write
    carried one. It selects both the provider (the vector itself, through
    :class:`embeddings.WriterSuppliedEmbeddings`) and the chunking (one chunk, since the
    vector describes the artifact rather than any part of it).

    Returns :data:`ARM_WRITTEN` / :data:`ARM_SKIPPED` / :data:`ARM_FAILED`; see
    :func:`_sse_index_artifact` for why the status is a return value and not a log line.

    On a node with no provisioned AnchorSet this returns :data:`ARM_SKIPPED` for every
    artifact — the write still succeeds and still indexes lexically. That is the fresh-install
    state and it does not clear itself; see the AnchorSet check below and
    :mod:`mantle.search.anchors.store`.
    """
    content = fields.get("content", "")
    if not content:
        return ARM_SKIPPED

    if not collection_id:
        return ARM_SKIPPED

    artifact_root = artifact.root_id or artifact.id

    if vector is not None:
        # One vector, one chunk. The writer embedded this artifact, not a window of it,
        # so the record's text is the artifact's text and there is nothing to split.
        embedded_chunks = [{"chunk_id": 0, "text": content}]
        embedder = Embeddings(WriterSuppliedEmbeddings(vector.values, vector.space_id))
        supplied_space_id = vector.space_id
    else:
        # Chunk + embed (shared chunker — see _content_chunks). On the bulk-reindex
        # path these exact texts were already embedded in batch, so this call is a
        # cache hit (no per-artifact round-trip).
        embedded_chunks = [c for c in _content_chunks(content) if c.get("text")]
        embedder = _embeddings
        supplied_space_id = None

    texts = [c["text"] for c in embedded_chunks]
    if not texts:
        return ARM_SKIPPED
    try:
        embeddings = embedder(texts)
    except Exception:
        logger.warning(
            "MANTLE: embedding failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return ARM_FAILED
    if not any(embeddings):
        return ARM_SKIPPED

    # The AnchorSet is the one coordinate system (canonical plan §3). It is seeded by a client,
    # never derived here (deriving it locally mints region ids no peer computes), so its absence
    # is a deployment state, not an error: skip the vector arm and let the commit succeed.
    try:
        from mantle.search.anchors.store import (
            AnchorSetNotProvisioned,
            require_live_anchorset,
        )
        anchorset = require_live_anchorset()
    except AnchorSetNotProvisioned:
        # One line, no traceback: the exception's message is self-contained, and this
        # state is a provisioning gap rather than a fault in this call path.
        logger.warning(
            "MANTLE: no AnchorSet provisioned; skipping vector index for %s",
            artifact.id,
        )
        return ARM_SKIPPED
    except Exception:
        logger.warning(
            "MANTLE: AnchorSet unavailable; skipping vector index for %s",
            artifact.id, exc_info=True,
        )
        return ARM_SKIPPED

    # Provenance is the space the vector actually came from. A writer-supplied vector
    # lives in the space the writer named, so its own name is recorded rather than the
    # AnchorSet's — labelling it with the local space is what would make two
    # incomparable vectors look comparable later.
    chunk_model_id = supplied_space_id or anchorset.model_id or emb_model_id()

    mantle_chunks = []
    for i, emb in enumerate(embeddings):
        if not emb:
            continue
        record = {
            "artifact_id": artifact_root,
            "chunk_id": int(embedded_chunks[i].get("chunk_id", i)),
            "embedding": emb,
            "text": embedded_chunks[i].get("text", ""),
            "model_id": chunk_model_id,
        }
        if supplied_space_id:
            # The space a writer named for their own vector, kept as they gave it. It is the
            # record of which coordinate system these numbers are statements in, which is the
            # one thing that cannot be recovered from the numbers.
            record["space_id"] = supplied_space_id
        mantle_chunks.append(record)
    if not mantle_chunks:
        return ARM_SKIPPED

    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        from mantle.search.mantle.wiring import build_indexer
    except Exception:
        logger.debug("MANTLE wiring unavailable; skipping vector index", exc_info=True)
        return ARM_SKIPPED

    try:
        store_db = next(get_store_db())
    except Exception:
        logger.debug("MANTLE: lattice handle unavailable; skipping", exc_info=True)
        return ARM_SKIPPED

    indexer = build_indexer(store_db, segment=segment)
    if indexer is None:
        logger.debug("MANTLE indexer prerequisites missing; skipping")
        return ARM_SKIPPED

    # The cell-key principal is the collection's immutable origin root (not
    # created_by / ownership) — index and query resolve it identically, so the
    # same key is derived at both ends. See search.mantle.principal.
    principal_id = resolve_cell_principal(store_db, collection_id)
    if not principal_id:
        return ARM_SKIPPED

    try:
        touched = indexer.index_artifact(
            principal_id, collection_id, mantle_chunks,
            _ingest_key_request(principal_id),
        )
        logger.info(
            "MANTLE indexed artifact %s (principal=%s collection=%s, %d cells)",
            artifact.id, principal_id, collection_id, touched,
        )
        return ARM_WRITTEN
    except Exception:
        logger.warning(
            "MANTLE indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return ARM_FAILED



def _ingest_key_request(principal_id: str):
    """The key request for a write into ``principal_id``'s cells, as the acting caller.

    The acting principal reaches this layer (:mod:`services.acting_principal`), so the
    question the light cone answers is the real one: *may this caller write into this
    context?* — ``GRANT`` with ``action="update"``. The grant ledger checks it, in the
    same place and by the same code path as the query arm.

    System-initiated indexing — ``collection_service`` auto-index on create, the
    ``init_search`` bulk reindex, the seed path — has no request context and must
    therefore declare its identity explicitly with
    :func:`services.acting_principal.system_acting_context`. It runs as the platform
    system principal and is checked like any other principal; it is not exempt.
    Anything that forgets raises ``NoActingPrincipal`` rather than quietly indexing
    under an unchecked identity.
    """
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest

    from mantle.services.acting_principal import require_acting_principal

    actor = require_acting_principal()
    return KeyRequest(requester_id=actor.principal_id, purpose=KeyPurpose.GRANT,
                      requester_type=actor.principal_type, action="update")


def _read_key_request(principal_id: str):
    """The key request for reading a principal's cells back (not writing them).

    Same identity rule as :func:`_ingest_key_request`, but ``action="read"`` — so a
    caller who may read but not write is not refused, and a read never mints a
    master key that would decrypt nothing.
    """
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest

    from mantle.services.acting_principal import require_acting_principal

    actor = require_acting_principal()
    return KeyRequest(requester_id=actor.principal_id, purpose=KeyPurpose.GRANT,
                      requester_type=actor.principal_type, action="read")


def _mantle_remove_artifact(
    principal_id: str, collection_id: str, artifact_id: str,
    *, segment: str = "committed",
) -> None:
    """Strip an artifact's chunks from MANTLE cells in one index segment."""
    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.wiring import build_indexer
        store_db = next(get_store_db())
        indexer = build_indexer(store_db, segment=segment)
        if indexer is None:
            return
        indexer.remove_artifact(
            principal_id, collection_id, artifact_id,
            _ingest_key_request(principal_id),
        )
    except Exception:
        logger.warning(
            "MANTLE remove failed for artifact %s (owner=%s, collection=%s)",
            artifact_id, principal_id, collection_id, exc_info=True,
        )


# ============================================================
#  MANTLE-SSE hook — encrypted lexical, posting lists
# ============================================================


def _sse_index_artifact(
    artifact: Artifact,
    collection_id: str,
    fields: dict[str, str],
    *,
    segment: str = "committed",
) -> str:
    """Write per-field text into the SSE blind-token posting lists.

    Returns one of :data:`ARM_WRITTEN` / :data:`ARM_SKIPPED` / :data:`ARM_FAILED`,
    which the caller reports. Exceptions stay swallowed here so the other arm and
    the commit survive; the return value is what carries the result out.
    """
    if not fields:
        return ARM_SKIPPED

    if not collection_id:
        return ARM_SKIPPED
    artifact_id = artifact.root_id or artifact.id

    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        from mantle.search.mantle.wiring import build_sse_indexer
    except Exception:
        logger.debug("SSE wiring unavailable; skipping lexical index", exc_info=True)
        return ARM_SKIPPED

    try:
        store_db = next(get_store_db())
    except Exception:
        logger.debug("SSE: lattice handle unavailable; skipping", exc_info=True)
        return ARM_SKIPPED

    indexer = build_sse_indexer(store_db, segment=segment)
    if indexer is None:
        logger.debug("SSE indexer prerequisites missing; skipping")
        return ARM_SKIPPED

    # Same principal as the vector arm: the collection's origin root.
    principal_id = resolve_cell_principal(store_db, collection_id)
    if not principal_id:
        return ARM_SKIPPED

    try:
        n = indexer.index_artifact(
            principal_id, collection_id, artifact_id, fields,
            _ingest_key_request(principal_id),
        )
        logger.info(
            "SSE indexed artifact %s (principal=%s collection=%s, %d tokens)",
            artifact.id, principal_id, collection_id, n,
        )
        return ARM_WRITTEN
    except Exception:
        logger.warning(
            "SSE indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return ARM_FAILED


def _sse_remove_artifact(
    principal_id: str, artifact_id: str, *, segment: str = "committed",
) -> None:
    """Strip an artifact's references from the SSE index in one segment."""
    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.wiring import build_sse_indexer
        store_db = next(get_store_db())
        indexer = build_sse_indexer(store_db, segment=segment)
        if indexer is None:
            return
        indexer.remove_artifact(
            principal_id, artifact_id, _ingest_key_request(principal_id),
        )
    except Exception:
        logger.warning(
            "SSE remove failed for artifact %s (owner=%s)",
            artifact_id, principal_id, exc_info=True,
        )


# ============================================================
#  Public API: index / batch / delete
# ============================================================


def move_artifact_segments(
    artifact: Artifact, collection_id: str, *, remove_from: list[str],
) -> None:
    """Remove the artifact's root from the given index segments.

    Called at a state transition to vacate the segment(s) the artifact is leaving
    (the new segment is (re)indexed separately). The index is root-keyed, and a
    root can legitimately occupy two segments at once — a committed version and a
    WIP draft of the same root coexist — so we never blanket-purge "all others";
    the caller names exactly the segment(s) being left:

        commit    (draft→committed):  remove_from=["draft"]
        archive   (→archived):        remove_from=["committed", "draft"]
        unarchive (archived→draft):   remove_from=["archived"]

    Best-effort and gated on the index backend being available, so it's a fast
    no-op where search isn't wired (no principal query is run).
    """
    if not collection_id or not remove_from:
        return
    try:
        from mantle.search.mantle.wiring import build_indexer
        # Availability gate first — build_indexer doesn't use the db handle, and
        # resolving the handle is the expensive part. This makes the whole thing
        # a fast no-op where search isn't wired (e.g. tests with no S3/oracle),
        # before any DB connection is attempted.
        if build_indexer(None) is None:
            return
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        store_db = next(get_store_db())
        principal_id = resolve_cell_principal(store_db, collection_id)
    except Exception:
        logger.warning(
            "segment move: cell principal unresolved for collection %s — the "
            "artifact was NOT vacated from segments %s and will keep matching "
            "searches of a state it has left",
            collection_id, remove_from, exc_info=True,
        )
        return
    if not principal_id:
        return
    root = artifact.root_id or artifact.id
    for seg in remove_from:
        if seg not in _SEGMENTS:
            continue
        _mantle_remove_artifact(principal_id, collection_id, root, segment=seg)
        _sse_remove_artifact(principal_id, root, segment=seg)


def get_artifact_embeddings(artifact: Artifact, collection_id: str) -> list[dict]:
    """Return the artifact's stored MANTLE vector chunk records for its current state's
    segment: each ``{chunk_id, embedding, model_id, ...}``, ordered by chunk_id.

    Reads the vectors back out of the encrypted cells — the inverse of indexing.
    Empty if the vector arm isn't wired (no cell store) or the artifact has
    nothing stored (e.g. a container, or a lexical-only deploy with no embeddings
    provider configured; see search/embeddings.py)."""
    if not collection_id:
        return []
    segment = _segment_for_state(artifact.state)
    try:
        from mantle.services.dependencies import get_store_db
        from mantle.search.mantle.principal import resolve_cell_principal
        from mantle.search.mantle.wiring import build_indexer
        store_db = next(get_store_db())
        indexer = build_indexer(store_db, segment=segment)
        if indexer is None:
            return []
        principal_id = resolve_cell_principal(store_db, collection_id)
        if not principal_id:
            return []
        root = artifact.root_id or artifact.id
        chunks = [
            c for c in indexer.collection_chunks(
                # read, not update: this reads vectors back out.
                principal_id, collection_id, _read_key_request(principal_id),
            )
            if c.get("artifact_id") == root
        ]
        chunks.sort(key=lambda c: c.get("chunk_id", 0))
        return chunks
    except MasterKeyMissing:
        # Never indexed, so there is nothing to read back. Genuinely empty.
        return []
    except KeyCustodyDenied:
        raise
    except Exception:
        logger.debug(
            "get_artifact_embeddings failed for %s", getattr(artifact, "id", "?"),
            exc_info=True,
        )
        return []


def index_artifact(
    artifact: Artifact,
    collection_id: str,
    *,
    is_head: bool = True,
    fields: Optional[dict[str, str]] = None,
    vector=None,
) -> IndexOutcome:
    """Index one artifact into the index segment for its current state.

    ``draft`` / ``committed`` / ``archived`` each have a separate physical index
    (separate S3 prefixes per arm), so the artifact is written into the segment
    matching its state. This does not touch the other segments — a root may hold
    a committed version *and* a WIP draft simultaneously; vacating a segment on a
    state transition is the caller's job via :func:`move_artifact_segments`.

    ``is_head`` is preserved for caller compatibility and does not drive index
    branching (versioning is artifact-level). ``fields`` may be supplied by the
    caller — the bulk reindex extracts them once and prewarms the embeddings
    cache, then passes them here so this path neither re-extracts nor makes a
    per-artifact embed round-trip. When ``None`` they are extracted here.

    ``vector`` is a writer-supplied :class:`api.vectors.SuppliedVector` when the write
    carried one; it feeds the vector arm and is ignored by the lexical arm, which reads
    text. Nothing else in the pipeline changes shape when it is ``None``.

    Returns an :class:`IndexOutcome` naming what each arm did. It is truthy iff no
    arm failed, so ``if index_artifact(...)`` keeps its meaning, and "both arms were
    refused" is distinguishable from "both arms wrote".
    """
    segment = _segment_for_state(artifact.state)
    try:
        if not is_indexable(artifact):
            logger.debug(
                "Artifact %s is platform trust config (%s); not a search target",
                artifact.id, getattr(artifact, "content_type", None),
            )
            return IndexOutcome(
                sse=ARM_SKIPPED, vector=ARM_SKIPPED, reason="non-indexable content type",
            )
        if fields is None:
            fields = _extract_artifact_fields(artifact)
        if not fields:
            logger.debug(
                "Artifact %s has no analyzable fields; skipping", artifact.id,
            )
            return IndexOutcome(
                sse=ARM_SKIPPED, vector=ARM_SKIPPED, reason="no analyzable fields",
            )
        outcome = IndexOutcome(
            sse=_sse_index_artifact(artifact, collection_id, fields, segment=segment),
            vector=_mantle_index_artifact(
                artifact, collection_id, fields, segment=segment, vector=vector,
            ),
        )
        # Name the per-arm result rather than logging a single unconditional success line,
        # so an artifact whose every arm was refused doesn't read as indexed.
        logger.log(
            logging.WARNING if outcome.failed else logging.INFO,
            "Indexed artifact %s in collection %s (segment=%s): sse=%s vector=%s",
            artifact.id, collection_id, segment, outcome.sse, outcome.vector,
        )
        return outcome
    except Exception:
        logger.error(
            "Indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return IndexOutcome(sse=ARM_FAILED, vector=ARM_FAILED, reason="pipeline error")


# ============================================================
#  Bulk reindex prep: extract fields once + batch-warm embeddings
# ============================================================

# Texts per Embeddings() call when prewarming the bulk reindex. A batch resolves from
# the long-term cache only — texts already embedded resolve, uncached texts come back
# empty — so this bounds per-call list size.
EMBED_BATCH_SIZE = 64


def prepare_reindex_items(
    items: list[tuple[str, Artifact]],
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[tuple[str, Artifact, dict[str, str]]]:
    """Prepare a bulk-reindex work list with the embeddings cache prewarmed.

    For each ``(collection_id, artifact)``: extract its analyzable fields once
    (skipping archived / field-less artifacts) and collect every unique MANTLE
    chunk text across all artifacts. Then embed those texts in batched HTTP
    calls, which populates the long-term embeddings cache.

    The returned ``(collection_id, artifact, fields)`` tuples feed
    :func:`index_artifact` (pass ``fields=...``); its per-artifact embed then
    hits the warm cache instead of making one round-trip per artifact.

    Built for a cold cache — the batched prewarm is the fast path, not an
    optimization that assumes prior warmth. Identical texts (boilerplate shared
    across artifacts) are embedded once.
    """
    prepared: list[tuple[str, Artifact, dict[str, str]]] = []
    all_texts: list[str] = []
    seen: set[str] = set()

    for collection_id, artifact in items:
        # All states are indexed (into their own segment); the prewarm cache
        # is keyed by chunk text, so it's segment-agnostic.
        try:
            fields = _extract_artifact_fields(artifact)
        except Exception:
            logger.warning(
                "reindex prep: field extraction failed for %s",
                artifact.id, exc_info=True,
            )
            continue
        if not fields:
            continue
        prepared.append((collection_id, artifact, fields))
        for chunk in _content_chunks(fields.get("content", "")):
            text = chunk.get("text")
            if text and text not in seen:
                seen.add(text)
                all_texts.append(text)

    if all_texts:
        batches = (len(all_texts) + batch_size - 1) // batch_size
        logger.info(
            "Reindex prewarm: embedding %d unique chunk texts in %d batch(es) of <=%d",
            len(all_texts), batches, batch_size,
        )
        start = time.time()
        embedded = 0
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i:i + batch_size]
            try:
                _embeddings(batch)  # populates the long-term cache
                embedded += len(batch)
            except Exception:
                logger.warning(
                    "Reindex prewarm: batch embed failed at offset %d", i, exc_info=True,
                )
        logger.info(
            "Reindex prewarm complete: %d/%d texts embedded in %.2fs",
            embedded, len(all_texts), time.time() - start,
        )

    return prepared


def index_artifacts_batch(
    artifacts: list[Artifact],
    collection_id: str,
    *,
    is_head: bool = True,
) -> bool:
    """Bulk-index a list of artifacts.

    Each artifact runs its own SSE + MANTLE flow. Embedding batching
    happens inside :func:`_mantle_index_artifact` — the MANTLE indexer
    handles per-artifact embedding without cross-artifact batching.
    For very large bulk reindex jobs, the admin command runs many of
    these in parallel via the index queue.
    """
    if not artifacts:
        return True

    start_time = time.time()
    logger.info("Starting bulk index of %d artifacts", len(artifacts))
    indexed = 0
    skipped = 0
    failed = 0

    for artifact in artifacts:
        # index_artifact routes each artifact to its state's segment (no skips).
        outcome = index_artifact(artifact, collection_id, is_head=is_head)
        if outcome.failed:
            failed += 1
        elif outcome.wrote_nothing:
            skipped += 1
        else:
            indexed += 1

    total_time = time.time() - start_time
    logger.log(
        logging.WARNING if failed else logging.INFO,
        "Bulk index: %d written, %d skipped, %d FAILED of %d in %.3fs",
        indexed, skipped, failed, len(artifacts), total_time,
    )
    return failed == 0


def delete_artifact_from_index(
    version_id: str,
    root_id: Optional[str] = None,
    *,
    principal_id: Optional[str] = None,
    collection_id: Optional[str] = None,
) -> bool:
    """Remove an artifact from MANTLE vector + MANTLE-SSE lexical indexes.

    ``principal_id`` is required for both arms. ``collection_id`` is required
    for the MANTLE vector arm (cells are scoped per collection); the SSE
    arm scans the artifact's manifest and removes from every posting
    list it appears in regardless of collection.

    Callers without ``principal_id`` get a no-op — there's nothing to
    remove without identity.
    """
    try:
        root = root_id or version_id
        # Resolve identity here rather than at seven call sites — the same derivation the write
        # path uses (`resolve_cell_principal`, above), so index and de-index agree on the key by
        # construction instead of by convention. Callers need only pass `collection_id`.
        # The artifact row is already deleted by the time this runs (workspace_service purges
        # the lattice first), so the principal cannot be recovered from the artifact — it has to come
        # from the container, which still exists. That is why `collection_id` is the argument
        # callers supply.
        if collection_id and not principal_id:
            try:
                from mantle.services.dependencies import get_store_db
                from mantle.search.mantle.principal import resolve_cell_principal
                # `next(...)` is not optional: `get_store_db` is a generator function, so a bare
                # call yields a generator object, not a Database — `get_origin_root` would then
                # raise, and the caller must not swallow that into a wrong-but-plausible fallback.
                # Matches every other use of `get_store_db()` in this file.
                principal_id = resolve_cell_principal(next(get_store_db()), collection_id)
            except Exception:
                logger.warning(
                    "delete_artifact_from_index(%s): could not resolve the cell principal for "
                    "collection %s — nothing will be removed from the index",
                    version_id, collection_id, exc_info=True,
                )
        # Hard delete: the artifact is gone for good, so purge it from every
        # segment (we don't track which state it was last indexed under).
        for seg in _SEGMENTS:
            if principal_id and collection_id:
                _mantle_remove_artifact(principal_id, collection_id, root, segment=seg)
            if principal_id:
                _sse_remove_artifact(principal_id, root, segment=seg)
        if not (principal_id and collection_id):
            logger.warning(
                "delete_artifact_from_index(%s): no principal_id/collection_id supplied, so NOTHING "
                "was removed from the search index — the artifact's chunks and postings (including "
                "their plaintext text) remain searchable. Caller must pass identity.", version_id,
            )
            return False
        logger.info("Deleted artifact %s from search", version_id)
        return True
    except Exception:
        logger.error(
            "Failed to delete artifact %s from search",
            version_id, exc_info=True,
        )
        return False


# ============================================================
#  Enqueue helpers
# ============================================================


def enqueue_index_artifact(
    artifact: Artifact,
    collection_id: str,
    *,
    is_head: bool = True,
    tenant_id: Optional[str] = None,
    vacate: Optional[list[str]] = None,
    vector=None,
) -> None:
    """Enqueue an artifact for async indexing; falls back to sync.

    ``vacate`` names index segment(s) the artifact is leaving on a state
    transition — they're removed in the same job, right after (re)indexing into
    the new state's segment. Folding the move into this one job keeps the
    transition atomic from the caller's view and gives a single mock point.

    ``vector`` rides the job. It is captured by the closure rather than re-read later,
    because the write that carried it has already returned by the time the job runs.
    """
    def _act() -> bool:
        outcome = index_artifact(artifact, collection_id, is_head=is_head, vector=vector)
        if vacate:
            move_artifact_segments(artifact, collection_id, remove_from=vacate)
        # bool(outcome) is False iff an arm failed — a skip is not a job failure.
        return bool(outcome)

    desc = f"index artifact {artifact.id} -> {collection_id}"
    if index_queue:
        try:
            index_queue.enqueue(_act, description=desc, tenant_id=tenant_id)
            return
        except RuntimeError:
            pass
    logger.debug("Index queue unavailable, indexing synchronously: %s", desc)
    _act()


def enqueue_index_artifacts_batch(
    artifacts: list[Artifact],
    collection_id: str,
    *,
    is_head: bool = True,
    tenant_id: Optional[str] = None,
) -> None:
    """Enqueue a batch for async bulk indexing; falls back to sync."""
    def _act() -> bool:
        return index_artifacts_batch(artifacts, collection_id, is_head=is_head)

    desc = f"batch index {len(artifacts)} artifacts -> {collection_id}"
    if index_queue:
        try:
            index_queue.enqueue(_act, description=desc, tenant_id=tenant_id)
            return
        except RuntimeError:
            pass
    logger.debug("Index queue unavailable, indexing batch synchronously: %s", desc)
    _act()


__all__ = [
    "ARM_FAILED",
    "ARM_SKIPPED",
    "ARM_WRITTEN",
    "IndexOutcome",
    "NON_INDEXABLE_CONTENT_TYPES",
    "is_indexable",
    "delete_artifact_from_index",
    "enqueue_index_artifact",
    "enqueue_index_artifacts_batch",
    "get_artifact_embeddings",
    "index_artifact",
    "index_artifacts_batch",
    "move_artifact_segments",
    "prepare_reindex_items",
]
