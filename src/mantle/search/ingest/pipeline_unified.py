"""Unified artifact indexing pipeline (post-OpenSearch retirement).

Every artifact — regardless of content type — goes through the same path:

    artifact (any type, any collection)
      → SSE: tokenize title/description/tags/content → encrypted posting lists
      → MANTLE: chunk content text → embed → encrypted IVF cells

Both arms are unconditional once the wiring prerequisites (Oracle, S3,
Arango) are met. No feature flags. The router converts missing
prerequisites to 503 (no plaintext fallback by design).

OpenSearch was retired in Step 2.6.9 part 2 — the previous BM25 path,
the `bulk_index_documents` calls, and the `_prepare_base_doc` shape
that mirrored the OpenSearch document went away with it.

See `.dev/features/mantle-mvp.md` and
`.dev/features/mantle-sse-lexical-index.md`.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from embeddings import Embeddings, model_id as emb_model_id
from entities.artifact import Artifact

from search.ingest.chunking import (
    chunk_text,
    extract_text_from_context,
    should_chunk_content,
)
from search.ingest.tags import (
    normalize_tags,
    parse_tags_from_context,
)
from services.ingest_runner_service import extract_text_from_artifact

logger = logging.getLogger(__name__)

_embeddings = Embeddings()


# Each artifact STATE indexes into its own physically separate index segment
# (separate S3 prefixes — see search.mantle.wiring._segment_prefixes). The
# segment name equals the state name. States are mutually exclusive, so an
# artifact's entry lives in exactly one segment; a transition (draft→committed,
# →archived, unarchive) MOVES it (index into the new segment, purge the others).
_SEGMENTS = ("committed", "draft", "archived")


def _segment_for_state(state: str) -> str:
    """Map an artifact state to its index segment (1:1; unknown → committed)."""
    return state if state in _SEGMENTS else Artifact.STATE_COMMITTED


# Optional async queue
try:
    from search.ingest import index_queue
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


def _reconcile_native(embeddings: list):
    """Reconcile raw embeddings → native anchor-relative codes against the live
    AnchorSet. Returns ``(codes_or_None, anchorset_model_id_or_None)``, where
    ``codes`` aligns 1:1 with ``embeddings`` (``None`` per item whose dimension
    doesn't match the anchors).

    Geometry layer (canonical plan §1) — no keys/auth. The caller ensures the
    AnchorSet exists first (``require_live_anchorset``); this returns ``None``
    only on an unexpected absence (defensive).
    """
    try:
        from search.anchors.reconciler import Reconciler
        from search.anchors.store import get_crosswalks, get_live_anchorset
    except Exception:
        return None, None
    aset = get_live_anchorset()
    if aset is None or len(aset) == 0:
        return None, None
    try:
        rec = Reconciler(aset, crosswalks=get_crosswalks())
        codes = [
            rec.to_native(emb).to_dict()
            if (emb is not None and len(emb) == aset.dim)
            else None
            for emb in embeddings
        ]
        return codes, aset.model_id
    except Exception:
        logger.debug("MANTLE: native reconcile skipped", exc_info=True)
        return None, aset.model_id


def _density_layers(embeddings: list):
    """Per-chunk density-zoom layer (L0/L1/L2 + density) over the live AnchorSet,
    aligned 1:1 with ``embeddings`` (``None`` per item that can't be placed).
    The caller ensures the AnchorSet exists first; returns ``None`` only on an
    unexpected absence (defensive). Geometry layer (§1)."""
    try:
        from search.anchors.store import get_density_zoom
    except Exception:
        return None
    dz = get_density_zoom()
    if dz is None:
        return None
    dim = dz.anchorset.dim
    out = []
    for emb in embeddings:
        if emb is None or len(emb) != dim:
            out.append(None)
        else:
            layer, dens = dz.layer(emb)
            out.append((layer, round(float(dens), 4)))
    return out


# ============================================================
#  MANTLE vector hook — encrypted IVF, chunks + embeddings
# ============================================================


def _mantle_index_artifact(
    artifact: Artifact,
    collection_id: str,
    fields: dict[str, str],
    *,
    segment: str = "committed",
) -> None:
    """Chunk + embed the artifact's content, write to MANTLE cells."""
    content = fields.get("content", "")
    if not content:
        return

    if not collection_id:
        return

    artifact_root = artifact.root_id or artifact.id

    # Chunk + embed (shared chunker — see _content_chunks). On the bulk-reindex
    # path these exact texts were already embedded in batch, so this call is a
    # cache hit (no per-artifact round-trip).
    chunks = _content_chunks(content)
    texts = [c["text"] for c in chunks if c.get("text")]
    if not texts:
        return
    try:
        embeddings = _embeddings(texts)
    except Exception:
        logger.warning(
            "MANTLE: embedding failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return
    if not any(embeddings):
        return

    # The AnchorSet is the one coordinate system (canonical plan §3). Ensure it
    # exists — bootstrapping from the seed corpus on first use — so reconcile,
    # density, and routing all see the same anchors. No flat fallback: if it
    # can't be created, skip the vector arm (the commit still succeeds).
    try:
        from search.anchors.store import require_live_anchorset
        require_live_anchorset()
    except Exception:
        logger.warning(
            "MANTLE: AnchorSet unavailable; skipping vector index for %s",
            artifact.id, exc_info=True,
        )
        return

    # Native language: reconcile raw vectors → sparse anchor-relative codes,
    # plus a density-zoom layer. Provenance: model_id per chunk.
    native_codes, anchorset_model_id = _reconcile_native(embeddings)
    density = _density_layers(embeddings)
    chunk_model_id = anchorset_model_id or emb_model_id()

    mantle_chunks = []
    for i, emb in enumerate(embeddings):
        if emb is None:
            continue
        record = {
            "artifact_id": artifact_root,
            "chunk_id": int(chunks[i].get("chunk_id", i)),
            "embedding": emb,
            "text": chunks[i].get("text", ""),
            "model_id": chunk_model_id,
        }
        if native_codes is not None and native_codes[i] is not None:
            record["native"] = native_codes[i]
        if density is not None and density[i] is not None:
            record["density_layer"], record["density"] = density[i]
        mantle_chunks.append(record)
    if not mantle_chunks:
        return

    try:
        from services.dependencies import get_arango_db
        from search.mantle.principal import resolve_cell_principal
        from search.mantle.wiring import build_indexer
    except Exception:
        logger.debug("MANTLE wiring unavailable; skipping vector index", exc_info=True)
        return

    try:
        arango_db = next(get_arango_db())
    except Exception:
        logger.debug("MANTLE: arango handle unavailable; skipping", exc_info=True)
        return

    indexer = build_indexer(arango_db, segment=segment)
    if indexer is None:
        logger.debug("MANTLE indexer prerequisites missing; skipping")
        return

    # The cell-key principal is the collection's immutable origin root (NOT
    # created_by / ownership) — index and query resolve it identically, so the
    # same key is derived at both ends. See search.mantle.principal.
    principal_id = resolve_cell_principal(arango_db, collection_id)
    if not principal_id:
        return

    try:
        touched = indexer.index_artifact(principal_id, collection_id, mantle_chunks)
        logger.info(
            "MANTLE indexed artifact %s (principal=%s collection=%s, %d cells)",
            artifact.id, principal_id, collection_id, touched,
        )
    except Exception:
        logger.warning(
            "MANTLE indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )


def _mantle_remove_artifact(
    principal_id: str, collection_id: str, artifact_id: str,
    *, segment: str = "committed",
) -> None:
    """Strip an artifact's chunks from MANTLE cells in one index segment."""
    try:
        from services.dependencies import get_arango_db
        from search.mantle.wiring import build_indexer
        arango_db = next(get_arango_db())
        indexer = build_indexer(arango_db, segment=segment)
        if indexer is None:
            return
        indexer.remove_artifact(principal_id, collection_id, artifact_id)
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
) -> None:
    """Write per-field text into the SSE blind-token posting lists."""
    if not fields:
        return

    if not collection_id:
        return
    artifact_id = artifact.root_id or artifact.id

    try:
        from services.dependencies import get_arango_db
        from search.mantle.principal import resolve_cell_principal
        from search.mantle.wiring import build_sse_indexer
    except Exception:
        logger.debug("SSE wiring unavailable; skipping lexical index", exc_info=True)
        return

    try:
        arango_db = next(get_arango_db())
    except Exception:
        logger.debug("SSE: arango handle unavailable; skipping", exc_info=True)
        return

    indexer = build_sse_indexer(arango_db, segment=segment)
    if indexer is None:
        logger.debug("SSE indexer prerequisites missing; skipping")
        return

    # Same principal as the vector arm: the collection's origin root.
    principal_id = resolve_cell_principal(arango_db, collection_id)
    if not principal_id:
        return

    try:
        n = indexer.index_artifact(principal_id, collection_id, artifact_id, fields)
        logger.info(
            "SSE indexed artifact %s (principal=%s collection=%s, %d tokens)",
            artifact.id, principal_id, collection_id, n,
        )
    except Exception:
        logger.warning(
            "SSE indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )


def _sse_remove_artifact(
    principal_id: str, artifact_id: str, *, segment: str = "committed",
) -> None:
    """Strip an artifact's references from the SSE index in one segment."""
    try:
        from services.dependencies import get_arango_db
        from search.mantle.wiring import build_sse_indexer
        arango_db = next(get_arango_db())
        indexer = build_sse_indexer(arango_db, segment=segment)
        if indexer is None:
            return
        indexer.remove_artifact(principal_id, artifact_id)
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

    Called at a STATE TRANSITION to vacate the segment(s) the artifact is leaving
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
        from search.mantle.wiring import build_indexer
        # Availability gate FIRST — build_indexer doesn't use the db handle, and
        # resolving the handle is the expensive part. This makes the whole thing
        # a fast no-op where search isn't wired (e.g. tests with no S3/oracle),
        # before any DB connection is attempted.
        if build_indexer(None) is None:
            return
        from services.dependencies import get_arango_db
        from search.mantle.principal import resolve_cell_principal
        arango_db = next(get_arango_db())
        principal_id = resolve_cell_principal(arango_db, collection_id)
    except Exception:
        logger.debug("segment move: principal unresolved", exc_info=True)
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
    """Return the artifact's stored MANTLE vector chunk records (its CURRENT state's
    segment): each ``{chunk_id, embedding, model_id, ...}``, ordered by chunk_id.

    Reads the vectors back out of the encrypted cells — the inverse of indexing.
    Empty if the vector arm isn't wired (no EMBEDDINGS_URI / no cell store) or the
    artifact has nothing stored (e.g. a container, or lexical-only deploy)."""
    if not collection_id:
        return []
    segment = _segment_for_state(artifact.state)
    try:
        from services.dependencies import get_arango_db
        from search.mantle.principal import resolve_cell_principal
        from search.mantle.wiring import build_indexer
        arango_db = next(get_arango_db())
        indexer = build_indexer(arango_db, segment=segment)
        if indexer is None:
            return []
        principal_id = resolve_cell_principal(arango_db, collection_id)
        if not principal_id:
            return []
        root = artifact.root_id or artifact.id
        chunks = [
            c for c in indexer.collection_chunks(principal_id, collection_id)
            if c.get("artifact_id") == root
        ]
        chunks.sort(key=lambda c: c.get("chunk_id", 0))
        return chunks
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
) -> bool:
    """Index one artifact into the index segment for its CURRENT state.

    ``draft`` / ``committed`` / ``archived`` each have a separate physical index
    (separate S3 prefixes per arm), so the artifact is written into the segment
    matching its state. This does NOT touch the other segments — a root may hold
    a committed version *and* a WIP draft simultaneously; vacating a segment on a
    state transition is the caller's job via :func:`move_artifact_segments`.
    (Previously only committed was indexed and archived was skipped entirely.)

    ``is_head`` is preserved for caller compatibility but no longer drives index
    branching (versioning is artifact-level). ``fields`` may be supplied by the
    caller — the bulk reindex extracts them once and prewarms the embeddings
    cache, then passes them here so this path neither re-extracts nor makes a
    per-artifact embed round-trip. When ``None`` they are extracted here.
    """
    segment = _segment_for_state(artifact.state)
    try:
        if fields is None:
            fields = _extract_artifact_fields(artifact)
        if not fields:
            logger.debug(
                "Artifact %s has no analyzable fields; skipping", artifact.id,
            )
            return False
        _sse_index_artifact(artifact, collection_id, fields, segment=segment)
        _mantle_index_artifact(artifact, collection_id, fields, segment=segment)
        logger.info(
            "Indexed artifact %s in collection %s (segment=%s)",
            artifact.id, collection_id, segment,
        )
        return True
    except Exception:
        logger.error(
            "Indexing failed for artifact %s",
            artifact.id, exc_info=True,
        )
        return False


# ============================================================
#  Bulk reindex prep: extract fields once + batch-warm embeddings
# ============================================================

# Texts embedded per HTTP call when prewarming the bulk reindex. bge-m3 handles
# this batch comfortably on the GPU; it collapses N per-artifact round-trips
# into ceil(total_unique_chunks / EMBED_BATCH_SIZE) calls — the dominant cost of
# a cold-cache reindex is the round-trip COUNT, not GPU compute.
EMBED_BATCH_SIZE = 64


def prepare_reindex_items(
    items: list[tuple[str, Artifact]],
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> list[tuple[str, Artifact, dict[str, str]]]:
    """Prepare a bulk-reindex work list with the embeddings cache prewarmed.

    For each ``(collection_id, artifact)``: extract its analyzable fields once
    (skipping archived / field-less artifacts) and collect every UNIQUE MANTLE
    chunk text across ALL artifacts. Then embed those texts in batched HTTP
    calls, which populates the long-term embeddings cache.

    The returned ``(collection_id, artifact, fields)`` tuples feed
    :func:`index_artifact` (pass ``fields=...``); its per-artifact embed then
    hits the warm cache instead of making one round-trip per artifact.

    Built for a COLD cache — the batched prewarm IS the fast path, not an
    optimization that assumes prior warmth. Identical texts (boilerplate shared
    across artifacts) are embedded once.
    """
    prepared: list[tuple[str, Artifact, dict[str, str]]] = []
    all_texts: list[str] = []
    seen: set[str] = set()

    for collection_id, artifact in items:
        # All states are indexed now (into their own segment); the prewarm cache
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
    handles per-artifact embedding without cross-artifact batching
    after OpenSearch retirement (the previous bulk path was OpenSearch-
    specific). For very large bulk reindex jobs, the admin command
    runs many of these in parallel via the index queue.
    """
    if not artifacts:
        return True

    start_time = time.time()
    logger.info("Starting bulk index of %d artifacts", len(artifacts))
    indexed = 0
    skipped = 0

    for artifact in artifacts:
        # index_artifact routes each artifact to its state's segment (no skips).
        if index_artifact(artifact, collection_id, is_head=is_head):
            indexed += 1
        else:
            skipped += 1

    total_time = time.time() - start_time
    logger.info(
        "Bulk indexed %d artifacts (%d skipped) in %.3fs",
        indexed, skipped, total_time,
    )
    return True


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

    Callers without ``principal_id`` (legacy pre-Step-2.6 paths) get a
    no-op — there's nothing to remove without identity.
    """
    try:
        root = root_id or version_id
        # Hard delete: the artifact is gone for good, so purge it from EVERY
        # segment (we don't track which state it was last indexed under).
        for seg in _SEGMENTS:
            if principal_id and collection_id:
                _mantle_remove_artifact(principal_id, collection_id, root, segment=seg)
            if principal_id:
                _sse_remove_artifact(principal_id, root, segment=seg)
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
) -> None:
    """Enqueue an artifact for async indexing; falls back to sync.

    ``vacate`` names index segment(s) the artifact is LEAVING on a state
    transition — they're removed in the same job, right after (re)indexing into
    the new state's segment. Folding the move into this one job keeps the
    transition atomic from the caller's view and gives a single mock point.
    """
    def _act() -> bool:
        ok = index_artifact(artifact, collection_id, is_head=is_head)
        if vacate:
            move_artifact_segments(artifact, collection_id, remove_from=vacate)
        return ok

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
    "delete_artifact_from_index",
    "enqueue_index_artifact",
    "enqueue_index_artifacts_batch",
    "get_artifact_embeddings",
    "index_artifact",
    "index_artifacts_batch",
    "move_artifact_segments",
    "prepare_reindex_items",
]
