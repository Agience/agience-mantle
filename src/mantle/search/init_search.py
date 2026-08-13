"""Search initialization.

There is no per-startup index creation step — both MANTLE vector cells and
MANTLE-SSE posting lists are objects created lazily on first commit (the indexer
auto-bootstraps owner domains), on S3 or on local disk.

Lazy bootstrap covers index STORAGE, and only that. It does not cover the semantic arm's other
prerequisite: an AnchorSet is provisioned by an operator and is never created on first commit,
on first boot, or by anything in this module. Startup on a node without one is silent — the
first evidence is a per-write WARNING from the ingest path, or the ``vector_arm`` field
:func:`reindex_all_artifacts` returns.

What lives here:

- :func:`reindex_all_artifacts` — bulk re-encryption walker. Used as a
  one-shot admin command to populate the encrypted indexes from
  existing artifacts, e.g. after a key rotation or a fresh deploy.
- :func:`init_search` — startup hook (no-op shape preserved for
  callers; logs that there's nothing to initialize).
- :func:`shutdown_search` — startup hook (no-op).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from mantle.services.acting_principal import propagate
from mantle.services.system_identity import system_acting_context

logger = logging.getLogger(__name__)


def init_search() -> None:
    """Startup hook — nothing to initialize.

    Cells / posting lists / stats blobs are created on first commit by
    the indexer's auto-bootstrap path. This function is kept so the
    main.py lifespan call site stays stable; it just logs.
    """
    logger.info(
        "Search init: MANTLE + MANTLE-SSE use lazy bootstrap (no startup work)."
    )


def shutdown_search() -> None:
    """Shutdown hook — nothing to close. Kept for caller compatibility."""
    pass


def reindex_all_artifacts(*, max_workers: Optional[int] = None) -> dict:
    """Bulk-reindex every non-archived artifact into MANTLE + MANTLE-SSE.

    One-shot admin operation. Use cases:

    - Migrating to a new index.
    - Re-encrypting after an owner key rotation.
    - Recovering from a corrupted S3 prefix.

    Queries all artifacts directly from the artifact store and runs the
    standard :func:`pipeline_unified.index_artifact` path on each.
    collection_id is read from the artifact document; root artifacts
    (no parent) self-reference their own id, matching the convention
    used by resolve_authorized_contexts. Idempotent — re-running is safe.

    Performance (cold cache is the design point, never assume a warm one):

    1. ``prepare_reindex_items`` extracts each artifact's fields once and
       embeds every unique chunk text in BATCHED HTTP calls — collapsing one
       round-trip per artifact into a handful of batches. This is the dominant
       win: a remote GPU embedder is latency-bound, so call COUNT is the cost.
    2. The AnchorSet is loaded once, single-threaded, before fan-out, so workers
       don't contend on the load. It is provisioned, not derived; when absent,
       the vector arm is off for the whole run and each artifact is lexical-only.
    3. Fan-out runs at ``INDEX_QUEUE_MAX_WORKERS`` (default 16), and each worker
       passes its precomputed ``fields`` to ``index_artifact`` so there is no
       re-extraction and no per-artifact embed round-trip (cache hits only).

    Returns ``{"indexed", "skipped", "failed", "total", "vector_arm"}``, where
    ``indexed`` counts artifacts for which at least one arm wrote. ``skipped`` is the
    third outcome — nothing to write, or an arm whose prerequisites (an AnchorSet, a
    wired indexer) are absent — and it keeps ``failed`` meaning failed.

    Runs as the platform system principal: it has no request context, and it both
    reads every artifact (decrypting content) and writes every index cell, so it
    needs an identity the grant ledger can check. The wrapper is here rather than at
    the call sites because this function is reached from startup, an admin endpoint
    and the CLI — one wrapper covers all three and none can forget it.
    """
    with system_acting_context(scope="platform.reindex"):
        return _reindex_all_artifacts(max_workers=max_workers)


def _reindex_all_artifacts(*, max_workers: Optional[int] = None) -> dict:
    """Body of :func:`reindex_all_artifacts`; assumes an acting principal is set."""
    from mantle import config
    from mantle.db.backend import COLLECTION_ARTIFACTS, query_documents
    from mantle.entities.artifact import Artifact as ArtifactEntity
    from mantle.search.ingest.pipeline_unified import (
        index_artifact,
        is_indexable,
        prepare_reindex_items,
    )
    from mantle.services.dependencies import get_store_db

    #
    # Without usable index storage, the reindex is triggered on every boot (`is_post_setup =
    # not _platform_seeded`, and `platform.bootstrap.seeded` is False on this store because
    # Mantle explicitly does not seed itself — the flag has no writer). A full pass cannot
    # finish before the next restart, so it begins again from zero: a loop that never
    # converges, wearing the disk and burning CPU for no persisted result.
    #
    # Refusing here is not a fallback, it is the honest answer to "is there anywhere to put this?".
    # The check is a measurement (`head_bucket` for S3, a created root directory for the local
    # store — see `wiring._build_sse_stores`), not an inference from an object existing.
    # [[verification-that-cannot-fail]]
    #
    # It asks about the selected backend, not about S3 specifically: a standalone install can
    # hold a local index, and asking about S3 alone would skip the reindex on a store that has
    # somewhere perfectly good to put one, leaving `POST /artifacts/recall` answering nothing on
    # the install that needs it most.
    try:
        from mantle.search.mantle.wiring import sse_index_storage_available
        if not sse_index_storage_available():
            logger.warning(
                "Reindex SKIPPED: no usable index storage, so a full pass could not persist "
                "anything. Nothing was scanned. Point the edge S3 bucket at a reachable endpoint "
                "(or provide credentials), or set MANTLE_SSE_DIR to a writable directory, and "
                "reindex explicitly.")
            return {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}
    except ImportError:
        pass                       # wiring unavailable → fall through and let the arms decide

    #
    # A flag nobody sets cannot answer "has this been done". The honest condition is a record of the
    # work actually completing, written by the thing that completes it — so the marker is set at the
    # end of a successful pass below, and read here. First boot indexes; later boots skip; a rebuild
    # is an explicit call, which is what a rebuild should be. [[never-impose-knowledge-derive-it]]
    _marker = "search.reindex.completed_at"
    try:
        from mantle.services.dependencies import get_store_db as _gsd
        from mantle.services.platform_settings_service import settings as _ps
        _g = _gsd(); _db0 = next(_g)
        try:
            _done = _ps.get(_marker) if hasattr(_ps, "get") else None
        finally:
            _g.close()
        if _done:
            logger.info(
                "Reindex SKIPPED: a full pass already completed (%s = %s). Call "
                "reindex_all_artifacts() explicitly to rebuild.", _marker, _done)
            return {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}
    except Exception:
        pass                       # cannot read the marker → do the work rather than skip blindly

    logger.info("Starting full reindex of all artifacts...")

    db_gen = get_store_db()
    db = next(db_gen)

    try:
        all_artifacts = list(query_documents(db, ArtifactEntity, COLLECTION_ARTIFACTS, {}))
        logger.info("Found %d artifacts to reindex", len(all_artifacts))

        items: List[Tuple[str, ArtifactEntity]] = []
        not_indexable = 0
        for artifact in all_artifacts:
            if artifact.state == ArtifactEntity.STATE_ARCHIVED:
                continue
            # Platform trust config is not user content and has no search context —
            # filtered here as well as at the gate so it is never counted as work
            # this run was supposed to do. See pipeline_unified.NON_INDEXABLE_CONTENT_TYPES.
            if not is_indexable(artifact):
                not_indexable += 1
                continue
            # Root artifacts (no parent collection) self-reference their own id —
            # consistent with resolve_authorized_contexts: doc.get("collection_id") or doc.get("_key").
            collection_id = artifact.collection_id or artifact.id
            items.append((collection_id, artifact))

        if not_indexable:
            logger.info(
                "Reindex: %d platform trust-config artifact(s) are not search targets",
                not_indexable,
            )

        if not items:
            logger.info("Reindex complete: no artifacts to index")
            return {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}

        # Extract fields once + batch-warm the embeddings cache (the fast path).
        prepared = prepare_reindex_items(items)
        if not prepared:
            logger.info("Reindex complete: no indexable artifacts")
            return {"indexed": 0, "skipped": 0, "failed": 0, "total": 0}

        # Load the AnchorSet once up front (single-threaded): workers then see an
        # existing set rather than racing on the load, and its embeds hit the warm cache.
        vector_arm_available = True
        try:
            from mantle.search.anchors.store import (
                AnchorSetNotProvisioned,
                require_live_anchorset,
            )
            require_live_anchorset()
        except AnchorSetNotProvisioned:
            # The AnchorSet is provisioned, never derived, so its absence is a
            # deployment state that retrying cannot change. Recorded once here; the
            # per-artifact path would otherwise repeat the same result N times.
            vector_arm_available = False
            logger.warning(
                "Reindex: no AnchorSet provisioned; the vector arm is OFF for this "
                "run and %d artifact(s) will be lexical-only until the canonical "
                "AnchorSet artifact is provisioned",
                len(prepared),
            )
        except Exception:
            vector_arm_available = False
            logger.warning(
                "Reindex: AnchorSet unavailable up front; per-artifact path "
                "will retry (vector arm may be skipped if it can't be built)",
                exc_info=True,
            )

        workers = max(
            1,
            min(
                max_workers or getattr(config, "INDEX_QUEUE_MAX_WORKERS", 16),
                len(prepared),
            ),
        )
        logger.info("Reindexing %d artifacts (%d workers)...", len(prepared), workers)

        indexed = 0
        skipped = 0
        failed = 0

        def index_safe(
            collection_id: str, artifact: ArtifactEntity, fields: dict
        ):
            """Return the artifact's :class:`IndexOutcome`, or ``None`` if it raised.
            """
            try:
                return index_artifact(
                    artifact, collection_id, is_head=True, fields=fields
                )
            except Exception as exc:
                logger.warning(
                    "Reindex failed for %s: %s", artifact.id, exc, exc_info=True,
                )
                return None

        # `propagate` captures this thread's context (which carries the system
        # acting principal, set below) so each pool worker runs under it. Without
        # it a worker starts from an empty context and every index write fails
        # closed with NoActingPrincipal — see services.acting_principal.propagate.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(propagate(index_safe), coll_id, artifact, fields)
                for coll_id, artifact, fields in prepared
            ]
            for i, future in enumerate(futures, 1):
                try:
                    outcome = future.result()
                    if outcome is None or outcome.failed:
                        failed += 1
                    elif outcome.wrote_nothing:
                        skipped += 1
                    else:
                        indexed += 1
                except Exception as exc:
                    failed += 1
                    logger.warning("Unexpected reindex error: %s", exc)
                if i % 50 == 0:
                    logger.info(
                        "  Progress: %d/%d (%d indexed, %d skipped, %d failed)",
                        i, len(futures), indexed, skipped, failed,
                    )

        result = {
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "total": len(prepared),
            "vector_arm": "on" if vector_arm_available else "off (no AnchorSet)",
        }
        # A reindex that wrote nothing, or that had failures, logs at WARNING.
        logger.log(
            logging.WARNING if (failed or not indexed) else logging.INFO,
            "Reindex complete: %s", result,
        )

        # ── The writer the boot gate never had ───────────────────────────────────────────────
        # Recorded only on a clean pass: work was done and nothing failed. A run with failures
        # leaves the marker unset, so the next boot tries again — which is the point. Claiming
        # completion after a partial pass would turn a one-off loop into a permanently half-built
        # index that reports itself finished, which is worse than repeating the work.
        if indexed and not failed:
            try:
                from datetime import datetime, timezone
                from mantle.services.platform_settings_service import settings as _ps
                _stamp = datetime.now(timezone.utc).isoformat()
                _ps.set_setting(db, "search.reindex.completed_at", _stamp, category="search")
                logger.info("Reindex marker set: search.reindex.completed_at=%s "
                            "(later boots skip; call reindex_all_artifacts() to rebuild)", _stamp)
            except Exception:
                logger.warning("Reindex finished but the completion marker could not be written — "
                               "the next boot will reindex again", exc_info=True)
        return result

    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


# Backwards-compatible aliases for legacy call sites.
init_search_indices = init_search


def reindex_in_background() -> None:
    """Spawn :func:`reindex_all_artifacts` in a background thread.

    Convenient for admin endpoints that don't want to block on a long
    reindex operation. The thread is daemon-flagged so process shutdown
    doesn't wait on it.
    """
    threading.Thread(target=reindex_all_artifacts, daemon=True).start()


__all__ = [
    "init_search",
    "init_search_indices",
    "reindex_all_artifacts",
    "reindex_in_background",
    "shutdown_search",
]
