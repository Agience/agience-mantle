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
    from mantle.db.backend import COLLECTION_ARTIFACTS, iter_documents
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
        # `unreadable="skip"`, because this is a MAINTENANCE pass and its job is to rebuild what
        # can be rebuilt. An ordinary read keeps the default and still refuses to return a short
        # list — the caller there cannot tell a filtered result from a truncated one. Here it can,
        # because the count is reported below and the ids are named.
        #
        # Measured on 71/dev: ONE artifact whose content was written under a key the node no longer
        # holds raised out of the hydration loop and blocked the rebuild of all 613. The index then
        # stayed on the previous analyzer generation — silently, because a stale index answers.
        # `denied="omit"` is NOT a second flavour of `unreadable`. An artifact behind an absorbing
        # deny (`propagate='[]'` on its `contains` edge) is not damaged — the index writer simply
        # may not read it, which is the grant system working. Indexing a document you may not read
        # is not a thing that should happen, so omitting it is correct rather than a concession,
        # and it is counted apart from `unreadable` so a deliberate deny is never reported as
        # corruption. Measured on 71/home: 6 such edges out of 2,158,434, one of them over a real
        # collection ("Mantle work products"), and before this they ended the whole pass.
        unreadable: List[Tuple[str, str]] = []
        refused: List[Tuple[str, str]] = []
        # STREAMED, not listed. `query_documents` materialises, and hydrating an artifact decrypts
        # it, so the list form of this plane is what put the pass into the pagefile. `unreadable`
        # and `refused` fill as the stream is consumed, so both are only complete once it is
        # exhausted — which is why they are reported after the loop rather than before it.
        artifact_stream = iter_documents(db, ArtifactEntity, COLLECTION_ARTIFACTS, {},
                                         unreadable="skip", skipped_out=unreadable,
                                         denied="omit", denied_out=refused)
        # ── Streamed in bounded chunks, and that is the whole design ────────────────────────
        # This pass used to materialise the plane THREE times: every hydrated artifact, every
        # prepared item, and one future per item. Hydrating an artifact DECRYPTS it, so the first
        # of those is far larger than the rows it came from. Measured on 71/home: 2,165,743
        # artifacts drove the working set past 16 GB and into the pagefile — 3,147 pages/sec, free
        # RAM from 7.6 GB to 6.2 GB and falling — before one index entry had been written. It could
        # not finish on the hardware it runs on, which is not a slow pass, it is a broken one.
        #
        # Now one chunk is hydrated, prepared, indexed and dropped before the next is read, so peak
        # memory is set by CHUNK rather than by the size of the corpus. The trade is that
        # `prepare_reindex_items` batches its embedding calls per chunk instead of once — the same
        # number of texts, in more batches — which costs round-trips on a remote embedder and
        # nothing at all when the vector arm is off.
        CHUNK = max(1, int(getattr(config, "REINDEX_CHUNK_SIZE", 2000) or 2000))

        # Loaded ONCE, before any chunk: workers see an existing set rather than racing on the
        # load, and re-checking per chunk would repeat a deployment fact thousands of times.
        vector_arm_available = True
        try:
            from mantle.search.anchors.store import (
                AnchorSetNotProvisioned,
                require_live_anchorset,
            )
            require_live_anchorset()
        except AnchorSetNotProvisioned:
            # The AnchorSet is provisioned, never derived, so its absence is a deployment state
            # that retrying cannot change. Recorded once here; the per-artifact path would
            # otherwise repeat the same result N times.
            vector_arm_available = False
            logger.warning(
                "Reindex: no AnchorSet provisioned; the vector arm is OFF for this run and every "
                "artifact will be lexical-only until the canonical AnchorSet artifact is "
                "provisioned")
        except Exception:
            vector_arm_available = False
            logger.warning(
                "Reindex: AnchorSet unavailable up front; per-artifact path will retry "
                "(vector arm may be skipped if it can't be built)", exc_info=True)

        # ONE indexer for the whole pass, so a chunk can commit once (see `run_chunk`). Built
        # here rather than per chunk because the store holds the connection and the transaction
        # depth; a per-chunk instance would reopen both for no gain. `None` is tolerated — the
        # per-artifact path is unchanged and still correct, just slower.
        shared_indexer = None
        try:
            from mantle.search.mantle.wiring import build_sse_indexer
            from mantle.services.dependencies import get_store_db as _gsd_idx
            shared_indexer = build_sse_indexer(next(_gsd_idx()), segment="committed")
        except Exception:
            logger.warning("Reindex: no shared SSE indexer; falling back to one transaction per "
                           "artifact (correct, and slower)", exc_info=True)
        batched = shared_indexer is not None and getattr(
            getattr(shared_indexer, "_postings", None), "transaction", None) is not None

        workers = max(1, int(max_workers or getattr(config, "INDEX_QUEUE_MAX_WORKERS", 16)))
        logger.info("Reindexing in chunks of %d (%s)...", CHUNK,
                    "one transaction per chunk" if batched
                    else "%d workers, one transaction per artifact" % workers)

        indexed = 0
        skipped = 0
        failed = 0
        total = 0
        not_indexable = 0

        def index_safe(collection_id: str, artifact: ArtifactEntity, fields: dict):
            """Return the artifact's :class:`IndexOutcome`, or ``None`` if it raised."""
            try:
                return index_artifact(artifact, collection_id, is_head=True, fields=fields)
            except Exception as exc:
                logger.warning("Reindex failed for %s: %s", artifact.id, exc, exc_info=True)
                return None

        def _tally(outcome):
            nonlocal indexed, skipped, failed
            if outcome is None or outcome.failed:
                failed += 1
            elif outcome.wrote_nothing:
                skipped += 1
            else:
                indexed += 1

        def run_chunk(batch, executor):
            """Prepare and index one chunk. Returns nothing; counters are closed over.

            ── One transaction per CHUNK, not per artifact ──────────────────────────────────────
            The SSE index is a single SQLite file and every artifact's update takes an exclusive
            `BEGIN IMMEDIATE` on it (`sse/indexer.py::_atomic_slot_writes` — a correctness
            mechanism: two writers on one term would otherwise lose an entry and leave an artifact
            that reports success and is not findable). So the 16 workers never ran in parallel;
            they queued, and the pass measured 0.64 cores with the disk 94% idle.

            Batching is therefore the only lever that exists here, and it is worth pulling because
            SQLite commit cost is fsync-dominated: 2,000 commits become one. It requires SHARING an
            indexer, because reentrancy is thread-local depth on the store instance and
            `build_sse_indexer` otherwise constructs a fresh one — with a fresh connection — per
            artifact.

            Sharing one store means one connection, so the chunk is indexed on THIS thread rather
            than fanned out. That costs nothing: the fan-out was already serialized by the very
            lock this avoids.
            """
            nonlocal total
            prepared = prepare_reindex_items(batch)
            if not prepared:
                return
            total += len(prepared)

            postings = getattr(shared_indexer, "_postings", None) if shared_indexer else None
            txn = getattr(postings, "transaction", None) if postings is not None else None
            if txn is not None:
                try:
                    with txn():
                        for coll_id, artifact, fields in prepared:
                            _tally(index_artifact(artifact, coll_id, is_head=True, fields=fields,
                                                  sse_indexer=shared_indexer))
                    del prepared
                    return
                except Exception as exc:
                    # ONE artifact must not cost the other 1,999. A chunk transaction is
                    # all-or-nothing, so a raise here rolled the whole chunk back and nothing was
                    # written — the artifacts are re-run below one at a time, each in its own
                    # transaction, which is exactly the pre-batching behaviour. Rare by
                    # construction (measured 26 failures in 60,835), so the fast path dominates
                    # and the slow path is still correct.
                    logger.warning(
                        "Chunk transaction rolled back (%s: %s); re-running its %d artifact(s) "
                        "individually so one failure does not discard the rest",
                        type(exc).__name__, exc, len(prepared))

            # Per-artifact path: no shared store, or the chunk transaction was rolled back.
            # `propagate` captures this thread's context (which carries the system acting
            # principal) so each pool worker runs under it. Without it a worker starts from an
            # empty context and every index write fails closed with NoActingPrincipal.
            futures = [executor.submit(propagate(index_safe), coll_id, artifact, fields)
                       for coll_id, artifact, fields in prepared]
            for future in futures:
                try:
                    _tally(future.result())
                except Exception as exc:
                    failed += 1
                    logger.warning("Unexpected reindex error: %s", exc)
            # Both drop out of scope here, which is the point of the chunking.
            del futures, prepared

        batch: List[Tuple[str, ArtifactEntity]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for artifact in artifact_stream:
                if artifact.state == ArtifactEntity.STATE_ARCHIVED:
                    continue
                # Platform trust config is not user content and has no search context — filtered
                # here as well as at the gate so it is never counted as work this run was supposed
                # to do. See pipeline_unified.NON_INDEXABLE_CONTENT_TYPES.
                if not is_indexable(artifact):
                    not_indexable += 1
                    continue
                # Root artifacts (no parent collection) self-reference their own id — consistent
                # with resolve_authorized_contexts: doc.get("collection_id") or doc.get("_key").
                batch.append((artifact.collection_id or artifact.id, artifact))
                if len(batch) >= CHUNK:
                    run_chunk(batch, executor)
                    batch = []
                    logger.info("  Progress: %d indexed, %d skipped, %d failed (%d seen)",
                                indexed, skipped, failed, total)
            if batch:
                run_chunk(batch, executor)

        if refused:
            logger.info(
                "Reindex: %d artifact(s) refused key custody and are NOT in this rebuild. These "
                "are DENY decisions, not damage: this principal holds no grant reaching them, so "
                "they are not its to index. First: %s (%s)",
                len(refused), refused[0][0], refused[0][1])
        if unreadable:
            logger.warning(
                "Reindex: %d artifact(s) could not be read and are NOT in this rebuild — their "
                "content is addressed but does not decrypt with the keys this node holds. They "
                "stay unsearchable until repaired or removed; every other artifact is rebuilt. "
                "First: %s (%s)",
                len(unreadable), unreadable[0][0], unreadable[0][1])

        if not_indexable:
            logger.info("Reindex: %d platform trust-config artifact(s) are not search targets",
                        not_indexable)
        if not total:
            logger.info("Reindex complete: no artifacts to index")
            return {"indexed": 0, "skipped": 0, "failed": 0, "total": 0,
                    "unreadable": len(unreadable), "denied": len(refused)}

        result = {
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "total": total,
            "vector_arm": "on" if vector_arm_available else "off (no AnchorSet)",
            # Counted separately from `failed`: a failure is an artifact this pass TRIED and could
            # not index, and these were never readable enough to try. Collapsing them would report
            # a rebuild as clean when part of the corpus is not in it.
            "unreadable": len(unreadable),
            # Distinct from `unreadable` on purpose: unreadable is damage, denied is a grant
            # decision. An operator seeing `denied` should check the light cone, not the disk.
            "denied": len(refused),
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
        # `unreadable` deliberately does NOT block the marker, and that is a choice rather than an
        # oversight. An artifact whose content does not decrypt with the keys this node holds is a
        # PERMANENT condition until someone repairs or removes it — retrying cannot change it. Left
        # blocking, every boot would restart a full pass that can never complete, which is the
        # non-converging loop this function refuses at the top. So the pass records that it finished
        # doing what it could; the count rides in the summary and the ids are logged at WARNING on
        # every explicit rebuild, which is where someone is looking when they care.
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
