"""MantleQueryEngine — encrypted IVF search over authorized cells.

Step 2.3 implementation. The query-time inverse of :class:`MantleIndexer`:

    query_vec ─► nearest_clusters ─► oracle.derive_cell_key ─► cell_store.get
                                                                     │
                                  unpack_cell ◄────────────────────────┘
                                       │
                                ANN within decrypted vectors
                                       │
                                dedup by (artifact_id, chunk_id)
                                       │
                                       ▼
                                    scored hits

Composes the substrate built in 2.2:

- :class:`OracleService` (2.2a) — derives cell keys; refuses keys for
  revoked grants
- :func:`cell.unpack_cell` (2.2b.i) — decrypts + deserializes
- :mod:`clustering` (2.2b.ii) — routes the query to its nearest clusters
- :class:`CellStore` + :class:`CentroidStore` (2.2b.iii) — opaque storage

Cell cache: an in-memory dict with TTL (configurable, default 60s) keeps
recently-decrypted cells around so repeat queries don't pay the crypto
cost twice. Plaintext is held only for the cache window.

See `.dev/features/mantle-mvp.md` § Layer 2c.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import cell as cell_mod
from .oracle import KeyRequest, OracleService
from .stores import CellStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class SystemicKeyFailure(RuntimeError):
    """Every readable cell failed GCM authentication — the key is wrong, not the data.

    Raised instead of returning an empty result set, because "no results" and "your
    entire corpus is unreadable because the master key is gone" MUST NOT be the
    same observable outcome. See the raise site in :meth:`MantleQueryEngine.search`.
    """


class _CellOutcome(enum.Enum):
    """What happened when a cell was fetched — the evidence the caller reasons over."""

    OK = "ok"                # decrypted and authenticated
    TAMPERED = "tampered"    # GCM auth failed: evidence ABOUT THE KEY
    MALFORMED = "malformed"  # unparseable blob: a storage defect, says nothing about the key

@dataclass(frozen=True)
class MantleHit:
    """One decrypted, scored hit from the MANTLE query path.

    ``score`` is cosine similarity in [-1, 1]; higher is closer. When fused
    with BM25 in the accessor (Step 2.4), it'll be fed into RRF.
    """
    artifact_id: str
    chunk_id: int
    score: float
    principal_id: str
    collection_id: str


# ---------------------------------------------------------------------------
# Cell cache
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    chunks: List[dict]
    expires_at: float


class _CellCache:
    """Thread-safe TTL cache keyed by ``(principal_id, collection_id)``.

    Trades plaintext-in-memory window against crypto round-trip cost. The
    default 60s TTL matches the MANTLE MVP spec's grant-check cache.
    """

    def __init__(self, ttl_seconds: int = 60) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str, str], _CacheEntry] = {}

    def get(
        self, principal_id: str, collection_id: str, cluster_id: str = ""
    ) -> Optional[List[dict]]:
        key = (principal_id, collection_id, cluster_id)
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at < now:
                self._entries.pop(key, None)
                return None
            return entry.chunks

    def put(
        self,
        principal_id: str,
        collection_id: str,
        chunks: List[dict],
        cluster_id: str = "",
    ) -> None:
        key = (principal_id, collection_id, cluster_id)
        with self._lock:
            self._entries[key] = _CacheEntry(
                chunks=chunks, expires_at=time.time() + self._ttl
            )

    def evict(self, principal_id: str, collection_id: str, cluster_id: str = "") -> None:
        key = (principal_id, collection_id, cluster_id)
        with self._lock:
            self._entries.pop(key, None)

    def evict_context(self, principal_id: str, collection_id: str) -> None:
        """Evict every cached cluster of one ``(owner, collection)`` — an
        artifact's chunks span several anchor cells, so re-index invalidation
        must drop them all, not just one cluster."""
        with self._lock:
            stale = [
                k for k in self._entries
                if k[0] == principal_id and k[1] == collection_id
            ]
            for k in stale:
                self._entries.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# MantleQueryEngine
# ---------------------------------------------------------------------------

class MantleQueryEngine:
    """Encrypted anchor-routed ANN query path."""

    def __init__(
        self,
        oracle: OracleService,
        cell_store: Optional[CellStore] = None,
        *,
        cell_cache_ttl_s: int = 60,
        nprobe: int = 8,
        enable_selfsize: Optional[bool] = None,
        selfsize_z: float = 3.5,
        selfsize_min_k: int = 5,
    ) -> None:
        self._oracle = oracle
        self._cells = cell_store
        self._cache = _CellCache(ttl_seconds=cell_cache_ttl_s)
        self._nprobe = nprobe
        # Entroptics self-sizing result cut (MANTLE-SEARCH-SPEC Change B1). **Default
        # OFF** — it changes result COUNTS and must be A/B'd on retrieval quality
        # before enabling. Toggle via env MANTLE_SEARCH_SELFSIZE=1 (no redeploy).
        import os as _os
        if enable_selfsize is None:
            enable_selfsize = _os.getenv("MANTLE_SEARCH_SELFSIZE", "") in ("1", "true", "True")
        self._selfsize = bool(enable_selfsize)
        self._selfsize_z = float(_os.getenv("MANTLE_SEARCH_SELFSIZE_Z", str(selfsize_z)))
        self._min_k = int(_os.getenv("MANTLE_SEARCH_SELFSIZE_MIN_K", str(selfsize_min_k)))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: Sequence[float],
        authorized_contexts: Iterable[Tuple[str, str]],
        request: "KeyRequest",
        *,
        top_k: int = 50,
    ) -> List[MantleHit]:
        """Run a vector search over the union of authorized cells.

        ``authorized_contexts`` is an iterable of ``(principal_id, collection_id)``
        tuples — the result of :class:`LightConeResolver`.resolve() filtered
        through the artifact ownership graph.

        ``request`` is the REQUIRED :class:`~search.mantle.oracle.KeyRequest`
        identifying the principal on whose behalf the search runs. It is passed
        down to every cell-key derivation, so the oracle re-verifies the grant for
        each context rather than trusting that ``authorized_contexts`` was built
        honestly. That redundancy is the point: the caller computing the context
        list and the custodian issuing the keys check the same grants
        independently.

        For each context, fetch the routed cells, decrypt, run cosine ANN, and
        merge results. Deduplicates by ``(artifact_id, chunk_id)`` — when a
        chunk appears in multiple authorized contexts, the highest score wins.

        Returns up to ``top_k`` hits sorted by descending score.

        Raises :class:`ValueError` when ``query_embedding`` is empty or
        when the cell store isn't wired up.
        """
        if self._cells is None:
            raise ValueError(
                "MantleQueryEngine requires cell_store"
            )

        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim != 1 or q.size == 0:
            raise ValueError("query_embedding must be a non-empty 1-D vector")
        # Normalize once — ANN uses the same metric as clustering (cosine).
        norm = float(np.linalg.norm(q))
        if norm == 0:
            raise ValueError("query_embedding has zero norm")
        q = q / norm

        contexts = list(authorized_contexts)
        if not contexts:
            return []

        if top_k <= 0:
            return []

        # Route the query to its nearest-anchor clusters (canonical plan §5.1):
        # decrypt only those cells per authorized context, not the whole union.
        # The AnchorSet is mandatory (bootstrapped from the seed corpus on first
        # use); there is one path and no flat fallback.
        from mantle.search.anchors.routing import route_query
        from mantle.search.anchors.store import require_live_anchorset

        clusters = route_query(require_live_anchorset(), q, nprobe=self._nprobe)

        # Best-score-wins dedup.
        best: dict[Tuple[str, int], MantleHit] = {}

        # FIX 2 bookkeeping — see `_load_cell` and the systemic-fault check below.
        attempted = 0
        failed_auth = 0

        for principal_id, collection_id in contexts:
            if not principal_id or not collection_id:
                continue
            for cluster_id in clusters:
                chunks, outcome = self._load_cell(
                    principal_id, collection_id, cluster_id, request
                )
                # Only OK and TAMPERED are evidence about the key; MALFORMED and
                # absent cells are excluded from BOTH sides of the ratio so a
                # storage defect can neither trigger nor mask a key fault.
                if outcome is _CellOutcome.OK:
                    attempted += 1
                elif outcome is _CellOutcome.TAMPERED:
                    attempted += 1
                    failed_auth += 1
                if chunks:
                    self._score_chunks(q, chunks, principal_id, collection_id, best)

        # ⛔ TOTAL KEY LOSS USED TO BE INDISTINGUISHABLE FROM AN EMPTY CORPUS.
        # Every cell failing GCM auth was logged and treated as a cache miss, so a
        # wrong/rotated/destroyed master key returned `[]` — "no results" — with
        # only a warning in a log nobody reads. That is the worst possible
        # presentation of unrecoverable data loss: it looks like a working system.
        #
        # The original DoS reasoning is CORRECT and is preserved: one tampered cell
        # must not fail a whole query, because then anyone who can corrupt a single
        # cell can deny the entire corpus. So the two conditions are separated by
        # the only signal that actually distinguishes them — the RATIO, not the
        # event:
        #
        #   some cells decrypt, some don't  → per-cell corruption. Degrade, as before.
        #   EVERY cell we could read fails  → the key itself is wrong. Systemic; raise.
        #
        # A per-cell attacker cannot reach the second branch without already having
        # corrupted every cell in every authorized context, at which point the
        # corpus is gone anyway and saying so is the correct behaviour.
        #
        # ⚠ Known edge: a principal whose ONLY cell is genuinely tampered raises
        # instead of degrading. That is accepted deliberately — with a sample size
        # of one there IS no evidence distinguishing "this cell is corrupt" from
        # "the key is wrong", and of the two possible errors, wrongly reporting a
        # systemic fault is recoverable while wrongly reporting an empty corpus is
        # the failure mode that already cost us production data.
        if attempted and failed_auth == attempted:
            raise SystemicKeyFailure(
                f"all {attempted} readable cell(s) across {len(contexts)} authorized "
                f"context(s) failed GCM authentication — this is a key fault "
                f"(wrong/rotated/destroyed master key), not an empty corpus; "
                f"refusing to report 'no results'"
            )

        # Sort by score descending.
        hits = sorted(best.values(), key=lambda h: h.score, reverse=True)

        # Optional entroptics noise-calibrated self-sizing (MANTLE-SEARCH-SPEC B1,
        # default OFF). Random cosine in d dims ~ N(0, 1/d) → noise scale ~1/sqrt(d);
        # keep only hits a z-sigma above the null, never below min_k, never above the
        # caller's top_k. This is the entroptics principle (read the noise floor from
        # the data) applied to the score distribution — NOT the search itself.
        if self._selfsize and hits:
            floor = self._selfsize_z / (q.size ** 0.5)
            k = sum(1 for h in hits if h.score > floor)
            hits = hits[: max(self._min_k, k)]

        return hits[:top_k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_cell(
        self,
        principal_id: str,
        collection_id: str,
        cluster_id: str,
        request: "KeyRequest",
    ) -> Tuple[List[dict], "Optional[_CellOutcome]"]:
        """Cache-aware cell fetch. **Authorizes before it serves anything.**

        Returns ``(chunks, outcome)``. ``outcome`` is ``None`` when no decryption
        was attempted (cache hit, or the cell is simply absent — absence is not a
        failure and must not count toward the systemic-fault ratio), otherwise the
        :class:`_CellOutcome` of the attempt.

        A cell that fails GCM authentication still degrades to a miss HERE, exactly
        as before — the DoS argument for that is sound. What changed is that the
        outcome is now REPORTED to the caller, which can see the difference between
        one bad cell and a wholesale key fault. Losing that distinction was the
        defect, not the local degradation.

        ⚠ ``request`` IS REQUIRED AND HAS NO DEFAULT. It previously defaulted to
        ``None``, which meant a caller that simply forgot it reached the oracle with
        ``None`` and got a ``TypeError`` — an unauthenticated request surfacing as a
        crash rather than a denial, and only if it got as far as the oracle at all.
        """
        # ⚠ ORDERING IS LOAD-BEARING — AUTHORIZE BEFORE THE CACHE READ.
        #
        # This cache holds DECRYPTED CHUNKS keyed by (principal, collection,
        # cluster) with NO requester component, so serving a hit grants exactly what
        # handing over the cell key would. Reading it first meant an unauthorized
        # caller was served another principal's plaintext for the 60s TTL, with the
        # oracle never invoked — the grant coupling was real on a cache miss and
        # absent on a hit.
        #
        # This mirrors what the SSE arm already does (`sse/query.py`: derive the key
        # first, then let the loaders read their caches) and what this oracle's own
        # master-key cache does (`oracle.__init__`: authorize on every call, cached
        # or not). Both had it right; only this arm inverted it. `authorize()` runs
        # the identical check without paying for a derivation on every hit.
        self._oracle.authorize(principal_id, collection_id, request)

        cached = self._cache.get(principal_id, collection_id, cluster_id)
        if cached is not None:
            return cached, None

        blob = self._cells.get(principal_id, collection_id, cluster_id)
        if blob is None:
            return [], None

        aad = cell_mod.cell_aad(collection_id, cluster_id)
        try:
            key = self._oracle.derive_cell_key(
                principal_id, collection_id, cluster_id, request
            )
            chunks = cell_mod.unpack_cell(blob, key, collection_id=aad)
        except cell_mod.CellTampered:
            logger.warning(
                "Cell (%s, %s, %s) failed GCM auth — skipping in search",
                principal_id, collection_id, cluster_id,
            )
            return [], _CellOutcome.TAMPERED
        except cell_mod.CellMalformed:
            logger.warning(
                "Cell (%s, %s, %s) is malformed — skipping in search",
                principal_id, collection_id, cluster_id,
            )
            # Malformed ≠ wrong key: a truncated/garbage blob is a storage defect and
            # says nothing about key custody, so it does NOT count toward the
            # systemic-key ratio. Only GCM auth failure is evidence about the key.
            return [], _CellOutcome.MALFORMED

        self._cache.put(principal_id, collection_id, chunks, cluster_id)
        return chunks, _CellOutcome.OK

    def _score_chunks(
        self,
        query: np.ndarray,
        chunks: List[dict],
        principal_id: str,
        collection_id: str,
        best: dict[Tuple[str, int], MantleHit],
    ) -> None:
        """Cosine-score every chunk in a cell against ``query``; update ``best``.

        ``query`` is already L2-normalized. Embeddings are stacked and scored in a
        single matrix-vector product (BLAS) rather than a Python per-chunk loop —
        **behavior-identical** (same cosine, same best-score-wins dedup, same
        missing-field / dim-mismatch / zero-norm skips), ~10-100x faster on dense
        cells (MANTLE-SEARCH-SPEC Change A).
        """
        ids: List[Tuple[str, int]] = []
        vecs: List[np.ndarray] = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            artifact_id = chunk.get("artifact_id")
            chunk_id = chunk.get("chunk_id")
            if embedding is None or artifact_id is None or chunk_id is None:
                continue
            try:
                vec = np.asarray(embedding, dtype=np.float32)
            except (TypeError, ValueError):
                continue
            if vec.ndim != 1 or vec.size != query.size:
                # Mismatched dimensions — skip silently rather than fail the query.
                continue
            ids.append((artifact_id, chunk_id))
            vecs.append(vec)
        if not vecs:
            return
        M = np.stack(vecs)                          # [n, dim]
        norms = np.linalg.norm(M, axis=1)          # [n]
        keep = norms > 0                            # drop zero-norm vectors (parity)
        if not keep.any():
            return
        # query is pre-normalized, so cosine == (unit rows) @ query, one matmul.
        scores = (M[keep] / norms[keep, None]) @ query
        kept_ids = [ids[i] for i in np.nonzero(keep)[0]]
        for (artifact_id, chunk_id), s in zip(kept_ids, scores):
            score = float(s)
            key = (artifact_id, chunk_id)
            existing = best.get(key)
            if existing is None or score > existing.score:
                best[key] = MantleHit(
                    artifact_id=artifact_id,
                    chunk_id=chunk_id,
                    score=score,
                    principal_id=principal_id,
                    collection_id=collection_id,
                )

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def evict_cache(
        self,
        principal_id: Optional[str] = None,
        collection_id: Optional[str] = None,
    ) -> None:
        """Drop cached cells. With no args, clears the whole cache;
        with both args, evicts that one entry."""
        if principal_id is None and collection_id is None:
            self._cache.clear()
        elif principal_id and collection_id:
            self._cache.evict_context(principal_id, collection_id)
        else:
            raise ValueError(
                "evict_cache: pass either no args (clear all) or both principal_id and collection_id"
            )


__all__ = ["MantleHit", "MantleQueryEngine"]
