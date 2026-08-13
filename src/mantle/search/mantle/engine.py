"""MantleQueryEngine — encrypted IVF search over authorized cells.

The query-time inverse of :class:`MantleIndexer`:

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

Composes:

- :class:`OracleService` — derives cell keys; refuses keys for
  revoked grants
- :func:`cell.unpack_cell` — decrypts + deserializes
- :mod:`clustering` — routes the query to its nearest clusters
- :class:`CellStore` + :class:`CentroidStore` — opaque storage

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
from typing import AbstractSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import cell as cell_mod
from .oracle import KeyRequest, OracleService
from .stores import CellStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Where the result set stops
# ---------------------------------------------------------------------------
#
# `search/beacon/cut.py` is the reduced instrument: a Tracy-Widom signal-rank read of a
# query-relative multi-head screen, cut at the largest relative gap. It is parameter-free —
# no fixed k, no MAD multiple, no significance level, no keep-fraction — and it is the
# project's stated position on result cuts, which is why the arm takes it rather than a
# noise-floor z-score somebody chose.
#
# It applies HERE and to nothing else, and the reason is the shape of its input rather than a
# preference. `beacon.cut.select` reads `(item_embs, query_emb)`: it needs a vector per
# candidate. Only this arm has those. The blind-token narrowing holds a SET and a pair of
# integer counts per artifact — no vectors at all — so there is nothing there for a silhouette
# to read, and `router_accessor._by_coverage` consequently takes no cut: a coverage-ordered
# recall returns the whole narrowed set.
#
# The rank fusion this cut was once at risk of being pointed at is gone with BM25, and the
# hazard is worth keeping written down because it is what a future fusion would walk back into:
# an RRF score is `Σ 1/(k + rank)`, so the consecutive ratios of a single arm's list are
# `(k+r+1)/(k+r)` — monotonically decreasing, largest at the very first pair, always cutting
# after one item. Those gaps are manufactured by `k`, not measured from the data, so reading a
# silhouette off them measures the fusion constant. Beacon cuts a ranked list that has a
# spectrum.
CUT_BEACON = "beacon"
CUT_NONE = "none"
VALID_CUTS = (CUT_BEACON, CUT_NONE)


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
    TAMPERED = "tampered"    # GCM auth failed: evidence about the key
    MALFORMED = "malformed"  # unparseable blob: a storage defect, says nothing about the key

@dataclass(frozen=True)
class MantleHit:
    """One decrypted, scored hit from the MANTLE query path.

    ``score`` is cosine similarity in [-1, 1]; higher is closer. Nothing fuses it with
    anything: this arm is the only ranker, so the number the accessor puts on a
    ``SearchHit`` for a cosine-ordered recall IS this one.
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
        cut: Optional[str] = None,
    ) -> None:
        self._oracle = oracle
        self._cells = cell_store
        self._cache = _CellCache(ttl_seconds=cell_cache_ttl_s)
        self._nprobe = nprobe
        # How the result set STOPS. See `_beacon_cut`. `MANTLE_SEARCH_CUT=none` returns the
        # whole scored horizon and lets the caller's top_k be the only bound.
        import os as _os
        if cut is None:
            cut = (_os.getenv("MANTLE_SEARCH_CUT", CUT_BEACON) or CUT_BEACON).strip().lower()
        if cut not in VALID_CUTS:
            logger.error(
                "MANTLE_SEARCH_CUT=%r is not one of %s — using %r rather than guessing.",
                cut, ", ".join(VALID_CUTS), CUT_BEACON,
            )
            cut = CUT_BEACON
        self._cut = cut

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
        authorized_artifacts: Optional[AbstractSet[str]] = None,
    ) -> List[MantleHit]:
        """Run a vector search over the union of authorized cells.

        ``authorized_contexts`` is an iterable of ``(principal_id, collection_id)``
        tuples — the result of :class:`LightConeResolver`.resolve() filtered
        through the artifact ownership graph.

        ``authorized_artifacts`` is the artifact-granular set from that same
        resolve(). A cell is the unit of ENCRYPTION, not of authorization: holding
        the key to a cell means every chunk inside it decrypts, including chunks
        belonging to artifacts the requester was never granted. Scoring every
        decrypted chunk therefore turns a grant on one artifact into recall over
        its whole collection. When supplied, chunks outside this set are dropped
        before scoring — a meet over the cell-level cut, never a widening of it.
        ``None`` leaves the engine at cell granularity, which is the engine-level default for
        callers and tests; the production recall path
        (:meth:`~.sse.router_accessor.MantleSseSearchAccessor.search`) always passes a concrete
        set, so the fail-closed decision lives in one place rather than being re-derived here.

        ``request`` is the required :class:`~search.mantle.oracle.KeyRequest`
        identifying the principal on whose behalf the search runs. It is passed
        down to every cell-key derivation, so the oracle re-verifies the grant for
        each context rather than trusting that ``authorized_contexts`` was built
        honestly. That redundancy is the point: the caller computing the context
        list and the custodian issuing the keys check the same grants
        independently.

        For each context, fetch the routed cells, decrypt, run cosine ANN, and
        merge results. Deduplicates by ``(artifact_id, chunk_id)`` — when a
        chunk appears in multiple authorized contexts, the highest score wins.

        Returns AT MOST ``top_k`` hits sorted by descending score. ``top_k`` is the
        horizon; how many of it belong together is read off the spectrum by
        :meth:`_beacon_cut`, so a query with three good answers returns three rather than
        fifty with a tail of noise.

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

        # Empty-but-present is "the light cone authorized nothing" and must not be
        # read as "unspecified" — only `None` means that. Short-circuiting here also
        # avoids decrypting cells whose every chunk would be discarded.
        if authorized_artifacts is not None and not authorized_artifacts:
            return []

        # Route the query to its nearest-anchor clusters (canonical plan §5.1):
        # decrypt only those cells per authorized context, not the whole union.
        #
        # The AnchorSet is mandatory and PROVISIONED — nothing derives one, here or on first
        # use — so on an unprovisioned node this line raises `AnchorSetNotProvisioned` on every
        # query. There is one path and no flat fallback: without anchors this arm reads nothing,
        # and `unified` catches the raise so recall still answers from the lexical arm alone.
        from mantle.search.anchors.routing import route_query
        from mantle.search.anchors.store import require_live_anchorset

        clusters = route_query(require_live_anchorset(), q, nprobe=self._nprobe)

        # Best-score-wins dedup.
        best: dict[Tuple[str, int], MantleHit] = {}
        # The unit vector behind each surviving hit, kept only when the cut will read it.
        # A cell's decrypted chunks are already resident in `_CellCache` for the TTL window,
        # so this holds one float32 row per DEDUPED hit on top of plaintext already in memory.
        vectors: Optional[dict[Tuple[str, int], np.ndarray]] = (
            {} if self._cut == CUT_BEACON else None
        )

        # Systemic-fault bookkeeping — see `_load_cell` and the check below.
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
                    self._score_chunks(
                        q, chunks, principal_id, collection_id, best,
                        authorized_artifacts=authorized_artifacts,
                        vectors=vectors,
                    )

        #
        # One tampered cell must not fail a whole query, because then anyone who can
        # corrupt a single cell can deny the entire corpus. So the two conditions are
        # separated by the only signal that actually distinguishes them — the ratio,
        # not the event:
        #
        #   some cells decrypt, some don't  → per-cell corruption. Degrade.
        #   every cell we could read fails  → the key itself is wrong. Systemic; raise.
        #
        # A per-cell attacker cannot reach the second branch without already having
        # corrupted every cell in every authorized context, at which point the
        # corpus is gone anyway and saying so is the correct behaviour.
        #
        if attempted and failed_auth == attempted:
            raise SystemicKeyFailure(
                f"all {attempted} readable cell(s) across {len(contexts)} authorized "
                f"context(s) failed GCM authentication — this is a key fault "
                f"(wrong/rotated/destroyed master key), not an empty corpus; "
                f"refusing to report 'no results'"
            )

        # Sort by score descending.
        hits = sorted(best.values(), key=lambda h: h.score, reverse=True)

        # `top_k` is the horizon, not the answer: it bounds what the caller is willing to
        # look at, and the cut decides how much of that actually belongs together.
        hits = hits[:top_k]
        if self._cut == CUT_BEACON and vectors is not None:
            hits = self._beacon_cut(q, hits, vectors)
        return hits

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

        A cell that fails GCM authentication degrades to a miss here — the DoS argument
        for that is sound — and the outcome is reported to the caller, which can see the
        difference between one bad cell and a wholesale key fault.
        """
        #
        # This cache holds decrypted chunks keyed by (principal, collection,
        # cluster) with no requester component, so serving a hit grants exactly what
        # handing over the cell key would. Authorization therefore runs before the
        # cache is read: checking the cache first would serve an unauthorized caller
        # another principal's plaintext for the TTL window, since the oracle would
        # never be invoked on a hit.
        #
        # This mirrors what the SSE arm does (`sse/query.py`: derive the key first,
        # then let the loaders read their caches) and what this oracle's own
        # master-key cache does (`oracle.__init__`: authorize on every call, cached
        # or not) — the same pattern applied consistently. `authorize()` runs the
        # identical check without paying for a derivation on every hit.
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
            # says nothing about key custody, so it does not count toward the
            # systemic-key ratio. Only GCM auth failure is evidence about the key.
            return [], _CellOutcome.MALFORMED

        self._cache.put(principal_id, collection_id, chunks, cluster_id)
        return chunks, _CellOutcome.OK

    def _beacon_cut(
        self,
        query: np.ndarray,
        hits: List[MantleHit],
        vectors: dict[Tuple[str, int], np.ndarray],
    ) -> List[MantleHit]:
        """Where this result set stops, read off its own spectrum. See the module header.

        ``hits`` is already the bounded, score-ordered horizon, which is the pool
        :func:`beacon.cut.select` documents itself as wanting: it reads the structure of the
        set it is handed, so handing it the whole corpus asks a different question. Order is
        preserved — beacon decides membership, cosine decides rank.

        Degrades to the uncut horizon on anything unexpected. A cut is a refinement of an
        answer that already exists; failing to take one must not cost the answer.
        """
        if len(hits) <= 2:
            # Two candidates have no spectrum to read — `select` would keep both anyway, and
            # a cut that never had a reading must not look like one that did.
            return hits
        try:
            from mantle.search.beacon.cut import select

            pool = np.stack([vectors[(h.artifact_id, h.chunk_id)] for h in hits])
            keep = select(pool, query)
            cut = [hits[i] for i in sorted(int(i) for i in keep)]
        except Exception:
            logger.warning(
                "MANTLE: the beacon cut could not be taken over %d candidates; returning the "
                "uncut horizon", len(hits), exc_info=True,
            )
            return hits
        return cut or hits

    def _score_chunks(
        self,
        query: np.ndarray,
        chunks: List[dict],
        principal_id: str,
        collection_id: str,
        best: dict[Tuple[str, int], MantleHit],
        *,
        authorized_artifacts: Optional[AbstractSet[str]] = None,
        vectors: Optional[dict[Tuple[str, int], np.ndarray]] = None,
    ) -> None:
        """Cosine-score every AUTHORIZED chunk in a cell against ``query``; update ``best``.

        ``query`` is already L2-normalized. Embeddings are stacked and scored in a
        single matrix-vector product (BLAS) rather than a Python per-chunk loop —
        **behavior-identical** (same cosine, same best-score-wins dedup, same
        missing-field / dim-mismatch / zero-norm skips), ~10-100x faster on dense
        cells (MANTLE-SEARCH-SPEC Change A).

        ``authorized_artifacts`` narrows "every chunk in the cell" to "every chunk
        of an artifact this principal may read". The cell's contents are wider than
        the grant whenever the grant is artifact-scoped, so this is the point where
        decryptability stops standing in for authorization.

        ``vectors``, when supplied, collects the unit vector behind each surviving hit for
        :meth:`_beacon_cut`. It is filled here rather than re-derived later because these are
        the rows already normalized for the cosine — recomputing them would mean holding the
        decrypted chunks past the point the scorer needs them.
        """
        ids: List[Tuple[str, int]] = []
        vecs: List[np.ndarray] = []
        for chunk in chunks:
            embedding = chunk.get("embedding")
            artifact_id = chunk.get("artifact_id")
            chunk_id = chunk.get("chunk_id")
            if embedding is None or artifact_id is None or chunk_id is None:
                continue
            if authorized_artifacts is not None and artifact_id not in authorized_artifacts:
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
        unit = M[keep] / norms[keep, None]
        scores = unit @ query
        kept_ids = [ids[i] for i in np.nonzero(keep)[0]]
        for row, ((artifact_id, chunk_id), s) in enumerate(zip(kept_ids, scores)):
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
                if vectors is not None:
                    vectors[key] = unit[row]

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
