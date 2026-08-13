"""Embeddings read seam — vectors that already exist, never vectors Mantle makes.

Mantle does not embed on its own behalf, so there is no provider to configure in the
usual direction. Two sources of a real vector exist, and neither is a model:

* **The writer.** :class:`WriterSuppliedEmbeddings` carries a vector handed to Mantle
  on the write that produced the text — the seam runs inward instead of outward. This
  is the semantic arm's ingress (``api/vectors.py`` validates the shape).
* **The long-term cache** (``embeddings_cache.py``) — a text already at rest resolves;
  a new text comes back empty.

With neither, :class:`_UnconfiguredEmbeddings` returns empty vectors and search
degrades to lexical for that text.

Call as ``Embeddings()(["text 1", "text 2"]) -> [[float], [float]]``. Search ingest and
the accessor call this signature directly and treat an empty vector as "degrade to
lexical", which is why inverting the seam changed no call site: the same call now
resolves against what a writer supplied rather than against nothing.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Protocol, Sequence

from mantle import config

logger = logging.getLogger(__name__)


def model_id() -> str:
    """Commons-format embedding model id (``<ns>:<path>@<ver>``).

    Provenance for every vector / native code (FACET embedding-registry). The
    AnchorSet is the authoritative source once provisioned — and a node nobody has
    provisioned has none, so on a fresh install this fallback, derived from
    ``EMBEDDINGS_MODEL``, is the only answer there is.

    No provider currently produces vectors under this id; it labels vectors
    already at rest (embeddings cache keys, stored cell records). The
    default stays ``BAAI/bge-m3`` because that is the model that produced
    that stored data — changing it would orphan the stored data's
    provenance.
    """
    raw = (os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3").strip()
    ver = (os.getenv("EMBEDDINGS_MODEL_VERSION", "1.0") or "1.0").strip()
    if any(raw.startswith(p) for p in ("hf:", "openai:", "custom:", "facet:")):
        return raw if "@" in raw else f"{raw}@{ver}"
    return f"hf:{raw}@{ver}"


class EmbeddingsProvider(Protocol):
    """Contract every embeddings backend conforms to.

    Implementations should be deterministic for the same input + model
    version, return vectors in declared `EMBEDDINGS_DIM` dimensions, and
    surface failures by returning an empty list (callers fall back to
    a lexically narrowed recall rather than crashing the request).
    """

    def __call__(self, input: List[str]) -> List[List[float]]: ...


# ---------------------------------------------------------------------------
# The ingress provider: a vector the writer already had
# ---------------------------------------------------------------------------

class WriterSuppliedEmbeddings:
    """The seam inverted — the vector arrives with the write instead of being fetched.

    Still not a model. This class holds numbers a writer computed elsewhere and hands
    them back in the shape the ingest path already expects, so nothing downstream has
    to learn that a vector can come from the request. The rule is unchanged: Mantle
    receives vectors, it never produces them.

    One supplied vector describes one write. The first text gets it; any further text
    gets an empty vector, because splitting one vector across several chunks would
    invent a claim the writer did not make. The ingest arm collapses a vector-bearing
    write to a single chunk for exactly that reason, so the second case is a guard
    rather than a path.

    ``space_id`` travels with the numbers and is recorded as the chunk's provenance.
    Two vectors are only comparable within one space; a vector stored without its
    space name is a vector nothing can safely be compared to later.
    """

    def __init__(self, values: Sequence[float], space_id: str) -> None:
        self._values = [float(v) for v in values]
        self.space_id = space_id

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        return [list(self._values)] + [[] for _ in input[1:]]


# ---------------------------------------------------------------------------
# The fallback provider: lexical degrade
# ---------------------------------------------------------------------------

class _UnconfiguredEmbeddings:
    """Returns empty lists — search degrades to lexical.

    The answer when no writer supplied a vector and the cache has never seen this
    text. There is deliberately no seam for a *model* here. A remote embedding
    endpoint is still trained weights; hosting it elsewhere changes who pays for the
    disk, not what the capability is. The question is "does this return learned
    vectors?", not "does this import a model runtime?" — the second is a leaky proxy
    for the first. Accepting a vector a writer already computed answers neither
    question with a yes.

    Logs once on first call so the operator knows the vector arm has nothing to
    rank with; subsequent calls stay silent to avoid log spam.
    """

    def __init__(self) -> None:
        self._warned = False

    def __call__(self, input: List[str]) -> List[List[float]]:
        if input and not self._warned:
            logger.warning(
                "embeddings: Mantle produces no vectors of its own; text with no "
                "writer-supplied vector and no cache entry returns an empty vector — "
                "search is lexical-only for it"
            )
            self._warned = True
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider: EmbeddingsProvider | None = None

# Long-term embeddings cache (lazy singleton). See embeddings_cache.py.
_cache = None
_cache_loaded = False


def _build_provider() -> EmbeddingsProvider:
    """Return the sole provider: :class:`_UnconfiguredEmbeddings`.

    A factory with one product is still the right shape here: it is the single
    place a provider is chosen, so the "no provider" answer is stated once
    instead of assumed at every call site.
    """
    return _UnconfiguredEmbeddings()


def reset_provider() -> None:
    """Drop the cached provider and the cache singleton so the next call rebuilds.

    The reset seam for anything that changes where cached vectors live.
    """
    global _provider, _cache, _cache_loaded
    _provider = None
    _cache = None
    _cache_loaded = False


def _get_cache():
    """Lazy singleton embeddings cache. ``None`` when disabled/unavailable."""
    global _cache, _cache_loaded
    if _cache_loaded:
        return _cache
    if os.getenv("EMBEDDINGS_CACHE", "1").strip().lower() in {"0", "false", "no", "off"}:
        _cache, _cache_loaded = None, True
        return None
    try:
        from mantle.search.embeddings_cache import EmbeddingsCache
        path = os.getenv("EMBEDDINGS_CACHE_PATH") or str(
            config.BASE_DIR / ".data" / "mantle" / "mantle.embeddings_cache.sqlite"
        )
        _cache = EmbeddingsCache(path)
        logger.info("Embeddings cache enabled: %s (%d entries)", path, _cache.count())
    except Exception:
        logger.warning("Embeddings cache unavailable; continuing without it", exc_info=True)
        _cache = None
    _cache_loaded = True
    return _cache


class Embeddings:
    """Provider facade with a transparent long-term cache.

    Call as ``Embeddings()([texts])``. Cached vectors (keyed by model_id + text)
    short-circuit the provider: a text already in the cache resolves, a new text comes
    back empty. Empty/degraded results are never cached, so a degraded run cannot
    poison the cache.

    ``Embeddings(provider=WriterSuppliedEmbeddings(...))`` is the ingress form — the
    ingest arm builds one per vector-bearing write. It runs with the cache OFF, and
    deliberately: the cache is keyed by :func:`model_id`, which names the space Mantle
    would have labelled a vector with, and a writer's vector lives in the space the
    writer named. Storing it under the other name would make two incomparable vectors
    look like siblings — the exact confusion ``space_id`` exists to prevent.
    """

    def __init__(
        self,
        provider: Optional[EmbeddingsProvider] = None,
        *,
        cache: Optional[bool] = None,
    ) -> None:
        self._override = provider
        #: Default: cache on for the shared provider, off for an injected one.
        self._use_cache = (provider is None) if cache is None else bool(cache)

    def __call__(self, input: List[str]) -> List[List[float]]:
        global _provider
        provider = self._override
        if provider is None:
            if _provider is None:
                _provider = _build_provider()
            provider = _provider
        if not input:
            return []

        cache = _get_cache() if self._use_cache else None
        if cache is None:
            return _pad(provider(input), len(input))

        mid = model_id()
        cached = cache.get_many(mid, input)
        misses = [i for i, v in enumerate(cached) if v is None]
        if misses:
            fresh = provider([input[i] for i in misses]) or []
            store_texts: List[str] = []
            store_vecs: List[List[float]] = []
            for j, i in enumerate(misses):
                v = fresh[j] if j < len(fresh) else None
                cached[i] = v
                if v:
                    store_texts.append(input[i])
                    store_vecs.append(v)
            if store_vecs:
                try:
                    cache.put_many(mid, store_texts, store_vecs)
                except Exception:
                    logger.debug("embeddings cache write failed", exc_info=True)
        return [v if v else [] for v in cached]


def _pad(vectors: List[List[float]], n: int) -> List[List[float]]:
    """Align a provider's answer 1:1 with its input, padding short returns with ``[]``.

    A provider is allowed to return fewer vectors than texts (``_UnconfiguredEmbeddings``
    returns none at all). Every caller indexes the result positionally against its own
    chunk list, so the alignment is enforced here once instead of at each of them.
    """
    out = [list(v) if v else [] for v in (vectors or [])][:n]
    out.extend([] for _ in range(n - len(out)))
    return out


__all__ = [
    "Embeddings",
    "EmbeddingsProvider",
    "WriterSuppliedEmbeddings",
    "model_id",
    "reset_provider",
]
