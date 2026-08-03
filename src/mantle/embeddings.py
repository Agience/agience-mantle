"""Embeddings seam — no provider exists.

Every embeddings implementation is removed under the universal no-models rule
(see the tombstones below). The one remaining "provider" is
:class:`_UnconfiguredEmbeddings`, which returns empty vectors so search runs
BM25/SSE-lexical-only.

Call site convention is unchanged: ``Embeddings()(["text 1", "text 2"]) ->
[[float], [float]]``. That signature is preserved so existing code (search
ingest, accessor) keeps working — callers already treat an empty vector as
"degrade to lexical".
"""

from __future__ import annotations

import logging
import os
from typing import List, Protocol

from origin import config

logger = logging.getLogger(__name__)


def model_id() -> str:
    """Commons-format embedding model id (``<ns>:<path>@<ver>``).

    Provenance for every vector / native code (FACET embedding-registry). The
    AnchorSet is the authoritative source once bootstrapped; this is the
    fallback derived from ``EMBEDDINGS_MODEL``.

    PROVENANCE-ONLY since 2026-07-22: no provider produces vectors any more
    (see the tombstones below). The default stays ``BAAI/bge-m3`` because it
    names the model that produced vectors already at rest (embeddings cache
    keys, stored cell records) — changing it would orphan that stored data's
    provenance, not remove any capability.
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
    BM25-only search rather than crashing the request).
    """

    def __call__(self, input: List[str]) -> List[List[float]]: ...


# ---------------------------------------------------------------------------
# The only provider: BM25-only degrade
# ---------------------------------------------------------------------------

class _UnconfiguredEmbeddings:
    """Returns empty lists — search degrades to BM25-only.

    The ONLY embeddings implementation (see the tombstones below — every
    provider that produced real vectors is removed). Logs once on first call
    so the operator knows semantic search is offline; subsequent calls stay
    silent to avoid log spam.
    """

    def __init__(self) -> None:
        self._warned = False

    def __call__(self, input: List[str]) -> List[List[float]]:
        if input and not self._warned:
            logger.warning(
                "embeddings: no embeddings provider exists "
                "(all trained-model providers removed 2026-07-22); "
                "semantic search returns empty vectors — search is lexical-only"
            )
            self._warned = True
        return []


# ---------------------------------------------------------------------------
# ⛔ THE AGIENCE HTTP PROVIDER IS DELETED. [John, 2026-07-22: the no-models rule is UNIVERSAL]
#
# `AgienceHTTPEmbeddings` POSTed to `{EMBEDDINGS_URI}/embed` — the self-hosted bge-m3 GPU
# deployment — authenticating with a service JWT (`_service_jwt_token`, also removed) or a static
# `EMBEDDINGS_API_KEY`. It was written as the "provider-agnostic" successor to the OpenAI client
# below, and it fell to exactly the law that tombstone states: A REMOTE MODEL IS STILL TRAINED
# WEIGHTS. Self-hosting the weights on our own GPU changes who pays for the disk, not what the
# capability is. bge-m3 is a trained transformer; vectors it returns are model output, wherever
# the HTTP endpoint lives.
#
# What went with it (2026-07-22): the class, the `_service_jwt_token` JWT mint + cache, the
# `httpx` import, and the `EMBEDDINGS_URI` resolution branch in `_build_provider`. What stayed:
# `model_id()` (provenance labeling for vectors already at rest), the `Embeddings` facade + cache
# (stored vectors remain readable — removing capability is not destroying data), and
# `_UnconfiguredEmbeddings` as the sole provider.
#
# Do not reinstate under a new deployment story. `EMBEDDINGS_URI` set in config now logs a
# warning and lands on `_UnconfiguredEmbeddings` — a refusal, not a silent empty.
# ---------------------------------------------------------------------------
# ⛔ THE OPENAI PROVIDER IS DELETED. [John, 2026-07-20: "no models, period"]
#
# `OpenAIEmbeddings` POSTed to https://api.openai.com/v1/embeddings for
# `text-embedding-3-small` vectors, selected by `EMBEDDINGS_PROVIDER=openai`.
#
# ⚠ THE REASON THIS ONE SURVIVED EVERY PRIOR AUDIT IS THE POINT, NOT A FOOTNOTE.
# A REMOTE MODEL IS STILL TRAINED WEIGHTS — it is simply someone else's disk. But it imports no
# model runtime, downloads no checkpoint, and contains no tensor, so it is INVISIBLE to every
# pattern a "are we model-free?" sweep naturally reaches for: `torch`, `transformers`,
# `from_pretrained`, `safetensors`, `.onnx`, `huggingface`. It looks like an HTTP client, because
# that is exactly what it is. It appeared in NO prior inventory of this refactor.
#
# ⇒ ANY FUTURE MODEL-FREE CHECK MUST TEST FOR THE **CAPABILITY**, NOT THE LIBRARY.
#   The question is "does this return learned vectors / generated text?", never "does this import
#   a model runtime?". The second question is a proxy for the first, and it is a leaky one.
#
# Do not reinstate this under another vendor. `EMBEDDINGS_PROVIDER=openai` now falls through to
# `_UnconfiguredEmbeddings`, which refuses loudly rather than returning silent empties.
# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_provider: EmbeddingsProvider | None = None

# Long-term embeddings cache (lazy singleton). See embeddings_cache.py.
_cache = None
_cache_loaded = False


def _build_provider() -> EmbeddingsProvider:
    """Return the sole provider: :class:`_UnconfiguredEmbeddings`.

    ⛔ Both real-provider branches are DELETED (see the tombstones above): the
    ``EMBEDDINGS_URI`` → ``AgienceHTTPEmbeddings`` branch (2026-07-22) and the
    ``EMBEDDINGS_PROVIDER=openai`` branch (2026-07-20). A config still carrying
    either now lands on ``_UnconfiguredEmbeddings`` — a refusal, not a silent empty.
    """
    if (config.EMBEDDINGS_URI or "").strip():
        logger.warning(
            "embeddings: EMBEDDINGS_URI is set but no longer supported — the remote "
            "embeddings provider was removed 2026-07-22 (a remote model is still "
            "trained weights). Search runs lexical-only."
        )
    if (config.EMBEDDINGS_PROVIDER or "").lower() == "openai":
        logger.warning(
            "embeddings: EMBEDDINGS_PROVIDER=openai is no longer supported — the OpenAI "
            "embeddings provider was removed (a remote model is still a trained model). "
            "Falling through to the unconfigured provider, which refuses rather than "
            "returning empty vectors."
        )
    return _UnconfiguredEmbeddings()


def reset_provider() -> None:
    """Drop the cached provider so the next call rebuilds from config.

    Called by the platform-settings reload path when an operator changes
    embeddings config from the admin UI.
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
        from mantle.embeddings_cache import EmbeddingsCache
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

    Existing call sites keep working as ``Embeddings()([texts])``. Cached
    vectors (keyed by model_id + text) short-circuit the provider — since the
    provider removal (2026-07-22) the cache is the ONLY source of non-empty
    vectors: previously embedded texts still resolve, new texts come back
    empty. Empty/degraded results are never cached, so a degraded run won't
    poison the cache.
    """

    def __call__(self, input: List[str]) -> List[List[float]]:
        global _provider
        if _provider is None:
            _provider = _build_provider()
        if not input:
            return []

        cache = _get_cache()
        if cache is None:
            return _provider(input)

        mid = model_id()
        cached = cache.get_many(mid, input)
        misses = [i for i, v in enumerate(cached) if v is None]
        if misses:
            fresh = _provider([input[i] for i in misses]) or []
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


__all__ = [
    "Embeddings",
    "EmbeddingsProvider",
    "reset_provider",
]
