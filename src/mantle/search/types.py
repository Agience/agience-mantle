"""Search request / response shapes (post lexical-backend retirement, Step 2.6.9 part 2).

The `SearchQuery` / `SearchHit` / `SearchResult` dataclasses are the
public contract between the artifacts router and whatever search engine
is wired in. After the lexical-backend retirement, the engine is MANTLE-SSE
(encrypted lexical) + MANTLE vector via RRF — see
`mantle.search.mantle.sse.router_accessor`. These dataclasses are
engine-independent so the router code is unchanged.

Originally lived in `search.accessor.search_accessor` alongside the
legacy-lexical-specific :class:`SearchAccessor` class. Extracted here when
that module went away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .query_parser import ParsedQuery


@dataclass
class SearchQuery:
    """Unified search query parameters."""

    query_text: str
    user_id: str

    # Raw query vector. When set, the accessor uses it directly for kNN and skips
    # the text→embedding step (query_text may be "" — pure "embedding activation").
    query_embedding: Optional[List[float]] = None

    # Explicit container scope (optional — restricts search to these collection/
    # workspace IDs). Set only when the caller provides body.scope or when
    # the principal is an API-key with narrower resource access than the user.
    # The accessor runs the full light-cone when this is None.
    scope: Optional[List[str]] = None

    # Scope filters (optional — for filtering results)
    collection_ids: Optional[List[str]] = None
    grant_keys: Optional[List[str]] = None

    # Pagination
    from_: int = 0
    size: int = 20

    # Hybrid flag — historically toggled BM25 + kNN fusion; the SSE
    # accessor folds both arms unconditionally and reports the result via
    # `SearchResult.used_hybrid`. Kept on the query for back-compat with
    # existing callers; ignored.
    use_hybrid: Optional[bool] = None

    # Sorting
    sort: Optional[Literal["relevance", "recency"]] = "relevance"

    # UI features
    highlight: bool = False
    # ⛔ `aperture: float = 0.75` REMOVED 2026-07-30. Declared "0.0 strict → 1.0 permissive" and
    # NEVER READ — no code path in mantle consumed it. It was written into SearchQuery at four
    # call sites with three different values (UI 0.5, sage/retrieval 0.7, mantle 0.75), exposed
    # to HTTP callers as a tunable, and surfaced in the UI as "Precise / Focused / Balanced /
    # Wide". A parameter that is triple-valued AND inert was never even tested. Relevance width
    # is derived from the score series (beam.resolution.signal_end), not dialled.

    # Control parameters from @ namespace
    controls: Optional[Dict[str, str]] = None


@dataclass
class SearchHit:
    """Single search result hit."""

    doc_id: str
    score: float
    root_id: str
    version_id: str

    # Content
    title: str
    description: str
    content: str
    tags: List[str]
    metadata: Dict[str, Any]

    # Context fields
    collection_id: Optional[str] = None
    principal_id: Optional[str] = None
    state: Optional[str] = None
    is_head: Optional[bool] = None

    # Highlighting
    highlights: Optional[Dict[str, List[str]]] = None


@dataclass
class SearchResult:
    """Search result with hits, facets, and metadata."""

    hits: List[SearchHit]
    total: int

    # Query metadata
    parsed_query: ParsedQuery
    corrections: List[str]
    used_hybrid: bool

    # ⚠ Was `total` a match count? No — it is the number of CANDIDATES the fused arms returned,
    # which the accessor has already capped at `top_k`. This flag says when that cap was hit, so
    # a caller rendering "N results" or a page count can tell a real total from a truncated one.
    # Defaulted, so existing constructors are unaffected.
    total_is_capped: bool = False

    # Facets (optional)
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = None


__all__ = ["SearchQuery", "SearchHit", "SearchResult"]
