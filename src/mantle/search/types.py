"""Search request / response shapes.

The `SearchQuery` / `SearchHit` / `SearchResult` dataclasses are the
public contract between the artifacts router and the search engine
wired in — the blind-token index NARROWS and the encrypted vector index RANKS what
survives, see `mantle.search.mantle.sse.router_accessor`. These dataclasses are
engine-independent, so the router does not need to know engine internals.

One ranker means the response has no "which arm answered" bit to report. What it reports
instead is `SearchResult.ordering`, which is a different question with a genuinely varying
answer: what actually put these hits in this order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .query_parser import ParsedQuery

#: A cosine ordered these hits — the semantic arm ran and ranked the narrowed set.
ORDER_SEMANTIC = "semantic"
#: Nothing could rank them, so they are most-recently-updated first. See
#: :attr:`SearchResult.ordering`.
ORDER_RECENCY = "recency"
#: HOW MUCH OF THE QUERY EACH HIT MATCHED ordered them — the count of distinct query stems the
#: narrowing found in each artifact, most first, ties broken by recency.
#:
#: Named for the measurement rather than for a quality. `"lexical"` would name an arm and imply
#: it scored something; `"relevance"` would claim the number means the hit is better. What is
#: true is narrower and is the whole of it: the narrowing looked up N stems and this artifact
#: carried k of them. That is coverage of the query, it is an integer, and it is not a metric on
#: the data — which is the distinction the removal of BM25 exists to hold.
ORDER_COVERAGE = "coverage"

#: What ordered a result. Three values, because there are three things that can: a cosine, the
#: query's own coverage, or the clock. There is no value meaning "we would rather not say".
Ordering = Literal["semantic", "coverage", "recency"]


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

    # Authorization scope, NOT the caller's `field:value` filter — two different narrowings
    # that the word "filter" would blur. These are container/grant identities.
    collection_ids: Optional[List[str]] = None
    grant_keys: Optional[List[str]] = None

    # No `filters` field. A `field:value` filter travels INSIDE `query_text`, which is where
    # the caller wrote it, and `sse.router_accessor.plan_recall` is the one place it is lifted
    # out and compiled. A parallel structured field here would be a second way to say the same
    # thing, and the two could disagree — which is the incoherence `state` already illustrates
    # (see `field_filters.REFUSED_FIELDS`).

    # Pagination
    from_: int = 0
    size: int = 20

    # No `use_hybrid`: there is nothing for it to switch. The lexical arm NARROWS on every
    # recall and the semantic arm RANKS what survives, so neither is optional and neither is
    # selectable. What varies is whether a query vector exists for the ranker to use, and the
    # field that carries the vector says that already. `SearchResult.ordering` reports what
    # actually ordered the result.

    #: Which of the two orderings the caller wants.
    #:
    #: ``"recency"`` asks for most-recently-updated first and gets it, vector or no vector,
    #: coverage or no coverage — it is the one request that turns every ranking off.
    #: ``"relevance"`` asks for the best ordering this recall can produce — a cosine when a
    #: query vector reached the ranker, otherwise the query's own coverage, and recency when
    #: there was neither. It is a REQUEST, so it cannot promise an outcome;
    #: ``SearchResult.ordering`` is the outcome, and the caller reads which it got there.
    sort: Optional[Literal["relevance", "recency"]] = "relevance"

    # UI features
    highlight: bool = False

    # The `@name:value` namespace does not travel here. It is a LEXER feature of
    # `query_parser` — those tokens are lifted out of the term stream and recorded on
    # `ParsedQuery.controls`, which rides back out on `SearchResult.parsed_query`. A copy
    # of them on the REQUEST would be a field the router fills and the accessor reads for
    # a decision no retrieval step takes, which is how `@hybrid:on` came to read as a
    # working switch. The request-side controls are the fields above, each of which names
    # the thing it carries.


@dataclass
class SearchHit:
    """Single search result hit."""

    doc_id: str

    #: WHAT PUT THIS HIT WHERE IT IS, as a number — and ``None`` when nothing did.
    #:
    #: Three cases, told apart by :attr:`SearchResult.ordering` and never by the number itself:
    #:
    #: - ``"semantic"`` — the cosine that ranked it. A float in ``[-1, 1]``.
    #: - ``"coverage"`` — HOW MANY DISTINCT QUERY STEMS this artifact matched. An INTEGER count,
    #:   carried in a float-typed field because the field is one field. It is a count and not a
    #:   score: it is not normalised, not weighted by document frequency or term frequency, and
    #:   not comparable between two different queries — a 2 on a five-stem query and a 2 on a
    #:   two-stem query say different things. What it is comparable across is the hits of ONE
    #:   response, which is exactly what ordering them needs. A normalised float here would
    #:   invite a client to threshold on a number that measures nothing.
    #: - ``"recency"`` — ``None``. Those hits are ordered by the clock, so no number measured
    #:   them against the query at all, and 0.0 or a rank would be a value a client could
    #:   threshold or re-sort by that would mean nothing.
    score: Optional[float]
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

    #: WHAT ORDERED THESE HITS. Not what was asked for — what happened.
    #:
    #: `"semantic"` means a cosine did: a query vector reached the ranker, it ranked the
    #: narrowed set, and the beacon cut decided where that ranking stops. `SearchHit.score`
    #: carries the cosine.
    #:
    #: `"coverage"` means the QUERY'S OWN COVERAGE did: no vector reached the ranker, so the
    #: hits are ordered by how much of the query each one matched — the count of distinct query
    #: stems the narrowing found in it, most first — with a quoted phrase's bigram count beneath
    #: that and recency beneath both. `SearchHit.score` carries the stem count as an integer.
    #: This is not a relevance model: there is no IDF, no term frequency and no length
    #: normalisation anywhere on the path, because the count is a by-product of the narrowing
    #: rather than a measurement performed on the corpus. A ONE-STEM QUERY IS THEREFORE EXACTLY
    #: RECENCY ORDER, since every surviving hit has the same count of 1 and the tiebreak is the
    #: whole ordering.
    #:
    #: `"recency"` means nothing could rank them, so the hits are most-recently-updated first
    #: and every `SearchHit.score` is `None`. Two situations arrive here and the field does not
    #: separate them: the caller asked for `sort="recency"`, or the recall had no query terms to
    #: cover at all — an embedding-only request on a node that could not rank it, which is what
    #: a node with no provisioned AnchorSet does with every vector. Node-level readiness is a
    #: node fact, constant until an operator changes it, and it is answered on demand rather
    #: than restated in every response body — see `search/mantle/anchors/store.py` and
    #: `python -m mantle.system.manage_anchors --action inspect`.
    #:
    #: There is no value for "empty result". A body with no hits reports whichever ordering the
    #: recall would have used; an empty list is trivially in every order, and the field
    #: describes the hits, of which there are none.
    #:
    #: A caller MUST be able to tell these apart, which is why this is a field and not an
    #: inference from `score`: a cosine, a count and a null are three different kinds of number
    #: and only this field says which one arrived.
    ordering: Ordering

    total_is_capped: bool = False

    #: The ``field:value`` filters that actually NARROWED this result, in canonical spelling.
    #:
    #: Distinct from `parsed_query`, which is the whole parse and still carries the inert
    #: `@name:value` controls. This list is retrieval's own account of itself: every entry
    #: was compiled into the predicate that narrowed the authorized artifact set before either
    #: arm ran, and a filter that could not be compiled never reaches here because it fails the
    #: request instead. So "parsed" and "applied" cannot drift apart silently — the only way a
    #: filter leaves the parse and does not appear here is a 400.
    applied_filters: List[str] = field(default_factory=list)

    # Facets (optional)
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = None


__all__ = [
    "ORDER_COVERAGE",
    "ORDER_RECENCY",
    "ORDER_SEMANTIC",
    "Ordering",
    "SearchQuery",
    "SearchHit",
    "SearchResult",
]
