"""MantleSseSearchAccessor — SearchResult-shaped adapter. Lexical narrows, semantic ranks.

The two encrypted indexes answer two different questions, and this accessor asks each the one
it is good at. The blind-token postings answer MEMBERSHIP — which artifacts carry these stems —
which is a set, and a set composes with the light cone by intersection. The cells answer
PROXIMITY — how close is each of these to the query — which is a ranking, and a ranking is what
a result page is. Neither is asked for the other's answer, so there is nothing to fuse and no
rank-fusion constant standing between a score and its meaning.

When no vector reaches the ranker, the ORDER comes from the narrowing itself: how much of the
query each artifact matched, counted while narrowing. That is a third ordering
(``ORDER_COVERAGE``) rather than a second ranker — nothing is measured about the corpus, and
the count exists whether or not anything reads it. See :meth:`_by_coverage`.

That count orders the survivors but says nothing about what the query is ABOUT, and on a node
where a host has bound an ontology there is something that does: :mod:`mantle.search.ranking`
measures how far each survivor's own position reaches toward the need's. When it can run it
re-orders the coverage set and cuts it (``ORDER_REACH``); when it cannot — no ontology, no
coordinate, nothing reachable — the coverage order stands and the response says so. It is the
same two-tier shape as the beacon and the aperture: the thin arm is not a degraded one.

Adapter responsibilities:

1. Plan the query (:func:`plan_recall`): parse it, compile its ``field:value`` filters into a
   doc predicate, and reduce the string retrieval sees to the terms only. `parsed_query`,
   `applied_filters` and `corrections` on the :class:`SearchResult` are what the router echoes
   back, and `applied_filters` is a list of what narrowed the recall rather than of what merely
   parsed — a filter this path cannot apply raises instead of reaching the echo.
2. Resolve the light cone via :func:`~mantle.search.mantle.lightcone.resolve_authorized_scope`,
   with BOTH narrowings riding INTO the resolve: the compiled field filter as a doc predicate,
   and the query's terms as a blind-token lookup
   (:meth:`~.narrowing.TokenNarrower.lookup_for`). Both are membership questions over artifact
   ids, which is what the light cone is too, so both compose there as a MEET and can only ever
   remove ids. That is the security property, and it is structural rather than checked: a token
   or filter naming content the requester cannot read lands in the same empty set as one
   matching nothing, with no observable between them.
   The resolve keeps both granularities — the ``(principal, collection)`` contexts that decide
   which encrypted indexes may be opened, and the artifact ids the principal may actually read.
   Discarding the second turns an artifact-scoped grant into recall over its whole collection.
3. Supply the ranker's query: a caller-provided ``query_embedding`` verbatim when the request
   carries one, otherwise whatever :class:`Embeddings` resolves for the query text. Neither an
   absent vector nor an embedding error fails the search — see 4.
4. Rank the survivors by cosine (:class:`~..engine.MantleQueryEngine`, which takes the beacon
   cut over its own spectrum), or, when no vector reached it, by how much of the query each one
   matched, or — when the caller asked for it, or when there was no query text to cover —
   most-recently-updated first. `SearchResult.ordering` says which happened and
   `SearchHit.score` carries the cosine, the stem count, or `None` respectively. A text query
   with no vector is an ordinary request, not an error: a caller that cannot embed still
   narrowed to a real set, and the narrowing already knows which members of it matched most of
   what was asked.
5. Hydrate each surviving artifact into a :class:`SearchHit` by reading its metadata from the
   lattice (since neither index stores plaintext text).

See ``.dev/features/mantle-sse-lexical-index.md`` § Query Flow.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence

from mantle.search.embeddings import Embeddings
from mantle.search.field_filters import QueryFilterError
from mantle.search.field_filters import describe as describe_filters
from mantle.search.field_filters import _title as _title_of
from mantle.search.field_filters import parse_context
from mantle.search.types import (
    ORDER_COVERAGE,
    ORDER_REACH,
    ORDER_RECENCY,
    ORDER_SEMANTIC,
)
from mantle.services.acting_principal import KeyCustodyDenied

from ..engine import MantleHit, MantleQueryEngine, SystemicKeyFailure
from ..lightcone import AuthorizedScope, LightConeResolver, _raw_artifact
from ..lightcone import resolve_authorized_scope
from ..oracle import KeyPurpose, KeyRequest
from .narrowing import Coverage, TokenNarrower

logger = logging.getLogger(__name__)


#: The caller-side spelling of :data:`~mantle.search.types.ORDER_RECENCY` on
#: ``SearchQuery.sort``. They are the same word for the same thing — a request for it and a
#: report of it — so the request vocabulary is not reinvented here.
SORT_RECENCY = "recency"


#: Has this process already said that no AnchorSet is provisioned? Once per process, because
#: the thing being reported does not vary between two recalls: an AnchorSet arrives by an
#: operator loading one. The alternative is one ERROR and one traceback per query, for as long
#: as the node runs, describing the install default.
_warned_unprovisioned = False


def _warn_unprovisioned_once() -> None:
    """Report the unprovisioned node once, at WARNING.

    WARNING, not ERROR: nothing failed. The ranker was asked for a routed read on a node with
    no coordinate system to route against, which is the state of every install nobody has
    provisioned. Recall still answers — it narrowed lexically, and it returns what survived
    ordered by how much of the query each hit matched — and ``SearchResult.ordering`` says so.
    No traceback, because there is no call path to inspect and the exception's own message is
    the whole finding.
    """
    global _warned_unprovisioned
    if _warned_unprovisioned:
        return
    _warned_unprovisioned = True
    logger.warning(
        "MantleSseSearchAccessor: no AnchorSet is provisioned, so no cosine can rank a recall "
        "on this node and every text response orders by query coverage (ordering=%r, score = "
        "the count of query stems matched). This is the install default, not a fault, and it "
        "does not change until an operator loads the canonical AnchorSet — see "
        "mantle.search.anchors.store and `python -m mantle.system.manage_anchors --action "
        "inspect`. Reported once per process, because the state is the same for every query.",
        ORDER_COVERAGE,
    )


def _key_request(query, action: str = "read") -> KeyRequest:
    """The requester identity for every key this search will need.

    Built from ``query.user_id`` — the principal actually making the request —
    not from the objects being read.

    Raises ``ValueError`` on a missing ``user_id`` rather than defaulting to an
    anonymous request: an unauthenticated search must fail closed, and a search
    with no requester is exactly that.
    """
    user_id = getattr(query, "user_id", None)
    if not user_id:
        raise ValueError(
            "search requires a requesting principal (query.user_id); refusing to "
            "issue keys for an anonymous request"
        )
    return KeyRequest(
        requester_id=str(user_id), purpose=KeyPurpose.GRANT,
        requester_type="user", action=action,
    )


#: A document the tie-break could not read. Larger than any real term count, so an unreadable
#: document loses a comparison it could not take part in rather than winning it by default.
_UNREADABLE = 1 << 30

#: The content type whose `lemmas` are NAMES rather than extracted body terms. Same constant
#: `pipeline_unified` reads for the same reason — one field, two meanings, two writers.
_LEXICON_CT = "text/x-wordnet"

#: A title split into the words a person would type. Hyphens separate, unlike `_WORDS`,
#: because a coined name like `prism-protocol` is named by either half.
_NAME_WORDS = re.compile(r"[a-z]+")

#: Words, for counting a document's own terms in `_terms_outside`.
_WORDS = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _narrowing_text(plan: "RecallPlan") -> str:
    """The string the narrower analyzes — ``retrieval_text``, with a phrase's quotes put back.

    ``plan_recall`` reduces a query to its terms joined by spaces, and a phrase term arrives
    there already unquoted: the parser lifted the quotes when it decided the term WAS a
    phrase, recording that on ``ParsedQuery.terms[i].is_phrase``. So the marker survives the
    parse and does not survive the join, and
    :func:`~.narrowing.phrase_stems` reads it off a quoted string — the one reading of what a
    phrase is on this path, shared by the gate and by the bigram count.

    Re-quoting is therefore restoring information the plan had, not inventing a second
    convention. It applies to a query that is ONE phrase and nothing else, because that is the
    whole of what the narrower's all-or-nothing phrase reading can express: a query mixing a
    phrase with loose terms analyzes as the union of its stems, which is what it did before
    the phrase was written.
    """
    terms = getattr(plan.parsed, "terms", ()) or ()
    if len(terms) == 1 and getattr(terms[0], "is_phrase", False):
        return '"%s"' % plan.retrieval_text
    return plan.retrieval_text


class _Ranked(NamedTuple):
    """One artifact that survived narrowing, in the position the result presents it.

    The unit between ordering and hydration. There is no per-arm provenance on it and no fused
    score, because there is one ranker and nothing to fuse.

    ``score`` is whatever put it here: the cosine on the semantic path, the integer count of
    distinct query stems it matched on the coverage path, and ``None`` on the recency path,
    where nothing measured it against anything. ``collection_id`` / ``principal_id`` come from
    the ranker when there was one; hydration reads the collection off the doc when there was
    not. ``coverage`` rides along on the coverage path so hydration can echo both counts
    without recomputing either.
    """

    artifact_id: str
    score: Optional[float]
    collection_id: Optional[str]
    principal_id: Optional[str]
    coverage: Optional[Coverage] = None


class RecallPlan(NamedTuple):
    """What a query string means, once its filters have been taken out of it.

    ``retrieval_text`` is the terms only. The filter tokens are gone from it: handing
    ``query.query_text`` to the lexical arm whole would let `type:pdf` reach the index as the two
    ordinary tokens `type` and `pdf` — unfiltered and unstripped, scoring documents that merely
    contain the word "type". Retrieval sees only what the caller is actually searching for.
    ``@name:value`` controls are already absent for the same reason: the parser lifts them out of
    the term stream too, so they never reach the index.
    """

    parsed: object
    predicate: Optional[Callable[[dict], bool]]
    retrieval_text: str


def plan_recall(query_text: str, *, has_vector: bool = False) -> RecallPlan:
    """Parse a query, compile its filters, and reduce it to the text retrieval should see.

    Every way a query string can be unusable is decided HERE, so that the ranked path, the
    candidates path and anything built on either refuse the same inputs. Raises
    :class:`~mantle.search.field_filters.QueryFilterError` for a filter this path cannot apply.
    Refusing is the contract: a filter that parsed and was then dropped returns a result set
    indistinguishable from an answer, which is exactly what this replaces.

    Only a KNOWN filterable field parses as a filter, so what reaches the refusals below is a
    filter the caller meant as one. A `word:value` whose word is not a field never becomes a
    filter and is never refused — it is a search term, and it is in ``retrieval_text`` whole.

    ``has_vector`` says whether the request carries a query vector, which is the difference
    between a filter-only string that is refused and one that legitimately narrows a kNN
    recall. A query of nothing but filters and no vector names no topic and carries no
    direction: a filter narrows a recall, it does not constitute one, and answering it with
    zero hits would be the same silent shrug this path exists to remove.

    The other empty case — no terms, no filters, no vector — is deliberately NOT decided here:
    ``search()`` answers it with an empty result and ``candidates()`` calls it malformed, and
    that difference is theirs to keep.
    """
    from mantle.search.field_filters import compile_filters, describe
    from mantle.search.query_parser import parse_query

    parsed = parse_query(query_text or "")
    predicate = compile_filters(parsed.filters)
    retrieval_text = " ".join(t.text for t in parsed.terms).strip()

    if parsed.has_filters() and not parsed.has_topics() and not has_vector:
        raise QueryFilterError(
            "a filter narrows a recall, it is not a recall by itself: %r carries only filters "
            "(%s) and no search terms or vector, so there is nothing to rank. Add query terms, "
            "or send `vector` + `space_id`."
            % (query_text, ", ".join(describe(parsed.filters)))
        )

    return RecallPlan(parsed=parsed, predicate=predicate, retrieval_text=retrieval_text)


class MantleSseSearchAccessor:
    """Canonical router-shape search accessor.

    Returns the :class:`SearchResult` shape the router's response-mapping
    code expects.
    """

    def __init__(
        self,
        lightcone: LightConeResolver,
        *,
        store_db,
        embeddings: Optional[Embeddings] = None,
        narrower: Optional[TokenNarrower] = None,
        ranker: Optional[MantleQueryEngine] = None,
        segment: str = "committed",
    ) -> None:
        """``narrower`` narrows, ``ranker`` ranks. There is no third collaborator.

        :meth:`candidates` and :meth:`search` both run the same narrow-then-order path, so there
        is one retrieval story and these are its two halves.

        Both are what :func:`~..wiring.build_sse_search_accessor` supplies on every node it can
        build an accessor for at all — it refuses to build one otherwise, so a production
        accessor never has ``None`` in either. They stay keyword-optional so a caller can
        construct one for a path that needs neither; both entry points refuse loudly rather
        than degrading when the narrower is missing.

        ``segment`` is the per-state index this accessor reads, and it is held here because
        **hydration needs it too**. The index is keyed on ``root_id``, so a ranked hit names a
        lineage; turning that back into bytes means asking for this segment's version of it.
        Without the segment, hydration can only take the row sitting at the root id, which is
        the version that was indexed for every artifact where ``id == root_id`` and the wrong
        one as soon as a committed member is edited into a draft.
        """
        self._lightcone = lightcone
        self._store_db = store_db
        self._embeddings = embeddings or Embeddings()
        self._narrower = narrower
        self._ranker = ranker
        self._segment = segment

    def search(self, query) -> "object":
        """Narrow lexically, rank semantically, hydrate. Returns a :class:`SearchResult`."""
        from mantle.search.types import SearchResult

        if self._store_db is None:
            raise ValueError(
                "MantleSseSearchAccessor needs an store_db for hydration"
            )
        if self._narrower is None:
            # Loud, not degraded. A recall narrows on the blind-token index before anything
            # ranks, and an accessor with no narrower has exactly two things it could do
            # instead: hand back everything authorized, which WIDENS every query into a
            # collection dump, or hand back nothing, which is an empty 200 for a query that
            # matched. Both are wrong and neither is visible from the response, so this refuses
            # instead. `build_sse_search_accessor` will not build one without a posting store,
            # so reaching this means an accessor was constructed by hand.
            raise ValueError(
                "MantleSseSearchAccessor needs a TokenNarrower to narrow a recall; refusing to "
                "answer by widening to everything authorized or by matching nothing"
            )

        provided_embedding = getattr(query, "query_embedding", None)
        plan = plan_recall(query.query_text, has_vector=bool(provided_embedding))
        parsed = plan.parsed
        applied_filters = describe_filters(parsed.filters)

        def _nothing(ordering: str = ORDER_RECENCY) -> "SearchResult":
            """The empty body. One shape for every way a recall ends with no hits."""
            return SearchResult(
                hits=[],
                total=0,
                parsed_query=parsed,
                applied_filters=applied_filters,
                corrections=parsed.corrections,
                ordering=ordering,
            )

        # Empty text is only a no-op when there's also no raw query vector;
        # an embedding-only query ("embedding activation") proceeds to kNN. A query of
        # nothing but FILTERS never reaches here — `plan_recall` refused it.
        if not parsed.has_topics() and not provided_embedding:
            return _nothing()

        # The blind-token narrowing, compiled from the query's terms only — the same string
        # the ranker's embedding is resolved from, so the two halves of the recall are reading
        # one query. `None` when the terms analyze to no stems: "nothing to look up" is
        # deliberately NOT "narrow to nothing", or a vector-only recall would meet the empty
        # set instead of the whole authorized one.
        #
        # The key request is minted only when there is something to look up AND a requester to
        # mint it for. `resolve_authorized_scope` answers an anonymous search with an empty
        # light cone, which is already correct and already fails closed; building a key request
        # for it here would turn that correct empty answer into a raise.
        token_lookup, coverage = self._narrowing(query, plan)

        # Light-cone authorization — the single ACL path. Both granularities travel
        # together from here down; see `AuthorizedScope`.
        #
        # Both narrowings ride IN to the resolve rather than being applied to its result. The
        # filter is a function of a doc and the loop is already holding one; the token lookup
        # is a function of the CONTEXTS, which are what that loop produces, so it meets in a
        # second phase inside the same call. Either way the set that comes back is a subset of
        # what the light cone returned, by construction rather than by discipline.
        authorized = resolve_authorized_scope(
            self._store_db,
            principal_id=query.user_id,
            lightcone=self._lightcone,
            artifact_predicate=plan.predicate,
            token_lookup=token_lookup,
        )
        contexts = authorized.contexts

        # An empty surviving set ends the search HERE, before a key is derived or a cell is
        # decrypted — everything below ranks WITHIN this set, so there is nothing left to do.
        # It is one door for four different facts: the light cone authorized nothing, the
        # filter matched nothing, the terms matched nothing, and the filter or the terms named
        # something this principal cannot read. The last one is the security property: an
        # unauthorized artifact's doc is never shown to the predicate and its id is met away by
        # the intersection, so "no match" and "not yours" leave with the same empty body and the
        # same `total`, and nothing in the response distinguishes them.
        if not authorized.artifact_ids:
            return _nothing()

        # Respect explicit scope from the router (body.scope → query.scope).
        # When scope is set, restrict to only the requested containers.
        # This is distinct from collection_ids (which carries the full authorized
        # set); scope is set only for user-explicit or principal-restricted searches.
        scope = getattr(query, "scope", None)
        if scope:
            allowed = set(scope)
            contexts = [(principal, col) for principal, col in contexts if col in allowed]

        logger.info(
            "MantleSseSearchAccessor: '%s' narrowed to %d artifacts over %d contexts",
            query.query_text, len(authorized.artifact_ids), len(contexts),
        )

        if not contexts:
            return _nothing()

        # The ranker's query: a caller-provided vector directly ("embedding activation"), else
        # whatever the embeddings cache holds for the query text. `None` from either is not a
        # failure — it means this recall has nothing to rank BY, which the recency path answers.
        embedding = (
            list(provided_embedding)
            if provided_embedding
            else self._embed_or_none(plan.retrieval_text, parsed)
        )

        _from = max(0, int(getattr(query, "from_", 0) or 0))
        _want = _from + query.size
        # A horizon wider than the page, because the beacon cut reads where a ranking stops off
        # the ranking's own spectrum and needs a spectrum to read.
        horizon = max(_want * 3, 50)

        ordering, ranked = self._order(
            query, authorized, contexts, embedding, coverage, top_k=horizon,
        )

        hits = self._hydrate(ranked[_from:_want])
        return SearchResult(
            hits=hits,
            # Every narrowing is already inside this count — the filter and the token lookup
            # both cut the set before anything ranked it — so `total` counts narrowed matches
            # and page N is the Nth page of them. A cut applied after hydration could say
            # neither of those things.
            total=len(ranked),
            # Only the two RANKED paths have a ceiling: each stops at the horizon above. The
            # coverage and recency paths order the whole narrowed set — both read a number the
            # resolve already produced, so neither has a retrieval budget to run out of — and
            # their `total` is exact however large it is. A reach-ordered result that reaches the
            # horizon means the cut kept the entire pool, so there is more behind it; one the cut
            # bit into is complete, and says so by not being capped.
            total_is_capped=(
                ordering in (ORDER_SEMANTIC, ORDER_REACH) and len(ranked) >= horizon
            ),
            parsed_query=parsed,
            applied_filters=applied_filters,
            corrections=parsed.corrections,
            ordering=ordering,
        )

    # ------------------------------------------------------------------
    # Ordering — the one place a result learns what put it in this order
    # ------------------------------------------------------------------

    def _order(
        self,
        query,
        authorized: AuthorizedScope,
        contexts: Sequence[tuple],
        embedding: Optional[list],
        coverage: Mapping[str, Coverage],
        *,
        top_k: int,
    ) -> tuple[str, List[_Ranked]]:
        """Order the narrowed set, and say what ordered it.

        Four answers, tried in that order:

        ``ORDER_SEMANTIC`` — a cosine did.

        ``ORDER_REACH`` — no cosine could, but an ontology is bound into
        :mod:`mantle.search.ranking`, so the coverage set was re-ordered by measured reach and
        cut where that reach stops. Tried before coverage and on top of it: the coverage order
        is what it re-ranks, so this arm can only ever be reached on a set the narrowing already
        produced and the light cone already met.

        ``ORDER_COVERAGE`` — no cosine could, but the query had terms, so the narrowing counted
        how much of it each survivor matched and that count is the order. This is not a
        fallback ranking invented for the occasion: the counts were produced by the narrowing
        that decided membership, and they exist whether or not this branch reads them.

        ``ORDER_RECENCY`` — neither. Two causes, and it does not separate them: the caller asked
        for ``sort="recency"``, or the recall had no query terms to cover (an embedding-only
        request whose vector nothing could rank).

        ``sort`` is the caller's request and this is the outcome. ``"recency"`` is honoured
        whether or not a vector or a coverage map exists, and honoured BEFORE anything runs, so
        asking for it costs no cell decryption. ``"relevance"`` asks for the best ordering
        available and walks down this list.
        """
        if getattr(query, "sort", None) == SORT_RECENCY:
            return ORDER_RECENCY, self._by_recency(authorized)

        if embedding is not None:
            ranked = self._by_cosine(
                query, contexts, embedding, authorized, top_k=top_k,
            )
            # `None` is "the arm could not run"; an empty LIST is "it ran and nothing
            # survived". Only the first may fall through — re-ordering a ranker's empty answer
            # would hand back every artifact it just excluded.
            if ranked is not None:
                return ORDER_SEMANTIC, ranked

        if coverage:
            ordered = self._by_coverage(authorized, coverage)
            # `_by_reach` derives this for itself; the coverage arm returns before it runs, so
            # §129 needs it here too. (`text` is not in scope in `_order` — the first version of
            # §129 referred to it anyway and the bench produced no output at all.)
            text = (getattr(query, "query_text", "") or "").strip()
            # ── when coverage has already named the answer, do not ask reach ─────────────────
            page = int(getattr(query, "size", 10) or 10)
            named = self._coverage_names_the_answer(ordered, top_k)
            if named is not None and not self._lead_is_not_merely_broad(named):
                named = None
            if named is None:
                named = self._break_a_tied_lead(ordered, top_k, page)
            if named is not None:
                # ── the name is evidence whichever arm ordered (§129) ────────────────────────
                # `_coverage_names_the_answer` and `_break_a_tied_lead` return BEFORE `_by_reach`
                # runs, so §117's naming and §128's cut protection never saw these queries at
                # all. Measured over the 60 pinned modifier questions, the coverage arm answered
                # 13 of them and got 3 right — 23%, against reach's 79% on the other 47.
                #
                # A candidate whose own lemma IS the word asked about is a fact about the
                # candidate, not about the ranker that happened to order it. So it applies here
                # too, as a SECONDARY key: coverage's own count still leads, and the name breaks
                # what that count cannot — which on a definition question is most of it, since
                # every member of a derivational family matches the same single stem.
                return ORDER_COVERAGE, self._named_break_the_tie(named, text)
            # `None` is "the arm could not run", exactly as in the cosine branch above, and for
            # the same reason: an empty LIST from a ranker that DID run is a real answer and
            # falling through to coverage would hand back everything it just excluded.
            reached = self._by_reach(query, ordered, top_k=top_k)
            if reached is not None:
                return ORDER_REACH, reached
            return ORDER_COVERAGE, ordered
        return ORDER_RECENCY, self._by_recency(authorized)

    def _by_reach(
        self, query, ordered: List[_Ranked], *, top_k: int,
    ) -> Optional[List[_Ranked]]:
        """Re-rank the coverage-ordered set by measured reach and cut it. ``None`` when the arm
        could not run at all.

        ``mantle.search.ranking`` is the whole of the measurement and none of it lives here. This
        method is the seam between a recall — which knows what the caller may read — and a
        ranking, which knows what a question is about, and the two facts it adds are both about
        that boundary.

        **The result is met with the pool, never unioned with it.** ``ranking.rank`` may ADD a
        candidate: the synset the need itself names is a position by construction, which is what
        lets a corpus answer "what is a glacier" with `glacier.n.01` even when no retrieval arm
        surfaced it. That is right for a corpus and WRONG here — this node's lattice carries those
        vertices too, and an id that entered the ranking from the ontology rather than from the
        narrowing never passed the light cone. So the returned order is filtered back to the ids
        that went in. That filter is not a check some other path could forget: it is the only
        route by which a reach-ordered id reaches a response at all.

        **The pool is the horizon, not the whole set.** Reach costs a propagation per position, so
        the arm reads the first ``top_k`` by coverage and orders those. Same budget the cosine arm
        takes, and ``total_is_capped`` reports it the same way.

        The cut is ``ranking.relevance_cut`` over this arm's own spectrum — the aperture's
        ``k_signal`` where an instrument is registered, ``_knee`` where none is. Coverage does not
        cut and must not: its ``total`` is documented as every narrowed match. Reach's is
        documented as what survived the cut, which is why this is a separate ordering rather than
        a better coverage.
        """
        text = (getattr(query, "query_text", "") or "").strip()
        if not text or self._store_db is None:
            return None
        pool = self._pool_for_reach(ordered, top_k)
        # ── the POOL may not discard a named candidate either (§132) ─────────────────────────
        # §128 stopped the reach CUT from throwing away a candidate the question named. The pool
        # is the same decision one stage earlier and was still doing it: `_pool_for_reach` sorts
        # by coverage and slices at `top_k`, so on a one-stem query — which `_by_coverage` says
        # ties EVERYWHERE by construction — the slice falls inside a tie and the answer can be on
        # the wrong side of it.
        #
        # Measured: `what does solar mean` narrows 379 candidates, the answer among them with
        # `matched=('solar',)`, and it never reaches the pool. Naming runs on the pool, so it
        # never gets asked. The same fix, at the stage that actually drops it.
        if len(pool) < len(ordered):
            in_pool = {row.artifact_id for row in pool}
            outside = [row for row in ordered if row.artifact_id not in in_pool]
            if outside:
                named_out = self._named_by_the_question(
                    [row.artifact_id for row in outside], text)
                if named_out:
                    pool = pool + [row for row in outside if row.artifact_id in named_out]
        if not pool:
            return None

        from mantle.search import ranking

        # BM25's sign convention, which `ranking` both takes and returns: most-negative is best.
        # A coverage score counts up, so it enters negated and the pool arrives already in the
        # best-first order it had. `content_type` is read by the ranking only for the `wn-`
        # position shortcut, and an artifact's is not loaded at this point, so it is "".
        cand = [(r.artifact_id, "", -float(r.score or 0.0)) for r in pool]
        try:
            # `own_position=False` — this arm filters the result back to the ids that went in,
            # so a candidate the ranking adds from the ontology can never be returned. Asking for
            # it and discarding it is not neutral: the added row folds the narrowed row into
            # itself and is then discarded with it. See `_reach_rank`.
            ranked, account, _reached = ranking.rank(
                cand, text, self._store_db, own_position=False,
            )
        except Exception:  # noqa: BLE001 — a ranking that raises must not fail a valid recall
            logger.debug("reach ranking raised; serving the coverage order", exc_info=True)
            return None
        if account.get("reach") != "measured":
            # "unavailable" (nothing bound), "no-coordinate" (the query fired nothing), or
            # "unreached" (no path from the need to anything here). Each is a real statement and
            # none of them is an order, so the coverage order stands.
            return None

        by_id = {r.artifact_id: r for r in pool}
        rows = [(cid, score) for cid, _ct, score in ranked if cid in by_id]
        if not rows:
            return None
        # `cut_for` rather than `relevance_cut`: the sharp cut needs the frame the ranked synsets
        # span, and a caller that passes only scores gets the thin `_knee` while believing it asked
        # for the instrument. Deriving the frame inside the ranking is what stops that argument
        # from being forgettable — see `ranking._cut_for`.
        cut = ranking.cut_for([(cid, "", score) for cid, score in rows],
                              query=text, store=self._store_db)
        cut = max(1, min(int(cut), len(rows)))
        survived = [by_id[cid]._replace(score=-float(score)) for cid, score in rows[:cut]]

        # ── the cut may not discard a candidate the question NAMED (§128) ────────────────────
        # `cut_for` reads this arm's own reach spectrum and decides where reach stops. It knows
        # nothing about names, and it should not: it is a measurement. But a candidate whose own
        # lemma IS the word asked about is not a weak reading of that spectrum, it is evidence of
        # a different kind — the same kind §104 admits — and reach is not entitled to throw it
        # away for scoring poorly on the only axis it can see.
        #
        # This is where the modifier tier was being lost. Attributed over the 60 pinned modifier
        # questions BEFORE this existed:
        #
        #     rank-1                     32        in answer, not first        7
        #     in pool, CUT by reach      13        ranked but past the page    4
        #     narrowed, not in pool       2        NOT NARROWED                2
        #
        # Thirteen answers reached the pool and were cut — nearly twice the mis-ranked bucket —
        # and §117's naming runs on the SURVIVORS, so it could only ever reorder what the cut had
        # already spared. A modifier projects to the same noun as every other modifier about that
        # noun (§96), so its reach is genuinely indistinguishable from theirs; the name is the
        # only thing that separates them and it was arriving too late to be heard.
        named = self._named_by_the_question([cid for cid, _score in rows], text)
        if named:
            kept = {row.artifact_id for row in survived}
            survived = survived + [by_id[cid]._replace(score=-float(score))
                                   for cid, score in rows[cut:]
                                   if cid in named and cid not in kept]
        return self._coverage_outranks_reach(survived, text, named=named)

    def _named_break_the_tie(self, ordered: List[_Ranked], text: str) -> List[_Ranked]:
        """Among rows the coverage count cannot separate, prefer the one the question NAMED.

        Stable, and secondary to `stems`: coverage's own measurement still leads and this only
        decides what it left tied. What it replaces as the tie-break is `_by_coverage`'s
        timestamp, which §81 already records as a non-signal — "a tied group has named a set and
        not an answer, and returning it would return a timestamp".
        """
        if len(ordered) < 2 or not text:
            return ordered
        named = self._named_by_the_question([r.artifact_id for r in ordered], text)
        if not named:
            return ordered
        zero = Coverage()
        return sorted(
            ordered,
            key=lambda row: ((row.coverage or zero).stems,
                             1 if row.artifact_id in named else 0),
            reverse=True,
        )

    def _named_by_the_question(self, artifact_ids, text: str) -> set:
        """Which candidates the question called by name (§117).

        A definition question names a word. One candidate's own lemma IS that word and the others'
        are not, and that is a fact about the candidate the ranker can check against the query
        text — the same kind of evidence §104 admits, and not a weight or a threshold.

        Stem-level coverage cannot supply it. `squeaking`, `squeakiness` and `squeaker` all stem to
        `squeak`, which is what stemming is FOR and is right everywhere else; it is exactly why
        `what does squeaking mean` answers `squeakiness` (§116). The lemmas are stored on every
        synset, so the identity is available without inferring it.

        Only SALIENT words are matched. "what does squeaking mean" contains `mean`, and the corpus
        holds a synset whose lemma is `mean` — matching on it would name a frame word as the
        subject. The salient measure already answers which words carry this question (`squeak`
        alone, here), and a surface word is kept when its own stem survives that measure, so the
        two paths agree by construction rather than by a second stop-list.
        """
        wanted = [str(a) for a in artifact_ids]
        if not wanted or self._store_db is None or not text:
            return set()
        words = {w.lower() for w in _WORDS.findall(text)}
        if not words:
            return set()

        from .tokenizer import tokenize
        measure = self._salient_measure()
        if measure is not None:
            try:
                stems = [t for t in (tokenize(text) or []) if t]
                keep = set(measure(stems) or stems)
                words = {w for w in words
                         if any(t in keep for t in (tokenize(w) or []))}
            except Exception:      # noqa: BLE001 — unmeasurable salience matches every word
                pass
        if not words:
            return set()

        named = set()
        try:
            conn = self._store_db.artifacts.db.read()
            rows = conn.execute(
                "SELECT id, doc FROM vertex WHERE id IN (%s)" % ",".join("?" * len(wanted)),
                wanted,
            )
        except Exception:          # noqa: BLE001 — a store read raises broadly
            return named
        for artifact_id, blob in rows:
            try:
                doc = json.loads(blob) if isinstance(blob, (str, bytes)) else (blob or {})
            except Exception:      # noqa: BLE001 — a malformed row names nothing
                continue
            if not isinstance(doc, dict):
                continue
            # `lemmas` means two different things, and only one of them is a name.
            # On a lexicon entry they are the words that MEAN the concept — the thing this
            # function is asking about. On a canon or wiki artifact they are key terms
            # `astra/doc_index` extracted from the BODY, which is a description of what the
            # document discusses and not what it is called. `pipeline_unified.
            # _extract_artifact_fields` already carries this warning and guards the same way,
            # by reading the TYPE rather than the field's presence.
            #
            # Measured, without this guard: `prism protocol` answered `canon:README`,
            # `canon:SIGNAL-PROTOCOL` and `canon:MCP-VS-SIGNAL-AUDIT` ahead of
            # `canon:prism-protocol`, because every document that discusses protocols carries
            # `protocol` among its body terms and was therefore "named" by the question.
            # by-title fell from 33/40 to 15/40 on that alone.
            # Prose is not named here. Matching query words against a document's title is worth
            # +6 on by-title only while every OEWN distance reads 0, and exactly 0 with the metric
            # working: for a prose candidate the title's words are already indexed, so coverage
            # counts them and the name adds nothing
            # coverage did not already have. A lexicon entry is different — its lemmas are names
            # that stemming collapses (`squeaking`/`squeakiness`/`squeaker`), which is the one
            # thing coverage cannot see.
            if str(doc.get("content_type") or "") != _LEXICON_CT:
                continue
            # ── a prose artifact's NAME is its TITLE (§120) ──────────────────────────────────
            # §111 settled which field holds what: `title` is the name, `description` is the
            # offer, and `lemmas` on prose are key terms `astra/doc_index` pulled from the BODY.
            # The body terms are why this is guarded at all — every document that DISCUSSES
            # protocols carries `protocol` among them (§117). The TITLE is a different field
            # doing a different job, and asking whether the question named the document means
            # asking about the title.
            #
            # Measured: `prism protocol` ranked `canon:README#read-in-this-order` first because
            # its 124 body positions outweighed `canon:prism-protocol`'s 12 under a size-aware
            # score (§118). README's title says nothing about prisms or protocols; the wanted
            # document's title is the query.
            for lemma in (doc.get("lemmas") or []):
                if str(lemma).strip().lower() in words:
                    named.add(str(artifact_id))
                    break
        return named

    def _coverage_outranks_reach(self, survived: List[_Ranked], text: str = "",
                                 named: Optional[set] = None) -> List[_Ranked]:
        """Among the rows reach kept, one carrying more of the question comes first (§104).

        Reach decides who survives — the cut is taken on its own spectrum, above, and nothing here
        touches it. This decides only the order of the survivors, and it states the one thing
        coverage knows that reach does not:

            coverage is evidence the ranker can check — these are the query's own stems, and this
            artifact carries them. Reach is a measured distance between positions. A distance may
            order candidates the evidence cannot separate; it may not overturn strictly better
            evidence.

        Without this the two were one ranking, and reach simply won. Asked `prism protocol`, the
        corpus holds nineteen sections of a document named exactly that, every one of them matching
        BOTH stems, and the synset `prism` matching one:

            canon:prism-protocol#1     stems=2  matched=prism,protocol     rank 2
            wn-oewn-13907168-n         stems=1  matched=prism              rank 1

        The synset is at distance 0 from the query's own position, so reach is right about what it
        measures and answering with it is still wrong. Every miss on `bench_canon --by-title` had
        this shape: a coined multi-word name losing to one of its own words.

        ## Why the tie-break arms do not already cover this

        They decline, correctly, and for reasons that are about coverage rather than about reach.
        `_coverage_names_the_answer` needs a leading band with a STRICTLY best member, and here
        nineteen sections of one document tie exactly. `_break_a_tied_lead` then needs that band to
        fit the caller's page, and nineteen does not fit ten — a lead too large to be a page has
        not been narrowed to an answer, which is true, and does not mean the band should lose to a
        row beneath it.

        Those arms answer "has coverage identified THE answer". This answers the weaker and much
        more often available question: "has coverage ruled this row out". Nineteen rows carrying
        twice as much of the query as the row above them is a ruling, whether or not it names a
        winner.

        ## It is stable, so reach still ranks equals

        `sorted` is stable, so within one stem count the order is exactly the order reach returned.
        This adds a coarser key ABOVE reach's, it does not replace it, and on a query where every
        candidate covers the same amount — a one-stem query, which `_by_coverage` says ties
        everywhere by construction — every row is in one stratum and the order is unchanged.
        """
        zero = Coverage()
        # ── the same evidence requirement the coverage arm already imposes ───────────────────
        # `_coverage_names_the_answer` refuses to pre-empt reach unless it can say WHICH stems
        # hit, because a bare count is not checkable: a broad document carries more of every
        # query without being about any of them, which is
        # `test_order_tries_reach_before_coverage_and_names_what_answered` — `doc-broad` covers
        # both query stems across three subjects and `doc-subject` covers one and IS the subject.
        # Reach is right there and this must not overturn it.
        #
        # A count with no `matched` behind it is exactly that un-checkable evidence, so it does
        # not reorder anything. This is not a special case for a test: it is the rule §86 already
        # argued, applied at the one place that had skipped it.
        if any(not (row.coverage or zero).matched for row in survived):
            return survived
        # ── and among rows coverage cannot separate, the one the question NAMED (§117) ────────
        # Coverage first: carrying more of the query is the stronger evidence. The name breaks
        # what is left, which on a definition question is everything — every member of a
        # derivational family matches the same single stem.
        if named is None:
            named = self._named_by_the_question([r.artifact_id for r in survived], text)
        return sorted(
            survived,
            key=lambda row: ((row.coverage or zero).stems,
                             1 if row.artifact_id in named else 0),
            reverse=True,
        )

    def _by_cosine(
        self,
        query,
        contexts: Sequence[tuple],
        embedding: list,
        authorized: AuthorizedScope,
        *,
        top_k: int,
    ) -> Optional[List[_Ranked]]:
        """Cosine rank over the narrowed set. ``None`` when the arm could not run at all.

        The engine takes the beacon cut over its own spectrum before returning, which is where
        that cut belongs and the only place it can be taken: it reads ``(item_embs, query_emb)``
        and this is the one arm holding a vector per candidate.

        ``authorized_artifacts`` is the narrowed set, and it is a MEET rather than a filter
        bolted on afterwards: a cell is the unit of ENCRYPTION, so holding its key decrypts
        chunks of artifacts the requester was never granted. Passing the set in makes the
        ranking's universe the light cone's, not the key's.
        """
        if self._ranker is None:
            return None

        # Function-local, the way every use of the anchors package from a query path is: the
        # narrowing must not pay for the ranker's imports.
        from mantle.search.anchors.store import AnchorSetNotProvisioned

        try:
            hits: Sequence[MantleHit] = self._ranker.search(
                embedding,
                contexts,
                _key_request(query),
                top_k=top_k,
                authorized_artifacts=authorized.artifact_ids,
            )
        except (KeyCustodyDenied, SystemicKeyFailure):
            # A custody refusal and a wholesale key fault are answers, not absences. Neither
            # may be re-told as "nothing could rank this" — the second exists precisely so an
            # unreadable corpus cannot come back looking like an ordinary result.
            raise
        except AnchorSetNotProvisioned:
            # A configuration state, not a fault, and the install default. Reported once per
            # process, separately from the blanket `except` below, which keeps its traceback.
            _warn_unprovisioned_once()
            return None
        except Exception:  # noqa: BLE001 — a ranking fault must not fail a narrowed recall
            logger.exception(
                "MantleSseSearchAccessor: the semantic arm raised; ordering by recency."
            )
            return None

        # Chunks collapse to artifacts, best cosine wins. A page is a page of artifacts, and
        # the same artifact appearing twice because two of its chunks are close is a ranking
        # artefact of chunking rather than a second result.
        #
        # The meet is RESTATED here on the engine's output. The engine was handed the same set
        # and honours it, so this is idempotent for the production ranker — but the guarantee
        # has to hold at this boundary whichever engine is wired in, and an injected one that
        # ignored the argument would otherwise widen a result past what the light cone
        # resolved. Restating costs a set membership per hit.
        best: dict[str, MantleHit] = {}
        for hit in hits:
            if hit.artifact_id not in authorized.artifact_ids:
                continue
            current = best.get(hit.artifact_id)
            if current is None or hit.score > current.score:
                best[hit.artifact_id] = hit
        return [
            _Ranked(h.artifact_id, h.score, h.collection_id, h.principal_id)
            for h in sorted(best.values(), key=lambda h: h.score, reverse=True)
        ]

    @staticmethod
    def _by_coverage(
        authorized: AuthorizedScope, coverage: Mapping[str, Coverage],
    ) -> List[_Ranked]:
        """The narrowed set, most of the query first, with the stem count as the score.

        The key is ``(stems, bigrams, updated_at, artifact_id)``, all descending. Every part of
        it is a number or a string the recall already holds:

        - ``stems`` — how many DISTINCT query stems the narrowing found this artifact under.
          A count of lookups that hit, not a score. There is no document frequency in it, no
          term frequency, no field length and no weighting of any kind: a stem found in the
          title and a stem found in a 40 000-word body each contribute exactly 1, and a stem
          found four times in one field contributes exactly 1. That is what makes it an
          ordinal on the query rather than a metric on the data.
        - ``bigrams`` — how many of a quoted phrase's bigrams it matched. Uniform across the
          survivors of a phrase query and zero for every unquoted one, so it breaks no tie
          today; see :mod:`.narrowing` for what making it discriminate would cost.
        - ``updated_at`` then ``artifact_id`` — the existing recency tiebreak, unchanged. A
          one-stem query is therefore byte-identical to :meth:`_by_recency`: every survivor
          scores 1, so the whole order is the tiebreak.

        Iteration is over ``authorized.artifact_ids`` and never over ``coverage``. The coverage
        map is the narrowing's own answer and is a superset of the surviving set — it can name
        artifacts the light cone refused, exactly as the narrowing's key set can, which is why
        the resolver meets that key set rather than trusting it. Walking the map here would
        undo the meet. ``.get(a)`` on an id with no entry yields a zero coverage, which sorts
        last; that is unreachable for a coverage-ordered recall (every survivor matched at
        least one stem, or it would not have survived) and is stated so the sort cannot raise
        on a hand-built scope.
        """
        stamps = authorized.updated_at
        zero = Coverage()
        ordered = sorted(
            authorized.artifact_ids,
            key=lambda a: (
                coverage.get(a, zero).stems,
                coverage.get(a, zero).bigrams,
                stamps.get(a) or "",
                a,
            ),
            reverse=True,
        )
        return [
            _Ranked(
                artifact_id,
                # An INTEGER count in a float-typed field. See `SearchHit.score`: a normalised
                # float here would be a number a client could threshold on across queries, and
                # it would mean nothing; a count says exactly what it is.
                coverage.get(artifact_id, zero).stems,
                None, None,
                coverage.get(artifact_id, zero),
            )
            for artifact_id in ordered
        ]

    @staticmethod
    def _by_recency(authorized: AuthorizedScope) -> List[_Ranked]:
        """The narrowed set, most-recently-updated first, with no score on any of it.

        Reached two ways, and it is a real answer to a real request in both. The caller asked
        for ``sort="recency"``, in which case a ranking would be the wrong answer rather than a
        missing one; or the recall had no query terms at all, in which case there is nothing to
        cover and nothing to be close to, and handing the authorized set back newest first is
        an answer where a 400 would refuse a query that worked.

        The timestamps are the light cone's own read: the loop that resolved these ids was
        already holding each doc, so this costs no store read, and it orders by the same field
        `updated_at:` FILTERS by. The artifact id breaks ties, so the same query pages the same
        way twice. An id whose doc could not be read has no timestamp and sorts last, because
        an unread doc is not evidence that it is old.
        """
        stamps = authorized.updated_at
        return [
            _Ranked(artifact_id, None, None, None)
            for artifact_id in sorted(
                authorized.artifact_ids,
                key=lambda a: (stamps.get(a) or "", a),
                reverse=True,
            )
        ]

    def candidates(
        self,
        query,
        *,
        candidate_budget: int = 200,
        include_vectors: bool = False,
    ) -> dict:
        """Raw retrieval primitive — the single authorization chokepoint.

        Resolves the light cone for ``query.user_id`` and returns the recall's candidate set,
        unranked and unhydrated. Search *flavors* (the open standard one, or an external
        premium one like Beacon) rank within this set — they can never widen it, so MANTLE §1
        holds by construction. See ``.dev/features/search-as-artifact.md``.

        Same universe as :meth:`search`. Same plan, same field filter, same blind-token
        narrowing, same meet. The only differences are that nothing here orders the set by
        anything the query says, and nothing here hydrates it.

        Why it narrows
        --------------
        Skipping the token narrowing would follow from the argument that a flavor asks for the
        authorized set to rank within, and that narrowing it decides part of the ranking on the
        flavor's behalf. That argument does not survive the narrowing becoming the ranking, and
        it did not survive contact with what the alternative actually returns.

        A narrowing is not a ranking; it is what the query means. Skipping it does not hand a
        flavor an unranked candidate set, it hands it the whole light cone for every query —
        the caller's entire readable corpus, capped at a budget, with the query text having
        selected nothing. :meth:`search` refuses to do exactly that: it raises rather than
        answer by "widening to everything authorized", because the widening is invisible in the
        response. One accessor cannot hold both positions, and the one it holds in the method
        that answers users is the one that survives.

        What IS the flavor's to decide is the ORDER, and that is what this method now declines
        to state. The coverage counts :meth:`search` orders by are computed here too — they are
        a by-product of the narrowing — and they are deliberately not published, so a flavor
        ranks a set rather than re-ranking a ranking.

        Per-candidate shape: ``artifact_id``, ``collection_id``, ``principal_id``. The fused
        vocabulary — ``sse_score``, ``rrf_score``, ``source`` — is GONE rather than nulled,
        because a null would say "this hit had no BM25 score" where the truth is that no BM25
        score exists anywhere in this system. ``vector_score`` goes with them: nothing on this
        path computes a cosine.

        ``include_vectors`` is accepted for the premium re-rank path and still surfaces no
        embedding, for the same reason as before: reaching one means threading candidate
        vectors out of ``MantleQueryEngine``, which is a change to that engine and not to this
        method.

        Budget order is RECENCY — the query-independent one. A budget has to cut somewhere, and
        cutting by anything the query said would be the ranking decision this method just
        declined to make.
        """
        if self._store_db is None:
            raise ValueError("MantleSseSearchAccessor needs an store_db")
        if self._narrower is None:
            # The same refusal `search()` makes, for the same reason and with more at stake:
            # this method is documented as the chokepoint every flavor ranks within, so an
            # accessor that answered it by widening would hand that widening to every flavor
            # at once. `build_sse_search_accessor` will not build one without a posting store.
            raise ValueError(
                "MantleSseSearchAccessor needs a TokenNarrower to narrow a recall; refusing to "
                "answer by widening to everything authorized or by matching nothing"
            )

        # The same plan `search()` builds, so a flavor ranking within this set is ranking
        # within the filtered set. A chokepoint that honoured filters differently from the
        # ranked path would make the filter mean one thing per caller — including which
        # queries it refuses, which is why the refusals live in `plan_recall` and not here.
        provided_embedding = getattr(query, "query_embedding", None)
        plan = plan_recall(query.query_text, has_vector=bool(provided_embedding))
        parsed = plan.parsed
        if not parsed.has_topics() and not provided_embedding:
            # A query that parses to no terms and carries no embedding is a
            # malformed request, not a search that legitimately matched zero
            # documents — the two must not share a return shape.
            raise ValueError(
                "empty query: %r parsed to no terms and no embedding was supplied; "
                "there is nothing to retrieve on" % (query.query_text,)
            )

        token_lookup, _coverage = self._narrowing(query, plan)
        authorized = resolve_authorized_scope(
            self._store_db,
            principal_id=query.user_id,
            lightcone=self._lightcone,
            artifact_predicate=plan.predicate,
            token_lookup=token_lookup,
        )
        contexts = authorized.contexts
        if not authorized.artifact_ids:
            # The same one door `search()` has, for the same four facts: the light cone
            # authorized nothing, the filter matched nothing, the terms matched nothing, or
            # either named something this principal cannot read. Reached without opening an
            # index beyond the ones the narrowing already read.
            return {"candidates": [], "model_id": None}
        scope = getattr(query, "scope", None)
        if scope:
            allowed = set(scope)
            contexts = [(p, c) for p, c in contexts if c in allowed]

        logger.info(
            "raw query: '%s' narrowed to %d artifacts over %d contexts (budget=%d)",
            query.query_text, len(authorized.artifact_ids), len(contexts),
            candidate_budget,
        )
        if not contexts:
            # Not a malformed request: the light cone is truthfully reporting zero authorized
            # contexts, which is the correct answer for a brand-new user with no collections
            # yet. Raising here would route through the caller's "Recall failed: …" 500
            # handler, converting a correct authorization result into a server error for the
            # most common first-run state. `model_id` is None: nothing ran.
            return {"candidates": [], "model_id": None}

        budget = max(int(candidate_budget), 1)
        # `None` when the caller named no scope, which is the case the "same universe as
        # `search()`" claim is about: no scope, no per-candidate cut, the meet is the answer.
        # A named scope is the caller's own narrowing and is applied to the candidates as well as
        # to the contexts, because those contexts do not reach a retrieval arm that would apply it
        # on this path.
        allowed_collections = {c for _p, c in contexts} if scope else None

        out = []
        for hit in self._by_recency(authorized)[:budget]:
            doc = self._safe_get(self._store_db, hit.artifact_id) or {}
            collection_id = doc.get("collection_id") or doc.get("_key") or doc.get("id")
            collection_id = str(collection_id) if collection_id else None
            if allowed_collections is not None and collection_id not in allowed_collections:
                continue
            rec = {
                "artifact_id": hit.artifact_id,
                "collection_id": collection_id,
                # The CELL principal — the collection's origin root, which is what a key is
                # derived per. Read through the one resolver that answers that question, so a
                # candidate names the same owner the index was written under.
                "principal_id": self._cell_principal(collection_id),
            }
            if include_vectors:
                rec["vector"] = None  # no path here holds a candidate embedding
            out.append(rec)
        # `model_id` names the embedding model a candidate set was retrieved under. Nothing
        # here retrieves by embedding, so there is none — and there was none before either:
        # the fused accessor this read it off never carried the attribute, so the key has
        # always been null. Kept because it is a published key, not because it varies.
        return {"candidates": out, "model_id": None}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _narrowing(self, query, plan: "RecallPlan"):
        """Compile the query's terms into ``(token_lookup, coverage)``.

        ``token_lookup`` is what :func:`~..lightcone.resolve_authorized_scope` meets — a
        callback over the CONTEXTS returning an artifact-id SET. ``coverage`` is a dict this
        method hands back EMPTY and the callback fills in when it runs, so the counts the
        narrowing produced are readable afterwards without a second lookup and without the
        resolver having to carry them.

        The values never travel through the meet. The resolver is handed ``.keys()`` and
        nothing else, so the intersection is between two sets exactly as it was, and a count
        belonging to an artifact the light cone refused is simply never looked up — see
        :meth:`_by_coverage`.

        ``(None, {})`` when the terms analyze to no stems: "nothing to look up" is deliberately
        NOT "narrow to nothing", or a vector-only recall would meet the empty set instead of
        the whole authorized one. Also ``(None, {})`` for an anonymous request — the resolver
        answers that with an empty light cone, which is already correct and already fails
        closed, where minting a key request for it here would turn that into a raise.
        """
        coverage: Dict[str, Coverage] = {}
        if not plan.retrieval_text or not getattr(query, "user_id", None):
            return None, coverage
        compiled = self._narrower.lookup_for(
            _narrowing_text(plan), _key_request(query),
            salient=self._salient_measure(),
        )
        if compiled is None:
            return None, coverage

        def _token_lookup(pairs):
            found = compiled(pairs)
            coverage.update(found)
            return found.keys()

        return _token_lookup, coverage

    @staticmethod
    def _coverage_names_the_answer(ordered: List[_Ranked], top_k: int):
        """The leading coverage group when there IS one and it has a single best member.

        ``None`` means coverage has not named anything and the reach arm should rank, which is
        what happens for a question like "what is a glacier": after salience the query is two
        stems, most candidates match one, and the spectrum is flat. Coverage genuinely says
        nothing there and reach is the whole of the answer.

        It says a great deal about a QUOTE. Measured, asking for a phrase that appears verbatim in
        exactly one canon document:

            canon:server-plugin-architecture   coverage rank 1, stems=7 (best in pool)
                                               after the reach arm: rank 38
            canon:PRISM-AND-CAPABILITIES#1     coverage rank 1, stems=4 (best in pool)
                                               after the reach arm: cut away entirely

        Reach measures conceptual distance, and for a known-item query that is the wrong question:
        the answer is the document that COVERS the query, while a lexicon synset for one of its
        nouns sits at distance 0 and wins. One ranker was being asked two different questions.

        ## Both tests are the corpus speaking, not a rule about queries

        `signal_end` — `prism.resolution`, maximum between-class variance, non-parametric and
        scale-invariant — reports whether the coverage spectrum SEPARATES. A proper subset leading
        the series is a leading group; `k == len(pool)` is the instrument saying it found nothing.
        Nothing here reads the query text, its length, or its shape.

        The second test is why this costs nothing. A leading group whose members TIE has named a
        set and not an answer, and returning it would return `_by_coverage`'s tiebreak — a
        timestamp (§81). That is exactly the dictionary case: "what is a volcano" leaves many
        documents matching both salient stems. So the gate also requires the top candidate to be
        STRICTLY better than the one behind it. Measured, with and without that second test:

                                          canon rank-1   dictionary rank-1
            reach always (before)              2/40            26/36
            leading group only                27/40            23/36
            leading group AND unique top      26/40            26/36

        One canon answer is given up and three dictionary answers come back. Uniqueness is the
        corpus's own statement that it holds one best answer for this query.

        ## It returns everything, in order — it does not cut

        Returning the leading group alone would violate the contract: `_by_reach` states it —
        "its `total` is documented as every narrowed match. Reach's is documented as what survived
        the cut." Truncating here would make `ORDER_COVERAGE` mean something new, and a caller
        paging through a coverage answer would lose every row behind the leading group.

        What this arm changes is which ordering is used, not how much of it is returned. The whole
        narrowed set comes back, ordered by the same key that chose the leading group, so the
        leaders are at the front and everything else is behind them where it was.
        """
        if len(ordered) < 2:
            return None
        zero = Coverage()

        ranked = sorted(ordered, key=MantleSseSearchAccessor._rarity_key(ordered), reverse=True)
        # ── coverage may only pre-empt reach on evidence it can CHECK ────────────────────────
        # A lead is trusted here on two grounds: it covers more of the query, and the query covers
        # it at least as well as the runner-up (`_lead_is_not_merely_broad`). The second needs
        # `matched` — WHICH stems hit — and a `Coverage` carrying only a count cannot supply it.
        #
        # Without it the count alone would decide, and the count alone is what
        # `test_order_tries_reach_before_coverage_and_names_what_answered` shows going wrong: a
        # broad document carries more of every query without being about any of them. So an
        # uncheckable lead defers to reach, the same outcome this path produces when the arm is
        # not engaged at all.
        if not (ranked[0].coverage or zero).matched:
            return None
        # The band is read over the pool the reach arm WOULD have measured, since that is the set
        # whose ordering is in question; the answer returned is the whole of `ranked`.
        window = ranked[:max(top_k, 2)]
        if MantleSseSearchAccessor._leading_band(window) is None:
            return None                      # flat: no leading group

        def _identity(row: _Ranked):
            cover = row.coverage or zero
            return (cover.stems, cover.bigrams, tuple(sorted(cover.matched)))

        if not _identity(ranked[0]) > _identity(ranked[1]):
            return None                      # a tied group is a set, not an answer
        return ranked

    @staticmethod
    def _leading_band(ranked: List[_Ranked]):
        """How many rows share the highest stem count — the leading group, exactly.

        This used `prism.resolution.signal_end`, which was a category error. That instrument finds
        where a CONTINUOUS series separates, by maximum between-class variance; `stems` is an
        integer count of lookups that hit, and a discrete spectrum's leading band is not a thing to
        estimate. It is the rows holding the maximum.

        Measured, the cost of estimating it. Querying the murre's gloss the pool is

            [6, 6, 3 x 19, 2 x 29]

        and `signal_end` answers 50 — no separation — because with three levels no single Otsu cut
        dominates. On the same data truncated to ten rows it answers 2. An estimator that depends
        on how much of the tail it is shown is the wrong tool for a question with an exact answer,
        and the murre case failed for that reason and no other.

        `None` when every row ties at the top, which is a flat spectrum with no leading group.
        """
        if not ranked:
            return None
        zero = Coverage()
        best = (ranked[0].coverage or zero).stems
        band = 0
        for row in ranked:
            if (row.coverage or zero).stems != best:
                break
            band += 1
        return None if band >= len(ranked) else band

    def _lead_is_not_merely_broad(self, ranked: List[_Ranked]) -> bool:
        """Does the leading document EARN its coverage, or merely contain a lot of words?

        Coverage counts how much of the query a document carries. A broad document carries more of
        every query, without being about any of them — and then a coverage lead is an artefact of
        the document's size rather than a statement about the question. That case is exactly what
        `test_order_tries_reach_before_coverage_and_names_what_answered` states:

            doc-broad     2 query stems, standing on glacier + trail + park
            doc-subject   1 query stem,  standing on glacier

        Coverage ranks the broad one first and the subject is the answer. Reach gets that right, so
        coverage must not pre-empt it.

        The test is the §91 asymmetry one level earlier. There, two documents covered the query
        equally and the query covered one of them more. Here the leading document covers the query
        MORE and is covered by it LESS, and that is a coverage lead not worth trusting.

        Compared against the RUNNER-UP rather than a bar: the leader keeps the answer when the query
        covers it at least as well as it covers the next candidate. Nothing is chosen, and a corpus
        where every document is broad is judged on its own scale.

        Reads two documents, and only when the gate was otherwise about to fire.
        """
        if self._store_db is None or len(ranked) < 2:
            return True
        zero = Coverage()
        asked = set((ranked[0].coverage or zero).matched)
        if not asked:
            return True
        outside = self._terms_outside(
            [ranked[0].artifact_id, ranked[1].artifact_id], asked)
        leader = outside.get(ranked[0].artifact_id, _UNREADABLE)
        runner = outside.get(ranked[1].artifact_id, _UNREADABLE)
        if leader == _UNREADABLE or runner == _UNREADABLE:
            return True                      # unmeasured: leave the decision where it was

        # ── breadth per query term answered, not breadth ────────────────────────────────────
        # Comparing the raw counts made a short document win every time. A tiny gloss carries few
        # terms outside any query, so it reads as "more specific" than a long one even when it
        # answers half as much of the question — and the guard then vetoed a leader that was
        # right. Measured, it cost five of twenty-five, every one of this shape:
        #
        #     query   the phoenix dactylifera gloss, 6 salient stems
        #     leader  phoenix dactylifera  — answers all 6, carries ~4 other terms
        #     runner  feather palm         — answers 3, carries 2 other terms
        #
        # 2 < 4, so the runner "won" on breadth while answering half the query.
        #
        # A document that answers more of the question has earned proportionally more of its own
        # subject matter, so the comparison is `outside / stems` — unasked terms per query term
        # answered. Both numbers are measured on the two documents in hand; nothing is weighted
        # and no constant enters. On the case this guard exists for it still declines: `doc-broad`
        # carries 2 unasked terms for 2 answered (1.0) against `doc-subject`'s 0 for 1 (0.0).
        zero = Coverage()
        lead_stems = max(1, (ranked[0].coverage or zero).stems)
        run_stems = max(1, (ranked[1].coverage or zero).stems)
        return (leader / lead_stems) <= (runner / run_stems)

    def _break_a_tied_lead(self, ordered: List[_Ranked], top_k: int, page: int):
        """A leading group that TIES, separated by how much of each document the query covers.

        `_coverage_names_the_answer` declines a tied lead, because returning it would return
        `_by_coverage`'s timestamp. That is right when there is nothing to separate the leaders and
        wrong when there is — and on a CONTAINMENT there is. Querying the murre's gloss:

            murre  "black-and-white diving bird of northern seas"
            auk    "black-and-white short-necked web-footed diving bird of northern seas"

        The auk's gloss contains the murre's word for word, so both carry all six salient stems and
        both are `stems=6`. Coverage has saturated. Handing that to reach returned neither — it
        answered `aquatic bird / sea bird`, the hypernyms.

        ## The other direction of the same question

        `stems` counts how much of the QUERY a document carries. It says nothing about how much of
        the DOCUMENT is the query, and that is exactly what separates these two: the murre's
        document IS the query, the auk's is the query plus `short-necked` and `web-footed`. So
        among documents covering the query equally, prefer the one the query covers most.

        Nothing is weighted and no length is normalised by a chosen exponent — it is a count of
        terms outside the query, and the smallest wins. A document that cannot be read counts as
        maximally distant rather than winning by default: an unreadable row must not take a
        tie-break it could not take part in.

        ## Why the work is bounded

        This reads documents, which the ranking otherwise avoids. It runs only when coverage
        produced a tied lead that already fits the caller's own page, so it is a handful of reads
        on a query where the alternative is giving up. `page` is the caller's envelope, not a size
        chosen here; a lead too large to be a page has not been narrowed to an answer.

        Measured on 90 reverse-dictionary questions: 48/90 -> 54/90 rank-1, 62/90 -> 68/90 in
        answer, all six discordant questions moving the same way (z = 2.44, the ceiling at six).
        """
        if len(ordered) < 2 or self._store_db is None:
            return None
        zero = Coverage()
        ranked = sorted(ordered, key=self._rarity_key(ordered), reverse=True)
        window = ranked[:max(top_k, 2)]
        leading = self._leading_band(window)
        if leading is None or leading > max(1, page):
            return None

        asked = set((ranked[0].coverage or zero).matched)
        group = ranked[:leading]
        outside = self._terms_outside([row.artifact_id for row in group], asked)
        scored = [(outside.get(row.artifact_id, _UNREADABLE), row) for row in group]
        scored.sort(key=lambda pair: pair[0])
        if len(scored) < 2 or scored[0][0] >= scored[1][0]:
            return None                      # still tied on both directions: coverage has no answer
        return [row for _outside, row in scored] + ranked[leading:]

    def _terms_outside(self, artifact_ids, asked: set):
        """Per artifact, how many of its own terms the query did not ask for.

        Read with one batched `id IN (...)` over `vertex`, the same statement `_reach_rank` uses
        and for the same reason it batches: the lattice is a 9.7 GB file and a point lookup per
        candidate is a round trip per candidate. Calling `artifacts.get_artifact` once per
        document — the decrypting read, correct but unbatched — triples this bench's wall clock,
        from 94s to 238s, for information already sitting in the row.

        `_UNREADABLE` rather than 0 for a document that does not come back, so absence loses a
        comparison it could not take part in instead of winning it.
        """
        from .narrowing import phrase_stems

        wanted = [str(a) for a in artifact_ids]
        out = {a: _UNREADABLE for a in wanted}
        if not wanted or self._store_db is None:
            return out
        try:
            conn = self._store_db.artifacts.db.read()
            rows = conn.execute(
                "SELECT id, doc FROM vertex WHERE id IN (%s)" % ",".join("?" * len(wanted)),
                wanted,
            )
        except Exception:                    # noqa: BLE001 — a store read raises broadly
            return out
        for artifact_id, blob in rows:
            try:
                doc = json.loads(blob) if isinstance(blob, (str, bytes)) else (blob or {})
            except Exception:                # noqa: BLE001 — a malformed row is unreadable
                continue
            if not isinstance(doc, dict):
                continue
            text = " ".join(str(doc.get(field) or "")
                            for field in ("title", "description", "content"))
            if not text.strip():
                continue
            try:
                stems, _phrase = phrase_stems(" ".join(_WORDS.findall(text)))
            except Exception:                # noqa: BLE001
                continue
            out[str(artifact_id)] = len(set(stems) - asked)
        return out

    @staticmethod
    def _pool_for_reach(ordered: List[_Ranked], top_k: int) -> List[_Ranked]:
        """The ``top_k`` the reach arm will measure — chosen by how much of the QUESTION each
        candidate covers, not by which of them was written most recently.

        ``ordered`` arrives sorted by ``(stems, bigrams, updated_at, artifact_id)``. Only the
        first two say anything about the query; the rest is a tiebreak, and `_by_coverage` is
        explicit that it is one. So slicing at ``top_k`` slices INSIDE a tie whenever one spans
        the boundary — and a one-stem query ties EVERYWHERE by construction, which `_by_coverage`
        also says. A timestamp was deciding which equally-covered half the ranker could see:

            recall("what is a vaccine")   662 narrowed, every one at stems=1, horizon 50
                                          `wn-oewn-04524830-n`, titled `vaccine`, sat at
                                          position 88 and was never measured
                                          answer: degenerate / Caddo / visionary / informant

        Widening the page to 250 put it at rank 1 with the highest reach in the pool — so nothing
        was wrong with the ranking. The answer was discarded before it ran. Measuring the whole
        tied band instead is correct and unaffordable: 23/36 against 19, and 530 seconds for 36
        questions, because reach costs a propagation per distinct position.

        So the tie is broken instead, by which stems each artifact matched rather than how many.
        A stem carried by few of these artifacts distinguishes them; one carried by most does not.
        Both are ``stems=1``, and they are not equally good candidates.

        ## The frequencies are counted HERE, and that is the whole point

        The narrowing could count them far more cheaply — it holds one id set per stem already.
        It must not. Those sets are the raw index, including artifacts the light cone refuses, so
        a frequency taken there changes depending on whether a document the caller may not see
        exists. That is an existence oracle, `TestTheCountsAreNotAnExistenceOracle` forbids it,
        and it caught exactly this. Counting over ``ordered`` — which is post-meet — the number
        can only describe documents the caller may already read.

        `N` is this pool's own size, so every frequency is at most `N` and every weight is >= 0;
        a negative one would rank an artifact matching MORE stems lower, inverting coverage.
        """
        if not ordered:
            return []
        zero = Coverage()

        return sorted(ordered, key=MantleSseSearchAccessor._rarity_key(ordered),
                      reverse=True)[:top_k]

    @staticmethod
    def _rarity_key(ordered: List[_Ranked]):
        """A `row -> (stems, bigrams, rarity)` sort key, with the stem frequencies counted ONCE.

        Factored out because two orderings need it — the reach arm's pool and the coverage arm's
        answer — and two copies would drift.

        The counting is deliberately outside the returned closure. Recomputing the frequency map
        per row makes the sort O(n^2), and a narrowing here reaches 2,880 candidates on an ordinary
        question; one pass over `ordered` and a dict lookup per stem is the same number, arrived at
        once.

        Counted POST-MEET, over the authorized set, never in the narrowing: those id sets include
        artifacts the light cone refuses, so a frequency taken there changes depending on whether a
        document the caller may not see EXISTS, which `TestTheCountsAreNotAnExistenceOracle`
        forbids.
        """
        zero = Coverage()
        frequency: Dict[str, int] = {}
        for row in ordered:
            for stem in (row.coverage or zero).matched:
                frequency[stem] = frequency.get(stem, 0) + 1
        log_n = math.log(max(1, len(ordered)))

        def key(row: _Ranked):
            cover = row.coverage or zero
            rarity = sum(log_n - math.log(max(1, frequency.get(stem, 1)))
                         for stem in cover.matched)
            return (cover.stems, cover.bigrams, rarity)

        return key

    def _salient_measure(self):
        """The corpus's measure of which query terms carry the question — or ``None``.

        ``None`` means "not measurable here", and the narrowing then keeps every stem — the same
        behavior this path has when no measure is reachable at all. It is never a stop-list and
        never a constant: `match.salient_terms` keeps a term carrying at least the query's own
        mean IDF, read off the corpus's own document frequencies.

        Reached through the ranking's seam resolver rather than imported, because this layer may
        not import `ember` — and it is the same seam `ranking._reach_rank` resolves for
        `fired_field`, so recall and reach cannot end up measuring with two different corpora.
        That is the invariant `match.fired_field`'s "one measure, both paths" note states.
        """
        if self._store_db is None:
            return None
        try:
            from mantle.search import ranking
            seam = ranking._resolve_seam("match")          # noqa: SLF001 — the declared resolver
            measure = getattr(seam, "salient_terms", None)
        except Exception:  # noqa: BLE001 — no seam bound is the base install, not an error
            return None
        if measure is None:
            return None                                    # an older seam: keep every stem
        store = self._store_db
        return lambda terms: measure(terms, store)

    def _cell_principal(self, collection_id: Optional[str]) -> Optional[str]:
        """The collection's cell principal, or ``None`` when it cannot be resolved.

        Function-local import for the reason every `..principal` use is: `candidates()` is the
        only caller, and the module is not on the narrowing path.
        """
        if not collection_id:
            return None
        from ..principal import resolve_cell_principal

        try:
            return resolve_cell_principal(self._store_db, collection_id) or None
        except Exception:  # noqa: BLE001 — store reads raise broadly
            return None

    def _embed_or_none(self, query_text: str, parsed) -> Optional[list[float]]:
        """Embed the query (or its semantic-flagged terms) for the vector arm.

        Returns ``None`` if embedding fails — the recall is still a recall, ordered by
        recency instead of by cosine. ``~``-flagged terms, when the query carries any, are the
        text sent; otherwise the whole query is. That selection is the entire effect of the
        ``~`` modifier, and the resolution it reaches is the embeddings cache (Mantle runs no
        model), so a text nothing has embedded before returns ``None`` here — which is the
        common case, and the reason the recency path is an ordinary outcome rather than a rare
        one.
        """
        try:
            from mantle.search.query_parser import TermModifier
        except Exception:
            TermModifier = None  # type: ignore[assignment]

        text_for_embedding = query_text
        if TermModifier is not None:
            semantic_terms = [
                t.text for t in parsed.terms
                if getattr(t, "modifier", None) == TermModifier.SEMANTIC
            ]
            if semantic_terms:
                text_for_embedding = " ".join(semantic_terms)

        if not text_for_embedding.strip():
            return None
        try:
            results = self._embeddings([text_for_embedding])
        except Exception:
            logger.exception("MantleSseSearchAccessor: embedding failed")
            return None
        if not results:
            return None
        # The degraded answer is a non-empty list holding an EMPTY vector, not an
        # empty list: `Embeddings` pads its result 1:1 with its input
        # (`embeddings._pad`), so a text with no vector comes back as `[[]]`. A truth
        # test on the outer list therefore reads "we got something". Returning that
        # inner `[]` is a vector by type and not by content — `_order` guards the arm
        # on `is not None`, so it would pass the guard and `MantleQueryEngine.search`
        # would raise on the first line it reads. Degrading has to look like degrading.
        vector = results[0]
        if not vector:
            return None
        return list(vector)

    def _hydrate(self, ranked: Sequence[_Ranked]) -> list:
        """Read each artifact's metadata from the lattice and produce SearchHits.

        Neither index stores plaintext title / description / tags / content — those live in
        the ``artifacts`` collection. Hydration is one lattice ``get`` per ranked hit. Failed
        lookups produce a SearchHit with empty metadata fields rather than dropping the hit
        entirely (the doc may have been removed between ranking and hydration).

        ``metadata`` echoes whatever produced this hit's position, under a name that says which
        kind of number it is — never a fused shape with a per-arm score each and a ``source``
        naming which arm found the hit, since that would describe a result two arms had both
        scored, and there are not two arms. A cosine-ranked hit carries ``vector_score``; a
        coverage-ordered one carries ``matched_stems`` and ``matched_bigrams``, which is where
        the second count lives since ``score`` is one field; a recency-ordered one carries
        neither, because nothing measured it.
        """
        from mantle.search.types import SearchHit

        out: list = []
        for hit in ranked:
            doc = self._safe_get(self._store_db, hit.artifact_id, self._segment)
            if doc:
                # Decrypt inline content before it enters a search hit (raw-doc path).
                from mantle.db.doc_boundary import decrypt_artifact_content as _decrypt_artifact_content
                # Non-strict: a bad row degrades to no content, never ciphertext,
                # and never fails the whole result set.
                _decrypt_artifact_content(doc, strict=False)
            ctx = self._parse_context(doc)
            tags_raw = ctx.get("tags") or ctx.get("tags_canonical") or []
            tags = (
                [str(t) for t in tags_raw if str(t).strip()]
                if isinstance(tags_raw, list)
                else []
            )
            out.append(
                SearchHit(
                    doc_id=hit.artifact_id,
                    score=hit.score,
                    root_id=(
                        (doc or {}).get("root_id")
                        or hit.artifact_id
                    ),
                    version_id=(
                        (doc or {}).get("_key")       # lattice doc-id shape
                        or (doc or {}).get("id")      # lattice doc-id shape
                        or hit.artifact_id
                    ),
                    # Through `field_filters._title`, not re-spelled here: the two must agree
                    # about what a doc's title IS, or `title:x` selects hits whose echoed title is
                    # not `x`. That reader also finds a title a bulk ingest wrote top-level.
                    title=str(_title_of(doc, ctx) or ""),
                    description=str(ctx.get("description") or ""),
                    content=str((doc or {}).get("content") or ""),
                    tags=tags,
                    metadata=(
                        {"matched_stems": hit.coverage.stems,
                         "matched_bigrams": hit.coverage.bigrams}
                        if hit.coverage is not None
                        else {"vector_score": hit.score}
                    ),
                    # The ranker knows which collection it read this out of; the recency path
                    # was never in a collection-shaped conversation, so the doc answers for it.
                    collection_id=hit.collection_id or (doc or {}).get("collection_id"),
                    principal_id=hit.principal_id,
                    state=(doc or {}).get("state"),
                    is_head=None,
                    highlights=None,
                )
            )
        return out

    @staticmethod
    def _safe_get(store_db, artifact_id: str, segment: str = "committed"):
        """This segment's version of the lineage ``artifact_id`` names.

        ``artifact_id`` on a ranked hit is a ROOT id — both arms index by root — so this is a
        lineage read, not a document read. `find_version_in_state` resolves within the segment
        already being searched and falls back to the row at the root, which is the same answer
        a direct lookup gave for every artifact whose id is its own root.
        """
        from mantle.db.backend import find_version_in_state
        try:
            return find_version_in_state(store_db, artifact_id, segment)
        except Exception:  # noqa: BLE001 — store reads raise broadly
            try:
                return _raw_artifact(store_db, artifact_id)
            except Exception:  # noqa: BLE001
                return None

    @staticmethod
    def _parse_context(doc) -> dict:
        """One reader, shared with the filter predicate.

        `field_filters` resolves `title:` / `tags:` off the same `context` blob this hydrates
        from. Two readers of a field that arrives as a dict on some paths and a JSON string on
        others would eventually disagree, and the visible symptom would be `title:x` selecting
        hits whose echoed title is not `x`."""
        return parse_context(doc)


__all__ = [
    "MantleSseSearchAccessor",
    "RecallPlan",
    "plan_recall",
]
