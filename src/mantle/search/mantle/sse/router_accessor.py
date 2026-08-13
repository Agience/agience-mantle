"""MantleSseSearchAccessor — SearchResult-shaped adapter. LEXICAL NARROWS, SEMANTIC RANKS.

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

Adapter responsibilities:

1. Plan the query (:func:`plan_recall`): parse it, compile its ``field:value`` filters into a
   doc predicate, and reduce the string retrieval sees to the TERMS ONLY. `parsed_query`,
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

import logging
from typing import Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence

from mantle.search.embeddings import Embeddings
from mantle.search.field_filters import QueryFilterError
from mantle.search.field_filters import describe as describe_filters
from mantle.search.field_filters import parse_context
from mantle.search.types import ORDER_COVERAGE, ORDER_RECENCY, ORDER_SEMANTIC
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

    ``retrieval_text`` is the TERMS ONLY. The filter tokens are gone from it, and that is the
    half of this change that has nothing to do with narrowing: the accessor used to hand
    ``query.query_text`` to the lexical arm whole, so `type:pdf` reached the index as the two
    ordinary tokens `type` and `pdf` — it neither filtered nor got stripped, and it scored
    documents that merely contained the word "type". Retrieval now sees what the caller was
    actually searching for. ``@name:value`` controls are already absent for the same reason:
    the parser lifts them out of the term stream too, and they were only ever reaching the
    index because the raw string bypassed the parse.
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

        The fused accessor that used to sit here served :meth:`candidates` alone, and
        :meth:`candidates` now runs the same narrow-then-order path :meth:`search` does — so
        there is one retrieval story and these are its two halves.

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

        # The blind-token narrowing, compiled from the query's TERMS ONLY — the same string
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
        # It is ONE DOOR for four different facts: the light cone authorized nothing, the
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
            # Only the cosine path has a ceiling: it stops at the horizon above. The coverage
            # and recency paths order the whole narrowed set — both read a number the resolve
            # already produced, so neither has a retrieval budget to run out of — and their
            # `total` is exact however large it is.
            total_is_capped=(ordering == ORDER_SEMANTIC and len(ranked) >= horizon),
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

        Three answers, tried in that order:

        ``ORDER_SEMANTIC`` — a cosine did.

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
            return ORDER_COVERAGE, self._by_coverage(authorized, coverage)
        return ORDER_RECENCY, self._by_recency(authorized)

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
            # A CONFIGURATION STATE, NOT A FAULT, and the install default. Reported once per
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
          ONE-STEM QUERY IS THEREFORE BYTE-IDENTICAL TO :meth:`_by_recency`: every survivor
          scores 1, so the whole order is the tiebreak.

        ITERATION IS OVER ``authorized.artifact_ids`` AND NEVER OVER ``coverage``. The coverage
        map is the narrowing's own answer and is a SUPERSET of the surviving set — it can name
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

        SAME UNIVERSE AS :meth:`search`. Same plan, same field filter, same blind-token
        narrowing, same meet. The only differences are that nothing here orders the set by
        anything the query says, and nothing here hydrates it.

        Why it narrows, having previously not
        -------------------------------------
        It used to skip the token narrowing, on the argument that a flavor asks for the
        authorized set TO RANK WITHIN and narrowing it would decide part of the ranking on the
        flavor's behalf. That argument does not survive the narrowing becoming the ranking, and
        it did not survive contact with what the alternative actually returns.

        A narrowing is not a ranking; it is what the query MEANS. Skipping it does not hand a
        flavor an unranked candidate set, it hands it THE WHOLE LIGHT CONE for every query —
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
        # A named scope is a caller's own narrowing and is applied to the candidates as well as
        # to the contexts, because those contexts no longer reach a retrieval arm that would
        # have applied it for us.
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
        )
        if compiled is None:
            return None, coverage

        def _token_lookup(pairs):
            found = compiled(pairs)
            coverage.update(found)
            return found.keys()

        return _token_lookup, coverage

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
        kind of number it is. The fused shape it used to hold — a per-arm score each and a
        ``source`` naming which arm found the hit — described a result two arms had both
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
                    title=str(ctx.get("title") or ctx.get("name") or ""),
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
