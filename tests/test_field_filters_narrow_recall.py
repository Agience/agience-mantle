"""`field:value` narrows a recall — on both arms, inside `total`, and never upward.

What this file pins
===================

A parsed filter is a membership predicate over artifact ids — the same kind of object the light
cone produces, applied at the same place and through the same parameter (`authorized_artifacts`).
That is what makes "both arms honour it identically" a structural fact rather than two
implementations kept in step, and it is why the security property below holds by construction: the
predicate is only ever shown docs of artifacts the light cone already authorized, so its result
cannot be anything but a subset.

An inert filter is the failure this pins. If `router_accessor` handed `query.query_text` to the
lexical arm whole, `type:pdf` would reach the index as the two ordinary tokens `type` and `pdf` —
neither filtering nor stripped, and changing the ranking by scoring documents that merely contain
the word "type".

The stack here is the production one. Only the lattice and the light cone are doubles, exactly
as in `test_recall_honours_artifact_scoped_grants.py`; both indexes are real, both engines are
real, and the corpus is built so that EVERY authorized artifact is a genuine match on both arms.
A test whose corpus only matched the artifact it expected back would pass without a filter.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
from cryptography.fernet import Fernet

from mantle.search.field_filters import QueryFilterError, compile_filters
from mantle.search.mantle import MantleIndexer, MantleQueryEngine, OracleService
from mantle.search.mantle.oracle import FernetMasterKeyStore, KeyPurpose, KeyRequest
from mantle.search.mantle.sse.sqlite_stores import SqlitePostingStore
from mantle.search.mantle.sse.indexer import SseIndexer
from mantle.search.mantle.sse.narrowing import TokenNarrower
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor, plan_recall
from mantle.search.mantle.stores import InMemoryCellStore
from mantle.search.query_parser import parse_query

ALICE = "user-alice"
COLLECTION = "col-1"
CELL_PRINCIPAL = COLLECTION          # self-rooted, as in the production index path
DIM = 16

#: A term in every AUTHORIZED artifact, so the lexical arm matches all of them and any
#: narrowing seen below is the filter's doing rather than the corpus's.
TERM = "quasar"
#: A term in NO artifact. With a query vector supplied it makes a recall vector-only, which
#: is how the vector arm gets exercised against the same filter as the lexical one.
ABSENT_TERM = "zzznosuchword"

PDF = "art-pdf"
NOTE = "art-note"
IMAGE = "art-image"
#: Carries the words `type` and `pdf` in its TEXT and is itself a PDF, so the field filter
#: does not exclude it. The only thing that can keep it out of a `quasar type:...` recall is
#: the filter tokens not reaching the index. See `TestFilterSyntaxDoesNotReachTheIndex`.
#: It also carries a URL, which is the other half: `https://example.com` is not a filter, so
#: it must reach the index AS A SEARCH and find this. See `TestOnlyAKnownFieldMakesAFilter`.
DECOY = "art-decoy"
#: Exists, is indexed, is a PDF — and is NOT in Alice's light cone. The security property.
SECRET = "art-secret"

_TEXT = {
    PDF:    f"{TERM} quarterly budget catalogue",
    NOTE:   f"{TERM} loose notes on the budget",
    IMAGE:  f"{TERM} cover plate",
    DECOY:  "the type of pdf documents filed here, see https://example.com",
    SECRET: f"{TERM} the acquisition nobody shared",
}

_META = {
    PDF:    ("application/pdf",  ["budget", "q1"], "Quarterly Budget", "2024-01-01T00:00:00Z"),
    NOTE:   ("text/plain",       ["budget"],       "Budget Notes",     "2025-06-01T00:00:00Z"),
    IMAGE:  ("image/png",        ["media"],        "Cover Plate",      "2026-01-01T00:00:00Z"),
    DECOY:  ("application/pdf",  ["misc"],         "Filing Guide",     "2025-01-15T00:00:00Z"),
    SECRET: ("application/pdf",  ["budget"],       "Acquisition",      "2025-03-01T00:00:00Z"),
}

AUTHORIZED = {PDF, NOTE, IMAGE, DECOY}
PDFS_ALICE_CAN_READ = {PDF, DECOY}


def _act(principal_id: str, principal_type: str = "user") -> None:
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal

    set_acting_principal(
        ActingPrincipal(principal_id=principal_id, principal_type=principal_type,
                        source="field-filter-test")
    )


def _write_request():
    """A write: a read action never mints a master key, so the corpus must be indexed under
    one or every query below would see an empty index and pass for the wrong reason."""
    _act(CELL_PRINCIPAL, "principal")
    return KeyRequest(requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
                      requester_type="principal", action="update")


class _AliceHoldsTheCollectionKey:
    """Key custody, decided the way `LightConeGrantVerifier` decides it — per collection.

    Alice legitimately holds this key. Everything the tests below assert therefore happens
    strictly downstream of correct custody: a verifier that denied her would hide whether the
    filter did anything.
    """

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action) -> bool:
        if requester_id == principal_id:
            return True
        return (requester_id == ALICE and principal_id == CELL_PRINCIPAL
                and collection_id in (None, COLLECTION))


class _FakeArtifacts:
    def __init__(self, docs: dict) -> None:
        self._docs = docs

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        return self._docs.get(artifact_id)


class _FakeGraph:
    def edges_of(self, node, label=None, direction="out", limit=None) -> list:
        return []


def _doc(artifact_id: str) -> dict:
    content_type, tags, title, created = _META[artifact_id]
    return {
        "id": artifact_id,
        "root_id": artifact_id,
        "collection_id": COLLECTION,
        "content_type": content_type,
        "created_by": "user-bob",
        "created_time": created,
        "modified_time": created,
        "state": "committed",
        "context": {"title": title, "tags": tags, "description": "a filed artifact"},
        "content": _TEXT[artifact_id],
    }


class _FakeStoreDB:
    def __init__(self) -> None:
        docs = {a: _doc(a) for a in _META}
        docs[COLLECTION] = {"id": COLLECTION, "root_id": COLLECTION, "context": {}}
        self.artifacts = _FakeArtifacts(docs)
        self.graph = _FakeGraph()


class _GrantedResource:
    """A grant as `mask_of` reads one: a resource, an effect, and the action flags."""

    def __init__(self, resource_id):
        self.resource_id = resource_id
        self.effect = "allow"
        for _a in ("create", "read", "update", "delete", "evict", "invoke", "add", "share",
                   "admin"):
            setattr(self, "can_" + _a, True)


class _FakeLightCone:
    def __init__(self, authorized: set) -> None:
        self._authorized = set(authorized)

    def resolve(self, principal_id, action="read", *, principal_type="user") -> set:
        return set(self._authorized) if principal_id == ALICE else set()

    def _grants_for(self, principal_id: str, principal_type: str = "user"):
        """The same reach this double already describes, expressed as grants.

        `resolve_authorized_scope` authorizes each candidate by walking up from it and testing the
        resources on the way against the caller's grants, so a double saying "these ids are
        authorized" says it by granting them — each id its own granted resource, which is what an
        artifact-scoped grant looks like. Derived from `resolve` rather than restated, so whatever
        that method gates on carries here too.
        """
        return [_GrantedResource(a) for a in self.resolve(principal_id,
                                                          principal_type=principal_type)]


@pytest.fixture(autouse=True)
def _live_anchorset():
    """One anchor: every dim-16 vector routes to a single cell, so decrypting it hands the
    query every chunk in the collection. The adversarial routing for a narrowing test."""
    from mantle.search.anchors import store
    from mantle.search.anchors.anchorset import AnchorSet
    from mantle.search.anchors.repo import InMemoryAnchorRepo

    store.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", DIM)
    aset.add_text("anchor-0", np.ones(DIM, dtype=np.float32))
    store.save_live_anchorset(aset)
    yield
    store.set_anchor_repo(None)


def _vec(seed: int) -> list:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DIM).tolist()


#: One query direction and near-identical doc vectors, so every artifact is a real vector
#: match. Any narrowing on the vector arm is the filter, never similarity.
QUERY_VEC = _vec(11)
_DOC_VECS = {a: _vec(11) for a in _META}


@pytest.fixture
def stack(tmp_path):
    oracle = OracleService(
        FernetMasterKeyStore(Fernet(Fernet.generate_key())),
        grant_verifier=_AliceHoldsTheCollectionKey(),
    )
    root = os.path.join(str(tmp_path), "sse-index")
    postings = SqlitePostingStore(os.path.join(root, "mantle-sse.db"))
    cells = InMemoryCellStore()

    sse_indexer = SseIndexer(oracle, postings)
    vec_indexer = MantleIndexer(oracle, cells)
    for artifact_id in _META:
        sse_indexer.index_artifact(
            CELL_PRINCIPAL, COLLECTION, artifact_id,
            {"content": _TEXT[artifact_id]}, _write_request(),
        )
        vec_indexer.index_artifact(
            CELL_PRINCIPAL, COLLECTION,
            [{"artifact_id": artifact_id, "chunk_id": 0,
              "embedding": _DOC_VECS[artifact_id], "text": _TEXT[artifact_id]}],
            _write_request(),
        )
    return {"vector": MantleQueryEngine(oracle, cells),
            "narrower": TokenNarrower(oracle, postings)}


class _NoVector:
    """An embedder whose answer is a vector by type and not by content.

    `Embeddings` pads its result 1:1 with its input, so a text nothing has embedded comes back
    as `[[]]` — this reproduces that shape, and `_embed_or_none` turns it into `None`, which is
    the lexical-only path."""

    def __call__(self, texts):
        return [[] for _ in texts]


def _accessor(stack, *, authorized=AUTHORIZED, embeddings=None) -> MantleSseSearchAccessor:
    """The production composition: the postings narrow, the cells rank.

    Both are the real objects over the real encrypted indexes; only the lattice and the light
    cone are doubles. `candidates()` runs on the same two, which is why there is no third.
    """
    return MantleSseSearchAccessor(
        _FakeLightCone(authorized),
        store_db=_FakeStoreDB(),
        embeddings=embeddings or (lambda texts: [list(QUERY_VEC) for _ in texts]),
        narrower=stack["narrower"],
        ranker=stack["vector"],
    )


def _query(text: str, *, size: int = 20, user_id: str = ALICE, vector=None):
    """A recall request with the acting principal bound as the router would bind it — the
    oracle refuses a key to a requester that is not the authenticated actor.

    ``vector`` supplies the ranker's query directly, which is how a recall reaches the
    ranking path with NO search terms: a filter narrows and the cosine orders what is left.
    """
    from mantle.search.types import SearchQuery

    _act(user_id, "user")
    return SearchQuery(
        query_text=text, user_id=user_id, size=size,
        query_embedding=list(vector) if vector is not None else None,
    )


def _ids(result) -> set:
    return {h.doc_id for h in result.hits}


# ---------------------------------------------------------------------------
# The corpus is real. Without these, everything below could pass vacuously.
# ---------------------------------------------------------------------------

class TestTheCorpusWouldOtherwiseReturnEverything:
    def test_the_terms_narrow_to_every_artifact_carrying_them(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM))
        assert _ids(result) == {PDF, NOTE, IMAGE}, (
            "every authorized artifact but the decoy carries the term and must match, or a "
            "filtered result below proves nothing"
        )

    def test_the_ranker_reaches_every_authorized_artifact(self, stack):
        """Two complementary filters, each partitioning the authorized set in half.

        Read together they show the ranker can reach all four artifacts and that WHICH of
        them comes back is decided by the filter alone — the corpus is one query direction
        with near-identical document vectors, so similarity cannot be doing the choosing.

        These queries carry a vector and no terms, which is how the ranking path is reached with
        nothing narrowing it lexically. A term absent from the corpus narrows the recall to
        nothing rather than leaving the vector to carry it, which is what
        `TestTermsNarrowBeforeAnythingRanks` measures directly.

        Two filters rather than one unfiltered query because of `MantleQueryEngine._beacon_cut`:
        it reads where a result set's spectrum stops, and over four near-identical vectors it
        legitimately stops at one. Its documented passthrough is a pool of two or fewer ("two
        candidates have no spectrum to read"), which each half here is. That is the engine
        being itself, not the filter; asserting on a four-way pool would be asserting on the
        cut's threshold rather than on the filter.
        """
        acc = _accessor(stack)
        pdfs = acc.search(_query("type:application/pdf", vector=QUERY_VEC))
        others = acc.search(_query("type:text/plain,image/png", vector=QUERY_VEC))

        assert _ids(pdfs) == {PDF, DECOY}
        assert _ids(others) == {NOTE, IMAGE}
        assert _ids(pdfs) | _ids(others) == AUTHORIZED
        for result in (pdfs, others):
            assert result.ordering == "semantic"
            assert all(h.score is not None for h in result.hits), (
                "a cosine ordered these, so every hit carries the number that ordered it"
            )


class TestTermsNarrowBeforeAnythingRanks:
    """The switch itself: the terms decide membership, and they decide it first.

    The terms are the narrowing, so `zzznosuchword type:application/pdf` returns nothing, and
    returns it before a cell is decrypted. Left to the ranking arm, a term in no document would
    bring back every PDF Alice can read.
    """

    def test_a_term_in_no_document_narrows_the_recall_to_nothing(self, stack):
        result = _accessor(stack).search(_query(f"{ABSENT_TERM} type:application/pdf"))
        assert result.hits == []
        assert result.total == 0

    def test_a_supplied_vector_does_not_re_admit_what_the_terms_excluded(self, stack):
        """A vector orders a set; it cannot add to one. Narrowing and ranking answer
        different questions, so the ranking has no vocabulary for widening."""
        result = _accessor(stack).search(_query(ABSENT_TERM, vector=QUERY_VEC))
        assert result.hits == []

    def test_the_decoy_is_reachable_by_its_own_words(self, stack):
        """The control. `ABSENT_TERM` narrowing to nothing has to be the term's doing, not a
        narrowing that never matches anything."""
        result = _accessor(stack, embeddings=_NoVector()).search(_query("filed"))
        assert DECOY in _ids(result)

    def test_the_secret_is_indexed_and_reachable_with_the_right_key(self, stack):
        """It is a real document in the same index, not an absent one. Otherwise the
        security test below would be proving that nothing exists."""
        _act(CELL_PRINCIPAL, "principal")
        lookup = stack["narrower"].lookup_for(
            TERM,
            KeyRequest(requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
                       requester_type="principal", action="read"),
        )
        assert SECRET in set(lookup([(CELL_PRINCIPAL, COLLECTION)]))


# ---------------------------------------------------------------------------
# Both arms, one filter
# ---------------------------------------------------------------------------

class TestBothOrderingsHonourTheSameFilter:
    """The filter is applied to the authorized set before anything orders it, so it is
    ARM-INDEPENDENT by construction — the same one parameter, the same one place, whether a
    cosine or the clock decides what order the survivors come back in. It is the claim, so it
    is measured on both."""

    def test_a_coverage_ordered_recall_honours_the_filter(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(
            _query(f"{TERM} type:application/pdf")
        )
        assert result.ordering == "coverage", "no cosine can order this one"
        assert _ids(result) == {PDF}

    def test_a_cosine_ordered_recall_honours_the_filter(self, stack):
        result = _accessor(stack).search(_query("type:application/pdf", vector=QUERY_VEC))
        assert result.ordering == "semantic"
        assert _ids(result) == PDFS_ALICE_CAN_READ

    def test_neither_ordering_returns_anything_the_filter_excludes(self, stack):
        """One filter, run once per ordering. They present different sets — one narrowed by
        the term as well, one not — so the claim that holds for both is that neither can
        return an artifact the filter excluded.

        They agree because they are told the same thing: the filter narrows
        `AuthorizedScope.artifact_ids` inside the light-cone resolve, before either ordering
        exists. There is no per-ordering filter implementation to drift.
        """
        excluded = AUTHORIZED - PDFS_ALICE_CAN_READ
        coverage = _accessor(stack, embeddings=_NoVector()).search(
            _query(f"{TERM} type:application/pdf")
        )
        semantic = _accessor(stack).search(_query("type:application/pdf", vector=QUERY_VEC))

        assert _ids(coverage) <= PDFS_ALICE_CAN_READ
        assert _ids(semantic) <= PDFS_ALICE_CAN_READ
        assert not (_ids(coverage) | _ids(semantic)) & excluded
        # And between them they do reach the filtered set, so neither is empty by accident.
        assert _ids(coverage) | _ids(semantic) == PDFS_ALICE_CAN_READ

    def test_a_recall_that_both_narrows_and_ranks_honours_it(self, stack):
        """Terms AND a filter AND a vector: the two narrowings meet, then the cosine orders
        what is left. The decoy is a PDF and passes the filter, and does not carry the term,
        so it is not here — which is the narrowing doing its job on the live path."""
        result = _accessor(stack).search(_query(f"{TERM} type:application/pdf"))
        assert _ids(result) == {PDF}
        assert result.ordering == "semantic"

    def test_candidates_returns_the_same_universe_search_does(self, stack):
        """`candidates()` is the chokepoint every external search flavor ranks within, and it
        now runs the same narrowing `search()` does — the filter AND the query's terms.

        It used to run the filter alone, on the argument that narrowing by the terms would
        decide part of the ranking for the flavor. What that actually returned for
        `quasar type:application/pdf` was `{PDF, DECOY}`: the decoy is a PDF and passes the
        filter, and does not carry the term. A flavor asking what matched `quasar` was handed
        a document that does not contain it. `search()` refuses to answer that way — it raises
        rather than widen to everything authorized — and one accessor cannot hold both
        positions. What `candidates()` still declines to state is the ORDER.
        """
        out = _accessor(stack).candidates(_query(f"{TERM} type:application/pdf"))
        ranked = _accessor(stack).search(_query(f"{TERM} type:application/pdf"))
        assert {c["artifact_id"] for c in out["candidates"]} == {PDF}
        assert {c["artifact_id"] for c in out["candidates"]} == _ids(ranked)
        assert DECOY not in {c["artifact_id"] for c in out["candidates"]}, (
            "the decoy passes the filter and does not carry the term; a flavor asking what "
            "matched the term must not be handed it"
        )

    def test_candidates_publishes_no_score_of_any_kind(self, stack):
        """The order is the flavor's to decide, so nothing here states one.

        `sse_score` was a BM25 score, `rrf_score` a rank-fusion constant's output and `source`
        a which-arm-found-it flag; none of those quantities exists. They are ABSENT rather than
        null, because a null would say "this candidate had no BM25 score" where the truth is
        that no BM25 score exists anywhere. The coverage counts `search()` orders by are
        computed on this path too, and are deliberately not published.
        """
        out = _accessor(stack).candidates(_query(f"{TERM} type:application/pdf"))
        assert out["candidates"], "nothing came back — the assertion below would be vacuous"
        for candidate in out["candidates"]:
            assert set(candidate) == {"artifact_id", "collection_id", "principal_id"}
        assert out["model_id"] is None


# ---------------------------------------------------------------------------
# THE SECURITY PROPERTY
# ---------------------------------------------------------------------------

class TestAFilterOnlyEverNarrows:
    def test_a_filter_naming_an_unreadable_artifact_matches_nothing(self, stack):
        result = _accessor(stack).search(_query(f"{TERM} id:{SECRET}"))
        assert result.hits == []
        assert result.total == 0

    def test_that_answer_is_identical_to_a_filter_matching_nothing(self, stack):
        """The property, stated as an indistinguishability.

        `id:art-secret` names a real, indexed, matching artifact Alice may not read.
        `id:art-no-such-thing` names nothing at all. If the two answers differed in ANY
        observable — hits, `total`, `applied_filters`, `ordering`, or an error that said
        "no such value" in one case and "not authorized" in the other — the filter would be
        an oracle for the existence of artifacts outside the light cone.
        """
        acc = _accessor(stack)
        unreadable = acc.search(_query(f"{TERM} id:{SECRET}"))
        nonexistent = acc.search(_query(f"{TERM} id:art-no-such-thing"))

        assert unreadable.hits == nonexistent.hits == []
        assert unreadable.total == nonexistent.total == 0
        assert unreadable.ordering == nonexistent.ordering
        assert unreadable.applied_filters == [f"id:{SECRET}"]
        assert nonexistent.applied_filters == ["id:art-no-such-thing"]
        # The only difference between the two responses is the caller's own echoed input.
        assert (unreadable.total, unreadable.hits, unreadable.ordering) == (
            nonexistent.total, nonexistent.hits, nonexistent.ordering,
        )

    def test_a_filter_cannot_widen_past_the_light_cone(self, stack):
        """A filter that is TRUE of the secret and of nothing Alice holds still returns
        nothing. There is no filter value that adds an id the resolver did not."""
        result = _accessor(stack, authorized={NOTE}).search(
            _query(f"{TERM} type:application/pdf")
        )
        assert result.hits == []

    def test_a_negated_filter_cannot_widen_either(self, stack):
        """Negation is where a filter would most plausibly leak: `!x` over a set the caller
        does not own could be read as "everything but x". It is not — it is "everything I
        may read, but x"."""
        result = _accessor(stack, authorized={NOTE}).search(_query(f"{TERM} !type:image/png"))
        assert _ids(result) == {NOTE}

    def test_the_predicate_is_never_shown_an_unauthorized_doc(self, stack):
        """The mechanism behind the three tests above, pinned directly.

        `resolve_authorized_scope` evaluates the predicate only inside its loop over the light
        cone's own ids, so an unauthorized doc is never read and never offered to it. This
        records that with a predicate that raises if it ever sees one.
        """
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        seen: set = set()

        def _spy(doc: dict) -> bool:
            seen.add(doc.get("id"))
            return True

        scope = resolve_authorized_scope(
            _FakeStoreDB(), ALICE, lightcone=_FakeLightCone(AUTHORIZED),
            artifact_predicate=_spy,
        )
        assert SECRET not in seen
        assert seen <= AUTHORIZED
        assert scope.artifact_ids <= AUTHORIZED


# ---------------------------------------------------------------------------
# Filter syntax must not reach the index as search terms
# ---------------------------------------------------------------------------

class TestFilterSyntaxDoesNotReachTheIndex:
    def test_the_terms_only_string_is_what_retrieval_sees(self):
        plan = plan_recall("budget type:pdf @lang:en +urgent")
        assert plan.retrieval_text == "budget urgent", (
            "retrieval must see the terms only — no filter tokens, no @controls"
        )

    def test_a_document_matching_only_the_filter_tokens_is_not_recalled(self, stack):
        """The decoy is a PDF, so the FIELD FILTER does not exclude it. Its text carries
        `type` and `pdf` and no query term. It came back before this change because the raw
        string reached the index and those two tokens scored it."""
        result = _accessor(stack, embeddings=_NoVector()).search(
            _query(f"{TERM} type:application/pdf")
        )
        assert DECOY not in _ids(result)
        assert _ids(result) == {PDF}

    def test_an_unfiltered_query_is_unchanged(self):
        """A query with no filter must reach retrieval exactly as before."""
        assert plan_recall("machine learning").retrieval_text == "machine learning"
        assert plan_recall("machine learning").predicate is None


# ---------------------------------------------------------------------------
# Only a KNOWN field makes a filter
# ---------------------------------------------------------------------------

#: The ordinary searches a colon used to break, each a real thing a caller types: a URL, a
#: clock time, a Windows path, an aspect ratio. None of them names a field.
NOT_FILTERS = [
    "https://example.com",
    "meeting at 3:30",
    r"C:\Users\example",
    "ratio 16:9",
]


class TestOnlyAKnownFieldMakesAFilter:
    """`word:value` is a filter only when `word` is a field the resolver owns.

    A colon is an ordinary character in ordinary text, and reading every one of them as
    filter syntax turned the four searches above into 400s — a filter feature breaking
    queries that were never trying to filter. The grammar asks
    `field_filters.is_filter_field`, so the words that make a filter are exactly the words
    that can be resolved, and everything else stays a term.
    """

    @pytest.mark.parametrize("query", NOT_FILTERS)
    def test_it_parses_as_terms_and_not_as_a_filter(self, query):
        parsed = parse_query(query)
        assert parsed.filters == []
        assert parsed.has_topics()

    @pytest.mark.parametrize("query", NOT_FILTERS)
    def test_it_plans_a_recall_instead_of_raising(self, query):
        plan = plan_recall(query)
        assert plan.predicate is None
        assert plan.retrieval_text == query, (
            "the token reaches retrieval whole: the parser neither strips the colon, nor "
            "splits on it, nor quotes around it"
        )

    def test_what_a_url_actually_reaches_the_index_as(self):
        """Measured, not inferred from the grammar.

        The parser keeps the URL as ONE term. The SSE analysis pipeline then splits it the
        way it splits every string — on non-word characters — into `http`, `exampl`, `com`.
        That is not the parser giving up: index time and query time call the same `tokenize`,
        so a document containing the URL writes those same three stems and the match is
        exact. A URL is not carried as a single token anywhere in this store, and the query
        path does not claim otherwise.
        """
        from mantle.search.mantle.sse.tokenizer import tokenize

        plan = plan_recall("https://example.com")
        assert [t.text for t in plan.parsed.terms] == ["https://example.com"]
        assert tokenize(plan.retrieval_text) == ["http", "exampl", "com"]
        assert tokenize("see https://example.com for details")[1:4] == \
            ["http", "exampl", "com"], "the same three stems the indexer wrote"

    def test_a_url_query_recalls_the_document_carrying_it(self, stack):
        """End to end on the production lexical arm: the URL is SEARCHABLE, not merely
        un-refused. The decoy is the one artifact whose text carries it."""
        result = _accessor(stack, embeddings=_NoVector()).search(_query("https://example.com"))
        assert _ids(result) == {DECOY}

    def test_a_misspelled_field_is_a_search_term_and_not_an_error(self):
        """THE ACCEPTED COST, pinned so it stays deliberate.

        `titel` is not a field, so `titel:foo` searches for that text and finds nothing
        instead of naming the typo. Recoverable — no results, look again — against a hard
        failure on every legitimate colon-bearing search, which is the trade. The field list
        is on the endpoint and in the README so a caller with no results can work out why.
        """
        plan = plan_recall("titel:foo")
        assert plan.predicate is None
        assert plan.retrieval_text == "titel:foo"

    def test_the_accepted_cost_has_no_warning_channel(self):
        """No did-you-mean and no warning field. Guessing intent from a colon is the
        behaviour being removed; half-removing it would leave the guess in a channel nobody
        reads."""
        parsed = parse_query("titel:foo")
        assert parsed.corrections == []
        assert parsed.controls == {}


class TestOneRosterForTheGrammarAndTheResolver:
    """"Is this a field" and "can I resolve this field" are one mapping, read twice.

    Two lists is how this class of bug returns. A word the grammar calls a field and the
    resolver cannot resolve is a 400 on an ordinary search; a word the resolver owns that the
    grammar does not is a filter silently searched as text. Neither is representable while
    there is one roster, and these fail if a second one ever appears.
    """

    def _roster(self) -> set:
        from mantle.search.field_filters import (
            FIELD_ALIASES, FILTERABLE_FIELDS, REFUSED_FIELDS,
        )

        return set(FILTERABLE_FIELDS) | set(FIELD_ALIASES) | set(REFUSED_FIELDS)

    def test_every_field_the_resolver_owns_parses_as_a_filter(self):
        roster = self._roster()
        assert roster, "an empty roster would make every assertion here vacuous"
        for name in sorted(roster):
            assert parse_query(f"{name}:v term").filters, (
                f"`{name}` is a field to the resolver and not to the grammar, so a caller "
                f"filtering on it would be silently searching for the text instead"
            )

    def test_the_grammar_and_the_resolver_agree_word_for_word(self):
        """Both directions at once: the parser makes a filter for exactly the words
        `is_filter_field` claims, including the near-misses that made this necessary."""
        from mantle.search.field_filters import is_filter_field

        words = sorted(self._roster()) + [
            "colour", "titel", "typ", "tags_", "https", "c", "3", "16", "note",
        ]
        for word in words:
            assert bool(parse_query(f"{word}:v term").filters) is is_filter_field(word), (
                f"`{word}:` parses one way and resolves the other"
            )

    def test_a_field_added_to_the_resolver_is_a_field_to_the_grammar(self):
        """The single-sourcing itself, made falsifiable.

        A field is added to the resolver's own mapping and NOTHING is told to the parser. If
        the parser ever grows a copy of the roster, this is the test that fails.
        """
        from mantle.search import field_filters

        assert parse_query("colour:red term").filters == []

        added = field_filters._Field(lambda d, c: d.get("colour"))
        with patch.dict(field_filters.FILTERABLE_FIELDS, {"colour": added}):
            parsed = parse_query("colour:red term")
            assert [str(f) for f in parsed.filters] == ["colour:red"]
            assert compile_filters(parsed.filters) is not None, (
                "and the resolver resolves the very field the grammar just accepted"
            )

        assert parse_query("colour:red term").filters == []

    def test_a_refusal_added_to_the_resolver_is_a_field_to_the_grammar_too(self):
        """`REFUSED_FIELDS` is part of the roster on purpose: a field refused WITH A REASON
        has to reach the resolver to be refused by name. A grammar that only knew the
        filterable half would turn every refusal into a silent search."""
        from mantle.search import field_filters

        with patch.dict(field_filters.REFUSED_FIELDS, {"colour": "no colour is stored"}):
            with pytest.raises(QueryFilterError) as exc:
                plan_recall("colour:red term")
            assert "colour" in str(exc.value)

    def test_an_alias_is_a_field_because_what_it_aliases_is(self):
        """`is_filter_field` canonicalizes before it looks, so the alias row needs no second
        mention. Canonicalization of the PARSED filter still happens where it did — `tags` in
        the parser, `content_type` in `describe` — and neither is a roster."""
        from mantle.search.field_filters import describe

        assert [str(f) for f in parse_query("type:pdf x").filters] == ["type:pdf"]
        assert describe(parse_query("type:pdf x").filters) == ["content_type:pdf"]
        assert [str(f) for f in parse_query("tag:budget x").filters] == ["tags:budget"]


# ---------------------------------------------------------------------------
# The echo is what was APPLIED
# ---------------------------------------------------------------------------

class TestTheEchoReportsWhatNarrowed:
    def test_applied_filters_lists_the_filters_that_ran(self, stack):
        result = _accessor(stack).search(_query(f"{TERM} type:application/pdf tag:budget"))
        assert result.applied_filters == ["content_type:application/pdf", "tags:budget"]

    def test_an_unfiltered_recall_applies_none(self, stack):
        assert _accessor(stack).search(_query(TERM)).applied_filters == []

    def test_nothing_reaches_the_echo_without_being_applied(self, stack):
        """The echo cannot overstate, because the only other outcome is a raised request.

        A filter either compiles — and then it narrowed, and appears here — or it fails the
        whole recall. There is no third state in which one is reported and not applied, which
        is the state the old `parsed_query` echo described.

        Written against `state:`, a field this path refuses, rather than an invented word: an
        invented word is not a field, so it is a search term and never reaches the echo to
        begin with. The claim is about filters, so it is made with one.
        """
        with pytest.raises(QueryFilterError):
            _accessor(stack).search(_query(f"{TERM} state:draft"))


# ---------------------------------------------------------------------------
# Operators: what is in
# ---------------------------------------------------------------------------

def _matching(query: str, docs=None) -> set:
    """The ids a query's filters select, evaluated directly against the docs."""
    predicate = compile_filters(parse_query(query).filters)
    assert predicate is not None
    pool = docs if docs is not None else list(_META)
    return {a for a in pool if predicate(_doc(a))}


class TestSupportedOperators:
    def test_equals_is_case_insensitive(self):
        assert _matching("type:APPLICATION/PDF") == {PDF, DECOY, SECRET}
        assert _matching("tag:BUDGET") == _matching("tag:budget")

    def test_equals_accepts_an_any_of_list(self):
        assert _matching("type:text/plain,image/png") == {NOTE, IMAGE}

    def test_exact_is_case_sensitive_and_taken_whole(self):
        assert _matching('title:="Quarterly Budget"') == {PDF}
        assert _matching('title:="quarterly budget"') == set(), (
            "`field:=` is the case-sensitive operator; folding it would leave no way to "
            "select a value that differs from another only by case"
        )

    def test_negation_removes_matches(self):
        assert _matching("!type:application/pdf") == {NOTE, IMAGE}

    def test_a_list_field_matches_any_member(self):
        assert _matching("tag:budget") == {PDF, NOTE, SECRET}
        assert _matching("tags:q1") == {PDF}

    def test_ranges_apply_to_timestamps(self):
        assert _matching("created_at:>2025-01-01") == {NOTE, IMAGE, DECOY, SECRET}
        assert _matching("created_at:<2025-01-01") == {PDF}

    def test_filters_conjoin(self):
        assert _matching("type:application/pdf tag:budget") == {PDF, SECRET}

    def test_the_alias_pairs_resolve_to_one_field(self):
        assert _matching("type:image/png") == _matching("content_type:image/png")
        assert _matching("tag:media") == _matching("tags:media")


# ---------------------------------------------------------------------------
# Operators and fields: what is out, and how loudly
# ---------------------------------------------------------------------------

class TestUnsupportedInputIsRefusedByName:
    """What is refused is what a caller MEANT as a filter — a word that is a field.

    A word that is not a field is not an unsupported filter, it is a search term, and the
    tests for that live in `TestOnlyAKnownFieldMakesAFilter`. Everything here is a genuine
    field, and narrowing which words are fields does not soften any of it.
    """

    def test_an_unknown_field_reaching_the_resolver_directly_is_refused_and_named(self):
        """No query string can produce this — the parser only builds a `FieldFilter` for a
        field `is_filter_field` knows — so it is the contract for a programmatic caller that
        constructs one itself. It stays loud for the reason every refusal here is loud: a
        filter that is dropped returns a result set that looks like an answer to a question
        nobody asked."""
        from mantle.search.query_parser import FieldFilter

        with pytest.raises(QueryFilterError) as exc:
            compile_filters([FieldFilter(field="colour", value="red")])
        assert "colour" in str(exc.value)
        assert "content_type" in str(exc.value), "the error must list what IS filterable"

    def test_a_refused_field_is_not_silently_dropped(self):
        """The bug this replaces, stated as its own test: the old path kept the filter in
        `ParsedQuery.filters`, read it nowhere, and answered as if the caller had not asked.

        `content:` rather than an invented word, because an invented word is not a filter at
        all now. `content` IS a field a caller can reasonably ask for, so it is the case where
        dropping would still be possible and must not happen."""
        with pytest.raises(QueryFilterError):
            plan_recall("budget content:secret")

    def test_a_refused_field_does_not_fall_through_to_a_search_term(self):
        """The other half, and the line between the two behaviours. Falling through is right
        for a word that is not a field and wrong for one that is: `content:secret` names a real
        field with a real reason it cannot be answered, and searching for the text instead
        would answer a different question and look like an answer to this one."""
        with pytest.raises(QueryFilterError):
            plan_recall("content:secret")

    def test_quoting_still_forces_a_term(self):
        """Quoting is how a caller searches for a FIELD's name literally — the one case that
        still needs an escape, now that a non-field keeps its colon on its own."""
        plan = plan_recall('"type:pdf"')
        assert plan.predicate is None
        assert plan.retrieval_text == "type:pdf"

    def test_the_semantic_field_operator_is_refused_and_named(self):
        with pytest.raises(QueryFilterError) as exc:
            compile_filters(parse_query("tag:~ai").filters)
        assert "~" in str(exc.value)
        assert "models" in str(exc.value), (
            "tag expansion needs an embedding of the value and Mantle runs no models — the "
            "reason belongs in the message"
        )

    def test_a_range_on_an_unordered_field_is_refused_and_named(self):
        with pytest.raises(QueryFilterError) as exc:
            compile_filters(parse_query("title:>m").filters)
        assert ">" in str(exc.value)
        assert "created_at" in str(exc.value), "it must say which fields DO take a range"

    def test_size_is_refused_with_a_reason(self):
        """`size:>10MB` parses — `tests/test_search.py` covers that — and cannot be answered:
        an artifact row carries no size."""
        with pytest.raises(QueryFilterError) as exc:
            compile_filters(parse_query("files size:>10MB").filters)
        assert "size" in str(exc.value)

    def test_content_is_refused_because_it_is_encrypted(self):
        with pytest.raises(QueryFilterError) as exc:
            compile_filters(parse_query("content:secret").filters)
        assert "encrypted" in str(exc.value)


class TestStateIsARequestFieldNotAFilter:
    """The overlap, resolved in one direction.

    `state` selects the index SEGMENT — `committed`, `draft` and `archived` are separately
    keyed encrypted trees under distinct object-storage prefixes, chosen when the accessor is
    built and before any query runs. A `state:draft` filter could not narrow a committed-segment
    recall to drafts: no draft is in the tree being read, so there is nothing to narrow. Two
    spellings of one control where only one of them can work is the incoherence, so the query
    spelling is refused and pointed at the request field.
    """

    def test_a_state_filter_is_refused(self):
        with pytest.raises(QueryFilterError) as exc:
            compile_filters(parse_query("budget state:draft").filters)
        assert "state" in str(exc.value)
        assert "segment" in str(exc.value)

    def test_the_refusal_names_the_request_field_that_does_work(self):
        with pytest.raises(QueryFilterError) as exc:
            compile_filters(parse_query("state:archived").filters)
        assert "request field" in str(exc.value)

    def test_state_is_not_in_the_filterable_registry(self):
        from mantle.search.field_filters import FILTERABLE_FIELDS, filterable_field_names

        assert "state" not in FILTERABLE_FIELDS
        assert "state" not in filterable_field_names()

    def test_the_parser_keeps_no_field_list_of_its_own(self):
        """`QueryParser.STANDARD_FIELDS` named `state` and was read by nothing. A registry
        no one consults describes intentions, not behaviour — and this one disagreed with the
        retrieval path about the one field that matters here.

        The parser now DOES ask which words are fields, and that is why it must still own no
        list: it asks `field_filters.is_filter_field`. A second copy is what would let
        `state` be a field to the grammar and not to the resolver all over again."""
        from mantle.search.query_parser import QueryParser

        assert not hasattr(QueryParser, "STANDARD_FIELDS")
        assert parse_query("state:draft x").filters, (
            "`state` is a field the resolver refuses BY NAME, so the parser must still hand "
            "it on — a grammar that dropped it would turn the refusal into a silent search"
        )


class TestAFilterIsNotARecall:
    def test_a_query_of_only_filters_is_refused(self, stack):
        with pytest.raises(QueryFilterError) as exc:
            _accessor(stack).search(_query("type:application/pdf"))
        assert "not a recall by itself" in str(exc.value)

    def test_candidates_refuses_it_too(self, stack):
        with pytest.raises(QueryFilterError):
            _accessor(stack).candidates(_query("type:application/pdf"))

    def test_a_filter_beside_a_term_is_fine(self, stack):
        assert _ids(_accessor(stack).search(_query(f"{TERM} type:application/pdf")))

    def test_a_filter_beside_a_query_VECTOR_is_fine(self, stack):
        """A vector is a query. `filters + vector` is a narrowed kNN, not an empty request —
        the refusal above is about having nothing to rank on, and a direction is something."""
        from mantle.search.types import SearchQuery

        _act(ALICE, "user")
        query = SearchQuery(
            query_text="type:application/pdf", user_id=ALICE, size=20,
            query_embedding=list(QUERY_VEC),
        )
        result = _accessor(stack).search(query)
        assert _ids(result) == PDFS_ALICE_CAN_READ
        assert result.applied_filters == ["content_type:application/pdf"]


# ---------------------------------------------------------------------------
# The unfiltered path is untouched
# ---------------------------------------------------------------------------

class TestNoFilterMeansNoChange:
    def test_resolve_without_a_predicate_returns_the_resolvers_set_verbatim(self):
        """Including ids whose doc could not be read — the fail-open half that is an
        AUTHORIZATION decision and must not be altered by adding a filter parameter."""
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        store_db = _FakeStoreDB()
        authorized = AUTHORIZED | {"art-vanished"}
        scope = resolve_authorized_scope(
            store_db, ALICE, lightcone=_FakeLightCone(authorized),
        )
        assert scope.artifact_ids == frozenset(authorized)

    def test_an_unreadable_doc_cannot_satisfy_a_filter(self):
        """With a filter running, a doc that cannot be read is dropped: an unread doc cannot
        be SHOWN to match. That is still a narrowing, so it is safe in the same direction."""
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        scope = resolve_authorized_scope(
            _FakeStoreDB(), ALICE,
            lightcone=_FakeLightCone(AUTHORIZED | {"art-vanished"}),
            artifact_predicate=lambda doc: True,
        )
        assert "art-vanished" not in scope.artifact_ids
        assert scope.artifact_ids == frozenset(AUTHORIZED)

    def test_compile_returns_none_for_no_filters(self):
        assert compile_filters(parse_query("plain query").filters) is None


# ---------------------------------------------------------------------------
# The public API: POST /artifacts/recall
# ---------------------------------------------------------------------------

def _recall_result(applied):
    """A `SearchResult` the router can map, with the filters reported as applied."""
    from mantle.search.types import SearchResult

    return SearchResult(
        hits=[], total=0, parsed_query=parse_query("x"),
        applied_filters=applied, corrections=[], ordering="recency",
    )


def _accessor_running_the_real_validation():
    """A stand-in accessor that runs the REAL query plan and then answers emptily.

    The 400s below are the router's contract, and they are only worth asserting if the
    thing that raises them is the production validation rather than a mock configured to
    raise. `plan_recall` is that validation.
    """
    from unittest.mock import MagicMock

    def _search(query):
        plan = plan_recall(
            query.query_text, has_vector=query.query_embedding is not None,
        )
        _search.seen_text = plan.retrieval_text
        from mantle.search.field_filters import describe

        return _recall_result(describe(plan.parsed.filters))

    accessor = MagicMock()
    accessor.search.side_effect = _search
    return accessor, _search


@pytest.mark.asyncio
class TestTheRecallApiContract:
    async def _post(self, client, mock_builder, body):
        accessor, spy = _accessor_running_the_real_validation()
        mock_builder.return_value = accessor
        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json=body,
        )
        return resp, spy

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_applied_filters_is_on_the_response(self, builder, client):
        resp, _ = await self._post(client, builder, {"query_text": "budget type:pdf"})
        assert resp.status_code == 200
        assert resp.json()["applied_filters"] == ["content_type:pdf"]

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_an_unfiltered_recall_reports_no_applied_filters(self, builder, client):
        resp, _ = await self._post(client, builder, {"query_text": "budget"})
        assert resp.status_code == 200
        assert resp.json()["applied_filters"] == []

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_only_the_terms_reach_retrieval(self, builder, client):
        resp, spy = await self._post(
            client, builder, {"query_text": "budget type:pdf @lang:en"},
        )
        assert resp.status_code == 200
        assert spy.seen_text == "budget"

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_word_that_is_not_a_field_searches_instead_of_400ing(self, builder, client):
        """The API half of `TestOnlyAKnownFieldMakesAFilter`. `colour` is not a field, so
        `colour:red` is text: the request succeeds, nothing is reported as applied, and the
        whole token reaches retrieval."""
        resp, spy = await self._post(client, builder, {"query_text": "budget colour:red"})
        assert resp.status_code == 200
        assert resp.json()["applied_filters"] == []
        assert spy.seen_text == "budget colour:red"

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_refused_field_is_still_a_400_naming_it(self, builder, client):
        """And the 400 did not go soft for a word that IS a field."""
        resp, _ = await self._post(client, builder, {"query_text": "budget content:secret"})
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "content" in detail
        assert "content_type" in detail, "the 400 must list what is filterable"

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_an_unsupported_operator_is_a_400_naming_it(self, builder, client):
        resp, _ = await self._post(client, builder, {"query_text": "budget tag:~ai"})
        assert resp.status_code == 400
        assert "~" in resp.json()["detail"]

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_range_on_an_unordered_field_is_a_400(self, builder, client):
        resp, _ = await self._post(client, builder, {"query_text": "budget title:>m"})
        assert resp.status_code == 400
        assert "created_at" in resp.json()["detail"]

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_state_filter_is_a_400_pointing_at_the_request_field(self, builder, client):
        resp, _ = await self._post(client, builder, {"query_text": "budget state:draft"})
        assert resp.status_code == 400
        assert "request field" in resp.json()["detail"]

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_the_state_request_field_still_works(self, builder, client):
        """The refusal above is only coherent if the field it names does the job."""
        resp, _ = await self._post(
            client, builder, {"query_text": "budget", "state": "draft"},
        )
        assert resp.status_code == 200

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_filter_only_query_is_a_400(self, builder, client):
        resp, _ = await self._post(client, builder, {"query_text": "type:pdf"})
        assert resp.status_code == 400
        assert "not a recall by itself" in resp.json()["detail"]

    async def test_content_types_is_gone_from_the_request_model(self):
        """It was declared and read by nothing — the handler never passed it on and
        `SearchQuery` had nowhere to put it — so it narrowed no recall, ever. `content_type:`
        is the one way to say it, and it does narrow. Removing it costs a client nothing:
        unknown fields are ignored, so a request still sending it gets the search it got."""
        from mantle.routers.artifacts_router import ArtifactRecallRequest

        assert "content_types" not in ArtifactRecallRequest.model_fields

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_client_still_sending_content_types_is_not_rejected(self, builder, client):
        resp, _ = await self._post(
            client, builder, {"query_text": "budget", "content_types": ["application/pdf"]},
        )
        assert resp.status_code == 200


# ── recall scope must not answer "does this id exist?" ─────────────────────────────────────

def test_a_scope_naming_only_unreachable_ids_searches_NOTHING() -> None:
    """The existence oracle, and the scope-widening bug behind it.

    `recall_artifacts` used to filter `body.scope` through `_artifact_exists` — a raw store read
    with no authorization — and then collapse an empty result with `or None`. That made the
    probe's answer observable: a real id filtered the contexts to empty and returned 0 hits,
    while a nonexistent id dropped the scope entirely and re-ran the SAME query unscoped over
    the caller's whole light cone. Binary, deterministic and timing-free over the entire id
    space, in a system whose stated invariant is that denial and nonexistence are
    indistinguishable.

    The second half is a plain authorization bug: naming only containers you cannot read is a
    request to search NOTHING, and `or None` turned it into a request to search EVERYTHING.

    This test pins the property rather than the implementation: whatever the router does with
    `scope`, a scope the caller cannot reach must never return more than an empty scope does.
    """
    from mantle.routers.artifacts_router import ArtifactRecallRequest

    body = ArtifactRecallRequest(query_text="anything", scope=["does-not-exist"])
    assert body.scope == ["does-not-exist"]

    import inspect

    from mantle.routers import artifacts_router

    src = inspect.getsource(artifacts_router.recall_artifacts)
    assert "col_ids or None" not in src, (
        "the `or None` collapse is back: an unreachable scope silently widens to the full "
        "light cone, and the widening is observable as an existence oracle"
    )
    assert "_artifact_exists(store_db, cid)" not in src, (
        "scope is being probed for existence again; `router_accessor` already applies scope as "
        "an intersection, so the probe buys nothing and leaks existence"
    )
