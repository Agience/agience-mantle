"""Lexical narrows, semantic ranks — the live recall path, end to end.

The two encrypted indexes answer two different questions and this file measures that they are
each asked the one they can answer. The blind-token postings answer MEMBERSHIP: which
artifacts carry these stems. The cells answer PROXIMITY: how close is each of these. Nothing
fuses them, because a set and a ranking are not competing answers — so
``SearchHit.score`` is the cosine, with no rank-fusion constant standing between the number
and its meaning.

What this file pins that its neighbours do not:

- `test_blind_token_narrowing.py` proves the narrowing composes safely AT THE RESOLVER. This
  proves the same property where a CALLER observes it: through `MantleSseSearchAccessor`,
  over the real encrypted indexes, in the response body.
- `test_field_filters_narrow_recall.py` proves a `field:value` filter narrows. This proves the
  query's TERMS narrow, which is the arm that changed.
- The TEXT-ONLY path: a recall that narrowed to a real set with nothing to rank it. It is an
  ordinary request — a shell script or a webhook that cannot embed still searches — so it
  returns the set most-recently-updated first rather than a 400 or an arbitrary order.
- The beacon cut is still the cut, and it is still taken inside the arm that holds a vector
  per candidate.

The stack is the production one. Only the lattice and the light cone are doubles; both
indexes are real, the oracle is real, and the corpus is built so that similarity cannot be
doing the selecting — one query direction, near-identical document vectors — which means any
narrowing seen here is the terms' doing.
"""

from __future__ import annotations

import os
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle import MantleIndexer, MantleQueryEngine, OracleService
from mantle.search.mantle.engine import CUT_NONE
from mantle.search.mantle.oracle import FernetMasterKeyStore, KeyPurpose, KeyRequest
from mantle.search.mantle.sse.file_stores import FilePostingStore
from mantle.search.mantle.sse.indexer import SseIndexer
from mantle.search.mantle.sse.narrowing import TokenNarrower
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor
from mantle.search.mantle.stores import InMemoryCellStore

ALICE = "user-alice"
COLLECTION = "col-1"
CELL_PRINCIPAL = COLLECTION          # self-rooted, as in the production index path
DIM = 16

#: In every AUTHORIZED artifact, so a narrowing seen below is a narrowing and not a corpus
#: that only ever matched one thing.
TERM = "quasar"
#: In NO artifact at all.
ABSENT_TERM = "zzznosuchword"
#: In the SECRET and nowhere else. Searching it is the existence question the light cone must
#: refuse to answer.
SECRET_TERM = "helioseismology"

OLD = "art-old"
MID = "art-mid"
NEW = "art-new"
#: Indexed, matching, and NOT in Alice's light cone.
SECRET = "art-secret"

_TEXT = {
    OLD: f"{TERM} the earliest filing",
    MID: f"{TERM} the middle filing",
    NEW: f"{TERM} the latest filing",
    SECRET: f"{TERM} {SECRET_TERM} the acquisition nobody shared",
}

#: Deliberately NOT in id order and not in insertion order, so an assertion on recency order
#: cannot pass by accident on either.
_MODIFIED = {
    OLD:    "2024-01-01T00:00:00Z",
    MID:    "2025-06-01T00:00:00Z",
    NEW:    "2026-02-01T00:00:00Z",
    SECRET: "2027-01-01T00:00:00Z",
}

AUTHORIZED = {OLD, MID, NEW}
#: Newest first — what the text-only path must return.
BY_RECENCY = [NEW, MID, OLD]


def _act(principal_id: str, principal_type: str = "user") -> None:
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal

    set_acting_principal(
        ActingPrincipal(principal_id=principal_id, principal_type=principal_type,
                        source="stage-b-test")
    )


def _write_request():
    """A write: a read action never mints a master key, so the corpus must be indexed under
    one or every query below would see an empty index and pass for the wrong reason."""
    _act(CELL_PRINCIPAL, "principal")
    return KeyRequest(requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
                      requester_type="principal", action="update")


class _AliceHoldsTheCollectionKey:
    """Key custody, decided the way `LightConeGrantVerifier` decides it — per collection.

    Alice legitimately holds this key, so everything below happens strictly downstream of
    correct custody: a verifier that denied her would hide whether the narrowing did anything.
    In particular, she can open the very posting lists the secret's terms are written into —
    which is what makes the security assertions non-vacuous.
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
    return {
        "id": artifact_id,
        "root_id": artifact_id,
        "collection_id": COLLECTION,
        "content_type": "text/plain",
        "created_by": "user-bob",
        "created_time": _MODIFIED[artifact_id],
        "modified_time": _MODIFIED[artifact_id],
        "state": "committed",
        "context": {"title": artifact_id, "tags": [], "description": "a filed artifact"},
        "content": _TEXT[artifact_id],
    }


class _FakeStoreDB:
    def __init__(self) -> None:
        docs = {a: _doc(a) for a in _TEXT}
        docs[COLLECTION] = {"id": COLLECTION, "root_id": COLLECTION, "context": {}}
        self.artifacts = _FakeArtifacts(docs)
        self.graph = _FakeGraph()


class _FakeLightCone:
    def __init__(self, authorized: set) -> None:
        self._authorized = set(authorized)

    def resolve(self, principal_id, action="read", *, principal_type="user") -> set:
        return set(self._authorized) if principal_id == ALICE else set()


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
#: match. Any narrowing below is the terms, never similarity.
QUERY_VEC = _vec(11)
_DOC_VECS = {a: _vec(11) for a in _TEXT}


class _NoVector:
    """An embedder whose answer is a vector by type and not by content.

    `Embeddings` pads its result 1:1 with its input, so a text nothing has embedded comes back
    as `[[]]` — this reproduces that shape, `_embed_or_none` turns it into `None`, and the
    recall has nothing to rank by. It is the DEFAULT state of a Mantle node, because Mantle
    runs no model.
    """

    def __call__(self, texts):
        return [[] for _ in texts]


@pytest.fixture
def stack(tmp_path):
    oracle = OracleService(
        FernetMasterKeyStore(Fernet(Fernet.generate_key())),
        grant_verifier=_AliceHoldsTheCollectionKey(),
    )
    root = os.path.join(str(tmp_path), "sse-index")
    postings = FilePostingStore(root, prefix="mantle-sse")
    cells = InMemoryCellStore()

    sse_indexer = SseIndexer(oracle, postings)
    vec_indexer = MantleIndexer(oracle, cells)
    for artifact_id in _TEXT:
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
    return {
        "oracle": oracle,
        "vector": MantleQueryEngine(oracle, cells),
        "narrower": TokenNarrower(oracle, postings),
        "cells": cells,
    }


def _reach(stack, text: str, request) -> set:
    """The narrowing's OWN answer over the whole collection, un-met."""
    lookup = stack["narrower"].lookup_for(text, request)
    return set() if lookup is None else set(lookup([(CELL_PRINCIPAL, COLLECTION)]))


def _accessor(stack, *, authorized=AUTHORIZED, embeddings=None,
              ranker=None) -> MantleSseSearchAccessor:
    return MantleSseSearchAccessor(
        _FakeLightCone(authorized),
        store_db=_FakeStoreDB(),
        embeddings=embeddings or (lambda texts: [list(QUERY_VEC) for _ in texts]),
        narrower=stack["narrower"],
        ranker=ranker or stack["vector"],
    )


def _query(text: str, *, size: int = 20, user_id: str = ALICE, sort: str = "relevance"):
    """A recall request with the acting principal bound as the router would bind it — the
    oracle refuses a key to a requester that is not the authenticated actor."""
    from mantle.search.types import SearchQuery

    _act(user_id, "user")
    return SearchQuery(query_text=text, user_id=user_id, size=size, sort=sort)


def _ids(result) -> set:
    return {h.doc_id for h in result.hits}


# ---------------------------------------------------------------------------
# The corpus is real. Without these, everything below could pass vacuously.
# ---------------------------------------------------------------------------

class TestTheCorpusWouldOtherwiseReturnEverything:
    def test_every_artifact_including_the_secret_carries_the_term(self, stack):
        _act(CELL_PRINCIPAL, "principal")
        assert _reach(stack, TERM, KeyRequest(
            requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
            requester_type="principal", action="read")) == set(_TEXT)

    def test_the_secret_is_the_only_carrier_of_its_own_term(self, stack):
        _act(CELL_PRINCIPAL, "principal")
        assert _reach(stack, SECRET_TERM, KeyRequest(
            requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
            requester_type="principal", action="read")) == {SECRET}


# ---------------------------------------------------------------------------
# The terms narrow, on the live path
# ---------------------------------------------------------------------------

class TestTheTermsNarrowTheRecall:
    def test_a_term_every_authorized_artifact_carries_returns_all_of_them(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM))
        assert _ids(result) == AUTHORIZED

    def test_a_term_one_artifact_carries_returns_that_one(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(_query("latest"))
        assert _ids(result) == {NEW}

    def test_a_term_in_no_document_returns_nothing(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(_query(ABSENT_TERM))
        assert result.hits == []
        assert result.total == 0

    def test_a_quoted_phrase_is_gated_on_adjacency(self, stack):
        """The bigram gate, on the live path. Both artifacts carry both words; only one
        carries them adjacent."""
        acc = _accessor(stack, embeddings=_NoVector())
        assert _ids(acc.search(_query('"latest filing"'))) == {NEW}
        assert acc.search(_query('"earliest latest"')).hits == []


# ---------------------------------------------------------------------------
# THE SECURITY PROPERTY, where a caller observes it
# ---------------------------------------------------------------------------

class TestNarrowingOnlyEverNarrows:
    """A token naming content the requester cannot read is indistinguishable from a token
    matching nothing.

    Alice holds the collection key, so the narrowing genuinely OPENS the posting list the
    secret's term is written into and genuinely finds its id there. What stops it being an
    existence oracle is not custody and not a check — it is that the id meets a set that is
    already a subset of the light cone, and intersection has no vocabulary for adding one.
    """

    def test_a_term_only_the_secret_carries_returns_nothing(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(_query(SECRET_TERM))
        assert result.hits == []
        assert result.total == 0

    def test_that_answer_is_identical_to_a_term_nothing_carries(self, stack):
        """Stated as an indistinguishability, which is the property itself.

        `helioseismology` is carried by a real, indexed, matching artifact outside Alice's
        light cone. `zzznosuchword` is carried by nothing anywhere. If the two responses
        differed in ANY observable — hits, `total`, `ordering`, `applied_filters`,
        `corrections`, or an error in one case and not the other — the recall would report
        the existence of content Alice cannot read.
        """
        acc = _accessor(stack, embeddings=_NoVector())
        unreadable = acc.search(_query(SECRET_TERM))
        nonexistent = acc.search(_query(ABSENT_TERM))

        assert unreadable.hits == nonexistent.hits == []
        assert (unreadable.total, unreadable.ordering, unreadable.applied_filters,
                unreadable.corrections) == (
            nonexistent.total, nonexistent.ordering, nonexistent.applied_filters,
            nonexistent.corrections,
        )

    def test_the_same_holds_with_a_vector_supplied(self, stack):
        """A query vector cannot re-admit what the narrowing removed. It orders a set; the
        set was decided before it ran, and the cells are read with that set as their
        universe."""
        acc = _accessor(stack)
        assert acc.search(_query(SECRET_TERM)).hits == []
        assert acc.search(_query(ABSENT_TERM)).hits == []

    def test_the_secret_never_reaches_a_recall_that_matches_everything_else(self, stack):
        """The control for all three: a term the secret DOES share with the authorized set
        comes back with the authorized set and without the secret."""
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM))
        assert SECRET not in _ids(result)
        assert _ids(result) == AUTHORIZED


# ---------------------------------------------------------------------------
# The text-only path
# ---------------------------------------------------------------------------

class TestTextOnlyRecallIsAnOrdinaryRequest:
    """A query with terms and no vector narrows to a set no cosine can order.

    That is not an error and not an empty result: the caller asked a real question and got a
    real set, and the narrowing that produced it already knows how much of the query each
    member matched. So the order is that — most of the query first — with recency beneath it.
    `ordering` says `"coverage"` and each `score` is the INTEGER COUNT of query stems matched.

    A ONE-STEM QUERY IS EXACTLY RECENCY ORDER, which is what `TERM` is here: every survivor
    scores 1 and the tiebreak is the whole ordering. That is the degeneracy stated as a test
    rather than as a claim.
    """

    def test_it_returns_the_narrowed_set_newest_first(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM))
        assert [h.doc_id for h in result.hits] == BY_RECENCY

    def test_it_reports_coverage_and_an_integer_count(self, stack):
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM))
        assert result.ordering == "coverage"
        assert [h.score for h in result.hits] == [1, 1, 1], (
            "a one-stem query matched one stem in each survivor — the count says exactly "
            "that, and a normalised float would invite a threshold on a number meaning nothing"
        )
        assert all(isinstance(h.score, int) for h in result.hits)

    def test_a_fuller_match_outranks_a_thinner_one(self, stack):
        """The ordering doing work, against recency pulling the other way.

        `OLD` is the OLDEST artifact and the only one carrying `earliest`, so a two-stem query
        for `quasar earliest` must put it FIRST — ahead of two artifacts the clock would have
        put ahead of it. Ties below it fall back to recency, unchanged.
        """
        result = _accessor(stack, embeddings=_NoVector()).search(
            _query("%s earliest" % TERM))
        assert [h.doc_id for h in result.hits] == [OLD, NEW, MID]
        assert [h.score for h in result.hits] == [2, 1, 1]

    def test_the_counts_are_echoed_beside_the_score(self, stack):
        """`score` is one field and coverage is two numbers, so the second lives in
        `metadata` rather than being dropped or folded into the first."""
        result = _accessor(stack, embeddings=_NoVector()).search(
            _query('"latest filing"'))
        assert [h.doc_id for h in result.hits] == [NEW]
        assert result.hits[0].metadata == {"matched_stems": 2, "matched_bigrams": 1}

    def test_the_total_is_exact_rather_than_capped(self, stack):
        """The coverage path orders the whole narrowed set, so it has no horizon to stop at."""
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM))
        assert result.total == 3
        assert result.total_is_capped is False

    def test_it_paginates_over_that_order(self, stack):
        from mantle.search.types import SearchQuery

        _act(ALICE, "user")
        page = _accessor(stack, embeddings=_NoVector()).search(
            SearchQuery(query_text=TERM, user_id=ALICE, size=1, from_=1),
        )
        assert [h.doc_id for h in page.hits] == [MID]
        assert page.total == 3

    def test_no_cell_is_decrypted_for_it(self, stack):
        """Nothing to rank by means nothing to decrypt. The recency order is read off
        timestamps the light cone already held."""
        cells = stack["cells"]
        with patch.object(cells, "get", side_effect=AssertionError("cell read")) as spy:
            result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM))
        assert spy.call_count == 0
        assert _ids(result) == AUTHORIZED


class TestSortIsRead:
    """`SearchQuery.sort` was plumbed from the router and read by nothing. It is read now."""

    def test_recency_is_honoured_even_when_a_vector_exists(self, stack):
        result = _accessor(stack).search(_query(TERM, sort="recency"))
        assert result.ordering == "recency"
        assert [h.doc_id for h in result.hits] == BY_RECENCY
        assert all(h.score is None for h in result.hits)

    def test_asking_for_recency_costs_no_cell_decryption(self, stack):
        cells = stack["cells"]
        with patch.object(cells, "get", side_effect=AssertionError("cell read")) as spy:
            _accessor(stack).search(_query(TERM, sort="recency"))
        assert spy.call_count == 0

    def test_relevance_with_a_vector_is_a_cosine(self, stack):
        result = _accessor(stack).search(_query(TERM, sort="relevance"))
        assert result.ordering == "semantic"
        assert all(h.score is not None for h in result.hits)

    def test_relevance_with_no_vector_resolves_to_coverage(self, stack):
        """What `relevance` MEANS when no cosine can measure it: the best ordering available,
        which is how much of the query each hit matched. It is a request, so it cannot promise
        a cosine — and the response does not pretend it delivered one."""
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM, sort="relevance"))
        assert result.ordering == "coverage"
        assert all(h.score == 1 for h in result.hits)

    def test_recency_is_still_reachable_and_is_still_scoreless(self, stack):
        """`sort="recency"` turns EVERY ordering off, coverage included. It is the one request
        that asks for no ranking at all, and its hits carry no number because nothing measured
        them."""
        result = _accessor(stack, embeddings=_NoVector()).search(_query(TERM, sort="recency"))
        assert result.ordering == "recency"
        assert all(h.score is None for h in result.hits)


# ---------------------------------------------------------------------------
# The cut
# ---------------------------------------------------------------------------

class TestTheBeaconCutIsStillTheCut:
    """Where a ranked result set STOPS is still read off its own spectrum, inside the one arm
    that holds a vector per candidate.

    It could not move: `beacon.cut.select` reads `(item_embs, query_emb)`, and the narrowing
    has no vectors at all — it has a set. Removing fusion removed the thing the cut was NOT
    allowed to be pointed at (an RRF spectrum, whose gaps are manufactured by `k`), and left
    the cut exactly where it was.
    """

    def test_the_cut_runs_on_a_ranked_recall(self, stack):
        # Resolved through `sys.modules`, the way `_beacon_cut`'s own `from ... import select`
        # resolves it, and NOT with `import mantle.search.beacon.cut as cut_mod`. The two can
        # be different objects: `tests/test_beacon_is_the_cut.py` deletes the beacon modules
        # from `sys.modules` to prove they re-import cleanly, and the parent package keeps an
        # attribute bound to the replacement. Patching the wrong one of the two silently spies
        # on a function nobody calls.
        import importlib

        cut_mod = importlib.import_module("mantle.search.beacon.cut")

        with patch.object(cut_mod, "select", wraps=cut_mod.select) as spy:
            result = _accessor(stack).search(_query(TERM))
        assert spy.call_count == 1, "the ranked path must take the cut, once"
        assert result.ordering == "semantic"

    def test_the_cut_is_what_bounds_the_ranked_set(self, stack):
        """Measured against the same ranking with the cut turned off. Over three
        near-identical vectors the cut legitimately stops early; `MANTLE_SEARCH_CUT=none`
        returns the whole scored horizon. A difference proves the cut decided membership."""
        uncut = MantleQueryEngine(stack["oracle"], stack["cells"], cut=CUT_NONE)
        with_cut = _accessor(stack).search(_query(TERM))
        without_cut = _accessor(stack, ranker=uncut).search(_query(TERM))

        assert _ids(without_cut) == AUTHORIZED
        assert _ids(with_cut) <= _ids(without_cut)
        assert len(with_cut.hits) < len(without_cut.hits), (
            "the cut must be deciding where this ranking stops, or it is not on the path"
        )

    def test_the_cut_never_widens_past_the_narrowing(self, stack):
        """Whatever the cut keeps is a subset of what the terms admitted. It is a refinement
        of an answer, and a refinement cannot add to one."""
        result = _accessor(stack).search(_query("latest"))
        assert _ids(result) <= {NEW}


# ---------------------------------------------------------------------------
# The response contract, on the wire
# ---------------------------------------------------------------------------

def _recency_result():
    """A `SearchResult` off the text-only path, for the router to map."""
    from mantle.search.query_parser import parse_query
    from mantle.search.types import SearchHit, SearchResult

    hit = SearchHit(
        doc_id=NEW, score=None, root_id=NEW, version_id=NEW,
        title="t", description="d", content="c", tags=[], metadata={},
        collection_id=COLLECTION,
    )
    return SearchResult(
        hits=[hit], total=1, parsed_query=parse_query(TERM),
        applied_filters=[], corrections=[], ordering="recency",
    )


@pytest.mark.asyncio
class TestTheRecallApiContract:
    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_null_score_survives_the_response_model(self, builder, client):
        """`RecallHitResponse.score` had to become optional for this. A required float would
        have turned every text-only recall into a 500 at serialization."""
        from types import SimpleNamespace

        builder.return_value = SimpleNamespace(search=lambda q: _recency_result())
        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json={"query_text": TERM},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ordering"] == "recency"
        assert body["hits"][0]["score"] is None

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_used_hybrid_is_gone_from_the_body(self, builder, client):
        """One ranker makes "did a vector reach this ranking" a constant, and a constant
        restated in every response reads as a query outcome. `ordering` replaces it with a
        question that genuinely varies."""
        from types import SimpleNamespace

        builder.return_value = SimpleNamespace(search=lambda q: _recency_result())
        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json={"query_text": TERM},
        )
        assert resp.status_code == 200
        assert "used_hybrid" not in resp.json()

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_sort_reaches_the_accessor(self, builder, client):
        """It is plumbed from the request body all the way to the thing that reads it."""
        from types import SimpleNamespace

        seen: dict = {}

        def _search(query):
            seen["sort"] = query.sort
            return _recency_result()

        builder.return_value = SimpleNamespace(search=_search)
        resp = await client.post(
            "/artifacts/recall",
            headers={"Authorization": "Bearer fake-token"},
            json={"query_text": TERM, "sort": "recency"},
        )
        assert resp.status_code == 200
        assert seen["sort"] == "recency"
