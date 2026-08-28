"""MantleSseSearchAccessor.candidates() — the raw retrieval primitive.

`candidates()` returns the SAME universe `search()` does — same plan, same field filter, same
blind-token narrowing, same meet — unordered and unhydrated. This file isolates its own shaping;
the "same universe" claim itself is measured over real encrypted indexes in
`test_field_filters_narrow_recall.py` and `test_recall_honours_artifact_scoped_grants.py`, where
there is a corpus for a narrowing to disagree with a filter about.

Context resolution is patched here (it is covered by the `search()` tests), which leaves three
things this file is the only cover for: the per-candidate KEY SET, the `include_vectors` switch,
and the empty answer for a principal with no authorized contexts.
"""

from unittest.mock import patch

import pytest

from mantle.search.mantle.lightcone import AuthorizedScope
from mantle.search.mantle.sse.narrowing import Coverage
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor

_RESOLVE = "mantle.search.mantle.sse.router_accessor.resolve_authorized_scope"


def _scope(contexts, artifacts=("a1", "a2")):
    """A light-cone answer at BOTH granularities, with timestamps.

    `candidates()` narrows to `artifact_ids` as well as to `contexts` (an artifact-scoped grant
    must not read back as its whole collection), so a stub that returned only the pairs would
    starve every assertion below of its candidates. The timestamps are what orders the budget
    cut — the query-independent order — so they are part of the answer rather than incidental.
    """
    stamps = {"a1": "2026-01-01T00:00:00Z", "a2": "2025-01-01T00:00:00Z"}
    return AuthorizedScope(
        list(contexts), frozenset(artifacts),
        {a: t for a, t in stamps.items() if a in artifacts},
    )


class _FakeNarrower:
    """Narrows to `matches`, with one matched stem each. `candidates()` computes the counts
    like `search()` does and publishes none of them, which is what the key-set test pins."""

    def __init__(self, matches=("a1", "a2")):
        self.matches = set(matches)

    def lookup_for(self, query_text, request, *, salient=None):
        # `salient` is the corpus's measure of which stems carry the question, handed to
        # the real narrower by `_salient_measure`. A double accepts and ignores it: these
        # tests state what the ACCESSOR does with a narrowing's answer, and filtering the
        # stems first would change the question they ask.
        return lambda pairs: {a: Coverage(1, 0) for a in self.matches}


class _FakeEmbeddings:
    def __call__(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeArtifacts:
    def __init__(self, docs):
        self._docs = docs

    def get_artifact(self, artifact_id):
        return self._docs.get(artifact_id)


class _FakeGraph:
    def edges_of(self, node, label=None, direction="out", limit=None):
        return []


class _FakeStoreDB:
    """Enough lattice for a candidate to name its collection and that collection's owner."""

    def __init__(self):
        docs = {
            "a1": {"id": "a1", "root_id": "a1", "collection_id": "c1", "context": {}},
            "a2": {"id": "a2", "root_id": "a2", "collection_id": "c1", "context": {}},
            "c1": {"id": "c1", "root_id": "c1", "context": {}},
        }
        self.artifacts = _FakeArtifacts(docs)
        self.graph = _FakeGraph()


def _query(text="alpha"):
    from mantle.search.types import SearchQuery
    return SearchQuery(query_text=text, user_id="u1", size=20)


def _accessor(matches=("a1", "a2")):
    return MantleSseSearchAccessor(
        object(), store_db=_FakeStoreDB(), embeddings=_FakeEmbeddings(),
        narrower=_FakeNarrower(matches),
    )


def test_candidates_returns_the_narrowed_set_over_authorized_contexts():
    acc = _accessor()
    with patch(_RESOLVE, return_value=_scope([("c1", "c1")])):
        out = acc.candidates(_query(), candidate_budget=50)
    # Recency order — the query-independent one, which is what a budget may cut by.
    assert [c["artifact_id"] for c in out["candidates"]] == ["a1", "a2"]
    assert out["candidates"][0]["collection_id"] == "c1"
    assert out["candidates"][0]["principal_id"] == "c1"
    assert "vector" not in out["candidates"][0]  # include_vectors defaults False


def test_candidates_publish_three_keys_and_no_score():
    """The fused vocabulary is GONE rather than nulled.

    `sse_score` was a BM25 score, `rrf_score` a rank-fusion constant's output, `source` a
    which-arm-found-it flag, `vector_score` a cosine nothing on this path computes. A null for
    any of them would say "this candidate had no such score" where the truth is that the
    quantity does not exist. `model_id` stays and is `null`: nothing here retrieves by
    embedding, and nothing did before either.
    """
    acc = _accessor()
    with patch(_RESOLVE, return_value=_scope([("c1", "c1")])):
        out = acc.candidates(_query())
    for candidate in out["candidates"]:
        assert set(candidate) == {"artifact_id", "collection_id", "principal_id"}
    assert out["model_id"] is None


def test_the_budget_cuts_and_cuts_by_recency():
    acc = _accessor()
    with patch(_RESOLVE, return_value=_scope([("c1", "c1")])):
        out = acc.candidates(_query(), candidate_budget=1)
    assert [c["artifact_id"] for c in out["candidates"]] == ["a1"]


def test_candidates_include_vectors_adds_reserved_placeholder():
    acc = _accessor()
    with patch(_RESOLVE, return_value=_scope([("c1", "c1")], artifacts=("a1",))):
        out = acc.candidates(_query(), include_vectors=True)
    assert out["candidates"][0]["vector"] is None  # reserved until the engine surfaces vectors


def test_candidates_empty_when_no_authorized_contexts():
    acc = _accessor()
    with patch(_RESOLVE, return_value=_scope([], artifacts=())):
        out = acc.candidates(_query())
    assert out == {"candidates": [], "model_id": None}


def test_candidates_empty_when_the_narrowing_matched_nothing():
    """One door, four facts — the same one `search()` has. The light cone authorized nothing,
    the filter matched nothing, the terms matched nothing, or either named something
    unreadable: all of them leave here."""
    acc = _accessor(matches=())
    with patch(_RESOLVE, return_value=_scope([("c1", "c1")], artifacts=())):
        out = acc.candidates(_query())
    assert out == {"candidates": [], "model_id": None}


def test_candidates_refuses_to_answer_without_a_narrower():
    """The same refusal `search()` makes, and for a stronger reason: this method is the
    chokepoint every flavor ranks within, so answering by widening to everything authorized
    would hand that widening to every flavor at once."""
    acc = MantleSseSearchAccessor(object(), store_db=_FakeStoreDB())
    with pytest.raises(ValueError, match="TokenNarrower"):
        acc.candidates(_query())
