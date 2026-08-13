"""The reader-side vector ingress — the mirror of ``test_vector_ingress.py``.

A writer hands Mantle a vector on the write (``api/vectors.py``); a reader hands
Mantle a vector on the recall. Mantle never produces one, so those two are the only
ways a vector reaches the vector arm, and they are held to the same contract: the
same validation, the same ``space_id`` rule, the same 400.

Grouped by the question each group answers:

- ``TestDegradedEmbeddingIsNotAVector`` — does a degraded embed produce ``None`` and
  not a zero-length vector the engine will reject?
- ``TestOrderingReportsWhatHappened`` — does the response body say a cosine ordered it only
  when a cosine actually did?
- ``TestQueryVectorValidation`` — is a supplied query vector held to the writer's contract?
- ``TestQueryVectorReachesTheRanker`` — does it arrive at the ranker unchanged?
- ``TestTextAndVectorTogether`` — are the two query forms alternatives or companions?
"""

from __future__ import annotations

import json
from typing import Optional

import pytest

from mantle.search.mantle.engine import MantleHit
from mantle.search.mantle.sse.narrowing import Coverage
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeArtifactStore:
    def __init__(self, docs: dict[str, dict]) -> None:
        self._docs = docs

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        return self._docs.get(artifact_id)


class _FakeGraph:
    def edges_of(self, node, label=None, direction="out", limit=None) -> list:
        return []


class _FakeStoreDB:
    def __init__(self, docs: Optional[dict[str, dict]] = None) -> None:
        self.artifacts = _FakeArtifactStore(docs or {})
        self.graph = _FakeGraph()


class _FakeLightCone:
    def __init__(self, authorized: Optional[list[str]] = None) -> None:
        self._authorized = authorized or []

    def resolve(self, principal_id: str, *, action: str = "read",
                principal_type: str = "user") -> set:
        return set(self._authorized)


class _FakeNarrower:
    """The lexical half — records the text it was compiled from, narrows to `matches`.

    Returns the coverage MAPPING the real narrower returns: keys are the membership answer,
    values are the counts an unranked recall orders by. One stem per match, since these cases
    are about the vector path rather than about the counts.
    """

    def __init__(self, matches: Optional[set] = None) -> None:
        self.matches = matches
        self.calls: list[str] = []

    def lookup_for(self, query_text, request):
        self.calls.append(query_text)
        if self.matches is None:
            return None
        return lambda pairs: {a: Coverage(1, 0) for a in self.matches}


class _FakeRanker:
    """The semantic half — records the vector it was handed, returns canned MantleHits."""

    def __init__(self, hits: Optional[list[MantleHit]] = None) -> None:
        self.calls: list[dict] = []
        self._hits = hits or []

    def search(self, query_embedding, contexts, request=None, *, top_k,
               authorized_artifacts=None):
        self.calls.append({
            "query_embedding": list(query_embedding),
            "top_k": top_k,
        })
        return list(self._hits)


class _DegradedEmbeddings:
    """What ``Embeddings()`` actually returns with no provider and no cache entry.

    Not ``[]``: the facade pads its answer 1:1 with its input (``embeddings._pad``),
    so one text with no vector comes back as one EMPTY vector inside a non-empty
    list. A truth test on the outer list therefore says "we got something".
    """

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class _FakeEmbeddings:
    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _artifact_doc(artifact_id: str) -> dict:
    return {
        "id": artifact_id,
        "root_id": artifact_id,
        "context": json.dumps({"title": "t", "description": "d", "tags": []}),
        "content": "",
        "state": "committed",
        "created_by": "user-1",
        "principal_id": "user-1",
    }


def _ranked(artifact_id: str, score: float = 0.9) -> MantleHit:
    return MantleHit(
        artifact_id=artifact_id, chunk_id=0, score=score,
        principal_id="user-1", collection_id="col-1",
    )


def _query(text: str = "alpha", *, vector=None, size: int = 20):
    from mantle.search.types import SearchQuery
    return SearchQuery(
        query_text=text,
        query_embedding=vector,
        user_id="user-1",
        size=size,
    )


def _accessor(*, embeddings, docs=None, authorized=("art-1",),
              narrower=None, ranker=None):
    return MantleSseSearchAccessor(
        _FakeLightCone(authorized=list(authorized)),
        store_db=_FakeStoreDB(docs or {"art-1": _artifact_doc("art-1")}),
        embeddings=embeddings,
        narrower=narrower if narrower is not None else _FakeNarrower(),
        ranker=ranker if ranker is not None else _FakeRanker(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDegradedEmbeddingIsNotAVector:
    """A degraded embed must produce ``None``, never a zero-length vector.

    ``MantleQueryEngine.search`` raises ``ValueError("query_embedding must be a
    non-empty 1-D vector")`` on ``[]``, and the guards on both paths test ``is not None``
    — so handing it ``[]`` puts a full traceback through a blanket ``except`` on every
    recall wherever a cell store exists. The degrade is supposed to be silent.
    """

    def test_an_empty_vector_becomes_none(self):
        acc = _accessor(embeddings=_DegradedEmbeddings())
        from mantle.search.query_parser import parse_query
        assert acc._embed_or_none("alpha", parse_query("alpha")) is None

    def test_the_ranker_is_not_run_at_all_rather_than_run_on_an_empty_vector(self):
        ranker = _FakeRanker(hits=[_ranked("art-1")])
        acc = _accessor(embeddings=_DegradedEmbeddings(), ranker=ranker)
        result = acc.search(_query("alpha"))
        assert ranker.calls == []
        # And the recall still answers — the narrowed set, by recency.
        assert result.ordering == "recency"
        assert [h.doc_id for h in result.hits] == ["art-1"]

    def test_the_candidates_path_never_reaches_the_ranker_at_all(self):
        """`candidates()` publishes no order, so it has nothing to embed FOR. A degraded
        embed cannot degrade a path that was never going to call the ranker."""
        ranker = _FakeRanker(hits=[_ranked("art-1")])
        acc = _accessor(embeddings=_DegradedEmbeddings(), ranker=ranker,
                        narrower=_FakeNarrower({"art-1"}))
        out = acc.candidates(_query("alpha"))
        assert ranker.calls == []
        assert [c["artifact_id"] for c in out["candidates"]] == ["art-1"]


class TestOrderingReportsWhatHappened:
    """``ordering`` is a claim about the result, not about the request.

    A caller reads it to know whether the order it is looking at is a ranking it can
    threshold or a page of the authorized set it cannot. ``embedding is not None`` answers a
    different question — "did we manage to build a query vector" — and answers it wrong
    besides, since an empty vector is not ``None`` and a vector that reaches an unprovisioned
    node ranks nothing.
    """

    def test_recency_when_the_embedding_degraded(self):
        acc = _accessor(embeddings=_DegradedEmbeddings(),
                        ranker=_FakeRanker(hits=[_ranked("art-1")]))
        result = acc.search(_query("alpha"))
        assert result.ordering == "recency"
        assert result.hits[0].score is None

    def test_semantic_when_a_cosine_ordered_the_result(self):
        acc = _accessor(embeddings=_FakeEmbeddings(),
                        ranker=_FakeRanker(hits=[_ranked("art-1", 0.42)]))
        result = acc.search(_query("alpha"))
        assert result.ordering == "semantic"
        assert result.hits[0].score == pytest.approx(0.42)

    def test_semantic_on_an_empty_ranking(self):
        # The ranker ran over the narrowed set and kept none of it. That is a ranking, and
        # the body must not re-order the excluded set by recency and call it an answer.
        acc = _accessor(embeddings=_FakeEmbeddings(),
                        ranker=_FakeRanker(hits=[]))
        result = acc.search(_query("alpha"))
        assert result.hits == []
        assert result.ordering == "semantic"


class TestQueryVectorValidation:
    """The reader's vector is validated by the writer's function, not a second one."""

    def test_a_query_vector_is_validated_by_the_writer_s_contract(self):
        from mantle.api.vectors import VectorIngressError, validate_vector

        with pytest.raises(VectorIngressError):
            validate_vector([0.1, 0.2], None)           # no space_id
        with pytest.raises(VectorIngressError):
            validate_vector([], "space-a")              # no components
        with pytest.raises(VectorIngressError):
            validate_vector([0.0, 0.0], "space-a")      # no direction
        assert validate_vector([0.1, 0.2], "space-a").space_id == "space-a"


class TestQueryVectorReachesTheRanker:
    def test_a_supplied_vector_is_passed_through_unembedded(self):
        ranker = _FakeRanker(hits=[_ranked("art-1")])
        acc = _accessor(embeddings=_FakeEmbeddings(), ranker=ranker)
        acc.search(_query("alpha", vector=[0.5, 0.5, 0.5]))
        # The supplied numbers reach the ranker verbatim — not the embedder's.
        assert ranker.calls[0]["query_embedding"] == [0.5, 0.5, 0.5]

    def test_a_vector_only_query_still_runs(self):
        # Empty text with a vector is not an empty query: it is a kNN request. It narrows
        # nothing, because there are no terms to narrow with, and ranks the whole authorized
        # set — "nothing to look up" is not "narrow to nothing".
        narrower = _FakeNarrower()
        ranker = _FakeRanker(hits=[_ranked("art-1")])
        acc = _accessor(embeddings=_DegradedEmbeddings(),
                        narrower=narrower, ranker=ranker)
        result = acc.search(_query("", vector=[0.5, 0.5, 0.5]))
        assert ranker.calls[0]["query_embedding"] == [0.5, 0.5, 0.5]
        assert narrower.calls == [], "no terms means nothing to compile a lookup from"
        assert result.ordering == "semantic"
        assert [h.doc_id for h in result.hits] == ["art-1"]


class TestForeignSpaceIsRefused:
    """A query vector from another space is refused, by name, with both names in the message.

    Not accepted quietly. Routing places the query against the anchors and scoring takes a raw
    cosine inside whatever cells that opened — both are statements in the AnchorSet's basis — so
    a foreign vector yields a number that is not a similarity while being indistinguishable from
    one. The client seeds the AnchorSet, so it owns both halves of the match and the refusal is
    something it can act on.
    """

    @pytest.fixture
    def anchors(self):
        import numpy as np
        from mantle.search.anchors import store as _store
        from mantle.search.anchors.anchorset import Anchor
        from mantle.search.anchors.repo import InMemoryAnchorRepo

        repo = InMemoryAnchorRepo()
        rng = np.random.default_rng(7)
        repo.bulk_add([
            Anchor.make(f"a{i}", rng.standard_normal(8).astype("float32"), "space-home")
            for i in range(4)
        ])
        _store.set_anchor_repo(repo)
        try:
            yield repo
        finally:
            _store.set_anchor_repo(None)

    def test_the_home_space_passes_through_untouched(self, anchors):
        from mantle.api.vectors import project_to_anchor_space, validate_vector

        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        out = project_to_anchor_space(validate_vector(values, "space-home"))
        assert out == values

    def test_a_foreign_space_is_refused_and_the_message_names_the_fix(self, anchors):
        from mantle.api.vectors import (
            VectorIngressError,
            project_to_anchor_space,
            validate_vector,
        )

        vec = validate_vector([0.1] * 8, "space-elsewhere")
        with pytest.raises(VectorIngressError) as exc:
            project_to_anchor_space(vec)
        message = str(exc.value)
        # Both space names are in the message: the caller cannot fix this without them.
        assert "space-elsewhere" in message
        assert "space-home" in message
        # And the move, not only the diagnosis. A caller who reads this should not have to guess
        # whether the answer is a different query or a different AnchorSet.
        assert "TO FIX" in message
        assert "manage_anchors --action load" in message

    def test_with_no_anchorset_every_space_is_foreign_and_the_vector_is_refused(self):
        """No AnchorSet is the case where EVERY name is unusable, not where any name will do.

        This used to pass the numbers through, on the reading that there was no basis to be
        foreign to. What that produced was a 200 carrying the caller's whole visible corpus in
        recency order — the same body, field for field, that a query with no vector at all
        comes back as. A caller that supplied a vector could not tell it had been ignored, and
        `ordering` could not tell it either.
        """
        from mantle.search.anchors import store as _store
        from mantle.search.anchors.repo import InMemoryAnchorRepo
        from mantle.api.vectors import (
            VectorIngressError,
            project_to_anchor_space,
            validate_vector,
        )

        _store.set_anchor_repo(InMemoryAnchorRepo())     # empty → no live AnchorSet
        try:
            with pytest.raises(VectorIngressError) as exc:
                project_to_anchor_space(validate_vector([0.1, 0.2, 0.3], "whatever"))
        finally:
            _store.set_anchor_repo(None)

        message = str(exc.value)
        assert "whatever" in message
        assert "no AnchorSet" in message
        # Both ways out, because the caller cannot know which one it wants: seed the set, or
        # send the same recall as a text query, which genuinely works on this node.
        assert "TO FIX" in message
        assert "manage_anchors --action load" in message
        assert "without `vector`" in message

    def test_a_writer_is_still_accepted_on_the_same_unseeded_node(self):
        """The refusal is the READER's door only. A stored vector is provenance for data at
        rest and is placed when a set arrives, so refusing the write would refuse something a
        later seeding makes good — and `POST /artifacts` never calls the projection."""
        from mantle.search.anchors import store as _store
        from mantle.search.anchors.repo import InMemoryAnchorRepo
        from mantle.api.vectors import validate_vector

        _store.set_anchor_repo(InMemoryAnchorRepo())
        try:
            assert validate_vector([0.1] * 384, "whatever").space_id == "whatever"
        finally:
            _store.set_anchor_repo(None)

    def test_an_unreadable_anchorset_is_not_reported_as_an_absent_one(self, monkeypatch):
        """A set that exists and could not be READ is a different state, and says so.

        `search/anchors/store.py` refuses to collapse `AnchorSetCorrupt` into `None` because
        telling an operator "nobody seeded this node" about a node they DID seed sends them to
        the wrong command. The same distinction has to survive this door.
        """
        from mantle.api import vectors as _vectors
        from mantle.api.vectors import (
            VectorIngressError,
            project_to_anchor_space,
            validate_vector,
        )
        from mantle.search.anchors import store as _store

        vec = validate_vector([0.1, 0.2, 0.3], "space-home")

        def _boom():
            raise RuntimeError("anchor ids do not follow from their contents")

        monkeypatch.setattr(_store, "get_live_anchorset", _boom)
        with pytest.raises(VectorIngressError) as exc:
            project_to_anchor_space(vec)

        message = str(exc.value)
        assert "could not be read" in message
        assert "no AnchorSet" not in message
        assert "--action inspect" in message


class TestTextAndVectorTogether:
    """Text and vector are companions, not alternatives.

    They answer different questions on every recall: the text decides WHICH artifacts come
    back, read as membership off the blind-token index, and the vector decides what ORDER
    they come back in. A vector does not suppress the text, and text does not suppress the
    vector, because neither is competing for the other's job.
    """

    def test_the_text_narrows_and_the_vector_ranks_on_the_same_recall(self):
        narrower = _FakeNarrower(matches={"art-1"})
        ranker = _FakeRanker(hits=[_ranked("art-1")])
        acc = _accessor(embeddings=_DegradedEmbeddings(),
                        narrower=narrower, ranker=ranker)
        result = acc.search(_query("alpha", vector=[0.5, 0.5, 0.5]))
        assert narrower.calls == ["alpha"]
        assert ranker.calls[0]["query_embedding"] == [0.5, 0.5, 0.5]
        assert result.ordering == "semantic"
