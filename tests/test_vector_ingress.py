"""Vector ingress — the semantic arm's front door.

Mantle never embeds. The only way a vector reaches the vector arm is a writer handing
one over on the write that produced the text, so the whole of 5b-1 is: accept it,
check that it can be *placed*, and carry it down the pipeline that already exists.

Three properties are pinned here, and each has a refusal beside it:

* **Shape is checked; quality never is.** Every rejection below is a statement about
  the request — no components, a NaN, a zero norm, the wrong width for the anchors.
  There is deliberately no test asserting a vector is "good", because that is a claim
  about someone else's model and Mantle does not make it.
* **`space_id` is not optional decoration.** A vector without one is unusable: the name is
  the record of which coordinate system the numbers are statements in, and it cannot be
  recovered from the numbers, so a stored vector with no space name is one nothing can later
  be compared to. Supplied without a vector it is equally meaningless, and both are refused.
* **The empty-vector contract survives.** A write with no vector must behave exactly
  as it did before this existed — the arm receives nothing and skips. That is the
  regression that would make ingress look like it worked while quietly changing every
  other write.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from mantle.api.vectors import (
    MAX_VECTOR_DIM,
    SuppliedVector,
    VectorIngressError,
    validate_vector,
)
from mantle.search.embeddings import Embeddings, WriterSuppliedEmbeddings
from mantle.routers.artifacts_router import _parse_supplied_vector
from mantle.search.ingest import pipeline_unified

_SPACE = "hf:BAAI/bge-m3@1.0"
_VEC = [0.6, 0.8]


@pytest.fixture(autouse=True)
def _no_live_anchorset(monkeypatch):
    """Most cases are about the payload, not this node's geometry.

    With no AnchorSet provisioned the width check has nothing to compare against and
    steps aside — which is the deployment state a bare node is in. The two tests that
    care about the width install one explicitly.
    """
    monkeypatch.setattr("mantle.api.vectors.anchorset_dim", lambda: None)


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def test_a_well_formed_vector_is_accepted():
    v = validate_vector(_VEC, _SPACE)
    assert isinstance(v, SuppliedVector)
    assert v.values == _VEC and v.space_id == _SPACE


def test_an_unnormalized_vector_is_accepted():
    """Routing unit-normalizes before it compares, so requiring the caller to do it
    first would refuse a vector that routes to exactly the same cell."""
    assert validate_vector([6.0, 8.0], _SPACE).values == [6.0, 8.0]


def test_a_vector_with_no_space_is_refused():
    with pytest.raises(VectorIngressError, match="space_id is required"):
        validate_vector(_VEC, None)
    with pytest.raises(VectorIngressError, match="space_id is required"):
        validate_vector(_VEC, "   ")


def test_an_empty_vector_is_refused():
    with pytest.raises(VectorIngressError, match="at least one component"):
        validate_vector([], _SPACE)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_component_is_refused(bad):
    with pytest.raises(VectorIngressError, match="not finite"):
        validate_vector([0.1, bad], _SPACE)


def test_a_zero_vector_is_refused():
    """It names no direction, so there is no nearest anchor — and routing has no
    unrouted path to fall into."""
    with pytest.raises(VectorIngressError, match="zero norm"):
        validate_vector([0.0, 0.0, 0.0], _SPACE)


def test_an_absurd_dimension_is_refused():
    with pytest.raises(VectorIngressError, match="exceeds the maximum"):
        validate_vector([0.1] * (MAX_VECTOR_DIM + 1), _SPACE)


def test_the_wrong_width_for_this_nodes_anchors_is_refused(monkeypatch):
    """The AnchorSet is the one coordinate system. A vector of another width cannot be
    routed, and the writer should learn that at the write rather than from a background
    index job they never see."""
    monkeypatch.setattr("mantle.api.vectors.anchorset_dim", lambda: 1024)
    with pytest.raises(VectorIngressError, match="anchor geometry"):
        validate_vector(_VEC, _SPACE)
    assert validate_vector([0.1] * 1024, _SPACE).space_id == _SPACE


# ---------------------------------------------------------------------------
# The router's translation of that into HTTP
# ---------------------------------------------------------------------------

def test_no_vector_and_no_space_is_the_ordinary_write():
    assert _parse_supplied_vector(None, None) is None


def test_malformed_input_is_a_400_not_a_500():
    with pytest.raises(HTTPException) as exc:
        _parse_supplied_vector([0.0, 0.0], _SPACE)
    assert exc.value.status_code == 400


def test_a_space_with_no_vector_is_refused():
    """It names the space of a vector that is not here — a request that cannot mean
    anything, so it is answered rather than silently ignored."""
    with pytest.raises(HTTPException) as exc:
        _parse_supplied_vector(None, _SPACE)
    assert exc.value.status_code == 400


def test_a_vector_with_no_space_is_refused_at_the_router():
    with pytest.raises(HTTPException) as exc:
        _parse_supplied_vector(_VEC, None)
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# The inverted provider seam
# ---------------------------------------------------------------------------

def test_the_writer_supplied_provider_returns_what_it_was_given():
    provider = WriterSuppliedEmbeddings(_VEC, _SPACE)
    assert provider(["some text"]) == [_VEC]


def test_one_vector_does_not_spread_across_several_texts():
    """Splitting one vector over several chunks would attribute to each of them a claim
    the writer made about the whole artifact."""
    provider = WriterSuppliedEmbeddings(_VEC, _SPACE)
    assert provider(["a", "b", "c"]) == [_VEC, [], []]


def test_the_facade_passes_an_injected_provider_through():
    out = Embeddings(WriterSuppliedEmbeddings(_VEC, _SPACE))(["text"])
    assert out == [_VEC]


def test_the_empty_vector_contract_still_holds(monkeypatch):
    """No writer vector, no cache entry: an empty vector per text, exactly as before.
    This is what keeps every write that carries no vector unchanged."""
    monkeypatch.setattr("mantle.search.embeddings._get_cache", lambda: None)
    assert Embeddings()(["a", "b"]) == [[], []]


# ---------------------------------------------------------------------------
# Through the pipeline
# ---------------------------------------------------------------------------

def _artifact():
    return SimpleNamespace(
        id="art-1", root_id="art-1", state="committed", collection_id="ws-1",
        created_by="user-1", modified_by="user-1", content_type="text/plain",
        context="{}", content="hello world", name="", description="",
        created_time="2026-08-09T00:00:00Z",
    )


class _Indexer:
    def __init__(self):
        self.chunks = None

    def index_artifact(self, principal_id, collection_id, chunks, request):
        self.chunks = list(chunks)
        return len(self.chunks)


@pytest.fixture()
def wired(monkeypatch):
    """The vector arm with its prerequisites satisfied and the indexer observable."""
    indexer = _Indexer()
    # A seeded set that names no space: the record's `model_id` then falls through to the
    # writer's own `space_id`, which is what these tests are about.
    anchorset = SimpleNamespace(model_id=None)
    monkeypatch.setattr("mantle.search.anchors.store.require_live_anchorset", lambda: anchorset)
    monkeypatch.setattr("mantle.services.dependencies.get_store_db", lambda: iter([object()]))
    monkeypatch.setattr("mantle.search.mantle.principal.resolve_cell_principal",
                        lambda db, cid: "principal-1")
    monkeypatch.setattr("mantle.search.mantle.wiring.build_indexer",
                        lambda db, segment="committed": indexer)
    monkeypatch.setattr(pipeline_unified, "_ingest_key_request", lambda pid: object())
    return indexer


def test_a_supplied_vector_reaches_the_cells(wired):
    outcome = pipeline_unified._mantle_index_artifact(
        _artifact(), "ws-1", {"content": "hello world"},
        vector=SuppliedVector(values=_VEC, space_id=_SPACE),
    )
    assert outcome == pipeline_unified.ARM_WRITTEN
    assert wired.chunks is not None and len(wired.chunks) == 1
    record = wired.chunks[0]
    assert record["embedding"] == _VEC
    assert record["artifact_id"] == "art-1"
    assert record["text"] == "hello world"


def test_the_stored_provenance_is_the_space_the_writer_named(wired):
    """Labelling a writer's vector with the local space would make two incomparable
    vectors look like siblings — the exact confusion `space_id` exists to prevent."""
    pipeline_unified._mantle_index_artifact(
        _artifact(), "ws-1", {"content": "hello world"},
        vector=SuppliedVector(values=_VEC, space_id="acme:custom@3"),
    )
    record = wired.chunks[0]
    assert record["space_id"] == "acme:custom@3"
    assert record["model_id"] == "acme:custom@3"


def test_a_vector_bearing_write_is_one_chunk_however_long_the_content(wired):
    long_text = "paragraph. " * 4000
    pipeline_unified._mantle_index_artifact(
        _artifact(), "ws-1", {"content": long_text},
        vector=SuppliedVector(values=_VEC, space_id=_SPACE),
    )
    assert len(wired.chunks) == 1


def test_a_write_without_a_vector_still_skips(wired):
    """The regression that matters: ingress must not change the write that carries no
    vector. With no provider producing anything, the arm has nothing to store."""
    outcome = pipeline_unified._mantle_index_artifact(
        _artifact(), "ws-1", {"content": "hello world"},
    )
    assert outcome == pipeline_unified.ARM_SKIPPED
    assert wired.chunks is None


def test_index_artifact_carries_the_vector_to_the_vector_arm_only():
    """The lexical arm reads text and must not be handed a vector it has no use for."""
    seen = {}

    def _vec(artifact, cid, fields, *, segment="committed", vector=None):
        seen["vector"] = vector
        return pipeline_unified.ARM_WRITTEN

    with (
        patch.object(pipeline_unified, "_sse_index_artifact", return_value=pipeline_unified.ARM_WRITTEN),
        patch.object(pipeline_unified, "_mantle_index_artifact", side_effect=_vec),
    ):
        supplied = SuppliedVector(values=_VEC, space_id=_SPACE)
        pipeline_unified.index_artifact(_artifact(), "ws-1", vector=supplied)

    assert seen["vector"] is supplied


def test_the_pipeline_writes_the_vector_the_router_validated():
    """End to end within the process: what `validate_vector` accepted is what lands in the
    cell record. Two validators that agreed by habit rather than by construction is the
    failure mode worth pinning."""
    supplied = _parse_supplied_vector([6.0, 8.0], _SPACE)
    assert isinstance(supplied, SuppliedVector)
    provider = WriterSuppliedEmbeddings(supplied.values, supplied.space_id)
    assert provider(["t"]) == [[6.0, 8.0]]


def test_the_enqueue_path_carries_the_vector():
    """The write that supplied the vector has returned by the time the job runs, so the
    vector has to ride the closure rather than be looked up again."""
    seen = {}

    def _index(artifact, collection_id, *, is_head=True, vector=None):
        seen["vector"] = vector
        return pipeline_unified.IndexOutcome(
            sse=pipeline_unified.ARM_WRITTEN, vector=pipeline_unified.ARM_WRITTEN,
        )

    supplied = SuppliedVector(values=_VEC, space_id=_SPACE)
    with (
        patch.object(pipeline_unified, "index_queue", None),
        patch.object(pipeline_unified, "index_artifact", side_effect=_index),
    ):
        pipeline_unified.enqueue_index_artifact(_artifact(), "ws-1", vector=supplied)

    assert seen["vector"] is supplied


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

class TestVectorOverHttp:
    """The request shape, exercised through the router the way a writer meets it."""

    @pytest.fixture(autouse=True)
    def _no_access_check(self):
        grant = SimpleNamespace(
            can_read=True, can_create=True, can_update=True, can_delete=True,
            can_invoke=True, can_add=True, can_share=True, resource_id=None,
        )
        with patch("mantle.routers.artifacts_router.check_access", return_value=grant):
            yield grant

    @pytest.mark.asyncio
    async def test_a_top_level_create_hands_the_vector_to_the_write_path(self, client):
        created = SimpleNamespace(to_dict=lambda: {"id": "art-new"})
        with patch("mantle.services.workspace_service.create_container",
                   return_value=created) as create:
            r = await client.post("/artifacts", json={
                "name": "n", "content": "hello",
                "vector": _VEC, "space_id": _SPACE,
            })
        assert r.status_code == 201
        supplied = create.call_args.kwargs["vector"]
        assert supplied.values == _VEC and supplied.space_id == _SPACE

    @pytest.mark.asyncio
    async def test_a_create_without_a_vector_passes_none(self, client):
        created = SimpleNamespace(to_dict=lambda: {"id": "art-new"})
        with patch("mantle.services.workspace_service.create_container",
                   return_value=created) as create:
            r = await client.post("/artifacts", json={"name": "n", "content": "hello"})
        assert r.status_code == 201
        assert create.call_args.kwargs["vector"] is None

    @pytest.mark.asyncio
    async def test_a_vector_without_a_space_is_a_400(self, client):
        with patch("mantle.services.workspace_service.create_container") as create:
            r = await client.post("/artifacts", json={"name": "n", "vector": _VEC})
        assert r.status_code == 400
        assert "space_id" in r.json()["detail"]
        create.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_malformed_vector_never_reaches_the_write(self, client):
        """Rejected before anything is created — a 400 that had already written a row would
        leave an artifact whose vector the caller believes was refused."""
        with patch("mantle.services.workspace_service.create_container") as create:
            r = await client.post("/artifacts", json={
                "name": "n", "vector": [0.0, 0.0], "space_id": _SPACE,
            })
        assert r.status_code == 400
        create.assert_not_called()
