"""A vector query on a node with no AnchorSet is refused. A text query on the same node is not.

The distinction is the whole of this file. A TEXT recall on an unseeded node genuinely works:
it narrows on the query's terms off the blind-token index and answers most-recently-updated
first, which is a real answer to a real request — `router_accessor._by_recency` argues exactly
that, and "a 400 would refuse a query that worked" is true of it.

A VECTOR recall on the same node works in no sense. There is no space to place the vector in,
so nothing about the request is honoured, and what came back was 200 over everything the caller
can read in recency order — the same body a query carrying no vector at all returns, down to
`ordering: "recency"`. The caller could not tell the two apart, which made "your vector was
ignored" and "here is your ranking" the same response.

Refusing also covers a hybrid text+vector recall, because the same door already refuses a hybrid
whose vector names a FOREIGN space — see `api/vectors.project_to_anchor_space`. Refusing a space
this node does not serve while accepting one that exists nowhere would be one request answered
two ways.
"""

from unittest.mock import MagicMock, patch

import pytest

from mantle.search.anchors import store as _anchor_store
from mantle.search.anchors.repo import InMemoryAnchorRepo


@pytest.fixture()
def unseeded():
    """The install default: an anchor repo with nothing in it, so no live AnchorSet."""
    _anchor_store.set_anchor_repo(InMemoryAnchorRepo())
    try:
        yield
    finally:
        _anchor_store.set_anchor_repo(None)


_VECTOR = [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_a_vector_recall_is_refused_and_the_message_names_both_ways_out(client, unseeded):
    resp = await client.post(
        "/artifacts/recall",
        json={"query_text": "", "vector": _VECTOR, "space_id": "hf:BAAI/bge-m3@1.0"},
    )

    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "no AnchorSet" in detail
    assert "hf:BAAI/bge-m3@1.0" in detail
    # Seed the set, or send it as a text query. The caller cannot pick without being told both.
    assert "manage_anchors --action load" in detail
    assert "without `vector`" in detail


@pytest.mark.asyncio
async def test_a_hybrid_recall_is_refused_too(client, unseeded):
    """The text half would have narrowed — and this door already refuses a hybrid whose vector
    names a foreign space, so accepting one whose space does not exist would be inconsistent."""
    resp = await client.post(
        "/artifacts/recall",
        json={"query_text": "quarterly report", "vector": _VECTOR, "space_id": "space-a"},
    )
    assert resp.status_code == 400, resp.text
    assert "no AnchorSet" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_the_refusal_never_reaches_the_search_backend(client, unseeded):
    """Refused at ingress, before authorization is resolved or a candidate is read.

    A vector that cannot be placed is not a cheaper search; it is not a search.
    """
    accessor = MagicMock()
    with patch(
        "mantle.search.mantle.wiring.build_sse_search_accessor", return_value=accessor
    ):
        resp = await client.post(
            "/artifacts/recall",
            json={"query_text": "hi", "vector": _VECTOR, "space_id": "space-a"},
        )

    assert resp.status_code == 400
    assert not accessor.search.called
    assert not accessor.candidates.called


@pytest.mark.asyncio
async def test_a_text_recall_on_the_same_node_is_not_refused(client, unseeded):
    """The control. Same node, same unseeded state, no vector — and it answers.

    This is what the refusal above must not break: a caller that CANNOT embed (a shell script,
    a webhook) still narrowed to a genuine set, and gets it back newest first.
    """
    accessor = MagicMock()
    accessor.candidates.return_value = {"candidates": [], "model_id": "hf:BAAI/bge-m3@1.0"}
    with patch(
        "mantle.search.mantle.wiring.build_sse_search_accessor", return_value=accessor
    ):
        resp = await client.post(
            "/artifacts/recall", json={"query_text": "quarterly report", "candidates": True}
        )

    assert resp.status_code == 200, resp.text
    assert accessor.candidates.called
