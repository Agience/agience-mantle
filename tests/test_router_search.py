"""Router tests for the raw candidate primitive: POST /artifacts/recall {candidates: true}.

It shares a handler with ranked search so both resolve authorization identically;
these pin the candidates arm of that handler — HTTP contract, validation, and the
503 no-fallback behavior.

The encrypted-search backend is mocked at the accessor seam (tests never touch a
real Oracle/S3/the lattice); these assert the HTTP contract + validation + the 503
no-fallback behavior.
"""

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from mantle.search.mantle import wiring

#: The candidate shape the accessor returns: identity only, no score of any kind, and a null
#: `model_id` because nothing on that path retrieves by embedding. See
#: `tests/test_router_accessor_candidates.py` for why the fused vocabulary is gone rather
#: than nulled.
_CANDIDATES = {
    "candidates": [
        {"artifact_id": "a1", "collection_id": "c1", "principal_id": "p1"},
    ],
    "model_id": None,
}


@pytest.mark.asyncio
async def test_query_returns_authorized_candidates(client):
    accessor = MagicMock()
    accessor.candidates.return_value = _CANDIDATES
    with patch("mantle.search.mantle.wiring.build_sse_search_accessor", return_value=accessor):
        resp = await client.post("/artifacts/recall",
                                 json={"query_text": "hello", "candidates": True,
                                       "candidate_budget": 50})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_id"] is None
    assert body["candidates"][0] == {
        "artifact_id": "a1", "collection_id": "c1", "principal_id": "p1"}
    assert accessor.candidates.called  # the auth chokepoint was exercised


@pytest.mark.asyncio
async def test_query_ignores_caller_supplied_embedding(client):
    """`embedding` must not be an accepted input: a caller-supplied vector is trained-model output,
    i.e. BYOK, which the no-models rule bans universally.

    Asserted against the request model rather than a status code. The endpoint ignores unknown
    fields by design, so every HTTP outcome here — 200 with the field dropped, 200 with the field
    HONOURED — looks the same from the outside; a status assertion cannot tell BYOK reopening from
    business as usual. `model_fields` can: if someone adds a real `embedding` field, this fails.
    The behavioural half (the accessor never receives one) is pinned in
    `test_router_artifacts.py::TestRecallSearch`.
    """
    from mantle.routers.artifacts_router import ArtifactRecallRequest

    assert "embedding" not in ArtifactRecallRequest.model_fields, (
        "ArtifactRecallRequest declares an `embedding` field — caller-supplied query vectors are "
        "BYOK and must not be an accepted input")
    # ...and the endpoint still ignores the unknown field rather than 422-ing a legacy client.
    resp = await client.post("/artifacts/recall",
                                 json={"query_text": "hi", "candidates": True,
                                       "embedding": [0.1, 0.2]})
    assert resp.status_code != 422, resp.text


@pytest.mark.asyncio
async def test_query_rejects_missing_query_text(client):
    resp = await client.post("/artifacts/recall", json={"candidates": True})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_query_503_when_search_backend_unavailable(client):
    # No plaintext fallback by design — missing Oracle/S3/the lattice → 503.
    with patch("mantle.search.mantle.wiring.build_sse_search_accessor", return_value=None):
        resp = await client.post("/artifacts/recall", json={"query_text": "hi", "candidates": True})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# The 503, constructed for real
#
# The test above patches `build_sse_search_accessor` to None — it asserts that the router converts
# None to 503, which is one line of router code. It cannot see a change that makes the builder stop
# returning None, and that is exactly the change a "helpful" local fallback would be.
#
# So these construct the no-backend condition for real and run the whole chain
# (`_build_sse_store` → `_build_query_stack` → `build_sse_search_accessor` → router). Because a
# local file-backed store exists and makes getting hits possible, the 503 must stay reachable
# when no encrypted backend exists.
# ---------------------------------------------------------------------------


def _no_edge_object_storage():
    """Edge object storage neither configured nor reachable — a standalone box."""
    return [
        patch("mantle.search.mantle.wiring.edge_object_storage_is_configured", lambda: False),
        patch("mantle.search.mantle.wiring.edge_s3_if_reachable", lambda _w: (None, None)),
    ]


@pytest.mark.asyncio
async def test_query_503_when_no_store_can_be_opened_at_all(client, monkeypatch, tmp_path):
    """A store backend that cannot fail would let this pass while the system is broken: the
    builder would have to return a working accessor for an install with nowhere to put an index —
    which is only possible if some backend defaults to a location nobody chose, or if the search
    path acquired a non-encrypted answer to fall back on. The condition here is constructed, not
    patched at the seam under test: `MANTLE_SSE_DIR` is pointed at a path whose parent is a
    regular file, so `os.makedirs` genuinely fails and there is genuinely no store.
    """
    blocker = tmp_path / "not-a-directory"
    blocker.write_bytes(b"")
    monkeypatch.setenv("MANTLE_SSE_STORE", "file")
    monkeypatch.setenv("MANTLE_SSE_DIR", str(blocker / "index"))

    with ExitStack() as stack:
        for p in _no_edge_object_storage():
            stack.enter_context(p)
        # These two assertions attribute the 503 to the right half: without them the test would
        # pass whenever anything in the chain was missing — including a test environment with no
        # encryption key — and would therefore keep passing after a local fallback made the store
        # half impossible to fail.
        assert wiring._build_oracle() is not None, "the oracle must be the healthy half here"
        assert wiring.sse_index_storage_available() is False
        resp = await client.post("/artifacts/recall",
                                 json={"query_text": "encryption", "candidates": True})

    assert resp.status_code == 503
    body = resp.json()
    assert "candidates" not in body, "a refusal must carry no result set, empty or otherwise"
    assert "detail" in body


@pytest.mark.asyncio
async def test_query_503_when_the_store_is_fine_but_no_key_custody_is(
    client, monkeypatch, tmp_path,
):
    """The local store being treated as sufficient is the half that can fail. The file-backed
    index is available here — a writable directory, a real store object — and the only missing
    piece is the KEK custody the oracle needs. If anything ever answers this request, it answered
    without the ability to derive a key, which means it read something that was not encrypted. A
    future fallback that says "no oracle, but we do have a local index, so let us return what we
    can" fails here rather than shipping.
    """
    monkeypatch.setenv("MANTLE_SSE_STORE", "file")
    monkeypatch.setenv("MANTLE_SSE_DIR", str(tmp_path / "index"))
    # The oracle is a process-level singleton; clear it so the builder actually runs.
    monkeypatch.setattr(wiring, "_oracle_singleton", None)

    def _no_kek():
        raise RuntimeError("no platform encryption key on this box")

    with ExitStack() as stack:
        for p in _no_edge_object_storage():
            stack.enter_context(p)
        stack.enter_context(
            patch("mantle.search.mantle.key_provider.build_key_provider", _no_kek)
        )
        # The store really is there — the 503 below is not about the store being absent.
        assert wiring.sse_index_storage_available() is True
        resp = await client.post("/artifacts/recall",
                                 json={"query_text": "encryption", "candidates": True})

    assert resp.status_code == 503
    assert "candidates" not in resp.json()


@pytest.mark.asyncio
async def test_query_503_when_s3_is_pinned_and_unreachable(client, monkeypatch, tmp_path):
    """`MANTLE_SSE_STORE=s3` must not quietly mean `auto`. An operator who pinned object storage
    has stated that a local index is not an acceptable substitute; silently building one would
    answer from an index that is empty now and divergent later, and would look like a successful
    search. Asserts both halves: the request returns 503, and no local index was created behind
    it.
    """
    local = tmp_path / "index"
    monkeypatch.setenv("MANTLE_SSE_STORE", "s3")
    monkeypatch.setenv("MANTLE_SSE_DIR", str(local))

    with patch("mantle.search.mantle.wiring.edge_s3_if_reachable", lambda _w: (None, None)):
        resp = await client.post("/artifacts/recall",
                                 json={"query_text": "encryption", "candidates": True})

    assert resp.status_code == 503
    assert not local.exists(), "an S3-pinned install silently grew a local index"
