"""Search — lexical ranking + the light-cone authorization chokepoint.

`POST /artifacts/recall` ranks & paginates; the same endpoint with `candidates: true`
returns the raw authorized candidate set (it can never widen access). Both require the search
backend (the lattice + MinIO) — 503 otherwise, which the suite treats as "skip".
Semantic ranking needs remote prism; without it search degrades to lexical, so
these assertions stay lexical.
"""
from __future__ import annotations

import uuid

import pytest


def _committed_child(api, text: str) -> tuple[str, str]:
    coll = api.create_collection(f"search-{uuid.uuid4().hex[:6]}")
    child = api.create_child(coll["id"], name="doc", content=text)
    api.commit(child["id"])
    return coll["id"], child["id"]


def _search_or_skip(api, query_text, **kw):
    r = api.search(query_text, **kw)
    if r.status_code == 503:
        pytest.skip("search backend unavailable (lattice/MinIO down) — 503")
    return r


@pytest.mark.search
def test_committed_artifact_is_findable(user):
    import time
    token = f"zqx{uuid.uuid4().hex[:8]}"  # rare token, unlikely to collide
    coll_id, child_id = _committed_child(user["api"], f"the magic word is {token}")
    # Indexing can lag the commit — retry a few times before asserting.
    hits = []
    for _ in range(8):
        r = _search_or_skip(user["api"], token, state="committed", scope=[coll_id])
        assert r.status_code == 200, r.text
        hits = r.json().get("hits", [])
        if any(h.get("id") == child_id or h.get("root_id") == child_id for h in hits):
            return
        time.sleep(0.5)
    assert False, f"artifact not searchable after retries: {hits}"


@pytest.mark.search
def test_search_validates_exactly_one_query_arm(user):
    # Neither query_text nor embedding → 400.
    r = user["api"].post("/artifacts/recall", json={"state": "committed"})
    if r.status_code == 503:
        pytest.skip("search backend unavailable")
    assert r.status_code == 400, r.text


@pytest.mark.search
def test_search_rejects_bad_state(user):
    r = user["api"].post("/artifacts/recall", json={"query_text": "x", "state": "bogus"})
    if r.status_code == 503:
        pytest.skip("search backend unavailable")
    assert r.status_code == 400, r.text


@pytest.mark.search
def test_search_is_light_cone_confined(user_factory):
    """A user only finds what they may read: a stranger's committed artifact does
    not surface in the searcher's results."""
    owner = user_factory("owner")
    searcher = user_factory("searcher")
    token = f"zqx{uuid.uuid4().hex[:8]}"
    owner_coll, owner_child = _committed_child(owner["api"], f"secret needle {token}")

    r = _search_or_skip(searcher["api"], token, state="committed")
    assert r.status_code == 200, r.text
    ids = {h.get("id") for h in r.json().get("hits", [])} | \
          {h.get("root_id") for h in r.json().get("hits", [])}
    assert owner_child not in ids, "search must not leak an unshared artifact"


@pytest.mark.search
def test_raw_query_primitive_returns_candidates(user):
    token = f"zqx{uuid.uuid4().hex[:8]}"
    coll_id, _ = _committed_child(user["api"], f"raw candidate {token}")
    r = user["api"].post("/artifacts/recall", json={
        "query_text": token, "state": "committed", "scope": [coll_id],
        "candidates": True, "candidate_budget": 50,
    })
    if r.status_code == 503:
        pytest.skip("search backend unavailable")
    assert r.status_code == 200, r.text
