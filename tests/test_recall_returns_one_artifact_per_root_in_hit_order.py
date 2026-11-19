"""`recall` with `artifacts: true` answers the artifacts themselves, one per root in hit order.

A display shows that list as it comes: it fetches nothing per hit, and puts nothing back in order.
Two versions of one root, or one version named twice, come back once, at the first hit's place,
and a hit the reader may not read is left out.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.db import lattice_api as store
from mantle.routers import artifacts_router as artifacts
from mantle.search.mantle import wiring
from mantle.services.dependencies import AuthContext

USER = "u-recall"


def _hit(doc_id, root_id):
    return SimpleNamespace(doc_id=doc_id, score=1.0, root_id=root_id, version_id=doc_id,
                           collection_id="col-1", title=doc_id, description=None, content=None,
                           tags=None)


@pytest.fixture
def db(tmp_path):
    database = store.LatticeDatabase(str(tmp_path / "recall.db"), origin="recall-test")
    for doc_id, root in (("a-1", "a"), ("a-2", "a"), ("b-1", "b"), ("c-1", "c")):
        database.artifacts.put_artifact({"id": doc_id, "root_id": root, "collection_id": "col-1",
                                         "state": "live", "content": doc_id})
    return database


async def _recall(db, monkeypatch, hits, **body):
    result = SimpleNamespace(hits=hits, total=len(hits), parsed_query="q", applied_filters=[],
                             corrections=[], ordering="coverage")
    accessor = SimpleNamespace(search=lambda query: result)
    monkeypatch.setattr(wiring, "build_sse_search_accessor", lambda db, segment="live": accessor)

    def _check(auth, aid, action, store_db):
        if aid.startswith("c"):
            raise HTTPException(status_code=404)
    monkeypatch.setattr(artifacts, "check_access", _check)
    auth = AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    return await artifacts.recall_artifacts(
        body=artifacts.ArtifactRecallRequest(query_text="q", **body), auth=auth, store_db=db)


async def test_each_root_comes_back_once_in_hit_order(db, monkeypatch):
    hits = [_hit("b-1", "b"), _hit("a-2", "a"), _hit("b-1", "b"), _hit("a-1", "a")]
    answer = await _recall(db, monkeypatch, hits, artifacts=True)
    assert [a["id"] for a in answer.artifacts] == ["b-1", "a-2"]
    assert all("container_ids" in a for a in answer.artifacts)
    assert len(answer.hits) == 4, "the hits themselves are answered as ranked"


async def test_an_unreadable_hit_is_left_out(db, monkeypatch):
    answer = await _recall(db, monkeypatch, [_hit("c-1", "c"), _hit("a-1", "a")], artifacts=True)
    assert [a["id"] for a in answer.artifacts] == ["a-1"]


async def test_artifacts_are_answered_only_when_asked(db, monkeypatch):
    answer = await _recall(db, monkeypatch, [_hit("a-1", "a")])
    assert answer.artifacts is None
