""": a warm sweep is bounded, and `materialized: 0` no longer means three things.

THE DEFECT. `POST /artifacts/{id}/warm` answered `{collection_id, materialized}` and nothing
else, over a sweep with no bound and a `try/except Exception: logger.warning` around its whole
body. So `materialized: 0` meant "nothing needed warming", OR "the member listing raised and was
swallowed", OR "it stopped early" — and all three answered `200`. The caller could not tell a
completed no-op from a failure.

This is the same confusion `_page` reports `total: None` to avoid, and the audit pointed at the
same precedent: `list_visible` grew an `EdgesTruncated` fallback for exactly this failure shape.

The sweep stays SYNCHRONOUS on purpose. It only ENQUEUES — indexing is already asynchronous —
so the request holds a worker for one `is_materialized` read plus one enqueue per member, which is
now bounded. Backgrounding the route would need a job handle this service does not have.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from mantle.db import lattice_api as store
from mantle.entities.artifact import Artifact as ArtifactEntity, WORKSPACE_CONTENT_TYPE
from mantle.services import workspace_service as ws


@pytest.fixture
def db():
    return store.LatticeDatabase(os.path.join(tempfile.mkdtemp(), "warm.db"), origin="warm-test")


def _container_with(db, n, user="u"):
    cid = ws.create_container(db, user, content_type=WORKSPACE_CONTENT_TYPE, name="c").id
    for i in range(n):
        aid = "m%d" % i
        store.create_artifact(db, ArtifactEntity(
            id=aid, root_id=aid, collection_id=cid, created_by=user,
            state=ArtifactEntity.STATE_COMMITTED, name=aid))
    return cid


def test_a_complete_sweep_says_so(db):
    out = ws.warm_collection(db, _container_with(db, 5))
    assert out["complete"] is True and out["truncated"] is False
    assert out["examined"] == 5 and out["materialized"] == 5 and out["failed"] == 0


def test_a_sweep_that_hits_its_bound_says_truncated(db):
    out = ws.warm_collection(db, _container_with(db, 7), limit=3)
    assert out["truncated"] is True, out
    assert out["examined"] == 3, out
    assert out["complete"] is True, "stopping at the bound is not a failure"


def test_a_failed_listing_is_not_reported_as_nothing_to_do(db, monkeypatch):
    """The point of the whole item. Before this, both cases answered `materialized: 0`."""
    cid = _container_with(db, 5)

    def boom(*_a, **_k):
        raise RuntimeError("listing failed")

    monkeypatch.setattr(ws.store, "list_collection_artifacts", boom)
    failed = ws.warm_collection(db, cid)
    monkeypatch.undo()

    assert failed["complete"] is False, "a swallowed failure still looks complete"
    assert failed["materialized"] == 0

    empty = ws.warm_collection(db, ws.create_container(
        db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="empty").id)
    assert empty["materialized"] == 0 and empty["complete"] is True
    # The two are now distinguishable — which they were not.
    assert failed["complete"] != empty["complete"]


def test_the_default_bound_exists_and_is_a_real_number():
    assert isinstance(ws.WARM_SWEEP_LIMIT, int) and ws.WARM_SWEEP_LIMIT > 0


def test_an_enqueue_failure_is_counted_not_swallowed(db, monkeypatch):
    cid = _container_with(db, 3)
    import mantle.search.ingest.pipeline_unified as pipe

    def boom(*_a, **_k):
        raise RuntimeError("enqueue failed")

    monkeypatch.setattr(pipe, "enqueue_index_artifact", boom)
    out = ws.warm_collection(db, cid)
    monkeypatch.undo()

    assert out["failed"] == 3, out
    assert out["materialized"] == 0
    assert out["complete"] is True, "per-member failures do not make the sweep incomplete"
