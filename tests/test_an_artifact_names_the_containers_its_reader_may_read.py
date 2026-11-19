"""An artifact read through the batch or children listing carries `containers`: the id and name of
every container holding it that the reader may read. A client shows where an artifact is from that
answer alone, and a container the reader may not read never has its name sent.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from mantle.db import lattice_api as store
from mantle.entities.artifact import COLLECTION_CONTENT_TYPE
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_there_is_no_promotion_state import (
    USER, _auth, _container, _write_workspace_create, real_node,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _deny(monkeypatch, *hidden):
    def check(_auth, artifact_id, *_a, **_k):
        if artifact_id in hidden:
            raise HTTPException(status_code=404)
    monkeypatch.setattr(artifacts, "check_access", check)


def _held_twice(db, monkeypatch):
    workspace = _container(db, "Website leads")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, workspace, "a lead from the website", monkeypatch)
        collection = ws.create_container(db, USER, content_type=COLLECTION_CONTENT_TYPE,
                                         name="Private review").id
    store.add_artifact_to_collection(db, collection, root, origin=False)
    return workspace, collection, root


async def _batch(db, root):
    with acting_as(USER, principal_type="user"):
        page = await artifacts.batch_fetch_artifacts(
            body=artifacts.BatchFetchRequest(artifact_ids=[root]), auth=_auth(), store_db=db)
    (card,) = page["items"]
    return card


async def test_the_batch_names_each_readable_container(db, monkeypatch):
    _deny(monkeypatch)
    workspace, collection, root = _held_twice(db, monkeypatch)
    card = await _batch(db, root)
    names = {c["id"]: c["name"] for c in card["containers"]}
    assert names == {workspace: "Website leads", collection: "Private review"}


async def test_an_unreadable_container_is_not_named(db, monkeypatch):
    workspace, collection, root = _held_twice(db, monkeypatch)
    _deny(monkeypatch, collection)
    card = await _batch(db, root)
    assert [c["id"] for c in card["containers"]] == [workspace]
    assert "Private review" not in repr(card["containers"])


async def test_the_children_listing_names_containers_too(db, monkeypatch):
    _deny(monkeypatch)
    workspace, collection, root = _held_twice(db, monkeypatch)
    request = Request({"type": "http", "method": "GET", "headers": [], "path": "/", "query_string": b""})
    with acting_as(USER, principal_type="user"):
        page = await artifacts.list_children(
            artifact_id=workspace, request=request, state="live", content_type=None, limit=100,
            offset=0, workspace_id=None, auth=_auth(), store_db=db)
    card = next(c for c in page["items"] if c["root_id"] == root)
    assert {c["name"] for c in card["containers"]} == {"Website leads", "Private review"}
