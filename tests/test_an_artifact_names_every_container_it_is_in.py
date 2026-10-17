"""An artifact says every container it is in: a workspace is a collection is an artifact.

`container_ids` names them all; `committed_collection_ids` keeps leaving workspaces out, as the
sharing checks read it. Both arrive on the children listing and on the batch fetch that search
results are hydrated through (2026-10-17, staging: a website lead opened from search said it was in
no collection, while its home was the Website leads workspace).
"""
from __future__ import annotations

import pytest
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


@pytest.fixture(autouse=True)
def _allowed(monkeypatch):
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)


def _card_in_a_workspace_and_a_collection(db, monkeypatch):
    workspace = _container(db, "Website leads")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, workspace, "a lead from the website", monkeypatch)
        collection = ws.create_container(db, USER, content_type=COLLECTION_CONTENT_TYPE,
                                         name="Leads").id
    store.add_artifact_to_collection(db, collection, root, origin=False)
    return workspace, collection, root


async def test_the_children_listing_names_every_container(db, monkeypatch):
    workspace, collection, root = _card_in_a_workspace_and_a_collection(db, monkeypatch)
    request = Request({"type": "http", "method": "GET", "headers": [], "path": "/", "query_string": b""})
    with acting_as(USER, principal_type="user"):
        page = await artifacts.list_children(
            artifact_id=workspace, request=request, state="live", content_type=None, limit=100,
            offset=0, workspace_id=None, auth=_auth(), store_db=db)
    card = next(c for c in page["items"] if c["root_id"] == root)
    assert card["container_ids"] == sorted([workspace, collection])
    assert card["committed_collection_ids"] == [collection], "the sharing field still leaves workspaces out"


async def test_the_batch_fetch_search_uses_names_every_container_too(db, monkeypatch):
    workspace, collection, root = _card_in_a_workspace_and_a_collection(db, monkeypatch)
    with acting_as(USER, principal_type="user"):
        page = await artifacts.batch_fetch_artifacts(
            body=artifacts.BatchFetchRequest(artifact_ids=[root]), auth=_auth(), store_db=db)
    (card,) = page["items"]
    assert card["container_ids"] == sorted([workspace, collection])
    assert card["committed_collection_ids"] == [collection]
