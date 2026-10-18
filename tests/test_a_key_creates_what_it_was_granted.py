"""A grant key creates inside a container it may add to; what it creates belongs to its issuer.

2026-10-18: `POST /artifacts` refused every caller without a `user_id`, and a grant key carries
none by design (it must never receive its issuer's own reach). So a key granted `can_create` on a
collection could not record anything there, whatever its bits said, and an agent working on a
key a person issued could read but never write.

Runs on a real store with real grants and the real access check: the key's reach is decided by its
own grants, never by its issuer's.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from mantle.routers import artifacts_router as artifacts
from mantle.services import grant_key_service
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import USER, real_node

COLLECTION = "application/vnd.agience.collection+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _collection(db, name):
    with acting_as(USER, principal_type="user"):
        return ws.create_container(db, USER, content_type=COLLECTION, name=name).id


def _key(db, resource_id, role):
    with acting_as(USER, principal_type="user"):
        grant, raw = grant_key_service.mint(db, user_id=USER, name="agent", resource_id=resource_id,
                                            role=role)
    root = grant_key_service.authenticate(db, raw)
    return AuthContext(principal_id=root.id, principal_type="grant_key", user_id=None,
                       grants=grant_key_service.resolve(db, root), grant_key_id=root.id,
                       bearer_grant=root)


def _request():
    return Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts",
                    "query_string": b""})


async def _create(db, auth, **body):
    with acting_as(auth.principal_id, principal_type=auth.principal_type):
        return await artifacts.create_artifact(
            request=_request(), response=Response(),
            body=artifacts.CreateArtifactRequest(**body), auth=auth, store_db=db)


async def test_a_key_with_create_on_a_collection_records_into_it(db):
    log = _collection(db, "Operations log")
    key = _key(db, log, "editor")

    doc = await _create(db, key, container_id=log, context=json.dumps({"title": "deployed v2.1"}),
                        content="deployed v2.1, three containers healthy")

    assert doc["created_by"] == USER, "what a key creates belongs to the person who issued it"
    with acting_as(USER, principal_type="user"):
        members = ws.list_workspace_artifacts(db, USER, log)
    assert doc["root_id"] in {m.root_id for m in members}


async def test_a_key_creates_nothing_top_level(db):
    key = _key(db, _collection(db, "Operations log"), "editor")
    with pytest.raises(HTTPException) as e:
        await _create(db, key, content_type=COLLECTION, name="a root of its own")
    assert e.value.status_code == 401


async def test_a_key_cannot_create_in_a_collection_it_was_not_granted(db):
    granted, other = _collection(db, "Operations log"), _collection(db, "Private notes")
    key = _key(db, granted, "editor")
    with pytest.raises(HTTPException) as e:
        await _create(db, key, container_id=other, context=json.dumps({"title": "x"}), content="x")
    assert e.value.status_code in (403, 404)


async def test_a_read_only_key_cannot_create(db):
    log = _collection(db, "Operations log")
    key = _key(db, log, "viewer")
    with pytest.raises(HTTPException) as e:
        await _create(db, key, container_id=log, context=json.dumps({"title": "x"}), content="x")
    assert e.value.status_code in (403, 404)
