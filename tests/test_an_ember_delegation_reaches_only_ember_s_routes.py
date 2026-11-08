"""Review 8, R3: a delegation Origin narrows to Ember (`act.sub` `ember`) reaches only what Ember does
for the person, and no delegation is a platform admin.

Ember's acts in Mantle (`ember.mantle_api`): the caller's memory collection (`POST /artifacts` with
`identity` `ember:memory`), a memory in it (`POST /artifacts` with that collection as
`container_id`), reading an artifact (`GET /artifacts/{id}`) and a Read grant (`POST /grants`).
Anything else on an Ember delegation is refused (403).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as deps
from mantle.services.artifact_identity import derive_artifact_id


def _auth(actor="ember", user="alice", scope="ember:memory ember:read"):
    """An Ember delegation as Origin's token exchange issues it: its actor and Ember's scopes."""
    return deps.AuthContext(principal_id=user, principal_type="user", user_id=user, actor=actor, delegation_aud=actor,
                            scope=scope if actor == "ember" else None)


def _req(method, path):
    return SimpleNamespace(method=method, url=SimpleNamespace(path=path))


@pytest.mark.parametrize("method,path", [
    ("POST", "/artifacts"), ("GET", "/grants/my-access")])
def test_ember_s_own_routes_are_open_to_its_delegation(method, path):
    deps.enforce_ember_bounds(_auth(), _req(method, path))


@pytest.mark.parametrize("method,path", [
    ("GET", "/artifacts/visible"), ("DELETE", "/artifacts/a-1"), ("PATCH", "/artifacts/a-1"),
    ("GET", "/grants"), ("POST", "/grants"), ("POST", "/grants/keys"), ("GET", "/system/users"),
    ("POST", "/artifacts/a-1/use"), ("GET", "/artifacts/a-1/content")])
def test_anything_else_is_refused_to_an_ember_delegation(method, path):
    with pytest.raises(HTTPException) as refused:
        deps.enforce_ember_bounds(_auth(), _req(method, path))
    assert refused.value.status_code == 403


def test_another_actor_is_not_bounded_by_ember_s_routes():
    deps.enforce_ember_bounds(_auth(actor=None), _req("DELETE", "/artifacts/a-1"))


def test_an_ember_delegation_creates_only_in_the_caller_s_memory_collection():
    mine = derive_artifact_id("alice", "ember:memory")
    deps.ember_create_allowed(_auth(), {"container_id": mine, "content": "x"})
    # The memory collection is Mantle's to make, at the person's sign-in (review 10).
    for body in ({"identity": "ember:memory"}, {"container_id": "someone-elses"},
                 {"identity": "other"}, {},
                 {"identity": "ember:memory", "container_id": mine},
                 {"source_artifact_id": "a-1", "container_id": mine}):
        with pytest.raises(HTTPException) as refused:
            deps.ember_create_allowed(_auth(), body)
        assert refused.value.status_code == 403


def test_no_delegation_is_a_platform_admin(monkeypatch):
    monkeypatch.setattr(deps, "is_platform_admin", lambda db, uid, **k: True)
    with pytest.raises(HTTPException) as refused:
        deps.require_platform_admin(_auth(actor="chorus"), object())
    assert refused.value.status_code == 403
    assert deps.require_platform_admin(_auth(actor=None), object()) == "alice"
