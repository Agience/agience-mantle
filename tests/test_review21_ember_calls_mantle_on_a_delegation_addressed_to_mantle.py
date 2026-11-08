"""Ember calls Mantle on a delegation Origin addressed to Mantle (review 21, F6).

Ember forwarded the delegation it was itself sent (`aud` `ember`) to Mantle, and Mantle could
replay it at Ember. Ember now exchanges it at Origin for the same person's delegation addressed to
Mantle (`aud` `mantle`, actor `ember`, Ember's scopes, never longer than the one it came from),
which Ember refuses as a bearer. Mantle bounds it exactly as an Ember delegation.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as deps
from mantle.services.artifact_identity import derive_artifact_id


def _auth():
    """Origin's exchange of Ember's delegation for one addressed to Mantle."""
    return deps.AuthContext(principal_id="alice", principal_type="user", user_id="alice",
                            actor="ember", delegation_aud="mantle", narrowed=True,
                            scope="ember:memory ember:read")


def _req(method, path):
    return SimpleNamespace(method=method, url=SimpleNamespace(path=path))


def test_it_is_an_ember_delegation():
    assert deps.is_ember(_auth()) and _auth().narrowed_audience == "ember"


@pytest.mark.parametrize("method,path", [("POST", "/artifacts"), ("GET", "/grants/my-access")])
def test_it_reaches_ember_s_routes(method, path):
    deps.enforce_narrowed_bounds(_auth(), _req(method, path))


@pytest.mark.parametrize("method,path", [("GET", "/artifacts/visible"), ("POST", "/grants"),
                                         ("POST", "/artifacts/a-1/use")])
def test_it_reaches_nothing_else(method, path):
    with pytest.raises(HTTPException) as refused:
        deps.enforce_narrowed_bounds(_auth(), _req(method, path))
    assert refused.value.status_code == 403


def test_it_creates_only_in_the_caller_s_memory_collection():
    mine = derive_artifact_id("alice", "ember:memory")
    deps.ember_create_allowed(_auth(), {"container_id": mine, "content": "x"})
    with pytest.raises(HTTPException):
        deps.ember_create_allowed(_auth(), {"container_id": "someone-elses"})


def test_a_platform_delegation_to_mantle_is_not_ember_s():
    platform = deps.AuthContext(principal_id="sys", principal_type="user", user_id="sys",
                                actor="ember", delegation_aud="mantle",
                                scope="platform.email.send")
    assert not deps.is_ember(platform)
