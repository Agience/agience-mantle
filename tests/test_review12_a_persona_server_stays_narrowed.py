"""Review 12, H5: narrowing survives the persona-to-server exchange.

A persona exchanges the caller's delegation narrowed to it for its own server delegation (`aud` =
`act.sub` = `agience-server-<name>`). Origin now stamps that server delegation with `narrowed: true`
and the narrowed scope (`persona:<name>`). Mantle treats it as narrowed: the persona server reaches
only the routes that persona calls, never `POST /grants` or a delete.
"""
from __future__ import annotations

import types

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as md


def _server(name, *, narrowed=True, scope=None):
    aud = f"agience-server-{name}"
    return types.SimpleNamespace(
        delegation_aud=aud, actor=aud, narrowed=narrowed,
        scope=scope if scope is not None else (f"persona:{name}" if narrowed else None),
        principal_id="p-1", user_id="p-1")


def _req(method, path):
    return types.SimpleNamespace(method=method, url=types.SimpleNamespace(path=path))


def test_a_narrowed_server_delegation_is_narrowed():
    assert md.is_narrowed(_server("lumen"))


def test_the_narrowed_claim_alone_marks_it():
    assert md.is_narrowed(_server("lumen", scope=""))


@pytest.mark.parametrize("name", ["iris", "lumen", "sage", "crystal"])
@pytest.mark.parametrize("method,path", [("POST", "/grants"), ("DELETE", "/grants/g-1"),
                                         ("DELETE", "/artifacts/a-1"),
                                         ("PATCH", "/artifacts/a-1"),
                                         ("POST", "/system/users/u/grant-admin")])
def test_a_persona_server_never_grants_or_deletes(name, method, path):
    with pytest.raises(HTTPException) as refused:
        md.enforce_narrowed_bounds(_server(name), _req(method, path))
    assert refused.value.status_code == 403


@pytest.mark.parametrize("name,method,path", [
    ("lumen", "GET", "/artifacts/a-1"), ("lumen", "POST", "/artifacts/a-1/use"),
    ("lumen", "GET", "/artifacts/visible"), ("lumen", "POST", "/artifacts/a-1/op/invoke"),
    ("iris", "POST", "/artifacts/sec-1/use"), ("iris", "GET", "/grants/my-access"),
    ("sage", "POST", "/artifacts/recall"), ("crystal", "GET", "/artifacts/a-1/children"),
])
def test_a_persona_server_reaches_the_routes_its_persona_calls(name, method, path):
    md.enforce_narrowed_bounds(_server(name), _req(method, path))


def test_a_persona_server_reaches_only_its_own_list():
    with pytest.raises(HTTPException):
        md.enforce_narrowed_bounds(_server("iris"), _req("POST", "/artifacts/recall"))


def test_a_plain_server_delegation_is_unaffected():
    plain = _server("lumen", narrowed=False)
    assert not md.is_narrowed(plain)
    md.enforce_narrowed_bounds(plain, _req("POST", "/grants"))


def test_the_claim_is_read_from_the_token():
    """`narrowed` rides from Origin's claims into the AuthContext."""
    assert "narrowed" in md.AuthContext.__dataclass_fields__
