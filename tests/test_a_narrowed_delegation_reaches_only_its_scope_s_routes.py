"""Review 9, M7: a narrowed delegation reaches, on every Mantle route, only what its declared scope
lists. Origin's token exchange narrows a person's token to one audience (`act.sub`: `ember`, or a
persona page's origin) and a few scopes; each scope is an explicit allowlist of (method, route), and
a narrowed delegation whose request matches none of its scopes' routes is refused (403), whatever
its audience. A narrowed delegation with no scope reaches nothing.

| scope          | routes                                                                |
|----------------|-----------------------------------------------------------------------|
| `ember:memory` | `POST /artifacts` (and only into the caller's memory collection)      |
| `ember:read`   | `GET /artifacts/{id}`                                                 |
| `ember:share`  | nothing: Ember shares nothing (review 11)                             |
| `persona`      | `GET /artifacts/{id}`, `GET /artifacts/{id}/children`,                |
|                | `GET /artifacts/{id}/content`, `POST /artifacts/recall`               |

A persona server's own delegation (`act.sub` `agience-server-<name>`, no narrowed scope) is not a
narrowed delegation and is bounded elsewhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as deps

ARIA = "https://aria.home.agience.ai"
ALL_ROUTES = [
    ("POST", "/artifacts"), ("GET", "/artifacts/a-1"), ("POST", "/grants"),
    ("GET", "/artifacts/a-1/children"), ("GET", "/artifacts/a-1/content"),
    ("POST", "/artifacts/recall"), ("GET", "/artifacts/visible"), ("PATCH", "/artifacts/a-1"),
    ("DELETE", "/artifacts/a-1"), ("PUT", "/artifacts/a-1/content"), ("GET", "/grants"),
    ("POST", "/grants/keys"), ("DELETE", "/grants/g-1"), ("GET", "/system/users"),
    ("POST", "/artifacts/a-1/use"), ("POST", "/artifacts/a-1/trash"), ("GET", "/events"),
    ("GET", "/grants/my-access"),
]
ALLOWED = {
    "ember:memory": {("POST", "/artifacts")},
    # A verdict, never content (review 10): Ember opens no person's content.
    "ember:read": {("GET", "/grants/my-access")},
    # Ember shares nothing (review 11): sharing happens only in the person's own session.
    "ember:share": set(),
    # No persona page calls Mantle on its narrowed token (review 10, R11: Aria's page sends
    # none, Sage's canon is the host's): the scope reaches nothing until a page needs a route.
    "persona": set(),
}


def _auth(actor, scope):
    return deps.AuthContext(principal_id="alice", principal_type="user", user_id="alice",
                            actor=actor, scope=scope, delegation_aud=actor)


def _req(method, path):
    return SimpleNamespace(method=method, url=SimpleNamespace(path=path))


def _reaches(auth, method, path):
    try:
        deps.enforce_token_bounds(auth, _req(method, path))
        return True
    except HTTPException as exc:
        assert exc.status_code == 403
        return False


@pytest.mark.parametrize("actor,scope", [
    ("ember", "ember:memory"), ("ember", "ember:read"), ("ember", "ember:share"),
    ("ember", "ember:memory ember:read ember:share"), (ARIA, "persona")])
@pytest.mark.parametrize("method,path", ALL_ROUTES)
def test_each_route_is_reached_only_by_a_scope_that_lists_it(actor, scope, method, path):
    allowed = set().union(*(ALLOWED[s] for s in scope.split()))
    assert _reaches(_auth(actor, scope), method, path) == ((method, path) in allowed)


@pytest.mark.parametrize("actor", ["ember", ARIA])
@pytest.mark.parametrize("method,path", ALL_ROUTES)
def test_a_narrowed_delegation_with_no_scope_reaches_nothing(actor, method, path):
    assert not _reaches(_auth(actor, None), method, path)


@pytest.mark.parametrize("method,path", ALL_ROUTES)
def test_a_persona_page_s_scope_on_ember_s_audience_reaches_only_its_list(method, path):
    """The scope decides, not the audience: an `ember` audience carrying `persona` reaches only
    the persona list, and an unknown scope reaches nothing."""
    assert _reaches(_auth("ember", "persona"), method, path) == ((method, path) in
                                                                  ALLOWED["persona"])
    assert not _reaches(_auth("ember", "ember:everything"), method, path)


def test_a_persona_server_s_own_delegation_is_not_a_narrowed_one():
    assert _reaches(_auth("agience-server-lumen", None), "PATCH", "/artifacts/a-1")
    assert _reaches(_auth(None, None), "DELETE", "/artifacts/a-1")
