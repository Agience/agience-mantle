"""`/v2` and `/git` answer an anonymous caller 401 — asserted, not assumed.

Why this file exists. Both routers' docstrings state that every route requires an authenticated
caller and that anonymous is 401. **Neither suite tested it**, because `conftest.override_dependencies`
is autouse and replaces `get_auth` for the whole session — so every existing `/v2` and `/git` test
runs as an authenticated user and could not have caught an open surface.

That gap matters more than most: these two routers are the first NEW public surfaces this service has
grown, they are mounted on every node, and the edge Caddy snippets proxy every path — so on a
deployed node they are reachable from the internet. `mantle_common.sh`'s deploy check already treats
"unauthenticated `POST /mcp` → 401" as the check that catches *a lattice open to the internet*; this
is the same property for the two surfaces that were added after that check was written.

The override is removed for these tests specifically, so the app's real auth dependency runs.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mantle.main import app
from mantle.services.dependencies import get_auth

client = TestClient(app)


@pytest.fixture
def anonymous():
    """Drop the autouse auth override so the REAL dependency answers.

    Restored afterwards rather than left cleared: the override is session-scoped and every other
    test in the run depends on it.
    """
    saved = app.dependency_overrides.pop(get_auth, None)
    try:
        yield
    finally:
        if saved is not None:
            app.dependency_overrides[get_auth] = saved


#: Every route the sovereign code plane mounts, with a method and a path that reaches it.
#: `/v2/` is the version endpoint a client hits first; the rest are the read paths.
SURFACES = [
    ("GET", "/v2/"),
    ("GET", "/v2/agience/agience-mantle/manifests/sha256:" + "a" * 64),
    ("HEAD", "/v2/agience/agience-mantle/manifests/sha256:" + "a" * 64),
    ("GET", "/v2/agience/agience-mantle/blobs/sha256:" + "a" * 64),
    ("HEAD", "/v2/agience/agience-mantle/blobs/sha256:" + "a" * 64),
    ("GET", "/git/prism/info/refs?service=git-upload-pack"),
    ("POST", "/git/prism/git-upload-pack"),
]


@pytest.mark.parametrize("method,path", SURFACES, ids=lambda v: str(v))
def test_an_anonymous_caller_is_refused(anonymous, method, path):
    """401, and NOT 200 — the property that keeps a node's registry and source off the open
    internet.

    A 404 would also be wrong here and is worth distinguishing: it would mean the route is not
    mounted, which is a different failure with a different fix. The assertion is on the code being
    401 specifically.
    """
    r = client.request(method, path)
    assert r.status_code == 401, (
        "%s %s answered %s to an ANONYMOUS caller — this surface is reachable without credentials"
        % (method, path, r.status_code))


def test_the_refusal_carries_a_challenge(anonymous):
    """A 401 without `WWW-Authenticate` tells a client it was refused and not how to proceed.

    `main.py` attaches the RFC 9728 challenge to every 401 through its exception handler, so this is
    really asserting that these routers go through it rather than answering 401 some other way.
    """
    r = client.get("/v2/")
    assert r.status_code == 401
    assert "www-authenticate" in {k.lower() for k in r.headers}, dict(r.headers)


def test_the_fixture_actually_removes_the_override():
    """The negative control, and it is not decoration.

    If `override_dependencies` were ever renamed or its key changed, the fixture above would pop
    nothing, every request would still run authenticated, and all the assertions in this file would
    pass while measuring nothing at all — the exact vacuous-green shape they exist to prevent.
    """
    before = client.get("/v2/")
    assert before.status_code == 200, (
        "with the override in place this should be an authenticated 200; got %s — the override is "
        "not doing what this file assumes, so the anonymous tests are not testing anonymity"
        % before.status_code)
