"""The app serves its own schema, on every node, with no flag to get wrong.

`/openapi.json` and `/docs` are unconditional. The schema is the API's contract, not a secret —
every route behind it enforces its own authorization — so a node that hid it would only be
withholding the shape of a surface it still refuses to serve. Unconditional is also what makes the
landing page's `service-doc` / `service-desc` links true wherever the node runs, including a ground
plane with no proxy in front of it.

The rest of this module pins the property that made the flag dangerous in the first place: opening
or closing the schema route must not take the router table with it.
"""
from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _client(monkeypatch):
    monkeypatch.setenv("AGIENCE_NO_DOTENV", "1")
    import mantle.main as m
    importlib.reload(m)
    return TestClient(m.app)


def test_the_schema_and_the_console_are_both_served(monkeypatch):
    """One surface, no switch. A console with no schema behind it is broken in a confusing way,
    and a node that serves neither cannot describe itself to a client or to a person."""
    c = _client(monkeypatch)
    assert c.get("/openapi.json").status_code == 200
    assert c.get("/docs").status_code == 200


def test_the_landing_page_advertises_links_that_resolve(monkeypatch):
    """`/` publishes `service-doc` and `service-desc`. Advertising a 404 is worse than
    advertising nothing, so the links are followed here rather than assumed."""
    c = _client(monkeypatch)
    links = c.get("/").json()["links"]
    assert c.get(links["service-doc"]).status_code == 200
    assert c.get(links["service-desc"]).status_code == 200


def test_the_schema_route_does_not_own_the_router_table(monkeypatch):
    """Pins against the schema route and the real surface being wired together: every endpoint
    404-ing while the node still looks healthy — uvicorn binds, `/` answers — is a failure mode
    that reads as fine from outside. Asserts the real surface still routes."""
    c = _client(monkeypatch)
    assert c.get("/").status_code == 200
    # 401, not 404: the route EXISTS and refused an unauthenticated caller.
    assert c.get("/artifacts/visible").status_code == 401
    assert c.post("/mcp", json={"jsonrpc": "2.0", "method": "ping", "id": 1}).status_code == 401


def test_the_callback_route_serves_the_browser_document(monkeypatch):
    """The page is the handler: the authorization code arrives as `?code=` and the JavaScript that
    exchanges it lives in the page, so `/auth/callback` must return the document, not a redirect
    or a stub. A 404 or an empty response there would leave the user completing sign-in at the IdP
    and landing on nothing, with a live code in the URL."""
    c = _client(monkeypatch)
    r = c.get("/auth/callback")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "REDIRECT_URI" in r.text and "code_challenge" in r.text


def test_the_callback_is_never_cached(monkeypatch):
    """It is reached with a live authorization code in the URL."""
    c = _client(monkeypatch)
    r = c.get("/auth/callback")
    assert "no-store" in r.headers.get("cache-control", "")
    assert r.headers.get("referrer-policy") == "no-referrer"
