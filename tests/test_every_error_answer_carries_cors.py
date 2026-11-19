"""Every answer Mantle gives the web app carries Access-Control-Allow-Origin, errors included.

A refusal without it reaches the browser as a CORS block and `net::ERR_FAILED`, so the web app
never sees the status and cannot sign in again or retry. Through the whole app: a refusal with
no token, one with a bad token, a 500 from a route that raises. Another origin gets none.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

ORIGIN = "https://workspace.example.test"


@pytest.fixture
def client(monkeypatch):
    from mantle import main
    monkeypatch.setattr(main, "_CORS_ORIGINS", [ORIGIN])

    async def boom():
        raise RuntimeError("a route that fails")

    main.app.add_api_route("/__test_boom", boom, methods=["GET"])
    try:
        yield TestClient(main.app, raise_server_exceptions=False)
    finally:
        main.app.router.routes[:] = [r for r in main.app.router.routes
                                     if getattr(r, "path", None) != "/__test_boom"]


@pytest.mark.parametrize("path, headers, status", [
    ("/artifacts/abc", {}, None),
    ("/artifacts/abc", {"Authorization": "Bearer not-a-token"}, None),
    ("/__test_boom", {}, 500),
])
def test_an_error_answer_carries_the_allow_origin(client, path, headers, status):
    r = client.get(path, headers={"Origin": ORIGIN, **headers})
    assert r.status_code == status if status else r.status_code >= 400
    assert r.headers.get("access-control-allow-origin") == ORIGIN


def test_another_origin_gets_no_allow_origin_on_an_error(client):
    r = client.get("/artifacts/abc", headers={"Origin": "https://evil.example"})
    assert r.status_code >= 400
    assert r.headers.get("access-control-allow-origin") is None


def test_cors_is_the_outermost_middleware():
    from starlette.middleware.cors import CORSMiddleware
    from mantle import main
    assert issubclass(main.app.user_middleware[0].cls, CORSMiddleware)
