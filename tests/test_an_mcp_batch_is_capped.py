"""A JSON-RPC batch on `POST /mcp` carries at most `MAX_BATCH` messages.

Each message dispatches a tool under the caller's principal, so an uncapped batch made one
request — one hit on the rate limiter — into as many store operations as the body could hold. An
over-limit batch is refused whole, before any message runs.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mantle.routers import mcp_router as mcp
from mantle.services.dependencies import AuthContext, get_auth, get_store_db


@pytest.fixture()
def dispatched(monkeypatch):
    seen = []

    async def fake_dispatch(message, **_kw):
        seen.append(message.get("id"))
        return {"jsonrpc": "2.0", "id": message.get("id"), "result": {}}

    monkeypatch.setattr(mcp, "_dispatch", fake_dispatch)
    return seen


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(mcp.mcp_router)
    app.dependency_overrides[get_auth] = lambda: AuthContext(user_id="u-1", principal_id="u-1",
                                                             principal_type="user")
    app.dependency_overrides[get_store_db] = lambda: object()
    return TestClient(app)


def _batch(n):
    return [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(n)]


def test_an_oversized_batch_is_refused_before_anything_runs(client, dispatched):
    resp = client.post("/mcp", json=_batch(1000))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == mcp.INVALID_REQUEST
    assert dispatched == []


def test_a_batch_at_the_limit_runs(client, dispatched):
    resp = client.post("/mcp", json=_batch(32))
    assert resp.status_code == 200
    assert len(resp.json()) == 32 and len(dispatched) == 32


def test_one_past_the_limit_is_refused(client, dispatched):
    assert client.post("/mcp", json=_batch(33)).status_code == 400
    assert dispatched == []
