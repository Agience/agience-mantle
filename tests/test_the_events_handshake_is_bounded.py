"""The `/events` WebSocket handshake carries the same bounds as an HTTP request.

Three gaps, one route. `@app.middleware("http")` never sees a websocket scope, so the handshake
was not rate limited. `_authenticate_ws` called `resolve_auth` directly on the event loop, so each
handshake blocked every other request for a JWT verification or a store round trip. And it called
`resolve_auth` rather than `get_auth`, so neither a platform-scoped token's scope nor a
`requires_nonce` key's challenge was enforced on it.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle import main  # noqa: E402
from mantle.routers import events_router as ev  # noqa: E402
from mantle.services.dependencies import AuthContext  # noqa: E402


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------

def test_the_app_mounts_the_websocket_limiter():
    assert any(m.cls is main._WebSocketRateLimit for m in main.app.user_middleware)


def test_handshakes_past_the_limit_are_refused(monkeypatch):
    monkeypatch.setattr(main, "_RL_HITS", OrderedDict())
    monkeypatch.setattr(main, "_RL_MAX", 2)
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())
    accepted = []

    async def endpoint(ws):
        await ws.accept()
        accepted.append(1)
        await ws.close()

    app = Starlette(routes=[WebSocketRoute("/events", endpoint)])
    app.add_middleware(main._WebSocketRateLimit)
    client = TestClient(app, client=("203.0.113.7", 40000))

    for _ in range(2):
        with client.websocket_connect("/events"):
            pass
    with pytest.raises(WebSocketDisconnect) as refused:
        with client.websocket_connect("/events"):
            pass
    assert refused.value.code == 1013
    assert accepted == [1, 1]


# ---------------------------------------------------------------------------
# Authentication: off the loop, and bounded like get_auth
# ---------------------------------------------------------------------------

class _FakeWs:
    def __init__(self, headers=None):
        self.headers = {"authorization": "Bearer tok", **(headers or {})}
        self.query_params = {}
        self.closed = None

    async def close(self, code=1000, reason=None):
        self.closed = code


@pytest.fixture()
def resolves_to(monkeypatch):
    """Make `resolve_auth` answer `auth`, recording the thread it ran on."""
    seen = {}

    def _install(auth):
        def fake_resolve(token, store_db, request=None):
            seen["thread"] = threading.get_ident()
            return auth
        monkeypatch.setattr(ev, "resolve_auth", fake_resolve)
        monkeypatch.setattr(ev, "get_store_db", lambda: iter([object()]))
        return seen

    return _install


def _run(ws):
    async def go():
        loop_thread = threading.get_ident()
        return loop_thread, await ev._authenticate_ws(ws)
    return asyncio.run(go())


def test_resolution_runs_off_the_event_loop(resolves_to):
    seen = resolves_to(AuthContext(user_id="u-1", principal_id="u-1", principal_type="user"))
    loop_thread, session = _run(_FakeWs())
    assert session is not None
    assert seen["thread"] != loop_thread, "resolve_auth ran on the event loop thread"


def test_a_platform_scoped_token_is_refused_on_the_socket(resolves_to):
    resolves_to(AuthContext(user_id="u-1", principal_id="u-1", principal_type="user",
                            scope="platform.email.send"))
    ws = _FakeWs()
    _, session = _run(ws)
    assert session is None
    assert ws.closed == 4403


_SECRET = "test-nonce-secret-32-bytes-long!!"


def _nonce_key_ctx():
    root = SimpleNamespace(id="key-1", requires_nonce=True, resource_id="col-1")
    return AuthContext(principal_id="key-1", principal_type="grant_key", user_id=None, grants=[],
                       grant_key_id="key-1", bearer_grant=root, target_artifact_id="col-1")


def _nonce(key_id, artifact_id):
    ts = str(int(time.time()))
    sig = hmac.new(_SECRET.encode(), f"{ts}:{artifact_id}:{key_id}".encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{ts}:{artifact_id}:{key_id}:{sig}".encode()).rstrip(b"=").decode()


def test_a_nonce_key_without_its_challenge_is_refused(resolves_to, monkeypatch):
    from mantle import config
    monkeypatch.setattr(config, "INBOUND_NONCE_SECRET", _SECRET, raising=False)
    resolves_to(_nonce_key_ctx())
    ws = _FakeWs()
    _, session = _run(ws)
    assert session is None
    assert ws.closed == 4403


def test_a_nonce_key_with_its_challenge_connects(resolves_to, monkeypatch):
    from mantle import config
    monkeypatch.setattr(config, "INBOUND_NONCE_SECRET", _SECRET, raising=False)
    resolves_to(_nonce_key_ctx())
    ws = _FakeWs({"X-Agience-Challenge": _nonce("key-1", "col-1")})
    _, session = _run(ws)
    assert session is not None and ws.closed is None
