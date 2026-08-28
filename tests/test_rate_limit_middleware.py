"""The limiter over the wire: a spoofed `X-Forwarded-For` buys nothing, and the store is bounded.

`test_rate_limit_client_identity.py` pins the identity function and the eviction sweep on their
own. These drive the middleware itself, because the two failures being closed are only visible
end to end: a caller that keeps getting 200s past its limit, and a store that keeps a key for
every value that caller invented.

The middleware is mounted on a bare app rather than on `mantle.main.app` — it reads its
configuration from module globals, so it is the same code under the same knobs, without a
lifespan, key material or a store standing between the request and the counter.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle import main  # noqa: E402

_LIMIT = 5


@pytest.fixture()
def store(monkeypatch):
    """A store of this test's own, so nothing leaks between cases or in from the app."""
    fresh: OrderedDict = OrderedDict()
    monkeypatch.setattr(main, "_RL_HITS", fresh)
    monkeypatch.setattr(main, "_RL_MAX", _LIMIT)
    return fresh


@pytest.fixture()
def app(store):
    app = FastAPI()
    app.middleware("http")(main._rate_limit)

    @app.get("/thing")
    def thing():
        return {"ok": True}

    return app


def _client(app, peer: str) -> TestClient:
    return TestClient(app, client=(peer, 40000))


def test_a_spoofed_header_does_not_lift_the_limit(app, store, monkeypatch):
    """The bypass, end to end. Every request names a different client; the limit still lands."""
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())
    client = _client(app, "203.0.113.7")

    codes = [
        client.get("/thing", headers={"X-Forwarded-For": f"198.51.100.{i}"}).status_code
        for i in range(_LIMIT + 3)
    ]

    assert codes[:_LIMIT] == [200] * _LIMIT
    assert codes[_LIMIT:] == [429, 429, 429]


def test_the_store_does_not_grow_with_the_spoofed_values(app, store, monkeypatch):
    """500 distinct claims about who is calling, one socket — and one entry in the store.

    The memory-growth path stated as a bound: without it an unauthenticated caller mints a
    permanent key per request just by varying a header it writes itself.
    """
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())
    monkeypatch.setattr(main, "_RL_MAX", 10_000)
    client = _client(app, "203.0.113.7")

    for i in range(500):
        client.get("/thing", headers={"X-Forwarded-For": f"10.{i // 256}.{i % 256}.1"})

    assert list(store) == ["203.0.113.7"]


def test_a_trusted_proxy_gets_the_client_its_own_limit(app, store, monkeypatch):
    """Behind a real edge the limit follows the forwarded client, not the edge's one address."""
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", main._parse_trusted_proxies("10.0.0.1"))
    client = _client(app, "10.0.0.1")

    for _ in range(_LIMIT):
        assert client.get("/thing", headers={"X-Forwarded-For": "198.51.100.9"}).status_code == 200
    assert client.get("/thing", headers={"X-Forwarded-For": "198.51.100.9"}).status_code == 429

    # A different forwarded client is a different window, and is unaffected by the first's.
    assert client.get("/thing", headers={"X-Forwarded-For": "198.51.100.10"}).status_code == 200
    assert list(store) == ["198.51.100.9", "198.51.100.10"]


def test_the_limit_is_per_socket_peer_when_no_proxy_is_trusted(app, store, monkeypatch):
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())

    for _ in range(_LIMIT):
        assert _client(app, "203.0.113.7").get("/thing").status_code == 200
    assert _client(app, "203.0.113.7").get("/thing").status_code == 429
    assert _client(app, "203.0.113.8").get("/thing").status_code == 200


def test_a_window_that_aged_out_is_dropped_rather_than_kept(app, store, monkeypatch):
    """Time is the only thing moved here — the entry goes because nothing is left in it."""
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())
    _client(app, "203.0.113.7").get("/thing")
    assert list(store) == ["203.0.113.7"]

    clock = [main.time.monotonic() + main._RL_WINDOW_S + 1.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])

    _client(app, "203.0.113.8").get("/thing")

    assert list(store) == ["203.0.113.8"]


def test_the_limiter_is_off_at_zero(app, store, monkeypatch):
    """`docker-compose.yml` sets 0, because one local stack shares one source address."""
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())
    monkeypatch.setattr(main, "_RL_MAX", 0)
    client = _client(app, "203.0.113.7")

    assert [client.get("/thing").status_code for _ in range(20)] == [200] * 20
    assert not store


def test_the_exempt_paths_are_never_counted(app, store, monkeypatch):
    """`/status` and friends answer a load balancer, which must not be able to exhaust a
    window that then refuses real traffic from the same address."""
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())

    @app.get("/status")
    def status():
        return {"status": "ok"}

    client = _client(app, "203.0.113.7")
    for _ in range(_LIMIT * 3):
        assert client.get("/status").status_code == 200
    assert not store
