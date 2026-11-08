"""Mantle answers a browser's cross-origin call only from the web app's own address (review 7, F1).

Facet calls Mantle's own host directly now that my.agience.ai no longer proxies `/api` to it, so
Mantle is reached cross-origin. `MANTLE_CORS_ORIGINS` (comma-separated exact origins) names who
may; unset, it is the web app's address, `FACET_URI`. Never `*`.
"""
from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _origins(monkeypatch, value):
    from mantle import config
    if value is None:
        monkeypatch.delenv("MANTLE_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("MANTLE_CORS_ORIGINS", value)
    monkeypatch.setattr(config, "FACET_URI", "https://my.agience.ai", raising=False)
    return config.cors_origins()


def test_the_allowed_origins_are_the_setting_or_the_web_app(monkeypatch):
    assert _origins(monkeypatch, None) == ["https://my.agience.ai"]
    assert _origins(monkeypatch, "https://my.agience.ai, https://my.agience.ai/x") == [
        "https://my.agience.ai"]
    assert _origins(monkeypatch, "*") == ["https://my.agience.ai"]          # never a wildcard


def test_a_preflight_from_another_origin_gets_no_allow_header(monkeypatch):
    from mantle import main
    monkeypatch.setattr(main, "_CORS_ORIGINS", ["https://my.agience.ai"])
    c = TestClient(main.app)

    def preflight(origin):
        return c.options("/status", headers={"Origin": origin,
                                             "Access-Control-Request-Method": "GET"})

    ok = preflight("https://my.agience.ai")
    assert ok.headers.get("access-control-allow-origin") == "https://my.agience.ai"
    bad = preflight("https://evil.example")
    assert bad.headers.get("access-control-allow-origin") is None
    assert "*" not in (ok.headers.get("access-control-allow-origin") or "")


def test_the_middleware_never_allows_every_origin():
    from mantle import main
    importlib.reload  # noqa: B018 — the app is built once; its CORS list is read here
    from starlette.middleware.cors import CORSMiddleware
    cors = [m for m in main.app.user_middleware if issubclass(m.cls, CORSMiddleware)]
    assert cors and all("*" not in (m.kwargs.get("allow_origins") or []) for m in cors)
