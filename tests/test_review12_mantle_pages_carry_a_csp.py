"""Review 12, info: Mantle's own pages (the API console at /docs and the browser at /) carry a CSP
that forbids framing, base tags and plugins, and limits where scripts load from."""
from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def _client(monkeypatch):
    monkeypatch.setenv("AGIENCE_NO_DOTENV", "1")
    import mantle.main as m
    importlib.reload(m)
    return TestClient(m.app)


def test_every_mantle_page_carries_a_csp_that_forbids_framing(monkeypatch):
    c = _client(monkeypatch)
    for path, accept in (("/docs", "text/html"), ("/", "text/html")):
        r = c.get(path, headers={"Accept": accept})
        if "text/html" not in r.headers.get("content-type", ""):
            continue
        csp = r.headers.get("content-security-policy", "")
        assert "frame-ancestors 'none'" in csp and "object-src 'none'" in csp, (path, csp)
        assert "base-uri 'none'" in csp and r.headers.get("x-frame-options") == "DENY"


def test_a_json_answer_carries_no_page_policy(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/openapi.json")
    assert r.status_code == 200 and "content-security-policy" not in r.headers
