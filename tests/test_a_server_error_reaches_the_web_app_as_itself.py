"""A server error reaches the web app as itself, not as a CORS failure (BUG-056).

Starlette answers an unhandled exception from its outermost error middleware, outside Mantle's
CORS middleware, so the catch-all handler's response carried no `Access-Control-Allow-Origin`.
A browser then reported every 500 as "blocked by CORS policy", and the web app could only say
"Failed to create artifact". The handler now adds the CORS headers for an allowed origin itself.

A refusal to seal (Mantle never writes plaintext) is a 503 that says why: on staging, an item made
inside a collection sealed under production's keys, which staging does not hold.
"""
from __future__ import annotations

import json

import pytest
from starlette.requests import Request

from mantle import main
from mantle.db.doc_boundary import ContentEncryptionError as FieldSealRefused
from mantle.services.content_service import ContentEncryptionError as BodySealRefused

APP = "https://my.example.test"


def _request(origin: str | None) -> Request:
    headers = [(b"origin", origin.encode())] if origin else []
    return Request({"type": "http", "method": "POST", "path": "/artifacts", "headers": headers,
                    "query_string": b"", "path_params": {}})


@pytest.fixture
def allowed(monkeypatch):
    monkeypatch.setattr(main, "_CORS_ORIGINS", [APP])


@pytest.mark.asyncio
async def test_a_500_carries_cors_for_an_allowed_origin(allowed):
    resp = await main.unhandled_exception_logger(_request(APP), RuntimeError("boom"))
    assert resp.status_code == 500
    assert resp.headers.get("access-control-allow-origin") == APP
    assert "Origin" in resp.headers.get("vary", "")


@pytest.mark.asyncio
async def test_a_500_carries_no_cors_for_another_origin(allowed):
    resp = await main.unhandled_exception_logger(_request("https://elsewhere.test"), RuntimeError("boom"))
    assert resp.status_code == 500
    assert "access-control-allow-origin" not in resp.headers
    resp = await main.unhandled_exception_logger(_request(None), RuntimeError("boom"))
    assert "access-control-allow-origin" not in resp.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [FieldSealRefused("context field encryption unavailable"),
                                 BodySealRefused("content encryption unavailable")])
async def test_a_refusal_to_seal_is_a_503_that_says_why(allowed, exc):
    resp = await main.unhandled_exception_logger(_request(APP), exc)
    assert resp.status_code == 503
    assert "can't be sealed" in json.loads(resp.body)["detail"]
    assert resp.headers.get("access-control-allow-origin") == APP
