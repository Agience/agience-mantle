"""A `requires_nonce` key presents its challenge on every route, not only on `POST /artifacts`.

`check_inbound_nonce` was called by `create_artifact` alone, so the website's inbound key — the
one credential deliberately shipped to browsers — reached every other route its grants allowed
with no challenge at all. `get_auth` now applies it to every authenticated request. These call
`get_auth` directly, the way FastAPI does, for a request to a route that is not the create path.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from mantle.services import dependencies as deps
from mantle.services.dependencies import AuthContext

_SECRET = "test-nonce-secret-32-bytes-long!!"
_KEY = "key-inbound-1"
_TARGET = "contact-collection-uuid"


def _nonce():
    ts = str(int(time.time()))
    sig = hmac.new(_SECRET.encode(), f"{ts}:{_TARGET}:{_KEY}".encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{ts}:{_TARGET}:{_KEY}:{sig}".encode()).rstrip(b"=").decode()


def _key_ctx(requires_nonce=True):
    root = SimpleNamespace(id=_KEY, requires_nonce=requires_nonce, resource_id=_TARGET)
    return AuthContext(principal_id=_KEY, principal_type="grant_key", user_id=None, grants=[],
                       grant_key_id=_KEY, bearer_grant=root, target_artifact_id=_TARGET)


def _request(headers=None):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "path": f"/artifacts/{_TARGET}",
                    "raw_path": b"", "query_string": b"", "headers": raw})


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    from mantle import config
    monkeypatch.setattr(config, "INBOUND_NONCE_SECRET", _SECRET, raising=False)


def _get_auth(ctx, request, monkeypatch):
    monkeypatch.setattr(deps, "resolve_auth", lambda token, store_db, request=None: ctx)
    return asyncio.run(deps.get_auth(token=f"{_TARGET}:agk_x", store_db=object(), request=request))


def test_a_read_without_the_challenge_is_refused(monkeypatch):
    with pytest.raises(HTTPException) as refused:
        _get_auth(_key_ctx(), _request(), monkeypatch)
    assert refused.value.status_code == 403


def test_a_read_with_a_bad_challenge_is_refused(monkeypatch):
    with pytest.raises(HTTPException) as refused:
        _get_auth(_key_ctx(), _request({"X-Agience-Challenge": "nope"}), monkeypatch)
    assert refused.value.status_code == 403


def test_a_read_with_the_challenge_is_authenticated(monkeypatch):
    auth = _get_auth(_key_ctx(), _request({"X-Agience-Challenge": _nonce()}), monkeypatch)
    assert auth.grant_key_id == _KEY


def test_a_key_that_does_not_require_one_is_unaffected(monkeypatch):
    auth = _get_auth(_key_ctx(requires_nonce=False), _request(), monkeypatch)
    assert auth.grant_key_id == _KEY
