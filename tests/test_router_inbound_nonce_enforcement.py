"""Inbound-nonce ENFORCEMENT tests for the write routes.

`test_inbound_nonce.py` unit-tests `verify_nonce`; these tests prove that the
bot-protected endpoint actually calls `check_inbound_nonce` so a GRANT KEY flagged
`requires_nonce=True` (the website's inbound card key) cannot create artifacts
without a valid `X-Agience-Challenge`.

Endpoints covered:
- POST /artifacts                       (contact form / subscribe)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.routers.artifacts_router import router  # noqa: E402
from mantle.services.dependencies import AuthContext, get_auth, get_store_db  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


_SECRET = "test-nonce-secret-32-bytes-long!!"
_ARTIFACT_ID = "contact-collection-uuid"
_KEY_ID = "key-inbound-1"


def _build_nonce(key_id: str, artifact_id: str, secret: str, *, ts: int | None = None) -> str:
    """Replicate Origin's `issue_nonce` token shape (see test_inbound_nonce.py)."""
    ts = int(time.time()) if ts is None else ts
    payload = f"{ts}:{artifact_id}:{key_id}"
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    raw = f"{ts}:{artifact_id}:{key_id}:{sig}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).rstrip(b"=").decode("utf-8")


@pytest.fixture()
def app():
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def mock_db():
    return MagicMock()


def _inbound_key_ctx() -> AuthContext:
    """AuthContext for an inbound grant key with requires_nonce=True.

    `requires_nonce` rides on the grant itself, so the nonce gate reads `bearer_grant`
    — the root grant the token resolved to.
    """
    root = SimpleNamespace(id=_KEY_ID, requires_nonce=True, resource_id=_ARTIFACT_ID)
    return AuthContext(
        principal_id=_KEY_ID,
        principal_type="grant_key",
        user_id=None,
        grants=[],
        grant_key_id=_KEY_ID,
        bearer_grant=root,
        target_artifact_id=_ARTIFACT_ID,
    )


@pytest.fixture()
def inbound_client(app, mock_db):
    app.dependency_overrides[get_auth] = lambda: _inbound_key_ctx()
    app.dependency_overrides[get_store_db] = lambda: mock_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _set_nonce_secret(monkeypatch):
    from mantle import config
    monkeypatch.setattr(config, "INBOUND_NONCE_SECRET", _SECRET, raising=False)


# ---------------------------------------------------------------------------
# POST /artifacts  (contact form / subscribe)
# ---------------------------------------------------------------------------

class TestCreateArtifactNonce:
    def test_create_without_challenge_is_403(self, inbound_client):
        resp = inbound_client.post(
            "/artifacts",
            json={"container_id": _ARTIFACT_ID, "content": "{}"},
        )
        assert resp.status_code == 403
        assert "Nonce required" in resp.json()["detail"]

    def test_create_with_bad_challenge_is_403(self, inbound_client):
        resp = inbound_client.post(
            "/artifacts",
            json={"container_id": _ARTIFACT_ID, "content": "{}"},
            headers={"X-Agience-Challenge": "not-a-valid-token"},
        )
        assert resp.status_code == 403
        assert "Invalid or expired nonce" in resp.json()["detail"]

    def test_create_with_valid_challenge_passes_nonce_gate(self, inbound_client, mock_db):
        """A valid challenge gets PAST the nonce gate and dies at the next check instead.

        The expected code is asserted rather than `!= 403`: the two sibling tests above show the
        nonce gate refuses with 403 and does so before this point, so the exact code the request
        reaches next is what says the gate was cleared. `!= 403` is satisfied by a 500 from a route
        that crashed on the way in, by a 200 from a route that stopped checking anything, and by
        every other status this endpoint could ever return.
        """
        mock_db.artifacts.get_artifact.return_value = None
        mock_db.artifacts.versions_of.return_value = []

        token = _build_nonce(_KEY_ID, _ARTIFACT_ID, _SECRET)
        resp = inbound_client.post(
            "/artifacts",
            json={"container_id": _ARTIFACT_ID, "content": "{}"},
            headers={"X-Agience-Challenge": token},
        )
        # 401 "User identification required": past the nonce gate, refused by the NEXT guard —
        # a grant key alone does not identify a user.
        assert resp.status_code == 401, resp.json()
        assert "nonce" not in resp.json()["detail"].lower(), resp.json()


# ---------------------------------------------------------------------------
# A normal user JWT is unaffected (no nonce required)
# ---------------------------------------------------------------------------

class TestUserPrincipalUnaffected:
    def test_user_create_without_challenge_not_blocked_by_nonce(self, app, mock_db):
        ctx = AuthContext(user_id="u-1", principal_type="user", grants=[])
        app.dependency_overrides[get_auth] = lambda: ctx
        app.dependency_overrides[get_store_db] = lambda: mock_db
        mock_db.artifacts.get_artifact.return_value = None
        mock_db.artifacts.versions_of.return_value = []

        client = TestClient(app)
        resp = client.post("/artifacts", json={"container_id": "ws-1", "content": "{}"})
        app.dependency_overrides.clear()

        # 404: the container does not exist, which is the create path's own refusal — reached, so
        # the nonce gate did not stand in the way. Asserted rather than guarded by
        # `if resp.status_code == 403:`, which checked nothing at all on every run where the
        # response was not a 403 — including a run where the gate started rejecting user tokens
        # with some other status.
        assert resp.status_code == 404, resp.json()
        assert "nonce" not in resp.json().get("detail", "").lower(), resp.json()
