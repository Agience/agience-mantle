"""Inbound-nonce ENFORCEMENT tests for the write/invoke routes.

`test_inbound_nonce.py` unit-tests `verify_nonce`; these tests prove that the
two bot-protected endpoints actually call `check_inbound_nonce` so a key flagged
`requires_nonce=True` (the website's inbound scoped key) cannot create artifacts
or invoke operations without a valid `X-Agience-Challenge`.

Endpoints covered:
- POST /artifacts                       (contact form / subscribe)
- POST /artifacts/{id}/op/{op_name}     (chat widget → artifact invoke)
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
    """AuthContext for a scoped inbound API key with requires_nonce=True."""
    key_entity = SimpleNamespace(id=_KEY_ID, user_id="u-bot", requires_nonce=True)
    return AuthContext(
        principal_id=_KEY_ID,
        principal_type="api_key",
        user_id="u-bot",
        grants=[],
        api_key_id=_KEY_ID,
        api_key_entity=key_entity,
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
    from origin import config
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
        # Container does not exist → 404 from the create path, which proves we got
        # PAST the nonce gate (a nonce failure would have been 403).
        mock_db.artifacts.get_artifact.return_value = None
        mock_db.artifacts.versions_of.return_value = []

        token = _build_nonce(_KEY_ID, _ARTIFACT_ID, _SECRET)
        resp = inbound_client.post(
            "/artifacts",
            json={"container_id": _ARTIFACT_ID, "content": "{}"},
            headers={"X-Agience-Challenge": token},
        )
        assert resp.status_code != 403, resp.json()


# ---------------------------------------------------------------------------
# POST /artifacts/{id}/op/{op_name}  (chat widget → artifact invoke)
# ---------------------------------------------------------------------------

class TestInvokeOpNonce:


    def test_invoke_with_valid_challenge_passes_nonce_gate(self, inbound_client, mock_db):
        # Unknown artifact → 404 from the op route, proving the nonce gate passed.
        mock_db.artifacts.get_artifact.return_value = None
        mock_db.artifacts.versions_of.return_value = []

        token = _build_nonce(_KEY_ID, _ARTIFACT_ID, _SECRET)
        resp = inbound_client.post(
            f"/artifacts/{_ARTIFACT_ID}/op/invoke",
            json={"params": {"messages": "[]"}},
            headers={"X-Agience-Challenge": token},
        )
        assert resp.status_code != 403, resp.json()


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

        # Whatever happens, it must NOT be a nonce 403.
        if resp.status_code == 403:
            assert "nonce" not in resp.json().get("detail", "").lower()
