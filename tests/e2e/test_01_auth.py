"""Auth & tokens — the identity floor every other test stands on.

bootstrap → operator; register + login; API keys; and the negative cases that
prove Mantle actually verifies (bad/absent tokens → 401).
"""
from __future__ import annotations

import uuid

from _api import Api, register


def test_operator_bootstrapped(operator):
    """The operator fixture yields a usable admin token + person id."""
    assert operator["token"]
    assert operator["person_id"] or operator["email"]


def test_register_then_login(user_factory):
    u = user_factory("auth")
    assert u["token"]
    # The token is a bearer usable against Mantle: visible must not 401.
    r = u["api"].get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 200, r.text


def test_register_rejects_short_password():
    uname = f"shortpw-{uuid.uuid4().hex[:8]}"
    r = register(uname, "short", name=uname, email=f"{uname}@agience.test")
    # min length is 12 (auth.password.min_length) → 400.
    assert r.status_code == 400, r.text


def test_login_wrong_password_401(user_factory):
    u = user_factory("auth")
    r = Api().post("/auth/password/login", on="origin",
                   json={"identifier": u["username"], "password": "definitely-wrong-pw"})
    assert r.status_code == 401


def test_mantle_rejects_missing_token():
    """No Authorization header → 401 on a user endpoint."""
    r = Api().get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 401


def test_mantle_rejects_garbage_token():
    r = Api("not-a-real-jwt").get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 401


def _create_api_key(user) -> str:
    # Sovereign: API keys are created on MANTLE (its own lattice store), not Origin.
    r = user["api"].post("/api-keys", json={"name": "e2e-key"})
    assert r.status_code in (200, 201), r.text
    raw = r.json().get("key")
    assert raw and raw.startswith("agc_"), r.json()
    return raw


def test_api_key_created_on_mantle(user):
    """Mantle issues a raw `agc_` personal key exactly once."""
    _create_api_key(user)


def test_api_key_usable_on_mantle(user):
    """The raw `agc_` key authenticates against Mantle end-to-end (same store)."""
    key_api = Api(_create_api_key(user))
    r = key_api.get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 200, r.text
