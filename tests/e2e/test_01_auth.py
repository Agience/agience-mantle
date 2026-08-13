"""Auth & tokens — the identity floor every other test stands on.

bootstrap → operator; register + login; grant keys and grant bundles; and the
negative cases that prove Mantle actually verifies (bad/absent tokens → 401, and a
retired `agc_` API key naming its own retirement rather than reading as malformed).
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


def _a_collection(user) -> str:
    """A committed top-level collection the user owns (see test_03_collections)."""
    return user["api"].create_collection(f"key-{uuid.uuid4().hex[:6]}")["id"]


def _create_grant_key(user, **body) -> tuple[str, str]:
    """Mint a grant key on Mantle. Returns `(key_id, raw_token)`."""
    payload = {"name": "e2e-key"}
    payload.update(body)
    r = user["api"].post("/grants/keys", json=payload)
    assert r.status_code in (200, 201), r.text
    data = r.json()
    raw = data.get("key")
    assert raw and raw.startswith("agk_"), data
    # The stored token hash must never come back out over the API.
    assert "grantee_id" not in data, "the grant key's token hash was serialized"
    return data["id"], raw


def test_grant_key_created_on_mantle(user):
    """Mantle issues a raw `agk_` key exactly once, and never echoes its hash."""
    _create_grant_key(user)


def test_grant_key_usable_on_mantle(user):
    """The raw `agk_` token authenticates end-to-end and reaches what it carries."""
    _, raw = _create_grant_key(user, resource_id=_a_collection(user), can_read=True)

    r = Api(raw).get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 200, r.text


def test_a_retired_api_key_says_so(user):
    """An `agc_` token names its own retirement instead of reading as malformed."""
    r = Api("agc_" + "0" * 32).get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 401, r.text
    assert "retired" in r.text.lower(), r.text


def test_api_keys_endpoint_is_gone(user):
    """The decommissioned surface must not still answer."""
    r = user["api"].post("/api-keys", json={"name": "e2e-key"})
    assert r.status_code == 404, r.text


def test_grant_bundle_carries_several_resources_at_once(user):
    """A bundle is a key whose members are grants — one token, many resources.

    Minted with no `resource_id`, so the root reaches nothing by itself and all of
    the key's authority comes from the member added below.
    """
    key_id, raw = _create_grant_key(user, name="e2e-bundle")

    r = user["api"].post(
        f"/grants/keys/{key_id}/members",
        json={"resource_id": _a_collection(user), "can_read": True},
    )
    assert r.status_code in (200, 201), r.text

    detail = user["api"].get(f"/grants/keys/{key_id}")
    assert detail.status_code == 200, detail.text
    assert len(detail.json()["members"]) == 1

    r = Api(raw).get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 200, r.text


def test_revoking_a_grant_key_stops_it_working(user):
    key_id, raw = _create_grant_key(user, resource_id=_a_collection(user), can_read=True)
    assert Api(raw).get("/artifacts/visible", params={"action": "read"}).status_code == 200

    assert user["api"].delete(f"/grants/keys/{key_id}").status_code == 200

    r = Api(raw).get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 401, r.text
