"""Platform-admin management (multi-admin) — Mantle owns it (sovereign).

An existing admin grants another user platform admin (an admin grant on the
authority collection); that user can then perform admin ops. `GET /platform/users`
is admin-gated, so it doubles as the "am I admin?" probe. The operator is
self-healed into a real grant on the first grant-admin so it never locks itself out.
"""
from __future__ import annotations

import uuid


def _admin_probe(api) -> int:
    """200 if the caller is a platform admin, 403 otherwise."""
    return api.get("/platform/users").status_code


def test_non_admin_cannot_list_or_grant(user):
    assert user["api"].get("/platform/users").status_code == 403
    r = user["api"].post(f"/platform/users/{uuid.uuid4()}/grant-admin")
    assert r.status_code == 403


def test_operator_grants_second_admin_and_keeps_own(operator_api, user_factory):
    b = user_factory("admin2")
    # Not admin yet.
    assert _admin_probe(b["api"]) == 403

    r = operator_api.post(f"/platform/users/{b['person_id']}/grant-admin")
    assert r.status_code == 200, r.text

    # B is now a platform admin — can do admin ops (register an issuer).
    assert _admin_probe(b["api"]) == 200
    iss = b["api"].post("/issuers", json={
        "issuer": f"https://b-idp-{uuid.uuid4().hex[:6]}.test/",
        "jwks": {"keys": []}, "role": "external",
    })
    # 200/201 = accepted as admin (jwks empty is fine for the authz check); the
    # point is it's NOT 403.
    assert iss.status_code != 403, iss.text

    # The operator did NOT lock itself out (self-heal persisted its grant).
    assert _admin_probe(operator_api) == 200


def test_revoke_admin_removes_access(operator_api, user_factory):
    b = user_factory("admin3")
    assert operator_api.post(f"/platform/users/{b['person_id']}/grant-admin").status_code == 200
    assert _admin_probe(b["api"]) == 200

    r = operator_api.delete(f"/platform/users/{b['person_id']}/revoke-admin")
    assert r.status_code == 200, r.text
    assert _admin_probe(b["api"]) == 403


def test_cannot_revoke_operator(operator_api, operator):
    r = operator_api.delete(f"/platform/users/{operator['person_id']}/revoke-admin")
    assert r.status_code == 400, r.text


def test_list_users_reports_admin_status(operator_api, operator):
    # Ensure the operator is provisioned into Mantle's people (fires on visible).
    operator_api.get("/artifacts/visible", params={"action": "read"})
    r = operator_api.get("/platform/users")
    assert r.status_code == 200, r.text
    users = r.json().get("users", [])
    # The operator appears and is flagged admin.
    me = [u for u in users if u.get("id") == operator["person_id"]]
    assert me and me[0].get("is_platform_admin") is True, users
