"""Users, provisioning, and multi-issuer / multi-tenant isolation.

The tenant lever: an external IdP token is namespaced `uuid5(tenant, sub)`, so the
SAME `sub` under two different issuers resolves to two DISTINCT Mantle users. A
tenant-B user must not see a native (tenant-A) user's artifacts.
"""
from __future__ import annotations

import uuid

import pytest

from _api import Api


def test_first_visible_provisions_and_returns_array(user_factory):
    """A brand-new user's first /artifacts/visible triggers seed provisioning and
    returns a JSON array (never 401/500)."""
    u = user_factory("prov")
    r = u["api"].get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


@pytest.mark.external_issuer
def test_external_issuer_token_is_accepted(second_issuer):
    """A self-signed JWT from the registered external issuer authenticates against
    Mantle once the issuer artifact is live."""
    sub = f"ext-{uuid.uuid4().hex[:10]}"
    token = second_issuer.mint(sub, email=f"{sub}@tenant-b.test", name="Tenant B User")
    r = Api(token).get("/artifacts/visible", params={"action": "read"})
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


@pytest.mark.external_issuer
def test_cross_tenant_isolation(second_issuer, user):
    """A native (tenant-A) user creates a collection. An external (tenant-B) user
    — even one with the same `sub` string as anyone — cannot see it."""
    coll = user["api"].create_collection(f"tenantA-{uuid.uuid4().hex[:6]}")
    coll_id = coll["id"]

    ext_sub = f"ext-{uuid.uuid4().hex[:10]}"
    ext_token = second_issuer.mint(ext_sub, email=f"{ext_sub}@tenant-b.test")
    ext = Api(ext_token)

    # Confinement: the tenant-B user gets 404 (not 403) on tenant-A's collection.
    r = ext.get_artifact(coll_id)
    assert r.status_code == 404, r.text

    # And it is absent from their visible set.
    ids = {a.get("id") for a in ext.visible()}
    assert coll_id not in ids


@pytest.mark.external_issuer
def test_same_sub_two_issuers_are_distinct_users(second_issuer, user):
    """Namespacing: an external token whose `sub` equals a native user's person_id
    is still a DIFFERENT Mantle user — it does not inherit the native user's view."""
    native_coll = user["api"].create_collection(f"native-{uuid.uuid4().hex[:6]}")
    # Mint an external token reusing the native user's own id as `sub`.
    collided_sub = user["person_id"] or "reused-sub"
    ext_token = second_issuer.mint(collided_sub, email="collide@tenant-b.test")
    ext = Api(ext_token)
    ids = {a.get("id") for a in ext.visible()}
    assert native_coll["id"] not in ids, "external tenant must not inherit native user's artifacts"
