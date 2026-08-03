"""Sharing — grants live in Mantle's own lattice (`/grants`, CRUDEASIO booleans);
Mantle resolves access by walking the grant graph upward (light-cone) and masks
denials as 404.

Covers: direct read-grant + propagation to children, read-only write→404, the
invite→claim flow, revoke→404, and non-grantee confinement (404-not-403).
"""
from __future__ import annotations

import uuid

import httpx
import pytest  # noqa: F401  (kept for the deny-effect skip marker below)


# Sovereign authorization: grants live in Mantle's own lattice and are managed via
# Mantle's `/grants` endpoints (creator/can_admin authorize the share). Origin is
# identity-only. The owner-grant created at collection-creation is in the SAME
# store the share-check reads, so an owner can share their own resource.


def _grant(owner_api, *, resource_id: str, grantee_id: str, **flags) -> httpx.Response:
    body = {"resource_id": resource_id, "grantee_type": "user", "grantee_id": grantee_id}
    body.update(flags)
    return owner_api.post("/grants", json=body)


def test_read_grant_propagates_to_children(user_factory):
    owner = user_factory("owner")
    reader = user_factory("reader")

    coll = owner["api"].create_collection(f"shared-{uuid.uuid4().hex[:6]}")
    child = owner["api"].create_child(coll["id"], name="doc", content="shared content")
    owner["api"].commit(child["id"])

    # Before the grant: reader is confined out (404, not 403).
    assert reader["api"].get_artifact(coll["id"]).status_code == 404

    r = _grant(owner["api"], resource_id=coll["id"], grantee_id=reader["person_id"], can_read=True)
    assert r.status_code in (200, 201), r.text

    # After: reader sees the collection AND (propagation) its committed child.
    assert reader["api"].get_artifact(coll["id"]).status_code == 200
    assert reader["api"].get_artifact(child["id"]).status_code == 200
    visible_ids = {a.get("id") for a in reader["api"].visible()}
    assert coll["id"] in visible_ids


def test_read_only_grantee_cannot_write(user_factory):
    owner = user_factory("owner")
    reader = user_factory("reader")
    coll = owner["api"].create_collection(f"ro-{uuid.uuid4().hex[:6]}")
    child = owner["api"].create_child(coll["id"], name="doc", content="v1")
    owner["api"].commit(child["id"])
    _grant(owner["api"], resource_id=coll["id"], grantee_id=reader["person_id"], can_read=True)

    # can_read but not can_update → PATCH is masked as 404 (confinement), never 200.
    r = reader["api"].patch(f"/artifacts/{child['id']}", json={"content": "hacked"})
    assert r.status_code == 404, r.text


def test_invite_then_claim(user_factory):
    owner = user_factory("owner")
    invitee = user_factory("invitee")
    coll = owner["api"].create_collection(f"inv-{uuid.uuid4().hex[:6]}")

    r = owner["api"].post("/grants", json={
        "resource_id": coll["id"], "grantee_type": "invite", "can_read": True,
        "name": "e2e invite",
    })
    assert r.status_code in (200, 201), r.text
    claim_token = r.json().get("claim_token")
    assert claim_token, r.json()

    # Invitee redeems the invite, then can read.
    c = invitee["api"].post("/grants/claim", json={"token": claim_token})
    assert c.status_code in (200, 201), c.text
    assert invitee["api"].get_artifact(coll["id"]).status_code == 200


def test_revoke_removes_access(user_factory):
    owner = user_factory("owner")
    reader = user_factory("reader")
    coll = owner["api"].create_collection(f"rev-{uuid.uuid4().hex[:6]}")

    g = _grant(owner["api"], resource_id=coll["id"], grantee_id=reader["person_id"], can_read=True)
    assert g.status_code in (200, 201), g.text
    grant_id = g.json().get("id")
    assert reader["api"].get_artifact(coll["id"]).status_code == 200

    d = owner["api"].delete(f"/grants/{grant_id}")
    assert d.status_code in (200, 204), d.text

    # Access is gone → confinement 404.
    assert reader["api"].get_artifact(coll["id"]).status_code == 404


def test_non_grantee_is_confined_404(user_factory):
    owner = user_factory("owner")
    stranger = user_factory("stranger")
    coll = owner["api"].create_collection(f"priv-{uuid.uuid4().hex[:6]}")
    # A stranger with no grant sees not-found, not forbidden.
    assert stranger["api"].get_artifact(coll["id"]).status_code == 404


@pytest.mark.skip(reason="deny-effect grants are not creatable via the public "
                         "/grants create API (no `effect` input field); "
                         "the deny short-circuit is exercised server-side. "
                         "404-on-absent-grant is covered by the confinement tests.")
def test_deny_effect_short_circuits():
    ...
