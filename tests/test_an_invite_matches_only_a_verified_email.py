"""An email or domain invite matches only an address an issuer vouched for.

Two ways an unverified address reached the match. `email_verified` was read as covering whatever
`_claim_email` returned — including `upn` or `preferred_username` when `email` was absent, which
the claim says nothing about. And with no verified address, `_verify_target_match` fell back to
the address on the Origin person record, which `/internal/persons/{id}` returns without saying
whether it was ever verified.
"""

from __future__ import annotations

import asyncio

import pytest

from mantle.db.lattice_api import LatticeDatabase
from mantle.entities.grant import Grant as GrantEntity
from mantle.entities.person import Person
from mantle.services import grant_service as gs
from mantle.services import person_service
from mantle.services.dependencies import AuthContext, _claim_email, _claim_email_verified

RESOURCE = "col-1"
OWNER = "user-owner"
CLAIMANT = "user-claimant"
TARGET = "intended@example.com"


# ---------------------------------------------------------------------------
# email_verified covers the `email` claim only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("claim", ["upn", "preferred_username"])
def test_email_verified_does_not_vouch_for_a_fallback_claim(claim):
    payload = {claim: TARGET, "email_verified": True}
    assert _claim_email(payload) == TARGET
    assert _claim_email_verified(payload) is False


def test_email_verified_does_not_vouch_for_a_fallback_behind_a_non_address_email():
    payload = {"email": "not-an-address", "upn": TARGET, "email_verified": True}
    assert _claim_email(payload) == TARGET
    assert _claim_email_verified(payload) is False


def test_email_verified_vouches_for_the_email_claim():
    assert _claim_email_verified({"email": TARGET, "email_verified": True}) is True
    assert _claim_email_verified({"email": TARGET, "email_verified": "false"}) is False
    assert _claim_email_verified({"email": TARGET}) is False


# ---------------------------------------------------------------------------
# No Origin fallback
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    return LatticeDatabase(str(tmp_path / "grants.db"), origin="test-node")


@pytest.fixture()
def invite_token(db):
    from mantle.db.backend import create_grant
    create_grant(db, GrantEntity(
        id="grant-owner", resource_id=RESOURCE, grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=OWNER, granted_by=OWNER, state=GrantEntity.STATE_ACTIVE,
        **{f: True for f in gs._ALL_FLAGS},
    ))
    _invite, token = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="viewer",
                                      target_email=TARGET)
    return token


@pytest.fixture()
def origin_says_target(monkeypatch):
    """Origin's person record carries the target address — unverified, as far as Mantle can tell."""
    calls = []

    def fake(db=None, id=None, **_kw):
        calls.append(id)
        return Person(id=id, email=TARGET)

    monkeypatch.setattr(person_service, "get_user_by_id", fake)
    return calls


def test_the_origin_person_address_does_not_claim_an_email_invite(db, invite_token, origin_says_target):
    with pytest.raises(gs.InviteIdentityMismatch):
        gs.claim_invite(db, CLAIMANT, invite_token)


def test_the_origin_person_address_does_not_open_the_invite_details(db, invite_token, origin_says_target):
    assert gs.get_invite_details(db, invite_token, CLAIMANT) == {"valid": True,
                                                                 "identity_mismatch": True}


def test_a_verified_token_address_claims_it(db, invite_token):
    claimed = gs.claim_invite(db, CLAIMANT, invite_token, TARGET)
    assert claimed.grantee_id == CLAIMANT


# ---------------------------------------------------------------------------
# The invite-details route passes the verified address, as /claim does
# ---------------------------------------------------------------------------

def _auth(email, verified):
    return AuthContext(user_id=CLAIMANT, principal_id=CLAIMANT, principal_type="user",
                       email=email, email_verified=verified)


def test_invite_details_reads_the_verified_token_address(db, invite_token):
    from mantle.routers.grants_router import get_invite_details_endpoint
    details = asyncio.run(get_invite_details_endpoint(invite_token, _auth(TARGET, True), db))
    assert details.get("identity_mismatch") is not True
    assert details["resource_id"] == RESOURCE


def test_invite_details_ignores_an_unverified_token_address(db, invite_token):
    from mantle.routers.grants_router import get_invite_details_endpoint
    details = asyncio.run(get_invite_details_endpoint(invite_token, _auth(TARGET, False), db))
    assert details == {"valid": True, "identity_mismatch": True}
