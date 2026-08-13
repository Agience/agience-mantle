"""`services/grant_service.py` — the invite/sharing access decisions.

`routers/grants_router.py` gates two access decisions on it (`can_admin` at :81, `can_share` at
:93) and runs the whole invite lifecycle through it, so a gap here is a gap in every route that
shares or administers a resource.

The clamp this file pins closes an escalation:

    owner shares collection C with X as "collaborator"   (can_share=True, can_admin=False)
    X: POST /grants {resource_id: C, role: "admin"}      -> passes the can_share gate
    X claims the returned token                          -> X now holds can_admin + can_delete
    X revokes the owner's access and deletes C.

Every test here asserts that the under-privileged caller holds no such access, alongside the
matching success for the privileged one — the same reasoning as `test_oracle_grant_coupling.py`.
"""
from __future__ import annotations

import pytest

from mantle.db.lattice_api import LatticeDatabase
from mantle.entities.grant import Grant as GrantEntity
from mantle.services import grant_service as gs

RESOURCE = "col-1"
OWNER = "user-owner"
COLLAB = "user-collab"      # can_share, not can_admin — the escalation's protagonist
STRANGER = "user-stranger"  # no grant at all


@pytest.fixture()
def db(tmp_path):
    return LatticeDatabase(str(tmp_path / "grants.db"), origin="test-node")


def _grant(db, grantee, **flags):
    """Mint a direct user grant carrying exactly `flags`."""
    from mantle.db.backend import create_grant
    g = GrantEntity(
        id="grant-%s-%s" % (grantee, len(flags)),
        resource_id=RESOURCE,
        grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=grantee,
        granted_by=OWNER,
        state=GrantEntity.STATE_ACTIVE,
        **flags,
    )
    return create_grant(db, g)


# ── the access decisions the router gates on ─────────────────────────────────
def test_a_principal_with_no_grant_can_neither_share_nor_admin(db):
    assert gs.can_share(db, STRANGER, RESOURCE) is False
    assert gs.can_admin(db, STRANGER, RESOURCE) is False


def test_creating_the_resource_grants_nothing_by_itself(db):
    """`created_by` is provenance only; it does not indicate ownership. A creator holds exactly
    what their explicit grant gives them — `create_new_collection` mints one, so nothing regresses
    for a normally-created resource, but the fast-path must not exist."""
    from mantle.db.backend import create_grant  # noqa: F401  (ensures the module import path is live)
    db.artifacts.put_artifact({"id": RESOURCE, "state": "committed", "created_by": OWNER,
                               "content_type": "application/vnd.agience.collection+json"})
    assert gs.can_share(db, OWNER, RESOURCE) is False
    assert gs.can_admin(db, OWNER, RESOURCE) is False
    assert gs.effective_flags(db, OWNER, RESOURCE)["can_admin"] is False


def test_can_share_does_not_confer_can_admin(db):
    """The two are distinct authorities and the escalation depends on them staying distinct."""
    _grant(db, COLLAB, can_read=True, can_share=True)
    assert gs.can_share(db, COLLAB, RESOURCE) is True
    assert gs.can_admin(db, COLLAB, RESOURCE) is False


def test_a_revoked_grant_confers_nothing(db):
    from mantle.db.backend import update_grant
    g = _grant(db, COLLAB, can_read=True, can_share=True, can_admin=True)
    assert gs.can_admin(db, COLLAB, RESOURCE) is True
    g.state = GrantEntity.STATE_REVOKED
    update_grant(db, g)
    assert gs.can_share(db, COLLAB, RESOURCE) is False
    assert gs.can_admin(db, COLLAB, RESOURCE) is False


# ── the delegation invariant: you cannot grant what you do not hold ──────────
def test_an_invite_cannot_grant_more_than_its_issuer_holds(db):
    """The escalation, pinned: a `collaborator` (can_share, no can_admin) asking for an `admin`
    invite must get it clamped, not honoured."""
    _grant(db, COLLAB, can_read=True, can_update=True, can_share=True)   # not can_admin, not delete
    invite, _token = gs.create_invite(db, user_id=COLLAB, resource_id=RESOURCE, role="admin")

    assert invite.can_admin is False, "an invite granted can_admin its issuer never held"
    assert invite.can_delete is False, "an invite granted can_delete its issuer never held"
    assert invite.can_read is True                       # held, so preserved
    assert invite.can_update is True


def test_the_clamp_survives_the_claim(db):
    """Clamping the invite is only half of it — the grant minted on claim is what confers access."""
    _grant(db, COLLAB, can_read=True, can_share=True)
    _invite, token = gs.create_invite(db, user_id=COLLAB, resource_id=RESOURCE, role="admin")
    claimed = gs.claim_invite(db, STRANGER, token)

    assert claimed.can_admin is False
    assert claimed.can_delete is False
    assert claimed.can_read is True
    assert gs.can_admin(db, STRANGER, RESOURCE) is False, "the escalation completed end to end"


def test_an_issuer_who_holds_admin_can_still_delegate_it(db):
    """The counterweight: the clamp must not break legitimate delegation. Without this, "grants
    nothing" would pass every escalation test vacuously."""
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    invite, _t = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="admin")
    assert invite.can_admin is True
    assert invite.can_delete is True


def test_an_invite_carries_can_evict(db):
    """`can_evict` is one of the flags `create_invite` must copy — otherwise an invited admin
    could not remove members from the container they were invited to administer. Pinned because a
    missing flag is invisible until someone tries the operation."""
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    invite, _t = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="editor")
    assert invite.can_evict is True


# ── invite lifecycle ─────────────────────────────────────────────────────────
def test_the_raw_token_is_never_stored(db):
    """The grant holds a SHA-256 hash; the raw token is returned exactly once. If the raw value were
    stored, read access to the grant row would be enough to claim the invite."""
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    invite, token = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="viewer")
    assert token
    assert invite.grantee_id != token
    assert token not in str(invite.__dict__)


def test_a_single_use_invite_cannot_be_claimed_twice(db):
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    _invite, token = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE,
                                      role="viewer", max_claims=1)
    gs.claim_invite(db, STRANGER, token)
    with pytest.raises(gs.InviteClaimError):
        gs.claim_invite(db, "user-second", token)


def test_an_unknown_token_is_not_found(db):
    with pytest.raises(gs.InviteNotFound):
        gs.claim_invite(db, STRANGER, "not-a-real-token")


def test_a_targeted_invite_refuses_the_wrong_identity(db):
    """`requires_identity` invites raise for a claimant who is not the intended recipient —
    otherwise the token alone is the authority and the targeting is decoration."""
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    _invite, token = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="viewer",
                                      target_email="intended@example.com")
    with pytest.raises(gs.InviteIdentityMismatch):
        gs.claim_invite(db, STRANGER, token)          # no person record -> no email -> no match


def test_an_unknown_role_is_refused(db):
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    with pytest.raises(ValueError):
        gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="superuser")


def test_reading_a_targeted_invite_reports_the_mismatch_rather_than_crashing(db):
    """`GET /grants/invite-details` on a TARGETED invite reached `_verify_target_match` with an
    undefined free variable and raised `NameError`, so the endpoint answered 500 for exactly the
    invites whose identity check matters. The caller must get the mismatch verdict instead —
    a 500 is indistinguishable from a broken node and tells the holder nothing."""
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    _invite, token = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="viewer",
                                      target_email="intended@example.com")

    assert gs.get_invite_details(db, token, STRANGER) == {"valid": True, "identity_mismatch": True}


def test_an_untargeted_invite_still_reports_its_details(db):
    """The control: the branch that never touched the broken name keeps working, and the details
    it returns carry no token and no target identity."""
    _grant(db, OWNER, **{f: True for f in gs._ALL_FLAGS})
    _invite, token = gs.create_invite(db, user_id=OWNER, resource_id=RESOURCE, role="viewer")

    details = gs.get_invite_details(db, token, STRANGER)
    assert details["valid"] is True and details["resource_id"] == RESOURCE
    assert "target_entity" not in details and token not in str(details)


# ── deny grants confer nothing here either (audit S1's share/admin sibling) ───
def test_a_deny_grant_confers_neither_share_nor_admin(db):
    """A deny grant sets the bits naming what it DENIES, so `getattr(g, "can_share")` answered
    True for one. See `tests/test_deny_grants_deny_everywhere.py` for the whole family."""
    _grant(db, COLLAB, effect="deny", can_share=True, can_admin=True)
    assert gs.can_share(db, COLLAB, RESOURCE) is False
    assert gs.can_admin(db, COLLAB, RESOURCE) is False


def test_a_deny_grant_is_not_an_invite_ceiling(db):
    """`effective_flags` is what `create_invite` clamps against, so a deny grant read as held
    would let its holder mint the very grant it denies and claim it."""
    _grant(db, COLLAB, effect="deny", can_admin=True, can_share=True)
    assert gs.effective_flags(db, COLLAB, RESOURCE) == {f: False for f in gs._ALL_FLAGS}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
