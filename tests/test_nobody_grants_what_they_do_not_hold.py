"""Minting is attenuation's blind spot, and it was unguarded on three of four paths.

Attenuation governs how authority COMPOSES down the lattice — a grant meets its edges, deny is
absorbing, the join is unrepresentable. All of that operates on authority already in the graph.
**Minting is where new authority enters**, and there the only question asked was `_require_admin`:
may this caller manage grants on this resource? That is a question about the issuer's standing, not
about the size of what they hand out.

So `can_admin` was a universal solvent — the one right that mints every other right, for yourself
or for anyone you name:

    hold  allow{can_admin} on R
    POST  /grants {resource_id: R, grantee_id: <me>, can_read: true, can_delete: true, ...}
    hold  everything on R, and on everything R's origin edges propagate to

The invite path, three branches up the same handler, refused the identical request — it has always
clamped against `effective_flags`. One ledger, two answers, and the widening one needed no invite.

`grant_service.clamp_to_issuer` is that clamp, lifted out so the four paths cannot drift again.
"""
from __future__ import annotations

import pytest

from mantle.db import lattice_api as api
from mantle.entities.grant import Grant as GrantEntity
from mantle.services import grant_key_service as gks
from mantle.services import grant_service

ISSUER = "user-issuer"
GRANTEE = "user-grantee"
RESOURCE = "col-1"

ALL_FLAGS = tuple(GrantEntity.PERMISSION_FLAGS)


@pytest.fixture()
def db(tmp_path):
    return api.open_database(str(tmp_path / "mantle-lattice.db"), origin="test-mantle")


def _grant(db, *, grantee: str, effect: str = "allow", **flags) -> GrantEntity:
    g = GrantEntity(
        id=f"g-{grantee}-{effect}-{len(flags)}-{sorted(flags)[0] if flags else 'none'}",
        resource_id=RESOURCE, grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=grantee, granted_by="seed", effect=effect,
        state=GrantEntity.STATE_ACTIVE,
        **{f: flags.get(f, False) for f in ALL_FLAGS},
    )
    api.create_grant(db, g)
    return g


def _clamp(db, requested):
    return grant_service.clamp_to_issuer(
        db, issuer_id=ISSUER, resource_id=RESOURCE, requested=requested)


# ---------------------------------------------------------------------------
# The escalation the clamp exists to stop
# ---------------------------------------------------------------------------

def test_admin_alone_does_not_mint_every_other_right(db):
    """The exploit, stated as the property it broke. An issuer whose only grant is admin may
    manage grants here — and may hand out admin. It may not invent read, delete or invoke."""
    _grant(db, grantee=ISSUER, can_admin=True)

    granted = _clamp(db, {f: True for f in ALL_FLAGS})

    assert granted["can_admin"] is True, "the issuer holds admin and may pass it on"
    for flag in ALL_FLAGS:
        if flag != "can_admin":
            assert granted[flag] is False, f"{flag} was minted from nothing"


def test_an_issuer_cannot_route_around_the_clamp_via_a_third_party(db):
    """Granting to someone else is the same act. Clamping only self-grants would leave the
    escalation intact with one extra account in it."""
    _grant(db, grantee=ISSUER, can_admin=True)

    granted = grant_service.clamp_to_issuer(
        db, issuer_id=ISSUER, resource_id=RESOURCE,
        requested={"can_read": True, "can_delete": True, "can_admin": True})

    assert granted == {"can_read": False, "can_delete": False, "can_admin": True}


def test_what_the_issuer_holds_passes_through_untouched(db):
    """The clamp is a meet, not a policy. It must not narrow a legitimate share — the ordinary
    case is a creator with all nine flags handing out a subset."""
    _grant(db, grantee=ISSUER, **{f: True for f in ALL_FLAGS})

    granted = _clamp(db, {"can_read": True, "can_update": True, "can_delete": False})

    assert granted == {"can_read": True, "can_update": True, "can_delete": False}


def test_a_deny_on_the_issuer_narrows_what_the_issuer_can_pass_on(db):
    """Deny is subtracted after the allow grants join, so a denied action cannot be laundered
    into a fresh allow grant for someone else. This is the property that makes deny worth
    writing down: without it, a deny is one `POST /grants` away from being undone."""
    _grant(db, grantee=ISSUER, can_read=True, can_update=True, can_admin=True)
    _grant(db, grantee=ISSUER, effect="deny", can_update=True)

    granted = _clamp(db, {"can_read": True, "can_update": True})

    assert granted["can_read"] is True
    assert granted["can_update"] is False, "a denied action was passed on to a third party"


def test_an_issuer_holding_nothing_grants_nothing(db):
    """No grants at all is the floor, not a bypass. `_require_admin` would refuse this caller
    upstream; the clamp does not depend on that having happened."""
    assert _clamp(db, {f: True for f in ALL_FLAGS}) == {f: False for f in ALL_FLAGS}


def test_flags_absent_from_the_request_stay_absent(db):
    """The clamp narrows; it never adds. A caller asking for two flags gets at most those two —
    notably not `can_read`, which `Grant.__init__` defaults to True."""
    _grant(db, grantee=ISSUER, **{f: True for f in ALL_FLAGS})

    granted = _clamp(db, {"can_invoke": True})

    assert granted == {"can_invoke": True}
    assert "can_read" not in granted


# ---------------------------------------------------------------------------
# Applied on every minting path, not just the one that was found
# ---------------------------------------------------------------------------

def test_a_grant_key_cannot_carry_more_than_its_issuer_holds(db):
    """A key is a bearer credential: its bits are exercised by whoever holds the token, so an
    unclamped mint is the same escalation with a longer reach."""
    _grant(db, grantee=ISSUER, can_read=True, can_admin=True)

    key, _raw = gks.mint(db, user_id=ISSUER, name="k", resource_id=RESOURCE,
                         flags={"can_read": True, "can_delete": True})

    assert key.can_read is True
    assert key.can_delete is False, "the key carries a right its issuer never held"


def test_a_bundle_member_cannot_carry_more_than_its_issuer_holds(db):
    """`_open_ceiling` deliberately gives a resource-less bundle root all nine bits, defended on
    the grounds that members are separately admin-checked. An admin check is not a ceiling, so
    the open root was narrowed by nothing on the way down."""
    _grant(db, grantee=ISSUER, can_read=True, can_admin=True)
    bundle, _raw = gks.mint(db, user_id=ISSUER, name="bundle")

    member = gks.add_member(db, bundle_id=bundle.id, resource_id=RESOURCE,
                            granted_by=ISSUER,
                            flags={"can_read": True, "can_update": True})

    assert member.can_read is True
    assert member.can_update is False, "the bundle laundered a right its issuer never held"


def test_the_role_preset_form_is_clamped_too(db):
    """`role="editor"` and an explicit flag dict resolve through the same `_flags_from`, so the
    clamp has to sit after that resolution — clamping the router's arguments would miss this."""
    _grant(db, grantee=ISSUER, can_read=True, can_admin=True)

    key, _raw = gks.mint(db, user_id=ISSUER, name="k", resource_id=RESOURCE, role="editor")

    assert key.can_read is True
    assert key.can_update is False, "a role preset minted a right the issuer never held"
