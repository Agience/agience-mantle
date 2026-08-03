"""`routers/platform_router._is_admin` — the platform-admin decision, previously untested.

Fourth module from the coverage survey (`NEXT.md §G.3` pattern): `platform_router.py` ranked second
on security surface with no test references. `_is_admin` gates `POST /platform/users/{id}/grant-admin`
and `DELETE …/revoke-admin` — i.e. **who may create and destroy platform administrators**.

⛔ THE SHAPE THAT NEEDED PINNING IS THE OPERATOR FAST-PATH:

    if operator_id and user_id == operator_id:
        return True

The `operator_id and` guard is not defensive noise — it is the whole safety of the line. Drop it and
`user_id == operator_id` compares two values that are BOTH commonly empty: an unresolved operator
(no operator row yet, a failed lookup returning "" or None) against an unauthenticated or malformed
`user_id`. `"" == ""` is True, so an anonymous caller becomes platform admin on a store that has not
finished provisioning. That is a fail-OPEN on the most privileged predicate in the service, reachable
precisely when the system is least set up.

The fast-path itself is deliberate and correct — someone must be able to mint the first admin, and
the operator is that someone. What must never happen is it firing when there is no operator.

Every test asserts a REFUSAL beside its matching success; a happy-path suite would pass against a
predicate that returned True unconditionally.
"""
from __future__ import annotations

import pytest

from mantle.db.lattice_api import LatticeDatabase
from mantle.entities.grant import Grant as GrantEntity
from mantle.routers.platform_router import _is_admin

AUTHORITY = "authority-collection"
OPERATOR = "user-operator"
ALICE = "user-alice"


@pytest.fixture()
def db(tmp_path):
    return LatticeDatabase(str(tmp_path / "plat.db"), origin="test-node")


def _grant(db, grantee, *, effect="allow", gid=None, resource=AUTHORITY, **flags):
    from mantle.db.backend import create_grant
    g = GrantEntity(
        id=gid or "g-%s-%s" % (grantee, "-".join(sorted(flags)) or "none"),
        resource_id=resource,
        grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=grantee,
        granted_by=OPERATOR,
        state=GrantEntity.STATE_ACTIVE,
        effect=effect,
        **flags,
    )
    return create_grant(db, g)


# ── the operator fast-path: correct, and correctly guarded ───────────────────
def test_the_operator_is_admin_without_a_grant(db):
    """Deliberate bootstrap — someone must be able to mint the first admin."""
    assert _is_admin(db, OPERATOR, OPERATOR, AUTHORITY) is True


@pytest.mark.parametrize("operator_id", ["", None])
@pytest.mark.parametrize("user_id", ["", None])
def test_an_unresolved_operator_never_makes_an_anonymous_caller_admin(db, operator_id, user_id):
    """⛔ THE FAIL-OPEN THIS GUARDS. Without `operator_id and`, `"" == ""` (and `None == None`) is
    True and an unauthenticated caller is platform admin on a store that has not finished
    provisioning — exactly when an unresolved operator is most likely."""
    assert _is_admin(db, user_id, operator_id, AUTHORITY) is not True


def test_a_non_operator_without_a_grant_is_not_admin(db):
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


# ── grant-derived admin ──────────────────────────────────────────────────────
def test_can_admin_on_the_authority_collection_confers_platform_admin(db):
    _grant(db, ALICE, can_read=True, can_admin=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is True


def test_can_update_does_not_confer_platform_admin(db):
    """⛔ NARROWED 2026-07-29 (John). The predicate accepted `can_admin OR can_update`, so WRITE
    access to the authority collection was enough to create and destroy platform administrators —
    an escalation from "may edit this container" to "may appoint admins".

    This test previously asserted the OPPOSITE, pinning the old behaviour so it could not drift
    silently. That is exactly what a policy pin is for: when the policy changes deliberately, the
    test fails and the change has to be made on purpose rather than noticed later."""
    _grant(db, ALICE, can_read=True, can_update=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


def test_read_alone_does_not_confer_platform_admin(db):
    _grant(db, ALICE, can_read=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


def test_a_deny_effect_confers_nothing(db):
    """`grant_is_allow` is positive-matching, so a deny — or any unrecognized effect — grants
    nothing even with the flag set."""
    _grant(db, ALICE, effect="deny", can_admin=True, can_read=False)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


def test_an_unrecognized_effect_confers_nothing(db):
    _grant(db, ALICE, effect="permit", can_admin=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


def test_admin_on_another_resource_does_not_confer_platform_admin(db):
    """Platform admin roots at the AUTHORITY collection specifically. Admin on some ordinary
    collection must not escalate to the platform."""
    _grant(db, ALICE, resource="col-ordinary", gid="g-other", can_read=True, can_admin=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


def test_a_revoked_admin_grant_confers_nothing(db):
    from mantle.db.backend import update_grant
    g = _grant(db, ALICE, can_read=True, can_admin=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is True
    g.state = GrantEntity.STATE_REVOKED
    update_grant(db, g)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
