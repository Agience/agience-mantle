"""`services.dependencies.is_platform_admin` — the platform-admin decision.

This predicate gates `POST /system/users/{id}/grant-admin` and `DELETE …/revoke-admin`
(via `require_platform_admin`, its raising form) — i.e. **who may create and destroy platform
administrators** — and simultaneously answers `is_platform_admin` in `GET /system/users`.

That it is one function is the point: if the listing (`system_router._is_admin`) and the gate
(`require_platform_admin`) were two separate implementations, they could disagree — say, the
listing accepting `can_admin` where the gate accepted `can_update`, or the listing honouring the
operator forever where the gate retired it at the end of the bootstrap window. Then the admin list
could name someone the API then refuses, or refuse to name someone it allows.

    if operator_id and user_id == operator_id:

The `operator_id and` guard is not defensive noise — it is the whole safety of the line. Drop it and
`user_id == operator_id` compares two values that are both commonly empty: an unresolved operator
(no operator row yet, a failed lookup returning "" or None) against an unauthenticated or malformed
`user_id`. `"" == ""` is True, so an anonymous caller becomes platform admin on a store that has not
finished provisioning. That is a fail-open on the most privileged predicate in the service, reachable
precisely when the system is least set up.

The fast-path itself is deliberate and correct — someone must be able to mint the first admin, and
the operator is that someone. What must never happen is it firing when there is no operator, or
after a real admin exists (see the bootstrap-window tests at the end).

Every test asserts a refusal beside its matching success; a happy-path suite would pass against a
predicate that returned True unconditionally.
"""
from __future__ import annotations

import pytest

from mantle.db.lattice_api import LatticeDatabase
from mantle.entities.grant import Grant as GrantEntity
from mantle.services.dependencies import is_platform_admin

from mantle.services.dependencies import get_id as _real_get_id

AUTHORITY = "authority-collection"
OPERATOR = "user-operator"
ALICE = "user-alice"


def _is_admin(db, user_id, operator_id, authority_id):
    """Positional shim for the canonical predicate.

    `authority_id` is passed explicitly so these tests never depend on the platform
    topology being resolved — the real callers pass it too.
    """
    return is_platform_admin(
        db, user_id, operator_id=operator_id, authority_id=authority_id)


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
    from mantle.services import dependencies

    dependencies.get_id = lambda slug: AUTHORITY               # type: ignore[assignment]
    try:
        assert _is_admin(db, OPERATOR, OPERATOR, AUTHORITY) is True
    finally:
        dependencies.get_id = _real_get_id                     # type: ignore[assignment]


@pytest.mark.parametrize("operator_id", ["", None])
@pytest.mark.parametrize("user_id", ["", None])
def test_an_unresolved_operator_never_makes_an_anonymous_caller_admin(db, operator_id, user_id):
    """Without `operator_id and`, `"" == ""` (and `None == None`) is True and an unauthenticated
    caller is platform admin on a store that has not finished provisioning — exactly when an
    unresolved operator is most likely."""
    assert _is_admin(db, user_id, operator_id, AUTHORITY) is not True


def test_a_non_operator_without_a_grant_is_not_admin(db):
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


# ── grant-derived admin ──────────────────────────────────────────────────────
def test_can_admin_on_the_authority_collection_confers_platform_admin(db):
    _grant(db, ALICE, can_read=True, can_admin=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is True


def test_can_update_does_not_confer_platform_admin(db):
    """Write access to the authority collection must not confer platform admin: the predicate
    accepts only `can_admin`, not `can_admin OR can_update`, which would let "may edit this
    container" escalate to "may appoint admins".

    This is a policy pin: when the policy changes deliberately, the test fails and the change has
    to be made on purpose rather than noticed later.

    Pins both the LISTING predicate and the gate (`require_platform_admin`) to the same
    `can_admin` requirement, so the two cannot drift apart and disagree.
    """
    _grant(db, ALICE, can_read=True, can_update=True)
    assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is False


def test_the_http_gate_agrees_with_the_predicate(db):
    """`require_platform_admin` must be exactly this predicate plus a 403.

    Asserted through the guard itself rather than by reading it, because the two drifting
    apart is the defect this file now exists to prevent.
    """
    from types import SimpleNamespace

    from fastapi import HTTPException

    from mantle.services import dependencies

    # Bind the authority collection id the guard resolves internally.
    dependencies.get_id = lambda slug: AUTHORITY               # type: ignore[assignment]
    try:
        _grant(db, ALICE, can_read=True, can_update=True)      # editor, NOT admin
        with pytest.raises(HTTPException) as exc:
            dependencies.require_platform_admin(
                SimpleNamespace(user_id=ALICE), db)
        assert exc.value.status_code == 403, (
            "can_update on the authority collection passed the HTTP gate — editing a "
            "container escalated to appointing administrators"
        )

        # POSITIVE CONTROL: a real admin grant passes, so the refusal above is the
        # policy and not a broken guard.
        _grant(db, ALICE, gid="g-alice-admin", can_read=True, can_admin=True)
        assert dependencies.require_platform_admin(
            SimpleNamespace(user_id=ALICE), db) == ALICE
    finally:
        dependencies.get_id = _real_get_id                     # type: ignore[assignment]


# ── the bootstrap window closes, and the operator retires with it ────────────
def test_the_operator_stops_being_admin_once_a_real_admin_exists(db):
    """The config fast-path is a bootstrap affordance, not standing trust.

    Once someone holds a real admin grant the window shuts and even the operator must
    act through a revocable, authority-rooted grant.
    """
    from mantle.services import dependencies

    dependencies.get_id = lambda slug: AUTHORITY               # type: ignore[assignment]
    try:
        assert _is_admin(db, OPERATOR, OPERATOR, AUTHORITY) is True   # window open
        _grant(db, ALICE, gid="g-alice-admin", can_read=True, can_admin=True)
        assert _is_admin(db, OPERATOR, OPERATOR, AUTHORITY) is False, (
            "the operator kept the config bypass after a real admin existed"
        )
        # POSITIVE CONTROL: the appointed admin does hold it.
        assert _is_admin(db, ALICE, OPERATOR, AUTHORITY) is True
    finally:
        dependencies.get_id = _real_get_id                     # type: ignore[assignment]


def test_an_editor_grant_does_not_close_the_bootstrap_window(db):
    """The window must close only on a grant that actually confers admin.

    Closing it on `can_update` would retire the operator while leaving nobody able to
    appoint anyone — unrecoverable through the API.
    """
    from mantle.services import dependencies

    dependencies.get_id = lambda slug: AUTHORITY               # type: ignore[assignment]
    try:
        _grant(db, ALICE, can_read=True, can_update=True)
        assert dependencies._authority_bootstrap_complete(db) is False
        assert _is_admin(db, OPERATOR, OPERATOR, AUTHORITY) is True
    finally:
        dependencies.get_id = _real_get_id                     # type: ignore[assignment]


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
