"""The operator CLI's revoke is a real revocation, so it invalidates like every other.

`LightConeGrantVerifier` memoizes `(requester, type, action)` for its TTL, and that memo
is read by every key derivation and every cell decryption. A revocation that edits the
ledger and stops there leaves the memo answering with the pre-revocation verdict —
grants still reachable, content keys still issued — for as long as the TTL lasts. The
ledger says `revoked` the whole time, which is what makes the gap hard to see.

Every other revocation path invalidates (`grants_router`, `grant_key_service`,
`system_router`, `workspace_service`). This one is bounded by the TTL rather than
unbounded, being a CLI, but a revocation that does not fully take effect is still a
revocation that does not fully take effect.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantle.entities.grant import Grant as GrantEntity


COLLECTION = "col-inbox-seeds"
USER = "user-1"


def _grant():
    return GrantEntity(
        id="g-1",
        resource_id=COLLECTION,
        grantee_type="user",
        grantee_id=USER,
        granted_by="admin",
        effect="allow",
        state="active",
        can_read=True,
    )


def _run_revoke(grants, *, dry_run=False):
    """Drive `action_revoke` against stubs, recording the order of the calls it makes."""
    from mantle.system import manage_seed

    calls: list = []
    grant = None

    def _update(_db, g):
        nonlocal grant
        grant = g
        calls.append(("update_grant", g.id, g.state))

    def _invalidate(_db, g):
        calls.append(("invalidate_for", g.id, g.state))

    with (
        patch.object(manage_seed, "get_id", return_value=COLLECTION),
        patch("mantle.db.backend.get_active_grants_for_principal_resource",
              return_value=grants),
        patch("mantle.db.backend.update_grant", side_effect=_update),
        patch("mantle.services.grant_key_service.invalidate_for", side_effect=_invalidate),
    ):
        manage_seed.action_revoke(MagicMock(), USER, dry_run)

    return calls, grant


def test_revoke_invalidates_the_light_cone_memo():
    calls, grant = _run_revoke([_grant()])

    assert grant.state == "revoked"
    assert ("invalidate_for", "g-1", "revoked") in calls


def test_the_memo_is_dropped_after_the_write_lands_not_before():
    """An entry dropped ahead of the commit can be refilled from the pre-change ledger
    by a concurrent request — the invalidation then applies to a state that no longer
    exists, and the stale verdict outlives it."""
    calls, _ = _run_revoke([_grant()])

    names = [c[0] for c in calls]
    assert names.index("update_grant") < names.index("invalidate_for")


def test_a_failed_invalidation_does_not_fail_the_revoke():
    """Best-effort, as on every other path: a stale memo is a delay the TTL bounds,
    while an exception here would abandon a ledger edit that already landed and report
    the revocation as having failed when it did not."""
    from mantle.system import manage_seed

    grant = _grant()
    with (
        patch.object(manage_seed, "get_id", return_value=COLLECTION),
        patch("mantle.db.backend.get_active_grants_for_principal_resource",
              return_value=[grant]),
        patch("mantle.db.backend.update_grant"),
        patch("mantle.services.grant_key_service.invalidate_for",
              side_effect=RuntimeError("oracle unreachable")),
    ):
        manage_seed.action_revoke(MagicMock(), USER, False)

    assert grant.state == "revoked"


def test_a_dry_run_neither_writes_nor_invalidates():
    calls, _ = _run_revoke([_grant()], dry_run=True)
    assert calls == []


def test_nothing_to_revoke_touches_nothing():
    calls, _ = _run_revoke([])
    assert calls == []
