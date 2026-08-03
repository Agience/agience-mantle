"""`services/grant_store.LocalBackend.check_grant` — the allow/deny primitive, previously untested.

Found the same way as `grant_service` (`NEXT.md §G.3`): a survey for modules no test references,
ranked by access-decision density. `grant_store.py` came top — 100 lines, and `check_grant` is the
evaluation that answers *"may this principal do this to this resource"* for the sovereign
authorization path. It had **no tests**.

⛔ THE INVARIANT THAT MATTERS IS DENY PRECEDENCE, AND IT IS ORDER-DEPENDENT CODE:

    if any(flag and grant_is_deny(g) for g in grants):
        allowed = False
    else:
        allowed = any(flag and grant_is_allow(g) for g in grants)

Collapse that to a single `any(... allow ...)` — a plausible "simplification", since deny grants are
rare — and a DENY silently stops working while every existing test keeps passing. The failure is
invisible until someone exercises a revocation-by-deny, and it fails OPEN.

The second invariant is **fail-closed on the unknown**: an action outside `_ACTION_FLAGS` maps to no
flag, and must therefore be refused rather than falling through to allowed. `grant_is_allow` is
deliberately not `not grant_is_deny(...)` for the same reason — an unrecognized `effect` confers
nothing. Both are positive-matching decisions that a refactor can quietly invert.

Every test asserts a REFUSAL alongside the matching success; a happy-path-only suite would pass
against a backend that allowed everything.

⚠ Stale docstring noted, not edited: the module header says grants live in "Mantle's own lattice".
the lattice is retired — `db.backend` delegates to the SQLite lattice. Wrong provenance in the header of a
security module is worth correcting, but it is someone else's attributed text; flagged in NEXT.md.
"""
from __future__ import annotations

import pytest

from mantle.db.lattice_api import LatticeDatabase
from mantle.entities.grant import Grant as GrantEntity
from mantle.services.grant_store import get_grant_backend, get_apikey_backend

RESOURCE = "col-1"
ALICE = "user-alice"
MALLORY = "user-mallory"


@pytest.fixture()
def db(tmp_path):
    return LatticeDatabase(str(tmp_path / "gs.db"), origin="test-node")


@pytest.fixture()
def backend():
    return get_grant_backend()


def _grant(db, grantee, *, effect="allow", gid=None, **flags):
    from mantle.db.backend import create_grant
    g = GrantEntity(
        id=gid or ("g-%s-%s-%s" % (grantee, effect, "-".join(sorted(flags)))),
        resource_id=RESOURCE,
        grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=grantee,
        granted_by="user-owner",
        state=GrantEntity.STATE_ACTIVE,
        effect=effect,
        **flags,
    )
    return create_grant(db, g)


# ── deny precedence ──────────────────────────────────────────────────────────
def test_a_deny_beats_an_allow_on_the_same_action(db, backend):
    """⛔ THE ORDER-DEPENDENT INVARIANT. Merge the two branches into one `any(allow)` and this is the
    only test that notices — and the defect fails OPEN."""
    _grant(db, ALICE, effect="allow", gid="g-allow", can_read=True)
    _grant(db, ALICE, effect="deny", gid="g-deny", can_read=True)
    out = backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE, action="read")
    assert out["allowed"] is False, "an explicit DENY did not beat an ALLOW"


def test_a_deny_on_a_different_action_does_not_block(db, backend):
    """Deny precedence must be per-ACTION. A blanket read of "any deny present" would make one narrow
    deny revoke everything the principal holds.

    ⚠ NOTE `can_read=False` ON THE DENY GRANT — it is not noise, see the next test."""
    _grant(db, ALICE, effect="allow", gid="g-allow", can_read=True)
    _grant(db, ALICE, effect="deny", gid="g-deny-del", can_delete=True, can_read=False)
    assert backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE,
                               action="read")["allowed"] is True
    assert backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE,
                               action="delete")["allowed"] is False


def test_a_deny_grant_denies_read_unless_can_read_is_explicitly_cleared(db, backend):
    """⚠ A LATENT TRAP, PINNED RATHER THAN SILENTLY WORKED AROUND. Found by this suite: my first
    draft of the test above wrote the obvious thing —

        Grant(effect="deny", can_delete=True)          # "deny delete"

    — and READ stopped being allowed. `can_read` is the ONLY CRUDEASIO flag defaulting to **True**
    (`entities/grant.py:105`; every other flag defaults False). That default is a sensible
    convenience for an ALLOW grant and a hazard for a DENY one: the natural expression of "deny
    delete" silently also denies read, and because deny takes precedence it REVOKES read access the
    principal legitimately holds elsewhere.

    ⚠ LATENT, NOT LIVE: no production code constructs deny grants today (verified — only tests do),
    so nothing is currently mis-denied. It becomes real for whoever writes the first one.

    This test asserts the CURRENT behaviour deliberately. Changing the default is a semantics call
    with blast radius (`upsert_user_collection_grant` also defaults `can_read=True`, correctly, for
    allow grants) — so it is FLAGGED for John, not quietly altered. If the default is ever made
    effect-aware, this test should fail and be rewritten, which is exactly the notification wanted."""
    _grant(db, ALICE, effect="allow", gid="g-allow", can_read=True)
    _grant(db, ALICE, effect="deny", gid="g-deny-default", can_delete=True)   # can_read defaults True
    assert backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE,
                               action="read")["allowed"] is False, (
        "the can_read=True default no longer leaks into deny grants — if that was deliberate, "
        "delete this test and note the semantics change")


def test_an_allow_alone_permits(db, backend):
    """The counterweight — without it, a backend that refused everything would pass the deny tests."""
    _grant(db, ALICE, effect="allow", can_read=True)
    assert backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE,
                               action="read")["allowed"] is True


# ── fail closed ──────────────────────────────────────────────────────────────
def test_no_grant_is_refused(db, backend):
    out = backend.check_grant(db, principal_id=MALLORY, resource_id=RESOURCE, action="read")
    assert out["allowed"] is False
    assert out["grants"] == []


def test_an_unknown_action_is_refused_not_ignored(db, backend):
    """An action outside `_ACTION_FLAGS` maps to no flag. It must be REFUSED — an unmapped verb that
    fell through to allowed would be a permanent hole opened by a typo."""
    _grant(db, ALICE, effect="allow", **{f: True for f in
                                         ("can_create", "can_read", "can_update", "can_delete",
                                          "can_evict", "can_invoke", "can_add", "can_share",
                                          "can_admin")})
    for action in ("publish", "", "READ", "can_read"):
        out = backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE, action=action)
        assert out["allowed"] is False, "unmapped action %r was allowed" % action


def test_an_unrecognized_effect_confers_nothing(db, backend):
    """`grant_is_allow` is positive-matching on purpose — it is NOT `not grant_is_deny(...)`, so a
    garbage `effect` is neither allow nor deny and grants nothing."""
    _grant(db, ALICE, effect="permit", can_read=True)          # not a valid effect
    assert backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE,
                               action="read")["allowed"] is False


def test_a_grant_without_the_action_flag_does_not_permit(db, backend):
    _grant(db, ALICE, effect="allow", can_read=True)
    assert backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE,
                               action="delete")["allowed"] is False


def test_a_grant_on_another_resource_does_not_leak(db, backend):
    _grant(db, ALICE, effect="allow", can_read=True)
    assert backend.check_grant(db, principal_id=ALICE, resource_id="col-OTHER",
                               action="read")["allowed"] is False


def test_the_verdict_carries_the_grants_it_was_based_on(db, backend):
    """The `grants` list is what makes a verdict auditable — a bare boolean cannot be reviewed."""
    _grant(db, ALICE, effect="allow", can_read=True)
    out = backend.check_grant(db, principal_id=ALICE, resource_id=RESOURCE, action="read")
    assert out["allowed"] is True
    assert len(out["grants"]) == 1 and out["grants"][0]["id"]


# ── api-key verification ─────────────────────────────────────────────────────
def test_an_unknown_api_key_verifies_to_nothing(db):
    assert get_apikey_backend().verify_api_key(db, "not-a-real-token") is None


def test_no_database_verifies_to_nothing_rather_than_raising(db):
    """`db is None` is a wiring fault; returning None keeps it a refusal instead of a 500 that a
    caller might treat as transient and retry into."""
    assert get_apikey_backend().verify_api_key(None, "anything") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
