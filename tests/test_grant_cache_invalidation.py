"""Revocation requires no re-encryption: removing a grant alone stops key issuance.

That is security invariant #4, and it is a claim about a CACHE as much as about the
ledger. `oracle.LightConeGrantVerifier` memoizes `(requester, type, action)` → authorized
contexts, and every key derivation and every cell decryption reads that memo. A revocation
that lands in the store but not in the memo is a revoked grant that keeps handing out
content keys, with the ledger already saying `revoked` and every metric reading healthy.

So these tests are about the join between the two, which is exactly what a test double
hides. They assert three separate things, because they can each fail independently:

1. **The window is real and is bounded by the TTL.** Revoking through the store, against
   the production-default verifier, does NOT deny the next request. That is pinned here
   rather than asserted away, because it is what invalidation is for and a suite that only
   exercised the invalidated path would imply a window that does not exist.
2. **Invalidation names the right principal.** The memo is keyed on the ACTING-PRINCIPAL
   id, which for a bearer key is the root grant's id and not `grantee_id` (the token hash),
   and for a bundle member is the root at the top of the chain and not the inner bundle.
   Naming the wrong id fails silently: the call is made, the entry stays.
3. **Every mutation path invalidates.** Create, claim, accept and revoke all change what a
   principal reaches; a path that skips it is stale for the length of the TTL.

The expiry case rides on the same memo: a grant that expires is a grant withdrawn by the
clock, and the store filters it at read while the memo would not.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from cryptography.fernet import Fernet

from mantle.db import lattice_api as api
from mantle.entities.artifact import Artifact
from mantle.entities.grant import Grant as GrantEntity
from mantle.search.mantle import wiring
from mantle.search.mantle.oracle import (
    FernetMasterKeyStore,
    LightConeGrantVerifier,
    OracleService,
)
from mantle.services import grant_key_service as gks

OWNER = "user-owner"
ALICE = "user-alice"
COLLECTION = "col-1"


def _now_iso(offset_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


@pytest.fixture()
def db(tmp_path):
    """A real lattice. The whole point is the join between store and memo."""
    store = api.open_database(str(tmp_path / "mantle-lattice.db"), origin="test-mantle")
    api.create_artifact(store, Artifact(id=COLLECTION, name=COLLECTION))
    return store


def _grant(store, grant_id: str, grantee: str = ALICE, **kw) -> GrantEntity:
    flags = {"can_read": True}
    flags.update(kw)
    grant = GrantEntity(
        id=grant_id, resource_id=COLLECTION, grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=grantee, granted_by=OWNER, state=GrantEntity.STATE_ACTIVE, **flags,
    )
    api.create_grant(store, grant)
    return grant


class _Clock:
    """A monotonic clock a test can advance without sleeping through a 30s TTL.

    Injected, so the verifier's own TTL arithmetic still runs for real — the alternative
    (rewriting a deadline inside `_cache`) would let the test pass against a verifier that
    never consulted the deadline at all.
    """

    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _authorized(verifier) -> bool:
    return verifier.authorized(
        requester_id=ALICE, requester_type="user",
        principal_id=COLLECTION, collection_id=COLLECTION, action="read",
    )


# ---------------------------------------------------------------------------
# 1. The window: revoking through the store is not, on its own, enough
# ---------------------------------------------------------------------------


def test_a_revoked_grant_keeps_authorizing_until_the_memo_ttl_lapses(db):
    """The production-default verifier — `ttl_s=30`, no test double anywhere.

    This is the honest shape of the hazard, and it is why `invalidate_grant_cache` exists
    rather than being an optimization someone can drop. Both halves are asserted: the
    entry survives the revocation, and the TTL is a real ceiling on how long it does.
    """
    clock = _Clock()
    grant = _grant(db, "g-alice")
    verifier = LightConeGrantVerifier(db, clock=clock)

    assert _authorized(verifier), "precondition: the grant authorizes and the memo is warm"

    grant.state = GrantEntity.STATE_REVOKED
    grant.revoked_at = _now_iso()
    api.update_grant(db, grant)
    # The ledger is unambiguous...
    assert api.get_grant_by_id(db, "g-alice").state == GrantEntity.STATE_REVOKED

    # ...and the memo is not. This is the window, stated in seconds.
    assert _authorized(verifier), (
        "expected the memoized decision to survive a revocation nobody told it about"
    )

    clock.advance(29.0)
    assert _authorized(verifier), "the TTL is 30s; 29s in, the entry is still live"

    clock.advance(1.5)
    assert not _authorized(verifier), (
        "the TTL failed to bound post-revocation key issuance — the memo outlived it"
    )


def test_invalidating_the_requester_closes_the_window_at_once(db):
    """What the mutation paths call. Same setup, one extra line, no waiting."""
    clock = _Clock()
    grant = _grant(db, "g-alice")
    verifier = LightConeGrantVerifier(db, clock=clock)
    assert _authorized(verifier)

    grant.state = GrantEntity.STATE_REVOKED
    api.update_grant(db, grant)
    verifier.invalidate(ALICE)

    assert not _authorized(verifier), "invalidation did not drop the entry it named"


def test_invalidating_a_different_principal_leaves_this_entry_alone(db):
    """Scoped, not global — otherwise every revocation costs every principal a re-walk."""
    clock = _Clock()
    grant = _grant(db, "g-alice")
    verifier = LightConeGrantVerifier(db, clock=clock)
    assert _authorized(verifier)

    grant.state = GrantEntity.STATE_REVOKED
    api.update_grant(db, grant)
    verifier.invalidate("user-someone-else")

    assert _authorized(verifier), "invalidation cleared an entry it was not asked to"


def test_a_ttl_of_zero_holds_no_memo_at_all(db):
    """The multi-worker configuration (`wiring._verifier_ttl_s`): revocation is immediate.

    `ttl_s <= 0` must mean *nothing is stored*, not *stored with a deadline in the past* —
    an entry that is written and never read is a leak, and one that is written and read by
    a clock that runs backwards is a window.
    """
    grant = _grant(db, "g-alice")
    verifier = LightConeGrantVerifier(db, ttl_s=0)

    assert _authorized(verifier)
    assert verifier._cache == {}, "a TTL of 0 stored an entry anyway"

    grant.state = GrantEntity.STATE_REVOKED
    api.update_grant(db, grant)
    assert not _authorized(verifier), "an unmemoized verifier still answered from a cache"


# ---------------------------------------------------------------------------
# 2. Expiry — revocation that arrives on a clock
# ---------------------------------------------------------------------------


def test_a_memo_entry_does_not_outlive_the_grant_it_was_derived_from(db):
    """A grant expiring in 5s may not be memoized for 30.

    The store filters expired grants at read; the memo has no such notion, so its deadline
    is clamped to the earliest expiry among the requester's grants. Without the clamp this
    entry would answer `authorized` for 25 seconds after the grant stopped existing.
    """
    clock = _Clock()
    _grant(db, "g-alice", expires_at=_now_iso(5))
    verifier = LightConeGrantVerifier(db, clock=clock)

    assert _authorized(verifier)

    (deadline, _), = verifier._cache.values()
    assert deadline <= clock.t + 5.5, (
        f"memo deadline {deadline - clock.t:.1f}s out for a grant expiring in 5s"
    )


def test_an_expired_grant_stops_authorizing_without_waiting_out_the_ttl(db):
    """End to end on the real clock: expiry, then denial, inside the TTL.

    Deliberately not clock-injected — the clamp reads wall-clock `expires_at` and the memo
    counts monotonic seconds, and this is the test that proves the two agree.
    """
    _grant(db, "g-alice", expires_at=_now_iso(0.5))
    verifier = LightConeGrantVerifier(db)          # production default: 30s

    assert _authorized(verifier), "precondition: an unexpired grant authorizes"

    time.sleep(0.8)
    assert not _authorized(verifier), (
        "an expired grant kept authorizing from the memo — 30s of the TTL to run"
    )


# ---------------------------------------------------------------------------
# 3. Every path invalidates, and names the right principal
# ---------------------------------------------------------------------------
#
# The memo key is the acting-principal id. These tests seed entries under several
# candidate ids and assert which survive, because an invalidation that clears everything
# passes a "was it dropped" assertion while destroying the memoization the cache exists
# for, and one that clears nothing passes nothing at all.


@pytest.fixture()
def memo(monkeypatch):
    """A real verifier behind the process oracle singleton, as `invalidate_grant_cache` finds it."""
    verifier = LightConeGrantVerifier(MagicMock())
    oracle = OracleService(
        FernetMasterKeyStore(Fernet(Fernet.generate_key())), grant_verifier=verifier,
    )
    monkeypatch.setattr(wiring, "_oracle_singleton", oracle)
    return verifier


def _seed(verifier, *principal_ids: str) -> None:
    """Warm one entry per candidate id, far enough out that only invalidation removes it."""
    for pid in principal_ids:
        verifier._cache[(pid, "grant_key", "read")] = (time.monotonic() + 3600, frozenset())


def _held(verifier) -> set:
    return {key[0] for key in verifier._cache}


def test_revoking_a_key_clears_the_entry_under_the_root_grants_id(db, memo):
    """Not `grantee_id`: for a grant key that field holds `sha256(raw_token)`.

    The token hash is a value the memo has never seen, so invalidating it is a call that
    does nothing at all — while the entry that IS holding a live authorization decision
    for this credential stays put.
    """
    key, raw = gks.mint(db, user_id=OWNER, name="k", resource_id=COLLECTION,
                        flags={"can_read": True})
    token_hash = gks.hash_token(raw)
    _seed(memo, key.id, token_hash)

    gks.revoke(db, key, OWNER)

    assert key.id not in _held(memo), "the root grant's id — the memo key — was not cleared"
    assert token_hash in _held(memo), "cleared the token hash, which is not a principal id"


def test_revoking_a_member_of_a_nested_bundle_clears_the_root(db, memo):
    """The acting principal is the ROOT key, however deep the member hangs.

    A member names its immediate parent in `grantee_id`. For a one-level bundle that IS the
    root; for a nested one it is an inner bundle, which no request ever acts as.
    """
    root, _ = gks.mint(db, user_id=OWNER, name="root")
    inner = gks.add_member(db, bundle_id=root.id, resource_id="col-inner", granted_by=OWNER)
    leaf = gks.add_member(db, bundle_id=inner.id, resource_id="col-leaf", granted_by=OWNER)
    _seed(memo, root.id, "user-unrelated")

    gks.revoke(db, leaf, OWNER)

    assert root.id not in _held(memo), "the inner bundle was named instead of the root"
    assert "user-unrelated" in _held(memo), "an unrelated principal lost its entry"


def test_adding_a_member_to_a_nested_bundle_clears_the_root(db, memo):
    """The widening direction, same translation: a member added deep in the chain is
    reachable by the root key immediately, and a warm memo would keep refusing it."""
    root, _ = gks.mint(db, user_id=OWNER, name="root")
    inner = gks.add_member(db, bundle_id=root.id, resource_id="col-inner", granted_by=OWNER)
    _seed(memo, root.id, "user-unrelated")

    gks.add_member(db, bundle_id=inner.id, resource_id="col-leaf", granted_by=OWNER)

    assert root.id not in _held(memo)
    assert "user-unrelated" in _held(memo)


def test_a_broken_bundle_chain_clears_everything_rather_than_guessing(db, memo):
    """Unsure which principal is affected → clear too much.

    An over-wide clear costs a light-cone re-walk. An under-wide one leaves a key issuing
    against a grant that no longer authorizes it, which is not a cost but a hole.
    """
    orphan = GrantEntity(
        id="m-orphan", resource_id="col-x", grantee_type=GrantEntity.GRANTEE_GRANT,
        grantee_id="bundle-that-does-not-exist", granted_by=OWNER,
        state=GrantEntity.STATE_ACTIVE, can_read=True,
    )
    api.create_grant(db, orphan)
    _seed(memo, "user-a", "user-b")

    gks.revoke(db, orphan, OWNER)

    assert _held(memo) == set(), "a chain that could not be walked left entries in place"


def test_a_user_grants_principal_is_its_grantee(db, memo):
    """The ordinary case still resolves to `grantee_id` — the translation narrows nothing."""
    assert gks.principal_ids_for(db, _grant(db, "g-alice")) == {ALICE}


# ---------------------------------------------------------------------------
# The router paths
# ---------------------------------------------------------------------------


def _auth(user_id: str = OWNER):
    from mantle.services.dependencies import AuthContext
    return AuthContext(principal_id=user_id, principal_type="user", user_id=user_id)


@pytest.fixture()
def admin(db):
    """OWNER can administer the collection — the gate every mutation path runs first."""
    return _grant(db, "g-owner-admin", grantee=OWNER, can_admin=True, can_share=True)


def test_delete_grants_on_a_key_clears_the_key_not_its_token_hash(db, memo, admin):
    """`DELETE /grants/{id}` matches a grant key's own id, so this path reaches a key.

    It is the reachable form of the same translation defect: the handler holds a grant whose
    `grantee_id` is a token hash and must not invalidate on it.
    """
    from mantle.routers.grants_router import revoke_grant

    key, raw = gks.mint(db, user_id=OWNER, name="k", resource_id=COLLECTION,
                        flags={"can_read": True})
    _seed(memo, key.id, gks.hash_token(raw))

    asyncio.run(revoke_grant(key.id, _auth(), db))

    assert api.get_grant_by_id(db, key.id).state == GrantEntity.STATE_REVOKED
    assert key.id not in _held(memo), "DELETE /grants/{id} left the key's memo entry warm"
    assert gks.hash_token(raw) in _held(memo)


def test_delete_grants_on_a_user_grant_clears_the_grantee(db, memo, admin):
    from mantle.routers.grants_router import revoke_grant

    _grant(db, "g-alice")
    _seed(memo, ALICE, "user-unrelated")

    asyncio.run(revoke_grant("g-alice", _auth(), db))

    assert ALICE not in _held(memo)
    assert "user-unrelated" in _held(memo)


def test_accepting_a_pending_grant_clears_the_grantees_memo(db, memo):
    """`pending_accept` → `active` is a reachability change: the store filters non-active
    grants at read, so a memo warmed while it was pending keeps refusing the grantee."""
    from mantle.routers.grants_router import accept_grant

    pending = GrantEntity(
        id="g-pending", resource_id=COLLECTION, grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=ALICE, granted_by=OWNER, can_read=True,
        state=GrantEntity.STATE_PENDING_ACCEPT,
    )
    api.create_grant(db, pending)
    _seed(memo, ALICE, "user-unrelated")

    asyncio.run(accept_grant("g-pending", _auth(ALICE), db))

    assert api.get_grant_by_id(db, "g-pending").state == GrantEntity.STATE_ACTIVE
    assert ALICE not in _held(memo), "the grantee stayed refused after accepting"
    assert "user-unrelated" in _held(memo)


def test_claiming_an_invite_clears_the_claimants_memo(db, memo, admin):
    """Claim → open the resource is one click, well inside the TTL."""
    from mantle.routers.grants_router import ClaimInviteRequest, claim_invite_endpoint
    from mantle.services import grant_service

    _, token = grant_service.create_invite(
        db, user_id=OWNER, resource_id=COLLECTION, role="viewer", max_claims=1,
    )
    _seed(memo, ALICE, "user-unrelated")

    asyncio.run(claim_invite_endpoint(ClaimInviteRequest(token=token), _auth(ALICE), db))

    assert ALICE not in _held(memo), "the claimant stayed refused after claiming"
    assert "user-unrelated" in _held(memo)


def test_creating_a_grant_clears_the_grantees_memo(db, memo, admin):
    """The widening direction, which is a hard failure rather than a delay: granted, then
    immediately refused for the length of the TTL."""
    from mantle.routers.grants_router import CreateGrantRequest, create_grant_endpoint

    _seed(memo, ALICE, "user-unrelated")

    asyncio.run(create_grant_endpoint(
        CreateGrantRequest(resource_id=COLLECTION, grantee_id=ALICE, can_read=True),
        _auth(), db,
    ))

    assert ALICE not in _held(memo)
    assert "user-unrelated" in _held(memo)


# ---------------------------------------------------------------------------
# 4. The memo is not held where invalidation cannot reach it
# ---------------------------------------------------------------------------
#
# `invalidate_grant_cache` touches `_oracle_singleton`, of which there is one per worker
# process. Under `MANTLE_WORKERS=N` a revocation served by one worker leaves N-1 memos
# untouched — the same shape `require_backplane_for_workers` refuses for the event bus,
# on the path where the cost is a live content key. `wiring._verifier_ttl_s` is where
# that is decided; these tests are the decision, written down.


@pytest.fixture(autouse=True)
def _quiet_ttl_report(monkeypatch):
    """The TTL choice is reported once per process; reset it so each test sees the branch."""
    monkeypatch.setattr(wiring, "_ttl_reported", False)


def test_a_single_worker_memoizes_for_the_default_ttl(monkeypatch):
    monkeypatch.delenv(wiring.GRANT_CACHE_TTL_SETTING, raising=False)
    monkeypatch.setenv("MANTLE_WORKERS", "1")
    assert wiring._verifier_ttl_s() == wiring.DEFAULT_GRANT_CACHE_TTL_S


def test_more_than_one_worker_turns_the_memo_off(monkeypatch, caplog):
    """Correct and slow over fast and issuing keys against revoked grants.

    The alternative — keeping a 30s memo per worker and hoping — is a silent window, and
    the operator has no way to observe it: every worker reports healthy, every revocation
    returns 200, and only the worker that served it stops issuing keys.
    """
    monkeypatch.delenv(wiring.GRANT_CACHE_TTL_SETTING, raising=False)
    monkeypatch.setenv("MANTLE_WORKERS", "4")

    with caplog.at_level("WARNING", logger="mantle.search.mantle.wiring"):
        assert wiring._verifier_ttl_s() == 0.0

    assert "MANTLE_WORKERS" in caplog.text and wiring.GRANT_CACHE_TTL_SETTING in caplog.text, (
        "the refusal must name both the setting that caused it and the way out"
    )


def test_an_operator_may_state_the_window_explicitly(monkeypatch):
    """An accepted window is a decision someone made; a default one is a decision that
    happened to them. The override is honoured over the worker count, both ways."""
    monkeypatch.setenv("MANTLE_WORKERS", "4")
    monkeypatch.setenv(wiring.GRANT_CACHE_TTL_SETTING, "5")
    assert wiring._verifier_ttl_s() == 5.0

    monkeypatch.setenv("MANTLE_WORKERS", "1")
    monkeypatch.setenv(wiring.GRANT_CACHE_TTL_SETTING, "0")
    assert wiring._verifier_ttl_s() == 0.0


def test_an_unreadable_ttl_setting_falls_to_the_safe_answer(monkeypatch):
    """A typo must not silently buy a 30s window."""
    monkeypatch.setenv("MANTLE_WORKERS", "1")
    monkeypatch.setenv(wiring.GRANT_CACHE_TTL_SETTING, "thirty")
    assert wiring._verifier_ttl_s() == 0.0
