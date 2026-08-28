"""The origin-chain walk behind `_roots_include` happens once per requester, not once per principal.

⚑ Measured on 71/home, 2026-08-28: `_roots_include` walked `_root_of(resource)` for every granted
resource on every call. The holder there has 7,268 of 7,388 grants and the SSE narrowing path asks
the question once per principal whose index it may open — 51 of them. One recall ran **103,111 SQL
statements to return zero hits**, `_roots_include` being 11.51s of a 13.93s profile.

`principal_id` is only compared against the walk's result, never used to steer it, so the walk is
shareable. This file holds that it is actually shared — by COUNTING the walks, because
📄 `status/RETRIEVAL-PATH-2026-08-25.md` records the same fix on the sibling walker being written up
as effective while never hitting once. A memo asserted is not a memo measured.

The other four tests are the ones that matter more than the speed: a cache on an authorization path
is only admissible if it cannot outlive the grants it was derived from.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mantle.db import lattice_api as api
from mantle.entities.artifact import Artifact
from mantle.entities.grant import Grant as GrantEntity
from mantle.search.mantle.oracle import LightConeGrantVerifier

OWNER = "user-owner"
ALICE = "user-alice"
NCOLL = 6


def _now_iso(offset_s: float = 0.0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


@pytest.fixture()
def db(tmp_path):
    """A real lattice with several separately-rooted collections.

    More than one root is the point: with a single collection every principal asked about is the
    same one, and a memo that returned a constant would pass.
    """
    store = api.open_database(str(tmp_path / "mantle-lattice.db"), origin="test-mantle")
    for i in range(NCOLL):
        api.create_artifact(store, Artifact(id=f"col-{i}", name=f"col-{i}"))
    return store


def _grant(store, gid: str, resource: str, grantee: str = ALICE, **kw) -> GrantEntity:
    flags = {"can_read": True}
    flags.update(kw)
    g = GrantEntity(id=gid, resource_id=resource, grantee_type=GrantEntity.GRANTEE_USER,
                    grantee_id=grantee, granted_by=OWNER, state=GrantEntity.STATE_ACTIVE, **flags)
    api.create_grant(store, g)
    return g


class _Clock:
    def __init__(self, t: float = 1_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _counting(verifier):
    """Wrap `_root_of` so the test can count walks. Returns the counter list."""
    calls = []
    inner = verifier._root_of

    def counted(artifact_id):
        calls.append(artifact_id)
        return inner(artifact_id)

    verifier._root_of = counted
    return calls


# ---------------------------------------------------------------------------
# 1. The walk is shared across principals — the defect this file exists for
# ---------------------------------------------------------------------------

def test_the_walk_runs_once_however_many_principals_are_asked_about(db):
    for i in range(NCOLL):
        _grant(db, f"g-{i}", f"col-{i}")
    verifier = LightConeGrantVerifier(db, clock=_Clock())
    walks = _counting(verifier)

    for i in range(NCOLL):
        verifier._roots_include(ALICE, "user", f"col-{i}", "read")

    assert len(walks) == NCOLL, (
        "expected one walk per GRANT (%d), shared across every principal asked about; got %d — "
        "the walk is being repeated per principal, which is the 103,111-statement recall"
        % (NCOLL, len(walks)))


# ---------------------------------------------------------------------------
# 2-3. It still answers the same thing
# ---------------------------------------------------------------------------

def test_it_answers_true_only_for_roots_the_requester_actually_holds(db):
    _grant(db, "g-0", "col-0")
    _grant(db, "g-1", "col-1")
    verifier = LightConeGrantVerifier(db, clock=_Clock())

    assert verifier._roots_include(ALICE, "user", "col-0", "read") is True
    assert verifier._roots_include(ALICE, "user", "col-1", "read") is True
    for i in range(2, NCOLL):
        assert verifier._roots_include(ALICE, "user", f"col-{i}", "read") is False, (
            "col-%d was never granted; a memo that returned a shared set must not widen it" % i)


def test_a_denied_resource_contributes_no_root(db):
    """Deny is filtered before the walk, exactly as it was when the loop returned early.

    A deny grant's bits name what it DENIES — `effect="deny", can_read=True` — see
    `test_deny_grants_deny_everywhere.py`, whose first assertion is that this is the hazard.
    """
    _grant(db, "g-0", "col-0")
    _grant(db, "g-deny", "col-0", effect="deny", can_read=True)
    verifier = LightConeGrantVerifier(db, clock=_Clock())
    roots = verifier._granted_roots(ALICE, "user", "read")
    assert "col-0" not in roots, "a denied resource must not put its root in the set"


# ---------------------------------------------------------------------------
# 4-6. The cache may not outlive the grants it came from
# ---------------------------------------------------------------------------

def test_ttl_zero_means_no_memo_at_all(db):
    """`ttl_s <= 0` is a configuration, not a degenerate case — it must still walk every call."""
    for i in range(3):
        _grant(db, f"g-{i}", f"col-{i}")
    verifier = LightConeGrantVerifier(db, ttl_s=0)
    walks = _counting(verifier)

    verifier._roots_include(ALICE, "user", "col-0", "read")
    first = len(walks)
    verifier._roots_include(ALICE, "user", "col-0", "read")

    assert len(walks) == first * 2 and first > 0, (
        "with ttl_s=0 nothing may be stored; got %d walks then %d" % (first, len(walks) - first))


def test_invalidate_clears_the_root_set(db):
    """`invalidate` filters on `k[0] == requester_id`, so the key must carry it first."""
    _grant(db, "g-0", "col-0")
    verifier = LightConeGrantVerifier(db, clock=_Clock())
    verifier._roots_include(ALICE, "user", "col-0", "read")
    walks = _counting(verifier)

    verifier._roots_include(ALICE, "user", "col-0", "read")
    assert not walks, "precondition: warm memo does not walk"

    verifier.invalidate(ALICE)
    verifier._roots_include(ALICE, "user", "col-0", "read")
    assert walks, "invalidate(ALICE) did not reach the root set — check the cache key shape"


def test_the_memo_expires_on_the_same_ttl_the_grants_do(db):
    """The post-revocation window must be the one that already existed, not a second one."""
    _grant(db, "g-0", "col-0")
    clock = _Clock()
    verifier = LightConeGrantVerifier(db, clock=clock)
    verifier._roots_include(ALICE, "user", "col-0", "read")
    walks = _counting(verifier)

    clock.advance(1.0)
    verifier._roots_include(ALICE, "user", "col-0", "read")
    assert not walks, "inside the TTL the memo should still serve"

    clock.advance(10_000.0)
    verifier._roots_include(ALICE, "user", "col-0", "read")
    assert walks, "past the TTL the walk must run again — the memo outlived its grants"
