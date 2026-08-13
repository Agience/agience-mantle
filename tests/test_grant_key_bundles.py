"""Grant keys and bundles end-to-end against a REAL lattice.

Everything else about grant keys is covered against mocks, which proves the branching
but not that the pieces fit: that a minted token actually hashes back to its grant in
storage, that a bundle member is actually found by the ordinary grantee lookup, and
that the ceiling is actually applied on the way out. Those are exactly the joins a
mock hides, and all three are load-bearing for authorization.

The design in one line: a bundle is a grant whose members are grants granted TO it, so
composition needs no new storage and no second traversal.
"""
from __future__ import annotations

import pytest

from mantle.db import lattice_api as api
from mantle.entities.grant import Grant as GrantEntity
from mantle.services import grant_key_service as gks

OWNER = "user-owner"


@pytest.fixture()
def db(tmp_path):
    return api.open_database(str(tmp_path / "mantle-lattice.db"), origin="test-mantle")


# ── minting ──────────────────────────────────────────────────────────────────

def test_a_minted_token_authenticates_back_to_its_grant(db):
    grant, raw = gks.mint(db, user_id=OWNER, name="k", resource_id="col-1",
                          flags={"can_read": True})

    assert raw.startswith(gks.KEY_PREFIX)
    found = gks.authenticate(db, raw)
    assert found is not None and found.id == grant.id


def test_the_raw_token_is_not_stored(db):
    """Only the hash is persisted — a store dump must not yield a working credential."""
    grant, raw = gks.mint(db, user_id=OWNER, name="k", resource_id="col-1")

    stored = api.get_grant_by_id(db, grant.id)
    assert stored.grantee_id == gks.hash_token(raw)
    assert raw not in str(stored.to_dict())


def test_a_wrong_token_authenticates_to_nothing(db):
    gks.mint(db, user_id=OWNER, name="k", resource_id="col-1")

    assert gks.authenticate(db, gks.KEY_PREFIX + "not-the-token") is None
    assert gks.authenticate(db, "") is None
    # A retired API key is not a near-miss to be hashed and looked up.
    assert gks.authenticate(db, "agc_something") is None


def test_partial_flags_do_not_inherit_the_entity_default(db):
    """`{"can_invoke": True}` must mean invoke ONLY.

    `Grant.__init__` defaults `can_read=True`, so a spec built by `dict.update` would
    hand out a read the caller never asked for.
    """
    grant, _ = gks.mint(db, user_id=OWNER, name="k", resource_id="col-1",
                        flags={"can_invoke": True})

    assert grant.can_invoke is True
    assert grant.can_read is False


def test_a_role_preset_can_be_used_instead_of_bits(db):
    grant, _ = gks.mint(db, user_id=OWNER, name="k", resource_id="col-1", role="viewer")

    assert grant.can_read is True
    assert grant.can_update is False


# ── bundles ──────────────────────────────────────────────────────────────────

def test_one_key_carries_several_resources_at_different_levels(db):
    """The point of the whole design: read here, read/write there, one token."""
    bundle, raw = gks.mint(db, user_id=OWNER, name="bundle")
    gks.add_member(db, bundle_id=bundle.id, resource_id="col-ro",
                   granted_by=OWNER, flags={"can_read": True})
    gks.add_member(db, bundle_id=bundle.id, resource_id="col-rw",
                   granted_by=OWNER, flags={"can_read": True, "can_update": True})

    effective = {g.resource_id: g for g in gks.resolve(db, gks.authenticate(db, raw))}

    assert set(effective) == {"col-ro", "col-rw"}
    assert effective["col-ro"].can_update is False
    assert effective["col-rw"].can_update is True


def test_a_bundle_root_reaches_nothing_by_itself(db):
    """It has no resource_id, so an empty bundle confers nothing at all."""
    bundle, raw = gks.mint(db, user_id=OWNER, name="bundle")

    assert gks.resolve(db, gks.authenticate(db, raw)) == []


def test_narrowing_the_bundle_narrows_every_member_at_once(db):
    """The ceiling is why a bundle is worth having over N separate keys."""
    bundle, raw = gks.mint(db, user_id=OWNER, name="bundle")
    for res in ("col-a", "col-b"):
        gks.add_member(db, bundle_id=bundle.id, resource_id=res, granted_by=OWNER,
                       flags={"can_read": True, "can_update": True})

    # Positive control: both writable to begin with.
    before = gks.resolve(db, gks.authenticate(db, raw))
    assert all(g.can_update for g in before)

    bundle.can_update = False
    api.update_grant(db, bundle)

    after = gks.resolve(db, gks.authenticate(db, raw))
    assert len(after) == 2
    assert not any(g.can_update for g in after), (
        "clearing a bit on the bundle left members writable — the ceiling is not "
        "applied, so narrowing a bundle would do nothing"
    )
    assert all(g.can_read for g in after), "the ceiling removed more than it should"


def test_revoking_the_bundle_revokes_the_whole_set(db):
    bundle, raw = gks.mint(db, user_id=OWNER, name="bundle")
    gks.add_member(db, bundle_id=bundle.id, resource_id="col-a", granted_by=OWNER,
                   flags={"can_read": True})

    gks.revoke(db, bundle, OWNER)

    assert gks.authenticate(db, raw) is None, "a revoked key still authenticated"


def test_removing_one_member_leaves_the_others(db):
    bundle, raw = gks.mint(db, user_id=OWNER, name="bundle")
    a = gks.add_member(db, bundle_id=bundle.id, resource_id="col-a",
                       granted_by=OWNER, flags={"can_read": True})
    gks.add_member(db, bundle_id=bundle.id, resource_id="col-b",
                   granted_by=OWNER, flags={"can_read": True})

    gks.revoke(db, a, OWNER)

    reached = {g.resource_id for g in gks.resolve(db, gks.authenticate(db, raw))}
    assert reached == {"col-b"}


def test_bundles_nest_and_narrowing_compounds(db):
    """An inner bundle can only ever narrow further, never recover a lost bit."""
    outer, raw = gks.mint(db, user_id=OWNER, name="outer", flags={"can_read": True})
    # The inner bundle is itself a member of the outer one.
    inner = gks.add_member(db, bundle_id=outer.id, resource_id="", granted_by=OWNER,
                           flags={"can_read": True, "can_update": True})
    gks.add_member(db, bundle_id=inner.id, resource_id="col-deep", granted_by=OWNER,
                   flags={"can_read": True, "can_update": True})

    effective = {g.resource_id: g for g in gks.resolve(db, gks.authenticate(db, raw))}

    assert "col-deep" in effective, "a nested bundle's member was not reached"
    assert effective["col-deep"].can_update is False, (
        "the outer read-only ceiling did not compound through the inner bundle"
    )


def test_a_cycle_terminates_rather_than_hanging_auth(db):
    """A malformed store must not make authentication spin.

    Cycles cannot be created through the API, but authentication reading a hostile or
    corrupted store must still return.
    """
    bundle, raw = gks.mint(db, user_id=OWNER, name="bundle")
    member = gks.add_member(db, bundle_id=bundle.id, resource_id="col-a",
                            granted_by=OWNER, flags={"can_read": True})
    # Point the bundle back at its own member: bundle -> member -> bundle.
    loop = GrantEntity(
        id="loop", resource_id="col-b", grantee_type=GrantEntity.GRANTEE_GRANT,
        grantee_id=member.id, granted_by=OWNER, can_read=True,
    )
    api.create_grant(db, loop)
    back = GrantEntity(
        id="back", resource_id="col-c", grantee_type=GrantEntity.GRANTEE_GRANT,
        grantee_id="loop", granted_by=OWNER, can_read=True,
    )
    api.create_grant(db, back)

    reached = {g.resource_id for g in gks.resolve(db, gks.authenticate(db, raw))}

    assert reached == {"col-a", "col-b", "col-c"}


# ── listing ──────────────────────────────────────────────────────────────────

def test_listing_keys_shows_only_the_issuers_own(db):
    gks.mint(db, user_id=OWNER, name="mine", resource_id="col-1")
    gks.mint(db, user_id="someone-else", name="theirs", resource_id="col-2")

    mine = gks.list_keys_issued_by(db, OWNER)

    assert [k.name for k in mine] == ["mine"]


def test_a_revoked_key_is_hidden_from_listing_by_default(db):
    grant, _ = gks.mint(db, user_id=OWNER, name="k", resource_id="col-1")
    gks.revoke(db, grant, OWNER)

    assert gks.list_keys_issued_by(db, OWNER) == []
    assert len(gks.list_keys_issued_by(db, OWNER, include_revoked=True)) == 1


def test_the_key_hint_identifies_a_key_without_revealing_it(db):
    grant, raw = gks.mint(db, user_id=OWNER, name="k", resource_id="col-1")

    assert grant.key_hint == raw[-4:]
    assert len(grant.key_hint) == 4
