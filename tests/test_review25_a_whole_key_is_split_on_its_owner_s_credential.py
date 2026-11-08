"""Review 25, H1: a key no one split is split on its owner's credential at their next call.

The cutover's split pass ran as the platform system principal under a job grant, so a principal who
never granted that job kept a master key wrapped by the KEK alone, which a Mantle that is taken
over unwraps, and under custody the owner could not open it either. Now the first time the owner's
own credential asks for their key, the threshold store splits it in place: its share here, Origin's
stored and activated on that credential, the escrow share written, and the whole row gone. Anyone
else's ask, or background work, is refused as before, and the key stays exposed until its owner
next calls (counted by `exposure_counts`).
"""
from __future__ import annotations

import json

import pytest

from mantle.search.mantle import threshold_custody as tc
from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore
from mantle.services.acting_principal import ForwardedCredential, acting_as

from .test_a_split_master_key_needs_two_custodians import ALICE_KEY, make_env


@pytest.fixture(autouse=True)
def _alice_is_acting():
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        yield


@pytest.fixture()
def env(tmp_path):
    e = make_env(tmp_path)
    LatticeMasterKeyStore(e.kek, lambda: e.db).put("alice", ALICE_KEY)    # stored whole
    return e


def _whole(e):
    row = e.db.artifacts.docs["master-key:alice"]
    return row.get("content_type") == LatticeMasterKeyStore.CONTENT_TYPE and bool(row.get("token"))


def test_the_owner_s_next_call_splits_the_whole_key(env):
    assert _whole(env)
    assert env.store.get("alice") == ALICE_KEY
    row = env.db.artifacts.docs["master-key:alice"]
    assert row["content_type"] == SHARE_CONTENT_TYPE and not _whole(env)
    assert ALICE_KEY.hex() not in json.dumps(row)
    assert env.origin.shares["alice"]["x"] == tc.ORIGIN_X
    assert env.escrow.for_row(row) is not None
    assert env.store.get("alice") == ALICE_KEY                   # and reads as a split key


def test_no_one_else_splits_it(env):
    with acting_as("bob", principal_type="user", source="request",
                   credential=ForwardedCredential("bob-token")):
        with pytest.raises(tc.MasterKeyUnavailable):
            env.store.get("alice")
    assert _whole(env) and not env.origin.shares


def test_the_operator_counts_what_is_still_whole(env):
    from mantle.system.split_master_keys import exposure_counts
    assert exposure_counts(env.db)["whole_master_keys"] == 1
    env.store.get("alice")
    assert exposure_counts(env.db)["whole_master_keys"] == 0
