"""Review 28 at Mantle.

M1. A root's master key is split, and what is under it re-keyed, on the call of a full
reader-Admin of the root, which is whom Origin releases it to; anyone else's call splits
nothing. Run against Origin's own rule (`origin_rule.RealRuleOrigin`), so a stand-in cannot pass
what Origin refuses.

L2. The owner re-key finds a row by its content key principal, wherever its root is, and never
writes its done record while a row under that principal's master key is left.

L4. A pending share Origin holds is replaced by a fresh split; only an activated one is completed
(`shares_through`). The storer redoes its own interrupted split.

L5. A refused row counts toward the batch, and a refusal is remembered for a while.

L3. Mantle's server-delegation routes name only services that hold a server delegation of their
own, by prism's one list.

I1. Only a whole non-person master key is removed when it keys nothing. I3. The split pass's exit
code is the non-person gate. I4. The planner's copy of the store is overwritten before removal.
"""
from __future__ import annotations

import base64
import uuid

import pyrage
import pytest
from origin.services.authority_reach import MemoryReplica

from mantle.db import doc_boundary as b
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.attenuation import ACTIONS
from mantle.search.mantle import oracle as o
from mantle.search.mantle import threshold_custody as tc
from mantle.services import content_crypto as cc
from mantle.services import owner_rekey
from mantle.services.acting_principal import ForwardedCredential, acting_as
from mantle.system import split_master_keys as smk

from .origin_rule import RealRuleOrigin, replica_grant
from .test_a_split_master_key_needs_two_custodians import (
    ALICE_KEY, FakeEscrow, _FakeKek, make_env,
)
from .test_each_collection_has_its_own_key import Clock, FakeCas

ALICE, BOB = str(uuid.uuid4()), str(uuid.uuid4())
TEAM, COLL = str(uuid.uuid4()), "c-team-docs"


@pytest.fixture(autouse=True)
def _no_remembered_refusals():
    """Refusals the owner re-key remembers live in the process (review 28, L5): none carried from
    another test."""
    from mantle.services import owner_rekey as _or
    _or.forget_refusals()
    yield
    _or.forget_refusals()


def _mec1(principal, master, coll, data):
    return cc.encrypt_content(principal, data, collection_id=coll,
                              master_key_provider=lambda p: master)


def _as(who, token, **kw):
    return acting_as(who, principal_type="user", source="request",
                     credential=ForwardedCredential(token), **kw)


# -- M1: a root's key on a full reader-Admin's call, by Origin's own rule --------------------------

@pytest.fixture
def team(tmp_path, monkeypatch):
    """TEAM (a root) holds COLL; Alice holds every action on TEAM; Bob holds Read and Admin on
    TEAM but a deny on Read on COLL. TEAM's master key is stored whole, with a row under it."""
    from mantle.search.mantle import wiring

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    db.artifacts.put_artifact({"id": TEAM, "content_type": "application/vnd.agience.collection+json",
                               "origin_root": TEAM, "root_id": TEAM})
    la.add_artifact_to_collection(db, TEAM, COLL, origin=True)
    for gid, who in (("g-at", ALICE), ("g-bt", BOB)):
        la.create_grant(db, Grant(id=gid, resource_id=TEAM, grantee_type="user", grantee_id=who,
                                  granted_by=ALICE, **{f"can_{a}": True for a in ACTIONS}))
    replica = MemoryReplica(
        grants={"g-at": replica_grant("g-at", ALICE, TEAM),
                "g-bt": replica_grant("g-bt", BOB, TEAM, ("read", "admin")),
                "g-bd": replica_grant("g-bd", BOB, COLL, ("read",), effect="deny")},
        edges={(TEAM, COLL): {"origin": True, "propagate": None}},
        roots={TEAM: {"root_id": TEAM}}, nodes={TEAM: {}, COLL: {}},
        persons={ALICE: ("a@x", True), BOB: ("b@x", True)})
    origin = RealRuleOrigin(replica, {"alice-token": ALICE, "bob-token": BOB})
    kek, master = _FakeKek(), bytes(range(32))
    o.LatticeMasterKeyStore(kek, lambda: db).put(TEAM, master)
    store = tc.ThresholdMasterKeyStore(kek, lambda: db, origin,
                                       str(pyrage.x25519.Identity.generate().to_public()),
                                       FakeEscrow(), tc.CustodyLock(str(tmp_path / "l.lock"),
                                                                    timeout=2))
    oracle = o.OracleService(store, grant_verifier=o.LightConeGrantVerifier(db, ttl_s=0),
                             clock=Clock())
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)
    cas = FakeCas()
    db.artifacts.put_artifact({
        "id": "t1", "collection_id": COLL, "origin_root": TEAM, b.CONTENT_KEY_PRINCIPAL: TEAM,
        "created_by": ALICE, "state": "live", "content_encrypted": True,
        "content": base64.b64encode(_mec1(TEAM, master, COLL, b"team note")).decode()})
    monkeypatch.setattr(owner_rekey, "_cas", lambda: cas.as_cas())
    monkeypatch.setattr(owner_rekey, "_start", lambda fn: fn())
    return db, store, origin


def test_a_full_reader_admin_s_call_splits_the_root_s_key_and_re_keys_under_it(team):
    db, store, origin = team
    with _as(ALICE, "alice-token"):
        assert store.get_scoped(TEAM, collection_id=COLL) is not None
    assert db.artifacts.get_artifact("master-key:" + TEAM)["content_type"] == o.SHARE_CONTENT_TYPE
    assert not list(owner_rekey.rows_under_master_key(db, TEAM))


def test_an_admin_origin_refuses_the_key_splits_nothing(team):
    db, store, origin = team
    with _as(BOB, "bob-token"):
        with pytest.raises(tc.ShareRefused):
            store.get_scoped(TEAM, collection_id=COLL)
    assert db.artifacts.get_artifact("master-key:" + TEAM)["content_type"] == \
        o.LatticeMasterKeyStore.CONTENT_TYPE
    assert not origin.shares and list(owner_rekey.rows_under_master_key(db, TEAM))


# -- L2: rows by their key principal, wherever their root is --------------------------------------

def _bob_row(db, rid="b1"):
    master = bytes(range(32))
    db.artifacts.put_artifact({
        "id": rid, "collection_id": COLL, "origin_root": TEAM, b.CONTENT_KEY_PRINCIPAL: BOB,
        "created_by": BOB, "state": "live", "content_encrypted": True,
        "content": base64.b64encode(_mec1(BOB, master, COLL, b"bob's note")).decode()})


def test_a_row_under_bob_s_key_in_another_root_is_found(tmp_path):
    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    _bob_row(db)
    assert [d["id"] for d, _w in owner_rekey.rows_under_master_key(db, BOB)] == ["b1"]


def test_no_done_record_while_a_row_is_left(tmp_path, monkeypatch):
    from mantle.system import rekey_collections as rk

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    _bob_row(db)
    monkeypatch.setattr(rk, "rekey_one", lambda *a, **k: "unscoped")    # left as it was
    owner_rekey.rekey_owner(db, BOB, cas=FakeCas().as_cas())
    assert not owner_rekey.is_done(db, BOB)


# -- L5: refusals count toward the batch, and are remembered ---------------------------------------

def test_refusals_count_toward_the_batch_and_are_remembered(tmp_path, monkeypatch):
    from mantle.system import rekey_collections as rk

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    for rid in ("b1", "b2", "b3"):
        _bob_row(db, rid)
    tried = []

    def refuse(_db, doc, *_a, **_k):
        tried.append(doc["id"])
        raise tc.ShareRefused("this reader may not open it")

    monkeypatch.setattr(rk, "rekey_one", refuse)
    monkeypatch.setattr(owner_rekey, "BATCH", 2)
    owner_rekey.forget_refusals()
    with _as(ALICE, "alice-token"):
        assert owner_rekey.rekey_owner(db, BOB, cas=FakeCas().as_cas())["left"] == 2
        assert len(tried) == 2                              # the batch bounds refusals too
        owner_rekey.rekey_owner(db, BOB, cas=FakeCas().as_cas())
        assert len(tried) == 3 and len(set(tried)) == 3     # the two refused are not walked again


# -- L4: a pending share is replaced; an activated one is completed --------------------------------

def _ck_env(tmp_path, origin_cls):
    e = make_env(tmp_path)
    kid = o.collection_key_id("c-team")
    o.LatticeMasterKeyStore(e.kek, lambda: e.db).put(kid, ALICE_KEY)
    replica = MemoryReplica(
        grants={"g-a": replica_grant("g-a", ALICE, "c-team", ("read",)),
                "g-c": replica_grant("g-c", "carol", "c-team", ("read",))},
        roots={"c-team": {"root_id": "c-team"}}, nodes={"c-team": {}},
        persons={ALICE: ("a@x", True)})
    e.store._custodian = e.origin = origin_cls(replica, {"alice-token": ALICE,
                                                         "carol-token": "carol"})
    return e, kid


def test_a_pending_share_is_replaced_by_a_fresh_split_on_its_storer_s_call(tmp_path):
    e, kid = _ck_env(tmp_path, RealRuleOrigin)
    e.escrow.down = True
    with _as(ALICE, "alice-token"):
        with pytest.raises(tc.MasterKeyUnavailable):
            e.store.get_scoped(kid, collection_id="c-team")         # stopped after Origin stored
    planted = dict(e.origin.shares[kid])
    e.escrow.down = False
    with _as("carol", "carol-token"):                               # another reader: refused
        with pytest.raises(tc.ShareRefused):
            e.store.get_scoped(kid, collection_id="c-team")
    with _as(ALICE, "alice-token"):                                 # the storer redoes it
        assert e.store.get_scoped(kid, collection_id="c-team") == ALICE_KEY
    assert e.origin.shares[kid]["y"] != planted["y"]                # a fresh line, not the found one
    assert e.db.artifacts.docs["master-key:" + kid]["content_type"] == o.SHARE_CONTENT_TYPE


def test_an_activated_share_is_completed(tmp_path):
    """A key written back whole by `--combine` keeps Origin's activated share: the next split
    completes that share's line rather than drawing one Origin would refuse to store."""
    e, kid = _ck_env(tmp_path, RealRuleOrigin)
    from prism.trust import shamir
    sh = dict(shamir.split(ALICE_KEY, k=2, n=3))
    e.origin.shares[kid] = {"x": 2, "y": base64.b64encode(sh[2]).decode(),
                            "kcv": tc.key_check_value(kid, ALICE_KEY)}      # activated, held
    with _as("carol", "carol-token"):
        assert e.store.get_scoped(kid, collection_id="c-team") == ALICE_KEY
    assert e.origin.shares[kid]["y"] == base64.b64encode(sh[2]).decode()


# -- L3 ---------------------------------------------------------------------------------------------

def test_mantle_s_server_delegation_routes_name_only_services_that_hold_one():
    from prism.trust.service_identity import SERVER_DELEGATION_SIGNERS

    from mantle.services.dependencies import NARROWED_SERVER_ROUTES
    assert set(NARROWED_SERVER_ROUTES) <= set(SERVER_DELEGATION_SIGNERS)
    assert "crystal" in SERVER_DELEGATION_SIGNERS


# -- I1, I3, I4 --------------------------------------------------------------------------------------

def test_only_a_whole_non_person_key_is_removed(tmp_path):
    e = make_env(tmp_path)
    o.LatticeMasterKeyStore(e.kek, lambda: e.db).put("r-whole", ALICE_KEY)
    e.db.artifacts.put_artifact({"id": "master-key:r-split", "content_type": o.SHARE_CONTENT_TYPE,
                                 "state": "active", "share": "x", "kcv": "k"})
    done = smk.retire_empty_master_keys(e.db, kinds={"r-whole": "root", "r-split": "root"})
    assert done["removed"] == ["r-whole"]
    assert e.db.artifacts.get_artifact("master-key:r-split") is not None


@pytest.mark.parametrize("left, counts, code", [
    ({"whole_non_person_master_keys": 1}, {"failed": 0, "changed": 0}, 1),
    ({"whole_non_person_master_keys": 0}, {"failed": 0, "changed": 0}, 0),
    ({"whole_non_person_master_keys": 0, "whole_person_master_keys": 5,
      "whole_collection_keys": 2}, {"failed": 0, "changed": 0}, 0),
    ({"whole_non_person_master_keys": 0}, {"failed": 1, "changed": 0}, 0),
])
def test_the_split_pass_s_exit_code_is_the_non_person_gate(left, counts, code):
    assert smk.gate_exit(left, counts) == code


def test_the_planner_s_copy_is_overwritten_before_it_is_removed(tmp_path):
    from mantle.system import seed_plan as sp

    copy = tmp_path / "plan-copy.db"
    copy.write_bytes(b"a store full of secrets" * 100)
    (tmp_path / "plan-copy.db-wal").write_bytes(b"its log" * 50)
    seen = {}
    sp._scrub_copy(str(tmp_path), observe=lambda p, data: seen.__setitem__(p, data))
    assert seen and all(set(data) <= {0} for data in seen.values())
    assert not copy.exists()
