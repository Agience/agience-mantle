"""Review 27 at Mantle.

HIGH. A top-level collection's (or any root's) own master key keys its legacy content, and its id
is a random UUID like a person's. It was counted as a person's, and nothing split it. Now:
  * a key principal's kind is Origin's replica's answer (`principal-kinds`), never an id's shape,
    and anything that is neither a person nor a collection key is non-person (failing closed when
    Origin does not answer);
  * a root's whole master key is split on the call of a person whose own credential Origin admits
    for it, and that call re-keys what is under it onto collection keys;
  * a non-person master key with nothing left under it is removed, its bytes settled out of the
    store file; one that still keys content is reported.

MEDIUM 2. A reader's split that stopped after Origin stored its share is finished by the next
reader: this node's share and the escrow share are derived from the key and Origin's share.

LOW 2. The owner re-key counts a permanent refusal as left, reads its done record only as its own
kind and origin, takes the reference counts once, and the mesh applies no inbound `custody:` row.

LOW 3. Origin is told a split key is active, and the store file settled, only once the row that
replaces the whole key has committed.
"""
from __future__ import annotations

import base64
import uuid

import pyrage
import pytest

from mantle.attenuation import ACTIONS
from mantle.db import doc_boundary as b
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.search.mantle import oracle as o
from mantle.search.mantle import threshold_custody as tc
from mantle.services import content_crypto as cc
from mantle.services import owner_rekey
from mantle.services.acting_principal import ForwardedCredential, acting_as
from mantle.system import split_master_keys as smk

from .test_a_split_master_key_needs_two_custodians import (
    ALICE_KEY, FakeEscrow, FakeOrigin, _FakeKek, make_env,
)
from .test_each_collection_has_its_own_key import Clock, FakeCas

ALICE = str(uuid.uuid4())
TEAM = str(uuid.uuid4())
COLL = "c-team-docs"


@pytest.fixture(autouse=True)
def _no_remembered_refusals():
    """Refusals the owner re-key remembers live in the process (review 28, L5): none carried from
    another test."""
    from mantle.services import owner_rekey as _or
    _or.forget_refusals()
    yield
    _or.forget_refusals()


class KindsOrigin(FakeOrigin):
    """Origin's answer to who is a person, for the counting tests alone; it decides no release.
    Every split below runs against Origin's own rule (`origin_rule.RealRuleOrigin`, review 28)."""

    def __init__(self, persons=(ALICE,)):
        super().__init__()
        self.persons = set(persons)

    def principal_kinds(self, ids):
        return {i: ("collection_key" if i.startswith("ck:") else
                    "person" if i in self.persons else "root") for i in ids}


# -- HIGH: who is a person is Origin's answer -------------------------------------------------------

@pytest.fixture()
def counted(tmp_path):
    e = make_env(tmp_path)
    o.LatticeMasterKeyStore(e.kek, lambda: e.db).put(TEAM, ALICE_KEY)
    o.LatticeMasterKeyStore(e.kek, lambda: e.db).put(ALICE, ALICE_KEY)
    e.db.artifacts.put_artifact({
        "id": str(uuid.uuid4()), "collection_id": TEAM, "origin_root": TEAM,
        "content_key_principal": TEAM, "created_by": ALICE, "content_encrypted": True,
        "content": base64.b64encode(b"MEC1" + b"\x00" * 40).decode()})
    return e


def test_a_root_s_master_key_is_non_person_by_origin_s_answer(counted):
    kinds = smk.classify([TEAM, ALICE, "ck:x"], KindsOrigin())
    assert smk.principal_kind(TEAM, kinds) == "non_person"
    assert smk.principal_kind(ALICE, kinds) == "person"
    assert smk.principal_kind("ck:x", kinds) == "collection"
    counts = smk.exposure_counts(counted.db, custodian=KindsOrigin())
    assert counts["whole_person_master_keys"] == 1
    assert counts["whole_non_person_master_keys"] == 1
    assert counts["non_person_master_keys_holding_content"] == 1
    assert counts["people_with_content_under_master_keys"] == 0


def test_without_origin_s_answer_every_non_collection_key_is_non_person(counted):
    class Down(KindsOrigin):
        def principal_kinds(self, ids):
            raise RuntimeError("Origin is unreachable")

    counts = smk.exposure_counts(counted.db, custodian=Down())
    assert counts["whole_person_master_keys"] == 0
    assert counts["whole_non_person_master_keys"] == 2


# -- HIGH: a root's key is split and its content re-keyed on a reader's call -------------------------

def _mec1(principal, master, coll, data):
    return cc.encrypt_content(principal, data, collection_id=coll,
                              master_key_provider=lambda p: master)


@pytest.fixture
def team(tmp_path, monkeypatch):
    """A top-level collection `TEAM` (its own root) holding `COLL`; Alice may read and write it.
    TEAM's master key is stored whole from before custody, and a row in COLL is under it."""
    from mantle.search.mantle import wiring

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    db.artifacts.put_artifact({"id": TEAM, "content_type": "application/vnd.agience.collection+json",
                               "origin_root": TEAM, "root_id": TEAM})
    la.add_artifact_to_collection(db, TEAM, COLL, origin=True)
    la.create_grant(db, Grant(id="g-at", resource_id=TEAM, grantee_type="user", grantee_id=ALICE,
                              granted_by=ALICE, **{f"can_{a}": True for a in ACTIONS}))
    kek = _FakeKek()
    master = bytes(range(32))
    o.LatticeMasterKeyStore(kek, lambda: db).put(TEAM, master)
    from origin.services.authority_reach import MemoryReplica

    from .origin_rule import RealRuleOrigin, replica_grant
    # Alice is a full reader-Admin of TEAM, which is whom Origin releases TEAM's key to.
    origin = RealRuleOrigin(MemoryReplica(
        grants={"g-at": replica_grant("g-at", ALICE, TEAM)},
        edges={(TEAM, COLL): {"origin": True, "propagate": None}},
        roots={TEAM: {"root_id": TEAM}}, nodes={TEAM: {}, COLL: {}},
        persons={ALICE: ("a@x", True)}), {"alice-token": ALICE})
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
    return db, store, origin, tmp_path


def _alice(**kw):
    return acting_as(ALICE, principal_type="user", source="request",
                     credential=ForwardedCredential("alice-token"), **kw)


def test_a_reader_s_call_splits_a_root_s_key_and_re_keys_what_is_under_it(team):
    db, store, origin, _ = team
    old = db.artifacts.get_artifact("t1")["content"]
    with _alice():
        assert tc.splits_on_this_call(TEAM, db)
        assert store.get_scoped(TEAM, collection_id=COLL) is not None
    row = db.artifacts.get_artifact("master-key:" + TEAM)
    assert row["content_type"] == o.SHARE_CONTENT_TYPE and TEAM in origin.shares
    assert db.artifacts.get_artifact("t1")["content"] != old
    assert not list(owner_rekey.rows_under_master_key(db, TEAM))


def test_a_persona_s_call_splits_no_root_s_key(team):
    db, store, origin, _ = team
    with _alice(scope="persona:lumen"):
        assert not tc.splits_on_this_call(TEAM, db)
    assert TEAM not in origin.shares


def test_an_emptied_non_person_key_is_removed_and_its_bytes_settled(team):
    db, store, origin, tmp_path = team
    with _alice():
        store.get_scoped(TEAM, collection_id=COLL)
    token = None
    # A second root, whole, with nothing under it: removed. TEAM's split key, emptied: removed.
    o.LatticeMasterKeyStore(_FakeKek(), lambda: db).put("r-empty", ALICE_KEY)
    origin.replica.roots["r-empty"] = {"root_id": "r-empty"}     # a root Origin knows (review 29)
    origin.replica.nodes["r-empty"] = {}
    token = db.artifacts.get_artifact("master-key:r-empty")["token"]
    done = smk.retire_empty_master_keys(db, origin)
    # Whole keys only (review 28, I1): TEAM's key is split, so it stays.
    assert done["removed"] == ["r-empty"]
    assert db.artifacts.get_artifact("master-key:" + TEAM) is not None
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("k.db") if p.is_file())
    raw += b"".join(p.read_bytes() for p in tmp_path.glob("k.db-wal") if p.is_file())
    assert token.encode() not in raw


def test_a_non_person_key_still_keying_content_is_kept_and_reported(team):
    db, _store, origin, _ = team
    done = smk.retire_empty_master_keys(db, origin)
    assert TEAM in done["holding_content"] and not done["removed"]
    assert db.artifacts.get_artifact("master-key:" + TEAM) is not None


def test_a_person_s_key_is_never_removed(tmp_path):
    e = make_env(tmp_path)
    o.LatticeMasterKeyStore(e.kek, lambda: e.db).put(ALICE, ALICE_KEY)
    assert not smk.retire_empty_master_keys(e.db, KindsOrigin())["removed"]
    assert e.db.artifacts.get_artifact("master-key:" + ALICE) is not None


# -- MEDIUM 2 -------------------------------------------------------------------------------------
# An interrupted split is redone by its storer with a fresh line while Origin's share is pending,
# and completed only once activated: `test_review28_custody` (review 28, L4).


# -- LOW 3: activation and settle after the replacing row commits --------------------------------

def test_activation_waits_for_the_replacing_row_to_commit(tmp_path, monkeypatch):
    from mantle.db import scrub

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    kek = _FakeKek()
    o.LatticeMasterKeyStore(kek, lambda: db).put("alice", ALICE_KEY)
    origin = FakeOrigin()
    store = tc.ThresholdMasterKeyStore(kek, lambda: db, origin,
                                       str(pyrage.x25519.Identity.generate().to_public()),
                                       FakeEscrow(), tc.CustodyLock(str(tmp_path / "l.lock"),
                                                                    timeout=2))
    settled = []
    monkeypatch.setattr(scrub, "settle_replaced", lambda d: settled.append(1) or True)
    monkeypatch.setattr(owner_rekey, "schedule", lambda *a, **k: False)
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        with pytest.raises(RuntimeError):
            with db.artifacts.db.write():
                store.get("alice")
                assert "alice" in origin.pending and not settled     # not yet: uncommitted
                raise RuntimeError("the caller's transaction fails")
    assert "alice" in origin.pending and not settled                 # rolled back: never told
    row = db.artifacts.get_artifact("master-key:alice")
    assert row["content_type"] == o.LatticeMasterKeyStore.CONTENT_TYPE   # still whole
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("alice-token")):
        store.get("alice")
    assert "alice" not in origin.pending and settled                 # committed: told


# -- LOW 2: the owner re-key's bookkeeping --------------------------------------------------------

def test_a_done_record_of_another_kind_or_origin_is_not_done(tmp_path):
    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    db.artifacts.put_artifact({"id": "custody:owner-rekeyed:p", "content_type": "text/plain"})
    assert not owner_rekey.is_done(db, "p")
    db.artifacts.put_artifact({"id": "custody:owner-rekeyed:q",
                               "content_type": owner_rekey.DONE_CT})      # this box's origin? no
    assert not owner_rekey.is_done(db, "q")


def test_a_permanent_refusal_is_left_not_failed(team, monkeypatch):
    from mantle.system import rekey_collections as rk

    db, _store, _origin, _ = team

    def refuse(*a, **k):
        raise tc.ShareRefused("Origin refused: the caller does not reach it")

    monkeypatch.setattr(rk, "rekey_one", refuse)
    counts = owner_rekey.rekey_owner(db, TEAM)
    assert counts["left"] == 1 and counts["failed"] == 0
    assert not owner_rekey.is_done(db, TEAM)


def test_reference_counts_are_taken_once_a_run(team, monkeypatch):
    from mantle.system import rekey_collections as rk

    db, _store, _origin, _ = team
    db.artifacts.put_artifact({**db.artifacts.get_artifact("t1"), "id": "t2", "root_id": "t2"})
    calls = []
    real = rk.key_refs
    monkeypatch.setattr(rk, "key_refs", lambda d: calls.append(1) or real(d))
    seen = []
    monkeypatch.setattr(rk, "rekey_one", lambda *a, **k: seen.append(k.get("refs")) or "already")
    owner_rekey.rekey_owner(db, TEAM)
    assert len(calls) == 1 and len(seen) == 2 and all(r is not None for r in seen)


def test_the_mesh_applies_no_inbound_custody_row():
    from mantle.mesh import sync

    applied = []

    class Store:
        class artifacts:
            @staticmethod
            def put_many(docs, **kw):
                applied.extend(d["id"] for d in docs)
                return len(docs)

    sync._apply_artifacts(Store, [{"id": "custody:owner-rekeyed:alice",
                                   "content_type": "text/plain"}])
    assert not applied
