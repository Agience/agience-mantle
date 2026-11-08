"""Review 26: what a call splits, and what a split leaves behind.

H1. A collection key stored whole (`ck:<collection>`) was split on no one's call: the owner-call
split compared the acting principal with the key's principal, which no person ever acts as. Now it
is split on the call of any person whose own credential Origin admits for the collection (Origin's
release rule, then its first-share rule, decide; Mantle only refuses a credential that is not a
person's own).

L1. The split reads Origin's stored share back and recombines it before the whole row is replaced.

L2. Only a person's own access token or an unscoped person delegation splits a key, never a
persona's or Ember's narrowed credential; no escrow object is written before Origin has admitted
the call; and a refusal is a `ShareRefused`, not a raw `AuthorityRefused`.

H2. The whole row the split replaces leaves no bytes in the store file (`secure_delete` on every
store connection, and a truncating checkpoint after a key-row replace).
"""
from __future__ import annotations

import json
import os

import pyrage
import pytest
from cryptography.fernet import Fernet

from mantle.clients.origin_key_share_client import AuthorityRefused
from mantle.search.mantle import threshold_custody as tc
from mantle.search.mantle.key_provider import LocalKeyProvider
from mantle.search.mantle.oracle import (
    SHARE_CONTENT_TYPE, LatticeMasterKeyStore, _unframe_master_key, collection_key_id,
)
from mantle.services.acting_principal import ForwardedCredential, acting_as

from .test_a_split_master_key_needs_two_custodians import (
    ALICE_KEY, FakeEscrow, FakeOrigin, make_env,
)

KID = collection_key_id("c-team")


def _whole(e, pid):
    row = e.db.artifacts.docs["master-key:" + pid]
    return row.get("content_type") == LatticeMasterKeyStore.CONTENT_TYPE and bool(row.get("token"))


@pytest.fixture()
def env(tmp_path):
    e = make_env(tmp_path)
    LatticeMasterKeyStore(e.kek, lambda: e.db).put(KID, ALICE_KEY)        # stored whole
    return e


def _as_alice(**kw):
    return acting_as("alice", principal_type="user", source="request",
                     credential=ForwardedCredential("alice-token"), **kw)


# -- H1: a collection key is split on a reader's call ----------------------------------------------

def test_a_whole_collection_key_is_split_on_a_reader_s_call(env):
    with _as_alice():
        assert env.store.get_scoped(KID, collection_id="c-team") == ALICE_KEY
    row = env.db.artifacts.docs["master-key:" + KID]
    assert row["content_type"] == SHARE_CONTENT_TYPE and not _whole(env, KID)
    assert env.origin.shares[KID]["x"] == tc.ORIGIN_X
    assert env.escrow.for_row(row) is not None
    from mantle.system.split_master_keys import exposure_counts
    assert exposure_counts(env.db)["whole_collection_keys"] == 0


@pytest.mark.parametrize("pid", ["alice", KID])
def test_a_caller_origin_does_not_admit_splits_nothing(tmp_path, pid):
    e = make_env(tmp_path)
    LatticeMasterKeyStore(e.kek, lambda: e.db).put(pid, ALICE_KEY)

    class Refusing(FakeOrigin):
        def get_share(self, principal_id, **kw):
            raise AuthorityRefused(403, "not released: the subject does not reach it")

        def put_share(self, principal_id, share, **kw):
            raise AuthorityRefused(403, "not stored: a first share is stored by its creator")

    e.store._custodian = e.origin = Refusing()
    with _as_alice():
        with pytest.raises(tc.ShareRefused):
            e.store.get_scoped(pid, collection_id="c-team")
    assert _whole(e, pid) and not e.escrow.objects            # no escrow before admission


@pytest.mark.parametrize("pid", ["alice", KID])
def test_a_store_refusal_after_the_release_check_is_a_share_refusal(tmp_path, pid):
    e = make_env(tmp_path)
    LatticeMasterKeyStore(e.kek, lambda: e.db).put(pid, ALICE_KEY)

    class StoreRefuses(FakeOrigin):
        def put_share(self, principal_id, share, **kw):
            raise AuthorityRefused(403, "not stored")

    e.store._custodian = e.origin = StoreRefuses()
    with _as_alice():
        with pytest.raises(tc.ShareRefused):
            e.store.get_scoped(pid, collection_id="c-team")
    assert _whole(e, pid) and not e.escrow.objects


# -- L2: only a person's own credential splits --------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"scope": "persona:lumen"},                 # a persona's server delegation
    {"narrowed": "ember", "scope": "ember:memory"},   # Ember's narrowed delegation
])
def test_a_persona_or_narrowed_credential_splits_nothing(env, kw):
    with _as_alice(**kw):
        with pytest.raises(tc.MasterKeyUnavailable):
            env.store.get_scoped(KID, collection_id="c-team")
    assert _whole(env, KID) and not env.origin.shares and not env.escrow.objects


def test_a_persona_credential_does_not_split_its_person_s_own_key(tmp_path):
    e = make_env(tmp_path)
    LatticeMasterKeyStore(e.kek, lambda: e.db).put("alice", ALICE_KEY)
    with _as_alice(scope="persona:lumen"):
        with pytest.raises(tc.MasterKeyUnavailable):
            e.store.get("alice")
    assert _whole(e, "alice") and not e.origin.shares


# -- L1: Origin's share is read back before the row is replaced ---------------------------------

@pytest.mark.parametrize("pid", ["alice", KID])
def test_the_split_reads_origin_s_share_back_before_replacing_the_row(tmp_path, pid):
    class Forgets(FakeOrigin):
        """Stores the share, then answers with another: the split must not trust its own put."""
        def get_share(self, principal_id, **kw):
            s = super().get_share(principal_id, **kw)
            if s is None:
                return None
            return {**s, "y": "AAAA" + s["y"][4:]}

    e = make_env(tmp_path)
    LatticeMasterKeyStore(e.kek, lambda: e.db).put(pid, ALICE_KEY)
    e.store._custodian = e.origin = Forgets()
    with _as_alice():
        with pytest.raises(tc.MasterKeyUnavailable):
            e.store.get_scoped(pid, collection_id="c-team")
    assert _whole(e, pid)


# -- H2: the replaced whole row leaves no bytes in the store file -------------------------------

def _file_bytes(tmp_path):
    out = b""
    for name in os.listdir(tmp_path):
        if name in ("live.db", "live.db-wal"):        # the -shm index holds no row bytes
            out += (tmp_path / name).read_bytes()
    return out


def test_the_owner_s_split_leaves_no_whole_key_in_the_store_file(tmp_path, monkeypatch):
    from mantle.services import owner_rekey

    from .test_there_is_no_promotion_state import real_node

    monkeypatch.setattr(owner_rekey, "_start", lambda fn: fn())     # no thread outliving the test
    db = real_node(tmp_path, monkeypatch)
    kek = LocalKeyProvider(Fernet(Fernet.generate_key()))
    LatticeMasterKeyStore(kek, lambda: db).put("alice", ALICE_KEY)
    whole_token = db.artifacts.get_artifact("master-key:alice")["token"]
    # Other rows, and a checkpoint, so the whole row sits in the main file beside live cells.
    for i in range(30):
        db.artifacts.put_artifact({"id": f"pad-{i}", "content_type": "x/pad", "body": "p" * 200})
    from mantle.db.schema import wal_checkpoint
    wal_checkpoint(db.artifacts.db.conn(), "TRUNCATE")
    assert whole_token.encode() in _file_bytes(tmp_path)

    ident = pyrage.x25519.Identity.generate()
    store = tc.ThresholdMasterKeyStore(kek, lambda: db, FakeOrigin(), str(ident.to_public()),
                                       FakeEscrow(),
                                       tc.CustodyLock(str(tmp_path / "l.lock"), timeout=0.3))
    with _as_alice():
        assert store.get("alice") == ALICE_KEY
    row = db.artifacts.get_artifact("master-key:alice")
    assert row["content_type"] == SHARE_CONTENT_TYPE and not row.get("token")
    residue = whole_token.encode() in _file_bytes(tmp_path)
    if residue:                                  # what a taken-over Mantle would do with it
        assert _unframe_master_key("alice", kek.unwrap(whole_token)) == ALICE_KEY
    assert not residue, "the KEK-wrapped whole key survives in the store file"


def test_every_store_connection_has_secure_delete_on(tmp_path, monkeypatch):
    from .test_there_is_no_promotion_state import real_node

    db = real_node(tmp_path, monkeypatch)
    assert db.artifacts.db.conn().execute("PRAGMA secure_delete").fetchone()[0] == 1


# -- L4: the operator's counts, by kind --------------------------------------------------------

def test_exposure_counts_by_kind(tmp_path, monkeypatch):
    from mantle.services import peer_signing
    from mantle.system.split_master_keys import exposure_counts

    monkeypatch.setattr(peer_signing, "get_system_principal_id",
                        lambda: "5d6ee3d1-7d5b-5f53-9c1c-0d1f6f0b6a01")
    e = make_env(tmp_path)
    store = LatticeMasterKeyStore(e.kek, lambda: e.db)
    store.put("8f14e45f-ceea-467a-9b36-7d0d5c2a9e11", ALICE_KEY)     # a person (uuid4)
    store.put("5d6ee3d1-7d5b-5f53-9c1c-0d1f6f0b6a01", ALICE_KEY)     # the system principal
    store.put("lumen", ALICE_KEY)                                    # a persona's principal
    store.put(KID, ALICE_KEY)                                        # a collection key
    e.db.artifacts.put_artifact({"id": "a1", "content_type": "text/plain",
                                 "content_key_scope": "master"})
    class Origin:                               # its replica says who is a person (review 27)
        @staticmethod
        def principal_kinds(ids):
            return {i: "person" if i == "8f14e45f-ceea-467a-9b36-7d0d5c2a9e11" else "platform"
                    for i in ids}

    counts = exposure_counts(e.db, custodian=Origin())
    assert counts["whole_person_master_keys"] == 1
    assert counts["whole_non_person_master_keys"] == 2
    assert counts["whole_collection_keys"] == 1
    assert "content_under_master_keys" in counts
    assert json.dumps(counts)                                          # counts only


# -- I2: key rows never cross the mesh ---------------------------------------------------------

def test_key_rows_are_not_replicated_and_a_peer_s_is_never_applied():
    from mantle.mesh import sync

    assert not sync._is_replicated(LatticeMasterKeyStore.CONTENT_TYPE)
    assert not sync._is_replicated(SHARE_CONTENT_TYPE)

    applied = []

    class Store:
        class artifacts:
            @staticmethod
            def put_many(docs, **kw):
                applied.extend(d["id"] for d in docs)
                return len(docs)

    peer = [{"id": "master-key:alice", "content_type": "text/plain", "_rev": "1"},
            {"id": "master-key:ck:c-team", "content_type": SHARE_CONTENT_TYPE}]
    sync._apply_artifacts(Store, peer)
    assert not [i for i in applied if i.startswith("master-key:")]
