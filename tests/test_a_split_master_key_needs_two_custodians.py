"""Under threshold custody a master key is 2-of-3 shares, and no one custodian holds a key.

`ThresholdMasterKeyStore` keeps share 1 in this node's lattice under the KEK, share 2 at Origin
(a fake here, holding Origin's overwrite rules), and share 3 sealed to an age recipient whose
identity this node never holds, in the durable object store (a fake here), never in the lattice.
It keeps `LatticeMasterKeyStore`'s contract: `get` answers None only for a key that exists nowhere,
and every other failure refuses rather than letting the oracle mint a replacement over a key that
exists. Every write happens under the custody lock and is conditional on what it read.
"""
from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pyrage
import pytest

from mantle.search.mantle import threshold_custody as tc
from mantle.search.mantle.oracle import (
    SHARE_CONTENT_TYPE, KeyPurpose, KeyRequest, LatticeMasterKeyStore, MasterKeyUnavailable,
    OracleService,
)
from mantle.services.acting_principal import acting_as
from prism.trust import shamir

ALICE_KEY = bytes(range(32))
OTHER_KEY = bytes(range(32, 64))

#: Made by rage (pyrage 1.4.0) for this identity, with a grease stanza a reader must skip.
REFERENCE_IDENTITY = "AGE-SECRET-KEY-1RU0D3RJDF78CE5WUDXSUDRLEAQFRA77KJNXNVU6WQNULAEW6PMZSWM6FCP"
REFERENCE_FILE = base64.b64decode(
    "YWdlLWVuY3J5cHRpb24ub3JnL3YxCi0+IFgyNTUxOSBQRzRWSUxMNTgxOE80RENtdGhQNHh0dS80dlN3amRrc2tmS2p2"
    "SlRSN1VJCjN5US9Sb1VsWGk0VHhjTVJ0ck1ya211TFlBSEpHNEZjMDBvMWJsRFZmNWcKLT4gbi5rXmt+Qy1ncmVhc2Ug"
    "Pm81LSBWRTx5XTk1ICEgX2kKV0FlWkgzSmVaUXd1OHBYeDBRbFpuQy9ROU5QVWxoWlE4SENrVDRSZzBkUGMKLS0tIGcx"
    "eDZGaDkzYjNCSlNlWGloZXcwSTlicE1wWWs3Z1BqUGNGbjkwN1F6cGMKL921MNMOmDUuUGRJrzS8qfBN0EEW69rWJI3n"
    "Sa7GidftzXQ2BuB8l6CM6pWV0VYjJTJt")


class _FakeKek:
    """Reversible, with an integrity check, so a moved token unwraps to the bytes it wrapped."""

    def wrap(self, plaintext: bytes) -> str:
        return "kek:" + plaintext.hex()

    def unwrap(self, token: str) -> bytes:
        if not token.startswith("kek:"):
            raise ValueError("not wrapped by this KEK")
        return bytes.fromhex(token[4:])


class _FakeArtifacts:
    def __init__(self):
        self.docs = {}

    def put_artifact(self, doc):
        self.docs[doc["id"]] = {**doc, "_seq": len(self.docs) + 1}

    def get_artifact(self, artifact_id):
        d = self.docs.get(artifact_id)
        return dict(d) if d else None

    def delete_artifact(self, artifact_id):
        self.docs.pop(artifact_id, None)

    def list_artifacts(self, *, content_type=None, **_kw):
        return [dict(d) for d in self.docs.values()
                if content_type is None or d.get("content_type") == content_type]


class _FakeDb:
    def __init__(self):
        self.artifacts = _FakeArtifacts()


class FakeOrigin:
    """Origin's `/internal/key-shares`, with `services.key_shares`' overwrite rules."""

    def __init__(self):
        self.shares = {}
        self.down = False
        self.puts = 0
        self.on_put = None

    def get_share(self, principal_id):
        if self.down:
            raise RuntimeError("Origin is unreachable")
        s = self.shares.get(principal_id)
        return dict(s) if s else None

    def put_share(self, principal_id, share, *, supersedes=None):
        if self.down:
            raise RuntimeError("Origin is unreachable")
        self.puts += 1
        if self.on_put:
            self.on_put(principal_id)
        old = self.shares.get(principal_id)
        if old == share:
            return "unchanged"
        if old and old["kcv"] != share["kcv"] and old["kcv"] != supersedes:
            raise RuntimeError("409 a share of another key is stored")
        self.shares[principal_id] = dict(share)
        return "replaced" if old else "stored"


class FakeEscrow:
    """The durable object store's `escrow/` prefix: objects by name, written only if absent."""

    def __init__(self):
        self.objects = {}
        self.down = False

    def put_if_absent(self, name, sealed):
        if self.down:
            raise RuntimeError("the durable store refused the write")
        if name in self.objects and self.objects[name] != sealed:
            raise RuntimeError("exists")
        self.objects.setdefault(name, sealed)

    def get(self, name):
        return self.objects.get(name)

    def for_row(self, row):
        return self.objects.get(tc.escrow_name(row))


class _SelfOnly:
    """A verifier that authorizes a principal for its own key and nothing else."""

    def authorized(self, *, requester_id, requester_type, principal_id, collection_id, action):
        return requester_id == principal_id


def make_env(tmp_path, **kw):
    ident = pyrage.x25519.Identity.generate()
    e = SimpleNamespace(db=_FakeDb(), origin=FakeOrigin(), escrow=FakeEscrow(), kek=_FakeKek(),
                        identity=str(ident), recipient=str(ident.to_public()),
                        lock=tc.CustodyLock(str(tmp_path / "lattice.db.custody.lock"),
                                            timeout=0.3))
    e.make = lambda **k: tc.ThresholdMasterKeyStore(e.kek, lambda: e.db, e.origin, e.recipient,
                                                    e.escrow, e.lock, **k)
    e.store = e.make(**kw)
    return e


@pytest.fixture()
def env(tmp_path):
    return make_env(tmp_path)


def _row(e, pid="alice"):
    return e.db.artifacts.docs["master-key:" + pid]


# -- the share flow -------------------------------------------------------------------------------

def test_a_key_round_trips_through_two_custodians(env):
    env.store.put("alice", ALICE_KEY)
    assert env.store.get("alice") == ALICE_KEY
    row = _row(env)
    assert row["content_type"] == SHARE_CONTENT_TYPE and row["state"] == "active"
    assert env.origin.shares["alice"]["x"] == tc.ORIGIN_X


def test_no_custodian_stores_the_key_or_two_shares(env):
    env.store.put("alice", ALICE_KEY)
    row = json.dumps(_row(env))
    assert ALICE_KEY.hex() not in row and base64.b64encode(ALICE_KEY).decode() not in row
    assert ALICE_KEY.hex() not in json.dumps(env.origin.shares)
    x, y = tc._unframe_share("alice", env.kek.unwrap(_row(env)["share"]))
    assert y != ALICE_KEY


def test_the_escrow_share_is_not_in_the_lattice(env):
    """The row holds share 1 and nothing else: the sealed escrow share is in the durable store,
    so losing the lattice loses one share, and Origin's with the escrow still recombine."""
    env.store.put("alice", ALICE_KEY)
    assert set(_row(env)) <= {"id", "content_type", "custody", "state", "share", "kcv",
                              "escrow_id", "_seq"}
    sealed = env.escrow.for_row(_row(env))
    name = tc.escrow_name(_row(env))
    assert sealed is not None and name.startswith("escrow/") and "alice" not in name
    assert _row(env)["kcv"] in name
    env.db.artifacts.docs.clear()                                 # the lattice is gone
    entry = json.loads(pyrage.decrypt(sealed, [pyrage.x25519.Identity.from_str(env.identity)]))
    o = env.origin.shares["alice"]
    assert shamir.combine([(entry["x"], base64.b64decode(entry["y"])),
                           (o["x"], base64.b64decode(o["y"]))]) == ALICE_KEY


def test_a_put_whose_escrow_write_fails_is_refused_and_never_active(env):
    env.escrow.down = True
    with pytest.raises(MasterKeyUnavailable, match="durable"):
        env.store.put("alice", ALICE_KEY)
    assert _row(env)["state"] == "pending" and not env.origin.shares
    assert env.store.get("alice") is None                        # never served; a retry replaces it


def test_the_row_alone_opens_nothing(env):
    env.store.put("alice", ALICE_KEY)
    env.origin.shares.clear()
    with pytest.raises(MasterKeyUnavailable, match="Origin holds no share"):
        env.store.get("alice")


def test_origin_unreachable_refuses(env):
    env.store.put("alice", ALICE_KEY)
    env.origin.down = True
    with pytest.raises(MasterKeyUnavailable):
        env.store.get("alice")


def test_a_share_of_another_key_at_origin_refuses(env):
    env.store.put("alice", ALICE_KEY)
    _row2, other_share, _e, _s = tc.split_key("alice", OTHER_KEY, env.kek, env.recipient)
    env.origin.shares["alice"] = other_share
    with pytest.raises(MasterKeyUnavailable, match="does not recombine"):
        env.store.get("alice")


def test_no_row_and_no_origin_share_is_first_use(env):
    assert env.store.get("nobody") is None


def test_no_row_while_origin_holds_a_share_refuses_to_mint(env):
    env.store.put("alice", ALICE_KEY)
    env.db.artifacts.docs.clear()
    with pytest.raises(MasterKeyUnavailable, match="refusing to mint"):
        env.store.get("alice")


def test_no_row_while_origin_is_unreachable_refuses(env):
    env.origin.down = True
    with pytest.raises(MasterKeyUnavailable):
        env.store.get("nobody")


def test_a_whole_key_row_is_refused_not_minted_over(env):
    LatticeMasterKeyStore(env.kek, lambda: env.db).put("alice", ALICE_KEY)
    with pytest.raises(MasterKeyUnavailable, match="split_master_keys"):
        env.store.get("alice")
    with pytest.raises(MasterKeyUnavailable):
        env.store.put("alice", ALICE_KEY)


def test_a_share_moved_to_another_principal_is_refused(env):
    env.store.put("alice", ALICE_KEY)
    env.db.artifacts.put_artifact(dict(_row(env), id="master-key:mallory"))
    env.origin.shares["mallory"] = dict(env.origin.shares["alice"])
    with pytest.raises(MasterKeyUnavailable, match="bound to 'alice'"):
        env.store.get("mallory")


def test_an_unwrappable_share_refuses(env):
    env.store.put("alice", ALICE_KEY)
    env.db.artifacts.docs["master-key:alice"]["share"] = "rotated:0000"
    with pytest.raises(MasterKeyUnavailable, match="could not be unwrapped"):
        env.store.get("alice")


# -- writing -------------------------------------------------------------------------------------

def test_put_refuses_to_overwrite_a_different_split_key(env):
    env.store.put("alice", ALICE_KEY)
    with pytest.raises(MasterKeyUnavailable, match="refusing to overwrite"):
        env.store.put("alice", OTHER_KEY)
    assert env.store.get("alice") == ALICE_KEY


def test_a_put_that_did_not_complete_is_replaced_by_the_next_mint(env):
    env.origin.down = True
    with pytest.raises(RuntimeError):
        env.store.put("alice", OTHER_KEY)
    assert _row(env)["state"] == "pending"
    env.origin.down = False
    assert env.store.get("alice") is None
    env.store.put("alice", ALICE_KEY)
    assert env.store.get("alice") == ALICE_KEY


def test_a_pending_put_whose_origin_share_landed_is_superseded(env):
    row, share, _e, _s = tc.split_key("alice", OTHER_KEY, env.kek, env.recipient, state="pending")
    env.db.artifacts.put_artifact(row)
    env.origin.shares["alice"] = share
    assert env.store.get("alice") is None
    env.store.put("alice", ALICE_KEY)
    assert env.store.get("alice") == ALICE_KEY


def test_a_row_changed_under_a_put_is_not_activated(env):
    """Another writer replaces the pending row while Origin is being called: activating it anyway
    would mark the other writer's row active under this put's key check."""
    def interleave(pid):
        row, _s, _e, _x = tc.split_key(pid, OTHER_KEY, env.kek, env.recipient, state="pending")
        env.db.artifacts.put_artifact(row)

    env.origin.on_put = interleave
    with pytest.raises(MasterKeyUnavailable, match="changed while"):
        env.store.put("alice", ALICE_KEY)
    assert _row(env)["state"] == "pending"


def test_a_put_waits_for_the_custody_lock_another_process_holds(env, tmp_path):
    """A second lock object on the same file stands in for `split_master_keys` in another
    process: the OS lock is per open file, so it excludes this one exactly as a process would."""
    other = tc.CustodyLock(env.lock.path)
    with other():
        with pytest.raises(MasterKeyUnavailable, match="custody lock"):
            env.store.put("alice", ALICE_KEY)
    assert "master-key:alice" not in env.db.artifacts.docs
    env.store.put("alice", ALICE_KEY)
    assert env.store.get("alice") == ALICE_KEY


def test_the_oracle_mints_once_and_reads_back_through_the_shares(env):
    oracle = OracleService(env.store, grant_verifier=_SelfOnly())
    with acting_as("alice", principal_type="user"):
        key = oracle.get_or_create_master_key(
            "alice", KeyRequest(requester_id="alice", purpose=KeyPurpose.SELF, action="create"))
        oracle.evict()
        assert oracle.get_or_create_master_key(
            "alice", KeyRequest(requester_id="alice", purpose=KeyPurpose.SELF)) == key
    assert env.origin.puts == 1


# -- the escrow ----------------------------------------------------------------------------------

def test_the_escrow_share_opens_only_with_the_offline_identity(env):
    env.store.put("alice", ALICE_KEY)
    sealed = env.escrow.for_row(_row(env))
    with pytest.raises(Exception):
        tc.open_sealed(sealed, str(pyrage.x25519.Identity.generate()))
    entry = json.loads(tc.open_sealed(sealed, env.identity))
    assert entry["principal_id"] == "alice" and entry["x"] == tc.ESCROW_X


def test_recovery_combines_with_the_escrow_only_when_selected(env):
    env.store.put("alice", ALICE_KEY)
    env.origin.shares.clear()
    with pytest.raises(MasterKeyUnavailable):
        env.store.get("alice")                                    # never automatic
    recovering = env.make(recovery_identity=env.identity)
    assert recovering.get("alice") == ALICE_KEY
    env.origin.down = True
    assert recovering.get("alice") == ALICE_KEY


def test_recovery_with_no_escrow_object_or_the_wrong_identity_refuses(env):
    env.store.put("alice", ALICE_KEY)
    env.origin.shares.clear()
    wrong = env.make(recovery_identity=str(pyrage.x25519.Identity.generate()))
    with pytest.raises(MasterKeyUnavailable, match="could not be opened"):
        wrong.get("alice")
    env.escrow.objects.clear()
    with pytest.raises(MasterKeyUnavailable, match="no escrow share"):
        env.make(recovery_identity=env.identity).get("alice")


def test_an_escrow_object_is_never_overwritten(env):
    """A re-split of the same key (after a rollback) makes a new escrow share; it goes to a new
    object, and the earlier one is left as it was."""
    env.store.put("alice", ALICE_KEY)
    first = dict(env.escrow.objects)
    env.db.artifacts.docs.clear()                            # as after `--combine` + a re-split
    env.origin.shares.clear()
    env.store.put("alice", ALICE_KEY)
    assert len(env.escrow.objects) == 2
    assert all(env.escrow.objects[k] == v for k, v in first.items())


def test_the_durable_store_writes_only_if_absent():
    class Client:
        def __init__(self):
            self.objects, self.puts = {"escrow/x.age": b"old"}, 0

        def head_object(self, Bucket, Key):
            if Key not in self.objects:
                raise type("E", (Exception,), {"response": {"Error": {"Code": "404"}}})()
            return {}

        def put_object(self, Bucket, Key, Body, **_kw):
            self.puts += 1
            self.objects[Key] = Body

    c = Client()
    store = tc.DurableEscrowStore(c, "durable")
    with pytest.raises(MasterKeyUnavailable, match="exists"):
        store.put_if_absent("escrow/x.age", b"new")
    assert c.objects["escrow/x.age"] == b"old" and c.puts == 0
    store.put_if_absent("escrow/y.age", b"new")
    assert c.objects["escrow/y.age"] == b"new"


def test_recovery_reads_the_object_named_by_the_row(env):
    env.store.put("alice", ALICE_KEY)
    env.origin.shares.clear()
    _r, _o, _e, decoy = tc.split_key("alice", OTHER_KEY, env.kek, env.recipient)
    env.escrow.objects["escrow/decoy.age"] = decoy
    assert env.make(recovery_identity=env.identity).get("alice") == ALICE_KEY


def test_calls_under_the_custody_lock_end_before_a_waiter_gives_up(monkeypatch, tmp_path):
    """A hung durable-store or Origin call must refuse, not hold the lock past the 30 s a waiter
    allows: the worst case of a put (escrow write, then Origin) fits inside it."""
    from mantle.clients import origin_key_share_client as oc

    _threshold_settings(monkeypatch, tmp_path)
    store = tc.build_threshold_store(_FakeKek(), _FakeDb)
    cfg = store._escrow._client.meta.config
    attempts = (cfg.retries or {}).get("total_max_attempts", 1)
    s3_worst = (cfg.connect_timeout + cfg.read_timeout) * attempts
    origin_worst = oc.CALL_TIMEOUT.connect + oc.CALL_TIMEOUT.read
    assert cfg.read_timeout <= 10 and oc.CALL_TIMEOUT.read <= 10
    assert s3_worst + origin_worst < tc.LOCK_TIMEOUT


# -- age, through pyrage, both directions --------------------------------------------------------

def test_a_seal_opens_with_the_reference_and_the_reference_opens_here(tmp_path):
    ident = pyrage.x25519.Identity.generate()
    sealed = tc.seal(b"share", str(ident.to_public()))
    assert pyrage.decrypt(sealed, [ident]) == b"share"
    assert tc.open_sealed(pyrage.encrypt(b"other", [ident.to_public()]), str(ident)) == b"other"
    assert tc.open_sealed(REFERENCE_FILE, REFERENCE_IDENTITY) == b"escrow share vector"
    keyfile = tmp_path / "identity.txt"
    keyfile.write_text(f"# public key: {ident.to_public()}\n{ident}\n")
    assert tc.recipient_from_file(str(keyfile)) == str(ident.to_public())
    assert tc.identity_from_file(str(keyfile)) == str(ident)
    with pytest.raises(ValueError):
        tc.check_recipient("age1notarecipient")


# -- the whole-key store refuses the share form ----------------------------------------------------

def test_the_lattice_store_refuses_a_split_key_rather_than_reading_it_as_absent(env):
    """Before this, `LatticeMasterKeyStore.get` read any non-key content type as absent, so a node
    switched back to whole-key custody without `--combine` would mint over every split key."""
    env.store.put("alice", ALICE_KEY)
    with pytest.raises(MasterKeyUnavailable, match="threshold shares"):
        LatticeMasterKeyStore(env.kek, lambda: env.db).get("alice")
    oracle = OracleService(LatticeMasterKeyStore(env.kek, lambda: env.db),
                           grant_verifier=_SelfOnly())
    with acting_as("alice", principal_type="user"), pytest.raises(MasterKeyUnavailable):
        oracle.get_or_create_master_key(
            "alice", KeyRequest(requester_id="alice", purpose=KeyPurpose.SELF, action="create"))
    assert _row(env)["content_type"] == SHARE_CONTENT_TYPE


# -- selection -----------------------------------------------------------------------------------

def _threshold_settings(monkeypatch, tmp_path, *, durable=True):
    from mantle import config
    from mantle.services import content_service

    pub = tmp_path / "escrow.pub"
    pub.write_text(f"# public key: {pyrage.x25519.Identity.generate().to_public()}\n")
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setenv("MANTLE_ESCROW_RECIPIENT_FILE", str(pub))
    monkeypatch.setenv("MANTLE_LATTICE_PATH", str(tmp_path / "lattice.db"))
    monkeypatch.delenv("MANTLE_MASTER_KEY_RECOVERY", raising=False)
    monkeypatch.delenv("MANTLE_ESCROW_IDENTITY_FILE", raising=False)
    monkeypatch.setattr(config, "ORIGIN_URI", "http://origin.test", raising=False)
    monkeypatch.setattr(content_service, "_DURABLE_BUCKET", "durable" if durable else "")
    monkeypatch.setattr(content_service, "_s3_durable", object() if durable else None)
    return pub


def test_the_default_custody_is_unchanged(monkeypatch):
    monkeypatch.delenv("MANTLE_MASTER_KEY_CUSTODY", raising=False)
    from mantle.search.mantle import wiring
    assert isinstance(wiring._master_key_store(_FakeKek(), _FakeDb), LatticeMasterKeyStore)


def test_threshold_custody_is_selected_by_setting(monkeypatch, tmp_path):
    _threshold_settings(monkeypatch, tmp_path)
    from mantle.search.mantle import wiring
    store = wiring._master_key_store(_FakeKek(), _FakeDb)
    assert isinstance(store, tc.ThresholdMasterKeyStore)
    assert store._lock.path == str(tmp_path / "lattice.db") + ".custody.lock"


def test_threshold_custody_without_a_durable_store_is_refused_at_startup(monkeypatch, tmp_path):
    _threshold_settings(monkeypatch, tmp_path, durable=False)
    from mantle.search.mantle import wiring
    with pytest.raises(ValueError, match="durable object store"):
        wiring._master_key_store(_FakeKek(), _FakeDb)


def test_threshold_custody_without_a_recipient_refuses(monkeypatch, tmp_path):
    _threshold_settings(monkeypatch, tmp_path)
    monkeypatch.delenv("MANTLE_ESCROW_RECIPIENT_FILE")
    from mantle.search.mantle import wiring
    with pytest.raises(ValueError, match="MANTLE_ESCROW_RECIPIENT_FILE"):
        wiring._master_key_store(_FakeKek(), _FakeDb)


@pytest.mark.parametrize("recovery,identity", [("escrow", ""), ("", "id.txt"), ("auto", "id.txt")])
def test_recovery_needs_both_settings(monkeypatch, tmp_path, recovery, identity):
    _threshold_settings(monkeypatch, tmp_path)
    monkeypatch.setenv("MANTLE_MASTER_KEY_RECOVERY", recovery)
    monkeypatch.setenv("MANTLE_ESCROW_IDENTITY_FILE", identity)
    with pytest.raises(ValueError):
        tc.build_threshold_store(_FakeKek(), _FakeDb)


def test_an_unknown_custody_is_refused(monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "shamir")
    with pytest.raises(ValueError):
        tc.custody_kind()
