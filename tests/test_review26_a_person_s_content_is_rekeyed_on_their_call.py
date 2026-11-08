"""Review 26, open item (a) (John, 2026-11-07): a person's content is re-keyed on their own call.

Content written before collections had keys is under its principal's master key, which Origin
releases to that principal alone, so only their own credential (or their own `platform.rekey` job
grant) could move it. Now, under threshold custody, the first time Mantle holds the person's own
credential for their master key, it re-keys their content under its collection keys
(`services/owner_rekey.py`, the rekey pass's own `rekey_one` on the owner's credential):

  * in the background of the request, bounded per call, resumable and idempotent; it never
    blocks the call;
  * every collection key it mints is split (the threshold store's `put`), never stored whole;
  * the old master-keyed bytes leave the store file (`secure_delete` and a settle, as for H2);
  * `exposure_counts` reports what remains under master keys.
"""
from __future__ import annotations

import base64
import threading

import pyrage
import pytest

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.search.mantle import oracle as o
from mantle.search.mantle import threshold_custody as tc
from mantle.services.acting_principal import ForwardedCredential, acting_as

from .test_a_split_master_key_needs_two_custodians import FakeEscrow, FakeOrigin, _FakeKek
from .test_each_collection_has_its_own_key import OWNER, Clock, FakeCas, _legacy_row


@pytest.fixture(autouse=True)
def _no_remembered_refusals():
    """Refusals the owner re-key remembers live in the process (review 28, L5): none carried from
    another test."""
    from mantle.services import owner_rekey as _or
    _or.forget_refusals()
    yield
    _or.forget_refusals()


@pytest.fixture
def node(tmp_path, monkeypatch):
    """`W` holds collections `C` and `C3` and a master key stored whole from before custody, and
    two rows under it; the node runs threshold custody."""
    from mantle.search.mantle import wiring
    from mantle.services import owner_rekey

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    for coll in ("C", "C3"):
        la.add_artifact_to_collection(db, OWNER, coll, origin=True)
    la.create_grant(db, Grant(id="g-bc", resource_id="C", grantee_type="user", grantee_id="bob",
                              granted_by=OWNER, **{f"can_{a}": a == "read" for a in ACTIONS}))
    kek = _FakeKek()
    master = bytes(range(32))
    o.LatticeMasterKeyStore(kek, lambda: db).put(OWNER, master)          # whole, pre-custody
    ident = pyrage.x25519.Identity.generate()
    store = tc.ThresholdMasterKeyStore(kek, lambda: db, FakeOrigin(), str(ident.to_public()),
                                       FakeEscrow(), tc.CustodyLock(str(tmp_path / "l.lock"),
                                                                    timeout=2))
    oracle = o.OracleService(store, grant_verifier=o.LightConeGrantVerifier(db, ttl_s=0),
                             clock=Clock())
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)
    cas = FakeCas()
    _legacy_row(db, master, "a1", "C", "one", cas=cas)
    _legacy_row(db, master, "a3", "C3", "three")
    monkeypatch.setattr(owner_rekey, "_cas", lambda: cas.as_cas())
    monkeypatch.setattr(owner_rekey, "_start", lambda fn: fn())          # run it here, in order
    return db, store, cas, tmp_path


def _owner(**kw):
    return acting_as(OWNER, principal_type="user", source="request",
                     credential=ForwardedCredential("w-token"), **kw)


def _magic(db, rid):
    return base64.b64decode(db.artifacts.get_artifact(rid)["content"])[:4]


def _under_master(db):
    from mantle.system.split_master_keys import exposure_counts
    return exposure_counts(db)["content_under_master_keys"]


def test_the_owner_s_call_rekeys_their_content_and_splits_each_new_key(node):
    db, store, _cas, tmp_path = node
    old = db.artifacts.get_artifact("a1")["content"]
    assert _under_master(db) == 2
    with _owner():
        store.get(OWNER)
    assert _under_master(db) == 0
    for row in db.artifacts.list_artifacts(content_type=o.LatticeMasterKeyStore.CONTENT_TYPE):
        assert not row.get("token"), f"{row['id']} is stored whole"
    cks = [r for r in db.artifacts.list_artifacts(content_type=o.SHARE_CONTENT_TYPE)
           if r["id"].startswith("master-key:ck:")]
    assert {r["id"] for r in cks} == {"master-key:ck:C", "master-key:ck:C3"}
    raw = b"".join(p.read_bytes() for p in tmp_path.glob("k.db*") if p.is_file())
    assert old.encode() not in raw                     # the master-keyed copy left the file


def test_it_is_idempotent_and_bounded(node, monkeypatch):
    from mantle.services import owner_rekey

    db, store, _cas, _ = node
    monkeypatch.setattr(owner_rekey, "BATCH", 1)
    with _owner():
        store.get(OWNER)
    assert _under_master(db) == 1                      # one row this call
    with _owner():
        store.get(OWNER)
    assert _under_master(db) == 0                      # and the rest on the next
    seen = []
    monkeypatch.setattr(owner_rekey, "rekey_owner",
                        lambda *a, **k: seen.append(a) or {})
    with _owner():
        store.get(OWNER)
    assert not seen                                    # done: nothing runs again


@pytest.mark.parametrize("who", [
    dict(principal="bob", kw={}),
    dict(principal=OWNER, kw={"scope": "persona:lumen"}),
])
def test_no_one_else_and_no_persona_rekeys_it(node, monkeypatch, who):
    from mantle.services import owner_rekey

    db, store, _cas, _ = node
    with _owner():
        store.get(OWNER)                               # split first, on the owner's call
    calls = []
    monkeypatch.setattr(owner_rekey, "schedule", lambda *a, **k: calls.append(a))
    with acting_as(who["principal"], principal_type="user", source="request",
                   credential=ForwardedCredential("x-token"), **who["kw"]):
        try:
            store.get(OWNER)
        except tc.MasterKeyUnavailable:
            pass
    assert not calls


def test_it_never_blocks_the_call(node, monkeypatch):
    from mantle.services import owner_rekey

    db, store, _cas, _ = node
    gate, started = threading.Event(), threading.Event()

    def slow(*a, **k):
        started.set()
        gate.wait(5)
        return {}

    monkeypatch.setattr(owner_rekey, "rekey_owner", slow)
    monkeypatch.setattr(owner_rekey, "_start",
                        lambda fn: threading.Thread(target=fn, daemon=True).start())
    with _owner():
        assert store.get(OWNER) is not None            # returns while the re-key is held
    assert started.wait(5)
    gate.set()
