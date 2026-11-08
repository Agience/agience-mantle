"""Review 12, M6 and M7.

M6: a cached key is never served without the requester's own verdict, on every path, the
double-check under the oracle's lock included.
M7: a verdict is remembered per credential, not per person: a narrowed delegation or a new token
does not reuse a verdict another credential earned. Forgetting a person's verdicts (their sign-in
bound moved) also forgets the key-release verdicts and the key cache.
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle.oracle import FernetMasterKeyStore, KeyPurpose, KeyRequest, OracleService
from mantle.services.acting_principal import ForwardedCredential, acting_as


class _Refusing(FernetMasterKeyStore):
    """A store whose requester verdict refuses everyone: as Origin would for a revoked reader."""

    def __init__(self):
        super().__init__(Fernet(Fernet.generate_key()))
        self.asked = []

    def verdict(self, principal_id, *, collection_id=None, artifact_id=None):
        self.asked.append(principal_id)
        raise PermissionError("no verdict")


class _Everyone:
    def authorized(self, *a, **k):
        return True


def _oracle(store):
    return OracleService(store, grant_verifier=_Everyone())


def test_the_master_key_double_check_still_asks_the_requester_s_verdict(monkeypatch):
    store = _Refusing()
    o = _oracle(store)
    o._cache["alice"] = (o._clock() + 30, b"k" * 32)
    seen = iter([None])                  # the fast path misses; the locked re-read hits
    real = o._cached
    monkeypatch.setattr(o, "_cached", lambda pid: next(seen, real(pid)))
    monkeypatch.setattr(o, "_authorize", lambda *a, **k: None)
    with pytest.raises(PermissionError):
        o.get_or_create_master_key("alice", KeyRequest(requester_id="bob", purpose=KeyPurpose.GRANT, action="read"))
    assert store.asked == ["alice"]


def test_the_collection_key_double_check_still_asks_the_requester_s_verdict(monkeypatch):
    from mantle.search.mantle.oracle import collection_key_id

    store = _Refusing()
    o = _oracle(store)
    kid = collection_key_id("C1")
    o._cache[kid] = (o._clock() + 30, b"k" * 32)
    seen = iter([None])
    real = o._cached
    monkeypatch.setattr(o, "_cached", lambda pid: next(seen, real(pid)))
    monkeypatch.setattr(o, "_authorize", lambda *a, **k: None)
    with pytest.raises(PermissionError):
        o.get_or_create_collection_key("alice", "C1",
                                       KeyRequest(requester_id="bob", purpose=KeyPurpose.GRANT, action="read"))
    assert store.asked == [kid]


def test_a_verdict_is_remembered_per_credential():
    from mantle.search.mantle.threshold_custody import ThresholdMasterKeyStore

    store = ThresholdMasterKeyStore.__new__(ThresholdMasterKeyStore)
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("token-one")):
        one = store._verdict_key("ck:C1", {"collection_id": "C1"})
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("token-two")):
        two = store._verdict_key("ck:C1", {"collection_id": "C1"})
    assert one != two


def test_forgetting_a_person_also_forgets_key_verdicts_and_the_key_cache(monkeypatch):
    from mantle.search.mantle import wiring
    from mantle.search.mantle.threshold_custody import ThresholdMasterKeyStore
    from mantle.services import authority_ledger

    store = ThresholdMasterKeyStore.__new__(ThresholdMasterKeyStore)
    with acting_as("alice", principal_type="user", source="request",
                   credential=ForwardedCredential("token-one")):
        store._remember_verdict("ck:C1", {"collection_id": "C1"})
    with acting_as("bob", principal_type="user", source="request",
                   credential=ForwardedCredential("token-b")):
        store._remember_verdict("ck:C1", {"collection_id": "C1"})
    o = OracleService(store, grant_verifier=_Everyone())
    o._cache["ck:C1"] = (o._clock() + 30, b"k" * 32)
    monkeypatch.setattr(wiring, "_oracle_singleton", o, raising=False)
    authority_ledger.forget_verdicts_of("alice")
    left = [k for k in store._verdicts]
    assert all(k[0][0] != "alice" for k in left) and any(k[0][0] == "bob" for k in left)
    assert o._cache == {}
