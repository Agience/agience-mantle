"""A combined master key stays in the oracle's cache for a short time only, then the store is
asked again.

Under threshold custody the store's read asks Origin, which applies its replica's grants at that
moment, so this is what bounds how long a revoked grant keeps opening a warm key: at most
`MASTER_KEY_CACHE_TTL_MAX` (five minutes), whatever the setting says.
"""
from __future__ import annotations

import pytest

from mantle.search.mantle import oracle as o
from mantle.services.acting_principal import acting_as

KEY = bytes(range(32))


class CountingStore:
    def __init__(self):
        self.reads = 0

    def get(self, principal_id):
        self.reads += 1
        return KEY

    def put(self, principal_id, key):  # pragma: no cover - not reached
        raise AssertionError("no mint expected")


class _Yes:
    def authorized(self, **_k):
        return True


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _get(oracle):
    with acting_as("alice", principal_type="user"):
        return oracle.get_or_create_master_key(
            "alice", o.KeyRequest(requester_id="alice", purpose=o.KeyPurpose.SELF))


def test_a_cached_key_is_read_again_after_its_time_to_live():
    store, clock = CountingStore(), Clock()
    oracle = o.OracleService(store, grant_verifier=_Yes(), cache_ttl_s=20, clock=clock)
    assert _get(oracle) == KEY and _get(oracle) == KEY
    assert store.reads == 1                         # warm within the TTL
    clock.t += 21
    assert _get(oracle) == KEY
    assert store.reads == 2                         # past it: the store (Origin) is asked again


@pytest.mark.parametrize("setting,expected", [("", 30.0), ("20", 20.0), ("120", 30.0),
                                              ("86400", 30.0), ("0", 0.0), ("-5", 0.0)])
def test_the_setting_never_holds_a_key_past_the_verdict_memo(monkeypatch, setting, expected):
    """Thirty seconds at most: the canon's verdict memo TTL bounds post-revocation validity
    (architecture §51), and a cached key is key-equivalent."""
    monkeypatch.setenv(o.MASTER_KEY_CACHE_TTL_SETTING, setting)
    assert o.master_key_cache_ttl() == expected
    store, clock = CountingStore(), Clock()
    oracle = o.OracleService(store, grant_verifier=_Yes(), clock=clock)
    _get(oracle)
    clock.t += 31
    _get(oracle)
    assert store.reads == 2


def test_a_constructor_value_is_clamped_too():
    store, clock = CountingStore(), Clock()
    oracle = o.OracleService(store, grant_verifier=_Yes(), cache_ttl_s=10_000, clock=clock)
    _get(oracle)
    clock.t += 31
    _get(oracle)
    assert store.reads == 2


def test_a_bad_setting_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(o.MASTER_KEY_CACHE_TTL_SETTING, "five minutes")
    with pytest.raises(ValueError, match=o.MASTER_KEY_CACHE_TTL_SETTING):
        o.master_key_cache_ttl()
