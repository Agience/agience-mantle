"""The rate limiter's two load-bearing properties: WHO it counts, and how much it remembers.

The limiter is the floor under credential stuffing (`.env.example`, "Rate limiting"), so the
only interesting question about it is whether a caller can choose its own bucket. It can, if
`X-Forwarded-For` is read from any peer — a different value per request is a different window,
and the limit never applies. The same header was also the key of a store that never dropped
anything, so the bypass and an unauthenticated memory-growth path were the same defect.

These tests exercise `main._rate_limit_client` and `main._rl_evict_expired` directly rather
than through a client, because both are decided before any route exists: the identity comes
off the connection and the header, and the bound is a property of the store, not of a response.
`test_rate_limit_middleware.py` covers the same two properties over the wire.
"""

from __future__ import annotations

import sys
from collections import OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle import main  # noqa: E402


def _request(peer: str | None, xff: str | None = None):
    """The two things the identity is computed from: the socket peer and the header."""
    headers = {"x-forwarded-for": xff} if xff is not None else {}
    return SimpleNamespace(
        client=SimpleNamespace(host=peer) if peer is not None else None,
        headers=headers,
    )


@pytest.fixture()
def trusted(monkeypatch):
    """One trusted edge at 10.0.0.1, plus a trusted CIDR block."""
    monkeypatch.setattr(
        main, "_RL_TRUSTED_PROXIES", main._parse_trusted_proxies("10.0.0.1,172.16.0.0/12")
    )


@pytest.fixture()
def untrusted(monkeypatch):
    """The default: no proxy is trusted, so no peer's word about the client is taken."""
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", ())


# ---------------------------------------------------------------------------
# WHO gets counted
# ---------------------------------------------------------------------------

def test_forwarded_header_is_ignored_from_an_untrusted_peer(untrusted):
    """The whole bypass in one assertion: the header does not move the bucket."""
    assert main._rate_limit_client(_request("203.0.113.7", "1.2.3.4")) == "203.0.113.7"


def test_every_spoofed_value_lands_in_the_SAME_bucket(untrusted):
    """A caller varying the header per request must not thereby vary its window.

    This is the property, stated as the attacker states it: a thousand different claims
    about who is calling, one socket, one bucket.
    """
    seen = {
        main._rate_limit_client(_request("203.0.113.7", f"10.1.{i // 256}.{i % 256}"))
        for i in range(1000)
    }
    assert seen == {"203.0.113.7"}


def test_forwarded_header_is_honoured_from_a_trusted_peer(trusted):
    assert main._rate_limit_client(_request("10.0.0.1", "198.51.100.9")) == "198.51.100.9"


def test_a_trusted_proxy_may_be_named_by_cidr(trusted):
    assert main._rate_limit_client(_request("172.20.4.5", "198.51.100.9")) == "198.51.100.9"


def test_the_rightmost_untrusted_hop_wins_not_the_leftmost(trusted):
    """A proxy APPENDS, so the left-most entry is the client's own claim about itself.

    Here the caller pre-set `X-Forwarded-For: 9.9.9.9` and the edge appended the address it
    actually saw. Reading left-to-right would hand the caller its bucket back through a real
    trusted edge — the bypass surviving the fix that was supposed to close it.
    """
    assert main._rate_limit_client(_request("10.0.0.1", "9.9.9.9, 198.51.100.9")) == "198.51.100.9"


def test_trusted_hops_are_skipped_walking_right_to_left(trusted):
    """A chain of trusted proxies resolves to the last hop none of them vouched for."""
    req = _request("10.0.0.1", "198.51.100.9, 172.16.0.8, 10.0.0.1")
    assert main._rate_limit_client(req) == "198.51.100.9"


def test_a_non_address_in_the_header_never_becomes_a_bucket(trusted):
    """A proxy emitting garbage is a misconfiguration, not a new client identity."""
    assert main._rate_limit_client(_request("10.0.0.1", "unknown, not-an-ip")) == "10.0.0.1"


def test_a_trusted_proxy_forwarding_nothing_is_still_limited_as_itself(trusted):
    assert main._rate_limit_client(_request("10.0.0.1", "")) == "10.0.0.1"
    assert main._rate_limit_client(_request("10.0.0.1")) == "10.0.0.1"


def test_a_peerless_connection_has_one_name(untrusted):
    assert main._rate_limit_client(_request(None)) == "unknown"


def test_an_unparseable_trusted_proxies_entry_is_dropped_not_widened():
    """A typo must not read as "trust everything"; what parsed is what is trusted."""
    nets = main._parse_trusted_proxies("10.0.0.1, nonsense, 172.16.0.0/12, ")
    assert len(nets) == 2
    assert str(nets[0]) == "10.0.0.1/32"
    assert str(nets[1]) == "172.16.0.0/12"


def test_no_trusted_proxies_configured_is_the_default():
    """Unset means take nobody's word — the value a node with no edge in front of it needs."""
    assert main._parse_trusted_proxies("") == ()


# ---------------------------------------------------------------------------
# How much the store remembers
# ---------------------------------------------------------------------------

def test_expired_windows_are_dropped_from_the_store(monkeypatch):
    """An entry whose newest hit aged out carries no state, so it is not kept."""
    store: OrderedDict = OrderedDict()
    for i in range(50):
        store[f"198.51.100.{i}"] = deque([100.0 + i])
    monkeypatch.setattr(main, "_RL_HITS", store)

    main._rl_evict_expired(cutoff=125.0)

    assert list(store) == [f"198.51.100.{i}" for i in range(25, 50)]


def test_eviction_stops_at_the_first_live_window(monkeypatch):
    """The dead entries are a PREFIX, so the sweep is O(1) amortized rather than a scan.

    A live entry ahead of a dead one would mean the ordering invariant broke — every touch
    re-seats its key at the back, so a later key was seen later.
    """
    store: OrderedDict = OrderedDict(
        [("a", deque([10.0])), ("b", deque([90.0])), ("c", deque([20.0]))]
    )
    monkeypatch.setattr(main, "_RL_HITS", store)

    main._rl_evict_expired(cutoff=50.0)

    assert list(store) == ["b", "c"]


def test_an_empty_window_is_dropped(monkeypatch):
    store: OrderedDict = OrderedDict([("a", deque())])
    monkeypatch.setattr(main, "_RL_HITS", store)
    main._rl_evict_expired(cutoff=0.0)
    assert not store


def test_the_store_drains_to_nothing_when_traffic_stops(monkeypatch):
    """The bound is derived, not chosen: what survives is one entry per client seen in the
    last window, so a quiet minute leaves an empty store rather than a high-water mark."""
    store: OrderedDict = OrderedDict((f"198.51.100.{i}", deque([float(i)])) for i in range(200))
    monkeypatch.setattr(main, "_RL_HITS", store)
    main._rl_evict_expired(cutoff=1e9)
    assert len(store) == 0
