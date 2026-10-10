"""The edge on the same host is trusted by default, so the limiter counts the real client.

In production the edge Caddy runs on the node's host, reaches Mantle over loopback, and sets
`X-Forwarded-For` to the client it saw. With nobody trusted by default, every client shared the
edge's one loopback address: one bucket for the whole internet. Loopback is trusted when
`AGIENCE_TRUSTED_PROXIES` is unset, because no peer off the machine can present a loopback
source address. The variable still overrides the default, and `none` trusts nobody.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle import main  # noqa: E402


def _request(peer, xff=None):
    headers = {"x-forwarded-for": xff} if xff is not None else {}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


@pytest.mark.skipif(bool(os.getenv("AGIENCE_TRUSTED_PROXIES")),
                    reason="the default is only observable with AGIENCE_TRUSTED_PROXIES unset")
def test_the_node_trusts_loopback_out_of_the_box():
    """The module's own configured value — not a monkeypatched one — honours the local edge."""
    assert main._rate_limit_client(_request("127.0.0.1", "198.51.100.9")) == "198.51.100.9"
    assert main._rate_limit_client(_request("::1", "198.51.100.9")) == "198.51.100.9"


@pytest.fixture()
def default(monkeypatch):
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", main._trusted_proxies_from_env(None))


def test_unset_and_empty_both_mean_loopback():
    assert main._trusted_proxies_from_env(None) == main._trusted_proxies_from_env("  ")
    assert {str(n) for n in main._trusted_proxies_from_env(None)} == {"127.0.0.0/8", "::1/128"}


def test_a_remote_peer_still_cannot_name_itself(default):
    """Trusting loopback trusts nobody else: an off-host caller's header moves nothing."""
    assert main._rate_limit_client(_request("203.0.113.7", "1.2.3.4")) == "203.0.113.7"


def test_a_loopback_peer_seen_through_a_dual_stack_socket_is_loopback(default):
    assert main._rate_limit_client(_request("::ffff:127.0.0.1", "198.51.100.9")) == "198.51.100.9"


def test_a_spoofed_prefix_through_the_local_edge_is_still_ignored(default):
    """The edge appends; the client's own claim sits to its left and is not read."""
    assert main._rate_limit_client(_request("127.0.0.1", "9.9.9.9, 198.51.100.9")) == "198.51.100.9"


def test_the_variable_replaces_the_default(monkeypatch):
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", main._trusted_proxies_from_env("10.0.0.1"))
    assert main._rate_limit_client(_request("10.0.0.1", "198.51.100.9")) == "198.51.100.9"
    assert main._rate_limit_client(_request("127.0.0.1", "198.51.100.9")) == "127.0.0.1"


def test_none_trusts_nobody(monkeypatch):
    assert main._trusted_proxies_from_env("none") == ()
    monkeypatch.setattr(main, "_RL_TRUSTED_PROXIES", main._trusted_proxies_from_env("none"))
    assert main._rate_limit_client(_request("127.0.0.1", "198.51.100.9")) == "127.0.0.1"
