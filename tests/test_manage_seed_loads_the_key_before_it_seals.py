"""The seed CLI loads the platform encryption key before it writes anything (2026-11-11).

The server loads it at startup; a CLI must load it itself. Without it the key oracle has no KEK,
every sealed field is refused ("context field encryption unavailable"), and a fresh store's seeded
servers and operators were left without their forms and wiring on the home test node.
"""
from __future__ import annotations

import sys

import pytest


def test_main_loads_the_key_before_it_opens_the_store(monkeypatch):
    from mantle.search.mantle import key_provider
    from mantle.system import manage_seed

    order = []

    class Stop(Exception):
        pass

    monkeypatch.setattr(key_provider, "load_platform_key", lambda *a, **k: order.append("key"))

    def connect():
        order.append("store")
        raise Stop
    monkeypatch.setattr(manage_seed, "connect", connect)
    monkeypatch.setattr(sys, "argv", ["manage_seed", "--action", "seed"])
    with pytest.raises(Stop):
        manage_seed.main()
    assert order == ["key", "store"]
