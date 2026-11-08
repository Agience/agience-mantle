"""Review 23, item 0: under pytest, Mantle never opens its default store.

A test that imported Mantle without this repo's pytest config opened the node's default store
(`<repo>/.data/mantle-lattice.db`), the developer's real one. The store's own open path now refuses
the default while pytest is loaded: a test names its store (`MANTLE_LATTICE_PATH`, or a `path`).
Here the default is pointed at a temporary file, so neither run of this test can reach `.data`.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def default_elsewhere(tmp_path, monkeypatch):
    from mantle.db import backend, lattice_api

    fake = tmp_path / "default" / "mantle-lattice.db"
    monkeypatch.setattr(lattice_api, "DEFAULT_LATTICE_PATH", fake)
    monkeypatch.delenv("MANTLE_LATTICE_PATH", raising=False)
    monkeypatch.setattr(backend, "_HANDLE", None, raising=False)
    monkeypatch.setattr(backend, "_CONTENT", None, raising=False)
    return fake


def test_the_default_store_is_refused_under_pytest(default_elsewhere):
    from mantle.db import lattice_api

    with pytest.raises(RuntimeError, match="MANTLE_LATTICE_PATH"):
        lattice_api.open_database()
    assert not default_elsewhere.exists()


def test_the_default_content_tier_is_refused_under_pytest(default_elsewhere):
    from mantle.db import backend

    with pytest.raises(RuntimeError, match="MANTLE_LATTICE_PATH"):
        backend.content_handle()
    assert not default_elsewhere.parent.exists()


def test_a_named_store_opens(tmp_path, monkeypatch):
    from mantle.db import lattice_api

    db = lattice_api.open_database(str(tmp_path / "named.db"), origin="t")
    assert db is not None
