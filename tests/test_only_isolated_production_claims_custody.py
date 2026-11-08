"""John, 2026-11-02: threshold custody is refused on the home or development plane, and wherever
isolation is absent. Only isolated production claims custody.

The signal is two things together, because either alone is a claim anyone can write:
* the plane the release tool renders into the service's env (`AGIENCE_PLANE=production`), which the
  home peer and a developer's machine never carry;
* evidence, read at start, that the process runs isolated as the release tool sets it up: not
  root, `NoNewPrivs` set (where the kernel reports it), and Mantle's keys folder owned by this
  process's own user with no access for anyone else (0700).
"""
from __future__ import annotations

import os
import sys

import pytest

from mantle.services import custody_plane as cp


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    keys = tmp_path / "keys"
    keys.mkdir()
    os.chmod(keys, 0o700)
    monkeypatch.setenv("AGIENCE_PLANE", "production")
    monkeypatch.delenv("MANTLE_CUSTODY_PLANE_OVERRIDE", raising=False)
    monkeypatch.setattr(cp, "_euid", lambda: 1001)
    monkeypatch.setattr(cp, "_no_new_privs", lambda: True)
    monkeypatch.setattr(cp, "_owner_and_mode", lambda p: (1001, 0o700))
    return keys


def test_isolated_production_claims_custody(isolated):
    cp.require_isolated_production(str(isolated))


@pytest.mark.parametrize("plane", ["home", "development", "", None])
def test_the_home_or_development_plane_is_refused(isolated, monkeypatch, plane):
    if plane is None:
        monkeypatch.delenv("AGIENCE_PLANE")
    else:
        monkeypatch.setenv("AGIENCE_PLANE", plane)
    with pytest.raises(cp.CustodyRefused, match="plane"):
        cp.require_isolated_production(str(isolated))


def test_root_is_refused(isolated, monkeypatch):
    monkeypatch.setattr(cp, "_euid", lambda: 0)
    with pytest.raises(cp.CustodyRefused, match="root"):
        cp.require_isolated_production(str(isolated))


def test_without_no_new_privs_is_refused(isolated, monkeypatch):
    monkeypatch.setattr(cp, "_no_new_privs", lambda: False)
    with pytest.raises(cp.CustodyRefused, match="NoNewPrivileges"):
        cp.require_isolated_production(str(isolated))


@pytest.mark.parametrize("owner,mode", [(0, 0o700), (1001, 0o750), (1001, 0o755)])
def test_a_keys_folder_not_its_own_or_open_to_others_is_refused(isolated, monkeypatch, owner,
                                                               mode):
    monkeypatch.setattr(cp, "_owner_and_mode", lambda p: (owner, mode))
    with pytest.raises(cp.CustodyRefused, match="keys folder"):
        cp.require_isolated_production(str(isolated))


def test_the_override_holds_only_under_pytest(isolated, monkeypatch):
    monkeypatch.setenv("AGIENCE_PLANE", "home")
    monkeypatch.setenv("MANTLE_CUSTODY_PLANE_OVERRIDE", "1")
    cp.require_isolated_production(str(isolated))                  # in a test run
    monkeypatch.setitem(sys.modules, "pytest", None)
    with pytest.raises(cp.CustodyRefused):
        cp.require_isolated_production(str(isolated))


def test_the_store_refuses_to_build_threshold_custody_off_isolated_production(monkeypatch):
    from mantle.search.mantle import wiring

    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    monkeypatch.setenv("AGIENCE_PLANE", "home")
    monkeypatch.delenv("MANTLE_CUSTODY_PLANE_OVERRIDE", raising=False)
    with pytest.raises(cp.CustodyRefused):
        wiring._master_key_store(object(), lambda: None)


# -- review 19: the same shape as Origin's ---------------------------------------------------------

def test_a_symlinked_folder_is_read_as_the_link(monkeypatch, tmp_path):
    """The owner and mode are the folder's own, never a link's target's."""
    seen = {}
    real = os.stat

    def stat(path, *a, **k):
        seen["follow"] = k.get("follow_symlinks", True)
        return real(path, *a, **k)
    monkeypatch.setattr(cp.os, "stat", stat)
    cp._owner_and_mode(str(tmp_path))
    assert seen["follow"] is False


def test_custody_needs_the_store_folder_its_own_at_0700(isolated, monkeypatch, tmp_path):
    store = tmp_path / "data" / "mantle-lattice.db"
    store.parent.mkdir()
    monkeypatch.setenv("MANTLE_LATTICE_PATH", str(store))
    assert cp.custody_refusal(str(isolated)) is None
    monkeypatch.setattr(cp, "_owner_and_mode",
                        lambda p: (1001, 0o750) if p == str(store.parent) else (1001, 0o700))
    why = cp.custody_refusal(str(isolated))
    assert why and "store" in why
    with pytest.raises(cp.CustodyRefused):
        cp.require_isolated_production(str(isolated))
