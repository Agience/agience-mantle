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
    # A home plane is refused too unless it is a home node (its issuer under home.agience.ai).
    with pytest.raises(cp.CustodyRefused, match="plane|home node"):
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


# -- The home node's test custody (John, 2026-11-11) ---------------------------------------------

def _home(monkeypatch, tmp_path):
    from mantle.services import custody_plane as cp
    monkeypatch.delenv("MANTLE_CUSTODY_PLANE_OVERRIDE", raising=False)
    monkeypatch.setenv("AGIENCE_PLANE", "home")
    monkeypatch.setenv("ORIGIN_URI", "https://origin.home.agience.ai")
    monkeypatch.delenv("AUTHORITY_ISSUER", raising=False)
    monkeypatch.delenv("EMBER_PEERS", raising=False)
    monkeypatch.setenv("KEYS_DIR", str(tmp_path / "keys"))
    from mantle import config
    monkeypatch.setattr(config, "AUTHORITY_ISSUER", "https://origin.home.agience.ai")
    return cp, str(tmp_path / "keys")


def test_the_home_node_holds_test_custody(monkeypatch, tmp_path):
    cp, keys = _home(monkeypatch, tmp_path)
    assert cp.custody_refusal(keys) is None


def test_home_test_custody_is_given_up_once_the_node_peers(monkeypatch, tmp_path):
    cp, keys = _home(monkeypatch, tmp_path)
    monkeypatch.setenv("EMBER_PEERS", "http://peer.example.invalid:8091")
    assert "peers" in cp.custody_refusal(keys)


@pytest.mark.parametrize("issuer", ["https://origin.agience.ai", "https://origin.staging.agience.ai",
                                    "https://home.agience.ai.example.invalid"])
def test_a_home_plane_whose_issuer_is_not_a_home_host_holds_none(monkeypatch, tmp_path, issuer):
    cp, keys = _home(monkeypatch, tmp_path)
    monkeypatch.setenv("AUTHORITY_ISSUER", issuer)
    assert "home node" in cp.custody_refusal(keys)


@pytest.mark.parametrize("plane", ["rc", "dev", "", "test"])
def test_every_other_plane_but_production_is_still_refused(monkeypatch, tmp_path, plane):
    cp, keys = _home(monkeypatch, tmp_path)
    monkeypatch.setenv("AGIENCE_PLANE", plane)
    assert "'production'" in cp.custody_refusal(keys)


def test_a_home_plane_signing_as_another_issuer_holds_none(monkeypatch, tmp_path):
    """Every issuer source must be a home host, as Origin decides (review, 2026-11-11)."""
    cp, keys = _home(monkeypatch, tmp_path)
    from mantle import config
    monkeypatch.setattr(config, "AUTHORITY_ISSUER", "https://origin.agience.ai")
    assert "home node" in cp.custody_refusal(keys)
    monkeypatch.setattr(config, "AUTHORITY_ISSUER", "https://origin.home.agience.ai")
    monkeypatch.setenv("AUTHORITY_ISSUER", "https://my.agience.ai")
    assert "home node" in cp.custody_refusal(keys)


def test_home_test_custody_needs_keys_dir_set_as_origin_reads_it(monkeypatch, tmp_path):
    cp, keys = _home(monkeypatch, tmp_path)
    monkeypatch.delenv("KEYS_DIR")
    assert "KEYS_DIR" in cp.custody_refusal(keys), "a default keys folder is not one the node was given"
