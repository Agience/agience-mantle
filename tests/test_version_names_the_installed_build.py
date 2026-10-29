"""`/version` names the build even where nothing stamped `build_info.json`.

2026-10-29: a release installs Mantle from its wheel into a slot, with no `build_info.json`, and
production Mantle answered `{"version": ""}`; the platform check could not tell which build it ran.
"""
from __future__ import annotations

from importlib.metadata import version

from mantle import main


def test_an_unstamped_install_names_its_installed_version(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "BUILD_INFO_PATH", str(tmp_path / "absent.json"))
    monkeypatch.setattr(main, "Path", lambda *a: tmp_path / "nowhere" / "x" / "y")
    info = main._load_build_info()
    assert info["version"] == version("agience-mantle") != ""
