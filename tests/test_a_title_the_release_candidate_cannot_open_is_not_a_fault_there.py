"""`seal_titles` fails on a title it could not open, except on the release candidate's copy.

The release candidate (`AGIENCE_PLANE=rc`) rehearses `seal_titles --unseal` on a copy of
production's store under keys of its own and with no custody, so it can open none of production's
sealed titles (2026-11-09: 612 denied). Anywhere else a title the node cannot open is a fault.
"""
from __future__ import annotations

import pytest

from mantle.system.seal_titles import exit_code


def test_nothing_denied_succeeds_on_every_plane(monkeypatch):
    for plane in ("rc", "production", "home"):
        monkeypatch.setenv("AGIENCE_PLANE", plane)
        assert exit_code({"sealed": 4, "unsealed": 4, "denied": 0}) == 0


@pytest.mark.parametrize("plane", ["production", "home", None])
def test_a_denied_title_fails_the_pass_on_a_node_that_holds_its_keys(monkeypatch, plane):
    if plane is None:
        monkeypatch.delenv("AGIENCE_PLANE", raising=False)
    else:
        monkeypatch.setenv("AGIENCE_PLANE", plane)
    assert exit_code({"sealed": 618, "unsealed": 6, "denied": 612}) == 1


def test_on_the_release_candidate_a_denied_title_is_reported_not_failed(monkeypatch, caplog):
    monkeypatch.setenv("AGIENCE_PLANE", "rc")
    with caplog.at_level("INFO"):
        assert exit_code({"sealed": 618, "unsealed": 6, "denied": 612}) == 0
    assert "612" in caplog.text
