"""Mantle never touches Origin's setup token.

Origin owns the setup wizard and its one-time token (`KEYS_DIR/setup.token`). A single-host install
(agience-observe) shares one `KEYS_DIR` between Origin and Mantle, so a Mantle that deleted the
file at startup, as it once did, left Origin with no token to finish setup after its next restart.
No module of Mantle's loads, deletes or reads it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import mantle

_SETUP_TOKEN_CALLS = {"init_setup_token", "delete_setup_token", "get_setup_token",
                      "verify_setup_token"}


def _names_in(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.alias):
            out.add(node.name.rsplit(".", 1)[-1])
    return out


def test_no_mantle_module_handles_the_setup_token():
    root = Path(mantle.__file__).parent
    hits = []
    for path in root.rglob("*.py"):
        if "tests" in path.parts:
            continue
        found = _names_in(path) & _SETUP_TOKEN_CALLS
        if found:
            hits.append((str(path.relative_to(root)), sorted(found)))
    assert hits == []
