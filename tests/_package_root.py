"""The one place that knows where the seeds/types package tree is.

Every test that needs the tree resolves its root through this module rather than computing its
own — a single source keeps `conftest.py`'s skip gate and each test's own path in agreement.

Resolution order, deliberately defaulting to an absent path so nothing changes by accident:
  1. `MANTLE_PACKAGE_ROOT` if set — the reproducible override;
  2. `agience-mantle/package` — the in-repo location, normally absent;
  3. never guessed from a sibling repo silently. A caller that wants the bundle tree names it.
"""
from __future__ import annotations

import os
from pathlib import Path

#: `<repo>/tests/` → parents[1] is the agience-mantle repo root. Asserted, not trusted: a
#: wrong `parents[N]` here can fail silently instead of raising.
_MANTLE_REPO = Path(__file__).resolve().parents[1]
assert (_MANTLE_REPO / "src" / "mantle").is_dir(), (
    "path depth is wrong: parents[1] should be the agience-mantle repo root, got %s" % _MANTLE_REPO)


def package_root() -> Path:
    env = os.environ.get("MANTLE_PACKAGE_ROOT")
    return Path(env).resolve() if env else _MANTLE_REPO / "package"


# The tree is split across two repos:
#   · seeds → `agience-bundle/package/seeds`
#   · types → `agience-crystal/src/types`
# Deriving both from a single `package_root()` cannot locate them, since no common parent
# contains both.
#
# Resolution order per subtree: explicit env override → the historical in-repo path if it exists →
# the sibling repo. The sibling leg is what switches the tree-dependent tests on in a workspace
# checkout; a standalone `agience-mantle` checkout finds nothing and skips cleanly, which is what
# the standalone-installable claim requires.
#
#     MANTLE_SEEDS_ROOT=<genesis>/agience-bundle/package/seeds \
#     MANTLE_TYPES_ROOT=<genesis>/agience-crystal/src/types \
#     python -m pytest -q tests

#: Sibling locations, used only when a full workspace checkout is present. A standalone
#: `agience-mantle` checkout finds nothing here and `have_tree()` answers False, so the
#: tree-dependent tests skip cleanly — the behaviour the standalone-installable claim needs. This
#: module is the only place that names a sibling repo; every caller asks it rather than guessing.
_WORKSPACE = _MANTLE_REPO.parent
_SIBLING_SEEDS = _WORKSPACE / "agience-bundle" / "package" / "seeds"
_SIBLING_TYPES = _WORKSPACE / "agience-crystal" / "src" / "types"


def seeds_root() -> Path:
    env = os.environ.get("MANTLE_SEEDS_ROOT")
    if env:
        return Path(env).resolve()
    derived = package_root() / "seeds"
    return derived if derived.is_dir() else _SIBLING_SEEDS


def types_root() -> Path:
    env = os.environ.get("MANTLE_TYPES_ROOT")
    if env:
        return Path(env).resolve()
    derived = package_root() / "types"
    return derived if derived.is_dir() else _SIBLING_TYPES


def have_tree() -> bool:
    """True when BOTH subtrees are present — the condition the skip gate uses."""
    return seeds_root().is_dir() and types_root().is_dir()
