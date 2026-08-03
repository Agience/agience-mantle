"""The ONE place that knows where the seeds/types package tree is.

🔴 WHY THIS EXISTS (2026-07-29, Contract Builder). 46 tests were skipped with the message *"requires
package/seeds|types tree (moved to Origin in the repo split)"*. The tree did **not** move to Origin — it
is git-tracked at `agience-bundle/package/{seeds,types}` — so the reason had been pointing readers at the
wrong repo for as long as it had stood.

Making `conftest.py`'s SKIP GATE overridable was necessary but not sufficient: **each test also computed
its own root**, hardcoded as `Path(__file__).resolve().parents[3] / "package"` in four files and six
places. So with the gate overridden the tests un-skipped and then measured an empty directory —
`test_platform_tree_loads_without_errors` asserted `0 == 11` collections against a tree that has 56 real
files in it. Two independent sources of one path is the defect; this module is the single source, and
`conftest.py` reads it too.

Resolution order, and the default is deliberately the ABSENT path so nothing changes by accident:
  1. `MANTLE_PACKAGE_ROOT` if set — the reproducible override;
  2. `agience-mantle/package` — the historical in-repo location, still absent;
  3. never guessed from a sibling repo silently. A caller that wants the bundle tree names it.
"""
from __future__ import annotations

import os
from pathlib import Path

#: `<repo>/tests/` → parents[1] is the agience-mantle repo root. Asserted, not trusted: this
#: workspace has been bitten twice by a wrong `parents[N]`, once silently.
# ⚠ DEPTH CHANGED 2026-07-31: the suite moved from `src/mantle/tests/` to `tests/`, so the repo root
# is now one level up, not three. The assertion below caught this on the first run after the move —
# which is why it is an assertion and not a comment.
_MANTLE_REPO = Path(__file__).resolve().parents[1]
assert (_MANTLE_REPO / "src" / "mantle").is_dir(), (
    "path depth is wrong: parents[3] should be the agience-mantle repo root, got %s" % _MANTLE_REPO)


def package_root() -> Path:
    env = os.environ.get("MANTLE_PACKAGE_ROOT")
    return Path(env).resolve() if env else _MANTLE_REPO / "package"


def seeds_root() -> Path:
    return package_root() / "seeds"


def types_root() -> Path:
    return package_root() / "types"


def have_tree() -> bool:
    """True when BOTH subtrees are present — the condition the skip gate uses."""
    return seeds_root().is_dir() and types_root().is_dir()
