"""Re-verify a failure that a concurrent edit could have caused, and say when that rescues one.

The problem, measured over four full runs. Seven interactive sessions edit this workspace, and
mantle's suite takes 6-9 minutes. **42 test files `ast.parse` source off disk at test time** and 4 of
them cross into sibling repos, so a run parses files that change underneath it. The four runs gave
failure sets of 2 / 4 / 0 / 9, largely disjoint, every one passing in isolation. Two were traced to
the minute: `artifacts_router.py` rewritten at 11:02 inside a run lasting 10m50s, and
`crystal/mantle.py` written by another session inside the window of the run that read it.

THE FIX IS A RE-VERIFY, NOT A RETRY. The distinction matters: a blanket retry hides real
flakiness, which is why a retry in a gate is a thing to add deliberately. This reruns a failed test
**only when the tree actually moved during the run**, and it SAYS SO both times — so a rescue is a
line in the log with a number attached, not a silence.

Why it is not hooked into the reads. The obvious design owns `read_source` and tracks exactly
which files each test parsed. Measured: those 42 files read source SEVEN different ways
(`path.read_text`, `io.open(mod.__file__)`, `inspect.getsource`, `_BUS_SRC.read_text`,...), so
owning the read means editing 42 files to gain precision this does not need. A failure is already
rare; a stat sweep on a failure is cheap; and the question *"did anything under these roots change
since the suite started"* answers the finding's own diagnosis exactly.

What it deliberately does not do. It does not rerun when the tree is quiet, so a genuinely flaky
test still fails and stays visible. It reruns at most ONCE. And a test that fails both times fails,
with the count of moved files appended — the *"self-diagnosing message"* half, which applies whether
or not the re-verify rescued anything.
"""
from __future__ import annotations

import os
import pathlib
import time
from typing import List, Optional, Tuple

#: Source roots a mantle test can legitimately read. The sibling entries are the 4 cross-repo cases
#: — `crystal/mantle.py` among them, which is the file a concurrent edit hit twice.
_ROOTS = ("agience-mantle/src", "agience-mantle/tests",
          "agience-crystal/src", "agience-prism/py/src", "agience-ember/src")

#: When the suite started. Anything newer than this was written while the run was in flight.
_SUITE_START: float = time.time()

#: Rescues, so the number John asked for exists: `[(nodeid, files_moved)]`.
RESCUES: List[Tuple[str, int]] = []

#: Failures that were re-verified and failed again — the ones the race does NOT explain.
CONFIRMED: List[Tuple[str, int]] = []

_WS: Optional[pathlib.Path] = None


def workspace_root() -> pathlib.Path:
    global _WS
    if _WS is None:
        _WS = pathlib.Path(__file__).resolve().parents[2]
    return _WS


def mark_suite_start() -> None:
    """Called once from the session hook, so the baseline is the run's start and not import time."""
    global _SUITE_START
    _SUITE_START = time.time()


def files_moved_since(when: float, limit: int = 400) -> List[str]:
    """Source files written since *when*. Stops at *limit* — the answer is "did the tree move", and
    the exact size of a large edit does not change what to do about it."""
    moved: List[str] = []
    root = workspace_root()
    for rel in _ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", "node_modules")]
            for f in filenames:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    if os.path.getmtime(p) > when:
                        moved.append(os.path.relpath(p, root).replace(os.sep, "/"))
                        if len(moved) >= limit:
                            return moved
                except OSError:
                    continue
    return moved


def the_tree_moved_during(started_at: float) -> List[str]:
    """Files written while one test was running. Scoped to the TEST, not the suite: a change that
    landed before this test began cannot have been parsed mid-parse by it, and rerunning for that
    would be the blanket retry this is not."""
    return files_moved_since(started_at)
