"""The re-verify fires only when the tree moved, and says so when it rescues something.

What it guards against is its own misuse: a rerun-on-failure is one edit away from a blanket
retry, and a blanket retry hides exactly the flakiness a suite exists to surface. Both halves are
required — re-verify the offender and log when a re-verify rescues a test — and the second half is
what keeps the first honest: a rescue is a line with a number attached, so "is this race worth more
effort" stays measurable instead of becoming a silence.

The race is real and dated. Four full mantle runs gave failure sets of 2 / 4 / 0 / 9, largely
disjoint, every one passing in isolation. 42 test files here `ast.parse` source off disk and 4 cross
into sibling repos; seven interactive sessions edit this workspace while a 6-9 minute run is in
flight. Two failures were traced to the minute — `artifacts_router.py` rewritten at 11:02 inside a
run lasting 10m50s.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import time

from . import source_race as race


def test_a_quiet_tree_moves_nothing():
    """The baseline. If this ever reports movement on an idle workspace, every failure would be
    rerun and the mechanism would BE the blanket retry it is written not to be."""
    now = time.time()
    time.sleep(0.05)
    assert race.files_moved_since(now) == []


def test_a_touched_source_file_is_seen(tmp_path):
    """Proven able to fire, against a real file under a real root — not a stubbed clock."""
    target = race.workspace_root() / "agience-mantle" / "src" / "mantle" / "__init__.py"
    if not target.is_file():
        import pytest
        pytest.skip("mantle source root is not present")
    started = time.time()
    time.sleep(0.05)
    target.touch()
    try:
        moved = race.the_tree_moved_during(started)
        assert any(m.endswith("mantle/__init__.py") for m in moved), moved
    finally:
        # Leave the mtime where a concurrent session would expect it: touched, not reverted to a
        # past time. Reverting would itself be a write, and a lie about when the file changed.
        pass


def test_the_roots_include_the_cross_repo_cases():
    """4 of the 42 source-reading tests cross into sibling repos, and `crystal/mantle.py` is the
    file a concurrent edit hit twice. A watcher scoped to mantle alone would miss the case that
    produced the finding."""
    assert "agience-crystal/src" in race._ROOTS
    assert "agience-mantle/src" in race._ROOTS


# ── the wiring, asserted at the source ───────────────────────────────────────────────────────────

def _conftest() -> str:
    return (pathlib.Path(__file__).parent / "conftest.py").read_text(encoding="utf-8-sig")


def _protocol_ast():
    """The `pytest_runtest_protocol` hook, as a syntax tree.

    Asked of the structure, not the text: a string-based assertion would break on a rewrite that
    changes nothing about the behaviour — pinning the literal `if not moved:` breaks the moment it
    becomes `if moved:`, and matching the word "while" breaks if it only appears in a comment. A
    test that a comment can turn red is testing prose.
    """
    import ast

    tree = ast.parse(_conftest())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "pytest_runtest_protocol":
            return node
    raise AssertionError("conftest no longer defines pytest_runtest_protocol")


def test_the_rerun_is_conditional_on_the_tree_having_moved():
    """The one branch that separates this from a retry. Rerunning unconditionally would hide the
    flakiness a suite exists to surface, and every other test in this file would still pass."""
    import ast

    fn = _protocol_ast()
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "runtestprotocol"]
    assert len(calls) == 2, (
        "expected exactly two `runtestprotocol` calls — the first attempt and one re-verify — "
        "found %d" % len(calls))

    consults = [n for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "the_tree_moved_during"]
    assert consults, (
        "the rerun no longer consults whether the tree moved, so it reruns every failure — a "
        "blanket retry")

    # The second attempt must sit inside a branch, not at the top level of the function body.
    top_level = {id(n) for n in fn.body}
    assert any(id(c) not in top_level for c in calls), (
        "both attempts run unconditionally; the re-verify is not gated on anything")


def test_it_reruns_at_most_once():
    """A loop here would turn a deterministic failure into a hang under an actively edited tree."""
    import ast

    fn = _protocol_ast()
    # NOT "no loops" — the hook legitimately iterates `reports` to log them, and asserting no
    # loop at all made this red over that. The property is that no loop may CONTAIN an attempt.
    offenders = []
    for loop in [n for n in ast.walk(fn) if isinstance(n, (ast.While, ast.For))]:
        for inner in ast.walk(loop):
            if isinstance(inner, ast.Call) and getattr(inner.func, "id", None) == "runtestprotocol":
                offenders.append(ast.unparse(loop)[:80])
    assert not offenders, (
        "a test attempt runs inside a loop, so a failure under a continuously-edited tree would "
        "retry without bound: %s" % offenders)


def test_both_outcomes_are_reported():
    """A rescue that is not printed is indistinguishable from a test that simply passed — which
    is the failure mode of every retry mechanism, and the reason the ruling asked for the log."""
    src = _conftest()
    assert "RESCUED" in src and "CONFIRMED" in src, (
        "the terminal summary no longer distinguishes a rescued failure from one that failed twice")


def test_a_confirmed_failure_still_fails():
    """Measured by running pytest, not by reading the hook. A test that fails both times must
    still fail the run — otherwise the mechanism converts every real defect into a green suite."""
    probe = pathlib.Path(__file__).parent / "_probe_always_fails.py"
    probe.write_text(
        "def test_always_fails():\n"
        "    assert False, 'planted'\n",
        encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:randomly"],
            cwd=str(pathlib.Path(__file__).parents[1]),
            capture_output=True, text=True, timeout=600)
        assert r.returncode != 0, (
            "a planted always-failing test PASSED the run — the re-verify is swallowing failures\n"
            + r.stdout[-2000:])
        assert "1 failed" in r.stdout, r.stdout[-2000:]
    finally:
        probe.unlink(missing_ok=True)


def test_a_failure_under_a_moving_tree_is_rescued_and_announced():
    """The whole mechanism, end to end, and the only test that proves it works.

    Everything above asserts that the re-verify does not fire when it should not. This one makes it
    fire: a probe that touches a real source file (so the tree genuinely moves during the test) and
    fails on its first invocation, passing on its second. It must end green and print RESCUED — a
    rescue that is not announced is the failure mode this exists to avoid.
    """
    probe = pathlib.Path(__file__).parent / "_probe_moves_the_tree.py"
    flag = probe.parent / "_probe_moves_the_tree.flag"
    flag.unlink(missing_ok=True)
    probe.write_text(
        "import pathlib\n"
        "def test_fails_once_while_the_tree_moves():\n"
        "    flag = pathlib.Path(__file__).with_suffix('.flag')\n"
        "    # Touch a real source file so `the_tree_moved_during` sees movement in THIS test.\n"
        "    target = pathlib.Path(__file__).parents[1] / 'src' / 'mantle' / '__init__.py'\n"
        "    target.touch()\n"
        "    if not flag.exists():\n"
        "        flag.write_text('seen', encoding='utf-8')\n"
        "        assert False, 'first invocation fails, as a concurrently-edited parse would'\n",
        encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:randomly"],
            cwd=str(pathlib.Path(__file__).parents[1]),
            capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, (
            "a failure under a genuinely moving tree was NOT rescued:\n" + r.stdout[-3000:])
        assert "RESCUED" in r.stdout, (
            "the run went green without announcing the rescue — indistinguishable from a test that "
            "simply passed:\n" + r.stdout[-3000:])
    finally:
        probe.unlink(missing_ok=True)
        flag.unlink(missing_ok=True)


def test_the_helper_module_is_the_only_place_the_roots_are_named():
    """One list, so a root added for the watcher cannot disagree with the root the rerun checks."""
    src = _conftest()
    assert "_ROOTS" not in src, "the source roots are duplicated into conftest; keep one copy"
