"""Guard: nothing may be hidden by `collect_ignore` while it still PASSES.

🔴 THE DEFECT THIS EXISTS FOR (measured 2026-07-29, Contract Builder). `src/mantle/tests/conftest.py`
carried four `collect_ignore` entries, each with a prose reason. `collect_ignore` is the worst quarantine
available: an ignored file is reported as neither run NOR skipped, so it disappears from every count with
nothing to notice — strictly less visible than `@pytest.mark.skip`, which at least prints an `s`.

Measured by collecting and running each one directly, the reasons were wrong for three of the four, and
one was badly wrong: **`test_operator_and_secret_material.py` collected 4 tests and ALL 4 PASSED**, while
its recorded reason claimed it imported `services.handler_registry` (removed). It does not import it. So
four passing tests covering operator and secret material had been silently dropped from the suite. That
is coverage loss wearing quarantine's clothes, and no count anywhere would ever have shown it.

WHY THIS TEST LIVES UNDER `db/lattice/` rather than beside the conftest it checks: the same reason
`test_embeddable_surface.py` does. `src/mantle/tests/conftest.py` imports `origin`, so a check placed
there cannot run in the bare-store environment — and a guard that only runs where the full stack exists
is a guard that silently stops running. It also must NOT import that conftest (importing it would apply
its collection hooks); the list is read as SOURCE via `ast`.

THE RULE, stated as the failure mode: an entry that passes is a coverage loss; an entry that no longer
exists is a phantom exemption; an entry with no stated reason is an unreviewable one. All three fail here.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

#: This file sits at `<repo>/src/mantle/db/lattice/`, so `parents[2]` is `src/mantle`. The depth is
#: ASSERTED immediately below rather than trusted: a wrong `parents[N]` here resolves to a directory that
#: simply does not exist, and this workspace has already been bitten twice by that idiom — once loudly
#: and once SILENTLY (a `sys.path.insert` of a non-existent dir is a no-op, so the test kept "passing"
#: while measuring nothing). Getting it wrong on the first run of this very file is what prompted the
#: assertion.
# ⚠ THE SUITE MOVED OUT OF THE PACKAGE ON 2026-07-31 (John): `src/mantle/tests/` -> `<repo>/tests/`.
# This file stays under `db/lattice/` for the reason in the header — it must run on the embeddable
# surface, where the unit suite's conftest (which imports `origin`) cannot. So it now reaches UP to
# the repo root rather than across to a sibling of the package.
_MANTLE = pathlib.Path(__file__).resolve().parents[2]          # …/src/mantle
_REPO = _MANTLE.parents[1]                                     # …/agience-mantle
TESTS_DIR = _REPO / "tests"
CONFTEST = TESTS_DIR / "conftest.py"

assert _MANTLE.name == "mantle" and CONFTEST.is_file(), (
    "path depth is wrong: expected src/mantle at %s and the suite conftest at %s (present: %s). "
    "Fix the depth — do not adjust the relative path." % (_MANTLE, CONFTEST, CONFTEST.is_file()))


def _literal(name: str):
    """Read a module-level literal out of the conftest WITHOUT importing it.

    Importing would install its `pytest_collection_modifyitems` hook into this run, which is both a
    side effect and a way for the thing under test to influence its own verdict.
    """
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8-sig", errors="replace"))
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(getattr(t, "id", None) == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError("%s not found in %s — it was renamed or removed, and this guard is now blind"
                         % (name, CONFTEST))


def test_the_conftest_and_its_ignore_list_are_readable():
    """Vacuous-pass guard, first: if the list cannot be read, every assertion below passes on nothing."""
    assert CONFTEST.is_file(), CONFTEST
    entries = _literal("collect_ignore")
    assert isinstance(entries, list) and entries, "collect_ignore is empty or not a list: %r" % (entries,)


def test_every_ignored_file_still_exists():
    """A phantom exemption outlives the file it excused and silently covers the next file of that name."""
    missing = [e for e in _literal("collect_ignore") if not (TESTS_DIR / e).is_file()]
    assert not missing, ("collect_ignore names files that do not exist — delete the entries: %r"
                         % missing)


def test_every_ignored_file_is_documented_as_uncollectable_or_failing():
    """The list may not grow silently: each entry is either declared import-dead or must fail when run."""
    entries = set(_literal("collect_ignore"))
    declared_dead = set(_literal("_IGNORE_UNCOLLECTABLE"))
    assert declared_dead <= entries, (
        "_IGNORE_UNCOLLECTABLE names files that are not ignored: %r" % sorted(declared_dead - entries))


@pytest.mark.parametrize("entry", sorted(set(_literal("collect_ignore"))))
def test_an_ignored_file_does_not_pass(entry):
    """THE LOAD-BEARING CHECK: run each ignored file in a subprocess; a clean pass is a coverage loss.

    A subprocess is required, not stylistic — this process has already imported half of mantle, and
    collecting a quarantined module in-process would both pollute `sys.modules` and let the parent run's
    fixtures change the outcome. `-p no:cacheprovider` keeps it from writing to the shared cache.

    Exit codes: 0 = all passed (the defect), 1 = tests failed (justified), 2 = collection/usage error
    (justified — import-dead), 5 = no tests collected (also a phantom entry).
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(TESTS_DIR / entry)],
        capture_output=True, text=True, timeout=600, cwd=str(TESTS_DIR.parents[2]),
    )
    assert proc.returncode != 0, (
        "%s is hidden by collect_ignore but PASSES CLEANLY — that is a silent coverage loss, not a "
        "quarantine. Remove it from collect_ignore.\n%s"
        % (entry, (proc.stdout or "")[-600:]))
    assert proc.returncode != 5, (
        "%s is ignored but collects NO tests — the entry is a phantom; delete it." % entry)
