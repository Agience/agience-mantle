"""Guard: the artifact op-dispatch route (`/{artifact_id}/op/...`) must not exist in mantle's live
HTTP surface.

That route has no auth-guard, invoke, commit, revert, move, or nonce coverage of its own in this
repo. Crystal's `test_dispatcher.py` covers grant-gated dispatch, but has zero coverage of `op/`,
`revert`, `nonce`, `challenge`, or `requires_user`. So if the route is ever added back to mantle, it
would run with none of that coverage in place, and this file is what would notice.

This file asserts the route's absence here; it does not claim crystal covers dispatch — a claim
about another repo's coverage would drift the moment that repo is reorganised. If the route returns,
the failure message says what has to be re-established rather than pretending it exists elsewhere.
"""
from __future__ import annotations

import pathlib
import re

import pytest

#: `<repo>/src/mantle/db/` → parents[1] is `src/mantle`. Depth asserted, never trusted: a wrong
#: `parents[N]` resolves silently to a directory that isn't `src/mantle`, so the assertion below checks
#: it explicitly rather than trusting the path arithmetic.
MANTLE = pathlib.Path(__file__).resolve().parents[1]

#: `src/mantle` → parents[0] is `src`, parents[1] is the repo root. The suite lives at `<repo>/tests`,
#: NOT at `src/mantle/tests` — that directory was deleted. Pointing an `rglob` at the old location is
#: the exact failure this file exists to prohibit: `Path.rglob` on a missing directory yields nothing
#: and raises nothing, so every conclusion drawn from it is drawn from an empty list.
REPO = MANTLE.parents[1]
TESTS_DIR = REPO / "tests"

#: Directories holding the live HTTP surface. Deliberately NOT the whole tree: `tests/` and this file
#: itself legitimately mention the route in prose, and matching those would make the check unfixable.
LIVE_DIRS = ("routers", "services")
LIVE_FILES = ("main.py",)

#: The route shape, in the spellings FastAPI would accept. `/op/` alone is too broad — it would match an
#: unrelated path segment — so this requires the artifact-scoped form.
_PATTERNS = (
    re.compile(r'["\']/?\{?artifact_id\}?/op/'),
    re.compile(r'["\']/\{[a-z_]+\}/op/'),
    re.compile(r'@router\.(post|get|patch|put)\([^)]*?/op/', re.S),
)


def _live_sources():
    assert MANTLE.name == "mantle", (
        "path depth is wrong: parents[1] should be src/mantle, got %s — fix the depth" % MANTLE)
    for d in LIVE_DIRS:
        base = MANTLE / d
        assert base.is_dir(), "expected live source dir missing: %s" % base
        for p in base.rglob("*.py"):
            if "__pycache__" not in p.parts and not p.name.startswith("test_"):
                yield p
    for f in LIVE_FILES:
        p = MANTLE / f
        assert p.is_file(), "expected live source file missing: %s" % p
        yield p


def _suite_test_files():
    """Every test module the repo actually collects, from both roots that hold one.

    Returns a list rather than yielding, and asserts it is non-empty, because the caller's verdict is
    an absence — "the roster is clean" and "we looked at nothing" are the same result otherwise.
    """
    assert MANTLE.name == "mantle", (
        "path depth is wrong: parents[1] should be src/mantle, got %s — fix the depth" % MANTLE)
    assert TESTS_DIR.is_dir(), (
        "the suite directory is missing at %s — this scan would silently examine nothing. Fix the "
        "path; do not delete the check." % TESTS_DIR)
    here = pathlib.Path(__file__).resolve()
    found = {}
    for root in (TESTS_DIR, MANTLE):
        for p in root.rglob("test_*.py"):
            if "__pycache__" in p.parts:
                continue
            rp = p.resolve()
            if rp != here:
                found[rp] = p
    files = sorted(found.values())
    assert len(files) > 50, (
        "only %d test modules found under %s and %s — the scan is not reaching the suite, and an "
        "absence-check over an empty file list passes on nothing" % (len(files), TESTS_DIR, MANTLE))
    return files


def test_the_suite_scan_actually_reaches_the_test_files():
    """Vacuous-pass guard for the roster check below — the sibling guard covers `_live_sources()`
    only, and did not notice when this file's own roster scan pointed at a deleted directory.

    Both roots must contribute: `<repo>/tests` holds the bulk of the suite, and `src/mantle` holds
    the store-local guards (this file among them). A re-added op test could land in either.
    """
    files = _suite_test_files()
    parents = {p.parent for p in files}
    assert any(p == TESTS_DIR or TESTS_DIR in p.parents for p in parents), (
        "no file under %s was scanned — the suite root is not being reached" % TESTS_DIR)
    assert any(MANTLE in p.parents for p in parents), (
        "no file under %s was scanned — the store-local guards are not being reached" % MANTLE)
    names = {p.name for p in files}
    assert "test_router_artifacts.py" in names, (
        "test_router_artifacts.py not scanned — that is the suite the op tests would return to")


def test_the_scan_actually_reaches_the_live_surface():
    """Vacuous-pass guard, checked first: if the globs stopped resolving, the real check passes on nothing.

    This is the failure mode that makes an absence-assertion worthless — "we found no op route" is also
    what you get from scanning an empty file list.
    """
    files = list(_live_sources())
    assert len(files) > 20, "only %d live source files found — the scan is not reaching mantle" % len(files)
    names = {p.name for p in files}
    assert "artifacts_router.py" in names, "artifacts_router.py not scanned — that is where it would land"


def test_no_artifact_op_dispatch_route_in_mantle():
    """The check: if this fails, the route is back and needs auth-guard, invoke, commit, revert,
    move, and nonce coverage re-established before it ships."""
    hits = []
    for p in _live_sources():
        text = p.read_text(encoding="utf-8-sig", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if any(pat.search(line) for pat in _PATTERNS):
                hits.append("%s:%d  %s" % (p.relative_to(MANTLE).as_posix(), i, line.strip()[:100]))
    assert not hits, (
        "an artifact op-dispatch route is back in mantle:\n  " + "\n  ".join(hits) +
        "\n\nThe route does not exist in mantle, so the eleven tests that once covered it are not "
        "present here either (auth guards on invoke/commit, invoke happy-path + undeclared-404, "
        "commit + commit_preview, revert, move, and three nonce/challenge cases). Crystal does NOT "
        "cover them — it has zero hits for op/, revert, nonce, challenge or requires_user. "
        "Re-establish that coverage here, or take the route back out.")


def test_the_check_can_fail(tmp_path):
    """Seeded-violation proof. An absence-assertion that has never been shown to fire is a belief."""
    seeded = tmp_path / "routers"
    seeded.mkdir()
    (seeded / "sneaky_router.py").write_text(
        '@router.post("/{artifact_id}/op/{op_name}")\nasync def dispatch_op(): ...\n', encoding="utf-8")
    found = [
        line for p in seeded.rglob("*.py")
        for line in p.read_text(encoding="utf-8").splitlines()
        if any(pat.search(line) for pat in _PATTERNS)
    ]
    assert found, "the patterns do not match a real op-dispatch route declaration — the check is inert"


@pytest.mark.parametrize("nodeid", [
    "TestAuthGuards::test_invoke_requires_user",
    "TestAuthGuards::test_commit_requires_user",
    "TestInvokeArtifact::test_happy_path_dispatches_invoke_op",
    "TestInvokeArtifact::test_undeclared_invoke_returns_404",
    "TestCommitArtifacts::test_commit_dispatches_through_op_endpoint",
    "TestCommitArtifacts::test_commit_preview_dispatches_through_op_endpoint",
    "TestRevertArtifact::test_revert_dispatches_through_op_endpoint",
    "TestMoveArtifact::test_happy_path_passes_collection_id_as_source",
    "TestInvokeOpNonce::test_invoke_without_challenge_is_403",
    "TestInvokeOpNonce::test_invoke_with_bad_challenge_is_403",
    "TestInvokeOpNonce::test_invoke_with_valid_challenge_passes_nonce_gate",
])
def test_the_deleted_tests_did_not_creep_back_as_skips(nodeid):
    """None of the tests named below may be re-added under an unconditional skip: that reproduces
    a silent pass, coverage that reads as present but asserts nothing.

    Re-adding one is fine if the route is back and the test actually runs. Re-adding it as an
    unconditional skip does not count, so the roster of nodeids is pinned by name here.
    """
    cls, method = nodeid.split("::")
    for p in _suite_test_files():
        text = p.read_text(encoding="utf-8-sig", errors="replace")
        if ("def %s" % method) in text and ("class %s" % cls) in text:
            assert "skip" not in text.split("def %s" % method)[0][-400:].lower(), (
                "%s was re-added under a skip marker in %s — a permanent skip is a silent pass. If the "
                "op route is back, let it RUN." % (nodeid, p))
