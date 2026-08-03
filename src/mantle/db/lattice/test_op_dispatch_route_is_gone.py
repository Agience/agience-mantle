"""Guard: the artifact op-dispatch route stays OUT of mantle — the enforced half of a deleted claim.

On 2026-07-29 ten tests were deleted from `tests/test_router_artifacts.py` and
`tests/test_router_inbound_nonce_enforcement.py`. They had been skipped UNCONDITIONALLY since Phase 2b
with the reason *"artifact op-dispatch (/op/{op}) moved to the gateway (crystal); covered in the
gateway's tests"* — a claim held in a comment, which is to say held nowhere. A permanently-skipped test
is a silent pass: dead code presenting as coverage, printing an `s` that reads like caution rather than
absence.

Deleting them was justified by measurement, not by the comment: every one of the ten posted to
`/artifacts/.../op/...`, and that route is absent from mantle's live source. But deleting tests removes
the only executable record that the route is not supposed to be here. This file is that record.

WHAT WOULD BREAK WITHOUT IT: someone re-adds an op-dispatch route to mantle — reasonably, to fix a
caller that still expects it — and nothing objects, because the tests that would have exercised it are
gone and the reasoning for their removal survives only as prose. Then mantle has a dispatch surface with
no auth-guard, invoke, commit, revert, move or nonce coverage at all, and the gateway has a second one.

Scope note: this asserts the route's ABSENCE, not that crystal covers dispatch. Crystal's
`test_dispatcher.py` does cover grant-gated dispatch; it has zero coverage of `op/`, `revert`, `nonce`,
`challenge` or `requires_user`. Asserting a claim about another repo's coverage from here would be a
check that drifts the moment that repo is reorganised — so if the route returns, the failure message
says what has to be re-established rather than pretending it exists elsewhere.
"""
from __future__ import annotations

import pathlib
import re

import pytest

#: `<repo>/src/mantle/db/lattice/` → parents[2] is `src/mantle`. Depth asserted, never trusted: a wrong
#: `parents[N]` resolves to a directory that does not exist, and this workspace has been bitten by that
#: idiom twice — once loudly, once SILENTLY.
MANTLE = pathlib.Path(__file__).resolve().parents[2]

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
        "path depth is wrong: parents[2] should be src/mantle, got %s — fix the depth" % MANTLE)
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


def test_the_scan_actually_reaches_the_live_surface():
    """Vacuous-pass guard FIRST: if the globs stopped resolving, the real check passes on nothing.

    This is the failure mode that makes an absence-assertion worthless — "we found no op route" is also
    what you get from scanning an empty file list.
    """
    files = list(_live_sources())
    assert len(files) > 20, "only %d live source files found — the scan is not reaching mantle" % len(files)
    names = {p.name for p in files}
    assert "artifacts_router.py" in names, "artifacts_router.py not scanned — that is where it would land"


def test_no_artifact_op_dispatch_route_in_mantle():
    """THE CHECK. If this fails, the route came back and the deleted coverage must be re-established."""
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
        "\n\nTen tests covering this were DELETED on 2026-07-29 because the route was absent (auth "
        "guards on invoke/commit, invoke happy-path + undeclared-404, commit + commit_preview, revert, "
        "move, and two nonce/challenge 403 cases). Crystal does NOT cover them — it has zero hits for "
        "op/, revert, nonce, challenge or requires_user. Re-establish that coverage here, or take the "
        "route back out.")


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
])
def test_the_deleted_tests_did_not_creep_back_as_skips(nodeid):
    """The deletions must not be re-added and re-quarantined — that is the state this replaced.

    Re-adding one is fine IF the route is back and the test runs. Re-adding it as an unconditional skip
    puts the silent pass straight back, so the roster is pinned by name.
    """
    cls, method = nodeid.split("::")
    for p in (MANTLE / "tests").rglob("test_*.py"):
        text = p.read_text(encoding="utf-8-sig", errors="replace")
        if ("def %s" % method) in text and ("class %s" % cls) in text:
            assert "skip" not in text.split("def %s" % method)[0][-400:].lower(), (
                "%s was re-added under a skip marker in %s — a permanent skip is a silent pass. If the "
                "op route is back, let it RUN." % (nodeid, p.name))
