"""A route this node deleted must have no caller left in the workspace.

`test_route_reshape::test_the_old_path_is_gone_entirely` removes
`/artifacts/search` and friends deliberately — 404, not 301 — and its docstring gives the
reason: "Every consumer is in-house; a redirect would let a stale client keep working while its
author believes it was migrated."

Measured 2026-08-26: that premise was false. Four in-house consumers were never migrated, and the
404 produced exactly the silent failure choosing it over a redirect was meant to prevent —

* `chorus/src/astra/server.py` — `deduplicate` swallowed the error and returned `"unique"` for
  every artifact, so it had never once found a duplicate;
* `chorus/src/sage/retrieval.py` — logged at INFO and returned `[]`, so grounding was silently
  skipped on every answer;
* `chorus/src/seraph/agent/mantle_client.py` — returned `[]`, so every search came back empty;
* `crystal/src/crystal/mantle.py` — raised. The only one that failed loudly.

The guard belongs here, beside the deletion, because the claim is made here: a test that states
its own premise and does not check it is the shape this whole class keeps taking.

Skipped when the siblings are not checked out — this repo must remain testable alone. The chorus
bundling suite uses the same escape for the same reason.
"""
from __future__ import annotations

import pathlib
import ast
import re

import pytest

_HERE = pathlib.Path(__file__).resolve()
_WORKSPACE = _HERE.parents[2]

#: The paths `test_the_old_path_is_gone_entirely` removes. Kept in step with it by
#: `test_the_removed_set_matches_the_removal_test` below, so this cannot quietly cover less.
REMOVED_PREFIXES = ("/artifacts/search", "/issuers", "/platform", "/servers")

#: Where an in-house caller would live. `_ci-work` is CI's own clone of every repo and is never a
#: workspace member — sweeping it counts every file twice and reports duplicates as findings.
SIBLINGS = ("agience-chorus", "agience-crystal", "agience-ember", "agience-observe",
            "agience-origin", "agience-prism/py")

_SKIP = [".git", "node_modules", "_ci-work", "dist", "build", "__pycache__", ".venv"]


def _sources():
    for name in SIBLINGS:
        root = _WORKSPACE / name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP for part in path.parts):
                continue
            #: Tests are not callers, and excluding them is the difference between a guard
            #: about behaviour and a ban on a word. A test that asserts a removed path stays
            #: removed has to name it, and this guard flagged exactly that — `crystal`'s own
            #: regression test for the leg deleted on 2026-08-26. Forbidding the name there would
            #: force the deletion of the test that keeps the deletion honest.
            #:
            #: The cost, stated rather than hidden: a genuinely dead call living inside a test
            #: is invisible here. That is the right trade — a test calling a 404 fails on its own,
            #: loudly, which is not true of the production callers this exists to catch.
            if path.name.startswith("test_") or "tests" in path.parts:
                continue
            yield path


#: A literal reaches a removed route if it is the path itself — `f"{base}/servers/register"`, the
#: dominant shape here — or a whole URL carrying it, `"http://host/servers/register"`.
#:
#: A hardcoded URL literal does not start with the prefix, so an unanchored prefix check alone
#: would miss it. Anchored deliberately — an unanchored search would match the prefix mid-path and
#: flag another service's unrelated route.
_REACHES = {
    p: re.compile(r"^(?:[a-z][a-z0-9+.-]*://[^/]*)?" + re.escape(p) + r"(?:/|$)")
    for p in REMOVED_PREFIXES
}


def _path_literals(tree: ast.AST):
    """Every string a call could send as a path — including f-string fragments.

    A regex over quoted text alone cannot see `f"{base}/servers/register"`: what sits between the
    quotes there is `{base}/servers/register`, which starts with `{`, so no prefix would ever
    match it — and `f"{BASE}/path"` is the dominant call shape in this workspace, the shape under
    which crystal was posting to a removed route while a quoted-text guard passed.

    Reading the AST instead: an f-string is a `JoinedStr` whose literal halves are separate
    `Constant` nodes, so `/servers/register` is visible on its own. Docstrings are skipped — prose
    naming a dead path is a record of the migration and must stay writable — and comments never
    reach the AST at all, which is the property that makes this approach right rather than merely
    different.
    """
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node.value
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield part.value


def _calls_in(text: str):
    """Path fragments that would REACH a removed route."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    hits = set()
    for literal in _path_literals(tree):
        bare = literal.split("?")[0]
        for prefix in REMOVED_PREFIXES:
            if _REACHES[prefix].match(bare):
                hits.add(bare)
    return hits


@pytest.mark.skipif(not (_WORKSPACE / "agience-chorus").is_dir(),
                    reason="siblings not checked out beside mantle")
def test_no_sibling_repo_calls_a_removed_route():
    offenders = []
    for path in _sources():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for hit in sorted(_calls_in(text)):
            offenders.append("%s -> %s" % (path.relative_to(_WORKSPACE).as_posix(), hit))
    assert not offenders, (
        "in-house callers of a route this node returns 404 for:" + chr(10) + "  "
        + (chr(10) + "  ").join(offenders))


def test_the_removed_set_matches_the_removal_test():
    """The two lists must not drift: a prefix removed there and absent here is unguarded."""
    src = (_HERE.parent / "test_route_reshape.py").read_text(encoding="utf-8")
    m = re.search(r'parametrize\("prefix",\s*\[([^\]]*)\]', src)
    assert m, "could not find the removal test's prefix list"
    theirs = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert theirs == set(REMOVED_PREFIXES), (
        "the removal test now covers %s; this guard covers %s"
        % (sorted(theirs), sorted(REMOVED_PREFIXES)))


def test_the_scan_reaches_real_files():
    """A guard that reads nothing passes for ever."""
    seen = sum(1 for _ in _sources())
    assert seen > 50, "only %d sibling source files scanned — the sweep is not reaching them" % seen
