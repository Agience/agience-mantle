"""`GET /artifacts/{id}/commits` answers any artifact, and says so.

What was removed, and why deleting beat correcting. A `400 "Commits only available for
collections"` stood in this handler with three faults at once:

  1. Its CONDITION was `not _artifact_exists(...)` — an existence test — while its MESSAGE described
     a type restriction. A missing artifact would have reported as a type error.
  2. It was UNREACHABLE. `_artifact_exists` calls `get_raw_artifact`; `check_access`, on the line
     above, calls the same function and raises `404 Not found` when it returns nothing.
  3. The restriction it announced was enforced NOWHERE — nothing on this path tests whether the
     artifact is a collection.

Correcting only the message would have left unreachable code asserting a rule nothing enforces.
Implementing the rule would have added a refusal for callers who ask today and are answered, buying
no safety: asking a plain artifact for its commits is a read, and the honest answer is an empty
page. So the branch went, and this file pins the behaviour that was always actually there.
"""
from __future__ import annotations

import ast
import inspect

from mantle.routers import artifacts_router as ar


def test_no_unreachable_existence_check_remains_in_commits():
    """The specific shape that was wrong: an existence test raising a 4xx AFTER `check_access` has
    already 404'd on the same lookup."""
    tree = ast.parse(inspect.getsource(ar.list_commits).lstrip())
    calls = {c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
             for c in ast.walk(tree) if isinstance(c, ast.Call)}
    assert "_artifact_exists" not in calls, (
        "`list_commits` tests existence again after `check_access` has already done it on the same "
        "`get_raw_artifact` lookup — the branch cannot be reached")


def test_commits_declares_no_400():
    """The 400 was the only one this route could name, and it could not happen. If a 400 returns
    here it should be because a real condition raises it — at which point this test should be
    updated deliberately, not deleted."""
    from mantle.main import app

    spec = app.openapi()
    path = next(p for p in spec["paths"] if p.endswith("/commits"))
    declared = set(spec["paths"][path]["get"]["responses"])
    assert "400" not in declared, (
        "commits declares a 400 again; the last one described a rule the code did not have")


def test_the_message_that_described_a_rule_the_code_lacked_is_gone():
    """Belt and braces on the wording itself. The sentence was the most misleading part: it told a
    reader there was a collection check, and there never was one."""
    src = inspect.getsource(ar.list_commits)
    assert "only available for collections" not in src.lower(), (
        "the message is back. If a collection restriction is genuinely wanted, it needs a condition "
        "that tests for a collection — the old one tested existence")
