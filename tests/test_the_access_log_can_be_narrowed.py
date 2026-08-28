""", L2, L4: the access log lied about its size and could not be narrowed by time.

L2, and a correction to this audit's own earlier decision. The route returned
`total=len(events), has_more=False`. `total` was the length of the page the caller was already
holding, and `has_more: false` said the log ended at the end of page one. An earlier decision in
the worksheet recorded L2 as closed by the `{items, total, has_more}` envelope — it was not: the
envelope arrived and kept being handed the page length.

L1: `result` was a free string, and the query tests `if result in ("allowed", "denied")`, so a
typo — `allow` — was SILENTLY DROPPED and the caller got both outcomes with a `200`. Not "no
validation": a wrong answer with no way to notice.

L4: `since` / `until` bound the walk in SQL, against the same `ts` the index orders by. Without
them the only way to reach an old event on an append-only log was to page through everything
newer than it.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from mantle.db import audit as db_audit
from mantle.db import lattice_api as store

def _executable(src: str) -> str:
    """`src` with comment lines removed.

    Matching raw text flags the very comment that argues against the defect. A check that
    forbids a literal, run over a file that must EXPLAIN why that literal was removed, fires on
    its own rationale — and the rationale is then deleted to reach green, which is the worst
    possible trade. This bit four times in one day before the technique was borrowed from
    `agience-cloud/tests/test_status_json_and_table_agree.py`, which carries the same note for
    the same reason.

    Comment LINES only. A docstring is code and stays — if a forbidden literal belongs in
    one, say it without quoting it."""
    return chr(10).join(ln for ln in src.splitlines()
                        if not ln.lstrip().startswith("#"))





@pytest.fixture
def db():
    return store.LatticeDatabase(os.path.join(tempfile.mkdtemp(), "al.db"), origin="al-test")


def test_the_time_bounds_narrow_the_query_in_sql():
    """L4, asserted on the SQL the function builds — the bound must be part of the WHERE, not a
    filter applied to rows already fetched, or it narrows nothing."""
    import inspect

    src = inspect.getsource(db_audit.access_log_of)
    assert "AND ts >= ?" in src and "AND ts <= ?" in src, src[-500:]
    assert src.index("AND ts >= ?") < src.index("ORDER BY ts DESC LIMIT"), (
        "the time bound must be applied before LIMIT, or it filters a page instead of the log")


def test_result_is_an_enum_so_a_typo_cannot_be_dropped():
    """L1. The query tests membership and ignores anything else, so validation has to happen at
    the boundary or a typo silently widens the answer."""
    from typing import get_args, get_type_hints

    import mantle.routers.artifacts_router as ar

    hints = get_type_hints(ar.get_artifact_access_log)
    allowed = set()
    for arg in get_args(hints["result"]):
        allowed |= set(get_args(arg))
    assert allowed == {"allowed", "denied"}, allowed


def test_the_route_reports_an_unknown_total_rather_than_the_page_length():
    """L2. `total` must not be the size of the page the caller is holding."""
    import ast
    import io as _io

    import mantle.routers.artifacts_router as ar

    tree = ast.parse(_io.open(ar.__file__, encoding="utf-8").read())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "get_artifact_access_log":
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_page":
                kw = {k.arg: k.value for k in call.keywords}
                assert isinstance(kw["total"], ast.Constant) and kw["total"].value is None, (
                    "`total` is not None — it is %s, and on this route the only cheap number "
                    "available is the page length, which is what the defect was"
                    % ast.dump(kw["total"])[:60])
                assert not (isinstance(kw["has_more"], ast.Constant)
                            and kw["has_more"].value is False), (
                    "`has_more` is hardcoded False again")
                return
    raise AssertionError("the access-log handler no longer calls _page — check this test")


# ---------------------------------------------------------------------------
# : the non-membership 404 said "Artifact not found" about an artifact that exists.
# (M2 + M4 removed the INVERSION itself on 2026-08-26 - the route is now
#  DELETE /artifacts/{artifact_id}/children/{child_id}, container first and bodyless.)
# ---------------------------------------------------------------------------


def test_removing_a_non_member_says_it_is_not_a_member():
    """M3. The message was "Artifact not found", which is wrong about the common case: the
    artifact usually EXISTS and simply is not in this container, so a caller could read "gone"
    and act on it — deleting a local copy of something it merely could not detach."""
    import inspect

    from mantle.services import workspace_service as ws

    src = _executable(inspect.getsource(ws.remove_artifact_from_container))
    assert "not a member of this container" in src, (
        "the non-membership 404 no longer distinguishes itself from a missing artifact")
    assert "Artifact not found" not in src, (
        "the misleading message is back")


def test_the_remove_route_declares_which_404_it_means():
    """The spec half of M3 — a `detail` a client never reads is not a contract."""
    from mantle.main import app as _app

    op = _app.openapi()["paths"]["/artifacts/{artifact_id}/children/{child_id}"]["delete"]
    desc = op["responses"]["404"]["description"]
    assert "NOT A MEMBER" in desc, desc
    # and it must still carry the indistinguishability sentence every 404 here owes
    assert "indistinguishable" in desc and "not permitted" in desc, desc
