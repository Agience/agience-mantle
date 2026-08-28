""": `GET /artifacts/{id}/commits` published a changeset in which nothing ever changed.

THE DEFECT. `adds` and `removes` are not fields of `Commit` — the changed version-ids live in
the `CommitItem`s that `item_ids` points at. The route read them as `getattr(c, "adds", [])`, so
every commit ever listed reported `adds: []` and `removes: []`. Measured 2026-08-26: a commit
recorded with two adds and one remove read back empty on both.

Nothing in this tree had ever read a `CommitItem` back. `record_collection_commit` wrote them
and every reader stopped at `item_ids`, so the data was written correctly and never surfaced.

The `getattr` default is what made it silent, which is the general lesson: a default on a field
that MUST exist does not protect against a defect, it converts one into an empty answer.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from mantle.db import lattice_api as store
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE
from mantle.services import collection_service as cs
from mantle.services import workspace_service as ws

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
    return store.LatticeDatabase(os.path.join(tempfile.mkdtemp(), "c.db"), origin="commit-test")


@pytest.fixture
def container(db):
    return ws.create_container(db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="c").id


def test_a_commit_reports_the_ids_it_added_and_removed(db, container):
    cs.record_collection_commit(
        db, "u", container, adds=["art-1", "art-2"], removes=["art-3"], message="m")
    commits = cs.get_commits_for_collection(db, "u", container)
    assert len(commits) == 1, commits
    c = commits[0]
    assert sorted(c.adds) == ["art-1", "art-2"], c.adds
    assert sorted(c.removes) == ["art-3"], c.removes


def test_an_add_only_commit_has_an_empty_removes_and_that_is_real(db, container):
    """The inverted case — `[]` must still be reachable, or the test above proves nothing."""
    cs.record_collection_commit(db, "u", container, adds=["only-add"], message="m")
    c = cs.get_commits_for_collection(db, "u", container)[0]
    assert sorted(c.adds) == ["only-add"]
    assert c.removes == []


def test_the_changeset_is_scoped_to_the_collection_asked_about(db):
    """A commit spanning two containers reports only what moved in the one being read."""
    a = ws.create_container(db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="a").id
    b = ws.create_container(db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="b").id
    cs.record_collection_commit(db, "u", a, adds=["in-a"], message="m")
    cs.record_collection_commit(db, "u", b, adds=["in-b"], message="m")

    assert [x for c in cs.get_commits_for_collection(db, "u", a) for x in c.adds] == ["in-a"]
    assert [x for c in cs.get_commits_for_collection(db, "u", b) for x in c.adds] == ["in-b"]


def test_the_route_no_longer_defaults_the_fields_that_must_exist():
    """K4's actual complaint. A default on a field that must exist turns a rename into `null`."""
    import io as _io

    from mantle.routers import artifacts_router

    src = _executable(_io.open(artifacts_router.__file__, encoding="utf-8").read())
    for gone in ('getattr(c, "adds", [])', 'getattr(c, "removes", [])',
                 'getattr(c, "id", None)', 'getattr(c, "message", None)'):
        assert gone not in src, (
            "%s is back — a default here reports an empty changeset instead of failing" % gone)


def test_the_commit_history_pages():
    """. Every other list on this surface pages; this one returned the whole history and
    hardcoded `has_more: False` — true, and useless as a bound.

    Paging bounds the RESPONSE, not the work: the store still scans every commit-item doc to
    decide which touch this container, and that is recorded in the handler rather than implied.

    Asserted against the published spec rather than the handler's source — a source-text check
    passes or fails on how the code is spelled, which is not the contract."""
    from mantle.main import app

    op = app.openapi()["paths"]["/artifacts/{artifact_id}/commits"]["get"]
    params = {p["name"]: p for p in op.get("parameters", [])}
    for name in ("limit", "offset"):
        assert name in params, "commits takes no `%s`: %s" % (name, sorted(params))
        assert params[name].get("description"), "`%s` is undescribed" % name
