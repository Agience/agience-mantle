"""The `materialized` marker is stamped when the index job RUNS, never when it is enqueued.

The defect, and it already happened. Three call sites in `services/workspace_service.py` stamped
the marker as soon as `enqueue_index_artifact` RETURNED. With a queue configured that returns
immediately — the job runs later, on a worker thread. So the marker meant *"this was queued"* while
its only reader treats it as *"this is indexed"*, and reads it as a SKIP condition.

A job that never ran therefore left an artifact **stamped done, carrying no postings, and never
re-enqueued**: stored and unfindable, with nothing reporting it. `mantle` was stopped and restarted
several times on `71/home` on 2026-08-25 — twice in an unplanned outage, once for the WAL
maintenance window, once when the supervisor moved under its scheduled task — so every job queued
at those moments was dropped.

The cost of the fix, stated because it is real: between the write and the job finishing,
`is_materialized` reads False, so a second write in that window enqueues a duplicate job. Indexing
overwrites, so that spends WORK. The arrangement it replaces spent CORRECTNESS, silently.
"""
from __future__ import annotations

import ast
import io
import os
from unittest.mock import patch

import pytest

from mantle.search.ingest import pipeline_unified as pu

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "mantle")


class _Art:
    def __init__(self, aid="a-1"):
        self.id = aid
        self.collection_id = "c-1"


# ── the ordering ─────────────────────────────────────────────────────────────────────────────

def test_the_marker_is_stamped_only_after_indexing_succeeds():
    marked: list[str] = []
    with patch.object(pu, "index_artifact", return_value=True), \
            patch.object(pu, "index_queue", None), \
            patch.object(pu, "_mark_indexed", side_effect=marked.append):
        pu.enqueue_index_artifact(_Art(), "c-1")
    assert marked == ["a-1"]


def test_a_failed_index_leaves_no_marker_so_the_work_is_retried():
    """The whole point. No marker ⇒ `is_materialized` stays False ⇒ the next access re-enqueues."""
    marked: list[str] = []
    with patch.object(pu, "index_artifact", return_value=False), \
            patch.object(pu, "index_queue", None), \
            patch.object(pu, "_mark_indexed", side_effect=marked.append):
        pu.enqueue_index_artifact(_Art(), "c-1")
    assert marked == [], "a failed index stamped the marker — the artifact is now unfindable"


def test_enqueueing_alone_stamps_nothing():
    """The defect itself, inverted: with a queue configured, `enqueue_index_artifact` returns
    before any work happens, and must leave no marker behind."""
    marked: list[str] = []
    queue = type("Q", (), {"enqueue": lambda self, fn, **kw: None})()
    with patch.object(pu, "index_queue", queue), \
            patch.object(pu, "index_artifact", return_value=True), \
            patch.object(pu, "_mark_indexed", side_effect=marked.append):
        pu.enqueue_index_artifact(_Art(), "c-1")
    assert marked == [], (
        "the marker was stamped at ENQUEUE time — it says work is done that has not started")


def test_the_batch_path_marks_every_artifact_it_indexed():
    marked: list[str] = []
    with patch.object(pu, "index_artifacts_batch", return_value=True), \
            patch.object(pu, "index_queue", None), \
            patch.object(pu, "_mark_indexed", side_effect=marked.append):
        pu.enqueue_index_artifacts_batch([_Art("a"), _Art("b")], "c-1")
    assert marked == ["a", "b"]


def test_a_failed_batch_marks_nothing():
    marked: list[str] = []
    with patch.object(pu, "index_artifacts_batch", return_value=False), \
            patch.object(pu, "index_queue", None), \
            patch.object(pu, "_mark_indexed", side_effect=marked.append):
        pu.enqueue_index_artifacts_batch([_Art("a")], "c-1")
    assert marked == []


def test_a_marker_that_cannot_be_written_does_not_fail_the_job():
    """`mark_materialized` is best-effort by contract: a missing marker costs a re-index, not
    correctness. It must not turn a successful index into a failed job."""
    with patch.object(pu, "index_artifact", return_value=True), \
            patch.object(pu, "index_queue", None), \
            patch("mantle.db.backend.store_handle", side_effect=RuntimeError("no store")):
        pu.enqueue_index_artifact(_Art(), "c-1")     # must not raise


# ── the guard ────────────────────────────────────────────────────────────────────────────────

def test_no_enqueue_site_stamps_the_marker_itself():
    """The three sites this was moved out of. If one comes back, the silent-unfindable failure
    comes back with it — and nothing else in the suite would notice, because the marker is only
    ever read as a reason to SKIP."""
    offenders = []
    for dirpath, _dirs, files in os.walk(_SRC):
        for fn in files:
            if not fn.endswith(".py") or fn.startswith("test_"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(io.open(path, encoding="utf-8").read())
            except SyntaxError:                                  # pragma: no cover
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                names = {c.func.attr if isinstance(c.func, ast.Attribute)
                         else getattr(c.func, "id", "")
                         for c in ast.walk(node) if isinstance(c, ast.Call)}
                if "mark_materialized" in names and "enqueue_index_artifact" in names:
                    offenders.append("%s:%s:%d" % (fn, node.name, node.lineno))
    assert not offenders, (
        "these stamp the materialization marker in the same function that ENQUEUES the work, "
        "which marks it done before it has happened: %r" % offenders)


def test_the_contract_docstring_no_longer_says_enqueue():
    """The docstring said *'called wherever indexing is enqueued'*. That was not a wording problem
    — it described, accurately, the thing that was wrong."""
    from mantle.db import lattice_api

    doc = lattice_api.mark_materialized.__doc__ or ""
    assert "called by the index JOB" in doc, doc[:200]
