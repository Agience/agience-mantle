"""A recall hit carries the store fields a caller needs to order hits itself.

Mantle reads; the Search artifact, run by Crystal, ranks. So each hit names who created it, when
it last changed and which query stems it matched, on every ordering that had query terms,
recency included.
"""
from __future__ import annotations

import pytest

from mantle.main import app
from mantle.search.mantle.lightcone import AuthorizedScope
from mantle.search.mantle.sse.narrowing import Coverage
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor


def _scope(ids, stamps):
    return AuthorizedScope(contexts=[("p", "c")], artifact_ids=frozenset(ids), updated_at=stamps)


def test_recency_rows_carry_what_each_matched():
    coverage = {"a": Coverage(1, 0, ("plan",)), "b": Coverage(2, 0, ("plan", "trip"))}
    rows = MantleSseSearchAccessor._by_recency(
        _scope(["a", "b", "c"], {"a": "2026-01", "b": "2026-02", "c": "2026-03"}), coverage,
    )
    assert [r.artifact_id for r in rows] == ["c", "b", "a"]
    assert [r.coverage.matched if r.coverage else None for r in rows] == [
        None, ("plan", "trip"), ("plan",)]
    assert all(r.score is None for r in rows)


def test_recency_without_terms_still_answers():
    rows = MantleSseSearchAccessor._by_recency(_scope(["a"], {}))
    assert [(r.artifact_id, r.coverage) for r in rows] == [("a", None)]


@pytest.fixture(scope="module")
def hit_props():
    return app.openapi()["components"]["schemas"]["RecallHitResponse"]["properties"]


def test_the_hit_declares_the_store_fields(hit_props):
    assert {"created_by", "modified_time", "matched"} <= set(hit_props)


def test_the_request_names_no_rank():
    """The rank keys live in the Search artifact."""
    props = app.openapi()["components"]["schemas"]["ArtifactRecallRequest"]["properties"]
    assert "rank" not in props and "search_id" not in props
