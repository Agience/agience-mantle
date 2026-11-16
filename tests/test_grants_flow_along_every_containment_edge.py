"""BUG-074 (John, 2026-11-14): a grant flows along every `contains` edge, not only the creation
edge.

One generic rule, read off the edge's stored mask and never off its kind: no mask propagates, a
list narrows, an empty list blocks. A deny on the node itself refuses; a deny above it cuts the
paths through it, and any other path still reaches.
"""
from __future__ import annotations

import pytest

from mantle.db import lattice_api as la
from mantle.search.mantle.lightcone import _reaches


@pytest.fixture
def db(tmp_path):
    return la.LatticeDatabase(str(tmp_path / "flow.db"), origin="flow")


def _shape(db):
    """Workspace A made X; X was then added to the shared workspace W."""
    la.add_artifact_to_collection(db, "A", "X", origin=True, propagate=None)
    la.add_artifact_to_collection(db, "X", "x-item", origin=True, propagate=None)
    la.add_artifact_to_collection(db, "W", "X", origin=False, propagate=None)


def test_a_grant_on_the_shared_workspace_reaches_a_collection_added_to_it(db):
    _shape(db)
    assert _reaches(db, "X", "read", {"W"}, set(), root_id="X")
    assert _reaches(db, "x-item", "read", {"W"}, set(), root_id="x-item")


def test_the_walk_yields_every_path_up_nearest_first(db):
    _shape(db)
    assert list(la.light_cone(db, "x-item", "read", root_id="x-item")) == ["x-item", "X", "A", "W"]


def test_a_blocked_edge_conducts_nothing(db):
    la.add_artifact_to_collection(db, "A", "X", origin=True, propagate=None)
    la.add_artifact_to_collection(db, "W", "X", origin=False, propagate=[])
    assert not _reaches(db, "X", "read", {"W"}, set(), root_id="X")


def test_a_narrowed_edge_conducts_only_what_it_names(db):
    la.add_artifact_to_collection(db, "W", "X", origin=False, propagate=["read"])
    assert _reaches(db, "X", "read", {"W"}, set(), root_id="X")
    assert not _reaches(db, "X", "update", {"W"}, set(), root_id="X")


def test_a_deny_above_cuts_its_paths_and_another_path_still_reaches(db):
    _shape(db)
    assert not _reaches(db, "X", "read", {"W"}, {"W"}, root_id="X")       # the one path, cut
    assert _reaches(db, "X", "read", {"A"}, {"W"}, root_id="X")           # through A instead


def test_a_deny_on_the_node_itself_refuses_every_path(db):
    _shape(db)
    assert not _reaches(db, "X", "read", {"A", "W"}, {"X"}, root_id="X")


def test_a_cycle_closes_and_reaches_nothing_it_should_not(db):
    la.add_artifact_to_collection(db, "P", "Q", origin=False, propagate=None)
    la.add_artifact_to_collection(db, "Q", "P", origin=False, propagate=None)
    assert not _reaches(db, "P", "read", {"Z"}, set(), root_id="P")
    assert _reaches(db, "P", "read", {"Q"}, set(), root_id="P")
