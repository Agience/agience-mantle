"""The MANTLE cell-key principal is the collection's immutable origin root.

Covers :func:`db.backend.get_origin_root` (walk the origin chain to the top,
cycle/depth-guarded) and :func:`search.mantle.principal.resolve_cell_principal`
(the index path and the query path resolve the SAME principal for the same
collection, so the derived keys match). There is no "owner" / ``created_by`` in
the crypto principal.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from mantle.db import backend as db_store
from mantle.search.mantle.principal import CellPrincipalUnresolved, resolve_cell_principal


# ---------------------------------------------------------------------------
# get_origin_root — walk the immutable origin chain to the top
# ---------------------------------------------------------------------------

def test_get_origin_root_walks_to_top():
    # c -> b -> a; a is the root (no origin parent).
    chain = {"c": ("b", None), "b": ("a", None), "a": None}
    with patch("mantle.db.lattice_api.get_origin_parent", side_effect=lambda db, rid: chain.get(rid)):
        assert db_store.get_origin_root(None, "c") == "a"
        assert db_store.get_origin_root(None, "b") == "a"


def test_get_origin_root_self_when_no_parent():
    with patch("mantle.db.lattice_api.get_origin_parent", return_value=None):
        assert db_store.get_origin_root(None, "solo") == "solo"


def test_get_origin_root_cycle_guarded():
    # x -> y -> x: a malformed cycle must not loop forever.
    chain = {"x": ("y", None), "y": ("x", None)}
    with patch("mantle.db.lattice_api.get_origin_parent", side_effect=lambda db, rid: chain.get(rid)):
        assert db_store.get_origin_root(None, "x") in {"x", "y"}


def test_get_origin_root_walks_past_the_old_depth_cap():
    """The walk has no depth limit — only a cycle guard (`visited`) — so a legitimate chain of any
    length resolves to its actual root rather than being cut off early."""
    chain = {str(n): str(n + 1) for n in range(100)}

    def parent(db, rid):
        nxt = chain.get(rid)
        return (nxt, None) if nxt else None

    with patch("mantle.db.lattice_api.get_origin_parent", side_effect=parent):
        assert db_store.get_origin_root(None, "0") == "100"


def test_get_origin_root_terminates_on_a_cycle():
    """`visited` is the real termination guard, not a depth number."""
    loop = {"a": "b", "b": "c", "c": "a"}

    def parent(db, rid):
        nxt = loop.get(rid)
        return (nxt, None) if nxt else None

    with patch("mantle.db.lattice_api.get_origin_parent", side_effect=parent):
        assert db_store.get_origin_root(None, "a") == "c"


# ---------------------------------------------------------------------------
# resolve_cell_principal — the shared index/query resolution
# ---------------------------------------------------------------------------

def test_resolve_is_origin_root():
    with patch.object(db_store, "get_origin_root", return_value="root-1"):
        assert resolve_cell_principal(None, "col-9") == "root-1"


def test_resolve_empty_collection_is_empty():
    assert resolve_cell_principal(None, "") == ""


def test_resolve_raises_rather_than_substituting_a_different_principal():
    """A failed lookup must raise, not fall back to ``collection_id``.

    A fallback would only engage while the lookup is failing, so index time and query time could
    disagree: cells get written under one key and sought under another. The corpus stays intact,
    the search finds nothing, and every metric reads healthy — indistinguishable from a delete
    that reports success while removing nothing.
    """
    with patch.object(db_store, "get_origin_root", side_effect=RuntimeError("boom")):
        with pytest.raises(CellPrincipalUnresolved):
            resolve_cell_principal(None, "col-9")


def test_resolve_self_rooted_collection_is_not_an_error():
    """The other side: no ancestors is a real answer, not a failure.

    ``get_origin_root`` returning falsy means the collection is its own origin
    root. This must keep resolving to the collection id — a top-level collection
    must not become an exception.
    """
    with patch.object(db_store, "get_origin_root", return_value=None):
        assert resolve_cell_principal(None, "col-top") == "col-top"


def test_index_and_query_resolve_the_same_principal():
    # The load-bearing invariant: the same collection_id resolves to the same
    # principal at index time and at query time, so the cell key matches.
    with patch.object(db_store, "get_origin_root", return_value="root-7"):
        principal_at_index = resolve_cell_principal(None, "col-A")
        principal_at_query = resolve_cell_principal(None, "col-A")
        assert principal_at_index == principal_at_query == "root-7"


def test_principal_is_not_created_by():
    # Two artifacts in the same collection created by different people must land
    # under the SAME principal (the collection's origin root) — never created_by.
    with patch.object(db_store, "get_origin_root", return_value="col-root"):
        # Same collection regardless of who created the artifact.
        assert resolve_cell_principal(None, "shared-col") == "col-root"
