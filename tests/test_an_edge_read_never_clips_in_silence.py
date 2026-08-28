"""A clipped edge list must not be indistinguishable from a complete one.

`edges_of` was the one read in `db/edge.py` without the cap discipline every other surface there
has. It took `limit: int = 1000`, applied a bare `LIMIT`, had no `ORDER BY`, and returned a plain
list — so a caller holding 1000 rows could not tell whether that was the answer or the ceiling.
Eight in-tree callers wanted every edge of a node.

Three consequences of a silent clip:

  * `lattice_api.list_origin_descendants` walks containment to build the **authorization light
    cone**, so past 1000 edges the cone was silently and non-deterministically incomplete — and
    that set is handed to `OracleService` to decide key custody.
  * `count_children` returned exactly 1000 for every container larger than that.
  * `remove_artifact_from_collection` returned `True` having removed nothing, because
    `_find_membership` looked for its edge in a list the edge had been clipped out of.

The fix is this store's existing discipline applied to one more read — `ListIndexUnbuilt` rather
than `[]`, `Page.truncated`, `edge_mark`'s `exhaustive`: refuse where degradation would be
invisible. A caller that genuinely wants a bounded peek says `partial_ok=True`.

The second half of this file is about order. `ORDER BY edge_key` is not tidiness: `edge_key =
blake2b(src ‖ dst ‖ label)` is node-invariant, so two nodes holding the same edges walk them in
the same order — and `get_relationship_target` returns "the first outbound edge with this label",
which without the ordering is whichever row SQLite happens to visit first.
"""
from __future__ import annotations

import pytest

from mantle.db import open_lattice
from mantle.db.edge import EDGES_OF_LIMIT, EdgesTruncated


ORIGIN = "test-origin"


def _fresh(tmp_path, name="lat.db"):
    return open_lattice(str(tmp_path / name), origin=ORIGIN, leaves=16)


def _members(L, collection_id, n, *, start=0):
    """`n` origin containment edges out of one container."""
    L.graph.add_edges([
        (collection_id, "art-%05d" % i, "contains",
         {"is_origin": True, "order_key": "a%05d" % i})
        for i in range(start, start + n)
    ])


class _DB:
    """The `db` shape `lattice_api` wants, over a bare lattice."""

    def __init__(self, lat):
        self.artifacts, self.graph, self.conn = lat.artifacts, lat.graph, lat.db


# ── refusing rather than clipping ─────────────────────────────────────────────────────────────


def test_more_edges_than_the_limit_raises_instead_of_clipping(tmp_path):
    """The whole point. Ten rows behind a limit of four is not four rows."""
    L = _fresh(tmp_path)
    _members(L, "col-1", 10)
    with pytest.raises(EdgesTruncated):
        L.graph.edges_of("col-1", direction="out", limit=4)


def test_exactly_the_limit_is_a_complete_answer(tmp_path):
    """The off-by-one that makes the difference detectable at all.

    `len(rows) == limit` is the ambiguous case — it is what both a complete answer and a clipped
    one look like — so the read fetches `limit + 1` and decides on the extra row. Ten rows behind a
    limit of ten must NOT raise.
    """
    L = _fresh(tmp_path)
    _members(L, "col-1", 10)
    assert len(L.graph.edges_of("col-1", direction="out", limit=10)) == 10


def test_a_deliberate_peek_says_so_and_gets_its_page(tmp_path):
    """`partial_ok=True` is the opt-in for callers whose question a bounded read answers —
    `has_children` asks for one row to answer yes/no, and a truncated list is what it wanted."""
    L = _fresh(tmp_path)
    _members(L, "col-1", 10)
    peek = L.graph.edges_of("col-1", direction="out", limit=3, partial_ok=True)
    assert len(peek) == 3, "a peek must return exactly the page it asked for, not the extra row"


def test_has_children_is_still_a_one_row_question(tmp_path):
    """It reads with `limit=1`, which is a truncating read by construction — so it must be the
    flagged kind, or asking whether a container has members would raise on every container that
    has more than one."""
    from mantle.db.lattice_api import has_children

    L = _fresh(tmp_path)
    db = _DB(L)
    assert has_children(db, "col-1") is False
    _members(L, "col-1", 50)
    assert has_children(db, "col-1") is True


def test_the_default_limit_is_a_memory_backstop_not_a_page_size(tmp_path):
    """A default of 1000 was reachable by an ordinary container. The point of the new default is
    that no real read hits it, so the refusal above never fires in normal operation — it is the
    bound on how much one read may materialize, never on how much of the graph a caller may see."""
    assert EDGES_OF_LIMIT >= 1_000_000
    L = _fresh(tmp_path)
    _members(L, "col-1", 1200)
    assert len(L.graph.edges_of("col-1", direction="out")) == 1200, (
        "1200 members is not an exotic container and must read without a limit argument"
    )


# ── the three measured consequences ──────────────────────────────────────────────────────────


def test_count_children_is_exact_past_the_old_cap(tmp_path):
    """It returned exactly 1000 for every container larger than 1000 — a number with no
    justification, which is the measurement discipline this store holds itself to."""
    from mantle.db.lattice_api import count_children

    L = _fresh(tmp_path)
    _members(L, "col-1", 1200)
    assert count_children(_DB(L), "col-1") == 1200


def test_removing_a_member_past_the_old_cap_actually_removes_it(tmp_path):
    """`remove_artifact_from_collection` reported success having removed nothing.

    `_find_membership` scans the container's edges for the pair; past the cap the edge it wanted
    was not in the list it was handed, so it concluded "already absent" — the idempotent answer —
    and returned True. Success that removed nothing is the worst of the three failures here,
    because the caller has been told the opposite of what happened.
    """
    from mantle.db.lattice_api import remove_artifact_from_collection

    L = _fresh(tmp_path)
    _members(L, "col-1", 1200)
    target = "art-01150"                                   # comfortably past the old 1000
    assert remove_artifact_from_collection(_DB(L), "col-1", target) is True
    remaining = {e["dst"] for e in L.graph.edges_of("col-1", direction="out")}
    assert target not in remaining, "reported success and removed nothing"
    assert len(remaining) == 1199


def test_the_light_cone_refuses_rather_than_under_reaching(tmp_path):
    """`list_origin_descendants` does not swallow truncation.

    Its `except Exception: continue` suits an unreadable edge row, where dropping a node
    under-reaches and is fail-closed. Truncation is a different fact — the edges are readable and
    there are more of them — and continuing past it produces an authorization answer quietly
    different from the one the graph supports.

    Forced with a raising stub rather than by building a million-edge container: what is under test
    is the handler, and the handler cannot tell where the exception came from.
    """
    from mantle.db import lattice_api

    L = _fresh(tmp_path)
    db = _DB(L)
    _members(L, "col-1", 3)
    assert lattice_api.list_origin_descendants(db, ["col-1"], "read")

    class _Truncating:
        def __getattr__(self, name):
            return getattr(L.graph, name)

        def edges_of(self, *a, **kw):
            raise EdgesTruncated("forced")

    db.graph = _Truncating()
    with pytest.raises(EdgesTruncated):
        lattice_api.list_origin_descendants(db, ["col-1"], "read")


@pytest.mark.parametrize("fn_name,args", [
    ("get_origin_parent", ("art-1",)),
    ("get_relationship_target", ("art-1", "lineage")),
    ("remove_all_edges_for_root", ("art-1",)),
    ("get_collection_ids_for_root", ("art-1",)),
])
def test_no_reader_absorbs_truncation_into_a_negative_answer(tmp_path, fn_name, args):
    """Every `except Exception` around an edge read in `lattice_api` re-raises this first.

    Each of these deliberately absorbs a failed edge read and returns a negative answer — `None`
    for "already a root", `[]` for "in no collection", a count for "this is what I removed". Those
    readings are right for a row that will not open and wrong for a read that stopped early, and
    the second one is the case where the negative answer is a lie rather than a caution.
    """
    from mantle.db import lattice_api

    L = _fresh(tmp_path)
    db = _DB(L)

    class _Truncating:
        def __getattr__(self, name):
            return getattr(L.graph, name)

        def edges_of(self, *a, **kw):
            raise EdgesTruncated("forced")

    db.graph = _Truncating()
    with pytest.raises(EdgesTruncated):
        getattr(lattice_api, fn_name)(db, *args)


def test_the_context_reader_does_not_absorb_it_either(tmp_path):
    """`context_service` states the fail-closed rule explicitly — "an unreadable edge withholds
    reach, it never grants it" — and that rule makes truncation the exception to itself: yielding
    `[]` would withhold reach the graph does grant."""
    from mantle.services import context_service

    L = _fresh(tmp_path)
    db = _DB(L)

    class _Truncating:
        def context_edges(self, *a, **kw):
            raise EdgesTruncated("forced")

    db.graph = _Truncating()
    with pytest.raises(EdgesTruncated):
        context_service._edges(db, "art-1", direction="in")


# ── deterministic, node-invariant order ──────────────────────────────────────────────────────


def test_the_order_does_not_depend_on_insertion_order(tmp_path):
    """Two stores holding the same edges must walk them identically.

    `ORDER BY edge_key` and `edge_key = blake2b(src ‖ dst ‖ label)` — derived from the ids, so it
    is the same value on every node in the mesh. Without an `ORDER BY` this was whatever order
    SQLite chose, which is stable enough to look deterministic in a test and is not a property
    anything may rely on.
    """
    pairs = [("col-1", "art-%d" % i) for i in (7, 2, 9, 4, 1, 8, 3)]
    a = _fresh(tmp_path, name="a.db")
    b = _fresh(tmp_path, name="b.db")
    for src, dst in pairs:
        a.graph.add_edge(src, dst, "contains", {"is_origin": True})
    for src, dst in reversed(pairs):
        b.graph.add_edge(src, dst, "contains", {"is_origin": True})

    order_a = [e["dst"] for e in a.graph.edges_of("col-1", direction="out")]
    order_b = [e["dst"] for e in b.graph.edges_of("col-1", direction="out")]
    assert order_a == order_b, (
        "the same edge set read in two different insertion orders came back differently ordered"
    )
    assert sorted(order_a) == sorted(dst for _s, dst in pairs)


def test_the_first_relationship_target_is_a_defined_edge(tmp_path):
    """`get_relationship_target` returns "the first outbound edge with this label", so before the
    ordering existed two nodes holding identical edges could disagree about a typed relationship's
    target. Asserted across two stores for the same reason as above."""
    from mantle.db.lattice_api import get_relationship_target

    targets = ["t-3", "t-1", "t-2"]
    a = _fresh(tmp_path, name="a.db")
    b = _fresh(tmp_path, name="b.db")
    for t in targets:
        a.graph.add_edge("art-1", t, "lineage", {})
    for t in reversed(targets):
        b.graph.add_edge("art-1", t, "lineage", {})

    assert get_relationship_target(_DB(a), "art-1", "lineage") == \
        get_relationship_target(_DB(b), "art-1", "lineage")
