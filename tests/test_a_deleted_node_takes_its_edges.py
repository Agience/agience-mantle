"""An edge must not outlive the node it names, because `edge_key` is derived from the id.

`edge_key = blake2b(src ‖ \\0 ‖ dst ‖ \\0 ‖ label, 16)` — the id STRINGS and nothing else. That is
the property that makes replay idempotent, and it is also what makes a dangling edge dangerous
rather than merely untidy: an edge left behind after its node is deleted is a live row that the
next artifact created with the same id inherits whole, `is_origin=1` and `propagate` mask
included.

A `db/vertex.py:delete_artifact` that removes the vertex row, its tasks, its listkeys and its
demand entry but no edges hands a delete-then-recreate the old artifact's position in the
containment graph, which is precisely the graph
`lattice_api.list_origin_descendants` walks to answer "who holds authority over this subtree". The
grant ledger was never consulted and never disagreed; the authority simply reappeared.

Deleting the edges is not the only thing that has to be true. A deletion in this store owes four
accountings per row — the merkle XOR-out, the edge total, the per-label extent, and the
proper-time vacate — and a bulk delete that skipped any of them would trade a visible
authorization bug for an invisible one in the store's own invariants. So this file asserts the
authority property AND that the store still balances afterwards.
"""
from __future__ import annotations

from mantle.db import open_lattice
from mantle.db.seq import seq_accounting
from mantle.db.schema import c_edge_total


ORIGIN = "test-origin"


def _fresh(tmp_path, name="lat.db"):
    return open_lattice(str(tmp_path / name), origin=ORIGIN, leaves=16)


def _doc(aid, **kw):
    d = {"id": aid, "content_type": "text/plain", "name": aid}
    d.update(kw)
    return d


def _member_edge(L, collection_id, member_id, *, propagate=None):
    """The edge that confers authority: origin containment, collection → member."""
    L.graph.add_edge(collection_id, member_id, "contains",
                     {"is_origin": True, "propagate": propagate, "order_key": "a0"})


# ── the authority property ───────────────────────────────────────────────────────────────────


def test_deleting_an_artifact_removes_the_edges_on_both_sides(tmp_path):
    """Both sides, and the inbound one is the one that matters.

    Authority descends along the arrow, so it is the INBOUND `contains` edge — the one naming this
    artifact as a member of someone's collection — that confers reach over it. A fix that removed
    only the outbound half would leave exactly the dangerous half behind, which is why the
    assertion names both directions instead of counting rows.
    """
    L = _fresh(tmp_path)
    L.artifacts.put_artifact(_doc("col-1"))
    L.artifacts.put_artifact(_doc("art-1"))
    L.artifacts.put_artifact(_doc("art-child"))
    _member_edge(L, "col-1", "art-1")               # inbound to art-1: confers authority
    _member_edge(L, "art-1", "art-child")           # outbound from art-1

    assert L.graph.edges_of("art-1", direction="in")
    assert L.graph.edges_of("art-1", direction="out")

    L.artifacts.delete_artifact("art-1")

    assert L.graph.edges_of("art-1", direction="in") == [], (
        "the inbound containment edge survived the delete — the next artifact created with this "
        "id inherits membership in col-1, and with it everything col-1's grants reach"
    )
    assert L.graph.edges_of("art-1", direction="out") == []


def test_a_recreated_id_does_not_inherit_the_deleted_node_s_authority(tmp_path):
    """THE RESURRECTION, end to end and through the authorization walk itself.

    Asserted through `list_origin_descendants` rather than by counting edges, because that is the
    function whose answer decides key custody: `lightcone.resolve` seeds from it, and the set it
    returns is handed to `OracleService` to derive content keys per (principal, collection). An
    artifact that reappears inside `col-1` does not merely look wrong in a listing — it receives a
    key.
    """
    from mantle.db.lattice_api import list_origin_descendants

    class _DB:
        def __init__(self, lat):
            self.artifacts, self.graph, self.conn = lat.artifacts, lat.graph, lat.db

    L = _fresh(tmp_path)
    db = _DB(L)
    L.artifacts.put_artifact(_doc("col-1"))
    L.artifacts.put_artifact(_doc("art-1"))
    _member_edge(L, "col-1", "art-1")
    assert list_origin_descendants(db, ["col-1"], "read") == {"art-1"}

    L.artifacts.delete_artifact("art-1")
    assert list_origin_descendants(db, ["col-1"], "read") == set(), (
        "the containment edge outlived the artifact, so the light cone still reaches an id that "
        "does not exist — and will reach whatever is created there next"
    )

    # The same id again, created by someone else, naming no collection.
    L.artifacts.put_artifact(_doc("art-1", name="a different artifact entirely"))
    assert list_origin_descendants(db, ["col-1"], "read") == set(), (
        "a same-id create inherited the deleted artifact's membership in col-1: authority "
        "resurrected from a dangling edge, with no grant anywhere saying so"
    )


def test_a_propagating_origin_edge_is_the_one_that_gets_removed(tmp_path):
    """The dangerous edge is specifically `is_origin=1` with a propagating mask, so it gets its
    own test — a fix that happened to drop only non-propagating edges would pass the tests above
    on a corpus that used the default mask."""
    L = _fresh(tmp_path)
    L.artifacts.put_artifact(_doc("col-1"))
    L.artifacts.put_artifact(_doc("art-1"))
    _member_edge(L, "col-1", "art-1", propagate=None)      # NULL == propagates everything
    edge = L.graph.edges_of("col-1", direction="out")[0]
    assert edge["is_origin"] == 1 and edge["propagate"] is None

    L.artifacts.delete_artifact("art-1")
    assert L.graph.edges_of("col-1", direction="out") == []


# ── and the store still balances ─────────────────────────────────────────────────────────────


def test_the_allocation_accounting_still_balances_after_the_edges_go(tmp_path):
    """`live_rows + vacated == last_seq`, per origin, over vertex ∪ edge.

    Every accounted removal must increment `vacated`, so retiring N edges inside the vertex
    delete has to vacate N times. A bulk `DELETE ... WHERE src=? OR dst=?` would have removed the
    rows without any of that, trading a visible authorization bug for an invisible one in the
    invariant this store checks itself against.
    """
    L = _fresh(tmp_path)
    for aid in ("col-1", "art-1", "art-2"):
        L.artifacts.put_artifact(_doc(aid))
    _member_edge(L, "col-1", "art-1")
    _member_edge(L, "col-1", "art-2")
    _member_edge(L, "art-1", "art-2")

    L.artifacts.delete_artifact("art-1")

    acc = seq_accounting(L.db, ORIGIN)
    assert acc["balanced"], (
        f"retiring edges with the vertex broke the allocation invariant: {acc}"
    )
    assert acc["unaccounted"] == 0


def test_the_edge_total_falls_by_exactly_what_was_removed(tmp_path):
    """The counter is what every extent read uses — `count(*)` is unservable on this schema — so a
    retirement that forgot to decrement it would report a corpus that only ever grows."""
    from mantle.db.seq import counter_of

    L = _fresh(tmp_path)
    for aid in ("col-1", "art-1", "art-2"):
        L.artifacts.put_artifact(_doc(aid))
    _member_edge(L, "col-1", "art-1")
    _member_edge(L, "col-1", "art-2")
    _member_edge(L, "art-1", "art-2")
    assert counter_of(L.db, c_edge_total()) == 3

    L.artifacts.delete_artifact("art-1")
    assert counter_of(L.db, c_edge_total()) == 1, (
        "two edges touched art-1 and one did not; the total must fall by exactly two"
    )


def test_a_self_edge_is_retired_once_not_twice(tmp_path):
    """A self-edge matches both `src = ?` and `dst = ?`.

    Retiring it twice would double the vacate and the counter decrement — the invariant would
    then be broken by a row that WAS correctly deleted, which is the hardest kind of accounting
    bug to find later.
    """
    from mantle.db.seq import counter_of

    L = _fresh(tmp_path)
    L.artifacts.put_artifact(_doc("art-1"))
    L.graph.add_edge("art-1", "art-1", "lineage", {"is_origin": False})
    assert counter_of(L.db, c_edge_total()) == 1

    L.artifacts.delete_artifact("art-1")
    assert counter_of(L.db, c_edge_total()) == 0
    acc = seq_accounting(L.db, ORIGIN)
    assert acc["balanced"] and acc["unaccounted"] == 0, (
        f"a self-edge was retired twice: {acc}"
    )


def test_the_merkle_contribution_is_xored_back_out(tmp_path):
    """Deleting the edges must return the digest to what it was before they existed.

    XOR is its own inverse, which is what makes the incremental Merkle work at all — but only if
    the XOR-out reconstructs byte-for-byte what was XOR-ed in. `_edge_row_content` is shared
    between the two directions for exactly this reason; a divergence would strand the
    contribution in the leaf permanently and no two nodes could agree on a root again.
    """
    L = _fresh(tmp_path)
    L.artifacts.put_artifact(_doc("col-1"))
    L.artifacts.put_artifact(_doc("art-1"))
    before = L.artifacts.merkle_leaves()

    _member_edge(L, "col-1", "art-1", propagate='["read"]')
    assert L.artifacts.merkle_leaves() != before, "the edge should have moved a leaf"

    L.artifacts.delete_artifact("art-1")
    L.artifacts.delete_artifact("col-1")
    # Both vertices are gone too, so compare against an empty store's leaves rather than `before`.
    empty = _fresh(tmp_path, name="empty.db").artifacts.merkle_leaves()
    assert L.artifacts.merkle_leaves() == empty, (
        "a deleted edge left its contribution behind in the merkle tree"
    )


# ── the documented exception ─────────────────────────────────────────────────────────────────


def test_evicting_a_cached_copy_leaves_its_edges_alone(tmp_path):
    """Eviction is not deletion, and the difference is load-bearing here.

    `evict_artifact` drops a cached copy of a row this node did not author and that still exists
    at its author; a later reach restores it. Edges are separately replicated objects with their
    own feed — `mesh/sync` consumes them through `graph.add_edges`, not as a property of the
    vertex — so taking them here would delete objects the eviction sweep never selected and that
    the reach would not bring back.

    Stated as a test rather than only as a comment because it is the one case where leaving an
    edge without its vertex is correct, and a later reader tightening the rule above would
    otherwise have nothing to tell them why not.
    """
    L = _fresh(tmp_path)
    L.artifacts.put_artifact(_doc("col-1"))
    L.artifacts.put_artifact(_doc("art-1"))
    _member_edge(L, "col-1", "art-1")

    L.artifacts.evict_artifact("art-1")

    assert L.artifacts.get_artifact("art-1") is None, "the cached row should be gone"
    assert L.graph.edges_of("art-1", direction="in"), (
        "eviction removed a replicated edge the authoritative node still holds"
    )
    acc = seq_accounting(L.db, ORIGIN)
    assert acc["balanced"], f"eviction unbalanced the accounting: {acc}"


def test_deleting_an_absent_id_is_still_a_no_op(tmp_path):
    """Idempotent, and it must not start retiring edges for an id that has none. A delete of
    something absent used to clear a stray demand entry and return; that stays true."""
    L = _fresh(tmp_path)
    L.artifacts.delete_artifact("never-existed")
    acc = seq_accounting(L.db, ORIGIN)
    assert acc["balanced"] and acc["last_seq"] == 0


def test_the_graph_store_exposes_the_same_retirement(tmp_path):
    """`delete_edges_touching` is the graph store's face on the same accounting, so a caller
    holding a graph store does not have to reach for the module function or write its own."""
    L = _fresh(tmp_path)
    L.artifacts.put_artifact(_doc("col-1"))
    L.artifacts.put_artifact(_doc("art-1"))
    _member_edge(L, "col-1", "art-1")
    _member_edge(L, "art-1", "art-2")

    assert L.graph.delete_edges_touching("art-1") == 2
    assert L.graph.edges_of("art-1", direction="in") == []
    assert L.graph.edges_of("art-1", direction="out") == []
    assert L.graph.delete_edges_touching("art-1") == 0, "idempotent"
    acc = seq_accounting(L.db, ORIGIN)
    assert acc["balanced"], f"{acc}"
