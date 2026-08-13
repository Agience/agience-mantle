"""Unit tests for the lattice store.

Every test here corresponds to an explicit contract clause, not to coverage
for its own sake. Run:

    python -m pytest src/mantle/db/test_lattice.py -q
    python src/mantle/db/test_lattice.py          # stdlib-only fallback
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading

#: This file sits at `<repo>/src/mantle/db/`, so three levels up is `<repo>/src` — the entry that
#: makes the fully qualified `mantle.*` imports below resolve in an uninstalled checkout. Asserted
#: rather than trusted: `sys.path.insert` of a wrong or non-existent directory is a SILENT no-op,
#: and the ImportError that follows names a module, not the path arithmetic that caused it.
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
assert os.path.basename(_SRC) == "src" and os.path.isdir(os.path.join(_SRC, "mantle", "db")), (
    "path depth is wrong: expected <repo>/src, resolved %s — fix the depth" % _SRC)
sys.path.insert(0, _SRC)

from mantle.db import (NEWER, OLDER, SAME, UNORDERED, compare_version,  # noqa: E402
                               edge_key, open_lattice, row_hash)


#: ONE directory in the system temp for this whole module, with every store below getting its own
#: numbered subdirectory inside it. Each test still runs against a private, empty directory — that
#: is unchanged — but the system temp directory gains one entry per RUN instead of ~100.
#:
#: The distinction is not tidiness. On NTFS the cost of creating a directory entry grows with the
#: number of entries already present (8.3 short-name generation scans same-prefixed siblings for a
#: free alias), so a `mkdtemp` per test both paid that cost and made the next run pay more of it.
#: Measured on Windows once ~100k leftovers had accumulated: 0.345s per `mkdtemp` in the system
#: temp against 0.0007s in an empty parent.
#:
#: Built lazily rather than at import because this module re-executes as `__main__` in the
#: subprocesses `test_seq_gap_free_across_os_processes` spawns, and a child that only writes to a
#: path it was handed has no reason to create a directory of its own.
_TMP_BASE = None
_TMP_SEQ = 0


def _tmpdir(prefix: str = "s") -> str:
    """A fresh empty directory for one test, parented on this module's single base."""
    global _TMP_BASE, _TMP_SEQ
    if _TMP_BASE is None:
        import atexit
        import shutil
        _TMP_BASE = tempfile.mkdtemp(prefix="lattice-tests-")
        atexit.register(shutil.rmtree, _TMP_BASE, True)
    _TMP_SEQ += 1
    d = os.path.join(_TMP_BASE, "%s%d" % (prefix, _TMP_SEQ))
    os.mkdir(d)
    return d


def _fresh():
    return open_lattice(os.path.join(_tmpdir("test-"), "lattice.db"), origin="test-origin")


def _doc(i, **kw):
    d = {"id": "a%04d" % i, "content_type": "text/markdown", "content": "x"}
    d.update(kw)
    return d


# ── 1. proper time: gap-freeness under concurrent writers ────────────────────

def test_seq_is_gap_free_and_monotonic_single_thread():
    L = _fresh()
    for i in range(200):
        L.artifacts.put_artifact(_doc(i))
    seqs = sorted(r["_seq"] for r in L.artifacts.page_by_origin(limit=1000))
    assert seqs == list(range(1, 201)), "seq must be 1..N with no holes: %r" % seqs[:20]


def test_seq_gap_free_under_concurrent_writers():
    """N threads authoring concurrently must between them consume exactly the
    integers 1..total — no gaps (which would break the accounting
    identity `live_rows + vacated == last_seq`) and no repeats (which would make
    `(_origin,_seq)` non-unique)."""
    L = _fresh()
    threads, per = 8, 40
    errors = []

    def worker(t):
        try:
            for i in range(per):
                L.artifacts.put_artifact(_doc(t * 1000 + i))
        except Exception as e:                      # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(t,)) for t in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors, errors
    seqs = sorted(r["_seq"] for r in L.artifacts.page_by_origin(limit=10000))
    total = threads * per
    assert len(seqs) == total, "lost writes: %d of %d" % (len(seqs), total)
    assert seqs == list(range(1, total + 1)), (
        "gaps or repeats: min=%s max=%s distinct=%d" % (seqs[0], seqs[-1], len(set(seqs))))


def _union_versions(L):
    """Every `(_origin, _seq)` across `vertex ∪ edge`. One counter per observer
    spans both tables (contract §4 RESOLVED-5), so the union is the only correct
    unit of analysis for uniqueness and for gap-freeness."""
    return ([(r["_origin"], r["_seq"]) for r in L.artifacts.page_by_origin(limit=100000)]
            + [(r["_origin"], r["_seq"]) for r in L.graph.page_by_origin(limit=100000)])


def test_naive_two_store_construction_shares_one_proper_time():
    """The allocator registry keeps two independently constructed stores on one
    proper time: reusing the allocator is automatic, not a caller obligation."""
    d = _tmpdir("naive-")
    from mantle.db import LatticeArtifactStore, LatticeConn, LatticeGraphStore
    db = LatticeConn(os.path.join(d, "l.db"))
    arts = LatticeArtifactStore(db, origin="71")
    graph = LatticeGraphStore(db, origin="71")        # naive: no allocator=
    assert arts.seq is graph.seq, "the two stores must share ONE allocator"

    for i in range(10):
        arts.put_artifact({"id": "v%d" % i, "content_type": "t/x"})
        graph.add_edges([("v%d" % i, "v%d" % (i + 1), "lineage", {})])

    union = ([(r["_origin"], r["_seq"]) for r in arts.page_by_origin(limit=1000)]
             + [(r["_origin"], r["_seq"]) for r in graph.page_by_origin(limit=1000)])
    assert len(union) == 20, len(union)
    assert len(set(union)) == 20, (
        "duplicate (_origin,_seq) across vertex+edge: %r"
        % sorted(k for k in set(union) if union.count(k) > 1))
    seqs = sorted(s for _, s in union)
    assert seqs == list(range(1, 21)), "union must be gap-free 1..20: %r" % seqs


def test_seq_gap_freeness_is_a_property_of_the_union_not_either_table():
    """Per-table contiguity is not the invariant, and asserting it produces false
    failures.

    With one counter per observer spanning both tables, interleaved writes give
    vertex `{1,3,5,...}` and edge `{2,4,6,...}`. Neither is contiguous; the union
    is. This test pins the correct unit of analysis."""
    L = _fresh()
    for i in range(6):
        L.artifacts.put_artifact(_doc(i))
        L.graph.add_edges([("a%04d" % i, "z", "lineage", {})])
    vseqs = sorted(r["_seq"] for r in L.artifacts.page_by_origin(limit=1000))
    eseqs = sorted(r["_seq"] for r in L.graph.page_by_origin(limit=1000))
    assert vseqs == [1, 3, 5, 7, 9, 11], vseqs
    assert eseqs == [2, 4, 6, 8, 10, 12], eseqs
    assert vseqs != list(range(1, len(vseqs) + 1)), "per-table contiguity is not the invariant"
    assert sorted(vseqs + eseqs) == list(range(1, 13)), "the UNION must be gap-free"


def test_second_allocator_for_same_origin_raises():
    """A registry overwrite must be loud: a silent one would let a duplicate allocator for the
    same origin go unnoticed, and two allocators issuing seqs for one origin produce duplicates."""
    from mantle.db import LatticeConn, SeqAllocator, allocator_for
    d = _tmpdir("alloc-")
    db = LatticeConn(os.path.join(d, "l.db"))
    a = allocator_for(db, "71")
    assert allocator_for(db, "71") is a, "must reuse, not mint a second"
    assert allocator_for(db, "71", override=a) is a, "re-registering the same one is fine"
    try:
        allocator_for(db, "71", override=SeqAllocator(db, "71"))
        raise AssertionError("a conflicting second allocator must RAISE")
    except ValueError as e:
        assert "duplicate" in str(e).lower() or "already registered" in str(e).lower()
    # an allocator bound to a different observer is never silently accepted
    try:
        allocator_for(db, "71", override=SeqAllocator(db, "other"))
        raise AssertionError("cross-origin allocator must RAISE")
    except ValueError:
        pass
    assert allocator_for(db, "other") is not a, "distinct origins get distinct allocators"


def test_pending_publish_spans_the_union():
    """Backlog counts events, and an event is a vertex or an edge — they share one
    proper-time sequence, so a vertex-only count would under-report the mesh's
    real publish debt."""
    L = _fresh()
    for i in range(5):
        L.artifacts.put_artifact(_doc(i))
    L.graph.add_edges([("a0000", "a0001", "lineage", {}), ("a0001", "a0002", "lineage", {})])
    pp = L.artifacts.pending_publish(vertex_cursor=0, edge_cursor=0)
    assert (pp["vertex"], pp["edge"], pp["total"]) == (5, 2, 7), pp


def test_check_tail_does_not_false_fail_on_an_accounted_delete():
    """§5.4: subtracting seq values does not answer a row question.

    A plain `delete_artifact()` of the newest row vacates that seq, so `max_observed`
    legitimately shrinks on a perfectly healthy store. The verdict is derived from
    accounting instead, so an accounted vacancy and a genuine raw-SQL tail loss produce
    distinguishable output rather than the same one."""
    from mantle.db import check_tail
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(5)])
    base = check_tail(L.db, "test-origin")
    assert base["verdict"] == "tail_intact", base
    assert base["max_observed"] == 5 and base["vacated"] == 0

    L.artifacts.delete_artifact("a0004")            # delete the newest row
    assert L.artifacts.seq_accounting(scan=True)["balanced"] is True

    cheap = check_tail(L.db, "test-origin")
    assert cheap["verdict"] != "loss_detected", (
        "an accounted delete of the newest row must NOT read as data loss: %r" % cheap)
    assert cheap["verdict"] == "undecidable", cheap
    assert cheap["rows_lost"] is None, "an undecidable check must not emit a count"
    assert "vacated=1" in cheap["reason"], cheap

    definitive = check_tail(L.db, "test-origin", scan=True)
    assert definitive["verdict"] == "intact", definitive
    assert definitive["rows_lost"] == 0


def test_check_tail_still_catches_a_genuine_loss_and_tells_them_apart():
    """An accounted delete and a genuine tail loss must produce distinguishable verdicts."""
    from mantle.db import check_tail
    accounted = _fresh()
    accounted.artifacts.put_many([_doc(i) for i in range(5)])
    accounted.artifacts.delete_artifact("a0004")

    lost = _fresh()
    lost.artifacts.put_many([_doc(i) for i in range(5)])
    with lost.db.write() as cur:                    # raw, behind the store's back
        cur.execute("DELETE FROM vertex WHERE _seq = 5")

    a = check_tail(accounted.db, "test-origin", scan=True)
    b = check_tail(lost.db, "test-origin", scan=True)
    assert a["max_observed"] == b["max_observed"] == 4, (a, b)
    assert a["verdict"] == "intact" and a["rows_lost"] == 0, a
    assert b["verdict"] == "loss_detected" and b["rows_lost"] == 1, b
    assert a["verdict"] != b["verdict"], "the two cases must be distinguishable"


def test_check_tail_undecidable_is_not_a_pass():
    """An undecidable result must never be indistinguishable from a healthy one.
    There is no boolean field: `truncated=None` would read as falsy and pass a
    naive `if r["truncated"]: fail`."""
    from mantle.db import check_tail
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(3)])
    L.artifacts.put_artifact(_doc(0, content="v2"))     # vacates a seq
    r = check_tail(L.db, "test-origin")
    assert r["verdict"] == "undecidable"
    assert "truncated" not in r, "a boolean verdict field is the trap; do not add one"
    assert "missing_tail" not in r, "renamed: it was never a tail-specific count"
    # a foreign origin can never be decided
    L.artifacts.put_many([dict(_doc(9), _origin="peer-1", _seq=77)], stamp_rev=False)
    f = check_tail(L.db, "peer-1", scan=True)
    assert f["verdict"] == "undecidable" and f["local_only"] is False, f


def test_check_tail_insert_only_endpoint_is_exact_but_narrow():
    """`vacated == 0` makes the endpoint exact and cheap. The verdict is
    `tail_intact`, not `intact` — it does not claim interior rows are present,
    because the endpoint check cannot see them."""
    from mantle.db import check_tail
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(6)])
    assert check_tail(L.db, "test-origin")["verdict"] == "tail_intact"

    with L.db.write() as cur:                       # interior loss, endpoint intact
        cur.execute("DELETE FROM vertex WHERE _seq = 3")
    cheap = check_tail(L.db, "test-origin")
    assert cheap["verdict"] == "tail_intact", "the endpoint IS still intact"
    assert cheap["max_observed"] == 6
    # only the row-counting scan sees it — which is why the cheap verdict is narrow
    deep = check_tail(L.db, "test-origin", scan=True)
    assert deep["verdict"] == "loss_detected" and deep["rows_lost"] == 1, deep


def test_high_water_never_moves_backwards():
    """The mark is the authority for what was allocated. If it could fall, the
    observer would re-issue live seqs after any loss."""
    from mantle.db import high_water
    L = _fresh()
    for i in range(5):
        L.artifacts.put_artifact(_doc(i))
    assert high_water(L.db, "test-origin") == 5
    with L.db.write() as cur:
        cur.execute("DELETE FROM vertex")
    assert high_water(L.db, "test-origin") == 5, "high-water must survive row loss"
    L.artifacts.put_artifact(_doc(99))
    assert L.artifacts.get_artifact("a0099")["_seq"] == 6, (
        "allocation must continue ABOVE the high-water, never restart at 1")


def test_update_vacates_a_seq_so_rows_are_legitimately_non_contiguous():
    """An update allocates a fresh `_seq` and vacates the old one, so surviving
    rows do not start at 1 and are not contiguous. Row contiguity is not the
    invariant; allocation accounting is."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(1))                 # a -> 1
    L.artifacts.put_artifact(_doc(2))                 # b -> 2
    L.artifacts.put_artifact(_doc(1, content="updated"))   # a' -> 3, vacating 1
    rows = sorted((r["doc"]["id"], r["_seq"]) for r in L.artifacts.page_by_origin(limit=100))
    assert rows == [("a0001", 3), ("a0002", 2)], rows
    seqs = sorted(s for _, s in rows)
    assert seqs == [2, 3], "contiguous but starting at 2 — NOT 1"
    assert seqs != list(range(1, len(seqs) + 1)), (
        "precondition: strict contiguity is FALSE here, which is why it is the "
        "wrong assertion for any store that has taken an update")

    acc = L.artifacts.seq_accounting(scan=True)
    assert acc["last_seq"] == 3 and acc["live_rows"] == 2 and acc["vacated"] == 1
    assert acc["balanced"] is True and acc["unaccounted"] == 0, acc
    assert acc["insert_only"] is False


def test_seq_accounting_balances_across_updates_deletes_and_edges():
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(20)])          # 20 inserts
    L.graph.add_edges([("a%04d" % i, "z", "lineage", {}) for i in range(5)])  # 5 inserts
    for i in range(6):
        L.artifacts.put_artifact(_doc(i, content="v2"))         # 6 updates -> 6 vacated
    L.graph.add_edges([("a0000", "z", "lineage", {})])          # replay -> 1 update
    for i in range(3):
        L.artifacts.delete_artifact("a%04d" % (10 + i))         # 3 deletes
    for mode in (False, True):
        acc = L.artifacts.seq_accounting(scan=mode)
        assert acc["balanced"] is True, (mode, acc)
        assert acc["unaccounted"] == 0, (mode, acc)
        assert acc["vacated"] == 10, acc                        # 6 + 1 + 3
    counter = L.artifacts.seq_accounting(scan=False)
    scanned = L.artifacts.seq_accounting(scan=True)
    assert counter["live_rows"] == scanned["live_rows"], (counter, scanned)


def test_seq_accounting_sees_a_row_lost_outside_the_write_path():
    """The property that made the contiguity check worth having. An accounted
    removal increments `vacated`; a row lost to a bad migration does not, so the
    balance breaks and the loss is visible.

    Only `scan=True` can see it — the counter was never decremented, so a
    counter-vs-counter comparison reports clean."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(10)])
    assert L.artifacts.seq_accounting(scan=True)["balanced"] is True

    with L.db.write() as cur:                       # a bad migration, not a delete
        cur.execute("DELETE FROM vertex WHERE id IN ('a0003','a0004')")

    lost = L.artifacts.seq_accounting(scan=True)
    assert lost["balanced"] is False, "row loss went undetected"
    assert lost["unaccounted"] == 2, lost
    assert lost["live_rows"] == 8 and lost["vacated"] == 0, lost
    # contrast: an accounted delete of the same size stays balanced
    L2 = _fresh()
    L2.artifacts.put_many([_doc(i) for i in range(10)])
    L2.artifacts.delete_artifact("a0003")
    L2.artifacts.delete_artifact("a0004")
    ok = L2.artifacts.seq_accounting(scan=True)
    assert ok["balanced"] is True and ok["vacated"] == 2, ok


def test_insert_only_store_permits_the_stronger_contiguity_assertion():
    """`vacated == 0` means the store has never taken an update or delete, and in
    that case strict contiguity 1..last_seq is exactly valid — the post-migration
    case, and a stronger claim than the union invariant alone guarantees."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(15)])
    L.graph.add_edges([("a0000", "z", "lineage", {})])
    acc = L.artifacts.seq_accounting(scan=True)
    assert acc["insert_only"] is True, acc
    seqs = sorted([r["_seq"] for r in L.artifacts.page_by_origin(limit=100)]
                  + [r["_seq"] for r in L.graph.page_by_origin(limit=100)])
    assert seqs == list(range(1, acc["last_seq"] + 1)), (
        "insert-only: strict contiguity MUST hold")

    L.artifacts.put_artifact(_doc(0, content="v2"))       # one update
    assert L.artifacts.seq_accounting()["insert_only"] is False, (
        "one update must retire the stronger assertion")


def test_seq_accounting_does_not_false_fail_on_foreign_origins():
    """A peer's seqs are observed, not allocated: we learn its high-water from
    whatever arrived while the rows behind it are still in flight. Asserting
    equality across every origin would flag every healthy peer on every sweep —
    the same false-failure class as asserting row contiguity, one level up."""
    from mantle.db import seq_accounting
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(5)])
    # observe peer-1 at seq 900 while receiving only one of its rows
    L.artifacts.put_many([dict(_doc(50), _origin="peer-1", _seq=900)], stamp_rev=False)

    peer = seq_accounting(L.db, "peer-1", scan=True)
    assert peer["local_only"] is False
    assert peer["last_seq"] == 900 and peer["live_rows"] == 1
    assert peer["unaccounted"] == 899, peer
    assert peer["unaccounted_means"] == "not_yet_received"
    assert peer["balanced"] is True, "replication lag must not read as data loss"

    loc = L.artifacts.seq_accounting(scan=True)
    assert loc["local_only"] is True and loc["unaccounted_means"] == "lost"
    assert loc["balanced"] is True and loc["last_seq"] == 5


def test_unordered_consume_does_not_move_a_row_between_origins():
    """A foreign version of a locally-authored vertex is UNORDERED, so `keep_local`
    declines it — and the local accounting must stay balanced, with no row and no
    vacancy invented."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(5)])
    n = L.artifacts.put_many([dict(_doc(2), _origin="peer-1", _seq=900)], stamp_rev=False)
    assert n == 1, "a decision was made -> HANDLED"
    assert L.artifacts.get_artifact("a0002")["_origin"] == "test-origin"
    acc = L.artifacts.seq_accounting(scan=True)
    assert acc["live_rows"] == 5 and acc["vacated"] == 0 and acc["balanced"] is True, acc
    assert L.artifacts.verify_counters()["drift"] == {}


def test_row_moving_between_origins_keeps_both_accounts_balanced():
    """When a foreign version legitimately supersedes a local row (same origin
    lineage, higher seq is not applicable across origins — so this is the
    take_remote path), the row must leave the local account and join the peer's."""
    L = _fresh()
    from mantle.db import seq_accounting
    L.artifacts.put_many([_doc(i) for i in range(5)])
    L.artifacts.put_many([dict(_doc(2), _origin="peer-1", _seq=900)],
                         stamp_rev=False, on_unordered="take_remote")
    assert L.artifacts.get_artifact("a0002")["_origin"] == "peer-1"
    loc = L.artifacts.seq_accounting(scan=True)
    assert loc["live_rows"] == 4, loc
    assert loc["vacated"] == 1, "the superseded local row must be an ACCOUNTED vacancy"
    assert loc["balanced"] is True, loc
    peer = seq_accounting(L.db, "peer-1", scan=True)
    assert peer["live_rows"] == 1 and peer["balanced"] is True, peer
    assert L.artifacts.verify_counters()["drift"] == {}


def test_seq_accounting_column_is_last_issued_not_next():
    """`last_seq` holds the seq already issued, not the seq to issue next —
    treating it as `next_seq` and computing `next_seq - 1` produces an off-by-one
    that under-counts by one allocation."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(1))
    r = L.db.read().execute("SELECT last_seq FROM seq_counter WHERE origin = ?",
                            ("test-origin",)).fetchone()
    assert r[0] == 1, "after ONE allocation last_seq must be 1, not 2"
    assert L.artifacts.seq_accounting()["last_seq"] == 1
    assert L.artifacts.get_artifact("a0001")["_seq"] == 1


def test_verify_counters_includes_seq_accounting_and_row_counters():
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(10)])
    L.graph.add_edges([("a0000", "z", "lineage", {})])
    L.artifacts.put_artifact(_doc(0, content="v2"))
    rep = L.artifacts.verify_counters()
    assert rep["drift"] == {}, rep["drift"]
    assert rep["seq_accounting"]["balanced"] is True, rep["seq_accounting"]
    assert rep["seq_accounting"]["source"] == "scan"
    # the per-origin row counter is judged against both tables
    with L.db.write() as cur:
        cur.execute("UPDATE counter SET n = 77 WHERE name = 'rows:test-origin'")
    assert L.artifacts.verify_counters()["drift"].get("rows:test-origin") == (77, 11)
    L.artifacts.verify_counters(repair=True)
    assert L.artifacts.verify_counters()["drift"] == {}


def test_publish_backlog_is_gone_and_must_not_come_back():
    """`LatticeArtifactStore` carries no `publish_backlog` method. `high_water -
    cursor` counts allocations rather than rows: it over-reports once any seq is
    vacated, it can never reach 0 with two feeds and one cursor, and
    `min(vc, ec)` only relocates that floor.

    `pending_publish()` is the replacement: it counts rows per feed, which is
    immune to vacancy by construction."""
    from mantle.db import LatticeArtifactStore
    assert not hasattr(LatticeArtifactStore, "publish_backlog"), (
        "publish_backlog is back. It cannot be made correct by subtraction — "
        "use pending_publish(), which counts rows per feed.")
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(5)])
    for r in range(3):
        L.artifacts.put_many([_doc(i, content="r%d" % r) for i in range(3)])
    # a scenario where allocation count and live-row count diverge (14 allocations,
    # 5 live rows), which is where a subtraction-based count would be wrong
    acc = L.artifacts.seq_accounting(scan=True)
    assert acc["last_seq"] == 14 and acc["live_rows"] == 5 and acc["vacated"] == 9, acc
    pp = L.artifacts.pending_publish(vertex_cursor=0, edge_cursor=0)
    assert pp["total"] == 5, "row count, not allocation count: %r" % pp

def test_pending_publish_reaches_zero_per_feed():
    """The convergence property. Each term drains to 0 on its own cursor, so the
    sum is 0 exactly when the node has published everything — which neither
    `high_water - cursor` nor `min(vertex_cursor, edge_cursor)` can achieve."""
    L = _fresh()
    # interleave so vertex and edge seqs are genuinely mixed
    for i in range(30):
        L.artifacts.put_artifact(_doc(i))
        if i % 3 == 0:
            L.graph.add_edges([("a%04d" % i, "z%d" % i, "lineage", {})])
    # Ending on an edge makes the vertex feed drain below last_seq, which is when
    # the single-cursor floor actually manifests; a fixture whose last write is a
    # vertex hides it entirely.
    L.graph.add_edges([("a0029", "zz", "lineage", {})])
    vs = [r["_seq"] for r in L.artifacts.page_by_origin(limit=1000)]
    es = [r["_seq"] for r in L.graph.page_by_origin(limit=1000)]
    assert vs and es and max(es) > max(vs), "fixture must end on an edge"

    pp = L.artifacts.pending_publish(vertex_cursor=0, edge_cursor=0)
    assert pp["total"] == len(vs) + len(es), pp
    # drain each feed against its own cursor
    drained = L.artifacts.pending_publish(vertex_cursor=max(vs), edge_cursor=max(es))
    assert drained["total"] == 0, "a fully-published node must read 0: %r" % drained
    # the naive single-cursor form does not reach 0
    lo = min(max(vs), max(es))
    assert L.artifacts.pending_publish(vertex_cursor=lo, edge_cursor=lo)["total"] > 0, (
        "min(vc, ec) relocates the floor rather than removing it")


def test_update_churn_keeps_accounting_exact():
    """`run_task` rewrites 14 operator artifacts per task, so the allocator sees
    far more update traffic than a naive workload would suggest. Heavy update
    churn must not drift the accounting or leak seqs."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(14)])
    for _round in range(50):                          # 50 tasks x 14 rewrites
        L.artifacts.put_many([_doc(i, content="r%d" % _round) for i in range(14)])
    acc = L.artifacts.seq_accounting(scan=True)
    assert acc["last_seq"] == 14 * 51, acc
    assert acc["live_rows"] == 14, acc
    assert acc["vacated"] == 14 * 50, acc
    assert acc["balanced"] is True and acc["unaccounted"] == 0, acc
    assert L.artifacts.verify_counters()["drift"] == {}


def test_seq_gap_free_across_os_processes():
    """The deployment shape. `worker.py --pool` spawns real OS subprocesses, which
    share no mutex — the in-process `threading.RLock` is an optimization, and the
    property that actually has to hold is `BEGIN IMMEDIATE` + `busy_timeout` at the
    file lock. A threads-only test would pass while cross-process writers silently
    duplicated or skipped seqs."""
    import subprocess
    d = _tmpdir("mp-")
    path = os.path.join(d, "l.db")
    open_lattice(path, origin="mp-origin")                  # create up front
    nproc, per = 6, 30
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__),
                               "--child", path, "p%d" % k]) for k in range(nproc)]
    rcs = [p.wait() for p in procs]
    assert all(r == 0 for r in rcs), "child process failed: %r" % rcs
    L = open_lattice(path, origin="mp-origin")
    seqs = sorted(r["_seq"] for r in L.artifacts.page_by_origin(limit=10000))
    total = nproc * per
    assert len(seqs) == total, "lost writes across processes: %d of %d" % (len(seqs), total)
    assert seqs == list(range(1, total + 1)), "gap or repeat across processes"
    assert L.artifacts.pending_publish()["total"] == total


def _child(path, tag, per=30):
    L = open_lattice(path, origin="mp-origin")
    for i in range(per):
        L.artifacts.put_artifact({"id": "%s-%03d" % (tag, i), "content_type": "t/x"})


def test_seq_is_not_a_clock():
    """`time.time_ns()` yields ~1 distinct value across a fast batch on Windows'
    15.6ms tick, which is why proper time cannot be a clock reading. `_seq` must
    be injective over the same batch."""
    import time
    clock = {time.time_ns() for _ in range(2000)}
    L = _fresh()
    for i in range(2000):
        L.artifacts.put_artifact(_doc(i))
    seqs = {r["_seq"] for r in L.artifacts.page_by_origin(limit=5000)}
    assert len(seqs) == 2000, "seq collapsed like a clock: %d distinct" % len(seqs)
    print("    [clock had %d distinct value(s) over 2000 calls; _seq had %d]"
          % (len(clock), len(seqs)))


def test_rollback_consumes_no_proper_time():
    """A gap would appear here if a failed write kept its allocated seq."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(1))
    n = L.artifacts.put_many([_doc(2), {"no_id": True}, _doc(3)])
    assert n == 2, "errored doc must not count as handled: %r" % n
    seqs = sorted(r["_seq"] for r in L.artifacts.page_by_origin(limit=100))
    assert seqs == [1, 2, 3], "rollback left a gap in proper time: %r" % seqs


def test_edge_idempotent_under_replay():
    """Mesh segments are replayed. Re-adding must update in place, never append."""
    L = _fresh()
    seg = [("a", "b", "lineage", {"w": 1}), ("b", "c", "lineage", {"w": 2})]
    for _ in range(10):                     # ten replays of the same segment
        handled = L.graph.add_edges(seg)
        assert handled == 2
    assert L.graph.count_edges() == 2, "duplicates accumulated: %d" % L.graph.count_edges()
    assert L.graph.neighbors("a", "lineage") == ["b"]
    rows = L.graph.edges_of("a", label="lineage")
    assert len(rows) == 1 and rows[0]["props"]["w"] == 1


def test_edge_replay_preserves_version_on_consume():
    """stamp_rev=False replays must not re-stamp, and an older replay must be
    LWW-rejected while still counting as handled."""
    L = _fresh()
    e = ("a", "b", "lineage", {"_origin": "peer-1", "_seq": 7})
    assert L.graph.add_edges([e], stamp_rev=False) == 1
    older = ("a", "b", "lineage", {"_origin": "peer-1", "_seq": 3})
    assert L.graph.add_edges([older], stamp_rev=False) == 1, "LWW reject must count HANDLED"
    row = L.graph.edges_of("a", label="lineage")[0]
    assert (row["_origin"], row["_seq"]) == ("peer-1", 7), row


def test_nul_separator_collision_resistance():
    """('a|b','c') must not collide with ('a','b|c'). Without the NUL separator
    these hash identically and one edge silently overwrites the other."""
    k1 = edge_key("a|b", "c", "L")
    k2 = edge_key("a", "b|c", "L")
    assert k1 != k2, "NUL separator missing — edge keys collide"
    for sep in ("/", ":", "-", "\t", " "):
        assert edge_key("a" + sep + "b", "c", "L") != edge_key("a", sep + "b" + "c", "L")
    # and the same concatenation hazard across the label boundary
    assert edge_key("a", "b", "c|d") != edge_key("a", "b|c", "d")

    L = _fresh()
    L.graph.add_edges([("a|b", "c", "L", {}), ("a", "b|c", "L", {})])
    assert L.graph.count_edges() == 2, "one edge overwrote the other"
    assert L.graph.neighbors("a|b") == ["c"]
    assert L.graph.neighbors("a") == ["b|c"]


def test_edge_is_origin_is_not_underscore_origin():
    """`is_origin` (grant-propagation bit) and `_origin` (authoring observer) are
    unrelated and must never be collapsed."""
    L = _fresh()
    L.graph.add_edges([("a", "b", "lineage", {"is_origin": 1})])
    row = L.graph.edges_of("a")[0]
    assert row["is_origin"] == 1
    assert row["_origin"] == "test-origin"
    assert row["is_origin"] != row["_origin"]


# ── 3. put_many: handled vs written ──────────────────────────────────────────

def test_put_many_counts_written():
    L = _fresh()
    assert L.artifacts.put_many([_doc(i) for i in range(10)]) == 10
    assert L.artifacts.count() == 10


def test_put_many_counts_lww_rejects_as_handled():
    """Declining to overwrite a newer local row is the right outcome — it is
    handled, not lost — so it must count, or the mesh raises 'partial apply' and
    wedges its cursor on a correctly-converged batch."""
    L = _fresh()
    fresh = [dict(_doc(i), _origin="peer-1", _seq=100 + i) for i in range(5)]
    assert L.artifacts.put_many(fresh, stamp_rev=False) == 5
    stale = [dict(_doc(i), _origin="peer-1", _seq=1 + i) for i in range(5)]
    assert L.artifacts.put_many(stale, stamp_rev=False) == 5, "LWW reject must count"
    for i in range(5):
        assert L.artifacts.get_artifact("a%04d" % i)["_seq"] == 100 + i


def test_put_many_errors_do_not_count():
    """The mesh's only data-loss guard. Returning len(docs) unconditionally
    disables it from inside the store layer."""
    L = _fresh()
    docs = [_doc(1), {"id": ""}, _doc(2), {"nope": 1}, _doc(3)]
    n = L.artifacts.put_many(docs)
    assert n == 3, "expected 3 handled of 5, got %r" % n
    assert n < len(docs), "the mesh guard `written < len(batch)` must fire"
    assert L.artifacts.count() == 3


def test_put_many_one_bad_doc_does_not_roll_back_the_batch():
    L = _fresh()
    n = L.artifacts.put_many([_doc(1), {"id": ""}, _doc(2)])
    assert n == 2
    assert L.artifacts.get_artifact("a0001") is not None
    assert L.artifacts.get_artifact("a0002") is not None


def test_stamp_rev_keyword_is_accepted_on_both_paths():
    """A backend that omits `stamp_rev` raises TypeError, which `reconcile_merkle`
    swallows into `applied: 0` — replication silently applying nothing while
    reporting clean. Both keywords must exist on both methods."""
    L = _fresh()
    L.artifacts.put_artifact(dict(_doc(1), _origin="peer-1", _seq=5), stamp_rev=False)
    L.artifacts.put_many([dict(_doc(2), _origin="peer-1", _seq=6)],
                         batch=500, stamp_rev=False)
    assert L.artifacts.get_artifact("a0001")["_origin"] == "peer-1"
    assert L.artifacts.get_artifact("a0001")["_seq"] == 5


def test_consume_preserves_origin_and_local_seq_untouched():
    """stamp_rev=False must not burn local proper time, or a replicating node's
    publish backlog inflates with rows it never authored."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(1))                     # local: seq 1
    L.artifacts.put_many([dict(_doc(i), _origin="peer-1", _seq=i) for i in range(2, 20)],
                         stamp_rev=False)
    assert L.artifacts.pending_publish()["total"] == 1, "consume burned local proper time"


def test_consume_without_version_is_a_loud_error():
    L = _fresh()
    assert L.artifacts.put_many([_doc(1)], stamp_rev=False) == 0, "must not silently re-stamp"
    try:
        L.artifacts.put_artifact(_doc(1), stamp_rev=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_unordered_origins_are_not_tiebroken():
    L = _fresh()
    L.artifacts.put_many([dict(_doc(1), _origin="peer-A", _seq=1)], stamp_rev=False)
    n = L.artifacts.put_many([dict(_doc(1), _origin="peer-B", _seq=999)], stamp_rev=False)
    assert n == 1, "a decision was made -> HANDLED"
    assert L.artifacts.get_artifact("a0001")["_origin"] == "peer-A", "no clock, no tiebreak"
    n = L.artifacts.put_many([dict(_doc(1), _origin="peer-B", _seq=999)],
                             stamp_rev=False, on_unordered="error")
    assert n == 0, "on_unordered='error' must NOT count as handled"


# ── 4. version comparison expresses "unordered" ──────────────────────────────

def test_compare_version():
    assert compare_version("o", 5, "o", 3) == NEWER
    assert compare_version("o", 3, "o", 5) == OLDER
    assert compare_version("o", 5, "o", 5) == SAME
    assert compare_version("a", 5, "b", 3) == UNORDERED, "different origins are UNORDERED"
    assert compare_version("a", 1, "b", 999) == UNORDERED
    assert compare_version("o", None, "o", 5) == UNORDERED


# ── 5. keyset pagination ─────────────────────────────────────────────────────

def test_keyset_pagination_visits_every_row_exactly_once():
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(500)])
    seen, cur, pages = [], "", 0
    while True:
        page = L.artifacts.page_by_id(after=cur, limit=37)
        if not page:
            break
        pages += 1
        seen += [r["id"] for r in page]
        cur = page[-1]["id"]
        assert pages < 100, "keyset loop did not advance"
    assert len(seen) == 500 and len(set(seen)) == 500, (len(seen), len(set(seen)))
    assert seen == sorted(seen)


def test_offset_pagination_is_refused():
    """`LIMIT ? OFFSET ?` is a defect, not a pattern: `skip` at a large offset costs
    roughly two orders of magnitude more than the equivalent keyset page."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(5)])
    assert len(list(L.artifacts.list_artifacts(skip=0))) == 5
    try:
        list(L.artifacts.list_artifacts(skip=10))
        raise AssertionError("skip must raise, not silently scan-and-discard")
    except ValueError as e:
        assert "keyset" in str(e).lower()


def test_page_by_origin_advances_with_strict_gt():
    """`_seq` is injective, so a strict `>` cursor is correct — no revision-group
    completion dance (the ~25 lines content_tier.py needs against `_rev` ties)."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(300)])
    seen, cur = [], 0
    while True:
        page = L.artifacts.page_by_origin(after_seq=cur, limit=25)
        if not page:
            break
        seen += [r["_seq"] for r in page]
        cur = page[-1]["_seq"]
    assert seen == list(range(1, 301)), "publish scan skipped or repeated rows"


def test_count_after_id_reports_inexactness():
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(100)])
    assert L.artifacts.count_after_id("", cap=1000) == {"n": 100, "exact": True}
    r = L.artifacts.count_after_id("", cap=10)
    assert r == {"n": 10, "exact": False}, "a truncated count must say so"


# ── 6. counters — no count(*) anywhere ───────────────────────────────────────

def _store_sources():
    """Every non-test module of the store package — the files whose SQL the guards below read.

    Returns a list and asserts it, because all three callers reach an ABSENCE verdict: "no
    `count(*)` reaches SQLite" and "this scan read no files" are the same answer otherwise, and the
    second one arrives silently — `os.listdir` on the wrong directory raises, but on a directory
    that merely holds different files it does not. So the resolved directory and the modules that
    must be in it are both named here rather than assumed."""
    here = os.path.dirname(os.path.abspath(__file__))
    assert os.path.basename(here) == "db", (
        "the store package is not where this scan expects it: resolved %s — fix the path" % here)
    names = sorted(fn for fn in os.listdir(here)
                   if fn.endswith(".py") and not fn.startswith("test_"))
    assert len(names) > 15, (
        "only %d store modules found in %s — the scan is not reaching the package, and a source "
        "guard over an empty file list passes on nothing" % (len(names), here))
    for required in ("vertex.py", "edge.py", "seq.py", "schema.py", "lattice_api.py", "store.py"):
        assert required in names, (
            "%s is not being scanned — the guards below would miss its SQL entirely" % required)
    return [(fn, os.path.join(here, fn)) for fn in names]


def _sql_literals(path):
    """Every string that reaches a `.execute()` / `.executemany()` call.

    An AST walk, not a grep, and scoped to the call argument rather than to the
    file — which makes it both stronger and exact:

      * stronger than a grep, because it follows SQL assembled by `%`, `+` or
        adjacent-literal concatenation: the operand Constants are still inside the
        argument subtree, so a banned construct hidden in a fragment is caught.
      * exact, because the invariant is "no `count(*)` / no OFFSET is ever sent to
        SQLite", not "these characters never appear in the file". A line-based grep
        cannot tell executable SQL from prose that warns about it, and this package
        deliberately quotes the banned constructs — in docstrings and in the
        ValueError message `list_artifacts` raises when a caller passes `skip` —
        precisely so the hazard is named at the point it matters. A guard that
        forced those strings silent would trade real documentation for a proxy."""
    import ast
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in ("execute", "executemany")):
            continue
        if not node.args:
            continue
        for sub in ast.walk(node.args[0]):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.append((sub.value, getattr(sub, "lineno", 0)))
    return out


def _cols(L, table):
    return [r[1] for r in L.db.read().execute("PRAGMA table_info(%s)" % table)]


def test_vertex_has_no_observer_dependent_columns():
    """Contract §2.2/§2.3.

    Pinned as an exact column set, not just an absence check: `state`, `_rev`, and any
    future candidate must satisfy §2.3 before joining the schema — would every observer
    compute the same value for it? If not, it is content and belongs in `doc`.

    `created_time` satisfies §2.3 on these grounds:

      * It is absent at the source for the 6.11M wiki rows (`wiki-en-1000071` returns
        None directly from the legacy graph engine), so there is no clock reading to
        disagree about for the overwhelming majority of the corpus.
      * Where absent, it is stamped `GENESIS_EPOCH` — a constant, so every observer
        computes it identically.
      * Where a real reading exists, it is an immutable claim carried from the author
        and never re-stamped, with `doc.created_time_origin` naming the claimant. A
        value that does not move under replication does not depend on the observer.

    `state` and `_rev` stay out: they fail §2.3 for their own reasons, independent of
    the `created_time` reasoning above."""
    L = _fresh()
    assert _cols(L, "vertex") == ["id", "ct", "offer", "content_ref", "created_by",
                                  "created_time", "origin_root", "root_id", "_origin", "_seq",
                                  "_leaf", "doc"], _cols(L, "vertex")
    for gone in ("state", "_rev"):
        assert gone not in _cols(L, "vertex"), "%r came back as a column" % gone


def _root_of(L, vid):
    row = L.db.read().execute(
        "SELECT origin_root FROM vertex WHERE id = ?", (vid,)
    ).fetchone()
    return row[0] if row else None


def test_origin_root_is_the_containment_root_not_the_version_root():
    """`origin_root` is the collection, `root_id` is the version lineage.

    Both are called "root" in conversation, but they are orthogonal: picking the
    wrong one keys content to a value that moves (versions advance constantly;
    containment does not)."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(1, collection_id="stage.0.lexicon", root_id="some-other-version"),
                   stamp_rev=True)
    assert _root_of(L, "a0001") == "stage.0.lexicon", (
        "origin_root must follow containment, not the version root"
    )


def test_origin_root_falls_back_to_self_for_a_top_level_artifact():
    """A vertex with no collection IS its own containment root — the live example is
    the `vtype.*` type definitions, which sit at the top with no parent."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(2), stamp_rev=True)          # no collection_id
    assert _root_of(L, "a0002") == "a0002"


def test_explicit_origin_root_is_never_recomputed():
    """Once resolved upstream it is authoritative.

    Recomputing a value whose entire contract is "never moves" is exactly how it
    moves. When subject trees arrive (GENESIS §5) the resolver supplies a root that
    is not the immediate collection, and this must survive the write untouched."""
    L = _fresh()
    L.artifacts.put_artifact(
        _doc(3, collection_id="subject.math.topology", origin_root="subject.math"),
        stamp_rev=True,
    )
    assert _root_of(L, "a0003") == "subject.math", (
        "an explicitly-resolved origin_root was overwritten by the immediate parent"
    )


def test_origin_root_never_falls_back_to_created_by():
    """`created_by` is provenance, not identity, so it must never become the key
    root. Provenance gets corrected — the lattice §3 identity fold is an example —
    and if it could become the key root, correcting an identity would orphan every
    blob written under the old value."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(4, created_by="john@ikailo.com"), stamp_rev=True)
    assert _root_of(L, "a0004") != "john@ikailo.com"
    assert _root_of(L, "a0004") == "a0004"


def test_no_time_derived_column_or_index_anywhere():
    """The general guard. A column is a standing invitation to `ORDER BY` it, so a
    frame-local value must not become one — in `vertex`, `edge`, or any index.

    `_seq` is exempt and is not a counterexample: it is never interpreted alone,
    and `(_origin, _seq)` is agreed by every observer once replicated. The pair is
    the value; the column is half of a composite.

    Scoped to the replicated tables, and the scope is the reasoned part. The
    `task` sidecar holds `claimed_at` / `next_retry_at` / `completed_at` and
    indexes `completed_at` — legitimately, because §1.2 classes work-pool lease
    expiry as node-local entirely. §2.3 asks whether every observer computes the
    same value; for node-local coordination state there is only one observer, so
    the question does not arise. `task` never replicates."""
    L = _fresh()
    banned = ("time", "clock", "stamp", "_rev", "date", "when", "epoch",
              "_at", "_ts", "seconds", "millis")
    # `created_time` is exempt: the column is sanctioned, the ordering is not. It passes §2.3
    # because it never carries a live clock reading — absent at source for 6.11M rows, stamped
    # with a constant genesis epoch where absent, and carried as an immutable author claim where
    # present (never re-stamped, claimant in `doc.created_time_origin`). A value that does not
    # move under replication does not depend on the observer.
    exempt = {"_seq", "_origin", "created_time"}   # half of a composite; see the docstring
    replicated = ("vertex", "edge")
    offenders = []
    for table in replicated:
        for c in _cols(L, table):
            if c in exempt:
                continue
            if any(b in c.lower() for b in banned):
                offenders.append("%s.%s" % (table, c))
    assert not offenders, (
        "observer-dependent column(s) %r — see contract §2.3. If ordering is "
        "needed it is edge.order_key within a frame and graph reachability "
        "across frames; UNORDERED IS A VALID ANSWER." % offenders)

    # Nothing derived from one may be indexed on a replicated table — an index is
    # an even stronger invitation to sort by it than a bare column.
    for name, sql in L.db.read().execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL"):
        tbl = (sql or "").lower().split(" on ")[-1].split("(")[0].strip()
        if tbl not in replicated:
            continue                       # task: node-local, see the docstring
        for b in banned:
            assert b not in (sql or "").lower(), (
                "index %r on replicated table %r references a time-derived value: %r"
                % (name, tbl, sql))

    # The guard must have teeth: prove it fires on a real violation.
    with L.db.write() as cur:
        cur.execute("ALTER TABLE vertex ADD COLUMN updated_at INTEGER")
    caught = [c for c in _cols(L, "vertex")
              if c not in exempt and any(b in c.lower() for b in banned)]
    assert caught == ["updated_at"], (
        "the guard did not catch an added time column: %r" % caught)


def test_created_time_survives_in_doc_attributed_to_its_claimant():
    """§2.2 requirement 2: the value is preserved, and recoverable as which
    observer claimed it — a claim with a named claimant, not an anonymous int."""
    L = _fresh()
    L.artifacts.put_artifact(_doc(1, created_time=1770000000))
    d = L.artifacts.get_artifact("a0001")
    assert d["created_time"] == 1770000000, "the value must survive in doc"
    assert d["created_time_origin"] == "test-origin", d

    # consume: a peer's clock reading keeps the peer's attribution
    L.artifacts.put_many([dict(_doc(2), created_time=42, _origin="peer-1", _seq=5)],
                         stamp_rev=False)
    p = L.artifacts.get_artifact("a0002")
    assert p["created_time"] == 42
    assert p["created_time_origin"] == "peer-1", (
        "re-stamping a peer's clock reading as ours is the same error as "
        "re-stamping _origin")

    # an existing attribution is preserved, never overwritten
    L.artifacts.put_artifact(dict(p, content="v2"))
    assert L.artifacts.get_artifact("a0002")["created_time_origin"] == "peer-1"

    # no created_time -> no attribution invented
    L.artifacts.put_artifact(_doc(3))
    assert "created_time_origin" not in L.artifacts.get_artifact("a0003")


def test_created_time_is_never_ordered_or_filtered_on():
    """The column's existence does not make ordering by `created_time` meaningful:

      * 97.8% of rows carry the same constant (`GENESIS_EPOCH`), so an ordering over them
        is an arbitrary tie-break wearing a chronology's clothes.
      * The remaining 2.2% are unreconciled clock readings from different authoring
        observers. Two nodes' clocks do not compose into one timeline, which is why the
        value carries `doc.created_time_origin` naming its claimant.

    The guard is therefore "no code may sort or filter on it" rather than "the column
    cannot exist" — a source-level check, in the same shape as
    `test_no_count_star_reaches_sqlite` below."""
    offenders = []
    for fn, path in _store_sources():
        for text, lineno in _sql_literals(path):
            low = " ".join(text.lower().split())
            if "order by created_time" in low or "created_time >" in low \
               or "created_time <" in low or "created_time between" in low:
                offenders.append("%s:%d %r" % (fn, lineno, text[:70]))
    assert not offenders, (
        "created_time is being ordered or filtered on: %r — 97.8%% of rows share one constant "
        "and the rest are unreconciled clocks. Use edge.order_key within a frame." % offenders)

    # The guard must have teeth: prove it fires on a real violation.
    import re as _re
    probe = "SELECT id FROM vertex ORDER BY created_time DESC"
    low = " ".join(probe.lower().split())
    assert "order by created_time" in low, "the detector would not catch a real ORDER BY"


def test_no_count_star_reaches_sqlite():
    """`count(*)` dereferences every record — EXPLAIN shows it loads every row to
    produce one integer, which can OOM a process on a large corpus. It must appear
    nowhere in any executable string in this package."""
    bad = []
    for fn, path in _store_sources():
        for text, lineno in _sql_literals(path):
            low = text.lower().replace(" ", "")
            if "count(*)" in low or "count (*)" in text.lower():
                bad.append("%s:%d %r" % (fn, lineno, text[:60]))
    assert not bad, "count(*) reaches SQLite in: %r" % bad


def test_no_offset_pagination_reaches_sqlite():
    """`LIMIT ? OFFSET ?` is a defect, not a pattern: SKIP at a large offset costs
    roughly two orders of magnitude more than the equivalent keyset page, because
    the engine walks and discards every skipped row. Keyset only."""
    bad = []
    for fn, path in _store_sources():
        for text, lineno in _sql_literals(path):
            low = text.lower()
            if " offset " in low or low.endswith(" offset") or " skip " in low:
                bad.append("%s:%d %r" % (fn, lineno, text[:60]))
    assert not bad, "OFFSET/SKIP pagination reaches SQLite in: %r" % bad


def test_the_source_guards_have_teeth():
    """A check that CANNOT fail is not a check that passed.

    The two guards above walk the AST and would report clean against an empty
    package, a syntax-error package, or a package whose SQL they failed to reach.
    So prove they fire: hand them a module that quotes both banned constructs in a
    docstring (must stay silent) and then executes them (must be caught)."""
    d = _tmpdir("guard-")
    clean = os.path.join(d, "clean.py")
    with open(clean, "w", encoding="utf-8") as f:
        f.write('"""Never use count(*), and never use LIMIT 10 OFFSET 500."""\n'
                'def f(cur):\n'
                '    raise ValueError("OFFSET is banned; count(*) is banned")\n'
                'def g(cur):\n'
                '    return cur.execute("SELECT id FROM vertex WHERE id > ? LIMIT ?")\n')
    assert _hits(clean) == [], (
        "guards fire on prose/error messages — they would force the docs silent")

    dirty = os.path.join(d, "dirty.py")
    with open(dirty, "w", encoding="utf-8") as f:
        f.write('def f(cur, t, n):\n'
                '    a = cur.execute("SELECT count(*) FROM vertex")\n'
                '    b = cur.execute("SELECT id FROM vertex LIMIT 10 OFFSET 500")\n'
                '    c = cur.execute("SELECT id FROM %s" % t + " LIMIT 5 OFFSET 9")\n'
                '    return a, b, c\n')
    hits = _hits(dirty)
    assert len(hits) == 3, (
        "guards missed a violation (concatenated SQL is the one that hides): %r" % hits)


def _hits(path):
    out = []
    for text, lineno in _sql_literals(path):
        low = text.lower()
        if "count(*)" in low.replace(" ", "") or " offset " in low or " skip " in low:
            out.append((lineno, text[:60]))
    return out


def test_counters_track_inserts_updates_and_deletes():
    L = _fresh()
    L.artifacts.put_many([_doc(i, collection_id="c1", state="committed") for i in range(20)])
    L.artifacts.put_many([_doc(100 + i, collection_id="c1", state="draft") for i in range(5)])
    assert L.artifacts.count() == 25
    assert L.artifacts.count(state="committed") == 20
    assert L.artifacts.count_in_collection("c1") == 25
    assert L.artifacts.count_in_collection("c1", committed_only=True) == 20
    assert L.artifacts.count_by_content_type("text/markdown") == 25

    L.artifacts.put_artifact(_doc(0, collection_id="c1", state="draft"))   # re-state
    assert L.artifacts.count() == 25, "upsert must not double-count"
    assert L.artifacts.count(state="committed") == 19
    assert L.artifacts.count_in_collection("c1", committed_only=True) == 19

    L.artifacts.delete_artifact("a0001")
    assert L.artifacts.count() == 24
    assert L.artifacts.count_in_collection("c1") == 24


def test_count_missing_field_is_incremental_and_exact():
    """genesis._count_null. Maintained as a counter, because `<field> IS NULL` over
    the corpus has no index that turns it into a seek — it is a scan or a counter,
    and the scan is not measured at 6.24M rows."""
    L = _fresh()
    L.artifacts.put_many([_doc(i, cited_from="c", provenance="p") for i in range(10)])
    L.artifacts.put_many([_doc(100 + i, provenance="p") for i in range(4)])      # no cited_from
    L.artifacts.put_many([_doc(200 + i) for i in range(3)])                      # neither
    assert L.artifacts.count_missing_field("cited_from") == 7
    assert L.artifacts.count_missing_field("provenance") == 3

    # a backfill filling the field in must decrement
    L.artifacts.put_artifact(_doc(200, cited_from="c", provenance="p"))
    assert L.artifacts.count_missing_field("cited_from") == 6
    assert L.artifacts.count_missing_field("provenance") == 2
    # and a delete must too
    L.artifacts.delete_artifact("a0201")
    assert L.artifacts.count_missing_field("provenance") == 1


def test_count_missing_field_treats_falsy_as_missing():
    """Must agree with `_scan_missing_field`'s fallback, which tests `not d.get(f)`.
    If the typed path used `is None` the audit would change answer by backend."""
    L = _fresh()
    L.artifacts.put_many([_doc(1, provenance=""), _doc(2, provenance=[]),
                          _doc(3, provenance=None), _doc(4, provenance="p")])
    assert L.artifacts.count_missing_field("provenance") == 3


def test_count_missing_field_agrees_with_the_keyset_fallback():
    """The two implementations must produce the same number — this is the property
    that makes adding the typed method safe. Mirrors the keyset fallback walk."""
    L = _fresh()
    L.artifacts.put_many([_doc(i, cited_from=("c" if i % 3 else ""),
                               state=("committed" if i % 2 else "draft"))
                          for i in range(60)])
    fallback, after = 0, ""
    while True:
        rows = L.artifacts.page_by_id(after=after, limit=7)
        if not rows:
            break
        fallback += sum(1 for r in rows if not (r["doc"] or {}).get("cited_from"))
        after = rows[-1]["id"]
    assert L.artifacts.count_missing_field("cited_from") == fallback, (
        "typed count %r disagrees with the keyset fallback %r"
        % (L.artifacts.count_missing_field("cited_from"), fallback))


def test_count_missing_field_default_is_the_safe_superset():
    """`state='committed'` cannot be expressed post-§2. The default must over-report:
    the consumer is `invariant_holds == 0`, so a superset can false-alarm but can
    never report clean over dirty."""
    L = _fresh()
    L.artifacts.put_many([_doc(1, state="committed"),          # missing, committed
                          _doc(2, state="draft"),              # missing, not committed
                          _doc(3, state="committed", provenance="p")])
    superset = L.artifacts.count_missing_field("provenance")
    exact = L.artifacts.count_missing_field("provenance", committed_only=True)
    assert exact == 1, exact
    assert superset == 2, superset
    assert superset >= exact, "the default must never under-report"


def test_count_missing_field_returns_a_bare_int_for_the_probe():
    """`genesis._scan_missing_field` does `int(typed(field))`. A capped dict would
    raise TypeError there and be swallowed into None — the method would appear to
    do nothing. Safe only because the counter is exact, with no cap to hide."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(5)])
    v = L.artifacts.count_missing_field("provenance")
    assert isinstance(v, int) and not isinstance(v, dict)
    assert int(v) == 5


def test_count_missing_field_rejects_unaudited_fields():
    """A closed set: `field` is validated against `NULL_AUDIT_FIELDS` rather than
    interpolated into SQL directly, so there is exactly one derivation of which
    fields are audited, not two that can silently disagree."""
    L = _fresh()
    from mantle.db import NULL_AUDIT_FIELDS, NullAuditField
    assert set(NULL_AUDIT_FIELDS) == {"cited_from", "provenance"}
    assert L.artifacts.count_missing_field(NullAuditField.PROVENANCE) == 0
    for bad in ("state", "id", "doc", "cited", "provenance_x", ""):
        try:
            L.artifacts.count_missing_field(bad)
            raise AssertionError("accepted unaudited field %r" % bad)
        except ValueError:
            pass


def test_verify_counters_detects_drift():
    """The counters are load-bearing — `count_missing_field` is the provenance
    audit — so drift must be detectable. A number nothing can check is not a
    measurement."""
    L = _fresh()
    L.artifacts.put_many([_doc(i, provenance=("p" if i % 2 else "")) for i in range(20)])
    clean = L.artifacts.verify_counters()
    assert clean["drift"] == {}, clean["drift"]
    assert clean["scanned"] == 20

    with L.db.write() as cur:                      # corrupt a counter behind its back
        cur.execute("UPDATE counter SET n = 999 WHERE name = 'missing:provenance'")
    bad = L.artifacts.verify_counters()
    assert bad["drift"].get("missing:provenance") == (999, 10), bad["drift"]

    fixed = L.artifacts.verify_counters(repair=True)
    assert fixed["repaired"] is True
    assert L.artifacts.verify_counters()["drift"] == {}
    assert L.artifacts.count_missing_field("provenance") == 10


def test_collection_count_is_per_collection_not_global():
    """`count_in_collection` must stay scoped to the collection, not the whole
    corpus — `advance_curriculum`'s resume offset depends on the count matching
    the collection it names."""
    L = _fresh()
    L.artifacts.put_many([_doc(i, collection_id="c1") for i in range(10)])
    L.artifacts.put_many([_doc(100 + i, collection_id="c2") for i in range(3)])
    assert L.artifacts.count() == 13
    assert L.artifacts.count_in_collection("c1") == 10
    assert L.artifacts.count_in_collection("c2") == 3
    assert L.artifacts.count_in_collection("nope") == 0


# ── 7. work pool — the typed replacements ────────────────────────────────────

TASK_CT = "application/vnd.agience.task+json"


def _task(i, **kw):
    d = {"id": "task-%03d" % i, "content_type": TASK_CT, "status": "pending",
         "priority": 0, "operator": "op.test", "task_key": "k%d" % i}
    d.update(kw)
    return d


def test_pending_window_is_not_silently_empty():
    """The pending window must return rows ordered by priority DESC. An empty or
    misordered window reads as a healthy queue in `queue_stats`'s pending counts
    while nothing is actually being claimed in priority order."""
    L = _fresh()
    L.artifacts.put_many([_task(i, priority=i % 3) for i in range(20)])
    win = L.artifacts.pending_window(TASK_CT, limit=8)
    assert len(win) == 8, "pending window came back empty/short: %r" % win
    assert [r["priority"] for r in win] == sorted(
        [r["priority"] for r in win], reverse=True), "priority DESC not honoured"


def test_pending_window_respects_retry_backoff():
    L = _fresh()
    L.artifacts.put_many([_task(1, next_retry_at="2030-01-01T00:00:00"),
                          _task(2, next_retry_at="2000-01-01T00:00:00"),
                          _task(3)])
    ids = {r["id"] for r in L.artifacts.pending_window(TASK_CT, now_iso="2026-07-20T00:00:00")}
    assert ids == {"task-002", "task-003"}, ids


def test_try_claim_is_atomic_exactly_one_winner():
    L = _fresh()
    L.artifacts.put_artifact(_task(1))
    wins = []
    errs = []

    def worker(w):
        try:
            if L.artifacts.try_claim("task-001", worker_id="w%d" % w, now_iso="t"):
                wins.append(w)
        except Exception as e:                       # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs, errs
    assert len(wins) == 1, "%d workers won the same task" % len(wins)
    assert L.artifacts.count_by_status(TASK_CT)["claimed"] == 1
    assert L.artifacts.count_by_status(TASK_CT)["pending"] == 0


def test_status_column_and_doc_never_disagree():
    """`status` must survive a re-upsert through `put_many`, exactly as a mesh
    consume performs it. A bulk path that resets the omitted `status` column to
    NULL would make a claimed task invisible to `claim`, `reclaim_stale`, and
    `queue_stats` simultaneously, and permanently orphaned."""
    L = _fresh()
    L.artifacts.put_artifact(_task(1))
    L.artifacts.try_claim("task-001", worker_id="w1", now_iso="t1")
    assert L.artifacts.get_artifact("task-001")["status"] == "claimed"
    assert L.artifacts.claimed(TASK_CT)[0]["claimed_by"] == "w1"
    # re-upsert through the bulk path, exactly as a mesh consume would
    doc = L.artifacts.get_artifact("task-001")
    L.artifacts.put_many([doc])
    assert L.artifacts.get_artifact("task-001")["status"] == "claimed"
    assert len(L.artifacts.claimed(TASK_CT)) == 1, "task went invisible after put_many"
    assert L.artifacts.count_by_status(TASK_CT)["claimed"] == 1


def test_lease_renew_release_and_terminal_window():
    L = _fresh()
    L.artifacts.put_many([_task(i) for i in range(5)])
    assert L.artifacts.try_claim("task-000", worker_id="w1", now_iso="t1")
    assert L.artifacts.renew_lease("task-000", now_iso="t2") is True
    assert L.artifacts.claimed(TASK_CT)[0]["claimed_at"] == "t2"
    assert L.artifacts.release("task-000") is True
    assert L.artifacts.renew_lease("task-000", now_iso="t3") is False, (
        "renewing an unclaimed lease must report False, not swallow it")
    assert L.artifacts.count_by_status(TASK_CT)["pending"] == 5

    L.artifacts.put_many([_task(9, status="done", completed_at="2026-01-0%d" % (i + 1))
                          for i in range(1)])
    L.artifacts.put_many([_task(10 + i, status="done",
                                completed_at="2026-01-%02d" % (i + 1)) for i in range(5)])
    L.artifacts.put_many([_task(20, status="failed", completed_at="2026-02-01")])
    recent = L.artifacts.recent_terminal(TASK_CT, limit=3)
    assert len(recent) == 3
    assert recent[0]["completed_at"] == "2026-02-01", recent
    assert [r["completed_at"] for r in recent] == sorted(
        [r["completed_at"] for r in recent], reverse=True)


def test_active_claims_shape():
    L = _fresh()
    L.artifacts.put_many([_task(i) for i in range(3)])
    L.artifacts.try_claim("task-001", worker_id="lt2-9", now_iso="t")
    L.artifacts.try_claim("task-002", worker_id="w71-1", now_iso="t")
    act = L.artifacts.active_claims(TASK_CT)
    assert [a["claimed_by"] for a in act] == ["lt2-9", "w71-1"]
    assert act[0]["operator"] == "op.test" and act[0]["task_key"] == "k1"


# ── 8. merkle ────────────────────────────────────────────────────────────────

def test_reshard_to_natural_leaves_preserves_the_tree():
    """The leaf count is DERIVED, not a constant: reshard() re-stamps `_leaf` and rebuilds the tree at
    `natural_leaves(corpus)` (the sqrt-law count). Afterwards the incremental tree still matches a
    full rescan, both tables moved in lockstep, and the resolution persists across a re-open."""
    import mantle.db.constants as K
    from mantle.db import open_lattice
    L = _fresh()
    for i in range(300):
        L.artifacts.put_artifact(_doc(i))
    L.graph.add_edges([("a%04d" % i, "a%04d" % ((i + 1) % 300), "lineage", {}) for i in range(300)])
    n = L.artifacts.count() + L.graph.count_edges()
    r = L.artifacts.reshard(graph=L.graph)
    assert r["resharded"] and r["leaves"] == K.natural_leaves(n)
    assert L.artifacts.leaves == L.graph.leaves == K.natural_leaves(n)          # lockstep
    inc = L.artifacts.merkle_leaves()
    L.artifacts.rebuild_merkle()
    assert inc == L.artifacts.merkle_leaves()                                   # incremental == rescan
    assert L.artifacts.maybe_reshard(graph=L.graph)["resharded"] is False       # idempotent
    L2 = open_lattice(L.db.path, origin=L.artifacts.origin)                     # persists across re-open
    assert L2.artifacts.leaves == K.natural_leaves(n)


def test_edge_merkle_incremental_matches_full_rescan():
    """One tree covers vertices and edges. Edge writes/updates/deletes XOR a node-invariant
    edge_hash into the same leaf_digest, so incremental maintenance must still match a full rescan
    once edges are present — the invariant that lets a fresh node converge on edges by Merkle alone."""
    L = _fresh()
    for i in range(20):
        L.artifacts.put_artifact(_doc(i))
    L.graph.add_edges([("a%04d" % i, "a%04d" % ((i + 1) % 20), "lineage", {}) for i in range(20)])
    L.graph.add_edges([("a0000", "a0001", "lineage", {"is_origin": 1, "force": "grant"})])  # prop update
    L.graph.delete_edge("a0005", "a0006", "lineage")                                        # XOR-out
    incremental = L.artifacts.merkle_leaves()
    L.artifacts.rebuild_merkle()
    assert incremental == L.artifacts.merkle_leaves(), "edge incremental merkle drifted from rescan"


def test_incremental_merkle_matches_full_rescan():
    """Incremental maintenance is required, not optional: a full-rescan publish
    runs well below the corpus's growth rate on a large, actively-written store."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(200)])
    L.artifacts.put_artifact(_doc(5, content="updated"))
    L.artifacts.delete_artifact("a0007")
    incremental = L.artifacts.merkle_leaves()
    L.artifacts.rebuild_merkle()
    assert incremental == L.artifacts.merkle_leaves(), "incremental merkle drifted from rescan"


def test_row_hash_uses_seq_not_rev():
    """Contract RESOLVED-1. Different `_seq` -> different row hash, so a mutation
    changes its leaf just as a new artifact does."""
    assert row_hash("a", 1) != row_hash("a", 2)
    assert row_hash("a", 1) == row_hash("a", 1)
    assert row_hash("a", None) == row_hash("a", 0), "absent seq hashes as 0, not skipped"


# ── 9. shim-replacement call sites ───────────────────────────────────────────

def test_list_by_content_type_reports_truncation():
    """A truncated page must be distinguishable from a complete one — `len(docs)
    <= cap` alone cannot tell truncation from a corpus that genuinely fits, so
    `exhaustive` reports which case occurred."""
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(50)])
    docs, exhaustive = L.artifacts.list_by_content_type("text/markdown", cap=1000)
    assert len(docs) == 50 and exhaustive is True
    docs, exhaustive = L.artifacts.list_by_content_type("text/markdown", cap=10)
    assert len(docs) == 10 and exhaustive is False, "truncation must be visible"
    docs, exhaustive = L.artifacts.list_by_content_type("nope/none", cap=10)
    assert docs == [] and exhaustive is True


def test_list_by_doc_field():
    """`genesis._shards_done` depends on this returning every matching id; an
    empty result here reads as no shard finished and re-ingests the fleet."""
    L = _fresh()
    L.artifacts.put_many([
        _doc(1, content_type="x/shard-done", source_name="wikipedia-en", shard="s1"),
        _doc(2, content_type="x/shard-done", source_name="wikipedia-en", shard="s2"),
        _doc(3, content_type="x/shard-done", source_name="other", shard="s3")])
    got = L.artifacts.list_by_doc_field(content_type="x/shard-done",
                                        field="source_name", value="wikipedia-en")
    assert sorted(d["shard"] for d in got) == ["s1", "s2"], got
    try:
        L.artifacts.list_by_doc_field(content_type="x", field="a'; DROP--", value=1)
        raise AssertionError("expected ValueError on a non-identifier field")
    except ValueError:
        pass


def test_dst_ids_by_label():
    """`genesis._consolidated_members` depends on this returning every destination
    id for the label; an empty result makes every artifact look like a generator."""
    L = _fresh()
    L.graph.add_edges([("canon", "m1", "consolidates", {}),
                       ("canon", "m2", "consolidates", {}),
                       ("canon", "m3", "lineage", {})])
    assert sorted(L.graph.dst_ids_by_label("consolidates")) == ["m1", "m2"]
    assert L.graph.dst_ids_by_label("nope") == []


def test_descendants_terminates_on_a_cycle():
    L = _fresh()
    L.graph.add_edges([("a", "b", "L", {}), ("b", "c", "L", {}), ("c", "a", "L", {})])
    assert sorted(L.graph.descendants("a", "L")) == ["b", "c"]


def test_vertex_and_edge_commit_atomically():
    """One connection, one transaction — there is no window where the artifact
    exists without its edges."""
    L = _fresh()
    with L.db.write() as cur:                       # nested writes must not deadlock
        L.artifacts.put_artifact(_doc(1))
        L.graph.add_edges([("a0001", "a0002", "lineage", {})])
        assert cur is not None
    assert L.artifacts.get_artifact("a0001") is not None
    assert L.graph.count_edges() == 1
    seqs = sorted([L.artifacts.get_artifact("a0001")["_seq"],
                   L.graph.edges_of("a0001")[0]["_seq"]])
    assert seqs == [1, 2], "vertex and edge must share one proper-time sequence: %r" % seqs


# ── runner ───────────────────────────────────────────────────────────────────
def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok   %s" % name)
        except Exception as e:
            failed += 1
            import traceback
            print("FAIL %s: %s" % (name, e))
            traceback.print_exc()
    print("\n%d passed, %d failed, %d total" % (len(tests) - failed, failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child(sys.argv[2], sys.argv[3])            # subprocess arm of the mp test
        sys.exit(0)
    sys.exit(_main())


# ── §6B: the version lineage ────────────────────────────────────────────────────────────────
def test_first_version_is_its_own_root():
    """Mantle's rule (`entities/artifact.py:73`), honoured by the lattice store too: a doc
    that names no root is its own first version."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown", "state": "committed",
                    "content": "one", "created_by": "u",
                    "created_time": "2026-01-01T00:00:00+00:00"})
    assert L.db.read().execute(
        "SELECT root_id FROM vertex WHERE id='a1'").fetchone()[0] == "a1"


def test_the_lineage_handle_cannot_be_repointed():
    """A lineage handle that can be rewritten is not a lineage handle. Re-pointing it would move a
    row into a different version history while the rows it left behind still claim it — a split
    lineage that no query can detect, because both halves look internally consistent."""
    L = _fresh()
    base = {"content_type": "text/markdown", "state": "committed", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00"}
    L.artifacts.put_artifact(dict(base, id="a1", content="one"))
    L.artifacts.put_artifact(dict(base, id="a2", root_id="a1", content="two"))
    assert L.db.read().execute(
        "SELECT root_id FROM vertex WHERE id='a2'").fetchone()[0] == "a1"
    # a re-put that tries to hijack the lineage must not take effect
    L.artifacts.put_artifact(dict(base, id="a2", root_id="HIJACK", content="two-b"))
    assert L.db.read().execute(
        "SELECT root_id FROM vertex WHERE id='a2'").fetchone()[0] == "a1"


def test_versions_of_walks_the_lineage_in_seq_order():
    """The `y` colimit: one artifact, many versions. Ordered by `(_origin,_seq)` — the version
    identity — not by `created_time`, which is a writer's claim."""
    L = _fresh()
    base = {"content_type": "text/markdown", "state": "committed", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00"}
    L.artifacts.put_artifact(dict(base, id="a1", content="one"))
    L.artifacts.put_artifact(dict(base, id="a2", root_id="a1", content="two"))
    L.artifacts.put_artifact(dict(base, id="a3", root_id="a1", content="three"))
    assert [v["id"] for v in L.artifacts.versions_of("a1")] == ["a1", "a2", "a3"]
    assert L.artifacts.versions_of("nonexistent") == []


# ── the plural reads: get_many / versions_of_many ────────────────────────────

def _traced(L):
    """Capture every statement the read connection executes, so a claim about the number of
    round trips is measured rather than asserted from the source."""
    seen = []
    L.artifacts.db.conn().set_trace_callback(seen.append)
    return seen


def test_get_many_answers_only_for_the_ids_that_exist_in_the_caller_s_order():
    """A miss is a MISSING KEY, never an empty doc: `{}` and "not stored" are different
    answers, and a mapping that carried `None` for a miss would make a caller check for it.
    Repeats collapse — one read, one entry."""
    L = _fresh()
    for i in range(5):
        L.artifacts.put_artifact(_doc(i, content="c%d" % i))

    got = L.artifacts.get_many(["a0003", "nope", "a0001", "a0003", "", None])
    assert list(got) == ["a0003", "a0001"], "caller order, misses dropped, repeats collapsed"
    assert got["a0003"]["content"] == "c3"
    assert L.artifacts.get_many([]) == {}
    assert L.artifacts.get_many(["nope"]) == {}


def test_get_many_of_five_thousand_ids_is_a_supported_call_not_a_ceiling():
    """SQLite caps host parameters per statement — 999 on builds before 3.32. Binding the
    caller's whole list would make the same request work on one machine and raise
    `too many SQL variables` on another, so the read is chunked and the id count is the
    caller's business alone."""
    from mantle.db.vertex import IN_CHUNK
    n = 1200
    L = _fresh()
    L.artifacts.put_many([_doc(i) for i in range(n)], batch=500)
    present = ["a%04d" % i for i in range(n)]
    asked = present + ["missing-%04d" % i for i in range(5000 - n)]
    assert len(asked) == 5000

    seen = _traced(L)
    try:
        got = L.artifacts.get_many(asked)
    finally:
        L.artifacts.db.conn().set_trace_callback(None)

    assert list(got) == present, "every stored id came back, in the order it was asked for"
    selects = [s for s in seen if "FROM vertex WHERE id IN" in s]
    assert len(selects) == -(-len(asked) // IN_CHUNK) == 6, (
        "5,000 ids must be a fixed number of chunked statements: %d" % len(selects))
    assert IN_CHUNK <= 999, (
        "a chunk must stay under the LOWEST SQLITE_MAX_VARIABLE_NUMBER (999, pre-3.32), not "
        "under whichever library this process happens to link — otherwise the same call that "
        "passes here raises `too many SQL variables` on an older build")


def test_versions_of_many_returns_exactly_what_versions_of_returns_per_root():
    """The plural must not be a second opinion. Same rows, same `(_origin, _seq)` order within
    each lineage, and a root with no versions absent rather than mapped to `[]`."""
    L = _fresh()
    base = {"content_type": "text/markdown", "state": "committed", "content": "x"}
    L.artifacts.put_artifact(dict(base, id="r1"))
    L.artifacts.put_artifact(dict(base, id="r1v2", root_id="r1"))
    L.artifacts.put_artifact(dict(base, id="r2"))
    L.artifacts.put_artifact(dict(base, id="r1v3", root_id="r1"))

    got = L.artifacts.versions_of_many(["r1", "r2", "gone", "r1"])
    assert list(got) == ["r1", "r2"]
    for root in ("r1", "r2"):
        assert [d["id"] for d in got[root]] == [d["id"] for d in L.artifacts.versions_of(root)]
    assert [d["id"] for d in got["r1"]] == ["r1", "r1v2", "r1v3"]
    assert L.artifacts.versions_of_many([]) == {}


def test_versions_of_many_chunks_its_root_list_too():
    """The lineage read has the same host-parameter ceiling as the id read, and the same
    answer: chunk it."""
    from mantle.db.vertex import IN_CHUNK
    n = 1000
    L = _fresh()
    L.artifacts.put_many([_doc(i, root_id="a%04d" % i) for i in range(n)], batch=500)
    roots = ["a%04d" % i for i in range(n)] + ["absent-%04d" % i for i in range(2000)]

    seen = _traced(L)
    try:
        got = L.artifacts.versions_of_many(roots)
    finally:
        L.artifacts.db.conn().set_trace_callback(None)

    assert len(got) == n
    selects = [s for s in seen if "FROM vertex WHERE root_id IN" in s]
    assert len(selects) == -(-len(roots) // IN_CHUNK)
    assert IN_CHUNK <= 999


def test_backfill_root_id_reports_what_remains_not_just_what_it_filled():
    """`filled: 0` is ambiguous on its own (already complete, or never ran?), so the assertion that
    matters is `remaining == 0`."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown", "state": "committed",
                    "content": "one", "created_by": "u",
                    "created_time": "2026-01-01T00:00:00+00:00"})
    with L.db.write() as cur:                       # simulate a pre-migration row
        cur.execute("UPDATE vertex SET root_id=NULL WHERE id='a1'")

    dry = L.artifacts.backfill_root_id(dry_run=True)
    assert dry["pending"] is True and dry["complete"] is False
    assert L.db.read().execute(                     # dry run changed NOTHING
        "SELECT root_id FROM vertex WHERE id='a1'").fetchone()[0] is None

    r = L.artifacts.backfill_root_id()
    assert r["ok"] is True and r["filled"] == 1 and r["complete"] is True
    assert L.db.read().execute(
        "SELECT root_id FROM vertex WHERE id='a1'").fetchone()[0] == "a1"
    again = L.artifacts.backfill_root_id()                    # idempotent
    assert again["filled"] == 0 and again["complete"] is True


def test_revise_creates_a_new_version_and_never_destroys_the_old():
    """`revise()` must create a new version while leaving the prior row intact: a caller
    reading the original id afterward still sees the pre-revision content, archived."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown", "state": "committed",
                              "content": "VERSION ONE", "created_by": "u",
                              "created_time": "2026-01-01T00:00:00+00:00"})
    v2 = L.artifacts.revise("a1", {"content": "VERSION TWO"})

    assert v2["id"] != "a1" and v2["root_id"] == "a1"
    # both rows exist.
    assert len(L.artifacts.versions_of("a1")) == 2
    old = L.artifacts.get_artifact("a1")
    assert old["content"] == "VERSION ONE", "the prior version was destroyed"
    assert old["state"] == "archived" and old["superseded_by"] == v2["id"]
    # the head is the new version.
    assert L.artifacts.head_of("a1")["content"] == "VERSION TWO"


def test_revise_version_id_is_derived_not_counted():
    """A counter-derived id (`root~2`) would make two observers revising concurrently mint the
    same id for different content, and the store would silently keep one. The id is the
    fingerprint of the doc, so identical content is idempotent and different content cannot collide."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown", "state": "committed",
                              "content": "one", "created_by": "u",
                              "created_time": "2026-01-01T00:00:00+00:00"})
    v2 = L.artifacts.revise("a1", {"content": "two"})
    before = len(L.artifacts.versions_of("a1"))
    again = L.artifacts.revise(v2["id"], {"content": "two"})   # identical content
    assert again["id"] == v2["id"]
    assert len(L.artifacts.versions_of("a1")) == before, "a no-op revision minted a version"
    # different content -> a genuinely different id
    v3 = L.artifacts.revise(v2["id"], {"content": "three"})
    assert v3["id"] not in (v2["id"], "a1") and v3["root_id"] == "a1"


def test_revise_refuses_a_caller_supplied_state():
    """`state` is a reserved field on `revise()`."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "g1", "content_type": "application/vnd.agience.grant+json",
                              "state": "active", "grantee_id": "alice"})
    try:
        L.artifacts.revise("g1", {"state": "revoked"})
    except ValueError as e:
        assert "RESERVED" in str(e)
    else:
        raise AssertionError("revise() accepted `state` — it will be silently overwritten, and for "
                             "a grant that means re-granting revoked access")
    assert L.artifacts.get_artifact("g1")["state"] == "active"      # untouched, not half-applied


def test_revise_refuses_an_archived_version_instead_of_forking_the_lineage():
    """Calling `revise()` twice on the same original id must raise rather than fork the lineage
    into two live committed heads on one root_id — a silent fork would leave `head_of` still
    answering, hiding the divergence."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown", "content": "one"})
    v2 = L.artifacts.revise("a1", {"content": "two"})
    assert L.artifacts.get_artifact("a1")["state"] == "archived"
    try:
        L.artifacts.revise("a1", {"content": "three"})              # the stale-id call
    except ValueError as e:
        assert "ARCHIVED" in str(e) and "head_of" in str(e)         # names the fix
    else:
        raise AssertionError("revise() forked the lineage off an archived version")
    root = v2["root_id"]
    live = [d for d in L.artifacts.versions_of(root) if d.get("state") != "archived"]
    assert len(live) == 1, "expected exactly one live head, found %d" % len(live)
    # …and revising the actual head still works.
    v3 = L.artifacts.revise(L.artifacts.head_of(root)["id"], {"content": "three"})
    assert L.artifacts.head_of(root)["id"] == v3["id"]


def test_list_by_content_type_is_head_only_by_default():
    """Must return only the head version by default. An external consumer builds grant revocation
    on this read, so an archived pre-revoke grant answering here would still grant access."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown", "content": "one"})
    L.artifacts.revise("a1", {"content": "two"})                    # a1 -> archived, v2 -> head
    docs, exhaustive = L.artifacts.list_by_content_type("text/markdown")
    assert exhaustive
    assert [d["content"] for d in docs] == ["two"], "archived version answered a query"
    assert all(d.get("state") != "archived" for d in docs)


def test_list_by_content_type_can_still_enumerate_everything():
    """Replication and repair need the complete set, not the current one — `mesh/sync.py` subtracts
    an enumerated operational set, and silently dropping archived rows there would make a peer diff
    a leaf whose object 404s and stay permanently divergent."""
    L = _fresh()
    L.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown", "content": "one"})
    L.artifacts.revise("a1", {"content": "two"})
    docs, exhaustive = L.artifacts.list_by_content_type("text/markdown", include_archived=True)
    assert exhaustive and len(docs) == 2
    assert {d.get("state") for d in docs} == {"archived", "committed"}


def test_revise_refuses_an_unknown_artifact():
    """`revise()` raises for a lineage that does not exist, rather than silently minting an orphan
    version under one."""
    L = _fresh()
    try:
        L.artifacts.revise("nope", {"content": "x"})
        raise AssertionError("revise() minted a version under a lineage that does not exist")
    except KeyError:
        pass


# ── S3.8: the store must work with every remote source unreachable ────────────────────────────
def test_the_lattice_store_has_no_network_dependency():
    """Acceptance check S3.8 requires content end to end with every external store unreachable, so
    `mantle.db.vertex` must import and function with no client library for boto3,
    botocore, requests, aiohttp, or urllib3 available — an unconditional import of any of them
    would violate S3.8 by construction.

    This test poisons those modules rather than merely checking they are absent: a machine that
    happens to have boto3 installed would pass a grep-style check while the dependency still
    exists. Poisoning makes any import attempt fail immediately, rather than only on a node that
    has been disconnected for a decommission test."""
    import importlib
    import sys

    poisoned = ("boto3", "botocore", "requests", "aiohttp", "urllib3")
    saved = {m: sys.modules.get(m) for m in poisoned}
    dropped = [m for m in list(sys.modules) if m.split(".")[0] == "mantle"]
    saved_mantle = {m: sys.modules[m] for m in dropped}
    try:
        for m in poisoned:
            sys.modules[m] = None          # any `import m` now raises ImportError
        for m in dropped:
            del sys.modules[m]
        mod = importlib.import_module("mantle.db.vertex")
        assert hasattr(mod, "LatticeArtifactStore")
        # and it must function, not merely import
        lat = importlib.import_module("mantle.db")
        import os
        import tempfile
        L = lat.open_lattice(os.path.join(_tmpdir("s38-"), "s38.db"), origin="s38")
        L.artifacts.put_artifact({"id": "x1", "content_type": "text/markdown",
                                  "state": "committed", "content": "local only",
                                  "created_by": "u",
                                  "created_time": "2026-01-01T00:00:00+00:00"})
        assert L.artifacts.get_artifact("x1")["content"] == "local only"
    finally:
        for m, v in saved.items():
            if v is None:
                sys.modules.pop(m, None)
            else:
                sys.modules[m] = v
        sys.modules.update(saved_mantle)


# ── §6B.4: content decides versioning — no forced route ───────────────────────────────────────
def test_same_content_ref_is_a_redescribe_not_a_new_version():
    """The same content_ref means the bytes are unchanged, so updating a derived field
    (offer/context) is a redescribe, applied in place — no snapshot, no forked lineage. The
    offers pass does this 6.11M times, and it must stay a no-op-shaped in-place update."""
    L = _fresh()
    base = {"id": "a1", "content_type": "text/markdown", "state": "committed", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00", "content_ref": "cas/" + "a" * 64}
    L.artifacts.put_artifact(dict(base, context="first offer"))
    L.artifacts.put_artifact(dict(base, context="re-described offer"))   # same content_ref
    rows = L.db.read().execute("SELECT id FROM vertex WHERE id LIKE 'a1%'").fetchall()
    assert [r[0] for r in rows] == ["a1"], "a same-content re-describe minted a version: %s" % rows
    assert L.artifacts.get_artifact("a1")["context"] == "re-described offer"


def test_changed_content_ref_snapshots_the_prior_and_handle_stays_latest():
    """A different content_ref means the content changed, so `put_artifact` itself (no forced
    `revise()` call) snapshots the prior version under a derived archived id and keeps the
    handle pointing at the latest. Nothing is overwritten in the silent-loss sense."""
    L = _fresh()
    base = {"id": "a1", "content_type": "text/markdown", "state": "committed", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00"}
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "1" * 64, content="VERSION ONE"))
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "2" * 64, content="VERSION TWO"))

    # the handle holds the latest
    assert L.artifacts.get_artifact("a1")["content"] == "VERSION TWO"
    # the prior version is preserved (snapshot), archived, and points at the handle
    snaps = [json.loads(r[0]) for r in L.db.read().execute(
        "SELECT doc FROM vertex WHERE id LIKE 'a1@%'").fetchall()]
    assert len(snaps) == 1
    assert snaps[0]["content"] == "VERSION ONE"
    assert snaps[0]["state"] == "archived" and snaps[0]["superseded_by"] == "a1"
    assert snaps[0]["root_id"] == "a1"          # shares the lineage
    # "VERSION ONE" is still reachable — not silently lost.
    assert any(s["content"] == "VERSION ONE" for s in snaps)


def test_content_change_is_idempotent_no_snapshot_per_oscillation():
    """The snapshot id is derived from the old content_ref, so re-applying the same change does not
    mint a snapshot each time: two writes of the same content leave exactly one snapshot."""
    L = _fresh()
    base = {"id": "a1", "content_type": "text/markdown", "state": "committed", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00"}
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "1" * 64, content="ONE"))
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "2" * 64, content="TWO"))
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "2" * 64, content="TWO"))  # same again
    snaps = L.db.read().execute("SELECT count(*) FROM vertex WHERE id LIKE 'a1@%'").fetchone()[0]
    assert snaps == 1, "an idempotent re-put minted a duplicate snapshot"


def test_versioning_needs_no_forced_route_put_artifact_alone_suffices():
    """A caller that only ever calls `put_artifact` still gets full version retention: no
    `revise()` call is required, and no forced route exists to work around."""
    L = _fresh()
    base = {"id": "doc", "content_type": "text/markdown", "state": "committed", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00"}
    for i, ref in enumerate(("1", "2", "3")):
        L.artifacts.put_artifact(dict(base, content_ref="cas/" + ref * 64, content="v%d" % i))
    assert L.artifacts.get_artifact("doc")["content"] == "v2"        # handle = latest
    prior = L.db.read().execute("SELECT count(*) FROM vertex WHERE id LIKE 'doc@%'").fetchone()[0]
    assert prior == 2, "three contents should leave two archived priors, got %d" % prior


# ── §6B.3: only the head answers a query (archived snapshots hidden by default) ────────────────
def test_list_artifacts_hides_archived_snapshots_by_default():
    """After a content-decides revision (§6B.4), the prior version is archived. A default
    `list_artifacts` call must return only the head, or versioning double-answers."""
    L = _fresh()
    base = {"id": "a1", "content_type": "text/markdown", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00"}
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "1" * 64, content="ONE"))
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "2" * 64, content="TWO"))
    # a snapshot now exists (a1@...); default listing must not include it
    ids = [a["id"] for a in L.artifacts.list_artifacts(content_type="text/markdown")]
    assert ids == ["a1"], "default list returned an archived snapshot: %s" % ids
    contents = [a["content"] for a in L.artifacts.list_artifacts(content_type="text/markdown")]
    assert contents == ["TWO"], "the head is not the latest content"


def test_history_is_reachable_on_explicit_opt_in():
    """Hidden by default is not gone: `include_archived=True` (or an explicit `state='archived'`)
    returns the priors, so history is never lost — only quiet."""
    L = _fresh()
    base = {"id": "a1", "content_type": "text/markdown", "created_by": "u",
            "created_time": "2026-01-01T00:00:00+00:00"}
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "1" * 64, content="ONE"))
    L.artifacts.put_artifact(dict(base, content_ref="cas/" + "2" * 64, content="TWO"))
    allv = list(L.artifacts.list_artifacts(content_type="text/markdown", include_archived=True))
    assert {a["content"] for a in allv} == {"ONE", "TWO"}
    archived = list(L.artifacts.list_artifacts(state="archived"))
    assert [a["content"] for a in archived] == ["ONE"]


def test_head_only_is_a_noop_when_there_are_no_snapshots():
    """On a corpus where every artifact has one version, the default filter changes nothing."""
    L = _fresh()
    for i in range(3):
        L.artifacts.put_artifact({"id": "d%d" % i, "content_type": "text/markdown",
                                  "created_by": "u", "created_time": "2026-01-01T00:00:00+00:00",
                                  "content_ref": "cas/" + str(i) * 64, "content": "x"})
    ids = [a["id"] for a in L.artifacts.list_artifacts(content_type="text/markdown")]
    assert ids == ["d0", "d1", "d2"]


