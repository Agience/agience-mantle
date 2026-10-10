"""`system/manage_state_collapse` on a fixture store holding every class of legacy row.

The fixture is a real `LatticeDatabase` on disk:

    A   a draft-only lineage
    B   a committed row with a newer draft beside it, and the reverse (a newer committed row)
    C   two drafts under one root in one collection
    D   a draft replicated from another node (its `_origin` is not this one)
    B"  a local draft beside a committed row replicated from another node
    E   a draft whose upload never completed
        plus plain committed rows, a live row, an archived row, a stateless row and a grant,
        none of which is a draft

What is pinned: the report writes nothing; an apply settles this node's drafts (A, B, C, E, B")
by the newest-wins rule, archives only rows this node authored, relabels another node's rows in
place, and leaves every counter exact (`verify_counters` finds no drift); a second apply writes
nothing; the bulk relabel moves no `_seq` and no merkle leaf and rewrites under the write lock
only the ids it read before taking it; and the legacy draft index is reported and never
deleted.
"""
from __future__ import annotations

import json
import os

import pytest

from mantle.db import lattice_api as store
from mantle.system import manage_state_collapse as tool

ORIGIN = "node-a"


def _put(db, doc_id, root, state, collection="col-1", **extra):
    doc = {"id": doc_id, "root_id": root, "collection_id": collection, "content": doc_id,
           "created_by": "u-1"}
    if state is not None:
        doc["state"] = state
    doc.update(extra)
    return db.artifacts.put_artifact(doc)


@pytest.fixture
def db(tmp_path):
    d = store.LatticeDatabase(str(tmp_path / "legacy.db"), origin=ORIGIN)
    d.artifacts.put_artifact({"id": "col-1", "root_id": "col-1", "state": "committed",
                              "content_type": "application/vnd.agience.workspace+json"})
    _put(d, "a", "a", "draft")                                         # A
    _put(d, "b", "b", "committed")                                     # B: older committed
    _put(d, "b~2", "b", "draft")                                       #    newer draft
    _put(d, "f~1", "f", "draft")                                       # B': older draft
    _put(d, "f", "f", "committed")                                     #     newer committed
    _put(d, "c~1", "c", "draft")                                       # C
    _put(d, "c~2", "c", "draft")
    d.artifacts.put_artifact({"id": "d", "root_id": "d", "collection_id": "col-1",
                              "state": "draft", "content": "d",
                              "_origin": "node-z", "_seq": 7}, stamp_rev=False)  # D
    _put(d, "h", "h", "draft")                                         # B": local draft
    d.artifacts.put_artifact({"id": "h-peer", "root_id": "h", "collection_id": "col-1",
                              "state": "committed", "content": "h-peer",
                              "_origin": "node-z", "_seq": 8}, stamp_rev=False)  # beside a peer's
    _put(d, "e", "e", "draft",                                         # E
         context=json.dumps({"upload": {"status": "uploading", "mode": "proxied"}}))
    for i in range(7):                                                 # plain committed rows
        _put(d, "p-%d" % i, "p-%d" % i, "committed")
    _put(d, "l", "l", "live")
    _put(d, "old", "old", "archived")
    _put(d, "bare", "bare", None)
    d.artifacts.put_artifact({"id": "g-1", "content_type": "application/vnd.agience.grant+json",
                              "state": "active", "resource_id": "col-1"})
    return d


def _snapshot(db):
    """Every row's doc and `_seq`, plus every counter — what "writes nothing" is checked against."""
    rows = {r["id"]: (r["_seq"], r["doc"]) for r in db.conn.read().execute(
        "SELECT id, _seq, doc FROM vertex")}
    counters = {r["name"]: r["n"] for r in db.conn.read().execute("SELECT name, n FROM counter")}
    return rows, counters


def _state(db, doc_id):
    return (db.artifacts.get_artifact(doc_id) or {}).get("state")


def test_the_report_classifies_every_class(db):
    rep = tool.report(db, measure_prefixes=False)
    by_root = {g["root_id"]: g for g in rep["groups"]}
    assert {r: g["class"] for r, g in by_root.items()} == {
        "a": "A", "b": "B", "f": "B", "c": "C", "d": "D", "e": "E", "h": "B"}
    assert rep["classes"] == {"A": 1, "B": 3, "C": 1, "D": 1, "E": 1}
    assert by_root["b"]["winner"] == "b~2" and by_root["b"]["losers"] == ["b"]
    assert by_root["f"]["winner"] == "f" and by_root["f"]["losers"] == ["f~1"]
    assert by_root["c"]["winner"] == "c~2" and by_root["c"]["losers"] == ["c~1"]
    # `(_origin, _seq)` order puts the peer's row last: it is the newest, and it is not ours.
    assert (by_root["h"]["winner"], by_root["h"]["winner_is_local"]) == ("h-peer", False)
    assert by_root["h"]["losers"] == ["h"] and by_root["h"]["foreign_losers"] == []
    assert rep["counters"] == {"live": 1, "committed": 11, "draft": 8, "archived": 1}
    assert rep["committed_to_relabel"] == 11
    assert rep["foreign_draft_rows"] == 1


def test_the_report_writes_nothing(db):
    before = _snapshot(db)
    tool.report(db, measure_prefixes=False)
    assert _snapshot(db) == before


def test_apply_settles_every_class_and_leaves_the_counters_exact(db):
    rep = tool.report(db, measure_prefixes=False)
    reindexed = []
    out = tool.apply(db, rep["groups"], batch=3, reindex=lambda art: reindexed.append(art.id))

    assert out["settled"] == {"A": 1, "B": 3, "C": 1, "E": 1}
    assert out["skipped_d"] == 1
    assert out["archived"] == 4
    assert out["foreign_drafts_relabelled"] == 1
    # 7 plain + the collection + the peer's committed row; B' wrote its winner live itself.
    assert out["relabelled"] == 9
    assert sorted(reindexed) == ["a", "b~2", "c~2", "e", "f", "h-peer"]

    # A, E: live. B, C: the newest row is live and each older one points at it.
    for doc_id in ("a", "e", "b~2", "f", "c~2", "h-peer"):
        assert _state(db, doc_id) == "live", doc_id
    for loser, winner in (("b", "b~2"), ("f~1", "f"), ("c~1", "c~2"), ("h", "h-peer")):
        doc = db.artifacts.get_artifact(loser)
        assert (doc["state"], doc.get("superseded_by")) == ("archived", winner), loser
    # Another node's rows are relabelled in place: same author, same proper time.
    for doc_id, seq in (("d", 7), ("h-peer", 8)):
        doc = db.artifacts.get_artifact(doc_id)
        assert (doc["state"], doc["_origin"], doc["_seq"]) == ("live", "node-z", seq), doc_id
    # Everything else committed is live; nothing outside the artifact axis moved.
    assert {_state(db, "p-%d" % i) for i in range(7)} == {"live"}
    assert _state(db, "old") == "archived"
    assert _state(db, "bare") is None
    assert _state(db, "g-1") == "active"

    ver = tool.verify(db)
    assert ver == {"committed": 0, "draft": 0, "local_drafts": [], "ok": True}
    audit = db.artifacts.verify_counters()
    assert audit["drift"] == {}, audit["drift"]


def test_a_second_apply_is_a_no_op(db):
    tool.apply(db, tool.report(db, measure_prefixes=False)["groups"], reindex=lambda art: None)
    before = _snapshot(db)

    rep = tool.report(db, measure_prefixes=False)
    assert rep["classes"] == {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    assert rep["committed_to_relabel"] == 0
    out = tool.apply(db, rep["groups"], reindex=lambda art: pytest.fail("reindexed again"))

    assert out["relabelled"] == 0 and out["archived"] == 0
    assert out["foreign_drafts_relabelled"] == 0
    assert _snapshot(db) == before


def test_the_bulk_relabel_moves_no_seq_and_no_merkle_leaf(db):
    """A relabel is not an authored event: `_seq` and the merkle leaves are what a peer compares,
    and neither may move, or every relabelled row is republished to the mesh."""
    a = db.artifacts
    seqs = {r["id"]: r["_seq"] for r in db.conn.read().execute("SELECT id, _seq FROM vertex")}
    leaves = a.merkle_leaves()
    committed = a.count_recorded("committed")

    out = a.relabel_state("committed", "live", batch=4)

    assert out == {"relabelled": committed, "batches": -(-committed // 4)}
    assert {r["id"]: r["_seq"] for r in db.conn.read().execute(
        "SELECT id, _seq FROM vertex")} == seqs
    assert a.merkle_leaves() == leaves
    assert a.count_recorded("committed") == 0
    assert a.verify_counters()["drift"] == {}


def test_the_bulk_relabel_rewrites_only_a_legacy_value_to_live(db):
    for old, new in (("committed", "archived"), ("live", "live"), ("archived", "live")):
        with pytest.raises(ValueError):
            db.artifacts.relabel_state(old, new)


def test_a_draft_relabel_moves_every_counter_it_touches(db):
    """A legacy draft is outside the `committed_only` buckets and enters them when relabelled,
    so the draft path applies the per-row deltas; `exclude_origin` leaves this node's rows."""
    a = db.artifacts
    out = a.relabel_state("draft", "live", exclude_origin=ORIGIN)
    assert out["relabelled"] == 1
    assert _state(db, "d") == "live" and _state(db, "a") == "draft"
    assert a.verify_counters()["drift"] == {}


def test_a_local_draft_left_unsettled_fails_verify(db):
    """The failure a skipped group would produce is reported by name, not absorbed."""
    tool.apply(db, [], reindex=lambda art: None)          # relabel only; no group settled
    ver = tool.verify(db)
    assert not ver["ok"]
    assert sorted(ver["local_drafts"]) == ["a", "b~2", "c~1", "c~2", "e", "f~1", "h"]


def test_the_write_locked_section_touches_only_the_ids_it_read(db):
    """The candidate scan walks the table (there is no index on `$.state`), so it runs before
    the write lock is taken. Inside `BEGIN IMMEDIATE … COMMIT` every statement names its rows by
    id: no statement there scans for candidates."""
    conn = db.conn.conn()
    trace: list = []
    conn.set_trace_callback(trace.append)
    try:
        db.artifacts.relabel_state("committed", "live", batch=4)
    finally:
        conn.set_trace_callback(None)

    inside, locked, scans_outside = False, [], 0
    for stmt in trace:
        s = " ".join(stmt.split()).upper()
        if s.startswith("BEGIN IMMEDIATE"):
            inside = True
        elif s.startswith(("COMMIT", "ROLLBACK")):
            inside = False
        elif inside and "FROM VERTEX" in s or inside and s.startswith("UPDATE VERTEX"):
            locked.append(s)
        elif "ORDER BY ID LIMIT" in s:
            scans_outside += 1
    assert scans_outside >= 3, "the positive control: the candidate reads were traced"
    assert locked, "the rewrite was not traced"
    assert all(" ID IN (" in s for s in locked), [s for s in locked if " ID IN (" not in s]
    assert not any("ORDER BY ID LIMIT" in s for s in locked)


def test_the_bulk_relabel_can_be_bounded_and_resumed(db):
    a = db.artifacts
    first = a.relabel_state("committed", "live", batch=2, limit=3)
    assert first["relabelled"] == 3
    assert a.count_recorded("committed") == 8
    rest = a.relabel_state("committed", "live", batch=2)
    assert rest["relabelled"] == 8
    assert a.verify_counters()["drift"] == {}


def test_the_cli_reports_by_default_and_prints_the_draft_index_deletion(db, tmp_path,
                                                                         monkeypatch, capsys):
    """`main()` with no flag writes nothing; with `--apply` it migrates, and names the legacy
    draft index for the operator without touching it."""
    from mantle.db import backend
    from mantle.search.mantle import wiring

    monkeypatch.setattr(backend, "_HANDLE", db)
    monkeypatch.setenv("MANTLE_CELL_DIR", str(tmp_path / "cells"))
    monkeypatch.setenv("MANTLE_SSE_DIR", str(tmp_path / "sse"))
    monkeypatch.setattr(wiring, "edge_object_storage_is_configured", lambda: False)
    cells = tmp_path / "cells" / "mantle-cells-draft" / "p1"
    cells.mkdir(parents=True)
    (cells / "cell.enc").write_bytes(b"x" * 100)
    (tmp_path / "sse").mkdir()
    sse_db = tmp_path / "sse" / "mantle-sse-draft.db"
    sse_db.write_bytes(b"y" * 40)
    monkeypatch.setattr(tool, "_default_reindex", lambda art: None)

    before = _snapshot(db)
    assert tool.main([]) == 0
    printed = capsys.readouterr().out
    assert "REPORT ONLY" in printed
    assert "100 bytes" in printed and "40 bytes" in printed
    assert _snapshot(db) == before

    assert tool.main(["--apply"]) == 0
    printed = capsys.readouterr().out
    assert "-> OK" in printed
    assert "rm -rf %s" % os.path.join(str(tmp_path / "cells"), "mantle-cells-draft") in printed
    assert "rm -f %s" % str(sse_db) in printed
    assert cells.exists() and sse_db.exists(), "the legacy draft index was deleted"


def test_an_edit_between_the_read_and_the_write_is_not_overwritten(db):
    """Another writer edits a winner while the migration settles it. The group is read and
    written in one write transaction, so the edit lands after the settle rather than being
    overwritten by a copy read before it."""
    import threading

    rep = tool.report(db, measure_prefixes=False)
    groups = [g for g in rep["groups"] if g["root_id"] == "a"]
    path = db.conn.path

    def _edit_elsewhere():
        other = store.LatticeDatabase(path, origin=ORIGIN)
        doc = other.artifacts.get_artifact("a")
        other.artifacts.put_artifact(dict(
            {k: v for k, v in doc.items() if k not in ("_origin", "_seq")}, content="edited"))

    fired: dict = {}

    def _hook(read):
        def hooked(key, *a, **k):
            rows = read(key, *a, **k)
            if key == "a" and not fired:
                fired["t"] = threading.Thread(target=_edit_elsewhere)
                fired["t"].start()
                fired["t"].join(0.5)           # lets an unguarded write finish first
            return rows
        return hooked

    db.artifacts.versions_of = _hook(db.artifacts.versions_of)
    db.artifacts.get_artifact = _hook(db.artifacts.get_artifact)
    tool.apply(db, groups, reindex=lambda art: None)
    fired["t"].join(10)
    assert not fired["t"].is_alive()
    assert store.LatticeDatabase(path, origin=ORIGIN).artifacts.get_artifact("a")["content"] \
        == "edited", "the migration overwrote an edit made while it was settling the row"


@pytest.mark.parametrize("old", ["committed", "draft"])
def test_the_relabel_stays_exact_when_a_row_changes_between_its_read_and_its_write(db, old):
    """Another writer changes two candidates after the unlocked read and before the locked
    rewrite: one to `archived`, one to `live`. Neither is rewritten, and the counters move by
    exactly what was."""
    import contextlib

    a = db.artifacts
    for i in range(6):
        a.put_artifact({"id": "q%d" % i, "root_id": "q%d" % i, "collection_id": "c",
                        "state": old, "content": "x", "content_type": "text/plain"})
    before = a.count_recorded(old)
    orig = a.db.write
    fired = {"v": False}

    @contextlib.contextmanager
    def hooked(*args, **kw):
        with orig(*args, **kw) as cur:
            if not fired["v"]:
                fired["v"] = True
                for doc_id, state in (("q2", "archived"), ("q4", "live")):
                    d = a.get_artifact(doc_id)
                    d["state"] = state
                    a.put_artifact(d)
            yield cur
    a.db.write = hooked
    try:
        out = a.relabel_state(old, "live", batch=100, exclude_origin="nobody")
    finally:
        a.db.write = orig
    assert out["relabelled"] == before - 2
    assert a.count_recorded(old) == 0
    assert a.get_artifact("q2")["state"] == "archived"
    assert a.verify_counters()["drift"] == {}
