"""The per-relation edge extent is a counter; `count(*)` never reaches the lattice.

`crystal.ontology.seed_lattice._relation_signature` reads a relation's extent from this counter.
No edge index leads with `label` (they are `(src,label)` and `(dst,label)`), so
`SELECT COUNT(*) FROM edge WHERE label=?` is unservable by any index — it dereferences the whole
edge table to produce one integer, the query shape `db/schema.py`'s own header bans.

Four things have to be true for the counter to be worth having, and each has a test that can fail.
Three of them are here, because they are about mantle's counter:

  1. the counter tracks the write path in both directions — a counter that only ever increases is
     a leak that reads as growth;
  2. an absent counter reads as unknown, never zero — a relation with a million edges and one that
     has never been counted must give different answers;
  3. the backfill measures a store written before the counter existed, without `count(*)`;

The fourth — the oracle, proving the signature unchanged against the `count(*)` it replaced on a
store small enough that both paths can be computed — is about the reader, which lives in
`agience-crystal/tests/test_relation_signature.py`, along with the two tests pinning the
signature's own absence handling and a `crystal/src` copy of section 5's scan.
"""
from __future__ import annotations

import ast
import os

import pytest

from mantle.db import open_lattice
from mantle.db.schema import c_edge_label, c_edge_label_built

# The reader lives in `crystal.ontology.seed_lattice`; this file does not import upward. mantle
# is the standalone database and stays testable with nothing above it installed. Everything below
# tests the counter — mantle's own — against mantle's own store.

# ── fixtures ─────────────────────────────────────────────────────────────────────────────────


def _fresh(tmp_path, name="lat.db"):
    return open_lattice(str(tmp_path / name), origin="test-origin", leaves=16)


def _corpus(L):
    """A small corpus with three labels, an inverse pair, and a symmetric relation."""
    L.graph.add_edges([
        ("wn-dog", "wn-canine", "hypernym", {}),
        ("wn-cat", "wn-feline", "hypernym", {}),
        ("wn-canine", "wn-dog", "hyponym", {}),
        ("wn-feline", "wn-cat", "hyponym", {}),
        ("wn-hot", "wn-cold", "antonym", {}),
        ("wn-cold", "wn-hot", "antonym", {}),
    ])


def _strip_to_legacy(L):
    """Make the store look like one whose edges were written before the counter existed."""
    with L.db.write() as cur:
        cur.execute("DELETE FROM counter WHERE name LIKE ?", (c_edge_label("") + "%",))
        cur.execute("DELETE FROM counter WHERE name = ?", (c_edge_label_built(),))


# ── 1. the counter tracks the write path, in BOTH directions ─────────────────────────────────


def test_per_label_counter_follows_adds_and_removes(tmp_path):
    L = _fresh(tmp_path)
    assert L.graph.edge_labels_measured(), "a store with no edges is trivially fully counted"
    assert L.graph.count_edges_by_label("hypernym") == 0, "measured zero on an empty store"

    _corpus(L)
    assert L.graph.count_edges_by_label("hypernym") == 2
    assert L.graph.count_edges_by_label("hyponym") == 2
    assert L.graph.count_edges_by_label("antonym") == 2
    assert L.graph.count_edges_by_label("meronym") == 0, "a label never written is a measured 0"
    assert L.graph.count_edges() == 6, "the total and the extents are maintained together"

    # The direction that matters: a counter that only ever rises is a leak that reads as growth.
    assert L.graph.delete_edge("wn-dog", "wn-canine", "hypernym") is True
    assert L.graph.count_edges_by_label("hypernym") == 1
    assert L.graph.count_edges_by_label("hyponym") == 2, "an unrelated extent must not move"
    assert L.graph.count_edges() == 5

    assert L.graph.delete_edge("wn-cat", "wn-feline", "hypernym") is True
    assert L.graph.count_edges_by_label("hypernym") == 0, "back to zero, not stuck at the peak"

    # And it comes back up.
    L.graph.add_edge("wn-dog", "wn-canine", "hypernym")
    assert L.graph.count_edges_by_label("hypernym") == 1

    # A delete of an edge that is not there changes nothing.
    assert L.graph.delete_edge("wn-x", "wn-y", "hypernym") is False
    assert L.graph.count_edges_by_label("hypernym") == 1


def test_replay_of_the_same_edge_does_not_double_count(tmp_path):
    """Mesh segments are replayed — idempotency is the load-bearing property of `add_edges`."""
    L = _fresh(tmp_path)
    for _ in range(4):
        L.graph.add_edges([("a", "b", "hypernym", {"force": "r"})])
    assert L.graph.count_edges_by_label("hypernym") == 1
    assert L.graph.count_edges() == 1


def test_the_extent_never_drifts_from_the_rows(tmp_path):
    """The counter against an independent oracle: the rows themselves, enumerated in Python."""
    L = _fresh(tmp_path)
    _corpus(L)
    L.graph.add_edges([("q%d" % i, "r%d" % i, "part_of", {}) for i in range(37)])
    L.graph.delete_edge("q3", "r3", "part_of")
    L.graph.delete_edge("wn-hot", "wn-cold", "antonym")

    truth = {}
    for r in L.db.read().execute("SELECT label FROM edge"):
        truth[str(r["label"])] = truth.get(str(r["label"]), 0) + 1
    for lab, n in truth.items():
        assert L.graph.count_edges_by_label(lab) == n, lab
    assert sorted(L.graph.labels_with_edges()) == sorted(truth), (
        "labels_with_edges() must equal DISTINCT label over the rows")
    assert L.graph.count_edges_by_label("antonym") == 1


# ── 2. absent is UNKNOWN, not zero ───────────────────────────────────────────────────────────


def test_absent_counter_reads_unknown_and_never_zero(tmp_path):
    L = _fresh(tmp_path)
    _corpus(L)
    _strip_to_legacy(L)

    assert L.graph.edge_labels_measured() is False
    assert L.graph.count_edges_by_label("hypernym") is None, (
        "an uncounted relation must be None. 0 here is a fabricated measurement: this store "
        "holds two hypernym edges.")
    assert L.graph.labels_with_edges() is None

    # And the two states are distinguishable, which is the whole requirement.
    L.graph.backfill_edge_label_counters()
    assert L.graph.count_edges_by_label("hypernym") == 2      # measured, non-zero
    assert L.graph.count_edges_by_label("meronym") == 0       # measured zero
    assert L.graph.count_edges_by_label("meronym") is not None


def test_reopening_a_populated_legacy_store_does_not_certify_itself(tmp_path):
    """`ensure_schema` certifies only a store with no edges; a populated one stays loud."""
    L = _fresh(tmp_path)
    _corpus(L)
    _strip_to_legacy(L)
    L.db.close()

    L2 = open_lattice(str(tmp_path / "lat.db"), origin="test-origin", leaves=16)
    assert L2.graph.edge_labels_measured() is False, (
        "opening a populated store must not mint a certificate it did not earn")
    assert L2.graph.count_edges_by_label("hypernym") is None
# `test_the_unmeasured_signature_puts_no_number_on_anything` and
# `test_relation_vertices_refuses_rather_than_publishing_a_fabricated_extent` live in
# `agience-crystal/tests/test_relation_signature.py`, verbatim, because the function they
# certify is `crystal.ontology.seed_lattice._relation_signature`. What stays here is the
# counter itself, which is mantle's: sections 1, 2 and 3 test the store, not its reader.
#
# mantle is the standalone database, below crystal (`ARCHITECTURE-TARGET.md` §2); a mantle test
# importing `crystal.ontology` would mean this repo could not be tested without crystal
# installed.

# ── 3. the backfill, on a store written before the counter existed ───────────────────────────


def test_backfill_measures_a_legacy_store_exactly(tmp_path):
    L = _fresh(tmp_path)
    _corpus(L)
    L.graph.add_edges([("s%d" % i, "t%d" % i, "part_of", {}) for i in range(53)])
    L.graph.delete_edge("s7", "t7", "part_of")
    truth = {}
    for r in L.db.read().execute("SELECT label FROM edge"):
        truth[str(r["label"])] = truth.get(str(r["label"]), 0) + 1

    _strip_to_legacy(L)
    out = L.graph.backfill_edge_label_counters(chunk=7)      # several pages, deliberately
    assert out["already"] is False
    assert out["built"] is True
    assert out["scanned"] == sum(truth.values())
    assert out["labels"] == len(truth)
    for lab, n in truth.items():
        assert L.graph.count_edges_by_label(lab) == n, lab

    # Idempotent, and a re-run on a certified store is a no-op rather than a re-measurement.
    again = L.graph.backfill_edge_label_counters()
    assert again["already"] is True and again["scanned"] == 0
    for lab, n in truth.items():
        assert L.graph.count_edges_by_label(lab) == n, lab


def test_backfill_drops_a_stale_extent_for_a_vanished_label(tmp_path):
    """It sets from the rows, so a counter for a label with no rows left goes to zero."""
    L = _fresh(tmp_path)
    _corpus(L)
    _strip_to_legacy(L)
    with L.db.write() as cur:
        cur.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?, ?)",
                    (c_edge_label("ghost"), 999))
    L.graph.backfill_edge_label_counters()
    assert L.graph.count_edges_by_label("ghost") == 0


def test_backfill_reads_only_keyset_pages_and_no_count_star(tmp_path):
    """The rule bans `count(*)`, not iteration — so prove the iteration is what actually runs.

    Every statement the backfill issues is captured off the connection's trace hook. The failure
    mode being tested is the easy one to write by accident: bootstrapping the counter with the
    `count(*)` the counter exists to abolish."""
    L = _fresh(tmp_path)
    _corpus(L)
    _strip_to_legacy(L)

    seen = []
    L.db.conn().set_trace_callback(seen.append)
    try:
        L.graph.backfill_edge_label_counters(chunk=2)
    finally:
        L.db.conn().set_trace_callback(None)

    low = [" ".join(s.lower().split()) for s in seen]
    assert low, "the trace hook captured nothing — this check could not have failed"
    for s in low:
        assert "count(*)" not in s.replace(" ", ""), "backfill issued count(*): %r" % s
        assert " offset " not in s and " skip " not in s, "backfill paged with OFFSET: %r" % s
    assert any("where rowid >" in s and "order by rowid" in s for s in low), (
        "the backfill did not keyset-page the edge table: %r" % low)


def test_verify_counters_judges_and_repairs_the_per_label_extent(tmp_path):
    """`node-repair.py` is the test suite: a load-bearing counter nothing can check is not a
    measurement. Corrupt one extent by hand and confirm the drift audit sees it."""
    L = _fresh(tmp_path)
    _corpus(L)
    with L.db.write() as cur:
        cur.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?, ?)",
                    (c_edge_label("hypernym"), 41))

    audit = L.artifacts.verify_counters()
    assert audit["drift"].get(c_edge_label("hypernym")) == (41, 2), audit["drift"]
    assert c_edge_label_built() not in audit["drift"], (
        "the build MARKER is a state flag, not a count — judging it would report permanent "
        "drift on every healthy store and repair would un-certify it")

    L.artifacts.verify_counters(repair=True)
    assert L.graph.count_edges_by_label("hypernym") == 2
    assert L.artifacts.verify_counters()["drift"] == {}


def test_verify_counters_repair_certifies_a_legacy_store(tmp_path):
    """A repair scan visits every edge row in one transaction, so the extents it leaves are
    complete — the marker it writes is a measurement, not an assumption."""
    L = _fresh(tmp_path)
    _corpus(L)
    _strip_to_legacy(L)
    assert L.graph.edge_labels_measured() is False
    L.artifacts.verify_counters(repair=True)
    assert L.graph.edge_labels_measured() is True
    assert L.graph.count_edges_by_label("hypernym") == 2


def test_a_read_only_audit_writes_no_certificate(tmp_path):
    L = _fresh(tmp_path)
    _corpus(L)
    _strip_to_legacy(L)
    L.artifacts.verify_counters()             # repair=False
    assert L.graph.edge_labels_measured() is False


# 4. the oracle
#
# `_oracle_signature`, `test_relation_signature_is_identical_to_the_count_star_oracle` and
# `test_relation_vertices_publishes_the_same_extents` live in
# `agience-crystal/tests/test_relation_signature.py`, with the function they audit. That oracle is
# the whole argument that the counter is an optimisation and not a change of answer, so it needs
# both paths computable on one store — the path it audits is crystal's.

# ── 5. the guard: no NEW `count(*)` against the lattice, anywhere in mantle/src ───────────────

#: Every `count(*)` that may remain in `mantle/src`, pinned as an EXACT per-file ceiling so a new
#: one cannot hide behind an existing exemption. Each is a count against a DIFFERENT database —
#: none of them touches the lattice, which is what the rule is about:
#:
#:   search/embeddings_cache.py
#:                         the embeddings sidecar DB (its own file, `embeddings` table). Not the
#:                         lattice. IN SCOPE of the scan, OUT OF SCOPE of the ban — but pinned,
#:                         so if it ever grows a second one somebody has to say why here.
#:   shard/sqlite_store.py the legacy shard store, whose module header already names `count(*)`
#:                         and `LIMIT ? OFFSET ?` as "the three things the lattice store exists
#:                         to fix". A different backend, kept for the shard path.
#:   mesh/sync.py          the legacy graph-engine backend (`FROM Artifact`), reached only when
#:                         `_vertices(store) is None` — i.e. explicitly NOT a lattice; the
#:                         lattice branch returns above it without a scan.
_COUNT_STAR_CEILING = {
    os.path.join("mantle", "search", "embeddings_cache.py"): 1,
    os.path.join("mantle", "shard", "sqlite_store.py"): 2,
    os.path.join("mantle", "mesh", "sync.py"): 1,
}

#: Call names whose FIRST argument is SQL. Scoped to the call argument, not to the file, for the
#: reason `db/test_lattice.py` gives: this codebase deliberately quotes `count(*)` in
#: docstrings and error messages to name the hazard, and a guard that forced those silent would
#: trade real documentation for a proxy. `query` is here because the legacy graph backend spells
#: it that way — an `execute`-only walk would have missed `sync.py` entirely.
_SQL_CALLS = ("execute", "executemany", "executescript", "query")


def _sql_literals(path):
    """Every string constant reaching a SQL call's first argument — an AST walk, not a grep.

    Follows SQL assembled by `%`, `+` or adjacent literals: the operand constants are still
    inside the argument subtree, so a banned construct hidden in a fragment is still caught."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name not in _SQL_CALLS:
            continue
        for sub in ast.walk(node.args[0]):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                out.append((sub.value, getattr(sub, "lineno", 0)))
    return out


def _count_star_hits(path):
    return [(text, ln) for text, ln in _sql_literals(path)
            if "count(*)" in text.lower().replace(" ", "")]


def _src_root():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _mantle_modules():
    root = _src_root()
    for dirpath, _dirs, files in os.walk(os.path.join(root, "mantle")):
        for fn in sorted(files):
            if fn.endswith(".py") and not fn.startswith("test_"):
                full = os.path.join(dirpath, fn)
                yield full, os.path.relpath(full, root)


def test_no_new_count_star_anywhere_in_mantle_src():
    """`count(*)` dereferences every record to produce one integer — expensive at the scale
    this store runs at.

    `db/test_lattice.py` guards the lattice package; files outside the package are not
    covered there, so this scan covers all of `mantle/src`, and pins the surviving non-lattice
    counts as an exact per-file ceiling. A new `count(*)` in a clean file fails; so does a
    second one in an exempted file."""
    modules = list(_mantle_modules())
    assert len(modules) > 100, "the walk found almost nothing — it could not have failed"

    found, offenders = {}, []
    for full, rel in modules:
        hits = _count_star_hits(full)
        if not hits:
            continue
        found[rel] = len(hits)
        allowed = _COUNT_STAR_CEILING.get(rel, 0)
        if len(hits) > allowed:
            offenders.append("%s: %d hit(s), ceiling %d -> %r"
                             % (rel, len(hits), allowed,
                                [(ln, t[:60]) for t, ln in hits]))
    assert not offenders, (
        "count(*) reaches a database from: %s\nIt is banned against the lattice. If the new one "
        "is against some OTHER database, raise its entry in _COUNT_STAR_CEILING and say which "
        "database and why." % offenders)

    stale = {rel: n for rel, n in _COUNT_STAR_CEILING.items() if found.get(rel, 0) < n}
    assert not stale, (
        "the ceiling is now looser than the code: %r. Lower it — an exemption nobody needs is "
        "an open door." % stale)


def test_the_count_star_guard_has_teeth(tmp_path):
    """A check that cannot fail proves nothing. State the failure mode, then produce it.

    Failure mode: the AST walk silently reaches no SQL — wrong call name, wrong argument
    position, unparsed file — and reports every module clean forever."""
    clean = tmp_path / "clean.py"
    clean.write_text(
        '"""Never use count(*) — it dereferences every record."""\n'
        'def f(cur):\n'
        '    raise ValueError("count(*) is banned on the lattice")\n'
        'def g(cur):\n'
        '    return cur.execute("SELECT n FROM counter WHERE name = ?")\n', encoding="utf-8")
    assert _count_star_hits(str(clean)) == [], (
        "the guard fires on prose and error messages — it would force the hazard undocumented")

    dirty = tmp_path / "dirty.py"
    dirty.write_text(
        'def f(cur, t):\n'
        '    a = cur.execute("SELECT COUNT(*) FROM edge WHERE label=?")\n'
        '    b = cur.execute("SELECT count (*) FROM vertex")\n'
        '    c = conn.query("SELECT count(*) AS c FROM Artifact")\n'
        '    d = cur.execute("SELECT count(*) FROM %s" % t)\n'
        '    return a, b, c, d\n', encoding="utf-8")
    hits = _count_star_hits(str(dirty))
    assert len(hits) == 4, (
        "the guard missed a violation — concatenated SQL and the `query` spelling are the two "
        "that hide: %r" % hits)

# `crystal/ontology/seed_lattice.py` lives outside `mantle/src`, so this walk does not cover
# it; a `crystal/src` copy of the scan below covers that file instead. This walk covers
# `mantle/src`, and a file that leaves `mantle/src` leaves the walk — without the copy, that
# file would go unscanned.
