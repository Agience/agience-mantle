"""`ix_v_collection` must be a FULL expression index, not a partial one.

WHY THIS FILE EXISTS. `collection_id` is one of the most-queried `doc` fields in the system —
`data_integrity_check`'s `artifacts_naming_a_missing_collection` and `orphan_content_roots` both
filter on it — and it had no index. Measured 2026-08-25 on the live lattice, same 200,000 rows:
**0.22 s** for a column-only aggregate against **9.34 s** once `json_extract` entered the predicate.

THE TRAP THIS PINS. The two expression indexes already in `schema.py` (`ix_v_grantee`,
`ix_v_resource`) are PARTIAL — `WHERE json_extract(...) IS NOT NULL`. Copying that shape here looks
consistent and is wrong, measured on 200,000 real rows:

    index                        has a collection   has none     plan
    FULL                              0.03 s         0.03 s      COVERING INDEX
    PARTIAL (IS NOT NULL)             0.95 s         0.94 s      SCAN TABLE

A partial index cannot cover a query whose result set may include the rows it excludes, and
*"which artifacts have NO collection"* is exactly such a query. The grantee/resource indexes are
partial because their queries only ever seek a value that is present; this one is asked both ways.

And the win is that it COVERS, not that it seeks: the plan is `SCAN ... USING COVERING INDEX`, so
SQLite answers from the index and never reads the table — which is where the per-row JSON parse was.
"""
from __future__ import annotations

import sqlite3

from mantle.db import schema

HAS = ("SELECT COUNT(*) FROM vertex v "
       "WHERE COALESCE(json_extract(v.doc,'$.collection_id'),'') <> ''")
HAS_NOT = ("SELECT COUNT(*) FROM vertex v "
           "WHERE COALESCE(json_extract(v.doc,'$.collection_id'),'') = ''")


def _ddl() -> str:
    hits = [d for d in schema.ALL_DDL if "ix_v_collection" in d]
    assert len(hits) == 1, "expected exactly one ix_v_collection declaration, found %d" % len(hits)
    return hits[0]


def test_the_index_is_declared() -> None:
    assert "json_extract(doc, '$.collection_id')" in _ddl(), (
        "ix_v_collection must index the expression, not a column — this store has a recorded case "
        "of a column disagreeing with its own JSON field on 99.97%% of rows")


def test_the_index_is_not_partial() -> None:
    """The load-bearing assertion. A `WHERE` clause here would make it unusable for the
    has-no-collection query, which is the one the collectionless sweep turns on."""
    assert "WHERE" not in _ddl().upper(), (
        "ix_v_collection became partial. Measured: a partial index falls back to SCAN TABLE for "
        "BOTH directions of this query — it would cost 64 MB and never be used. See this module's "
        "docstring for the numbers.")


def _store(tmp_path, with_index: bool):
    c = sqlite3.connect(str(tmp_path / ("y" if with_index else "n")))
    c.execute("CREATE TABLE vertex(id TEXT PRIMARY KEY, ct TEXT, doc TEXT)")
    c.executemany(
        "INSERT INTO vertex VALUES (?,?,?)",
        [("a%d" % i, "text/markdown",
          '{"collection_id":"c1"}' if i % 3 else "{}") for i in range(4000)])
    if with_index:
        c.execute(_ddl())
    c.commit()
    c.execute("ANALYZE")
    return c


def test_the_index_covers_both_directions(tmp_path) -> None:
    """Both questions must be answered from the index, without reading the table."""
    c = _store(tmp_path, with_index=True)
    for name, q in (("has a collection", HAS), ("has none", HAS_NOT)):
        plan = " / ".join(r[-1] for r in c.execute("EXPLAIN QUERY PLAN " + q))
        assert "COVERING INDEX" in plan, (
            "%r does not use the covering index: %s — the JSON parse is still per-row" % (name, plan))
    c.close()


def test_the_index_does_not_change_the_answer(tmp_path) -> None:
    """An index may make a query cheaper. It may never make it different."""
    with_idx, without = _store(tmp_path, True), _store(tmp_path, False)
    for q in (HAS, HAS_NOT):
        assert with_idx.execute(q).fetchone()[0] == without.execute(q).fetchone()[0]
    assert with_idx.execute(HAS).fetchone()[0] + with_idx.execute(HAS_NOT).fetchone()[0] == 4000, (
        "the two directions must partition the table between them")
    with_idx.close()
    without.close()
