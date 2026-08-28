"""The lattice schema — LATTICE-CONTRACT.md §2, authoritative.

`vertex` and `edge` are reproduced exactly as the contract specifies them.
Everything else in this module is bookkeeping that exists to make two hard rules
enforceable rather than aspirational:

  * **No `count(*)`, ever.** `count(*)` dereferences every record — EXPLAIN shows
    it loads every row to produce one integer, which can OOM a process on a large
    corpus. So every count this store can answer is maintained incrementally in
    `counter`, in the same transaction as the write that changes it.
  * **No `SKIP`/`OFFSET`, ever.** `SKIP` at a large offset costs roughly two orders
    of magnitude more than the equivalent keyset page. Every paging API here
    is `WHERE id > :cur ORDER BY id LIMIT n`.

1. **`task` sidecar table.** Contract §2 says there is no `state` column and that
   a migration shim "lives inside `doc` JSON where it can never become a query
   predicate". The work pool nevertheless needs an indexed, selective predicate
   on task status — `pool.claim()` scanning the whole task history is the exact
   pathology contract §5 records as "queue silently dead on sqlite". Resolution:
   task coordination state lives in a separate table keyed on `vertex.id`, never
   as a column on `vertex`. The contract's intent (artifact *lifecycle* `state`
   must not be a query predicate) is preserved intact; task `status` is a
   different field with a different owner — it is the work pool's coordination
   state, not the artifact's lifecycle.

2. **`leaf_digest` table.** Merkle leaves are maintained incrementally (XOR out
   the old row hash, XOR in the new) rather than by full rescan. This is not an
   optimization: a full rescan publish is far slower than incremental maintenance,
   slow enough that the corpus can keep growing during that same pass on a node
   that is still catching up. A rescan-built tree is stale before it finishes on
   such a node, and publishing it advertises a root that never matched any real
   state.
"""
from __future__ import annotations

import sqlite3
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# The rule for adding a column — contract §2.3. Apply it before adding one.
#
#     A column is legitimate only if every observer would compute the same
#     value for it.
#
#   deterministic / content-addressed  -> column, indexable, a query predicate
#   frame-local / observer-dependent   -> content, in `doc`, never a predicate
#
# Neither `state` nor `_rev` is a column, because both fail this test. `created_time` also fails
# it and is present anyway, as a deliberate override — see the note on VERTEX_DDL. An override is
# not a counter-example to the rule; it is a decision taken with the rule understood.
#
# The point is not that observer-dependent values are worthless — they are
# content, and content lives in `doc`. The point is that a column is a standing
# invitation to a query predicate: nothing but this rule stops `ORDER BY created_time`
# from being accepted by the engine, so treat `created_time` strictly as local
# annotation, never as a cross-observer sort key.
#
# Ordering belongs on edges. Placement in an ordering is a relation between
# things — relative and frame-dependent — so it is `edge.order_key`, a
# fractional index. `created_time` is a second, worse ordering primitive
# sitting next to a working one. Across frames the answer is graph
# reachability (§1.2), and unordered is a valid answer (§C.7) — not a gap to
# paper over with a timestamp.
# ─────────────────────────────────────────────────────────────────────────────

# ── contract §2, verbatim ────────────────────────────────────────────────────
VERTEX_DDL: List[str] = [
    # `created_time` is kept as a deliberate override of the column test: two observers reading
    # their own clocks disagree and no function reconciles them, so by the column test it is
    # content, not a column. `state` and `_rev` remain removed.
    #
    # ── Deprecated: `created_by` and `created_time` ───────────────────────────────────────────
    # Both columns are declared and populated. Each has a named deterministic replacement, and
    # each may be dropped once that replacement is populated and verified on the live store.
    # `test_deprecated_columns.py` is the removal gate and asserts the replacements behave, so
    # "we have a replacement" is a measurement rather than a claim.
    #
    #   created_by   -> the creation edge (`label=created`, `is_origin=1`, `propagate="r"`).
    #                   Contract §2.1 already makes the edge authoritative and the column the
    #                   non-authoritative duplicate: "the who is expressed twice and the edge is
    #                   authoritative, because authorization flows through it and it carries the
    #                   `propagate` mask, which a column cannot."
    #                   Removal precondition: the `creation` stage has run and every artifact with
    #                   a who has its edge. That stage reads this column to build the edges, so the
    #                   column is the source — it outlives the edges it generates.
    #
    #   created_time -> for ordering: `(_origin, _seq)` within an origin (gap-free within each
    #                   origin) and `edge.order_key` within a frame.
    #                   -> for provenance: the `provenance` rung + `cited_from` + `origin_root`.
    #                   Removal precondition: `_seq` is gap-free per origin, and every consumer
    #                   reads `(_origin, _seq)` as the pair.
    #                   And the honest limit: ordering holds within an origin. Two nodes' counters
    #                   run independently, so equal `_seq` from different origins are concurrent.
    #                   A cross-origin ordering is graph reachability rather than a timestamp —
    #                   unordered is a valid answer.
    """CREATE TABLE IF NOT EXISTS vertex (
         id           TEXT PRIMARY KEY,
         ct           TEXT,
         offer        TEXT,
         content_ref  TEXT,
         created_by   TEXT,
         created_time TEXT,
         origin_root  TEXT,
         root_id      TEXT,
         _origin      TEXT,
         _seq         INTEGER,
         _leaf        INTEGER,
         doc          TEXT
       )""",
    "CREATE INDEX IF NOT EXISTS ix_v_ct     ON vertex(ct)",
    "CREATE INDEX IF NOT EXISTS ix_v_offer  ON vertex(offer)",
    "CREATE INDEX IF NOT EXISTS ix_v_leaf   ON vertex(_leaf)",
    "CREATE INDEX IF NOT EXISTS ix_v_origin ON vertex(_origin, _seq)",
    # The version lineage. Without this index a lineage walk is a full scan, which is the
    # difference between versioning being usable and being nominal.
    "CREATE INDEX IF NOT EXISTS ix_v_root_id ON vertex(root_id)",
    "CREATE INDEX IF NOT EXISTS ix_v_root   ON vertex(origin_root)",
    # ── the authorization plane: grants, by who holds one and what it reaches ──
    #
    # Every grant shares one `ct`, so `ix_v_ct` narrows a grant lookup to "all grants in the
    # store" and nothing further — the grantee and the resource are then sifted in Python.
    # That makes an authorization O(total active grants system-wide) rather than O(this
    # principal's grants), and it sits on `LightConeResolver.resolve`, i.e. on nearly every
    # authenticated request.
    #
    # Both fields live inside `doc`, so the predicate is an expression and the index has to be
    # one too. `json_extract` is deterministic, which is what makes it indexable at all.
    #
    # PARTIAL, on `IS NOT NULL`, for two reasons that both matter:
    #   * Size. Only grants carry these fields (`entities/grant.py` is the sole writer of
    #     either), so the index holds one entry per grant rather than one per vertex — it does
    #     not grow with a multi-million-row artifact corpus that has no grants in it.
    #   * It costs the caller nothing. `<expr> = ?` cannot be true when `<expr>` is NULL, and
    #     SQLite knows it: the planner discharges the partial-index WHERE from the equality
    #     term alone, so no query has to name the `IS NOT NULL` predicate to get the seek.
    #     Verified with `EXPLAIN QUERY PLAN`.
    #
    # `ct` is the second column so `WHERE ct = ? AND json_extract(...) = ?` — the shape
    # `LatticeArtifactStore.list_by_doc_field` emits — seeks on both terms, and `id` is third so
    # its `ORDER BY id` needs no temp b-tree and a keyset page (`AND id > ?`) stays an indexed
    # range. No `count(*)` and no `OFFSET` are introduced or implied.
    "CREATE INDEX IF NOT EXISTS ix_v_grantee ON vertex("
    "  json_extract(doc, '$.grantee_id'), ct, id"
    ") WHERE json_extract(doc, '$.grantee_id') IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS ix_v_resource ON vertex("
    "  json_extract(doc, '$.resource_id'), ct, id"
    ") WHERE json_extract(doc, '$.resource_id') IS NOT NULL",
    #
    # `collection_id` is one of the most-queried `doc` fields in the system and had no index:
    # `data_integrity_check`'s `artifacts_naming_a_missing_collection` and `orphan_content_roots`
    # both filter on it, and it is the field the collectionless-replication sweep turns on. Measured
    # 2026-08-25 on the live lattice, bounded probes over the same 200,000 rows: **0.22 s** for a
    # column-only `GROUP BY ct` against **9.34 s** once `json_extract(doc,'$.collection_id')` was in
    # the predicate. 42x, and all of it the JSON parse.
    #
    # The win is that it covers, not that it seeks. The plan is
    # `SCAN vertex USING COVERING INDEX` — SQLite answers from the index and never reads the
    # table, so the per-row JSON parse disappears entirely. That is why a scan-shaped query gets
    # faster at all.
    #
    # Deliberately not partial, unlike the two above: copying their `WHERE ... IS NOT NULL` shape
    # would build 64 MB of index that SQLite never uses, measured against the same rows:
    #
    #     index                      has a collection   has none     plan
    #     FULL                            0.03 s         0.03 s      COVERING INDEX
    #     PARTIAL (IS NOT NULL)           0.95 s         0.94 s      SCAN TABLE
    #
    # A partial index cannot cover a query whose result set may include the rows it excludes, and
    # "which artifacts have NO collection" is exactly such a query. The grantee/resource indexes are
    # partial because their queries only ever seek a present value; this one is asked both ways.
    #
    # An index, not a column, and the distinction is load-bearing: this store has a recorded case
    # of a column disagreeing with its own JSON field, `vertex.root_id` against
    # `json_extract(doc,'$.root_id')` differing on 2,169,111 of 2,169,683 rows (99.97%). An
    # expression index computes the identical expression, so it cannot diverge; it only makes the
    # same answer cheaper.
    #
    # Cost on first apply, measured at ~1.7 s and +5 MB per 200,000 rows: roughly 23 s and +64 MB
    # against the 2.66M-row store. It is a one-time build on the next schema application, not a
    # per-query cost.
    "CREATE INDEX IF NOT EXISTS ix_v_collection ON vertex("
    "  json_extract(doc, '$.collection_id')"
    ")",
]

#
# `origin_root` is the collection's immutable origin root: the top of the creation-lineage tree,
# fixed at creation and never moved thereafter. It does not depend on the observer, on a clock, on
# replication order, or on who is asking. Two nodes that replicate the same row compute the same
# root. That is precisely the property `created_time` lacked and `_origin`/`_seq` were built to
# supply.
#
# It is a column rather than derived on read because it is the key root for content encryption
# (`content_cache.collection_key(root_secret, origin_root)`). The grant gates whether a key is
# issued; this determines which key. Deriving it on read would put a lineage walk on every
# decrypt; deriving it lazily would mean a blob's key root could differ between two reads. Neither
# is acceptable for a value whose whole job is to never move.
#
# Today no collection nests: every collection artifact (`universe`, `ontology`, `stage.0..3`,
# `subjects`) has no parent, so `origin_root == collection_id` and the "walk" is depth-1. It is
# stored anyway because that is flat only today: GENESIS §5 designs for subject trees
# (`subject.math.topology.homotopy`), and if nesting arrives while content is keyed on the flat
# `collection_id`, every blob written under it would need re-keying. Storing `origin_root` now
# makes nesting free.

EDGE_DDL: List[str] = [
    """CREATE TABLE IF NOT EXISTS edge (
         edge_key   BLOB PRIMARY KEY,
         src        TEXT NOT NULL,
         dst        TEXT NOT NULL,
         label      TEXT NOT NULL,
         force      TEXT,
         propagate  TEXT,
         is_origin  INTEGER,
         order_key  TEXT,
         _origin    TEXT,
         _seq       INTEGER,
         _leaf      INTEGER,
         props      TEXT
       )""",
    "CREATE INDEX IF NOT EXISTS ix_e_src    ON edge(src, label)",
    "CREATE INDEX IF NOT EXISTS ix_e_dst    ON edge(dst, label)",
    "CREATE INDEX IF NOT EXISTS ix_e_origin ON edge(_origin, _seq)",
    "CREATE INDEX IF NOT EXISTS ix_e_leaf   ON edge(_leaf)",
]

# ── bookkeeping ──────────────────────────────────────────────────────────────
SUPPORT_DDL: List[str] = [
    # Proper time. One row per observer.
    #
    """CREATE TABLE IF NOT EXISTS seq_counter (
         origin   TEXT PRIMARY KEY,
         last_seq INTEGER NOT NULL,
         is_local INTEGER NOT NULL DEFAULT 0
       )""",

    # Every count this store can answer. Maintained incrementally, in the write
    # transaction. A missing row means zero. Never backfilled with count(*).
    """CREATE TABLE IF NOT EXISTS counter (
         name TEXT PRIMARY KEY,
         n    INTEGER NOT NULL
       )""",

    # Incremental merkle. `digest` is 8 raw bytes, not an INTEGER, because
    # row hashes are unsigned 64-bit and SQLite INTEGER is signed — storing them
    # as integers silently wraps half the key space negative and the XOR
    # round-trip stops being an identity.
    """CREATE TABLE IF NOT EXISTS leaf_digest (
         leaf   INTEGER PRIMARY KEY,
         digest BLOB NOT NULL
       )""",

    # Small key/value metadata that is not a counter and must not be recomputed by
    # the counter-drift audit. `merkle.leaves` lives here: the store's operating
    # Merkle resolution (the derived `natural_leaves(corpus)`), recorded so every
    # `open_lattice` resolves the same value and `reshard()` can move it in one place.
    """CREATE TABLE IF NOT EXISTS meta (
         k TEXT PRIMARY KEY,
         v TEXT
       )""",

    # The demand cache index (local, never replicated). One row per reached artifact,
    # carrying its accumulated `mass` and last-touch `ts`. Decay (demurrage) is applied
    # in the ember layer (which owns the one `prism.law` kernel); the store only holds
    # the raw scalar and its timestamp. A row here is an evictable cache copy; a held
    # row with no demand row is own-authored and is never evicted. This is what lets a
    # limited ember shed what it stops being asked for and stay within its envelope.
    """CREATE TABLE IF NOT EXISTS demand (
         id   TEXT PRIMARY KEY,
         mass REAL NOT NULL,
         ts   REAL NOT NULL
       )""",

    # Work-pool coordination state. See the module docstring for why this is a
    # sidecar and not a `vertex` column.
    """CREATE TABLE IF NOT EXISTS task (
         id            TEXT PRIMARY KEY,
         ct            TEXT NOT NULL,
         status        TEXT NOT NULL,
         priority      INTEGER NOT NULL DEFAULT 0,
         operator      TEXT,
         task_key      TEXT,
         claimed_by    TEXT,
         claimed_at    TEXT,
         next_retry_at TEXT,
         completed_at  TEXT
       )""",
    # The claim scan: (ct, status) seeks the pending bucket, then priority/id are
    # already in index order — so `ORDER BY priority DESC, id LIMIT n` is a
    # bounded index walk with no sort. The legacy store materialised and sorted
    # the entire task history per worker per 5s, then shuffled the result away.
    "CREATE INDEX IF NOT EXISTS ix_t_pending  ON task(ct, status, priority DESC, id)",
    # The /status panel: bounded window off the indexed terminal buckets.
    "CREATE INDEX IF NOT EXISTS ix_t_terminal ON task(ct, status, completed_at DESC)",

    # ── the keyed arm: multi-valued doc fields, discriminated by content type ──
    #
    # A dictionary lookup is keyed retrieval (word -> senses), not similarity, and a JSON list
    # cannot be indexed in place: SQLite indexes scalars, and `json_each` is a table-valued
    # function no index can cover. So the list is unrolled into rows here.
    #
    # `(field, value, ct)` is therefore the index key: "the wordnet senses of 'dog'" is one
    # index seek whose LIMIT is spent entirely on synsets. The leading `(field, value)` prefix
    # still serves an undiscriminated lookup, so `content_type=None` remains expressible for
    # callers that genuinely want every type (the browse UI).
    """CREATE TABLE IF NOT EXISTS listkey (
         aid   TEXT NOT NULL,
         field TEXT NOT NULL,
         value TEXT NOT NULL,
         ct    TEXT
       )""",
    "CREATE INDEX IF NOT EXISTS ix_lk_lookup ON listkey(field, value, ct)",
    # The maintenance side: an upsert deletes this artifact's old postings first.
    "CREATE INDEX IF NOT EXISTS ix_lk_aid    ON listkey(aid)",

    # ── access audit log (standalone-Mantle web service) ─────────────────────
    # Append-only history rows — "who touched what, when, allowed or denied". Not edges:
    # the edge table upserts by (src,dst,label), which would collapse repeated accesses of
    # the same (principal, artifact, action) into one row and destroy the history an audit
    # log exists to keep. Not vertices: log rows are operational bookkeeping (like `task`),
    # not observations in the artifact universe. See `db/audit.py`.
    """CREATE TABLE IF NOT EXISTS access_event (
         principal_id TEXT,
         artifact_id  TEXT NOT NULL,
         action       TEXT NOT NULL,
         result       TEXT NOT NULL,
         ts           TEXT NOT NULL,
         ctx          TEXT
       )""",
    # The one read shape: an artifact's history, newest first — a bounded index walk.
    "CREATE INDEX IF NOT EXISTS ix_ae_artifact ON access_event(artifact_id, ts DESC)",
]

PRAGMAS: List[str] = [
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=OFF",
    #
    # Without this, the WAL grows without bound and nothing reclaims it. Measured on a real node:
    # a 10.3 GB database beside a 32.7 GB write-ahead log, having doubled from 16.7 GB in a day of
    # bulk ingests. Every cold read scans that WAL — measured at >45 s cold against 6.28 s warm, a
    # 7x penalty paid once per session by every reader — and `du` over the directory stopped
    # returning at all, which is what hung the node's supervisor in its own preflight.
    #
    # This bounds future growth; it does not shrink what is there. `wal_autocheckpoint` runs
    # passive checkpoints, which fold committed frames back into the database and then reuse the
    # WAL from the start — the file stays at its high-water mark forever. Only
    # `wal_checkpoint(TRUNCATE)` resets it to zero, and that needs the node quiet because an active
    # reader's snapshot blocks it, so an existing oversized WAL is a separate, one-off maintenance
    # action.
    #
    # `agience-ember`'s `src/ember/genesis.py::_wal_checkpoint` carries the same measurement for
    # the same reason — an uncheckpointed bulk re-ingest grows its own lattice WAL and every read
    # then scans the whole thing — and calls TRUNCATE every N bulk batches. Same file, same
    # magnitude, same consequence. See
    # `_archive/2026-08-26-tighten-threads-closed/LATTICE-WAL-NEVER-CHECKPOINTS.md`.
    #
    # 2000 pages against SQLite's inherited default of 1000: this store's pages are 4 KB, so a
    # checkpoint is attempted roughly every 8 MB of WAL rather than every 4 MB. Doubling it halves
    # the checkpoint attempts on a write-heavy ingest while still keeping the WAL two orders of
    # magnitude below what it reached unattended. It is written out rather than inherited so that a
    # future SQLite changing its default cannot change this store's behaviour silently.
    "PRAGMA wal_autocheckpoint=2000",
    #
    # `wal_autocheckpoint` bounds the steady state; this bounds what is left after a blocked
    # period, and they are not the same guarantee. Measured on scratch stores, both pragmas held
    # constant except this one:
    #
    #     journal_size_limit  autocheckpoint=2000, steady-state WAL
    #     -1                  8,408,952
    #     65536               8,408,952      ← identical: with checkpoints RUNNING it changes nothing
    #
    # So it is not a steady-state fix. It matters when checkpoints are blocked — a reader holding
    # an older snapshot stops frames being reclaimed, the WAL grows past any threshold, and the
    # file then keeps that high-water mark forever:
    #
    #     journal_size_limit  after writes   after PASSIVE   after the NEXT write
    #     -1                   2,768,672      2,768,672       2,768,672   ← peak held
    #     65536                2,768,672      2,768,672          65,536   ← truncated
    #
    # The timing is counterintuitive: the truncation happens on the first write after a resetting
    # checkpoint, not at the checkpoint itself. A reader measuring the file immediately after
    # `wal_checkpoint(PASSIVE)` sees no change and would conclude the pragma does nothing.
    #
    # 64 MiB against an 8.19 MiB autocheckpoint threshold (2000 × 4 KiB pages) is eight times the
    # steady state, so a normal ingest burst never touches it, while the residual after a
    # reader-blocked stretch is bounded at 64 MiB rather than growing without limit. It is a
    # ceiling, not a target — nothing shrinks to it on a store that never exceeds it.
    "PRAGMA journal_size_limit=67108864",
    #
    # The reason it cannot help: these stages do a single-pass sequential scan. Each page is
    # read once and evicted, so a larger cache has nothing to re-serve, while `mmap` adds
    # per-page mapping overhead to every one of them. A page cache pays for repeated access,
    # which this workload does not have.
    #
    # The real cost is the scan itself, which grows with corpus size and can run for
    # many minutes on a large corpus. That is fixed by not scanning — an index on the
    # extracted JSON key — not by buffering.
]

ALL_DDL: List[str] = VERTEX_DDL + EDGE_DDL + SUPPORT_DDL


def apply_pragmas(conn: sqlite3.Connection) -> None:
    for p in PRAGMAS:
        conn.execute(p)


def wal_checkpoint(conn: sqlite3.Connection, mode: str = "TRUNCATE") -> tuple[int, int, int]:
    """Fold the write-ahead log back into the database and reset it. Returns SQLite's own triple.

    The counterpart of `agience-ember`'s `genesis.py::_wal_checkpoint`, in the same shape for the
    same reason: an uncheckpointed bulk write grows the WAL until every read scans the whole thing.
    `journal_mode=WAL` alone is not enough; this is the missing half.

    `TRUNCATE`, not `PASSIVE`, and the difference is the whole point. `PASSIVE` folds committed
    frames back and then reuses the WAL from its start — the file stays at its high-water mark
    forever, which is why `wal_autocheckpoint` alone never recovers a byte on its own. Only `TRUNCATE`
    resets the file to zero.

    AND IT CAN LEGITIMATELY DO NOTHING. A checkpoint cannot pass an active reader's snapshot: any
    connection holding an older view blocks the frames after it. SQLite reports that by returning
    `busy=1` rather than by raising, so a caller that ignored the return value would log a
    successful checkpoint that moved nothing. The triple is returned for that reason, and the
    return is `(busy, log_pages, checkpointed_pages)` exactly as SQLite defines it.

    Not called automatically from here. Where a bulk path should checkpoint is a decision about that
    path's batching - `genesis.py` does it every N batches and again after each pass - and guessing
    at it from the schema module would put the policy in the wrong place.
    """
    mode = mode.upper()
    if mode not in ("PASSIVE", "FULL", "RESTART", "TRUNCATE"):
        raise ValueError("unknown checkpoint mode %r - SQLite defines PASSIVE, FULL, RESTART, "
                         "TRUNCATE" % mode)
    row = conn.execute("PRAGMA wal_checkpoint(%s)" % mode).fetchone()
    # A connection with no WAL (an in-memory or journal-mode database) returns no row rather than
    # zeros. Reporting that as `(0, 0, 0)` would be indistinguishable from a checkpoint that ran and
    # found nothing to do.
    if row is None:
        return (0, -1, -1)
    return (int(row[0]), int(row[1]), int(row[2]))


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent. Safe to call on every open, from every thread and process."""
    _migrate_vertex_columns(conn)
    _migrate_edge_columns(conn)
    for stmt in ALL_DDL:
        conn.execute(stmt)
    _migrate_next_seq(conn)
    _mark_empty_store_indexed(conn)
    _mark_empty_store_edge_labels(conn)


# Columns added to `vertex` after stores already existed in the field, newest last.
# `CREATE TABLE IF NOT EXISTS` does not add columns to an existing table, so without
# this every pre-existing store fails to open with `no such column`.
#
# Live stores exist with `origin_root` and `listkey` missing, and some also lack
# `created_time` — this is a live migration path, not a defensive one.
#
# Every entry must be nullable with no default. A backfill is a separate, explicit pass:
# `origin_root` is derived from the creation lineage (`vertex._origin_root`), and silently
# defaulting it here would write a computed value that no observer agreed on — the exact
# sentinel-as-measurement failure the `ic = 0.0` comment in enrich_wordnet.py catalogues.
_VERTEX_ADDED_COLUMNS = (
    ("created_time", "TEXT"),
    ("origin_root", "TEXT"),
    ("root_id", "TEXT"),
)


def _migrate_vertex_columns(conn: sqlite3.Connection) -> None:
    """Add any `vertex` column introduced after this store was created.

    No-op on a fresh store (the table does not exist yet; `ALL_DDL` creates it complete)
    and on an up-to-date one."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(vertex)")}
    if not cols:                      # fresh store — nothing to migrate
        return
    for name, decl in _VERTEX_ADDED_COLUMNS:
        if name not in cols:
            conn.execute("ALTER TABLE vertex ADD COLUMN %s %s" % (name, decl))


# Columns added to `edge` after stores already existed. Same rule as the vertex list: nullable, no
# default, and a NULL value means "not yet backfilled" — never a silently-computed one.
_EDGE_ADDED_COLUMNS = (
    ("_leaf", "INTEGER"),
)


def _migrate_edge_columns(conn: sqlite3.Connection) -> None:
    """Add any `edge` column introduced after this store was created. No-op on a fresh or up-to-date
    store. Must run before `ALL_DDL` for the same reason as the vertex migration: `ix_e_leaf` indexes
    `_leaf`, and an index over a column a legacy edge table lacks aborts the whole open."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(edge)")}
    if not cols:                      # fresh store — nothing to migrate
        return
    for name, decl in _EDGE_ADDED_COLUMNS:
        if name not in cols:
            conn.execute("ALTER TABLE edge ADD COLUMN %s %s" % (name, decl))


def _mark_empty_store_indexed(conn: sqlite3.Connection) -> None:
    """Certify `listkey` as built iff this store has no vertices yet.

    A store with zero vertices is trivially, exhaustively indexed, and every write from here on
    maintains `listkey` inside the same transaction as the vertex row — so the certificate stays
    true without anyone rescanning. This is what keeps the guard in `c_list_index_built` from
    firing on every newly-created store. A populated store is left uncertified:
    `rebuild_list_index()` is then the only way to earn the marker."""
    built = conn.execute("SELECT n FROM counter WHERE name = ?",
                         (c_list_index_built(),)).fetchone()
    if built is not None:
        return
    row = conn.execute("SELECT n FROM counter WHERE name = ?", (c_vertex_total(),)).fetchone()
    if row is None:
        # No vertex counter at all. On a genuinely fresh store this is the same instant the
        # tables were created, so fall back to the one question that is cheap and exact on an
        # empty table: does `vertex` hold a single row? `LIMIT 1` is a seek, not a scan — it is
        # not `count(*)`, and it stops after the first row on a 6M-row store just as fast.
        if conn.execute("SELECT 1 FROM vertex LIMIT 1").fetchone() is not None:
            return
    elif int(row["n"] if hasattr(row, "keys") else row[0]) != 0:
        return
    conn.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?, 1)",
                 (c_list_index_built(),))


def _mark_empty_store_edge_labels(conn: sqlite3.Connection) -> None:
    """Certify the per-label edge counters as complete iff this store has no edges yet.

    Exactly the argument `_mark_empty_store_indexed` makes, for `edge:label:*`: a store with no
    edge rows is trivially, exhaustively counted, and every `_add_one`/`delete_edge` from here on
    maintains the per-label counter in the same transaction as the row — so the certificate stays
    true without anyone rescanning.

    A populated store is left uncertified. Where this counter has not yet been computed for its
    edges, `counter_of` would read 0 for a relation holding a million edges — a fabricated
    measurement, which is worse than the slow query it replaced. Absent means not measured,
    `count_edges_by_label` returns `None` for it, and
    `LatticeGraphStore.backfill_edge_label_counters()` is the only way to earn the marker.

    The emptiness question is asked of the ROWS, not of `c_edge_total()`: `SELECT 1 ... LIMIT 1`
    stops at the first row, so it costs the same on an empty store and on a 6M-edge one, and it
    cannot be fooled by a total counter that has itself drifted."""
    if conn.execute("SELECT n FROM counter WHERE name = ?",
                    (c_edge_label_built(),)).fetchone() is not None:
        return
    if conn.execute("SELECT 1 FROM edge LIMIT 1").fetchone() is not None:
        return
    conn.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?, 1)",
                 (c_edge_label_built(),))


def _migrate_next_seq(conn: sqlite3.Connection) -> None:
    """Rename the legacy `seq_counter.next_seq` column to `last_seq` if present.

    No-op once a store already has `last_seq`. Nothing outside this package
    references `next_seq`; this exists only so an older store file does not
    become unreadable."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(seq_counter)")}
    if "next_seq" in cols and "last_seq" not in cols:
        conn.execute("ALTER TABLE seq_counter RENAME COLUMN next_seq TO last_seq")


# ── counter names — the only sanctioned way to answer "how many" ─────────────
# Kept as functions so a typo is an AttributeError at import, not a silently
# wrong number at read time (a missing counter row reads as 0, which is exactly
# the wrong-answer-not-empty-answer failure class contract §5 catalogues).

def c_vertex_total() -> str:
    return "vertex"


def c_edge_total() -> str:
    return "edge"


def c_edge_label(label: str) -> str:
    """Edges carrying one `label` — a relation's extent.

    `c_edge_total()` answers "how many edges", which is the wrong question for
    anything that reasons about one relation: `ontology.seed_lattice.
    _relation_signature` needs `hypernym`'s extent to decide whether the relation
    exists at all and to size its sample stride. Without this counter that needs
    `SELECT COUNT(*) FROM edge WHERE label=?`, and there is no index on `label`
    alone (the two edge indexes are `(src,label)` and `(dst,label)`), so that is a
    full scan of the edge table per relation — 85 of them per build. Exactly the
    6M-record dereference the module docstring bans, run 85 times.

    Maintained in the write transaction, beside `c_edge_total()`, at every site
    that changes the edge count. `label` is part of `edge_key`, so an upsert can
    never move an edge from one label to another — a re-add is the same row and
    bumps nothing, which is what keeps replay idempotent here too."""
    return "edge:label:" + label


def c_edge_label_built() -> str:
    """1 once `edge:label:*` covers every edge in this store; absent otherwise.

    The same two-state design as `c_list_index_built`, and for the same reason: a
    missing `counter` row reads as 0 through `counter_of`, so without a separate
    marker "nobody has ever counted this relation" and "this relation has exactly
    zero edges" would be the same answer. They are not the same answer. Reporting
    `edges: 0` for a relation holding a million edges is a fabricated measurement,
    and a fabricated measurement is worse than the slow query it replaced.

    So: absent -> `count_edges_by_label` returns `None` -> `_relation_signature`
    reports `measured: False` rather than putting a number on it. `ensure_schema`
    sets the marker on a store with no edges (trivially complete, and every write
    thereafter maintains it); `backfill_edge_label_counters()` sets it after a
    keyset-paged pass over the rows. A populated store that has had neither
    matches neither, which is exactly the state that must stay loud.

    Not spelled `edge:label:built`. That is precisely the counter name a relation
    literally called `built` would take, and the marker and that relation's extent
    would then be the same row — each silently overwriting the other. The marker
    lives outside the `edge:label:` namespace on purpose."""
    return "edgelabel:built"


def c_ct(ct: str) -> str:
    return "ct:" + ct


def c_collection(collection_id: str, *, committed_only: bool = False) -> str:
    return "col:" + collection_id + (":committed" if committed_only else "")


def c_task_status(ct: str, status: str) -> str:
    return "task:" + ct + ":" + status


def c_rows(origin: str) -> str:
    """Live rows currently carrying `origin`, across `vertex` and `edge`.

    The left-hand side of the allocation-accounting invariant. Because allocation
    is injective and each allocation writes exactly one row, this equals
    `COUNT(DISTINCT _seq)` for that origin — maintained incrementally so the
    invariant can be checked without a scan."""
    return "rows:" + origin


def c_vacated(origin: str) -> str:
    """Seqs whose row no longer exists by an accounted write — an update that
    superseded it, or a delete.

    Counting vacancies restores an exact invariant that holds in steady state and
    post-migration:

        live_rows + vacated == last_seq          (per origin, vertex ∪ edge)

    and it keeps the property that made the contiguity check worth having: a row
    lost outside the write path — a bad migration, corruption — decrements the
    left side without incrementing `vacated`, so the balance breaks and the loss
    is visible. An accounted removal stays distinguishable from an unaccounted
    one."""
    return "vacated:" + origin


def c_list_index_built() -> str:
    """1 once `listkey` is known to cover every vertex in this store; absent otherwise.

    The state is recorded explicitly, and a lookup made while it is absent carries no authority
    (see `vertex.ListIndexUnbuilt`). `ensure_schema` sets it on a store with no vertices — a fresh
    store is trivially fully indexed, and every write thereafter maintains it in the same
    transaction — and `rebuild_list_index()` sets it after a backfill. A populated store that
    has never been backfilled matches neither, which is exactly the state that must be loud."""
    return "listkey:built"


def c_missing(field: str, *, committed_only: bool = False) -> str:
    """Vertices whose audited provenance field is missing (falsy).

    Maintained incrementally because `<field> IS NULL` over the whole corpus is
    exactly the predicate shape that makes `count(*)` a 6M-record dereference —
    there is no index that turns it into a seek, so it is a scan or it is a
    counter, and a scan at production scale returns not measured."""
    return "missing:" + field + (":committed" if committed_only else "")
