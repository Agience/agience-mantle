"""The lattice schema — LATTICE-CONTRACT.md §2, authoritative.

`vertex` and `edge` are reproduced EXACTLY as the contract specifies them.
Everything else in this module is bookkeeping that exists to make two hard rules
enforceable rather than aspirational:

  * **No `count(*)`, ever.** `count(*)` dereferences every record — EXPLAIN proves
    it loads 6M rows to produce one integer, which OOMs node 71's acceptor thread.
    So every count this store can answer is maintained INCREMENTALLY in `counter`,
    in the same transaction as the write that changes it.
  * **No `SKIP`/`OFFSET`, ever.** Measured on the live corpus: `SKIP` at depth 5M
    took 142,136ms; the equivalent keyset page took 743ms. Every paging API here
    is `WHERE id > :cur ORDER BY id LIMIT n`.

⚠ TWO DELIBERATE DEVIATIONS FROM CONTRACT §2, both flagged in the unit report:

1. **`task` sidecar table.** Contract §2 says there is NO `state` column and that
   a migration shim "lives inside `doc` JSON where it can never become a query
   predicate". The work pool nevertheless MUST have an indexed, selective
   predicate on task status — `pool.claim()` scanning the whole task history is
   the exact pathology contract §5 records as "queue silently dead on sqlite".
   Resolution: task coordination state lives in a SEPARATE table keyed on
   `vertex.id`, never as a column on `vertex`. The contract's intent (artifact
   *lifecycle* `state` must not be a query predicate) is preserved intact; task
   `status` is a different field with a different owner — it is the work pool's
   coordination state, not the artifact's lifecycle.

2. **`leaf_digest` table.** Merkle leaves are maintained incrementally (XOR out
   the old row hash, XOR in the new) rather than by full rescan. This is not an
   optimization: MEASURED on 71, a full rescan publish ran at 6,286 rows/sec —
   ~7 min for 2.7M rows — while the corpus grew 2.23M→2.73M during the same
   session. A rescan-built tree is STALE BEFORE IT FINISHES on a catching-up
   node, and publishing it advertises a root that never matched any real state.
"""
from __future__ import annotations

import sqlite3
from typing import List

# ═════════════════════════════════════════════════════════════════════════════
# THE RULE FOR ADDING A COLUMN — contract §2.3. Apply it BEFORE adding one.
#
#     A COLUMN IS LEGITIMATE ONLY IF EVERY OBSERVER WOULD COMPUTE THE SAME
#     VALUE FOR IT.
#
#   deterministic / content-addressed  -> column, indexable, a query predicate
#   frame-local / observer-dependent   -> content, in `doc`, NEVER a predicate
#
# `state` and `_rev` failed that test and are gone. `created_time` ALSO fails it —
# and is present anyway, by John's explicit dated override (2026-07-20, "just in
# case"). See the note on VERTEX_DDL. An override is not a counter-example to the
# rule; it is a decision taken WITH the rule understood, which is why it is
# recorded with its author and date rather than quietly absorbed.
#
# The point is not that observer-dependent values are worthless — they are
# content, and content lives in `doc`. The point is that a COLUMN IS A STANDING
# INVITATION TO A QUERY PREDICATE. `created_time` was documented as "local
# annotation ONLY, never a cross-observer sort key" and that rule was enforced by
# nothing but a comment; `ORDER BY created_time` would have been accepted by the
# engine every time. A rule enforced only by comment is not enforced.
#
# ⚠ THE `created_by` / `created_time` ASYMMETRY, because it looks inconsistent:
# both are claims, but IDENTITY IS DETERMINISTIC — every observer computes the
# same `uuid5(tenant, sub)` from the same issuer assertion — while two observers
# reading their own clocks disagree and there is NO FUNCTION THAT RECONCILES
# THEM. Same category, opposite answers, because the test is agreement and not
# provenance.
#
# ⚠ `_seq` IS NOT A COUNTEREXAMPLE. It is observer-dependent alone and is never
# interpreted alone: `(_origin, _seq)` is immutable, travels with the row, and
# every observer agrees on the pair once replicated. The pair is the value; the
# column is half of a composite. Contrast `created_time`, which every observer
# would compute DIFFERENTLY for the same event.
#
# ORDERING BELONGS ON EDGES. Placement in an ordering is a relation BETWEEN
# things — relative and frame-dependent — so it is `edge.order_key`, a fractional
# index. `created_time` was a second, worse ordering primitive sitting next to a
# working one. Across frames the answer is graph reachability (§1.2), and
# UNORDERED IS A VALID ANSWER (§C.7) — not a gap to paper over with a timestamp.
# ═════════════════════════════════════════════════════════════════════════════

# ── contract §2, verbatim ────────────────────────────────────────────────────
VERTEX_DDL: List[str] = [
    # ⚠⚠ `created_time` IS AN EXPLICIT, DATED OVERRIDE OF CONTRACT §2.2 BY JOHN, 2026-07-20.
    #
    #     John: "Leave `created_time` in. Just in case. We'll remove it later. Add it back."
    #
    # §2.2 removed it, and the reasoning there is NOT retracted: two observers reading their own
    # clocks disagree and no function reconciles them, so by THE COLUMN TEST it is content, not a
    # column. The override is a deliberate, temporary exception made with that known — recorded
    # here with its date and author so the next reader does not "fix" the schema back and
    # silently drop WHEN from 6.25M rows.
    #
    # ⚠ BE HONEST ABOUT WHAT THIS PROTECTS: NOTHING, YET. `created_time` is 100% EMPTY across the
    # 6.25M-row extract (measured 2026-07-20: 0 non-NULL). The column is PRECAUTIONARY —
    # "just in case" — not a rescue of existing data. An earlier justification in
    # LATTICE-OUTSTANDING §10 implied it was protecting captured timestamps; that was WRONG and
    # is corrected here.
    #
    # ⚠ TEXT, NOT INTEGER. §10 wrote `created_time INTEGER`; the values the legacy graph engine actually holds
    # are ISO-8601 STRINGS ('2026-07-17T15:19:38.173855+00:00'), measured on node 45. SQLite's
    # typing is advisory so INTEGER would not have failed loudly — it would just have made every
    # stored value a string in a column claiming otherwise.
    #
    # ⚠ THE COLUMN TEST STILL APPLIES TO THE VALUE. It is only safe as a column while it is the
    # AUTHORING observer's claim, carried immutably across replication — never re-stamped by
    # whoever imported the row. `vertex._attribute_time` records the claimant in
    # `doc.created_time_origin` for exactly this reason, and that stays.
    #
    # STILL REMOVED, and not up for revival: NO `state`, NO `_rev`.
    # ── ⚠ DEPRECATED: `created_by` AND `created_time` [John, 2026-07-21] ─────────────────────
    # "we should probably mark created_time and created_by as deprecated.. but make sure we have
    #  a deterministic method to track provenance and ordering before fully removing it."
    #
    # Both stay for now. Neither may be REMOVED until its replacement is populated AND verified on
    # the live store — see `test_deprecated_columns_have_deterministic_replacements`.
    #
    #   created_by   -> THE CREATION EDGE (`label=created`, `is_origin=1`, `propagate="r"`).
    #                   Contract §2.1 already makes the edge AUTHORITATIVE and the column the
    #                   non-authoritative duplicate: "the WHO is expressed twice and THE EDGE IS
    #                   AUTHORITATIVE, because authorization flows through it and it carries the
    #                   `propagate` mask, which a column cannot."
    #                   ⛔ REMOVAL PRECONDITION: the `creation` stage has run and every artifact
    #                   with a WHO has its edge. The stage READS this column to build the edges,
    #                   so the column is the SOURCE — dropping it before the edges exist destroys
    #                   the ability to rebuild them.
    #
    #   created_time -> for ORDERING: `(_origin, _seq)` within an origin (gap-free, MEASURED 0
    #                   gaps across all three origins 2026-07-21) and `edge.order_key` within a
    #                   frame.
    #                   -> for PROVENANCE: the `provenance` rung + `cited_from` + `origin_root`.
    #                   ⚠ AND THE HONEST LIMIT: there is NO total order ACROSS origins, and there
    #                   should not be. Two nodes' clocks do not compose into one timeline. If a
    #                   cross-origin ordering is ever wanted, that is graph reachability, not a
    #                   timestamp — UNORDERED IS A VALID ANSWER.
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
    # The VERSION LINEAGE. Without this index a lineage walk is a full scan, which is the
    # difference between versioning being usable and being nominal.
    "CREATE INDEX IF NOT EXISTS ix_v_root_id ON vertex(root_id)",
    "CREATE INDEX IF NOT EXISTS ix_v_root   ON vertex(origin_root)",
]

# ⚠ `origin_root` PASSES THE §2.3 COLUMN TEST — "would EVERY OBSERVER COMPUTE THE
# SAME VALUE?" — which is why it is a column and not a `doc` key.
#
# It is the collection's IMMUTABLE origin root: the top of the creation-lineage
# tree, fixed at creation and never moved thereafter. It does not depend on the
# observer, on a clock, on replication order, or on who is asking. Two nodes that
# replicate the same row compute the same root, forever. That is precisely the
# property `created_time` lacked and `_origin`/`_seq` were built to supply.
#
# ⛔ IT IS NOT `root_id`. Two different roots, orthogonal:
#     root_id     — the VERSION root. Versions move through time, all pointing at one.
#     origin_root — the CONTAINMENT root. Does not move at all, ever.
#
# WHY A COLUMN RATHER THAN DERIVED ON READ: it is the key root for content
# encryption (`content_cache.collection_key(root_secret, origin_root)`, P9.3
# resolved 2026-06-07: "`created_by`/'owner' is gone from the crypto path"). The
# grant gates WHETHER a key is issued; this determines WHICH key. Deriving it on
# read would put a lineage walk on every decrypt; deriving it lazily would mean a
# blob's key root could differ between two reads. Neither is acceptable for a value
# whose whole job is to never move.
#
# MEASURED 2026-07-21 on node 71: NO COLLECTION NESTS. Every collection artifact
# (`universe`, `ontology`, `stage.0..3`, `subjects`) has no parent, so today
# `origin_root == collection_id` for all 6.25M rows and the "walk" is depth-1.
# It is stored anyway BECAUSE it is flat today: GENESIS §5 designs for subject
# trees (`subject.math.topology.homotopy`), and if nesting arrives while content is
# keyed on the flat `collection_id`, every blob written under it needs re-keying —
# the exact orphaning P9.3 exists to prevent. Storing it now makes nesting free.

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
    # ⚠ `_leaf` = leaf_of(edge_key) — the Merkle leaf this edge XORs into (the SAME leaf_digest
    # table vertices use, so one tree covers both). Indexed because publish rebuilds a CHANGED leaf's
    # object from `WHERE _leaf = ?` — the read side of the anti-entropy plane, one indexed equality
    # instead of a corpus scan. Edges are the second half of the one sync path.
    "CREATE INDEX IF NOT EXISTS ix_e_leaf   ON edge(_leaf)",
]

# ── bookkeeping ──────────────────────────────────────────────────────────────
SUPPORT_DDL: List[str] = [
    # Proper time. One row per observer.
    #
    # ⚠ THE COLUMN IS `last_seq` AND IT HOLDS THE HIGHEST SEQ *ALREADY ISSUED*
    # (or observed, for foreign origins) — NOT the next one to hand out. It was
    # briefly named `next_seq`, which is an off-by-one trap for anyone writing an
    # accounting check: `next_seq - 1` under-counts by exactly one allocation.
    # Renamed rather than documented, because a name that has to be corrected in
    # prose will be read wrong by whoever does not read the prose. Use
    # `seq.seq_accounting()` and never do this arithmetic by hand.
    """CREATE TABLE IF NOT EXISTS seq_counter (
         origin   TEXT PRIMARY KEY,
         last_seq INTEGER NOT NULL,
         is_local INTEGER NOT NULL DEFAULT 0
       )""",

    # Every count this store can answer. Maintained incrementally, in the write
    # transaction. A missing row means zero. NEVER backfilled with count(*).
    """CREATE TABLE IF NOT EXISTS counter (
         name TEXT PRIMARY KEY,
         n    INTEGER NOT NULL
       )""",

    # Incremental merkle. `digest` is 8 raw bytes, not an INTEGER, because
    # row hashes are UNSIGNED 64-bit and SQLite INTEGER is signed — storing them
    # as integers silently wraps half the key space negative and the XOR
    # round-trip stops being an identity.
    """CREATE TABLE IF NOT EXISTS leaf_digest (
         leaf   INTEGER PRIMARY KEY,
         digest BLOB NOT NULL
       )""",

    # Small key/value metadata that is NOT a counter and must not be recomputed by
    # the counter-drift audit. `merkle.leaves` lives here: the store's operating
    # Merkle resolution (the DERIVED `natural_leaves(corpus)`), recorded so every
    # `open_lattice` resolves the same value and `reshard()` can move it in one place.
    """CREATE TABLE IF NOT EXISTS meta (
         k TEXT PRIMARY KEY,
         v TEXT
       )""",

    # The DEMAND cache index (local, never replicated). One row per REACHED artifact,
    # carrying its accumulated `mass` and last-touch `ts`. Decay (demurrage) is applied
    # in the ember layer (which owns the one `prism.law` kernel); the store only holds
    # the raw scalar and its timestamp. A row here is an EVICTABLE cache copy; a held
    # row with NO demand row is own-authored and is never evicted. This is what lets a
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
    # bounded index walk with NO sort. The legacy store materialised and sorted
    # the entire task history per worker per 5s, then shuffled the result away.
    "CREATE INDEX IF NOT EXISTS ix_t_pending  ON task(ct, status, priority DESC, id)",
    # The /status panel: bounded window off the indexed terminal buckets.
    "CREATE INDEX IF NOT EXISTS ix_t_terminal ON task(ct, status, completed_at DESC)",

    # ── the KEYED arm: multi-valued doc fields, DISCRIMINATED BY CONTENT TYPE ──
    #
    # A dictionary lookup is KEYED retrieval (word -> senses), not similarity, and a JSON list
    # cannot be indexed in place: SQLite indexes scalars, and `json_each` is a table-valued
    # function no index can cover. So the list is unrolled into rows here.
    #
    # ⚠⚠ `ct` IS IN THE TABLE AND IS THE FIRST-CLASS POINT OF THE DESIGN, NOT DENORMALISATION
    # FOR SPEED. The legacy predecessor indexed `(field, value)` ONLY, so `lemmas` was one
    # undiscriminated namespace shared by every content type in the corpus. MEASURED on the live
    # system: `lookup_by_lemma('spaceship', 200)` returned 200 hits and ZERO synsets, because
    # 6,063,979 `wiki-*` rows share that index with 117,659 `wn-*` rows. Callers "fixed" this by
    # post-filtering the result (`code.py:75`), which CANNOT work — the LIMIT has already been
    # spent on distractors before the filter runs, so filtering an all-distractor page yields an
    # empty list and the caller reports "not found" for a word the store holds. Type
    # discrimination must happen INSIDE the seek, ahead of the LIMIT, or it does not happen.
    #
    # `(field, value, ct)` is therefore the index key: "the wordnet senses of 'dog'" is ONE
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
    # APPEND-ONLY history rows — "who touched what, when, allowed or denied". NOT edges:
    # the edge table upserts by (src,dst,label), which would collapse repeated accesses of
    # the same (principal, artifact, action) into one row and destroy the history an audit
    # log exists to keep. NOT vertices: log rows are operational bookkeeping (like `task`),
    # not observations in the artifact universe. See `db/lattice/audit.py`.
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
    # ⛔ DO NOT ADD `cache_size` / `mmap_size` HERE FOR SCAN SPEED — MEASURED, IT IS SLOWER.
    # The migration stages looked cache-starved: 5,880 random read IOPS at ~1.1 KB/read, device
    # 92% utilised, 30 MB RSS on a 23 GB box. Raising the cache to 2 GiB and mapping 16 GiB
    # measured **0.92x — a 8% REGRESSION** (120k rows: 22.25s default vs 24.30s tuned, node 45,
    # 2026-07-21), and the tuned run had the warmer page cache of the two.
    #
    # The reason it cannot help: these stages do a SINGLE-PASS sequential scan. Each page is
    # read once and evicted, so a larger cache has nothing to re-serve, while `mmap` adds
    # per-page mapping overhead to every one of them. A page cache pays for REPEATED access,
    # which this workload does not have.
    #
    # The real cost is the scan itself (~5,400 rows/s => ~19 min per pass over 6.25M rows).
    # That is fixed by not scanning — an index on the extracted JSON key — not by buffering.
]

ALL_DDL: List[str] = VERTEX_DDL + EDGE_DDL + SUPPORT_DDL


def apply_pragmas(conn: sqlite3.Connection) -> None:
    for p in PRAGMAS:
        conn.execute(p)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotent. Safe to call on every open, from every thread and process."""
    # ⚠ COLUMN MIGRATION RUNS FIRST, BEFORE ALL_DDL — NOT AFTER. `ALL_DDL` contains
    # `CREATE INDEX ix_v_root ON vertex(origin_root)`, and an index over a column the
    # legacy table does not have raises `no such column` and aborts the whole open.
    # Running the migration afterwards would therefore never be reached.
    _migrate_vertex_columns(conn)
    _migrate_edge_columns(conn)
    for stmt in ALL_DDL:
        conn.execute(stmt)
    _migrate_next_seq(conn)
    _mark_empty_store_indexed(conn)


# Columns added to `vertex` after stores already existed in the field, newest last.
# `CREATE TABLE IF NOT EXISTS` does NOT add columns to an existing table, so without
# this every pre-existing store fails to open with `no such column`.
#
# MEASURED on node 45 (2026-07-21): `lattice/work/lattice.db` has neither `origin_root`
# nor `listkey`, and `work/resumed.db` additionally lacks `created_time` — so this is a
# live migration path, not a defensive one.
#
# Every entry must be NULLable with no default. A backfill is a separate, explicit pass:
# `origin_root` is derived from the creation lineage (`vertex._origin_root`), and silently
# defaulting it here would write a computed value that no observer agreed on — the exact
# sentinel-as-measurement failure the `ic = 0.0` comment in enrich_wordnet.py catalogues.
_VERTEX_ADDED_COLUMNS = (
    ("created_time", "TEXT"),
    ("origin_root", "TEXT"),
    # ⭐ THE VERSION LINEAGE (John, 2026-07-21: "lattice needs the root_id", "version_id is
    # latest"). Mantle's rule, reproduced: the FIRST version of an artifact has `id == root_id`,
    # so `id` is the VERSION identity and `root_id` groups the lineage. That is why this needs no
    # primary-key change — versions coexist as separate rows, each with its own id.
    #
    # ⛔ ADDING THE COLUMN IS NOT THE FIX ON ITS OWN. Existing rows get NULL, and a NULL
    # discriminator is invisible to every query that filters on it — precisely how the `wn-*`
    # rows with `ct IS NULL` went missing. `backfill_root_id()` must run before any lineage query
    # is trusted, and it reports what it filled rather than assuming.
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


# Columns added to `edge` after stores already existed. Same rule as the vertex list: NULLable, no
# default, and a NULL value means "not yet backfilled" — never a silently-computed one.
_EDGE_ADDED_COLUMNS = (
    # ⭐ `_leaf` = leaf_of(edge_key). Adding the column is NOT the whole fix: existing edges get NULL
    # and are invisible to a leaf's `WHERE _leaf = ?` rebuild until `backfill_edge_leaf()` stamps them
    # and XORs each into `leaf_digest` — the exact story `root_id`/`_leaf` had for vertices. Until
    # backfilled the edge tree is incomplete, which `merkle_coverage` reports; a fresh store is
    # complete by construction because `edge._add_one` stamps `_leaf` on every write.
    ("_leaf", "INTEGER"),
)


def _migrate_edge_columns(conn: sqlite3.Connection) -> None:
    """Add any `edge` column introduced after this store was created. No-op on a fresh or up-to-date
    store. Must run BEFORE `ALL_DDL` for the same reason as the vertex migration: `ix_e_leaf` indexes
    `_leaf`, and an index over a column a legacy edge table lacks aborts the whole open."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(edge)")}
    if not cols:                      # fresh store — nothing to migrate
        return
    for name, decl in _EDGE_ADDED_COLUMNS:
        if name not in cols:
            conn.execute("ALTER TABLE edge ADD COLUMN %s %s" % (name, decl))


def _mark_empty_store_indexed(conn: sqlite3.Connection) -> None:
    """Certify `listkey` as built IFF this store has no vertices yet.

    A store with zero vertices is trivially, exhaustively indexed, and every write from here on
    maintains `listkey` inside the same transaction as the vertex row — so the certificate stays
    true without anyone rescanning. This is what keeps the guard in `c_list_index_built` from
    firing on every newly-created store.

    ⚠ IT MUST NOT FIRE ON A POPULATED ONE. The vertex count is read from the incrementally
    maintained counter, never `count(*)` — and the ABSENCE of that counter row reads as 0, which
    would certify a populated legacy store as indexed and re-open the silent-empty hole this
    marker closes. So a missing `vertex` counter is treated as UNKNOWN (not zero) and the store
    is left uncertified: `rebuild_list_index()` is then the only way to earn the marker."""
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


def _migrate_next_seq(conn: sqlite3.Connection) -> None:
    """Rename the short-lived `seq_counter.next_seq` column to `last_seq`.

    Nothing outside this package ever referenced it (checked), and no lattice
    store is deployed — Phase 5.0 is a rebuild. This exists only so a store file
    created during today's parallel unit work does not become unreadable."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(seq_counter)")}
    if "next_seq" in cols and "last_seq" not in cols:
        conn.execute("ALTER TABLE seq_counter RENAME COLUMN next_seq TO last_seq")


# ── counter names — the ONLY sanctioned way to answer "how many" ─────────────
# Kept as functions so a typo is an AttributeError at import, not a silently
# wrong number at read time (a missing counter row reads as 0, which is exactly
# the wrong-answer-not-empty-answer failure class contract §5 catalogues).

def c_vertex_total() -> str:
    return "vertex"


def c_edge_total() -> str:
    return "edge"


def c_ct(ct: str) -> str:
    return "ct:" + ct


def c_collection(collection_id: str, *, committed_only: bool = False) -> str:
    return "col:" + collection_id + (":committed" if committed_only else "")


def c_task_status(ct: str, status: str) -> str:
    return "task:" + ct + ":" + status


def c_rows(origin: str) -> str:
    """LIVE rows currently carrying `origin`, across `vertex` AND `edge`.

    The left-hand side of the allocation-accounting invariant. Because allocation
    is injective and each allocation writes exactly one row, this equals
    `COUNT(DISTINCT _seq)` for that origin — maintained incrementally so the
    invariant can be checked without a scan."""
    return "rows:" + origin


def c_vacated(origin: str) -> str:
    """Seqs whose row no longer exists BY AN ACCOUNTED WRITE — an update that
    superseded it, or a delete.

    ⚠ AN UPDATE ALLOCATES A FRESH `_seq` AND VACATES THE OLD ONE, so surviving
    rows are legitimately non-contiguous. `put(a)->1, put(b)->2, put(a')->3`
    leaves `[(b,2), (a,3)]` — contiguous but starting at 2. Asserting row
    contiguity therefore FAILS on any store that has ever taken an update, which
    is every real store; earlier units only passed because they ran against
    seeded, insert-only fixtures.

    Counting vacancies restores an exact invariant that holds in steady state AND
    post-migration:

        live_rows + vacated == last_seq          (per origin, vertex ∪ edge)

    and it KEEPS the property that made the contiguity check worth having: a row
    lost outside the write path — a bad migration, corruption — decrements the
    left side WITHOUT incrementing `vacated`, so the balance breaks and the loss
    is visible. That is the whole point; an accounted removal must be
    distinguishable from an unaccounted one."""
    return "vacated:" + origin


def c_list_index_built() -> str:
    """1 once `listkey` is known to cover every vertex in this store; absent otherwise.

    ⚠ THIS MARKER EXISTS BECAUSE AN UNBUILT INDEX AND AN ABSENT WORD ARE THE SAME QUERY RESULT.
    `listkey` is a derived index, and it was added to a schema that already has populated stores
    in the field (node 45's migrated store predates it). On such a store the table exists and is
    EMPTY, so `lookup_by_lemma('dog')` returns `[]` — which is not "I could not answer", it is
    "the lexicon does not contain 'dog'". That is a WRONG ANSWER, not an empty one, and it is
    the precise failure class contract §5 catalogues six times over.

    So the state is recorded explicitly and lookups REFUSE when it is absent (see
    `vertex.ListIndexUnbuilt`). `ensure_schema` sets it on a store with no vertices — a fresh
    store is trivially fully indexed, and every write thereafter maintains it in the same
    transaction — and `rebuild_list_index()` sets it after a backfill. A populated store that
    has never been backfilled matches neither, which is exactly the state that must be loud."""
    return "listkey:built"


def c_missing(field: str, *, committed_only: bool = False) -> str:
    """Vertices whose audited provenance field is MISSING (falsy).

    Maintained incrementally because `<field> IS NULL` over the whole corpus is
    exactly the predicate shape that makes `count(*)` a 6M-record dereference —
    there is no index that turns it into a seek, so it is a scan or it is a
    counter, and a scan at production scale returns NOT MEASURED."""
    return "missing:" + field + (":committed" if committed_only else "")
