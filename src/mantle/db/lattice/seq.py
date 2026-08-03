"""Proper time — the `(_origin, _seq)` allocator — and the write-lock discipline.

This module is the whole reason `_rev` could be deleted, so it is worth being
precise about what `_seq` is and, more importantly, what it is NOT.

    _origin   the AUTHORING OBSERVER — immutable across replication
    _seq      that observer's PROPER TIME — monotonic, gap-free, local,
              incremented once per AUTHORED EVENT

**`_seq` IS NOT A CLOCK.** It is never derived from wall time. MEASURED on this
box: 2000 successive `time.time_ns()` calls yield **1 distinct value** — Windows'
system clock ticks at 15.6ms, so a timestamp-derived "version" collapses an entire
batch onto one value. The legacy `_rev` was exactly `time.time_ns()`, and every
downstream defect it caused traces back to that collapse:

  * `content_tier.py:317` carries a two-line warning in red ink — "`_rev` IS NOT
    UNIQUE — A BARE `_rev > :r` CURSOR SKIPS ROWS PERMANENTLY" — and a 20-line
    workaround to finish a revision GROUP by equality before advancing. That
    entire mechanism exists solely because the clock was not injective.
  * probe fixtures stamped with a `_rev` four months in the future pinned publish
    cursors beyond every real row and MUTED FIVE NODES PERMANENTLY. A counter
    cannot be four months in the future; a clock can.

Allocating from the store instead buys three properties a clock cannot give:

  1. **Injective.** One event, one value. `_seq > :cursor` is a correct cursor
     with no group-completion dance.
  2. **Gap-free.** Allocation happens inside the SAME transaction as the write it
     versions, so a rolled-back write rolls back its seq. That is what makes the
     ACCOUNTING IDENTITY exact — `live_rows + vacated == last_seq` — which is how
     a row lost outside the write path is detected at all.
     ⚠ It does NOT make a backlog computable by subtraction. `high_water - cursor`
     counts ALLOCATIONS, and once updates vacate seqs that is not a row count; the
     method that did so has been DELETED. Backlog is `pending_publish()`, which
     counts rows per feed. See vertex.py.
  3. **Bounded.** It counts this observer's own events and nothing else, so it
     cannot be poisoned by another node's clock, timezone, or fixture.

⚠ **GAP-FREENESS IS GUARANTEED FOR THE LOCAL ORIGIN ONLY.** Foreign origins are
tracked in the same table as a HIGH-WATER MARK: a replicating node observes a
peer's seqs, it does not allocate them, and it may observe them out of order or
with holes it has not yet received. `local_max()` is gap-free; `high_water()` is
not, and callers must not assume otherwise.

⚠ **ONE COUNTER PER OBSERVER, SHARED ACROSS `vertex` AND `edge`** (contract §4
RESOLVED-5). `seq_counter` has `origin` as its sole PRIMARY KEY, deliberately.
The argument is uniqueness, not taste: §1.2 states that `(_origin, _seq)` is
"a globally unique version identity that survives replication", and with
per-table counters a vertex and an edge would both be allocated `(71, 5)` — a
collision that makes the documented claim false.

Two consequences that catch people out, both load-bearing:

  * **GAP-FREENESS IS A PROPERTY OF THE UNION, NOT OF EITHER TABLE.** Vertex seqs
    may legitimately read `{1,3,7}` while edge seqs read `{2,4,5,6}`. Any
    per-table contiguity assertion is WRONG and will produce false failures.
    Check `vertex ∪ edge`.
  * **There must be exactly ONE `SeqAllocator` per `(LatticeConn, origin)`.** Two
    instances over one durable row both cache in-transaction — which is precisely
    what makes allocation gap-free and cheap — so they can each read
    `last_seq = N` and both hand out `N`. `allocator_for()` is therefore the only
    sanctioned way to obtain one, it REUSES by default, and a conflicting second
    instance RAISES rather than silently overwriting the registry.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from . import schema as _schema

_BUSY_TIMEOUT_MS = 30_000
_BEGIN_RETRIES = 60
_BEGIN_BACKOFF_S = 0.05


class LatticeConn:
    """One SQLite database, one connection per thread, one explicit write lock.

    Autocommit (`isolation_level=None`) so transactions are ours to demarcate —
    the stdlib's implicit-BEGIN behaviour would open a DEFERRED transaction that
    upgrades to a write lock mid-statement, which is precisely how you get
    SQLITE_BUSY on a statement that already did half its work."""

    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        # Serializes writers within THIS process so they queue on a cheap mutex
        # rather than on SQLITE_BUSY retries. Cross-process writers still
        # contend at the file lock, which `busy_timeout` + the retry loop handle.
        self._wlock = threading.RLock()
        # THE allocator registry, keyed by origin, held ON THE CONNECTION.
        # It lives here rather than in a module-level dict keyed on `id(db)`
        # because `id()` is recycled after garbage collection, so a global
        # registry can hand a new connection the dead one's allocator — a second
        # way to produce duplicate `(_origin, _seq)`. Instance state has the
        # exact lifetime we want and cannot leak.
        self._allocators: "dict[str, SeqAllocator]" = {}
        self._alloc_lock = threading.Lock()
        with self.write() as cur:
            _schema.ensure_schema(cur)

    # ── connections ──────────────────────────────────────────────────────────
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000.0,
                                isolation_level=None)
            c.row_factory = sqlite3.Row
            _schema.apply_pragmas(c)
            c.execute("PRAGMA busy_timeout=%d" % _BUSY_TIMEOUT_MS)
            self._local.c = c
            self._local.depth = 0
        return c

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def read(self) -> sqlite3.Cursor:
        """A cursor for read-only work. No transaction is opened."""
        return self.conn().cursor()

    # ── the write lock ───────────────────────────────────────────────────────
    @contextmanager
    def write(self) -> Iterator[sqlite3.Cursor]:
        """`BEGIN IMMEDIATE` … `COMMIT`. Reentrant: a nested `write()` JOINS the
        outer transaction rather than opening a second one, so a store method may
        freely call another store method without deadlocking itself or —
        far worse — committing half of a caller's atomic unit."""
        c = self.conn()
        if self._depth() > 0:                     # already inside a write txn
            self._local.depth += 1
            try:
                yield c.cursor()
            finally:
                self._local.depth -= 1
            return

        with self._wlock:
            self._begin_immediate(c)
            self._local.depth = 1
            # EVERY allocator on this connection, not just one. Driving a single
            # allocator's lifecycle is how the duplicate-seq bug hid: the second
            # store's allocator was never begun, flushed or ended, so its cache
            # persisted across transactions while its counter never reached disk.
            allocs = self.allocators()
            for a in allocs:
                a.begin()
            try:
                cur = c.cursor()
                yield cur
                for a in allocs:
                    a.flush(cur)
                c.execute("COMMIT")
            except BaseException:
                try:
                    c.execute("ROLLBACK")
                except Exception:
                    pass
                for a in allocs:                  # a rolled-back write consumes no proper time
                    a.discard()
                raise
            finally:
                self._local.depth = 0
                for a in allocs:
                    a.end()

    def allocators(self) -> "list[SeqAllocator]":
        with self._alloc_lock:
            return list(self._allocators.values())

    @staticmethod
    def _begin_immediate(c: sqlite3.Connection) -> None:
        """Take the write lock UP FRONT. Retried explicitly because a `timeout=`
        alone does not cover the BEGIN itself on every platform/build."""
        last: Optional[Exception] = None
        for attempt in range(_BEGIN_RETRIES):
            try:
                c.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) and "busy" not in str(e):
                    raise
                last = e
                time.sleep(_BEGIN_BACKOFF_S * min(8, attempt + 1))
        raise sqlite3.OperationalError(
            "could not acquire the lattice write lock after %d attempts: %s"
            % (_BEGIN_RETRIES, last))

    def close(self) -> None:
        c = getattr(self._local, "c", None)
        if c is not None:
            c.close()
            self._local.c = None


# ── proper-time allocation ───────────────────────────────────────────────────

class SeqAllocator:
    """Allocates `_seq` for ONE origin, inside a `LatticeConn.write()` block.

    The in-transaction cache is what makes this gap-free AND cheap: the counter
    row is read once per transaction and written back once at COMMIT, while
    individual allocations are pure Python increments. A ROLLBACK discards the
    cache, so the next transaction re-reads the durable value and reissues the
    seqs the rolled-back one had taken. No gap."""

    def __init__(self, db: LatticeConn, origin: str):
        self.db = db
        self.origin = origin
        self._local = threading.local()

    # -- transaction lifecycle (driven by LatticeConn.write) ------------------
    def begin(self) -> None:
        self._local.cache = None
        self._local.dirty = False

    def discard(self) -> None:
        self._local.cache = None
        self._local.dirty = False

    def end(self) -> None:
        self._local.cache = None
        self._local.dirty = False

    def flush(self, cur: sqlite3.Cursor) -> None:
        if not getattr(self._local, "dirty", False):
            return
        # MAX, never a bare assignment. For a single correct allocator the cache is
        # always >= the durable value, so MAX is a no-op — but it makes the
        # high-water mark structurally incapable of moving BACKWARDS, which is the
        # property the whole scheme rests on. A plain `= excluded.last_seq` would
        # let any future bug that lowers a cache silently re-issue live seqs.
        cur.execute(
            "INSERT INTO seq_counter(origin, last_seq, is_local) VALUES(?,?,1) "
            "ON CONFLICT(origin) DO UPDATE SET "
            "last_seq = MAX(last_seq, excluded.last_seq), is_local = 1",
            (self.origin, int(self._local.cache)))
        self._local.dirty = False

    # -- allocation ----------------------------------------------------------
    def allocate(self, cur: sqlite3.Cursor) -> int:
        """The next `_seq` for the local origin. MUST be called inside `write()`.

        Monotonic and gap-free: values are handed out 1, 2, 3, … with no holes
        and no repeats, for the lifetime of the store, across threads and
        processes."""
        if getattr(self._local, "cache", None) is None:
            self._local.cache = self._load(cur)
        self._local.cache += 1
        self._local.dirty = True
        return int(self._local.cache)

    def mark(self) -> Any:
        """Snapshot the in-transaction cache so a SAVEPOINT rollback can restore
        it. Without this, a per-document savepoint that rolls back a failed write
        leaves the Python counter advanced while the durable row did not change —
        which is a GAP, and a gap breaks `local_max() - cursor` as an exact
        backlog. `put_many` uses this on every doc."""
        return getattr(self._local, "cache", None)

    def restore(self, snapshot: Any) -> None:
        self._local.cache = snapshot

    def _load(self, cur: sqlite3.Cursor) -> int:
        row = cur.execute("SELECT last_seq FROM seq_counter WHERE origin = ?",
                          (self.origin,)).fetchone()
        if row is not None:
            return int(row[0])
        return self._recover(cur)

    def _recover(self, cur: sqlite3.Cursor) -> int:
        """Bootstrap the counter for an origin that has rows but no counter row —
        a store file restored from backup, or one written before this table
        existed. `MAX(_seq)` over `ix_v_origin` is an index-only seek to the end
        of one origin's range: it is O(log n) and touches ONE row. It is not a
        `count(*)` and must never be replaced by one.

        Scans BOTH tables and takes the max, because one counter is shared across
        `vertex` and `edge` (RESOLVED-5) — recovering from `vertex` alone would
        re-issue every seq an edge already holds.

        ⚠ **THIS PATH IS THE TAIL-TRUNCATION BLIND SPOT.** If the counter row is
        gone AND the highest-`_seq` rows were lost, `MAX(_seq)` recovers a value
        BELOW the true high-water and this observer silently re-issues seqs that
        already exist on its peers. Recovery from data is a last resort, not a
        routine: `seq_counter` is the authority, it is written on every commit, and
        `check_tail()` exists to detect the discrepancy when both survive."""
        best = 0
        for table in ("vertex", "edge"):
            r = cur.execute(
                "SELECT _seq FROM %s WHERE _origin = ? ORDER BY _origin DESC, _seq DESC LIMIT 1"
                % table, (self.origin,)).fetchone()
            if r is not None and isinstance(r[0], int):
                best = max(best, int(r[0]))
        return best

    # -- read-side -----------------------------------------------------------
    def local_max(self) -> int:
        """Highest `_seq` this observer has ALLOCATED. Gap-free, so
        `local_max() - cursor` is the exact publish backlog — O(1)."""
        r = self.db.read().execute(
            "SELECT last_seq FROM seq_counter WHERE origin = ?", (self.origin,)).fetchone()
        return int(r[0]) if r else 0


def observe_seq(cur: sqlite3.Cursor, origin: str, seq: Any) -> None:
    """Record a FOREIGN origin's seq as a high-water mark.

    ⚠ This is NOT allocation and the result is NOT gap-free — we observe a peer's
    proper time, we do not generate it, and replication delivers it out of order
    and with holes. `is_local` stays 0 so nothing can later mistake this row for
    an allocator and hand out a seq some peer already used."""
    if not isinstance(seq, int):
        return
    cur.execute(
        "INSERT INTO seq_counter(origin, last_seq, is_local) VALUES(?,?,0) "
        "ON CONFLICT(origin) DO UPDATE SET last_seq = MAX(last_seq, excluded.last_seq)",
        (origin, int(seq)))


def high_water(db: LatticeConn, origin: str) -> int:
    """Highest `_seq` SEEN for `origin`, local or foreign. May have holes below
    it for foreign origins — see `observe_seq`.

    This is the PERSISTED per-origin high-water mark. It is written on every
    commit and is monotonically non-decreasing by construction (`flush` uses MAX),
    which is what lets `check_tail` see losses that the data alone cannot show."""
    r = db.read().execute("SELECT last_seq FROM seq_counter WHERE origin = ?",
                          (origin,)).fetchone()
    return int(r[0]) if r else 0


def write_mark(db: LatticeConn) -> Tuple[Tuple[str, int], ...]:
    """EVERY origin's `last_seq`, sorted — **the store's whole-of-store WRITE MARK.**

    `high_water` answers for ONE origin, which is the wrong question for a reader
    holding a DERIVED cache and asking *"has anything under me changed since I last
    looked?"*. This answers exactly that question, and it answers it from bookkeeping
    the writer already has to maintain in order to write at all:

      · **`seq_counter.last_seq`, per origin** — every INSERT and every UPDATE allocates
        a fresh `_seq`, over `vertex` AND `edge`, because contract §4 RESOLVED-5 gives
        one counter per OBSERVER spanning both tables. A REPLICATED row `observe_seq`s
        its foreign origin, so a change arriving from a peer moves this too, and `flush`
        uses MAX, so it is monotonically non-decreasing and cannot be walked backwards.
      · **the `vertex` and `edge` row totals** — ⛔ AND THEY ARE HERE BECAUSE A DELETE
        MOVES NEITHER `last_seq` NOR ANY `_seq`. `delete_artifact` / `delete_edge` remove
        the row, XOR the merkle leaf and `vacate` the origin; they allocate nothing. So a
        mark built on `seq_counter` alone reports "nothing has changed" across a deletion
        — a cache would go on serving an artifact that no longer exists, which is a WRONG
        answer, not a stale one. `test_freshness.py::test_write_mark_moves_on_every_write`
        caught this by asserting all four kinds; the first version of this function had
        it wrong and every other test still passed.

    Those two counters are the same incremental counters the store already maintains in
    the write transaction (`_recount`), never a `count(*)`.

    ⛔ IT IS NOT A SIDE-CAR AND NOBODY HAS TO REMEMBER TO BUMP IT
    ([[everything-is-an-artifact]]). It is not a version number written beside the data
    by whoever last touched it; it is the counter the WRITER already advances inside the
    write transaction in order to allocate at all. A writer that forgot to move it could
    not write. That is the whole reason a cache may hang its validity on it: an
    invalidation that depends on someone remembering to emit an event is an invalidation
    that will one day not happen.

    **The reading a cache actually wants is the NEGATIVE one.** Unchanged mark ⇒ nothing
    in this store was written ⇒ every derivation built from it is still the derivation of
    what is in it. One indexed read over a table with one row per observer (MEASURED on
    71's 5.7 GB lattice: **9.3 µs** at this store, 14.1 µs through `ember.ontology.freshness`),
    and it is the only read a warm cache needs
    to make. A mark that HAS moved says only that *something* changed — attributing it to
    a particular artifact is `version_of`'s job, not this one's."""
    # ONE round trip, not three. This is polled on every public entry into `ember.ontology.
    # wn_store`, and three separate `execute()`s measured 13.8 µs against 5.6 µs for the union —
    # the whole point of the gate is that a warm cache can afford to take it every time.
    # `'#' ||` namespaces the counters so a counter name can never collide with an origin.
    rows = db.read().execute(
        "SELECT origin AS k, last_seq AS v FROM seq_counter "
        "UNION ALL "
        "SELECT '#' || name, n FROM counter WHERE name IN (?, ?)",
        (_schema.c_vertex_total(), _schema.c_edge_total())).fetchall()
    return tuple(sorted((str(r[0]), int(r[1])) for r in rows))


def max_observed(db: LatticeConn, origin: str) -> int:
    """Highest `_seq` actually PRESENT in `vertex ∪ edge` for `origin`. Two
    index-only seeks; never a scan, never a `count(*)`."""
    best = 0
    cur = db.read()
    for table in ("vertex", "edge"):
        r = cur.execute(
            "SELECT _seq FROM %s WHERE _origin = ? ORDER BY _origin DESC, _seq DESC LIMIT 1"
            % table, (origin,)).fetchone()
        if r is not None and isinstance(r[0], int):
            best = max(best, int(r[0]))
    return best


def check_tail(db: LatticeConn, origin: str, *, scan: bool = False) -> Dict[str, Any]:
    """Endpoint health for one origin's proper time. **Verdict, not a boolean.**

    Returns `{"origin", "last_seq", "max_observed", "vacated", "live_rows",
    "local_only", "rows_lost", "verdict", "method"}` where `verdict` is one of:

        "intact"        fully accounted: no rows lost anywhere. (scan only)
        "tail_intact"   the ENDPOINT is consistent. Does NOT rule out interior
                        loss — see below. (cheap, insert-only stores only)
        "loss_detected" `rows_lost` rows are gone outside the write path.
        "undecidable"   this call cannot answer. NOT a pass — report SKIP.

    ⚠⚠ **THE PREVIOUS VERSION OF THIS FUNCTION WAS WRONG AND FALSE-FAILED HEALTHY
    STORES.** It computed `missing_tail = high_water - max_observed`, i.e. it
    subtracted SEQ VALUES to answer a ROW question. An ordinary
    `delete_artifact()` of the newest row vacates that seq, so `max_observed`
    legitimately shrinks and the check reported data loss on a perfectly healthy,
    fully-accounted store. MEASURED (Unit P): after `delete_artifact("a4")` on a
    5-row store it reported `missing_tail=1, truncated=True` while
    `seq_accounting(scan=True)` correctly reported `vacated=1, balanced=True`.

    Worse, it emitted **byte-identical output** for that accounted delete and for a
    genuine raw-SQL tail loss — so the check that fired was the one that could not
    tell you anything, while the one that could stayed quiet, and two checks in the
    same suite contradicted each other about the same store.

    **The verdict is now derived from ACCOUNTING, never from subtraction.** A
    removal that went through the store increments `vacated`; one that did not,
    does not. That difference is the entire signal.

    ⚠ **WHAT IS IRREDUCIBLE HERE, stated rather than guessed:** once `vacated > 0`,
    *tail* loss cannot be distinguished from *interior* loss, and a shrunken
    `max_observed` cannot be distinguished from a legitimately vacated endpoint,
    **without recording WHICH seqs were vacated — and we only COUNT them.** A
    per-seq vacancy record would be a second write-amplified index to answer a
    question the accounting identity already answers globally, so it is not built.
    Consequently the honest report is "rows lost", not "tail truncated", and the
    cheap endpoint answer exists only where it is exact.

    So this function no longer has a cheap universal mode, and does not pretend to:

      * `vacated == 0` (insert-only — the post-migration case this was written for)
        the endpoint IS exact: every allocated seq must still have a row, so
        `last_seq - max_observed` is a true count of lost tail rows. Two index
        seeks. Verdict `tail_intact` deliberately does NOT claim `intact`, because
        an interior row could still be missing.
      * `scan=True` — defer to `seq_accounting`, which counts rows and is
        vacancy-correct. This detects loss ANYWHERE, tail or interior, and is the
        authoritative answer.
      * otherwise — `undecidable`. Call with `scan=True`.

    A foreign origin is always `undecidable`: its seqs are observed, not allocated,
    so a gap below the high-water is replication lag, not loss."""
    acc = seq_accounting(db, origin, scan=scan)
    obs = max_observed(db, origin)
    base = {"origin": origin, "last_seq": acc["last_seq"], "max_observed": obs,
            "vacated": acc["vacated"], "live_rows": acc["live_rows"],
            "local_only": acc["local_only"]}

    if not acc["local_only"]:
        return dict(base, rows_lost=None, verdict="undecidable", method="none",
                    reason="foreign origin: seqs are observed, not allocated, so a "
                           "gap below the high-water is replication lag, not loss")
    if scan:
        lost = max(0, acc["unaccounted"])
        return dict(base, rows_lost=lost, method="accounting_scan",
                    verdict="loss_detected" if lost else "intact")
    if acc["vacated"] == 0:
        lost = max(0, acc["last_seq"] - obs)
        return dict(base, rows_lost=lost, method="endpoint_insert_only",
                    verdict="loss_detected" if lost else "tail_intact")
    return dict(base, rows_lost=None, verdict="undecidable", method="none",
                reason="vacated=%d: a vacated endpoint is indistinguishable from a "
                       "lost one without a per-seq vacancy record. Call with "
                       "scan=True for the authoritative answer." % acc["vacated"])


def allocator_for(db: LatticeConn, origin: str, *,
                  override: Optional[SeqAllocator] = None) -> SeqAllocator:
    """**THE only sanctioned way to obtain a `SeqAllocator`.** Reuses by default.

    There must be exactly ONE allocator per `(LatticeConn, origin)`, because
    `(_origin, _seq)` is required to be a globally unique version identity and one
    counter per OBSERVER is shared across `vertex` and `edge` (contract §4
    RESOLVED-5). Two allocator instances over one durable `seq_counter` row both
    cache in-transaction, so each can read `last_seq = N` and both hand out `N`.
    MEASURED against the pre-fix code: 10 vertices + 10 edges written the naive way
    produced vertex seqs `1..10` and edge seqs `2..11` — **9 duplicate
    `(_origin, _seq)` pairs out of 20 rows.**

    THE SAFE PATH IS THE DEFAULT. Constructing the two stores the obvious way now
    shares one allocator automatically:

        db    = LatticeConn(path)
        arts  = LatticeArtifactStore(db, origin="71")
        graph = LatticeGraphStore(db, origin="71")     # reuses arts' allocator

    `override` exists for unusual cases (a caller that built its own allocator).
    Passing an allocator that CONFLICTS with one already registered RAISES —
    silently overwriting the registry is exactly how this stayed invisible."""
    if override is not None and override.origin != origin:
        raise ValueError(
            "allocator_for: override allocator is for origin %r, not %r. An "
            "allocator is bound to one observer's proper time and cannot be "
            "reused across origins." % (override.origin, origin))
    with db._alloc_lock:
        existing = db._allocators.get(origin)
        if override is not None:
            if existing is not None and existing is not override:
                raise ValueError(
                    "allocator_for: a DIFFERENT SeqAllocator is already registered "
                    "for origin %r on this connection. Two allocators over one "
                    "seq_counter row both cache in-transaction and will hand out "
                    "the same _seq, producing duplicate (_origin, _seq) — which "
                    "breaks the uniqueness of the version identity the mesh relies "
                    "on. Pass the existing allocator, or omit `allocator=` and let "
                    "it be reused." % (origin,))
            db._allocators[origin] = override
            return override
        if existing is None:
            existing = SeqAllocator(db, origin)
            db._allocators[origin] = existing
        return existing


def register_allocator(db: LatticeConn, alloc: SeqAllocator) -> SeqAllocator:
    """Back-compat shim for `allocator_for(db, alloc.origin, override=alloc)`.
    Raises on a conflicting registration rather than overwriting."""
    return allocator_for(db, alloc.origin, override=alloc)


# ── incremental counters — the ONLY way this store answers "how many" ────────

def bump(cur: sqlite3.Cursor, name: str, delta: int) -> None:
    """Adjust a counter inside the caller's write transaction.

    Never call `count(*)` to "fix up" a counter. On the live corpus EXPLAIN shows
    `count(*)` loading 6M records to produce one integer — it is a QUERY defect,
    not a heap shortage, and on node 71 it OOMs the acceptor thread and zombies
    the node. If a counter is wrong, the write path that failed to maintain it is
    the bug."""
    if not delta:
        return
    cur.execute(
        "INSERT INTO counter(name, n) VALUES(?,?) "
        "ON CONFLICT(name) DO UPDATE SET n = n + excluded.n", (name, int(delta)))


def vacate(cur: sqlite3.Cursor, origin: Any, *, rows_delta: int = -1) -> None:
    """Record that one seq belonging to `origin` has been VACATED.

    Called in the writing transaction whenever an existing row's `_seq` is
    superseded (an update allocates a fresh seq and abandons the old one) or
    removed (a delete). Both are ACCOUNTED removals; a row that disappears any
    other way is not counted here, which is exactly what makes the imbalance
    visible."""
    if not isinstance(origin, str) or not origin:
        return
    bump(cur, _schema.c_rows(origin), rows_delta)
    bump(cur, _schema.c_vacated(origin), 1)


def seq_accounting(db: LatticeConn, origin: str, *, scan: bool = False) -> Dict[str, Any]:
    """**The allocation-accounting invariant. Read this instead of doing the
    arithmetic — the terms are easy to get subtly wrong.**

        live_rows + vacated == last_seq        (per origin, over vertex ∪ edge)

    This is what "the allocator never skips" actually means. Row CONTIGUITY is
    NOT the invariant and asserting it produces false failures: an update
    allocates a fresh `_seq` and vacates the old one, so
    `put(a)->1, put(b)->2, put(a')->3` leaves rows at seqs `{2, 3}` — perfectly
    correct, and not contiguous from 1.

    Returns::

        {"origin", "last_seq", "live_rows", "vacated", "accounted",
         "unaccounted", "balanced", "insert_only", "source"}

    `unaccounted = last_seq - (live_rows + vacated)`. **Non-zero means rows went
    missing outside the write path** — a bad migration, a truncated file — because
    every accounted removal increments `vacated`. `balanced` is the assertion.

    `insert_only` is `vacated == 0`. In that case the store has never taken an
    update or delete, and **strict row contiguity `1..last_seq` IS exactly valid** —
    it is a strictly stronger assertion and worth making in that specific case
    (the post-migration check this was originally written for).

    `scan=False` (default) reads the incremental counter: O(1), safe on every
    health tick. `scan=True` recomputes `live_rows` from the rows themselves,
    which is the mode that can actually SEE a row lost outside the write path —
    the counter cannot, since it was decremented only by accounted removals.
    node-repair should use `scan=True`; anything on a request path must not.

    ⚠ **THE EQUALITY HOLDS FOR THE LOCAL ORIGIN ONLY**, and the result says which
    case you are in rather than making the caller remember:

        local_only=True    `balanced` is the EQUALITY. `unaccounted > 0` means rows
                           are LOST. `unaccounted_means == "lost"`.
        local_only=False   a peer's seqs are OBSERVED, not allocated: `last_seq` is
                           a high-water mark learned from whatever arrived while
                           the rows behind it are still in flight. `balanced` is
                           the INEQUALITY, and `unaccounted` is replication lag —
                           `unaccounted_means == "not_yet_received"`.

    A sweep that asserted the equality across every origin would flag every healthy
    peer on every pass; that is the same false-failure class as asserting row
    contiguity, one level up. Assert `balanced`, and read `unaccounted` through
    `unaccounted_means`."""
    hw = high_water(db, origin)
    vac = counter_of(db, _schema.c_vacated(origin))
    if scan:
        # COUNT over a UNION of two indexed ranges. This is NOT the banned
        # `count(*)`: the ban exists because `count(*)` DEREFERENCES EVERY RECORD
        # (EXPLAIN shows 6M records loaded for one integer). `_seq` is the second
        # column of `ix_v_origin`/`ix_e_origin`, so this is served entirely from
        # the index for ONE origin's range and touches no record at all. UNION
        # (not UNION ALL) dedupes across the two tables, so a seq duplicated
        # between vertex and edge is caught rather than double-counted.
        r = db.read().execute(
            "SELECT COUNT(_seq) FROM ("
            "  SELECT _seq FROM vertex WHERE _origin = ?"
            "  UNION"
            "  SELECT _seq FROM edge   WHERE _origin = ?)", (origin, origin)).fetchone()
        live = int(r[0]) if r else 0
        source = "scan"
    else:
        live = counter_of(db, _schema.c_rows(origin))
        source = "counter"
    accounted = live + vac
    unaccounted = hw - accounted
    local = _is_local(db, origin)
    # ⚠ THE INVARIANT IS AN EQUALITY ONLY FOR THE LOCAL ORIGIN.
    # For a FOREIGN origin we OBSERVE a peer's proper time rather than allocating
    # it: `last_seq` is a high-water mark learned from whatever arrived, while the
    # rows behind it are still in flight. `accounted < last_seq` there is ordinary
    # replication lag, NOT loss — so the correct test is `<=`, and reporting an
    # equality failure would flag every healthy peer on every sweep. That is the
    # same false-failure class as asserting row contiguity, one level up.
    return {"origin": origin, "last_seq": hw, "live_rows": live, "vacated": vac,
            "accounted": accounted, "unaccounted": unaccounted,
            "balanced": (unaccounted == 0) if local else (unaccounted >= 0),
            "unaccounted_means": "lost" if local else "not_yet_received",
            "insert_only": vac == 0, "local_only": local, "source": source}


def _is_local(db: LatticeConn, origin: str) -> bool:
    r = db.read().execute("SELECT is_local FROM seq_counter WHERE origin = ?",
                          (origin,)).fetchone()
    return bool(r[0]) if r else False


def counter_of(db: LatticeConn, name: str) -> int:
    r = db.read().execute("SELECT n FROM counter WHERE name = ?", (name,)).fetchone()
    return int(r[0]) if r else 0


# ── incremental merkle leaves ────────────────────────────────────────────────

def xor_leaf(cur: sqlite3.Cursor, leaf: int, value: int) -> None:
    """XOR `value` into leaf `leaf`. XOR is its own inverse, so a row UPDATE is
    `xor_leaf(old_hash)` then `xor_leaf(new_hash)` and a DELETE is just
    `xor_leaf(old_hash)` — which is exactly why XOR-of-row-hashes was chosen over
    a sorted hash. Digests are stored as 8 RAW BYTES, never as INTEGER: row
    hashes are unsigned 64-bit and SQLite INTEGER is signed, so an integer column
    wraps half the key space negative and breaks the XOR round-trip."""
    if not value:
        return
    r = cur.execute("SELECT digest FROM leaf_digest WHERE leaf = ?", (leaf,)).fetchone()
    old = int.from_bytes(r[0], "big") if r else 0
    new = old ^ (value & 0xFFFFFFFFFFFFFFFF)
    cur.execute(
        "INSERT INTO leaf_digest(leaf, digest) VALUES(?,?) "
        "ON CONFLICT(leaf) DO UPDATE SET digest = excluded.digest",
        (int(leaf), new.to_bytes(8, "big")))
