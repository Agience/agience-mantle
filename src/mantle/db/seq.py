"""Proper time — the `(_origin, _seq)` allocator — and the write-lock discipline.

    _origin   the authoring observer — immutable across replication
    _seq      that observer's proper time — monotonic, gap-free, local,
              incremented once per authored event

A wall clock cannot serve as a cursor field: it is not injective (two events can
tie), and a fixture or clock skewed into the future can pin a cursor beyond every
real row, silently starving every reader behind it.

Allocating from the store buys two properties a clock cannot give:

  1. **Injective.** One event, one value. `_seq > :cursor` is a correct cursor
     with no group-completion dance.
  2. **Gap-free.** Allocation happens inside the same transaction as the write it
     versions, so a rolled-back write rolls back its seq. That is what makes the
     accounting identity exact — `live_rows + vacated == last_seq` — which is how
     a row lost outside the write path is detected at all.

Two consequences that catch people out, both load-bearing:

  * **Gap-freeness is a property of the union, not of either table.** Vertex seqs
    may legitimately read `{1,3,7}` while edge seqs read `{2,4,5,6}`. Any
    per-table contiguity assertion produces false failures — check `vertex ∪ edge`.
  * **There must be exactly one `SeqAllocator` per `(LatticeConn, origin)`.** Two
    instances over one durable row both cache in-transaction — which is precisely
    what makes allocation gap-free and cheap — so they can each read
    `last_seq = N` and both hand out `N`. `allocator_for()` is therefore the only
    sanctioned way to obtain one: it reuses by default, and a conflicting second
    instance raises rather than silently overwriting the registry.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from . import reach_journal as _journal
from . import schema as _schema

_BUSY_TIMEOUT_MS = 30_000
_BEGIN_RETRIES = 60
_BEGIN_BACKOFF_S = 0.05


class _RecordingCursor(sqlite3.Cursor):
    """A cursor that, while its connection records, keeps every statement it runs and its
    parameters exactly as given (`LatticeConn.write`'s co-sign path replays them)."""

    def execute(self, sql, parameters=()):
        log = self.connection.recording
        if log is not None:
            log.append(("one", sql, parameters))
        return super().execute(sql, parameters)

    def executemany(self, sql, seq_of_parameters):
        log = self.connection.recording
        if log is not None:
            seq_of_parameters = [tuple(p) if not isinstance(p, dict) else dict(p)
                                 for p in seq_of_parameters]
            log.append(("many", sql, seq_of_parameters))
        return super().executemany(sql, seq_of_parameters)

    def executescript(self, sql_script):
        if self.connection.recording is not None:
            raise sqlite3.ProgrammingError("executescript inside a recorded write transaction")
        return super().executescript(sql_script)


class _RecordingConnection(sqlite3.Connection):
    recording = None                     # a list while a write transaction records

    def cursor(self, factory=_RecordingCursor):
        return super().cursor(factory)

    def execute(self, sql, parameters=()):
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql, seq_of_parameters):
        return self.cursor().executemany(sql, seq_of_parameters)


def _replay(c: sqlite3.Connection, log: list) -> None:
    for kind, sql, params in log:
        if kind == "one":
            c.execute(sql, params)
        else:
            c.executemany(sql, params)


class ReachChangeConflict(_journal.ReachChangeRefused):
    """Something the transaction depended on changed while Origin was asked; the co-signed
    entries were withdrawn and nothing was written. Safe to run the change again."""


def _receipt_seq(c, key: str):
    r = c.execute("SELECT seq FROM authority_receipt WHERE subject = ?", (key,)).fetchone()
    return r[0] if r is not None else None


class _Reserved(BaseException):
    """Ends the first run of a `cosigned` store call once it has recorded its changes."""


#: How many times a `cosigned` store call is run again when what it depends on changed while
#: Origin was asked.
COSIGN_ATTEMPTS = 5


def cosigned(conn_of):
    """For a store method that may change reach, when it is the outermost write: run it once to
    learn its changes and roll that back (holding the write lock only for the run), have Origin
    co-sign them with no lock held, then run it again under the lock and commit only if it makes
    exactly the changes Origin co-signed. Otherwise those are withdrawn and it is tried again, up
    to `COSIGN_ATTEMPTS`; then `ReachChangeConflict`. A call that changes no reach commits on its
    first run, as it always did. `conn_of(args)` gives the `LatticeConn`."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            conn = conn_of(args)
            if conn._depth() > 0 or not _journal.installed() or _journal.is_suspended():
                return fn(*args, **kwargs)
            for attempt in range(COSIGN_ATTEMPTS):
                conn._local.cosign_mode = "reserve"
                try:
                    return fn(*args, **kwargs)            # changed no reach: committed
                except _Reserved:
                    pending = conn._local.cosign_reserved
                    basis = conn._local.cosign_basis
                finally:
                    conn._local.cosign_mode = None
                    conn._local.cosign_reserved = None
                    conn._local.cosign_basis = None
                c = conn.conn()
                if _journal.due(c):                       # retried first, no lock held
                    _journal.retry_due(c)
                    if _journal.required() and _journal.due(c):
                        raise _journal.ReachChangeRefused(
                            "a withdrawal the authority has not taken is outstanding; under "
                            "threshold custody no change to reach is written until it clears")
                done = _journal.cosign(pending, c)        # no lock held
                conn._local.cosign_mode, conn._local.cosign_expect = "expect", done
                conn._local.cosign_basis = basis
                try:
                    return fn(*args, **kwargs)
                except ReachChangeConflict:
                    if attempt == COSIGN_ATTEMPTS - 1:
                        raise
                finally:
                    conn._local.cosign_mode = None
                    conn._local.cosign_expect = None
                    conn._local.cosign_basis = None
        return wrapper
    return deco


def _run_after_commit(fns) -> None:
    import logging
    for fn in fns:
        try:
            fn()
        except Exception:  # noqa: BLE001 — the write landed; what follows it is logged
            logging.getLogger(__name__).warning("an after-commit step failed", exc_info=True)


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
                                isolation_level=None, factory=_RecordingConnection)
            c.row_factory = sqlite3.Row
            _schema.apply_pragmas(c)
            c.execute("PRAGMA busy_timeout=%d" % _BUSY_TIMEOUT_MS)
            self._local.c = c
            self._local.depth = 0
        return c

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    def after_commit(self, fn) -> None:
        """Run `fn` once the write transaction open on this thread commits, or now when none is
        open (review 27, L3). A transaction that rolls back drops it: what `fn` tells another
        party (Origin that a key is active) is told only of a write that landed. `fn` never fails
        the write; an error is logged."""
        if self._depth() > 0 and getattr(self._local, "after_commit", None) is not None:
            self._local.after_commit.append(fn)
            return
        _run_after_commit([fn])

    def read(self) -> sqlite3.Cursor:
        """A cursor for read-only work. No transaction is opened."""
        return self.conn().cursor()

    # ── the write lock ───────────────────────────────────────────────────────
    @contextmanager
    def write(self) -> Iterator[sqlite3.Cursor]:
        """`BEGIN IMMEDIATE` … `COMMIT`. Reentrant: a nested `write()` joins the
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

        # Changes to reach this transaction makes are recorded as it makes them
        # (`reach_journal`). With an authority installed they are co-signed OUTSIDE the write
        # lock, so ordinary writers never queue behind Origin:
        #   1. reserve: run the transaction under the lock, keep every statement it ran, note the
        #      store's write mark (`write_mark`: every origin's last `_seq`, and the vertex and
        #      edge totals, which any write to the store moves, on this connection or another),
        #      and roll it back;
        #   2. ask: submit the recorded changes with no lock held;
        #   3. commit: under the lock, only if no other connection committed since the
        #      reservation, replay the statements and the receipts and commit. Otherwise nothing
        #      is written, the co-signed entries are withdrawn, and `ReachChangeConflict` says so.
        # A refusal at 2 writes nothing; a failure at 3 withdraws what 2 co-signed.
        allocs = self.allocators()
        self._wlock.acquire()
        held = True
        committed = False
        self._local.after_commit = []
        _journal.begin()
        try:
            self._begin_immediate(c)
            self._local.depth = 1
            # Every allocator on this connection is driven together, not just one.
            # Driving a single allocator's lifecycle would leave any other store's
            # allocator un-begun and un-flushed, so its cache would persist across
            # transactions while its counter never reaches disk.
            for a in allocs:
                a.begin()
            c.recording = [] if _journal.installed() else None
            try:
                cur = c.cursor()
                yield cur
                for a in allocs:
                    a.flush(cur)
                pending = _journal.take_pending() if c.recording is not None else []
                mode = getattr(self._local, "cosign_mode", None)
                self._local.cosign_mode = None            # one transaction's worth
                if mode == "expect":
                    # The second run of a `cosigned` store call: written only if it makes exactly
                    # the changes Origin co-signed for the first.
                    expected = self._local.cosign_expect
                    c.recording = None
                    self._local.cosign_withdraw = expected    # withdrawn if not committed
                    basis = getattr(self._local, "cosign_basis", None) or {}
                    if [(e, k) for e, _r, k in expected] != pending or any(
                            _receipt_seq(c, k) != seq_ for k, seq_ in basis.items()):
                        raise ReachChangeConflict(
                            "what this change depends on changed while Origin was asked")
                    _journal.write_receipts(cur, expected)
                    c.execute("COMMIT")
                    committed = True
                    self._local.to_confirm = expected
                    self._local.cosign_withdraw = None
                    return
                if not pending:
                    c.recording = None
                    _journal.flush(cur)
                    c.execute("COMMIT")
                    committed = True
                    return
                if mode == "reserve":
                    # The first run of a `cosigned` store call: hand back what it would change.
                    c.recording = None
                    self._local.cosign_reserved = pending
                    # What the change is made against: the co-signed change each row had then.
                    self._local.cosign_basis = {k: _receipt_seq(c, k) for _e, k in pending}
                    raise _Reserved()
                log, c.recording = c.recording, None
                outstanding = bool(_journal.due(cur))
                c.execute("ROLLBACK")
                version = write_mark(self)       # the store as it was, still under the lock
            except BaseException:
                c.recording = None
                try:
                    c.execute("ROLLBACK")
                except Exception:
                    pass
                for a in allocs:                  # a rolled-back write consumes no proper time
                    a.discard()
                _journal.abort(c)
                withdraw = getattr(self._local, "cosign_withdraw", None)
                if withdraw:
                    self._local.cosign_withdraw = None
                    _journal.compensate([(e, r) for e, r, _k in withdraw], c)
                raise
            for a in allocs:
                a.discard()
            self._local.depth = 0               # no transaction is open while Origin is asked
            self._wlock.release()
            held = False

            if outstanding:                       # retried first, still with no lock held
                _journal.retry_due(c)
                if _journal.required() and _journal.due(c):
                    raise _journal.ReachChangeRefused(
                        "a withdrawal the authority has not taken is outstanding; under "
                        "threshold custody no change to reach is written until it clears")
            done = _journal.cosign(pending, c)    # no lock held

            self._wlock.acquire()
            held = True
            try:
                self._begin_immediate(c)
                self._local.depth = 1
                if write_mark(self) != version:
                    raise ReachChangeConflict(
                        "the store changed while Origin was asked; the change was withdrawn "
                        "and nothing was written; run it again")
                _replay(c, log)
                cur = c.cursor()
                _journal.write_receipts(cur, done)
                c.execute("COMMIT")
                committed = True
                self._local.to_confirm = done
            except BaseException:
                try:
                    c.execute("ROLLBACK")
                except Exception:
                    pass
                _journal.compensate([(e, r) for e, r, _k in done], c)
                raise
        finally:
            c.recording = None
            _journal.end()
            self._local.depth = 0
            for a in allocs:
                a.end()
            if held:
                self._wlock.release()
            # Committed: tell the authority, with no lock held, so the entries can no longer be
            # withdrawn. One it does not hear is recorded and told again (`confirm`).
            confirmed = getattr(self._local, "to_confirm", None)
            if confirmed:
                self._local.to_confirm = None
                _journal.confirm(confirmed, c)
            after, self._local.after_commit = getattr(self._local, "after_commit", None), None
            if committed and after:
                _run_after_commit(after)

    def allocators(self) -> "list[SeqAllocator]":
        with self._alloc_lock:
            return list(self._allocators.values())

    @staticmethod
    def _begin_immediate(c: sqlite3.Connection) -> None:
        """Take the write lock up front. Retried explicitly because a `timeout=`
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
    """Allocates `_seq` for one origin, inside a `LatticeConn.write()` block.

    The in-transaction cache is what makes this gap-free and cheap: the counter
    row is read once per transaction and written back once at commit, while
    individual allocations are pure Python increments. A rollback discards the
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
        # high-water mark structurally incapable of moving backwards, which is the
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
        """The next `_seq` for the local origin. Must be called inside `write()`.

        Monotonic and gap-free: values are handed out 1, 2, 3, … with no holes
        and no repeats, for the lifetime of the store, across threads and
        processes."""
        if getattr(self._local, "cache", None) is None:
            self._local.cache = self._load(cur)
        self._local.cache += 1
        self._local.dirty = True
        return int(self._local.cache)

    def mark(self) -> Any:
        """Snapshot the in-transaction cache so a savepoint rollback can restore
        it. Without this, a per-document savepoint that rolls back a failed write
        leaves the Python counter advanced while the durable row did not change —
        which is a gap, and a gap breaks `local_max() - cursor` as an exact
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
        of one origin's range: it is O(log n) and touches one row. It is not a
        `count(*)` and must never be replaced by one.

        Scans BOTH tables and takes the max, because one counter is shared across
        `vertex` and `edge` (RESOLVED-5) — recovering from `vertex` alone would
        re-issue every seq an edge already holds.
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
    """Every origin's `last_seq`, sorted — the store's whole-of-store write mark.

    `high_water` answers for one origin, which is the wrong question for a reader
    holding a derived cache and asking "has anything under me changed since I last
    looked?". This answers exactly that question, and it answers it from bookkeeping
    the writer already has to maintain in order to write at all:

      · **`seq_counter.last_seq`, per origin** — every INSERT and every UPDATE allocates
        a fresh `_seq`, over `vertex` and `edge`, because contract §4 RESOLVED-5 gives
        one counter per observer spanning both tables. A replicated row `observe_seq`s
        its foreign origin, so a change arriving from a peer moves this too, and `flush`
        uses MAX, so it is monotonically non-decreasing and cannot be walked backwards.
      · **the `vertex` and `edge` row totals** —

    Those two counters are the same incremental counters the store already maintains in
    the write transaction (`_recount`), never a `count(*)`.

    The reading a cache actually wants is the negative one. Unchanged mark means nothing
    in this store was written, so every derivation built from it is still the derivation of
    what is in it — one indexed read over a table with one row per observer, and it is the
    only read a warm cache needs to make. A mark that has moved says only that *something*
    changed — attributing it to a particular artifact is `version_of`'s job, not this one's."""
    # One round trip, not three: this is polled on every public entry into `ember.ontology.
    # wn_store`, and a warm cache can only afford to take the gate every time if the union
    # costs one round trip rather than several separate `execute()`s.
    # `'#' ||` namespaces the counters so a counter name can never collide with an origin.
    rows = db.read().execute(
        "SELECT origin AS k, last_seq AS v FROM seq_counter "
        "UNION ALL "
        "SELECT '#' || name, n FROM counter WHERE name IN (?, ?)",
        (_schema.c_vertex_total(), _schema.c_edge_total())).fetchall()
    return tuple(sorted((str(r[0]), int(r[1])) for r in rows))


def max_observed(db: LatticeConn, origin: str) -> int:
    """Highest `_seq` actually present in `vertex ∪ edge` for `origin`. Two
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
    """Endpoint health for one origin's proper time. A verdict, not a boolean.

    Returns `{"origin", "last_seq", "max_observed", "vacated", "live_rows",
    "local_only", "rows_lost", "verdict", "method"}` where `verdict` is one of:

        "intact"        fully accounted: no rows lost anywhere. (scan only)
        "tail_intact"   the endpoint is consistent. Does not rule out interior
                        loss — see below. (cheap, insert-only stores only)
        "loss_detected" `rows_lost` rows are gone outside the write path.
        "undecidable"   this call cannot answer. Not a pass — report skip.

    The verdict is derived from accounting, never from subtraction: a removal
    that went through the store increments `vacated`; one that did not, does
    not. That difference is the entire signal, which is why this function has
    no cheap universal mode:

      * `vacated == 0` (insert-only — the post-migration case this was written for)
        the endpoint is exact: every allocated seq must still have a row, so
        `last_seq - max_observed` is a true count of lost tail rows. Two index
        seeks. Verdict `tail_intact` deliberately does not claim `intact`, because
        an interior row could still be missing.
      * `scan=True` — defer to `seq_accounting`, which counts rows and is
        vacancy-correct. This detects loss anywhere, tail or interior, and is the
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
    """The only sanctioned way to obtain a `SeqAllocator`. Reuses by default.

    There must be exactly one allocator per `(LatticeConn, origin)`, because
    `(_origin, _seq)` is required to be a globally unique version identity and one
    counter per observer is shared across `vertex` and `edge` (contract §4
    RESOLVED-5). Two allocator instances over one durable `seq_counter` row both
    cache in-transaction, so each can read `last_seq = N` and both hand out `N`,
    producing duplicate `(_origin, _seq)` pairs.

    The safe path is the default: constructing the two stores the obvious way
    shares one allocator automatically:

        db    = LatticeConn(path)
        arts  = LatticeArtifactStore(db, origin="71")
        graph = LatticeGraphStore(db, origin="71")     # reuses arts' allocator

    `override` exists for unusual cases (a caller that built its own allocator).
    Passing an allocator that conflicts with one already registered raises,
    rather than silently overwriting the registry."""
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


# ── incremental counters — the only way this store answers "how many" ────────

def bump(cur: sqlite3.Cursor, name: str, delta: int) -> None:
    """Adjust a counter inside the caller's write transaction.

    Never call `count(*)` to "fix up" a counter. On the live corpus EXPLAIN shows
    `count(*)` loading 6M records to produce one integer — it is a query defect,
    not a heap shortage, and at that scale it can OOM the process serving the
    request. If a counter is wrong, the write path that failed to maintain it is
    the bug."""
    if not delta:
        return
    cur.execute(
        "INSERT INTO counter(name, n) VALUES(?,?) "
        "ON CONFLICT(name) DO UPDATE SET n = n + excluded.n", (name, int(delta)))


def vacate(cur: sqlite3.Cursor, origin: Any, *, rows_delta: int = -1) -> None:
    """Record that one seq belonging to `origin` has been vacated.

    Called in the writing transaction whenever an existing row's `_seq` is
    superseded (an update allocates a fresh seq and abandons the old one) or
    removed (a delete). Both are accounted removals; a row that disappears any
    other way is not counted here, which is exactly what makes the imbalance
    visible."""
    if not isinstance(origin, str) or not origin:
        return
    bump(cur, _schema.c_rows(origin), rows_delta)
    bump(cur, _schema.c_vacated(origin), 1)


def seq_accounting(db: LatticeConn, origin: str, *, scan: bool = False) -> Dict[str, Any]:
    """The allocation-accounting invariant. Read this instead of doing the
    arithmetic — the terms are easy to get subtly wrong.

        live_rows + vacated == last_seq        (per origin, over vertex ∪ edge)

    This is what "the allocator never skips" actually means. Row contiguity is
    not the invariant, and asserting it produces false failures: an update
    allocates a fresh `_seq` and vacates the old one, so
    `put(a)->1, put(b)->2, put(a')->3` leaves rows at seqs `{2, 3}` — perfectly
    correct, and not contiguous from 1.

    Returns::

        {"origin", "last_seq", "live_rows", "vacated", "accounted",
         "unaccounted", "balanced", "insert_only", "source"}

    `unaccounted = last_seq - (live_rows + vacated)`. Non-zero means rows went
    missing outside the write path — a bad migration, a truncated file — because
    every accounted removal increments `vacated`. `balanced` is the assertion.

    `insert_only` is `vacated == 0`. In that case the store has never taken an
    update or delete, and strict row contiguity `1..last_seq` is exactly valid —
    it is a strictly stronger assertion and worth making in that specific case
    (the post-migration check).

    `scan=False` (default) reads the incremental counter: O(1), safe on every
    health tick. `scan=True` recomputes `live_rows` from the rows themselves,
    which is the mode that can actually see a row lost outside the write path —
    the counter cannot, since it was decremented only by accounted removals.
    node-repair should use `scan=True`; anything on a request path must not.

        local_only=True    `balanced` is the equality. `unaccounted > 0` means rows
                           are lost. `unaccounted_means == "lost"`.
        local_only=False   a peer's seqs are observed, not allocated: `last_seq` is
                           a high-water mark learned from whatever arrived while
                           the rows behind it are still in flight. `balanced` is
                           the inequality, and `unaccounted` is replication lag —
                           `unaccounted_means == "not_yet_received"`.

    A sweep that asserted the equality across every origin would flag every healthy
    peer on every pass; that is the same false-failure class as asserting row
    contiguity, one level up. Assert `balanced`, and read `unaccounted` through
    `unaccounted_means`."""
    hw = high_water(db, origin)
    vac = counter_of(db, _schema.c_vacated(origin))
    if scan:
        # COUNT over a UNION of two indexed ranges. This is not the banned
        # `count(*)`: the ban exists because `count(*)` dereferences every record
        # (EXPLAIN shows 6M records loaded for one integer). `_seq` is the second
        # column of `ix_v_origin`/`ix_e_origin`, so this is served entirely from
        # the index for one origin's range and touches no record at all. UNION
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
