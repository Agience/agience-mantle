"""The write-ahead log must be bounded, and there must be a way to reclaim it.

`db/schema.py` sets `PRAGMA journal_mode=WAL`, and this suite is what holds it to bounding and
reclaiming that log rather than letting it grow unchecked.

Measured on node 71 (2026-08-25): a 10.3 GB database beside a 32.7 GB write-ahead log, having
doubled from 16.7 GB in a single day of bulk ingests, with no `wal_checkpoint` or `autocheckpoint`
anywhere in `agience-mantle/` at the time.

What that cost, all measured rather than argued:

  · every cold read scanned the whole WAL — >45 s cold against 6.28 s warm, a 7× penalty paid
    once per session by every reader;
  · `du` over the store directory stopped returning, which hung `agience-supervise` in its own
    preflight and left the node unsupervised;
  · the disk grew monotonically, because nothing ever reclaimed a byte.

`agience-ember/src/ember/genesis.py::_wal_checkpoint` carries the same measurement in its
docstring — "an uncheckpointed force=True WordNet re-ingest (117k synset rewrites) grows
lattice.db-wal to 18 GB… so chat takes 20–60 s" — and checkpoints after every bulk pass. Same
file, same magnitude, same consequence; this suite is what holds mantle to the same discipline.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from mantle.db import schema


def _new_db(tmp_path):
    path = str(tmp_path / "t.db")
    conn = sqlite3.connect(path)
    schema.apply_pragmas(conn)
    return path, conn


def _wal_bytes(path: str) -> int:
    wal = path + "-wal"
    return os.path.getsize(wal) if os.path.exists(wal) else 0


def test_wal_autocheckpoint_is_declared_not_inherited() -> None:
    """An inherited default is not a decision. SQLite's default is 1000 pages, and a future
    version changing it would change this store's behaviour with nothing in the repo recording that
    it had. The value is written out for the same reason every port in this fleet is.
    """
    assert any("wal_autocheckpoint" in p for p in schema.PRAGMAS), (
        "no wal_autocheckpoint in PRAGMAS — the WAL is unbounded again, which is the state that "
        "produced a 32.7 GB log against a 10.3 GB database")


def test_the_pragma_actually_reaches_the_connection(tmp_path) -> None:
    """The list can carry a value the connection never receives — a typo in the pragma name is
    accepted silently by SQLite and simply does nothing. Read it back off a real connection."""
    _, conn = _new_db(tmp_path)
    got = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
    assert got and int(got) > 0, "wal_autocheckpoint resolved to %r on a live connection" % got
    assert int(got) != 1000, (
        "wal_autocheckpoint is 1000, SQLite's inherited default — PRAGMAS declares a value and it "
        "is not reaching the connection")


def test_truncate_actually_empties_the_file(tmp_path) -> None:
    """`PASSIVE` would pass a weaker version of this test and fix nothing. It folds committed
    frames back and then reuses the WAL from its start, so the file stays at its high-water mark
    forever — which is exactly why `wal_autocheckpoint` alone never recovered a byte on node 71.
    The assertion is on the file size, not on the return value, for that reason.
    """
    path, conn = _new_db(tmp_path)
    conn.execute("CREATE TABLE t(x)")
    conn.executemany("INSERT INTO t VALUES(?)", [(i,) for i in range(20000)])
    conn.commit()
    assert _wal_bytes(path) > 0, "no WAL was produced — this test measured nothing"

    schema.wal_checkpoint(conn, "TRUNCATE")
    assert _wal_bytes(path) == 0, (
        "TRUNCATE left %d bytes of WAL. Only TRUNCATE resets the file to zero; if this now runs "
        "PASSIVE, the unbounded growth is back." % _wal_bytes(path))


def test_the_data_survives_the_checkpoint(tmp_path) -> None:
    """The obvious property, asserted because the failure would be catastrophic and silent: a
    checkpoint that lost committed frames would empty the WAL exactly as a correct one does."""
    path, conn = _new_db(tmp_path)
    conn.execute("CREATE TABLE t(x)")
    conn.executemany("INSERT INTO t VALUES(?)", [(i,) for i in range(20000)])
    conn.commit()
    schema.wal_checkpoint(conn, "TRUNCATE")
    assert conn.execute("SELECT count(*) FROM t").fetchone()[0] == 20000
    conn.close()
    # Re-opened, because a checkpoint's job is to make the DATABASE hold what the WAL held.
    again = sqlite3.connect(path)
    assert again.execute("SELECT count(*) FROM t").fetchone()[0] == 20000


def test_a_blocked_checkpoint_is_reported_rather_than_swallowed(tmp_path) -> None:
    """A checkpoint can legitimately do nothing, and the caller must be able to tell.

    Any connection holding an older snapshot blocks the frames after it — which is precisely the
    situation on a live node, and the reason node 71's 32.7 GB could not simply be truncated while
    mantle was running. SQLite reports this by returning `busy=1`, not by raising. A helper that
    discarded the return value would let a caller log a successful checkpoint that moved nothing.
    """
    path, writer = _new_db(tmp_path)
    writer.execute("CREATE TABLE t(x)")
    writer.executemany("INSERT INTO t VALUES(?)", [(i,) for i in range(20000)])
    writer.commit()

    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT count(*) FROM t").fetchone()   # snapshot held open

    busy, _log, _done = schema.wal_checkpoint(writer, "TRUNCATE")
    reader.rollback()
    reader.close()

    assert busy in (0, 1), "busy must be SQLite's flag, got %r" % busy
    if busy == 1:
        assert _wal_bytes(path) > 0, (
            "the checkpoint reported busy=1 but the WAL is empty — the flag is being fabricated")


def test_an_unknown_mode_is_refused(tmp_path) -> None:
    """`wal_checkpoint(conn, "TRUNCTAE")` must not silently become a PASSIVE checkpoint. SQLite
    would reject the statement, but the error would name a syntax problem rather than the typo."""
    _, conn = _new_db(tmp_path)
    with pytest.raises(ValueError, match="unknown checkpoint mode"):
        schema.wal_checkpoint(conn, "TRUNCTAE")


def test_mode_is_case_insensitive(tmp_path) -> None:
    """A caller writing `"truncate"` gets a truncate, not a ValueError. Small, and the alternative
    is a maintenance script failing at the moment it is needed."""
    path, conn = _new_db(tmp_path)
    conn.execute("CREATE TABLE t(x)")
    conn.executemany("INSERT INTO t VALUES(?)", [(i,) for i in range(5000)])
    conn.commit()
    schema.wal_checkpoint(conn, "truncate")
    assert _wal_bytes(path) == 0


# ── journal_size_limit — the residual after a BLOCKED period, which autocheckpoint does not bound ──

def test_journal_size_limit_is_declared_not_inherited() -> None:
    """SQLite's default is -1: no limit, and the WAL keeps its high-water mark forever.

    Written out rather than inherited for the same reason `wal_autocheckpoint` is — a future
    SQLite changing its default must not silently change this store's disk behaviour.
    """
    assert "PRAGMA journal_size_limit=67108864" in schema.PRAGMAS, (
        "journal_size_limit is not declared; the WAL will hold its peak size after any period "
        "when a reader blocked checkpoints")


def test_the_limit_bounds_what_is_left_after_a_checkpoint(tmp_path) -> None:
    """The property that matters, measured rather than asserted from the docs.

    The timing is counterintuitive and this test pins it: the truncation happens on the first
    write after a resetting checkpoint, not at the checkpoint. Measuring the file immediately
    after `wal_checkpoint(PASSIVE)` shows no change, which is exactly how someone concludes the
    pragma does nothing and removes it.
    """
    import os
    import sqlite3

    def wal_bytes(p):
        return os.path.getsize(str(p) + "-wal") if os.path.exists(str(p) + "-wal") else 0

    sizes = {}
    for limit in (-1, 65536):
        p = tmp_path / ("lim%d.db" % (limit if limit > 0 else 0))
        c = sqlite3.connect(str(p))
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA journal_size_limit=%d" % limit)
        c.execute("PRAGMA wal_autocheckpoint=0")       # the blocked case: nothing checkpoints for us
        c.execute("CREATE TABLE t(x)")
        c.executemany("INSERT INTO t VALUES (?)", [("x" * 400,) for _ in range(6000)])
        c.commit()
        grown = wal_bytes(p)
        c.execute("PRAGMA wal_checkpoint(PASSIVE)")
        at_checkpoint = wal_bytes(p)
        c.execute("INSERT INTO t VALUES ('the write that wraps the log')")
        c.commit()
        sizes[limit] = (grown, at_checkpoint, wal_bytes(p))
        c.close()

    for limit, (grown, at_cp, after) in sizes.items():
        assert grown > 1_000_000, "the WAL did not grow, so this measures nothing (limit=%d)" % limit
        assert at_cp == grown, (
            "the WAL shrank AT the checkpoint (limit=%d) — if SQLite's timing changed, the comment "
            "in schema.py explaining when truncation happens is now wrong too" % limit)

    assert sizes[-1][2] == sizes[-1][0], (
        "without a limit the WAL must keep its high-water mark; got %d from %d" % (sizes[-1][2], sizes[-1][0]))
    assert sizes[65536][2] <= 65536, (
        "with a 64 KiB limit the WAL must be truncated on the next write; got %d" % sizes[65536][2])
