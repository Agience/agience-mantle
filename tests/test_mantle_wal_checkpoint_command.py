"""`mantle-wal-checkpoint` — the command that reclaims the write-ahead log without a restart.

Companion to `test_the_wal_is_bounded_and_reclaimable.py`, which pins the schema half
(`wal_autocheckpoint=2000` and `schema.wal_checkpoint`). This pins the OPERATOR half: the console
script, its exit codes, and the one behaviour an operator can be misled by.

Every test here holds a second connection open, and that is not incidental. SQLite checkpoints
and DELETES the WAL when the LAST connection to a database closes — so a test that let the command's
own connection be the last one would watch SQLite tidy up, see a zero-byte WAL, and pass no matter
what the command did. Measured while writing these: with no holder, `--mode PASSIVE` also leaves a
0-byte WAL, because the close did the work. A holder keeps the database open so the command's
result is the command's own.

The holder also reproduces the real node: node 71's lattice is read by `mantle` AND `ember`, which
is why its WAL survived every restart and reached 30.4 GiB — neither process's exit was ever the
last close.
"""
from __future__ import annotations

import json
import os
import sqlite3

import pytest

from mantle.db import open_lattice
from mantle.scripts import wal_checkpoint as cmd


def _lattice_with_a_wal(tmp_path, edges: int = 3000):
    """A store with real committed frames in its WAL, plus the handle that wrote them."""
    p = str(tmp_path / "lat.db")
    L = open_lattice(p, origin="test-origin", leaves=16)
    L.graph.add_edges([("v%d" % i, "v%d" % (i + 1), "lineage", {}) for i in range(edges)])
    assert os.path.getsize(p + "-wal") > 0, (
        "no WAL was produced — every assertion below would be vacuous")
    return p, L


@pytest.fixture
def store(tmp_path):
    p, L = _lattice_with_a_wal(tmp_path)
    holder = sqlite3.connect(p)          # see the module docstring: NOT incidental
    yield p, holder
    try:
        holder.rollback()
    except Exception:
        pass
    holder.close()
    L.artifacts.db.close()


def test_truncate_reclaims_the_log(store, capsys):
    p, _holder = store
    before = os.path.getsize(p + "-wal")
    assert cmd.main(["--path", p]) == 0
    after = os.path.getsize(p + "-wal") if os.path.exists(p + "-wal") else 0
    assert after == 0, "TRUNCATE left %d bytes of WAL" % after
    assert before > 0
    out = capsys.readouterr().out
    assert "reclaimed" in out


def test_a_reader_holding_a_snapshot_makes_it_exit_NON_ZERO(store, capsys):
    """The whole reason the command exists in this shape.

    SQLite reports a blocked checkpoint by RETURNING busy=1, never by raising. Measured on node 71
    on 2026-08-25: the first attempt with the node up folded 832 of 948 log pages and truncated
    NOTHING, and a caller that ignored the return value would have logged a successful checkpoint
    that reclaimed nothing. An operator must be able to tell those apart from the exit code alone.
    """
    p, holder = store
    holder.execute("BEGIN")
    holder.execute("SELECT count(*) FROM edge").fetchone()   # now holding a snapshot

    before = os.path.getsize(p + "-wal")
    assert cmd.main(["--path", p, "--timeout", "2"]) == 1, (
        "a checkpoint that reclaimed nothing exited 0 — that is the failure this command exists "
        "to make impossible")
    after = os.path.getsize(p + "-wal")
    assert after == before, "the WAL moved despite a held snapshot"
    assert "busy=1" in capsys.readouterr().err


def test_passive_folds_frames_and_still_does_not_shrink_the_file(store):
    """`PASSIVE` vs `TRUNCATE` is the distinction the whole fix rests on.

    PASSIVE folds committed frames back and then REUSES the WAL from its start — the file stays at
    its high-water mark for ever, which is why `wal_autocheckpoint` alone never recovered a byte on
    node 71. If this ever starts shrinking the file, the argument for TRUNCATE has changed and the
    docstrings that rest on it need re-reading."""
    p, _holder = store
    before = os.path.getsize(p + "-wal")
    assert cmd.main(["--path", p, "--mode", "PASSIVE"]) == 0
    assert os.path.getsize(p + "-wal") == before, (
        "PASSIVE shrank the file — then TRUNCATE is not the only mode that reclaims, and "
        "`db/schema.py::wal_checkpoint`'s reasoning needs revisiting")


def test_a_missing_lattice_is_refused_and_NOT_created(tmp_path, capsys):
    """`sqlite3.connect` CREATES an empty database for a path that is not there. Without this
    guard a typo leaves a stray 0-byte file, reports a clean checkpoint of nothing, and exits 0 —
    the failure looking exactly like the success."""
    missing = str(tmp_path / "nowhere" / "lat.db")
    assert cmd.main(["--path", missing]) == 2
    assert not os.path.exists(missing), "the command created the database it was asked to check"
    assert "refusing to create" in capsys.readouterr().err


def test_json_carries_the_sizes_and_sqlite_s_own_triple(store, capsys):
    p, _holder = store
    before = os.path.getsize(p + "-wal")
    assert cmd.main(["--path", p, "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["busy"] == 0
    assert doc["wal_bytes_before"] == before
    assert doc["wal_bytes_after"] == 0
    assert doc["reclaimed_bytes"] == before
    for k in ("path", "mode", "log_pages", "checkpointed_pages", "db_bytes"):
        assert k in doc, k


def test_an_unknown_mode_is_rejected_rather_than_silently_downgraded():
    """A misspelling must not become a PASSIVE checkpoint that reclaims nothing and exits 0."""
    with pytest.raises(SystemExit) as exc:
        cmd.main(["--mode", "TRUNCTAE"])
    assert exc.value.code != 0


def test_the_console_script_is_declared_and_points_here():
    """An install should yield the command. A module nobody can invoke is not an operator tool."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as f:
        pyproject = f.read()
    assert 'mantle-wal-checkpoint = "mantle.scripts.wal_checkpoint:main"' in pyproject
    assert callable(cmd.main)
