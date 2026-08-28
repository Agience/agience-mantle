#!/usr/bin/env python3
"""Collapse the lattice's write-ahead log. The `mantle-wal-checkpoint` console script.

    mantle-wal-checkpoint                     # TRUNCATE the configured lattice
    mantle-wal-checkpoint --path /srv/lattice.db
    mantle-wal-checkpoint --mode PASSIVE      # fold frames, do NOT reset the file
    mantle-wal-checkpoint --json              # for a cron or a health script

Why this is a command and not an automatic behaviour. On node 71 the WAL reached **30.4 GiB**
against a 9.58 GiB database, and every cold read paid for it — one query measured **>45 s cold
against 6.28 s warm**. `db/schema.py` now sets `wal_autocheckpoint=2000`, which BOUNDS growth, and
`main.py` truncates on clean shutdown, which RECLAIMS it. Neither covers a node that has been up
for weeks and cannot be restarted, which is exactly the state that produced the 30 GiB. That case
needs a human choosing a moment, so it gets a command rather than a timer.

It needs the node quiet, and it will tell you if it was not. A checkpoint cannot pass an active
reader's snapshot. SQLite reports that by RETURNING `busy=1`, never by raising — so a caller that
ignored the return value would log a successful checkpoint that moved nothing. Measured on
2026-08-25: the first attempt with the node up returned `busy=1`, folded 832 of 948 log pages and
truncated nothing; stopping `mantle` and `ember` (the lattice's readers) and re-running returned
`busy=0` and took 4m40s. **This command exits non-zero when `busy=1`**, so an operator or a script
cannot mistake "did nothing" for "done".

`TRUNCATE` (the default) is the only mode that shrinks the FILE. `PASSIVE` folds committed frames
back and then reuses the WAL from its start, leaving it at its high-water mark forever — which is
why `wal_autocheckpoint` alone never recovered a byte here.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

from mantle.config import DEFAULT_LATTICE_PATH
from mantle.db.schema import wal_checkpoint

#: Read the same way `db/backend.py` reads it, so this command and the service cannot disagree
#: about which file is "the lattice". A second spelling here would be a second answer.
_ENV = "MANTLE_LATTICE_PATH"


def _sidecars(db_path: str) -> tuple[int, int]:
    """`(wal_bytes, db_bytes)` — `stat`, never a directory walk.

    `deploy_common.sh::check_store_path` used `du -sk` on the store directory and, at 30 GiB of
    WAL, stopped returning: `agience-supervise` hung in its own preflight and NOTHING was
    supervised. Sizing two named files is O(1) and cannot repeat that."""
    def _size(p: str) -> int:
        try:
            return os.stat(p).st_size
        except OSError:
            return 0
    return _size(db_path + "-wal"), _size(db_path)


def _human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return str(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="mantle-wal-checkpoint",
        description="Fold the lattice's write-ahead log back into the database and reset it.")
    ap.add_argument("--path", default=None,
                    help="lattice file (default: $%s, else %s)" % (_ENV, DEFAULT_LATTICE_PATH))
    ap.add_argument("--mode", default="TRUNCATE",
                    choices=("PASSIVE", "FULL", "RESTART", "TRUNCATE"),
                    help="TRUNCATE (default) is the only mode that shrinks the file")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds to wait for writers before giving up (default: 120)")
    ap.add_argument("--json", action="store_true", help="machine-readable result on stdout")
    args = ap.parse_args(argv)

    path = args.path or os.getenv(_ENV) or str(DEFAULT_LATTICE_PATH)

    # Refuse a missing file rather than creating one. `sqlite3.connect` CREATES an empty database
    # for a path that is not there, so a typo would leave a stray 0-byte file, report a clean
    # checkpoint of nothing, and exit 0 — the failure looking exactly like the success.
    if not os.path.exists(path):
        print("no lattice at %s — refusing to create one. Pass --path or set %s."
              % (path, _ENV), file=sys.stderr)
        return 2

    wal_before, db_bytes = _sidecars(path)
    conn = sqlite3.connect(path, timeout=args.timeout)
    try:
        conn.execute("PRAGMA busy_timeout=%d" % int(args.timeout * 1000))
        busy, log_pages, done = wal_checkpoint(conn, args.mode)
    finally:
        conn.close()
    wal_after, _ = _sidecars(path)

    result = {
        "path": path, "mode": args.mode, "busy": busy,
        "log_pages": log_pages, "checkpointed_pages": done,
        "wal_bytes_before": wal_before, "wal_bytes_after": wal_after,
        "db_bytes": db_bytes, "reclaimed_bytes": wal_before - wal_after,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("lattice   %s" % path)
        print("mode      %s" % args.mode)
        print("wal       %s -> %s  (reclaimed %s)"
              % (_human(wal_before), _human(wal_after), _human(wal_before - wal_after)))
        print("db        %s" % _human(db_bytes))
        print("sqlite    busy=%d log_pages=%d checkpointed=%d" % (busy, log_pages, done))

    if busy:
        # The whole reason `wal_checkpoint` returns the triple. A reader held a snapshot the
        # checkpoint could not pass, so some frames stayed; on TRUNCATE the file was not reset.
        print("busy=1 — a reader held a snapshot, so the log was NOT fully reclaimed. Stop the "
              "lattice's readers (mantle, ember) and run again.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":                                   # pragma: no cover - console entry
    sys.exit(main())
