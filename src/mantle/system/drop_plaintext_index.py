"""Drop the plaintext lexical index an earlier Ember wrote into this store.

WHY. Ember's corpus index (`ember.corpus.fts`) lives in the store's file, beside the rows it
indexes. It was written in plaintext (`fts`, `fts_map`, `fts_vocab` and FTS5's shadow tables): every
indexed word sat in the file as a posting, outside every key. Ember now writes keyed tokens to
tables of its own (`fts_kt*`). The plaintext tables are read by nothing, so this pass drops them,
with the `fts:total` counter that counted their rows. 2026-10-30, measured on production: 149 rows
and 754 terms, among them grants, member profiles and leads.

Idempotent, and no reverse: an older Ember that finds no index reports its term frequencies as
unmeasurable, which it already does on a node that never built one.

    python -m mantle.system.drop_plaintext_index --dry-run
    python -m mantle.system.drop_plaintext_index
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("drop_plaintext_index")

#: The plaintext index's tables, the vocabulary view first: `fts5vocab` reads `fts`, and FTS5 drops
#: its shadow tables with `fts`. The same names as `ember.corpus.fts.PLAINTEXT_TABLES`; mantle does
#: not import ember.
PLAINTEXT_TABLES = ("fts_vocab", "fts", "fts_map", "fts_data", "fts_idx", "fts_content",
                    "fts_docsize", "fts_config")
_COUNTER = "fts:total"


def drop_plaintext_index(db, *, dry_run: bool = False) -> dict:
    """Drop the plaintext index's tables from `db`'s file. Returns the tables found and dropped."""
    lattice = db.artifacts.db
    have = {r[0] for r in lattice.read().execute("SELECT name FROM sqlite_master WHERE type='table'")}
    found = [name for name in PLAINTEXT_TABLES if name in have]
    counts = {"found": found, "dropped": []}
    if dry_run or not found:
        return counts
    with lattice.write() as cur:
        for name in found:
            cur.execute(f"DROP TABLE IF EXISTS {name}")
            counts["dropped"].append(name)
        cur.execute("DELETE FROM counter WHERE name = ?", (_COUNTER,))
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    counts = drop_plaintext_index(db, dry_run=args.dry_run)
    logger.info("%s: %s", "would drop" if args.dry_run else "dropped", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
