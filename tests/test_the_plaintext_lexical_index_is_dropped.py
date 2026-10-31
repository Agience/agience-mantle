"""The plaintext lexical index an earlier Ember wrote into the store's file is dropped by the release.

2026-10-30, measured on production: `fts`, `fts_map` and `fts_vocab` held 149 rows and 754 terms in
plaintext postings, among them grants, member profiles and leads, in the same file as the sealed
rows. Ember now writes keyed tokens (`fts_kt*`), and nothing reads the old tables.
"""
from __future__ import annotations

import sqlite3

import pytest

from mantle.system.drop_plaintext_index import PLAINTEXT_TABLES, drop_plaintext_index

from .test_there_is_no_promotion_state import real_node

WORD = "larkspurwren"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _tables(db) -> set:
    return {r[0] for r in db.artifacts.db.read().execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _write_the_old_index(db):
    """The index as an earlier Ember wrote it: `porter unicode61`, contentless, with its vocab."""
    with db.artifacts.db.write() as cur:
        try:
            cur.execute("CREATE VIRTUAL TABLE fts USING fts5(title, description, tags, content, "
                        "content='', contentless_delete=1, tokenize='porter unicode61')")
        except sqlite3.OperationalError:                  # pragma: no cover
            pytest.skip("this sqlite build has no fts5")
        cur.execute("CREATE TABLE fts_map (fts_rowid INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "vertex_id TEXT NOT NULL UNIQUE)")
        cur.execute("CREATE VIRTUAL TABLE fts_vocab USING fts5vocab(fts, 'row')")
        cur.execute("INSERT INTO fts(rowid, title, description, tags, content) VALUES (1, ?, '', '', '')",
                    (f"Board figures {WORD}",))
        cur.execute("INSERT INTO fts_map(vertex_id) VALUES ('a')")
        cur.execute("INSERT INTO counter(name, n) VALUES ('fts:total', 1) "
                    "ON CONFLICT(name) DO UPDATE SET n = excluded.n")


def test_the_old_index_is_dropped_with_its_counter(db):
    _write_the_old_index(db)
    terms = [r[0] for r in db.artifacts.db.read().execute("SELECT term FROM fts_vocab")]
    assert any(WORD.startswith(t[:6]) for t in terms), "the control: the word is readable in the file"

    assert set(drop_plaintext_index(db, dry_run=True)["found"]) >= {"fts", "fts_map", "fts_vocab"}
    assert "fts" in _tables(db), "a dry run changes nothing"

    counts = drop_plaintext_index(db)
    assert not (_tables(db) & set(PLAINTEXT_TABLES)), sorted(_tables(db) & set(PLAINTEXT_TABLES))
    assert db.artifacts.db.read().execute(
        "SELECT n FROM counter WHERE name = 'fts:total'").fetchone() is None
    assert counts["dropped"]


def test_a_store_without_one_is_left_as_it_is(db):
    before = _tables(db)
    assert drop_plaintext_index(db) == {"found": [], "dropped": []}
    assert _tables(db) == before
