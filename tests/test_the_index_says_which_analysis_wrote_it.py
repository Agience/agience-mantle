"""An SSE index records the analysis that wrote it, so a stale one is a fact rather than a silence.

## Why a version number is load-bearing here and not elsewhere

A blind token is `HMAC(key, "field:term")` over an *analysed* term. The store holds hashes, so it
cannot re-derive a term it has never seen and no in-place migration is possible — a client whose
pipeline has moved queries terms the index was never filed under. The result is empty, the query is
well-formed and the store is healthy, so nothing anywhere reports a problem.

That is the failure this stamp converts into a sentence. It does not repair anything: the remedy is
`search.init_search.reindex_all_artifacts`, which already exists and already backgrounds itself.

## What is deliberately NOT done

The mismatch is reported, not refused. Generation 2 added ASCII folding, which is the identity on
ASCII, so the terms that moved are exactly those containing combining marks — refusing would take a
working index offline over a subset of its content. That is a judgement about blast radius, and it
is stated in `wiring._report_analyzer_generation` so a future generation with a wider blast radius
can make the other call knowingly.

## The distinction this file exists to protect

An unstamped store is only stale **if it holds something**. An empty unstamped store has not been
written under any analysis, and the first index write claims it. Conflating those two would make
every fresh install log a rebuild warning on boot, and a warning that fires when nothing is wrong is
one nobody reads when something is.
"""
from __future__ import annotations

import logging
import os
import tempfile

import pytest

from mantle.search.mantle.sse.posting import (
    InMemoryPostingStore, analyzer_generation_of, stamp_analyzer_generation)
from mantle.search.mantle.sse.sqlite_stores import SqlitePostingStore
from mantle.search.mantle.sse.tokenizer import ANALYZER


@pytest.fixture()
def sqlite_store(tmp_path):
    return SqlitePostingStore(str(tmp_path / "sse.db"))


def test_a_fresh_store_has_no_stamp(sqlite_store):
    """`None` means "cannot say", and a store nobody has written to genuinely cannot."""
    assert sqlite_store.analyzer_generation() is None


def test_a_stamp_survives_reopening_the_file(tmp_path):
    """The point of the stamp is that it outlives the process that wrote it. An in-memory field
    would satisfy every other test in this file and none of the purpose."""
    path = str(tmp_path / "sse.db")
    SqlitePostingStore(path).record_analyzer_generation(ANALYZER)
    assert SqlitePostingStore(path).analyzer_generation() == ANALYZER


def test_the_last_writer_wins(sqlite_store):
    """A store written by two generations is already broken; the last stamp is the honest one,
    because it names the analysis whose terms are now mixed into it."""
    sqlite_store.record_analyzer_generation(1)
    sqlite_store.record_analyzer_generation(2)
    assert sqlite_store.analyzer_generation() == 2


def test_the_in_memory_store_holds_one_too(monkeypatch):
    """The test default implements the same protocol, so a suite exercising the indexer exercises
    the stamp rather than skipping past it."""
    store = InMemoryPostingStore()
    assert store.analyzer_generation() is None
    store.record_analyzer_generation(ANALYZER)
    assert store.analyzer_generation() == ANALYZER


def test_a_store_that_cannot_hold_a_stamp_reads_as_unknown_and_does_not_raise():
    """The methods are optional. A third-party or older `PostingStore` must degrade to "cannot say"
    — a diagnostic that can break a working index is worse than the condition it diagnoses."""
    class Bare:
        pass

    assert analyzer_generation_of(Bare()) is None
    stamp_analyzer_generation(Bare(), 2)          # must not raise

    class Hostile:
        def analyzer_generation(self):
            raise RuntimeError("no")

        def record_analyzer_generation(self, generation):
            raise RuntimeError("no")

    assert analyzer_generation_of(Hostile()) is None
    stamp_analyzer_generation(Hostile(), 2)       # must not raise


def test_a_non_integer_stamp_reads_as_unknown(sqlite_store):
    """Corruption reads as "cannot say" rather than as a generation. A garbage value parsed
    optimistically would compare unequal to `ANALYZER` and produce a rebuild warning that a rebuild
    cannot clear."""
    with sqlite_store.transaction() as cur:
        cur.execute("INSERT INTO index_meta (key, value) VALUES ('analyzer', 'banana') "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value")
    assert sqlite_store.analyzer_generation() is None


# ── what the indexer and the wiring do with it ───────────────────────────────────────────────────

def test_indexing_stamps_the_store():
    """The stamp is taken on the write path, so it records what actually wrote, not what happened
    to be installed when a store was opened."""
    from mantle.search.mantle.sse.indexer import SseIndexer

    class _Oracle:
        def derive_sse_key(self, principal_id, request):
            return bytes(range(32))

    store = InMemoryPostingStore()
    indexer = SseIndexer(_Oracle(), store)
    assert store.analyzer_generation() is None, "constructing an indexer must not claim a store"

    indexer.index_artifact("p1", "c1", "a1", {"title": "hello world"}, request=None)
    assert store.analyzer_generation() == ANALYZER


def test_an_empty_unstamped_store_is_not_reported_as_stale(tmp_path, caplog):
    """A fresh install must boot quietly. A warning that fires when nothing is wrong is one nobody
    reads when something is."""
    from mantle.search.mantle.wiring import _report_analyzer_generation

    store = SqlitePostingStore(str(tmp_path / "sse.db"))
    with caplog.at_level(logging.WARNING):
        _report_analyzer_generation(store, "fresh")
    assert not [r for r in caplog.records if "analyzer generation" in r.message], \
        "an empty store was reported as stale"


def test_a_populated_unstamped_store_is_reported_as_generation_one(tmp_path, caplog):
    """The upgrade case: an index written before stamping existed. It holds content under the old
    analysis, so it is stale by definition and must say so."""
    from mantle.search.mantle.wiring import _report_analyzer_generation

    store = SqlitePostingStore(str(tmp_path / "sse.db"))
    store.add_entry("p1", "deadbeef", "a1", "c1", b"blob")     # populated, never stamped
    with caplog.at_level(logging.WARNING):
        _report_analyzer_generation(store, "upgraded")
    messages = [r.getMessage() for r in caplog.records]
    assert any("generation 1" in m and "reindex_all_artifacts" in m for m in messages), \
        "a populated pre-stamp index did not report a rebuild is needed: %s" % messages


def test_a_current_store_is_reported_as_nothing_at_all(tmp_path, caplog):
    """The common path, and the one that must stay silent."""
    from mantle.search.mantle.wiring import _report_analyzer_generation

    store = SqlitePostingStore(str(tmp_path / "sse.db"))
    store.add_entry("p1", "deadbeef", "a1", "c1", b"blob")
    store.record_analyzer_generation(ANALYZER)
    with caplog.at_level(logging.WARNING):
        _report_analyzer_generation(store, "current")
    assert not caplog.records, "a current index logged something: %s" % [
        r.getMessage() for r in caplog.records]


def test_the_report_names_the_remedy_and_not_just_the_problem(tmp_path, caplog):
    """An operator reading this line has to know what to run. Asserted because a message that says
    only "mismatch" leaves the index stale for as long as it takes someone to read the source."""
    from mantle.search.mantle.wiring import _report_analyzer_generation

    store = SqlitePostingStore(str(tmp_path / "sse.db"))
    store.add_entry("p1", "deadbeef", "a1", "c1", b"blob")
    store.record_analyzer_generation(ANALYZER - 1)
    with caplog.at_level(logging.WARNING):
        _report_analyzer_generation(store, "stale")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "reindex_all_artifacts" in text, "the report does not name the rebuild"
    assert "unreachable" in text, "the report does not say what the mismatch costs"


# ── asking a store its generation, rather than hoping the boot log was read ───────────────────────

def _generation_cli(tmp_path, monkeypatch, *, populated: bool, stamp):
    """Run `manage_sse_index --generation` against a throwaway store; return `(exit_code, output)`.

    In-process rather than a subprocess: the flag's job is to read a store and decide, and a
    subprocess would additionally test this machine's console encoding, which is not the property.
    """
    import mantle.system.manage_sse_index as cli
    from mantle.search.mantle.wiring import _segment_prefixes, local_sse_path

    monkeypatch.setenv("MANTLE_SSE_DIR", str(tmp_path))
    _, prefix = _segment_prefixes("committed")
    store = SqlitePostingStore(local_sse_path(prefix))
    if populated:
        store.add_entry("p1", "deadbeef", "a1", "c1", b"blob")
    if stamp is not None:
        store.record_analyzer_generation(stamp)
    store.close()

    monkeypatch.setattr("sys.argv", ["manage_sse_index", "--generation"])
    printed = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    code = cli.main()
    return code, "\n".join(printed)


def test_the_cli_reports_a_current_index_and_exits_zero(tmp_path, monkeypatch):
    code, out = _generation_cli(tmp_path, monkeypatch, populated=True, stamp=ANALYZER)
    assert code == 0, out
    assert "CURRENT" in out, out


def test_the_cli_reports_a_stale_index_and_exits_non_zero(tmp_path, monkeypatch):
    """Non-zero so this is usable from a health check, not only by a human reading the output."""
    code, out = _generation_cli(tmp_path, monkeypatch, populated=True, stamp=ANALYZER - 1)
    assert code != 0, out
    assert "STALE" in out and "reindex_all_artifacts" in out, out


def test_the_cli_calls_a_populated_unstamped_index_generation_one(tmp_path, monkeypatch):
    code, out = _generation_cli(tmp_path, monkeypatch, populated=True, stamp=None)
    assert code != 0, out
    assert "generation 1" in out and "unstamped" in out, out


def test_the_cli_does_not_call_an_empty_index_stale(tmp_path, monkeypatch):
    """Same distinction the boot path makes: nothing has been written under any analysis yet."""
    code, out = _generation_cli(tmp_path, monkeypatch, populated=False, stamp=None)
    assert code == 0, out
    assert "CURRENT" in out and "nothing yet" in out, out
