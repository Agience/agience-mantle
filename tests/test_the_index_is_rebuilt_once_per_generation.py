"""The search index is rebuilt once for each change to what it holds (`mantle.system.reindex`).

2026-10-29, a tester: 0.1.6 finds a hyphenated name by its parts, but only for artifacts saved
after it; one stored before was found only by the whole word. The release runs this migration on
every promote, so it must rebuild when the store is behind and do nothing once it is not, and the
rebuild must not be skipped by the marker that keeps a boot from reindexing twice.
"""
from __future__ import annotations

import pytest

from mantle.system import reindex


class Settings:
    def __init__(self, generation=None):
        self.values = {} if generation is None else {reindex.SETTING: str(generation)}

    def get_int(self, key, default=0):
        return int(self.values.get(key, default))

    def set_setting(self, db, key, value, category):
        self.values[key] = value


def test_a_store_behind_is_rebuilt_and_records_the_generation():
    s, calls = Settings(), []
    out = reindex.reindex_once(None, s, rebuild=lambda: calls.append(1) or {"indexed": 3, "failed": 0})
    assert calls == [1] and out["rebuilt"]
    assert s.values[reindex.SETTING] == str(reindex.GENERATION)


def test_a_store_at_the_generation_is_left_alone():
    calls = []
    out = reindex.reindex_once(None, Settings(reindex.GENERATION), rebuild=lambda: calls.append(1) or {})
    assert calls == [] and not out["rebuilt"]


def test_a_pass_with_failures_records_nothing_so_the_next_release_tries_again():
    s = Settings()
    out = reindex.reindex_once(None, s, rebuild=lambda: {"indexed": 2, "failed": 1})
    assert out["failed"] == 1 and not out["rebuilt"]
    assert reindex.SETTING not in s.values


def test_check_changes_nothing():
    s, calls = Settings(), []
    out = reindex.reindex_once(None, s, rebuild=lambda: calls.append(1) or {}, check=True)
    assert calls == [] and out["would_rebuild_to"] == reindex.GENERATION and not s.values


class _Reached(Exception):
    pass


def test_a_forced_rebuild_gets_past_the_completion_marker(monkeypatch):
    """Without `force`, a store that once completed a full pass is never reindexed again, and the
    migration would rebuild nothing."""
    from mantle.db import backend
    from mantle.search import init_search
    from mantle.search.mantle import wiring
    from mantle.services import dependencies
    from mantle.services import platform_settings_service as pss

    def one_db():
        yield object()
    monkeypatch.setattr(wiring, "sse_index_storage_available", lambda: True)
    monkeypatch.setattr(dependencies, "get_store_db", one_db)
    monkeypatch.setattr(pss.settings, "get",
                        lambda key, default=None: "2026-10-01T00:00:00+00:00"
                        if key == "search.reindex.completed_at" else default)

    def reached(*a, **k):
        raise _Reached
    monkeypatch.setattr(backend, "iter_documents", reached)

    assert init_search._reindex_all_artifacts()["total"] == 0, "unforced: skipped at the marker"
    with pytest.raises(_Reached):
        init_search._reindex_all_artifacts(force=True)
