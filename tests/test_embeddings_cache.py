"""Long-term embeddings cache (SQLite) + the cached Embeddings facade."""

import pytest

from mantle.search.embeddings_cache import EmbeddingsCache


def test_cache_roundtrip_and_namespacing(tmp_path):
    c = EmbeddingsCache(str(tmp_path / "c.sqlite"))
    assert c.get_many("m1", ["a", "b"]) == [None, None]

    assert c.put_many("m1", ["a", "b"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) == 2
    got = c.get_many("m1", ["a", "c", "b"])
    assert got[1] is None                                  # 'c' is a miss
    assert got[0] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    assert got[2] == pytest.approx([0.4, 0.5, 0.6], abs=1e-6)

    assert c.get_many("m2", ["a"]) == [None]               # model-namespaced
    assert c.put_many("m1", ["x"], [[]]) == 0              # empties not cached
    assert c.count() == 2


def test_embeddings_facade_caches(tmp_path, monkeypatch):
    import mantle.search.embeddings as E

    monkeypatch.setenv("EMBEDDINGS_CACHE", "1")
    monkeypatch.setenv("EMBEDDINGS_CACHE_PATH", str(tmp_path / "facade.sqlite"))

    calls = {"n": 0, "texts": []}

    def stub(texts):
        calls["n"] += 1
        calls["texts"].extend(texts)
        return [[float(len(t)), 1.0] for t in texts]

    monkeypatch.setattr(E, "_build_provider", lambda: stub)
    E.reset_provider()  # also resets the cache singleton so the tmp path loads
    emb = E.Embeddings()

    r1 = emb(["hello", "world"])
    assert r1[0] == [5.0, 1.0] and calls["n"] == 1

    r2 = emb(["hello", "world"])          # all cached → provider NOT called again
    assert calls["n"] == 1
    assert r2[0] == pytest.approx([5.0, 1.0], abs=1e-6)

    before = calls["n"]
    emb(["hello", "new"])                 # only the miss is embedded
    assert calls["n"] == before + 1
    assert calls["texts"][-1] == "new"

    E.reset_provider()


def test_the_connection_is_reused_across_calls(tmp_path, monkeypatch):
    """A fresh `sqlite3.connect` per call means a file open, a page-header read and a
    `journal_mode=WAL` round trip on every lookup — the cache would spend more on its own
    bookkeeping than the lookup it exists to save. `journal_mode` is a durable property of the
    FILE, so it belongs on the connection, not on the query."""
    import sqlite3

    opens = {"n": 0}
    real_connect = sqlite3.connect

    def counting_connect(*a, **kw):
        opens["n"] += 1
        return real_connect(*a, **kw)

    monkeypatch.setattr(sqlite3, "connect", counting_connect)

    c = EmbeddingsCache(str(tmp_path / "reuse.sqlite"))
    after_construction = opens["n"]
    assert after_construction == 1

    for _ in range(5):
        c.put_many("m1", ["a"], [[1.0, 2.0]])
        assert c.get_many("m1", ["a"])[0] == pytest.approx([1.0, 2.0], abs=1e-6)
        c.count()

    assert opens["n"] == after_construction


def test_each_thread_gets_its_own_connection(tmp_path):
    """A `sqlite3.Connection` cannot be used from two threads, so the reuse has to be
    per-thread — and every thread must still see the same durable rows."""
    import threading

    c = EmbeddingsCache(str(tmp_path / "threads.sqlite"))
    c.put_many("m1", ["shared"], [[7.0]])

    seen: dict[int, object] = {}
    errors: list[BaseException] = []

    def worker(i):
        try:
            c.put_many("m1", ["t%d" % i], [[float(i)]])
            assert c.get_many("m1", ["shared"])[0] == pytest.approx([7.0], abs=1e-6)
            seen[i] = c._local.conn
        except BaseException as exc:      # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len({id(conn) for conn in seen.values()}) == 4
    assert c.count() == 5
