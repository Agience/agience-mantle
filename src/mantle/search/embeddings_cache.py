"""Long-term embeddings cache (SQLite) — keyed by ``(model_id, sha256(text))``.

We run / test / rebuild a lot and the embedder is a paid GPU; this avoids
re-embedding the same texts (seeds, reindex, repeated queries) across runs,
rebuilds, and restarts. Persisted under the data dir so it survives process
restarts and image rebuilds (when the data volume is mounted).

Disable with ``EMBEDDINGS_CACHE=0``. Path via ``EMBEDDINGS_CACHE_PATH``
(default ``<BASE_DIR>/.data/mantle/embeddings_cache.sqlite``).

Vectors are stored as little-endian float32 blobs. Empty/None vectors are never
cached, so a degraded (unconfigured-provider) run does not poison the cache.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import struct
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)


class EmbeddingsCache:
    """SQLite-backed cache of text → embedding vector, namespaced by model id."""

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = str(path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        # One connection PER THREAD, held for the life of the thread. A sqlite3 connection
        # cannot cross threads, and opening one per call means a file open, a page-header read
        # and a `journal_mode=WAL` round trip on every lookup — the cache would spend more on
        # its own bookkeeping than the lookup it exists to save. `journal_mode` is a durable
        # property of the FILE, so it is set once per connection, not once per query.
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings "
                "(k TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
            )

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        """The calling thread's connection, inside a transaction that commits on clean exit.

        A context manager rather than a bare accessor so the call sites keep the `with conn:`
        commit/rollback semantics they had when each one owned a fresh connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=30.0)
            # WAL keeps concurrent readers (search) from blocking the writer (ingest).
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:
                pass
            self._local.conn = conn
        with conn:
            yield conn

    def close(self) -> None:
        """Close this thread's connection, if it has opened one."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            self._local.conn = None
            conn.close()

    @staticmethod
    def _key(model_id: str, text: str) -> str:
        h = hashlib.sha256()
        h.update((model_id or "").encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def get_many(self, model_id: str, texts: List[str]) -> List[Optional[List[float]]]:
        """Return cached vectors aligned 1:1 with ``texts`` (``None`` on miss)."""
        if not texts:
            return []
        keys = [self._key(model_id, t) for t in texts]
        found: dict[str, List[float]] = {}
        uniq = list(dict.fromkeys(keys))
        with self._conn() as conn:
            for i in range(0, len(uniq), 500):  # stay under SQLite's variable limit
                batch = uniq[i:i + 500]
                placeholders = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT k, dim, vec FROM embeddings WHERE k IN ({placeholders})",
                    batch,
                )
                for k, dim, blob in rows:
                    found[k] = list(struct.unpack(f"<{int(dim)}f", blob))
        return [found.get(k) for k in keys]

    def put_many(
        self,
        model_id: str,
        texts: List[str],
        vectors: List[Optional[List[float]]],
    ) -> int:
        """Cache non-empty vectors. Returns the number of rows written."""
        rows = []
        for text, vec in zip(texts, vectors):
            if not vec:
                continue
            v = [float(x) for x in vec]
            rows.append((self._key(model_id, text), len(v), struct.pack(f"<{len(v)}f", *v)))
        if not rows:
            return 0
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (k, dim, vec) VALUES (?, ?, ?)", rows
            )
        return len(rows)

    def count(self) -> int:
        with self._conn() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
