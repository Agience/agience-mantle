""":class:`SqlitePostingStore` — the local MANTLE-SSE index, one row per slot.

WHY THIS REPLACED THE FILE STORE. The file-backed store put every posting list and manifest in its
own file under a two-level sharded tree, and that single decision produced four separate measured
problems:

**Write cost linear in corpus size.** 3.59 MB rewritten per single-artifact write at 400 docs;
1,200 docs did not finish in two minutes in memory. A term's posting list is one blob, so adding
one artifact to a term rewrites that term's whole list — and an artifact touches every term it
carries.

**A file explosion.** One object per (owner × term × field), plus one per manifest. `keys.py`
claimed less leakage than the object count and blob sizes actually revealed.

**An accelerator that had to exist.** A recall probes every query term against every authorized
owner, and on the file store each probe was an `open()` — 4,520 of them for a ten-term query over
194 owners, mostly misses. The owner-index blob existed purely to collapse that into one read per
owner, and it brought its own failure modes: a partial index read as complete made a whole prior
corpus unfindable, and its read-modify-write lost concurrent writers. Here a probe is an indexed
`SELECT` against a primary key, so there is nothing to accelerate and the blob is gone.

**No atomicity anywhere.** `_atomic_write` made one blob's *publication* atomic and nothing else.
A posting update is read-decrypt-modify-encrypt-write, and two writers interleaving on one term
silently dropped one of them. :meth:`SqlitePostingStore.transaction` is what makes that a real
transaction rather than a mutex over an unprotected sequence.

WHAT IS DELIBERATELY NOT DIFFERENT. This store receives ciphertext and returns ciphertext, exactly
as the file and S3 stores do — the blind tokens are already HMACs, the blobs are already AES-GCM
sealed against their slot's AAD, and none of that moves. A store chooses where bytes land, never
what they are. The schema below therefore holds opaque `BLOB`s keyed by opaque token strings: SQLite
is never asked to index, search or interpret a single byte of content.

The index is rebuildable and the lattice is not, so this is its own database file rather than more
tables in the lattice. A corrupted or deleted SSE index costs a reindex; mixing it into the
authoritative store would put rebuildable derived data behind the same backup and recovery story as
the corpus, and would make `reshard`'s leaf arithmetic share a file with data that has no leaves.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Matches the lattice's (`db/seq.py`). A writer that has to wait for another process's transaction
#: should wait, not fail — the alternative is a reindex job that dies on contention it could have
#: outlasted.
_BUSY_TIMEOUT_MS = 30_000

_PRAGMAS = (
    # WAL: readers never block the writer and the writer never blocks readers. A recall is all
    # reads and runs concurrently with indexing, which is the whole point.
    "PRAGMA journal_mode=WAL",
    # NORMAL, not FULL: the index is rebuildable by definition, so trading an fsync per commit for
    # throughput risks work that `manage_sse_index` can redo. The lattice makes the opposite trade
    # for the opposite reason.
    "PRAGMA synchronous=NORMAL",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA foreign_keys=ON",
)

#: ``WITHOUT ROWID`` on every table: the primary key IS the lookup and there is no second index to
#: keep, so the row lives in the PK b-tree instead of a rowid table plus a unique index over it.
#: That is one page walk per probe rather than two, and it is the read the recall path makes
#: thousands of times.
#:
#: `posting_entry` is one row per `(principal, token, artifact, collection)` — the layout this store
#: exists for. Its primary key is ordered so every operation the index path makes is a prefix of it:
#: `get_entries` scans `(principal, token)`, `delete_entries_for_artifact` scans
#: `(principal, token, artifact)`, and `add_entry` addresses the whole key. Nothing needs a secondary
#: index, and nothing needs a scan.
#:
#: `posting` (the whole-slot blob) is read-only. An index written before the entry layout holds one
#: blob per token, and splitting one needs the owner's SSE key, which no store has. The table stays
#: so those blobs keep opening while `narrowing` reads them and `indexer` absorbs them on the next
#: write.
_DDL = (
    """
    CREATE TABLE IF NOT EXISTS posting_entry (
        principal_id  TEXT NOT NULL,
        blind_token   TEXT NOT NULL,
        artifact_id   TEXT NOT NULL,
        collection_id TEXT NOT NULL,
        blob          BLOB NOT NULL,
        PRIMARY KEY (principal_id, blind_token, artifact_id, collection_id)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS posting (
        principal_id TEXT NOT NULL,
        blind_token  TEXT NOT NULL,
        blob         BLOB NOT NULL,
        PRIMARY KEY (principal_id, blind_token)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS manifest (
        principal_id TEXT NOT NULL,
        artifact_id  TEXT NOT NULL,
        blob         BLOB NOT NULL,
        PRIMARY KEY (principal_id, artifact_id)
    ) WITHOUT ROWID
    """,
    # Store-wide facts about the index itself, not about any principal's content. Cleartext by
    # nature: `analyzer` names a code generation, which is a property of the software that wrote
    # the file and reveals nothing about what is in it. Keyed rather than single-column so a
    # second such fact does not need a migration.
    """
    CREATE TABLE IF NOT EXISTS index_meta (
        key   TEXT NOT NULL PRIMARY KEY,
        value TEXT NOT NULL
    ) WITHOUT ROWID
    """,
)

#: The `index_meta` key naming which `tokenizer.ANALYZER` wrote this file.
_ANALYZER_KEY = "analyzer"


class SqlitePostingStore:
    """:class:`~.posting.PostingStore` over one SQLite file per index segment.

    Args:
        path: The database file. Parent directories are created. Required and non-empty for the
            same reason the file store's root was: an empty path would put the encrypted index in
            whatever the process's working directory happened to be, which is not a location
            anybody chose.

    One connection per thread, autocommit, one explicit in-process write lock — the same three
    decisions `db/seq.LatticeConn` makes, for the same reasons, and not a reuse of it: that class
    creates the lattice schema in its constructor and drives proper-time allocators, neither of
    which belongs in a rebuildable derived index.

    Segments (``committed`` / ``draft`` / ``archived``) stay physically separate by being separate
    files, which is what the file store's `prefix` did with directories.
    """

    def __init__(self, path: str) -> None:
        if not path or not str(path).strip():
            raise ValueError(
                "SqlitePostingStore: a database path is required — an empty path would put the "
                "encrypted index in the process's current working directory, which is not a "
                "location anyone chose"
            )
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        # Serializes writers within THIS process so they queue on a cheap mutex rather than on
        # SQLITE_BUSY retries. Cross-process writers contend at the file lock, which
        # `busy_timeout` covers — and unlike the file store, contending is now *correct* rather
        # than merely slow, because the losing writer waits instead of clobbering.
        self._wlock = threading.RLock()
        with self.transaction() as cur:
            for stmt in _DDL:
                cur.execute(stmt)

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            # `isolation_level=None` — autocommit, so transactions are ours to demarcate. The
            # stdlib's implicit BEGIN opens DEFERRED and upgrades to a write lock mid-statement,
            # which is how a statement that already did half its work gets SQLITE_BUSY.
            c = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000.0,
                                isolation_level=None)
            c.row_factory = sqlite3.Row
            for p in _PRAGMAS:
                c.execute(p)
            c.execute("PRAGMA busy_timeout=%d" % _BUSY_TIMEOUT_MS)
            self._local.c = c
            self._local.depth = 0
        return c

    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """``BEGIN IMMEDIATE`` … ``COMMIT``, reentrant.

        This is what makes a posting update atomic, and it is the capability a file store cannot
        offer. Indexing one artifact is a read-decrypt-modify-encrypt-write per token: the store
        cannot do it alone, because only the caller holds the key. So the caller brackets the whole
        sequence in this, and two writers on one term serialize instead of one silently overwriting
        the other's entries. `SseIndexer` reaches it through `getattr` so a store without it is
        slower and never wrong, exactly as with every other optional method here.

        Reentrant: a nested call joins the open transaction rather than opening a second one, so a
        store method may call another store method without deadlocking itself or — far worse —
        committing half of a caller's atomic unit.
        """
        c = self._conn()
        if self._depth() > 0:
            self._local.depth += 1
            try:
                yield c.cursor()
            finally:
                self._local.depth -= 1
            return

        with self._wlock:
            c.execute("BEGIN IMMEDIATE")
            self._local.depth = 1
            try:
                cur = c.cursor()
                yield cur
                c.execute("COMMIT")
            except BaseException:
                try:
                    c.execute("ROLLBACK")
                except Exception:                  # noqa: BLE001 - rollback of a dead txn
                    logger.debug("rollback failed on %s", self.path, exc_info=True)
                raise
            finally:
                self._local.depth = 0

    def close(self) -> None:
        """Close this thread's connection. Other threads' connections are theirs to close.

        Provided for tests and tools that open a store, finish with it and want the file released —
        on Windows an open handle prevents the deletion of a temporary directory, which is a real
        failure mode in a test suite and not a hypothetical.
        """
        c = getattr(self._local, "c", None)
        if c is not None:
            c.close()
            self._local.c = None

    # ------------------------------------------------------------------
    # PostingStore Protocol — entries (the write path)
    # ------------------------------------------------------------------

    def add_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                  collection_id: str, blob: bytes) -> None:
        """One upsert, which is what the layout exists for.

        A whole-slot shape makes this `get_posting` → decrypt every entry → scan → re-encrypt every
        entry → `put_posting`: O(artifacts already carrying this term) for a write about one
        artifact. Here it is a single row addressed by the full primary key, independent of how many
        artifacts
        the term already reaches — which is what makes indexing a body affordable at all.
        """
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("SqlitePostingStore.add_entry expects bytes")
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO posting_entry"
                " (principal_id, blind_token, artifact_id, collection_id, blob)"
                " VALUES(?,?,?,?,?) "
                "ON CONFLICT(principal_id, blind_token, artifact_id, collection_id)"
                " DO UPDATE SET blob = excluded.blob",
                (principal_id, blind_token, artifact_id, collection_id or "",
                 sqlite3.Binary(bytes(blob))))

    def get_entries(self, principal_id: str, blind_token: str) -> List[Tuple[str, str, bytes]]:
        """Every entry in one slot with its identity — a range scan over the key's first two
        columns.

        The identity travels with the bytes so the reader can authenticate each entry against its
        own AAD; see the Protocol for why returning it is what makes that binding real.
        """
        rows = self._conn().execute(
            "SELECT artifact_id, collection_id, blob FROM posting_entry "
            "WHERE principal_id = ? AND blind_token = ?",
            (principal_id, blind_token)).fetchall()
        return [(r["artifact_id"], r["collection_id"], bytes(r["blob"])) for r in rows]

    def delete_entries_for_artifact(self, principal_id: str, blind_token: str,
                                    artifact_id: str) -> int:
        """One `DELETE` over the key's first three columns. The deletion path knows the artifact and
        not which collections it appears in, and this is that question exactly."""
        with self.transaction() as cur:
            cur.execute(
                "DELETE FROM posting_entry WHERE principal_id = ? AND blind_token = ? "
                "AND artifact_id = ?", (principal_id, blind_token, artifact_id))
            return int(cur.rowcount or 0)

    def delete_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                     collection_id: str) -> bool:
        """Exactly one entry — partial revocation removes an artifact from ONE collection while it
        remains in others, so this must not reach its siblings."""
        with self.transaction() as cur:
            cur.execute(
                "DELETE FROM posting_entry WHERE principal_id = ? AND blind_token = ? "
                "AND artifact_id = ? AND collection_id = ?",
                (principal_id, blind_token, artifact_id, collection_id or ""))
            return int(cur.rowcount or 0) > 0

    # ------------------------------------------------------------------
    # PostingStore Protocol — the legacy whole-slot blob, read-only
    # ------------------------------------------------------------------

    def get_posting(self, principal_id: str, blind_token: str) -> Optional[bytes]:
        row = self._conn().execute(
            "SELECT blob FROM posting WHERE principal_id = ? AND blind_token = ?",
            (principal_id, blind_token)).fetchone()
        return bytes(row["blob"]) if row is not None else None

    def put_posting(self, principal_id: str, blind_token: str, blob: bytes) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("SqlitePostingStore.put_posting expects bytes")
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO posting(principal_id, blind_token, blob) VALUES(?,?,?) "
                "ON CONFLICT(principal_id, blind_token) DO UPDATE SET blob = excluded.blob",
                (principal_id, blind_token, sqlite3.Binary(bytes(blob))))

    def delete_posting(self, principal_id: str, blind_token: str) -> None:
        with self.transaction() as cur:
            cur.execute("DELETE FROM posting WHERE principal_id = ? AND blind_token = ?",
                        (principal_id, blind_token))

    def list_tokens_for_owner(self, principal_id: str) -> List[str]:
        """Both layouts. A lister that saw only `posting_entry` would make a re-key pass silently
        skip every slot still held as a legacy blob — which is all of them, on a store written
        before the entry layout."""
        rows = self._conn().execute(
            "SELECT blind_token FROM posting_entry WHERE principal_id = ? "
            "UNION SELECT blind_token FROM posting WHERE principal_id = ?",
            (principal_id, principal_id)).fetchall()
        return [r["blind_token"] for r in rows]

    def list_owners(self) -> List[str]:
        """Every principal with anything stored here — entries, a legacy blob, or manifests.

        All three tables, because an owner whose postings were all evicted can still hold manifests,
        and a rebuild that could not see it would leave those manifests referencing nothing forever.
        """
        rows = self._conn().execute(
            "SELECT principal_id FROM posting_entry "
            "UNION SELECT principal_id FROM posting "
            "UNION SELECT principal_id FROM manifest ORDER BY principal_id").fetchall()
        return [r["principal_id"] for r in rows]

    # ------------------------------------------------------------------
    # PostingStore Protocol — manifests
    # ------------------------------------------------------------------

    def get_manifest(self, principal_id: str, artifact_id: str) -> Optional[bytes]:
        row = self._conn().execute(
            "SELECT blob FROM manifest WHERE principal_id = ? AND artifact_id = ?",
            (principal_id, artifact_id)).fetchone()
        return bytes(row["blob"]) if row is not None else None

    def put_manifest(self, principal_id: str, artifact_id: str, blob: bytes) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("SqlitePostingStore.put_manifest expects bytes")
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO manifest(principal_id, artifact_id, blob) VALUES(?,?,?) "
                "ON CONFLICT(principal_id, artifact_id) DO UPDATE SET blob = excluded.blob",
                (principal_id, artifact_id, sqlite3.Binary(bytes(blob))))

    def delete_manifest(self, principal_id: str, artifact_id: str) -> None:
        with self.transaction() as cur:
            cur.execute("DELETE FROM manifest WHERE principal_id = ? AND artifact_id = ?",
                        (principal_id, artifact_id))

    # ------------------------------------------------------------------
    # Which analysis wrote this index
    # ------------------------------------------------------------------

    def analyzer_generation(self) -> Optional[int]:
        """The stamp, or ``None`` for a file written before stamping existed.

        A non-integer value reads as ``None`` rather than raising: this is a diagnostic, and one
        that can fail an otherwise healthy index is worse than the condition it reports."""
        try:
            row = self._conn().execute(
                "SELECT value FROM index_meta WHERE key = ?", (_ANALYZER_KEY,)).fetchone()
        except sqlite3.Error:
            return None
        if not row:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    def record_analyzer_generation(self, generation: int) -> None:
        """Stamp the writing generation. Idempotent; last writer wins."""
        with self.transaction() as cur:
            cur.execute(
                "INSERT INTO index_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_ANALYZER_KEY, str(int(generation))))


__all__ = ["SqlitePostingStore"]
