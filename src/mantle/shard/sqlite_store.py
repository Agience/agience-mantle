"""A local, infra-free store shard — SQLite (artifacts+graph) + filesystem (content).

Everything is local: SQLite for artifacts/edges, plain files for content-addressed blobs. No
Docker, no JDK, no server.

Why: a peer box with idle CPU but no way to host a DB server can still run a full ingest shard —
its writes stay local (no cross-LAN round-trip per article), which is the write-scaling win. The
two shards hold the corpus between them; `op.mesh.pull` federates/merges when needed.

Content model: content_ref = cas/<sha256(plaintext)>, Fernet-encrypted at rest.

What this module is
--------------------
`open_sqlite_store()` — what `EMBER_STORE_BACKEND=sqlite` selects — returns the lattice store
(`mantle.db`), which has typed methods, `(_origin, _seq)` proper time, unique non-NULL
`edge_key`, incremental counters, and keyset pagination.

`SqliteArtifactStore` / `SqliteGraphStore` below are not part of that path. They stamp `_rev` from
`time.time_ns()` (not injective under high call rates on a coarse clock tick), count with
`count(*)`, and page with `LIMIT ? OFFSET ?` — the three things the lattice store exists to avoid.
Do not build on them. They are kept only because `FsContentStore` lives in this module. They
have no `.c` connection-shim property, so any caller that reaches for raw SQL through one fails
closed rather than getting a silent wrong answer.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time as _time            # module scope: importing `time` locally inside a function on
                                # this path is a NameError hazard (shadowing) — see MESH.md 0a.
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


class SqliteArtifactStore:
    """Artifacts in SQLite: full doc as JSON + hot filter columns + a lemma/calls index table."""

    def __init__(self, path: str):
        self._path = path
        self._local = threading.local()
        self.ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self._path, timeout=30, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.row_factory = sqlite3.Row
            self._local.c = c
        return c

    def ensure_schema(self) -> None:
        c = self._conn()
        c.execute("""CREATE TABLE IF NOT EXISTS artifacts(
            id TEXT PRIMARY KEY, doc TEXT NOT NULL,
            content_type TEXT, state TEXT, collection_id TEXT, created_by TEXT, status TEXT)""")
        # a shard file may have no status column — add it if missing (idempotent)
        cols = {r[1] for r in c.execute("PRAGMA table_info(artifacts)")}
        if "status" not in cols:
            c.execute("ALTER TABLE artifacts ADD COLUMN status TEXT")
        for col in ("content_type", "state", "collection_id", "status"):
            c.execute(f"CREATE INDEX IF NOT EXISTS ix_art_{col} ON artifacts({col})")
        # the queue's hot path: claim scans (content_type, status)
        c.execute("CREATE INDEX IF NOT EXISTS ix_art_ct_status ON artifacts(content_type, status)")
        # keyed multi-valued index: (artifact_id, field, value) — powers lemma/calls lookup
        c.execute("""CREATE TABLE IF NOT EXISTS listkeys(
            aid TEXT, field TEXT, value TEXT)""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_lk ON listkeys(field, value)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_lk_aid ON listkeys(aid, field)")

    @staticmethod
    def _stamp(doc: Dict[str, Any], stamp_rev: bool) -> Dict[str, Any]:
        """Stamps `_rev` with a fresh time_ns on a local write; preserved when the caller passes
        stamp_rev=False (mesh consume, which must keep the origin's rev so replicated rows do not
        echo around the mesh forever)."""
        if not stamp_rev:
            return doc
        d = dict(doc)
        d["_rev"] = _time.time_ns()
        return d

    def _index_lists(self, c, doc: Dict[str, Any]) -> None:
        aid = doc["id"]
        c.execute("DELETE FROM listkeys WHERE aid = ?", (aid,))
        rows = []
        for field in ("lemmas", "calls"):
            for v in (doc.get(field) or []):
                rows.append((aid, field, str(v).lower()))
        if rows:
            c.executemany("INSERT INTO listkeys(aid, field, value) VALUES(?,?,?)", rows)

    def put_artifact(self, doc: Dict[str, Any], *, stamp_rev: bool = True) -> Dict[str, Any]:
        """Accepts `stamp_rev` to match the ArtifactStore contract the mesh writes against:
        `worker.py` and `sync.py` call `put_artifact(..., stamp_rev=False)` unconditionally, so the
        parameter must be accepted regardless of whether this store needs it."""
        if not doc.get("id"):
            raise ValueError("put_artifact: doc has no 'id'")
        doc = self._stamp(doc, stamp_rev)
        c = self._conn()
        c.execute("INSERT OR REPLACE INTO artifacts(id,doc,content_type,state,collection_id,created_by,status)"
                  " VALUES(?,?,?,?,?,?,?)",
                  (doc["id"], json.dumps(doc), doc.get("content_type"), doc.get("state"),
                   doc.get("collection_id"), doc.get("created_by"), doc.get("status")))
        self._index_lists(c, doc)
        return doc

    def put_many(self, docs: Iterable[Dict[str, Any]], *, batch: int = 500,
                 stamp_rev: bool = True) -> int:
        """Bulk upsert. Signature matches `ArcadeArtifactStore.put_many` — the mesh calls it
        positionally-by-keyword and does not know which backend it has, so both `stamp_rev` and
        the full `status` column must be honoured here exactly as `put_artifact` honours them:

        * `stamp_rev` is accepted and applied the same way `_stamp` applies it elsewhere in this
          class. `sync._apply_artifacts` calls `put_many(batch, batch=500, stamp_rev=False)` on
          every consume batch, and that call must not raise; `_rev` is honoured the same way
          `put_artifact` honours it, so a sqlite shard also produces entries for the `_rev`
          change-feed.

        * All 7 columns — including `status` — are written on every upsert. `INSERT OR REPLACE`
          deletes the row and reinserts it, so any omitted column would reset to NULL even though
          the doc JSON still carries the value; `claim`, `reclaim_stale` and `queue_stats` all read
          the `status` column rather than the JSON, so it must stay in sync."""
        c = self._conn()
        n = 0
        c.execute("BEGIN")
        try:
            for doc in docs:
                if not doc.get("id"):
                    raise ValueError("put_many: doc has no 'id'")
                doc = self._stamp(doc, stamp_rev)
                c.execute("INSERT OR REPLACE INTO artifacts(id,doc,content_type,state,collection_id,created_by,status)"
                          " VALUES(?,?,?,?,?,?,?)",
                          (doc["id"], json.dumps(doc), doc.get("content_type"), doc.get("state"),
                           doc.get("collection_id"), doc.get("created_by"), doc.get("status")))
                self._index_lists(c, doc)
                n += 1
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        return n

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        r = self._conn().execute("SELECT doc FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        return json.loads(r["doc"]) if r else None

    def list_artifacts(self, *, state: Optional[str] = None, content_type: Optional[str] = None,
                       collection_id: Optional[str] = None, created_by: Optional[str] = None,
                       limit: Optional[int] = None, skip: int = 0) -> Iterator[Dict[str, Any]]:
        where, params = [], []
        for col, val in (("state", state), ("content_type", content_type),
                         ("collection_id", collection_id), ("created_by", created_by)):
            if val is not None:
                where.append(f"{col} = ?"); params.append(val)
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        q = f"SELECT doc FROM artifacts{clause} LIMIT ? OFFSET ?"
        params += [limit if limit is not None else -1, int(skip)]
        for r in self._conn().execute(q, params):
            yield json.loads(r["doc"])

    def count(self, *, state: Optional[str] = None) -> int:
        if state is None:
            r = self._conn().execute("SELECT count(*) AS n FROM artifacts").fetchone()
        else:
            r = self._conn().execute("SELECT count(*) AS n FROM artifacts WHERE state = ?", (state,)).fetchone()
        return int(r["n"])

    def delete_artifact(self, artifact_id: str) -> None:
        c = self._conn()
        c.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
        c.execute("DELETE FROM listkeys WHERE aid = ?", (artifact_id,))

    def lookup_by_lemma(self, word: str, *, limit: int = 12) -> List[Dict[str, Any]]:
        return self.lookup_by_list_field("lemmas", word.lower(), limit=limit)

    def lookup_by_list_field(self, field: str, value: str, *, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT a.doc FROM artifacts a JOIN listkeys l ON a.id = l.aid "
            "WHERE l.field = ? AND l.value = ? AND a.state <> 'archived' LIMIT ?",
            (field, str(value).lower(), int(limit))).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    # No `.c` property on this class. Its absence is load-bearing, not an omission:
    # `stats._raw_ok` and `pool._tasks` ask "is there a connection known to execute SQL
    # faithfully?", and the honest answer for this store is no. They degrade to "not measured"
    # and to a loud refusal respectively.


class SqliteGraphStore:
    """Labeled edges in SQLite (mirrors the Arcade LINK-with-label model)."""

    def __init__(self, path: str):
        self._path = path
        self._local = threading.local()
        self.ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "c", None)
        if c is None:
            c = sqlite3.connect(self._path, timeout=30, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL")
            c.row_factory = sqlite3.Row
            self._local.c = c
        return c

    def ensure_schema(self) -> None:
        c = self._conn()
        c.execute("CREATE TABLE IF NOT EXISTS edges(src TEXT, dst TEXT, label TEXT, props TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_e_src ON edges(src, label)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_e_dst ON edges(dst, label)")

    def add_edge(self, from_id: str, to_id: str, label: str, props: Optional[Dict[str, Any]] = None) -> None:
        self._conn().execute("INSERT INTO edges(src,dst,label,props) VALUES(?,?,?,?)",
                             (from_id, to_id, label, json.dumps(props or {})))

    def add_edges(self, edges: Iterable[tuple], *, batch: int = 500) -> int:
        rows = [(e[0], e[1], e[2], json.dumps(dict(e[3]) if len(e) > 3 and e[3] else {})) for e in edges]
        c = self._conn()
        c.executemany("INSERT INTO edges(src,dst,label,props) VALUES(?,?,?,?)", rows)
        return len(rows)

    def neighbors(self, node_id: str, label: Optional[str] = None, *, direction: str = "out") -> List[str]:
        if direction == "out":
            q, col = "SELECT dst AS nid FROM edges WHERE src = ?", "src"
        elif direction == "in":
            q, col = "SELECT src AS nid FROM edges WHERE dst = ?", "dst"
        else:
            outn = self.neighbors(node_id, label, direction="out")
            inn = self.neighbors(node_id, label, direction="in")
            return list(dict.fromkeys(outn + inn))
        params = [node_id]
        if label:
            q += " AND label = ?"; params.append(label)
        return list(dict.fromkeys(r["nid"] for r in self._conn().execute(q, params)))

    def descendants(self, root_id: str, label: str, *, direction: str = "out") -> List[str]:
        seen, frontier, out = {root_id}, [root_id], []
        while frontier:
            nxt = []
            for nid in frontier:
                for m in self.neighbors(nid, label, direction=direction):
                    if m not in seen:
                        seen.add(m); nxt.append(m); out.append(m)
            frontier = nxt
        return out

    # `sync.py` interpolates `store.graph.EDGE_TYPE` into the edge change-feed queries
    # (`SELECT ... FROM {store.graph.EDGE_TYPE} WHERE _rev > :r`), so it must be defined here.
    EDGE_TYPE = "edges"

    # No `.c` property on this class either. Its absence is load-bearing: `store.graph.c` on this
    # backend raises AttributeError, and three call sites treat that as "nothing to do" rather
    # than a fault —
    #   * `genesis._consolidated_members`  -> `except: return set()`
    #   * `sync._iter_edges_rev`           -> `except: return []`
    #   * `sync`'s edge-publish idle check -> `except: pass`


class FsContentStore:
    """Content-addressed blobs on the local filesystem. Same cas/<sha256> key layout as the S3
    mirror; the bytes are already Fernet-encrypted by content.put_content, so files are ciphertext
    at rest."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        safe = key.replace("cas/", "")
        return self.root / safe[:2] / safe

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self._p(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def delete(self, key: str) -> None:
        try:
            self._p(key).unlink()
        except FileNotFoundError:
            pass


def _import_content_cache():
    """`(FileContentCache, collection_key, shared_content_key)` from the lattice, however mantle is
    on the path. Returns None if the lattice content-cache module isn't importable (older mantle /
    seed deploy) — including a mantle predating `shared_content_key`, which then falls back to the
    seed FsContentStore rather than writing under a scheme that mantle would not read back."""
    try:
        from mantle.db.content_cache import (FileContentCache, collection_key,
                                                     shared_content_key)
        return FileContentCache, collection_key, shared_content_key
    except ImportError:
        pass
    try:
        from db.content_cache import (FileContentCache, collection_key,
                                              shared_content_key)
        return FileContentCache, collection_key, shared_content_key
    except ImportError:
        return None


def _open_lattice_content(root, keys_dir, db_path):
    """The migrated corpus's content store: a `FileContentCache` over `<root>/cas`, keyed with one
    node-wide at-rest key. None (→ seed FsContentStore fallback) if the cas dir, the content key, or
    the lattice content-cache module is absent.

    At-rest blobs are AES-GCM under `shared_content_key(root_secret)`, where the root secret is
    derived from `content.key` — a derivation change, not a key-distribution change, so existing
    encrypted objects on disk are read successfully and a corpus migrates itself with no re-fetch
    and no downtime."""
    import hashlib
    cas = Path(root) / "cas"
    keyfile = Path(keys_dir) / "content.key"
    imp = _import_content_cache()
    if imp is None or not cas.is_dir() or not keyfile.exists():
        return None
    FileContentCache, collection_key, shared_content_key = imp
    root_secret = hashlib.blake2b(keyfile.read_bytes().strip(), digest_size=32).digest()
    # collection_id -> origin_root, from the collection artifacts (a handful of rows), read-only.
    # Only the legacy per-collection fallback needs these; the shared key does not depend on the
    # collection at all, so an empty map here is not fatal.
    roots = {}
    try:
        import sqlite3
        c = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
        try:
            for cid, doc in c.execute(
                    "SELECT id, doc FROM vertex WHERE ct='application/vnd.agience.collection+json'"):
                d = json.loads(doc)
                if d.get("origin_root"):
                    roots[str(cid)] = str(d["origin_root"])
        finally:
            c.close()
    except Exception:
        roots = {}                                    # no legacy map — new objects still work

    def legacy_key_for(collection: str) -> bytes:
        origin_root = roots.get(collection)
        if not origin_root:
            raise KeyError("no origin_root for collection %r" % collection)
        return collection_key(root_secret, origin_root)

    # ── keys from EARLIER ERAS, so a rotation does not orphan what is already on disk ────────
    # A rotation replaces `content.key` and rewrites nothing, so the store immediately holds two
    # populations that differ only in which key opens them. Measured 2026-08-25 on 71/home after
    # `content.key` was replaced on 08-24 at 11:05: of 197 blobs sampled across 12 random shards,
    # 138 opened only under the previous key and 59 only under the current one, and NONE were
    # corrupt. Every CAS-backed artifact on the node — 313,982 of them — had stopped being
    # readable, and the surfaced error named a missing object rather than a key.
    #
    # Named by CONFIGURATION, never discovered: `MANTLE_CONTENT_KEY_PREVIOUS` is a path list
    # (`os.pathsep`-separated). A key file that merely appears next to the current one must not
    # silently change what this node can decrypt — that is the same class of surprise as the
    # rotation itself. An unreadable or wrong-length entry is skipped with a warning rather than
    # failing the open: a node that cannot read one previous era must still serve the current one.
    import logging as _logging
    log = _logging.getLogger(__name__)
    previous: list = []
    for candidate in (os.environ.get("MANTLE_CONTENT_KEY_PREVIOUS") or "").split(os.pathsep):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            secret = hashlib.blake2b(Path(candidate).read_bytes().strip(), digest_size=32).digest()
            previous.append(shared_content_key(secret))
        except Exception:
            log.warning("previous content key %r could not be read; skipping it", candidate)
    if previous:
        log.info("content cache: %d previous key era(s) configured for reads", len(previous))

    return FileContentCache(str(cas), key=shared_content_key(root_secret),
                            legacy_key_for_collection=legacy_key_for if roots else None,
                            previous_keys=previous)


def _import_content_tier():
    """`(TieredContentStore, content_cipher)` from the lattice, however mantle is on the path.
    None if the tier module isn't importable (older mantle) — same convention as
    `_import_content_cache`, and the same consequence: the node runs local-only."""
    try:
        from mantle.db.content_tier import TieredContentStore, content_cipher
        return TieredContentStore, content_cipher
    except ImportError:
        pass
    try:
        from db.content_tier import TieredContentStore, content_cipher
        return TieredContentStore, content_cipher
    except ImportError:
        return None


def _open_content_tier(content, keys_dir, *, cred_dir=None):
    """The one read path on the lattice backend: `FileContentCache` inside the mantle
    `TieredContentStore`, with the S3/CDN mirror behind it when the box has creds.

    Always built when the store's content is the lattice cache — a box with no S3 credentials
    gets a local-only tier (`remote=None`): same surface, reads stay local, which makes the
    air-gap case a first-class configuration rather than a different code path. None only when
    there is nothing collection-keyed to front (the seed `FsContentStore` bootstrap seam) or the
    tier module itself is absent (older mantle).

    When the remote is configured, a broken cipher is not swallowed: `_open_lattice_content`
    already proved `content.key` exists (it derived the cache's root secret from it), so a
    failure here is a real configuration fault and stops the node at open — the same refusal
    discipline as the keys-dir and db-path checks above."""
    imp = _import_content_tier()
    if imp is None:
        return None
    TieredContentStore, content_cipher = imp
    # `keys_dir` holds the store's CIPHER; `cred_dir` (the service keys dir) holds the mirror's
    # CREDENTIALS. They are different kinds of secret and need not sit together, so each is tried
    # rather than assuming one directory has both — a node whose `content.key` lives beside its
    # store and whose `ovh.*` keys live in `KEYS_DIR` is a normal configuration, not a broken one.
    remote = None
    for _d in (cred_dir, keys_dir):
        if not _d:
            continue
        try:
            from mantle.shard.content_tier import open_ovh_store
            remote = open_ovh_store(Path(_d))
        except Exception:
            remote = None
        if remote is not None:
            break
    if type(content).__name__ != "FileContentCache":
        # No per-collection local cache to front — e.g. a node holding only an imported index (no
        # cas/ dir and no collection origin_roots, exactly the wiki-index-on-lumen case). Without a
        # remote, resolve_text could never reach a body it had not already stored, so an imported
        # index would answer with titles and no article text. A shared-cipher S3 remote closes
        # that: `cache=None` makes every body a verified WAN read (remote.get → shared-cipher
        # decrypt → sha256-against-ref) — correct, just uncached. Only when there is also no
        # remote is there genuinely nothing to add, so this returns None (true local-only).
        if remote is None:
            return None
        ciph = content_cipher(keys_dir)
        return TieredContentStore(None, remote, decrypt=ciph.decrypt, encrypt=ciph.encrypt)
    if remote is None:
        return TieredContentStore(content, None)  # air-gapped: local-only, same one surface
    ciph = content_cipher(keys_dir)              # content.key proven present; failures are loud
    return TieredContentStore(content, remote, decrypt=ciph.decrypt, encrypt=ciph.encrypt)


def _open_lattice():
    """`mantle.db.open_lattice`, however mantle happens to be on the path.

    `ember.cmd` puts `agience-mantle/src/mantle` on PYTHONPATH (so the package is `db.*`), while
    the tests and unit A's harness put `agience-mantle/src` on it (so it is `mantle.db.*`). Both
    are load-bearing and neither is wrong, so both are tried. The lattice package itself uses
    only relative imports, so it works identically under either name.

    A failure here raises with both attempted names; it must never fall back to the seed store,
    which would silently select a store with no working queue."""
    try:
        from mantle.db import open_lattice          # agience-mantle/src on the path
        return open_lattice
    except ImportError:
        pass
    try:
        from db import open_lattice                 # agience-mantle/src/mantle on the path
        return open_lattice
    except ImportError as e:
        raise ImportError(
            "EMBER_STORE_BACKEND=sqlite needs the lattice store, and neither "
            "`mantle.db` nor `db` is importable (%s). Put "
            "agience-mantle/src (or agience-mantle/src/mantle) on PYTHONPATH — ember.cmd "
            "already does. Refusing to fall back to the seed SqliteArtifactStore: it has no "
            "working work queue, and falling back would silently degrade the backend." % e) from e


def open_sqlite_store(root: Optional[str] = None):
    """A LocalStore backed by the lattice store (SQLite) + filesystem content — what
    `EMBER_STORE_BACKEND=sqlite` selects.

    One file, one connection: `open_lattice` binds both stores to a single `LatticeConn` and a
    single `SeqAllocator`, which is not a tidiness point — an artifact and its edges then commit in
    one transaction, and they draw `_seq` from one counter per observer spanning both tables
    (contract §4 RESOLVED-5). Two allocators over one origin would hand out the same `_seq` twice
    and destroy the uniqueness of `(_origin, _seq)`.

    `origin` must be pinned, never generated. A node that changes origin forks its own proper time
    and every peer then sees two unrelated, permanently-unordered event streams — the same failure
    mode as an unpinned `EMBER_NODE_ID`, which is exactly where it is read from. There is no
    default: a store that cannot name its observer refuses to open rather than inventing one.
    root defaults to EMBER_SQLITE_DIR; the db filename defaults to EMBER_SQLITE_DB / lattice.db."""
    from mantle.shard.local_store import LocalStore
    root = root or os.getenv("EMBER_SQLITE_DIR") or str(Path.home() / "genesis-shard")
    Path(root).mkdir(parents=True, exist_ok=True)

    # The keys dir is never auto-created. It holds `content.key`; if the volume is not mounted,
    # creating an empty directory here would let the first write mint a brand-new key that no
    # peer holds, after which this node's published segments are undecryptable fleet-wide while
    # every count, ρ and keyed_coverage metric stays healthy — a silent partition, matching the
    # same refusal `content.py` applies at the moment a key is read or minted. Refusing here, at
    # open, makes the node fail to start instead of failing to be decryptable.
    #
    keys_dir = Path(os.getenv("EMBER_STORE_KEYS_DIR") or (Path(root) / "keys"))
    if not keys_dir.is_dir():
        raise RuntimeError(
            "keys dir %s does not exist — refusing to create it. It holds content.key, and an "
            "auto-created empty keys dir lets the first write mint a key no peer holds: this "
            "node's content then becomes undecryptable fleet-wide while every health metric "
            "stays green (a SILENT PARTITION). Mount the keys volume, or set "
            "EMBER_STORE_KEYS_DIR to the provisioned directory. This mirrors content.py's "
            "existing refusal; it is deliberately not a mkdir." % keys_dir)

    origin = (os.getenv("EMBER_NODE_ID") or "").strip()
    if not origin:
        raise RuntimeError(
            "EMBER_STORE_BACKEND=sqlite requires EMBER_NODE_ID: it is this observer's stable "
            "identity, and `(_origin, _seq)` is the store's only version identity. Generating "
            "one per boot would fork this node's proper time on every restart, leaving peers "
            "with two unrelated and permanently-unordered event streams. Pin it.")

    # `open_lattice` creates-or-opens, so pointing the node at the wrong directory — an
    # unmounted volume, a typo, a stale path after a move — would silently create a brand-new
    # empty store; `LocalStore.ready()` would then report True for it, because a fresh lattice
    # answers a keyed probe perfectly well. To avoid that: the filename is configurable
    # (`EMBER_SQLITE_DB`), and an absent file is a refusal by default rather than a silent create
    # (opt in with EMBER_SQLITE_CREATE=1).
    #
    db_name = (os.getenv("EMBER_SQLITE_DB") or "lattice.db").strip()
    db_path = Path(root) / db_name
    if not db_path.exists() and (os.getenv("EMBER_SQLITE_CREATE") or "").strip().lower() not in (
            "1", "true", "yes"):
        raise RuntimeError(
            "no lattice store at %s — refusing to create one. `open_lattice` creates-or-opens, "
            "so a wrong EMBER_SQLITE_DIR would mint an EMPTY store here and this node would "
            "report ready() == True while serving nothing. Check EMBER_SQLITE_DIR (currently "
            "%r) and EMBER_SQLITE_DB (currently %r); if this is a new shard, set "
            "EMBER_SQLITE_CREATE=1 to bootstrap it deliberately." % (db_path, root, db_name))

    L = _open_lattice()(str(db_path), origin=origin)
    # Content lives in the migration's per-collection-keyed CAS at `<root>/cas` (2-level fan-out),
    # not the single-key `<root>/content`. Use the lattice `FileContentCache` when that cache is
    # present so `resolve_text` returns real article text; fall back to the seed store otherwise.
    content = _open_lattice_content(root, keys_dir, db_path) or FsContentStore(str(Path(root) / "content"))
    # The wide-area tier rides beside `content`, not around it: writers hand `content` ciphertext
    # (content.put_content), which the tier's plaintext-verifying surface would loudly refuse.
    # Readers (content.resolve_text) prefer the tier when present — local cache first, then the
    # S3/CDN mirror with sha256 verify-on-pull. None ⇒ this box serves its local working set only.
    content_tier = _open_content_tier(content, keys_dir)
    # `conn=None` is deliberate and is not a missing value. `LocalStore.conn` is an
    # `ArcadeConnection` — a network handle whose `ready()` answers "is the server up?". The
    # lattice store is a local file opened in-process; there is no server, and there is nothing
    # for a raw-SQL escape hatch to connect to. `LocalStore.ready()` handles it by asking the
    # store itself. Anything reaching for `store.conn` on this backend gets an immediate
    # AttributeError, which is the correct, loud outcome.
    return LocalStore(artifacts=L.artifacts, graph=L.graph, content=content,
                      conn=None, keys_dir=keys_dir, content_tier=content_tier)
