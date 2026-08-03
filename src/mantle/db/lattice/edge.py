"""`LatticeGraphStore` — the edge half of the lattice store.

An edge IS AN OBSERVATION: it records that some observer looked and found a
relation. That is why edges carry `(_origin, _seq)` exactly as vertices do — they
replicate, so they need a replication-stable version identity — and it is why
`add_edges` must be idempotent rather than merely convenient.

**IDEMPOTENCY IS THE LOAD-BEARING PROPERTY HERE.** Mesh segments ARE replayed:
consume is retried on any held cursor, so the same edge arrives again and again,
by design. The primary key `edge_key = blake2b(src || \\0 || dst || \\0 || label, 16)`
makes a re-add an UPDATE IN PLACE. The seed store issued a plain
`INSERT INTO edges(...)` with no key at all and accumulated duplicate rows without
bound on every replay — the two shipped backends disagreed on exactly this point,
which is the kind of divergence a declared seam exists to prevent.

⚠ **`is_origin` and `_origin` are DIFFERENT THINGS.** `is_origin` is the
GRANT-PROPAGATION bit: this is the artifact's creation edge, so authorization
flows through it. `_origin` is the AUTHORING OBSERVER. They are unrelated. The old
schema overloaded the word; the split is deliberate and must not be collapsed.

⚠ Do not call the causal-ordering use of edges a "light cone". The corpus already
uses that term for grant propagation (`lightcone.py` — authorization, spatial).
Causal ordering of observations is a second, genuinely relativistic use of the
same word. Call this one CAUSAL ORDER or PROPER TIME.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import constants as K
from . import schema as _schema
from . import seq as _seq
from .seq import LatticeConn, SeqAllocator

try:
    from ..store import GraphStore as _GraphStoreABC
except Exception:
    _GraphStoreABC = object             # type: ignore[assignment,misc]

# props keys that are promoted to real columns rather than left in the JSON blob
_PROMOTED = ("force", "propagate", "is_origin", "order_key")


class LatticeGraphStore(_GraphStoreABC):  # type: ignore[misc,valid-type]
    """Typed, idempotent, replicating edges.

    Pass the SAME `LatticeConn` the artifact store uses when vertices and edges
    live in one file — then an artifact and its edges commit ATOMICALLY, and
    `LatticeConn.write()`'s reentrancy makes that composition safe. Two separate
    connections means two separate transactions and a window in which an artifact
    exists with no edges."""

    EDGE_TYPE = "edge"      # `sync.py` interpolates this into edge-feed queries

    def __init__(self, path_or_conn: Any, *, origin: str,
                 allocator: Optional[SeqAllocator] = None,
                 leaves: int = K.DEFAULT_LEAVES):
        """`path_or_conn` is a filesystem path OR an existing `LatticeConn`.

        ⚠ Passing the artifact store's `LatticeConn` is the NORMAL case, and the
        allocator is now reused automatically — `allocator=` used to be the only
        way to avoid minting a second one, which made the SAFE path the non-default
        and the naive construction silently broken. It is now an override for
        unusual cases; a conflicting one raises."""
        if not origin:
            raise ValueError("LatticeGraphStore requires an explicit `origin`")
        self.origin = origin
        if isinstance(path_or_conn, LatticeConn):
            self.db = path_or_conn
        else:
            self.db = LatticeConn(str(path_or_conn))
        self.seq = _seq.allocator_for(self.db, origin, override=allocator)
        # Edges XOR into the SAME leaf_digest tree vertices use, so the leaf modulus MUST match the
        # artifact store's — `open_lattice` resolves ONE value (the DERIVED `natural_leaves(corpus)`,
        # or a stored/legacy value) and hands it to both. A mismatch would put an edge and a vertex
        # with the "same" leaf index into different buckets and no two nodes could ever agree on a
        # root. `reshard()` keeps the two in lockstep when the corpus grows.
        self.leaves = int(leaves)

    def ensure_schema(self) -> None:
        with self.db.write() as cur:
            _schema.ensure_schema(cur)

    # ── writes ───────────────────────────────────────────────────────────────
    def add_edge(self, from_id: str, to_id: str, label: str,
                 props: Optional[Dict[str, Any]] = None, *,
                 stamp_rev: bool = True) -> None:
        """Singular add. Delegates to the bulk path so there is ONE write path and
        the two cannot drift apart in idempotency semantics."""
        self.add_edges([(from_id, to_id, label, props or {})], stamp_rev=stamp_rev)

    def add_edges(self, edges: Iterable[Any], *, batch: int = 500,
                  stamp_rev: bool = True) -> int:
        """Bulk upsert. **RETURNS THE NUMBER HANDLED, NOT WRITTEN** — same contract
        as `put_many`: written counts, correctly LWW-rejected counts, ERRORED MUST
        NOT COUNT. Callers use the shortfall as a data-loss guard.

        `edges` is an iterable of `(src, dst, label, props)`; `props` may carry the
        promoted keys `force` / `propagate` / `is_origin` / `order_key`, which land
        in real columns, plus anything else, which stays in the JSON blob.

        **MUST BE, AND IS, IDEMPOTENT.** Re-adding an existing edge UPDATEs in
        place via the `edge_key` primary key. Replay of the same segment N times
        leaves exactly one row and one edge counter increment.

        `stamp_rev` carries the same meaning as on the artifact store:
        `True` = locally authored, allocate fresh proper time; `False` = mesh
        consume, PRESERVE the incoming `(_origin, _seq)`."""
        handled = 0
        chunk: List[Any] = []
        for e in edges:
            chunk.append(e)
            if len(chunk) >= max(1, int(batch)):
                handled += self._add_chunk(chunk, stamp_rev)
                chunk = []
        if chunk:
            handled += self._add_chunk(chunk, stamp_rev)
        return handled

    def _add_chunk(self, chunk: List[Any], stamp_rev: bool) -> int:
        handled = 0
        with self.db.write() as cur:
            for i, e in enumerate(chunk):
                sp = "esp_%d" % i
                snapshot = self.seq.mark()
                cur.execute("SAVEPOINT " + sp)
                try:
                    handled += self._add_one(cur, e, stamp_rev)
                    cur.execute("RELEASE " + sp)
                except Exception:
                    cur.execute("ROLLBACK TO " + sp)
                    cur.execute("RELEASE " + sp)
                    self.seq.restore(snapshot)   # a rolled-back write consumes no proper time
        return handled

    def _add_one(self, cur: sqlite3.Cursor, e: Any, stamp_rev: bool) -> int:
        src, dst, label, props = self._unpack(e)
        key = K.edge_key(src, dst, label)
        leaf = K.leaf_of(key.hex(), self.leaves)

        # ONE read of the prior row — used for version comparison, accounting, AND the merkle XOR-out.
        old = cur.execute(
            "SELECT _origin, _seq, force, propagate, is_origin, order_key, props "
            "FROM edge WHERE edge_key = ?", (key,)).fetchone()

        if stamp_rev:
            origin, seq_val = self.origin, self.seq.allocate(cur)
        else:
            origin = props.get("_origin")
            seq_val = props.get("_seq")
            if not isinstance(origin, str) or not origin or not isinstance(seq_val, int):
                raise ValueError(
                    "add_edges(stamp_rev=False) requires (_origin, _seq) on the "
                    "incoming edge props — that is what 'preserve the origin's "
                    "version' means. Got _origin=%r _seq=%r for (%r,%r,%r)."
                    % (origin, seq_val, src, dst, label))
            _seq.observe_seq(cur, origin, seq_val)
            if old is not None:
                verdict = K.compare_version(origin, seq_val, old["_origin"], old["_seq"])
                if verdict in (K.OLDER, K.SAME):
                    return 1                     # correctly rejected -> HANDLED
                if verdict == K.UNORDERED:
                    _seq.bump(cur, "conflict:unordered_edge", 1)
                    return 1                     # keep local; a decision -> HANDLED

        blob = {k: v for k, v in props.items()
                if k not in _PROMOTED and not k.startswith("_")}
        is_origin = props.get("is_origin")
        norm_io = None if is_origin is None else int(bool(is_origin))
        cur.execute(
            "INSERT INTO edge(edge_key, src, dst, label, force, propagate, is_origin,"
            " order_key, _origin, _seq, _leaf, props) VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(edge_key) DO UPDATE SET force=excluded.force,"
            " propagate=excluded.propagate, is_origin=excluded.is_origin,"
            " order_key=excluded.order_key, _origin=excluded._origin,"
            " _seq=excluded._seq, _leaf=excluded._leaf, props=excluded.props",
            (key, src, dst, label, props.get("force"), props.get("propagate"),
             norm_io, props.get("order_key"), origin, seq_val, leaf, json.dumps(blob)))

        # ── incremental merkle: the SAME leaf_digest vertices use, so ONE tree covers both ──
        # An edge is an IDENTITY-FACT, so its contribution is edge_hash(edge_key, content) —
        # NODE-INVARIANT (excludes _origin/_seq). XOR out the OLD content, XOR in the NEW. An
        # idempotent replay of an identical edge XORs the same value out then in and cancels to a
        # no-op, so replay never churns the leaf; a genuine props change updates it exactly.
        if old is not None:
            _seq.xor_leaf(cur, leaf, K.edge_hash(key, self._hash_content(old)))
        _seq.xor_leaf(cur, leaf, K.edge_hash(key, {
            "force": props.get("force"), "propagate": props.get("propagate"),
            "is_origin": norm_io, "order_key": props.get("order_key"), **blob}))

        # ── allocation accounting ──
        # An idempotent replay that UPDATES in place still allocates a fresh seq and vacates the old
        # one — edges replicate, so they are accounted exactly as vertices are.
        if old is None:
            _seq.bump(cur, _schema.c_edge_total(), 1)
        else:
            _seq.vacate(cur, old["_origin"])
        _seq.bump(cur, _schema.c_rows(origin), 1)
        return 1

    @staticmethod
    def _hash_content(row: sqlite3.Row) -> Dict[str, Any]:
        """The node-invariant content of an existing edge row, for `edge_hash` on the XOR-out. Must
        reconstruct EXACTLY what was hashed in on the prior write: promoted columns + the blob props."""
        try:
            blob = json.loads(row["props"]) if row["props"] else {}
        except Exception:
            blob = {}
        return {"force": row["force"], "propagate": row["propagate"],
                "is_origin": row["is_origin"], "order_key": row["order_key"], **blob}

    @staticmethod
    def _unpack(e: Any) -> Tuple[str, str, str, Dict[str, Any]]:
        if isinstance(e, dict):
            src, dst, label = e.get("src"), e.get("dst"), e.get("label")
            props = {k: v for k, v in e.items() if k not in ("src", "dst", "label")}
        else:
            t = tuple(e)
            src, dst, label = t[0], t[1], t[2]
            props = dict(t[3]) if len(t) > 3 and t[3] else {}
        if not src or not dst or not label:
            raise ValueError("edge requires non-empty (src, dst, label); got %r" % (e,))
        return str(src), str(dst), str(label), props

    def delete_edge(self, from_id: str, to_id: str, label: str) -> bool:
        key = K.edge_key(from_id, to_id, label)
        with self.db.write() as cur:
            prev = cur.execute(
                "SELECT _origin, force, propagate, is_origin, order_key, props "
                "FROM edge WHERE edge_key = ?", (key,)).fetchone()
            if prev is None:
                return False
            cur.execute("DELETE FROM edge WHERE edge_key = ?", (key,))
            if cur.rowcount < 1:
                return False
            # XOR the edge's contribution back out of the shared merkle tree — a delete is just
            # xor_leaf(old_hash), XOR being its own inverse (same reason vertex delete works).
            _seq.xor_leaf(cur, K.leaf_of(key.hex(), self.leaves),
                          K.edge_hash(key, self._hash_content(prev)))
            _seq.bump(cur, _schema.c_edge_total(), -1)
            _seq.vacate(cur, prev["_origin"])   # an ACCOUNTED removal
            return True

    # ── reads ────────────────────────────────────────────────────────────────
    def edge_mark(self, node_id: str, *, direction: str = "out",
                  cap: int = 256) -> Tuple[int, int, bool]:
        """`(degree, max _seq, exhaustive)` over the edges on one side of `node_id` — **that node's
        edge set's own freshness stamp**, the edge half of `version_of`.

        A vertex's `_seq` does not move when only its EDGES change, and on this corpus the
        edges are not decoration: MEASURED on 71, `wn-dog.n.01` carries no `hypernyms` field
        at all, so `wn_store._synset_from_doc` reads the taxonomy from `edge WHERE src=?`.
        A reader that verified only the vertex would serve a synset whose parents had moved.

        Both halves are needed and neither alone is enough: `max(_seq)` alone misses the
        deletion of any edge that was not the newest, and `count` alone misses an update
        (which reallocates `_seq` in place). Together they move on every write to this
        node's edges — every write allocates a fresh `_seq`, so an add, a re-add and a
        delete each change one term or the other.

        Seeks `ix_e_src`/`ix_e_dst`, i.e. an index range over THIS node's handful of edges
        — never a scan and never a `count(*)` over the table. MEASURED on 71's 5.7 GB
        lattice, over `wn-dog.n.01`'s 2 edges: **9.7 µs**."""
        return edge_mark(self.db, node_id, direction=direction, cap=cap)

    def neighbors(self, node_id: str, label: Optional[str] = None, *,
                  direction: str = "out") -> List[str]:
        """Neighbor ids. `direction` in {'out','in','both'}. Seeks `ix_e_src` /
        `ix_e_dst`, which are `(src, label)` / `(dst, label)` — so a labelled walk
        is a composite seek, not a filtered scan."""
        if direction == "both":
            out = self.neighbors(node_id, label, direction="out")
            inn = self.neighbors(node_id, label, direction="in")
            return list(dict.fromkeys(out + inn))
        if direction == "out":
            sel, key = "dst", "src"
        elif direction == "in":
            sel, key = "src", "dst"
        else:
            raise ValueError("direction must be out|in|both, got %r" % (direction,))
        params: List[Any] = [node_id]
        clause = ""
        if label:
            clause = " AND label = ?"
            params.append(label)
        rows = self.db.read().execute(
            "SELECT %s AS nid FROM edge WHERE %s = ?%s" % (sel, key, clause), params)
        return list(dict.fromkeys(r["nid"] for r in rows))

    def descendants(self, root_id: str, label: str, *,
                    direction: str = "out") -> List[str]:
        """Transitive reachable ids — the PRUNE/BFS analogue. `seen` includes the
        root, so a cycle terminates rather than looping; on a single-owner leaf
        there is no authorization prune.

        ⛔ `max_depth=64` REMOVED 2026-07-30. `seen` includes the root and every vertex is
        admitted once over a finite graph, so the BFS always drains — the cap was a bare claim
        that nothing 65 hops away matters. This is the primitive every light-cone sits on, and
        the same lattice was described as 4, 10, 25, 32 and 64 deep across five files.
        """
        seen = {root_id}
        frontier = [root_id]
        out: List[str] = []
        while frontier:
            nxt: List[str] = []
            for nid in frontier:
                for m in self.neighbors(nid, label, direction=direction):
                    if m not in seen:
                        seen.add(m)
                        nxt.append(m)
                        out.append(m)
            frontier = nxt
        return out

    def edges_of(self, node_id: str, *, label: Optional[str] = None,
                 direction: str = "out", limit: int = 1000) -> List[Dict[str, Any]]:
        """Full edge rows, not just neighbor ids — needed wherever `force`,
        `propagate`, `is_origin` or `order_key` matter (grant propagation reads
        `is_origin`; ordered collections read `order_key`)."""
        key = "src" if direction == "out" else "dst"
        params: List[Any] = [node_id]
        clause = ""
        if label:
            clause = " AND label = ?"
            params.append(label)
        params.append(int(limit))
        rows = self.db.read().execute(
            "SELECT src, dst, label, force, propagate, is_origin, order_key, "
            "_origin, _seq, props FROM edge WHERE %s = ?%s LIMIT ?" % (key, clause),
            params).fetchall()
        return [self._row(r) for r in rows]

    def dst_ids_by_label(self, label: str, *, limit: int = 1_000_000) -> List[str]:
        """Every id with an INCOMING edge of `label` — one query, not N.

        Replaces `genesis.py:1804` (`_consolidated_members`), which issued an
        openCypher `MATCH` the shim did not recognise and answered with `[]`, then
        `except: return set()`. Every artifact looked like a generator rather than
        a reconstructible member, which is a wrong answer feeding the ρ ledger."""
        rows = self.db.read().execute(
            "SELECT DISTINCT dst FROM edge WHERE label = ? LIMIT ?",
            (label, int(limit))).fetchall()
        return [r["dst"] for r in rows]

    def src_ids_by_label(self, label: str, *, limit: int = 1_000_000) -> List[str]:
        rows = self.db.read().execute(
            "SELECT DISTINCT src FROM edge WHERE label = ? LIMIT ?",
            (label, int(limit))).fetchall()
        return [r["src"] for r in rows]

    def page_by_origin(self, *, origin: Optional[str] = None, after_seq: int = 0,
                       limit: int = 200) -> List[Dict[str, Any]]:
        """The EDGE PUBLISH SCAN: `WHERE _origin = :me AND _seq > :cursor`, an
        indexed range walk on `ix_e_origin`. Edges replicate, so they get the same
        cursor primitive vertices do — and the same freedom from the `_rev`-tie
        group-completion workaround, because `_seq` is injective."""
        origin = origin or self.origin
        rows = self.db.read().execute(
            "SELECT src, dst, label, force, propagate, is_origin, order_key, "
            "_origin, _seq, props FROM edge WHERE _origin = ? AND _seq > ? "
            "ORDER BY _seq LIMIT ?", (origin, int(after_seq), int(limit))).fetchall()
        return [self._row(r) for r in rows]

    def count_edges(self) -> int:
        """Counter lookup. No `count(*)` on any path."""
        return _seq.counter_of(self.db, _schema.c_edge_total())

    def list_by_leaf(self, leaf: int, *,
                     cap: int = 200_000) -> Tuple[List[Dict[str, Any]], bool]:
        """Edges whose `_leaf == leaf`, as SHIPPED WIRE RECORDS `{"f","t","label","props"}`, plus an
        `exhaustive` flag — the edge half of the read side of the Merkle tree.

        Publish rebuilds a changed leaf's object from this (indexed by `ix_e_leaf`), and `_apply_edges`
        consumes EXACTLY this shape, so a leaf object can carry vertex docs and these edge records
        side by side and each applier takes its own lines. `props` carries the promoted grant/order
        columns so the consumer can rebuild them; `_origin`/`_seq` are NOT shipped — a consumer stamps
        its own reserved edge origin, which is why the edge merkle hash excludes them (node-invariant).

        `(records, exhaustive)` for the same reason as the vertex `list_by_leaf`: a truncated page must
        never be mistaken for a genuinely empty leaf, or a peer diffs a leaf whose object underserves
        it and stays divergent."""
        cap = max(0, int(cap))
        rows = self.db.read().execute(
            "SELECT src, dst, label, force, propagate, is_origin, order_key, props "
            "FROM edge WHERE _leaf = ? ORDER BY edge_key LIMIT ?",
            (int(leaf), cap + 1)).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows[:cap]:
            try:
                props = json.loads(r["props"]) if r["props"] else {}
            except Exception:
                props = {}
            for k in ("force", "propagate", "is_origin", "order_key"):
                if r[k] is not None:
                    props[k] = r[k]
            out.append({"f": r["src"], "t": r["dst"], "label": r["label"], "props": props})
        return out, len(rows) <= cap

    def backfill_edge_leaf(self, *, batch: int = 5000) -> Dict[str, Any]:
        """Stamp `_leaf` and XOR into `leaf_digest` for edges written before `_leaf` existed — the
        edge analogue of the vertex `_leaf` backfill. A one-shot pass over `WHERE _leaf IS NULL`.

        IDEMPOTENT: a row it stamps gets a non-NULL `_leaf` and is never revisited, so re-running
        double-counts nothing. A fresh store never needs this (every `_add_one` stamps `_leaf`); it
        exists only to bring an existing store's edges into the one tree without a rebuild."""
        stamped = 0
        while True:
            with self.db.write() as cur:
                rows = cur.execute(
                    "SELECT edge_key, force, propagate, is_origin, order_key, props "
                    "FROM edge WHERE _leaf IS NULL LIMIT ?", (int(batch),)).fetchall()
                if not rows:
                    break
                for r in rows:
                    key = r["edge_key"]
                    leaf = K.leaf_of(key.hex(), self.leaves)
                    cur.execute("UPDATE edge SET _leaf = ? WHERE edge_key = ?", (leaf, key))
                    _seq.xor_leaf(cur, leaf, K.edge_hash(key, self._hash_content(r)))
                    stamped += 1
            if len(rows) < int(batch):
                break
        return {"stamped": stamped}

    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        d = {k: r[k] for k in ("src", "dst", "label", "force", "propagate",
                               "is_origin", "order_key", "_origin", "_seq")}
        try:
            d["props"] = json.loads(r["props"]) if r["props"] else {}
        except Exception:
            d["props"] = {}
        return d


# ── the edge half of a derivation's freshness stamp ──────────────────────────────────────────
def edge_mark(db: LatticeConn, node_id: str, *, direction: str = "out",
              cap: int = 256) -> Tuple[int, int, bool]:
    """`(degree, max _seq, exhaustive)` over the edges on one side of `node_id`. Module-level so a
    reader that holds only the ARTIFACT store — which shares this `LatticeConn` — can take the
    stamp without also being handed a graph store. `LatticeGraphStore.edge_mark` is this function.

    See the method's docstring for why BOTH terms are needed and why neither alone is enough.

    ⛔ THE ROWS ARE COUNTED IN PYTHON, NOT BY `count(*)`, AND THAT IS NOT STYLE. `count(*)`
    dereferences every matching record ([[count-star-dereferences-every-record]]) and is banned
    package-wide with an AST guard (`test_no_count_star_reaches_sqlite`) because on node 71 it OOMs
    the acceptor thread and zombies the node. The first version of this function wrote
    `SELECT count(*), max(_seq)`, the guard caught it, and it was the guard that was right: a
    freshness stamp lives on a READ path, which is the worst possible place for an unbounded
    dereference. `LIMIT cap+1` over the `ix_e_src`/`ix_e_dst` range bounds the work absolutely and
    `exhaustive=False` REPORTS a node too wide to stamp rather than returning a prefix's count as
    if it were the whole — a mark over a prefix is stable while the tail moves, i.e. a check that
    cannot fail. Callers must treat `exhaustive=False` as UNVERIFIABLE, never as approximate."""
    if direction == "out":
        key = "src"
    elif direction == "in":
        key = "dst"
    else:
        raise ValueError("direction must be out|in, got %r" % (direction,))
    cap = max(0, int(cap))
    rows = db.read().execute(
        "SELECT _seq FROM edge WHERE %s = ? LIMIT ?" % key, (node_id, cap + 1)).fetchall()
    hi = 0
    for r in rows:
        v = r[0]
        if isinstance(v, int) and v > hi:
            hi = v
    return (min(len(rows), cap), hi, len(rows) <= cap)
