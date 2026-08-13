"""`LatticeGraphStore` — the edge half of the lattice store.

An edge is an observation: it records that some observer looked and found a
relation. That is why edges carry `(_origin, _seq)` exactly as vertices do — they
replicate, so they need a replication-stable version identity — and it is why
`add_edges` must be idempotent rather than merely convenient.

Idempotency is the load-bearing property here. Mesh segments are replayed:
consume is retried on any held cursor, so the same edge arrives again and again,
by design. The primary key `edge_key = blake2b(src || \\0 || dst || \\0 || label, 16)`
makes a re-add an update in place, so replaying a segment any number of times
leaves exactly one row per edge.
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
    from .store import GraphStore as _GraphStoreABC
except Exception:
    _GraphStoreABC = object             # type: ignore[assignment,misc]

# props keys that are promoted to real columns rather than left in the JSON blob
_PROMOTED = ("force", "propagate", "is_origin", "order_key")

#: The context relation: **`src` is the context, `dst` is what sits in it.**
#:
#: The direction is load-bearing, and it is the opposite of how the relation reads in prose
#: ("an artifact has an edge to its context"). `propagate` means *what authority passes from
#: src to dst* everywhere else in this store; pointing the edge the other way and keeping the
#: column would silently invert that meaning for one label, which is the kind of local
#: exception that turns a mask into a footgun. So authority descends along the arrow here
#: exactly as it does along `contains`, and "what is this node's context?" is the inbound
#: read — one indexed seek on `ix_e_dst`, no more expensive than the outbound one.
#:
#: A context edge is a distinct label, so `edge_key = blake2b(src ‖ dst ‖ label)` never
#: collides with the containment edge between the same pair: the two relations can coexist
#: on one pair and a re-add of either updates only its own row. That is what makes this
#: additive — no existing walk sees a new edge, because every existing walk names its label.
CONTEXT_LABEL = "context"

#: What a context edge propagates when its writer does not say.
#:
#: **The empty mask — nothing.** `Mask.from_propagate` decodes `'[]'` to `attenuation.NOTHING`:
#: a permitted, navigable path that transmits no authority. It is not a new encoding; it is the
#: value `artifacts_router` already writes on a non-lineage link, read by the one decoder.
#:
#: The default used to be `None`, which the column defines as *unrestricted* and the decoder
#: turns into :data:`attenuation.TOP` — so a context edge written with no arguments propagated
#: **every** CRUDEASIO action. A default that confers maximum authority is the wrong default for
#: a security-relevant edge: it makes the careless write the dangerous one, and it means an
#: endpoint that forwards a caller-supplied pair without a mask hands out full authority over
#: whatever it names. Fail-closed is the only defensible direction here, and unlike a
#: containment edge (whose NULL default is load-bearing for a corpus already on disk) no context
#: edge exists yet to be re-interpreted — the relation has no production writer, so changing the
#: default costs nothing and is free only until it is not.
#:
#: A writer that wants authority to flow states which actions: `propagate=["read"]`,
#: `propagate=Mask.of(("read", "update"))`, or `propagate=None` for genuinely unrestricted.
#: `None` keeps meaning what the column has always meant — this changes the *default*, never the
#: decoding of a stored value.
DEFAULT_CONTEXT_PROPAGATE: Tuple[str, ...] = ()


class LatticeGraphStore(_GraphStoreABC):  # type: ignore[misc,valid-type]
    """Typed, idempotent, replicating edges.

    Pass the same `LatticeConn` the artifact store uses when vertices and edges
    live in one file — then an artifact and its edges commit atomically, and
    `LatticeConn.write()`'s reentrancy makes that composition safe. Two separate
    connections means two separate transactions and a window in which an artifact
    exists with no edges."""

    EDGE_TYPE = "edge"      # `sync.py` interpolates this into edge-feed queries

    def __init__(self, path_or_conn: Any, *, origin: str,
                 allocator: Optional[SeqAllocator] = None,
                 leaves: int = K.DEFAULT_LEAVES):
        """`path_or_conn` is a filesystem path or an existing `LatticeConn`. The
        allocator is reused automatically; there is no default `origin`, since a
        generated one would change on restart and fork this node's proper time."""
        if not origin:
            raise ValueError("LatticeGraphStore requires an explicit `origin`")
        self.origin = origin
        if isinstance(path_or_conn, LatticeConn):
            self.db = path_or_conn
        else:
            self.db = LatticeConn(str(path_or_conn))
        self.seq = _seq.allocator_for(self.db, origin, override=allocator)
        # Edges XOR into the same leaf_digest tree vertices use, so the leaf modulus must match the
        # artifact store's — `open_lattice` resolves one value (the derived `natural_leaves(corpus)`,
        # or a stored value) and hands it to both. A mismatch would put an edge and a vertex with the
        # "same" leaf index into different buckets and no two nodes could ever agree on a root.
        # `reshard()` keeps the two in lockstep when the corpus grows.
        self.leaves = int(leaves)

    def ensure_schema(self) -> None:
        with self.db.write() as cur:
            _schema.ensure_schema(cur)

    # ── writes ───────────────────────────────────────────────────────────────
    def add_edge(self, from_id: str, to_id: str, label: str,
                 props: Optional[Dict[str, Any]] = None, *,
                 stamp_rev: bool = True) -> None:
        """Singular add. Delegates to the bulk path so there is one write path and
        the two cannot drift apart in idempotency semantics."""
        self.add_edges([(from_id, to_id, label, props or {})], stamp_rev=stamp_rev)

    def add_edges(self, edges: Iterable[Any], *, batch: int = 500,
                  stamp_rev: bool = True) -> int:
        """Bulk upsert. Returns the number handled, not written — same contract
        as `put_many`: written counts, a correct LWW rejection counts, an error
        does not count. Callers use the shortfall as a data-loss guard.

        `edges` is an iterable of `(src, dst, label, props)`; `props` may carry the
        promoted keys `force` / `propagate` / `is_origin` / `order_key`, which land
        in real columns, plus anything else, which stays in the JSON blob.

        Idempotent: re-adding an existing edge updates in place via the `edge_key`
        primary key, so replaying the same segment any number of times leaves
        exactly one row and one edge counter increment.

        `stamp_rev` carries the same meaning as on the artifact store:
        `True` = locally authored, allocate fresh proper time; `False` = mesh
        consume, preserve the incoming `(_origin, _seq)`."""
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

        # One read of the prior row — used for version comparison, accounting, and the merkle XOR-out.
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
                    return 1                     # correctly rejected -> handled
                if verdict == K.UNORDERED:
                    _seq.bump(cur, "conflict:unordered_edge", 1)
                    return 1                     # keep local; a decision -> handled

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

        # ── incremental merkle: the same leaf_digest vertices use, so one tree covers both ──
        # An edge is an identity fact, so its contribution is edge_hash(edge_key, content) —
        # node-invariant (excludes _origin/_seq). XOR out the old content, XOR in the new. An
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
            # The per-relation extent, in the same transaction and the same branch as the total —
            # so the two cannot drift apart, and an idempotent replay (which takes the `else`
            # branch) double-counts neither. `label` is inside `edge_key`, so an upsert can never
            # move a row between labels and there is no relabel case to account for.
            _seq.bump(cur, _schema.c_edge_label(label), 1)
        else:
            _seq.vacate(cur, old["_origin"])
        _seq.bump(cur, _schema.c_rows(origin), 1)
        return 1

    @staticmethod
    def _hash_content(row: sqlite3.Row) -> Dict[str, Any]:
        """The node-invariant content of an existing edge row, for `edge_hash` on the XOR-out. Must
        reconstruct exactly what was hashed in on the prior write: promoted columns + the blob props."""
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
            _seq.bump(cur, _schema.c_edge_label(label), -1)   # the extent falls too
            _seq.vacate(cur, prev["_origin"])   # an accounted removal
            return True

    # ── the context relation ─────────────────────────────────────────────────
    # A typed name over `add_edge` / `edges_of`, not a second write path: `add_context_edge`
    # delegates to `add_edges` like every other write, so idempotency, proper time and the
    # merkle XOR are the ones already proved above rather than a copy of them.
    def add_context_edge(self, context_id: str, node_id: str, *,
                         propagate: Any = DEFAULT_CONTEXT_PROPAGATE,
                         is_origin: bool = True,
                         order_key: Optional[str] = None,
                         stamp_rev: bool = True) -> None:
        """Record that `node_id` sits in the context `context_id`.

        A node may sit in several contexts — that is the point of making context an artifact
        — so this is an add, never a set. Re-adding the same pair updates one row in place.

        `is_origin` carries the same weight it does on a containment edge and defaults the
        same way: **True marks the defining context, through which authority propagates**;
        a shared or referential context edge must pass `is_origin=False`. Without that
        distinction anyone able to write an edge could name someone else's artifact as
        sitting in their context and inherit authority over it — the exact reason
        `list_origin_descendants` refuses to walk a non-origin link.

        `propagate` accepts what the column accepts: `None` (unrestricted), a list of action
        names, an already-encoded string, or an `attenuation.Mask`. See :meth:`_encode`.

        It **defaults to** :data:`DEFAULT_CONTEXT_PROPAGATE`, the empty mask — an edge written
        with no mask carries no authority at all. See that constant for why the old `None`
        default was the wrong one. `is_origin` and `propagate` are two independent narrowings and
        both must be opened for authority to flow: `is_origin=False` says *this is not a
        defining context*, `propagate=[]` says *and nothing passes through it either*.
        """
        self.add_edge(context_id, node_id, CONTEXT_LABEL,
                      {"propagate": self._encode(propagate),
                       "is_origin": bool(is_origin),
                       "order_key": order_key},
                      stamp_rev=stamp_rev)

    def delete_context_edge(self, context_id: str, node_id: str) -> bool:
        """Remove one context edge. Leaves any containment edge on the same pair alone —
        different label, different `edge_key`, different row."""
        return self.delete_edge(context_id, node_id, CONTEXT_LABEL)

    def context_edges(self, node_id: str, *, direction: str = "out",
                      limit: int = 1000) -> List[Dict[str, Any]]:
        """Full context-edge rows on one side of `node_id`.

        Rows, not neighbour ids, because `is_origin` and `propagate` are the whole question a
        traversal asks of a context edge; an id list would drop both and any walk built on it
        would confer authority a link was never meant to carry.

        `direction="out"` reads *what sits in this context*; `direction="in"` reads *the
        contexts this node sits in*. Both are composite index seeks (`ix_e_src` / `ix_e_dst`
        are `(src, label)` / `(dst, label)`), so neither direction is the expensive one.
        """
        return self.edges_of(node_id, label=CONTEXT_LABEL, direction=direction, limit=limit)

    @staticmethod
    def _encode(propagate: Any) -> Any:
        """A `propagate` value in the form the TEXT column stores.

        Byte-identical to `lattice_api._ser_propagate` for the list form, and to
        `attenuation.Mask.to_propagate()` for the mask form — the encoding is not reinvented
        here, it is just applied at the one point a context edge is written. A raw string
        (including the substrate's compact `"r"`) passes through untouched, because this is a
        writer for a column that already holds three shapes and correcting one of them here
        would rewrite the meaning of live edges.
        """
        encoder = getattr(propagate, "to_propagate", None)
        if callable(encoder):
            return encoder()
        if isinstance(propagate, (list, tuple)):
            return json.dumps([str(v) for v in propagate])
        if isinstance(propagate, (set, frozenset)):
            # Sorted, because an unordered container has no order to preserve and two writes
            # of the same authority must produce the same bytes — the edge's merkle hash is
            # taken over the stored value, so a set that serialized differently on two nodes
            # would give the same edge two leaf contributions and they could never converge.
            return json.dumps(sorted(str(v) for v in propagate))
        return propagate

    # ── reads ────────────────────────────────────────────────────────────────
    def edge_mark(self, node_id: str, *, direction: str = "out",
                  cap: int = 256) -> Tuple[int, int, bool]:
        """`(degree, max _seq, exhaustive)` over the edges on one side of `node_id` — that node's
        edge set's own freshness stamp, the edge half of `version_of`.

        A vertex's `_seq` does not move when only its edges change, and on this corpus the
        edges are not decoration: `wn-dog.n.01` carries no `hypernyms` field at all, so
        `wn_store._synset_from_doc` reads the taxonomy from `edge WHERE src=?`. A reader that
        verified only the vertex would serve a synset whose parents had moved.

        Both halves are needed and neither alone is enough: `max(_seq)` alone misses the
        deletion of any edge that was not the newest, and `count` alone misses an update
        (which reallocates `_seq` in place). Together they move on every write to this
        node's edges — every write allocates a fresh `_seq`, so an add, a re-add and a
        delete each change one term or the other.

        Seeks `ix_e_src`/`ix_e_dst`, i.e. an index range over this node's handful of edges
        — never a scan and never a `count(*)` over the table."""
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
        """Every id with an incoming edge of `label` — one query, not N."""
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
        """The edge publish scan: `WHERE _origin = :me AND _seq > :cursor`, an
        indexed range walk on `ix_e_origin`. Edges replicate, so they get the same
        cursor primitive vertices do — a strict `>` cursor with no revision-group
        completion workaround, because `_seq` is injective."""
        origin = origin or self.origin
        rows = self.db.read().execute(
            "SELECT src, dst, label, force, propagate, is_origin, order_key, "
            "_origin, _seq, props FROM edge WHERE _origin = ? AND _seq > ? "
            "ORDER BY _seq LIMIT ?", (origin, int(after_seq), int(limit))).fetchall()
        return [self._row(r) for r in rows]

    def count_edges(self) -> int:
        """Counter lookup. No `count(*)` on any path."""
        return _seq.counter_of(self.db, _schema.c_edge_total())

    # ── the per-relation extent ──────────────────────────────────────────────
    def edge_labels_measured(self) -> bool:
        """Do the `edge:label:*` counters cover every edge in this store?

        See `schema.c_edge_label_built`. False is not a failure — it is the state of a store
        whose edges have not yet been counted, and the only correct response to it
        is to say not measured (or to run the backfill), never to read 0 out of an absent row."""
        return _seq.counter_of(self.db, _schema.c_edge_label_built()) == 1

    def count_edges_by_label(self, label: str) -> Optional[int]:
        """How many edges carry `label` — or `None`, meaning not measured.

        A counter lookup: one indexed read of one row, no `count(*)`, no scan. A plain
        `SELECT COUNT(*) FROM edge WHERE label=?` has no index to seek (the edge indexes lead
        with `src`/`dst`), so it would dereference the whole edge table for one integer.

        `None` and `0` are different answers and callers must keep them different. `0` is a
        measured zero: this store has been counted and holds no such relation. `None` is the
        absence of a measurement, and any caller that maps it to 0 has invented a number."""
        if not self.edge_labels_measured():
            return None
        return _seq.counter_of(self.db, _schema.c_edge_label(label))

    def labels_with_edges(self) -> Optional[List[str]]:
        """Every label with a non-zero extent, off the counters — or `None` if unmeasured.

        The counter-served answer to `SELECT DISTINCT label FROM edge`, which is a full scan
        (no index leads with `label`). Zero-extent counters are filtered out so a relation whose
        last edge was deleted stops being listed, exactly as `DISTINCT` over the rows would."""
        if not self.edge_labels_measured():
            return None
        pre = _schema.c_edge_label("")
        rows = self.db.read().execute(
            "SELECT name, n FROM counter WHERE name LIKE ? ORDER BY name", (pre + "%",)).fetchall()
        return [str(r["name"])[len(pre):] for r in rows if int(r["n"]) > 0]

    def backfill_edge_label_counters(self, *, chunk: int = 5000) -> Dict[str, Any]:
        """One-time measurement of `edge:label:*` for a store where the counter has not yet been
        computed.

        It may not use `count(*)`, which is the entire point of the counter — the ban is on
        `count(*)` dereferencing every record to produce one integer, not on iteration, so this
        walks the rows in keyset pages (`WHERE rowid > :cur ORDER BY rowid LIMIT n`, never
        `OFFSET`) reading only `(rowid, label)`, and tallies in Python. The working set is one
        page plus one integer per distinct label.

        One write transaction, paged reads inside it. `LatticeConn.write()` is `BEGIN IMMEDIATE`,
        so no other writer can interleave, which is what makes the tally exact rather than merely
        likely: a concurrent `add_edges` during a multi-transaction pass would either bump a
        counter for a row this scan then counts again (double), or insert below a cursor already
        passed (missed). Under WAL, readers are never blocked by this. It is an operator-run
        migration — the same standing as `rebuild_list_index` and `verify_counters` — never a
        request path.

        It sets rather than adds, and clears the whole `edge:label:` namespace first, so it is
        idempotent and it also drops a stale counter for a label whose rows are all gone. The
        marker is written last and in the same transaction: an interrupted backfill leaves the
        store uncertified and still answering not measured, which is the honest state.

        Returns `{"built", "scanned", "labels", "already"}`. Re-running on a certified store is a
        no-op (`already=True`) — the counters are maintained from the write path by then, and a
        re-measurement could only overwrite a live value with a stale one."""
        if self.edge_labels_measured():
            return {"built": True, "scanned": 0, "labels": 0, "already": True}
        tally: Dict[str, int] = {}
        scanned = 0
        with self.db.write() as cur:
            cursor = 0
            while True:
                rows = cur.execute(
                    "SELECT rowid AS rid, label FROM edge WHERE rowid > ? "
                    "ORDER BY rowid LIMIT ?", (cursor, int(chunk))).fetchall()
                if not rows:
                    break
                for r in rows:
                    lab = str(r["label"])
                    tally[lab] = tally.get(lab, 0) + 1
                    scanned += 1
                cursor = int(rows[-1]["rid"])
            cur.execute("DELETE FROM counter WHERE name LIKE ?",
                        (_schema.c_edge_label("") + "%",))
            for lab, n in sorted(tally.items()):
                cur.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?,?)",
                            (_schema.c_edge_label(lab), int(n)))
            cur.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?, 1)",
                        (_schema.c_edge_label_built(),))
        return {"built": True, "scanned": scanned, "labels": len(tally), "already": False}

    def list_by_leaf(self, leaf: int, *,
                     cap: int = 200_000) -> Tuple[List[Dict[str, Any]], bool]:
        """Edges whose `_leaf == leaf`, as shipped wire records `{"f","t","label","props"}`, plus an
        `exhaustive` flag — the edge half of the read side of the Merkle tree.

        Publish rebuilds a changed leaf's object from this (indexed by `ix_e_leaf`), and `_apply_edges`
        consumes exactly this shape, so a leaf object can carry vertex docs and these edge records
        side by side and each applier takes its own lines. `props` carries the promoted grant/order
        columns so the consumer can rebuild them; `_origin`/`_seq` are not shipped — a consumer stamps
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
        """Stamp `_leaf` and XOR into `leaf_digest` for edges where `_leaf` has not yet been
        computed — the edge analogue of the vertex `_leaf` backfill. A one-shot pass over
        `WHERE _leaf IS NULL`.

        Idempotent: a row it stamps gets a non-NULL `_leaf` and is never revisited, so re-running
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
    reader that holds only the artifact store — which shares this `LatticeConn` — can take the
    stamp without also being handed a graph store. `LatticeGraphStore.edge_mark` is this function.

    See the method's docstring for why both terms are needed and why neither alone is enough.
    Callers must treat `exhaustive=False` as unverifiable, never as approximate."""
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
