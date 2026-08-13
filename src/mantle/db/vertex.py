"""`LatticeArtifactStore` — the vertex half of the lattice store.

Implements `mantle.db.store.ArtifactStore` with typed methods in place of raw SQL call
sites. A typed method cannot drift the way a pattern-dispatched SQL string can: a
renamed method is an `AttributeError` at the call site rather than a query silently
falling through to `[]` and reporting a wrong count as if it were the real one.

Three invariants this module enforces:

  * **No `count(*)`.** Every count is a `counter` row maintained in the write
    transaction. See `seq.bump`.
  * **No `SKIP`/`OFFSET`.** Keyset only — `SKIP` at depth 5M costs roughly two
    orders of magnitude more than a keyset walk. `list_artifacts(skip=...)`
    raises rather than quietly doing the slow thing.
  * **`_seq` is proper time, never a clock.** See `seq.py`.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from . import constants as K
from . import schema as _schema
from . import seq as _seq
# `edge` imports only constants/schema/seq, so this is not a cycle — see `edge_mark` below for why
# the artifacts face publishes the edge half of a freshness stamp.
from . import edge as _edge
from .seq import LatticeConn, SeqAllocator

try:                                    # the declared seam, when mantle is importable
    from .store import ArtifactStore as _ArtifactStoreABC
except Exception:                       # stand-alone: the package must import on its own
    _ArtifactStoreABC = object          # type: ignore[assignment,misc]


#: Rows per keyset page over the `demand` (reached-copy) cache. Exported because the eviction
#: sweep in `mesh/demand.py` walks the same table through `demand_page` and must terminate on the
#: same value — "a short page is the last page" is only true against the `LIMIT` that was asked for,
#: so the page size and every end-of-walk test read this one name.
#:
#: This is a per-round memory bound, not a judgement about the data: the walk's answer is identical
#: at any page size, only the peak row count held at once changes. A different value would be right
#: if the envelope measurement (see `mesh/demand.budget_rows`) showed a different number of demand
#: rows fitting alongside the sweep's scored list — the bound is the envelope, not this number.
DEMAND_PAGE = 5000

#: Ids bound into one `IN (...)` statement by the plural reads (`get_many`, `versions_of_many`).
#:
#: SQLite caps host parameters per statement — `SQLITE_MAX_VARIABLE_NUMBER`, 999 on builds before
#: 3.32 and 32766 after. That ceiling belongs to whichever library the process happens to link, not
#: to the caller's request, so a plural read that binds the whole list would work on one machine and
#: raise `sqlite3.OperationalError: too many SQL variables` on another with the same input. Chunking
#: below the lowest ceiling makes the cost a function of the request alone: a 5,000-id read is six
#: statements here and six statements everywhere.
#:
#: Below 999 with room to spare, because a chunk is not the only thing bound — a plural read that
#: later grows a filter term binds it alongside the ids.
IN_CHUNK = 900


class Page(list):
    """A list of results that carries whether it is the whole answer.

    A `LIMIT`-truncated result and a complete one look identical as a bare list; `Page`
    makes the difference visible on the object itself rather than leaving a caller to
    infer completeness from a row count. `Page` is a real `list`, so every existing
    caller is unaffected; callers that care can ask.

        page = store.lookup_by_lemma("dog", content_type="text/x-wordnet")
        page.truncated      # True  => there are more; `limit` decided what you got
        page.limit          # the cap that was applied
    """

    __slots__ = ("truncated", "limit")

    def __init__(self, rows, *, truncated: bool = False, limit: Optional[int] = None):
        super().__init__(rows)
        self.truncated = bool(truncated)
        self.limit = limit


class ListIndexUnbuilt(RuntimeError):
    """Raised when the keyed arm is asked a question `listkey` cannot yet answer.

    `[]` from a keyed lookup means "this store does not contain that word" — a
    different claim from "this store has no index to ask". Raising instead of
    returning `[]` keeps those two claims distinct. See `schema.c_list_index_built`."""


class LatticeArtifactStore(_ArtifactStoreABC):  # type: ignore[misc,valid-type]
    """Artifacts as vertices. One SQLite file, one authoring observer.

    `origin` is this observer's identity and is stamped into `_origin` on every
    locally-authored write. It must be stable across restarts of the same node —
    a node that changes its origin forks its own proper time and every peer sees
    two unrelated, permanently-unordered event streams. (This is the same failure
    mode as an unpinned `EMBER_NODE_ID` on the S3 mesh plane.)"""

    def __init__(self, path_or_conn: Any, *, origin: str,
                 leaves: int = K.DEFAULT_LEAVES,
                 allocator: Optional[SeqAllocator] = None):
        """`path_or_conn` is a filesystem path or an existing `LatticeConn`.

        Pass a shared `LatticeConn` when the graph store uses the same file — then
        vertices and edges commit atomically and share one proper-time sequence,
        which contract §4 RESOLVED-5 requires (one counter per observer, spanning
        both tables). The allocator is reused automatically; `allocator=` is an
        override for unusual cases, not the way to be correct."""
        if not origin:
            raise ValueError(
                "LatticeArtifactStore requires an explicit `origin` — the authoring "
                "observer's stable identity. There is no default: a generated one "
                "would change on restart and fork this node's proper time.")
        self.origin = origin
        self.leaves = int(leaves)
        self.db = path_or_conn if isinstance(path_or_conn, LatticeConn) else LatticeConn(str(path_or_conn))
        self.seq = _seq.allocator_for(self.db, origin, override=allocator)

    # ── schema ───────────────────────────────────────────────────────────────
    def ensure_schema(self) -> None:
        with self.db.write() as cur:
            _schema.ensure_schema(cur)

    # ── column projection ────────────────────────────────────────────────────
    def _columns(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Project the hot filter columns out of the doc.

        `ct` and `offer` are declared in contract §2 as pointing at content-type
        and offer vertices, which Phase 3 mints. Until then `ct` holds the MIME
        string verbatim (a valid vertex id when Phase 3 promotes it) and `offer`
        is populated only from an explicit string `offer` field. `doc["context"]`
        is conceptually the offer (see `db/store.py`) but is frequently a dict,
        so it is not auto-projected here — flattening a dict into a TEXT column
        is how you get a query predicate that matches nothing and reports zero."""
        offer = doc.get("offer")
        ct_time = doc.get("created_time")
        return {
            "ct": doc.get("content_type"),
            "offer": offer if isinstance(offer, str) else None,
            "content_ref": doc.get("content_ref"),
            "created_by": doc.get("created_by"),
            # `created_time` is a column (a deliberate override — see the note on
            # `schema.VERTEX_DDL`). Projected here so the column is populated rather than merely
            # present: a column that is always NULL protects nothing.
            #
            "created_time": None if ct_time is None else str(ct_time),
            # The collection's immutable origin root — the key root for content
            # encryption. See `schema.VERTEX_DDL` for why it is a column.
            "origin_root": self._origin_root(doc),
            "root_id": doc.get("root_id") or doc.get("id"),
        }

    @staticmethod
    def _origin_root(doc: Dict[str, Any]) -> Optional[str]:
        """The containment root this vertex is permanently keyed under.

        Precedence, and each step is deliberate:

          1. an explicit ``origin_root`` — already resolved upstream; never recomputed,
             because recomputing a value whose contract is "never moves" is how it moves;
          2. ``collection_id`` — no collection nests today, so the origin-lineage walk is
             depth-1 and the parent is the root;
          3. the vertex's own id — a top-level artifact with no collection is its own
             root (the `vtype.*` type definitions are the live example).

        Returns ``None`` only when the doc has no id at all, which `put_artifact`
        rejects separately. There is no guess and no fallback to `created_by`: that field
        is provenance and mutates, and keeping it out of the crypto path is what makes
        correcting an identity unable to orphan a blob.
        """
        explicit = doc.get("origin_root")
        if explicit:
            return str(explicit)
        collection_id = doc.get("collection_id")
        if collection_id:
            return str(collection_id)
        vid = doc.get("id")
        return str(vid) if vid else None

    # The doc key recording which observer claimed `created_time`. See
    # `_attribute_time` — the value is a claim, so it names its claimant.
    TIME_CLAIM_KEY = "created_time_origin"

    # ── the keyed arm ────────────────────────────────────────────────────────
    # The multi-valued doc fields unrolled into `listkey`. A closed set, deliberately:
    # `lookup_by_list_field` refuses a field outside it rather than seeking an index that was
    # never populated for it and reporting the resulting `[]` as "no matches" — the same
    # wrong-answer-not-empty-answer trap the build marker exists to close, one level down.
    LIST_INDEX_FIELDS: Tuple[str, ...] = ("lemmas", "calls")

    def _index_lists(self, cur: sqlite3.Cursor, aid: str, doc: Optional[Dict[str, Any]],
                     ct: Any, *, had_prev: bool = True) -> int:
        """Re-post this artifact's indexed list fields; returns the number of postings written.

        Delete-then-insert rather than a diff: the posting set is small (a synset has ~3 lemmas)
        and a diff has to be correct about removals, which is where an index silently drifts out
        of step with its table. `doc=None` is the delete path, so an artifact removed entirely
        gets the same delete-then-insert handling as one whose list fields were cleared."""
        if had_prev:
            cur.execute("DELETE FROM listkey WHERE aid = ?", (aid,))
        if doc is None:
            return 0
        rows = []
        ct_s = None if ct is None else str(ct)
        for field in self.LIST_INDEX_FIELDS:
            vals = doc.get(field)
            if not isinstance(vals, (list, tuple, set)):
                continue            # a scalar in a list field is malformed; index nothing
            for v in vals:
                if v is None:
                    continue
                rows.append((aid, field, str(v).lower(), ct_s))
        if rows:
            cur.executemany(
                "INSERT INTO listkey(aid, field, value, ct) VALUES(?,?,?,?)", rows)
        return len(rows)

    def _attribute_time(self, d: Dict[str, Any], origin: str) -> None:
        """Attribute `doc["created_time"]` to the observer that claimed it.

        `created_time` is a deliberate override of the column rule (contract §2.2, see
        `schema.VERTEX_DDL`): a clock reading is observer-dependent, so by that rule it is
        content — the claim "observer X read its clock as T" — and content lives in `doc`. An
        anonymous integer is a weaker claim than the truth supports, because two observers reading
        their own clocks disagree with no function to reconcile them. Naming the claimant makes
        the value interpretable: it is not "the time" but "the time according to 71".

        The claim key (`TIME_CLAIM_KEY`) adds no column and no index. If you need ordering, that
        is `edge.order_key` within a frame and graph reachability across frames — unordered is a
        valid answer."""
        if d.get("created_time") is None:
            return
        if not d.get(self.TIME_CLAIM_KEY):
            d[self.TIME_CLAIM_KEY] = origin

    # ── the single write path ────────────────────────────────────────────────
    def _write_row(self, cur: sqlite3.Cursor, doc: Dict[str, Any],
                   origin: str, seq_val: Optional[int]) -> Dict[str, Any]:
        aid = doc["id"]
        d = dict(doc)
        d["_origin"] = origin
        d["_seq"] = seq_val
        self._attribute_time(d, origin)     # a clock reading is a claim; name the claimant

        prev = cur.execute(
            "SELECT ct, _seq, _origin, doc FROM vertex WHERE id = ?", (aid,)).fetchone()

        # `created_time` is first-write-wins: an existing row's value and its claimant carry
        # forward unchanged. That is what makes the §2.3 exemption hold — the field is admissible
        # as a column only while it does not move under replication. Pinned by
        # `test_lattice.py::test_created_time_survives_in_doc_attributed_to_its_claimant`.
        if prev is not None:
            try:
                pdoc = json.loads(prev["doc"]) if prev["doc"] else {}
            except Exception:
                pdoc = {}
            if pdoc.get("created_time") is not None:
                d["created_time"] = pdoc["created_time"]
                if pdoc.get(self.TIME_CLAIM_KEY) is not None:
                    d[self.TIME_CLAIM_KEY] = pdoc[self.TIME_CLAIM_KEY]

        leaf = K.leaf_of(aid, self.leaves)
        cols = self._columns(d)
        cur.execute(
            "INSERT INTO vertex(id, ct, offer, content_ref, created_by, created_time,"
            " origin_root, root_id, _origin, _seq, _leaf, doc)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET ct=excluded.ct, offer=excluded.offer,"
            " content_ref=excluded.content_ref, created_by=excluded.created_by,"
            " created_time=excluded.created_time, origin_root=excluded.origin_root,"
            " _origin=excluded._origin,"
            " _seq=excluded._seq, _leaf=excluded._leaf, doc=excluded.doc",
            (aid, cols["ct"], cols["offer"], cols["content_ref"], cols["created_by"],
             cols["created_time"], cols["origin_root"], cols["root_id"], origin, seq_val, leaf,
             json.dumps(d)))

        # ── incremental merkle: XOR out the old identity, XOR in the new ──
        if prev is not None:
            _seq.xor_leaf(cur, leaf, K.row_hash(aid, prev["_seq"]))
        _seq.xor_leaf(cur, leaf, K.row_hash(aid, seq_val))

        # ── allocation accounting ──
        # An update allocates a fresh seq and vacates the old one, so live rows go
        # non-contiguous. Recording the vacancy here — in the same transaction —
        # is what keeps `live_rows + vacated == last_seq` exact, and what keeps a
        # row lost outside this path distinguishable from one retired by it.
        if prev is not None:
            _seq.vacate(cur, prev["_origin"])
        _seq.bump(cur, _schema.c_rows(origin), 1)

        # ── incremental counters ──
        old_doc = json.loads(prev["doc"]) if prev is not None else None
        self._recount(cur, old_doc, d, prev["ct"] if prev is not None else None, cols["ct"])
        self._sync_task(cur, aid, old_doc, d, cols["ct"])
        # The keyed arm's index, maintained in this transaction — so a lemma posting and the
        # doc it points at can never be observed disagreeing, and an update that changes `ct`
        # re-posts under the new type rather than leaving a stale discriminator behind.
        self._index_lists(cur, aid, d, cols["ct"], had_prev=prev is not None)
        return d

    def _recount(self, cur: sqlite3.Cursor, old: Optional[Dict[str, Any]],
                 new: Dict[str, Any], old_ct: Any, new_ct: Any) -> None:
        if old is None:
            _seq.bump(cur, _schema.c_vertex_total(), 1)
        for name, delta in self._counter_deltas(old, new, old_ct, new_ct).items():
            _seq.bump(cur, name, delta)

    @staticmethod
    def _counter_deltas(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]],
                        old_ct: Any, new_ct: Any) -> Dict[str, int]:
        d: Dict[str, int] = {}

        def add(name: str, delta: int) -> None:
            d[name] = d.get(name, 0) + delta

        # THE `state` COUNTERS PARTITION ON THE RECORDED VALUE, NOT ON `K.state_of`.
        #
        # `K.STATE_WHEN_ABSENT` says what a doc with no `state` IS — and every reader of an
        # ARTIFACT derives from it. These counters are not that reader. `vertex` is a mixed
        # plane: people, grants, commits, commit items and the materialized markers all live in
        # this table, most of them carrying no `state` at all, and grants carry one on an
        # orthogonal axis (a grant lifecycle — see `revise`, which refuses a caller-supplied
        # `state` for exactly this collision). Defaulting them would fold every person and every
        # commit row into `state:committed`, which `scripts/collect_usage_metrics` reports as
        # `committed_artifact_versions_total`.
        #
        # So a doc that records no state is counted in no `state:*` bucket and in no
        # `committed_only` bucket, and `count(state=…)` stays exactly the O(1) sibling of
        # `list_artifacts(state=…)` — both answer "what does the row record". The absent-state
        # default is a claim about artifact docs, and this path cannot tell an artifact doc from a
        # person doc: `root_id` and `collection_id` are both populated on non-artifacts.
        for doc, sign in ((old, -1), (new, 1)):
            if doc is None:
                continue
            ct = old_ct if sign < 0 else new_ct
            committed = doc.get("state") == "committed"
            if ct:
                add(_schema.c_ct(str(ct)), sign)
            if doc.get("state"):
                add("state:" + str(doc["state"]), sign)
            if doc.get("collection_id"):
                add(_schema.c_collection(str(doc["collection_id"])), sign)
                if committed:
                    add(_schema.c_collection(str(doc["collection_id"]),
                                             committed_only=True), sign)
            # The provenance null-audit. Counted here, on the single write path, so a
            # doc that arrives without provenance and a backfill that fills it in
            # are the same bookkeeping operation with opposite signs.
            for field in K.NULL_AUDIT_FIELDS:
                if K.is_missing(doc.get(field)):
                    add(_schema.c_missing(field), sign)
                    if committed:
                        add(_schema.c_missing(field, committed_only=True), sign)
        return {k: v for k, v in d.items() if v}

    # ── the task sidecar ─────────────────────────────────────────────────────
    def _sync_task(self, cur: sqlite3.Cursor, aid: str, old: Optional[Dict[str, Any]],
                   new: Optional[Dict[str, Any]], ct: Any) -> None:
        """Keep `task` in step with `doc.status`, transactionally.

        There is exactly one write path here and it derives every sidecar field
        from the doc, so the two cannot disagree."""
        old_status = str(old.get("status")) if old and old.get("status") else None
        old_ct = str(old.get("content_type")) if old and old.get("content_type") else None
        new_status = str(new.get("status")) if new and new.get("status") else None

        if old_status and old_ct:
            _seq.bump(cur, _schema.c_task_status(old_ct, old_status), -1)
        if new is None or not new_status or not ct:
            cur.execute("DELETE FROM task WHERE id = ?", (aid,))
            return

        _seq.bump(cur, _schema.c_task_status(str(ct), new_status), 1)
        cur.execute(
            "INSERT INTO task(id, ct, status, priority, operator, task_key,"
            " claimed_by, claimed_at, next_retry_at, completed_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET ct=excluded.ct, status=excluded.status,"
            " priority=excluded.priority, operator=excluded.operator,"
            " task_key=excluded.task_key, claimed_by=excluded.claimed_by,"
            " claimed_at=excluded.claimed_at, next_retry_at=excluded.next_retry_at,"
            " completed_at=excluded.completed_at",
            (aid, str(ct), new_status, int(new.get("priority") or 0),
             new.get("operator"), new.get("task_key"), new.get("claimed_by"),
             new.get("claimed_at"), new.get("next_retry_at"), new.get("completed_at")))

    # ── ArtifactStore: single put ────────────────────────────────────────────
    def put_artifact(self, doc: Dict[str, Any], *, stamp_rev: bool = True) -> Dict[str, Any]:
        """Idempotent upsert keyed on `id`. Returns the stored doc.

            stamp_rev=True   Local authorship. Allocates a fresh `_seq` from this
                             observer's proper time and stamps `_origin = self.origin`.
                             This is a new authored event.

            stamp_rev=False  Mesh consume. Preserves the incoming `(_origin, _seq)`
                             exactly. A replicated doc keeps the version identity
                             it was published with, so it does not echo around the
                             mesh forever, and the peer's merkle leaf for that row
                             matches ours once applied.

        With `stamp_rev=False` a doc carrying no `(_origin, _seq)` raises `ValueError`
        rather than being re-stamped locally: re-stamping would claim this node
        authored a peer's event, corrupting the only causal record the system has."""
        if not doc.get("id"):
            raise ValueError("put_artifact: doc has no 'id'")
        with self.db.write() as cur:
            # ── content decides whether this is a new version ────────────────────────────────
            # The deciding information is `content_ref` = cas/<sha256-of-the-content>:
            #   · same content_ref — the bytes are unchanged; this is a re-describe (offer,
            #     context, any derived field), applied in place. Not a version.
            #   · different content_ref — the bytes changed; the prior version is snapshotted
            #     under a derived archived id (sharing root_id) before the handle is overwritten,
            #     so `get(id)` always returns the latest ("version_id is latest") and the prior
            #     content is never lost. `revise()` is a convenience, not a gate: the single write
            #     path decides, driven by the data, not by which method was called.
            # Local authorship only (stamp_rev): a mesh-consumed row is already version-distinct
            # by its peer (_origin,_seq), so snapshotting it here would double-count a peer's history.
            if stamp_rev and doc.get("content_ref"):
                prev = cur.execute(
                    "SELECT content_ref, root_id, doc FROM vertex WHERE id = ?",
                    (doc["id"],)).fetchone()
                if prev is not None and prev["content_ref"] \
                        and prev["content_ref"] != doc["content_ref"]:
                    self._snapshot_prior(cur, doc["id"], prev)
            origin, seq_val = self._version_for(cur, doc, stamp_rev)
            if not stamp_rev:
                _seq.observe_seq(cur, origin, seq_val)
            return self._write_row(cur, doc, origin, seq_val)

    def _snapshot_prior(self, cur: sqlite3.Cursor, aid: str, prev: sqlite3.Row) -> None:
        """Preserve the prior version of `aid` under a derived, archived id before its content is
        replaced in place. The snapshot shares the handle's `root_id`; the handle stays the head.

        The snapshot is written through `_write_row`, so it picks up the merkle XOR, `_seq`
        allocation and counter accounting without duplicating any of them. Idempotent: the derived
        id is a function of the prior `content_ref`, so a repeated call finds the row and returns."""
        try:
            pdoc = json.loads(prev["doc"]) if prev["doc"] else {}
        except Exception:
            return
        if not pdoc:
            return
        root = prev["root_id"] or aid
        old_ref = prev["content_ref"] or ""
        snap_id = "%s@%s" % (aid, hashlib.blake2b(old_ref.encode("utf-8"),
                                                  digest_size=8).hexdigest())
        # degenerate, or already snapshotted (idempotent)
        if snap_id == aid or cur.execute(
                "SELECT 1 FROM vertex WHERE id = ?", (snap_id,)).fetchone():
            return
        snap = dict(pdoc)
        snap["id"] = snap_id
        snap["root_id"] = root
        snap["state"] = "archived"
        snap["superseded_by"] = aid
        origin, seq_val = self._version_for(cur, snap, True)
        self._write_row(cur, snap, origin, seq_val)

    def _version_for(self, cur: sqlite3.Cursor, doc: Dict[str, Any],
                     stamp_rev: bool) -> Tuple[str, int]:
        if stamp_rev:
            return self.origin, self.seq.allocate(cur)
        origin = doc.get("_origin")
        sq = doc.get("_seq")
        if not isinstance(origin, str) or not origin or not isinstance(sq, int):
            raise ValueError(
                "put(stamp_rev=False) requires the incoming doc to carry "
                "(_origin, _seq) — that is what 'preserve the origin's version' "
                "means. Got _origin=%r _seq=%r for id=%r. Re-stamping locally "
                "would claim this node authored a peer's event."
                % (origin, sq, doc.get("id")))
        return origin, sq

    # ── ArtifactStore: bulk put ──────────────────────────────────────────────
    def put_many(self, docs: Iterable[Dict[str, Any]], *, batch: int = 500,
                 stamp_rev: bool = True, on_unordered: str = "keep_local") -> int:
        """Bulk upsert. Returns the number *handled*, which is not the number written.

        The distinction is load-bearing. `mesh.sync._apply_artifacts` uses this
        return as its only guard against advancing the consume cursor:

            written = store.artifacts.put_many(batch, batch=500, stamp_rev=False)
            if written < len(batch): raise RuntimeError("partial apply: ...")

        A segment recorded as applied advances `last_key` behind a monotone
        `StartAfter` marker, so anything not written is unrecoverable. "Handled"
        therefore has a precise meaning every implementation must honour:

            doc written                     -> counts
            doc correctly rejected by LWW   -> counts  (declining to overwrite a
                                               newer local row is the right
                                               outcome; it is handled, not lost)
            doc that errored                -> does not count

        An implementation returning `len(docs)` unconditionally disables the mesh's
        only data-loss guard from inside the store layer.

        `on_unordered` decides what to do when an incoming doc and the local row
        have different `_origin`s — genuinely concurrent authorship of one vertex,
        which `(_origin, _seq)` cannot and must not order (contract RESOLVED-3: no
        clock, ever; no synthesized tiebreak). It is an explicit parameter rather
        than a buried default because there is no universally right answer:

            "keep_local"  keep ours, count as handled, bump `conflict:unordered`.
                          Nothing is lost — the peer still holds its copy.
            "take_remote" apply theirs, count as handled. For a consumer that has
                          decided the peer is authoritative for this vertex.
            "error"       do not count, so the mesh holds its cursor and the
                          divergence surfaces instead of being absorbed.

        Per-document savepoints: one bad doc must not roll back the batch, because
        a whole-batch rollback with a `len(docs)` return would report success for
        rows that were never written."""
        if on_unordered not in ("keep_local", "take_remote", "error"):
            raise ValueError("on_unordered must be keep_local|take_remote|error, got %r"
                             % (on_unordered,))
        handled = 0
        chunk: List[Dict[str, Any]] = []
        for doc in docs:
            chunk.append(doc)
            if len(chunk) >= max(1, int(batch)):
                handled += self._put_chunk(chunk, stamp_rev, on_unordered)
                chunk = []
        if chunk:
            handled += self._put_chunk(chunk, stamp_rev, on_unordered)
        return handled

    def _put_chunk(self, chunk: List[Dict[str, Any]], stamp_rev: bool,
                   on_unordered: str) -> int:
        handled = 0
        with self.db.write() as cur:
            for i, doc in enumerate(chunk):
                sp = "sp_%d" % i
                snapshot = self.seq.mark()
                cur.execute("SAVEPOINT " + sp)
                try:
                    handled += self._put_one(cur, doc, stamp_rev, on_unordered)
                    cur.execute("RELEASE " + sp)
                except Exception:
                    cur.execute("ROLLBACK TO " + sp)
                    cur.execute("RELEASE " + sp)
                    # Restore proper time: the rolled-back write consumed none.
                    self.seq.restore(snapshot)
                    # Not counted as handled. The mesh will hold its cursor.
        return handled

    def _put_one(self, cur: sqlite3.Cursor, doc: Dict[str, Any], stamp_rev: bool,
                 on_unordered: str) -> int:
        if not doc.get("id"):
            raise ValueError("put_many: doc has no 'id'")
        origin, seq_val = self._version_for(cur, doc, stamp_rev)
        if not stamp_rev:
            _seq.observe_seq(cur, origin, seq_val)
            verdict = self._lww(cur, doc["id"], origin, seq_val)
            if verdict == K.OLDER or verdict == K.SAME:
                return 1                       # correctly rejected -> handled
            if verdict == K.UNORDERED:
                if on_unordered == "keep_local":
                    _seq.bump(cur, "conflict:unordered", 1)
                    return 1                   # a decision was made -> handled
                if on_unordered == "error":
                    raise ValueError(
                        "unordered concurrent authorship of vertex %r: incoming "
                        "_origin=%r vs local row from a different origin. There is "
                        "no clock and no tiebreak." % (doc["id"], origin))
        self._write_row(cur, doc, origin, seq_val)
        return 1

    def _lww(self, cur: sqlite3.Cursor, aid: str, origin: str, seq_val: int) -> str:
        row = cur.execute("SELECT _origin, _seq FROM vertex WHERE id = ?", (aid,)).fetchone()
        if row is None:
            return K.NEWER                     # nothing local -> incoming wins
        return K.compare_version(origin, seq_val, row["_origin"], row["_seq"])

    # ── ArtifactStore: reads ─────────────────────────────────────────────────
    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        r = self.db.read().execute(
            "SELECT doc FROM vertex WHERE id = ?", (artifact_id,)).fetchone()
        return json.loads(r["doc"]) if r else None

    def get_many(self, artifact_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """`{id: doc}` for the ids that exist — the plural of `get_artifact`.

        One `SELECT id, doc FROM vertex WHERE id IN (...)` per `IN_CHUNK` ids, seeking the PK
        index. A page of ids costs `ceil(n / IN_CHUNK)` statements rather than `n`, which is
        what lets a batching caller stop looping `get_artifact` — and what keeps it from
        hand-rolling this `IN (...)` at the call site ([[never-handroll-probes]]).

        Ids are deduplicated, so a repeated id costs one read and yields one entry. Ids with no
        row are ABSENT from the mapping — a miss is a missing key, never an empty doc. The
        mapping is built in the caller's id order, so iterating it is the caller's own order
        with the misses dropped, not chunk-arrival order.

        No `count(*)` and no `OFFSET`: this is a keyed lookup, and the set of keys is the
        bound."""
        ids = list(dict.fromkeys(str(a) for a in (artifact_ids or ()) if a))
        found: Dict[str, Dict[str, Any]] = {}
        if not ids:
            return found
        con = self.db.read()
        for i in range(0, len(ids), IN_CHUNK):
            chunk = ids[i:i + IN_CHUNK]
            rows = con.execute(
                "SELECT id, doc FROM vertex WHERE id IN (%s)" % ",".join(["?"] * len(chunk)),
                chunk).fetchall()
            for r in rows:
                found[r["id"]] = json.loads(r["doc"])
        return {aid: found[aid] for aid in ids if aid in found}

    def versions_of_many(self, root_ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        """`{root_id: [docs, oldest first]}` — the plural of `versions_of`.

        The root-lineage read for a set of roots: `WHERE root_id IN (...)` over `ix_v_root_id`,
        chunked at `IN_CHUNK`. A caller resolving "the current version of each of these roots"
        reads every lineage in `ceil(n / IN_CHUNK)` statements instead of one per root.

        Ordered by `(_origin, _seq)` exactly as `versions_of` is — the globally-unique version
        identity, gap-free per origin, rather than the writer's `created_time` claim. Grouping
        by root preserves that order within each lineage, so each list here is the list
        `versions_of(root)` returns.

        Roots are deduplicated. A root with no versions is ABSENT rather than mapped to `[]`,
        keeping "this root has no lineage" distinguishable from a root the caller never
        asked about."""
        roots = list(dict.fromkeys(str(r) for r in (root_ids or ()) if r))
        by_root: Dict[str, List[Dict[str, Any]]] = {}
        if not roots:
            return by_root
        con = self.db.read()
        for i in range(0, len(roots), IN_CHUNK):
            chunk = roots[i:i + IN_CHUNK]
            rows = con.execute(
                "SELECT root_id, doc FROM vertex WHERE root_id IN (%s) ORDER BY _origin, _seq"
                % ",".join(["?"] * len(chunk)), chunk).fetchall()
            for r in rows:
                by_root.setdefault(r["root_id"], []).append(json.loads(r["doc"]))
        return {rid: by_root[rid] for rid in roots if rid in by_root}

    def version_of(self, artifact_id: str) -> Optional[Tuple[Optional[str], Optional[int]]]:
        """`(_origin, _seq)` for one vertex, without hydrating the doc. `None` if
        absent. Feed both values to `constants.compare_version`.
        gate this pairs with."""
        r = self.db.read().execute(
            "SELECT _origin, _seq FROM vertex WHERE id = ?", (artifact_id,)).fetchone()
        return (r["_origin"], r["_seq"]) if r else None

    def write_mark(self) -> Tuple[Tuple[str, int], ...]:
        """The store's whole-of-store write mark — see `seq.write_mark` for what it is
        and why a cache may hang its validity on it. Published here because a reader must
        never hand-roll a probe against this schema ([[never-handroll-probes]]): a missing
        stat is a stat to add to the store, not a `SELECT` to write at the call site."""
        return _seq.write_mark(self.db)

    def edge_mark(self, artifact_id: str, *, direction: str = "out",
                  cap: int = 256) -> Tuple[int, int, bool]:
        """`(degree, max _seq, exhaustive)` over the edges on one side of `artifact_id` — the edge
        half of `version_of`, published on the artifacts face.

        A one-line delegation to `edge.edge_mark`, published here so a reader that holds only the
        artifact store — which shares this `LatticeConn` — can take the stamp without also being
        handed a graph store ([[never-handroll-probes]]: a stat a reader has to hand-roll is a
        stat the store failed to publish). `version_of` alone verifies half of what it claims to,
        since a vertex's `_seq` does not move when only its edges change — `wn-dog.n.01` on node
        71, for example, carries no `hypernyms` field at all.

        `exhaustive=False` means unverifiable, never approximate — see `edge.edge_mark`. Seeks
        `ix_e_src`/`ix_e_dst`; never a scan and never a `count(*)`."""
        return _edge.edge_mark(self.db, artifact_id, direction=direction, cap=cap)

    def list_artifacts(self, *, state: Optional[str] = None,
                       content_type: Optional[str] = None,
                       collection_id: Optional[str] = None,
                       created_by: Optional[str] = None,
                       limit: Optional[int] = None,
                       skip: int = 0,
                       include_archived: bool = False) -> Iterator[Dict[str, Any]]:
        """Filtered stream in `id` order.

        `skip` raises rather than paging by `OFFSET`. Keyset paging is the sanctioned
        primitive here; `page_by_id(after=<last id>, limit=n)` is the replacement."""
        if skip:
            raise ValueError(
                "list_artifacts(skip=%r): OFFSET pagination is not supported. "
                "Use page_by_id(after=<last id>, limit=n) — keyset paging. "
                "SKIP at a large offset costs roughly two orders of magnitude "
                "more than the equivalent keyset page." % (skip,))
        where, params = [], []
        for col, val in (("ct", content_type), ("created_by", created_by)):
            if val is not None:
                where.append(col + " = ?")
                params.append(val)
        for field, val in (("state", state), ("collection_id", collection_id)):
            if val is not None:
                where.append("json_extract(doc, '$.%s') = ?" % field)
                params.append(val)
        # Head-only by default: exclude archived unless the caller asked for a specific state or for
        # history. `IS NOT` is null-safe — a row with no `state` is kept, which is
        # `K.STATE_WHEN_ABSENT` ("committed") expressed as a predicate.
        #
        # The `state = ?` filter above is deliberately NOT widened by that default: it partitions
        # on the recorded value, the same reading the `state:*` counters take (see
        # `_counter_deltas`), so `count(state=…)` and this list agree row for row.
        if state is None and not include_archived:
            where.append("json_extract(doc, '$.state') IS NOT 'archived'")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        q = "SELECT doc FROM vertex" + clause + " ORDER BY id"
        if limit is not None and int(limit) >= 0:
            q += " LIMIT ?"
            params.append(int(limit))
        for r in self.db.read().execute(q, params):
            yield json.loads(r["doc"])

    def page_by_ct(self, *, content_type: str, after_seq: int = 0,
                   limit: int = 512) -> List[Dict[str, Any]]:
        """The carrier scan: `WHERE ct = :ct AND _seq > :cursor ORDER BY _seq`, returning
        `[{"_seq": int, "doc": {...}}]` so a caller can advance its own watermark.

        Exists because `list_artifacts(content_type=...)` returns every row of that type and
        `json.loads`s each one. A ground-plane carrier polls it twice per answer, so that cost
        grows with every leaf ever placed and a chat gets slower without bound — the poll cost
        scales with total leaves ever placed, not with what actually changed, so latency
        compounds turn over turn.

        `_seq` is injective (see `seq.py`), so a strict `>` is a correct cursor with no
        revision-group completion dance and no `OFFSET` — the same property `page_by_origin`
        relies on.
        The `_seq` predicate is applied before the row's `doc` is materialised, so an already-seen
        leaf costs an index entry and an integer compare instead of a JSON parse; that, not the
        walk, is where the time went.

        Not scoped by origin, unlike `page_by_origin`. That method answers "what have I authored
        that I still owe a peer" and origin-scoping is its correctness. This one answers "what is on
        the ground plane that I have not yet read", and a leaf reconciled in from a peer is exactly
        what a poll must not miss."""
        rows = self.db.read().execute(
            "SELECT _seq, doc FROM vertex WHERE ct = ? AND _seq > ? ORDER BY _seq LIMIT ?",
            (content_type, int(after_seq), int(limit))).fetchall()
        return [{"_seq": int(r["_seq"]), "doc": json.loads(r["doc"])} for r in rows]

    def count(self, *, state: Optional[str] = None) -> int:
        """Total vertices, or vertices in one `state`. Read off an incrementally
        maintained counter — this store issues no `count(*)` on any path."""
        name = ("state:" + state) if state else _schema.c_vertex_total()
        return _seq.counter_of(self.db, name)

    def delete_artifact(self, artifact_id: str, *, accounted: bool = True) -> None:
        with self.db.write() as cur:
            r = cur.execute("SELECT ct, _seq, _origin, _leaf, doc FROM vertex WHERE id = ?",
                            (artifact_id,)).fetchone()
            if r is None:
                cur.execute("DELETE FROM demand WHERE id = ?", (artifact_id,))   # no orphan demand
                return
            old = json.loads(r["doc"])
            cur.execute("DELETE FROM vertex WHERE id = ?", (artifact_id,))
            _seq.xor_leaf(cur, int(r["_leaf"]), K.row_hash(artifact_id, r["_seq"]))
            _seq.bump(cur, _schema.c_vertex_total(), -1)
            if accounted:
                # A real deletion, accounted in this observer's proper time.
                _seq.vacate(cur, r["_origin"])
            # else: eviction of a cached copy. The authoritative row still exists at its author, and
            # this node never authored it — dropping our copy is not a deletion in anyone's
            # proper-time ledger, so it does not vacate a (foreign) origin's sequence.
            for name, delta in self._counter_deltas(old, None, r["ct"], None).items():
                _seq.bump(cur, name, delta)
            self._sync_task(cur, artifact_id, old, None, r["ct"])
            self._index_lists(cur, artifact_id, None, r["ct"])   # retract the postings too
            cur.execute("DELETE FROM demand WHERE id = ?", (artifact_id,))   # clear any demand entry

    def evict_artifact(self, artifact_id: str) -> None:
        """Drop a cached (reached) copy — a non-accounted removal. The authoritative row survives in
        the substrate, so a later reach restores it. This is how a limited ember stays within its
        envelope. Never call it on a row this node authored (own rows carry no demand entry, so the
        eviction sweep never selects them)."""
        self.delete_artifact(artifact_id, accounted=False)

    # ── demand cache (local, never replicated) — raw storage; ember owns the decay ──
    def demand_set(self, artifact_id: str, mass: float, ts: float) -> None:
        with self.db.write() as cur:
            cur.execute("INSERT INTO demand(id, mass, ts) VALUES(?,?,?) "
                        "ON CONFLICT(id) DO UPDATE SET mass = excluded.mass, ts = excluded.ts",
                        (str(artifact_id), float(mass), float(ts)))

    def demand_get(self, artifact_id: str) -> Optional[Dict[str, float]]:
        r = self.db.read().execute("SELECT mass, ts FROM demand WHERE id = ?",
                                   (str(artifact_id),)).fetchone()
        return {"mass": float(r["mass"]), "ts": float(r["ts"])} if r else None

    def demand_page(self, *, after: str = "", limit: int = DEMAND_PAGE) -> List[Dict[str, Any]]:
        """One keyset page of demand rows (`id, mass, ts`). The cache is bounded by design, so the
        eviction sweep can afford to walk it and rank the coldest."""
        rows = self.db.read().execute(
            "SELECT id, mass, ts FROM demand WHERE id > ? ORDER BY id LIMIT ?",
            (after, int(limit))).fetchall()
        return [{"id": r["id"], "mass": float(r["mass"]), "ts": float(r["ts"])} for r in rows]

    def demand_count(self) -> int:
        """How many reached (evictable) rows the cache holds. Walked in keyset pages, NOT `count(*)`:
        `count(*)` is banned package-wide (it dereferences every row on the corpus), and the cache is
        bounded by design, so a page-walk is cheap and keeps the one rule with no exception."""
        n, after = 0, ""
        while True:
            rows = self.db.read().execute(
                "SELECT id FROM demand WHERE id > ? ORDER BY id LIMIT ?",
                (after, DEMAND_PAGE)).fetchall()
            if not rows:
                break
            n += len(rows)
            after = rows[-1]["id"]
            # "A short page is the last page" is a fact about the `LIMIT` that was asked for, so it
            # is compared against that same value and never against a second literal. Two copies
            # of the page size is the defect this shape removes: raise one and the walk stops a
            # page early (undercount), lower one and it re-reads the tail forever.
            if len(rows) < DEMAND_PAGE:
                break
        return n

    # ── keyset pagination — the only sanctioned paging primitive ─────────────
    def page_by_id(self, *, after: str = "", limit: int = 200,
                   content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """`WHERE id > :after ORDER BY id LIMIT :n`, walking the PK index.

        Cost is constant in the depth of the page. Drive the next call with
        `rows[-1]["id"]`; an empty list means the end, and it means it truthfully."""
        params: List[Any] = [after]
        clause = ""
        if content_type is not None:
            clause = " AND ct = ?"
            params.append(content_type)
        params.append(int(limit))
        rows = self.db.read().execute(
            "SELECT id, content_ref, _origin, _seq, doc FROM vertex "
            "WHERE id > ?" + clause + " ORDER BY id LIMIT ?", params).fetchall()
        return [{"id": r["id"], "content_ref": r["content_ref"],
                 "_origin": r["_origin"], "_seq": r["_seq"],
                 "doc": json.loads(r["doc"])} for r in rows]

    def page_by_origin(self, *, origin: Optional[str] = None, after_seq: int = 0,
                       limit: int = 200) -> List[Dict[str, Any]]:
        """The publish scan: `WHERE _origin = :me AND _seq > :cursor ORDER BY _seq`.

        An indexed range walk on `ix_v_origin`, with no `OFFSET` and no revision-group
        completion dance: `_seq` is injective (see `seq.py`), so a strict `>` cursor
        cannot skip a row and needs no tie-breaking machinery around it."""
        origin = origin or self.origin
        rows = self.db.read().execute(
            "SELECT id, content_ref, _origin, _seq, doc FROM vertex "
            "WHERE _origin = ? AND _seq > ? ORDER BY _seq LIMIT ?",
            (origin, int(after_seq), int(limit))).fetchall()
        return [{"id": r["id"], "content_ref": r["content_ref"],
                 "_origin": r["_origin"], "_seq": r["_seq"],
                 "doc": json.loads(r["doc"])} for r in rows]

    def count_after_id(self, cursor: str, *, cap: int = 100_000) -> Dict[str, Any]:
        """Rows with `id > cursor`, **bounded**. Returns `{"n": int, "exact": bool}`.

        Counting an arbitrary id range cannot be a counter lookup, and it must not
        become a `count(*)`. So it is bounded: walk the PK index up to `cap+1` rows
        and report `exact=False` when the cap is hit. A caller that needs an exact
        publish backlog wants `pending_publish()`, which counts rows per feed.

        The two-field return is deliberate — a bare integer here is indistinguishable
        from a truncated one, and "returns a plausible wrong number" is the exact
        failure class contract §5 exists to record."""
        cap = max(0, int(cap))
        rows = self.db.read().execute(
            "SELECT 1 FROM vertex WHERE id > ? ORDER BY id LIMIT ?",
            (cursor, cap + 1)).fetchall()
        n = len(rows)
        return {"n": min(n, cap), "exact": n <= cap}

    def pending_publish(self, *, vertex_cursor: int = 0, edge_cursor: int = 0,
                        origin: Optional[str] = None, graph: Any = None,
                        cap: int = 100_000) -> Dict[str, Any]:
        """The publish backlog. Rows this observer authored above each table's own
        cursor, counted per table and summed.

        Returns `{"vertex", "edge", "total", "exact"}`.

        A `high_water - cursor` backlog (`last_seq - cursor`) is wrong two separate ways:

        1. **Vacated seqs.** `last_seq - cursor` counts allocations. An update
           allocates a new seq and vacates the old one, so the arithmetic counts
           work with no row behind it — unboundedly, under operator-rewrite
           churn. Counting rows is immune by construction.
        2. **Two feeds, two cursors.** A vertex scan cannot consume an edge's seq,
           so one counter shared across both tables means neither cursor alone
           ranges over the union. Measuring against either counts the other table's
           rows as outstanding forever, giving the backlog a floor it can never
           cross and making `converged` structurally unreachable. `min(vc, ec)` does
           not fix this — it relocates the floor to whichever table stopped lower.

        Each term is an indexed range on `ix_v_origin` / `ix_e_origin`, bounded by
        `cap`, and reaches 0 exactly when its own feed drains — so the sum is 0
        exactly when the node has published everything. That is the property a
        convergence signal needs.

        `graph` is the `LatticeGraphStore` sharing this connection; omit it and the
        edge term is counted directly off the shared connection."""
        origin = origin or self.origin
        v = self.count_after_seq(origin=origin, after_seq=vertex_cursor, cap=cap)
        rows = self.db.read().execute(
            "SELECT 1 FROM edge WHERE _origin = ? AND _seq > ? ORDER BY _seq LIMIT ?",
            (origin, int(edge_cursor), int(cap) + 1)).fetchall()
        e_n = len(rows)
        e = {"n": min(e_n, cap), "exact": e_n <= cap}
        return {"vertex": v["n"], "edge": e["n"], "total": v["n"] + e["n"],
                "exact": bool(v["exact"] and e["exact"])}

    def count_after_seq(self, *, origin: str, after_seq: int,
                        cap: int = 100_000) -> Dict[str, Any]:
        """Bounded count over one origin's proper-time range. Same contract as
        `count_after_id`. Use for foreign origins, where holes make subtraction
        unsafe."""
        cap = max(0, int(cap))
        rows = self.db.read().execute(
            "SELECT 1 FROM vertex WHERE _origin = ? AND _seq > ? ORDER BY _seq LIMIT ?",
            (origin, int(after_seq), cap + 1)).fetchall()
        n = len(rows)
        return {"n": min(n, cap), "exact": n <= cap}

    # ── typed counts ─────────────────────────────────────────────────────────
    def count_in_collection(self, collection_id: str, *, committed_only: bool = False) -> int:
        """Members of one collection. Counter lookup, O(1).

        `committed_only` exists because two kinds of caller need different things:
        `advance_curriculum` uses the count as a resume offset and must count every
        state — filtering to committed there under-counts, hands the ingester a low
        offset, and re-ingests records it already has."""
        return _seq.counter_of(
            self.db, _schema.c_collection(collection_id, committed_only=committed_only))

    def count_by_content_type(self, content_type: str) -> int:
        return _seq.counter_of(self.db, _schema.c_ct(content_type))

    def page_by_origin_root(self, origin_root: str, *, after: str = "",
                            limit: int = 5000) -> List[str]:
        """One keyset page of a collection's member ids by its indexed containment root
        (`origin_root == collection_id`, flat today) over `ix_v_root` — never a scan. A generic
        enumerator; the store keeps no opinion about what a collection means. The mesh uses it to
        enumerate the members of grant-gated collections (which never leave the node); the access
        decision itself lives in `ember.access`, computed from grants."""
        rows = self.db.read().execute(
            "SELECT id FROM vertex WHERE origin_root = ? AND id > ? ORDER BY id LIMIT ?",
            (str(origin_root), after, int(limit))).fetchall()
        return [r["id"] for r in rows]

    def count_missing_field(self, field: Any, *, committed_only: bool = False) -> int:
        """Vertices whose audited provenance field is missing. Counter lookup, O(1).

        A capped keyset walk becomes unmeasurably slow at 6.24M rows, which would leave the
        provenance audit unavailable exactly at production scale. `field` comes from the closed
        `NullAuditField` set and is a value, never interpolated into SQL.

        Returns a bare `int`. `genesis._scan_missing_field` probes this as `int(typed(field))`,
        so the capped `{"n":…, "exact":…}` dict used by `count_after_id` would raise `TypeError`
        there and be swallowed into `None`. An int is safe here because the value is exact: it is
        an incrementally maintained counter, never a truncated scan, so there is no cap to hide.

        `committed_only=True` is exact and is maintained alongside, resting on the `state`
        key surviving in `doc` JSON. It is not the default, so that the audit's answer stays
        comparable with every other backend's fallback."""
        f = K.NullAuditField.coerce(field)
        return _seq.counter_of(
            self.db, _schema.c_missing(f.value, committed_only=committed_only))

    # ── content-type fetch (stats.py:515) ────────────────────────────────────
    def list_by_content_type(self, content_type: str, *, cap: int = 2000,
                             include_archived: bool = False
                             ) -> Tuple[List[Dict[str, Any]], bool]:
        """Returns `(docs, exhaustive)`. Probes with `LIMIT cap+1` and reports
        `exhaustive=False` if a full page came back.

        Returning the exhaustive flag alongside the rows makes "I did not finish"
        unrepresentable as "there is nothing" — a truncated page and a genuinely empty
        result are opposite claims, and only the flag tells them apart."""
        cap = max(0, int(cap))
        where = "ct = ?"
        params: List[Any] = [content_type]
        if not include_archived:
            # `IS NOT` is null-safe — a row with no `state` is kept: `K.STATE_WHEN_ABSENT`
            # ("committed") as a predicate.
            where += " AND json_extract(doc, '$.state') IS NOT 'archived'"
        params.append(cap + 1)
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE " + where + " ORDER BY id LIMIT ?", params).fetchall()
        docs = [json.loads(r["doc"]) for r in rows]
        return (docs[:cap], len(docs) <= cap)

    def content_type_mark(self, content_type: str, *, cap: int = 2000
                          ) -> Tuple[int, int, bool]:
        """`(rows, max _seq, exhaustive)` over one content type — the freshness stamp of a
        derivation built from a whole content type, the set-sized sibling of `version_of`.

        `_offers` in ember is derived from all 48 `…operator+json` artifacts and costs 215–270 ms to
        rebuild; the whole-store `write_mark` would drop it whenever anything at all was written,
        which on a live node means every chat message. This asks the narrower question: has
        anything of this type been written since we read it?

        Both terms are needed. `max(_seq)` alone misses the deletion of any row that was not the
        newest; `rows` alone misses an update, which reallocates `_seq` in place. Both are read
        from one `SELECT`, so they cannot disagree about which rows they mean."""
        cap = max(0, int(cap))
        rows = self.db.read().execute(
            "SELECT _seq FROM vertex WHERE ct = ? ORDER BY id LIMIT ?",
            (content_type, cap + 1)).fetchall()
        n = len(rows)
        hi = 0
        for r in rows:
            v = r["_seq"]
            if isinstance(v, int) and v > hi:
                hi = v
        return (min(n, cap), hi, n <= cap)

    def list_by_leaf(self, leaf: int, *,
                     cap: int = 200_000) -> Tuple[List[Dict[str, Any]], bool]:
        """Returns `(docs, exhaustive)` for every row whose `_leaf == leaf`.

        The read side of the Merkle tree. `publish_merkle`'s incremental path needs a
        changed leaf's rows to rebuild that leaf's authoritative object, and
        `ix_v_leaf(_leaf)` makes it an indexed equality lookup over ~corpus/leaves rows
        instead of the full-corpus `_scan_rows`.

        `(docs, exhaustive)` for the same reason as `list_by_content_type`: a truncated
        page must not be indistinguishable from a genuinely empty leaf — that conflation
        is what makes a peer diff a leaf whose object 404s and stay permanently divergent.
        The caller still filters on `_is_replicated`: on lattice every row (operational
        included) carries `_leaf`, so a leaf is not a replication filter by itself, and
        the mesh subtracts operational rows explicitly — see `_refresh_leaves_lattice`."""
        cap = max(0, int(cap))
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE _leaf = ? ORDER BY id LIMIT ?",
            (int(leaf), cap + 1)).fetchall()
        docs = [json.loads(r["doc"]) for r in rows]
        return (docs[:cap], len(docs) <= cap)

    def list_by_doc_field(self, *, content_type: str, field: str, value: Any,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Docs of one content type whose JSON `field` equals `value`.

        Scoped by `ct` first, which is indexed, so the JSON predicate only ever
        evaluates over one content-type bucket. `field` is interpolated into a JSON
        path, so it must be a literal from the caller, never user input — it is
        validated as an identifier here rather than trusted."""
        if not field.replace("_", "").isalnum():
            raise ValueError("list_by_doc_field: field %r is not a plain identifier" % (field,))
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE ct = ? AND json_extract(doc, '$.%s') = ? "
            "ORDER BY id LIMIT ?" % field, (content_type, value, int(limit))).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    # ── the keyed arm: lemma / list-field lookup ─────────────────────────────
    def list_index_built(self) -> bool:
        """Does `listkey` cover every vertex in this store? See `schema.c_list_index_built`."""
        return _seq.counter_of(self.db, _schema.c_list_index_built()) == 1

    def _require_list_index(self, field: str) -> None:
        if field not in self.LIST_INDEX_FIELDS:
            raise ValueError(
                "lookup_by_list_field: %r is not an indexed list field (indexed: %s). Returning "
                "[] here would report 'no matches' for a field that was never indexed, which is "
                "a wrong answer, not an empty one. Add it to LatticeArtifactStore."
                "LIST_INDEX_FIELDS and rebuild_list_index()." % (field, ", ".join(self.LIST_INDEX_FIELDS)))
        if not self.list_index_built():
            raise ListIndexUnbuilt(
                "the `listkey` index has not been built for this store, so a keyed lookup "
                "cannot be answered. This store predates the index (it was migrated before "
                "`listkey` existed). Run `LatticeArtifactStore.rebuild_list_index()` once — it "
                "is a one-time backfill and a WRITE, so it needs the store writable. Refusing "
                "to return [] : that would claim this store does not contain the word, which is "
                "a different and false statement.")

    def lookup_by_lemma(self, word: str, *, limit: int = 12,
                        content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """The dictionary path: a word -> the artifacts that carry it as a lemma.

        `content_type` is the type discriminator and callers asking a typed question should
        pass it — `lexical.define` passes `text/x-wordnet`, `code` passes the symbol type. See
        `lookup_by_list_field` for why filtering afterwards is not equivalent."""
        return self.lookup_by_list_field("lemmas", word, limit=limit, content_type=content_type)

    def lookup_by_list_field(self, field: str, value: Any, *, limit: int = 20,
                             content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Artifacts whose indexed list `field` contains `value`, optionally of one content type.

        `content_type=None` keeps the undiscriminated lookup expressible — the browse UI wants
        every type — and it is served by the same index's `(field, value)` prefix. It is a
        deliberate choice by the caller rather than the only available behaviour.

        Ordered by `aid`, so a `LIMIT`ed result is the same page on every run."""
        self._require_list_index(field)
        params: List[Any] = [field, str(value).lower()]
        clause = ""
        if content_type is not None:
            clause = " AND l.ct = ?"
            params.append(str(content_type))
        n = int(limit)
        params.append(n + 1)
        rows = self.db.read().execute(
            "SELECT v.doc FROM listkey l JOIN vertex v ON v.id = l.aid "
            "WHERE l.field = ? AND l.value = ?" + clause +
            " ORDER BY l.aid LIMIT ?", params).fetchall()
        truncated = len(rows) > n
        return Page([json.loads(r["doc"]) for r in rows[:n]], truncated=truncated, limit=n)

    def backfill_root_id(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Give every row without one its version-lineage handle. Idempotent.

        The fill is Mantle's rule (`entities/artifact.py:73`): a doc that names a `root_id` keeps
        it; one that does not is its own first version, so `root_id = id`. That is correct for an
        initial load, where each artifact has precisely one version.
        """
        con = self.db.read()
        cols = {r[1] for r in con.execute("PRAGMA table_info(vertex)")}
        if "root_id" not in cols:
            # Refusal, not a quiet zero: the caller asked for a lineage the store cannot hold.
            return {"ok": False, "filled": 0, "complete": False,
                    "reason": "vertex has no root_id column — run ensure_schema() first"}

        def _pending() -> bool:
            return bool(self.db.read().execute(
                "SELECT EXISTS(SELECT 1 FROM vertex WHERE root_id IS NULL)").fetchone()[0])

        if dry_run:
            return {"ok": True, "filled": 0, "pending": _pending(), "dry_run": True,
                    "complete": not _pending()}
        filled = 0
        if _pending():
            with self.db.write() as cur:
                c = cur.execute(
                    "UPDATE vertex SET root_id = COALESCE(json_extract(doc, '$.root_id'), id) "
                    "WHERE root_id IS NULL")
                filled = c.rowcount if c.rowcount is not None and c.rowcount >= 0 else 0
        remaining = _pending()
        return {"ok": not remaining, "filled": filled, "complete": not remaining}

    def revise(self, artifact_id: str, changes: Dict[str, Any], *,
               author: Optional[str] = None) -> Dict[str, Any]:
        """Author a new version under the same lineage. A convenience, not a required route.

        The shape matches `entities/artifact.py` and Ember's `LocalCache.revise`:
          · the new version is a new row with its own `id`, sharing `root_id`;
          · the prior head is archived, never deleted — it stays queryable forever;
          · `version_id` is the committed head ("version_id is latest").

        `state` here means lineage position (draft/committed/archived) and is the method's own
        output — a revision is by definition the new committed head. The same field also carries
        grant lifecycle, two orthogonal axes on one name, so a caller passing `state` in `changes`
        is refused rather than silently overridden.

        """
        if "state" in changes:
            raise ValueError(
                "revise() does not accept `state` (%r): it is RESERVED. `state` is lineage "
                "position and revise() sets it — the new version is the committed head. Accepting "
                "a caller-supplied `state` here would let a revision to 'revoked' be silently "
                "overwritten by the forced 'committed', reinstating access that had been revoked. "
                "Set lifecycle on a field of your own, or archive via "
                "put_artifact()." % (changes["state"],))
        prev = self.get_artifact(artifact_id)
        if prev is None:
            raise KeyError("cannot revise %r: no such artifact" % artifact_id)
        if prev.get("state") == "archived":
            root_hint = prev.get("root_id") or artifact_id
            raise ValueError(
                "cannot revise %r: it is ARCHIVED (superseded by %r). Revising a superseded "
                "version forks the lineage into two live heads on one root_id, and nothing "
                "downstream can tell which is current. Resolve the head first: "
                "head_of(%r)." % (artifact_id, prev.get("superseded_by"), root_hint))
        root = prev.get("root_id") or self.db.read().execute(
            "SELECT root_id FROM vertex WHERE id = ?", (artifact_id,)).fetchone()[0] or artifact_id

        doc = dict(prev)
        doc.update(changes)
        doc["root_id"] = root
        doc["state"] = "committed"
        if author:
            doc["created_by"] = author
        canon = {k: v for k, v in doc.items()
                 if k != "id" and not (k.startswith("_") and k in ("_origin", "_seq", "_leaf", "_rev"))}
        body = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
        vid = "%s~%s" % (root, hashlib.blake2b(body, digest_size=8).hexdigest())
        doc.pop("id", None)
        doc["id"] = vid
        if vid == artifact_id:
            return prev                      # identical content — the revision is a no-op

        # Order matters: write the new version before archiving the old. A crash between them
        # leaves two committed versions (visible, reconcilable) rather than none (an artifact
        # that momentarily does not exist for readers resolving the head).
        self.put_artifact(doc)
        old = dict(prev)
        old["state"] = "archived"
        old["superseded_by"] = vid
        self.put_artifact(old)
        return doc

    def head_of(self, root_id: str) -> Optional[Dict[str, Any]]:
        """The current version of a lineage ("version_id is latest")."""
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE root_id = ? ORDER BY _origin, _seq", (root_id,)
        ).fetchall()
        live = [json.loads(r["doc"]) for r in rows]
        live = [d for d in live if K.state_of(d) != "archived"]
        return live[-1] if live else None

    def versions_of(self, root_id: str) -> List[Dict[str, Any]]:
        """Every version in a lineage, oldest first. The `y` colimit — each version is another
        order, the anti-entropy projection.

        Ordered by `(_origin, _seq)` — the globally-unique version identity, gap-free per origin —
        rather than by `created_time`, which is a claim by the writer.
        """
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE root_id = ? ORDER BY _origin, _seq", (root_id,)
        ).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    def rebuild_list_index(self, *, chunk: int = 5000) -> Dict[str, Any]:
        """One-time backfill of `listkey` for a store without it. Returns what it did.

        The only sanctioned way to earn the build marker on a populated store, and a write: a
        store opened read-only cannot be repaired into answering, and refusing is the right
        outcome there.

        Walks by keyset (`WHERE id > :cur ORDER BY id`), never `OFFSET`, and commits per chunk so
        a 6M-row corpus does not hold one transaction open for the whole backfill. The marker is
        set only after the final chunk, so an interrupted rebuild leaves the store uncertified
        and still refusing — a half-built index answers wrongly, and plausibly."""
        cursor, indexed, posted = "", 0, 0
        while True:
            rows = self.db.read().execute(
                "SELECT id, ct, doc FROM vertex WHERE id > ? ORDER BY id LIMIT ?",
                (cursor, int(chunk))).fetchall()
            if not rows:
                break
            with self.db.write() as cur:
                for r in rows:
                    posted += self._index_lists(cur, r["id"], json.loads(r["doc"]), r["ct"])
                    indexed += 1
                cursor = rows[-1]["id"]
        with self.db.write() as cur:
            cur.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?, 1)",
                        (_schema.c_list_index_built(),))
        return {"vertices": indexed, "postings": posted,
                "fields": list(self.LIST_INDEX_FIELDS), "built": True}

    # ── work pool — typed replacements for pool.py's raw SQL ─────────────────
    def pending_window(self, content_type: str, *, limit: int = 24,
                       now_iso: Optional[str] = None,
                       operator: Optional[str] = None) -> List[Dict[str, Any]]:
        """Head of the pending queue for `content_type`, retry-backoff already applied.

        `ORDER BY priority DESC, id` is free here — `ix_t_pending` is
        `(ct, status, priority DESC, id)`, so this is a bounded index walk with no
        sort, rather than materialising and sorting the whole task history per
        worker per poll. Callers still shuffle this window to avoid claim
        contention; that is a herd-avoidance measure, not an ordering opinion.

        `operator` narrows the window to ONE operator's work, and exists because a content
        type MAY be a shared pool: `application/vnd.agience.task+json` is every operator's
        queue, so a drain that implements exactly one of them would otherwise spend its whole
        window claiming, and then having to put back, work it cannot do. Expressed here rather
        than in the loop so the narrowing is a property of what the drain asks for, not of what
        it does with the answer — and so `try_claim` is never reached for a row this caller has
        no business claiming.

        A caller whose content type is its own still passes it, and should: it costs one column
        comparison, and it is what keeps the query correct if a second operator is ever written
        under that type. `services/mirror_drain.py` is that caller.

        It is a filter over the same `(ct, status)` bucket walk, not a new access path:
        `operator` is a column on the row, so the cost stays bounded by `limit` plus
        whatever other operators' rows are interleaved in the pending bucket. There is no
        `(ct, operator)` index and none is added — the pending bucket is the live queue,
        which is small by construction on any node that is draining it."""
        params: List[Any] = [content_type]
        clause = ""
        if operator is not None:
            clause += " AND operator = ?"
            params.append(operator)
        if now_iso is not None:
            clause += " AND (next_retry_at IS NULL OR next_retry_at <= ?)"
            params.append(now_iso)
        params.append(int(limit))
        rows = self.db.read().execute(
            "SELECT id, priority, next_retry_at FROM task "
            "WHERE ct = ? AND status = 'pending'" + clause +
            " ORDER BY priority DESC, id LIMIT ?", params).fetchall()
        return [dict(r) for r in rows]

    def try_claim(self, task_id: str, *, worker_id: str, now_iso: str) -> bool:
        """Atomically move one task pending -> claimed. `True` iff this caller won.

        The `WHERE status = 'pending'` predicate plus SQLite's single-writer lock
        makes this a genuine compare-and-set: concurrent workers see `rowcount == 0`
        and move on. It never returns a task the caller did not win, so no defensive
        `get_artifact(...)['claimed_by'] == me` re-read is needed at the call site."""
        with self.db.write() as cur:
            cur.execute(
                "UPDATE task SET status='claimed', claimed_by=?, claimed_at=? "
                "WHERE id = ? AND status = 'pending'", (worker_id, now_iso, task_id))
            if cur.rowcount < 1:
                return False
            self._patch_doc(cur, task_id,
                            {"status": "claimed", "claimed_by": worker_id,
                             "claimed_at": now_iso})
            row = cur.execute("SELECT ct FROM task WHERE id = ?", (task_id,)).fetchone()
            if row is not None:
                _seq.bump(cur, _schema.c_task_status(row["ct"], "pending"), -1)
                _seq.bump(cur, _schema.c_task_status(row["ct"], "claimed"), 1)
            return True

    def renew_lease(self, task_id: str, *, now_iso: str) -> bool:
        """Heartbeat a claimed task's lease. `True` iff the task was still claimed.

        A `False` here is information the caller needs: the lease was already
        reclaimed and it is working on a task someone else now owns."""
        with self.db.write() as cur:
            cur.execute("UPDATE task SET claimed_at = ? WHERE id = ? AND status = 'claimed'",
                        (now_iso, task_id))
            if cur.rowcount < 1:
                return False
            self._patch_doc(cur, task_id, {"claimed_at": now_iso})
            return True

    def claimed(self, content_type: str, *, limit: int = 1000) -> List[Dict[str, Any]]:
        """Every currently-claimed task with its lease timestamp — the input to
        stale-claim reclamation. Index seek over the claimed bucket alone."""
        rows = self.db.read().execute(
            "SELECT id, claimed_by, claimed_at, operator, task_key FROM task "
            "WHERE ct = ? AND status = 'claimed' ORDER BY id LIMIT ?",
            (content_type, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def active_claims(self, content_type: str, *, limit: int = 1000) -> List[Dict[str, Any]]:
        """The /status fleet panel: who is working on what, ordered by worker.
        Ordering happens in SQL over the small claimed set."""
        rows = self.db.read().execute(
            "SELECT id, claimed_by, operator, task_key, claimed_at FROM task "
            "WHERE ct = ? AND status = 'claimed' ORDER BY claimed_by, id LIMIT ?",
            (content_type, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def recent_terminal(self, content_type: str, *, limit: int = 8,
                        statuses: Iterable[str] = ("done", "failed")) -> List[Dict[str, Any]]:
        """Most recently completed/failed tasks.

        `ix_t_terminal` is `(ct, status, completed_at DESC)`, so each status bucket
        yields its top-N as an index walk — no sort over the whole never-pruned
        done+failed set, which grows without bound while /status displays 8 rows."""
        out: List[Dict[str, Any]] = []
        cur = self.db.read()
        for st in statuses:
            rows = cur.execute(
                "SELECT id, operator, task_key, status, completed_at FROM task "
                "WHERE ct = ? AND status = ? ORDER BY completed_at DESC LIMIT ?",
                (content_type, st, int(limit))).fetchall()
            out += [dict(r) for r in rows]
        out.sort(key=lambda r: str(r.get("completed_at") or ""), reverse=True)
        return out[:int(limit)]

    def release(self, task_id: str, *, to_status: str = "pending",
                next_retry_at: Optional[str] = None) -> bool:
        """Return a task to `to_status` and clear its claim. `True` iff it was
        claimed. Used by stale-claim reclamation and by retry-with-backoff."""
        with self.db.write() as cur:
            row = cur.execute("SELECT ct FROM task WHERE id = ? AND status = 'claimed'",
                              (task_id,)).fetchone()
            if row is None:
                return False
            cur.execute(
                "UPDATE task SET status=?, claimed_by=NULL, next_retry_at=? "
                "WHERE id = ? AND status = 'claimed'", (to_status, next_retry_at, task_id))
            if cur.rowcount < 1:
                return False
            self._patch_doc(cur, task_id, {"status": to_status, "claimed_by": None,
                                           "next_retry_at": next_retry_at})
            _seq.bump(cur, _schema.c_task_status(row["ct"], "claimed"), -1)
            _seq.bump(cur, _schema.c_task_status(row["ct"], to_status), 1)
            return True

    def settle(self, task_id: str, *, worker_id: str, to_status: str,
               next_retry_at: Optional[str] = None,
               completed_at: Optional[str] = None,
               fields: Optional[Dict[str, Any]] = None) -> bool:
        """Move a task OUT of THIS worker's claim, or report that the claim is gone.

        `release` is the same transition owned by a THIRD party — stale-claim reclamation,
        which by definition acts on a lease its caller does not hold, so it compares only on
        `status = 'claimed'`. This one is the claimer's own exit, and it compares on
        `claimed_by` as well. That difference is the whole point:

          * a worker whose lease was reclaimed while it worked writes NOTHING, and learns it
            (`False`) instead of overwriting the row the new owner is now working on;
          * a task rewritten back to `pending` underneath a claim — which is exactly what a
            fresh failed upload for the same `content_key` does — cannot be silently stamped
            `done` by the in-flight attempt for the superseded bytes. The refresh wins, the
            claim notices.

        `fields` are doc-only (`attempts`, `last_error`, `result`): facts about the attempt
        that no column projects. They are written in the same transaction as the status, so a
        reader can never see a terminal status without the reason it is terminal.

        `completed_at` is passed rather than inferred from `to_status`, because it is the sort
        key `ix_t_terminal` (and `recent_terminal`) walks — a terminal row without one is
        invisible in the window an operator actually reads.
        """
        with self.db.write() as cur:
            row = cur.execute(
                "SELECT ct FROM task WHERE id = ? AND status = 'claimed' AND claimed_by = ?",
                (task_id, worker_id)).fetchone()
            if row is None:
                return False
            cur.execute(
                "UPDATE task SET status=?, claimed_by=NULL, next_retry_at=?, completed_at=? "
                "WHERE id = ? AND status = 'claimed' AND claimed_by = ?",
                (to_status, next_retry_at, completed_at, task_id, worker_id))
            if cur.rowcount < 1:
                return False
            patch: Dict[str, Any] = dict(fields or {})
            patch.update({"status": to_status, "claimed_by": None,
                          "next_retry_at": next_retry_at, "completed_at": completed_at})
            self._patch_doc(cur, task_id, patch)
            _seq.bump(cur, _schema.c_task_status(row["ct"], "claimed"), -1)
            _seq.bump(cur, _schema.c_task_status(row["ct"], to_status), 1)
            return True

    def count_by_status(self, content_type: str,
                        statuses: Iterable[str] = ("pending", "claimed", "done",
                                                   "failed", "dead")) -> Dict[str, int]:
        """Queue depth per status. One counter lookup per status, no scan, no `count(*)`."""
        return {st: _seq.counter_of(self.db, _schema.c_task_status(content_type, st))
                for st in statuses}

    def _patch_doc(self, cur: sqlite3.Cursor, aid: str, fields: Dict[str, Any]) -> None:
        """Keep `doc` JSON in step with a sidecar-only UPDATE.

        The doc is the artifact of record; the `task` table is a projection. They
        are updated in one transaction, so they cannot disagree — column/doc
        divergence here is what orphans a claimed task indefinitely.

        No new `_seq` is allocated: a lease renewal is bookkeeping, not an authored
        event, and stamping one would churn the merkle leaf and republish the row on
        every heartbeat. A claim does change the artifact, so a caller that wants the
        claim replicated follows with an explicit `put_artifact(..., stamp_rev=True)`."""
        r = cur.execute("SELECT ct, doc FROM vertex WHERE id = ?", (aid,)).fetchone()
        if r is None:
            return
        d = json.loads(r["doc"])
        d.update(fields)
        cur.execute("UPDATE vertex SET doc = ? WHERE id = ?", (json.dumps(d), aid))
        if any(k in self.LIST_INDEX_FIELDS for k in fields):
            self._index_lists(cur, aid, d, r["ct"])

    # ── merkle ───────────────────────────────────────────────────────────────
    def merkle_leaves(self) -> List[int]:
        """The incrementally maintained leaf digests, read straight off `leaf_digest`.
        Never a rescan — `_scan_rows` is the rebuild path and is separate."""
        acc = [0] * self.leaves
        for r in self.db.read().execute("SELECT leaf, digest FROM leaf_digest"):
            leaf = int(r["leaf"])
            if 0 <= leaf < self.leaves:
                acc[leaf] = int.from_bytes(r["digest"], "big")
        return acc

    def verify_counters(self, *, repair: bool = False) -> Dict[str, Any]:
        """Recompute every vertex counter from the rows and report drift.

        Verification and bootstrap only — this is a full scan. Run it from
        `node-repair.py`, never on a request path. The counters are load-bearing:
        `count_missing_field` is the provenance audit, and a drifted counter reports a
        clean audit over a dirty corpus. A number nothing can check is not a measurement.

        Also the fill path: counters are maintained from the moment a row is written,
        so a store holding rows older than a counter (a newly audited field, a file
        restored from before that table) reads 0 until this runs.

        Returns `{"scanned", "drift": {name: (stored, actual)}, "repaired": bool}`.
        An empty `drift` is the assertion worth making in a health check."""
        actual: Dict[str, int] = {}

        def add(name: str, delta: int) -> None:
            actual[name] = actual.get(name, 0) + delta

        scanned = 0
        with self.db.write() as cur:
            for r in cur.execute("SELECT ct, _origin, doc FROM vertex"):
                scanned += 1
                doc = json.loads(r["doc"])
                add(_schema.c_vertex_total(), 1)
                if r["_origin"]:
                    add(_schema.c_rows(str(r["_origin"])), 1)
                for name, delta in self._counter_deltas(None, doc, None, r["ct"]).items():
                    add(name, delta)
            # `rows:<origin>` spans both tables, so the edge side must be counted
            # too or every store with edges reports permanent phantom drift.
            for r in cur.execute("SELECT _origin, label FROM edge"):
                if r["_origin"]:
                    add(_schema.c_rows(str(r["_origin"])), 1)
                add(_schema.c_edge_total(), 1)
                # The per-relation extent is derivable from the same row, so it is judged here
                # too. `edge:label:*` is load-bearing — `_relation_signature` sizes its sample
                # from it — and a load-bearing counter nothing can check is not a measurement.
                add(_schema.c_edge_label(str(r["label"])), 1)
            stored = {row["name"]: int(row["n"])
                      for row in cur.execute("SELECT name, n FROM counter")}
            # Not derivable from the rows, so not judged here:
            #   task:*      maintained from the `task` sidecar
            #   conflict:*  records decisions, not rows
            #   vacated:*   historical — a vacated seq has no row left to count, so it
            #               can only be accumulated forward. That is why
            #               `seq_accounting(scan=True)` is the check that can see a row
            #               lost outside the write path: it compares the scanned live
            #               count against `last_seq - vacated`, which no recomputation
            #               of `vacated` could do.
            #   listkey:*   a state marker, not a count. `listkey:built` is a boolean
            #               ("this store's keyed index covers every row") living in the
            #               counter table because that is where this store keeps durable
            #               scalars. Recomputing it from the rows yields 0 — there is no
            #               row-derived quantity it corresponds to — so judging it would
            #               report permanent drift on every healthy store, and
            #               `repair=True` would write that 0 back, un-certifying a
            #               correctly built index and turning every keyed lookup into a
            #               refusal.
            #   edgelabel:  a state marker, not a count — `edgelabel:built` says the
            #               `edge:label:*` counters cover every edge row. Same argument as
            #               `listkey:`. (The extents themselves carry the `edge:label:`
            #               prefix and are judged, three lines up.)
            drift = {}
            for name in set(stored) | set(actual):
                if name.startswith(("task:", "conflict:", "vacated:", "listkey:", "edgelabel:")):
                    continue
                s, a = stored.get(name, 0), actual.get(name, 0)
                if s != a:
                    drift[name] = (s, a)
            if repair and drift:
                for name, (_s, a) in drift.items():
                    cur.execute(
                        "INSERT INTO counter(name, n) VALUES(?,?) "
                        "ON CONFLICT(name) DO UPDATE SET n = excluded.n", (name, a))
            if repair:
                # This scan visited every edge row inside one write transaction, so the
                # `edge:label:*` values now standing are exact — which is the claim the marker
                # makes. Certifying here is a measurement rather than an assumption, and it
                # means a store repaired by node-repair needs no separate backfill.
                # Only under `repair`: a read-only audit must not write a certificate.
                cur.execute("INSERT OR REPLACE INTO counter(name, n) VALUES(?, 1)",
                            (_schema.c_edge_label_built(),))
        out = {"scanned": scanned, "drift": drift, "repaired": bool(repair and drift)}
        # The allocation-accounting invariant, verified against the rows (scan=True)
        # rather than the counters — a counter-vs-counter comparison could never
        # detect a row that vanished outside the write path.
        out["seq_accounting"] = self.seq_accounting(scan=True)
        return out

    def seq_accounting(self, *, origin: Optional[str] = None,
                       scan: bool = False) -> Dict[str, Any]:
        """`live_rows + vacated == last_seq` for one origin. See `seq.seq_accounting`.

        The accessor `node-repair.py` calls: it returns every term plus a `balanced`
        verdict, so no caller has to reconstruct the arithmetic. `last_seq` holds the
        last issued seq, not the next — an easy off-by-one to write by hand.

        Use `scan=True` in node-repair: it recomputes `live_rows` from the rows and
        is the only mode that can see a row lost outside the write path."""
        return _seq.seq_accounting(self.db, origin or self.origin, scan=scan)

    def rebuild_merkle(self) -> int:
        """Full rescan into `leaf_digest`. Bootstrap and verification only.

        A rescan publish is far slower than incremental maintenance, so on a node that is
        still catching up the corpus can grow faster than the rescan, and a rescan-built
        tree can be stale before it finishes. Steady state is the incremental path in
        `_write_row`."""
        acc: Dict[int, int] = {}
        with self.db.write() as cur:
            cur.execute("DELETE FROM leaf_digest")
            n = 0
            for r in cur.execute("SELECT id, _leaf, _seq FROM vertex"):
                leaf = int(r["_leaf"]) if r["_leaf"] is not None else K.leaf_of(r["id"], self.leaves)
                acc[leaf] = acc.get(leaf, 0) ^ K.row_hash(r["id"], r["_seq"])
                n += 1
            # One tree covers both tables, so a rebuild XORs the edges back in too — otherwise it
            # would drop every edge and the "rescan == incremental" invariant would fail the
            # moment any edge exists. Edge identity is node-invariant (edge_hash excludes
            # _origin/_seq), the same value `edge._add_one` XORs in on write.
            for r in cur.execute("SELECT edge_key, _leaf, force, propagate, is_origin, order_key, "
                                 "props FROM edge"):
                key = r["edge_key"]
                leaf = int(r["_leaf"]) if r["_leaf"] is not None else K.leaf_of(key.hex(), self.leaves)
                try:
                    blob = json.loads(r["props"]) if r["props"] else {}
                except Exception:
                    blob = {}
                content = {"force": r["force"], "propagate": r["propagate"],
                           "is_origin": r["is_origin"], "order_key": r["order_key"], **blob}
                acc[leaf] = acc.get(leaf, 0) ^ K.edge_hash(key, content)
                n += 1
            for leaf, dig in acc.items():
                cur.execute("INSERT INTO leaf_digest(leaf, digest) VALUES(?,?)",
                            (leaf, (dig & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "big")))
        return n

    # ── derived resolution: reshard to natural_leaves(corpus) ─────────────────
    def maybe_reshard(self, *, graph: Any = None) -> Dict[str, Any]:
        """Reshard only when the corpus has crossed a sqrt boundary since the last one. Cheap to
        call every publish cycle — a counter read and a compare — so only the rare boundary
        crossing pays the O(N) re-stamp. This keeps the operating leaf count equal to the derived
        `natural_leaves(corpus)` as the store grows, with no background job and no constant."""
        n = self.count() + (graph.count_edges() if graph is not None else 0)
        if K.natural_leaves(n) == self.leaves:
            return {"resharded": False, "leaves": self.leaves}
        return self.reshard(graph=graph)

    def reshard(self, *, graph: Any = None, target: Optional[int] = None) -> Dict[str, Any]:
        """Re-stamp `_leaf` on every row and rebuild the tree at the derived resolution
        `natural_leaves(corpus)` — the sqrt-law count where publishing the tree and pulling a leaf
        cost the same. O(N), so it runs rarely (only on a power-of-two sqrt boundary). Idempotent: a
        no-op when already at target.

        Both tables move in lockstep — they share one `leaf_digest`, so a vertex and an edge must
        never disagree on the modulus. The `graph` store's `.leaves` is updated too when passed, and
        the recorded `merkle.leaves` meta is set so the next `open_lattice` resolves the same value."""
        n = self.count() + (graph.count_edges() if graph is not None else 0)
        tgt = max(1, int(target if target else K.natural_leaves(n)))
        if tgt == self.leaves:
            return {"resharded": False, "leaves": self.leaves, "reason": "already-natural"}
        # 1) re-stamp _leaf, paged in bounded transactions (never load a whole table into memory)
        self._restamp_leaf("vertex", "id", tgt)
        if graph is not None:
            self._restamp_leaf("edge", "edge_key", tgt)
        # 2) adopt the new resolution, then rebuild the digest from the re-stamped `_leaf`
        self.leaves = tgt
        if graph is not None:
            graph.leaves = tgt
        self.rebuild_merkle()
        # 3) record it so it survives a re-open (in `meta`, NOT `counter` — it is not a counter and
        #    must never be recomputed by the counter-drift audit)
        with self.db.write() as cur:
            cur.execute("INSERT INTO meta(k, v) VALUES('merkle.leaves', ?) "
                        "ON CONFLICT(k) DO UPDATE SET v = excluded.v", (str(tgt),))
        return {"resharded": True, "leaves": tgt, "rows": n}

    def _restamp_leaf(self, table: str, key: str, leaves: int, *, batch: int = 5000) -> None:
        """`_leaf = leaf_of(<key>, leaves)` for every row of `table`, keyset-paged by the primary key
        so memory stays bounded and each batch is its own transaction. The coordinate matches the
        write path exactly: the id string for a vertex, `edge_key.hex()` for an edge."""
        after = None
        while True:
            with self.db.write() as cur:
                if after is None:
                    rows = cur.execute("SELECT %s AS k FROM %s ORDER BY %s LIMIT ?"
                                       % (key, table, key), (batch,)).fetchall()
                else:
                    rows = cur.execute("SELECT %s AS k FROM %s WHERE %s > ? ORDER BY %s LIMIT ?"
                                       % (key, table, key, key), (after, batch)).fetchall()
                if not rows:
                    break
                for r in rows:
                    k = r["k"]
                    coord = k.hex() if isinstance(k, (bytes, bytearray)) else str(k)
                    cur.execute("UPDATE %s SET _leaf = ? WHERE %s = ?" % (table, key),
                                (K.leaf_of(coord, leaves), k))
                after = rows[-1]["k"]
            if len(rows) < int(batch):
                break
