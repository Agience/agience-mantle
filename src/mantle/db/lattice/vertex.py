"""`LatticeArtifactStore` — the vertex half of the lattice store.

Implements `mantle.db.store.ArtifactStore` and replaces the raw-SQL call sites
that `_SqliteConnShim` served with TYPED methods. The shim was pattern-dispatch
over a fixed set of the legacy graph engine SQL strings that **returned `[]` for anything it did
not recognise**, which turned every query drift into a wrong answer rather than
an error. Contract §5 catalogues six live instances; the two worst returned the
GLOBAL CORPUS COUNT as a per-collection count and the TOTAL CORPUS SIZE as the
mesh backlog. Typed methods cannot drift, because a renamed method is an
AttributeError at the call site instead of an empty list.

THREE INVARIANTS THIS MODULE ENFORCES, none of them negotiable:

  * **No `count(*)`.** Every count is a `counter` row maintained in the write
    transaction. See `seq.bump`.
  * **No `SKIP`/`OFFSET`.** Keyset only. MEASURED: SKIP at depth 5M = 142,136ms
    vs keyset 743ms. `list_artifacts(skip=...)` RAISES rather than quietly doing
    the slow thing.
  * **`_seq` is proper time, never a clock.** See `seq.py`.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from . import constants as K
from . import schema as _schema
from . import fts as _fts
from . import seq as _seq
from .seq import LatticeConn, SeqAllocator

try:                                    # the declared seam, when mantle is importable
    from ..store import ArtifactStore as _ArtifactStoreABC
except Exception:                       # stand-alone: the package must import on its own
    _ArtifactStoreABC = object          # type: ignore[assignment,misc]


class Page(list):
    """A list of results that KNOWS whether it is the whole answer.

    ⛔ WHY THIS EXISTS. `lookup_by_lemma(limit=12)` returned a plain list, so a caller could not
    distinguish "these are all 12 artifacts carrying this lemma" from "there are 30 and you are
    seeing 12". MEASURED 2026-07-21: `define` on four unrelated words returned EXACTLY 12 every
    time — the cap, not the answer — and nothing in the result said so.

    That is the defect class this codebase keeps hitting: AN ABSENCE ENCODED INVISIBLY. `ic = 0.0`
    meaning both "corpus root" and "not measured"; `signature("")` returning 128 zeros that score
    Jaccard 1.0 against everything; `K_signal = 0` reading as "maximally compact"; `fit_error =
    NaN` passing as a certificate. Silent truncation is the same shape: a partial answer wearing a
    complete answer's clothes.

    A count is a FACT, not a measurement against a noise floor, so there is no entroptics read to
    derive `limit` from — the honest fix is not a better number but making the number's EFFECT
    visible. `Page` is a real `list`, so every existing caller is unaffected; callers that care
    can ask.

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

    The alternative is returning `[]`, and `[]` from a keyed lookup does not mean "I have no
    index" — it means "this store does not contain that word". Those are opposite claims and
    this store refuses to conflate them. See `schema.c_list_index_built`."""


class LatticeArtifactStore(_ArtifactStoreABC):  # type: ignore[misc,valid-type]
    """Artifacts as vertices. One SQLite file, one authoring observer.

    `origin` is THIS observer's identity and is stamped into `_origin` on every
    locally-authored write. It must be stable across restarts of the same node —
    a node that changes its origin forks its own proper time and every peer sees
    two unrelated, permanently-unordered event streams. (This is the same failure
    mode as an unpinned `EMBER_NODE_ID` on the S3 mesh plane.)"""

    def __init__(self, path_or_conn: Any, *, origin: str,
                 leaves: int = K.DEFAULT_LEAVES,
                 allocator: Optional[SeqAllocator] = None):
        """`path_or_conn` is a filesystem path OR an existing `LatticeConn`.

        Pass a shared `LatticeConn` when the graph store uses the same file — then
        vertices and edges commit atomically AND share one proper-time sequence,
        which contract §4 RESOLVED-5 requires (one counter per OBSERVER, spanning
        both tables). The allocator is REUSED automatically; `allocator=` is an
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
        and offer VERTICES, which Phase 3 mints. Until then `ct` holds the MIME
        string verbatim (a valid vertex id when Phase 3 promotes it) and `offer`
        is populated only from an explicit string `offer` field. `doc["context"]`
        is the offer CONCEPTUALLY (see `db/store.py`) but is frequently a dict,
        so it is NOT auto-projected here — flattening a dict into a TEXT column
        is how you get a query predicate that matches nothing and reports zero."""
        offer = doc.get("offer")
        ct_time = doc.get("created_time")
        return {
            "ct": doc.get("content_type"),
            "offer": offer if isinstance(offer, str) else None,
            "content_ref": doc.get("content_ref"),
            "created_by": doc.get("created_by"),
            # `created_time` is a column again by John's dated override (2026-07-20) — see the
            # note on `schema.VERTEX_DDL`. Projected here so the column is POPULATED rather than
            # merely present: a column that is always NULL protects nothing, and would have made
            # the override look landed while carrying no data at all.
            #
            # ⚠ Coerced to `str`. The corpus holds ISO-8601 strings, but nothing stops a caller
            # passing an int epoch, and letting both shapes into one column is how a later
            # comparison silently orders lexicographically against numerics. The doc keeps
            # whatever was passed, verbatim; only the column is normalised.
            "created_time": None if ct_time is None else str(ct_time),
            # The collection's immutable origin root — the key root for content
            # encryption. See `schema.VERTEX_DDL` for why it is a column.
            "origin_root": self._origin_root(doc),
            # ⭐ THE VERSION LINEAGE (John, 2026-07-21). Mantle's rule verbatim
            # (`entities/artifact.py:73`): the FIRST version of an artifact has `id == root_id`,
            # so a doc that names no root IS its own root. `id` is the VERSION identity;
            # `root_id` is the stable handle every version of it shares.
            #
            # ⛔ IT IS NOT `origin_root`. Two different roots, and conflating them would key
            # content encryption off a version lineage: `origin_root` is the CONTAINMENT root
            # (which collection's key decrypts this) and never moves; `root_id` is the VERSION
            # root and is exactly the thing that accumulates new rows over time. schema.py:165
            # already carries this warning — it is restated here because this is the line where
            # someone would reach for the wrong one.
            "root_id": doc.get("root_id") or doc.get("id"),
        }

    @staticmethod
    def _origin_root(doc: Dict[str, Any]) -> Optional[str]:
        """The containment root this vertex is permanently keyed under.

        Precedence, and each step is deliberate:

          1. an explicit ``origin_root`` — already resolved upstream; never recomputed,
             because recomputing a value whose contract is "never moves" is how it moves;
          2. ``collection_id`` — MEASURED 2026-07-21: no collection on node 71 nests, so
             the origin-lineage walk is depth-1 and the parent IS the root. When subject
             trees arrive (GENESIS §5), the resolver upstream supplies (1) and this branch
             stops being reached for nested rows;
          3. the vertex's own id — a top-level artifact with no collection IS its own
             root (the `vtype.*` type definitions are the live example).

        Returns ``None`` only when the doc has no id at all, which `put_artifact`
        rejects separately. It never GUESSES: there is no fallback to `created_by`,
        which is provenance and mutates — P9.3 removed it from the crypto path
        precisely so that correcting an identity cannot orphan a blob.
        """
        explicit = doc.get("origin_root")
        if explicit:
            return str(explicit)
        collection_id = doc.get("collection_id")
        if collection_id:
            return str(collection_id)
        vid = doc.get("id")
        return str(vid) if vid else None

    # The doc key recording WHICH OBSERVER claimed `created_time`. See
    # `_attribute_time` — the value is a claim, so it names its claimant.
    TIME_CLAIM_KEY = "created_time_origin"

    # ── the keyed arm ────────────────────────────────────────────────────────
    # The multi-valued doc fields unrolled into `listkey`. A CLOSED set, deliberately:
    # `lookup_by_list_field` refuses a field outside it rather than seeking an index that was
    # never populated for it and reporting the resulting `[]` as "no matches" — the same
    # wrong-answer-not-empty-answer trap the build marker exists to close, one level down.
    LIST_INDEX_FIELDS: Tuple[str, ...] = ("lemmas", "calls")

    def _index_lists(self, cur: sqlite3.Cursor, aid: str, doc: Optional[Dict[str, Any]],
                     ct: Any, *, had_prev: bool = True) -> int:
        """Re-post this artifact's indexed list fields; returns the number of postings written.

        Delete-then-insert rather than a diff: the posting set is small (a synset has ~3 lemmas)
        and a diff has to be correct about removals, which is where an index silently drifts out
        of step with its table. `doc=None` is the delete path.

        ⚠ `had_prev=False` SKIPS THE DELETE, and it is a correctness-preserving optimisation, not
        a shortcut: `_write_row` has already established that no `vertex` row existed for this
        id, and postings only ever exist alongside a row (they are written in the same
        transaction and retracted in `delete_artifact`). So there is provably nothing to delete.
        This matters because a bulk ingest is ALL inserts — MEASURED, leaving the unconditional
        DELETE in place cost one index seek per artifact across the whole corpus for a guaranteed
        zero-row result. The default stays `True` so any caller that has NOT proven the row is
        new gets the safe behaviour."""
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

        `created_time` is no longer a column (contract §2.2): it is CONTENT — the
        claim "observer X read its clock as T" — and content lives in `doc`. But an
        anonymous integer is a weaker claim than the truth supports, because two
        observers reading their own clocks disagree with no function to reconcile
        them. Naming the claimant makes the value interpretable: it stops being
        "the time" and becomes "the time ACCORDING TO 71", which is what it always
        actually was.

        ⚠ Set once and PRESERVED thereafter. On mesh consume `origin` is the
        peer's origin, so a replicated row keeps its own attribution; re-stamping
        would relabel a peer's clock reading as ours, which is the same category of
        error as re-stamping `_origin` itself.

        This adds no column and no index. If you need ordering, that is
        `edge.order_key` within a frame and graph reachability across frames —
        UNORDERED IS A VALID ANSWER."""
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
        self._attribute_time(d, origin)     # a clock reading is a CLAIM; name the claimant

        prev = cur.execute(
            "SELECT ct, _seq, _origin, doc FROM vertex WHERE id = ?", (aid,)).fetchone()

        # ⛔ THE TIME CLAIM IS IMMUTABLE — BOTH HALVES, TOGETHER. MEASURED 2026-07-21: a second
        # `put_artifact` overwrote `created_time` (2020 -> 2099) while `_attribute_time` correctly
        # PRESERVED `created_time_origin`. On a replicated row that produced the worst possible
        # state: claimant `peer-1` beside a value peer-1 never wrote — the row asserting "peer-1
        # read its clock as 2098". A mismatched claim is more misleading than either field being
        # wrong on its own, because the claimant makes it look attributed and checked.
        #
        # This also makes the §2.3 exemption TRUE rather than aspirational. `created_time` is only
        # admissible as a column while it does not move under replication; before this it did.
        # See `test_deprecated_columns.py` and `test_created_time_survives_in_doc_attributed_to_
        # its_claimant`.
        #
        # ⚠ The claim is the PAIR. Preserving one without the other is what created the mismatch,
        # so they are restored together or not at all.
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
            # ⛔ `root_id` IS DELIBERATELY ABSENT FROM THE UPDATE SET. A lineage handle that can
            # be rewritten is not a lineage handle: re-pointing it would silently move a row into
            # a different version history, and the rows it left behind would still claim it.
            # COALESCE keeps whatever the row was first written with; only a row that has none
            # (a pre-migration row) can acquire one, via `backfill_root_id()`.
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
        # An UPDATE allocates a fresh seq and VACATES the old one, so live rows go
        # non-contiguous. Recording the vacancy here — in the same transaction —
        # is what keeps `live_rows + vacated == last_seq` exact, and what keeps a
        # row lost OUTSIDE this path distinguishable from one retired by it.
        if prev is not None:
            _seq.vacate(cur, prev["_origin"])
        _seq.bump(cur, _schema.c_rows(origin), 1)

        # ── incremental counters ──
        old_doc = json.loads(prev["doc"]) if prev is not None else None
        self._recount(cur, old_doc, d, prev["ct"] if prev is not None else None, cols["ct"])
        self._sync_task(cur, aid, old_doc, d, cols["ct"])
        # The keyed arm's index, maintained in THIS transaction — so a lemma posting and the
        # doc it points at can never be observed disagreeing, and an update that changes `ct`
        # re-posts under the new type rather than leaving a stale discriminator behind.
        self._index_lists(cur, aid, d, cols["ct"], had_prev=prev is not None)
        # ⭐ THE LEXICAL INDEX IS MAINTAINED BY THE STORE, IN THIS TRANSACTION, for the same reason
        # the keyed arm's is: an index is DERIVED data, and derived data that some CALLER has to
        # remember to update is derived data that will be stale. MEASURED 2026-07-23: it was built
        # only by a migration script, so an ordinarily-ingested store had 120,684 rows and no index
        # at all — every question raised `no such table` and no health signal moved. Briefly it was
        # then maintained by each ingester calling an ember-side adapter, which is the same defect
        # one layer up: five call sites to keep in step, and a store written to by anything else
        # would be silently unsearchable. The writer maintains it. [John, 2026-07-23: "leave one
        # path. the only path."]
        _fts.index_artifacts(cur, [d])
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
            # The provenance null-audit. Counted here, on the ONE write path, so a
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

        ⚠ THIS IS THE FIX FOR A SPECIFIC, VERIFIED DATA-LOSS BUG. In the seed
        store `put_artifact` wrote 7 columns and `put_many` wrote 6 — and
        `INSERT OR REPLACE` DELETES the row and reinserts it, so the omitted
        `status` column reset to NULL. A task at `status='claimed'` re-upserted
        through the mesh ended with `status = NULL` in the column while the doc
        JSON still read `claimed`. `claim`, `reclaim_stale` and `queue_stats` all
        read the COLUMN, so the task became invisible to every one of them: not
        pending, not claimed, never reclaimed. Permanently orphaned, silently.

        There is exactly ONE write path here and it derives every sidecar field
        from the doc, so the two can no longer disagree."""
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

        ⚠ **`stamp_rev` IS A COMPATIBILITY KEYWORD AND IT IS NOT OPTIONAL.**
        `_rev` no longer exists, but every production caller passes `stamp_rev=`
        (`worker.py`, ~10 sites in `mesh/sync.py`), and a backend written honestly
        without it raises `TypeError` — which `reconcile_merkle` SWALLOWS into
        `applied: 0`, i.e. replication silently applies nothing while reporting a
        clean zero. That is not hypothetical; it is the outage the sqlite backend
        shipped with until 2026-07-20. So the keyword stays and its MEANING is
        mapped onto `(_origin, _seq)`:

            stamp_rev=True   LOCAL AUTHORSHIP. Allocates a fresh `_seq` from this
                             observer's proper time and stamps `_origin = self.origin`.
                             This is a new authored event.

            stamp_rev=False  MESH CONSUME. PRESERVES the incoming `(_origin, _seq)`
                             exactly. A replicated doc keeps the version identity
                             it was published with, so it does not echo around the
                             mesh forever, and the peer's merkle leaf for that row
                             matches ours once applied.

        With `stamp_rev=False` a doc carrying no `(_origin, _seq)` is a hard
        ValueError, NOT a silent local re-stamp. Re-stamping would claim this node
        authored a peer's event, corrupting the only causal record the system has."""
        if not doc.get("id"):
            raise ValueError("put_artifact: doc has no 'id'")
        with self.db.write() as cur:
            # ── CONTENT DECIDES whether this is a new version — no forced route ──────────────
            # John, 2026-07-21: "put artifact shouldnt be a forced route. let the information
            # decide." The information is `content_ref` = cas/<sha256-of-the-content>:
            #   · SAME content_ref  -> the bytes are unchanged; this is a RE-DESCRIBE (offer,
            #     context, any derived field), applied in place. Not a version.
            #   · DIFFERENT content_ref -> the bytes changed; the prior version is SNAPSHOTTED
            #     under a derived archived id (sharing root_id) BEFORE the handle is overwritten,
            #     so `get(id)` always returns the LATEST ("version_id is latest") and the prior
            #     content is never silently lost. `revise()` becomes a convenience, not a gate:
            #     the single write path decides, driven by the data, not by which method was called.
            # Only for LOCAL authorship (stamp_rev): a mesh-consumed row is already version-distinct
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

        ⚠ IDEMPOTENT and DETERMINISTIC. The snapshot id is `<aid>@<blake2b(old content_ref)>`, so
        re-writing the same prior content re-derives the same id and the existing-row guard skips
        it — a re-put that oscillates between two contents does not mint a snapshot per oscillation.
        Reuses `_write_row` (a NEW id, so the content-decides branch above cannot re-fire) to get
        the merkle XOR, `_seq` allocation and accounting right without duplicating them."""
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
        """Bulk upsert. **RETURNS THE NUMBER *HANDLED*, NOT THE NUMBER *WRITTEN*.**

        The distinction is load-bearing. `mesh.sync._apply_artifacts` uses this
        return as its ONLY guard against advancing the consume cursor:

            written = store.artifacts.put_many(batch, batch=500, stamp_rev=False)
            if written < len(batch): raise RuntimeError("partial apply: ...")

        A segment recorded as applied advances `last_key` behind a MONOTONE
        `StartAfter` marker, so anything not written is GONE. "Handled" therefore
        has a precise meaning every implementation must honour:

            doc WRITTEN                     -> counts
            doc correctly REJECTED by LWW   -> counts  (declining to overwrite a
                                               newer local row IS the right
                                               outcome; it is handled, not lost)
            doc that ERRORED                -> MUST NOT COUNT

        An implementation returning `len(docs)` unconditionally silently disables
        the mesh's only data-loss guard from inside the store layer.

        `on_unordered` decides what to do when an incoming doc and the local row
        have DIFFERENT `_origin`s — genuinely concurrent authorship of one vertex,
        which `(_origin, _seq)` cannot and must not order (contract RESOLVED-3: no
        clock, ever; no synthesized tiebreak). It is an explicit parameter rather
        than a buried default because there is no universally right answer:

            "keep_local"  keep ours, count HANDLED, bump `conflict:unordered`.
                          Nothing is lost — the peer still holds its copy.
            "take_remote" apply theirs, count HANDLED. For a consumer that has
                          decided the peer is authoritative for this vertex.
            "error"       count NOT handled, so the mesh holds its cursor and the
                          divergence surfaces instead of being absorbed.

        ⚠ Under "keep_local"/"take_remote" the two sides stay divergent and their
        merkle leaves keep mismatching, so anti-entropy will re-offer the row every
        round. That is honest (the disagreement is real) but it is not free — see
        the unit report's open questions.

        Per-document SAVEPOINTs: one bad doc must not roll back the batch, because
        a whole-batch rollback with a `len(docs)` return would report success for
        rows that no longer exist."""
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
                    # NOT counted as handled. The mesh will hold its cursor.
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
                return 1                       # correctly rejected -> HANDLED
            if verdict == K.UNORDERED:
                if on_unordered == "keep_local":
                    _seq.bump(cur, "conflict:unordered", 1)
                    return 1                   # a decision was made -> HANDLED
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

    def version_of(self, artifact_id: str) -> Optional[Tuple[Optional[str], Optional[int]]]:
        """`(_origin, _seq)` for one vertex, without hydrating the doc. `None` if
        absent. Feed both values to `constants.compare_version`.

        ⭐ ALSO THE ARTIFACT'S OWN FRESHNESS STAMP. `_seq` is reallocated on every write
        to this row, so a reader holding something DERIVED from this artifact can ask
        "is what I derived it from still what is here?" for the price of one PK seek and
        no doc hydration — MEASURED on 71: **5.1 µs against 17.7 µs for `get_artifact`**,
        and against 370 µs to rebuild one `wn_store.Synset`. It is not a side-car version
        field someone maintains beside the data; it is a column of the row, written by the
        one writer inside the write transaction. See `seq.write_mark` for the whole-store
        gate this pairs with."""
        r = self.db.read().execute(
            "SELECT _origin, _seq FROM vertex WHERE id = ?", (artifact_id,)).fetchone()
        return (r["_origin"], r["_seq"]) if r else None

    def write_mark(self) -> Tuple[Tuple[str, int], ...]:
        """The store's whole-of-store WRITE MARK — see `seq.write_mark` for what it is
        and why a cache may hang its validity on it. Published here because a reader must
        never hand-roll a probe against this schema ([[never-handroll-probes]]): a missing
        stat is a stat to ADD, not a SELECT to write at the call site."""
        return _seq.write_mark(self.db)

    def list_artifacts(self, *, state: Optional[str] = None,
                       content_type: Optional[str] = None,
                       collection_id: Optional[str] = None,
                       created_by: Optional[str] = None,
                       limit: Optional[int] = None,
                       skip: int = 0,
                       include_archived: bool = False) -> Iterator[Dict[str, Any]]:
        """Filtered stream in `id` order.

        ⭐ ONLY THE HEAD ANSWERS BY DEFAULT (§6B.3). Content-decides versioning (§6B.4) snapshots a
        prior version under a derived id with `state='archived'`; without this, a query would
        return the head AND every archived prior as separate hits. So archived rows are excluded
        unless the caller either passes an explicit `state=` (they know exactly what they want,
        including `state='archived'` for history) or sets `include_archived=True`. On a store with
        no snapshots yet this changes nothing; it is the guard that keeps versioning from
        double-answering the moment a content-changing write lands.

        ⚠ **`skip` IS ACCEPTED ONLY AS `0` AND RAISES OTHERWISE.** It exists on the
        ABC and is kept so callers do not hit an unexpected-keyword TypeError, but
        OFFSET pagination is banned outright: MEASURED, `SKIP` at depth 5M took
        142,136ms against 743ms for the equivalent keyset page, because the engine
        walks and discards every skipped row. Use `page_by_id`. Failing loudly here
        is deliberate — the seed's `LIMIT ? OFFSET ?` was a defect, not a pattern."""
        if skip:
            raise ValueError(
                "list_artifacts(skip=%r): OFFSET pagination is not supported. "
                "Use page_by_id(after=<last id>, limit=n) — keyset paging. "
                "Measured: SKIP at depth 5M = 142,136ms vs keyset 743ms." % (skip,))
        where, params = [], []
        for col, val in (("ct", content_type), ("created_by", created_by)):
            if val is not None:
                where.append(col + " = ?")
                params.append(val)
        for field, val in (("state", state), ("collection_id", collection_id)):
            if val is not None:
                where.append("json_extract(doc, '$.%s') = ?" % field)
                params.append(val)
        # HEAD-ONLY DEFAULT: exclude archived unless the caller asked for a specific state or for
        # history. `IS NOT` is null-safe — a row with no `state` (committed-by-absence) is kept.
        if state is None and not include_archived:
            where.append("json_extract(doc, '$.state') IS NOT 'archived'")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        q = "SELECT doc FROM vertex" + clause + " ORDER BY id"
        if limit is not None and int(limit) >= 0:
            q += " LIMIT ?"
            params.append(int(limit))
        for r in self.db.read().execute(q, params):
            yield json.loads(r["doc"])

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
                # A real deletion, accounted in THIS observer's proper time.
                _seq.vacate(cur, r["_origin"])
            # else: EVICTION of a CACHED copy. The authoritative row still exists at its author, and
            # this node never authored it — dropping our copy is not a deletion in anyone's
            # proper-time ledger, so it MUST NOT vacate a (foreign) origin's sequence.
            for name, delta in self._counter_deltas(old, None, r["ct"], None).items():
                _seq.bump(cur, name, delta)
            self._sync_task(cur, artifact_id, old, None, r["ct"])
            self._index_lists(cur, artifact_id, None, r["ct"])   # retract the postings too
            # ⭐ AND THE LEXICAL INDEX — the write path maintains it (`_write_row`), so the delete
            # path must retract it. It did not until 2026-07-30: `FtsIndex.delete` had no caller
            # here, so a deleted artifact kept its `fts_map` row and lexical search returned a hit
            # whose `get_artifact()` is None — a WRONG answer, not an empty one. `retract_artifact`
            # drops the postings AND decrements `fts:total`, in THIS transaction, so the index and
            # the row it points at can never be observed disagreeing (same rule as `_index_lists`).
            _fts.retract_artifact(cur, artifact_id)
            cur.execute("DELETE FROM demand WHERE id = ?", (artifact_id,))   # clear any demand entry

    def evict_artifact(self, artifact_id: str) -> None:
        """Drop a CACHED (reached) copy — a non-accounted removal. The authoritative row survives in
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

    def demand_page(self, *, after: str = "", limit: int = 5000) -> List[Dict[str, Any]]:
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
                "SELECT id FROM demand WHERE id > ? ORDER BY id LIMIT ?", (after, 5000)).fetchall()
            if not rows:
                break
            n += len(rows)
            after = rows[-1]["id"]
            if len(rows) < 5000:
                break
        return n

    # ── keyset pagination — the ONLY sanctioned paging primitive ─────────────
    def page_by_id(self, *, after: str = "", limit: int = 200,
                   content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """`WHERE id > :after ORDER BY id LIMIT :n`, walking the PK index.

        Replaces `content_tier.py:307` (`SELECT id, content_ref FROM Artifact
        WHERE id > :cur ORDER BY id LIMIT :n`), which the shim did not recognise
        and answered with `[]` — setting `backfill_done = True` on an empty page
        and permanently retiring a one-shot backfill that never ran.

        Cost is constant in the DEPTH of the page. Drive the next call with
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
        """The PUBLISH SCAN: `WHERE _origin = :me AND _seq > :cursor ORDER BY _seq`.

        An indexed range walk on `ix_v_origin` with NO SKIP and — because `_seq` is
        injective, unlike the `time.time_ns()` it replaced — **no revision-group
        completion dance.** `content_tier.py:317-343` carries ~25 lines that exist
        only to work around `_rev` ties ("a bare `_rev > :r` cursor skips rows
        permanently"). Against `_seq` that entire mechanism is unnecessary: a
        strict `>` is correct."""
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
        """**The correct publish backlog.** Rows this observer authored above each
        table's OWN cursor, counted per table and summed.

        Returns `{"vertex", "edge", "total", "exact"}`.

        Two defects this exists to avoid. Both applied to the former
        `publish_backlog` (`high_water - cursor`), which was wrong three separate
        ways and has been DELETED rather than documented a fourth time — a 25-line
        warning docstring did not stop it being wired into the health path, so the
        defence that kept failing was documentation:

        1. **Vacated seqs.** `last_seq - cursor` counts ALLOCATIONS. An update
           allocates a new seq and vacates the old one, so the arithmetic counts
           work whose rows no longer exist — unboundedly, under operator-rewrite
           churn. Counting ROWS is immune by construction.
        2. **Two feeds, two cursors.** A vertex scan cannot consume an edge's seq,
           so one counter shared across both tables means neither cursor alone
           ranges over the union. Measuring against either counts the other table's
           rows as outstanding forever, giving the backlog a floor it can never
           cross and making `converged` structurally unreachable. `min(vc, ec)` does
           NOT fix this — it relocates the floor to whichever table stopped lower.

        Each term is an indexed range on `ix_v_origin` / `ix_e_origin`, bounded by
        `cap`, and **reaches 0 exactly when its own feed drains** — so the sum is 0
        exactly when the node has published everything. That is the property a
        convergence signal needs and neither broken form has.

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
        `count_after_id`. Use for FOREIGN origins, where holes make subtraction
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

        Replaces `genesis.py:1313`, where the shim tested `"collection_id" in params`
        while the caller passed `{"c": ...}` — so it fell through to the bare-total
        branch and **returned the GLOBAL CORPUS COUNT as the per-collection count**,
        which then fed `advance_curriculum`'s resume offset. A wrong answer, not an
        empty one.

        `committed_only` is preserved because the two callers need different things:
        `advance_curriculum` uses the count as a RESUME OFFSET and must count every
        state — filtering to committed there under-counts, hands the ingester a low
        offset, and re-ingests records it already has."""
        return _seq.counter_of(
            self.db, _schema.c_collection(collection_id, committed_only=committed_only))

    def count_by_content_type(self, content_type: str) -> int:
        return _seq.counter_of(self.db, _schema.c_ct(content_type))

    def page_by_origin_root(self, origin_root: str, *, after: str = "",
                            limit: int = 5000) -> List[str]:
        """One keyset page of a collection's member ids by its indexed CONTAINMENT root
        (`origin_root == collection_id`, flat today) over `ix_v_root` — never a scan. A generic
        enumerator; the store keeps no opinion about what a collection means. The mesh uses it to
        enumerate the members of GRANT-GATED collections (which never leave the node); the access
        decision itself lives in `ember.access`, computed from grants."""
        rows = self.db.read().execute(
            "SELECT id FROM vertex WHERE origin_root = ? AND id > ? ORDER BY id LIMIT ?",
            (str(origin_root), after, int(limit))).fetchall()
        return [r["id"] for r in rows]

    def count_missing_field(self, field: Any, *, committed_only: bool = False) -> int:
        """Vertices whose audited provenance field is MISSING. Counter lookup, O(1).

        Replaces `genesis._count_null` / `_scan_missing_field`'s capped keyset walk,
        which returns **NOT MEASURED at 6.24M rows** — so the provenance audit was
        unavailable exactly at production scale, where it matters. `field` comes from
        the closed `NullAuditField` set and is a VALUE, never interpolated into SQL.

        **Returns a bare `int`, deliberately.** `genesis._scan_missing_field` probes
        this as `int(typed(field))`; the capped `{"n":…, "exact":…}` dict used by
        `count_after_id` would raise `TypeError` there and be swallowed into `None` —
        i.e. adding this method would appear to do nothing. An int is safe here only
        because the value is exact: it is an incrementally maintained counter, never a
        truncated scan, so there is no cap to hide.

        ⚠ **DEFAULT IS THE SUPERSET, AND THAT IS THE SAFE DIRECTION.** The original
        predicate was `<field> IS NULL AND state = 'committed'`; contract §2 deletes
        `state` and §3.3 replaces it with a verification edge that does not exist yet.
        The default therefore counts rows missing `field` ACROSS EVERY STATE, matching
        `_scan_missing_field`'s fallback exactly. The only consumer is
        `invariant_holds = (miss_cite == 0 and miss_prov == 0)`, so a superset can
        raise a FALSE ALARM but can NEVER report a clean audit over a dirty corpus. An
        audit that misses dirty rows is worthless; one that flags extra rows is merely
        noisy.

        `committed_only=True` is exact and is maintained alongside — but it is
        TRANSITIONAL, resting on the `state` key surviving in `doc` JSON. When Phase
        3.3 lands the verification edge, scope to that and delete the flag. Do not
        make it the default in the meantime: it would silently change the audit's
        answer relative to every other backend's fallback."""
        f = K.NullAuditField.coerce(field)
        return _seq.counter_of(
            self.db, _schema.c_missing(f.value, committed_only=committed_only))

    # ── content-type fetch (stats.py:515) ────────────────────────────────────
    def list_by_content_type(self, content_type: str, *, cap: int = 2000,
                             include_archived: bool = False
                             ) -> Tuple[List[Dict[str, Any]], bool]:
        """Returns `(docs, exhaustive)`. Probes with `LIMIT cap+1` and reports
        `exhaustive=False` if a full page came back.

        Replaces `stats.py:515`. The shim matched no pattern there and returned
        `[]` — and because `len([]) <= cap`, the guard at `:517` accepted it as the
        fast answer and the exhaustive fallback NEVER FIRED. `[]` was served as
        authoritative. Returning the flag alongside the rows makes "I did not
        finish" unrepresentable as "there is nothing".

        ⭐ HEAD-ONLY BY DEFAULT, matching `list_artifacts` (§6B.3). This was the ONE typed read
        that still returned archived rows, and the inconsistency had teeth: EREA (2026-07-28) built
        grant revocation on it and found that after a revoke the ARCHIVED pre-revoke grant was
        still returned — and still granted access. Mantle's own grant path was unaffected only
        because `lattice_api` passes an explicit `state` to `list_artifacts` instead.

        ⚠ PASS `include_archived=True` TO ENUMERATE A COMPLETE SET. Replication and repair need
        every row, not the current ones: `mesh/sync.py` subtracts an enumerated operational set,
        and silently dropping archived rows there would make a peer diff a leaf whose object 404s
        and stay permanently divergent — exactly the conflation the `(docs, exhaustive)` contract
        exists to prevent. The rule is: answering a QUESTION wants heads, rebuilding STATE wants
        everything."""
        cap = max(0, int(cap))
        where = "ct = ?"
        params: List[Any] = [content_type]
        if not include_archived:
            # `IS NOT` is null-safe — a row with no `state` (committed-by-absence) is kept.
            where += " AND json_extract(doc, '$.state') IS NOT 'archived'"
        params.append(cap + 1)
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE " + where + " ORDER BY id LIMIT ?", params).fetchall()
        docs = [json.loads(r["doc"]) for r in rows]
        return (docs[:cap], len(docs) <= cap)

    def content_type_mark(self, content_type: str, *, cap: int = 2000
                          ) -> Tuple[int, int, bool]:
        """`(rows, max _seq, exhaustive)` over one content type — **the freshness stamp of a
        derivation built from a whole content type**, the set-sized sibling of `version_of`.

        `_offers` in ember is derived from all 48 `…operator+json` artifacts and costs 215–270 ms to
        rebuild; the whole-store `write_mark` would drop it whenever anything at all was written,
        which on a live node means every chat message. This is the exact question instead: has
        anything OF THIS TYPE been written since we read it?

        Both terms are needed. `max(_seq)` alone misses the deletion of any row that was not the
        newest; `rows` alone misses an update, which reallocates `_seq` in place.

        ⛔ CAPPED, AND THE CAP IS NOT A TUNING KNOB — IT IS WHAT KEEPS THIS FROM BECOMING THE
        DEFECT IT GUARDS. `count(*)` over a content type dereferences every matching record
        ([[count-star-dereferences-every-record]]), so an uncapped version of this pointed at
        `text/x-wordnet` (117k) or `application/x-concept` (1.17M) would scan the 5.7 GB lattice on
        a read path — the exact failure that zombied the acceptor. `LIMIT cap+1` bounds the work
        absolutely, and `exhaustive=False` REPORTS that the mark covers only a prefix rather than
        quietly returning a number that looks whole. **A caller that gets `exhaustive=False` must
        treat the stamp as unusable, not as approximate**, because a mark over a prefix is stable
        while the tail changes underneath it — a check that cannot fail. The default matches
        `list_by_content_type`'s, so a set small enough to enumerate is a set small enough to
        stamp, and the two can never disagree about which rows they mean."""
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
        CHANGED leaf's rows to rebuild that leaf's authoritative object, and
        `ix_v_leaf(_leaf)` makes it an indexed equality lookup over ~corpus/leaves rows
        instead of the full-corpus `_scan_rows`. `_refresh_leaves_lattice` records that
        the store "exposes no page_by_leaf" and pays a full keyset pass per refresh as a
        result; this is the accessor that removes that cost.

        `(docs, exhaustive)` for the SAME reason as `list_by_content_type`: a truncated
        page must never be indistinguishable from a genuinely empty leaf — that
        conflation is exactly what makes a peer diff a leaf whose object 404s and stay
        permanently divergent. The caller still filters on `_is_replicated`: on lattice
        every row (operational included) carries `_leaf`, so a leaf is NOT a replication
        filter by itself (the mesh subtracts operational rows explicitly — see
        `_refresh_leaves_lattice`)."""
        cap = max(0, int(cap))
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE _leaf = ? ORDER BY id LIMIT ?",
            (int(leaf), cap + 1)).fetchall()
        docs = [json.loads(r["doc"]) for r in rows]
        return (docs[:cap], len(docs) <= cap)

    def list_by_doc_field(self, *, content_type: str, field: str, value: Any,
                          limit: int = 1000) -> List[Dict[str, Any]]:
        """Docs of one content type whose JSON `field` equals `value`.

        Replaces `genesis.py:562` (`_shards_done`: `WHERE content_type = :ct AND
        source_name = :n`), which the shim answered with `[]` — a silent no-op that
        made the fleet re-ingest finished shards.

        Scoped by `ct` FIRST, which is indexed, so the JSON predicate only ever
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
        """The DICTIONARY path: a word -> the artifacts that carry it as a lemma.

        `content_type` is the type discriminator and callers asking a typed question should
        pass it — `lexical.define` passes `text/x-wordnet`, `code` passes the symbol type. See
        `lookup_by_list_field` for why filtering afterwards is not equivalent.

        ⚠ `limit` IS A PAGE SIZE, NOT AN ANSWER SIZE. It is a chosen number and cannot be derived
        — how many artifacts carry a lemma is a fact about the corpus, not a signal with a noise
        floor. What CAN be honest is whether it truncated: the result is a `Page`, and
        `page.truncated` says so. A caller that ignores it is asserting completeness it was never
        given."""
        return self.lookup_by_list_field("lemmas", word, limit=limit, content_type=content_type)

    def lookup_by_list_field(self, field: str, value: Any, *, limit: int = 20,
                             content_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Artifacts whose indexed list `field` contains `value`, optionally of ONE content type.

        ⚠ `content_type` IS APPLIED IN THE SEEK, AHEAD OF THE LIMIT, AND THAT IS THE WHOLE
        POINT. The legacy original had no such parameter, so every caller that wanted one type
        post-filtered the returned page — and post-filtering cannot recover what the LIMIT
        already spent. MEASURED on the live corpus: `lookup_by_lemma('spaceship', 200)` returned
        200 rows and ZERO synsets, because 6M `wiki-*` rows share the lemma namespace with
        117,659 `wn-*` rows; `code.py:75`'s `[a for a in ... if a['content_type'] == SYMBOL]`
        therefore filtered an all-distractor page down to `[]` and reported "not found" for
        symbols the store held. Here the predicate is part of the index key `(field, value, ct)`,
        so the LIMIT is spent entirely on rows of the requested type.

        `content_type=None` keeps the undiscriminated lookup expressible — the browse UI wants
        every type — and it is served by the same index's `(field, value)` prefix. It is a
        deliberate choice by the caller rather than the only available behaviour.

        Ordered by `aid` so a repeated query returns a stable page; the previous implementation
        had no ORDER BY, which made a LIMIT'ed result nondeterministic between runs."""
        self._require_list_index(field)
        params: List[Any] = [field, str(value).lower()]
        clause = ""
        if content_type is not None:
            clause = " AND l.ct = ?"
            params.append(str(content_type))
        # ⚠ FETCH ONE MORE THAN ASKED FOR. That extra row is the only way to know whether the
        # limit truncated the answer, and it costs one index step. Without it a full page and a
        # complete answer are byte-identical to the caller.
        n = int(limit)
        params.append(n + 1)
        rows = self.db.read().execute(
            "SELECT v.doc FROM listkey l JOIN vertex v ON v.id = l.aid "
            "WHERE l.field = ? AND l.value = ?" + clause +
            " ORDER BY l.aid LIMIT ?", params).fetchall()
        truncated = len(rows) > n
        return Page([json.loads(r["doc"]) for r in rows[:n]], truncated=truncated, limit=n)

    def backfill_root_id(self, *, dry_run: bool = False) -> Dict[str, Any]:
        """Give every pre-existing row its VERSION LINEAGE handle. Idempotent.

        ⛔ ADDING THE COLUMN IS NOT THE FIX. A row written before `root_id` existed has NULL, and
        a NULL discriminator is INVISIBLE to every query that filters on it — the same failure as
        the `wn-*` rows whose `ct` was NULL, which no typed query could see and no index covered.
        A lineage feature over a column that is NULL for 6.25M rows is nominal, not real.

        The fill is Mantle's rule (`entities/artifact.py:73`): a doc that names a `root_id` keeps
        it; one that does not IS its own first version, so `root_id = id`. That is exactly correct
        for an initial load, where each artifact has precisely one version.

        ⚠ REPORTS `remaining` EVEN AT ZERO. "Filled 0 rows" from an already-complete store and
        "filled 0 rows" from a store where the column does not exist are the same sentence with
        opposite meanings, so the count that matters is what is STILL NULL afterwards.
        """
        con = self.db.read()
        cols = {r[1] for r in con.execute("PRAGMA table_info(vertex)")}
        if "root_id" not in cols:
            # Refusal, not a quiet zero: the caller asked for a lineage the store cannot hold.
            return {"ok": False, "filled": 0, "complete": False,
                    "reason": "vertex has no root_id column — run ensure_schema() first"}

        # ⛔ NO `count(*)`. It dereferences every record — on a 6.25M-row store this is the query
        # that zombied node 71 — and `test_no_count_star_reaches_sqlite` enforces the ban. EXISTS
        # against `ix_v_root_id` is a SEEK that stops at the first match, and the honest answer to
        # "is the backfill complete?" is a BOOLEAN, not a number nobody can produce cheaply.
        # `filled` comes from the UPDATE's own rowcount, which is exact and free.
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
        # ⚠ Re-checked AFTER the write, never inferred from `filled`. "Filled 0" from an already
        # complete store and "filled 0" from a store where nothing worked are the same sentence
        # with opposite meanings; `complete` is what the caller must gate on.
        remaining = _pending()
        return {"ok": not remaining, "filled": filled, "complete": not remaining}

    def revise(self, artifact_id: str, changes: Dict[str, Any], *,
               author: Optional[str] = None) -> Dict[str, Any]:
        """Author a NEW VERSION under the same lineage. A CONVENIENCE, not a required route.

        ⚠ THIS IS NO LONGER THE ONLY WAY TO VERSION, and it is deliberately not a gate. John,
        2026-07-21: *"put artifact shouldnt be a forced route. let the information decide."*
        `put_artifact` now versions on its own when the `content_ref` changes (the content is the
        information that decides), so a caller need never reach for a separate method to avoid a
        silent overwrite. `revise()` remains for callers who want to express "make a new version
        with exactly these field changes" without re-assembling the whole doc — it computes the
        changed doc and hands it to the same content-decides write path.

        Shape, matching the two implementations that already do this correctly
        (`entities/artifact.py`, Ember's `LocalCache.revise`):
          · the new version is a NEW ROW with its own `id`, sharing `root_id`;
          · the prior head is ARCHIVED, never deleted — it stays queryable forever;
          · `version_id` (John: "version_id is latest") is the committed head.

        ⛔ `state` IS RESERVED — PASSING IT IN `changes` RAISES.
        It used to be accepted and then silently overwritten: the method applied the caller's
        changes and *then* forced `doc["state"] = "committed"`, so `revise(id, {"state":
        "revoked"})` returned success and had no effect. EREA hit this implementing grant
        revocation (2026-07-28) and worked around it with a separate `grant_active` field.

        ⚠ THE SILENT VERSION WAS A SECURITY DEFECT, NOT JUST A TRAP. Grants OVERLOAD this field —
        `lattice_api.py` carries a grant's lifecycle (`active` / `revoked` / `pending_accept`) on
        the vertex `state` (see `_grant_docs`). So the forced `"committed"` would have flipped a
        REVOKED grant back to committed, silently reinstating access that had been withdrawn.

        `state` here means LINEAGE POSITION (draft/committed/archived) and is the method's own
        output — a revision is by definition the new committed head. That it also carries grant
        lifecycle is a real overload of one field by two orthogonal axes; untangling it is a
        FLAGGED SEAM, deliberately not attempted here. Refusing loudly closes the hole either way.

        ⭐ HEAD-ONLY READS ARE NOW ENFORCED (the old "KNOWN GAP" note is obsolete).
        `list_artifacts()` defaults to `state != 'archived'` (see its docstring) and
        `list_by_content_type()` gained the same default, so a revision no longer double-answers.
        `fts` still has no `archived` predicate — a search index hit must be resolved through the
        head to be trusted.

        ⚠ THE VERSION ID IS DERIVED, NOT COUNTED. `f"{root}~{n}"` off a version count would make
        two observers who revised concurrently mint the SAME id for DIFFERENT content, and the
        lattice would silently keep one. It is the fingerprint of the canonical doc instead:
        deterministic (all observers agree), collision-resistant, and idempotent — re-revising
        with identical content re-derives the same id and writes nothing new.
        """
        if "state" in changes:
            raise ValueError(
                "revise() does not accept `state` (%r): it is RESERVED. `state` is lineage "
                "position and revise() sets it — the new version is the committed head. It was "
                "silently overwritten before, which meant a caller revising a grant to 'revoked' "
                "got success and no effect, and the forced 'committed' would REINSTATE access that "
                "had been revoked. Set lifecycle on a field of your own, or archive via "
                "put_artifact()." % (changes["state"],))
        prev = self.get_artifact(artifact_id)
        if prev is None:
            raise KeyError("cannot revise %r: no such artifact" % artifact_id)
        if prev.get("state") == "archived":
            # ⛔ REVISING AN ARCHIVED VERSION FORKS THE LINEAGE, INVISIBLY. Called twice with the
            # same original id, the old code produced TWO live committed heads on one root_id (plus
            # a stale `superseded_by` on the newer), and nothing raised — `head_of` kept answering,
            # so the fork was undetectable from the outside. A revision must extend the CURRENT
            # head; resolve it first.
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
        # ⛔ STORE BOOKKEEPING IS EXCLUDED FROM THE VERSION FINGERPRINT, AND THIS IS A §2.3
        # REQUIREMENT, NOT A TIDINESS PREFERENCE. MEASURED: a round-trip through the store leaves
        # the doc byte-identical EXCEPT `_seq` (allocation order). Hashing it would make the SAME
        # content mint DIFFERENT version ids depending on which node wrote it and in what order —
        # an observer-dependent identity, which is exactly what the contract forbids, and it would
        # also break idempotency (re-revising identical content would fork the lineage forever).
        # `id` is excluded because it is the value being derived.
        canon = {k: v for k, v in doc.items()
                 if k != "id" and not (k.startswith("_") and k in ("_origin", "_seq", "_leaf", "_rev"))}
        body = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode("utf-8")
        vid = "%s~%s" % (root, hashlib.blake2b(body, digest_size=8).hexdigest())
        doc.pop("id", None)
        doc["id"] = vid
        if vid == artifact_id:
            return prev                      # identical content — the revision is a no-op

        # ORDER MATTERS: write the new version BEFORE archiving the old. A crash between them
        # leaves TWO committed versions (visible, reconcilable) rather than ZERO (an artifact
        # that momentarily does not exist for readers resolving the head).
        self.put_artifact(doc)
        old = dict(prev)
        old["state"] = "archived"
        old["superseded_by"] = vid
        self.put_artifact(old)
        return doc

    def head_of(self, root_id: str) -> Optional[Dict[str, Any]]:
        """The current version of a lineage — John's "version_id is latest".

        ⚠ Returns None for an unknown root AND for a lineage whose every version is archived.
        Those are different situations and neither is an error, but a caller that treats None as
        "not found" while the rows exist will report absence for content that is present.
        """
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE root_id = ? ORDER BY _origin, _seq", (root_id,)
        ).fetchall()
        live = [json.loads(r["doc"]) for r in rows]
        live = [d for d in live if d.get("state") != "archived"]
        return live[-1] if live else None

    def versions_of(self, root_id: str) -> List[Dict[str, Any]]:
        """Every version in a lineage, oldest first. The `y` colimit (John: "each version is
        another order - the anti-entropy projection").

        Ordered by `(_origin, _seq)` — the globally-unique version identity, gap-free per origin.
        NOT by `created_time`, which is a CLAIM by the writer and is exactly what the deprecation
        gate is trying to stop depending on.
        """
        rows = self.db.read().execute(
            "SELECT doc FROM vertex WHERE root_id = ? ORDER BY _origin, _seq", (root_id,)
        ).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    def rebuild_list_index(self, *, chunk: int = 5000) -> Dict[str, Any]:
        """One-time backfill of `listkey` for a store that predates it. Returns what it did.

        This is the ONLY sanctioned way to earn the build marker on a populated store, and it is
        a WRITE — a store opened read-only cannot be repaired into answering, which is correct:
        the alternative is answering wrongly.

        Walks by keyset (`WHERE id > :cur ORDER BY id`), never `OFFSET`, and commits per chunk so
        a 6M-row corpus does not hold one transaction open for the whole backfill. The marker is
        set only after the final chunk, so an interrupted rebuild leaves the store UNCERTIFIED
        and still refusing — a half-built index that answered would be worse than one that
        does not, because its wrong answers would be plausible."""
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
                       now_iso: Optional[str] = None) -> List[Dict[str, Any]]:
        """Head of the pending queue for `content_type`, retry-backoff already applied.

        Replaces `pool.py:84`. The shim's guard required the literal
        `"order by priority"`, which `pool.claim` had REMOVED — so the dispatch fell
        through to `[]` and **the queue was silently dead on sqlite**: workers idled
        48 ticks and self-exited while `queue_stats` reported healthy pending counts.

        `ORDER BY priority DESC, id` is free here — `ix_t_pending` is
        `(ct, status, priority DESC, id)`, so this is a bounded index walk with no
        sort. The legacy store materialised and sorted the ENTIRE task history per
        worker per 5s and then `random.shuffle`d the result away. Callers still
        shuffle this window to avoid claim contention; that is a herd-avoidance
        measure, not an ordering opinion."""
        params: List[Any] = [content_type]
        clause = ""
        if now_iso is not None:
            clause = " AND (next_retry_at IS NULL OR next_retry_at <= ?)"
            params.append(now_iso)
        params.append(int(limit))
        rows = self.db.read().execute(
            "SELECT id, priority, next_retry_at FROM task "
            "WHERE ct = ? AND status = 'pending'" + clause +
            " ORDER BY priority DESC, id LIMIT ?", params).fetchall()
        return [dict(r) for r in rows]

    def try_claim(self, task_id: str, *, worker_id: str, now_iso: str) -> bool:
        """Atomically move ONE task pending -> claimed. `True` iff THIS caller won.

        The `WHERE status = 'pending'` predicate plus SQLite's single-writer lock
        makes this a genuine compare-and-set: concurrent workers see `rowcount == 0`
        and move on. Never returns a task the caller did not win, so the caller does
        not need pool.py's defensive `get_artifact(...)['claimed_by'] == me` re-read."""
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

        A `False` here is INFORMATION — the lease was already reclaimed and the
        caller is working on a task someone else now owns. pool.py swallowed this
        path in a bare `except: pass`."""
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
        Replaces the shim's `select claimed_by, operator, task_key, claimed_at`
        branch. Ordering happens in SQL over the small claimed set."""
        rows = self.db.read().execute(
            "SELECT id, claimed_by, operator, task_key, claimed_at FROM task "
            "WHERE ct = ? AND status = 'claimed' ORDER BY claimed_by, id LIMIT ?",
            (content_type, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def recent_terminal(self, content_type: str, *, limit: int = 8,
                        statuses: Iterable[str] = ("done", "failed")) -> List[Dict[str, Any]]:
        """Most recently completed/failed tasks.

        `ix_t_terminal` is `(ct, status, completed_at DESC)`, so each status bucket
        yields its top-N as an index walk — no sort, and no index on the whole
        never-pruned done+failed set. The legacy store sorted that entire
        unbounded set per /status load just to display 8 rows."""
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

    def count_by_status(self, content_type: str,
                        statuses: Iterable[str] = ("pending", "claimed", "done",
                                                   "failed", "dead")) -> Dict[str, int]:
        """Queue depth per status. Five counter lookups, no scan, no `count(*)`.
        Replaces `pool.queue_stats`, which issued one `count(*)` per status per
        /status load and per supervise cycle."""
        return {st: _seq.counter_of(self.db, _schema.c_task_status(content_type, st))
                for st in statuses}

    def _patch_doc(self, cur: sqlite3.Cursor, aid: str, fields: Dict[str, Any]) -> None:
        """Keep `doc` JSON in step with a sidecar-only UPDATE.

        The doc is the artifact of record; the `task` table is a projection. They
        are updated in ONE transaction so they cannot disagree — the seed store's
        column/doc divergence is what silently orphaned claimed tasks forever.

        This does NOT allocate a new `_seq`: a lease renewal is bookkeeping, not an
        authored event, and stamping one would churn the merkle leaf and republish
        the row on every heartbeat. A CLAIM does change the artifact, so callers
        that want the claim replicated should follow with an explicit
        `put_artifact(..., stamp_rev=True)`."""
        r = cur.execute("SELECT ct, doc FROM vertex WHERE id = ?", (aid,)).fetchone()
        if r is None:
            return
        d = json.loads(r["doc"])
        d.update(fields)
        cur.execute("UPDATE vertex SET doc = ? WHERE id = ?", (json.dumps(d), aid))
        # ⚠ THIS IS THE ONE WRITE THAT BYPASSES `_write_row`, SO IT MUST MAINTAIN THE KEYED
        # INDEX ITSELF. Today it only ever patches task coordination fields (status, claimed_by,
        # claimed_at, next_retry_at), none of which are indexed — so this is a no-op in practice.
        # It is here because "in practice" is doing all the work in that sentence: the day
        # someone patches a field that IS indexed, `listkey` would silently disagree with `doc`,
        # and a keyed index that disagrees with its table answers confidently and wrongly. Cheap
        # to keep correct now, invisible to debug later.
        if any(k in self.LIST_INDEX_FIELDS for k in fields):
            self._index_lists(cur, aid, d, r["ct"])

    # ── merkle ───────────────────────────────────────────────────────────────
    def merkle_leaves(self) -> List[int]:
        """The incrementally maintained leaf digests. NOT a rescan.

        ⚠ Roots computed here will NOT equal the legacy graph engine node's roots, because
        `row_hash` now hashes `(id, _seq)` rather than `(id, _rev)` — contract
        RESOLVED-1. Phase 5.0 is a rebuild, not a byte-copy. Do not report that
        inequality as a migration failure."""
        acc = [0] * self.leaves
        for r in self.db.read().execute("SELECT leaf, digest FROM leaf_digest"):
            leaf = int(r["leaf"])
            if 0 <= leaf < self.leaves:
                acc[leaf] = int.from_bytes(r["digest"], "big")
        return acc

    def verify_counters(self, *, repair: bool = False) -> Dict[str, Any]:
        """Recompute every vertex counter from the rows and report DRIFT.

        **VERIFICATION AND BOOTSTRAP ONLY — this is a full scan.** It exists because
        the counters are now load-bearing: `count_missing_field` IS the provenance
        audit, and a counter that has silently drifted reports a clean audit over a
        dirty corpus, which is precisely the wrong-answer-not-empty-answer class this
        whole unit exists to eliminate. A number nothing can check is not a
        measurement. Run it from `node-repair.py`, never on a request path.

        Also the migration path: counters are maintained from the moment a row is
        written, so a store whose rows predate a counter (a new audited field, a file
        restored from before this table) reads 0 until this runs.

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
            # `rows:<origin>` spans BOTH tables, so the edge side must be counted
            # too or every store with edges reports permanent phantom drift.
            for r in cur.execute("SELECT _origin FROM edge"):
                if r["_origin"]:
                    add(_schema.c_rows(str(r["_origin"])), 1)
                add(_schema.c_edge_total(), 1)
            # The lexical index IS derivable from its own map, so it is JUDGED rather than
            # excused. Excluding it would have made `fts:total` a number nothing can check, and
            # the whole point of this method is that a load-bearing counter must be checkable.
            #
            # 🔴 BUT `fts:total` vs `fts_map` IS A CHECK THAT COULD NOT FAIL FOR THE DEFECT IT
            # MATTERED FOR (found 2026-07-30). Both sides are the INDEX'S OWN bookkeeping, and the
            # missing-retraction defect above left them BOTH standing — the counter agreed with the
            # map, so `drift` was empty while node 71 carried `fts:total` 2,030,948 against
            # `vertex` 2,030,946. Self-consistency between two derived artifacts is not
            # verification; the index has to be judged against the ROWS it claims to index.
            # So the map row is now resolved against `vertex` in the same pass — an index-only PK
            # probe, and NO `count(*)` (banned package-wide; the tally is kept in Python).
            fts_orphans = 0
            try:
                for _rid, _resolved in cur.execute(
                        "SELECT m.vertex_id, v.id FROM %s m "
                        "LEFT JOIN vertex v ON v.id = m.vertex_id" % _fts.MAP_TABLE):
                    add(_fts.C_FTS_TOTAL, 1)
                    if _resolved is None:
                        fts_orphans += 1
            except Exception:
                pass                    # no index table yet: actual 0, and drift will say so
            stored = {row["name"]: int(row["n"])
                      for row in cur.execute("SELECT name, n FROM counter")}
            # Not derivable from the rows, so not judged here:
            #   task:*      maintained from the `task` sidecar
            #   conflict:*  records decisions, not rows
            #   vacated:*   HISTORICAL — a vacated seq has no row left to count, so
            #               it can only ever be accumulated forward. That is why
            #               `seq_accounting(scan=True)` is the check that can see a
            #               row lost outside the write path: it compares the SCANNED
            #               live count against `last_seq - vacated`, which no
            #               recomputation of `vacated` could ever do.
            #   listkey:*   A STATE MARKER, NOT A COUNT. `listkey:built` is a boolean
            #               ("this store's keyed index covers every row") that happens
            #               to live in the counter table because that is where this
            #               store keeps durable scalars. Recomputing it from the rows
            #               yields 0 — there is no row-derived quantity it corresponds
            #               to — so it would be reported as permanent drift on every
            #               healthy store, and `repair=True` would then WRITE that 0
            #               back and silently un-certify a correctly built index,
            #               turning every subsequent keyed lookup into a refusal.
            drift = {}
            for name in set(stored) | set(actual):
                if name.startswith(("task:", "conflict:", "vacated:", "listkey:")):
                    continue
                s, a = stored.get(name, 0), actual.get(name, 0)
                if s != a:
                    drift[name] = (s, a)
            if repair and drift:
                for name, (_s, a) in drift.items():
                    cur.execute(
                        "INSERT INTO counter(name, n) VALUES(?,?) "
                        "ON CONFLICT(name) DO UPDATE SET n = excluded.n", (name, a))
        out = {"scanned": scanned, "drift": drift, "repaired": bool(repair and drift),
               # Map rows pointing at a vertex that is gone. A hit on one of these resolves to
               # `get_artifact() -> None` — a WRONG answer, not an empty one — and no counter
               # comparison can see it, because the counter and the map drift together.
               # NOT auto-repaired: dropping index rows is a destructive act, and this number's
               # job is to make the condition visible to `node-repair.py`, which decides.
               "fts_orphans": fts_orphans}
        # The allocation-accounting invariant, verified against the ROWS (scan=True)
        # rather than the counters — a counter-vs-counter comparison could never
        # detect a row that vanished outside the write path.
        out["seq_accounting"] = self.seq_accounting(scan=True)
        return out

    def seq_accounting(self, *, origin: Optional[str] = None,
                       scan: bool = False) -> Dict[str, Any]:
        """`live_rows + vacated == last_seq` for one origin. See `seq.seq_accounting`.

        **This is the accessor `node-repair.py` should call** — it returns every
        term plus a `balanced` verdict, so no caller has to reconstruct the
        arithmetic (and `last_seq` holds the LAST issued seq, not the next, which
        is an easy off-by-one to write by hand).

        Use `scan=True` in node-repair: it recomputes `live_rows` from the rows and
        is the only mode that can see a row lost outside the write path."""
        return _seq.seq_accounting(self.db, origin or self.origin, scan=scan)

    def rebuild_merkle(self) -> int:
        """Full rescan into `leaf_digest`. BOOTSTRAP AND VERIFICATION ONLY.

        MEASURED on 71: a rescan publish ran at 6,286 rows/sec (~7 min for 2.7M
        rows) while the corpus grew 2.23M -> 2.73M in the same session, so a
        rescan-built tree is stale before it finishes on a catching-up node.
        Steady state is the incremental path in `_write_row`."""
        acc: Dict[int, int] = {}
        with self.db.write() as cur:
            cur.execute("DELETE FROM leaf_digest")
            n = 0
            for r in cur.execute("SELECT id, _leaf, _seq FROM vertex"):
                leaf = int(r["_leaf"]) if r["_leaf"] is not None else K.leaf_of(r["id"], self.leaves)
                acc[leaf] = acc.get(leaf, 0) ^ K.row_hash(r["id"], r["_seq"])
                n += 1
            # ONE tree covers BOTH tables, so a rebuild must XOR the edges back in too — otherwise a
            # rebuild would silently DROP every edge and the "rescan == incremental" invariant would
            # fail the moment any edge exists. Edge identity is NODE-INVARIANT (edge_hash excludes
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
        """Reshard IFF the corpus has crossed a sqrt boundary since the last one. Cheap to call every
        publish cycle — a counter read and a compare — so only the rare boundary crossing pays the
        O(N) re-stamp. This is what keeps the operating leaf count equal to the DERIVED
        `natural_leaves(corpus)` as the store grows, without a background job or a magic constant."""
        n = self.count() + (graph.count_edges() if graph is not None else 0)
        if K.natural_leaves(n) == self.leaves:
            return {"resharded": False, "leaves": self.leaves}
        return self.reshard(graph=graph)

    def reshard(self, *, graph: Any = None, target: Optional[int] = None) -> Dict[str, Any]:
        """Re-stamp `_leaf` on every row and rebuild the tree at the DERIVED resolution
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
