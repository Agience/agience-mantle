"""Anti-entropy mesh sync — memory-bounded, content-addressed, adaptive.

Two peers converge by comparing a compact DIGEST, never by re-scanning each other. The digest is a
per-bucket XOR of artifact-id hashes: XOR is commutative (order-independent → streamable) and
homomorphic over sets, so two boxes holding the same id-set produce identical per-bucket values, and a
bucket whose values differ is exactly where their sets diverge. Only the artifacts MISSING on the
puller are transferred. O(diff), not the O(N^2) SKIP crawl of op.mesh.pull.

Memory bounds (the hard rule — nothing loads the whole DB):
  • digest  = a fixed array of `nbuckets` ints (+counts). Independent of DB size. The scan pages ids
              in bounded chunks; only one page of ids is in memory at a time.
  • diff    = compare two digest arrays (small).
  • resolve = per differing bucket, exchange only that bucket's id list (~ids/bucket) and fetch only
              the missing docs, in batches.

Adaptive: `nbuckets` targets ~64 ids/bucket (Merkle-style — deepen as the corpus grows), so a bucket's
id list stays small from 10^6 to 10^8 artifacts. Everything is a fraction of a measured limit.

Direction-agnostic: the same primitive pulls peers→71 (71 prioritized) AND 71→peers (peers go full).
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional
from prism.canonical import canonical_string as _jcs_string

_OP_EXCLUDE = {                # operational state is per-box, never replicated (would pollute the mesh)
    "application/vnd.agience.task+json", "application/vnd.agience.mesh-cursor+json",
    "application/vnd.agience.content-cursor+json", "application/vnd.agience.shard-done+json",
    "application/vnd.agience.s3sync-cursor+json",   # S3-sync publish/consume cursors are per-box
    # ⚠ ADDED 2026-07-20 by John's ruling, found by the LATTICE migration smoke test.
    # `application/x-ember-state` is node-local operational state (`ember.restart.budget`) and it
    # was the ONLY such type still crossing the mesh — every sibling above is already excluded.
    # It surfaced because the migration's type-declaration gate (P4) refused to describe a type
    # in use with no declared format, which forced someone to look at what it actually was.
    "application/x-ember-state",
}
# ⚠ THIS SET IS HELD IN LOCKSTEP WITH `db/arcade.py::_NO_LEAF_TYPES` by
# `test_merkle.py::test_op_exclude_sets_are_in_lockstep`, and that check is RIGHT: a type excluded
# here but still `_leaf`-stamped there would be hashed into the Arcade replication tree while this
# node kept it out, so two converged nodes could never agree on a root. Adding an entry here is
# therefore a TWO-REPO change. Unit E does not own `arcade.py`, which is why the declined-version
# marker below reuses `_S3SYNC_CT` instead of minting a new content type — it is genuinely a
# per-box sync cursor, so the honest name and the available name coincide.


def _is_replicated(content_type) -> bool:
    """Whether an artifact type crosses the mesh at all.

    Probe/throwaway types are excluded BY PREFIX rather than by exact name because the damage they
    did was disproportionate: leftover `probe-tw-rev-*` fixtures with `_rev` four months in the
    future replicated to all five nodes and pinned publish cursors beyond every real row, muting
    nodes permanently. A test fixture must never be able to reach production replication, and
    matching on a prefix means a future probe type nobody remembered to register is still safe."""
    ct = str(content_type or "")
    if ct in _OP_EXCLUDE:
        return False
    return not (ct.startswith("application/x-probe")
                or ct.startswith("application/vnd.agience.probe")
                or "throwaway" in ct)


# ── the store: THE LATTICE, and only the lattice ──────────────────────────────────────────────────
# ⛔ THIS WAS A TWO-BACKEND SEAM and it is gone [John, 2026-07-23: "leave one path. the only path.
# No constants, no fitting. no forcing."]. Ten accessors each had an `else` branch carrying the
# ArcadeDB dialect (`SKIP`, `@rid`, `IN [...]`, `_rev`), kept so Phase 8 could "retire it by deleting
# ten branches". Phase 8 is now. A second branch is not free while it waits: every accessor had to be
# written twice, every doc had to say which engine it meant, and the two could drift silently because
# only one of them was ever exercised.
#
# ⚠ WHAT MUST NOT COME BACK IS THE EMPTY-LIST FALLBACK. `_SqliteConnShim` answered every unrecognised
# statement with `[]`, which is why contract §5 catalogues six WRONG-ANSWER defects rather than six
# crashes. A store that is not a lattice store RAISES. "I could not run this query" and "the query
# returned nothing" must never be the same value again.


# ⚠⚠ OPERATIONAL ROWS ARE NOT EVENTS IN ANY OBSERVER'S PROPER TIME. READ THIS BEFORE CHANGING IT.
#
# Every cursor in this file was written with `put_artifact(..., stamp_rev=False)`, which under `_rev`
# meant "do not stamp a mesh revision on this row" — correct, because `_S3SYNC_CT` is in
# `_OP_EXCLUDE` and a cursor must never reach the mesh. Under `(_origin, _seq)` the SAME KEYWORD
# MEANS SOMETHING ELSE: "preserve the version this doc arrived with". A locally-authored cursor
# carries no version, so the lattice store raises ValueError — correctly, since re-stamping would
# claim this node authored a peer's event. All 20 cursor writes in this file hit that.
#
# `stamp_rev=True` is NOT the fix, and the reason is a live-lock, not a preference. A stamped cursor
# allocates a fresh `_seq` on every publish cycle, so the publish scan's next page ALWAYS finds at
# least one row — the cursor it just wrote. It is filtered out of the segment by `_is_replicated`,
# the cursor advances past it, and writing that advance allocates the next `_seq`. The feed can
# never read idle and `publish_backlog` can never reach 0: the convergence signal becomes
# structurally unreachable, which is the exact defect `mesh_lag` was taught about when it asymptoted
# to 49 on a retired stream and `converged` could never become True.
#
# So operational rows are stamped with a RESERVED origin and `_seq = 0`. This is not a trick to get
# past a validator — it is the accurate statement. `_origin` names the AUTHORING OBSERVER of a
# replicated event; a publish cursor is not one. Pinning it outside every real observer's sequence
# makes `page_by_origin(origin=me)` skip it structurally rather than by a filter someone can forget,
# so operational state cannot enter the publish feed even by accident.
#
# ⚠ THE VERSION IDENTITY IS `("_local:<id>", 1)` — ONE DEGENERATE ORIGIN PER OPERATIONAL ROW.
# Two earlier shapes were tried against node-repair and both were wrong, so do not "simplify" this
# back to either:
#
#   ("_local", 0) for every row  -> FAILS `(_origin,_seq) unique`. A version identity must be
#                                   unique, and a shared constant gives every cursor the same one.
#   ("_local", hash(id))         -> WARNS on `_seq contiguity (peers)`, forever, on every healthy
#                                   node. `_local` then looks like a peer observer whose proper time
#                                   is full of enormous gaps, and the check cannot tell that from
#                                   "a consume cursor advanced past a segment that did not apply" —
#                                   which is the one thing that check exists to catch. An alarm that
#                                   fires permanently is an alarm nobody reads.
#
# Giving each operational row its own origin makes its sequence trivially `1..1` — gap-free by
# construction, unique by construction, stable across rewrites (so a cursor update does not churn a
# merkle leaf), and still invisible to `page_by_origin(origin=me)`. It is also the accurate
# statement: each of these rows is its own degenerate non-observer, not a member of some shared
# pseudo-observer's timeline.
_LOCAL_ORIGIN = "_local"        # reserved prefix; never a real node id, never replicated


def _op_origin(artifact_id: str) -> str:
    return "%s:%s" % (_LOCAL_ORIGIN, artifact_id)


# ⚠⚠ CONSUMED EDGES GET A RESERVED ORIGIN TOO. THE ONE THING THAT MUST NEVER CHANGE IS THAT IT IS
# NOT `me`. Read this before "simplifying" `_apply_edges`.
#
# The edge segment format does NOT carry `(_origin, _seq)` — an edge is `{f, t, label, props}` on
# the wire and nothing else — so a consumed edge arrives with no provenance to preserve. That is a
# real gap and John ruled on it directly: edge provenance was never guaranteed, so the edge
# `_origin` does not need to be ACCURATE (contract §5.8.2). It does need to not be `me`.
#
# The reason is not tidiness, it is echo suppression, and it is the SAME mechanism `_op_origin`
# relies on. `page_by_origin(origin=me, after_seq=cursor)` is the publish scan. A row whose
# `_origin` is not `me` is excluded from it STRUCTURALLY — by the index range, not by a filter
# someone can forget. Stamp a consumed edge as locally authored (which is what
# `add_edges(edges)` with the default `stamp_rev=True` does) and every one of the fleet's 246,784
# edges enters this node's publish feed: A publishes, B consumes and re-authors, B publishes,
# A consumes and re-authors, forever. And because `row_hash` keys on `_seq` (contract RESOLVED-1),
# every re-authoring churns a merkle leaf, so anti-entropy chases a difference it is itself
# creating and never converges.
#
# THE SHAPE IS `("_local:edge:<digest>", 1)` — one degenerate origin per edge, exactly as
# `_op_origin` gives one per operational row, and for exactly the reasons recorded above:
#   ("_local", 0) for every edge  -> FAILS `(_origin,_seq) unique`
#   ("_local", hash(key))         -> WARNs `_seq contiguity (peers)` forever
# Its own origin makes each edge's sequence trivially `1..1`: unique and gap-free by construction.
#
# `<digest>` is DETERMINISTIC in the edge triple (contract §3's NUL-separated blake2b, the same
# construction as `edge_key`), and that buys the property this whole change is measured on:
# re-consuming a segment produces the IDENTICAL `(_origin, _seq)`, so `add_edges(stamp_rev=False)`
# compares versions, reads SAME, and DOES NOT WRITE. A replayed segment allocates no proper time
# and churns no leaf. Under the old code every replay allocated a fresh `_seq` per edge.
#
# ⚠ DO NOT generalise this to incoming ARTIFACTS. Stamping a local version on a consumed artifact
# was tried, shipped fleet-wide, and REVERTED (see `_apply_artifacts` below): it disables the
# anti-downgrade guard, so an ancient backlog copy silently overwrites the current row. Edges are
# different in the way that matters — they are idempotent by `edge_key` and carry no mutable
# payload, so there is no downgrade for the guard to prevent.
_CONSUMED_EDGE_NS = _LOCAL_ORIGIN + ":edge:"


def _consumed_edge_origin(src: str, dst: str, label: str) -> str:
    """A reserved, deterministic, per-edge origin. NUL-separated, per contract §3 — the separator
    is load-bearing: without it ("ab","c") and ("a","bc") hash alike."""
    h = hashlib.blake2b(("%s\0%s\0%s" % (src, dst, label)).encode("utf-8"), digest_size=16)
    return _CONSUMED_EDGE_NS + h.hexdigest()


def _put_op(store, doc: Dict[str, Any]) -> None:
    """Write a LOCAL OPERATIONAL row (cursor, watermark). Never replicated, never versioned.

    One function for all 20 sites, so the invariant is stated once and cannot drift between them —
    the two cursor writes in `publish_to_s3` were once the only ones in this file missing
    `stamp_rev=False`, and that one inconsistency dragged the publish watermark past uncommitted
    writes and silently lost them."""
    # A row written through here is pinned OUTSIDE every observer's proper time, so if it were a
    # replicated type it would be unpublishable and invisible to every peer — silently. Catch the
    # mistake at the write instead of discovering it as missing data.
    if _is_replicated(doc.get("content_type")):
        raise ValueError(
            "_put_op refuses id=%r content_type=%r: that type REPLICATES, and an operational write "
            "pins (_origin,_seq) outside every observer's sequence — the row could never be "
            "published. Use put_artifact(stamp_rev=True), or add the type to _OP_EXCLUDE."
            % (doc.get("id"), doc.get("content_type")))
    if _vertices(store) is not None:
        d = dict(doc)
        d["_origin"] = _op_origin(doc.get("id"))
        d["_seq"] = 1
        store.artifacts.put_artifact(d, stamp_rev=False)
        return
    store.artifacts.put_artifact(doc, stamp_rev=False)


def _vertices(store):
    """The typed vertex store if this is a lattice node, else None.

    `page_by_origin` is the discriminator because it is the method the whole `_rev` -> `(_origin,
    _seq)` conversion is built on: a store that has it can serve every accessor below, and a store
    that lacks it cannot serve any of them."""
    a = getattr(store, "artifacts", None)
    return a if hasattr(a, "page_by_origin") else None


def _edges(store):
    g = getattr(store, "graph", None)
    return g if hasattr(g, "page_by_origin") else None


def _store_leaves(store) -> int:
    """This store's DERIVED Merkle resolution — `natural_leaves(corpus)`, held on the store and kept
    current by `reshard`. The single source of truth for how many leaves the tree has; the whole mesh
    reads it here rather than a constant, so there is no `4096` to drift against."""
    from . import merkle
    v = _vertices(store)
    n = int(getattr(v, "leaves", 0) or 0) if v is not None else 0
    return n if n > 0 else merkle.DEFAULT_LEAVES


def _require_lattice(store_part, what: str):
    """The typed vertex/edge store, or a hard error naming what could not be served.

    Never a fallback and never an empty result: an unservable query and an empty answer are
    different facts, and conflating them is the defect class this whole module is written against."""
    if store_part is None:
        raise RuntimeError(
            "mesh.sync: %s needs the lattice store's typed methods and this store has none. "
            "Refusing to return an empty result." % (what,))
    return store_part


def _origin_of(store) -> str:
    """This observer's identity as the STORE understands it.

    Deliberately the store's `origin`, not `_node_id()`. `_seq` is scoped `WHERE _origin = :me`, so
    a publish scan keyed on a different string than the one stamped at write returns zero rows —
    forever, silently, on a node that is writing normally. The env var and the store must agree, and
    when they do not it is the store that is right, because the store is what stamped the rows."""
    v = _vertices(store)
    return str(getattr(v, "origin", "") or "") if v is not None else _node_id()


# ── the publish backlog — Merkle-native (one path) ───────────────────────────────────────────────
# "Unpublished local work" is no longer a feed-cursor question. There are no feed cursors: the tree
# maintained incrementally on write (`_MERKLE_LIVE`) minus what is actually backed by an uploaded
# leaf object (`_MERKLE_PUBLISHED`) IS the backlog — the CHANGED LEAVES a peer cannot yet see. It is
# O(leaves) integer compares, immune to the seq-vacancy and two-cursor defects the feed backlog
# spent three rewrites on, because it counts leaf STATE, never subtracts seq allocations.


def publish_backlog_now(store) -> Dict[str, Any]:
    """How many of this node's Merkle leaves have CHANGED but are not yet published (LIVE != PUBLISHED).

    Read by health monitoring (`stats._activity`). Returns `publish_backlog_unset` — never 0 — before
    any tree has been built, so a node that has never published does not read as converged (the
    unit-C+D rule, preserved). Exact by construction: it compares two leaf arrays, it does not
    subtract seq allocations, so update churn cannot inflate it."""
    cur = store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}
    live = cur.get(_MERKLE_LIVE)
    if not live:
        return {"publish_backlog_unset": True, "reason": "no-merkle-tree-yet"}
    live = [int(x) for x in live]
    prev = cur.get(_MERKLE_PUBLISHED) or []
    pub = ([int(x) for x in prev] + [-1] * len(live))[:len(live)]     # -1 = never uploaded
    behind = sum(1 for i in range(len(live)) if pub[i] != live[i])
    published = len(live) - behind
    return {"publish_backlog": behind, "publish_backlog_exact": True,
            "publish_backlog_at": time.time(),
            "publish_cursor": published,        # leaves backed by an uploaded object
            "basis": "merkle: changed leaves not yet published"}


def _total_count(store) -> int:
    """Every row, replicated or not. A COUNTER LOOKUP on lattice — no `count(*)` reaches SQLite.

    On the live corpus EXPLAIN shows `count(*)` loading 6M records to produce one integer, and on
    node 71 that OOMs the acceptor thread and zombies the node."""
    v = _vertices(store)
    if v is not None:
        return int(v.count())
    return int(_require_lattice(v, "_total_count"))


def _count_of_type(store, ct: str) -> int:
    v = _vertices(store)
    if v is not None:
        return int(v.count_by_content_type(ct))
    return int(_require_lattice(v, "_count_of_type"))

# ── S3 mesh plane — S3 is the AUTHORITATIVE store AND the exchange plane (John's decision) ────────────
#   Content is content-addressed in `cas/<sha>`. The GRAPH is authoritative in S3 too, as each node's
#   MERKLE TREE: a 32 KB summary plus per-leaf objects. One path — no segment log, no cursors. A box
#   publishes its tree and pulls the leaves that differ; a wiped/new box rebuilds by the same pull.
#   Every object is Fernet ciphertext under the shared content.key (OVH sees opaque bytes).
_MESH_MERKLE_PREFIX = "mesh/merkle/"     # per-node tree summary (root + leaf digests) — 32 KB, JSON
_MESH_LEAF_PREFIX = "mesh/leaf/"         # per-node, per-leaf row set (vertices + edges) — unit of transfer
_S3_MERKLE_CURSOR = "s3.merkle.cursor"
# TWO DISTINCT DIGEST KEYS ON THAT CURSOR — they are NOT the same thing and conflating them made
# publish silently skip uploads:
#   digests   = LIVE: what the local corpus hashes to right now. `refresh_leaves` writes this after
#               every reconcile/mutation WITHOUT uploading anything to S3.
#   published = what is actually BACKED BY AN UPLOADED LEAF OBJECT in S3.
# `changed` must be computed against `published`. Computed against `digests`, a leaf that
# refresh_leaves had already recomputed looks unchanged, so no leaf file is ever written — yet the
# root advertises it. Peers then fetch a key that 404s and stay divergent forever, silently.
_MERKLE_LIVE = "digests"
_MERKLE_PUBLISHED = "published"
_S3SYNC_CT = "application/vnd.agience.s3sync-cursor+json"


def _mesh_s3(store):
    """The authoritative OVH S3 client (a GarageContentStore: has ._s3 + .bucket + get/put/exists).
    Reuse the tiered content store's origin if present, else open OVH directly. None if creds absent."""
    remote = getattr(getattr(store, "content", None), "remote", None)
    if remote is not None:
        return remote
    try:
        from mantle.shard.content_tier import open_ovh_store
        return open_ovh_store(store.keys_dir)
    except Exception:
        return None


def _node_id() -> str:
    try:
        from prism.envelope import node_id as nid
        return nid()
    except Exception:
        import os
        return os.getenv("EMBER_NODE_ID") or "node"


# ⛔ `publish_tail_to_s3` STOOD HERE AND IS DELETED. It published the NEWEST rows first, keyed on
# `@rid` — ArcadeDB's PHYSICAL row address — because the backfill pass started at the beginning of
# the @rid space and a fresh write sat behind the entire history. A lattice store has no @rid, so
# on every store that exists the function's first act was to return
# `{"retired": True, "reason": "no-@rid-on-lattice", "use": "publish_updates_to_s3"}`.
# Its whole body — ~120 lines of measured @rid paging notes — was unreachable.
# `publish_updates_to_s3` is the feed. [John, 2026-07-23: "dont just retire. remove it."]


def _ignored_nodes() -> set:
    """Publisher ids to skip entirely (`EMBER_MESH_IGNORE`, comma-separated).

    Before identity failed closed, any process started without `EMBER_NODE_ID` published under its
    HOSTNAME — so the mesh accumulated phantom publishers that are really an existing node wearing a
    different name (`JOHN-HOME-LT` = 71, `e8ae2618c8ac` = t5's container id). Their segments are
    byte-duplicates of the real node's, so every peer burns real cycles re-applying data it already
    has: 190 graph segments of JOHN-HOME-LT alone, on every consumer, forever.

    Skipping is deliberately preferred over DELETING the S3 segments: deletion is irreversible and
    these streams are the only copy of anything they contain that was written while the node was
    mis-identified. Ignoring is reversible, needs no coordination, and costs nothing to undo."""
    import os
    raw = os.getenv("EMBER_MESH_IGNORE", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def mesh_lag(store) -> Dict[str, Any]:
    """How many LEAVES differ from each peer — the Merkle-native 'how far behind'.

    READ-ONLY: fetch each peer's 32 KB tree summary and diff it against my LIVE tree; never pull.
    This is the mesh's self-observation, the signal the 5-minute online SLO is measured against.
    `converged` is the POSITIVE fact that every peer's root matched (0 differing leaves) — and it is
    None (UNKNOWN), NEVER True, whenever a listing failed or a peer's tree could not be read. The SLO
    must never read green because it went blind. `segments_behind` keeps its name for the health
    board and worker; the unit is now differing LEAVES (there are no segments)."""
    import json
    from . import merkle
    s3 = _mesh_s3(store)
    if s3 is None:
        return {"reason": "no-s3"}
    me = _node_id()
    skip = _ignored_nodes()
    local = (store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}).get(_MERKLE_LIVE)
    if not local:
        # No local tree yet -> nothing to compare -> UNKNOWN, not converged.
        return {"node": me, "peers": {}, "segments_behind": 0, "converged": None,
                "blind": ["no-local-tree"], "stuck": []}
    local = [int(x) for x in local]
    peers: Dict[str, Any] = {}
    total, blind = 0, []
    try:
        keys = _s3_keys_after(s3, _MESH_MERKLE_PREFIX)     # flat `mesh/merkle/<node>.json` objects
    except MeshListingError as e:
        # CANNOT LIST != CAUGHT UP. Refuse to call this converged.
        return {"node": me, "peers": {}, "segments_behind": 0, "converged": None,
                "blind": ["merkle: %s" % str(e)[:120]], "stuck": []}
    for key in keys:
        node = key[len(_MESH_MERKLE_PREFIX):].strip("/")
        if not node.endswith(".json") or "/" in node:
            continue
        node = node[:-len(".json")]
        if not node or node == me or node in skip:
            continue
        try:
            theirs = merkle.load(json.loads(
                s3.get("%s%s.json" % (_MESH_MERKLE_PREFIX, node)).decode("utf-8")))
        except Exception as e:
            blind.append("%s: %s" % (node, str(e)[:80]))     # unreadable peer tree -> blind, not 0
            continue
        if not theirs:
            blind.append("%s: tree-shape" % node)
            continue
        d = len(merkle.diff(local, theirs))     # folds to the common resolution — different-sized ok
        peers[node] = {"differing": d}
        total += d
    return {"node": me, "peers": peers, "segments_behind": total,
            "converged": None if blind else (total == 0),
            "blind": blind, "stuck": []}


def _fernet(store):
    from mantle.shard import content as C
    return C._content_key(store.keys_dir)


def _proj(a: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in a.items() if not str(k).startswith("@")}


    # ⛔ THE ARCADE EDGE FEED FOLLOWED THIS LINE — a composite `(_rev, from, to, label)` cursor
    # walked with cypher, ~90 lines that could not be reached: the delegation above returns
    # unconditionally on any store with a lattice graph, which is every store there is.
    # [John, 2026-07-23: "leave one path. the only path."]


class MeshListingError(RuntimeError):
    """S3 could not be LISTED. Distinct from 'S3 listed successfully and was empty'.

    These two were indistinguishable: both helpers below swallowed every exception and returned [],
    so an unreachable bucket produced 'no peers, no segments' -> `total_behind == 0` -> and
    `mesh_lag` reported **converged: True EXACTLY WHEN THE MESH WAS BLIND**. The SLO signal read
    healthiest at the moment it knew least. Raising instead makes the caller decide, and no caller
    is allowed to decide 'converged'."""


def _s3_keys_after(s3, prefix: str, after_key: str = "") -> List[str]:
    out: List[str] = []
    try:
        pag = s3._s3.get_paginator("list_objects_v2")
        kw = {"Bucket": s3.bucket, "Prefix": prefix}
        if after_key:
            kw["StartAfter"] = after_key
        for page in pag.paginate(**kw):
            for o in page.get("Contents", []) or []:
                out.append(o["Key"])
    except Exception as e:
        raise MeshListingError("list keys %r after %r failed: %s" % (prefix, after_key, e)) from e
    return out


def _new_docs(store, batch: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """The docs in `batch` whose id is NOT present locally — the GENUINELY-NEW rows, distinguished from
    no-op re-applies. Uses the SAME keyed `version_of` read `_split_unordered` already does (indexed by id
    PK), so it is cheap and backend-agnostic. Returns None on a non-lattice store (no `version_of`), where
    the split is not computable — an honest 'unknown', never a guess.

    ⭐ This is the split `_apply_artifacts` documents as computable-for-free. `applied` counts UPSERTS: a
    no-op re-apply of a row we already hold counts identically to genuinely-new knowledge, so `applied>0`
    cannot tell 'converging' from 'busily rewriting what we already have' (measured: ~723k applied for a NET
    GAIN OF 0). It is ALSO the gate #2b needs — peer-apply must reach the persona only on genuinely-new
    signals, never on the no-op re-applies. Computed by a READ: it does NOT touch the write path, `put_many`,
    or `_rev` (the 2026-07-20 write-path attempt was REVERTED for re-enabling a data-destruction mode; this
    is the safe, read-only form of the same information)."""
    v = _vertices(store)
    if v is None:
        return None
    new = []
    for doc in batch:
        try:
            local = v.version_of(str(doc.get("id")))
        except Exception:
            local = None
        if local is None:                            # absent locally → genuinely new (not a re-apply)
            new.append(doc)
    return new


def _apply_artifacts(store, items: List[Dict[str, Any]], *, stats: Optional[Dict[str, int]] = None) -> int:
    """Upsert consumed artifact docs. stamp_rev=False PRESERVES the origin `_rev` so a replicated row
    keeps its origin revision and does not echo around the mesh forever.

    `stats` (optional out-param) collects the GENUINELY-NEW split additively — `stats['new']` /
    `stats['reapplied']` (via `_new_docs`, a READ) — so a caller can tell real convergence from no-op
    re-applies. Populating it changes NOTHING about what is written or the load-bearing `handled` return /
    cursor guard; the default (None) is a pure no-op. The DEEPER wire — routing `new` into
    `worker._healthy_progress` (so 'progress' means new>0, not applied>0) and into #2b (fire peer-apply
    reaches only on the new docs) — is the consumer, flagged separately; this supplies the number safely."""
    batch = [d for d in items if d.get("id") and _is_replicated(d.get("content_type"))]
    if not batch:
        return 0
    nd = None
    if stats is not None or _DELIVER_PEER_SIGNALS:   # additive metric — a READ, never the write path
        nd = _new_docs(store, batch)
    if stats is not None and nd is not None:
        stats["new"] = stats.get("new", 0) + len(nd)
        stats["reapplied"] = stats.get("reapplied", 0) + (len(batch) - len(nd))
    # RETURN WHAT WAS ACTUALLY WRITTEN, not what we hoped to write. This returned len(batch)
    # regardless, so when put_many wrote nothing the caller still counted a full segment applied
    # AND advanced last_key past it — permanently losing the segment behind a monotone StartAfter
    # marker. That is precisely the loss the "HOLD THE CURSOR" logic below claims to have fixed;
    # this line quietly re-opened it one level up.
    # ⚠ `applied` COUNTS UPSERTS, NOT NEW KNOWLEDGE — measured, unfixed, deliberately.
    # On a node that already holds what its peers are backfilling, every one of these is a no-op
    # UPDATE. Measured on MANTLE 2026-07-20: ~723,000 docs "applied" across 12 consume cycles for
    # a NET GAIN OF 0 ROWS, while `segments_behind` ROSE 6,529 -> 6,924 the whole time. So this
    # number cannot distinguish "converging" from "busily rewriting what it already has", and
    # `worker._healthy_progress` reads `applied > 0` as proof of progress either way.
    # The new-vs-existing split IS computable for free (put_many's LWW path already reads which
    # ids exist locally) — but plumbing it out means adding a keyword to `put_many`, which is an
    # `ArtifactStore` interface method with four implementations, one of which (sqlite_store) does
    # not accept `stamp_rev` either. Attempted 2026-07-20 and REVERTED: the TypeError is swallowed
    # by reconcile_merkle's handler and silently becomes `applied: 0`. Widen `db/store.py` first.
    # ⛔ A ROW THAT ARRIVES WITH NO `_rev` IS INVISIBLE TO THE CONTENT DRAIN, PERMANENTLY.
    # `stamp_rev=False` exists to PRESERVE the origin's revision so a replicated row does not echo
    # around the mesh forever. But a doc arriving WITHOUT a `_rev` has no origin revision to
    # preserve, and writing it unstamped drops it below every `_rev` walk in the fleet: an LSM index
    # SKIPS NULLS, so `content_tier.promote_local_content`'s primary walk never returns it and its
    # content is never promoted to durable S3. The id backfill that would have caught it runs
    # exactly ONCE and is never re-armed (`id_backfill_done`), so after that pass such a row is
    # unreachable by BOTH walks — it exists only on local Garage and dies with the box.
    # content_tier stated this as an accepted residual risk whose fix was "re-arm the backfill on a
    # consume event". That fix is wrong: re-arming costs a full id walk of the whole corpus, which
    # is the scan the `_rev` walk was introduced to eliminate. Stamping the ABSENT case here closes
    # the hole at its only entry point instead — preservation is untouched (a present `_rev` is
    # never overwritten), and it costs no scan, no re-arm and no timer.
    # ⛔⛔ REVERTED, AND MUST NOT BE REINSTATED. DO NOT STAMP `_rev` HERE.
    # I added a stamp on this line earlier today to close a real hole (a rev-less row is invisible
    # to content_tier's `_rev` walk, so its content is never promoted). It closed that hole by
    # DISABLING THE ANTI-DOWNGRADE GUARD, which is far worse.
    # `arcade.py::_keep` (the `stamp_rev=False` path) rejects an incoming doc that carries NO `_rev`
    # when the local row HAS one — "an unversioned copy can never be shown to be newer, so it must
    # never win over one that carries a version." ~97.6% of rows are rev-less, and the mesh/graph
    # backlog is made of exactly those. Stamping a fresh `time.time_ns()` before `put_many` sees
    # them makes `ir >= lr` true every time, so ANCIENT rev-less backlog copies overwrite CURRENT
    # rows — and because consume writes with `stamp_rev=False`, the downgraded row sits below the
    # publish cursor and is never re-shipped. Silently gone from the whole fleet.
    # That is the precise data-destruction mode `_keep` was written to stop, re-enabled by a
    # one-line "fix" to an unrelated problem, and deployed fleet-wide.
    # The drain-invisibility hole is what the ONE-TIME id backfill exists for; it must be solved on
    # that side, never by forging a version this node cannot vouch for.
    declined = 0
    if _vertices(store) is not None:
        batch, declined = _split_unordered(store, batch)
        if not batch:
            return declined
    written = store.artifacts.put_many(batch, batch=500, stamp_rev=False)
    n = written if isinstance(written, int) else len(batch)
    if n < len(batch):
        # A short write means the segment did NOT fully apply. Raise so the caller holds the cursor
        # and records failed_key, rather than silently banking partial progress.
        raise RuntimeError("partial apply: wrote %d of %d" % (n, len(batch)))
    if _vertices(store) is not None:
        _enqueue_consumed(store, batch)
    _deliver_new(store, nd)
    # A DECLINED ROW IS HANDLED, NOT LOST. It was examined, a decision was recorded, and the peer
    # still holds its copy — exactly the status `put_many` gives an LWW-rejected doc. Counting it
    # keeps the caller's `written < len(batch)` guard meaning "something ERRORED", which is the one
    # thing that must hold the cursor.
    return n + declined


# ── PEER-APPLY REACHES COGNITION (John: "fix it properly" / "do it") ─────────────────────────────
#
# An arriving peer SIGNAL used to land as a row and stop there: `_apply_artifacts` wrote it and
# nothing routed it through the electroweak switch, so a signal addressed to this observer was
# replicated and never heard. `signal.deliver` is the joint that closes it and has existed, pure and
# unwired, this whole time.
#
# ⚠ IT FIRES ON GENUINELY-NEW DOCS ONLY, and that is the entire reason this was gated rather than
# just switched on. `applied` counts UPSERTS, not new knowledge: measured on MANTLE 2026-07-20,
# ~723,000 docs were "applied" across 12 consume cycles for a NET GAIN OF 0 ROWS. Delivering on
# `batch` would re-fire cognition for three quarters of a million no-op re-applies every cycle.
# `_new_docs` already computes the split; this is the consumer the docstring above says is missing.
#
# INACTIVE BY DEFAULT (`EMBER_DELIVER_PEER_SIGNALS`), and `forward` is not passed at all: local
# grounding is the whole story on a plain node and nothing is dropped. The onward persona hop needs a
# live carrier + reactor and stays a separate step — wiring a reach here without one would be fitting
# code to imagination.
#
# FIRE-AND-FORGET, AND IT MUST NOT TOUCH THE RETURN VALUE. The caller's `written < len(batch)` guard
# is what holds the mesh cursor; letting a delivery failure change `handled` would make a cognition
# error look like a partial write and strand a segment behind a monotone marker. So this runs after
# the write, records failures LOUDLY, and returns nothing.
log = logging.getLogger(__name__)

_DELIVER_PEER_SIGNALS = os.getenv("EMBER_DELIVER_PEER_SIGNALS", "").strip().lower() in (
    "1", "true", "yes", "on")


#: Where new peer docs go for COGNITION. A callable `(store, new_docs) -> None`, or None.
#:
#: ⚠ AN INJECTION SEAM, NOT AN IMPORT (2026-07-31). This block used to reach into
#: `ember.signal` and `ember.runtime.delegate` directly. Sync is the STORE replicating rows; that it
#: also woke the runner's cognition made the store depend on the runner, and it was the last thing
#: keeping `store/` and `mesh/` from landing in mantle — 26 files held by one optional call.
#:
#: The direction was wrong regardless of the move: a store tells you rows arrived; deciding that
#: this means something is the runner's job. Ember wires this at boot; anything else replicating
#: rows simply does not, and nothing is missing on a plain node.
_PEER_SIGNAL_SINK = None


def set_peer_signal_sink(fn) -> None:
    """Wire what happens when peer replication brings in genuinely NEW docs.

    `fn(store, new_docs)`. Pass None to unwire. Public because the wiring crosses a repo boundary —
    the alternative is a caller assigning a module private from another package, which is the exact
    back-door shape removed from `seed_lattice`/`wn_store` on 2026-07-30."""
    global _PEER_SIGNAL_SINK
    _PEER_SIGNAL_SINK = fn


def _deliver_new(store, new_docs) -> None:
    if not _DELIVER_PEER_SIGNALS or not new_docs:
        return
    sink = _PEER_SIGNAL_SINK
    if sink is None:
        # ⚠ ASKED-FOR AND UNWIRED IS A STATE WORTH SAYING OUT LOUD. The operator set
        # EMBER_DELIVER_PEER_SIGNALS expecting cognition on peer arrivals; silently doing nothing
        # would look exactly like "no new docs", which is the class of defect the comment above
        # already describes for a swallowed delivery.
        log.warning("EMBER_DELIVER_PEER_SIGNALS is on but no peer-signal sink is wired — "
                    "%d new doc(s) were stored and NOT delivered for cognition", len(new_docs))
        return
    try:
        sink(store, new_docs)
    except Exception as exc:                       # noqa: BLE001 — reported, never fatal, never silent
        # A swallow here would recreate the exact defect this closes: a signal that arrives, is
        # written, and is never heard, with nothing anywhere recording that cognition failed.
        log.error("peer-apply delivery failed for %d new doc(s): %s: %s",
                  len(new_docs), type(exc).__name__, exc)


# ── CONSUMED PEER ROWS ARE INVISIBLE TO EVERY LOCAL-ORIGIN WALK. CLOSED HERE, AT THE SOURCE ──────
#
# `page_by_origin` walks `WHERE _origin = :me`. A consumed row carries the PEER's `_origin` by
# design — that is what "preserve the origin's version" means and it is not negotiable — so it is
# invisible to every local-origin scan on this node, including `content_tier.promote_local_content`.
# Its content therefore never gets promoted to durable S3: it exists only on local Garage and dies
# with the box. The one-shot id backfill that would have caught it runs exactly ONCE and is never
# re-armed, so after that pass such a row is unreachable by BOTH walks.
#
# ⚠ THIS IS NOT A REGRESSION AND MUST NOT BE "FIXED" THE WAY IT WAS BEFORE. The `_rev` walk had the
# identical hole; it was merely clock-dependent and therefore invisible. The obvious repair — stamp
# a local version on the incoming row so the local walk sees it — WAS TRIED, SHIPPED FLEET-WIDE, AND
# REVERTED, and the note above `put_many` explains why in detail: stamping disables the anti-
# downgrade guard, so ANCIENT rev-less backlog copies overwrite CURRENT rows, and because consume
# writes with `stamp_rev=False` the downgraded row sits below the publish cursor and is never
# re-shipped. Silently gone from the whole fleet. Re-arming the full id backfill is the other
# tempting fix and costs a full corpus walk — the exact scan the `_rev` walk existed to eliminate.
#
# So: neither forge a version nor rescan. RECORD WHAT WAS CONSUMED, at the only point that knows.
# A bounded local queue of consumed ids needing content promotion is O(1) per consumed row, forges
# nothing, claims nothing about authorship, and gives the content drain an exact work list instead
# of a scan. `_origin` and `_seq` on the row are untouched.
_CONSUMED_ID = "mesh.consumed.pending"
_CONSUMED_CAP = 50000


def _enqueue_consumed(store, batch: List[Dict[str, Any]]) -> None:
    """Queue consumed peer rows that carry content, for the content drain to promote.

    Only rows with a `content_ref` are queued — a row with no content has nothing to promote and
    would just dilute the queue against its cap."""
    ids = [str(d.get("id")) for d in batch
           if d.get("id") and d.get("content_ref")
           and str(d.get("_origin") or "") not in ("", _origin_of(store))]
    if not ids:
        return
    cur = store.artifacts.get_artifact(_CONSUMED_ID) or {}
    pend = cur.get("pending")
    pend = list(pend) if isinstance(pend, list) else []
    merged = list(dict.fromkeys(pend + ids))
    overflow = bool(cur.get("overflow")) or len(merged) > _CONSUMED_CAP
    if overflow:
        # ⚠ OVERFLOW IS STICKY AND IT MEANS "FALL BACK TO A FULL SWEEP". Silently dropping the tail
        # would leave those rows in exactly the state this queue exists to prevent — present, and
        # invisible to the drain forever. The flag is the instruction to the consumer: the queue is
        # no longer a complete work list, so a full id sweep is required to be sure.
        merged = merged[:_CONSUMED_CAP]
    _put_op(store, {"id": _CONSUMED_ID, "content_type": _S3SYNC_CT, "state": "committed",
                    "pending": merged, "n": len(merged), "overflow": overflow})


def consumed_pending(store) -> Dict[str, Any]:
    """The content drain's work list: peer-authored rows this node holds whose content may not be
    promoted yet. `drain_consumed(ids)` removes them once promoted.

    `overflow: True` means the queue is INCOMPLETE and a full id sweep is required — treat the list
    as a hint, not as the whole set."""
    d = store.artifacts.get_artifact(_CONSUMED_ID) or {}
    p = d.get("pending")
    p = list(p) if isinstance(p, list) else []
    return {"pending": p, "n": len(p), "overflow": bool(d.get("overflow"))}


def drain_consumed(store, ids) -> int:
    """Remove ids the content drain has promoted. Clears `overflow` only when the queue empties —
    an overflow that is forgotten while rows remain would silently downgrade the caller from
    'sweep required' back to 'this list is complete'."""
    done = {str(x) for x in ids}
    d = store.artifacts.get_artifact(_CONSUMED_ID) or {}
    p = [x for x in (d.get("pending") or []) if str(x) not in done]
    _put_op(store, {"id": _CONSUMED_ID, "content_type": _S3SYNC_CT, "state": "committed",
                    "pending": p, "n": len(p),
                    "overflow": bool(d.get("overflow")) and bool(p)})
    return len(done)


# ── UNORDERED CONCURRENT AUTHORSHIP — the decision Unit L left open (its report, A4) ─────────────
#
# THE PROBLEM. Two observers author the same vertex id independently. `(_origin, _seq)` cannot order
# them and MUST NOT: contract RESOLVED-3 removes the clock permanently and §C.7 states that
# "unordered is a valid answer". Under `on_unordered="keep_local"` both sides stay divergent, so
# their merkle leaves keep mismatching and anti-entropy re-offers the row EVERY ROUND, forever.
# Honest, but not free.
#
# THE DECISION: record a local DECLINATION. Do not scope the row out of anti-entropy.
#
# Why this does not violate §C.7. A declination asserts nothing about order. It is a statement about
# THIS observer's history — "I have seen version (O_r, S_r) of vertex X and did not adopt it" — not
# about which version is newer. `compare_version` still answers UNORDERED, the peer still holds its
# copy, and nothing is deleted anywhere. This is precisely the "private per-observer state" §1.2
# calls for when it says a shared vertex carries only what is universal and interpretation stays
# local: shared observation, private interpretation. Synthesizing a winner by clock, node id, or
# arrival order would all be inventing information; declining to adopt is not.
#
# Why NOT the alternative (scope the divergence out of the anti-entropy set). That stops the re-offer
# completely, which is why it is tempting. It also stops the row being CHECKED — so a later, genuine
# divergence on the same row goes unnoticed forever. That trades a visible cost for invisible data
# loss, which is the failure class this entire file is a monument to. Rejected.
#
# ⚠ WHAT THIS DOES AND DOES NOT BUY — stated plainly, because overclaiming here would itself be the
# defect. It does NOT stop the leaf being re-fetched: the digests genuinely differ, and no bookkeeping
# can make them agree without picking a winner. The irreducible cost of a permanent unordered
# conflict is one leaf fetch per peer per round. What it DOES buy is that the divergence stops being
# invisible churn: it is counted, it is attributable to specific ids, and it is reported out of
# `consume_from_s3` / `reconcile_merkle` so an operator sees a number instead of a node that is
# mysteriously busy forever. Per the standing rule, a missing signal gets ADDED to health monitoring
# rather than worked around.
_DECLINED_ID = "mesh.unordered.declined"
# Reuses `_S3SYNC_CT` rather than minting a type: see the lockstep note on `_OP_EXCLUDE`. It IS a
# per-box sync cursor — a record of where this observer's adoption decisions stand — so the reuse is
# accurate rather than expedient. The id keeps it distinct from the publish/consume cursors.
_DECLINED_CT = _S3SYNC_CT
_DECLINED_CAP = 10000       # bounded on purpose; overflow is itself the alarm — see below


def _load_declined(store) -> Dict[str, Any]:
    d = store.artifacts.get_artifact(_DECLINED_ID) or {}
    seen = d.get("seen")
    return seen if isinstance(seen, dict) else {}


def _split_unordered(store, batch: List[Dict[str, Any]]):
    """Partition incoming docs into (appliable, declined_count).

    Compares each doc's `(_origin, _seq)` against the local row BEFORE `put_many` sees it. `put_many`
    would reach the same verdict — it does its own LWW read — but it reports only an aggregate
    counter, so which vertex is diverging, and whether it is the SAME one every round, is
    unrecoverable from it. The extra keyed read per doc is what turns "the conflict counter went up
    again" into "vertex X has been unordered against origin Y for 400 rounds"."""
    from mantle.db.lattice import constants as K       # noqa: WPS433 — optional dep, lattice only
    v = _vertices(store)
    keep, newly = [], {}
    seen = _load_declined(store)
    for doc in batch:
        origin, sq = doc.get("_origin"), doc.get("_seq")
        if not isinstance(origin, str) or not isinstance(sq, int):
            keep.append(doc)                  # put_many will raise on it — its error, not ours
            continue
        local = v.version_of(str(doc.get("id")))
        if local is None or local[0] is None:
            keep.append(doc)
            continue
        if K.compare_version(origin, sq, local[0], local[1]) != K.UNORDERED:
            keep.append(doc)
            continue
        newly[str(doc.get("id"))] = [origin, sq]
    if newly:
        merged = dict(seen)
        merged.update(newly)
        overflow = len(merged) > _DECLINED_CAP
        if overflow:
            # DO NOT SILENTLY EVICT AND CARRY ON. Past this many permanently-divergent vertices the
            # per-id record is no longer the right instrument and the situation is an incident, not
            # a statistic. Keep a bounded, deterministic slice so the artifact stays small, and say
            # loudly that the set is truncated so nobody reads its size as the real number.
            merged = {k: merged[k] for k in sorted(merged)[:_DECLINED_CAP]}
        _put_op(store, {"id": _DECLINED_ID, "content_type": _DECLINED_CT, "state": "committed",
                        "seen": merged, "n": len(merged), "truncated": overflow,
                        "note": "unordered concurrent authorship; no order was synthesized"})
    return keep, len(newly)


def unordered_report(store) -> Dict[str, Any]:
    """The operator-facing view of permanent divergence. Read by health monitoring.

    `n` is how many vertices this observer has declined and NOT adopted. A number that grows without
    bound means two observers are actively authoring the same ids, which is a topology problem, not a
    replication problem — no cursor fix and no merkle change will move it."""
    d = store.artifacts.get_artifact(_DECLINED_ID) or {}
    seen = d.get("seen") if isinstance(d.get("seen"), dict) else {}
    return {"declined": len(seen), "truncated": bool(d.get("truncated")),
            "sample": sorted(seen)[:20],
            "basis": "unordered (_origin,_seq); no tiebreak synthesized (contract RESOLVED-3/§C.7)"}


def _apply_edges(store, items: List[Dict[str, Any]]) -> int:
    """Rebuild consumed edges. Endpoints arrive via the artifact feeds first; a missing endpoint just
    yields no edge (add_edges' FROM/TO SELECT finds nothing) — harmless, retried next round.

    ⚠ A CONSUMED EDGE IS NOT LOCALLY AUTHORED. On the lattice path each one is stamped with the
    reserved per-edge origin defined at the top of this file, so it is invisible to
    `page_by_origin(origin=me)` and cannot echo back onto the mesh. See `_CONSUMED_EDGE_NS`."""
    edges = [(e["f"], e["t"], e.get("label") or "link", e.get("props") or {})
             for e in items if e.get("f") and e.get("t")]
    if not edges:
        return 0
    # DO NOT SWALLOW THE WRITE FAILURE. This used to `except Exception: return 0`, and 0 is
    # indistinguishable from "nothing to apply" — so the caller saw no exception, recorded no
    # failed_key, and ADVANCED THE CURSOR past an edge segment that never landed. Every failed
    # edge segment was lost permanently and stayed invisible to mesh_lag()["stuck"]. Letting it
    # raise is what makes the cursor-hold and the stuck report work for edges at all.
    if _edges(store) is None:
        # ArcadeDB: `add_edges` has no `stamp_rev` keyword and `_rev` carries no authorship, so
        # there is nothing to preserve and nothing to reserve. Live publish path for MANTLE/45/71 —
        # unchanged on purpose (contract §6.1 / Unit E §6).
        store.graph.add_edges(edges, batch=500)
        return len(edges)

    me = _origin_of(store)
    stamped = []
    for src, dst, label, props in edges:
        p = dict(props)
        o, sq = p.get("_origin"), p.get("_seq")
        if not (isinstance(o, str) and o and o != me and isinstance(sq, int)):
            # No usable provenance on the wire (the normal case — the edge segment format does not
            # carry it), or a peer claiming WE authored its edge. Either way: reserved origin.
            p["_origin"] = _consumed_edge_origin(src, dst, label)
            p["_seq"] = 1
        stamped.append((src, dst, label, p))
    handled = store.graph.add_edges(stamped, batch=500, stamp_rev=False)
    if handled < len(stamped):
        # `_add_chunk` rolls a failing edge back per-savepoint and CARRIES ON, so a partial apply
        # does not raise on its own — the shortfall is the only evidence it happened. Unguarded,
        # `_consume_stream` would see no exception, write a later `last_key`, and drop those edges
        # permanently behind a monotone StartAfter marker. Same guard `_apply_artifacts` puts on
        # `put_many`, and for the same reason: handled < offered is data loss.
        raise RuntimeError(
            "edge segment partially applied: %d of %d edges handled. Holding the consume cursor "
            "so this segment is retried rather than silently skipped." % (handled, len(stamped)))
    return len(stamped)


def _private_set(store, *, cap: int = 200000):
    """The ids the mesh MUST NOT publish — the members (and container) of every GRANT-GATED collection —
    as `(ids, exhaustive)`. What leaves the node is the ungated PUBLIC set (the un-keyed TOP).

    The access decision is grants, computed by `ember.access` (the same light-cone the mantle service
    uses): a collection is non-public iff a grant gates it. This is NOT a flag and NOT the `private.`
    name. With no grant present, nothing is gated and the whole Merkle path is byte-identical to before.

    Membership is enumerated via the indexed `origin_root`. Non-exhaustive (over `cap`, or the grant
    system cannot be consulted) -> the caller REFUSES to publish rather than leak the remainder.

    ⛔ FAILS CLOSED. If `access`/`lattice_api` is importable, an ERROR resolving grants returns
    non-exhaustive (refuse). Only a genuinely grant-less store returns the empty, exhaustive set."""
    v = _vertices(store)
    if v is None or not hasattr(v, "page_by_origin_root"):
        return set(), True
    try:
        from mantle.db.lattice import access
        access._api()                                  # probe: is the grant subsystem importable?
    except Exception:
        # No grant subsystem reachable at all (a minimal env without mantle's lattice_api on the path).
        # There are then no grants to consult, so nothing is gated — empty is correct, not a leak.
        return set(), True
    try:
        gated_cols = access.gated_collections(store)
        # The COMMONS light-cone: collections/artifacts made public (a Read grant to the public entity)
        # are NOT withheld even though they sit in a gated collection — they mesh out like any public row.
        public_reach = access.reachable_collections(store, access.PUBLIC_PRINCIPAL)
    except Exception:
        # The subsystem IS present but resolving grants failed. FAIL CLOSED: refuse rather than leak.
        return set(), False
    withheld, seen = set(), 0
    for root in gated_cols:
        if str(root) in public_reach:
            continue                                   # whole collection made public -> replicates
        withheld.add(str(root))                        # the private container does not mesh out
        after = ""
        while True:
            page = v.page_by_origin_root(str(root), after=after, limit=5000)
            if not page:
                break
            withheld.update(m for m in page if str(m) not in public_reach)   # keep made-public members
            after = page[-1]
            seen += len(page)
            if seen > cap:
                return withheld, False
            if len(page) < 5000:
                break
    return withheld, True


def _refresh_leaves_lattice(store, leaves_to_refresh, n: int, cur: Dict[str, Any],
                            digests: List[int]) -> Dict[str, Any]:
    """Refresh leaf digests on a lattice store — from the INCREMENTAL tree, minus operational rows.

    WHY NOT JUST RE-QUERY THE LEAF. The Arcade form is one indexed `WHERE _leaf = :li` per leaf, and
    the lattice store exposes no `page_by_leaf`, so the equivalent would be a full keyset pass of the
    corpus PER REFRESH — roughly 75 minutes at the 6M-row target, per reconcile round. That is not a
    slower version of the same thing; it removes the reason this function exists (a full rescan is
    ~7 min on 2.7M rows and the corpus changes faster than the scan completes, so a rescan-built tree
    is stale before it finishes).

    WHY NOT JUST USE `merkle_leaves()`. It is O(1) and incrementally maintained, which is exactly
    right — but `vertex._write_row` XORs EVERY row into its leaf, including `_OP_EXCLUDE` cursors and
    tasks. Publishing that tree would give two perfectly converged nodes different roots forever,
    over state that must never replicate. (Flagged to Unit L: the durable fix is for `_write_row` to
    skip `xor_leaf` on non-replicated content types. Until then it is corrected here, in the file
    that owns the exclusion policy.)

    SO: take the incremental tree and XOR THE OPERATIONAL ROWS BACK OUT. XOR is its own inverse —
    that is why XOR-of-row-hashes was chosen over a sorted hash — so removing a row's contribution is
    the same operation as adding it. The operational set is enumerated with `list_by_content_type`,
    a typed indexed lookup, once per `_OP_EXCLUDE` type. Cost is O(operational rows), not O(corpus).

    ⚠ RESIDUAL, STATED RATHER THAN HIDDEN: the PREFIX bans (`application/x-probe*`,
    `application/vnd.agience.probe*`, `throwaway`) cannot be enumerated by exact-name lookup, so a
    probe fixture present on this node is NOT subtracted here. It is reported as
    `prefix_bans_unverified`, and it self-heals: `publish_merkle`'s full scan filters on
    `_is_replicated` (prefix bans included) and rewrites `_MERKLE_LIVE` wholesale. This function only
    ever moves the tree between two publishes."""
    from . import merkle
    v = _vertices(store)
    want = sorted(x for x in {int(y) for y in leaves_to_refresh} if 0 <= x < n)
    live = v.merkle_leaves()
    if len(live) != n:
        # Different tree shape: the store's leaf count and the mesh's disagree. Refuse rather than
        # publish a tree indexed on a different modulus — every leaf would appear to differ.
        return {"refreshed": 0, "error": "leaf-count mismatch: store=%d mesh=%d" % (len(live), n)}
    tree = list(live)
    excluded = 0
    for ct in sorted(_OP_EXCLUDE):
        # ⛔ `include_archived=True` IS LOAD-BEARING HERE — this is REBUILDING STATE, not answering a
        # question. `list_by_content_type` became head-only by default (mantle §6c, 2026-07-29) so a
        # revision stops double-answering queries; that default is wrong for subtraction. An archived
        # operational row still contributed to the incremental tree, so omitting it would leave its
        # hash in place while `publish_merkle`'s full scan — which filters on `_is_replicated`, not on
        # state — removes it. The two would then disagree forever: peers diff a leaf whose object
        # 404s and stay PERMANENTLY DIVERGENT, which is the exact conflation the `(docs, exhaustive)`
        # contract exists to prevent. XOR is its own inverse only if both passes see the same rows.
        docs, exhaustive = v.list_by_content_type(ct, cap=200000, include_archived=True)
        for d in docs:
            aid = d.get("id")
            if not aid:
                continue
            ver = v.version_of(str(aid))
            if ver is None:
                continue
            li = merkle.leaf_of(str(aid), n)
            tree[li] ^= merkle.row_hash(str(aid), ver[1])
            excluded += 1
        if not exhaustive:
            # Truncated enumeration means an unknown number of operational rows are still in the
            # tree. Do not write a digest we know to be wrong.
            return {"refreshed": 0,
                    "error": "operational enumeration truncated for %r; cannot subtract exactly" % ct}
    # GRANT-GATED (non-public) rows are subtracted the SAME way — held and queried locally, never
    # advertised. What leaves the node is the ungated public set (the un-keyed TOP). The access
    # decision is grants (`ember.access`), not a flag. XOR is its own inverse, so removing a row's
    # contribution is the same operation that added it on write.
    priv_ids, priv_exhaustive = _private_set(store)
    if not priv_exhaustive:
        return {"refreshed": 0,
                "error": "grant-gated enumeration could not be resolved exactly; refusing to publish "
                         "a tree that would leak withheld rows"}
    for aid in priv_ids:
        ver = v.version_of(str(aid))
        if ver is None:
            continue
        li = merkle.leaf_of(str(aid), n)
        tree[li] ^= merkle.row_hash(str(aid), ver[1])
    for li in want:
        digests[li] = tree[li] & 0xFFFFFFFFFFFFFFFF
    # PRESERVE `published`. This recomputes LIVE digests and uploads NOTHING, so it must not touch
    # the record of which leaf objects exist in S3 — overwriting that is precisely what let
    # publish_merkle conclude "nothing changed" while no leaf file had ever been written.
    _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT, "state": "committed",
                    _MERKLE_LIVE: digests, _MERKLE_PUBLISHED: cur.get(_MERKLE_PUBLISHED) or [],
                    "root": merkle.root(digests)})
    return {"refreshed": len(want), "root": merkle.root(digests), "excluded_op_rows": excluded,
            "prefix_bans_unverified": True, "basis": "incremental-minus-operational"}


def refresh_leaves(store, leaves_to_refresh, *, leaves: int = 0) -> Dict[str, Any]:
    """Recompute the digests of specific leaves from the indexed `_leaf` column.

    This is what makes Merkle affordable to run continuously. A full rescan costs ~7 minutes on a
    2.7M-row corpus under load (measured), and the corpus changes faster than that while a node is
    catching up — so a rescan-built tree is stale before it finishes. Refreshing only the leaves
    that actually changed is an indexed lookup over ~1500 rows each instead.

    Note this recomputes a leaf from its CURRENT rows rather than XOR-ing a delta. Both are correct
    (XOR is its own inverse, so a delta would work), but recomputing is self-healing: it converges
    to the truth even if a previous digest was wrong, whereas a delta chain carries any past error
    forward silently. Given the day's recurring lesson — components asserting more than they had
    done — the self-correcting form is worth the extra query.

    Rows written before `_leaf` existed have NULL and belong to no leaf, so they are invisible here
    until backfilled; `merkle_coverage` reports exactly how many, and publish refuses on the gap."""
    from . import merkle
    n = int(leaves or _store_leaves(store))
    cur = store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}
    digests = [int(x) for x in (cur.get(_MERKLE_LIVE) or [0] * n)]
    if len(digests) != n:
        digests = [0] * n
    done = 0
    if _vertices(store) is not None:
        return _refresh_leaves_lattice(store, leaves_to_refresh, n, cur, digests)
    for li in sorted(set(int(x) for x in leaves_to_refresh)):
        if not (0 <= li < n):
            continue
        try:
            # `_leaf = {li}` is already the replication filter: operational rows are never stamped,
            # so they carry NULL and NULL matches no equality — they cannot enter a leaf digest.
            rows = store.artifacts.c.query(
                f"SELECT id, _rev FROM Artifact WHERE _leaf = {li}") or []
        except Exception:
            continue
        d = 0
        for r in rows:
            aid = r.get("id")
            if aid:
                d ^= merkle.row_hash(aid, r.get("_rev"))
        digests[li] = d
        done += 1
    # PRESERVE `published`. This function recomputes LIVE digests and uploads NOTHING, so it must
    # not touch the record of which leaf objects exist in S3 — overwriting it is precisely what let
    # publish_merkle conclude "nothing changed" while no leaf file had ever been written.
    _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT,
                                  "state": "committed", _MERKLE_LIVE: digests,
                                  _MERKLE_PUBLISHED: cur.get(_MERKLE_PUBLISHED) or [],
                                  "root": merkle.root(digests)})
    return {"refreshed": done, "root": merkle.root(digests)}


def publish_merkle_incremental(store, *, leaves: int = 0,
                               max_seconds: float = 0.0) -> Dict[str, Any]:
    """Steady-state Merkle publish WITHOUT a full corpus rescan.

    `publish_merkle` scans every row (~5-7 min on millions) to rebuild both the LIVE digests and the
    per-leaf id buckets it needs to upload. At steady state that cost is wrong: the lattice store
    already maintains the leaf digests incrementally on write, so LIVE is recomputed by
    `refresh_leaves` in O(operational rows), and only the leaves whose digest actually MOVED since the
    last publish need their object rebuilt — each an indexed `list_by_leaf` lookup, not a scan. On a
    converged node that is one 32 KB summary and nothing else.

    This is the path the aggregator loop drives. `publish_merkle` (full rescan) stays as the
    bootstrap/verification path AND the only safe path while `merkle_coverage < 100%` — a legacy
    Arcade store with unbackfilled `_leaf` — because only its full scan self-heals the prefix bans
    (`_refresh_leaves_lattice` cannot subtract a prefix-banned probe type by exact-name lookup).

    LATTICE ONLY: it rests on the incremental-minus-operational tree and on `list_by_leaf`. On a
    non-lattice store it DECLINES rather than silently full-scanning under a name that promises
    'incremental' — the honest-refusal rule applied to a capability the backend lacks.

    Every write to `_S3_MERKLE_CURSOR` keeps LIVE and PUBLISHED distinct and advances PUBLISHED only
    as far as leaves actually landed (contract M6); the summary is uploaded LAST so a peer never sees
    a root claiming a leaf object that is not up yet."""
    import json
    from . import merkle
    v = _vertices(store)
    if v is None:
        return {"published": 0, "reason": "not-a-lattice-store",
                "note": "publish_merkle_incremental needs the incremental leaf tree; use "
                        "publish_merkle for the Arcade rescan path"}
    s3 = _mesh_s3(store)
    if s3 is None:
        return {"published": 0, "reason": "no-s3"}
    if not leaves:
        # Keep the operating resolution equal to `natural_leaves(corpus)`. Cheap every cycle (a
        # counter compare); pays the O(N) re-stamp only on a rare sqrt boundary. Done BEFORE reading
        # `n` so a reshard's new tree is what this round publishes — the whole tree re-publishes at
        # the new resolution, which is correct (every leaf moved).
        try:
            v.maybe_reshard(graph=_edges(store))
        except Exception:
            pass
    n = int(leaves or _store_leaves(store))
    node, f = _node_id(), _fernet(store)
    t0 = time.time()
    # 1) Recompute LIVE from the store's incremental tree, minus operational rows (M7). This writes
    #    _MERKLE_LIVE and PRESERVES _MERKLE_PUBLISHED — refresh uploads nothing.
    rl = refresh_leaves(store, range(n), leaves=n)
    if rl.get("error"):
        return {"published": 0, "reason": "refresh-failed", "error": rl["error"]}
    cur = store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}
    live = [int(x) for x in (cur.get(_MERKLE_LIVE) or [0] * n)]
    if len(live) != n:
        return {"published": 0, "reason": "live-shape", "have": len(live), "want": n}
    # 2) changed = LIVE != PUBLISHED. -1 sentinel = "no object uploaded for this leaf yet", so a
    #    never-published leaf always compares changed (unsigned-64 digests can never be -1).
    prev = cur.get(_MERKLE_PUBLISHED) or []
    pub = ([int(x) for x in prev] + [-1] * n)[:n]
    changed = [i for i in range(n) if i >= len(prev) or int(prev[i]) != live[i]]
    # 3) Rebuild ONLY the changed leaves from the indexed `_leaf` column; upload; advance PUBLISHED.
    #    A leaf object is a MIXED NDJSON: vertex docs ({"id",…}) and edge records ({"f","t",…}). The
    #    tree covers BOTH tables, so a changed leaf's object must carry both — `_apply_artifacts` takes
    #    the id-lines and `_apply_edges` the f/t-lines, each ignoring the other.
    g = _edges(store)
    # The SAME withheld set the digest was computed against (`_refresh_leaves_lattice` subtracted it),
    # so each changed leaf's object and its advertised digest agree: neither carries a grant-gated row.
    # Refuse if it cannot be resolved exactly — publishing the object would leak the remainder.
    priv_set, priv_exhaustive = _private_set(store)
    if not priv_exhaustive:
        return {"published": 0, "reason": "grant-gated-enumeration-unresolved",
                "note": "refusing to build leaf objects that could leak withheld rows"}
    uploaded, truncated, incomplete = 0, False, False
    for li in changed:
        docs, v_exh = v.list_by_leaf(li)
        erecs, e_exh = g.list_by_leaf(li) if g is not None else ([], True)
        if not (v_exh and e_exh):
            # A leaf we cannot enumerate exactly must not be published as its authoritative object —
            # that would advertise a root over rows we did not include. Hold this leaf for next round.
            incomplete = True
            continue
        lines = [json.dumps(_proj(d), separators=(",", ":"))
                 for d in docs if d and _is_replicated(d.get("content_type"))
                 and d.get("id") not in priv_set]
        lines += [json.dumps(er, separators=(",", ":")) for er in erecs]
        # An EMPTY leaf is still uploaded as an empty object (contract M6): a missing file is an
        # unresolvable 404 diff that keeps a genuinely-empty range "differing" forever.
        s3.put("%s%s/%05d.ndjson.enc" % (_MESH_LEAF_PREFIX, node, li),
               f.encrypt(("\n".join(lines)).encode("utf-8")), "application/octet-stream")
        uploaded += 1
        pub[li] = live[li]      # only now, backed by an object, is this leaf PUBLISHED
        if max_seconds and time.time() - t0 >= max_seconds:
            truncated = True
            break
    if truncated or incomplete:
        # Do NOT publish a summary whose leaves are only partly up. Record LIVE (the truth about local
        # state) and advance PUBLISHED exactly as far as the uploads got; the previous consistent
        # summary stays in place and peers keep converging on the last good root. Next round finishes.
        _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT, "state": "committed",
                        _MERKLE_LIVE: live, _MERKLE_PUBLISHED: pub, "partial_upload": True})
        return {"published": 0,
                "reason": "upload-truncated" if truncated else "leaf-enumeration-truncated",
                "changed": len(changed), "uploaded": uploaded, "node": node}
    # 4) Summary LAST (M6); then, and ONLY here, equate PUBLISHED to LIVE.
    s3.put("%s%s.json" % (_MESH_MERKLE_PREFIX, node),
           json.dumps(merkle.summary(live)).encode("utf-8"), "application/json")
    _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT, "state": "committed",
                    _MERKLE_LIVE: live, _MERKLE_PUBLISHED: list(live), "root": merkle.root(live)})
    return {"leaves": n, "changed": len(changed), "uploaded": uploaded,
            "root": merkle.root(live), "secs": round(time.time() - t0, 1),
            "basis": "incremental", "node": node}


def _replicated_count(store) -> int:
    """How many rows the mesh is supposed to replicate — `count(*)` MINUS operational state.

    Every coverage/completeness ratio must use this as its denominator. `_OP_EXCLUDE` rows are
    per-box by definition and are deliberately never stamped with `_leaf`, so any check that
    compares a `_leaf`-filtered numerator to a raw `count(*)` is guaranteed to under-report and
    will refuse to publish (or report <100% coverage) forever, on a node that is perfectly
    healthy."""
    # `content_type IS NULL OR ...` is required, not defensive: in SQL's three-valued logic
    # `NULL NOT IN [...]` is NULL, i.e. NOT matched — so a row with no content_type would drop out
    # of the denominator while still being stamped with `_leaf` and counted in the numerator,
    # producing a coverage figure above 100%.
    # COMPUTED BY SUBTRACTION, NOT BY NEGATION. The old form
    #   `WHERE content_type IS NULL OR content_type NOT IN [...]`
    # stacks three separate index disqualifiers (IS NULL, OR, NOT IN) and is therefore a guaranteed
    # full scan. The identical answer comes from indexed queries only: an unfiltered `count(*)` is
    # O(1)-ish (~0-40ms), and every excluded type is an EQUALITY on the indexed `content_type`.
    # Subtracting also preserves the three-valued-logic property the old comment was defending --
    # rows with a NULL content_type match none of the equality counts, so they stay in the total,
    # which is exactly where they belong.
    # ON LATTICE BOTH TERMS ARE COUNTER LOOKUPS AND NO `count(*)` IS ISSUED AT ALL. The subtraction
    # shape is unchanged — it is still total minus each excluded type — but `count()` and
    # `count_by_content_type()` read incrementally maintained counter rows, so this is O(|_OP_EXCLUDE|)
    # keyed reads instead of a scan. That matters beyond speed: EXPLAIN on the live corpus shows
    # `count(*)` loading 6M records to produce one integer, and on node 71 it OOMs the acceptor
    # thread and zombies the node.
    try:
        total = _total_count(store)
        excluded = 0
        for ct in sorted(_OP_EXCLUDE):
            excluded += _count_of_type(store, ct)
        # Grant-gated (non-public) rows are held locally but never replicated, so they are not part of
        # what the mesh is "supposed to replicate" — subtract them too. Empty (no subtraction) by
        # default (no grants -> nothing gated).
        priv, _priv_exh = _private_set(store)
        excluded += len(priv)
        return max(0, total - excluded)
    except Exception:
        try:
            return _total_count(store)
        except Exception:
            return 0


def merkle_coverage(store) -> Dict[str, Any]:
    """How much of the corpus carries `_leaf` yet. Merkle is only trustworthy at ~100%: a row with
    NULL `_leaf` is in no leaf, so it is invisible to refresh AND to any tree built from it — the
    tree would look healthy while silently omitting data. Report it rather than assume it.

    The denominator is REPLICATED rows only (`_replicated_count`). Operational rows are intentionally
    left with NULL `_leaf`, so counting them as "missing" would peg coverage permanently below 100%
    and make the very signal this function exists to give — "is Merkle trustworthy yet?" — read
    as broken on a node where nothing is wrong."""
    # THIS ONE STAYS A SCAN, AND THAT IS THE CORRECT ANSWER — not an oversight and not a missing
    # index. Three independent reasons, any one of which is sufficient:
    #
    #  1. NO INDEX CAN CONTAIN THE ANSWER. `Artifact(_leaf)` is an LSM index with nullStrategy SKIP,
    #     so NULL `_leaf` rows are not IN the index. `IS NOT NULL` therefore cannot be index-served,
    #     and neither can its complement — the rows this function exists to count (the ones MISSING
    #     `_leaf`) are exactly the rows no index has an entry for. Recommending an index would be
    #     recommending an index over nulls, which this engine does not have.
    #  2. AN INDEX WOULD NOT HELP EVEN IF IT EXISTED. `_leaf` is stamped on every replicated row, so
    #     the predicate matches ~100% of the table. A range that spans the whole index reads one
    #     entry per row — the same O(N) as the scan, with an extra indirection. Index selectivity is
    #     the entire mechanism, and here there is none to have.
    #  3. THE OBVIOUS REWRITE IS ACTIVELY UNSAFE ON THIS ENGINE. `WHERE _leaf > -1` is the tempting
    #     index-served spelling of the same predicate, and both of its failure modes are MEASURED on
    #     this table: `WHERE _rev > -1` throws "Detected infinite loop while iterating index
    #     Artifact_0_*" (see `_scan_rows`), and `>=` over a duplicated key on a NOTUNIQUE LSM index
    #     returns ONE row per key group (6 rows sharing R -> `count(*) WHERE _rev >= R` = 1). `_leaf`
    #     has only 4096 distinct values over millions of rows — the most duplicated key on the type —
    #     so it is the worst possible candidate for that shape. A coverage number that is silently
    #     4096 instead of 6,000,000 would make `publish_merkle` refuse forever on a healthy node.
    #
    # What makes it affordable is instead that it is COLD: the only caller is scripts/backfill_leaf.py
    # (an operator-run backfill), never the sync cycle. And the denominator beside it is now fully
    # indexed (`_replicated_count` = one unfiltered count + 5 equality counts, ~0-40ms each), so this
    # call costs exactly ONE scan rather than the two it used to.
    # ON LATTICE THE GAP THIS FUNCTION MEASURES CANNOT EXIST, AND SAYING SO IS THE HONEST ANSWER —
    # not a scan that trivially returns 100%. `_leaf` is computed and stamped inside
    # `vertex._write_row` on every single write, so there is no "written before `_leaf` existed"
    # population to backfill and `scripts/backfill_leaf.py` has nothing to do. Reporting
    # `pct: 100.0` from a real scan would be true and useless; reporting the BASIS lets the caller
    # tell "verified complete" from "complete by construction".
    #
    # ⚠ The residual is the OPPOSITE problem and it is named rather than buried: on lattice every row
    # carries `_leaf`, INCLUDING the operational rows that must never replicate. So `_leaf` coverage
    # is no longer a proxy for "the tree is trustworthy" — `_refresh_leaves_lattice` subtracts those
    # rows explicitly, and that is where the trust now comes from.
    if _vertices(store) is not None:
        tot = _replicated_count(store)
        return {"total": tot, "with_leaf": tot, "missing": 0, "pct": 100.0,
                "basis": "lattice: _leaf stamped at write; no unstamped population can exist",
                "note": "operational rows ARE stamped here; see _refresh_leaves_lattice"}
    try:
        tot = _replicated_count(store)
        have = int((store.artifacts.c.query(
            "SELECT count(*) AS c FROM Artifact WHERE _leaf IS NOT NULL") or [{}])[0].get("c", 0))
    except Exception as e:
        return {"error": str(e)[:120]}
    return {"total": tot, "with_leaf": have, "missing": max(0, tot - have),
            "pct": round(100.0 * have / tot, 2) if tot else 0.0}


def reconcile_merkle(store, *, max_leaves: int = 64, max_seconds: float = 0.0) -> Dict[str, Any]:
    """Pull ONLY the leaves that differ from each peer. This is the O(diff) replacement for reading
    every peer's whole log.

    Steady state costs one 32 KB tree fetch per peer and nothing else — if the roots match we are
    provably identical and stop. That flat steady-state cost is the entire reason this scales to
    500 nodes where the log feed cannot.

    Uses the LOCAL TREE CACHED by publish_merkle rather than rescanning: the scan is ~117s on a
    2.2M-row store, so rescanning per peer would make reconciliation cost O(peers x corpus) — the
    very thing being fixed. If no local tree has been published yet we do nothing rather than
    guess, because comparing against an empty tree would claim every leaf differs and pull the
    entire corpus from every peer at once.

    Applying a leaf is idempotent and order-independent (upsert by id), so a partial or repeated
    transfer is always safe — there is no cursor to corrupt and nothing to roll back."""
    import json
    from . import merkle
    s3 = _mesh_s3(store)
    if s3 is None:
        return {"applied": 0, "reason": "no-s3"}
    local = (store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}).get(_MERKLE_LIVE)
    if not local:
        return {"applied": 0, "reason": "no-local-tree"}
    local = [int(x) for x in local]
    me, f = _node_id(), _fernet(store)
    skip = _ignored_nodes()
    t0 = time.time()
    applied, peers, fetched = 0, {}, 0
    stats = {"new": 0, "reapplied": 0}      # the genuinely-new split (#3), accumulated across peers/leaves
    # LIST KEYS, NOT PREFIXES. `mesh/merkle/<node>.json` is a FLAT object — there is no `/` after the
    # prefix, so a delimited list returns it under Contents and CommonPrefixes is always EMPTY. Using
    # _s3_prefixes here meant reconcile enumerated zero peers on every node, forever: it returned
    # `{"applied": 0}` and looked perfectly healthy while doing nothing at all. (The dead
    # `.replace(".json", "")` on a value that could never contain it is the fossil of that mistake.)
    for key in _s3_keys_after(s3, _MESH_MERKLE_PREFIX):
        node = key[len(_MESH_MERKLE_PREFIX):].strip("/")
        if not node.endswith(".json") or "/" in node:
            continue
        node = node[:-len(".json")]
        if not node or node == me or node in skip:
            continue
        try:
            theirs = merkle.load(json.loads(s3.get("%s%s.json" % (_MESH_MERKLE_PREFIX, node)).decode("utf-8")))
        except Exception:
            continue
        if not theirs:
            continue                        # unreadable/empty peer tree: not comparable, skip
        # `diff` compares at the COMMON resolution t = min(len(local), len(theirs)) — a node never
        # re-shards just to talk to a peer; it compares at the coarser of the two. A differing coarse
        # index `li` maps to the peer's NATIVE leaves {li, li+t, li+2t, …} (the peer published its
        # objects at ITS resolution). Fetch each and apply the mixed vertex+edge object.
        d = merkle.diff(local, theirs)
        if not d:
            peers[node] = 0                 # converged with this peer — the common case, costs 1 fetch
            continue
        t = min(len(local), len(theirs))
        span = max(1, len(theirs) // t)     # peer-native leaves folded into one coarse index
        got = 0
        stop = False
        for li in d[:max_leaves]:
            for m in range(span):
                idx = li + m * t
                try:
                    raw = f.decrypt(s3.get("%s%s/%05d.ndjson.enc" % (_MESH_LEAF_PREFIX, node, idx)))
                    items = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
                    # MIXED object: each applier takes its own lines (id vs f/t).
                    got += _apply_artifacts(store, items, stats=stats)
                    got += _apply_edges(store, items)
                    fetched += 1
                except Exception:
                    continue                # a missing/bad leaf just stays divergent for next round
                if max_seconds and time.time() - t0 >= max_seconds:
                    stop = True
                    break
            if stop:
                break
        applied += got
        peers[node] = {"differing": len(d), "pulled": min(len(d), max_leaves), "applied": got}
        if stop:
            break
    # RE-HASH THE LOCAL TREE, BEFORE RETURNING. Without this the local tree still describes the store
    # as it was BEFORE the leaves were applied, so the next round diffs identically against the same
    # peer, re-downloads the same leaves, applies the same rows, and updates nothing again — a
    # livelock that burns S3 bandwidth forever and never converges. We refresh the WHOLE local tree
    # (not just the coarse indices we pulled): applied rows land in LOCAL leaves at the local
    # resolution, which need not equal the coarse indices we fetched, and `refresh_leaves` recomputes
    # from the incremental `merkle_leaves()` in one pass anyway.
    refreshed = (refresh_leaves(store, range(len(local)), leaves=len(local))["refreshed"]
                 if fetched else 0)
    out_r = {"applied": applied, "peers": peers, "leaves_fetched": fetched, "refreshed": refreshed,
             "secs": round(time.time() - t0, 1), "node": me}
    # A leaf that keeps differing round after round with `applied` making no progress is the
    # signature of unordered concurrent authorship, not of a stuck transfer. Report the two side by
    # side so they cannot be mistaken for each other.
    if _vertices(store) is not None:
        out_r["unordered"] = unordered_report(store)
        # ⭐ The genuinely-NEW split (#3) — lattice nodes ONLY, where `version_of` makes it computable.
        # `new` = rows this cycle we did not already hold; `applied` counts those PLUS no-op re-applies.
        # `worker._healthy_progress` judges consumer health on `new` when present (so 'busily rewriting
        # what we already have' no longer reads as progress); it falls back to `applied` when absent.
        out_r["new"] = stats["new"]
        out_r["reapplied"] = stats["reapplied"]
    return out_r


def reconcile_via_s3(store, *, max_leaves: int = 256,
                     max_seconds: float = 0.0) -> Dict[str, Any]:
    """One anti-entropy round over the Merkle plane — THE sync path. Publish this node's tree
    incrementally, then pull ONLY the leaves that differ from each peer (vertices AND edges).

    This is the whole of sync: there is no segment feed beside it. Bulk catch-up and steady state are
    the SAME operation — a fresh node finds every leaf differing and pulls them all; a converged node
    exchanges one 32 KB tree and stops. `max_leaves=0` makes a round publish-only (the shard-node
    role). Both halves are idempotent and order-independent, so a crash mid-round just retries next
    round with nothing to undo.

    Driven by `op.mesh.reconcile`, called every cycle by the aggregator/daemon loop. Publish is the
    incremental path (no full rescan), so a round is cheap enough to run continuously. The time budget
    is split evenly: a slow publish must not starve reconcile, nor the reverse."""
    half = (max_seconds / 2.0) if max_seconds else 0.0
    pub = publish_merkle_incremental(store, max_seconds=half)
    rec = reconcile_merkle(store, max_leaves=max_leaves, max_seconds=half)
    return {"published_leaves": pub.get("uploaded", 0), "publish_reason": pub.get("reason"),
            "root": pub.get("root") or rec.get("root"),
            "applied": rec.get("applied", 0), "peers": rec.get("peers", {}),
            "leaves_fetched": rec.get("leaves_fetched", 0),
            "refreshed": rec.get("refreshed", 0), "node": _node_id()}


def reach_index(store, missing_id: str, *, max_publishers: int = 4) -> Dict[str, Any]:
    """REACH — pull a MISSED index row from the authoritative substrate and cache it locally.

    This is what lets an ember be LIMITED: it need not hold the whole graph. When a lookup misses (an
    id the node does not hold), reach finds the leaf that WOULD contain it in a publisher's Merkle tree
    in S3, pulls that one leaf (the existing content-addressed transfer unit), and applies it — so the
    id, and its leaf-neighbours, are now held. Same machinery as `reconcile_merkle`, but TARGETED at
    one need instead of converging the whole tree. Bodies still fetch on-miss from CAS; this is the
    index half of the same idea.

    A miss is a NEED, not an error (offers/needs): if no publisher holds it, reach returns
    `reached=False` and the caller refuses honestly — it does not fabricate. Eviction/demurrage of what
    is reached is a LATER layer (the demand cache); here reach only FETCHES, and what it applies is
    held like any consumed row until that layer bounds it.

    The publisher's leaf resolution is read from ITS published summary (`natural_leaves` differs per
    node), so `leaf_of(id, len(their_tree))` is the leaf on THAT node — no shared constant needed."""
    import json
    from . import merkle
    v = _vertices(store)
    if v is not None and v.get_artifact(str(missing_id)) is not None:
        _demand_touch(store, missing_id)          # re-reaching a held row is still demand
        return {"reached": True, "already_held": True}
    s3 = _mesh_s3(store)
    if s3 is None:
        return {"reached": False, "reason": "no-s3"}
    me, f = _node_id(), _fernet(store)
    skip = _ignored_nodes()
    # WHICH PEERS TO TRY, MEASURED-BEST FIRST. Peers this ember KNOWS (peer-artifacts, each carrying a
    # peer's node id + CAS-addressed manifest) come first, ranked by the demand mass on each — attention
    # flows to the peer that has actually been useful. Then any publisher not yet known as a peer, for
    # DISCOVERY. Nothing pre-judged and no hardcoded peer list: the order is what use has measured.
    order, seen = [], {me} | set(skip)
    for nid in _reach_candidates(store):
        if nid and nid not in seen:
            order.append(nid); seen.add(nid)
    blind = None
    try:
        for key in _s3_keys_after(s3, _MESH_MERKLE_PREFIX):
            n = key[len(_MESH_MERKLE_PREFIX):].strip("/")
            if not n.endswith(".json") or "/" in n:
                continue
            n = n[:-len(".json")]
            if n and n not in seen:
                order.append(n); seen.add(n)
    except MeshListingError as e:
        blind = str(e)[:120]
        if not order:
            return {"reached": False, "reason": "blind", "err": blind}
    tried = 0
    for node in order:
        try:
            theirs = merkle.load(json.loads(
                s3.get("%s%s.json" % (_MESH_MERKLE_PREFIX, node)).decode("utf-8")))
        except Exception:
            continue
        if not theirs:
            continue
        li = merkle.leaf_of(str(missing_id), len(theirs))     # the leaf on THAT peer's tree
        try:
            raw = f.decrypt(s3.get("%s%s/%05d.ndjson.enc" % (_MESH_LEAF_PREFIX, node, li)))
            items = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
        except Exception:
            continue                                          # this peer's leaf is absent/bad
        tried += 1
        got = _apply_artifacts(store, items) + _apply_edges(store, items)
        if v is not None and v.get_artifact(str(missing_id)) is not None:
            # The whole leaf is now CACHED. Mark every fetched row as demand (evictable) so nothing a
            # node did not author is silently pinned; the requested id is the hot one (touched twice).
            for it in items:
                iid = it.get("id")
                if iid and _is_replicated(it.get("content_type")):
                    _demand_touch(store, iid)
            _demand_touch(store, missing_id)      # the specifically-requested id — hotter
            _demand_touch(store, node)            # the PEER was useful — attention accretes on ITS
            #                                       peer-artifact, so next reach tries it sooner
            return {"reached": True, "from": node, "leaf": li, "applied": got}
        if tried >= max_publishers:
            break
    return {"reached": False, "reason": "not-found", "publishers_tried": tried, "blind": blind}


def _demand_touch(store, artifact_id) -> None:
    """Record a reach/access as demand (mass), so the cache knows what to keep and what to shed.
    Best-effort: demand tracking must never fail an answer."""
    try:
        from . import demand as _demand
        _demand.touch(store, str(artifact_id))
    except Exception:
        pass


def resolve(store, artifact_id: str, *, reach: bool = True) -> Optional[Dict[str, Any]]:
    """The one-call front door for a LIMITED ember: return the artifact if held, else REACH for it and
    return it. `reach=False` answers only from what is held (offline / no substrate). Returns None when
    the id is genuinely unreachable — an honest miss the caller turns into an honest refusal."""
    v = _vertices(store)
    if v is None:
        return None
    a = v.get_artifact(str(artifact_id))
    if a is not None:
        _demand_touch(store, artifact_id)         # a hit is demand too — it keeps the row warm
        return a
    if not reach:
        return None
    reach_index(store, artifact_id)               # reach records its own demand on success
    return v.get_artifact(str(artifact_id))


# ── peers are artifacts too — each carrying a peer's CAS address (John, 2026-07-28) ───────────────
# An observer is not special. It is an artifact whose CONTENT is a CAS-addressed manifest of its
# MEASURED state (its Merkle root = a fingerprint of what it holds, its leaf count, its measured
# envelope). An ember publishes ITSELF as one, and receives every other ember's through the mesh like
# any artifact — so it holds the CAS address of each peer. There is no separate directory file and no
# declared peer list: peers are discovered as artifacts and attended to by measured demand.
_OBSERVER_CT = "application/vnd.agience.observer+json"


def _envelope_bytes(store) -> Optional[int]:
    """This node's MEASURED envelope (data-volume free / cgroup mem), or **None if unmeasurable**.
    Part of the manifest so a peer can see what this ember can carry — measured, never declared."""
    try:
        from prism import envelope as resource
        kd = str(getattr(store, "keys_dir", "") or ".")
        # ⛔ FELL THROUGH TO `mem_limit_bytes()`, WHICH FABRICATED 8 GB — so an unmeasurable node
        # PUBLISHED 8 GiB to its peers as a measured envelope, and the docstring's own promise
        # ("measured, never declared") was false on exactly that path. 0 now means unmeasurable,
        # and `publish_manifest` omits the key rather than advertising a number nobody read.
        # ⛔ AND THEN RETURNED 0, WHICH `publish_manifest` PUBLISHED ANYWAY. The comment above
        # promised the key would be omitted; it never was (`"envelope": _envelope_bytes(store)`,
        # unconditional), so an unmeasurable node advertised `envelope: 0` — a NUMBER, meaning "I
        # can carry nothing", which is a claim rather than an absence. 0 is also what a genuinely
        # full disk measures, so the two were the same wire value. None now means unmeasurable and
        # the key really is omitted; 0 is reserved for a real, measured zero.
        free = resource.disk_free_bytes(kd)
        if free is not None:
            return int(free)
        mem = resource.mem_limit_bytes()
        return int(mem) if mem is not None else None
    except Exception:
        return None


def publish_manifest(store) -> Dict[str, Any]:
    """Publish THIS ember as a peer-artifact with a CAS-ADDRESSED manifest — because a peer is an
    artifact too. The manifest is the ember's measured state; it is content-addressed into
    `cas/<sha256>` (encrypted under the fleet key, like all content), and the self peer-artifact
    (`id = this node`, `ct = observer`) points at it via `content_ref`. A changed state → a new sha →
    a new content_ref → the peer-artifact versions itself, and propagates through the mesh. Other
    embers then hold the CAS address of this peer."""
    import json
    import hashlib
    from . import merkle
    v = _vertices(store)
    if v is None:
        return {"published": False, "reason": "not-a-lattice-store"}
    s3, f = _mesh_s3(store), _fernet(store)
    node = _node_id()
    cur = store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}
    live = [int(x) for x in (cur.get(_MERKLE_LIVE) or [])]
    manifest = {"node": node, "leaves": len(live) or _store_leaves(store),
                "root": merkle.root(live) if live else 0}
    # OMITTED when unmeasurable, which is what the envelope contract has always claimed and never
    # did: an absent key is an absence; `envelope: 0` is this node telling every peer it can carry
    # nothing. A peer must be able to tell "unknown" from "measured zero".
    _env = _envelope_bytes(store)
    if _env is not None:
        manifest["envelope"] = _env
    body = _jcs_string(manifest).encode("utf-8")
    ref = "cas/" + hashlib.sha256(body).hexdigest()
    if s3 is not None:
        try:
            if not s3.exists(ref):
                s3.put(ref, f.encrypt(body), "application/octet-stream")
        except Exception:
            pass
    # the self peer-artifact — content-addressed, versioned as the state changes, replicated like any
    store.artifacts.put_artifact({"id": node, "content_type": _OBSERVER_CT, "content_ref": ref,
                                  "node": node, "leaves": manifest["leaves"],
                                  "root": manifest["root"]})
    return {"published": True, "node": node, "content_ref": ref, "root": manifest["root"]}


def peers(store) -> List[Dict[str, Any]]:
    """The peer-artifacts this ember holds — one per peer it knows, each carrying that peer's CAS
    address (`content_ref`) and node id. THIS is the observer directory: not a special file, just
    artifacts, discovered through the mesh and attended to by measured demand (§ _reach_candidates)."""
    v = _vertices(store)
    if v is None:
        return []
    me = _node_id()
    docs, _ = v.list_by_content_type(_OBSERVER_CT, cap=100000)
    return [d for d in docs if d.get("node") and d.get("node") != me]


def _reach_candidates(store) -> List[str]:
    """Peer NODES to try for a miss, MEASURED-BEST FIRST: the peers this ember knows (peer-artifacts),
    ranked by the demand mass on each (raw mass — the eviction sweep decays; for ranking, the accreted
    mass is monotone in usefulness). Attention flows to the peer that has been useful; nothing is
    pre-judged. Publishers not yet known as peers are appended by `reach_index` for discovery."""
    v = _vertices(store)
    if v is None or not hasattr(v, "demand_get"):
        return []
    scored = []
    for p in peers(store):
        nid = p.get("node")
        if not nid:
            continue
        d = v.demand_get(nid) or {}
        scored.append((float(d.get("mass", 0.0)), nid))
    scored.sort(reverse=True)                 # most-attended peer first
    return [nid for _m, nid in scored]


# `_q` (`aid.replace("'", "")`) IS DELETED, NOT MOVED. It was never an escape — it DELETED the
# apostrophe from the value, and both its callers fed it a KEYSET CURSOR. That silently skipped rows:
# stripping the quote from `foo's` yields `foos`, and `'` (0x27) sorts before `s` (0x73), so
# `foos` > `foo's` — the cursor jumped FORWARD past every id between the two and those artifacts were
# never yielded to `digest`/`bucket_ids` at all. Ids come from ingested content (wiki titles etc.) so
# apostrophes are common, and the corruption was invisible: a short scan looks exactly like a small
# corpus. Left un-stripped the same literal terminates the string and throws "Error on transaction
# commit" — measured, it broke T5/TU sync on every cycle when the id keyset first shipped. Both
# failure modes vanish under parameterization, which is what `publish_to_s3` already does at the
# `id > :cur` page above and what arcade.py does everywhere else.




