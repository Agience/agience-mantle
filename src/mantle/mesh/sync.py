"""Anti-entropy mesh sync — memory-bounded, content-addressed, adaptive.

Two peers converge by comparing a compact DIGEST, never by re-scanning each other. The digest is a
per-bucket XOR of artifact-id hashes: XOR is commutative (order-independent → streamable) and
homomorphic over sets, so two boxes holding the same id-set produce identical per-bucket values, and a
bucket whose values differ is exactly where their sets diverge. Only the artifacts missing on the
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
    "application/vnd.agience.task+json",
    # `services/mirror_drain.py`'s own pool, separate from the shared one above so no other
    # service's worker can claim from it. A per-box obligation either way: this node's list of
    # what it has not managed to upload is not an observation about the universe.
    "application/vnd.agience.mirror-task+json",
    "application/vnd.agience.mesh-cursor+json",
    "application/vnd.agience.content-cursor+json", "application/vnd.agience.shard-done+json",
    "application/vnd.agience.s3sync-cursor+json",   # S3-sync publish/consume cursors are per-box
    # `db/lattice_api.py::mark_materialized` records that this box has indexed an artifact, and
    # `services/workspace_service.py:1164,1197` read it as a skip condition. Replicated, a marker
    # written on node A would make node B skip indexing an artifact B has never indexed — silently,
    # with nothing to see but a search result that is not there.
    "application/vnd.agience.materialized-marker+json",
    # A sensor reading describes one box: a node's own services, store, authority, certificate and
    # code, written into `host.<id>`. Replicated, a reading
    # written on node A would claim node B's services are whatever A measured, its disk is A's disk,
    # and its certificate expires when A's does. Every one of those is confidently wrong rather than
    # missing, which is the worse failure: an operator would act on it.
    #
    # `sensor_common.py` carries this as a warning — "this content type must not be minted on a
    # node with MESH_ROLE set" — and this line is the enforcement: a rule written in a comment
    # beside the code that would violate it is not a rule.
    "application/vnd.agience.sensor+json",
    "application/x-ember-state",
}


def _is_replicated(content_type) -> bool:
    """Whether an artifact type crosses the mesh at all.

    Probe/throwaway types are excluded by prefix rather than by exact name: a probe fixture whose
    `_rev` is stamped far in the future can replicate to every node and pin publish cursors beyond
    every real row, muting nodes permanently. A test fixture must never reach production
    replication, and matching on a prefix means a future probe type nobody remembered to register
    is still excluded."""
    ct = str(content_type or "")
    if ct in _OP_EXCLUDE:
        return False
    return not (ct.startswith("application/x-probe")
                or ct.startswith("application/vnd.agience.probe")
                or "throwaway" in ct)


# ── the store: the lattice, and only the lattice ──────────────────────────────────────────────────
#
# `_origin` names the authoring observer of a replicated event; a publish cursor is not one, so
# operational rows are stamped with a reserved origin that pins them outside every real observer's
# sequence: `page_by_origin(origin=me)` skips them structurally rather than by a filter someone can
# forget, so operational state cannot enter the publish feed even by accident.
#
# The shape is `("_local:<artifact_id>", 1)` — one degenerate origin per operational row, built by
# `_op_origin` below. A shared origin for every row would fail the `(_origin,_seq)` uniqueness
# constraint, and a hashed shared origin would look to node-repair like a peer whose proper time is
# full of gaps — indistinguishable from a consume cursor that advanced past a segment that did not
# apply, the one thing that check exists to catch. A per-row origin avoids both: each row's sequence
# is trivially `1..1`, gap-free, unique, and stable across rewrites, so a cursor update does not
# churn a merkle leaf.
_LOCAL_ORIGIN = "_local"        # reserved prefix; never a real node id, never replicated


def _op_origin(artifact_id: str) -> str:
    return "%s:%s" % (_LOCAL_ORIGIN, artifact_id)


#
# An edge is `{f, t, label, props}` on the wire and carries no `(_origin, _seq)`, so a consumed edge
# arrives with no provenance to preserve. Edge provenance was never guaranteed, so the edge
# `_origin` need not be accurate (contract §5.8.2) — it must only not be `me`.
#
# That is echo suppression, the same mechanism `_op_origin` relies on: a row whose `_origin` is not
# `me` is excluded from the publish scan structurally, by the index range.
#
# The shape is `("_local:edge:<digest>", 1)` — one degenerate origin per edge, for the reasons
# recorded above `_op_origin`. `<digest>` is deterministic in the edge triple (contract §3's
# NUL-separated blake2b, the same construction as `edge_key`), so re-consuming a segment produces
# the identical `(_origin, _seq)`: `add_edges(stamp_rev=False)` compares versions, reads the same
# value, and does not write. A replayed segment allocates no proper time and churns no leaf.
#
_CONSUMED_EDGE_NS = _LOCAL_ORIGIN + ":edge:"


def _consumed_edge_origin(src: str, dst: str, label: str) -> str:
    """A reserved, deterministic, per-edge origin. NUL-separated, per contract §3 — the separator
    is load-bearing: without it ("ab","c") and ("a","bc") hash alike."""
    h = hashlib.blake2b(("%s\0%s\0%s" % (src, dst, label)).encode("utf-8"), digest_size=16)
    return _CONSUMED_EDGE_NS + h.hexdigest()


def _put_op(store, doc: Dict[str, Any]) -> None:
    """Write a local operational row (cursor, watermark). Never replicated, never versioned.

    One function for every call site, so the invariant — `stamp_rev=False`, on a row pinned outside
    every observer's proper time — is stated once and cannot drift between them."""
    # A row written through here is pinned outside every observer's proper time, so if it were a
    # replicated type it would be unpublishable and invisible to every peer, silently. Catch the
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
    """This store's derived Merkle resolution — `natural_leaves(corpus)`, held on the store and kept
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
    """This observer's identity as the store understands it.

    Deliberately the store's `origin`, not `_node_id`. `_seq` is scoped `WHERE _origin =:me`, so
    a publish scan keyed on a different string than the one stamped at write returns zero rows —
    forever, silently, on a node that is writing normally. The env var and the store must agree, and
    when they do not it is the store that is right, because the store is what stamped the rows."""
    v = _vertices(store)
    return str(getattr(v, "origin", "") or "") if v is not None else _node_id()


# ── the publish backlog — Merkle-native (one path) ───────────────────────────────────────────────
# There are no feed cursors. "Unpublished local work" is the tree maintained incrementally on write
# (`_MERKLE_LIVE`) minus what is actually backed by an uploaded leaf object (`_MERKLE_PUBLISHED`) —
# the changed leaves a peer cannot yet see. It is O(leaves) integer compares: it counts leaf state
# directly and never subtracts seq allocations, so update churn cannot inflate it.


def publish_backlog_now(store) -> Dict[str, Any]:
    """How many of this node's Merkle leaves have changed but are not yet published (live != published).

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
    """Every row, replicated or not. A counter lookup on lattice — no `count(*)` reaches SQLite:
    EXPLAIN shows it loads every row to produce one integer, which can OOM a process on a large
    corpus."""
    v = _vertices(store)
    if v is not None:
        return int(v.count())
    return int(_require_lattice(v, "_total_count"))


def _count_of_type(store, ct: str) -> int:
    v = _vertices(store)
    if v is not None:
        return int(v.count_by_content_type(ct))
    return int(_require_lattice(v, "_count_of_type"))

# ── S3 mesh plane — S3 is the authoritative store and the exchange plane ───────────────────────────
#   Content is content-addressed in `cas/<sha>`. The graph is authoritative in S3 too, as each node's
#   Merkle tree: a 32 KB summary plus per-leaf objects. One path — no segment log, no cursors. A box
#   publishes its tree and pulls the leaves that differ; a wiped/new box rebuilds by the same pull.
#   Every object is Fernet ciphertext under the shared content.key (OVH sees opaque bytes).
_MESH_MERKLE_PREFIX = "mesh/merkle/"     # per-node tree summary (root + leaf digests) — 32 KB, JSON
_MESH_LEAF_PREFIX = "mesh/leaf/"         # per-node, per-leaf row set (vertices + edges) — unit of transfer
_S3_MERKLE_CURSOR = "s3.merkle.cursor"
# Two distinct digest keys live on that cursor, and conflating them makes publish silently skip
# uploads:
#   digests   = live: what the local corpus hashes to right now. `refresh_leaves` writes this after
#               every reconcile/mutation without uploading anything to S3.
#   published = what is actually backed by an uploaded leaf object in S3.
# `changed` must be computed against `published`. Computed against `digests`, a leaf that
# refresh_leaves had already recomputed looks unchanged, so no leaf file is ever written even
# though the root advertises it — peers then fetch a key that 404s and stay divergent, silently.
_MERKLE_LIVE = "digests"
_MERKLE_PUBLISHED = "published"
_S3SYNC_CT = "application/vnd.agience.s3sync-cursor+json"


#: A directory to use as the mesh plane instead of S3, decomposed as trust -> plane -> sweep: the
#: mesh plane is for peering, and durability is a separate concern from it.
#:
#: The mesh plane still requires S3 today: `mantle_common.sh` states that the mesh plane IS the
#: durable content tier — there is no separate mesh bucket — and `CONTENT_DURABLE_BUCKET` is
#: refused-if-absent. "Peering, not backup" has not escaped S3; it only changes what the S3 is for.
#:
#: `mesh/carrier.SpoolPlane` — a directory that is a mesh plane — is exported with no caller:
#: measured 2026-08-26, `grep SpoolPlane` finds only its definition and its `__all__` entry.
#: `MESH_SPOOL_DIR` is that caller.
MESH_SPOOL_DIR = "MANTLE_MESH_SPOOL_DIR"


def _mesh_s3(store):
    """The mesh plane: a spool directory if one is configured, else the authoritative OVH S3 client.

    A plane is three verbs — `put` / `get` / `exists` — plus a boto-shaped `_s3` for listing. That is
    the whole contract, which is why a directory can stand in for a bucket and *"the mesh cannot tell
    the difference"*. Returns None when neither is available, and the caller treats that as "no
    plane" exactly as before.

    The spool is checked first, deliberately: it is an explicit operator choice — the variable is
    unset everywhere today — and a node that names a spool has said which plane it means. Falling
    through to S3 after that would silently prefer the bucket over the instruction.

    It is a plane, not a backup, and the difference is durability. A spool directory is exactly as
    durable as the directory: on a local disk it is a single-box plane, useful for a loopback mesh, a
    test, or a carrier that spools frames off the air. Pointed at shared storage it is a real
    multi-node plane. It is not a second copy of anything — nothing here replicates the spool
    itself, and treating it as durable because peering runs over it is the confusion "peering, not
    backup" exists to end.
    """
    import os

    spool = (os.getenv(MESH_SPOOL_DIR) or "").strip()
    if spool:
        from mantle.mesh.carrier import SpoolPlane
        return SpoolPlane(spool)

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




def _ignored_nodes() -> set:
    """Publisher ids to skip entirely (`EMBER_MESH_IGNORE`, comma-separated).

    A process started without `EMBER_NODE_ID` publishes under its hostname, so the mesh can
    accumulate phantom publishers that are really an existing node wearing a different name. Their
    segments are byte-duplicates of the real node's, so every peer burns cycles re-applying data it
    already has.

    Skipping is preferred over deleting the S3 segments: deletion is irreversible, and these streams
    may be the only copy of anything they contain that was written while the node was
    mis-identified. Ignoring is reversible, needs no coordination, and costs nothing to undo."""
    import os
    raw = os.getenv("EMBER_MESH_IGNORE", "")
    return {p.strip() for p in raw.split(",") if p.strip()}


def mesh_lag(store) -> Dict[str, Any]:
    """How many leaves differ from each peer — the Merkle-native 'how far behind'.

    Read-only: fetches each peer's 32 KB tree summary and diffs it against the local live tree;
    never pulls. This is the mesh's self-observation, the signal the 5-minute online SLO is measured
    against. `converged` is the positive fact that every peer's root matched (0 differing leaves),
    and it is None (unknown) rather than True whenever a listing failed or a peer's tree could not
    be read, so the SLO cannot read green while blind. `segments_behind` keeps its name for the
    health board and worker; the unit is differing leaves (there are no segments)."""
    import json
    from . import merkle
    s3 = _mesh_s3(store)
    if s3 is None:
        return {"reason": "no-s3"}
    me = _node_id()
    skip = _ignored_nodes()
    local = (store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}).get(_MERKLE_LIVE)
    if not local:
        # No local tree yet -> nothing to compare -> unknown, not converged.
        return {"node": me, "peers": {}, "segments_behind": 0, "converged": None,
                "blind": ["no-local-tree"], "stuck": []}
    local = [int(x) for x in local]
    peers: Dict[str, Any] = {}
    total, blind = 0, []
    try:
        keys = _s3_keys_after(s3, _MESH_MERKLE_PREFIX)     # flat `mesh/merkle/<node>.json` objects
    except MeshListingError as e:
        # A failed listing is not the same as being caught up. Refuse to call this converged.
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




class MeshListingError(RuntimeError):
    """S3 could not be listed. Distinct from 'S3 listed successfully and was empty' — a swallowed
    listing failure and a genuinely empty bucket both look like 'no peers, no segments', which would
    let `mesh_lag` report converged exactly when the mesh is blind. Raising instead makes the caller
    decide, and no caller is allowed to decide 'converged' on a failed listing."""


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
    """The docs in `batch` whose id is not present locally — the genuinely-new rows, distinguished
    from no-op re-applies. Uses the same keyed `version_of` read `_split_unordered` already does
    (indexed by id), so it is cheap and backend-agnostic. Returns None on a non-lattice store (no
    `version_of`), where the split is not computable — an honest 'unknown', never a guess."""
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


#: Content types that confer authority. An inbound row of one of these decides what a principal may
#: do, so applying it unverified means write access to the mesh plane is lattice authority.
_AUTHORITY_TYPES = ("+grant+json", "grant+json", ".grant", "vnd.agience.grant")


def _confers_authority(content_type) -> bool:
    ct = str(content_type or "").lower()
    return any(marker in ct for marker in _AUTHORITY_TYPES)


def _withhold_unverified_authority(batch, *, stats=None):
    """Refuse inbound rows that confer authority while nothing verifies them. `(kept, refused)`.

    The peering daemon has exactly one sync mechanism — its own comment says "the one path …
    no second mechanism" — and it verifies nothing: `_apply_artifacts` filters on `id` +
    content-type and upserts, and `_is_replicated` returns True for grants. An inbound grant from
    anyone who could write to the plane becomes a grant in this lattice, with no signature, no
    principal and no authority checked — write access to the plane would equal full lattice
    authority.

    This fails closed, and visibly, rather than quietly dropping grants out of `_is_replicated`. A
    type silently removed from the replicated set is indistinguishable from one nobody publishes,
    and the day signing lands nobody would know to put it back. Refusing at apply time keeps the
    intent ("grants replicate") and states the precondition ("when they can be verified"), counts
    every refusal into `stats`, and logs the first one per batch.

    No node declares `MANTLE_RUN_MESH` or `MESH_ROLE` across either peer tier — the mesh runs
    nowhere today — so this closes the hole before it opens rather than changing the behaviour of a
    running system.

    The verifier is not wired here: `mesh/node.py::sync_from` takes an `authority_pub` and raises on
    tamper, but it belongs to the shard/region subsystem (`anchor_routing`, `directory`, `service`)
    and is not an alternative wiring of this daemon — signing this path is work, not a config
    choice. Until that work lands, "verify what it applies" can only mean "refuse what it cannot
    verify".
    """
    if not batch:
        return batch, 0
    kept, refused = [], []
    for d in batch:
        (refused if _confers_authority(d.get("content_type")) else kept).append(d)
    if refused:
        if stats is not None:
            stats["refused_unverified_authority"] = (
                stats.get("refused_unverified_authority", 0) + len(refused))
        log.warning(
            "mesh apply refused %d inbound authority row(s) — nothing on this path verifies a "
            "grant's signature or issuing principal, so applying one would make write access to "
            "the plane equal lattice authority (ruling 1, TRUST). First: id=%r type=%r",
            len(refused), refused[0].get("id"), refused[0].get("content_type"))
    return kept, len(refused)


def _apply_artifacts(store, items: List[Dict[str, Any]], *, stats: Optional[Dict[str, int]] = None) -> int:
    """Upsert consumed artifact docs. `stamp_rev=False` preserves the origin `_rev` so a replicated
    row keeps its origin revision and does not echo around the mesh forever.

    `stats` (optional out-param) collects the genuinely-new split additively — `stats['new']` /
    `stats['reapplied']` (via `_new_docs`, a read) — so a caller can tell real convergence from
    no-op re-applies. Populating it changes nothing about what is written or the load-bearing
    `handled` return / cursor guard; the default (None) is a pure no-op."""
    batch = [d for d in items if d.get("id") and _is_replicated(d.get("content_type"))]
    batch, refused_authority = _withhold_unverified_authority(batch, stats=stats)
    if not batch:
        return refused_authority
    nd = None
    if stats is not None or _DELIVER_PEER_SIGNALS:   # additive metric — a read, never the write path
        nd = _new_docs(store, batch)
    if stats is not None and nd is not None:
        stats["new"] = stats.get("new", 0) + len(nd)
        stats["reapplied"] = stats.get("reapplied", 0) + (len(batch) - len(nd))
    #
    #
    declined = 0
    if _vertices(store) is not None:
        batch, declined = _split_unordered(store, batch)
        if not batch:
            return declined
    written = store.artifacts.put_many(batch, batch=500, stamp_rev=False)
    n = written if isinstance(written, int) else len(batch)
    n += refused_authority
    if n - refused_authority < len(batch):
        # A short write means the segment did NOT fully apply. Raise so the caller holds the cursor
        # and records failed_key, rather than silently banking partial progress.
        raise RuntimeError("partial apply: wrote %d of %d" % (n, len(batch)))
    if _vertices(store) is not None:
        _enqueue_consumed(store, batch)
    _deliver_new(store, nd)
    # A declined row is handled, not lost. It was examined, a decision was recorded, and the peer
    # still holds its copy — exactly the status `put_many` gives an LWW-rejected doc. Counting it
    # keeps the caller's `written < len(batch)` guard meaning "something errored", which is the one
    # thing that must hold the cursor.
    return n + declined


# ── peer-apply reaches cognition ──────────────────────────────────────────────────────────────────
#
# An arriving peer signal is routed through the electroweak switch by `signal.deliver`.
#
# Inactive by default (`EMBER_DELIVER_PEER_SIGNALS`), and `forward` is not passed at all: local
# grounding is the whole story on a plain node and nothing is dropped. The onward persona hop needs a
# live carrier and reactor and stays a separate step.
#
# Fire-and-forget: delivery must not touch the return value. The caller's `written < len(batch)`
# guard is what holds the mesh cursor; letting a delivery failure change `handled` would make a
# cognition error look like a partial write and strand a segment behind a monotone marker. So this
# runs after the write, records failures loudly, and returns nothing.
log = logging.getLogger(__name__)

_DELIVER_PEER_SIGNALS = os.getenv("EMBER_DELIVER_PEER_SIGNALS", "").strip().lower() in (
    "1", "true", "yes", "on")


#: Where new peer docs go for cognition. A callable `(store, new_docs) -> None`, or None.
#:
#: A store tells you rows arrived; deciding that this means something is the runner's job.
#: Ember wires this at boot; anything else replicating rows simply does not, and nothing is
#: missing on a plain node.
_PEER_SIGNAL_SINK = None


def set_peer_signal_sink(fn) -> None:
    """Wire what happens when peer replication brings in genuinely new docs.

    `fn(store, new_docs)`. Pass None to unwire. Public because the wiring crosses a repo boundary —
    the alternative is a caller assigning a module private from another package."""
    global _PEER_SIGNAL_SINK
    _PEER_SIGNAL_SINK = fn


def _deliver_new(store, new_docs) -> None:
    if not _DELIVER_PEER_SIGNALS or not new_docs:
        return
    sink = _PEER_SIGNAL_SINK
    if sink is None:
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


# ── consumed peer rows are invisible to every local-origin walk ──────────────────────────────────
#
# `page_by_origin` walks `WHERE _origin =:me`. A consumed row carries the peer's `_origin` by
# design — that is what "preserve the origin's version" means — so it is invisible to every
# local-origin scan on this node, including `content_tier.promote_local_content`. Its content
# therefore never gets promoted to durable S3 by that path: it exists only on local Garage and dies
# with the box unless something else records that it was consumed.
#
# So: neither forge a version nor rescan. Record what was consumed, at the only point that knows.
# A bounded local queue of consumed ids needing content promotion is O(1) per consumed row, forges
# nothing, claims nothing about authorship, and gives the content drain an exact work list instead
# of a scan. `_origin` and `_seq` on the row are untouched.
_CONSUMED_ID = "mesh.consumed.pending"
_CONSUMED_CAP = 50000

#: Rows per keyset page while enumerating a gated collection's members by `origin_root`.
#: Read both as the limit and by the end-of-walk test in `_private_set`, so the two can never
#: disagree — a test against a value larger than the limit would end a walk after one page and
#: under-report `withheld` while still claiming the enumeration was exhaustive.
#:
#: A per-round memory bound, not a judgement: the withheld set is identical at any page size.
#: The bound is `cap`, which the caller passes and which turns a non-exhaustive enumeration into
#: a refusal to publish — this constant only sizes the walk, not the outcome.
_GATED_PAGE = 5000


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
        merged = merged[:_CONSUMED_CAP]
    _put_op(store, {"id": _CONSUMED_ID, "content_type": _S3SYNC_CT, "state": "committed",
                    "pending": merged, "n": len(merged), "overflow": overflow})


def consumed_pending(store) -> Dict[str, Any]:
    """The content drain's work list: peer-authored rows this node holds whose content may not be
    promoted yet. `drain_consumed(ids)` removes them once promoted.

    `overflow: True` means the queue is incomplete and a full id sweep is required — treat the list
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


# ── unordered concurrent authorship ───────────────────────────────────────────────────────────────
#
# Two observers can author the same vertex id independently. `(_origin, _seq)` cannot order them and
# must not: contract RESOLVED-3 removes the clock permanently and §C.7 states that "unordered is a
# valid answer". Under `on_unordered="keep_local"` both sides stay divergent, so their merkle leaves
# keep mismatching and anti-entropy re-offers the row every round.
#
# The row stays in the anti-entropy set — a divergence that were scoped out would also stop being
# checked, so a later, genuine divergence on the same row would go unnoticed forever. Instead this
# records a local declination: "I have seen version (O_r, S_r) of vertex X and did not adopt it",
# a statement about this observer's history rather than about which version is newer. Declining to
# adopt asserts nothing about order — `compare_version` still answers unordered, the peer still
# holds its copy, and nothing is deleted anywhere — whereas synthesizing a winner by clock, node id,
# or arrival order would invent information the contract does not license. This is the "private
# per-observer state" §1.2 describes: a shared vertex carries only what is universal, and
# interpretation of a genuine conflict stays local.
#
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

    Compares each doc's `(_origin, _seq)` against the local row before `put_many` sees it. `put_many`
    would reach the same verdict — it does its own LWW read — but it reports only an aggregate
    counter, so which vertex is diverging, and whether it is the same one every round, is
    unrecoverable from it. The extra keyed read per doc is what turns "the conflict counter went up
    again" into "vertex X has been unordered against origin Y for 400 rounds"."""
    from mantle.db import constants as K       # noqa: WPS433 — optional dep, lattice only
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
            # Past this many permanently-divergent vertices the per-id record is no longer the right
            # instrument and the situation is an incident, not a statistic. Keep a bounded,
            # deterministic slice so the artifact stays small, and record that the set is truncated
            # so nobody reads its size as the real number.
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
    `page_by_origin(origin=me)` and cannot echo back onto the mesh. See `_CONSUMED_EDGE_NS`."""
    edges = [(e["f"], e["t"], e.get("label") or "link", e.get("props") or {})
             for e in items if e.get("f") and e.get("t")]
    if not edges:
        return 0
    # A write failure must not be swallowed: returning 0 on exception would be indistinguishable
    # from "nothing to apply", so the caller would see no exception, record no failed_key, and
    # advance the cursor past an edge segment that never landed — a failed edge segment lost
    # permanently and invisible to mesh_lag["stuck"]. Raising is what makes the cursor-hold and
    # the stuck report work for edges at all.
    if _edges(store) is None:
        # ArcadeDB: `add_edges` has no `stamp_rev` keyword and `_rev` carries no authorship, so
        # there is nothing to preserve and nothing to reserve (contract §6.1 / Unit E §6).
        store.graph.add_edges(edges, batch=500)
        return len(edges)

    me = _origin_of(store)
    stamped = []
    for src, dst, label, props in edges:
        p = dict(props)
        o, sq = p.get("_origin"), p.get("_seq")
        if not (isinstance(o, str) and o and o != me and isinstance(sq, int)):
            # No usable provenance on the wire (the normal case — the edge segment format does not
            # carry it), or a peer claiming we authored its edge. Either way: reserved origin.
            p["_origin"] = _consumed_edge_origin(src, dst, label)
            p["_seq"] = 1
        stamped.append((src, dst, label, p))
    handled = store.graph.add_edges(stamped, batch=500, stamp_rev=False)
    if handled < len(stamped):
        # `_add_chunk` rolls a failing edge back per-savepoint and carries on, so a partial apply
        # does not raise on its own — the shortfall is the only evidence it happened. Unguarded,
        # `_consume_stream` would see no exception, write a later `last_key`, and drop those edges
        # permanently behind a monotone StartAfter marker. Same guard `_apply_artifacts` puts on
        # `put_many`, and for the same reason: handled < offered is data loss.
        raise RuntimeError(
            "edge segment partially applied: %d of %d edges handled. Holding the consume cursor "
            "so this segment is retried rather than silently skipped." % (handled, len(stamped)))
    return len(stamped)


def _withheld_endpoints(v, erecs, priv_set, *, cap: int = 200000):
    """The endpoint ids an edge must not be published toward, as `(ids, exhaustive)`.

    Leaf assembly filters vertices by `_is_replicated` and ships edges unconditionally. Excluding a
    content type that is an edge endpoint therefore sends the edge and withholds its vertex, and the
    peer ends up holding a membership edge to an artifact that is not there — the
    `contains_edges_to_missing_vertex` condition, which a deployment integrity gate pins at 0.

    Measured on 71/home: 18 `contains` edges point at `application/vnd.agience.shard-done+json`
    vertices, which are in `_OP_EXCLUDE` — the current store publishes 18 dangling edges the moment
    the mesh is switched on.

    Three reasons an endpoint is withheld:

      * its content type does not replicate (`_is_replicated`);
      * it is grant-gated (`priv_set`) — an edge would otherwise leak the existence of a withheld id
        even though the row itself is correctly held back;
      * it is not in this store at all — publishing an edge to a vertex we do not have propagates a
        dangling edge rather than creating one, which is no better.

    Bounded by the leaf, not by the store: only the endpoints named by `erecs` are looked up, so
    this stays O(edges in the leaf) however large the excluded population grows. Non-exhaustive
    makes the caller refuse to publish the leaf, the same contract `_private_set` has — a leaf whose
    edges cannot be resolved exactly must not be advertised as authoritative.
    """
    ids = set()
    for er in erecs or ():
        f, t = er.get("f"), er.get("t")
        if f:
            ids.add(f)
        if t:
            ids.add(t)
    if not ids:
        return set(), True
    if len(ids) > cap:
        return set(), False
    if v is None or not hasattr(v, "get_many"):
        # No typed vertex store: we cannot tell whether an endpoint replicates. Fail closed.
        return set(), False
    try:
        docs = v.get_many(ids)
    except Exception:
        return set(), False
    withheld = set()
    for aid in ids:
        d = docs.get(aid)
        if not d:
            withheld.add(aid)                       # not here: do not propagate a dangling edge
        elif not _is_replicated(d.get("content_type")):
            withheld.add(aid)
        elif aid in priv_set:
            withheld.add(aid)
    return withheld, True


def _private_set(store, *, cap: int = 200000):
    """The ids the mesh must not publish — the members (and container) of every grant-gated
    collection — as `(ids, exhaustive)`. What leaves the node is the ungated public set.

    The access decision is grants, computed by `ember.access` (the same light-cone the mantle
    service uses): a collection is non-public if and only if a grant gates it — not a flag and not
    a naming convention. With no grant present, nothing is gated and the whole Merkle path is
    byte-identical to a store with no access system at all.

    Membership is enumerated via the indexed `origin_root`. Non-exhaustive — over `cap`, or the
    grant system cannot be consulted — makes the caller refuse to publish rather than leak the
    remainder. Only a genuinely grant-less store returns the empty, exhaustive set."""
    v = _vertices(store)
    if v is None or not hasattr(v, "page_by_origin_root"):
        return set(), True
    try:
        from mantle.db import access
        access._api()                                  # probe: is the grant subsystem importable?
    except Exception:
        # No grant subsystem reachable at all (a minimal env without mantle's lattice_api on the path).
        # There are then no grants to consult, so nothing is gated — empty is correct, not a leak.
        return set(), True
    try:
        gated_cols = access.gated_collections(store)
        # The commons light-cone: collections/artifacts made public (a Read grant to the public entity)
        # are not withheld even though they sit in a gated collection — they mesh out like any public row.
        public_reach = access.reachable_collections(store, access.PUBLIC_PRINCIPAL)
    except Exception:
        # The subsystem is present but resolving grants failed. Fail closed: refuse rather than leak.
        return set(), False
    withheld, seen = set(), 0
    for root in gated_cols:
        if str(root) in public_reach:
            continue                                   # whole collection made public -> replicates
        withheld.add(str(root))                        # the private container does not mesh out
        after = ""
        while True:
            page = v.page_by_origin_root(str(root), after=after, limit=_GATED_PAGE)
            if not page:
                break
            withheld.update(m for m in page if str(m) not in public_reach)   # keep made-public members
            after = page[-1]
            seen += len(page)
            if seen > cap:
                return withheld, False
            # A short page is the last page — true against the limit that was asked for, so it is
            # tested against that same name. A test against a value larger than the limit would end
            # the walk after one page and leave the remaining members of a gated collection out of
            # `withheld` while `exhaustive` still returned True.
            if len(page) < _GATED_PAGE:
                break
    return withheld, True


def _refresh_leaves_lattice(store, leaves_to_refresh, n: int, cur: Dict[str, Any],
                            digests: List[int]) -> Dict[str, Any]:
    """Refresh leaf digests on a lattice store — from the incremental tree, minus operational rows.

    A re-query of the leaf is not viable: the Arcade form is one indexed `WHERE _leaf =:li` per
    leaf, but the lattice store exposes no `page_by_leaf`, so the equivalent would be a full keyset
    pass of the corpus per refresh — O(corpus) per reconcile round instead of O(operational rows).
    That defeats the reason this function exists: a full rescan is already expensive, and
    the corpus changes faster than the scan completes, so a rescan-built tree is stale before it
    finishes.

    `merkle_leaves` alone is not enough either. It is O(1) and incrementally maintained, which is
    the right shape, but `vertex._write_row` XORs every row into its leaf, including `_OP_EXCLUDE`
    cursors and tasks — publishing that tree as-is would give two perfectly converged nodes
    different roots forever, over state that must never replicate. This function corrects for that
    at the point that owns the exclusion policy: it takes the incremental tree and XORs the
    operational rows back out. XOR is its own inverse — that is why XOR-of-row-hashes was chosen
    over a sorted hash — so removing a row's contribution is the same operation as adding it. The
    operational set is enumerated with `list_by_content_type`, a typed indexed lookup, once per
    `_OP_EXCLUDE` type, so the cost is O(operational rows), not O(corpus)."""
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
    # Grant-gated (non-public) rows are subtracted the same way — held and queried locally, never
    # advertised. What leaves the node is the ungated public set. The access decision is grants
    # (`ember.access`), not a flag. XOR is its own inverse, so removing a row's contribution is the
    # same operation that added it on write.
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
    # `published` is preserved. This recomputes live digests and uploads nothing, so it must not
    # touch the record of which leaf objects exist in S3 — overwriting that would let a later
    # publish_merkle conclude "nothing changed" while no leaf file had ever been written.
    _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT, "state": "committed",
                    _MERKLE_LIVE: digests, _MERKLE_PUBLISHED: cur.get(_MERKLE_PUBLISHED) or [],
                    "root": merkle.root(digests)})
    return {"refreshed": len(want), "root": merkle.root(digests), "excluded_op_rows": excluded,
            "prefix_bans_unverified": True, "basis": "incremental-minus-operational"}


def refresh_leaves(store, leaves_to_refresh, *, leaves: int = 0) -> Dict[str, Any]:
    """Recompute the digests of specific leaves from the indexed `_leaf` column.

    This is what makes Merkle affordable to run continuously. A full rescan costs meaningfully more
    than an incremental refresh under load, and the corpus can change faster than a rescan completes
    while a node is catching up — so a rescan-built tree is stale before it finishes. Refreshing
    only the leaves that actually changed is an indexed lookup over ~1500 rows each instead.

    Note this recomputes a leaf from its current rows rather than XOR-ing a delta. Both are correct
    (XOR is its own inverse, so a delta would work), but recomputing is self-healing: it converges
    to the truth even if a previous digest was wrong, whereas a delta chain carries any past error
    forward silently. The self-correcting form is worth the extra query.

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
    # `published` is preserved. This function recomputes live digests and uploads nothing, so it
    # must not touch the record of which leaf objects exist in S3 — overwriting it would let a
    # later publish_merkle conclude "nothing changed" while no leaf file had ever been written.
    _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT,
                                  "state": "committed", _MERKLE_LIVE: digests,
                                  _MERKLE_PUBLISHED: cur.get(_MERKLE_PUBLISHED) or [],
                                  "root": merkle.root(digests)})
    return {"refreshed": done, "root": merkle.root(digests)}


def publish_merkle_incremental(store, *, leaves: int = 0,
                               max_seconds: float = 0.0) -> Dict[str, Any]:
    """Steady-state Merkle publish without a full corpus rescan.

    `publish_merkle` scans every row — meaningfully more expensive than the incremental path — to
    rebuild both the live digests and the per-leaf id buckets it needs to upload. At steady state
    that cost is avoidable: the lattice store
    already maintains the leaf digests incrementally on write, so live is recomputed by
    `refresh_leaves` in O(operational rows), and only the leaves whose digest actually moved since the
    last publish need their object rebuilt — each an indexed `list_by_leaf` lookup, not a scan. On a
    converged node that is one 32 KB summary and nothing else.

    This is the path the aggregator loop drives. `publish_merkle` (full rescan) stays the
    bootstrap/verification path, and the only safe path while `merkle_coverage < 100%` — a legacy
    Arcade store with unbackfilled `_leaf` — because only its full scan self-heals the prefix bans
    (`_refresh_leaves_lattice` cannot subtract a prefix-banned probe type by exact-name lookup).

    Lattice only: it rests on the incremental-minus-operational tree and on `list_by_leaf`. On a
    non-lattice store it declines rather than silently full-scanning under a name that promises
    'incremental' — the honest-refusal rule applied to a capability the backend lacks.

    Every write to `_S3_MERKLE_CURSOR` keeps live and published distinct and advances published only
    as far as leaves actually landed (contract M6); the summary is uploaded last so a peer never sees
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
        # counter compare); pays the O(N) re-stamp only on a rare sqrt boundary. Done before reading
        # `n` so a reshard's new tree is what this round publishes — the whole tree re-publishes at
        # the new resolution, which is correct (every leaf moved).
        try:
            v.maybe_reshard(graph=_edges(store))
        except Exception:
            pass
    n = int(leaves or _store_leaves(store))
    node, f = _node_id(), _fernet(store)
    t0 = time.time()
    # 1) Recompute live from the store's incremental tree, minus operational rows (M7). This writes
    #    _MERKLE_LIVE and preserves _MERKLE_PUBLISHED — refresh uploads nothing.
    rl = refresh_leaves(store, range(n), leaves=n)
    if rl.get("error"):
        return {"published": 0, "reason": "refresh-failed", "error": rl["error"]}
    cur = store.artifacts.get_artifact(_S3_MERKLE_CURSOR) or {}
    live = [int(x) for x in (cur.get(_MERKLE_LIVE) or [0] * n)]
    if len(live) != n:
        return {"published": 0, "reason": "live-shape", "have": len(live), "want": n}
    # 2) changed = live != published. -1 sentinel = "no object uploaded for this leaf yet", so a
    #    never-published leaf always compares changed (unsigned-64 digests can never be -1).
    prev = cur.get(_MERKLE_PUBLISHED) or []
    pub = ([int(x) for x in prev] + [-1] * n)[:n]
    changed = [i for i in range(n) if i >= len(prev) or int(prev[i]) != live[i]]
    # 3) Rebuild only the changed leaves from the indexed `_leaf` column; upload; advance published.
    #    A leaf object is a mixed NDJSON: vertex docs ({"id",…}) and edge records ({"f","t",…}). The
    #    tree covers both tables, so a changed leaf's object must carry both — `_apply_artifacts` takes
    #    the id-lines and `_apply_edges` the f/t-lines, each ignoring the other.
    g = _edges(store)
    # The same withheld set the digest was computed against (`_refresh_leaves_lattice` subtracted it),
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
        # An edge publishes only if BOTH its endpoints do — see `_withheld_endpoints`. Without
        # this, excluding a content type ships the edge and withholds the vertex, which hands the
        # peer a dangling membership edge.
        held, held_exh = _withheld_endpoints(v, erecs, priv_set)
        if not held_exh:
            incomplete = True
            continue
        lines += [json.dumps(er, separators=(",", ":")) for er in erecs
                  if er.get("f") not in held and er.get("t") not in held]
        # An empty leaf is still uploaded as an empty object (contract M6): a missing file is an
        # unresolvable 404 diff that keeps a genuinely-empty range "differing" forever.
        s3.put("%s%s/%05d.ndjson.enc" % (_MESH_LEAF_PREFIX, node, li),
               f.encrypt(("\n".join(lines)).encode("utf-8")), "application/octet-stream")
        uploaded += 1
        pub[li] = live[li]      # only now, backed by an object, is this leaf published
        if max_seconds and time.time() - t0 >= max_seconds:
            truncated = True
            break
    if truncated or incomplete:
        # A summary whose leaves are only partly up is not published. Live (the truth about local
        # state) is recorded and published advances exactly as far as the uploads got; the previous
        # consistent summary stays in place and peers keep converging on the last good root.
        _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT, "state": "committed",
                        _MERKLE_LIVE: live, _MERKLE_PUBLISHED: pub, "partial_upload": True})
        return {"published": 0,
                "reason": "upload-truncated" if truncated else "leaf-enumeration-truncated",
                "changed": len(changed), "uploaded": uploaded, "node": node}
    # 4) Summary last (M6); then, and only here, published is equated to live.
    s3.put("%s%s.json" % (_MESH_MERKLE_PREFIX, node),
           json.dumps(merkle.summary(live)).encode("utf-8"), "application/json")
    _put_op(store, {"id": _S3_MERKLE_CURSOR, "content_type": _S3SYNC_CT, "state": "committed",
                    _MERKLE_LIVE: live, _MERKLE_PUBLISHED: list(live), "root": merkle.root(live)})
    return {"leaves": n, "changed": len(changed), "uploaded": uploaded,
            "root": merkle.root(live), "secs": round(time.time() - t0, 1),
            "basis": "incremental", "node": node}


def _replicated_count(store) -> int:
    """How many rows the mesh is supposed to replicate — `count(*)` minus operational state.

    Every coverage/completeness ratio must use this as its denominator. `_OP_EXCLUDE` rows are
    per-box by definition and are deliberately never stamped with `_leaf`, so any check that
    compares a `_leaf`-filtered numerator to a raw `count(*)` under-reports and will refuse to
    publish (or report <100% coverage) forever, even on a healthy node.

    Computed by subtraction rather than negation: a filter like
    `WHERE content_type IS NULL OR content_type NOT IN [...]` stacks three separate index
    disqualifiers and is a guaranteed full scan, and in SQL's three-valued logic
    `NULL NOT IN [...]` evaluates to NULL — not matched — so a row with no `content_type` would
    drop out of that denominator while still being stamped with `_leaf` and counted in the
    numerator, producing a coverage figure above 100%. Subtracting each excluded type as an
    equality on the indexed `content_type` gives the identical answer from indexed queries only
    (an unfiltered `count(*)` is O(1)-ish, ~0-40ms), and a row with a NULL `content_type` matches
    none of the equality counts, so it stays correctly in the total.

    On lattice both terms are counter lookups and no `count(*)` is issued at all: `count` and
    `count_by_content_type` read incrementally maintained counter rows, so this is
    O(|_OP_EXCLUDE|) keyed reads instead of a scan — an unfiltered `count(*)` loads every row to
    produce one integer, which can OOM a process on a large corpus."""
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
    NULL `_leaf` is in no leaf, so it is invisible to refresh and to any tree built from it — the
    tree would look healthy while silently omitting data. Report it rather than assume it.

    The denominator is replicated rows only (`_replicated_count`). Operational rows are intentionally
    left with NULL `_leaf`, so counting them as "missing" would peg coverage permanently below 100%
    and make the very signal this function exists to give — "is Merkle trustworthy yet?" — read
    as broken on a node where nothing is wrong."""
    # This stays a full scan rather than an indexed query, for three independent reasons:
    #
    #  1. No index can contain the answer. `Artifact(_leaf)` is an LSM index with nullStrategy SKIP,
    #     so NULL `_leaf` rows are not in the index. `IS NOT NULL` therefore cannot be index-served,
    #     and neither can its complement — the rows this function exists to count (the ones missing
    #     `_leaf`) are exactly the rows no index has an entry for.
    #  2. An index would not help even if one existed. `_leaf` is stamped on every replicated row,
    #     so the predicate matches ~100% of the table. A range that spans the whole index reads one
    #     entry per row — the same O(N) as the scan, with an extra indirection.
    #  3. The obvious index-served rewrite is unsafe on this engine. `WHERE _leaf > -1` looks
    #     equivalent to the intended predicate, but is not safe to substitute: `_rev > -1` is
    #     rejected by the query engine as an infinite-loop pattern (see `_scan_rows`), and `>=`
    #     over a duplicated key on a NOTUNIQUE LSM index returns one row per key group (6 rows
    #     sharing R -> `count(*) WHERE _rev >= R` = 1). `_leaf` has only 4096 distinct values over
    #     millions of rows — the most duplicated key on the type — so a coverage number computed
    #     that way could silently read 4096 instead of 6,000,000 and make `publish_merkle` refuse
    #     forever on a healthy node.
    #
    # What makes the scan affordable is that it is cold: the only caller is
    # scripts/backfill_leaf.py (an operator-run backfill), never the sync cycle. The denominator
    # beside it is fully indexed (`_replicated_count` = one unfiltered count plus one equality count
    # per excluded type, ~0-40ms each), so this call costs one scan.
    #
    # On lattice the gap this function measures cannot exist. `_leaf` is computed and stamped inside
    # `vertex._write_row` on every write, so there is no "written before `_leaf` existed" population
    # to backfill and `scripts/backfill_leaf.py` has nothing to do there. Reporting the basis (rather
    # than just `pct: 100.0`) lets the caller tell "verified complete" from "complete by
    # construction".
    if _vertices(store) is not None:
        tot = _replicated_count(store)
        return {"total": tot, "with_leaf": tot, "missing": 0, "pct": 100.0,
                "basis": "lattice: _leaf stamped at write; no unstamped population can exist",
                "note": "operational rows are stamped here; see _refresh_leaves_lattice"}
    try:
        tot = _replicated_count(store)
        have = int((store.artifacts.c.query(
            "SELECT count(*) AS c FROM Artifact WHERE _leaf IS NOT NULL") or [{}])[0].get("c", 0))
    except Exception as e:
        return {"error": str(e)[:120]}
    return {"total": tot, "with_leaf": have, "missing": max(0, tot - have),
            "pct": round(100.0 * have / tot, 2) if tot else 0.0}


def reconcile_merkle(store, *, max_leaves: int = 64, max_seconds: float = 0.0) -> Dict[str, Any]:
    """Pull only the leaves that differ from each peer. This is the O(diff) replacement for reading
    every peer's whole log.

    Steady state costs one 32 KB tree fetch per peer and nothing else — if the roots match we are
    provably identical and stop. That flat steady-state cost is the entire reason this scales to
    500 nodes where the log feed cannot.

    Uses the local tree cached by publish_merkle rather than rescanning: the scan is expensive
    enough on a large store that rescanning per peer would make reconciliation cost O(peers x
    corpus) — the very thing being fixed. If no local tree has been published yet we do nothing
    rather than guess, because comparing against an empty tree would claim every leaf differs and
    pull the entire corpus from every peer at once.

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
    # Lists keys, not prefixes. `mesh/merkle/<node>.json` is a flat object — there is no `/` after
    # the prefix, so a delimited list returns it under Contents and CommonPrefixes is always empty.
    # A prefix-based listing would enumerate zero peers on every node while still returning
    # `{"applied": 0}`, indistinguishable from a healthy converged mesh.
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
        # `diff` compares at the common resolution t = min(len(local), len(theirs)) — a node never
        # re-shards just to talk to a peer; it compares at the coarser of the two. A differing coarse
        # index `li` maps to the peer's native leaves {li, li+t, li+2t, …} (the peer published its
        # objects at its own resolution). Fetch each and apply the mixed vertex+edge object.
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
                    # Mixed object: each applier takes its own lines (id vs f/t).
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
    # The local tree is re-hashed before returning. Without this the local tree still describes the
    # store as it was before the leaves were applied, so the next round would diff identically
    # against the same peer, re-download the same leaves, apply the same rows, and update nothing
    # again — a livelock that burns S3 bandwidth and never converges. The whole local tree is
    # refreshed (not just the coarse indices pulled): applied rows land in local leaves at the local
    # resolution, which need not equal the coarse indices fetched, and `refresh_leaves` recomputes
    # from the incremental `merkle_leaves` in one pass anyway.
    refreshed = (refresh_leaves(store, range(len(local)), leaves=len(local))["refreshed"]
                 if fetched else 0)
    out_r = {"applied": applied, "peers": peers, "leaves_fetched": fetched, "refreshed": refreshed,
             "secs": round(time.time() - t0, 1), "node": me}
    # A leaf that keeps differing round after round with `applied` making no progress is the
    # signature of unordered concurrent authorship, not of a stuck transfer. Report the two side by
    # side so they cannot be mistaken for each other.
    if _vertices(store) is not None:
        out_r["unordered"] = unordered_report(store)
        out_r["new"] = stats["new"]
        out_r["reapplied"] = stats["reapplied"]
    return out_r


def reconcile_via_s3(store, *, max_leaves: int = 256,
                     max_seconds: float = 0.0) -> Dict[str, Any]:
    """One anti-entropy round over the Merkle plane — the sync path. Publish this node's tree
    incrementally, then pull only the leaves that differ from each peer (vertices and edges).

    This is the whole of sync: there is no segment feed beside it. Bulk catch-up and steady state are
    the same operation — a fresh node finds every leaf differing and pulls them all; a converged node
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
    """Reach — pull a missed index row from the authoritative substrate and cache it locally.

    This is what lets an ember be limited: it need not hold the whole graph. When a lookup misses (an
    id the node does not hold), reach finds the leaf that would contain it in a publisher's Merkle tree
    in S3, pulls that one leaf (the existing content-addressed transfer unit), and applies it — so the
    id, and its leaf-neighbours, are now held. Same machinery as `reconcile_merkle`, but targeted at
    one need instead of converging the whole tree. Bodies still fetch on-miss from CAS; this is the
    index half of the same idea.

    A miss is a need, not an error (offers/needs): if no publisher holds it, reach returns
    `reached=False` and the caller answers honestly that the id is unavailable, rather than
    fabricating a value. Eviction/demurrage of what is reached is the demand cache's job; reach only
    fetches, and what it applies is held like any consumed row until the demand cache bounds it.

    The publisher's leaf resolution is read from its own published summary (`natural_leaves` differs per
    node), so `leaf_of(id, len(their_tree))` is the leaf on that node — no shared constant needed."""
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
    # Which peers to try, measured-best first. Peers this ember knows (peer-artifacts, each carrying a
    # peer's node id + CAS-addressed manifest) come first, ranked by the demand mass on each — attention
    # flows to the peer that has actually been useful. Then any publisher not yet known as a peer, for
    # discovery. Nothing pre-judged and no hardcoded peer list: the order is what use has measured.
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
        li = merkle.leaf_of(str(missing_id), len(theirs))     # the leaf on that peer's tree
        try:
            raw = f.decrypt(s3.get("%s%s/%05d.ndjson.enc" % (_MESH_LEAF_PREFIX, node, li)))
            items = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
        except Exception:
            continue                                          # this peer's leaf is absent/bad
        tried += 1
        got = _apply_artifacts(store, items) + _apply_edges(store, items)
        if v is not None and v.get_artifact(str(missing_id)) is not None:
            # The whole leaf is now cached. Mark every fetched row as demand (evictable) so nothing a
            # node did not author is silently pinned; the requested id is the hot one (touched twice).
            for it in items:
                iid = it.get("id")
                if iid and _is_replicated(it.get("content_type")):
                    _demand_touch(store, iid)
            _demand_touch(store, missing_id)      # the specifically-requested id — hotter
            _demand_touch(store, node)            # the peer was useful — attention accretes on its
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
    """The one-call front door for a limited ember: return the artifact if held, else reach for it and
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


# ── peers are artifacts too — each carrying a peer's CAS address ───────────────────────────────────
# An observer is not special. It is an artifact whose content is a CAS-addressed manifest of its
# measured state (its Merkle root = a fingerprint of what it holds, its leaf count, its measured
# envelope). An ember publishes itself as one, and receives every other ember's through the mesh like
# any artifact — so it holds the CAS address of each peer. There is no separate directory file and no
# declared peer list: peers are discovered as artifacts and attended to by measured demand.
_OBSERVER_CT = "application/vnd.agience.observer+json"


def _envelope_bytes(store) -> Optional[int]:
    """This node's measured envelope (data-volume free / cgroup mem), or None if unmeasurable.
    Part of the manifest so a peer can see what this ember can carry — measured, never declared."""
    try:
        from prism import envelope as resource
        kd = str(getattr(store, "keys_dir", "") or ".")
        free = resource.disk_free_bytes(kd)
        if free is not None:
            return int(free)
        mem = resource.mem_limit_bytes()
        return int(mem) if mem is not None else None
    except Exception:
        return None


def publish_manifest(store) -> Dict[str, Any]:
    """Publish this ember as a peer-artifact with a CAS-addressed manifest — because a peer is an
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
    # Omitted when unmeasurable: an absent key is an absence, while `envelope: 0` would tell every
    # peer this node can carry nothing. A peer must be able to tell "unknown" from "measured zero".
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
    address (`content_ref`) and node id. This is the observer directory: not a special file, just
    artifacts, discovered through the mesh and attended to by measured demand (§ _reach_candidates)."""
    v = _vertices(store)
    if v is None:
        return []
    me = _node_id()
    docs, _ = v.list_by_content_type(_OBSERVER_CT, cap=100000)
    return [d for d in docs if d.get("node") and d.get("node") != me]


def _reach_candidates(store) -> List[str]:
    """Peer nodes to try for a miss, measured-best first: the peers this ember knows (peer-artifacts),
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


# Keyset cursor values (ids) are never stripped or escaped by string manipulation — they are always
# passed as bind parameters. Stripping the apostrophe from a value like `foo's` to avoid breaking a
# literal is unsafe: `'` (0x27) sorts before `s` (0x73), so the stripped `foos` sorts after `foo's`,
# and a keyset cursor built from it would jump forward past every id between the two, silently
# skipping those artifacts from `digest`/`bucket_ids`. Ids come from ingested content (wiki titles
# etc.), so apostrophes are common, and a skip of this kind is invisible: a short scan looks exactly
# like a small corpus. Left un-stripped, the same literal terminates the string and raises on commit.
# Parameterization avoids both failure modes, the same as `publish_to_s3`'s `id >:cur` page above
# and as `arcade.py` throughout.




