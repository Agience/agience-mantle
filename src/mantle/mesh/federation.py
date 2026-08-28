"""The mesh: two (or more) Ember shards, each a full local Ember/Mantle install with its own
ArcadeDB+Garage, holding a disjoint half of the corpus (partitioned by EMBER_SHARDS). Writes stay
local on each box — that is the write-scaling win. This module federates the read side: it fans out
to peers over their HTTP /health endpoint and merges the numbers so /status shows one universe.

Federation is metric-level and content-blind here (counts, ρ, curriculum) — no artifact bytes cross
the wire, so it is cheap and leaks nothing. Deeper federation (op.mesh.search across shards) builds
on the same peer list.

Peers are configured out-of-band, never hardcoded:
    EMBER_PEERS = "http://<peer-host>:8091,http://…"   # comma-separated peer base URLs
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict, List

# A peer's /health runs a metrics scan (several seconds on a large shard), so give it room and
# cache the aggregate briefly — /status polling must not re-hit the peer on every request.
_PEER_TIMEOUT = 20.0
_CACHE: Dict[str, Any] = {"ts": 0.0, "val": None}
_CACHE_TTL = 30.0


def peers() -> List[str]:
    raw = os.getenv("EMBER_PEERS", "") or ""
    return [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]


def _get(url: str, *, timeout: float = _PEER_TIMEOUT) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def peer_stats(base: str, *, timeout: float = 6.0) -> Dict[str, Any]:
    """A peer's snapshot (its stats.json via /stats) — instant, no scan on the peer side. This is
    what the dashboard aggregates: N tiny snapshot reads, never N full-corpus scans."""
    try:
        d = _get(base + "/stats", timeout=timeout)
        d["peer"] = base
        d["reachable"] = "error" not in d
        return d
    except Exception as e:
        return {"peer": base, "reachable": False, "error": str(e)[:120]}


def peer_health(base: str, *, timeout: float = _PEER_TIMEOUT) -> Dict[str, Any]:
    """One peer's headline numbers (from its /health), tagged with the peer's address. Never raises
    — an unreachable peer returns {reachable: False} so the mesh view degrades, not crashes."""
    try:
        h = _get(base + "/health", timeout=timeout)
        w = h.get("worker") or {}
        return {"peer": base, "reachable": True, "status": h.get("status"),
                "artifacts": h.get("artifacts"), "rho": h.get("rho"),
                "worker_alive": w.get("alive"), "worker_age_s": w.get("age_s"),
                "recent_errors": w.get("recent_errors")}
    except Exception as e:
        return {"peer": base, "reachable": False, "error": str(e)[:120]}


def mesh_status(local_artifacts: int, *, local_rho: float | None = None) -> Dict[str, Any]:
    """Aggregate this shard + every configured peer into one universe view. `local_artifacts` is
    passed in (the caller already has the cheap count) so this stays a pure network fan-out."""
    now = time.time()
    if _CACHE["val"] is not None and (now - _CACHE["ts"]) < _CACHE_TTL:
        ps = _CACHE["val"]
    else:
        ps = [peer_health(b) for b in peers()]
        _CACHE["val"] = ps
        _CACHE["ts"] = now
    reachable = [p for p in ps if p.get("reachable")]
    peer_total = sum(int(p.get("artifacts") or 0) for p in reachable)
    return {
        "shards": 1 + len(ps),
        "peers": ps,
        "peers_reachable": len(reachable),
        "local_artifacts": int(local_artifacts or 0),
        "peer_artifacts": peer_total,
        "total_artifacts": int(local_artifacts or 0) + peer_total,
        "local_rho": local_rho,
    }


# ── authoritative replication: D: (.71) pulls every peer's artifacts into itself ─────────────────
# The mesh's remote shards ingest in parallel (fast local writes, CPU-heavy describe offloaded), but
# the owner wants one authoritative store (D:) that contains everything. op.mesh.pull replicates each
# peer's artifacts into the local store. content_ref = cas/sha256(plaintext) is key-independent, so
# re-encrypting a pulled artifact's plaintext under D:'s own key yields the same ref — the doc stays
# valid. Edges are not shipped: collection/lemma edges re-derive from the doc on upsert, and semantic
# edges (crosswalk/consolidation) are re-run locally as operators. Cursor-checkpointed + idempotent.
import base64  # noqa: E402


def export_page(store, offset: int = 0, limit: int = 25, *, after: str = "",
                with_content: bool = False) -> Dict[str, Any]:
    """Peer side: return a page of artifacts, in storage order. By default docs only (metadata +
    lemmas + context + 300-char preview + content-addressed ref) — fast, and enough for the
    authoritative store to hold the complete knowledge graph + keyed lookup. with_content=True
    also ships full decrypted plaintext (slow; for the blob back-fill).

    Pages by keyset when the store exposes `page_by_id` — a lattice store's own sanctioned
    pagination primitive, immune to the index inconsistency below and O(limit) rather than
    O(offset) per page (measured: 743ms against 142,136ms for an equivalent `SKIP` page at
    depth 5M). Falls back to `offset`/`limit` SKIP/LIMIT SQL against the legacy ArcadeDB store,
    where a peer under heavy concurrent ingest can have a transiently-inconsistent unique id
    index (a duplicate-key race at cold start), and an index range scan then NPEs — a sequential
    SKIP/LIMIT scan sidesteps the index entirely. Any dup rows a peer holds collapse on arrival
    (the authoritative store's own id index is healthy).

    Both cursors are always returned: `next_offset` stays monotonic for a peer still on the
    legacy SKIP contract, and `next_after` carries the keyset cursor for one backed by a lattice
    store. `pull_from_peers` prefers `next_after` when present, so neither side is stranded
    mid-sweep by the other's backend.

      * The SKIP fallback is only reachable when EMBER_PEERS is set (pull_from_peers iterates
        peers(); pool.py:316 gates the whole mesh-pull job on the same var). Peers are configured
        out-of-band, never hardcoded, so whether this path is live depends on deployment config —
        the S3 mesh plane (mesh/sync.py, per-node encrypted segment logs) is the other route
        between shards and does not depend on EMBER_PEERS.
      * `offset` / `next_offset` are the wire contract with `pull_from_peers`' persisted cursor
        artifact (`page_offset`, an int). A peer still on the legacy path depends on it staying
        monotonic regardless of which backend serves the page.
    """
    # `SKIP` paging is banned by contract §0.6; the keyset path above is what satisfies it for a
    # lattice store. The dual `next_offset`/`next_after` return is what keeps a legacy peer on the
    # SKIP contract working unchanged while a lattice-backed peer never pays for it.
    offset = max(0, int(offset))          # already cast before interpolation — never raw
    after = str(after or "")
    v = getattr(store, "artifacts", None)
    if hasattr(v, "page_by_id"):
        if offset > 0 and not after:
            raise RuntimeError(
                "op.mesh.export: this store pages by KEYSET but the request carried offset=%d with "
                "no `after` cursor, so the dispatcher dropped it. Add `after=args.get(\"after\",\"\")` "
                "to the op.mesh.export handler in genesis.py. Refusing to serve page 1 as though it "
                "were page %d." % (offset, offset // max(1, int(limit))))
        rows = [r["doc"] for r in v.page_by_id(after=after, limit=int(limit))]
    else:
        rows = store.artifacts.c.query(f"SELECT FROM Artifact SKIP {offset} LIMIT {int(limit)}")
    # Operational state (the work-queue, mesh cursors) is per-box, not shared knowledge — never
    # replicate it (it would pollute the authoritative store's queue).
    #
    #
    # There is one definition of "does this replicate", in the module that owns the policy: a
    # second copy of a security-relevant predicate is a copy that will drift.
    from .sync import _is_replicated
    out = []
    scanned = 0
    last_id = after
    for r in rows:
        scanned += 1
        doc = {k: v for k, v in r.items() if not str(k).startswith("@")}
        aid = doc.get("id")
        if aid and str(aid) > last_id:
            last_id = str(aid)
        if not aid or not _is_replicated(doc.get("content_type")):
            continue
        item: Dict[str, Any] = {"doc": doc}
        ref = doc.get("content_ref")
        if with_content and ref and store.content is not None and store.keys_dir is not None:
            try:
                from mantle.shard import content as C
                item["content_b64"] = base64.b64encode(
                    C.get_content(store.content, store.keys_dir, ref)).decode("ascii")
            except Exception:
                pass
        out.append(item)
    # next_offset advances by rows scanned (incl. excluded); exhausted when a short page comes back.
    # `next_after` is the keyset cursor — the value a caller should resume on. Both are returned so
    # an old caller keeps working unchanged and a new one never pays the SKIP cost.
    return {"artifacts": out, "next_offset": offset + scanned, "next_after": last_id,
            "scanned": scanned, "exhausted": scanned < int(limit), "count": len(out)}


def _post(base: str, operator: str, arguments: Dict[str, Any], *, timeout: float = 60.0) -> Dict[str, Any]:
    body = json.dumps({"operator": operator, "arguments": arguments}).encode()
    req = urllib.request.Request(base + "/v1/invoke", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _cursor_id(base: str) -> str:
    import hashlib
    return "mesh.cursor." + hashlib.sha256(base.encode()).hexdigest()[:12]


def pull_from_peers(store, *, max_pages: int = 40, page: int = 25, with_content: bool = False) -> Dict[str, Any]:
    """Local (authoritative) side: for each peer, page through its artifacts and upsert them here.
    Docs only by default (fast — the complete graph + keyed index + previews + content-addressed
    refs); with_content=True also re-stores full plaintext under this store's key (the slow bulk
    back-fill). Resumable via a per-peer cursor artifact; bounded pages per call so it never runs
    long when scheduled."""
    from mantle.shard import content as C
    from prism import grounding as genesis   # contract, not the runner: only the provenance
    # rung + citation id + timestamp are used here.
    # mantle may reach prism; it may not reach ember.
    summary = []
    for base in peers():
        cid = _cursor_id(base)
        cur = store.artifacts.get_artifact(cid) or {}
        offset = int(cur.get("page_offset", 0) or 0)   # NOT 'offset' — reserved word in ArcadeDB SQL
        after = str(cur.get("page_after", "") or "")   # the keyset cursor; "" = start of the id space
        pulled = 0
        pages = 0
        err = None
        exhausted = False
        while pages < max_pages:
            data = None
            for _try in range(3):                 # a transient peer 500 shouldn't abandon the peer
                try:
                    # Send both cursors. A peer on the old code ignores `after` and pages on
                    # `offset`; a peer on the new code pages on `after` and returns `next_after`.
                    res = _post(base, "op.mesh.export",
                                {"offset": offset, "after": after, "limit": page,
                                 "with_content": with_content})
                    data = res.get("result") or res
                    if isinstance(data, dict) and "error" in data and "artifacts" not in data:
                        err = str(data.get("error"))[:120]
                        data = None
                        time.sleep(0.3 * (_try + 1))
                        continue
                    err = None
                    break
                except Exception as e:
                    err = str(e)[:120]
                    time.sleep(0.3 * (_try + 1))
            if data is None:                      # still failing after retries → move on this call
                break
            docs = []
            for item in (data.get("artifacts") or []):
                doc = item.get("doc") or {}
                if not doc.get("id"):
                    continue
                b64 = item.get("content_b64")
                if b64 and store.content is not None and store.keys_dir is not None:
                    # There are two content-store shapes with incompatible `put` contracts:
                    #
                    #   FsContentStore.put(ref, ciphertext)                     — the caller encrypts
                    #   TieredContentStore.put(ref, plaintext, *, collection)   — the store encrypts
                    #
                    # `shard.content.put_content` handles both: it takes a `collection` and passes
                    # plaintext to the stores that scope, so this call works against a real node.
                    # Federated content goes into the artifact's own collection — the doc already
                    # carries it, and it is the only answer that keeps a pulled body reachable by
                    # exactly the grants its metadata says it should be.
                    #
                    # A doc with no `collection_id` is skipped, not defaulted: `put_content` refuses
                    # a scoping store with no collection rather than inventing one, and inventing one
                    # here would be the same authorization-by-accident a layer up. The artifact still
                    # imports; its body does not, which is the honest half-state.
                    #
                    # And the `pass` is still not a `raise`: a peer with no content tier at all is a
                    # legal node shape (`content_handle` returns None for one), so a failure to store
                    # a body must not fail the whole pull.
                    #
                    # `pull_from_peers` has no callers in `src/` today, which is the only reason a
                    # regression here would surface as a note rather than an incident — it stops
                    # being latent the moment federation is wired, which is now one config away since
                    # the mesh daemon became startable (`build/mantle/docker-compose.yml`, profile
                    # `mesh`). The same class of defect sits at `oci/store.py::_put`, where it is at
                    # least loud; `agience-cloud/build/mantle/oci_ingest.py` refuses up front and
                    # names it.
                    _coll = doc.get("collection_id") or ""
                    if _coll:
                        try:
                            C.put_content(store.content, store.keys_dir,
                                          base64.b64decode(b64), collection=_coll)
                        except Exception:
                            pass
                docs.append(doc)
            if docs:
                store.artifacts.put_many(docs, batch=100)
                pulled += len(docs)
            offset = int(data.get("next_offset", offset))
            # Prefer the keyset when the peer offers one. Absent `next_after` the peer is running
            # the old code and `offset` is all there is; present, it is authoritative and the O(offset)
            # SKIP never runs. A peer that returns `next_after == after` has made no forward progress
            # — stop rather than re-request the identical page forever.
            nxt = data.get("next_after")
            if isinstance(nxt, str) and nxt:
                if nxt == after:
                    exhausted = True
                    after = ""
                    break
                after = nxt
            pages += 1
            if data.get("exhausted"):
                exhausted = True
                offset = 0                        # next sweep restarts to catch newly-ingested rows
                after = ""
                break
        # Operational row -> `_put_op`, never a stamped write. A per-peer pull cursor is per-box
        # state; stamping it would allocate proper time on every sweep and keep this node's publish
        # feed permanently non-idle. Same rule as every cursor in mesh/sync.py.
        from .sync import _put_op
        _put_op(store, {
            "id": cid, "content_type": "application/vnd.agience.mesh-cursor+json",
            "state": "committed", "peer": base, "page_offset": offset, "page_after": after,
            "pulled_total": (cur.get("pulled_total", 0) + pulled),
            "content": f"mesh pull cursor for {base}", "provenance": genesis.P_HUMAN,
            "cited_from": genesis.CITE_GENESIS, "updated": _now()})
        entry = {"peer": base, "pulled_this_call": pulled, "page_offset": offset,
                 "page_after": after, "exhausted": exhausted}
        if err:
            entry["error"] = err
        summary.append(entry)
    return {"peers": summary}


def _now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
