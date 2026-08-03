"""The mesh: two (or more) Ember shards, each a full local Ember/Mantle install with its OWN
ArcadeDB+Garage, holding a DISJOINT half of the corpus (partitioned by EMBER_SHARDS). Writes stay
local on each box — that is the write-scaling win. This module federates the READ side: it fans out
to peers over their HTTP /health endpoint and merges the numbers so /status shows ONE universe.

Federation is metric-level and content-blind here (counts, ρ, curriculum) — no artifact bytes cross
the wire, so it is cheap and leaks nothing. Deeper federation (op.mesh.search across shards) builds
on the same peer list.

Peers are configured out-of-band, never hardcoded:
    EMBER_PEERS = "http://192.168.4.45:8091,http://…"   # comma-separated peer base URLs
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, Dict, List

# A peer's /health runs a metrics scan (several seconds on a large shard), so give it room and
# CACHE the aggregate briefly — /status polling must not re-hit the peer on every request.
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
    """A peer's SNAPSHOT (its stats.json via /stats) — instant, no scan on the peer side. This is
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
    """Aggregate THIS shard + every configured peer into one universe view. `local_artifacts` is
    passed in (the caller already has the cheap COUNT) so this stays a pure network fan-out."""
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


# ── authoritative replication: D: (.71) pulls every peer's artifacts INTO itself ─────────────────
# The mesh's remote shards ingest in parallel (fast local writes, CPU-heavy describe offloaded), but
# the OWNER wants ONE authoritative store (D:) that contains everything. op.mesh.pull replicates each
# peer's artifacts into the local store. content_ref = cas/sha256(PLAINTEXT) is key-independent, so
# re-encrypting a pulled artifact's plaintext under D:'s own key yields the SAME ref — the doc stays
# valid. Edges are NOT shipped: collection/lemma edges re-derive from the doc on upsert, and semantic
# edges (crosswalk/consolidation) are re-run locally as operators. Cursor-checkpointed + idempotent.
import base64  # noqa: E402


def export_page(store, offset: int = 0, limit: int = 25, *, after: str = "",
                with_content: bool = False) -> Dict[str, Any]:
    """Peer side: return a page of artifacts at SKIP `offset` LIMIT `limit`, in storage order. By
    default DOCS ONLY (metadata + lemmas + context + 300-char preview + content-addressed ref) —
    fast, and enough for the authoritative store to hold the complete knowledge GRAPH + keyed
    lookup. with_content=True also ships full decrypted plaintext (slow; for the blob back-fill).

    OFFSET paging (not `WHERE id > cursor`) is deliberate: a peer under heavy concurrent ingest can
    have a transiently-inconsistent UNIQUE id index (a duplicate-key race at cold start), and an
    index RANGE scan then NPEs — a sequential SKIP/LIMIT scan sidesteps the index entirely. Any dup
    rows a peer holds collapse on arrival (the authoritative store's own id index is healthy).

    ⚠ COLD PATH — SUPERSEDED, and O(offset). `SKIP n` is measured at 142,136ms at depth 5M (the
    keyset form `WHERE id > :cur ORDER BY id LIMIT n` is FLAT, ~743ms at the same depth), so a full
    replication sweep of a multi-million-row peer costs time QUADRATIC in its size — at page=25 that
    is unusable at any real scale. It is left in the O(offset) form ON PURPOSE:

      * It is only reachable when EMBER_PEERS is set (pull_from_peers iterates peers(); pool.py:316
        gates the whole mesh-pull job on the same var). Nothing in the tree sets EMBER_PEERS — the
        S3 mesh plane (mesh/sync.py, per-node encrypted segment logs) replaced HTTP peer-pull — so
        this is believed dead. "Believed", not proven: peers are configured out-of-band by design,
        so a live deployment could still set it. Not deleted on that evidence.
      * `offset` / `next_offset` are the WIRE CONTRACT with pull_from_peers' persisted cursor
        artifact (`page_offset`, an int). Converting to a keyset changes that contract and would
        strand any peer running the other version mid-sweep.
      * The NPE-avoidance reason above is real and would be given up by moving onto the id index.

    If this path is ever revived at scale, migrate BOTH sides together to an id keyset and accept
    the index risk — do not tune the page size, the cost is in the SKIP."""
    # ── RETIREMENT RECOMMENDED, NOT TAKEN. See the unit report; deletion is Phase 8. ──────────────
    #
    # ON LATTICE THIS USES A KEYSET AND BOTH SIDES MIGRATE TOGETHER, which is exactly what the
    # docstring above says is required. `SKIP` is banned by contract §0.6 and measured at 142,136ms
    # at depth 5M against 743ms for the equivalent keyset page, so an O(offset) sweep of a
    # multi-million-row peer costs time QUADRATIC in its size. A lattice store also cannot NPE on an
    # id range — the reason the SKIP form was defensible on ArcadeDB — so the one argument for
    # keeping it does not survive the backend change.
    #
    # THE WIRE CONTRACT IS EXTENDED, NOT BROKEN. `pull_from_peers` persists `page_offset` as an int,
    # and a peer running the older code still returns only `next_offset`. So this returns BOTH:
    # `next_offset` stays monotonic for a legacy caller, and `next_after` carries the real keyset
    # cursor. `pull_from_peers` below prefers `next_after` WHEN PRESENT and falls back to
    # `next_offset` otherwise, so neither side is stranded mid-sweep in either direction.
    offset = max(0, int(offset))          # already cast before interpolation — never raw
    after = str(after or "")
    v = getattr(store, "artifacts", None)
    if hasattr(v, "page_by_id"):
        # ⛔ CROSS-UNIT DEPENDENCY, GUARDED LOUDLY RATHER THAN LEFT TO ROT.
        # `op.mesh.export`'s dispatcher (`genesis.py:1747`, Unit B — blocked, not mine to edit) does
        # NOT forward `after` yet. Without this guard a caller advancing `offset` would get page ONE
        # every time on a lattice store, since the keyset ignores `offset` entirely: a sweep that
        # re-pulls the first 25 rows forever while reporting healthy `pulled_this_call` counts. That
        # is the shim's exact failure signature — a plausible wrong answer — and it must not be
        # reintroduced through an argument nobody plumbed.
        if offset > 0 and not after:
            raise RuntimeError(
                "op.mesh.export: this store pages by KEYSET but the request carried offset=%d with "
                "no `after` cursor, so the dispatcher dropped it. Add `after=args.get(\"after\",\"\")` "
                "to the op.mesh.export handler in genesis.py. Refusing to serve page 1 as though it "
                "were page %d." % (offset, offset // max(1, int(limit))))
        rows = [r["doc"] for r in v.page_by_id(after=after, limit=int(limit))]
    else:
        rows = store.artifacts.c.query(f"SELECT FROM Artifact SKIP {offset} LIMIT {int(limit)}")
    # OPERATIONAL state (the work-queue, mesh cursors) is per-box, NOT shared knowledge — never
    # replicate it (it would pollute the authoritative store's queue).
    #
    # ⛔ THIS WAS A DRIFTED DUPLICATE OF `sync._OP_EXCLUDE` AND IT LET REAL ROWS THROUGH. The local
    # copy listed four types and had fallen two behind — it was missing
    # `application/vnd.agience.s3sync-cursor+json` (the S3 publish/consume cursors) and
    # `application/vnd.agience.mesh-declined+json` — and it carried NONE of the three prefix bans
    # (`application/x-probe*`, `application/vnd.agience.probe*`, `"throwaway" in ct`). So this path
    # would ship a peer's per-box publish cursors AND its probe fixtures into the authoritative
    # store. Probe fixtures are the specific thing that once pinned publish cursors beyond every
    # real row and MUTED FIVE NODES PERMANENTLY.
    #
    # There is now ONE definition of "does this replicate", in the module that owns the policy. A
    # second copy of a security-relevant predicate is a copy that will drift again.
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
    # next_offset advances by rows SCANNED (incl. excluded); exhausted when a short page comes back.
    # `next_after` is the keyset cursor — the value a caller SHOULD resume on. Both are returned so
    # an old caller keeps working unchanged and a new one never pays the SKIP.
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
    DOCS ONLY by default (fast — the complete graph + keyed index + previews + content-addressed
    refs); with_content=True also re-stores full plaintext under THIS store's key (the slow bulk
    back-fill). Resumable via a per-peer cursor artifact; bounded pages per call so it never runs
    long when scheduled."""
    from mantle.shard import content as C
    from prism import grounding as genesis   # ⚠ CONTRACT, NOT THE RUNNER: only the provenance
    # rung + citation id + timestamp are used here, and those moved to prism on 2026-07-31.
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
                    # Send BOTH cursors. A peer on the old code ignores `after` and pages on
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
                    try:
                        C.put_content(store.content, store.keys_dir, base64.b64decode(b64))
                    except Exception:
                        pass
                docs.append(doc)
            if docs:
                store.artifacts.put_many(docs, batch=100)
                pulled += len(docs)
            offset = int(data.get("next_offset", offset))
            # PREFER THE KEYSET WHEN THE PEER OFFERS ONE. Absent `next_after` the peer is running
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
