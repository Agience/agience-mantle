"""The Ember mesh daemon — a box manages its own peering: sync + role/idle policy.

This replaces every hand-driven thing (Claude SSHing in, spawning tunnels, toggling ingest/compress).
Ember reads its role and runs itself. Claude never has a back door: it authors this, deploys it once,
and observes only via the reported stats.

Transport is S3: S3 is the authoritative store and the exchange plane. There is no
box-to-box connection at all — no SSH tunnels, no overlay, no VPS. Sync is one path: Merkle
anti-entropy. Each box publishes its Merkle tree (per-node leaf objects) to S3 and pulls only the
leaves that differ from each peer — vertices and edges. Because every box reaches S3, NAT is
irrelevant; because a leaf is the authoritative state of its key-range, any box converges to full just
by pulling differing leaves, and a wiped/new box rebuilds by the same operation. Content is already
S3-authoritative (cas/), so a box that has the graph can read any content on demand.

The daemon does two jobs on a loop:

  1. SYNC — op.mesh.reconcile: publish my tree incrementally + pull only the leaves that differ.
     Memory-bounded, idempotent, order-independent. Consumers converge to the whole graph; shard
     boxes reconcile publish-only (max_leaves=0).
  2. ROLE / IDLE POLICY — enforce the topology so no paid box is ever idle:
       role=ingest              -> always ingest + promote content->S3.
       role=full, purpose=tests -> sync only (71): no ingest.
       role=full, purpose=compress (t5) -> ingest UNTIL converged-to-full, then run compression;
                                           ingest again if it falls behind. Never idle.
     Invariant (checked here, not by a human): at least one ingest runpod + the ingest laptop are
     always ingesting — guaranteed because ingest-role boxes never stop.

Config (env, so it's data not code; adaptive sizing comes from resource.py):
  EMBER_ROLE     = "ingest" | "full"
  EMBER_PURPOSE  = "tests" | "compress" | ""        (for full boxes)
"""
from __future__ import annotations

import json
import os
import sys
import time

from prism import envelope as resource


def invoke_loud(store, operator_id: str, args: dict, rec: dict) -> dict:
    """Invoke an operator without swallowing an error envelope.

    The failure envelope is `{"error": ...}` with no `"operator"` key — the success envelope always
    carries `"operator"`. Do not sniff nested results (a composition step may legitimately carry its
    own error), which is the discrimination `pool.run_task` already settled on."""
    from mantle.system import runner_hooks   # the store ASKS the runner; it does not import one
    res = runner_hooks.invoke(store, operator_id, args)
    if isinstance(res, dict) and res.get("error") and not res.get("operator"):
        detail = "%s -> %s" % (operator_id, str(res["error"])[:200])
        rec.setdefault("invoke_errors", []).append(detail)
        print("[mantle.mesh.daemon] INVOKE FAILED: %s" % detail, file=sys.stderr, flush=True)
    return res if isinstance(res, dict) else {}


def _converged_to_full(store) -> bool:
    """A compress box is 'full' when a Merkle round pulls nothing new and every peer's tree matches —
    i.e. it is provably identical to the fleet, not merely that one probe read nothing.
    roots actually matched — the same discipline `mesh_lag` uses (`converged = None` when blind)."""
    try:
        from . import sync as _sync
        r = _sync.reconcile_via_s3(store)             # publish my tree + pull any differing leaves
        if r.get("reason") == "no-s3":
            return False
        peers = r.get("peers") or {}
        if not peers:
            return False                              # saw no peers -> cannot claim converged
        all_matched = all(p == 0 for p in peers.values())   # 0 == "converged with this peer"
        return int(r.get("leaves_fetched", 0) or 0) == 0 and all_matched
    except Exception:
        return False


def run(store, *, interval: float = 30.0):
    """The mesh daemon loop — S3 sync + role/idle policy. One process per box.

    Every operator call goes through `invoke_loud`, never `genesis.invoke` directly: an
    `{"error": ...}` envelope is a failure and this loop must not print a clean record over one."""
    role = os.getenv("EMBER_ROLE", "ingest")
    purpose = os.getenv("EMBER_PURPOSE", "")
    print(json.dumps({"mesh_daemon": "up", "transport": "s3", "sync": "merkle", "role": role,
                      "purpose": purpose,
                      "limits": resource.snapshot(
                          str(store.keys_dir) if store.keys_dir else ".")}), flush=True)
    while True:
        rec = {"mesh": True, "ts": time.time(), "role": role}
        # 1. SYNC — THE ONE PATH: Merkle anti-entropy over S3 (op.mesh.reconcile). Publish my tree
        #    incrementally (only changed leaves) and pull ONLY the leaves that differ from each peer —
        #    vertices AND edges. Bulk catch-up and steady state are the same operation: a fresh node
        #    pulls every differing leaf; a converged node exchanges one 32 KB tree and stops. There is
        #    no second mechanism, no segment feed, no cursor. See S3-SUBSTRATE-RESILIENCY.md.
        try:
            # Publish THIS ember as a peer-artifact (its measured state, CAS-addressed) so peers
            # discover it as an artifact — before reconcile, so it rides this cycle's tree.
            invoke_loud(store, "op.mesh.manifest", {}, rec)
            r = invoke_loud(store, "op.mesh.reconcile",
                            {"max_leaves": 256, "max_seconds": max(5.0, interval)}, rec)
            res = r.get("result", {}) or {}
            rec["applied"] = res.get("applied", 0)
            rec["published_leaves"] = res.get("published_leaves", 0)
            rec["leaves_fetched"] = res.get("leaves_fetched", 0)
            rec["peers"] = res.get("peers", {})
        except Exception as e:
            rec["sync_err"] = str(e)[:120]
        # 2. ROLE / IDLE POLICY.
        #
        #
        # `op.curriculum.advance` is what it meant and is the operator that does the job: "advances
        # the developmental curriculum by one bounded increment (ingest the next stage's records, or
        # promote it) — Ember driving its own ingestion" (its own registered description). It is
        # bounded per call, which is what a 30-second loop needs — a pool that "ensures ingest" by
        # spawning work would have to be idempotent against the previous tick, and the curriculum
        # operator already is (it advances one increment or reports the stage complete).
        try:
            if role == "ingest":
                invoke_loud(store, "op.curriculum.advance", {}, rec)      # keep ingesting (never idle)
            elif purpose == "compress":
                if _converged_to_full(store):
                    rec["policy"] = "compress"
                    invoke_loud(store, "op.consolidate.crosswalk",
                                {"apply": True, "bounded": True}, rec)
                else:
                    rec["policy"] = "ingest-until-full"
                    invoke_loud(store, "op.curriculum.advance", {}, rec)  # never idle while catching up
            # purpose == tests (71): sync only, no ingest/compress.
        except Exception as e:
            rec["policy_err"] = str(e)[:100]
        print(json.dumps(rec), flush=True)
        time.sleep(interval)
