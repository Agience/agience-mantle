"""The demand cache — what a LIMITED ember holds is shaped by use, and stays within its envelope.

A reached row carries a MASS. Each reach accretes it (+1); time decays it — demurrage, the 2nd law,
through the ONE decay kernel in `prism.law` (`cool(dt, tau) = exp(-dt/tau)`), never a bare `math.exp`.
When the cache exceeds the node's MEASURED envelope, the coldest rows (lowest decayed mass) are
evicted; the authoritative copy still lives in the substrate, so a later reach restores it. This is
what makes a node STAY limited rather than merely start empty.

Own-authored rows carry no demand entry (they were authored, not reached) and are therefore never
selected for eviction — a node never sheds its own observations, only its cache of others'.

Decay lives HERE, not in the store: the mantle lattice is dependency-free, and `prism.law` is the ember
layer's to import. The store holds only the raw `(mass, ts)`.

⚠ IT STAYS IN MANTLE, AND THE KERNEL MOVED INSTEAD (2026-07-31). It was briefly moved to ember to
escape the layering rule (mantle may not import beam) — but `mesh/sync.py` calls `demand.touch` on
every reached row, so the module belongs with the store that holds the demand rows. What did not
belong in beam was the KERNEL: `law.cool` is a primitive, and it now lives in `prism.law`, which
both mantle and beam may reach.

That also deleted the `except ImportError -> math.exp(-dt/tau)` fallback — a second decay kernel
hiding in an except branch. See `_cool`.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict

from prism import law as _law       # the ONE decay kernel (2nd law) — no fallback, see _cool


def _now() -> float:
    return time.time()


def _tau() -> float:
    """The demand decay time (seconds) — a FLAGGED SEAM, default honest (~1 day). It SHOULD be
    measured from the access trajectory (`optics.fit_dynamics(...).rates()`) where one exists; until
    then it is a declared tunable, not a hidden constant. Larger tau = a longer memory."""
    return max(1e-9, float(os.getenv("EMBER_DEMAND_TAU", "86400")))


def _cool(dt: float, tau: float) -> float:
    """Decay to now through THE kernel. No fallback, deliberately.

    ⚠ THE `except: _law = None` FALLBACK WAS DELETED WITH THE MOVE (2026-07-31). It read
    `math.exp(-dt / tau)` with the comment "identical kernel; only if beam is unavailable" — a
    SECOND implementation of the decay law, hiding in an except branch, which the single-source rule
    exists to forbid ("optics MEASURES the scale, law APPLIES the kernel; no bare exp(-x/scale)").
    Identical today is not identical after the next change to `law.cool`, and the divergence would
    appear only on hosts where the import failed — the hardest possible place to notice it.

    It existed because this module lived in mantle, which may not import beam. Now it lives in
    ember, which may, so there is nothing to fall back FROM."""
    return float(_law.cool(max(0.0, float(dt)), tau=tau))


def touch(store, artifact_id: str) -> None:
    """A reach or access is DEMAND. Decay the item's mass to now, then add one. A held row with no
    demand entry (own-authored) simply gains one on its first *reach* and is otherwise untouched."""
    v = getattr(store, "artifacts", None)
    if v is None or not hasattr(v, "demand_set"):
        return
    # Demand tracks HELD artifacts only — one demand row per held row, never an orphan. Touching an id
    # the node does not hold (e.g. a peer discovered but whose peer-artifact is not held) is a no-op,
    # so the cache invariant stays: a demand entry ⇒ a held, evictable row.
    if v.get_artifact(str(artifact_id)) is None:
        return
    now, tau = _now(), _tau()
    d = v.demand_get(artifact_id)
    mass = (float(d["mass"]) * _cool(now - float(d["ts"]), tau) + 1.0) if d else 1.0
    v.demand_set(artifact_id, mass, now)


def current_mass(d: Dict[str, float], *, now: float, tau: float) -> float:
    """A demand row's mass decayed to `now` — the value the eviction ranking compares."""
    return float(d["mass"]) * _cool(now - float(d["ts"]), tau)


def budget_rows(store) -> int:
    """Max cache rows, DERIVED from the MEASURED envelope: `envelope_bytes · fraction / bytes_per_row`.
    The envelope (cgroup mem / data-volume free) and the bytes-per-row are measured; the fraction of
    the envelope to spend on the cache index, and the bytes-per-row estimate when it can't be
    measured, are FLAGGED SEAMS (default honest), per 'no arbitrary caps — leave a flagged seam'."""
    frac = max(0.0, float(os.getenv("EMBER_CACHE_FRACTION", "0.25")))
    bpr = max(1.0, float(os.getenv("EMBER_BYTES_PER_ROW", "2048")))     # ~an index row; measured seam
    env_bytes = 0
    try:
        from prism import envelope as resource
        kd = str(getattr(store, "keys_dir", "") or ".")
        env_bytes = int(resource.disk_free_bytes(kd) or 0) or int(resource.mem_limit_bytes() or 0)
    except Exception:
        env_bytes = 0
    if not env_bytes:               # cannot measure -> a flagged, bounded fallback (never unlimited)
        return int(os.getenv("EMBER_CACHE_ROWS", "100000"))
    return max(1000, int(env_bytes * frac / bpr))


def evict(store, *, budget: int) -> Dict[str, Any]:
    """Evict the COLDEST cache rows (lowest decayed mass) until the cache is within `budget`. Only
    demand rows (reached copies) are candidates, so own-authored rows are never shed. The
    authoritative row survives in the substrate; a re-reach restores it."""
    v = getattr(store, "artifacts", None)
    if v is None or not hasattr(v, "demand_count"):
        return {"evicted": 0, "reason": "no-demand-cache"}
    n = v.demand_count()
    if n <= budget:
        return {"evicted": 0, "cache": n, "budget": budget}
    now, tau = _now(), _tau()
    scored, after = [], ""
    while True:                     # walk the (bounded) cache once, decaying each to now
        page = v.demand_page(after=after, limit=5000)
        if not page:
            break
        for d in page:
            scored.append((current_mass(d, now=now, tau=tau), d["id"]))
        after = page[-1]["id"]
        if len(page) < 5000:
            break
    scored.sort()                   # coldest (lowest decayed mass) first
    evicted = 0
    for _score, aid in scored[:max(0, len(scored) - budget)]:
        v.evict_artifact(aid)
        evicted += 1
    return {"evicted": evicted, "cache": v.demand_count(), "budget": budget}


def maybe_evict(store) -> Dict[str, Any]:
    """Evict down to the measured-envelope budget. Cheap to call each cycle: a counter read, and the
    O(cache) sweep only when actually over. This is the loop that keeps a limited ember limited."""
    return evict(store, budget=budget_rows(store))
