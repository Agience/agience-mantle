"""The demand cache — what a limited ember holds is shaped by use, and stays within its envelope.

A reached row carries a mass. Each reach accretes it (+1); time decays it — demurrage, the 2nd law,
through the exponential decay kernel (`cool(dt, tau) = exp(-dt/tau)`). When the cache exceeds the
node's measured envelope, the coldest rows (lowest decayed mass) are evicted; the authoritative copy
still lives in the substrate, so a later reach restores it. This is what keeps a node limited rather
than merely starting empty.

Own-authored rows carry no demand entry (they were authored, not reached) and are therefore never
selected for eviction — a node never sheds its own observations, only its cache of others'.

Decay lives here, not in the store: the mantle lattice is dependency-free, and this cache is not the
store. The formula is inlined rather than imported from `prism.law` — no other consumer needs this
node's local eviction ranking to match anything else bit-for-bit, so there is nothing to single-source
against. The store holds only the raw `(mass, ts)`.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict

#: Rows per keyset page while walking the demand cache. Declared here rather than imported from
#: `db.vertex` on purpose: this module duck-types the store (`hasattr(v, "demand_count")`)
#: so that any implementation can serve it, and a hard import would quietly make one of them
#: mandatory. It is passed as the LIMIT and is the same name the end-of-walk test reads, so the
#: two can never disagree.
#:
#: A per-round MEMORY bound only — the eviction decision is identical at any page size, since the
#: full `scored` list is built either way. A different value would be right if the measured
#: envelope (`budget_rows`) could not hold a page alongside that list.
_PAGE = 5000


def _now() -> float:
    return time.time()


def _tau() -> float:
    """The demand decay time (seconds) — a flagged seam, default honest (~1 day). Measurable from
    the access trajectory (`optics.fit_dynamics(...).rates()`) where one exists; a declared tunable,
    not a hidden constant, everywhere else. Larger tau = a longer memory."""
    return max(1e-9, float(os.getenv("EMBER_DEMAND_TAU", "86400")))


def _cool(dt: float, tau: float) -> float:
    """Decay to now: the fraction of a stock that survives `dt` at decay time `tau`.

    `exp(-dt/tau)`, `dt` clamped at 0 so a negative travel survives whole. `tau` is taken as a
    limit at `tau -> 0+`: an instantaneous decay survives at 0 for any positive `dt`, at 1 at
    `dt == 0` — the same limit `prism.law._survive` takes, reproduced here because this is the
    one caller of it in the whole codebase."""
    dt = max(0.0, float(dt))
    tau = float(tau)
    if not (tau > 0.0):
        return 0.0 if dt > 0.0 else 1.0
    return math.exp(-dt / tau)


def touch(store, artifact_id: str) -> None:
    """A reach or access is demand. Decay the item's mass to now, then add one. A held row with no
    demand entry (own-authored) simply gains one on its first *reach* and is otherwise untouched."""
    v = getattr(store, "artifacts", None)
    if v is None or not hasattr(v, "demand_set"):
        return
    # Demand tracks held artifacts only — one demand row per held row, never an orphan. Touching an id
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
    """Max cache rows, derived from the measured envelope: `envelope_bytes · fraction / bytes_per_row`.
    The envelope (cgroup mem / data-volume free) and the bytes-per-row are measured; the fraction of
    the envelope to spend on the cache index, and the bytes-per-row estimate when it can't be
    measured, are flagged seams (default honest), per 'no arbitrary caps — leave a flagged seam'."""
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
    """Evict the coldest cache rows (lowest decayed mass) until the cache is within `budget`. Only
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
        page = v.demand_page(after=after, limit=_PAGE)
        if not page:
            break
        for d in page:
            scored.append((current_mass(d, now=now, tau=tau), d["id"]))
        after = page[-1]["id"]
        # A short page is the last page — true against the limit that was asked for, so it is
        # tested against that same name and never a second literal.
        if len(page) < _PAGE:
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
