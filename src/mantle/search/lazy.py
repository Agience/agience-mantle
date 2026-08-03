"""Lazy indexing (latent → materialized) — Phase 1 of the Information Gauge DB build.

A write leaves an artifact **latent** — stored, owned, and timed (WHO + WHEN) but
NOT indexed — unless indexing is requested eagerly. The search / anchor index
(WHERE) is **materialized on first access**. This is a per-deployment storage-cost
control: never embed the long tail nobody reads. Purely local — no economics, no
sharing (see INFORMATION-GAUGE-DB-IMPLEMENTATION.md §5, first-release scope).

Controlled by ``MANTLE_LAZY_INDEX`` (default OFF → today's eager-on-write
behaviour, so enabling is an explicit opt-in) plus a per-write ``index`` hint.
"""
from __future__ import annotations

import os
from typing import Optional

# "valence" — the one word borrowed from the design: the count of live relations.
# WHO + WHEN are always on (2); +WHERE/WHAT/HOW once materialized (5). Derived.
VALENCE_LATENT = 2
VALENCE_MATERIALIZED = 5

_TRUTHY = {"1", "true", "yes", "on"}


def lazy_index_default() -> bool:
    """Deployment default: leave new artifacts latent (defer indexing to first
    access)? ``MANTLE_LAZY_INDEX`` — default OFF (eager-on-write, unchanged)."""
    return (os.getenv("MANTLE_LAZY_INDEX", "") or "").strip().lower() in _TRUTHY


def resolve_lazy(index_hint: Optional[str] = None) -> bool:
    """Should this write be latent (defer indexing)? A per-write ``index`` hint
    wins; otherwise the deployment default. ``"eager"`` indexes now (today's
    behaviour); ``"lazy"`` defers to first access."""
    if index_hint == "lazy":
        return True
    if index_hint == "eager":
        return False
    return lazy_index_default()


def valence(materialized: bool) -> int:
    """Derived valence for a vertex: 2 while latent, 5 once materialized."""
    return VALENCE_MATERIALIZED if materialized else VALENCE_LATENT
