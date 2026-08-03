"""Entitlement gate — platform enforcement for resource limits.

Core compares numbers. It never knows about Stripe, plan names, or
subscription state. Ophan pushes numeric limits via POST /internal/gate/set-limits;
this service reads them and compares against live counts or accumulated tallies.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from mantle.db.store import Database

from origin import config

logger = logging.getLogger(__name__)

# Collection names
# ⛔ SEVEN `if _is_lattice(): ... else: <AQL>` BRANCHES LIVED IN THIS FILE, plus six unreachable
# lattice tails after an early `return`. There is one store, so every condition was always true and
# every else was code that could not run but still had to be read, tested and kept plausible.
# [John, 2026-07-23: "leave one path. the only path. No constants, no fitting. no forcing."]
# The two names below were the lattice COLLECTION names for those dead paths.

# Free-tier defaults — used when no entitlement_cache row exists for a person.
_FREE_DEFAULTS = {
    "max_workspaces": 1,
    "max_artifacts": 500,
    "vu_limit": 100,
    "features": [],          # capability flags (e.g. "beacon") — none on free tier
}


def enforcement_enabled() -> bool:
    """True when billing enforcement is active (SaaS production only)."""
    return bool(config.BILLING_ENFORCEMENT_ENABLED)


# ---------------------------------------------------------------------------
# Lattice planes (MANTLE_DB=lattice): entitlement rows and tallies live as
# namespaced typed docs in the one store — same move as grants/commits/keys.
# ---------------------------------------------------------------------------
_ENT_CT = "application/vnd.agience.entitlement+json"
_TALLY_CT = "application/vnd.agience.usage-tally+json"


def _plane_get(db, doc_id: str, ct: str):
    raw = db.artifacts.get_artifact(doc_id)
    return raw if raw is not None and raw.get("content_type") == ct else None


def _plane_put(db, doc: dict) -> None:
    db.artifacts.put_artifact(doc)


# ---------------------------------------------------------------------------
# Entitlement cache (limits pushed by Ophan)
# ---------------------------------------------------------------------------

def get_limits(db: Database, person_id: str) -> Optional[dict]:
    """Read cached limits for a person, or None if no row."""
    doc = _plane_get(db, "entitlement:" + person_id, _ENT_CT)
    if doc is None:
        return None
    return {
        "max_workspaces": doc.get("max_workspaces"),
        "max_artifacts": doc.get("max_artifacts"),
        "vu_limit": doc.get("vu_limit"),
        "features": doc.get("features") or [],
    }


def get_or_default_limits(db: Database, person_id: str) -> dict:
    """Read cached limits, falling back to free-tier defaults."""
    return get_limits(db, person_id) or dict(_FREE_DEFAULTS)


def set_limits(
    db: Database,
    person_id: str,
    *,
    max_workspaces: Optional[int] = None,
    max_artifacts: Optional[int] = None,
    vu_limit: Optional[int] = None,
    features: Optional[list] = None,
) -> None:
    """Upsert entitlement cache row. Called by Ophan via the gate router.

    Numeric limits are always written. ``features`` (capability flags such as
    ``"beacon"``) are only written when provided — an update that omits them
    leaves any existing capability set untouched.
    """
    now = datetime.now(timezone.utc).isoformat()
    doc = {
        "_key": person_id,
        "person_id": person_id,
        "max_workspaces": max_workspaces,
        "max_artifacts": max_artifacts,
        "vu_limit": vu_limit,
        "updated_at": now,
    }
    if features is not None:
        doc["features"] = list(features)
    existing = _plane_get(db, "entitlement:" + person_id, _ENT_CT) or {}
    merged = {**existing, **{k: v for k, v in doc.items() if k != "_key"},
              "id": "entitlement:" + person_id, "content_type": _ENT_CT}
    merged.setdefault("features", [])
    _plane_put(db, merged)
    return


def has_feature(db: Database, person_id: str, feature: str) -> bool:
    """True if the person is entitled to the named capability (e.g. ``"beacon"``).

    When billing enforcement is off (local dev / self-hosted), all capabilities
    are granted — the gate only bites in SaaS production. In enforced mode a
    capability is present only when Ophan has pushed it into ``features``.
    """
    if not enforcement_enabled():
        return True
    limits = get_limits(db, person_id)
    return feature in ((limits or {}).get("features") or [])


# ---------------------------------------------------------------------------
# Usage tallies (consumable resources — VU only)
# ---------------------------------------------------------------------------

def get_tally(db: Database, person_id: str, dimension: str, period: str) -> int:
    """Read accumulated tally for a dimension/period, or 0."""
    key = f"{person_id}:{dimension}:{period}"
    doc = _plane_get(db, "tally:" + key, _TALLY_CT)
    return (doc or {}).get("total", 0)


def add_tally(
    db: Database,
    person_id: str,
    dimension: str,
    period: str,
    amount: int = 1,
) -> int:
    """Increment a tally, creating the row if needed. Returns new total."""
    key = f"{person_id}:{dimension}:{period}"
    now = datetime.now(timezone.utc).isoformat()
    doc = _plane_get(db, "tally:" + key, _TALLY_CT) or {
        "id": "tally:" + key, "content_type": _TALLY_CT,
        "person_id": person_id, "dimension": dimension, "period": period, "total": 0}
    doc["total"] = int(doc.get("total", 0)) + int(amount)
    doc["updated_at"] = now
    _plane_put(db, doc)
    return doc["total"]


def get_all_tallies(db: Database, person_id: str) -> dict:
    """Return all tallies for a person, grouped by dimension then period."""
    result: dict = {}
    for doc in db.artifacts.list_artifacts(content_type=_TALLY_CT):
        if doc.get("person_id") == person_id:
            result.setdefault(doc["dimension"], {})[doc["period"]] = doc.get("total", 0)
    return result


# ---------------------------------------------------------------------------
# Live counts (workspaces, artifacts — not tallied, queried in real time)
# ---------------------------------------------------------------------------

def count_workspaces(db: Database, person_id: str) -> int:
    """Count workspaces owned by this person, excluding the inbox.

    Unified store: a workspace is a `Collection` with content_type=workspace.
    The inbox collection is keyed by the person_id, so we exclude it explicitly.
    """
    from mantle.entities.collection import WORKSPACE_CONTENT_TYPE

    return sum(1 for c in db.artifacts.list_artifacts(
                   content_type=WORKSPACE_CONTENT_TYPE, created_by=person_id)
               if c.get("id") != person_id)


def count_artifacts(db: Database, person_id: str) -> int:
    """Count non-archived artifacts created by this person across all containers."""
    return sum(1 for _ in db.artifacts.list_artifacts(created_by=person_id))


def enforce_create_quota(db: Database, person_id: Optional[str], *, kind: str = "artifact") -> None:
    """Reject (HTTP 429) a create that would exceed the person's plan limit.

    No-op unless ``enforcement_enabled()`` — sovereign / self-hosted nodes do NOT
    gate (billing off); SaaS deployments push limits via the gate router and enforce
    here on the write path. Previously limits were defined but never enforced.
    """
    from fastapi import HTTPException

    if not enforcement_enabled() or not person_id:
        return
    limits = get_or_default_limits(db, person_id)
    if kind == "workspace":
        cap = limits.get("max_workspaces")
        if cap is not None and count_workspaces(db, person_id) >= cap:
            raise HTTPException(status_code=429, detail="Workspace limit reached for your plan")
    else:
        cap = limits.get("max_artifacts")
        if cap is not None and count_artifacts(db, person_id) >= cap:
            raise HTTPException(status_code=429, detail="Artifact limit reached for your plan")
