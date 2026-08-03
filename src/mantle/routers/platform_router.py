"""Mantle /platform/users — platform-admin management (sovereign).

Platform admin is a Mantle concept: a user holding an active ``can_admin`` /
``can_update`` grant on the authority collection (``agience-authorities``), OR the
bootstrap operator (``resolve_operator_id``). Origin no longer manages admins —
this router does, because Mantle owns the authority collection.

Endpoints (all require the caller to already be a platform admin):
- ``GET    /platform/users``                     — list users + admin status
- ``POST   /platform/users/{user_id}/grant-admin``
- ``DELETE /platform/users/{user_id}/revoke-admin``
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

from mantle.db.store import Database
from fastapi import APIRouter, Depends, HTTPException

from mantle.db import backend as db_store
from mantle.entities.grant import grant_is_allow
from mantle.services.bootstrap_types import AUTHORITY_COLLECTION_SLUG, PEOPLE_COLLECTION_SLUG
from mantle.services.dependencies import AuthContext, get_store_db, get_auth, require_platform_admin
from mantle.services.operator import resolve_operator_id
from mantle.services.platform_topology import get_id_optional

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/platform", tags=["Platform"])


def _authority_id(db: Database) -> str:
    # Ensure the authority collection exists (idempotent runtime provisioning) —
    # it's the canonical resource platform-admin grants root to.
    authority_id = get_id_optional(AUTHORITY_COLLECTION_SLUG)
    if authority_id:
        return authority_id
    from mantle.services.seed_provisioning.user_provisioning import ensure_authority_collection
    authority_id = ensure_authority_collection(db)
    if not authority_id:
        raise HTTPException(status_code=503, detail="Authority collection unavailable")
    return authority_id


def _is_admin(db: Database, user_id: str, operator_id: str, authority_id: str) -> bool:
    if operator_id and user_id == operator_id:
        return True
    grants = db_store.get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=authority_id
    )
    # ⛔ `can_admin` ONLY. `can_update` used to satisfy this too, which meant WRITE access to the
    # authority collection was enough to create and destroy platform administrators — a quiet
    # escalation from "may edit this container" to "may appoint admins". Narrowed 2026-07-29
    # (John). Anyone who legitimately administers the platform holds `can_admin`; anyone who only
    # held `can_update` was never meant to appoint.
    return any(
        g.is_active() and grant_is_allow(g) and g.can_admin
        for g in grants
    )


def _grant_admin_flags(db, *, user_id: str, authority_id: str, granted_by: str, is_admin: bool, name: str) -> None:
    db_store.upsert_user_collection_grant(
        db,
        user_id=user_id,
        collection_id=authority_id,
        granted_by=granted_by,
        can_read=True,
        can_update=is_admin,
        can_admin=is_admin,
        name=name,
    )


def _card_context(card: dict) -> dict:
    raw = card.get("context")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


@router.get("/users")
async def list_users(
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> dict:
    """List platform users (the person cards in the People collection) with their
    platform-admin status. Admin-only."""
    require_platform_admin(auth, store_db)
    authority_id = _authority_id(store_db)
    operator_id = resolve_operator_id(store_db)

    people_id = get_id_optional(PEOPLE_COLLECTION_SLUG)
    users: List[dict] = []
    seen: set[str] = set()
    if people_id:
        for card in db_store.list_collection_artifacts(store_db, people_id) or []:
            ctxd = _card_context(card)
            identity = ctxd.get("identity") if isinstance(ctxd.get("identity"), dict) else {}
            uid = str(card.get("created_by") or identity.get("agience_root_id") or "")
            if not uid or uid in seen:
                continue
            seen.add(uid)
            users.append({
                "id": uid,
                "email": ctxd.get("email") or "",
                "name": card.get("name") or ctxd.get("display_name") or "",
                "picture": ctxd.get("picture"),
                "is_platform_admin": _is_admin(store_db, uid, operator_id, authority_id),
                "created_time": card.get("created_time"),
            })
    return {"users": users}


@router.post("/seed")
async def seed_platform(
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> dict:
    """Apply the platform seed corpus (the ``platform/`` artifacts + grants that the
    user/admin grant seeds target). Admin-only, idempotent.

    Mantle bundles NO seeds and does NOT apply them at boot (bare). The seed corpus
    is an INSTALL-PACKAGE artifact, mounted at ``AGIENCE_SEEDS_ROOT``; the install
    package calls THIS endpoint once (after the operator is established, before
    users) so Mantle applies the platform artifacts on explicit request — the
    application on top drives seeding via the API, not the data layer at boot.
    """
    require_platform_admin(auth, store_db)

    base = os.getenv("AGIENCE_SEEDS_ROOT")
    if not base:
        raise HTTPException(
            status_code=409,
            detail="No seed corpus configured (AGIENCE_SEEDS_ROOT unset) — Mantle is bare.",
        )
    root = Path(base) / "platform"
    if not root.is_dir():
        raise HTTPException(status_code=409, detail=f"Platform seed tree not found at {root}")

    from mantle.services.seed_provisioning import seed_from_artifacts
    report = seed_from_artifacts(store_db, root)
    # Re-derive server/platform ids into the topology after seeding.
    try:
        from mantle.services import server_registry
        server_registry.populate_ids()
    except Exception:
        logger.debug("populate_ids after seed failed", exc_info=True)

    logger.info("platform seed applied: %s", report.summary())
    return {"applied": True, "summary": report.summary(), "errors": list(report.errors)}


@router.post("/users/{user_id}/grant-admin")
async def grant_platform_admin(
    user_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> dict:
    """Grant a user platform admin (an admin grant on the authority collection)."""
    admin_id = require_platform_admin(auth, store_db)
    authority_id = _authority_id(store_db)

    # Self-retire the bootstrap fast-path: persist the OPERATOR as a real admin
    # BEFORE the first grant closes the bootstrap window — otherwise the operator
    # (who was admin only via the fast-path) would lose access.
    operator_id = resolve_operator_id(store_db)
    if operator_id:
        _grant_admin_flags(
            store_db, user_id=operator_id, authority_id=authority_id,
            granted_by=admin_id, is_admin=True, name="Platform admin (operator)",
        )

    _grant_admin_flags(
        store_db, user_id=user_id, authority_id=authority_id,
        granted_by=admin_id, is_admin=True, name="Platform admin",
    )
    logger.info("platform admin granted: user=%s by=%s", user_id, admin_id)
    return {"status": "granted", "user_id": user_id}


@router.delete("/users/{user_id}/revoke-admin")
async def revoke_platform_admin(
    user_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> dict:
    """Revoke a user's platform admin (downgrade to read on the authority collection)."""
    admin_id = require_platform_admin(auth, store_db)
    authority_id = _authority_id(store_db)

    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="Cannot revoke your own platform admin access")
    operator_id = resolve_operator_id(store_db)
    if operator_id and user_id == operator_id:
        raise HTTPException(status_code=400, detail="Cannot revoke the platform operator")

    _grant_admin_flags(
        store_db, user_id=user_id, authority_id=authority_id,
        granted_by=admin_id, is_admin=False, name="Platform user (admin revoked)",
    )
    logger.info("platform admin revoked: user=%s by=%s", user_id, admin_id)
    return {"status": "revoked", "user_id": user_id}
