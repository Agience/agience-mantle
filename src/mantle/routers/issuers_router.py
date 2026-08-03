"""Admin API for trusted-issuer artifacts.

The privileged surface for managing which token issuers Mantle trusts. Every
endpoint requires the platform-admin grant. Created artifacts are owned by the
system principal (so the verifier's trust loader accepts them) but record the
authorizing admin — provenance roots to a person. Create/revoke fire the db
chokepoint event, so the verifier's trust set updates immediately (see the issuer
watcher). `vnd.agience.issuer+json` `create` is intentionally disabled on the
generic /artifacts API; this is the only way to mint a trusted issuer.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mantle.db.store import Database
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from mantle.services import issuers as issuers_svc
from mantle.services.dependencies import (
    AuthContext,
    get_store_db,
    get_auth,
    require_platform_admin,
)

issuers_router = APIRouter(prefix="/issuers", tags=["Issuers"])


class IssuerCreateRequest(BaseModel):
    issuer: str
    audience: Optional[str] = None
    jwks: Optional[Dict[str, Any]] = None
    jwks_uri: Optional[str] = None
    namespace: Optional[str] = None
    role: str = "external"


@issuers_router.post("", status_code=status.HTTP_201_CREATED)
async def create_issuer(
    body: IssuerCreateRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Add a trusted issuer (admin-only). Takes effect immediately."""
    admin_id = require_platform_admin(auth, store_db)
    try:
        art = issuers_svc.create_issuer_artifact(
            store_db, issuer=body.issuer, authorized_by=admin_id,
            jwks=body.jwks, jwks_uri=body.jwks_uri, audience=body.audience,
            namespace=body.namespace, role=body.role,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"id": art.id, "issuer": body.issuer, "role": body.role}


@issuers_router.get("")
async def list_issuers(
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List trusted issuers (admin-only)."""
    require_platform_admin(auth, store_db)
    out: List[dict] = []
    for a in issuers_svc.list_issuer_artifacts(store_db):
        try:
            ctx = json.loads(a.context) if isinstance(a.context, str) else (a.context or {})
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        out.append({
            "id": a.id, "issuer": ctx.get("issuer"), "role": ctx.get("role"),
            "audience": ctx.get("audience"), "namespace": ctx.get("namespace"),
            "authorized_by": ctx.get("authorized_by"),
        })
    return {"issuers": out}


@issuers_router.delete("/{artifact_id}")
async def revoke_issuer(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Revoke trust in an issuer (admin-only). Takes effect immediately."""
    admin_id = require_platform_admin(auth, store_db)
    if not issuers_svc.revoke_issuer_artifact(store_db, artifact_id, by=admin_id):
        raise HTTPException(status_code=404, detail="Trusted issuer not found")
    return {"revoked": artifact_id}
