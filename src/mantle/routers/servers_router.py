"""Persona server registration — the servers self-register with Mantle.

Mantle carries no pre-known list of persona servers (no manifest). Chorus (the
persona host) POSTs each persona here at startup; Mantle records it dynamically
(``services.server_registry``) so name→id / client_id resolution works for
Mantle's outbound persona calls (welcome email, describers, search flavors).

Auth: platform-service only (Chorus signs a service JWT, aud=mantle). Users and
api-keys cannot register servers.
"""
from __future__ import annotations

import logging
from typing import Optional

from mantle.db.store import Database
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mantle.services import server_registry
from mantle.services.dependencies import AuthContext, get_store_db, get_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/servers", tags=["Servers"])


class RegisterServerRequest(BaseModel):
    name: str
    client_id: str
    path: str = ""
    title: str = ""
    role: str = ""
    summary: str = ""


def _require_platform_service(auth: AuthContext) -> None:
    if getattr(auth, "principal_type", None) != "service":
        raise HTTPException(status_code=403, detail="Platform service token required")


@router.post("/register")
async def register_server(
    body: RegisterServerRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> dict:
    """A persona (Chorus) registers itself. Idempotent; returns its UUID."""
    _require_platform_service(auth)
    if not body.name or not body.client_id:
        raise HTTPException(status_code=400, detail="name and client_id are required")
    server_id: Optional[str] = server_registry.register(
        store_db,
        name=body.name,
        client_id=body.client_id,
        path=body.path,
        title=body.title,
        role=body.role,
        summary=body.summary,
    )
    return {"name": body.name, "id": server_id, "client_id": body.client_id}


@router.get("")
async def list_servers(
    auth: AuthContext = Depends(get_auth),
) -> dict:
    """List registered persona servers (platform-service only)."""
    _require_platform_service(auth)
    return {
        "servers": [
            {"name": e.name, "client_id": e.client_id, "path": e.path,
             "id": server_registry.get_id(e.name), "title": e.title, "role": e.role}
            for e in server_registry.all_entries()
        ]
    }
