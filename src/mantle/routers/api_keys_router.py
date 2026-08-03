"""Sovereign API-key management (`/api-keys`).

Mantle owns API keys in its own lattice (sovereign authorization — see
services/grant_store.py: `local` backend). Keys are created/listed/revoked here
by the authenticated USER (API-key principals cannot mint keys). The raw
`agc_...` key is returned exactly once, on creation.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from mantle.services.dependencies import get_store_db, get_auth, AuthContext
from mantle.api.api_key import (
    APIKeyResponse,
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyCreateResponse,
)
from mantle.entities.api_key import APIKey as APIKeyEntity
from mantle.db import backend as store_backend
from mantle.services import auth_service as auth_svc

router = APIRouter(prefix="/api-keys", tags=["API Keys"])

DEFAULT_SCOPES = [
    "resource:*:read",
    "resource:*:search",
    "resource:*:list",
    "resource:*:invoke",
]

DEFAULT_RESOURCE_FILTERS = {
    "workspaces": "*",
    "collections": "*",
}


def _reject_empty_resource_filters(resource_filters) -> None:
    """An explicitly-empty filter map is never a legitimate request.

    It cannot express any restriction, and historically it was the BROADEST
    possible value — `can_access_resource` returned True for every resource type
    when the map was empty, which is wider than DEFAULT_RESOURCE_FILTERS (that
    names only workspaces and collections, denying everything else). So
    `{"resource_filters": {}}` in a single request body produced a key with more
    reach than the documented default.

    Omit the field to accept the defaults; send a populated map to restrict.
    """
    if resource_filters is not None and not resource_filters:
        raise HTTPException(
            status_code=400,
            detail=(
                "resource_filters must not be empty — omit the field for the "
                "default scoping, or name the resource types to allow"
            ),
        )


def _require_user(auth: AuthContext) -> str:
    """API keys are user-owned; only a user JWT may manage them (not an api_key)."""
    if getattr(auth, "principal_type", None) == "api_key":
        raise HTTPException(status_code=403, detail="API keys cannot manage API keys")
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    return auth.user_id


def _to_response(k: APIKeyEntity) -> APIKeyResponse:
    return APIKeyResponse(
        id=k.id,
        user_id=k.user_id,
        name=k.name,
        client_id=getattr(k, "client_id", None),
        host_id=getattr(k, "host_id", None),
        server_id=getattr(k, "server_id", None),
        agent_id=getattr(k, "agent_id", None),
        display_label=getattr(k, "display_label", None),
        issued_by_user_id=getattr(k, "issued_by_user_id", None),
        created_from_client_id=getattr(k, "created_from_client_id", None),
        scopes=k.scopes,
        resource_filters=k.resource_filters,
        created_time=k.created_time,
        modified_time=k.modified_time,
        expires_at=k.expires_at,
        last_used_at=k.last_used_at,
        is_active=k.is_active,
    )


@router.post("", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    payload: APIKeyCreate,
    store_db=Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Create a personal API key for the authenticated user.

    Returns the raw `agc_...` key ONCE — the client must store it. When
    scopes/resource_filters are omitted, broad read-oriented defaults apply.
    """
    user_id = _require_user(auth)
    _reject_empty_resource_filters(payload.resource_filters)
    raw_key = auth_svc.generate_api_key()
    key_hash = auth_svc.hash_api_key(raw_key)

    now = datetime.now(timezone.utc).isoformat()
    entity = APIKeyEntity(
        id=str(uuid.uuid4()),
        user_id=user_id,
        key_hash=key_hash,
        name=payload.name,
        client_id=payload.client_id,
        host_id=payload.host_id,
        server_id=payload.server_id,
        agent_id=payload.agent_id,
        display_label=payload.display_label or "Default MCP Key",
        issued_by_user_id=user_id,
        created_from_client_id=payload.client_id,
        scopes=payload.scopes or list(DEFAULT_SCOPES),
        resource_filters=(
            payload.resource_filters if payload.resource_filters is not None
            else dict(DEFAULT_RESOURCE_FILTERS)
        ),
        created_time=now,
        modified_time=now,
        expires_at=payload.expires_at,
        last_used_at=None,
        is_active=True,
    )

    created = store_backend.create_api_key(store_db, entity)
    resp = _to_response(created)
    return APIKeyCreateResponse(**resp.model_dump(), key=raw_key)


@router.get("", response_model=List[APIKeyResponse])
async def list_api_keys(
    store_db=Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """List the authenticated user's API keys (metadata only, no secrets)."""
    user_id = _require_user(auth)
    keys = store_backend.get_api_keys_by_user(store_db, user_id)
    return [_to_response(k) for k in keys]


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    store_db=Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Read one of the user's own API keys."""
    user_id = _require_user(auth)
    key = store_backend.get_api_key_by_id(store_db, key_id)
    if not key or key.user_id != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    return _to_response(key)


@router.patch("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    payload: APIKeyUpdate,
    store_db=Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Update name / scopes / resource_filters / is_active on the user's key."""
    user_id = _require_user(auth)
    _reject_empty_resource_filters(getattr(payload, "resource_filters", None))
    key = store_backend.get_api_key_by_id(store_db, key_id)
    if not key or key.user_id != user_id:
        raise HTTPException(status_code=404, detail="API key not found")

    for field in ("name", "scopes", "resource_filters", "client_id",
                  "host_id", "server_id", "agent_id", "display_label", "is_active"):
        val = getattr(payload, field, None)
        if val is not None:
            setattr(key, field, val)
    key.modified_time = datetime.now(timezone.utc).isoformat()

    updated = store_backend.update_api_key(store_db, key)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update API key")
    return _to_response(updated)


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    key_id: str,
    store_db=Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Delete one of the user's own API keys."""
    user_id = _require_user(auth)
    key = store_backend.get_api_key_by_id(store_db, key_id)
    if not key or key.user_id != user_id:
        raise HTTPException(status_code=404, detail="API key not found")
    if not store_backend.delete_api_key(store_db, key_id):
        raise HTTPException(status_code=500, detail="Failed to delete API key")
