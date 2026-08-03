"""Pydantic schemas for the sovereign Mantle API-key endpoints (`/api-keys`).

Mantle owns API keys in its own store (sovereign authorization model —
see services/grant_store.py). These mirror the fields on `entities.api_key.APIKey`.
Scopes/resource_filters are accepted as opaque pass-through here; enforcement
lives in `services.dependencies` at request time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class APIKeyCreate(BaseModel):
    """Request body for `POST /api-keys`."""
    name: str = Field(..., min_length=1, max_length=100)
    scopes: Optional[List[str]] = None
    resource_filters: Optional[Dict[str, Any]] = None
    client_id: Optional[str] = None
    host_id: Optional[str] = None
    server_id: Optional[str] = None
    agent_id: Optional[str] = None
    display_label: Optional[str] = None
    expires_at: Optional[str] = None


class APIKeyUpdate(BaseModel):
    """Request body for `PATCH /api-keys/{id}`."""
    name: Optional[str] = None
    scopes: Optional[List[str]] = None
    resource_filters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    client_id: Optional[str] = None
    host_id: Optional[str] = None
    server_id: Optional[str] = None
    agent_id: Optional[str] = None
    display_label: Optional[str] = None


class APIKeyResponse(BaseModel):
    """Metadata for an API key (never includes the raw key or hash)."""
    id: str
    user_id: str
    name: str
    client_id: Optional[str] = None
    host_id: Optional[str] = None
    server_id: Optional[str] = None
    agent_id: Optional[str] = None
    display_label: Optional[str] = None
    issued_by_user_id: Optional[str] = None
    created_from_client_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    resource_filters: Dict[str, Any] = Field(default_factory=dict)
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    is_active: bool = True


class APIKeyCreateResponse(APIKeyResponse):
    """`POST /api-keys` response — carries the raw key ONCE."""
    key: str
