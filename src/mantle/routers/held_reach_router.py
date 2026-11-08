"""/held-reach — widenings of reach a peer sent, held until a person here accepts them.

Under threshold custody a mesh-consumed grant or `contains` edge that widens reach is not written
(`services.held_reach`). A person who manages the resource it would widen (`grant_service.can_admin`
on the grant's resource, or the edge's collection) sees it here and accepts or rejects it. To
accept is to write it on that person's own credential: the write is journaled, Origin checks that
this person may make it, and the co-signed write is the change. Refused, it stays held.
"""
from __future__ import annotations

import logging

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel

from mantle.api.errors import ERROR_DESCRIPTIONS

from mantle.db.reach_journal import ReachChangeRefused
from mantle.db.store import Database
from mantle.services import held_reach
from mantle.services.dependencies import AuthContext, get_auth, get_store_db, offload_sync

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/held-reach", tags=["Grants"])


class HeldReachChange(BaseModel):
    id: int
    kind: str
    resource_id: str
    received_from: Optional[str] = None
    held_at: Optional[str] = None
    grantee_id: Optional[str] = None
    grantee_type: Optional[str] = None
    member_id: Optional[str] = None


class HeldReachList(BaseModel):
    items: List[HeldReachChange]


class HeldReachAccepted(BaseModel):
    accepted: HeldReachChange


class HeldReachRejected(BaseModel):
    rejected: bool


def _errors(*codes: int, ok: type) -> dict:
    out = {c: {"description": ERROR_DESCRIPTIONS[c]} for c in codes}
    out[200] = {"model": ok}
    return out


def _resource(held: dict) -> str:
    p = held.get("payload") or {}
    return str(p.get("resource_id") or "") if held.get("kind") == "grant" else str(p.get("src") or "")


def _visible(db: Database, auth: AuthContext) -> list:
    from mantle.services import grant_service

    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    # Who manages each resource is asked once per resource, not once per held row.
    managed = [r for r in held_reach.held_resources(db)
               if grant_service.can_admin(db, auth.user_id, r)]
    return held_reach.list_held(db, managed)


def _summary(held: dict) -> dict:
    p = held.get("payload") or {}
    out = {"id": held["id"], "kind": held["kind"], "resource_id": _resource(held),
           "received_from": held.get("received_from"), "held_at": held.get("held_at")}
    if held["kind"] == "grant":
        out["grantee_id"] = p.get("grantee_id")
        out["grantee_type"] = p.get("grantee_type")
    else:
        out["member_id"] = p.get("dst")
    return out


def _held_for(db: Database, auth: AuthContext, held_id: int) -> dict:
    held = next((h for h in _visible(db, auth) if h["id"] == held_id), None)
    if held is None:
        raise HTTPException(status_code=404, detail="no held change here for you")
    return held


@router.get("", responses=_errors(401, ok=HeldReachList))
async def list_held_reach(auth: AuthContext = Depends(get_auth),
                          db: Database = Depends(get_store_db)):
    """The held widenings on resources the caller manages."""
    held = await offload_sync(_visible, db, auth)
    return {"items": [_summary(h) for h in held]}


@router.post("/{held_id}/accept", responses=_errors(401, 403, 404, ok=HeldReachAccepted))
async def accept_held_reach(held_id: int = Path(..., ge=1), auth: AuthContext = Depends(get_auth),
                            db: Database = Depends(get_store_db)):
    """Write a held widening as the caller. Origin co-signs it or refuses; refused, it stays."""
    def run():
        _held_for(db, auth, held_id)
        try:
            return held_reach.accept(db, held_id)
        except ReachChangeRefused as exc:
            raise HTTPException(status_code=403, detail=str(exc))

    held = await offload_sync(run)
    logger.info("held reach %s accepted by %s", held_id, auth.user_id)
    return {"accepted": _summary(held)}


@router.post("/{held_id}/reject", responses=_errors(401, 404, ok=HeldReachRejected))
async def reject_held_reach(held_id: int = Path(..., ge=1), auth: AuthContext = Depends(get_auth),
                            db: Database = Depends(get_store_db)):
    """Drop a held widening; nothing is written."""
    def run():
        _held_for(db, auth, held_id)
        return held_reach.reject(db, held_id)

    return {"rejected": await offload_sync(run)}
