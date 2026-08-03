"""Grant management endpoints (`/grants`) — sovereign, the lattice-backed.

Direct user→user grants, invite creation + claim, list, read, revoke, accept.
Mantle owns grants in its own lattice (see services/grant_store.py), so the whole
sharing surface lives here; Origin stays identity-only. Authorization for managing
grants is decided by `services.grant_service` (creator OR can_admin/can_share).
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from mantle.db.store import Database
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from mantle.services.dependencies import get_store_db, get_auth, AuthContext
from mantle.db.backend import (
    create_grant,
    get_grant_by_id,
    update_grant,
)
from mantle.entities.grant import Grant as GrantEntity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grants", tags=["Grants"])


# =============================================================================
# Request / Response Models
# =============================================================================

class ClaimInviteRequest(BaseModel):
    token: str


class CreateGrantRequest(BaseModel):
    resource_id: str
    # CRUDEASIO
    can_create: bool = False
    can_read: bool = True
    can_update: bool = False
    can_delete: bool = False
    can_invoke: bool = False
    can_add: bool = False
    can_share: bool = False
    can_admin: bool = False
    # Invite targeting (optional — makes this an invite grant)
    grantee_type: str = "user"              # "user" | "invite"
    grantee_id: Optional[str] = None        # user_id for direct grant; omit for invite
    target_entity: Optional[str] = None     # email, domain, etc. (invite only)
    target_entity_type: Optional[str] = None
    max_claims: Optional[int] = None
    requires_identity: bool = False
    name: Optional[str] = None
    notes: Optional[str] = None
    expires_at: Optional[str] = None
    state: str = "active"
    role: Optional[str] = None
    message: Optional[str] = None


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _grant_response(grant: GrantEntity) -> dict:
    return grant.to_dict()


def _require_admin(auth: AuthContext, resource_id: str, store_db: Database) -> None:
    """Raise 403 unless the caller can manage grants on the resource (creator OR can_admin)."""
    from mantle.services import grant_service
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    if not grant_service.can_admin(store_db, auth.user_id, resource_id):
        raise HTTPException(
            status_code=403,
            detail="Only the resource creator or an admin can manage grants",
        )


def _require_share_or_admin(auth: AuthContext, resource_id: str, store_db: Database) -> None:
    """Raise 403 unless the caller can create invites (creator OR can_share/can_admin)."""
    from mantle.services import grant_service
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    if not grant_service.can_share(store_db, auth.user_id, resource_id):
        raise HTTPException(
            status_code=403,
            detail="You need share or admin permission on this resource",
        )


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/invite-context")
async def get_invite_context_endpoint(
    token: str = Query(..., description="Raw invite claim token"),
    store_db: Database = Depends(get_store_db),
):
    """Non-PII invite metadata. Safe to call pre-auth."""
    from mantle.services import grant_service
    ctx = grant_service.get_invite_context(store_db, token)
    if not ctx:
        raise HTTPException(status_code=404, detail="Invite not found or expired")
    return ctx


@router.get("/invite-details")
async def get_invite_details_endpoint(
    token: str = Query(..., description="Raw invite claim token"),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Full invite details after verifying caller identity."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    from mantle.services import grant_service
    details = grant_service.get_invite_details(store_db, token, auth.user_id)
    if not details:
        raise HTTPException(status_code=404, detail="Invite not found or expired")
    return details


@router.get("/mine-sent")
async def list_invites_sent_endpoint(
    include_revoked: bool = Query(False, description="Include revoked / exhausted invites."),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List invite grants the caller has created (claim tokens never exposed)."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    from mantle.services import grant_service
    grants = grant_service.list_invites_sent(store_db, auth.user_id, include_revoked=include_revoked)
    return [_grant_response(g) for g in grants]


@router.post("/claim", status_code=status.HTTP_201_CREATED)
async def claim_invite_endpoint(
    body: ClaimInviteRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Claim an invite grant by presenting the raw invite token."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    from mantle.services import grant_service
    from mantle.services.grant_service import InviteNotFound, InviteExhausted, InviteIdentityMismatch
    try:
        created = grant_service.claim_invite(store_db, auth.user_id, body.token)
    except InviteIdentityMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except InviteExhausted as exc:
        raise HTTPException(status_code=410, detail=str(exc))
    except InviteNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _grant_response(created)


@router.get("")
async def list_grants_endpoint(
    resource_id: str = Query(..., description="Resource ID to list grants for"),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List all grants on a resource. Only the creator (or can_admin) may list."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    _require_admin(auth, resource_id, store_db)
    from mantle.db.backend import get_grants_for_collection
    grants = get_grants_for_collection(store_db, resource_id)
    return [_grant_response(g) for g in grants]


@router.get("/my-access")
async def my_access_endpoint(
    resource_id: str = Query(..., description="Resource to evaluate"),
    action: str = Query(..., description="CRUDEASIO action to evaluate (read/update/invoke/…)"),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """The caller's effective verdict for ONE action on ONE resource — server-derived.

    Added 2026-07-22 for crystal's op-level `requires_grant` gate: the gateway needs to ask
    "may the CALLER do X here" and nothing in the API answered it (grant listing is
    admin-only, and effective access is a light-cone computation that must never be
    re-derived client-side). This delegates to `check_access` — the SAME audited chokepoint
    every enforcing route uses, so the verdict here can never drift from the verdict at the
    data path, and each probe is witnessed in the access audit like any other decision.
    Denial and nonexistence both return allowed=false — no existence oracle."""
    from mantle.services.dependencies import check_access
    try:
        check_access(auth, resource_id, action, store_db)
        allowed = True
    except HTTPException:
        allowed = False
    return {"resource_id": resource_id, "action": action, "allowed": allowed}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_grant_endpoint(
    body: CreateGrantRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Create a new grant or invite.

    - Invite grants (``grantee_type == "invite"``) need can_share OR can_admin.
    - Direct user→user grants need can_admin (they bypass the claim/identity flow).
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")

    if body.grantee_type == GrantEntity.GRANTEE_INVITE:
        _require_share_or_admin(auth, body.resource_id, store_db)
        from mantle.services import grant_service

        target_email = None
        if (body.target_entity_type or "").lower() == "email":
            target_email = body.target_entity

        role = body.role or _role_from_bits(body) or "viewer"
        try:
            created, raw_token = grant_service.create_invite(
                store_db,
                user_id=auth.user_id,
                resource_id=body.resource_id,
                role=role,
                target_email=target_email,
                max_claims=body.max_claims if body.max_claims is not None else 1,
                name=body.name,
                notes=body.notes,
                expires_at=body.expires_at,
                message=body.message,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        response = _grant_response(created)
        response["claim_token"] = raw_token
        response["claim_url"] = grant_service.build_claim_url(raw_token)
        return response

    # Direct user->user grant: requires can_admin on the resource.
    _require_admin(auth, body.resource_id, store_db)

    now = _now_iso()
    grant = GrantEntity(
        id=str(uuid.uuid4()),
        resource_id=body.resource_id,
        grantee_type=body.grantee_type,
        grantee_id=body.grantee_id or "",
        granted_by=auth.user_id,
        can_create=body.can_create,
        can_read=body.can_read,
        can_update=body.can_update,
        can_delete=body.can_delete,
        can_invoke=body.can_invoke,
        can_add=body.can_add,
        can_share=body.can_share,
        can_admin=body.can_admin,
        requires_identity=body.requires_identity,
        target_entity=body.target_entity,
        target_entity_type=body.target_entity_type,
        max_claims=body.max_claims,
        state=body.state,
        name=body.name,
        notes=body.notes,
        granted_at=now,
        expires_at=body.expires_at,
        created_time=now,
        modified_time=now,
    )
    created = create_grant(store_db, grant)
    return _grant_response(created)


def _role_from_bits(body: CreateGrantRequest) -> Optional[str]:
    """Reverse-map CRUDEASIO bits on the request to an exact role preset, if any."""
    actual = {
        "can_create": body.can_create, "can_read": body.can_read,
        "can_update": body.can_update, "can_delete": body.can_delete,
        "can_invoke": body.can_invoke, "can_add": body.can_add,
        "can_share": body.can_share, "can_admin": body.can_admin,
    }
    enabled = {k for k, v in actual.items() if v}
    for role_name, preset in GrantEntity.ROLE_PRESETS.items():
        if enabled == {k for k, v in preset.items() if v}:
            return role_name
    return None


@router.get("/{grant_id}")
async def read_grant(
    grant_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Read a single grant. Visible to the grantee, the granter, or a resource admin."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    grant = get_grant_by_id(store_db, grant_id)
    if not grant:
        raise HTTPException(status_code=404, detail="Grant not found")
    if grant.grantee_id != auth.user_id and grant.granted_by != auth.user_id:
        try:
            _require_admin(auth, grant.resource_id, store_db)
        except HTTPException:
            raise HTTPException(status_code=404, detail="Grant not found")
    return _grant_response(grant)


@router.delete("/{grant_id}")
async def revoke_grant(
    grant_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Revoke a grant (soft-delete → state=revoked)."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    grant = get_grant_by_id(store_db, grant_id)
    if not grant:
        raise HTTPException(status_code=404, detail="Grant not found")

    # Granters may revoke their own pending invites; revoking anything else needs can_admin.
    is_revocable_invite = (
        grant.granted_by == auth.user_id
        and grant.grantee_type == GrantEntity.GRANTEE_INVITE
    )
    if not is_revocable_invite:
        _require_admin(auth, grant.resource_id, store_db)

    now = _now_iso()
    grant.state = GrantEntity.STATE_REVOKED
    grant.revoked_by = auth.user_id
    grant.revoked_at = now
    grant.modified_time = now
    if not update_grant(store_db, grant):
        raise HTTPException(status_code=500, detail="Failed to revoke grant")
    return {"id": grant_id, "state": "revoked"}


@router.post("/{grant_id}/accept")
async def accept_grant(
    grant_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Accept a pending_accept direct grant (grantee only)."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    grant = get_grant_by_id(store_db, grant_id)
    if not grant:
        raise HTTPException(status_code=404, detail="Grant not found")
    if grant.state != GrantEntity.STATE_PENDING_ACCEPT:
        raise HTTPException(status_code=400, detail="Grant is not pending acceptance")
    if grant.grantee_id != auth.user_id:
        raise HTTPException(status_code=403, detail="Only the grantee can accept this grant")

    now = _now_iso()
    grant.state = GrantEntity.STATE_ACTIVE
    grant.accepted_by = auth.user_id
    grant.accepted_at = now
    grant.modified_time = now
    if not update_grant(store_db, grant):
        raise HTTPException(status_code=500, detail="Failed to accept grant")
    return _grant_response(grant)
