"""Grant service — sovereign invite creation + claim logic (the lattice-backed).

Restored to Mantle for the sovereign authorization model: Mantle owns grants in
its own lattice, so grant management + the invite/claim flow live here, next to
the light-cone enforcement in `services.dependencies`. Origin stays identity-only.

Identity semantics on claim:
- When an invite declares a ``target_entity`` (email / domain / user_id), only the
  authenticated user that matches that target may claim it. Forwarding a link
  doesn't grant access.
- When no ``target_entity`` is set (open invite), ``max_claims`` controls who may
  claim.

Email delivery: Mantle has no email service (that moved to Origin). Invites are
created WITHOUT sending mail — the caller receives the raw claim token + URL and
delivers it. `send_email=True` is best-effort and simply no-ops if no mail
service is importable.
"""
from __future__ import annotations

import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

from mantle.db.store import Database

from mantle.db.backend import (
    create_grant,
    update_grant,
    get_active_grants_for_grantee,
    get_active_grants_for_principal_resource,
)
from mantle.entities.grant import Grant as GrantEntity
from mantle.services.auth_service import hash_api_key

logger = logging.getLogger(__name__)


_INVITE_TOKEN_PREFIX = "agc_"
_INVITE_TOKEN_BYTES = 32


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_claim_token() -> str:
    """Return a raw, unhashed invite claim token."""
    return f"{_INVITE_TOKEN_PREFIX}{secrets.token_urlsafe(_INVITE_TOKEN_BYTES)}"


# ---------------------------------------------------------------------------
#  Permission helpers
# ---------------------------------------------------------------------------

# `is_creator(db, user_id, resource_id)` was here — DELETED 2026-07-29 (John).
#
# It answered "is this user the `created_by` of the artifact", and its own docstring recorded that
# it GRANTS NO ACCESS (John, 2026-07-28: "is_creator does not indicate ownership. it is provenance
# only."). It was kept "for PROVENANCE/display" — but nothing displayed it: zero references
# anywhere in the tree, and no test covered it.
#
# A predicate that looks like an authorization check, is named like one, and is reachable from an
# authorization module is a loaded gun even when it is currently unused — the next person to need
# "is this theirs?" finds it and wires it into a gate. The canon it cites is the reason it must not
# exist, not the reason to keep it: OPERATOR-ARCHITECTURE.md §13.2 "no owner fast-path";
# agience-baseline.md §14 "created_by is provenance only; it grants no access. Access always
# requires an explicit grant, even for the creator."
#
# `created_by` is still on the artifact for provenance; read it directly if you need to DISPLAY it.


def user_has_any_flag(db: Database, user_id: str, resource_id: str, *flags: str) -> bool:
    """True when *user_id* holds any of the named ``can_*`` flags on the resource.

    There is no creator fast-path to combine with — access is the grant, and only the grant."""
    if not user_id:
        return False
    grants = get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=resource_id,
    )
    for g in grants:
        if not g.is_active():
            continue
        for flag in flags:
            if getattr(g, flag, False):
                return True
    return False


def can_share(db: Database, user_id: str, resource_id: str) -> bool:
    """Can *user_id* create invites on this resource? A GRANT with can_share OR can_admin — never a
    creator fast-path (provenance grants no access; the creator's explicit grant carries these)."""
    return user_has_any_flag(db, user_id, resource_id, "can_share", "can_admin")


def can_admin(db: Database, user_id: str, resource_id: str) -> bool:
    """Can *user_id* manage grants on this resource? A GRANT with can_admin — no creator fast-path."""
    return user_has_any_flag(db, user_id, resource_id, "can_admin")


# ---------------------------------------------------------------------------
#  Invite creation
# ---------------------------------------------------------------------------

_ALL_FLAGS = ("can_create", "can_read", "can_update", "can_delete", "can_evict",
              "can_invoke", "can_add", "can_share", "can_admin")


def effective_flags(db: Database, user_id: str, resource_id: str) -> Dict[str, bool]:
    """Every CRUDEASIO flag *user_id* actually holds on *resource_id* — the union of their active
    grants, and nothing else. NO creator fast-path: `created_by` is provenance and grants no access,
    so the creator holds exactly what their explicit grant gives (minted at creation)."""
    out = {f: False for f in _ALL_FLAGS}
    if not user_id:
        return out
    for g in get_active_grants_for_principal_resource(db, grantee_id=user_id, resource_id=resource_id):
        if not g.is_active():
            continue
        for f in _ALL_FLAGS:
            if getattr(g, f, False):
                out[f] = True
    return out


def create_invite(
    db: Database,
    *,
    user_id: str,
    resource_id: str,
    role: str = "viewer",
    target_email: Optional[str] = None,
    max_claims: Optional[int] = 1,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    expires_at: Optional[str] = None,
    message: Optional[str] = None,
    send_email: bool = False,
) -> Tuple[GrantEntity, str]:
    """Create an invite grant using a named role preset.

    Returns ``(grant, raw_claim_token)``. The raw token is returned exactly once;
    the grant stores only a SHA-256 hash. Raises :class:`ValueError` for an
    unknown role. ``send_email`` is best-effort and no-ops without a mail service.
    """
    preset = GrantEntity.permissions_for_role(role)

    # ⛔⛔ AN INVITE COULD GRANT MORE THAN ITS ISSUER HELD — can_share ESCALATED TO can_admin.
    # The gate on this endpoint is `can_share OR can_admin OR creator`, but the preset's bits were
    # copied VERBATIM with no comparison to what the issuer actually holds, and nothing stops the
    # issuer claiming their own invite. `collaborator` carries `can_share` but NOT `can_admin`, so:
    #   owner shares collection C with X as "collaborator"  (can_share=True, can_admin=False)
    #   X: POST /grants {resource_id: C, grantee_type: "invite", role: "admin"}   -> passes the gate
    #   X claims the returned token -> X now holds can_admin + can_delete on C
    #   X revokes the owner's access and deletes C.
    # The invariant being restored is the ordinary one for delegated authority: YOU CANNOT GRANT
    # WHAT YOU DO NOT HOLD. Clamping (rather than rejecting) keeps over-asking non-fatal, which is
    # how the rest of this codebase treats over-claims — an invite for more than the issuer has
    # simply comes back smaller, and the caller is told which bits were dropped.
    held = effective_flags(db, user_id, resource_id)
    granted = {f: bool(preset.get(f, False)) and held[f] for f in _ALL_FLAGS}
    dropped = sorted(f for f in _ALL_FLAGS if preset.get(f, False) and not held[f])
    if dropped:
        logger.warning(
            "invite on %s by %s requested role %r but the issuer does not hold %s — clamped",
            resource_id, user_id, role, ", ".join(dropped),
        )

    raw_token = _generate_claim_token()
    token_hash = hash_api_key(raw_token)
    now = _now_iso()

    grant = GrantEntity(
        id=str(uuid.uuid4()),
        resource_id=resource_id,
        grantee_type=GrantEntity.GRANTEE_INVITE,
        grantee_id=token_hash,
        granted_by=user_id,
        can_create=granted["can_create"],
        can_read=granted["can_read"],
        can_update=granted["can_update"],
        can_delete=granted["can_delete"],
        # `can_evict` was MISSING here while every other flag was copied, so an editor/collaborator/
        # admin invite silently produced a grant with can_evict=False — an invited admin could not
        # remove members from the container they were invited to administer.
        can_evict=granted["can_evict"],
        can_invoke=granted["can_invoke"],
        can_add=granted["can_add"],
        can_share=granted["can_share"],
        can_admin=granted["can_admin"],
        requires_identity=bool(target_email),
        target_entity=target_email.lower() if target_email else None,
        target_entity_type="email" if target_email else None,
        max_claims=max_claims,
        state=GrantEntity.STATE_ACTIVE,
        name=name,
        notes=notes,
        granted_at=now,
        expires_at=expires_at,
        created_time=now,
        modified_time=now,
    )

    created = create_grant(db, grant)

    email_sent = False
    if send_email and target_email:
        email_sent = _send_invite_email(db, user_id, resource_id, target_email, raw_token, message)
    _emit_invite_event(
        resource_id, "grant.invite.created",
        {"grant_id": created.id, "role": role, "target_email": target_email, "email_sent": email_sent},
        actor_id=user_id,
    )

    return created, raw_token


def build_claim_url(raw_token: str) -> str:
    """Public claim URL for an invite token. Central so every surface agrees."""
    try:
        from origin.config import FACET_URI as _BASE
    except Exception:
        try:
            from origin.config import AUTHORITY_ISSUER as _BASE
        except Exception:
            _BASE = ""
    return f"{str(_BASE).rstrip('/')}/invite/{raw_token}"


def _send_invite_email(db, user_id, resource_id, target_email, raw_token, message) -> bool:
    """Best-effort invite email. Mantle has no mail service, so this normally
    no-ops (ImportError → False). Kept so a future mail seam can light up."""
    try:
        from mantle.services import email_service  # type: ignore
        from mantle.services.person_service import get_user_by_id
    except Exception:
        logger.debug("invite email skipped: no mail service in Mantle")
        return False
    try:
        person = get_user_by_id(db=db, id=user_id)
        from_name = (getattr(person, "name", None) or "").strip() or "Someone"
    except Exception:
        from_name = "Someone"
    claim_url = build_claim_url(raw_token)
    try:
        return _run_async(email_service.send_invite(target_email, from_name, "a workspace", claim_url, message))
    except Exception as exc:
        logger.warning("send_invite email delivery failed: %s", exc)
        return False


def _run_async(coro):
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _emit_invite_event(resource_id: str, event_name: str, data: dict, *, actor_id: Optional[str] = None) -> None:
    """Emit a grant.invite.* event. Never raises."""
    try:
        from mantle.event_bus import emit_artifact_event_sync
        emit_artifact_event_sync(resource_id, event_name, data, actor_id=actor_id)
    except Exception as exc:
        logger.debug("invite event emission failed: %s", exc)


# ---------------------------------------------------------------------------
#  Claim
# ---------------------------------------------------------------------------

class InviteClaimError(Exception):
    """Base for expected claim failures. Subclasses map to HTTP statuses."""


class InviteNotFound(InviteClaimError):
    """Token doesn't match any active invite grant. → 404."""


class InviteExhausted(InviteClaimError):
    """Invite has been revoked or reached its claim limit. → 410."""


class InviteIdentityMismatch(InviteClaimError):
    """Authenticated user doesn't match the invite's target identity. → 403."""


def claim_invite(db: Database, user_id: str, raw_token: str) -> GrantEntity:
    """Claim an invite for *user_id* using *raw_token*. Returns the new user grant.

    Raises InviteNotFound / InviteExhausted / InviteIdentityMismatch.
    """
    invite = _lookup_active_invite(db, raw_token)

    if invite.target_entity and invite.target_entity_type:
        _verify_target_match(db, user_id, invite)

    if invite.max_claims is not None and invite.claims_count >= invite.max_claims:
        raise InviteExhausted("Invite has reached its claim limit")

    now = _now_iso()
    new_grant = GrantEntity(
        id=str(uuid.uuid4()),
        resource_id=invite.resource_id,
        grantee_type=GrantEntity.GRANTEE_USER,
        grantee_id=user_id,
        granted_by=invite.granted_by,
        can_create=invite.can_create,
        can_read=invite.can_read,
        can_update=invite.can_update,
        can_delete=invite.can_delete,
        can_evict=invite.can_evict,
        can_invoke=invite.can_invoke,
        can_add=invite.can_add,
        can_share=invite.can_share,
        can_admin=invite.can_admin,
        requires_identity=True,
        state=GrantEntity.STATE_ACTIVE,
        name=invite.name,
        notes=f"Claimed from invite {invite.id}",
        granted_at=now,
        expires_at=invite.expires_at,
        created_time=now,
        modified_time=now,
    )
    created = create_grant(db, new_grant)

    invite.claims_count += 1
    invite.modified_time = now
    if invite.max_claims == 1:
        invite.state = GrantEntity.STATE_REVOKED
        invite.revoked_at = now
        invite.revoked_by = user_id
    update_grant(db, invite)

    logger.info("invite claimed: invite=%s resource=%s user=%s", invite.id, invite.resource_id, user_id)
    _emit_invite_event(
        invite.resource_id, "grant.invite.claimed",
        {"grant_id": created.id, "invite_id": invite.id, "user_id": user_id},
        actor_id=user_id,
    )
    return created


def list_invites_sent(db: Database, user_id: str, include_revoked: bool = False) -> list[GrantEntity]:
    """List invite grants created by *user_id* (default: active only)."""
    from mantle.db.backend import query_documents, COLLECTION_GRANTS

    filters: dict = {"grantee_type": GrantEntity.GRANTEE_INVITE, "granted_by": user_id}
    if not include_revoked:
        filters["state"] = GrantEntity.STATE_ACTIVE
    return query_documents(db, GrantEntity, COLLECTION_GRANTS, filters)


# ---------------------------------------------------------------------------
#  Pre/post-auth context
# ---------------------------------------------------------------------------

def get_invite_context(db: Database, raw_token: str) -> Optional[dict]:
    """Non-PII invite metadata. Safe pre-auth. None if invalid."""
    try:
        invite = _lookup_active_invite(db, raw_token)
    except InviteClaimError:
        return None
    return {"valid": True, "has_target": bool(invite.target_entity),
            "target_type": invite.target_entity_type}


def get_invite_details(db: Database, raw_token: str, user_id: str) -> Optional[dict]:
    """Full invite details after verifying caller identity. None if invalid."""
    try:
        invite = _lookup_active_invite(db, raw_token)
    except InviteClaimError:
        return None
    if invite.target_entity and invite.target_entity_type:
        try:
            _verify_target_match(db, user_id, invite)
        except InviteIdentityMismatch:
            return {"valid": True, "identity_mismatch": True}
    return {"valid": True, "resource_id": invite.resource_id,
            "granted_by": invite.granted_by, "name": invite.name}


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _lookup_active_invite(db: Database, raw_token: str) -> GrantEntity:
    token_hash = hash_api_key(raw_token)
    candidates = get_active_grants_for_grantee(db, token_hash, "invite")
    if not candidates:
        raise InviteNotFound("Invite not found or expired")
    invite = candidates[0]
    if invite.grantee_type != GrantEntity.GRANTEE_INVITE:
        raise InviteNotFound("Not an invite grant")
    if invite.state != GrantEntity.STATE_ACTIVE:
        raise InviteExhausted("Invite is no longer active")
    return invite


def _verify_target_match(db: Database, user_id: str, invite: GrantEntity) -> None:
    """Raise InviteIdentityMismatch unless *user_id* matches the invite target."""
    match = False
    target_type = invite.target_entity_type
    target = invite.target_entity or ""

    if target_type == "user_id":
        match = user_id == target
    elif target_type in ("email", "domain"):
        person = None
        try:
            from mantle.services.person_service import get_user_by_id
            person = get_user_by_id(db=db, id=user_id)
        except Exception:
            person = None
        email = (getattr(person, "email", None) or "").lower() if person else ""
        if email:
            if target_type == "email":
                match = email == target.lower()
            else:  # domain
                match = email.endswith("@" + target.lower())

    if not match:
        raise InviteIdentityMismatch("You are not the intended recipient of this invite")
