"""Grant service — sovereign invite creation + claim logic (lattice-backed).

Mantle owns grants in its own lattice, so grant management and the invite/claim
flow live here, next to the light-cone enforcement in `services.dependencies`.
Origin stays identity-only.

Identity semantics on claim:
- When an invite declares a ``target_entity`` (email / domain / user_id), only the
  authenticated user that matches that target may claim it. Forwarding a link
  doesn't grant access.
- When no ``target_entity`` is set (open invite), ``max_claims`` controls who may
  claim.

Email delivery: Mantle has no email service. Invites are created without sending
mail — the caller receives the raw claim token + URL and delivers it.
`send_email=True` is best-effort and simply no-ops if no mail service is
importable.
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
from mantle.attenuation import ACTIONS, FLAG_OF, Mask
from mantle.entities.grant import grant_is_deny, Grant as GrantEntity, mask_of
from mantle.services.grant_key_service import hash_token

logger = logging.getLogger(__name__)


# Distinct from the grant-key prefix: an invite is claimed at /grants/claim, never
# presented as a Bearer credential, and the two must not be confusable by eye.
_INVITE_TOKEN_PREFIX = "agi_"
_INVITE_TOKEN_BYTES = 32


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_claim_token() -> str:
    """Return a raw, unhashed invite claim token."""
    return f"{_INVITE_TOKEN_PREFIX}{secrets.token_urlsafe(_INVITE_TOKEN_BYTES)}"


# ---------------------------------------------------------------------------
#  Permission helpers
# ---------------------------------------------------------------------------

# There is no `is_creator` predicate here: `created_by` is provenance only and grants
# no access (OPERATOR-ARCHITECTURE.md §13.2 "no owner fast-path"; agience-baseline.md
# §14). Access always requires an explicit grant, even for the creator. `created_by`
# is still on the artifact for provenance; read it directly if you need to display it.


def user_has_any_action(db: Database, user_id: str, resource_id: str, *actions: str) -> bool:
    """True when *user_id* holds a grant that AUTHORIZES any of *actions* on the resource.

    There is no creator fast-path to combine with — access is the grant, and only the grant.

    Deny-first, then allow, which is the order `services.dependencies.check_access` walks in and
    the reason this is not a plain "any grant that allows" fold: a principal holding both an
    allow and a deny on the same resource must be denied whichever order the store returns them
    in. A deny grant's CRUDEASIO bits name the actions it denies, so `carries` (the bit alone,
    effect-blind) is the right question on the deny pass and `allows` (bit AND effect) on the
    allow pass.

    Read as `getattr(g, flag, False)` over the raw flag names, with no effect check, a **deny**
    grant carrying `can_share`/`can_admin` confers sharing and grant administrationin
    the share/admin path. The flags are named by their CRUDEASIO action and resolved through
    `mask_of`, the one operator, so the bit and the effect
    can never again be answered separately.
    """
    if not user_id:
        return False
    grants = [g for g in get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=resource_id,
    ) if g.is_active()]
    masks = [mask_of(g) for g in grants]
    if any(m.is_deny and m.carries(a) for m in masks for a in actions):
        return False
    return any(m.allows(a) for m in masks for a in actions)


def user_has_any_flag(db: Database, user_id: str, resource_id: str, *flags: str) -> bool:
    """Back-compat shim: the same question spelled with ``can_*`` column names.

    Kept because callers outside this module name the stored columns. It translates to the
    action vocabulary rather than re-deriving the decision, so there is exactly one
    implementation. An unknown flag name maps to no action and therefore contributes nothing —
    a typo denies rather than opening a hole.
    """
    actions = [a for a in ACTIONS if FLAG_OF[a] in flags]
    return user_has_any_action(db, user_id, resource_id, *actions)


def can_share(db: Database, user_id: str, resource_id: str) -> bool:
    """Can *user_id* create invites on this resource? A GRANT with can_share OR can_admin — never a
    creator fast-path (provenance grants no access; the creator's explicit grant carries these)."""
    return user_has_any_action(db, user_id, resource_id, "share", "admin")


def can_admin(db: Database, user_id: str, resource_id: str) -> bool:
    """Can *user_id* manage grants on this resource? A GRANT with can_admin — no creator fast-path."""
    return user_has_any_action(db, user_id, resource_id, "admin")


# ---------------------------------------------------------------------------
#  Invite creation
# ---------------------------------------------------------------------------

#: The nine stored column names, in CRUDEASIO order. Derived from the attenuation vocabulary
#: rather than typed out again: a hand-written copy that disagreed about which columns exist
#: would silently drop a permission from every invite this module mints.
_ALL_FLAGS = tuple(FLAG_OF[a] for a in ACTIONS)


def effective_flags(db: Database, user_id: str, resource_id: str) -> Dict[str, bool]:
    """Every CRUDEASIO flag *user_id* actually holds on *resource_id* — the union of what their
    active ALLOW grants authorize, minus anything a deny grant names. NO creator fast-path:
    `created_by` is provenance and grants no access, so the creator holds exactly what their
    explicit grant gives (minted at creation).

    This is the ceiling `create_invite` clamps a role preset against, so a flag that is wrong
    here becomes a real grant on somebody's account one claim later. It used to fold
    `getattr(g, f, False)` over every grant with no effect check, which meant a **deny** grant
    carrying `can_admin` reported admin as held — and a holder of an ordinary `can_share` grant
    could then mint an admin invite, claim it, and convert a denial into an allow grant. Same
    defect as `user_has_any_action`, one layer up, and the escalation is the reason it is worth
    naming separately.

    Composed with the one operator: the allow grants join (union, `|` expressed as the union of
    each mask's authorized actions) and the deny grants are removed afterwards, in that order,
    so deny wins regardless of store ordering — the same precedence `check_access` enforces.
    """
    out = {f: False for f in _ALL_FLAGS}
    if not user_id:
        return out
    masks = [mask_of(g) for g in
             get_active_grants_for_principal_resource(db, grantee_id=user_id, resource_id=resource_id)
             if g.is_active()]
    held = set()
    for m in masks:
        if m.is_allow:
            held |= m.actions
    for m in masks:
        if m.is_deny:
            held -= m.actions          # a deny grant's bits name what it denies
    for action in held:
        out[FLAG_OF[action]] = True
    return out


def clamp_to_issuer(db: Database, *, issuer_id: str, resource_id: str,
                    requested: Dict[str, bool]) -> Dict[str, bool]:
    """Narrow *requested* to what *issuer_id* actually holds on *resource_id*.

    Nobody grants what they do not hold. Attenuation is the whole authority model — a grant
    composes down the lattice and never up — but the operator governs only authority already in the
    graph. Minting is where new authority enters, so it is checked here; unchecked, `can_admin` is a
    universal solvent, the only right needed to mint every other
    right, for yourself or anyone else.

    `_require_admin` is not that check. It asks whether the caller may manage grants here, which is
    a question about the ISSUER'S standing, not about the SIZE of what they are handing out. Both
    are needed, and only the first was being asked.

    This is the same clamp `create_invite` has always applied, lifted out so the direct path and
    the grant-key paths cannot drift from it. It is deliberately a meet against `effective_flags`
    — allow grants joined, deny grants subtracted after — so a deny on the issuer narrows what the
    issuer can pass on, which is the property that makes deny worth writing down at all.

    Flags absent from *requested* stay absent; a caller asking for nothing still gets nothing.
    """
    # THE OPERATOR, not a hand-written intersection. `Mask.__and__` is the meet the whole
    # authority model is defined by; a dict comprehension that happens to compute the same thing
    # is a second implementation that can drift, which is the defect
    # `test_attenuation_is_single_sourced` names.
    held = Mask.from_flags(effective_flags(db, issuer_id, resource_id))
    allowed = (Mask.from_flags(requested) & held).to_flags()
    # Narrowed back to what was actually asked for: `to_flags()` emits all nine, and feeding a
    # flag the caller never mentioned into a `Grant` is how the constructor's `can_read=True`
    # default silently widens.
    granted = {f: allowed[f] for f in requested}
    dropped = sorted(f for f, v in requested.items() if v and not allowed[f])
    if dropped:
        logger.warning(
            "grant on %s by %s requested %s but the issuer does not hold %s — clamped",
            resource_id, issuer_id, sorted(f for f, v in requested.items() if v),
            ", ".join(dropped),
        )
    return granted


def clamp_grant_to_issuer(db: Database, grant: GrantEntity, issuer_id: str) -> GrantEntity:
    """Narrow *grant*'s CRUDEASIO bits to what *issuer_id* holds on its resource, in place.

    The entity form of :func:`clamp_to_issuer`, and the one the routers use: it takes the grant
    the caller asked for and returns the grant they are entitled to mint. Working on the entity
    rather than on a flag dict keeps the bits inside `mask_of` / `Mask.to_flags` end to end, so
    no caller has to read a `can_*` field to make an authority decision.

    A deny grant is returned untouched. Its bits name what it FORBIDS, so meeting them with the
    issuer's held authority would narrow a prohibition — an issuer holding little could then only
    deny little, which is backwards. Nothing in the product mints deny grants today
    (`CreateGrantRequest` has no `effect` field); this is here so that stays true if one does.
    """
    if grant_is_deny(grant):
        return grant
    held = Mask.from_flags(effective_flags(db, issuer_id, grant.resource_id))
    for flag, value in (mask_of(grant) & held).to_flags().items():
        setattr(grant, flag, value)
    return grant


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

    held = effective_flags(db, user_id, resource_id)
    granted = {f: bool(preset.get(f, False)) and held[f] for f in _ALL_FLAGS}
    dropped = sorted(f for f in _ALL_FLAGS if preset.get(f, False) and not held[f])
    if dropped:
        logger.warning(
            "invite on %s by %s requested role %r but the issuer does not hold %s — clamped",
            resource_id, user_id, role, ", ".join(dropped),
        )

    raw_token = _generate_claim_token()
    token_hash = hash_token(raw_token)
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
        from mantle.config import FACET_URI as _BASE
    except Exception:
        try:
            from mantle.config import AUTHORITY_ISSUER as _BASE
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
        # No subject token here: this runs several frames below the route, off the
        # invite-creation path, and is handed a `user_id` rather than a request — so the call is
        # the unscoped one. It reads a display name for an email salutation, which is the
        # weakest use of this lookup in the tree and the easiest to drop if Origin ever requires
        # the subject token outright. Nothing is manufactured to stand in for it.
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
        from mantle.events.event_bus import emit_artifact_event_sync
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


def claim_invite(db: Database, user_id: str, raw_token: str,
                 claimant_email: str = "") -> GrantEntity:
    """Claim an invite for *user_id* using *raw_token*. Returns the new user grant.

    Raises InviteNotFound / InviteExhausted / InviteIdentityMismatch.
    """
    invite = _lookup_active_invite(db, raw_token)

    if invite.target_entity and invite.target_entity_type:
        _verify_target_match(db, user_id, invite, claimant_email)

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


def get_invite_details(db: Database, raw_token: str, user_id: str,
                       claimant_email: str = "") -> Optional[dict]:
    """Full invite details after verifying caller identity. None if invalid.

    ``claimant_email`` must be a verified address (the caller names that gate — see
    `_verify_target_match`). It is a parameter rather than a free variable, and defaulting to `""`
    keeps the single caller working and takes the Origin-lookup branch, which reads a verified
    address off the person
    record rather than trusting anything the request supplied.
    """
    try:
        invite = _lookup_active_invite(db, raw_token)
    except InviteClaimError:
        return None
    if invite.target_entity and invite.target_entity_type:
        try:
            _verify_target_match(db, user_id, invite, claimant_email)
        except InviteIdentityMismatch:
            return {"valid": True, "identity_mismatch": True}
    return {"valid": True, "resource_id": invite.resource_id,
            "granted_by": invite.granted_by, "name": invite.name}


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _lookup_active_invite(db: Database, raw_token: str) -> GrantEntity:
    token_hash = hash_token(raw_token)
    candidates = get_active_grants_for_grantee(db, token_hash, "invite")
    if not candidates:
        raise InviteNotFound("Invite not found or expired")
    invite = candidates[0]
    if invite.grantee_type != GrantEntity.GRANTEE_INVITE:
        raise InviteNotFound("Not an invite grant")
    if invite.state != GrantEntity.STATE_ACTIVE:
        raise InviteExhausted("Invite is no longer active")
    return invite


def _verify_target_match(db: Database, user_id: str, invite: GrantEntity,
                        claimant_email: str = "") -> None:
    """Raise InviteIdentityMismatch unless *user_id* matches the invite target.

    `claimant_email` is the caller's own verified address, which is why this works
    standalone: Mantle already verified a token carrying that address, so claim
    resolution doesn't need an HTTP round trip to Origin. (The fallback below still
    calls `person_service.get_user_by_id` — an HTTP GET to
    `{ORIGIN_URI}/internal/persons/{id}` — when no verified email is passed.)

    The caller must pass only a verified address. `grants_router` gates on
    `auth.email_verified`; this function trusts what it is handed, which is why the
    gate is named at the call site rather than left implicit.
    """
    match = False
    target_type = invite.target_entity_type
    target = invite.target_entity or ""

    if target_type == "user_id":
        match = user_id == target
    elif target_type in ("email", "domain"):
        email = (claimant_email or "").strip().lower()
        if not email:
            # Fallback: ask Origin. Correct on the full platform, unreachable standalone — which is
            # why the verified token claim is preferred above rather than added below.
            person = None
            try:
                from mantle.services.person_service import get_user_by_id
                # No subject token: this function is handed a `user_id` and a verified email, not
                # a request, so it has no bearer to forward. That is a second reason the arm
                # above is the preferred one — it needs no Origin call and therefore no scoping.
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
