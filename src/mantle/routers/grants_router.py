"""Grant management endpoints (`/grants`) — sovereign, the lattice-backed.

Direct user→user grants, invite creation + claim, list, read, revoke, accept.
Mantle owns grants in its own lattice (see services/grant_store.py), so the whole
sharing surface lives here; Origin stays identity-only. Authorization for managing
grants is decided by `services.grant_service` (creator OR can_admin/can_share).
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, List, Optional

#: Imported at module scope so `/my-access` can publish the same derived `action` enum that
#: `/artifacts/visible` does. `attenuation` is stdlib-only, so there is no cycle to defer around.
from mantle.attenuation import ACTIONS
from mantle.db.store import Database
from fastapi import APIRouter, Depends, HTTPException, Query, status, Path
from pydantic import BaseModel, Field

from mantle.services.dependencies import get_store_db, get_auth, AuthContext, offload_sync
from mantle.db.backend import (
    create_grant,
    get_grant_by_id,
    update_grant,
)
from mantle.entities.grant import Grant as GrantEntity
from mantle.api.errors import ERROR_DESCRIPTIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/grants", tags=["Grants"])


#: `responses=` for one route. Declaration only, no runtime effect.
#:
#: Declaring only `422` when the handlers raise 400, 401, 403, 404 and 410 would leave a
#: generated client a case for validation failure and nothing else — including no branch for
#: `410 Gone` on a claimed or expired invite, which is the one answer on this surface a person
#: actually needs to be shown.
#:
#: Each set is derived from its handler by AST — the `HTTPException` codes reachable from
#: it — never typed from the table, with one subtraction: the four explicit `500`s are
#: not declared, because they are unreachable. `update_grant` always returns
#: the entity and `GrantEntity` defines no `__bool__`, so `if not update_grant(...)` is dead
#: in all four. Declaring a code a handler cannot produce tells every client to write a branch
#: it will never enter — the same defect as omitting one, pointed the other way.
#:
#: The prose lives in `mantle/api/errors.py`, shared with `/artifacts`. Two homes for what a
#: `404` means is what that module exists to prevent.
#: What each path id names on this surface — measured 2026-08-26, because all eight were bare
#: `str` with no description while every query parameter had one.
#:
#: The three are not interchangeable, and `grant_id` is the one that surprises: it accepts a
#: user grant, a grant key, or a bundle member. The router says so where it matters — "this path
#: reaches a grant key (`DELETE /grants/{id}` matches a key's own id) and a bundle member as
#: readily as a user grant" — and a caller reading only the parameter name would not guess it.
_GRANT_PARAM = (
    "The grant to act on. ⚠ Accepts THREE kinds of id: an ordinary user grant, a grant KEY, or a "
    "bundle MEMBER — revocation reaches all three by the same path. So a `404` here "
    "means 'no grant, key or member with this id that you may see', not 'not a grant'."
)
_KEY_PARAM = (
    "The grant KEY — the shareable credential itself, not the grant it was minted from and not a "
    "member hanging off it."
)
_MEMBER_PARAM = (
    "The member WITHIN the key named by `key_id`. ⚠ Removing a member is a soft revoke and leaves "
    "the row in place: the bundle it hangs off is what stops resolving, so the member is left "
    "where it is rather than deleted."
)


def _errors(*codes: int, ok: Optional[type] = None, ok_code: int = 200) -> dict:
    #: `ok=` documents the success body. Deliberately `responses={200:
    #: {"model": …}}` and not `response_model=`, following the `/artifacts` P-1 decision the
    #: audit points at: `response_model` filters, so a handler that returns a field the model
    #: has not caught up with would silently stop sending it. Under-describing a response is a
    #: documentation bug; silently truncating one is data loss whose symptom is a missing key
    #: nobody can trace.
    out = {c: {"description": ERROR_DESCRIPTIONS[c]} for c in codes}
    #: `ok_code`, because four of these routes answer `201` and not `200`. Declaring the body
    #: under `200` on a `status_code=201` route documents a response the route never sends and
    #: leaves the one it does send undeclared — the defect wearing a disguise. The success
    #: ratchet carries the same warning from an earlier sweep that measured 12 here instead of
    #: 16 by looking only at `200`.
    if ok is not None:
        out[ok_code] = {"model": ok}
    return out




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


#: Grantee types whose `grantee_id` is a hashed SECRET rather than a principal id
#: (a bearer token for `grant_key`, a claim token for `invite`). Serializing one would
#: publish the stored form of a live credential to anyone who can list grants.
_SECRET_GRANTEE_TYPES = {GrantEntity.GRANTEE_GRANT_KEY, GrantEntity.GRANTEE_INVITE}


class GrantResponse(BaseModel):
    """A grant as this API returns it.

    ALL 16 OPERATIONS DECLARED AN EMPTY SUCCESS SCHEMA, so a generated client got `any` for
    every response on this surface and could not name one key it would be handed. That is the same
    defect `/artifacts` fixed as P-1, and the reasoning transfers whole.

    DECLARED, NOT ENFORCED — `responses={200: {"model": …}}`, never `response_model=`. The latter
    FILTERS: the day a handler returns a field this model has not caught up with, that field
    silently stops reaching clients. A response that is under-described is a documentation bug; one
    that is silently truncated is a data-loss bug whose symptom is a missing key nobody can trace.

    The key set is DERIVED from `Grant.__init__`, checked against a live `to_dict` — 37 fields,
    and the generator asserts the two agree rather than trusting either. `_grant_response` returns
    `to_dict()` whole, so declaring a SUBSET would have documented a smaller response than the one
    actually sent, which is the failure this is meant to end.

    Every field is Optional in the SCHEMA and that is deliberate: it describes what a client may
    receive, and a grant mid-lifecycle genuinely has `null` in most of the timestamp and
    `*_requires_identity` slots. Optionality here is not laxity — it is the honest shape.
    """

    accepted_at: Optional[str] = None
    accepted_by: Optional[str] = None
    can_add: Optional[bool] = None
    can_admin: Optional[bool] = None
    can_create: Optional[bool] = None
    can_delete: Optional[bool] = None
    can_evict: Optional[bool] = None
    can_invoke: Optional[bool] = None
    can_read: Optional[bool] = None
    can_share: Optional[bool] = None
    can_update: Optional[bool] = None
    claims_count: Optional[int] = Field(
        None,
        description="How many times an invite has been claimed, against `max_claims`.",
    )
    created_time: Optional[str] = None
    effect: Optional[str] = Field(
        None,
        description="`allow` or `deny`. A deny wins over any allow on the same verb.",
    )
    expires_at: Optional[str] = Field(
        None,
        description="ISO-8601, or `null` for a grant that does not expire on its own.",
    )
    granted_at: Optional[str] = None
    granted_by: Optional[str] = None
    grantee_id: Optional[str] = Field(
        None,
        description="Who the grant is for. ⛔ `null` for a grant key or an invite: those hold a "
                    "TOKEN HASH here and the router redacts it, so a `null` on those two "
                    "grantee types means WITHHELD, not absent.",
    )
    grantee_type: Optional[str] = None
    id: Optional[str] = None
    invoke_requires_identity: Optional[bool] = None
    key_hint: Optional[str] = Field(
        None,
        description="A non-secret fragment for recognising a grant key. Never the key.",
    )
    last_used_at: Optional[str] = None
    max_claims: Optional[int] = None
    modified_time: Optional[str] = None
    name: Optional[str] = None
    notes: Optional[str] = None
    read_requires_identity: Optional[bool] = None
    requires_identity: Optional[bool] = None
    requires_nonce: Optional[bool] = None
    resource_id: Optional[str] = None
    revoked_at: Optional[str] = None
    revoked_by: Optional[str] = None
    state: Optional[str] = Field(
        None,
        description="Lifecycle: active, revoked, expired or pending acceptance.",
    )
    target_entity: Optional[str] = None
    target_entity_type: Optional[str] = None
    write_requires_identity: Optional[bool] = None


class GrantWithClaimResponse(GrantResponse):
    """A freshly created INVITE — the grant, plus the one-time claim material.

    `claim_token` and `claim_url` are returned EXACTLY ONCE, on creation. They are not stored in
    retrievable form and no read path returns them, which is why they are a separate model rather
    than nullable fields on `GrantResponse`: a schema that offered them on every grant would invite
    a client to look for them on a read and find `null` for ever."""

    claim_token: Optional[str] = Field(None, description="The invite secret. Shown once.")
    claim_url: Optional[str] = Field(None, description="The link a person follows to claim it.")


class GrantKeyResponse(GrantResponse):
    """A grant key and the members it bundles."""

    members: List[GrantResponse] = Field(
        default_factory=list,
        description="The grants this key bundles. Empty for a key that bundles none.")


class GrantKeyCreatedResponse(GrantKeyResponse):
    """A grant key at creation — the only response that carries the key itself.

    Separate from `GrantKeyResponse` on purpose. The secret exists in the response to exactly
    one request and is never retrievable again; declaring it on the read model would document a
    field that is always `null` there, which teaches a client to check for something it can never
    be given."""

    key: Optional[str] = Field(
        None, description="The grant key. Returned once, at creation, and never again.")


class RevokedResponse(BaseModel):
    """What a revoke answers: the thing acted on, and the state it is now in."""

    id: Optional[str] = Field(None, description="The grant or key that was revoked.")
    state: Optional[str] = Field(None, description="Its lifecycle state after the call.")


class MyAccessResponse(BaseModel):
    """Whether this caller may perform one action on one resource.

    Answers a stranger too — see the note on the route. `allowed: false` is a real answer, not a
    refusal, which is why this operation declares no errors."""

    resource_id: Optional[str] = Field(None, description="The resource asked about, echoed back.")
    action: Optional[str] = Field(None, description="The CRUDEASIO verb asked about, echoed back.")
    allowed: Optional[bool] = Field(None, description="Whether this principal may do it.")


class InviteContextResponse(BaseModel):
    """The little a stranger holding an invite link may be told BEFORE authenticating.

    Deliberately thin. It says whether the link is live and what KIND of thing it points at,
    never what or whose — anything more would make an unauthenticated endpoint a lookup."""

    valid: Optional[bool] = Field(None, description="Whether the invite is still claimable.")
    has_target: Optional[bool] = Field(None, description="Whether it names a specific target.")
    target_type: Optional[str] = Field(None, description="The KIND of target, never its id.")


class InviteDetailsResponse(BaseModel):
    """What an AUTHENTICATED caller may learn about an invite.

    TWO BRANCHES, and the fields differ between them — which is why every field is optional and
    the shape is documented rather than enforced. A caller whose identity does not match the
    invite gets `{valid, identity_mismatch}` and none of the descriptive fields: telling them who
    granted what would answer, for someone the invite was not for, the question the invite exists
    to answer for someone it was."""

    valid: Optional[bool] = Field(None, description="Whether the invite is still claimable.")
    identity_mismatch: Optional[bool] = Field(
        None, description="Set when the invite is bound to a different identity than the caller.")
    granted_by: Optional[str] = Field(None, description="Who issued it. Absent on a mismatch.")
    name: Optional[str] = Field(None, description="Its name, if it has one. Absent on a mismatch.")
    resource_id: Optional[str] = Field(
        None, description="What it grants on. Absent on a mismatch.")


def _grant_response(grant: GrantEntity) -> dict:
    """Serialize a grant for the API, with credential material redacted."""
    data = grant.to_dict()
    if data.get("grantee_type") in _SECRET_GRANTEE_TYPES:
        data.pop("grantee_id", None)
    return data


# The handlers below are `async def` and the grant store is synchronous, so anything that
# issues more than one query runs through `offload_sync` — a whole call per hop, never a
# fragment of a transaction (`db/seq.py`). `get_grant_by_id` stays on the loop: it is
# one indexed seek, and the hop would cost more than the read.


def _invalidate_cache_for(store_db: Database, grant: GrantEntity) -> None:
    """Drop the oracle's memoized light-cone decisions for whoever *grant* affects.

    Every path in this file that changes what a grant authorizes — create, claim, accept,
    revoke — ends here, because the memo is read by every key derivation and every cell
    decryption. Revocation is the direction that matters: the ledger already says
    `revoked` while a memo entry keeps issuing content keys until its TTL lapses.

    Which principal to name is `grant_key_service.principal_ids_for`'s job and not this
    file's: for a bearer key it is the root grant's id, not `grantee_id` (which holds the
    token hash), and for a bundle member it is the root at the top of the chain. Naming
    the wrong id fails silently — the call is made, the entry stays.

    Best-effort: a stale cache is a delay for the create path, and the TTL still bounds
    it for the revoke path. Logged at warning for revocations by the caller when it is
    the security-relevant direction.
    """
    try:
        from mantle.services import grant_key_service
        grant_key_service.invalidate_for(store_db, grant)
    except Exception:
        logger.warning("grant-cache invalidation failed for grant %s", grant.id, exc_info=True)


def _require_admin(auth: AuthContext, resource_id: str, store_db: Database) -> None:
    """Raise 403 unless the caller can manage grants on the resource (creator OR can_admin).

    Stays synchronous so it can be handed to `offload_sync` whole — it resolves the resource's
    creator and then walks the light cone, which is several queries, and an `HTTPException`
    raised in the worker still becomes its response.
    """
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

@router.get("/invite-context",
    responses=_errors(404, ok=InviteContextResponse),
)
async def get_invite_context_endpoint(
    token: str = Query(..., description="Raw invite claim token"),
    store_db: Database = Depends(get_store_db),
):
    """Non-PII invite metadata. Safe to call pre-auth."""
    from mantle.services import grant_service
    ctx = await offload_sync(grant_service.get_invite_context, store_db, token)
    if not ctx:
        raise HTTPException(status_code=404, detail="Invite not found or expired")
    return ctx


@router.get("/invite-details",
    responses=_errors(401, 404, ok=InviteDetailsResponse),
)
async def get_invite_details_endpoint(
    token: str = Query(..., description="Raw invite claim token"),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Full invite details after verifying caller identity."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    from mantle.services import grant_service
    details = await offload_sync(grant_service.get_invite_details, store_db, token, auth.user_id)
    if not details:
        raise HTTPException(status_code=404, detail="Invite not found or expired")
    return details


@router.get("/mine-sent",
    responses=_errors(401, ok=List[GrantResponse]),
)
async def list_invites_sent_endpoint(
    include_revoked: bool = Query(False, description="Include revoked / exhausted invites."),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List invite grants the caller has created (claim tokens never exposed)."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    from mantle.services import grant_service
    grants = await offload_sync(
        grant_service.list_invites_sent, store_db, auth.user_id,
        include_revoked=include_revoked,
    )
    return [_grant_response(g) for g in grants]


@router.post("/claim", status_code=status.HTTP_201_CREATED,
    responses=_errors(401, 403, 404, 410, ok=GrantResponse, ok_code=201),
)
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
        # The address is passed only if the issuer vouched for it. Unverified, it is a string the
        # user chose, and matching an invite on it would let anyone claim an invite addressed to
        # anyone else. Withheld, the claim falls back to asking Origin — the full-platform path.
        _claimant_email = auth.email if getattr(auth, "email_verified", False) else ""
        created = await offload_sync(
            grant_service.claim_invite, store_db, auth.user_id, body.token, _claimant_email
        )
    except InviteIdentityMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except InviteExhausted as exc:
        raise HTTPException(status_code=410, detail=str(exc))
    except InviteNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # The claimant now holds a grant they did not hold a moment ago, and a memo entry warmed
    # by their own pre-claim request would keep refusing them for the rest of the TTL — the
    # claim → open-the-resource sequence is one click, well inside it.
    _invalidate_cache_for(store_db, created)
    return _grant_response(created)


@router.get("",
    responses=_errors(401, ok=List[GrantResponse]),
)
async def list_grants_endpoint(
    resource_id: str = Query(..., description="Resource ID to list grants for"),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List all grants on a resource. Only the creator (or can_admin) may list."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    await offload_sync(_require_admin, auth, resource_id, store_db)
    from mantle.db.backend import get_grants_for_collection
    grants = await offload_sync(get_grants_for_collection, store_db, resource_id)
    return [_grant_response(g) for g in grants]


#: No `responses=` here, and that is the measured answer, and it asks
#: in as many words that this asymmetry is never "tidied up by someone matching it to its
#: fifteen siblings". This is that note.
#:
#: It raises nothing. The missing auth guard is load-bearing rather than forgotten: a
#: request with no credentials becomes an anonymous principal and is answered by the access
#: rules — "may this principal do this?" is a question a stranger is entitled to ask about
#: itself, and the honest answer is `false`, not `401`. Declaring a 401 here would advertise
#: a refusal this endpoint deliberately does not make.
@router.get("/my-access",
    #: Success only — no error codes, and that is C-3's measured answer, not an omission.
    responses=_errors(ok=MyAccessResponse),
)
async def my_access_endpoint(
    resource_id: str = Query(..., description="Resource to evaluate"),
    action: str = Query(
        ...,
        #: Was "(read/update/invoke/…)" — three of the nine, with an ellipsis for the rest.
        #: `check_access` rejects anything outside `ACTIONS` with `400 Unknown action`, so an
        #: unpublished vocabulary is a 400 the caller could not have avoided. Derived, never typed.
        description=(
            "CRUDEASIO action to evaluate. Permitted values, published as this parameter's "
            "enum: " + ", ".join(ACTIONS) + "."
        ),
        json_schema_extra={"enum": list(ACTIONS)},
    ),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """The caller's effective verdict for ONE action on ONE resource — server-derived.

    Serves crystal's op-level `requires_grant` gate: the gateway needs to ask "may the CALLER do X
    here", and grant listing alone can't answer it (it's admin-only, and effective access is a
    light-cone computation that must never be re-derived client-side). This delegates to
    `check_access` — the SAME audited chokepoint every enforcing route uses, so the verdict here can
    never drift from the verdict at the data path, and each probe is witnessed in the access audit
    like any other decision. Denial and nonexistence both return allowed=false — no existence
    oracle."""
    from mantle.services.dependencies import check_access
    try:
        await offload_sync(check_access, auth, resource_id, action, store_db)
        allowed = True
    except HTTPException:
        allowed = False
    return {"resource_id": resource_id, "action": action, "allowed": allowed}


@router.post("", status_code=status.HTTP_201_CREATED,
    responses=_errors(400, 401, ok=GrantWithClaimResponse, ok_code=201),
)
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
        await offload_sync(_require_share_or_admin, auth, body.resource_id, store_db)
        from mantle.services import grant_service

        target_email = None
        if (body.target_entity_type or "").lower() == "email":
            target_email = body.target_entity

        role = body.role or _role_from_bits(body) or "viewer"
        try:
            created, raw_token = await offload_sync(
                grant_service.create_invite,
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
    await offload_sync(_require_admin, auth, body.resource_id, store_db)

    from mantle.services import grant_service

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
    # Narrowed to what the issuer holds, after the entity is built and before it is stored.
    # `_require_admin` above answers whether this caller may manage grants here and says nothing
    # about how much they may hand out. Without this clamp, `can_admin` mints every other right —
    # to the caller or to anyone they name — while the invite path in this same handler refuses the
    # identical request against the same ledger.
    grant = await offload_sync(grant_service.clamp_grant_to_issuer, store_db, grant, auth.user_id)

    created = await offload_sync(create_grant, store_db, grant)

    # A grant nobody can see yet is a grant that does not work yet. `LightConeGrantVerifier`
    # memoizes (requester, type, action) → authorized-contexts for its TTL, so a person refused
    # the resource within that window stays refused after being granted it unless this cache
    # entry is dropped. Every grant mutation in this file ends this way, including this one.
    #
    # Scoped to the affected principals rather than global: `principal_ids_for` names them, and
    # clearing every entry would throw away the memoization the cache exists for.
    _invalidate_cache_for(store_db, created)

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


# =============================================================================
# Grant keys and grant bundles
# =============================================================================
#
# These are registered BEFORE `/{grant_id}` — FastAPI matches in declaration order,
# and a path parameter would otherwise swallow `/keys` as a grant id.
#
# A grant key is a bearer credential that IS a grant (see services/grant_key_service).
# A bundle is one with members, so a single key can carry several resources at
# different permission levels; the bundle's own bits are the ceiling over all of them.


class CreateGrantKeyRequest(BaseModel):
    name: str
    #: Omit for a bundle — a bundle root reaches nothing itself and exists to carry
    #: members. Provide one for a plain single-resource key.
    resource_id: Optional[str] = None
    role: Optional[str] = None              # role preset; mutually exclusive with the bits
    can_create: Optional[bool] = None
    can_read: Optional[bool] = None
    can_update: Optional[bool] = None
    can_delete: Optional[bool] = None
    can_evict: Optional[bool] = None
    can_invoke: Optional[bool] = None
    can_add: Optional[bool] = None
    can_share: Optional[bool] = None
    can_admin: Optional[bool] = None
    expires_at: Optional[str] = None
    notes: Optional[str] = None


class AddBundleMemberRequest(BaseModel):
    resource_id: str
    role: Optional[str] = None
    can_create: bool = False
    can_read: bool = True
    can_update: bool = False
    can_delete: bool = False
    can_evict: bool = False
    can_invoke: bool = False
    can_add: bool = False
    can_share: bool = False
    can_admin: bool = False
    name: Optional[str] = None
    expires_at: Optional[str] = None


def _explicit_bits(body: BaseModel) -> Optional[dict]:
    """The CRUDEASIO flags the caller actually set, or None if they set none."""
    bits = {
        flag: getattr(body, flag)
        for flag in GrantEntity.PERMISSION_FLAGS
        if getattr(body, flag, None) is not None
    }
    return bits or None


def _require_key_owner(grant: GrantEntity, auth: AuthContext) -> None:
    """Only the issuer manages a key. 404 rather than 403 — someone who does not hold
    the key has no business learning that this id names one."""
    if grant.grantee_type != GrantEntity.GRANTEE_GRANT_KEY or grant.granted_by != auth.user_id:
        raise HTTPException(status_code=404, detail="Grant key not found")


@router.post("/keys", status_code=status.HTTP_201_CREATED,
    responses=_errors(400, 401, 403, ok=GrantKeyCreatedResponse, ok_code=201),
)
async def create_grant_key_endpoint(
    body: CreateGrantKeyRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Mint a grant key. The raw token is returned EXACTLY once.

    With `resource_id`, the key carries that one resource at the given permissions —
    which requires can_admin on it. Without, it is an empty bundle; add resources with
    `POST /grants/keys/{id}/members`, each of which is admin-checked in its own right.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    if auth.principal_type == "grant_key":
        # A key that could mint keys would let a leaked credential outlive its own
        # revocation by issuing a fresh one.
        raise HTTPException(status_code=403, detail="Grant keys cannot mint grant keys")

    if body.resource_id:
        await offload_sync(_require_admin, auth, body.resource_id, store_db)

    from mantle.services import grant_key_service
    try:
        grant, raw_token = await offload_sync(
            grant_key_service.mint,
            store_db,
            user_id=auth.user_id,
            name=body.name,
            resource_id=body.resource_id,
            flags=_explicit_bits(body),
            role=body.role,
            expires_at=body.expires_at,
            notes=body.notes,
        )
    except ValueError as exc:                 # unknown role preset
        raise HTTPException(status_code=400, detail=str(exc))

    response = _grant_response(grant)
    response["key"] = raw_token
    response["members"] = []
    return response


@router.get("/keys",
    responses=_errors(401, ok=List[GrantResponse]),
)
async def list_grant_keys_endpoint(
    include_revoked: bool = Query(False, description="Include revoked keys."),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """List the grant keys the caller has minted (token hashes never exposed)."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    from mantle.services import grant_key_service
    keys = await offload_sync(
        grant_key_service.list_keys_issued_by,
        store_db, auth.user_id, include_revoked=include_revoked,
    )
    return [_grant_response(k) for k in keys]


@router.get("/keys/{key_id}",
    responses=_errors(401, 404, ok=GrantKeyResponse),
)
async def read_grant_key_endpoint(
    key_id: Annotated[str, Path(description=_KEY_PARAM)],
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Read one key and its direct members."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    grant = get_grant_by_id(store_db, key_id)
    if not grant:
        raise HTTPException(status_code=404, detail="Grant key not found")
    _require_key_owner(grant, auth)

    from mantle.services import grant_key_service
    response = _grant_response(grant)
    members = await offload_sync(grant_key_service.list_members, store_db, key_id)
    response["members"] = [_grant_response(m) for m in members]
    return response


@router.delete("/keys/{key_id}",
    responses=_errors(401, 404, ok=RevokedResponse),
)
async def revoke_grant_key_endpoint(
    key_id: Annotated[str, Path(description=_KEY_PARAM)],
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Revoke a key. Members are left in place — with the bundle carrying them revoked,
    they resolve to nothing, and the key can be reconstructed if it was revoked in error."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    grant = get_grant_by_id(store_db, key_id)
    if not grant:
        raise HTTPException(status_code=404, detail="Grant key not found")
    _require_key_owner(grant, auth)

    from mantle.services import grant_key_service
    #: `revoke` returns nothing, deliberately: a guard computed as `update_grant(...) is not None`
    #: is always true, since `update_grant` always returns the entity, so such a check can never
    #: be false. A failed persist reaches the client by `put_artifact` raising, not by a guard here.
    await offload_sync(grant_key_service.revoke, store_db, grant, auth.user_id)
    return {"id": key_id, "state": "revoked"}


@router.post("/keys/{key_id}/members", status_code=status.HTTP_201_CREATED,
    responses=_errors(400, 401, 404, ok=GrantResponse, ok_code=201),
)
async def add_bundle_member_endpoint(
    key_id: Annotated[str, Path(description=_KEY_PARAM)],
    body: AddBundleMemberRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Attach a resource to the bundle with its own permissions.

    Requires can_admin on the resource being added — the bundle's own bits never widen
    anything, they only cap what its members already carry.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    bundle = get_grant_by_id(store_db, key_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Grant key not found")
    _require_key_owner(bundle, auth)
    await offload_sync(_require_admin, auth, body.resource_id, store_db)

    from mantle.services import grant_key_service
    try:
        member = await offload_sync(
            grant_key_service.add_member,
            store_db,
            bundle_id=key_id,
            resource_id=body.resource_id,
            granted_by=auth.user_id,
            flags={flag: getattr(body, flag) for flag in GrantEntity.PERMISSION_FLAGS},
            role=body.role,
            name=body.name,
            expires_at=body.expires_at,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _grant_response(member)


@router.delete("/keys/{key_id}/members/{member_id}",
    responses=_errors(401, 404, ok=RevokedResponse),
)
async def remove_bundle_member_endpoint(
    key_id: Annotated[str, Path(description=_KEY_PARAM)],
    member_id: Annotated[str, Path(description=_MEMBER_PARAM)],
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
):
    """Detach a resource from the bundle."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    bundle = get_grant_by_id(store_db, key_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Grant key not found")
    _require_key_owner(bundle, auth)

    member = get_grant_by_id(store_db, member_id)
    # Verify the member actually hangs off THIS bundle, or one bundle's owner could
    # revoke a member of another's by id.
    if (not member
            or member.grantee_type != GrantEntity.GRANTEE_GRANT
            or member.grantee_id != key_id):
        raise HTTPException(status_code=404, detail="Bundle member not found")

    from mantle.services import grant_key_service
    #: No guard. Same shape as the key revocation above: `not True`, every call.
    await offload_sync(grant_key_service.revoke, store_db, member, auth.user_id)
    return {"id": member_id, "state": "revoked"}


@router.get("/{grant_id}",
    responses=_errors(401, 404, ok=GrantResponse),
)
async def read_grant(
    grant_id: Annotated[str, Path(description=_GRANT_PARAM)],
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
            await offload_sync(_require_admin, auth, grant.resource_id, store_db)
        except HTTPException:
            raise HTTPException(status_code=404, detail="Grant not found")
    return _grant_response(grant)


@router.delete("/{grant_id}",
    responses=_errors(401, 404, ok=RevokedResponse),
)
async def revoke_grant(
    grant_id: Annotated[str, Path(description=_GRANT_PARAM)],
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
        await offload_sync(_require_admin, auth, grant.resource_id, store_db)

    now = _now_iso()
    grant.state = GrantEntity.STATE_REVOKED
    grant.revoked_by = auth.user_id
    grant.revoked_at = now
    grant.modified_time = now
    #: `update_grant` has one `return entity` and no error path, so a guard computed as
    #: `not update_grant(...)` is always false: the annotation says `Optional`, but the body
    #: never produces `None`.
    update_grant(store_db, grant)

    # Revocation is the direction that matters — see `_invalidate_cache_for`. This path reaches
    # a grant KEY (`DELETE /grants/{id}` matches a key's own id) and a bundle MEMBER as readily
    # as a user grant, which is exactly where `grantee_id` is the token hash or an inner bundle
    # rather than the memo's key; the offload is because the bundle walk is several seeks.
    await offload_sync(_invalidate_cache_for, store_db, grant)

    return {"id": grant_id, "state": "revoked"}


@router.post("/{grant_id}/accept",
    responses=_errors(400, 401, 403, 404, ok=GrantResponse),
)
async def accept_grant(
    grant_id: Annotated[str, Path(description=_GRANT_PARAM)],
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
    #: No guard. Same as the revoke path above.
    update_grant(store_db, grant)

    # `pending_accept` → `active` is a reachability change like any other: the store filters
    # non-active grants at read, so a memo warmed while this grant was pending answers without
    # it for the rest of the TTL, and the grantee who just accepted stays refused.
    _invalidate_cache_for(store_db, grant)
    return _grant_response(grant)
