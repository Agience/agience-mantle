"""Mantle ``/system`` — the platform-admin surface, in one place.

Everything that administers the node rather than its contents lives here: trusted
issuers, platform users and their admin status, the seed corpus, and erasure. They
were two routers; they are one namespace because they answer to one predicate. Every
endpoint below gates on ``require_platform_admin`` and there is no second definition
of who that is — ``services.dependencies.is_platform_admin`` (an active ``can_admin``
grant on the authority collection ``agience-authorities``, or the bootstrap operator
while no such grant exists yet) is the only one.

Routes
------
- ``GET    /system/issuers``                     — list trusted issuers
- ``POST   /system/issuers``                     — add a trusted issuer
- ``DELETE /system/issuers/{artifact_id}``       — revoke trust in an issuer
- ``GET    /system/users``                       — list users + admin status
- ``POST   /system/seed``                        — apply the platform seed corpus
- ``POST   /system/users/{user_id}/grant-admin``
- ``DELETE /system/users/{user_id}/revoke-admin``
- ``POST   /system/erasure/{person_id}``         — inventory (default) or erase one person

Why issuers are not just ``POST /artifacts`` with a content type
----------------------------------------------------------------
The trust boundary is OWNERSHIP, not this endpoint. ``services.issuers.load_issuer_configs``
queries ``created_by=<system principal>``, and ``CreateArtifactRequest`` has no ``created_by``
field — the generic create path stamps the authenticated caller. So ``/artifacts`` can
produce something shaped exactly like an issuer (context and content_type are
caller-supplied) and the loader will never read it. Minting a *trusted* issuer means
creating a system-owned artifact, which only this router does; it records the
authorizing admin in ``authorized_by``, so ownership roots to the system while provenance
still roots to a person.

That is deliberately stronger than blocking the content type on the generic API would
be: it holds for every write path, including ones added later, rather than depending on
each of them remembering a rule.

It also enforces two rules the generic path has no concept of (``services/issuers.py``):
``role`` is ``external`` or ``platform``, and an external issuer MUST bind an ``audience`` —
without that bind, one tenant's IdP can mint tokens Mantle accepts for another.

Why grant-admin is not just ``POST /grants`` on the authority collection
------------------------------------------------------------------------
It usually could be — but not when it matters. ``POST /grants`` gates on
``grant_service.can_admin``, which requires an existing ``can_admin`` grant and has no
creator fast-path. The bootstrap operator holds no grant at all, so that route refuses
them and the FIRST admin could never be minted. This endpoint is the only path open in
that window, and it closes the window safely by persisting the operator as a real admin
before appointing anyone else.

``revoke-admin`` likewise carries the two guards nothing else enforces: no self-revoke,
no revoking the operator — i.e. the lockout protection.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mantle.api.errors import ERROR_DESCRIPTIONS
from mantle.db.store import Database
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from mantle.db import backend as db_store
from mantle.services import issuers as issuers_svc
from mantle.services.bootstrap_types import AUTHORITY_COLLECTION_SLUG, PEOPLE_COLLECTION_SLUG


from mantle.services.dependencies import (
    AuthContext,
    get_store_db,
    get_auth,
    is_platform_admin,
    offload_sync,
    require_platform_admin,
    _authority_bootstrap_complete,
)
from mantle.services.operator import resolve_operator_id
from mantle.services.platform_topology import get_id_optional

from mantle.services.acting_principal import SystemPrincipalUnavailable



#: `responses=` for one route, applied to `/system` 2026-08-26.
#:
#: A LOCAL BUILDER, ON PURPOSE. `mantle/api/errors.py` says why in its own words: *"one home
#: for the prose, not one home for the builder — the per-route assembly is three lines and each
#: surface's differs, so the builders stay local and only this table is shared."* What must never
#: be duplicated is what a code MEANS on this node.
#:
#: Every route here is admin-gated, and that is where most of the missing codes came from.
#: `get_auth` answers `401` before a handler runs, and `require_platform_admin` raises `403` —
#: neither appears as a `raise` in these handlers, so a sweep that reads only handler bodies
#: reports "this operation cannot fail" about a surface where the two commonest failures are
#: authentication and authorisation.
def _errors(*codes: int, ok: Optional[type] = None, ok_code: int = 200) -> dict:
    out = {c: {"description": ERROR_DESCRIPTIONS[c]} for c in codes}
    if ok is not None:
        out[ok_code] = {"model": ok}
    return out

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["System"])

# Every handler here is `async def` over a synchronous store, so the store work runs through
# `offload_sync` — one whole call per hop, never a fragment of a transaction (see
# `db/seq.py`). `require_platform_admin` is offloaded on every route because it is the
# gate on every route: it resolves the operator, scans the authority collection's grants, and
# then asks the grant store again for the caller.


# =============================================================================
# Schemas
# =============================================================================

class IssuerCreateRequest(BaseModel):
    """Declare a token issuer this node will trust."""
    issuer: str = Field(..., description="The `iss` claim value to trust.")
    audience: Optional[str] = Field(
        None,
        description="Required for an external issuer — the `aud` this node accepts from it.",
    )
    jwks: Optional[Dict[str, Any]] = Field(None, description="Inline JWKS document.")
    jwks_uri: Optional[str] = Field(None, description="Where to fetch the issuer's JWKS.")
    namespace: Optional[str] = Field(None, description="Tenant namespace this issuer maps into.")
    role: str = Field("external", description="`external` or `platform`.")


class IssuerCreatedResponse(BaseModel):
    """The trusted-issuer artifact that was minted."""
    id: str
    issuer: str
    role: str


class IssuerResponse(BaseModel):
    """One trusted issuer, as recorded on its artifact."""
    id: str
    issuer: Optional[str] = None
    role: Optional[str] = None
    audience: Optional[str] = None
    namespace: Optional[str] = None
    authorized_by: Optional[str] = None


class IssuerListResponse(BaseModel):
    issuers: List[IssuerResponse]


class IssuerRevokedResponse(BaseModel):
    revoked: str


class PlatformUserResponse(BaseModel):
    """A person card, plus whether this node treats them as a platform admin."""
    id: str
    email: str = ""
    name: str = ""
    picture: Optional[str] = None
    is_platform_admin: bool = False
    created_time: Optional[str] = None


class PlatformUserListResponse(BaseModel):
    users: List[PlatformUserResponse]
    #: Total person cards found, before the page was taken — so a caller knows
    #: whether another page exists without asking for it.
    total: int
    limit: int
    offset: int


class SeedResponse(BaseModel):
    applied: bool
    summary: str
    errors: List[str] = Field(default_factory=list)


class AdminChangeResponse(BaseModel):
    status: str
    user_id: str


# =============================================================================
# Issuers
# =============================================================================

@router.post(
    "/issuers",
    responses=_errors(400, 401, 403, 500, 503),
    status_code=status.HTTP_201_CREATED,
    response_model=IssuerCreatedResponse,
    summary="Add a trusted issuer",
    description=(
        "Mint a system-owned trusted-issuer artifact. Takes effect immediately — the "
        "create fires the db chokepoint event, so the verifier's trust set updates without "
        "a restart. Admin-only."
    ),
)
async def create_issuer(
    body: IssuerCreateRequest,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> IssuerCreatedResponse:
    admin_id = await offload_sync(require_platform_admin, auth, store_db)
    try:
        art = await offload_sync(
            issuers_svc.create_issuer_artifact,
            store_db, issuer=body.issuer, authorized_by=admin_id,
            jwks=body.jwks, jwks_uri=body.jwks_uri, audience=body.audience,
            namespace=body.namespace, role=body.role,
        )
    except ValueError as exc:
        # Deliberately raised, with a curated message naming what the caller must change —
        # "role must be 'external' or 'platform'", "an external issuer must bind an 'audience'".
        # Returning that text IS the point of a 400.
        raise HTTPException(status_code=400, detail=str(exc))
    except SystemPrincipalUnavailable as exc:
        # NARROWED FROM `except RuntimeError` [P-9, 2026-08-26]. The broad handler returned
        # `str(exc)` for ANY RuntimeError the call stack could raise — a library's, sqlite's,
        # anything — straight into the response body. It was right about the one raise it was
        # written for and wrong about every other one, which is exactly the shape P-9 names:
        # *"an unhandled exception's text is internal detail, and it is being returned to the
        # caller."*
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        # Anything else is unhandled by definition: logged in full, and the caller is told only
        # that it failed. The same pattern `artifacts_router` already uses for its 500s — one of
        # which says so outright: *"The server log carries the underlying error."*
        logger.error("issuer creation failed for %s", body.issuer, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create the issuer")
    return IssuerCreatedResponse(id=art.id, issuer=body.issuer, role=body.role)


@router.get(
    "/issuers",
    responses=_errors(401, 403),
    response_model=IssuerListResponse,
    summary="List trusted issuers",
    description="Every issuer this node accepts tokens from, with the admin who authorized it. Admin-only.",
)
async def list_issuers(
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> IssuerListResponse:
    await offload_sync(require_platform_admin, auth, store_db)
    out: List[IssuerResponse] = []
    for a in await offload_sync(issuers_svc.list_issuer_artifacts, store_db):
        try:
            ctx = json.loads(a.context) if isinstance(a.context, str) else (a.context or {})
        except (json.JSONDecodeError, TypeError):
            ctx = {}
        out.append(IssuerResponse(
            id=a.id, issuer=ctx.get("issuer"), role=ctx.get("role"),
            audience=ctx.get("audience"), namespace=ctx.get("namespace"),
            authorized_by=ctx.get("authorized_by"),
        ))
    return IssuerListResponse(issuers=out)


@router.delete(
    "/issuers/{artifact_id}",
    responses=_errors(401, 403, 404),
    response_model=IssuerRevokedResponse,
    summary="Revoke trust in an issuer",
    description="Takes effect immediately; tokens from this issuer stop verifying. Admin-only.",
)
async def revoke_issuer(
    artifact_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> IssuerRevokedResponse:
    admin_id = await offload_sync(require_platform_admin, auth, store_db)
    if not await offload_sync(issuers_svc.revoke_issuer_artifact, store_db, artifact_id, by=admin_id):
        raise HTTPException(status_code=404, detail="Trusted issuer not found")
    return IssuerRevokedResponse(revoked=artifact_id)


# =============================================================================
# Platform administration
# =============================================================================

def _authority_id(db: Database) -> str:
    # Ensure the authority collection exists (idempotent runtime provisioning) —
    # it's the canonical resource platform-admin grants root to.
    authority_id = get_id_optional(AUTHORITY_COLLECTION_SLUG)
    if authority_id:
        return authority_id
    from mantle.services.seed_provisioning.user_provisioning import ensure_authority_collection
    #: NO GUARD — the C-4 family, one instance further out than the grants audit measured.
    #: `ensure_authority_collection` is annotated `-> str` and its id is
    #: `derive_uuid(...)` — a uuid5 string, never empty — so `if not authority_id` could not
    #: fire and the `503` was unreachable.
    #:
    #: C-4 swept for callers of three named functions and this one is a caller of a FOURTH,
    #: so the sweep could not see it. The shape is not "the annotation says Optional" — here
    #: the annotation is honest — it is "the caller guards against a value the callee cannot
    #: produce". That is the wider class, and it is not detectable by looking at annotations.
    authority_id = ensure_authority_collection(db)
    return authority_id


def _grant_admin_flags(db, *, user_id: str, authority_id: str, granted_by: str, is_admin: bool, name: str) -> None:
    """Set or clear a user's admin flags on the authority collection.

    The single choke point for both `grant-admin` and `revoke-admin`, so the cache invalidation
    below covers both directions from one place.
    """
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

    # The light-cone verdict is memoized per principal, so a revoke that only rewrites the ledger
    # keeps issuing keys until the entry lapses. Revocation is the direction that matters: the
    # grant direction merely delays an admin's own access, while the revoke direction leaves
    # authority standing after it was taken away. The grantee is a plain user id, so it is the
    # memo key as-is and needs no translation.
    try:
        from mantle.search.mantle.wiring import invalidate_grant_cache
        invalidate_grant_cache(user_id)
    except Exception:
        logger.warning("grant cache invalidation failed after admin flag change", exc_info=True)


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


@router.get(
    "/users",
    responses=_errors(401, 403),
    response_model=PlatformUserListResponse,
    summary="List platform users",
    description=(
        "The person cards in the People collection, each with its platform-admin status. "
        "Paginated — the People collection grows without bound, so a page is taken rather "
        "than the whole set. Admin-only."
    ),
)
async def list_users(
    limit: int = Query(100, ge=1, le=1000, description="Page size."),
    offset: int = Query(0, ge=0, description="How many users to skip."),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> PlatformUserListResponse:
    await offload_sync(require_platform_admin, auth, store_db)
    # One hop for the whole listing rather than one per store call: this handler scans the
    # People collection and then evaluates the admin predicate once per user on the page, so
    # awaiting each separately would spend more time crossing threads than reading.
    users, total = await offload_sync(_platform_user_page, store_db, limit, offset)
    return PlatformUserListResponse(users=users, total=total, limit=limit, offset=offset)


def _platform_user_page(
    store_db: Database, limit: int, offset: int
) -> tuple[List[PlatformUserResponse], int]:
    """The person cards on one page, each with its admin verdict, plus the pre-page total."""
    authority_id = _authority_id(store_db)
    operator_id = resolve_operator_id(store_db)
    # Resolved once and reused across the loop below: the predicate would otherwise
    # re-scan the authority collection's whole grant set for every user listed.
    bootstrap_open = not _authority_bootstrap_complete(store_db)

    people_id = get_id_optional(PEOPLE_COLLECTION_SLUG)

    # Deduplicate first, page second. A page taken over the raw card list would shift
    # under itself whenever a person holds two cards, so the identity set is settled
    # before any slicing happens.
    ordered_ids: List[str] = []
    cards_by_id: Dict[str, tuple[dict, dict]] = {}
    if people_id:
        for card in db_store.list_collection_artifacts(store_db, people_id) or []:
            ctxd = _card_context(card)
            identity = ctxd.get("identity") if isinstance(ctxd.get("identity"), dict) else {}
            uid = str(card.get("created_by") or identity.get("agience_root_id") or "")
            if not uid or uid in cards_by_id:
                continue
            ordered_ids.append(uid)
            cards_by_id[uid] = (card, ctxd)

    total = len(ordered_ids)
    page = ordered_ids[offset:offset + limit]

    users: List[PlatformUserResponse] = []
    for uid in page:
        card, ctxd = cards_by_id[uid]
        users.append(PlatformUserResponse(
            id=uid,
            email=ctxd.get("email") or "",
            name=card.get("name") or ctxd.get("display_name") or "",
            picture=ctxd.get("picture"),
            # The SAME predicate the write endpoints gate on — a second
            # implementation here would risk this list reporting an admin the
            # API then refuses, or vice versa.
            is_platform_admin=is_platform_admin(
                store_db, uid, operator_id=operator_id,
                authority_id=authority_id, bootstrap_open=bootstrap_open,
            ),
            created_time=card.get("created_time"),
        ))
    return users, total


@router.post(
    "/seed",
    responses=_errors(401, 403, 409),
    response_model=SeedResponse,
    summary="Apply the platform seed corpus",
    description="Idempotent. Admin-only. 409 when this node is bare (no seed corpus mounted).",
)
async def seed_platform(
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> SeedResponse:
    """Apply the platform seed corpus (the ``platform/`` artifacts + grants that the
    user/admin grant seeds target).

    Mantle bundles no seeds and does not apply them at boot (bare). The seed corpus
    is an install-package artifact, mounted at ``AGIENCE_SEEDS_ROOT``; the install
    package calls this endpoint once (after the operator is established, before
    users) so Mantle applies the platform artifacts on explicit request — the
    application on top drives seeding via the API, not the data layer at boot.
    """
    await offload_sync(require_platform_admin, auth, store_db)

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
    # Reads a whole seed tree off disk and writes every artifact and grant in it — the
    # longest-running call in this service, and it must run start to finish in one thread.
    report = await offload_sync(seed_from_artifacts, store_db, root)

    logger.info("platform seed applied: %s", report.summary())
    return SeedResponse(applied=True, summary=report.summary(), errors=list(report.errors))


@router.post(
    "/users/{user_id}/grant-admin",
    responses=_errors(401, 403),
    response_model=AdminChangeResponse,
    summary="Grant a user platform admin",
    description=(
        "Writes an admin grant on the authority collection. Also persists the bootstrap "
        "operator as a real admin first, so the fast-path can close without locking them out. "
        "Admin-only."
    ),
)
async def grant_platform_admin(
    user_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> AdminChangeResponse:
    admin_id = await offload_sync(require_platform_admin, auth, store_db)
    authority_id = await offload_sync(_authority_id, store_db)

    # Self-retire the bootstrap fast-path: persist the operator as a real admin
    # before the first grant closes the bootstrap window — otherwise the operator
    # (who was admin only via the fast-path) would lose access.
    operator_id = await offload_sync(resolve_operator_id, store_db)
    if operator_id:
        await offload_sync(
            _grant_admin_flags,
            store_db, user_id=operator_id, authority_id=authority_id,
            granted_by=admin_id, is_admin=True, name="Platform admin (operator)",
        )

    await offload_sync(
        _grant_admin_flags,
        store_db, user_id=user_id, authority_id=authority_id,
        granted_by=admin_id, is_admin=True, name="Platform admin",
    )
    logger.info("platform admin granted: user=%s by=%s", user_id, admin_id)
    return AdminChangeResponse(status="granted", user_id=user_id)


@router.delete(
    "/users/{user_id}/revoke-admin",
    responses=_errors(400, 401, 403),
    response_model=AdminChangeResponse,
    summary="Revoke a user's platform admin",
    description=(
        "Downgrades them to read on the authority collection. Refuses to revoke the caller "
        "themselves or the platform operator — the lockout guards. Admin-only."
    ),
)
async def revoke_platform_admin(
    user_id: str,
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> AdminChangeResponse:
    admin_id = await offload_sync(require_platform_admin, auth, store_db)
    authority_id = await offload_sync(_authority_id, store_db)

    if user_id == admin_id:
        raise HTTPException(status_code=400, detail="Cannot revoke your own platform admin access")
    operator_id = await offload_sync(resolve_operator_id, store_db)
    if operator_id and user_id == operator_id:
        raise HTTPException(status_code=400, detail="Cannot revoke the platform operator")

    await offload_sync(
        _grant_admin_flags,
        store_db, user_id=user_id, authority_id=authority_id,
        granted_by=admin_id, is_admin=False, name="Platform user (admin revoked)",
    )
    logger.info("platform admin revoked: user=%s by=%s", user_id, admin_id)
    return AdminChangeResponse(status="revoked", user_id=user_id)


# =============================================================================
# Erasure
# =============================================================================

@router.post(
    "/erasure/{person_id}",
    responses=_errors(400, 401, 403),
    summary="Inventory or erase everything grounded at one person",
    description=(
        "A dry run unless `apply=true`. The inventory comes back in the same shape either "
        "way, so a dry run can be diffed against the real one. Admin-only."
    ),
)
async def erase_person(
    person_id: str,
    apply: bool = Query(False, description="Execute the erasure. Omitted or false, this reports only."),
    include_identity: bool = Query(
        False,
        description="Also remove the person artifact and, when applying, the identity record. "
                    "Off means RESET: the person stays, everything they made goes.",
    ),
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> dict:
    """The front door to ``shard/erasure.py`` — the right-to-erasure primitive.

    **A dry run by default.** ``apply`` must be sent explicitly and true for anything to be
    deleted; the inventory comes back in the same shape either way, so an operator can diff a
    dry run against the real one and see exactly what changed. That ordering is the module's
    own contract: the inventory is the product, the deletion is a separate decision made by a
    human who has read it.

    ``include_identity`` chooses between two genuinely different acts. Off is a RESET — the
    person keeps their identity, everything grounded at them goes. On is a full erasure: the
    person artifact goes with it, and an applied run also drops the standalone identity record
    (email, provider subject, password hash), which lives on the identity plane and is not an
    artifact, so the erasure primitive cannot reach it.

    Erasure is defined positively — what is provably grounded at this person — so the report
    also lists what was found and deliberately left alone (``not_yours``: the commons, the
    registry, the operators). Reading "I found these and they are not yours to erase" is part
    of the answer, not noise.

    The response is an open dict rather than a declared model: it is the erasure module's own
    inventory, and pinning a schema here would silently drop any category the module learns to
    report. An erasure report that quietly omits a category is the one failure mode this
    endpoint must not have.
    """
    admin_id = await offload_sync(require_platform_admin, auth, store_db)
    if not person_id:
        raise HTTPException(status_code=400, detail="person_id required")

    from mantle.shard import erasure
    # A whole-store sweep, and on `apply` a whole-store delete. Offloaded as one call: the
    # inventory and the deletion it authorises are one operation and cannot be split.
    report = await offload_sync(
        erasure.erase, store_db, person_id, apply=apply, include_identity=include_identity
    )
    report["dry_run"] = not apply
    report["requested_by"] = admin_id

    # The identity record is a separate plane, so an applied full erasure has to say so
    # separately — reporting the artifact sweep as complete while the email row survives
    # would be the one claim this endpoint must never make.
    if include_identity and apply:
        from mantle.db import identity_backend as identity_store
        try:
            report["identity_record_removed"] = bool(identity_store.delete_person(store_db, person_id))
        except Exception:
            logger.warning("erasure: identity record delete failed for %s", person_id, exc_info=True)
            report["identity_record_removed"] = False
            report["complete"] = False

    logger.info("erasure %s for person=%s by=%s: total=%d removed=%s",
                "APPLIED" if apply else "dry-run", person_id, admin_id,
                report.get("total", 0), report.get("removed", "-"))
    return report
