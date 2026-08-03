# /services/dependencies.py
#
# Unified auth dependency layer.
#
# Public API:
#   AuthContext        — dataclass returned by get_auth()
#   get_auth()         — single FastAPI dependency for all endpoints
#   get_person()       — load Person entity for the authenticated user
#   resolve_auth()     — plain-function core (usable outside FastAPI DI)
#   require_platform_admin() — post-auth guard
#   get_end_user_claims() — user-only JWT guard (rejects API-key JWTs)
#   check_access()        — verify principal has permission on an artifact
#   _check_grant_permission() — grant permission helper

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from fastapi import HTTPException, Depends, Security, Request
from fastapi.security import (
    OAuth2AuthorizationCodeBearer,
)
from mantle.db.store import Database

from typing import Generator

from mantle.clients.origin_client import get_origin_client
from origin import config
from mantle.services.acting_principal import acting_from_auth, set_acting_principal
from mantle.services.person_service import get_user_by_id  # now expects Database
from mantle.services.auth_service import verify_token
from mantle.db import backend as db_store
from mantle.services.bootstrap_types import AUTHORITY_COLLECTION_SLUG
from mantle.services.platform_topology import get_id
from mantle.entities.person import Person
from mantle.entities.api_key import APIKey as APIKeyEntity
from mantle.entities.grant import Grant as GrantEntity, grant_is_allow, grant_is_deny

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB connection (FastAPI dependency)
# ---------------------------------------------------------------------------


def get_store_db() -> Generator[Database, None, None]:
    """The store handle for FastAPI dependency injection — backend-selected (see `db.backend`).

    Yields Mantle's OWN store (`db.lattice_api.LatticeDatabase`, one SQLite file opened once
    per process — THE STANDALONE DB; the same handle startup uses).
    """
    from mantle.db import backend
    yield backend.store_handle()


# ---------------------------------------------------------------------------
# AuthContext
# ---------------------------------------------------------------------------

@dataclass
class AuthContext:
    """Unified auth context returned by ``get_auth()``.

    Replaces all legacy auth dependency return types (tuples, Person,
    old AuthContext) with a single consistent shape.  Field names follow
    the Unified Artifact API spec.
    """

    principal_id: str = ""                              # user_id | api_key_id | server_client_id
    principal_type: str = "user"                        # "user" | "api_key" | "server" | "mcp_client" | "grant_key"
    user_id: Optional[str] = None                       # present for user, mcp_client, api_key, delegation
    grants: List[GrantEntity] = field(default_factory=list)  # loaded server-side
    api_key_id: Optional[str] = None                    # if auth was via API key
    api_key_entity: Optional[APIKeyEntity] = None       # full entity — needed by collection service
    server_id: Optional[str] = None                     # if auth was via server token
    actor: Optional[str] = None                         # delegation: acting server
    authority: Optional[str] = None                     # issuer / authority identity (external IdP = tenant)
    host_id: Optional[str] = None                       # host identity (platform instance)
    email: Optional[str] = None                         # profile email from token claims (external IdP login)
    name: Optional[str] = None                          # profile display name from token claims
    bearer_grant: Optional[GrantEntity] = None           # convenience: grant resolved from Bearer grant key
    target_artifact_id: Optional[str] = None             # artifact scoping from prefixed Bearer token ({id}:agc_xxx)


# ---------------------------------------------------------------------------
# Schemes & helpers
# ---------------------------------------------------------------------------

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="/auth/authorize",
    tokenUrl="/auth/token"
)


def is_api_key_jwt_payload(payload: Optional[dict]) -> bool:
    """Return True when JWT claims represent an API-key JWT token."""
    if not payload:
        return False
    return bool(payload.get("api_key_id"))


def _claim_email(payload: dict) -> Optional[str]:
    """Best-effort email from standard OIDC claims (email, then upn /
    preferred_username when they look like an address)."""
    for k in ("email", "upn", "preferred_username"):
        v = payload.get(k)
        if isinstance(v, str) and "@" in v:
            return v
    return None


def _claim_name(payload: dict) -> Optional[str]:
    """Best-effort display name from standard OIDC claims."""
    for k in ("name", "given_name", "preferred_username"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _validate_aud_for_principal(payload: dict) -> None:
    """Post-decode audience validation for multi-type token paths."""
    principal_type = payload.get("principal_type", "user")
    aud = payload.get("aud")
    if principal_type == "service":
        # Phase C platform mutual JWT: peer services (origin, chorus) calling
        # Mantle with `aud="mantle"`.
        if aud != "mantle":
            raise HTTPException(status_code=401, detail="Invalid token audience for platform service")
    elif principal_type == "server":
        if aud != "agience":
            raise HTTPException(status_code=401, detail="Invalid token audience for server credential")
    elif principal_type == "mcp_client":
        if not aud:
            raise HTTPException(status_code=401, detail="Missing aud in mcp_client token")
    elif principal_type == "delegation":
        # Delegation JWTs have aud=server_client_id (the server they were issued
        # TO).  When a persona server calls Core on behalf of a user, Core
        # accepts these because the JWT is Core-signed and carries sub=user_id
        # + act.sub=server_client_id.  Only require aud to be present.
        if not aud:
            raise HTTPException(status_code=401, detail="Missing aud in delegation token")
    else:
        # External OIDC IdP tokens carry the IdP's own audience (already validated
        # by the OidcVerifier against the configured client id), not AUTHORITY_ISSUER
        # — so don't re-check it here.
        from mantle.services.oidc import get_oidc_verifier
        if get_oidc_verifier().is_trusted(payload.get("iss", "")):
            return
        if aud != config.AUTHORITY_ISSUER:
            raise HTTPException(status_code=401, detail="Invalid token audience")


def _check_grant_permission(grants: List[GrantEntity], action: str, resource_id: str = None) -> bool:
    """Check if any allow-effect grant permits the requested action.

    Deny-effect grants are excluded — callers that need deny semantics
    should use check_access() instead.
    """
    perm_attr = f"can_{action}"
    for grant in grants:
        if grant_is_deny(grant):
            continue
        if not getattr(grant, perm_attr, False):
            continue
        if resource_id and getattr(grant, "resource_id", None) != resource_id:
            continue
        return True
    return False


def _get_end_user_token_payload(token: str) -> dict:
    """Decode user-only JWT, rejecting API-key JWTs."""
    payload = verify_token(token, expected_audience=config.AUTHORITY_ISSUER)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid or malformed token")
    if is_api_key_jwt_payload(payload):
        raise HTTPException(status_code=403, detail="API key token not valid for this endpoint")
    return payload


async def get_end_user_claims(
    token: str = Security(oauth2_scheme)
) -> dict:
    return _get_end_user_token_payload(token)


# ---------------------------------------------------------------------------
# Core resolution
# ---------------------------------------------------------------------------

def resolve_auth(
    token: str,
    store_db: Database,
    request: Optional[Request] = None,
) -> AuthContext:
    """Core auth resolution — usable from both FastAPI deps and ASGI middleware.

    Token dispatch:
    1. Parse optional artifact-id prefix (``{artifact_id}:agc_xxx``).
    2. ``agc_`` prefix → API key path.
    3. JWT (``ey`` prefix) → decode + dispatch by ``principal_type``.
    4. Otherwise → grant key in Bearer slot.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    raw_token = token.strip()
    target_artifact_id: Optional[str] = None

    # --- prefix parsing: {artifact_id}:agc_xxx ---
    if ":" in raw_token and not raw_token.startswith("ey"):
        parts = raw_token.split(":", 1)
        if len(parts) == 2 and parts[1].startswith("agc_"):
            target_artifact_id = parts[0]
            raw_token = parts[1]

    # --- API key path ---
    if raw_token.startswith("agc_"):
        # Pluggable authz backend (Sovereign-Stack): `local` (default) verifies
        # against Mantle's own api_keys + grants in the lattice — no Origin call;
        # `origin` delegates to Origin. Verify + grants come back together.
        from mantle.services.grant_store import get_apikey_backend
        verify_result = get_apikey_backend().verify_api_key(store_db, raw_token)
        if verify_result is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        api_key_entity, grants = verify_result

        return AuthContext(
            principal_id=str(getattr(api_key_entity, "id", "")),
            principal_type="api_key",
            user_id=str(api_key_entity.user_id) if api_key_entity.user_id else None,
            grants=grants,
            api_key_id=str(getattr(api_key_entity, "id", None)) if getattr(api_key_entity, "id", None) else None,
            api_key_entity=api_key_entity,
            target_artifact_id=target_artifact_id,
        )

    # --- JWT path ---
    payload = verify_token(raw_token)
    if payload is None and raw_token.count(".") == 2:
        # A well-formed JWT from an UNKNOWN issuer may be a newly-added issuer
        # artifact the verifier hasn't loaded yet — refresh trust from the store
        # (throttled, fail-open) and retry once.
        try:
            from mantle.services.oidc import get_oidc_verifier
            if get_oidc_verifier().refresh_if_unknown_iss(store_db, raw_token):
                payload = verify_token(raw_token)
        except Exception:
            logger.debug("issuer refresh-on-miss failed", exc_info=True)
    if payload and "sub" in payload:
        _validate_aud_for_principal(payload)

        if is_api_key_jwt_payload(payload):
            raise HTTPException(status_code=403, detail="API-key JWT not accepted; use direct API key")

        jwt_principal_type = payload.get("principal_type", "user")

        if jwt_principal_type == "service":
            # Phase C platform mutual JWT — origin/chorus identifying themselves
            # to Mantle. `iss` carries the service name (verified by the dispatch
            # in `verify_token`).
            return AuthContext(
                principal_id=str(payload.get("iss", "")),
                principal_type="service",
                authority=str(payload.get("iss", "")) or None,
            )

        if jwt_principal_type == "server":
            client_id = str(payload.get("client_id")) if payload.get("client_id") else None
            return AuthContext(
                principal_id=client_id or str(payload.get("sub", "")),
                principal_type="server",
                user_id=None,
                server_id=str(payload.get("server_id")) if payload.get("server_id") else None,
                authority=str(payload.get("authority", "")) or None,
                host_id=str(payload.get("host_id", "")) or None,
            )

        if jwt_principal_type == "mcp_client":
            return AuthContext(
                principal_id=str(payload.get("aud", "")),
                principal_type="mcp_client",
                user_id=str(payload.get("sub")) if payload.get("sub") else None,
            )

        if jwt_principal_type == "delegation":
            # All four identity-chain entities are required:
            # User (sub), Server (act.sub), Authority (iss), Host (host_id)
            d_sub = payload.get("sub")
            d_act_sub = (payload.get("act") or {}).get("sub")
            d_host = payload.get("host_id")
            if not d_sub:
                raise HTTPException(status_code=401, detail="Delegation token missing sub (user)")
            if not d_act_sub:
                raise HTTPException(status_code=401, detail="Delegation token missing act.sub (server)")
            if not d_host:
                raise HTTPException(status_code=401, detail="Delegation token missing host_id")
            return AuthContext(
                principal_id=str(d_sub),
                principal_type="user",
                user_id=str(d_sub),
                actor=str(d_act_sub),
                authority=str(payload.get("iss", "")) or None,
                host_id=str(d_host),
            )

        # Default: user JWT.
        #
        # Multi-tenant: an external IdP's `sub` is unique only within its issuer,
        # so namespace it by (tenant, sub) to keep tenants isolated. Origin's own
        # user tokens already carry a globally-unique Agience UUID as `sub`, so
        # they pass through unchanged (external_user_id returns None for them).
        from mantle.services.oidc import get_oidc_verifier

        _oidc = get_oidc_verifier()
        ext_uid = _oidc.external_user_id(payload)
        if ext_uid:
            return AuthContext(
                principal_id=ext_uid,
                principal_type="user",
                user_id=ext_uid,
                authority=_oidc.tenant_for(payload.get("iss")),  # tenant key
                email=_claim_email(payload),
                name=_claim_name(payload),
            )

        user_id = str(payload.get("sub")) if payload.get("sub") else None
        return AuthContext(
            principal_id=user_id or "",
            principal_type="user",
            user_id=user_id,
            # Capture profile from claims for platform users too (Origin tokens carry
            # email/name). tenant stays None — platform users are the native tenant.
            email=_claim_email(payload),
            name=_claim_name(payload),
        )

    # --- Grant key in Bearer slot ---
    key_grants = db_store.get_active_grants_by_key(store_db, raw_token)
    if key_grants:
        grant = key_grants[0]
        return AuthContext(
            principal_id=getattr(grant, "id", "") or "",
            principal_type="grant_key",
            user_id=None,
            grants=[grant],
            bearer_grant=grant,
        )

    raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

async def get_auth(
    token: str = Security(oauth2_scheme),
    store_db: Database = Depends(get_store_db),
    request: Request = None,
) -> AuthContext:
    """Single auth dependency for all endpoints.

    Also publishes the authenticated caller to the acting-principal contextvar, which
    is what lets the key oracle check WHO is asking (see
    :mod:`services.acting_principal`). Before this, every layer below the router
    re-derived a "principal" from the data it was handling, so the oracle's grant
    check compared two caller-supplied values.

    Set HERE rather than in an ASGI middleware because this dependency is where the
    caller is actually resolved, and ``test_auth_policy_matrix`` already asserts every
    critical route depends on it. A route that does NOT use it publishes nothing, and
    key issuance then fails closed rather than running under a stale or absent
    identity.
    """
    auth = resolve_auth(
        token=token or "",
        store_db=store_db,
        request=request,
    )
    if request is not None and auth.user_id:
        request.state.user_id = auth.user_id

    # Each request runs in its own task with its own context copy, so this cannot
    # leak into another request; it is deliberately not reset, because it must stay
    # readable for the whole handler including anything it awaits.
    if auth.principal_id or auth.user_id:
        set_acting_principal(acting_from_auth(auth))
    return auth


async def get_person(
    auth: AuthContext = Depends(get_auth),
    store_db: Database = Depends(get_store_db),
) -> Person:
    """Load the Person entity for the authenticated user.

    Use as a second dependency alongside ``get_auth`` when a router needs
    Person fields (email, name, preferences, etc.) — not just ``user_id``.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    person = get_user_by_id(db=store_db, id=auth.user_id)
    if not person:
        raise HTTPException(status_code=404, detail="User not found")
    return person


# ---------------------------------------------------------------------------
# Post-auth guards
# ---------------------------------------------------------------------------

def _authority_bootstrap_complete(store_db: Database) -> bool:
    """True once an admin grant exists on the authority collection.

    The bootstrap window (during which the ``platform.operator_id`` config fast-path
    is honored) is OPEN until this is true. Fail OPEN on error (return False) so a
    genuinely-bootstrapping operator isn't locked out by a transient query failure —
    the canonical grant check still gates everyone else.
    """
    try:
        grants = db_store.get_grants_for_collection(
            store_db, get_id(AUTHORITY_COLLECTION_SLUG)
        )
        return any(
            getattr(g, "state", "active") == "active"
            and grant_is_allow(g)
            and (getattr(g, "can_admin", False) or getattr(g, "can_update", False))
            for g in grants
        )
    except Exception:
        logger.debug("authority bootstrap-complete check failed", exc_info=True)
        return False


def require_platform_admin(
    auth: AuthContext, store_db: Database
) -> str:
    """Post-auth guard: require platform admin.

    Merged successor to ``require_admin`` + ``require_operator`` (2026-04-06).
    A platform admin is any user with a write grant on the authority
    collection. During the post-setup / pre-Phase-4 bootstrap window,
    the initial operator recorded in ``platform.operator_id`` settings
    is treated as a platform admin even before the authority collection
    has issued them a grant — this avoids a chicken-and-egg between the
    setup wizard and the grant system.

    Returns the user_id on success, raises HTTP 403 otherwise.
    """
    if not auth.user_id:
        raise HTTPException(status_code=403, detail="Platform admin access required")

    # Bootstrap fast-path: initial operator from setup wizard — CONFINED to the
    # window before any admin grant exists on the authority collection. Once the
    # operator (or the platform) holds a real authority grant, the window closes and
    # even the operator must act through that revocable, authority-rooted grant. This
    # keeps the operator out of the STANDING runtime trust boundary: genesis (writing
    # the manifest) is irreducible, but the config-flag bypass must self-retire once
    # the grant system can serve them. See .dev/features/issuer-merge-bootstrap.md §3b.
    # Operator resolution is sovereign-capable: Mantle setting → env
    # (AGIENCE_OPERATOR_ID, for standalone) → Origin fallback (full platform).
    from mantle.services.operator import resolve_operator_id
    stored_operator_id = resolve_operator_id(store_db)
    if (
        stored_operator_id
        and auth.user_id == stored_operator_id
        and not _authority_bootstrap_complete(store_db)
    ):
        return auth.user_id

    # Canonical check: write grant on the authority collection — via lattice.
    try:
        grants = db_store.get_active_grants_for_principal_resource(
            store_db,
            grantee_id=auth.user_id,
            resource_id=get_id(AUTHORITY_COLLECTION_SLUG),
        )
        if any(grant_is_allow(g) and g.can_update for g in grants):
            return auth.user_id
    except Exception:
        logger.debug("the lattice grant check failed in require_platform_admin", exc_info=True)

    raise HTTPException(status_code=403, detail="Platform admin access required")


# ---------------------------------------------------------------------------
# Access check
# ---------------------------------------------------------------------------

# Map action names to CRUDEASIO grant flag attributes.
_ACTION_FLAG_MAP = {
    "create": "can_create",
    "read": "can_read",
    "update": "can_update",
    "delete": "can_delete",
    "evict": "can_evict",
    "invoke": "can_invoke",
    "add": "can_add",
    "share": "can_share",
    "admin": "can_admin",
}

# ⛔ `_MAX_ORIGIN_DEPTH = 10` REMOVED 2026-07-30. It was a bare claim that grants never inherit
# from more than 10 ancestors, and exceeding it fell through to `raise HTTPException(404)` — a
# SILENT FALSE DENY on an artifact the principal is genuinely granted, indistinguishable from
# "no such artifact". Unlike the other depth caps this walk had NO cycle guard, so the cap was
# doing double duty. Termination is now `visited` (each ancestor once, over a finite graph).
#
# What remains is an OPERATIONAL bound, not a claim about lattice shape: a malformed store that
# returns an endless chain of fresh ids would hang an auth request. It is deliberately far above
# any real chain and it RAISES 500 — the point of the fix is that "you are not granted" (404) and
# "this lattice did not terminate" (500) are different answers, and the old cap collapsed them.
_ORIGIN_WALK_CEILING = 10_000


def _grant_from_check_response(response: dict) -> GrantEntity:
    """Construct a synthetic GrantEntity from a /grants/check response.

    The response carries `{allowed, grant_id, flags, effect}` — enough for
    callers that just need a non-None signal. Fields not in the response
    (granted_by, target_*, etc.) are left as their entity defaults.
    """
    flags = set(response.get("flags") or [])
    return GrantEntity(
        id=response.get("grant_id") or "",
        resource_id="",
        grantee_type="user",
        grantee_id="",
        granted_by="",
        effect=response.get("effect", "allow"),
        can_create="create" in flags,
        can_read="read" in flags,
        can_update="update" in flags,
        can_delete="delete" in flags,
        can_evict="evict" in flags,
        can_invoke="invoke" in flags,
        can_add="add" in flags,
        can_share="share" in flags,
        can_admin="admin" in flags,
        state="active",
    )


# CRUDEASIO action -> the API-key scope action that must also permit it.
# `read` covers the read-ish verbs; everything that mutates maps to `write`
# unless it has a dedicated scope action. An action absent from this map is
# treated as requiring `admin`, i.e. it fails closed for ordinary keys.
_ACTION_TO_SCOPE_ACTION = {
    "read": "read",
    "list": "read",
    "search": "search",
    "invoke": "invoke",
    "create": "create",
    "add": "create",
    "update": "write",
    "delete": "delete",
    "evict": "write",
    "share": "admin",
    "admin": "admin",
}


def _enforce_api_key_ceiling(auth: AuthContext, action: str) -> None:
    """Require the API key's OWN scopes to permit *action*.

    No-op for non-api_key principals. For an api_key principal this is a hard
    ceiling on top of whatever the owning user is granted — the key can only
    ever narrow, never widen.

    Fails closed on an unmapped action and on a missing key entity: if we cannot
    establish what the key is allowed to do, it is allowed to do nothing.
    """
    if auth.principal_type != "api_key":
        return

    key_entity = auth.api_key_entity
    if key_entity is None:
        raise HTTPException(status_code=403, detail="API key scope unavailable")

    scope_action = _ACTION_TO_SCOPE_ACTION.get(action)
    if scope_action is None:
        raise HTTPException(
            status_code=403, detail=f"API keys may not perform '{action}'"
        )

    try:
        permitted = key_entity.has_scope("resource", "*", scope_action)
    except Exception:
        logger.warning("api key scope evaluation failed; denying", exc_info=True)
        raise HTTPException(status_code=403, detail="API key scope check failed")

    if not permitted:
        raise HTTPException(
            status_code=403,
            detail=f"API key is not scoped for '{action}'",
        )


def check_access(
    auth: AuthContext,
    artifact_id: str,
    action: str,
    store_db: Database,
) -> GrantEntity:
    """Verify *auth* has permission to perform *action* on *artifact_id*.

    CRUDEASIO lives in Mantle (the lattice grants collection). Light-cone
    traversal: direct grant first, then walk origin edges upward checking
    propagated grants at each parent. Workspace IS A Collection IS An
    Artifact — all addressed by artifact _key.
    """
    flag_attr = _ACTION_FLAG_MAP.get(action)
    if not flag_attr:
        raise HTTPException(status_code=400, detail=f"Unknown action: {action}")

    try:
        from mantle.db.backend import get_raw_artifact
        doc = get_raw_artifact(store_db, artifact_id)
    except Exception:
        doc = None
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    if not auth.user_id:
        raise HTTPException(status_code=404, detail="Not found")

    # ⛔ AN API KEY USED TO INHERIT ITS OWNER'S FULL AUTHORITY.
    # Grants below resolve against `auth.user_id`, which for an api_key principal
    # is the OWNING USER — and the key's own scopes were never consulted. So a
    # read-only integration key handed to a third party carried the owner's
    # entire CRUDEASIO. The key's scoping was decorative.
    #
    # The key is a CEILING, applied before the owner's grant is even looked up:
    # effective permission = (owner's grant) ∩ (key's scopes). Checked first so a
    # scope-exceeding call cannot be witnessed as "allowed" by the audit path.
    _enforce_api_key_ceiling(auth, action)

    # --- Access-audit "force": witness this authorization decision (allow OR deny) ---
    # Emitted here in the authz layer so auditability is a property of access itself.
    # Wrapped so a failure in the audit path can NEVER deny or break legitimate access.
    _audit_ctx = {
        "via": auth.principal_type,
        "principal_id": auth.principal_id or None,
        "api_key_id": auth.api_key_id,
        "tenant": auth.authority,
    }

    def _witness(result: str, reason: Optional[str] = None) -> None:
        try:
            from mantle.services import audit_service
            ctx = dict(_audit_ctx)
            if reason:
                ctx["reason"] = reason
            audit_service.record_access(
                principal_id=auth.user_id,
                artifact_id=artifact_id,
                action=action,
                result=result,
                context=ctx,
            )
        except Exception:  # auditing must never break access
            logger.debug("access witness failed", exc_info=True)

    def _check_grants(resource_id: str) -> Optional[GrantEntity]:
        grants = db_store.get_active_grants_for_principal_resource(
            store_db, grantee_id=auth.user_id, resource_id=resource_id
        )
        for g in grants:
            if grant_is_deny(g) and getattr(g, flag_attr, False):
                _witness("denied", "deny_grant")
                raise HTTPException(status_code=404, detail="Not found")
        for g in grants:
            if grant_is_allow(g) and getattr(g, flag_attr, False):
                return g
        return None

    # --- Direct grant on the target ---
    direct = _check_grants(artifact_id)
    if direct is not None:
        _witness("allowed")
        return direct

    # --- Grant on the artifact's ROOT ---
    # ⛔ THIS RUNG WAS MISSING. The direct check above uses `artifact_id` (a
    # VERSION id) and the traversal below starts at `root_id` but only ever
    # checks that node's PARENTS — so a grant written on `root_id` itself was
    # never consulted when reading any non-root version. Ownership grants are
    # written against the root, so the owner of a versioned artifact could be
    # refused their own artifact.
    #
    # This is a fail-CLOSED gap (a false denial, not an escalation), which is why
    # it survived: it degrades availability, not confidentiality.
    root_id = doc.get("root_id")
    if root_id and root_id != artifact_id:
        at_root = _check_grants(root_id)
        if at_root is not None:
            _witness("allowed")
            return at_root

    # --- Light-cone: walk origin edges upward ---
    cursor_id = doc.get("root_id") or artifact_id
    visited = {cursor_id}
    while True:
        if len(visited) > _ORIGIN_WALK_CEILING:
            _witness("denied", "origin_chain_did_not_terminate")
            raise HTTPException(
                status_code=500,
                detail="origin chain did not terminate; the lattice is malformed",
            )
        parent = db_store.get_origin_parent(store_db, cursor_id)
        if parent is None:
            break
        parent_id, propagate_mask = parent
        if not parent_id or parent_id in visited:
            break

        if propagate_mask is not None and action not in propagate_mask:
            break

        inherited = _check_grants(parent_id)
        if inherited is not None:
            _witness("allowed")
            return inherited

        visited.add(parent_id)
        cursor_id = parent_id

    _witness("denied", "no_grant")
    raise HTTPException(status_code=404, detail="Not found")


def check_inbound_nonce(request: Request, auth: AuthContext) -> None:
    """Enforce nonce validation for keys with ``requires_nonce=True``.

    Must be called explicitly from any endpoint that should be bot-protected.
    No-ops for principals whose key does not have ``requires_nonce=True``, so
    the same endpoint can serve both authenticated users and nonce-gated callers.

    Raises 403 if the nonce is absent or invalid.
    """
    if auth.principal_type != "api_key":
        return
    key_entity = auth.api_key_entity
    if not key_entity or not getattr(key_entity, "requires_nonce", False):
        return

    from mantle.services.auth_service import verify_nonce as _verify_nonce
    from origin import config

    nonce = request.headers.get("X-Agience-Challenge", "")
    if not nonce:
        raise HTTPException(status_code=403, detail="Nonce required for inbound access")

    artifact_id = auth.target_artifact_id or ""
    key_id = auth.api_key_id or ""

    if not _verify_nonce(
        token=nonce,
        key_id=key_id,
        artifact_id=artifact_id,
        secret=config.INBOUND_NONCE_SECRET,
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired nonce")

