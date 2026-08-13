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
#   get_end_user_claims() — user-only JWT guard (rejects retired API-key JWTs)
#   check_access()        — verify principal has permission on an artifact
#   _check_grant_permission() — grant permission helper

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

from fastapi import HTTPException, Depends, Security, Request
from fastapi.security import (
    OAuth2AuthorizationCodeBearer,
)
from mantle.db.store import Database

from typing import Generator

from mantle import config
from mantle.services.acting_principal import acting_from_auth, set_acting_principal
from mantle.services.person_service import get_user_by_id  # expects Database
from mantle.services.auth_service import verify_token
from mantle.db import backend as db_store
from mantle.services.bootstrap_types import AUTHORITY_COLLECTION_SLUG
from mantle.services.platform_topology import get_id
from mantle.entities.person import Person
from mantle.attenuation import propagates
from mantle.entities.grant import Grant as GrantEntity, grant_is_allow, grant_is_deny
from mantle.services import grant_key_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB connection (FastAPI dependency)
# ---------------------------------------------------------------------------


def get_store_db() -> Generator[Database, None, None]:
    """The store handle for FastAPI dependency injection — backend-selected (see `db.backend`).

    Yields Mantle's OWN store (`db.lattice_api.LatticeDatabase`, one SQLite file opened once
    per process — THE STANDALONE DB; the same handle startup uses).

    Deliberately a SYNCHRONOUS generator, and it must stay one. FastAPI runs a sync generator
    dependency in its worker thread pool; rewriting it as `async def` would move
    `store_handle()` — which opens the SQLite file and runs `ensure_schema` on the first
    request of a process — onto the event loop, where it would block every other request.
    "Make the dependency async" is the wrong direction here: the offload is `offload_sync`
    below, applied to the work, not to the handle.
    """
    from mantle.db import backend
    yield backend.store_handle()


async def offload_sync(fn, *args, **kwargs):
    """Run a synchronous store call in a worker thread — the request-path offload primitive.

    Nothing in the store is awaitable: SQLite and the content stores are synchronous, so a
    `async def` handler that calls them directly holds the event loop for the whole call and
    every other in-flight request waits behind it, including ones that would have finished in
    microseconds. One slow read makes the process look single-threaded because, for the
    duration, it is.

    Applied, not merely available: `get_auth` / `get_person` here, and in the routers the
    list / read / search / create handlers plus the byte path (see `routers/artifacts_router.py`,
    `grants_router.py`, `system_router.py`). It is NOT applied to every store touch — the rule
    is that a call earns a hop by issuing more than one query or doing file/network/crypto I/O.
    A single indexed seek (`_artifact_exists`, `has_children`, `get_grant_by_id`) costs less than
    the thread hop that would wrap it, so those stay on the loop. Nor does a router whose handlers
    are plain `def` need it: FastAPI already runs a sync handler in its own worker pool, so
    wrapping there would be a hop taken inside a thread.

    Safe for this store, for reasons that are properties of `db/seq.py` rather than
    assumptions about it:

      * `LatticeConn` keeps its `sqlite3.Connection` in a `threading.local`, so a worker thread
        gets its own connection and none is ever used from two threads at once — the thing
        sqlite3 refuses to allow.
      * Writers serialise on a process-wide `RLock` plus `BEGIN IMMEDIATE`, so concurrent
        workers queue on a cheap mutex instead of racing to `SQLITE_BUSY`.
      * A transaction never spans threads. `write()`'s re-entrancy depth and every
        `SeqAllocator`'s in-transaction cache are thread-local, so one whole store call must
        run in one thread — which is exactly what this does. Splitting a transaction across
        threads would break gap-free `_seq` allocation; passing a *whole* call across one does
        not.

    Correctness notes for callers:

      * Contextvars are COPIED into the worker, not shared. A value the callee sets — the
        acting principal, notably — does not come back. Set those in the caller, after the
        awaited call returns, which is what `get_auth` does.
      * Exceptions propagate unchanged, so an `HTTPException` raised inside still becomes its
        response.
      * The pool is bounded (anyio's default limiter), which bounds the number of live
        thread-local SQLite connections too.

    This is single-process and needs no coordination. It is NOT the multi-worker change: the
    event bus is in-process, so running multiple uvicorn workers without a back-plane silently
    breaks the change feed. That is a separate change with its own precondition.
    """
    import anyio.to_thread

    if kwargs:
        return await anyio.to_thread.run_sync(partial(fn, *args, **kwargs))
    return await anyio.to_thread.run_sync(fn, *args)


# ---------------------------------------------------------------------------
# AuthContext
# ---------------------------------------------------------------------------

@dataclass
class AuthContext:
    """Unified auth context returned by ``get_auth()``.

    A single consistent shape for every auth path. Field names follow
    the Unified Artifact API spec.
    """

    #: Who the work is done on behalf of: user_id | grant_id | service name. For every
    #: principal that acts FOR someone — ``delegation``, ``mcp_client`` — this is the
    #: SUBJECT, not the machine; the machine is :attr:`actor`. A grant key is the
    #: exception that proves the rule: it acts for nobody, so it is its own subject.
    principal_id: str = ""
    principal_type: str = "user"                        # "user" | "grant_key" | "service" | "mcp_client"
    user_id: Optional[str] = None                       # present for user, mcp_client, delegation
    #: For a grant key this is the RESOLVED bundle — the root plus every member,
    #: each already narrowed by the root's bits. Consumers read it as a flat list.
    grants: List[GrantEntity] = field(default_factory=list)  # loaded server-side
    grant_key_id: Optional[str] = None                  # id of the root grant, if auth was via grant key
    #: The machine acting FOR the subject, when one is: the acting server for a
    #: delegation (``act.sub``), the OAuth client id for an ``mcp_client`` (``aud``).
    #: Provenance only — never an authorization input, which is why both principal types
    #: can name their actor here without either of them gaining reach by doing so.
    actor: Optional[str] = None
    authority: Optional[str] = None                     # issuer / authority identity (external IdP = tenant)
    host_id: Optional[str] = None                       # host identity (platform instance)
    email: Optional[str] = None                         # profile email from token claims (external IdP login)
    #: Did the issuer vouch for that address? Not the same as merely having one.
    #:
    #: `_claim_email` accepts `email` / `upn` / `preferred_username` — none of which is a claim that
    #: the address was verified. Matching an invite on an unverified address is the classic
    #: "Sign in with X" takeover: any IdP that lets a user type an arbitrary address becomes a way
    #: to claim an invite addressed to someone else. Anything that decides access from the email
    #: must read this too; anything merely displaying it need not.
    email_verified: bool = False
    name: Optional[str] = None                          # profile display name from token claims
    bearer_grant: Optional[GrantEntity] = None           # convenience: grant resolved from Bearer grant key
    target_artifact_id: Optional[str] = None             # artifact scoping from prefixed Bearer token ({id}:agk_xxx)


# ---------------------------------------------------------------------------
# Schemes & helpers
# ---------------------------------------------------------------------------

#: The authorization server, absolute. Mantle serves NEITHER of the paths below — it is a
#: protected RESOURCE, and `/auth/authorize` + `/auth/token` are Origin's endpoints (see
#: `main.py`: "the /auth/authorize authorization endpoint lives in Origin"). Declared against
#: Origin's public URI for exactly that reason: a RELATIVE url here is resolved by Swagger
#: against Mantle's own host, so the Authorize button posts to a 404 on this service.
#:
#: A node with no authority configured names no endpoint at all rather than inventing one, and a
#: deployment fronted by an external IdP authenticates against that IdP's own endpoints — neither
#: is knowable here, and `/.well-known/oauth-protected-resource` (RFC 9728, built from
#: `config.authorization_servers()`) is the machine-readable answer in both cases. This scheme is
#: Swagger affordance only; it is not a second claim about who the issuer is.
_AUTHORITY = (config.AUTHORITY_ISSUER or "").rstrip("/")

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{_AUTHORITY}/auth/authorize" if _AUTHORITY else "",
    tokenUrl=f"{_AUTHORITY}/auth/token" if _AUTHORITY else "",
)


def is_retired_api_key_jwt(payload: Optional[dict]) -> bool:
    """True for a JWT minted against the decommissioned API-key system.

    Such a token cannot be honoured — the keys behind it are gone — but it must be
    rejected on its own terms rather than treated as an ordinary user JWT, whose `sub`
    it would otherwise be read as.
    """
    if not payload:
        return False
    return bool(payload.get("api_key_id"))


def _claim_email_verified(payload: dict) -> bool:
    """Did the issuer assert `email_verified`? Absence means false.

    It is an optional OIDC claim. A provider that never sends it is not saying "verified", it is
    saying nothing, and collapsing the two is how an unverified address becomes an identity. The
    string form is accepted because claims survive JSON round-trips as strings — and `bool("false")`
    is `True`, so a naive cast would read an explicit denial as approval.
    """
    v = payload.get("email_verified")
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return False


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


def _claim_aud_str(payload: dict) -> Optional[str]:
    """``aud`` as a single readable string, for the audit field it is recorded in.

    RFC 7519 allows ``aud`` to be an array, and a naive ``str()`` would render one as
    ``"['a', 'b']"`` — a Python repr in an audit log, and a name that matches no client id
    anyone could search for. Only ever read for provenance, never to decide access, so the
    ambiguous multi-audience case is spelled out rather than resolved.
    """
    aud = payload.get("aud")
    if isinstance(aud, (list, tuple)):
        return ",".join(str(a) for a in aud) or None
    return str(aud) if aud else None


def _validate_aud_for_principal(payload: dict) -> None:
    """Post-decode audience validation for multi-type token paths."""
    principal_type = payload.get("principal_type", "user")
    aud = payload.get("aud")
    if principal_type == "service":
        # Platform mutual JWT: a peer service (origin) calling Mantle with
        # `aud="mantle"`.
        if aud != "mantle":
            raise HTTPException(status_code=401, detail="Invalid token audience for platform service")
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
    if is_retired_api_key_jwt(payload):
        raise HTTPException(status_code=401, detail="API keys have been retired; use a grant key")
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
    1. Parse optional artifact-id prefix (``{artifact_id}:agk_xxx``).
    2. ``agk_`` prefix → grant key: the bearer acts AS a grant (see
       :mod:`services.grant_key_service`).
    3. JWT (``ey`` prefix) → decode + dispatch by ``principal_type``.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    raw_token = token.strip()
    target_artifact_id: Optional[str] = None

    # --- prefix parsing: {artifact_id}:agk_xxx ---
    if ":" in raw_token and not raw_token.startswith("ey"):
        parts = raw_token.split(":", 1)
        if len(parts) == 2 and parts[1].startswith(grant_key_service.KEY_PREFIX):
            target_artifact_id = parts[0]
            raw_token = parts[1]

    # --- Grant key path ---
    #
    # The credential resolves to a grant, and the grant is the whole authorization —
    # there is no scope grammar layered on top of it. A bundle expands here, once, so
    # every downstream consumer sees a flat list of effective grants and none of them
    # needs to know whether the key was a single-resource key or a bundle.
    if raw_token.startswith(grant_key_service.KEY_PREFIX):
        root = grant_key_service.authenticate(store_db, raw_token)
        if root is None:
            raise HTTPException(status_code=401, detail="Invalid grant key")
        grants = grant_key_service.resolve(store_db, root)
        grant_key_service.touch(store_db, root)

        return AuthContext(
            principal_id=root.id or "",
            principal_type="grant_key",
            # A key is not a person. It carries no user_id, so nothing downstream can
            # mistake it for the issuing user and hand it that user's full light cone —
            # `check_access` resolves it through `ledger_identity` instead.
            user_id=None,
            grants=grants,
            grant_key_id=root.id or None,
            bearer_grant=root,
            target_artifact_id=target_artifact_id,
        )

    # A retired credential must say so. Falling through to the JWT branch would report
    # a decommissioned API key as a malformed token, which sends the holder looking in
    # the wrong place.
    if raw_token.startswith("agc_"):
        raise HTTPException(
            status_code=401,
            detail="API keys have been retired; use a grant key (see POST /grants/keys)",
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
        jwt_principal_type = payload.get("principal_type", "user")

        # A `server` names nothing here. Mantle registers no servers, issues no server
        # credentials and keeps no server JWK plane — a server is an ordinary
        # `vnd.agience.server+json` artifact, and an artifact is not a principal. Such a
        # token resolved to a principal that held no grants and no user id, so
        # `check_access` refused it on every route.
        #
        # Rejected BY NAME, and before the audience rule, rather than deleted and left to
        # fall through: the default branch below is the user branch, which would read
        # `sub` — the string `server/<client_id>` — as a person and hand it a user's
        # identity. A principal type this node cannot resolve is a 401, not a user.
        if jwt_principal_type == "server":
            raise HTTPException(
                status_code=401,
                detail="Server principals are not accepted; use a grant key (see POST /grants/keys)",
            )

        _validate_aud_for_principal(payload)

        if is_retired_api_key_jwt(payload):
            raise HTTPException(status_code=401, detail="API keys have been retired; use a grant key")

        if jwt_principal_type == "service":
            # Platform mutual JWT — a peer service (origin) identifying itself to
            # Mantle. `iss` carries the service name (verified by the dispatch in
            # `verify_token`).
            return AuthContext(
                principal_id=str(payload.get("iss", "")),
                principal_type="service",
                authority=str(payload.get("iss", "")) or None,
            )

        if jwt_principal_type == "mcp_client":
            # THE SUBJECT IS THE USER, THE CLIENT IS THE ACTOR — the same shape as
            # `delegation` below, because it is the same situation: a machine calling on a
            # person's behalf, where `sub` names the person and a second claim names the
            # machine. Origin mints exactly that (`auth_router`: `sub` = user id, `aud` =
            # the client), so `aud` is an audience, not an identity to hold grants under.
            #
            # Reading `aud` as `principal_id` split this principal in two. Everything that
            # keys on `user_id` — `check_access`, and `list_visible`'s light cone, which
            # resolves `auth.user_id` as a plain `"user"` — already saw the person and
            # already worked. Only `acting_principal.acting_from_auth` reads `principal_id`,
            # so the key oracle alone resolved a light cone for the CLIENT ID, found no
            # grants under it, and refused every content key. A client could therefore list
            # artifacts and never read one.
            #
            # So this WIDENS NOTHING. It does not hand the client a reach it lacked; it
            # makes key custody agree with the authorization decision two other call sites
            # in the same request had already made against `sub`. The reachable set is
            # unchanged — a request that was allowed is still allowed, a request that was
            # denied is still denied — and what changes is that content under an allowed
            # artifact now decrypts instead of failing closed.
            #
            # `scopes` is deliberately not consulted: Origin records it as "a record of the
            # request and not a per-client entitlement", so narrowing by it would read an
            # unvouched-for client's own ask as authority.
            #
            # `principal_type` stays `"mcp_client"` rather than collapsing to `"user"` as
            # delegation does. `ledger_grantee_type` already maps it onto the ledger's
            # `"user"` grantee vocabulary, so the grant lookup is identical either way —
            # and keeping the name means the audit trail says which KIND of caller acted,
            # and that a `principal_type == "user"` gate (Origin has two) can still tell a
            # machine from a person.
            mcp_sub = payload.get("sub")
            return AuthContext(
                principal_id=str(mcp_sub) if mcp_sub else "",
                principal_type="mcp_client",
                user_id=str(mcp_sub) if mcp_sub else None,
                # `aud` is retained here and nowhere else: WHICH client acted for this user
                # is worth keeping even though it is not the authorization key. `actor` is
                # the field that already means exactly that (`ActingPrincipal.actor`: "NOT
                # consulted for the authorization decision"), `acting_from_auth` carries it
                # into the acting principal, and nothing in Mantle reads it to decide access.
                actor=_claim_aud_str(payload),
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
                email_verified=_claim_email_verified(payload),
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
            email_verified=_claim_email_verified(payload),
            name=_claim_name(payload),
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
    is what lets the key oracle check who is asking (see
    :mod:`services.acting_principal`).

    Set here rather than in an ASGI middleware because this dependency is where the
    caller is actually resolved, and ``test_auth_policy_matrix`` asserts every
    critical route depends on it. A route that does not use it publishes nothing, and
    key issuance then fails closed rather than running under a stale or absent
    identity.

    Resolution runs in a worker thread (see :func:`offload_sync`): it verifies a JWT, and on
    the grant-key path reads and writes the store, none of it awaitable. On the event loop
    that work blocks every concurrent request, and it is on the front of every authenticated
    one. The acting principal is published below, back on the loop, because a contextvar set
    inside a worker thread does not survive the return.
    """
    auth = await offload_sync(
        resolve_auth,
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
    token: str = Security(oauth2_scheme),
) -> Person:
    """Load the Person entity for the authenticated user.

    Use as a second dependency alongside ``get_auth`` when a router needs
    Person fields (email, name, preferences, etc.) — not just ``user_id``.

    ``token`` is declared again here, rather than reached for through ``auth``, because
    :class:`AuthContext` is the RESULT of verification and deliberately carries no credential.
    It is forwarded to Origin as the subject token (see
    :data:`services.person_service.SUBJECT_TOKEN_HEADER`), and this is the one call site where
    doing so is sound without a further check: the person being read is ``auth.user_id``, which
    is the subject ``resolve_auth`` derived from this very token, so "the caller's bearer" and
    "the subject's own token" are the same string by construction rather than by coincidence.

    A grant key is never forwarded. ``resolve_auth`` gives a ``grant_key`` context
    ``user_id=None`` ("a key is not a person"), so the 401 below is reached first — an opaque
    ``agk_`` credential of this node's cannot leave it through this path.
    """
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    person = await offload_sync(get_user_by_id, db=store_db, id=auth.user_id,
                                subject_token=token or None)
    if not person:
        raise HTTPException(status_code=404, detail="User not found")
    return person


# ---------------------------------------------------------------------------
# Post-auth guards
# ---------------------------------------------------------------------------

def _authority_bootstrap_complete(store_db: Database) -> bool:
    """True once a real platform admin exists on the authority collection.

    The bootstrap window (during which the ``platform.operator_id`` config fast-path
    is honored) is OPEN until this is true. Fail OPEN on error (return False) so a
    genuinely-bootstrapping operator isn't locked out by a transient query failure —
    the canonical grant check still gates everyone else.

    Matches :func:`is_platform_admin` on ``can_admin`` deliberately. A weaker condition
    here would let the window close on a grant that confers no admin — leaving the
    operator retired and nobody able to appoint anyone, which is unrecoverable through
    the API.
    """
    try:
        grants = db_store.get_grants_for_collection(
            store_db, get_id(AUTHORITY_COLLECTION_SLUG)
        )
        return any(
            getattr(g, "state", "active") == "active"
            and grant_is_allow(g)
            and getattr(g, "can_admin", False)
            for g in grants
        )
    except Exception:
        logger.debug("authority bootstrap-complete check failed", exc_info=True)
        return False


def is_platform_admin(
    store_db: Database,
    user_id: Optional[str],
    *,
    operator_id: Optional[str] = None,
    authority_id: Optional[str] = None,
    bootstrap_open: Optional[bool] = None,
) -> bool:
    """THE platform-admin predicate — one answer, for gating and for reporting alike.

    Platform admin is an active, allow-effect ``can_admin`` grant on the authority
    collection, plus a bootstrap fast-path for the configured operator while no such
    grant exists yet (someone has to be able to appoint the first admin).

    ``can_admin`` and NOT ``can_update``: write access to the authority collection is
    permission to edit a container, and must not escalate to permission to appoint
    administrators. See ``tests/test_platform_admin_predicate.py``, which pins this.

    The optional arguments let a caller that already resolved these values reuse them —
    ``list_users`` evaluates this once per user and would otherwise re-scan the whole
    grant set every iteration. They are an optimization only; omitted, each is resolved
    here and the answer is identical.
    """
    if not user_id:
        return False

    if operator_id is None:
        from mantle.services.operator import resolve_operator_id
        operator_id = resolve_operator_id(store_db)

    # `operator_id and` is the whole safety of this line, not defensive noise: on a
    # part-provisioned store an unresolved operator and an unidentified caller are both
    # commonly "" or None, and `"" == ""` would make an anonymous caller platform admin
    # exactly when the system is least set up.
    if operator_id and user_id == operator_id:
        if bootstrap_open is None:
            bootstrap_open = not _authority_bootstrap_complete(store_db)
        if bootstrap_open:
            return True

    # `get_id` raises when the platform topology has not been resolved, so it belongs
    # inside the guard: an unresolved slug must deny (and be visible as a 403) rather
    # than escape as a 500 from the most privileged predicate in the service.
    try:
        if authority_id is None:
            authority_id = get_id(AUTHORITY_COLLECTION_SLUG)
        grants = db_store.get_active_grants_for_principal_resource(
            store_db, grantee_id=user_id, resource_id=authority_id,
        )
    except Exception:
        logger.debug("the lattice grant check failed in is_platform_admin", exc_info=True)
        return False
    return any(grant_is_allow(g) and getattr(g, "can_admin", False) for g in grants)


def require_platform_admin(
    auth: AuthContext, store_db: Database
) -> str:
    """Post-auth guard: require platform admin. The raising form of
    :func:`is_platform_admin`, which holds the policy — this adds only the HTTP shape.

    The bootstrap fast-path lives there too: the initial operator recorded in
    ``platform.operator_id`` is admin only while no real admin grant exists, so the
    config-flag bypass retires once the grant system can serve them and even the
    operator then acts through a revocable, authority-rooted grant. Genesis (writing
    the manifest) is irreducible; standing runtime trust is not. See
    .dev/features/issuer-merge-bootstrap.md §3b. Operator resolution is
    sovereign-capable: Mantle setting → env (AGIENCE_OPERATOR_ID, standalone) →
    Origin fallback.

    Returns the user_id on success, raises HTTP 403 otherwise.
    """
    if not auth.user_id:
        raise HTTPException(status_code=403, detail="Platform admin access required")
    if is_platform_admin(store_db, auth.user_id):
        return auth.user_id
    raise HTTPException(status_code=403, detail="Platform admin access required")


# ---------------------------------------------------------------------------
# Access check
# ---------------------------------------------------------------------------

# Map action names to CRUDEASIO grant flag attributes.
#: Re-exported from the entity, which owns it. Kept as a module name because the
#: resolver and several routers import it from here.
_ACTION_FLAG_MAP = GrantEntity.ACTION_FLAGS

#
# This is an operational bound, not a claim about lattice shape: a malformed store that
# returns an endless chain of fresh ids would hang an auth request. It is deliberately far above
# any real chain, and exceeding it raises 500 rather than 404 — "you are not granted" and
# "this lattice did not terminate" are different answers.
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

    Two kinds of principal reach this, and they differ only in where their grants
    come from: a user's are looked up by ``user_id``, a grant key's were already
    resolved (and narrowed by the bundle ceiling) at authentication. Everything after
    that — deny-first, root, light-cone walk — is identical, which is the point: a
    bearer key gets no separate, weaker enforcement path.
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

    is_key = auth.principal_type == "grant_key"
    # A principal with neither identity holds nothing. Nonexistence and denial return
    # the same 404 throughout this function — no existence oracle.
    if not auth.user_id and not is_key:
        raise HTTPException(status_code=404, detail="Not found")
    if is_key and not auth.grants:
        raise HTTPException(status_code=404, detail="Not found")

    # --- Access-audit "force": witness this authorization decision (allow or deny) ---
    # Emitted here in the authz layer so auditability is a property of access itself.
    # Wrapped so a failure in the audit path can NEVER deny or break legitimate access.
    _audit_ctx = {
        "via": auth.principal_type,
        "principal_id": auth.principal_id or None,
        "grant_key_id": auth.grant_key_id,
        "tenant": auth.authority,
        # WHICH machine acted for this subject, when one did — the OAuth client id for an
        # `mcp_client`, the acting server for a delegation. `principal_id` above records
        # only whose authority was used, and for both of those principal types that is a
        # person who was not the one making the call. Recorded, never consulted: the
        # decision below is made against the subject's grants alone.
        "actor": auth.actor,
    }

    def _witness(result: str, reason: Optional[str] = None) -> None:
        try:
            from mantle.services import audit_service
            ctx = dict(_audit_ctx)
            if reason:
                ctx["reason"] = reason
            audit_service.record_access(
                # A key acts on its own behalf, not the issuer's. Attributing its
                # actions to the issuing user would make the audit log say a person
                # did something a detached credential did.
                principal_id=auth.user_id or auth.principal_id,
                artifact_id=artifact_id,
                action=action,
                result=result,
                context=ctx,
            )
        except Exception:  # auditing must never break access
            logger.debug("access witness failed", exc_info=True)

    def _check_grants(resource_id: str) -> Optional[GrantEntity]:
        if is_key:
            # Already resolved and masked at authentication — re-querying by grantee
            # would find the members UNMASKED and hand back more than the bundle allows.
            grants = [g for g in auth.grants if g.resource_id == resource_id]
        else:
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

    # --- Grant on the artifact's root ---
    #
    # A false denial here degrades availability, not confidentiality — the gap
    # fails closed rather than open.
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

        # The light-cone prune, through the one attenuation operator rather than a second
        # copy of it. This read `propagate_mask is not None and action not in propagate_mask`
        # — the same rule `lattice_api.list_origin_descendants` spells out for the DOWNWARD
        # BFS, written out a second time for the upward walk. `propagates` decodes the column
        # with `Mask` and asks `allows`; the decoder is proved bit-for-bit equal to the old
        # expression over every known column shape × every action in
        # `tests/test_attenuation_algebra.py`, so no stored edge changes meaning. `action` is
        # already known to be a CRUDEASIO verb (the `_ACTION_FLAG_MAP` guard at the top of
        # this function), which is the only input on which the two could differ, and there
        # the new one is the closed direction.
        if not propagates(propagate_mask, action):
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
    if auth.principal_type != "grant_key":
        return
    root = auth.bearer_grant
    if not root or not getattr(root, "requires_nonce", False):
        return

    from mantle.services.auth_service import verify_nonce as _verify_nonce
    from mantle import config

    nonce = request.headers.get("X-Agience-Challenge", "")
    if not nonce:
        raise HTTPException(status_code=403, detail="Nonce required for inbound access")

    artifact_id = auth.target_artifact_id or ""
    key_id = auth.grant_key_id or ""

    if not _verify_nonce(
        token=nonce,
        key_id=key_id,
        artifact_id=artifact_id,
        secret=config.INBOUND_NONCE_SECRET,
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired nonce")

