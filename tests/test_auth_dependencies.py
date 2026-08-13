"""Tests for the unified auth resolution layer (resolve_auth / get_auth)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mantle.entities.grant import Grant as GrantEntity
from mantle.services.dependencies import resolve_auth, require_platform_admin, AuthContext
from mantle.config import AUTHORITY_ISSUER


def test_resolve_auth_rejects_retired_api_key_jwt():
    """A JWT from the decommissioned API-key system is rejected on its own terms.

    It carries a `sub`, so the danger is not that it is accepted as an API key but
    that it falls through and is honoured as an ordinary USER token for that subject.
    """
    payload = {
        "sub": "user-123",
        "aud": AUTHORITY_ISSUER,
        "api_key_id": "key-123",
        "scopes": ["resource:*:search"],
    }

    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="jwt-token", store_db=MagicMock())

    assert exc_info.value.status_code == 401
    assert "retired" in exc_info.value.detail


def test_resolve_auth_rejects_retired_agc_bearer_key():
    """A raw `agc_` key names its own retirement rather than reading as malformed."""
    with patch("mantle.services.dependencies.verify_token", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="agc_deadbeef", store_db=MagicMock())

    assert exc_info.value.status_code == 401
    assert "retired" in exc_info.value.detail


def test_resolve_auth_accepts_user_jwt():
    """Standard user JWTs return an AuthContext with principal_type=user."""
    payload = {
        "sub": "user-123",
        "aud": AUTHORITY_ISSUER,
    }

    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        ctx = resolve_auth(token="jwt-token", store_db=MagicMock())

    assert isinstance(ctx, AuthContext)
    assert ctx.principal_type == "user"
    assert ctx.user_id == "user-123"
    assert ctx.principal_id == "user-123"
    assert ctx.grant_key_id is None
    assert ctx.bearer_grant is None


def test_resolve_auth_accepts_grant_key_with_bundle_expanded():
    """A grant key resolves to its ROOT plus every member, each already masked.

    The expansion happens once, here, so no downstream consumer has to know whether
    the credential was a single-resource key or a bundle.
    """
    root = GrantEntity(
        id="bundle-1", resource_id="", grantee_type="grant_key",
        grantee_id="hash-1", granted_by="user-1",
        can_read=True, can_update=True,
    )
    member_rw = GrantEntity(
        id="m-rw", resource_id="col-rw", grantee_type="grant",
        grantee_id="bundle-1", granted_by="user-1",
        can_read=True, can_update=True,
    )
    # Carries delete, which the bundle does not — masking must strip it.
    member_ro = GrantEntity(
        id="m-ro", resource_id="col-ro", grantee_type="grant",
        grantee_id="bundle-1", granted_by="user-1",
        can_read=True, can_delete=True,
    )

    def _by_grantee(_db, grantee_id, grantee_type="user"):
        if grantee_type == "grant_key":
            return [root]
        if grantee_type == "grant" and grantee_id == "bundle-1":
            return [member_rw, member_ro]
        return []

    with patch("mantle.services.dependencies.verify_token", return_value=None), \
         patch("mantle.db.backend.get_active_grants_for_grantee", side_effect=_by_grantee), \
         patch("mantle.db.backend.update_grant", return_value=None):
        ctx = resolve_auth(token="agk_test_key", store_db=MagicMock())

    assert ctx.principal_type == "grant_key"
    assert ctx.user_id is None, "a key must not carry its issuer's identity"
    assert ctx.grant_key_id == "bundle-1"

    by_resource = {g.resource_id: g for g in ctx.grants}
    # The bundle root has no resource of its own, so it contributes nothing directly.
    assert set(by_resource) == {"col-rw", "col-ro"}
    assert by_resource["col-rw"].can_update is True
    assert by_resource["col-ro"].can_delete is False, (
        "a member kept a permission the bundle ceiling does not allow"
    )


def test_resolve_auth_parses_artifact_prefix():
    """`{artifact_id}:agk_xxx` populates target_artifact_id (card keys)."""
    grant = GrantEntity(
        id="key-1", resource_id="ws-1", grantee_type="grant_key",
        grantee_id="hash-1", granted_by="user-1", can_read=True,
    )

    def _by_grantee(_db, grantee_id, grantee_type="user"):
        return [grant] if grantee_type == "grant_key" else []

    with patch("mantle.services.dependencies.verify_token", return_value=None), \
         patch("mantle.db.backend.get_active_grants_for_grantee", side_effect=_by_grantee), \
         patch("mantle.db.backend.update_grant", return_value=None):
        ctx = resolve_auth(token="art_123:agk_test_key", store_db=MagicMock())

    assert ctx.target_artifact_id == "art_123"
    assert ctx.principal_type == "grant_key"


def test_resolve_auth_accepts_grant_key_in_bearer():
    """Grant keys in the Bearer slot return principal_type=grant_key.

    CRUDEASIO lives in Mantle — grant-key lookup goes to the lattice, not Origin.
    """
    grant = GrantEntity(
        id="grant-1",
        resource_id="col-1",
        grantee_type="grant_key",
        grantee_id="hash-1",
        granted_by="user-1",
        can_read=True,
    )

    def _by_grantee(_db, grantee_id, grantee_type="user"):
        return [grant] if grantee_type == "grant_key" else []

    with patch("mantle.services.dependencies.verify_token", return_value=None), \
         patch("mantle.db.backend.get_active_grants_for_grantee", side_effect=_by_grantee), \
         patch("mantle.db.backend.update_grant", return_value=None):
        ctx = resolve_auth(token="agk_grant-key-value", store_db=MagicMock())

    assert ctx.principal_type == "grant_key"
    assert ctx.user_id is None
    assert ctx.bearer_grant is grant
    assert len(ctx.grants) == 1


def test_resolve_auth_rejects_a_server_jwt():
    """A `server` principal names nothing here, so it is a 401 — not a user.

    Mantle registers no servers and issues no server credentials: a server is an
    ordinary `vnd.agience.server+json` artifact, and an artifact is not a principal.
    The rejection has to be by NAME, because the fall-through branch is the user
    branch and it would read `sub` — `server/<client_id>` — as a person.
    """
    payload = {
        "sub": "server/my-server",
        "aud": "agience",
        "principal_type": "server",
        "client_id": "my-server",
        "server_id": "srv-1",
    }

    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="server-jwt", store_db=MagicMock())

    assert exc_info.value.status_code == 401


def test_a_server_jwt_is_never_resolved_as_a_user():
    """The regression the 401 above exists to prevent: with no branch of its own, a
    server token would land in the default user branch and be handed a user identity
    built from `server/<client_id>`. Asserted on a payload carrying the audience the
    user branch accepts, so nothing but the principal-type check can be refusing it."""
    from mantle import config

    payload = {
        "sub": "server/my-server",
        "aud": config.AUTHORITY_ISSUER,
        "principal_type": "server",
        "client_id": "my-server",
    }

    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="server-jwt", store_db=MagicMock())

    assert exc_info.value.status_code == 401


def test_resolve_auth_invalid_token_raises_401():
    """Completely invalid tokens raise 401."""
    with patch("mantle.services.dependencies.verify_token", return_value=None), \
         patch("mantle.db.backend.get_active_grants_for_grantee", return_value=[]):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="garbage", store_db=MagicMock())

    assert exc_info.value.status_code == 401


def test_resolve_auth_missing_token_raises_401():
    """Empty/missing token raises 401."""
    with pytest.raises(HTTPException) as exc_info:
        resolve_auth(token="", store_db=MagicMock())

    assert exc_info.value.status_code == 401


# ---------------------------------------------------------------------------
# Delegation tokens — all four identity-chain entities required
# ---------------------------------------------------------------------------

def _delegation_payload(**overrides):
    """Base valid delegation JWT claims."""
    base = {
        "sub": "user-42",
        "aud": "agience-server-astra",
        "iss": AUTHORITY_ISSUER,
        "act": {"sub": "agience-server-astra"},
        "principal_type": "delegation",
        "host_id": "host-abc",
    }
    base.update(overrides)
    return base


def test_resolve_auth_delegation_all_entities():
    """Delegation tokens with all four entities produce a valid AuthContext."""
    with patch("mantle.services.dependencies.verify_token", return_value=_delegation_payload()):
        ctx = resolve_auth(token="delegation-jwt", store_db=MagicMock())

    assert ctx.principal_type == "user"
    assert ctx.user_id == "user-42"
    assert ctx.actor == "agience-server-astra"
    assert ctx.authority == AUTHORITY_ISSUER
    assert ctx.host_id == "host-abc"


def test_resolve_auth_delegation_missing_sub():
    """Delegation tokens without sub (user) are rejected."""
    payload = _delegation_payload(sub="")
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="delegation-jwt", store_db=MagicMock())
    assert exc_info.value.status_code == 401
    assert "sub" in exc_info.value.detail


def test_resolve_auth_delegation_missing_act_sub():
    """Delegation tokens without act.sub (server) are rejected."""
    payload = _delegation_payload(act={})
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="delegation-jwt", store_db=MagicMock())
    assert exc_info.value.status_code == 401
    assert "act.sub" in exc_info.value.detail


def test_resolve_auth_delegation_missing_act_entirely():
    """Delegation tokens without act claim at all are rejected."""
    payload = _delegation_payload()
    del payload["act"]
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="delegation-jwt", store_db=MagicMock())
    assert exc_info.value.status_code == 401
    assert "act.sub" in exc_info.value.detail


def test_resolve_auth_delegation_missing_host_id():
    """Delegation tokens without host_id are rejected."""
    payload = _delegation_payload(host_id="")
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="delegation-jwt", store_db=MagicMock())
    assert exc_info.value.status_code == 401
    assert "host_id" in exc_info.value.detail


def test_resolve_auth_delegation_missing_aud():
    """Delegation tokens without aud are rejected (via _validate_aud_for_principal)."""
    payload = _delegation_payload(aud="")
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as exc_info:
            resolve_auth(token="delegation-jwt", store_db=MagicMock())
    assert exc_info.value.status_code == 401
    assert "aud" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# Operator confinement — the platform.operator_id fast-path is honored only in the
# bootstrap window (before any admin grant exists on the authority collection), so
# the operator is not a standing config-flag backdoor into platform admin (and thus
# into minting trusted issuers). See .dev/features/issuer-merge-bootstrap.md §3b.
# ---------------------------------------------------------------------------

def _operator_auth(uid="op-1"):
    return AuthContext(principal_id=uid, principal_type="user", user_id=uid)


def _admin_grant():
    return SimpleNamespace(state="active", effect="allow", can_admin=True, can_update=True)


def test_operator_fastpath_open_before_any_authority_grant():
    """Bootstrap window open: operator config flag confers admin while no authority
    grant exists yet (resolves the chicken-and-egg)."""
    with (
        patch("mantle.services.platform_settings_service.settings.get", return_value="op-1"),
        patch("mantle.services.dependencies.get_id", return_value="auth-col"),
        patch("mantle.db.backend.get_grants_for_collection", return_value=[]),
    ):
        assert require_platform_admin(_operator_auth(), MagicMock()) == "op-1"


def test_operator_fastpath_closes_after_authority_admin_grant():
    """Bootstrap window closed: once an admin grant exists on the authority
    collection, the operator flag is inert — an operator without their own grant is
    rejected."""
    with (
        patch("mantle.services.platform_settings_service.settings.get", return_value="op-1"),
        patch("mantle.services.dependencies.get_id", return_value="auth-col"),
        patch("mantle.db.backend.get_grants_for_collection", return_value=[_admin_grant()]),
        patch("mantle.db.backend.get_active_grants_for_principal_resource", return_value=[]),
    ):
        with pytest.raises(HTTPException) as ei:
            require_platform_admin(_operator_auth(), MagicMock())
    assert ei.value.status_code == 403


def test_operator_with_own_grant_passes_after_window_closed():
    """After the window closes, the operator still passes — but via their real,
    revocable authority grant, not the config flag."""
    with (
        patch("mantle.services.platform_settings_service.settings.get", return_value="op-1"),
        patch("mantle.services.dependencies.get_id", return_value="auth-col"),
        patch("mantle.db.backend.get_grants_for_collection", return_value=[_admin_grant()]),
        patch("mantle.db.backend.get_active_grants_for_principal_resource", return_value=[_admin_grant()]),
    ):
        assert require_platform_admin(_operator_auth(), MagicMock()) == "op-1"


# ---------------------------------------------------------------------------
# Thread offload — the request-path seam
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_offload_sync_runs_the_call_off_the_event_loop():
    """Nothing in the store is awaitable, so a sync call made directly from an `async def`
    handler holds the loop for its whole duration and every concurrent request waits behind it.
    The seam is worth nothing if the work stays on the loop thread."""
    import threading

    from mantle.services.dependencies import offload_sync

    loop_thread = threading.get_ident()
    ran_on = await offload_sync(threading.get_ident)
    assert ran_on != loop_thread


@pytest.mark.anyio
async def test_offload_sync_passes_arguments_and_propagates_exceptions():
    """An `HTTPException` raised in the worker still has to become its response."""
    from mantle.services.dependencies import offload_sync

    assert await offload_sync(lambda a, b=0: a + b, 1, b=2) == 3

    def boom():
        raise HTTPException(status_code=418, detail="teapot")

    with pytest.raises(HTTPException) as exc_info:
        await offload_sync(boom)
    assert exc_info.value.status_code == 418


@pytest.mark.anyio
async def test_get_auth_publishes_the_acting_principal_on_the_loop_side():
    """Contextvars are COPIED into the worker, not shared: a value set inside it does not come
    back. So the acting principal must be published after the awaited call returns, or key
    issuance runs under an absent identity."""
    from mantle.services import dependencies as deps
    from mantle.services.acting_principal import current_acting_principal

    auth = AuthContext(principal_id="user-1", principal_type="user", user_id="user-1")
    with patch.object(deps, "resolve_auth", return_value=auth):
        got = await deps.get_auth(token="whatever", store_db=MagicMock(), request=None)

    assert got is auth
    acting = current_acting_principal()
    assert acting is not None and acting.principal_id == "user-1"


@pytest.mark.anyio
async def test_offload_sync_carries_the_acting_principal_INTO_the_worker():
    """The direction the route handlers depend on.

    A value set in the worker does not come back — which is why `get_auth` publishes on the loop
    side — but the context is COPIED IN, and that is load-bearing now that the list/read/search/
    create handlers run their store work through here: the key oracle reads the acting principal
    from inside those calls. If the copy did not happen, every offloaded read would run under an
    absent identity and key derivation would fail closed.
    """
    from mantle.services.acting_principal import (
        ActingPrincipal,
        current_acting_principal,
        set_acting_principal,
    )
    from mantle.services.dependencies import offload_sync

    set_acting_principal(ActingPrincipal(principal_id="user-9", principal_type="user"))
    seen = await offload_sync(current_acting_principal)
    assert seen is not None and seen.principal_id == "user-9"


def test_the_hot_handlers_do_not_call_the_store_from_the_event_loop():
    """`offload_sync` with two call sites was a primitive, not a change.

    `check_access` is the gate on the front of nearly every route and is several queries plus an
    audit write, so an `async def` handler that calls it bare holds the loop for all of it. This
    walks the routers' own ASTs rather than trusting a grep: a bare call inside an `async def` is
    the exact shape being ruled out, and the same call inside a plain `def` helper is fine —
    that helper is what gets handed across whole.
    """
    import ast
    import inspect

    from mantle.routers import artifacts_router, grants_router, system_router

    offenders = []
    for module in (artifacts_router, grants_router, system_router):
        tree = ast.parse(inspect.getsource(module))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name != "check_access":
                    continue
                # `offload_sync(check_access, ...)` passes it as a value, never calls it.
                offenders.append(f"{module.__name__}.{fn.name}")
    assert offenders == [], f"check_access called on the event loop in: {offenders}"


def test_get_store_db_stays_a_sync_generator():
    """FastAPI runs a sync generator dependency in its worker thread pool. Rewriting this as
    `async def` would move the store open — schema creation included, on the first request of a
    process — onto the event loop, where it blocks every other request. The offload belongs on
    the work, not on the handle."""
    import inspect

    from mantle.services.dependencies import get_store_db

    assert inspect.isgeneratorfunction(get_store_db)
    assert not inspect.isasyncgenfunction(get_store_db)
    assert not inspect.iscoroutinefunction(get_store_db)
