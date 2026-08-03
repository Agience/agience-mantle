"""§2.9 authorization: API-key ceiling, resource-filter breadth, deny symmetry.

Three live-fire defects on a running Mantle:

1. **API-key principals authorized as their owning USER.** `check_access`
   resolved grants against `auth.user_id` and never consulted the key's own
   scopes, so a read-only integration key carried the owner's full CRUDEASIO.
2. **`resource_filters: {}` meant "access everything"** — broader than the
   documented default, and settable from one request body.
3. **`Grant.effect` enforcement was asymmetric** — `== "deny"` to detect a deny
   but `!= "deny"` to detect an allow, so `"DENY"` counted as an ALLOW.

Positive controls accompany every rejection assertion (working rule 4).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.entities.api_key import APIKey
from mantle.entities.grant import Grant

# Resolved lazily, not imported at module scope. A module-level import raises
# ImportError against a tree without these symbols, collapsing every test in the
# file into one collection error — which proves nothing about behaviour. This way
# each test fails on its own assertion.


def _sym(module, name, why):
    import importlib

    fn = getattr(importlib.import_module(module), name, None)
    if fn is None:
        pytest.fail(f"{module}.{name} is missing — {why}")
    return fn


def grant_is_allow(g):
    return _sym(
        "mantle.entities.grant", "grant_is_allow",
        "allow is matched as `!= 'deny'`, so any unrecognized effect counts as an ALLOW",
    )(g)


def grant_is_deny(g):
    return _sym("mantle.entities.grant", "grant_is_deny", "deny detection is an unnormalized compare")(g)


def _enforce_api_key_ceiling(auth, action):
    return _sym(
        "mantle.services.dependencies", "_enforce_api_key_ceiling",
        "API keys authorize as their owning user and inherit full CRUDEASIO",
    )(auth, action)


# ---------------------------------------------------------------------------
# 3. Deny/allow symmetry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effect", ["DENY", "Deny", " deny ", "deny"])
def test_casing_variants_of_deny_are_denies(effect):
    """THE REGRESSION: any spelling of deny must deny, never silently allow."""
    g = SimpleNamespace(effect=effect)

    assert grant_is_deny(g) is True, f"{effect!r} must be treated as a deny"
    assert grant_is_allow(g) is False, (
        f"{effect!r} counted as an ALLOW — a deny grant doing the exact opposite "
        "of what it says is worse than no deny grant at all"
    )


@pytest.mark.parametrize("effect", ["allow", "ALLOW", " Allow "])
def test_casing_variants_of_allow_are_allows(effect):
    """POSITIVE CONTROL: real allows must still allow."""
    g = SimpleNamespace(effect=effect)

    assert grant_is_allow(g) is True
    assert grant_is_deny(g) is False


@pytest.mark.parametrize("effect", ["", None, "maybe", "permit", "0"])
def test_unknown_effects_confer_nothing(effect):
    """An unrecognized effect is neither allow nor deny — it must fail closed.

    This is why `is_allow` is positive matching rather than `not is_deny()`.
    """
    g = SimpleNamespace(effect=effect)

    assert grant_is_allow(g) is False, f"{effect!r} must not confer access"
    assert grant_is_deny(g) is False


def test_grant_entity_normalizes_effect_on_construction():
    assert Grant("r", "user", "u", "granter", effect="DENY").effect == "deny"
    assert Grant("r", "user", "u", "granter", effect=" Allow ").effect == "allow"
    # Empty falls back to the documented default rather than an unusable value.
    assert Grant("r", "user", "u", "granter", effect="").effect == "allow"


def test_effect_predicates_work_on_duck_typed_grants():
    """Grant-like objects reach enforcement from several producers.

    Coupling authorization to one class is how the first version of this fix
    broke five existing tests: the enforcement path receives AQL row shims and
    test doubles, not only `Grant` entities.
    """
    assert grant_is_deny(SimpleNamespace(effect="deny")) is True
    assert grant_is_allow(SimpleNamespace(effect="allow")) is True
    assert grant_is_allow(object()) is False  # no `effect` attribute at all


# ---------------------------------------------------------------------------
# 1. The API-key ceiling
# ---------------------------------------------------------------------------


def _auth_for_key(scopes):
    return SimpleNamespace(
        principal_type="api_key",
        api_key_entity=APIKey(id="k1", user_id="owner", scopes=scopes),
    )


def test_readonly_key_cannot_perform_writes():
    """THE REGRESSION: a read-only key must not inherit the owner's write access."""
    auth = _auth_for_key(["resource:*:read"])

    for action in ("update", "delete", "create", "admin", "share"):
        with pytest.raises(HTTPException) as exc:
            _enforce_api_key_ceiling(auth, action)
        assert exc.value.status_code == 403, (
            f"a read-only API key was permitted to '{action}' — it inherits the "
            "owning user's full CRUDEASIO"
        )


def test_readonly_key_can_still_read_positive_control():
    """POSITIVE CONTROL: without this, the test above passes if EVERYTHING is denied."""
    auth = _auth_for_key(["resource:*:read"])

    _enforce_api_key_ceiling(auth, "read")  # must not raise


def test_scoped_key_is_permitted_exactly_its_scopes():
    auth = _auth_for_key(["resource:*:read", "resource:*:write"])

    _enforce_api_key_ceiling(auth, "read")
    _enforce_api_key_ceiling(auth, "update")  # 'update' maps to the write scope

    with pytest.raises(HTTPException):
        _enforce_api_key_ceiling(auth, "delete")


def test_non_api_key_principals_are_unaffected():
    """The ceiling applies only to API keys; users authorize purely on grants."""
    for principal in ("user", "server", "mcp_client", "delegation"):
        _enforce_api_key_ceiling(
            SimpleNamespace(principal_type=principal, api_key_entity=None), "delete"
        )  # must not raise


def test_missing_key_entity_fails_closed():
    """If we cannot establish what the key may do, it may do nothing."""
    auth = SimpleNamespace(principal_type="api_key", api_key_entity=None)

    with pytest.raises(HTTPException) as exc:
        _enforce_api_key_ceiling(auth, "read")
    assert exc.value.status_code == 403


def test_unmapped_action_fails_closed():
    auth = _auth_for_key(["resource:*:read"])

    with pytest.raises(HTTPException):
        _enforce_api_key_ceiling(auth, "some_new_verb")


# ---------------------------------------------------------------------------
# 2. resource_filters breadth
# ---------------------------------------------------------------------------


def test_empty_filters_are_not_broader_than_the_default():
    """THE REGRESSION: `{}` used to return True for EVERY resource type."""
    key = APIKey(id="k1", user_id="u", resource_filters={})

    assert key.can_access_resource("workspaces") is True, "default reach preserved"
    assert key.can_access_resource("collections") is True, "default reach preserved"
    assert key.can_access_resource("tools") is False, (
        "an empty filter map granted access to every resource type — broader "
        "than the documented default, from one request body"
    )
    assert key.can_access_resource("secrets") is False


def test_explicit_filters_still_apply_positive_control():
    key = APIKey(id="k1", user_id="u", resource_filters={"collections": ["c1"]})

    assert key.can_access_resource("collections", "c1") is True
    assert key.can_access_resource("collections", "c2") is False
    assert key.can_access_resource("workspaces") is False


def test_router_rejects_explicitly_empty_resource_filters():
    from mantle.routers.api_keys_router import _reject_empty_resource_filters

    with pytest.raises(HTTPException) as exc:
        _reject_empty_resource_filters({})
    assert exc.value.status_code == 400

    # POSITIVE CONTROLS: omitted (None) and populated maps are both fine.
    _reject_empty_resource_filters(None)
    _reject_empty_resource_filters({"workspaces": "*"})


def test_api_key_router_handlers_have_no_undefined_payload_refs():
    """Regression on a bug introduced while writing this fix.

    `_reject_empty_resource_filters` was first inserted into `get_api_key`, which
    takes no `payload` — a guaranteed NameError on every GET /api-keys/{id}. The
    suite did not catch it because nothing covers that handler, so the shape is
    asserted directly here.
    """
    import ast
    import inspect

    from mantle.routers import api_keys_router

    tree = ast.parse(inspect.getsource(api_keys_router))
    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        used = {
            n.id for n in ast.walk(fn)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        if "payload" in used and "payload" not in params:
            offenders.append(fn.name)

    assert not offenders, f"handlers reference an undefined `payload`: {offenders}"


# ---------------------------------------------------------------------------
# get_workspace authorizes the VERB, not just read
# ---------------------------------------------------------------------------


def test_read_only_grant_cannot_update_or_delete_a_workspace():
    """THE REGRESSION: get_workspace authorized everything on `can_read`.

    It is the only authorization in front of update_workspace / delete_workspace
    / update_workspace_context / binding / rotate-key, so read access conferred
    the right to rewrite or destroy the workspace.
    """
    from unittest.mock import MagicMock, patch

    from mantle.services import workspace_service

    read_only = MagicMock(can_read=True, can_update=False, can_delete=False, effect="allow")

    with (
        patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=MagicMock()),
        patch(
            "mantle.services.workspace_service.store.get_active_grants_for_principal_resource",
            return_value=[read_only],
        ),
    ):
        # POSITIVE CONTROL: reading still works.
        workspace_service.get_workspace(MagicMock(), "user-1", "ws-1", required="read")

        for verb in ("update", "delete"):
            with pytest.raises(HTTPException) as exc:
                workspace_service.get_workspace(MagicMock(), "user-1", "ws-1", required=verb)
            assert exc.value.status_code == 404, (
                f"a read-only grant was allowed to '{verb}' the workspace"
            )


def test_deny_grant_blocks_workspace_access():
    """The old check was `any(g.can_read ...)` — it ignored `effect` entirely,
    so an explicit deny did nothing here."""
    from unittest.mock import MagicMock, patch

    from mantle.services import workspace_service

    deny = MagicMock(can_read=True, effect="deny")

    with (
        patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=MagicMock()),
        patch(
            "mantle.services.workspace_service.store.get_active_grants_for_principal_resource",
            return_value=[deny],
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            workspace_service.get_workspace(MagicMock(), "user-1", "ws-1", required="read")
        assert exc.value.status_code == 404, "a deny grant did not block access"


def test_every_workspace_write_path_declares_its_verb():
    """No call site may silently inherit the default `read`.

    The defect was that update/delete authorized on read. This asserts the
    mutating entry points each name their own verb, so a new one cannot quietly
    pick up read-level authorization.
    """
    import ast
    import inspect

    from mantle.services import workspace_service

    tree = ast.parse(inspect.getsource(workspace_service))
    WRITE_FNS = {
        "update_workspace", "delete_workspace", "update_workspace_context",
        "create_workspace_artifact", "create_workspace_artifacts_bulk",
        "update_artifact", "revert_artifact", "add_artifact_to_workspace",
        "move_workspace_artifact", "commit_workspace_to_collections",
    }

    offenders = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        if fn.name not in WRITE_FNS:
            continue
        for call in [c for c in ast.walk(fn) if isinstance(c, ast.Call)]:
            func = call.func
            if isinstance(func, ast.Name) and func.id == "get_workspace":
                if not any(kw.arg == "required" for kw in call.keywords):
                    offenders.append(fn.name)

    assert not offenders, (
        f"these write paths call get_workspace without `required=`, so they "
        f"authorize on read: {sorted(set(offenders))}"
    )


def test_grant_on_root_id_is_honoured_for_a_version():
    """A grant written on root_id must authorize its versions.

    The direct check uses the VERSION id and the light-cone starts at root_id but
    only checks its PARENTS — so a grant on root_id itself was skipped, and the
    owner of a versioned artifact could be refused their own artifact. Fail-closed
    (a false denial), which is why it went unnoticed.
    """
    from unittest.mock import MagicMock, patch

    from mantle.services import dependencies

    grant = MagicMock(can_read=True, effect="allow")
    db = MagicMock()
    db.artifacts.get_artifact.side_effect = (
        lambda aid: {"id": "v2", "root_id": "r1"} if aid == "v2" else None)

    def _grants(_db, grantee_id, resource_id):
        return [grant] if resource_id == "r1" else []

    auth = MagicMock(
        principal_type="user", user_id="u1", api_key_entity=None,
        principal_id="u1", api_key_id=None, authority=None,
    )

    with (
        patch("mantle.db.backend.get_active_grants_for_principal_resource", side_effect=_grants),
        patch("mantle.db.backend.get_origin_parent", return_value=None),
    ):
        result = dependencies.check_access(auth, "v2", "read", db)

    assert result is grant, "a grant on root_id did not authorize its own version"
