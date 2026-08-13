"""§2.9 authorization: the grant-key ceiling, bundle masking, deny symmetry.

Three invariants this file pins:

1. A grant-key principal is authorized against the grants the KEY holds, never
   against its issuer's — a leaked key must not widen to the person who minted it.
2. A bundle grant is a ceiling: a member is only ever as strong as the bundle
   carrying it, so clearing a bit on the bundle clears it across every member.
3. `Grant.effect` matching is symmetric: allow and deny are each matched
   positively, so no spelling of one can be read as the other.

Positive controls accompany every rejection assertion.

Invariants 1 and 2 protect the same property: a bearer credential narrows and never
widens. It is expressed directly in CRUDEASIO bits, not in a second permission grammar
layered on top of them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

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


# ---------------------------------------------------------------------------
# 3. Deny/allow symmetry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effect", ["DENY", "Deny", " deny ", "deny"])
def test_casing_variants_of_deny_are_denies(effect):
    """Any spelling of deny must deny, never silently allow."""
    g = SimpleNamespace(effect=effect)

    assert grant_is_deny(g) is True, f"{effect!r} must be treated as a deny"
    assert grant_is_allow(g) is False, (
        f"{effect!r} counted as an ALLOW — a deny grant doing the exact opposite "
        "of what it says is worse than no deny grant at all"
    )


@pytest.mark.parametrize("effect", ["allow", "ALLOW", " Allow "])
def test_casing_variants_of_allow_are_allows(effect):
    """Positive control: real allows must still allow."""
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
    """Grant-like objects reach enforcement from several producers: the enforcement path receives
    AQL row shims and test doubles, not only `Grant` entities, so the predicates must work on any
    duck-typed object with an `effect` attribute, not just the `Grant` class.
    """
    assert grant_is_deny(SimpleNamespace(effect="deny")) is True
    assert grant_is_allow(SimpleNamespace(effect="allow")) is True
    assert grant_is_allow(object()) is False  # no `effect` attribute at all


# ---------------------------------------------------------------------------
# 1. A grant key authorizes as ITSELF
# ---------------------------------------------------------------------------


def _key_auth(grants):
    """An AuthContext as `resolve_auth` builds one for a grant key: no user_id."""
    from mantle.services.dependencies import AuthContext

    return AuthContext(
        principal_id="key-grant-1",
        principal_type="grant_key",
        user_id=None,
        grants=grants,
        grant_key_id="key-grant-1",
        bearer_grant=grants[0] if grants else None,
    )


def _readable_grant(resource_id="res-1", **flags):
    bits = {"can_read": True}
    bits.update(flags)
    return Grant(resource_id, Grant.GRANTEE_GRANT_KEY, "hash", "owner", **bits)


def _check(auth, action, grants_for_resource):
    """Run check_access against a stub store in which `res-1` exists."""
    from unittest.mock import MagicMock, patch

    from mantle.services import dependencies

    db = MagicMock()
    with (
        patch("mantle.db.backend.get_raw_artifact",
              return_value={"id": "res-1", "root_id": "res-1"}),
        patch("mantle.db.backend.get_active_grants_for_principal_resource",
              return_value=grants_for_resource),
        patch("mantle.db.backend.get_origin_parent", return_value=None),
    ):
        return dependencies.check_access(auth, "res-1", action, db)


def test_readonly_key_cannot_perform_writes():
    """A read-only key must not reach a write, even where its issuer could."""
    auth = _key_auth([_readable_grant()])
    # The issuer holds everything on this resource. Were the key authorized as the
    # user, every one of these would succeed.
    issuer_grants = [Grant("res-1", Grant.GRANTEE_USER, "owner", "owner",
                           **{f: True for f in Grant.PERMISSION_FLAGS})]

    for action in ("update", "delete", "create", "admin", "share"):
        with pytest.raises(HTTPException) as exc:
            _check(auth, action, issuer_grants)
        assert exc.value.status_code == 404, (
            f"a read-only grant key was permitted to '{action}' — it authorized as "
            "its issuing user instead of as itself"
        )


def test_readonly_key_can_still_read_positive_control():
    """Positive control: without this, the test above would pass if all were denied."""
    auth = _key_auth([_readable_grant()])
    assert _check(auth, "read", []) is not None


def test_key_is_permitted_exactly_its_own_bits():
    auth = _key_auth([_readable_grant(can_update=True)])

    assert _check(auth, "read", []) is not None
    assert _check(auth, "update", []) is not None
    with pytest.raises(HTTPException):
        _check(auth, "delete", [])


def test_key_carrying_no_grants_reaches_nothing():
    """If we cannot establish what the key may do, it may do nothing."""
    with pytest.raises(HTTPException) as exc:
        _check(_key_auth([]), "read", [])
    assert exc.value.status_code == 404


def test_key_cannot_reach_a_resource_it_does_not_carry():
    """A key readable on one resource must not read another."""
    auth = _key_auth([_readable_grant(resource_id="other-resource")])
    with pytest.raises(HTTPException) as exc:
        _check(auth, "read", [])
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# 2. The bundle ceiling
# ---------------------------------------------------------------------------


def test_bundle_ceiling_narrows_its_members():
    """A member cannot exceed the bundle carrying it."""
    member = Grant("res-1", Grant.GRANTEE_GRANT, "bundle-1", "owner",
                   can_read=True, can_update=True, can_delete=True)
    read_only_bundle = Grant("", Grant.GRANTEE_GRANT_KEY, "hash", "owner", can_read=True)

    masked = member.masked_by(read_only_bundle)

    assert masked.can_read is True, "the ceiling must not remove what it allows"
    assert masked.can_update is False, (
        "a member kept write access through a read-only bundle — narrowing the "
        "bundle would then narrow nothing"
    )
    assert masked.can_delete is False


def test_masking_never_widens():
    """A permissive bundle must not add bits the member never had."""
    member = Grant("res-1", Grant.GRANTEE_GRANT, "bundle-1", "owner", can_read=True)
    open_bundle = Grant("", Grant.GRANTEE_GRANT_KEY, "hash", "owner",
                        **{f: True for f in Grant.PERMISSION_FLAGS})

    masked = member.masked_by(open_bundle)

    assert masked.can_read is True
    for flag in ("can_update", "can_delete", "can_admin", "can_share"):
        assert getattr(masked, flag) is False, f"{flag} was granted by the ceiling"


def test_masking_does_not_mutate_the_stored_member():
    """The ceiling is a property of the presented credential, not of the record."""
    member = Grant("res-1", Grant.GRANTEE_GRANT, "bundle-1", "owner",
                   can_read=True, can_update=True)
    member.masked_by(Grant("", Grant.GRANTEE_GRANT_KEY, "h", "owner", can_read=True))

    assert member.can_update is True, "masking wrote its narrowing back into the member"


def test_deny_bundle_makes_members_deny():
    """A deny ceiling must not resolve to a permissive member."""
    member = Grant("res-1", Grant.GRANTEE_GRANT, "bundle-1", "owner", can_read=True)
    deny_bundle = Grant("", Grant.GRANTEE_GRANT_KEY, "h", "owner",
                        effect="deny", can_read=True)

    assert member.masked_by(deny_bundle).is_deny() is True


# ---------------------------------------------------------------------------
# get_workspace authorizes the VERB, not just read
# ---------------------------------------------------------------------------


def test_read_only_grant_cannot_update_or_delete_a_workspace():
    """`get_workspace` is the only authorization in front of update_workspace / delete_workspace
    / update_workspace_context / binding / rotate-key, so it must check the verb requested rather
    than only `can_read` — otherwise read access would confer the right to rewrite or destroy the
    workspace.
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
    """An explicit deny grant must block access even when a matching `can_read` grant exists —
    the check must honor `effect`, not only the read flag."""
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
    """No call site may silently inherit the default `read`. Every mutating entry point must name
    its own verb explicitly, so a new one cannot quietly pick up read-level authorization.
    """
    import ast
    import inspect

    from mantle.services import workspace_service

    tree = ast.parse(inspect.getsource(workspace_service))
    WRITE_FNS = {
        "update_workspace", "delete_workspace", "update_workspace_context",
        "create_workspace_artifact", "create_workspace_artifacts_bulk",
        "update_artifact", "revert_artifact", "add_artifact_to_workspace",
        "move_workspace_artifact",
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
    """A grant written on root_id must authorize its versions. The direct check uses the version
    id, and the light-cone starting at root_id only walks its parents, so root_id itself must also
    be checked explicitly — otherwise a grant on root_id would not authorize the versions it is
    meant to cover.
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
        principal_type="user", user_id="u1", grants=[],
        principal_id="u1", grant_key_id=None, authority=None,
    )

    with (
        patch("mantle.db.backend.get_active_grants_for_principal_resource", side_effect=_grants),
        patch("mantle.db.backend.get_origin_parent", return_value=None),
    ):
        result = dependencies.check_access(auth, "v2", "read", db)

    assert result is grant, "a grant on root_id did not authorize its own version"
