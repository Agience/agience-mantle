"""The operator definitions the services carry are published into the platform operators'
collection by the platform's system principal, on one grant: create, and nothing more. Create does
not carry read, so the publisher reaches only what its own creations give it. The platform
operators that run a definition name it."""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, check_access
from mantle.services.seed_provisioning import platform_operators as P

from .test_platform_operators_are_granted_artifacts import PLATFORM, db  # noqa: F401

SYSTEM = "u-platform-system"


@pytest.fixture
def system(monkeypatch):
    from mantle.services import peer_signing
    monkeypatch.setattr(peer_signing, "get_system_principal_id", lambda: SYSTEM)
    return SYSTEM


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


def _grants(db, coll):
    from mantle.db.backend import get_active_grants_for_principal_resource as get_grants_for_principal_resource
    return list(get_grants_for_principal_resource(db, SYSTEM, coll))


def test_the_system_principal_is_given_create_on_the_collection_once(db, system):
    assert P.ensure_platform_operators(db) and P.ensure_platform_operators(db)
    coll = P.ids()["platform-operators"]
    (g,) = _grants(db, coll)
    assert g.grantee_type == "user" and g.granted_by == PLATFORM and g.can_create
    assert not any(getattr(g, "can_" + a) for a in
                   ("read", "update", "delete", "evict", "invoke", "add", "share", "admin"))


def test_the_grant_creates_and_does_not_read(db, system):
    P.ensure_platform_operators(db)
    coll = P.ids()["platform-operators"]
    with acting_as(SYSTEM, principal_type="user"):
        check_access(_auth(SYSTEM), coll, "create", db)
        with pytest.raises(HTTPException):
            check_access(_auth(SYSTEM), coll, "read", db)


def test_without_a_system_principal_no_grant_is_made(db, monkeypatch):
    from mantle.services import peer_signing
    monkeypatch.setattr(peer_signing, "get_system_principal_id", lambda: None)
    assert P.ensure_platform_operators(db)
    assert P.ensure_definitions_grant(db, PLATFORM, P.ids()["platform-operators"]) is False


@pytest.mark.parametrize("slug,definition", [("op-answer-from-my-notes", "op.retrieve"),
                                             ("op-use-a-server-tool", "op.mcp.call")])
def test_a_platform_operator_names_the_definition_it_runs(db, system, slug, definition):
    P.ensure_platform_operators(db)
    raw = dict(db.artifacts.get_artifact(P.ids()[slug]))
    with acting_as(PLATFORM, principal_type="user"):
        from mantle.db.doc_boundary import unseal_content_fields
        unseal_content_fields(raw)
    assert json.loads(raw["context"])["definition"] == definition


def test_an_operator_with_no_definition_names_none():
    (op,) = [o for o in P.OPERATORS if o["slug"] == "op-reconcile-accounts"]
    assert "definition" not in P._context(op)
