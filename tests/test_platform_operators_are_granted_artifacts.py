"""The platform's own operators are granted artifacts: seeded in a collection, owned by the platform
operator, and granted to every account one operator at a time — invoke and read, nothing more.

2026-10-21 (John): tools should carry grants; "we may not want everyone to be able to send email, or
someone else might provide a send email tool, and connect it to their own server/host." The workflow
builder offered every persona tool to everyone.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi import HTTPException

from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, check_access
from mantle.services.seed_provisioning import platform_operators as P

from .test_there_is_no_promotion_state import OTHER, USER, real_node

PLATFORM = "u-platform-operator"
NS = uuid.UUID("22222222-2222-4222-8222-222222222222")


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = real_node(tmp_path, monkeypatch)
    from mantle.services.seed_provisioning import loader, platform_email
    monkeypatch.setattr(loader, "get_instance_namespace", lambda: NS)
    monkeypatch.setattr(platform_email, "_operator_id", lambda db: PLATFORM)
    return d


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


def test_the_operators_are_seeded_once_and_converge(db):
    assert P.ensure_platform_operators(db) and P.ensure_platform_operators(db)
    known = P.ids()
    raw = db.artifacts.get_artifact(known["op-reconcile-accounts"])
    with acting_as(PLATFORM, principal_type="user"):
        from mantle.db.doc_boundary import unseal_content_fields
        raw = dict(raw)
        unseal_content_fields(raw)
    ctx = json.loads(raw["context"])
    assert ctx["title"] == "Reconcile accounts" and raw["created_by"] == PLATFORM
    assert ctx["operator"] == {"server": "lumen", "tool": "reconcile", "arguments": {
        "collection_id": "$.body.params.collection_id?", "inbox_to": "$.body.params.inbox_to?",
        "out_to": "$.body.params.out_to?"}}
    assert ctx["input_schema"]["properties"]["collection_id"]["format"] == "artifact-id"


def test_an_account_may_invoke_and_read_each_operator_and_nothing_more(db):
    P.ensure_platform_operators(db)
    op = P.ids()["op-raise-inbox-item"]
    with pytest.raises(HTTPException):
        with acting_as(USER, principal_type="user"):
            check_access(_auth(USER), op, "invoke", db)
    assert P.grant_to_account(db, USER) == len([o for o in P.OPERATORS if o.get("public")])
    with acting_as(USER, principal_type="user"):
        check_access(_auth(USER), op, "invoke", db)
        check_access(_auth(USER), op, "read", db)
        with pytest.raises(HTTPException):
            check_access(_auth(USER), op, "update", db)


def test_the_migration_grants_every_account_the_store_knows(db):
    P.ensure_platform_operators(db)
    from mantle.db.backend import upsert_user_collection_grant
    upsert_user_collection_grant(db, user_id=OTHER, collection_id="something", granted_by=OTHER,
                                 can_read=True, can_invoke=False, name="x")
    counts = P.grant_to_every_account(db)
    assert counts["granted"] >= 1
    with acting_as(OTHER, principal_type="user"):
        check_access(_auth(OTHER), P.ids()["op-reconcile-accounts"], "invoke", db)


async def test_the_platform_names_its_operators_and_which_the_caller_may_invoke(db):
    from mantle.routers import artifacts_router as artifacts
    P.ensure_platform_operators(db)
    with acting_as(OTHER, principal_type="user"):
        before = (await artifacts.list_platform_operators(auth=_auth(OTHER), store_db=db))["items"]
    assert sorted(i["slug"] for i in before) == sorted(o["slug"] for o in P.OPERATORS)
    assert not any(i["may_invoke"] for i in before)
    P.grant_to_account(db, OTHER)
    with acting_as(OTHER, principal_type="user"):
        got = (await artifacts.list_platform_operators(auth=_auth(OTHER), store_db=db))["items"]
    assert sorted((i["slug"], i["id"]) for i in got) == sorted((o["slug"], P.ids()[o["slug"]]) for o in P.OPERATORS)
    assert all(i["may_invoke"] for i in got)
    assert {(i["server"], i["tool"]) for i in got} >= {("lumen", "reconcile"), ("sage", "browse_collections")}


async def test_a_key_carries_what_its_owner_holds_now(db):
    """A timer's key reaches the operators its workflow runs. Its owner holds them as read and
    invoke, not admin, so a key may carry what its owner holds; and when the owner's grant goes,
    the key's goes with it, so withholding an operator reaches the keys already issued."""
    from mantle.db import backend as store
    from mantle.routers import grants_router
    from mantle.services import grant_key_service as keys

    P.ensure_platform_operators(db)
    P.grant_to_account(db, USER)
    op = P.ids()["op-raise-inbox-item"]
    root, _token = keys.mint(db, user_id=USER, name="Timer: morning")
    with acting_as(USER, principal_type="user"):
        await grants_router.add_bundle_member_endpoint(
            key_id=root.id, body=grants_router.AddBundleMemberRequest(resource_id=op, role="collaborator"),
            auth=_auth(USER), store_db=db)
    (member,) = [g for g in keys.resolve(db, root) if g.resource_id == op]
    assert member.can_invoke and member.can_read and not member.can_update, "clamped to what the owner holds"

    (held,) = store.get_active_grants_for_principal_resource(db, grantee_id=USER, resource_id=op)
    keys.revoke(db, held, revoked_by=PLATFORM)
    assert not [g for g in keys.resolve(db, root) if g.resource_id == op and g.can_invoke]


async def test_a_key_cannot_carry_what_its_owner_does_not_hold(db):
    from mantle.routers import grants_router
    from mantle.services import grant_key_service as keys

    P.ensure_platform_operators(db)
    root, _token = keys.mint(db, user_id=USER, name="Timer: morning")
    with acting_as(USER, principal_type="user"):
        with pytest.raises(HTTPException) as e:
            await grants_router.add_bundle_member_endpoint(
                key_id=root.id, body=grants_router.AddBundleMemberRequest(
                    resource_id=P.ids()["op-raise-inbox-item"], role="collaborator"),
                auth=_auth(USER), store_db=db)
    assert e.value.status_code == 403


def test_the_platform_operator_holds_what_it_provisions_and_can_open_its_titles(db):
    """2026-10-21, the RC's reverse migration: it could not unseal the six titles the operators were
    made with, because their owner held no grant on them; `created_by` confers nothing."""
    P.ensure_platform_operators(db)
    for slug, rid in P.ids().items():
        with acting_as(PLATFORM, principal_type="user"):
            check_access(_auth(PLATFORM), rid, "admin", db)
    from mantle.system.seal_titles import unseal_stored_titles
    counts = unseal_stored_titles(db, dry_run=True)
    assert counts["denied"] == 0, counts


async def test_a_timers_key_is_told_which_operators_it_may_invoke(db):
    """A timer fires on a grant key, which has no user id; Crystal's gate asks this route on its
    behalf for every persona tool a firing reaches, so the key must get an answer, not a 401."""
    from mantle.routers import artifacts_router as artifacts
    from mantle.services import grant_key_service as keys

    P.ensure_platform_operators(db)
    P.grant_to_account(db, USER)
    op = P.ids()["op-raise-inbox-item"]
    root, _token = keys.mint(db, user_id=USER, name="Timer", resource_id=op, role="collaborator")
    key = AuthContext(principal_id=root.id, principal_type="grant_key", user_id=None,
                      grants=keys.resolve(db, root), grant_key_id=root.id, bearer_grant=root)
    got = {i["slug"]: i["may_invoke"] for i in
           (await artifacts.list_platform_operators(auth=key, store_db=db))["items"]}
    assert got["op-raise-inbox-item"] is True and got["op-reconcile-accounts"] is False
