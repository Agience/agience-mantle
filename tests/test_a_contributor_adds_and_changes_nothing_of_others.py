"""A contributor reads a shared collection and adds their own, and changes nothing anyone else made.

2026-10-22, use case 08: a group keeps ideas in a shared collection so that each keeps its author.
The only role that could add was `editor`, which also rewrote and deleted another member's idea.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from mantle.entities.grant import Grant
from mantle.services import grant_service
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, check_access
from mantle.services import workspace_service as ws

from .test_there_is_no_promotion_state import OTHER, USER, real_node

ANA = "u-ana-contributor"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


def test_the_contributor_role_is_add_and_read():
    assert Grant.permissions_for_role("contributor") == {"can_create": True, "can_read": True}


def test_a_contributor_reads_and_adds_and_cannot_change_anothers_card(db):
    with acting_as(USER, principal_type="user"):
        ideas = ws.create_container(db, USER, content_type="application/vnd.agience.collection+json", name="Ideas").id
        first = ws.create_workspace_artifact(db, USER, ideas, context=json.dumps({"title": "Idea 1"}),
                                             content="Host's idea").id
    for who in (ANA, OTHER):
        with acting_as(USER, principal_type="user"):
            invite, token = grant_service.create_invite(db, user_id=USER, resource_id=ideas, role="contributor")
        grant_service.claim_invite(db, who, token, "")
    with acting_as(OTHER, principal_type="user"):
        check_access(_auth(OTHER), first, "read", db)
        check_access(_auth(OTHER), ideas, "create", db)
        for action in ("update", "delete"):
            with pytest.raises(HTTPException):
                check_access(_auth(OTHER), first, action, db)
        with pytest.raises(HTTPException):
            check_access(_auth(OTHER), ideas, "evict", db)


def _req():
    from starlette.requests import Request
    return Request({"type": "http", "method": "PATCH", "headers": [], "path": "/", "query_string": b""})


async def _add(db, who, container, text):
    from fastapi.responses import Response
    from mantle.routers import artifacts_router as artifacts

    with acting_as(who, principal_type="user"):
        return (await artifacts.create_artifact(
            request=_req(), response=Response(), auth=_auth(who), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=container, content=text,
                                                 context=json.dumps({"title": text}))))["id"]


async def test_what_a_contributor_adds_belongs_to_the_collection_s_owner(db):
    """John, 2026-11-01: a creator who does not already hold the owner's rights on the collection
    gets no creator grant on what they add. A `contributor` reads it, through the collection, and
    cannot change or delete it; the collection's owner can."""

    with acting_as(USER, principal_type="user"):
        ideas = ws.create_container(db, USER, content_type="application/vnd.agience.collection+json", name="Ideas").id
        invite, token = grant_service.create_invite(db, user_id=USER, resource_id=ideas, role="contributor")
    grant_service.claim_invite(db, ANA, token, "")
    mine = await _add(db, ANA, ideas, "Ana's idea")
    from mantle.db import lattice_api as la
    assert la.get_active_grants_for_principal_resource(db, grantee_id=ANA, resource_id=mine) == []
    with acting_as(ANA, principal_type="user"):
        check_access(_auth(ANA), mine, "read", db)
        for action in ("update", "delete"):
            with pytest.raises(HTTPException):
                check_access(_auth(ANA), mine, action, db)
    with acting_as(USER, principal_type="user"):
        check_access(_auth(USER), mine, "update", db)
        check_access(_auth(USER), mine, "delete", db)


async def test_an_inbox_sender_adds_asks_for_no_grant_and_reads_nothing(db):
    """A create-only sender's add succeeds; no grant is asked of the authority; afterwards they
    can read neither the item nor the inbox."""
    from mantle.attenuation import ACTIONS
    from mantle.db import lattice_api as la
    from mantle.db import reach_journal

    with acting_as(USER, principal_type="user"):
        inbox = ws.create_container(db, USER, content_type="application/vnd.agience.collection+json", name="Inbox").id
        la.create_grant(db, Grant(id="g-inbox-ana", resource_id=inbox, grantee_type="user",
                                  grantee_id=ANA, granted_by=USER,
                                  **{f"can_{a}": a == "create" for a in ACTIONS}))
    seen = []
    prior = reach_journal._authority
    reach_journal.install(lambda entry, **kw: seen.append(entry) or None)
    try:
        item = await _add(db, ANA, inbox, "a note for the host")
    finally:
        reach_journal._authority = prior
    assert item and not [e for e in seen if e["kind"] == "grant"]
    with acting_as(ANA, principal_type="user"):
        for target in (item, inbox):
            with pytest.raises(HTTPException):
                check_access(_auth(ANA), target, "read", db)
