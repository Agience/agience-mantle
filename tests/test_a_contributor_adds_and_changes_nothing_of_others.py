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


async def test_the_maker_edits_and_deletes_their_own_card_and_no_one_elses(db):
    """2026-10-22 (John): the maker of a card in someone else's collection edits and deletes it. Its
    owner's grant was minted at creation, but the update path asked for `update` on the container."""
    from fastapi.responses import Response
    from starlette.requests import Request
    from mantle.routers import artifacts_router as artifacts

    def req():
        return Request({"type": "http", "method": "PATCH", "headers": [], "path": "/", "query_string": b""})

    with acting_as(USER, principal_type="user"):
        ideas = ws.create_container(db, USER, content_type="application/vnd.agience.collection+json", name="Ideas").id
        hosts = ws.create_workspace_artifact(db, USER, ideas, context=json.dumps({"title": "Host's"}), content="h").id
        invite, token = grant_service.create_invite(db, user_id=USER, resource_id=ideas, role="contributor")
    grant_service.claim_invite(db, ANA, token, "")
    with acting_as(ANA, principal_type="user"):
        mine = (await artifacts.create_artifact(
            request=req(), response=Response(), auth=_auth(ANA), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=ideas, content="Ana's idea",
                                                 context=json.dumps({"title": "Idea A"}))))["id"]
        patched = await artifacts.update_artifact(
            artifact_id=mine, auth=_auth(ANA), store_db=db,
            body=artifacts.UpdateArtifactRequest(content="Ana's idea, refined"))
        assert (patched.get("content") if isinstance(patched, dict) else patched.content) == "Ana's idea, refined"
        with pytest.raises(HTTPException):
            await artifacts.update_artifact(artifact_id=hosts, auth=_auth(ANA), store_db=db,
                                            body=artifacts.UpdateArtifactRequest(content="Ana's now"))
        check_access(_auth(ANA), mine, "delete", db)
