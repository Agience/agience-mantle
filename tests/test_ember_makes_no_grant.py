"""Ember makes no grant (John, 2026-11-02, review 11).

Sharing happens only in the person's own session. An Ember delegation, with whatever scope it
carries, makes no grant: not on one artifact, not on a collection, not to a person, not to the
commons. The person's own token is unaffected: they share as their grants allow.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException

from mantle.routers import grants_router
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, EMBER_ACTOR

from .test_there_is_no_promotion_state import USER, real_node

BOB = str(uuid.uuid4())
COLLECTION = "application/vnd.agience.collection+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _ember(scope="ember:memory ember:read"):
    return AuthContext(principal_id=USER, principal_type="user", user_id=USER, actor=EMBER_ACTOR,
                       scope=scope, delegation_aud=EMBER_ACTOR)


def _soon(days=1):
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _world(db):
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type=COLLECTION, name="Shelf").id
        note = ws.create_workspace_artifact(db, USER, shelf, context=json.dumps({"title": "n"}),
                                            content="a note", content_type="text/markdown").id
    return shelf, note


async def _share(db, auth, **body):
    body = {"grantee_type": "user", "can_read": True, **body}
    with acting_as(USER, principal_type="user"):
        return await grants_router.create_grant_endpoint(
            body=grants_router.CreateGrantRequest(**body), auth=auth, store_db=db)


@pytest.mark.parametrize("scope", ["ember:memory ember:read", "ember:share", ""])
@pytest.mark.parametrize("what", ["note", "shelf", "public"])
async def test_ember_makes_no_grant_of_any_kind(db, scope, what):
    shelf, note = _world(db)
    resource = shelf if what == "shelf" else note
    grantee = "public" if what == "public" else BOB
    with pytest.raises(HTTPException) as refused:
        await _share(db, _ember(scope), resource_id=resource, grantee_id=grantee,
                     expires_at=_soon())
    assert refused.value.status_code == 403


async def test_the_person_s_own_token_is_unaffected(db):
    shelf, _note = _world(db)
    me = AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    g = await _share(db, me, resource_id=shelf, grantee_id=BOB)
    assert g["resource_id"] == shelf
