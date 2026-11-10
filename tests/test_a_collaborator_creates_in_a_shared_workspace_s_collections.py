"""A person granted create on a shared workspace creates in the collections it holds.

BUG-015, 2026-11-10: a Collaborator on a shared workspace was refused (404 "Workspace not found")
creating in the workspace's Backlog collection: the create asked for a grant on the collection
itself and never walked up to the workspace, as every read and update does
(`dependencies.check_access`). A person whose grant does not reach create is still refused.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException
from fastapi.responses import Response

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_card_shared_alone_opens_for_its_reader import _auth, _create, _req
from . import test_a_card_shared_alone_opens_for_its_reader as _card
from .test_there_is_no_promotion_state import OTHER, USER

#: The card test's node: the light-cone verifier, as production decides keys.
db = _card.db

WORKSPACE = "application/vnd.agience.workspace+json"
COLLECTION = "application/vnd.agience.collection+json"


async def _shared_backlog(db, actions):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type=WORKSPACE, name="Agience").id
    backlog = (await _create(db, container_id=space, content_type=COLLECTION, name="Backlog"))["id"]
    with acting_as(USER, principal_type="user"):
        la.create_grant(db, Grant(id=str(uuid.uuid4()), resource_id=space, grantee_type="user",
                                  grantee_id=OTHER, granted_by=USER, effect="allow", state=Grant.STATE_ACTIVE,
                                  **{f"can_{a}": a in actions for a in ACTIONS}))
    return space, backlog


async def _create_as_other(db, container_id):
    with acting_as(OTHER, principal_type="user"):
        return await artifacts.create_artifact(
            request=_req(), response=Response(), auth=_auth(OTHER), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=container_id, content_type="text/markdown",
                                                 content="FR-005", name="FR-005"))


async def test_a_collaborator_creates_in_a_collection_the_shared_workspace_holds(db):
    _, backlog = await _shared_backlog(db, {"read", "create", "add", "update"})
    made = await _create_as_other(db, backlog)
    assert made["id"]
    with acting_as(USER, principal_type="user"):
        members = {a["root_id"] for a in la.list_collection_artifacts(db, backlog)}
    assert made.get("root_id", made["id"]) in members


async def test_a_reader_of_the_shared_workspace_is_still_refused(db):
    _, backlog = await _shared_backlog(db, {"read"})
    with pytest.raises(HTTPException) as e:
        await _create_as_other(db, backlog)
    assert e.value.status_code == 404
