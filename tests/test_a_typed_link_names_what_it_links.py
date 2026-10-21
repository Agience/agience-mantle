"""A typed link says what the linked artifact is to the one it hangs from — the collection a workflow
step reconciles, the authorizer an operator sends with — and confers nothing.

2026-10-21 (John): "everything should be linked to other artifacts." A workflow's arguments named
artifacts as ids typed into strings; the store could link an artifact only as a plain member.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import OTHER, USER, real_node

TRANSFORM = "application/vnd.agience.transform+json"
COLLECTION = "application/vnd.agience.collection+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _auth(user=USER):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


def _req():
    return Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts", "query_string": b""})


async def _create(db, user=USER, **body):
    with acting_as(user, principal_type="user"):
        return await artifacts.create_artifact(request=_req(), response=Response(), auth=_auth(user),
                                               store_db=db, body=artifacts.CreateArtifactRequest(**body))


async def _list(db, artifact_id, user=USER, **kw):
    with acting_as(user, principal_type="user"):
        return await artifacts.list_children(artifact_id=artifact_id, request=_req(), store_db=db,
                                             auth=_auth(user), content_type=None, workspace_id=None,
                                             state="live", limit=100, offset=0, links=kw.get("links"))


async def _setup(db):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json", name="Routines").id
        accounts = ws.create_container(db, USER, content_type=COLLECTION, name="Accounts").id
    flow = await _create(db, container_id=space, content_type=TRANSFORM,
                         context=json.dumps({"title": "Morning check"}))
    note = await _create(db, container_id=space, context=json.dumps({"title": "a note"}), content="x")
    return space, accounts, flow, note


def _ids(page):
    items = page["items"] if isinstance(page, dict) else page
    return {(i.get("root_id") or i.get("id"), i.get("relationship")) for i in items}


async def test_a_link_carries_its_relationship_and_is_listed_as_a_link(db):
    _, accounts, flow, _ = await _setup(db)
    await _create(db, container_id=flow["id"], source_artifact_id=accounts, relationship="step1.collection_id")
    links = await _list(db, flow["id"], links="*")
    assert _ids(links) == {(accounts, "step1.collection_id")}
    assert _ids(await _list(db, flow["id"], links="step1")) == {(accounts, "step1.collection_id")}
    assert _ids(await _list(db, flow["id"], links="step2")) == set()
    assert _ids(await _list(db, flow["id"])) == set(), "a link is not a member"


async def test_a_link_grants_nothing_and_shows_only_what_the_reader_may_read(db):
    space, accounts, flow, note = await _setup(db)
    await _create(db, container_id=flow["id"], source_artifact_id=note["id"], relationship="input")
    from mantle.db import lattice_api as la
    from mantle.attenuation import ACTIONS
    from mantle.entities.grant import Grant
    import uuid
    la.create_grant(db, Grant(id=str(uuid.uuid4()), resource_id=flow["id"], grantee_type="user", grantee_id=OTHER,
                              granted_by=USER, effect="allow", state=Grant.STATE_ACTIVE,
                              **{f"can_{a}": a == "read" for a in ACTIONS}))
    seen = await _list(db, flow["id"], user=OTHER, links="*")
    assert _ids(seen) == set(), "reading the workflow does not reach what it links"


async def test_a_relationship_without_a_link_is_refused(db):
    space, _, _, _ = await _setup(db)
    with pytest.raises(HTTPException) as e:
        await _create(db, container_id=space, relationship="input", context=json.dumps({"title": "x"}))
    assert e.value.status_code == 400


async def test_removing_a_steps_links_leaves_members_and_the_linked_artifact_alone(db):
    space, accounts, flow, note = await _setup(db)
    await _create(db, container_id=flow["id"], source_artifact_id=accounts, relationship="s1.collection_id")
    await _create(db, container_id=flow["id"], source_artifact_id=note["id"], relationship="s2.source_id")
    await _create(db, container_id=flow["id"], source_artifact_id=note["id"])          # a member too
    with acting_as(USER, principal_type="user"):
        out = await artifacts.remove_links(artifact_id=flow["id"], relationship="s1", auth=_auth(), store_db=db)
    assert out["removed"] == 1
    assert _ids(await _list(db, flow["id"], links="*")) == {(note["id"], "s2.source_id")}
    assert _ids(await _list(db, flow["id"])) == {(note["id"], None)}, "the member stays"
    from mantle.db import backend as store
    assert store.get_artifact(db, accounts).state == "live", "the linked artifact is not archived"


async def test_removing_links_needs_evict_on_the_artifact(db):
    _, accounts, flow, _ = await _setup(db)
    await _create(db, container_id=flow["id"], source_artifact_id=accounts, relationship="s1.collection_id")
    with acting_as(OTHER, principal_type="user"):
        with pytest.raises(HTTPException) as e:
            await artifacts.remove_links(artifact_id=flow["id"], relationship="s1", auth=_auth(OTHER), store_db=db)
    assert e.value.status_code == 404
    assert _ids(await _list(db, flow["id"], links="*")) == {(accounts, "s1.collection_id")}
