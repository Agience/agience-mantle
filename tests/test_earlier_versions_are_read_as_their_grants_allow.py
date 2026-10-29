"""An artifact's earlier versions can be listed, read and restored, each as the light cone allows.

2026-10-23, a tester: the glossary says every edit is kept, and no view showed an earlier version.
A content edit keeps the version it replaces as an archived snapshot named by `parent_id`; these
routes walk that line. A version is an artifact, so whether a principal sees it is `check_access`
on that version's own id: a grant on the artifact reaches every version through its root, and a
grant or a deny on one version reaches that version alone (John, 2026-10-27: "only access to the
versions I can see"). A restore is refused when it would show a version to someone it is denied to.
"""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from mantle.db.backend import upsert_user_collection_grant
from mantle.entities.grant import Grant
from mantle.routers import artifacts_router as artifacts
from mantle.services import grant_service
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext
from mantle.services import workspace_service as ws

from .test_there_is_no_promotion_state import OTHER, USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


def _req():
    return Request({"type": "http", "method": "POST", "headers": [], "path": "/", "query_string": b""})


async def _edit(db, aid, title, content):
    with acting_as(USER, principal_type="user"):
        await artifacts.update_artifact(
            artifact_id=aid, auth=_auth(USER), store_db=db,
            body=artifacts.UpdateArtifactRequest(content=content, context={"title": title}))


async def _note_with_three_versions(db):
    with acting_as(USER, principal_type="user"):
        home = ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Home").id
        made = await artifacts.create_artifact(
            request=_req(), response=Response(), auth=_auth(USER), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=home, content="first",
                                                 context=json.dumps({"title": "One"})))
    aid = made["id"]
    await _edit(db, aid, "Two", "second")
    await _edit(db, aid, "Three", "third")
    return home, aid


async def _versions(db, user, aid):
    with acting_as(user, principal_type="user"):
        page = await artifacts.list_versions(artifact_id=aid, auth=_auth(user), store_db=db)
    return page["items"]


async def test_the_owner_sees_every_version_and_reads_an_earlier_one(db):
    _, aid = await _note_with_three_versions(db)
    items = await _versions(db, USER, aid)
    assert [i["title"] for i in items] == ["Three", "Two", "One"]
    assert items[0]["current"] and not any(i["current"] for i in items[1:])
    assert items[0]["parent_id"] == items[1]["id"] and items[1]["parent_id"] == items[2]["id"]
    with acting_as(USER, principal_type="user"):
        first = await artifacts.read_version(artifact_id=aid, version_id=items[2]["id"],
                                             auth=_auth(USER), store_db=db)
    assert first["content"] == "first"
    assert json.loads(first["context"])["title"] == "One"


async def test_a_reader_of_the_artifact_sees_its_versions_and_a_deny_on_one_hides_it(db):
    home, aid = await _note_with_three_versions(db)
    with acting_as(USER, principal_type="user"):
        _, token = grant_service.create_invite(db, user_id=USER, resource_id=home, role="viewer")
    grant_service.claim_invite(db, OTHER, token, "")
    items = await _versions(db, OTHER, aid)
    assert [i["title"] for i in items] == ["Three", "Two", "One"]

    hidden = items[1]["id"]
    deny = Grant(resource_id=hidden, grantee_type="user", grantee_id=OTHER, granted_by=USER,
                 effect="deny", can_read=True, state="active")
    with acting_as(USER, principal_type="user"):
        grant_service.create_grant(db, deny)
    assert [i["title"] for i in await _versions(db, OTHER, aid)] == ["Three", "One"]
    with acting_as(OTHER, principal_type="user"), pytest.raises(HTTPException) as refused:
        await artifacts.read_version(artifact_id=aid, version_id=hidden, auth=_auth(OTHER), store_db=db)
    assert refused.value.status_code == 404

    # Restoring it would show it to the reader it is denied to.
    with acting_as(USER, principal_type="user"), pytest.raises(HTTPException) as widen:
        await artifacts.restore_version(artifact_id=aid, version_id=hidden, auth=_auth(USER), store_db=db)
    assert widen.value.status_code == 409


async def test_a_grant_on_one_version_reads_that_version_alone(db):
    _, aid = await _note_with_three_versions(db)
    items = await _versions(db, USER, aid)
    only = items[2]["id"]
    with acting_as(USER, principal_type="user"):
        upsert_user_collection_grant(db, user_id=OTHER, collection_id=only, granted_by=USER, can_read=True)
        version = await artifacts.read_version(artifact_id=aid, version_id=only,
                                               auth=_auth(OTHER), store_db=db)
    assert version["content"] == "first"
    with acting_as(OTHER, principal_type="user"), pytest.raises(HTTPException):
        await artifacts.read_version(artifact_id=aid, version_id=items[1]["id"],
                                     auth=_auth(OTHER), store_db=db)


async def test_a_restore_is_a_new_version_and_loses_nothing(db):
    _, aid = await _note_with_three_versions(db)
    first = (await _versions(db, USER, aid))[2]["id"]
    with acting_as(USER, principal_type="user"):
        restored = await artifacts.restore_version(artifact_id=aid, version_id=first,
                                                   auth=_auth(USER), store_db=db)
    assert restored["content"] == "first"
    items = await _versions(db, USER, aid)
    assert [i["title"] for i in items] == ["One", "Three", "Two", "One"]
    assert items[0]["id"] == aid, "the artifact keeps its id; the version it replaced is kept"


async def test_a_version_of_another_artifact_is_not_read_through_this_one(db):
    _, aid = await _note_with_three_versions(db)
    _, other = await _note_with_three_versions(db)
    foreign = (await _versions(db, USER, other))[1]["id"]
    with acting_as(USER, principal_type="user"), pytest.raises(HTTPException) as nope:
        await artifacts.read_version(artifact_id=aid, version_id=foreign, auth=_auth(USER), store_db=db)
    assert nope.value.status_code == 404
