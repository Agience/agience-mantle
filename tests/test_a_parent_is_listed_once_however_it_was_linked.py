"""An artifact's `containers` names each parent once: adding it again, through `held` or `add`,
after the parent changed, or with a typed link from the same parent, writes or lists no second
entry."""
from __future__ import annotations

import pytest

from mantle.routers import artifacts_router as ar
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import USER, real_node

COLLECTION_CT = "application/vnd.agience.collection+json"
AUTH = AuthContext(principal_id=USER, principal_type="user", user_id=USER)


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


async def _held(db, child, **change):
    with acting_as(USER, principal_type="user"):
        out = await ar.change_memberships(ar.MembershipRequest(artifact_ids=[child], **change),
                                          auth=AUTH, store_db=db)
    return out["items"][0]


async def test_each_way_of_linking_a_parent_again_leaves_it_listed_once(db):
    with acting_as(USER, principal_type="user"):
        parent = ws.create_container(db, USER, content_type=COLLECTION_CT, name="Receives").id
        child = ws.create_container(db, USER, content_type=COLLECTION_CT, name="Pack").id
    await _held(db, child, held=[parent])
    with acting_as(USER, principal_type="user"):
        ws.update_workspace(db, USER, parent, name="Receives", description="changed")
    await _held(db, child, held=[parent])
    await _held(db, child, add=[parent])
    with acting_as(USER, principal_type="user"):
        await ar._default_create_artifact(ar.CreateArtifactRequest(
            container_id=parent, source_artifact_id=child, relationship="related_to"), AUTH, db)
    item = await _held(db, child)
    assert item["container_ids"] == [parent]
    assert item["containers"] == [{"id": parent, "name": "Receives"}]
