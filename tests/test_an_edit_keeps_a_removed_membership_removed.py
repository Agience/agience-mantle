"""An edit never puts an artifact back in a container it was removed from (BUG-034).

2026-11-10: an artifact created in workspace A, assigned into the Backlog, removed from A and then
edited came back in A on the next load. Its home stays A (its creation edge, which Origin's ledger
keeps for authority: John, 2026-11-10); only its membership in A is gone, and an edit must not
restore it.
"""
from __future__ import annotations

import json

from fastapi.responses import Response

from mantle.db import lattice_api as la
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from . import test_a_card_shared_alone_opens_for_its_reader as _card
from .test_a_card_shared_alone_opens_for_its_reader import _auth, _create, _req
from .test_there_is_no_promotion_state import USER

db = _card.db
WORKSPACE = "application/vnd.agience.workspace+json"
COLLECTION = "application/vnd.agience.collection+json"


def _roots(db, container):
    with acting_as(USER, principal_type="user"):
        return {a["root_id"] for a in la.list_collection_artifacts(db, container)}


async def test_an_edit_after_a_removal_does_not_restore_the_membership(db):
    with acting_as(USER, principal_type="user"):
        a = ws.create_container(db, USER, content_type=WORKSPACE, name="A").id
    backlog = (await _create(db, container_id=a, content_type=COLLECTION, name="Backlog"))["id"]
    item = await _create(db, container_id=a, content="first", context=json.dumps({"title": "Report"}))
    root = item.get("root_id") or item["id"]
    with acting_as(USER, principal_type="user"):
        la.add_artifact_to_collection(db, backlog, root)
        ws.remove_artifact_from_container(db, USER, a, item["id"], auth=_auth(USER))
    assert root not in _roots(db, a) and root in _roots(db, backlog)
    with acting_as(USER, principal_type="user"):
        await artifacts.update_artifact(artifact_id=item["id"], auth=_auth(USER), store_db=db,
                                        body=artifacts.UpdateArtifactRequest(content="second"))
    assert root in _roots(db, backlog)
    assert root not in _roots(db, a), "the edit put it back in the container it was removed from"
