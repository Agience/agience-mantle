"""A picker's search finds collections and workspaces by name: a recall whose query text narrows
to their types returns the containers whose names match and none of the artifacts inside them."""
from __future__ import annotations

import pytest

from mantle.routers import artifacts_router as artifacts
from mantle.services.acting_principal import acting_as
from mantle.services import workspace_service as ws

from .test_a_hyphenated_word_is_found_by_its_parts import _auth, _note
from .test_there_is_no_promotion_state import USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"
COLLECTION_CT = "application/vnd.agience.collection+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


async def test_a_type_narrowed_recall_finds_containers_by_name(db):
    with acting_as(USER, principal_type="user"):
        home = ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Orchard plans").id
        coll = ws.create_container(db, USER, content_type=COLLECTION_CT, name="Orchard notes").id
        other = ws.create_container(db, USER, content_type=COLLECTION_CT, name="Harbour").id
    note = await _note(db, home, "Orchard rows", "orchard planting")
    with acting_as(USER, principal_type="user"):
        result = await artifacts.recall_artifacts(
            body=artifacts.ArtifactRecallRequest(
                query_text=f"orchard type:{COLLECTION_CT},{WORKSPACE_CT}"),
            auth=_auth(), store_db=db)
    hits = result["hits"] if isinstance(result, dict) else result.hits
    found = {(h["root_id"] if isinstance(h, dict) else h.root_id) for h in hits}
    assert {home, coll} <= found
    assert other not in found and note not in found
