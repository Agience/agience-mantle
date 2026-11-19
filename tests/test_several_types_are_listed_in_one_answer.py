"""A picker that offers artifacts of several types asks once: `/artifacts/visible` takes the
content type more than once and answers every readable artifact of any of them, each once, with
the title it is shown by."""
from __future__ import annotations

import pytest

from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext
from mantle.services import workspace_service as ws

from .test_there_is_no_promotion_state import USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"
COLLECTION_CT = "application/vnd.agience.collection+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


async def _visible(db, types):
    from mantle.routers import artifacts_router as artifacts
    with acting_as(USER, principal_type="user"):
        page = await artifacts.list_visible(
            content_type=types, action="read", order="id", limit=100, offset=0,
            auth=AuthContext(principal_id=USER, principal_type="user", user_id=USER), store_db=db)
    return page["items"]


async def test_two_types_come_back_in_one_answer_each_once(db):
    with acting_as(USER, principal_type="user"):
        w = ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Plans").id
        c = ws.create_container(db, USER, content_type=COLLECTION_CT, name="Notes").id
    items = await _visible(db, [WORKSPACE_CT, COLLECTION_CT, WORKSPACE_CT])
    ids = [i["id"] for i in items]
    assert {w, c} <= set(ids) and len(ids) == len(set(ids))
    assert {i["content_type"] for i in items} <= {WORKSPACE_CT, COLLECTION_CT}
    names = {i["id"]: i.get("name") for i in items}
    assert names[w] == "Plans" and names[c] == "Notes"


async def test_one_type_named_alone_lists_only_it(db):
    with acting_as(USER, principal_type="user"):
        w = ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Plans").id
        c = ws.create_container(db, USER, content_type=COLLECTION_CT, name="Notes").id
    ids = {i["id"] for i in await _visible(db, WORKSPACE_CT)}
    assert w in ids and c not in ids
    ids = {i["id"] for i in await _visible(db, [COLLECTION_CT])}
    assert c in ids and w not in ids
