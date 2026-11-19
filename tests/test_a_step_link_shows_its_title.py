"""A workflow's step links are listed with the name a person knows them by: each linked artifact
comes back with its context, title included, to a reader who may read it. A persona's narrowed
delegation reads no sealed field, so to it the same links come back untitled (BUG-053).
"""
from __future__ import annotations

import json

import pytest

from .test_a_typed_link_names_what_it_links import _create, _list, _setup
from .test_there_is_no_promotion_state import real_node


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A standalone node whose key provider refuses what production's refuses: a narrowed
    delegation opens no person's key outside its job reach."""
    from mantle.services import content_crypto as cc
    from mantle.services.acting_principal import current_acting_principal
    database = real_node(tmp_path, monkeypatch)
    decrypt = cc.decrypt_content

    def refusing(pid, blob, **kw):
        cc._refuse_ember(current_acting_principal())
        return decrypt(pid, blob, **kw)
    monkeypatch.setattr(cc, "decrypt_content", refusing)
    return database


def _title(item):
    ctx = item.get("context")
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except ValueError:
            ctx = {}
    return (ctx or {}).get("title") or item.get("name") or ""


async def test_each_step_link_carries_the_linked_artifacts_title(db):
    _, accounts, flow, note = await _setup(db)
    await _create(db, container_id=flow["id"], source_artifact_id=accounts,
                  relationship="step1.collection_id")
    await _create(db, container_id=flow["id"], source_artifact_id=note["id"],
                  relationship="step2.source_id")
    page = await _list(db, flow["id"], links="*")
    titles = {i["relationship"]: _title(i) for i in page["items"]}
    assert titles == {"step1.collection_id": "Accounts", "step2.source_id": "a note"}


def _lumen(user):
    from mantle.services.dependencies import AuthContext
    return AuthContext(principal_id=user, principal_type="user", user_id=user,
                       actor="agience-server-lumen", delegation_aud="agience-server-lumen",
                       narrowed=True, scope="persona:lumen")


async def test_lumen_s_delegation_lists_the_step_links_without_their_sealed_titles(db):
    """Lumen reads the step links on the person's narrowed delegation, which opens no sealed field:
    the links come back, their titles empty, and the person's session names them."""
    from mantle.services.acting_principal import acting_as
    from .test_a_typed_link_names_what_it_links import COLLECTION, USER, _req
    from mantle.routers import artifacts_router as artifacts
    space, accounts, flow, _ = await _setup(db)
    shelf = await _create(db, container_id=space, content_type=COLLECTION,
                          context=json.dumps({"title": "Shelf"}))
    await _create(db, container_id=flow["id"], source_artifact_id=accounts,
                  relationship="step1.collection_id")
    await _create(db, container_id=flow["id"], source_artifact_id=shelf["id"],
                  relationship="step2.collection_id")
    with acting_as(USER, principal_type="user", actor="agience-server-lumen", narrowed="lumen",
                   scope="persona:lumen", source="request"):
        page = await artifacts.list_children(artifact_id=flow["id"], request=_req(), store_db=db,
                                             auth=_lumen(USER), content_type=None, workspace_id=None,
                                             state="live", limit=100, offset=0, links="*")
    titles = {i["relationship"]: _title(i) for i in page["items"]}
    assert titles == {"step1.collection_id": "", "step2.collection_id": ""}
