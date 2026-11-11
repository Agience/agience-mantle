"""A run whose grant lists a collection lists it and reads the notes it holds (BUG-052).

2026-11-11, production 2026.11.24: with a single note as evidence the run answered from it, and
with the note's collection as evidence it answered "no grounded evidence". The run's job delegation
must list the collection's children and open each note's body, as Sage's `ask` does.
"""
from __future__ import annotations

import json

from mantle.routers import artifacts_router as artifacts
from mantle.services import dependencies as D
from mantle.services.acting_principal import acting_from_auth, set_acting_principal

from . import test_a_card_shared_alone_opens_for_its_reader as _card
from .test_a_card_shared_alone_opens_for_its_reader import _create, _req
from .test_there_is_no_promotion_state import USER
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

db = _card.db
COLLECTION = "application/vnd.agience.collection+json"


def _run_auth(reach):
    return D.AuthContext(principal_id=USER, principal_type="user", user_id=USER,
                         actor="agience-server-sage", delegation_aud="mantle", narrowed=True,
                         scope="persona:sage", job_reach=tuple(reach))


async def test_a_run_lists_the_collection_and_reads_its_note(db, monkeypatch):
    from mantle.db import backend
    monkeypatch.setattr(backend, "_HANDLE", db)
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json",
                                    name="Acceptance space").id
    shelf = (await _create(db, container_id=space, content_type=COLLECTION,
                           context=json.dumps({"title": "Acceptance"})))["id"]
    note = await _create(db, container_id=shelf, content_type="text/markdown",
                         context=json.dumps({"title": "Smoke note"}), content="The smoke check confirms 42.")
    auth = _run_auth((shelf,))
    token = set_acting_principal(acting_from_auth(auth))
    try:
        page = await artifacts.list_children(artifact_id=shelf, request=_req(), store_db=db, auth=auth,
                                             content_type=None, workspace_id=None, state="live",
                                             limit=200, offset=0, links=None)
        items = page["items"] if isinstance(page, dict) else page
        ids = {i.get("id") for i in items} | {i.get("root_id") for i in items}
        assert note["id"] in ids or note.get("root_id") in ids, items
        read = await artifacts.read_artifact(artifact_id=note["id"], auth=auth, store_db=db)
        body = read if isinstance(read, dict) else json.loads(read.body)
        assert "42" in (body.get("content") or ""), body
    finally:
        set_acting_principal(None) if token is None else None
