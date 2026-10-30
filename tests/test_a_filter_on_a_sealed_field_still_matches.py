"""A `title:`, `description:` or `tags:` filter matches what its searcher can read, though every one of
those fields is sealed at rest.

2026-10-29, sealing everything: the filters were evaluated on the stored row
(`search/mantle/lightcone.py`), where the fields are ciphertext, so a filter on any of them matched
nothing. The predicate now reads the row opened for the searcher, who is authorized by then.
"""
from __future__ import annotations

import json

from fastapi.responses import Response
from starlette.requests import Request

from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_title_is_content_and_sealed_at_rest import _auth, db  # noqa: F401
from .test_there_is_no_promotion_state import USER, _recall


async def _tagged_note(db):  # noqa: F811
    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts", "query_string": b""})
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type="application/vnd.agience.collection+json",
                                    name="Shelf").id
        doc = await artifacts.create_artifact(
            request=req, response=Response(), auth=_auth(), store_db=db,
            body=artifacts.CreateArtifactRequest(
                container_id=shelf, content="quarterly figures quillfeather",
                context=json.dumps({"title": "Board figures", "description": "for the board",
                                    "tags": ["finance"], "kind": "note"})))
    return doc


async def test_each_sealed_field_filter_finds_the_note(db):  # noqa: F811
    note = await _tagged_note(db)
    root = note.get("root_id") or note["id"]
    assert root in _recall(db, "quillfeather"), "the control: the note is found without a filter"
    for query in ("quillfeather tags:finance", 'quillfeather title:"Board figures"',
                  'quillfeather description:"for the board"'):
        assert root in _recall(db, query), f"{query!r} did not find the note"
    assert root not in _recall(db, "quillfeather tags:holiday"), "a filter that should exclude it did not"
