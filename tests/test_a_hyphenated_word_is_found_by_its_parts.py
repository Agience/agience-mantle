"""A hyphenated word in a body is found by a search for its parts.

2026-10-23, a tester: a search for `quasar lantern` returned an uploaded file containing that phrase
and not the Note whose body held `quasar-lantern-731`. The analyzer splits a hyphenated word into
its parts, at index and at query alike, so the parts are terms of the Note.
"""
from __future__ import annotations

import json

import pytest
from fastapi.responses import Response
from starlette.requests import Request

from mantle.routers import artifacts_router as artifacts
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext
from mantle.services import workspace_service as ws

from .test_there_is_no_promotion_state import USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _auth():
    return AuthContext(principal_id=USER, principal_type="user", user_id=USER)


async def _note(db, home, title, body):
    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/", "query_string": b""})
    with acting_as(USER, principal_type="user"):
        made = await artifacts.create_artifact(
            request=req, response=Response(), auth=_auth(), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=home, content=body,
                                                 context=json.dumps({"title": title})))
    return made["id"]


async def _found(db, query):
    with acting_as(USER, principal_type="user"):
        result = await artifacts.recall_artifacts(
            body=artifacts.ArtifactRecallRequest(query_text=query), auth=_auth(), store_db=db)
    hits = result["hits"] if isinstance(result, dict) else result.hits
    return {(h["root_id"] if isinstance(h, dict) else h.root_id) for h in hits}


@pytest.mark.parametrize("query", ["quasar lantern", '="quasar lantern"', "quasar-lantern-731"])
async def test_a_hyphenated_word_is_found_by_its_parts(db, query):
    with acting_as(USER, principal_type="user"):
        home = ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Inbox").id
    note = await _note(db, home, "Search Gamma", "Regression marker quasar-lantern-731 for search.")
    assert note in await _found(db, query)
