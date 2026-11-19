"""A workflow's run is listed in its Runs collection.

A run is made the way the workflow runner makes it: a top-level collection, the run created into
it, then the run linked to its workflow as `workflow`. The collection's children list the run.
"""
from __future__ import annotations

import json

import pytest

from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_typed_link_names_what_it_links import _create, _list
from .test_there_is_no_promotion_state import USER, real_node

COLLECTION = "application/vnd.agience.collection+json"
RUN = "application/vnd.agience.run+json"
TRANSFORM = "application/vnd.agience.transform+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


async def test_a_run_is_listed_in_its_runs_collection(db):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json",
                                    name="Flows").id
    flow = (await _create(db, container_id=space, content_type=TRANSFORM,
                          context=json.dumps({"title": "Acceptance workflow"})))["id"]
    runs = (await _create(db, content_type=COLLECTION, name="Runs — Acceptance workflow"))["id"]
    await _create(db, container_id=flow, source_artifact_id=runs, relationship="runs")
    record = {"title": "Acceptance workflow", "state": "running"}
    run = (await _create(db, container_id=runs, content_type=RUN, name=record["title"],
                         context=json.dumps(record)))["id"]
    await _create(db, container_id=run, source_artifact_id=flow, relationship="workflow")
    page = await _list(db, runs)
    items = page["items"] if isinstance(page, dict) else page
    assert [i.get("root_id") or i.get("id") for i in items] == [run]
