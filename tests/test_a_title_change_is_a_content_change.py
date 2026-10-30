"""A title is part of an artifact's content — declared on the root type `*/*` and inherited by every
type — so changing it keeps the version it replaced, as a body change does; and each version names
the version it was made from (`parent_id`), so a lineage is a DAG.

2026-10-22 (John): "title is part of content. Artifact (root object type) has a title. Inherited."
A version was kept only when the body's `content_ref` changed, so a title edit overwrote the title
before it with no trace. No row records `superseded_by`: a version names its parent, which allows a
lineage more than one successor.
"""
from __future__ import annotations

import json

import pytest

from mantle.db import backend as store
from mantle.db.doc_boundary import unseal_content_fields
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_there_is_no_promotion_state import USER, real_node


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _title(doc):
    raw = dict(doc)
    with acting_as(USER, principal_type="user"):
        unseal_content_fields(raw)
    return json.loads(raw["context"]).get("title")


def _note(db):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json", name="W").id
        note = ws.create_workspace_artifact(db, USER, space, context=json.dumps({"title": "Draft one"}),
                                            content="body")
    return space, note


def test_a_title_edit_keeps_the_title_it_replaced_and_names_it_as_parent(db):
    space, note = _note(db)
    with acting_as(USER, principal_type="user"):
        ws.update_artifact(db, USER, space, note.id, caller_context=json.dumps({"title": "Draft two"}))
        ws.update_artifact(db, USER, space, note.id, caller_context=json.dumps({"title": "Final"}))
    lineage = db.artifacts.versions_of(note.root_id)
    live = [v for v in lineage if store.state_of(v) != "archived"]
    kept = [v for v in lineage if store.state_of(v) == "archived"]
    assert [v["id"] for v in live] == [note.id] and _title(live[0]) == "Final"
    assert sorted(_title(v) for v in kept) == ["Draft one", "Draft two"], "a replaced title was lost"
    parents = {v["id"]: v.get("parent_id") for v in lineage}
    assert parents[note.id] in {v["id"] for v in kept} and _title(
        next(v for v in kept if v["id"] == parents[note.id])) == "Draft two"
    assert not any("superseded_by" in v for v in lineage)


def test_a_description_or_a_tag_is_content_and_makes_a_version(db):
    """Sealing is the default (John, 2026-10-29), and what is sealed is content: a new description
    or tag set makes a version as a new title does."""
    space, note = _note(db)
    with acting_as(USER, principal_type="user"):
        ws.update_artifact(db, USER, space, note.id, description="what this note is for")
    assert len(db.artifacts.versions_of(note.root_id)) == 2, "a new description made no version"


def test_a_change_that_is_not_content_is_written_in_place(db):
    """Upload bookkeeping is structure the server writes, not content: a progress update makes no
    version (`doc_boundary.STRUCTURAL_CONTEXT_KEYS`)."""
    space, note = _note(db)
    with acting_as(USER, principal_type="user"):
        ws.update_artifact(db, USER, space, note.id,
                           caller_context=json.dumps({"title": "Draft one", "upload": {"progress": 0.5}}))
    lineage = db.artifacts.versions_of(note.root_id)
    assert [v["id"] for v in lineage] == [note.id], "a non-content change made a version"
