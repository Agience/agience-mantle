"""An uploaded file is an artifact of the file's type, from the moment it is created.

2026-10-23, a tester: an uploaded Markdown file showed as a generic card, and `type:text/markdown`
did not find it, though its context said `text/markdown`. The artifact was created untyped, and the
type is what a reader renders it as and what a `type:` filter matches.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from mantle.db import lattice_api as store
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

# This store has no grant ledger behind its key oracle (`conftest.content_keys`).
pytestmark = pytest.mark.usefixtures("content_keys")

CAS_REF = "cas/" + "a" * 64


@pytest.fixture
def db():
    return store.LatticeDatabase(os.path.join(tempfile.mkdtemp(), "up.db"), origin="upload-test")


def test_an_upload_takes_its_files_type_and_keeps_it_on_completion(db, monkeypatch):
    import mantle.services.content_service as cs
    monkeypatch.setattr(cs, "head_object", lambda key: None)
    monkeypatch.setattr(cs, "persist_object_to_durable", lambda key: False)
    monkeypatch.setattr(cs, "local_content_has", lambda ref: ref == CAS_REF)
    with acting_as("u", principal_type="user"):
        cid = ws.create_container(db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="c").id
        _, created = ws.initiate_upload_and_create_artifact(
            db, "u", cid, filename="notes.md", content_type="text/markdown", size=13)
        assert created.content_type == "text/markdown"
        ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=created.id,
                                context_patch={"content_cas_ref": CAS_REF, "content_sha256": "d" * 64})
        done = ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=created.id,
                                       status_value="complete")
    assert done.content_type == "text/markdown"
    assert json.loads(done.context)["content_type"] == "text/markdown"


def test_a_stored_untyped_upload_is_typed_once_in_place(db):
    """`mantle.system.type_uploads`: an upload stored before uploads were typed takes its file's
    type, in place (no version is kept), and a second run changes nothing."""
    from mantle.entities.artifact import Artifact as ArtifactEntity
    from mantle.system.type_uploads import type_stored_uploads
    with acting_as("u", principal_type="user"):
        cid = ws.create_container(db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="c").id
    ctx = {"content_source": "agience-content", "filename": "plan.md", "content_type": "text/markdown"}
    store.create_artifact(db, ArtifactEntity(
        id="old-upload", root_id="old-upload", collection_id=cid, created_by="u",
        state=ArtifactEntity.STATE_LIVE, context=json.dumps(ctx)))
    # A small text file the app read in whole: no `content_source`, a filename and a type.
    inline = {"filename": "notes.txt", "mime": "text/plain", "content_type": "text/plain"}
    store.create_artifact(db, ArtifactEntity(
        id="old-inline", root_id="old-inline", collection_id=cid, created_by="u",
        state=ArtifactEntity.STATE_LIVE, context=json.dumps(inline)))
    # A note: its type is in its context, but it is not a file and is left as it is.
    store.create_artifact(db, ArtifactEntity(
        id="a-note", root_id="a-note", collection_id=cid, created_by="u",
        state=ArtifactEntity.STATE_LIVE, context=json.dumps({"content_type": "text/plain"})))
    assert type_stored_uploads(db) == {"untyped": 2, "typed": 2}
    assert [d["id"] for d in db.artifacts.list_artifacts(content_type="text/markdown")] == ["old-upload"]
    assert [d["id"] for d in db.artifacts.list_artifacts(content_type="text/plain")] == ["old-inline"]
    assert len(db.artifacts.versions_of("old-upload") or []) <= 1, "typing is not a content change"
    assert type_stored_uploads(db) == {"untyped": 0, "typed": 0}
