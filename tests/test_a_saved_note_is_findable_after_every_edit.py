"""A note saved in a workspace is in the segment `recall` reads by default — and stays there.

`recall` does not filter by state; it selects an index SEGMENT before the query runs, and
`ArtifactRecallRequest.state` defaults to `live`. So "findable" is decided at write time, by the
state the write leaves the artifact in, and the index segment that state routes it to.

Two ends have to hold, and a fix to one alone passes a first-write test and fails the user's
second keystroke:

* **born real** — a note created in a workspace is written `live`;
* **edits stay real** — editing it rewrites that live version in place. Forking a second version
  on edit would leave the first row, and its index entry, answering recall with the content from
  BEFORE the edit.

These run through the real store (a `LatticeDatabase` on disk) and the real index job
(`enqueue_index_artifact` → `index_artifact`, synchronously); only the two index arms' writes are
recorded instead of performed, which is where the segment each write routes to becomes visible.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

import pytest

from mantle.db import backend
from mantle.db import lattice_api as store
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE
from mantle.routers.artifacts_router import ArtifactRecallRequest
from mantle.search.ingest import pipeline_unified
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

USER = "u-note"
CAS_REF = "cas/" + "b" * 64

#: The segment `recall` reads when the caller names none — read from the request model rather than
#: restated, so this file follows the published default if it ever moves.
DEFAULT_RECALL_SEGMENT = ArtifactRecallRequest.model_fields["state"].default


@pytest.fixture
def db():
    return store.LatticeDatabase(os.path.join(tempfile.mkdtemp(), "note.db"), origin="note-test")


@pytest.fixture
def workspace(db):
    return ws.create_container(db, USER, content_type=WORKSPACE_CONTENT_TYPE, name="Inbox").id


@pytest.fixture(autouse=True)
def _bodies_stay_readable(monkeypatch):
    """Keep each body on the row instead of addressing it in a CAS this test has none of.

    `conftest`'s stub returns a real-looking `cas/` address for bytes it never stores, so reading
    a note back — which `update_artifact` does before every edit — would fail to hydrate it. What
    is under test is the state and the index segment each write leaves, not where bytes live.
    """
    monkeypatch.setattr(
        ws, "_store_content_in_s3",
        lambda artifact_id, content, context_str, **_: (f"artifacts/{artifact_id}.content",
                                                        content, None),
    )
    # The REAL envelope with only the master-key source injected, as
    # `test_an_unreachable_mirror_leaves_a_record.py` does: a body on the row is sealed for its
    # owner, and the key oracle that normally supplies the owner's key is not wired here.
    from mantle.services import content_crypto

    real_encrypt, real_decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content

    def _provider(principal_id: str) -> bytes:
        return hashlib.sha256(("master:" + principal_id).encode("utf-8")).digest()

    monkeypatch.setattr(content_crypto, "encrypt_content",
                        lambda pid, pt, **kw: real_encrypt(pid, pt, master_key_provider=_provider, **kw))
    monkeypatch.setattr(content_crypto, "decrypt_content",
                        lambda pid, blob, **kw: real_decrypt(pid, blob, master_key_provider=_provider, **kw))


@pytest.fixture
def index_writes(monkeypatch):
    """Run every index job synchronously and record what each arm was asked to write, and where.

    Each entry is ``(arm, segment, root_id, content)``; vacated segments are recorded as
    ``("vacate", segment, root_id, None)``.
    """
    writes: list[tuple] = []

    def _sse(artifact, collection_id, fields, *, segment="live", indexer=None):
        writes.append(("lexical", segment, artifact.root_id or artifact.id, artifact.content))
        return pipeline_unified.ARM_WRITTEN

    def _vector(artifact, collection_id, fields, *, segment="live", vector=None, **_):
        writes.append(("vector", segment, artifact.root_id or artifact.id, artifact.content))
        return pipeline_unified.ARM_WRITTEN

    def _vacate(artifact, collection_id, *, remove_from):
        for seg in remove_from:
            writes.append(("vacate", seg, artifact.root_id or artifact.id, None))

    monkeypatch.setattr(pipeline_unified, "index_queue", None)
    monkeypatch.setattr(pipeline_unified, "_sse_index_artifact", _sse)
    monkeypatch.setattr(pipeline_unified, "_mantle_index_artifact", _vector)
    monkeypatch.setattr(pipeline_unified, "move_artifact_segments", _vacate)
    monkeypatch.setattr(pipeline_unified, "_mark_indexed", lambda artifact_id: None)
    return writes


def _lineage(db, root_id):
    return db.artifacts.versions_of(root_id) or []


def test_the_default_recall_segment_is_live():
    """The premise every other test here leans on, pinned so a change to it is seen."""
    assert DEFAULT_RECALL_SEGMENT == "live"


def test_a_new_note_is_written_into_the_default_recall_segment(db, workspace, index_writes):
    with acting_as(USER, principal_type="user"):
        note = ws.create_workspace_artifact(
            db, USER, workspace, context='{"content_type":"text/markdown"}',
            content="first thought", content_type="text/markdown",
        )

    assert note.state == DEFAULT_RECALL_SEGMENT
    assert [store.state_of(v) for v in _lineage(db, note.root_id)] == [DEFAULT_RECALL_SEGMENT]
    indexed = {seg for arm, seg, root, _ in index_writes if arm != "vacate" and root == note.root_id}
    assert indexed == {DEFAULT_RECALL_SEGMENT}, (
        f"the note was indexed into {indexed}, so a default recall cannot find it")


def test_the_second_edit_is_what_a_default_recall_answers_with(db, workspace, index_writes):
    """A save after the first one is what recall answers with: one live row, the newest content.

    A create that lands in the default segment is not enough on its own. If an edit left the first
    row live, it and its index entry would keep answering with the content from before the edit,
    so the first save would be findable and every later one not. Each body change keeps the
    version it replaced as an archived snapshot, which recall does not answer with.
    """
    with acting_as(USER, principal_type="user"):
        note = ws.create_workspace_artifact(
            db, USER, workspace, context='{"content_type":"text/markdown"}',
            content="first thought", content_type="text/markdown",
        )
        ws.update_artifact(db, USER, workspace, note.id, content="second thought")
        ws.update_artifact(db, USER, workspace, note.id, content="third thought")

    root = note.root_id
    lineage = _lineage(db, root)
    live = [v for v in lineage if store.state_of(v) != "archived"]
    assert [v["id"] for v in live] == [note.id], f"the lineage has {len(live)} live rows"
    assert len(lineage) == 3, "each body change keeps the version it replaced"

    # The row recall hydrates a live-segment hit through, and what it holds.
    hydrated = backend.find_newest_by_root(db, root)
    assert hydrated is not None and hydrated["id"] == note.id
    assert store.get_artifact(db, note.id).content == "third thought"

    # The last thing the index was told about this root, per arm: the newest content, in the
    # default segment. A stale entry there is exactly "recall answers with the old text".
    last = {}
    for arm, seg, r, content in index_writes:
        if r == root and arm != "vacate":
            last[arm] = (seg, content)
    assert last == {
        "lexical": (DEFAULT_RECALL_SEGMENT, "third thought"),
        "vector": (DEFAULT_RECALL_SEGMENT, "third thought"),
    }
    assert {seg for arm, seg, r, _ in index_writes
            if r == root and arm != "vacate"} == {DEFAULT_RECALL_SEGMENT}


def test_a_legacy_draft_is_written_live_by_its_next_edit(db, workspace, index_writes):
    """A `draft` row a store that has not been migrated holds reads as live, and its
    next edit writes it `live` and indexes it into the live segment."""
    root = "legacy-draft-1"
    db.artifacts.put_artifact({"id": root, "root_id": root, "collection_id": workspace,
                               "created_by": USER, "modified_by": USER, "state": "draft",
                               "context": '{"content_type":"text/plain"}', "content": ""})
    store.add_artifact_to_collection(db, workspace, root)

    with acting_as(USER, principal_type="user"):
        out = ws.update_artifact(db, USER, workspace, root, content="now saved")

    assert out.state == DEFAULT_RECALL_SEGMENT
    assert db.artifacts.get_artifact(root)["state"] == DEFAULT_RECALL_SEGMENT
    assert ("lexical", DEFAULT_RECALL_SEGMENT, root, "now saved") in index_writes


def test_an_upload_completes_into_the_default_recall_segment(db, workspace, index_writes,
                                                               monkeypatch):
    """Initiate → progress → complete, through the service calls the upload routes make."""
    import mantle.services.content_service as cs
    monkeypatch.setattr(cs, "head_object", lambda key: None)
    monkeypatch.setattr(cs, "persist_object_to_durable", lambda key: False)
    monkeypatch.setattr(cs, "local_content_has", lambda ref: ref == CAS_REF)
    # The uploaded body is read back from storage for indexing, and no bytes were really written
    # here. What it extracts to is not the question; which segment it is written into is.
    monkeypatch.setattr(pipeline_unified, "_extract_artifact_fields",
                        lambda artifact: {"title": "notes.txt", "content": "uploaded text"})

    with acting_as(USER, principal_type="user"):
        _descriptor, created = ws.initiate_upload_and_create_artifact(
            db, USER, workspace, filename="notes.txt", content_type="text/plain", size=13,
        )
        # What the proxied `PUT /artifacts/{id}/content` records once the bytes are in the CAS.
        ws.update_upload_status(
            db=db, user_id=USER, workspace_id=workspace, upload_id=created.id,
            context_patch={"content_cas_ref": CAS_REF},
        )
        done = ws.update_upload_status(
            db=db, user_id=USER, workspace_id=workspace, upload_id=created.id,
            status_value="complete",
        )

    assert "upload" not in json.loads(done.context)
    assert done.id == created.id, "the upload's writes forked the artifact"
    assert done.state == DEFAULT_RECALL_SEGMENT
    assert [store.state_of(v) for v in _lineage(db, created.root_id)] == [DEFAULT_RECALL_SEGMENT]
    indexed = [seg for arm, seg, r, _ in index_writes if r == created.root_id and arm != "vacate"]
    assert indexed and set(indexed) == {DEFAULT_RECALL_SEGMENT}, (
        f"the completed upload was indexed into {set(indexed) or 'nothing'}")


def test_with_content_addressed_bodies_recall_answers_with_the_newest_edit(
        db, workspace, index_writes, monkeypatch):
    """The same two edits with bodies addressed in a CAS, as production writes them.

    With a `content_ref` on the row, `db/vertex.put_artifact` snapshots the prior version under a
    derived id whenever the content changes, so the lineage grows by one archived row per edit.
    That is history, not a second answer: the head is still one row, and it is what recall's
    hydration and the index both describe.
    """
    cas: dict = {}

    def _addressed(artifact_id, content, context_str, **_):
        ref = "cas/" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        cas[ref] = content.encode("utf-8")
        return f"artifacts/{artifact_id}.content", content, ref

    monkeypatch.setattr(ws, "_store_content_in_s3", _addressed)
    import mantle.services.content_service as cs
    monkeypatch.setattr(cs, "get_bytes_decrypted",
                        lambda key, principal, *, cas_ref=None, **_: cas[cas_ref])

    with acting_as(USER, principal_type="user"):
        note = ws.create_workspace_artifact(
            db, USER, workspace, context='{"content_type":"text/markdown"}',
            content="first thought", content_type="text/markdown",
        )
        ws.update_artifact(db, USER, workspace, note.id, content="second thought")
        ws.update_artifact(db, USER, workspace, note.id, content="third thought")

    root = note.root_id
    lineage = _lineage(db, root)
    heads = [v for v in lineage if store.state_of(v) != "archived"]
    snapshots = [v for v in lineage if store.state_of(v) == "archived"]
    assert [v["id"] for v in heads] == [note.id], f"the lineage has {len(heads)} heads"
    assert len(snapshots) == 2, "each content change keeps the prior version as a snapshot"
    # The live version names the snapshot it was made from; the lineage is a DAG, not a pile.
    assert store.get_artifact(db, note.id).parent_id in {v["id"] for v in snapshots}
    assert not any(v.get("superseded_by") for v in lineage), "no row points forward to its successor"

    hydrated = backend.find_newest_by_root(db, root)
    assert hydrated["id"] == note.id
    assert store.get_artifact(db, note.id).content == "third thought"

    last = {}
    for arm, seg, r, content in index_writes:
        if r == root and arm != "vacate":
            last[arm] = (seg, content)
    assert last == {
        "lexical": (DEFAULT_RECALL_SEGMENT, "third thought"),
        "vector": (DEFAULT_RECALL_SEGMENT, "third thought"),
    }
