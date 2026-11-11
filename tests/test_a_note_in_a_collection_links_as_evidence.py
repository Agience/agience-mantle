"""An artifact whose body the caller can't open still links, as a collection (which has no body) does.

BUG-062 (2026-11-11, production): Lumen's Add step links a step's evidence under the persona's
delegation, which opens no person's content key. The link loaded the note strictly, the body
didn't open, and the call answered 500. Picking the note's collection worked. A link writes an
edge and needs no body, so it reads the artifact without refusing over the body.
"""
from __future__ import annotations

import json

import pytest

from mantle.db import lattice_api
from mantle.services import content_crypto, content_service
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_typed_link_names_what_it_links import COLLECTION, TRANSFORM, _create, _ids, _list
from .test_there_is_no_promotion_state import USER, real_node


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _no_key(*_a, **_k):
    raise RuntimeError("no content key for this delegation")


async def _filed(db):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json",
                                    name="Acceptance space").id
    shelf = await _create(db, container_id=space, content_type=COLLECTION,
                          context=json.dumps({"title": "Acceptance"}))
    note = await _create(db, container_id=shelf["id"], content_type="text/markdown",
                         context=json.dumps({"title": "Smoke note"}), content="The answer is 42.")
    flow = await _create(db, container_id=space, content_type=TRANSFORM,
                         context=json.dumps({"title": "Check"}))
    return note, flow


async def test_a_note_whose_body_the_caller_cannot_open_links_as_evidence(db, monkeypatch):
    note, flow = await _filed(db)
    monkeypatch.setattr(content_service, "get_bytes_decrypted", _no_key)
    monkeypatch.setattr(content_crypto, "decrypt_content", _no_key)
    root = note.get("root_id") or note["id"]
    with pytest.raises(Exception):
        lattice_api.get_artifact(db, note["id"]), "the body really doesn't open for this caller"
    await _create(db, container_id=flow["id"], source_artifact_id=note["id"], relationship="s1a2b3c4.evidence")
    assert _ids(await _list(db, flow["id"], links="*")) == {(root, "s1a2b3c4.evidence")}


async def test_a_note_inside_a_collection_links_as_evidence(db):
    note, flow = await _filed(db)
    await _create(db, container_id=flow["id"], source_artifact_id=note["id"], relationship="s1a2b3c4.evidence")
    assert _ids(await _list(db, flow["id"], links="*")) == {(note.get("root_id") or note["id"], "s1a2b3c4.evidence")}
