"""Completing an upload asks the tiers in the order the read path asks them.

The CAS is this node's own encrypted store; an object store is redundancy behind it. That split is
supposed to be invisible above the storage layer — `PUT /artifacts/{id}/content` says so in its own
docstring, "a node with no object store is a complete configuration: the write succeeds, nothing
warns" — and the read path honours it, resolving "local CAS first, object store behind it".

⛔ THE COMPLETION GATE DID NOT. It called `head_object`, which asks S3/MinIO and nothing else, and
raised `400 Object not found in S3` when that came back empty. On a node with no object store every
upload therefore wrote its bytes, verified them, served them on the next read — and could never be
marked complete. The artifact stayed in `uploading` and the card never left Draft. Measured end to
end 2026-09-13 against a live local node.

The failure is worth naming precisely: nothing was lost and nothing errored on the way in. The
write succeeded, and a later check for the write looked somewhere the write was never required to
reach.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
from fastapi import HTTPException

from mantle.db import lattice_api as store
from mantle.entities.artifact import Artifact as ArtifactEntity, WORKSPACE_CONTENT_TYPE
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

CAS_REF = "cas/" + "a" * 64


@pytest.fixture
def db():
    return store.LatticeDatabase(os.path.join(tempfile.mkdtemp(), "up.db"), origin="upload-test")


@pytest.fixture
def pending(db):
    """An artifact mid-upload whose bytes are in the local CAS, with no object store anywhere."""
    cid = ws.create_container(db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="c").id
    aid = "upload-1"
    ctx = {
        "upload": {"status": "uploading", "s3_key": "k", "mode": "proxied", "progress": 1.0},
        "content_key": "k",
        "content_cas_ref": CAS_REF,
        "content_sha256": "d" * 64,
        "content_type": "text/plain",
        "size": 13,
    }
    store.create_artifact(db, ArtifactEntity(
        id=aid, root_id=aid, collection_id=cid, created_by="u",
        state=ArtifactEntity.STATE_COMMITTED, name=aid,
        context=json.dumps(ctx)))
    return cid, aid


def _no_object_store(monkeypatch):
    """Every object-store answer is empty — the state of a node that has none configured."""
    import mantle.services.content_service as cs
    monkeypatch.setattr(cs, "head_object", lambda key: None)
    monkeypatch.setattr(cs, "persist_object_to_durable", lambda key: False)
    return cs


def test_completion_accepts_content_the_local_cas_holds(db, pending, monkeypatch):
    cid, aid = pending
    cs = _no_object_store(monkeypatch)
    monkeypatch.setattr(cs, "local_content_has", lambda ref: ref == CAS_REF)

    with acting_as("u", principal_type="user"):
        result = ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=aid, status_value="complete")

    ctx = json.loads(result.context)
    assert "upload" not in ctx, "a completed upload leaves no pending upload block"


def test_completion_still_refuses_when_neither_tier_has_the_bytes(db, pending, monkeypatch):
    """The gate is not removed — only asked to consult both tiers rather than the mirror alone."""
    cid, aid = pending
    cs = _no_object_store(monkeypatch)
    monkeypatch.setattr(cs, "local_content_has", lambda ref: False)

    with acting_as("u", principal_type="user"):
        with pytest.raises(HTTPException) as exc:
            ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=aid, status_value="complete")

    assert exc.value.status_code == 400
    # The message names both tiers: "not in S3" sent readers to a store the write never needed.
    assert "CAS" in str(exc.value.detail)


def test_a_mirror_that_answers_is_still_believed(db, pending, monkeypatch):
    """With an object store present its metadata still wins — size and type come from the head."""
    cid, aid = pending
    import mantle.services.content_service as cs
    monkeypatch.setattr(cs, "head_object",
                        lambda key: {"ContentLength": 99, "ContentType": "application/pdf"})
    monkeypatch.setattr(cs, "persist_object_to_durable", lambda key: False)
    monkeypatch.setattr(cs, "local_content_has", lambda ref: False)

    with acting_as("u", principal_type="user"):
        result = ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=aid, status_value="complete")

    ctx = json.loads(result.context)
    assert ctx["size"] == 99
    assert ctx["content_type"] == "application/pdf"
