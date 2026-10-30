"""A reader the key service will not serve, and that saves what it read, leaves the sealed fields as
they were sealed.

2026-10-29, from a reading of the store: `unseal_content_fields` took the sealed blob out of the
context before decrypting it, so a reader without custody (a background pass under the platform's
principal) got a context with neither the fields nor the blob, and a save of that context erased the
title for good. The more fields are sealed, the more that loses.
"""
from __future__ import annotations

import json

from mantle.db import doc_boundary
from mantle.routers import artifacts_router as artifacts
from mantle.services import content_crypto
from mantle.services.acting_principal import acting_as

from .test_a_title_is_content_and_sealed_at_rest import WORD, _auth, _note, db  # noqa: F401
from .test_there_is_no_promotion_state import USER


def _refused(*_a, **_k):
    raise PermissionError("no grant reaches this key")


async def test_a_reader_without_custody_that_saves_keeps_the_title(db, monkeypatch):  # noqa: F811
    _, note = await _note(db)
    raw = dict(db.artifacts.get_artifact(note["id"]))

    with monkeypatch.context() as m:
        m.setattr(content_crypto, "decrypt_content", _refused)
        doc_boundary.unseal_content_fields(raw)
    ctx = json.loads(raw["context"])
    assert "title" not in ctx, "a reader without custody got the title"
    assert doc_boundary.SEALED in ctx, "the sealed blob was dropped from what the reader holds"

    db.artifacts.put_artifact(raw)          # the background pass saves what it read

    with acting_as(USER, principal_type="user"):
        read = await artifacts.read_artifact(artifact_id=note["id"], auth=_auth(), store_db=db)
    assert json.loads(read["context"])["title"] == f"Board figures {WORD}", "the save erased the title"


async def test_a_response_never_carries_the_blob(db, monkeypatch):  # noqa: F811
    _, note = await _note(db)
    raw = dict(db.artifacts.get_artifact(note["id"]))
    with monkeypatch.context() as m:
        m.setattr(content_crypto, "decrypt_content", _refused)
        doc_boundary.unseal_content_fields(raw)
    body = artifacts._artifact_body(raw)
    assert doc_boundary.SEALED not in json.loads(body["context"])
