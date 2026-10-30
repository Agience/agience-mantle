"""A title is part of an artifact's content: sealed under the body's key at rest, restored for an
authorized reader, and carried by no event.

2026-10-19 (John): "Title is really just part of content, and the content type definition should
provide for that. It's a field inherited from the root artifact type." Measured 2026-10-18 on
staging: a title was in plain text in the artifact row (`vertex.doc`) and in the event log.

Runs on a real store with real content encryption.
"""
from __future__ import annotations

import json

from fastapi.responses import Response
from starlette.requests import Request

from mantle.db import doc_boundary
from mantle.events import event_bus
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import USER, real_node

import pytest

WORD = "quillfeather"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _auth():
    return AuthContext(principal_id=USER, principal_type="user", user_id=USER)


async def _note(db, title=f"Board figures {WORD}"):
    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts", "query_string": b""})
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type="application/vnd.agience.collection+json",
                                    name="Shelf").id
        doc = await artifacts.create_artifact(
            request=req, response=Response(), auth=_auth(), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=shelf, content="figures",
                                                 context=json.dumps({"title": title, "kind": "note"})))
    return shelf, doc


def _raw(db, artifact_id):
    return db.artifacts.get_artifact(artifact_id)


async def test_the_stored_row_carries_the_title_only_sealed(db):
    _, doc = await _note(db)
    raw = _raw(db, doc["id"])
    assert WORD not in json.dumps(raw), "the title is in the stored row in plain text"
    ctx = json.loads(raw["context"])
    # Sealing is the default and structure the exception (John, 2026-10-29): `kind` is sealed too.
    assert set(ctx) <= doc_boundary.STRUCTURAL_CONTEXT_KEYS | {doc_boundary.SEALED}, sorted(ctx)
    assert "title" not in ctx and "kind" not in ctx
    assert set(ctx[doc_boundary.SEALED]) == {"p", "blob"}


async def test_an_authorized_reader_gets_the_title_back(db):
    shelf, doc = await _note(db)
    with acting_as(USER, principal_type="user"):
        read = await artifacts.read_artifact(artifact_id=doc["id"], auth=_auth(), store_db=db)
        listed = ws.list_workspace_artifacts(db, USER, shelf)
    assert json.loads(read["context"])["title"] == f"Board figures {WORD}"
    assert doc_boundary.SEALED not in json.loads(read["context"])
    assert [json.loads(a.context)["title"] for a in listed] == [f"Board figures {WORD}"]


async def test_a_new_title_replaces_the_old_and_an_edit_without_one_keeps_it(db):
    _, doc = await _note(db)
    with acting_as(USER, principal_type="user"):
        await artifacts.update_artifact(artifact_id=doc["id"], auth=_auth(), store_db=db,
                                        body=artifacts.UpdateArtifactRequest(
                                            context=json.dumps({"title": "Renamed", "kind": "note"})))
        await artifacts.update_artifact(artifact_id=doc["id"], auth=_auth(), store_db=db,
                                        body=artifacts.UpdateArtifactRequest(content="new figures"))
        read = await artifacts.read_artifact(artifact_id=doc["id"], auth=_auth(), store_db=db)
    assert json.loads(read["context"])["title"] == "Renamed"
    assert "Renamed" not in json.dumps(_raw(db, doc["id"]))


async def test_a_reader_the_oracle_will_not_serve_gets_no_title_and_no_error(db, monkeypatch):
    """Fails soft: a reader the key oracle refuses (a background path with no acting principal)
    reads the structure, never the ciphertext and never an exception. `real_node` hands every
    reader every key, so the refusal is made here as the oracle makes it."""
    from mantle.services import content_crypto
    from mantle.services.acting_principal import KeyCustodyDenied
    _, doc = await _note(db)

    def refuse(*a, **k):
        raise KeyCustodyDenied("no acting principal")
    monkeypatch.setattr(content_crypto, "decrypt_content", refuse)
    raw = dict(_raw(db, doc["id"]))
    doc_boundary.unseal_content_fields(raw)
    ctx = json.loads(raw["context"])
    assert "title" not in ctx and "kind" not in ctx
    # The blob stays, so a save of what this reader holds cannot erase the sealed fields
    # (`test_a_failed_unseal_never_erases_what_it_seals.py`); a response drops it (`_artifact_body`).
    assert doc_boundary.SEALED in ctx


def test_the_change_feed_carries_the_structure_alone():
    ct = {"content_type": "text/plain"}
    sealed = {"id": "a", "name": "n", "context": json.dumps({"title": "t", "sealed": {"p": "x", "blob": "y"},
                                                             "kind": "k", **ct})}
    plain = {"id": "b", "description": "d", "context": {"title": "t", "kind": "k", "tags": ["x"], **ct}}
    out = event_bus.redact_content({"artifact": sealed, "artifacts": [plain]})
    assert json.loads(out["artifact"]["context"]) == ct and "name" not in out["artifact"]
    assert out["artifacts"][0]["context"] == ct and "description" not in out["artifacts"][0]


async def test_a_title_stored_before_sealing_is_sealed_by_the_pass(db):
    from mantle.system.seal_titles import seal_stored_titles
    _, doc = await _note(db, title="Settled")
    raw = dict(_raw(db, doc["id"]))
    ctx = json.loads(raw["context"])
    ctx.pop(doc_boundary.SEALED)
    ctx["title"] = f"Old {WORD}"                       # as a row written before sealing holds it
    raw["context"] = json.dumps(ctx)
    db.artifacts.put_artifact(raw)
    assert WORD in json.dumps(_raw(db, doc["id"])), "the control: the old row is plain"

    assert seal_stored_titles(db, dry_run=True)["plain"] >= 1
    counts = seal_stored_titles(db)
    assert counts["sealed"] >= 1 and counts["left_plain"] == 0
    assert WORD not in json.dumps(_raw(db, doc["id"]))
    with acting_as(USER, principal_type="user"):
        read = await artifacts.read_artifact(artifact_id=doc["id"], auth=_auth(), store_db=db)
    assert json.loads(read["context"])["title"] == f"Old {WORD}"
    assert seal_stored_titles(db, dry_run=True)["plain"] == 0


async def test_the_reverse_puts_the_titles_back_in_plain_text_for_a_rollback(db):
    """Before Mantle is rolled back to a version that does not read sealed titles, the release
    unseals them; the forward pass seals them again. 2026-10-20."""
    from mantle.system.seal_titles import seal_stored_titles, unseal_stored_titles
    _, doc = await _note(db, title=f"Kept {WORD}")
    assert WORD not in json.dumps(_raw(db, doc["id"]))

    counts = unseal_stored_titles(db)
    assert counts["unsealed"] >= 1 and counts["denied"] == 0
    ctx = json.loads(_raw(db, doc["id"])["context"])
    assert ctx["title"] == f"Kept {WORD}" and doc_boundary.SEALED not in ctx, "as an older Mantle reads it"
    assert unseal_stored_titles(db, dry_run=True)["sealed"] == 0

    assert seal_stored_titles(db)["sealed"] >= 1
    assert WORD not in json.dumps(_raw(db, doc["id"])), "and forward again"
    with acting_as(USER, principal_type="user"):
        read = await artifacts.read_artifact(artifact_id=doc["id"], auth=_auth(), store_db=db)
    assert json.loads(read["context"])["title"] == f"Kept {WORD}"


async def test_a_title_the_oracle_will_not_open_stays_sealed_rather_than_emptied(db, monkeypatch):
    from mantle.services import content_crypto
    from mantle.services.acting_principal import KeyCustodyDenied
    from mantle.system.seal_titles import unseal_stored_titles
    _, doc = await _note(db)
    before = _raw(db, doc["id"])["context"]

    def refuse(*a, **k):
        raise KeyCustodyDenied("no key")
    monkeypatch.setattr(content_crypto, "decrypt_content", refuse)
    counts = unseal_stored_titles(db)
    assert counts["denied"] >= 1 and _raw(db, doc["id"])["context"] == before


async def test_a_title_is_sealed_without_reading_the_body(db, monkeypatch):
    """2026-10-29, release 2026.10.14: the RC has no object store, and the pass read and rewrote
    each whole artifact, so 11 titles whose body lives only in the bucket could not be sealed again
    after the rollback check unsealed them. Sealing a title needs the row's key, not its body."""
    from mantle.services.content_service import ContentStoreUnavailable
    from mantle.system.seal_titles import seal_stored_titles, unseal_stored_titles
    _, doc = await _note(db, title=f"Far {WORD}")
    assert unseal_stored_titles(db)["unsealed"] >= 1
    body = _raw(db, doc["id"])["content"]

    def unreachable(*a, **k):
        raise ContentStoreUnavailable("not in this node's local content tier")
    reachable = doc_boundary._decrypt_body
    monkeypatch.setattr(doc_boundary, "_decrypt_body", unreachable)
    counts = seal_stored_titles(db)
    assert counts["left_plain"] == 0 and counts["sealed"] >= 1
    raw = _raw(db, doc["id"])
    assert WORD not in json.dumps(raw) and raw["content"] == body, "the title sealed, the body as stored"
    monkeypatch.setattr(doc_boundary, "_decrypt_body", reachable)
    with acting_as(USER, principal_type="user"):
        read = await artifacts.read_artifact(artifact_id=doc["id"], auth=_auth(), store_db=db)
    assert json.loads(read["context"])["title"] == f"Far {WORD}"


async def test_a_row_no_actor_can_read_back_is_left_plain_not_sealed_blind(db, monkeypatch):
    """2026-10-30, on the release candidate: the pass sealed 43 of the platform's rows that then no
    actor could open. A seal is written only if the actor that made it opens it again."""
    from mantle.services import content_crypto
    from mantle.services.acting_principal import KeyCustodyDenied
    from mantle.system.seal_titles import seal_stored_titles
    _, doc = await _note(db, title="Settled")
    raw = dict(_raw(db, doc["id"]))
    ctx = json.loads(raw["context"])
    ctx.pop(doc_boundary.SEALED)
    ctx["title"] = f"Old {WORD}"
    raw["context"] = json.dumps(ctx)
    db.artifacts.put_artifact(raw)

    def refuse(*a, **k):
        raise KeyCustodyDenied("no read grant")
    monkeypatch.setattr(content_crypto, "decrypt_content", refuse)    # sealing works; reading back does not
    counts = seal_stored_titles(db)
    assert counts["left_plain"] >= 1
    assert WORD in json.dumps(_raw(db, doc["id"])), "a seal nobody could open was written"
