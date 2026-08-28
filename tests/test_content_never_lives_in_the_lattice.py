"""Content does not live in the lattice. The vertex carries an ADDRESS; the bytes live in the CAS.

Why this file exists. Every piece of the content tier was already built and correct — the
`content_ref` column in `db/schema.py`, `db/vertex.py` reading it to tell a re-describe from a new
version, `db/test_lattice.py` covering both, `TieredContentStore` with dedup by content address,
`get_bytes_decrypted(cas_ref=...)` serving it back, `shard/content_tier.promote_local_content`
draining it to S3 — and nothing in the service layer ever set the ref. The artifact write path
called `put_text_direct`, which is deliberately ``cas=False``, so on a node with no object store
it raised and the caller kept the bytes inline, where `db/doc_boundary.encrypt_artifact_content`
encrypted them into the document.

Measured on 71/dev before this was fixed:

    vertices                          709
    with a content_ref                  0
    docs carrying inline content      240
    content bytes in the lattice      7.2 MB of 7.9 MB — 92% of all doc bytes

It cost nothing visible. Reads worked, writes returned 200, the suite was green, and the only
symptom was a log line reporting normal operation as a failure. That is exactly the shape of
regression a test has to catch, because nothing else will: **the correct behaviour and the broken
behaviour are indistinguishable from the outside until the store is too big.**

So these tests assert the property directly rather than any one path to it. A future refactor may
move where the ref is computed; it may not move the bytes back into the row.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.services import workspace_service as ws_svc


_REF = "cas/" + "b" * 64

# This file tests the real write path, so it opts out of the addressed stub `conftest` installs
# for every other unit test.
pytestmark = pytest.mark.real_content_store


def test_the_entity_carries_the_ref_through_a_storage_round_trip():
    """A ref that `from_dict` dropped would be recomputed as None on the next save, and the row
    would quietly go back to carrying its own bytes — the same failure as before, re-entered by
    a different door. `content_encrypted` and `origin_root` are modeled here for this reason."""
    a = ArtifactEntity(id="a1", content="body", content_ref=_REF)
    assert a.to_dict()["content_ref"] == _REF
    assert ArtifactEntity.from_dict(a.to_dict()).content_ref == _REF
    assert ArtifactEntity(id="a2").content_ref is None


def test_a_write_with_an_owner_addresses_the_bytes_rather_than_inlining_them():
    """The CAS leg runs first and its ref is what comes back."""
    with patch("mantle.services.content_service.put_bytes_encrypted",
               return_value=_REF) as put_cas:
        key, inline, ref = ws_svc._store_content_in_s3(
            "art-1", "the body", '{"content_type":"text/plain"}',
            owner_id="owner-1", collection_id="col-1",
        )
    assert ref == _REF, "the write must return the content address it stored the bytes at"
    assert put_cas.call_args.kwargs["cas"] is True, (
        "cas=True is the whole point — `put_text_direct` is cas=False and is what put 92% of a "
        "live lattice inside the document store"
    )
    assert put_cas.call_args.kwargs["collection_id"] == "col-1"
    assert key == "artifacts/art-1.content"


def test_the_address_is_recorded_where_BOTH_readers_look():
    """`GET /artifacts/{id}/content` and `mirror_drain` read `context["content_cas_ref"]`;
    `db/vertex.py` reads the doc's `content_ref`. Setting one and not the other leaves a reader
    blind, so they move together."""
    ctx = ws_svc._record_content_address('{"content_type":"text/plain"}',
                                         "artifacts/art-1.content", _REF)
    assert json.loads(ctx)["content_cas_ref"] == _REF
    assert json.loads(ctx)["content_key"] == "artifacts/art-1.content"
def test_create_stamps_the_ref_onto_the_artifact_it_persists():
    """The end-to-end property: what reaches the store carries the address.

    Asserted on the ENTITY handed to `store.create_artifact`, because that is what becomes the
    document — and the document is what `db/vertex.py` reads `content_ref` out of.
    """
    db = MagicMock()
    with (
        patch("mantle.services.workspace_service.get_workspace", return_value=MagicMock()),
        patch("mantle.services.workspace_service._store_content_in_s3",
              return_value=("artifacts/x.content", "", _REF)),
        patch("mantle.services.workspace_service.store.create_artifact") as created,
        patch("mantle.services.workspace_service.store.add_artifact_to_collection"),
        patch("mantle.services.workspace_service.store.get_last_order_key", return_value=None),
        patch("mantle.services.workspace_service._link_to_target_collections"),
        patch("mantle.services.workspace_service._emit_event"),
    ):
        ws_svc.create_workspace_artifact(
            db=db, user_id="u1", workspace_id="ws-1", context="{}", content="the body",
            enqueue_index=False,
        )
    entity = created.call_args.args[1]
    assert entity.content_ref == _REF, (
        "the artifact persisted with no content_ref is the regression this file exists for: its "
        "bytes end up encrypted into the lattice document instead of addressed in the CAS"
    )


def test_an_empty_artifact_needs_no_address():
    """No content, no bytes, no ref — and no NameError from a variable only bound under `if`."""
    db = MagicMock()
    with (
        patch("mantle.services.workspace_service.get_workspace", return_value=MagicMock()),
        patch("mantle.services.workspace_service.store.create_artifact") as created,
        patch("mantle.services.workspace_service.store.add_artifact_to_collection"),
        patch("mantle.services.workspace_service.store.get_last_order_key", return_value=None),
        patch("mantle.services.workspace_service._link_to_target_collections"),
        patch("mantle.services.workspace_service._emit_event"),
    ):
        ws_svc.create_workspace_artifact(
            db=db, user_id="u1", workspace_id="ws-1", context="{}", content="",
            enqueue_index=False,
        )
    assert created.call_args.args[1].content_ref is None


@pytest.mark.parametrize("path", ["put_text_direct"])
def test_the_direct_leg_stays_cas_false(path):
    """`put_text_direct` is correct AS ITSELF and must not be 'fixed' by flipping its flag.

    Its caller keeps the authoritative copy, so a CAS object written for it would be one nothing
    references. The bug was never that this function was wrong — it was that the artifact write
    path used it for content whose authoritative copy was supposed to be the CAS. Flipping the
    flag here would create unreferenced objects instead of fixing anything.
    """
    import inspect

    from mantle.services import content_service

    src = inspect.getsource(getattr(content_service, path))
    assert "cas=False" in src, (
        "put_text_direct must keep cas=False; route artifact content through "
        "put_bytes_encrypted(cas=True) instead"
    )


# ── the persistence boundary: the pair that makes "one copy, in the CAS" true ────────────────
#
# `db/doc_boundary` is where it is enforced, and it has to be a PAIR. Keeping content out of the
# document without hydrating it back on read empties every consumer of `artifact.content` —
# indexing, recall previews, `GET /artifacts/{id}` — and each of those fails silently, as an
# artifact that merely looks empty. Hydrating without keeping it out just stores it twice.

def test_an_addressed_doc_stores_no_body():
    """The write half. A doc carrying a ref must reach storage with an empty `content`."""
    from mantle.db import doc_boundary

    doc = {"id": "a1", "content": "the body", "content_ref": _REF, "created_by": "u1"}
    doc_boundary.encrypt_artifact_content(doc)
    assert doc["content"] == "", (
        "an addressed artifact must not also carry its bytes in the document — that is what made "
        "92% of a live lattice content"
    )
    assert doc["content_encrypted"] is False, "there is nothing inline left to have encrypted"
def test_the_read_half_hydrates_from_the_address():
    """The read half. A row with a ref and no body returns the bytes from the CAS."""
    from mantle.db import doc_boundary

    doc = {"id": "a1", "content": "", "content_ref": _REF, "created_by": "u1",
           "collection_id": "col-1", "context": json.dumps({"content_key": "artifacts/a1.content"})}
    with patch("mantle.services.content_service.get_bytes_decrypted",
               return_value=b"the body") as get:
        doc_boundary.decrypt_artifact_content(doc)
    assert doc["content"] == "the body"
    assert get.call_args.kwargs["cas_ref"] == _REF, "the read must go through the content address"


def test_an_unreadable_address_is_an_ERROR_not_an_empty_artifact():
    """The dangerous failure is the quiet one.

    A body that cannot be fetched must not come back as `content: ""` — that is indistinguishable
    from an artifact with nothing in it, and it would propagate into the index and the previews as
    though the content had been deleted. `strict` carries the same meaning it does for a failed
    decrypt: a single read raises, a list page drops the body and stays visibly incomplete.
    """
    from mantle.db import doc_boundary

    doc = {"id": "a1", "content": "", "content_ref": _REF, "created_by": "u1", "context": "{}"}
    with patch("mantle.services.content_service.get_bytes_decrypted",
               side_effect=OSError("tier is unreachable")):
        with pytest.raises(doc_boundary.ContentDecryptionError):
            doc_boundary.decrypt_artifact_content(dict(doc), strict=True)

        loose = dict(doc)
        doc_boundary.decrypt_artifact_content(loose, strict=False)
        assert loose["content"] == ""      # a page must not fail whole; the row is incomplete


def test_a_write_then_read_round_trip_keeps_the_bytes_out_of_the_doc():
    """Both halves together, which is the only way the property actually holds."""
    from mantle.db import doc_boundary

    body = "the axolotl reads only from the content address"
    doc = {"id": "a1", "content": body, "content_ref": _REF, "created_by": "u1",
           "collection_id": "col-1", "context": json.dumps({"content_key": "artifacts/a1.content"})}
    doc_boundary.encrypt_artifact_content(doc)
    assert body not in json.dumps(doc), "the persisted document must not contain the body anywhere"

    with patch("mantle.services.content_service.get_bytes_decrypted",
               return_value=body.encode("utf-8")):
        doc_boundary.decrypt_artifact_content(doc)
    assert doc["content"] == body, "and the reader must still see it"


def test_there_is_no_path_that_writes_a_body_into_a_document():
    """The property, stated as the absence it is: `_store_content_in_s3` has one destination.

    No branch returns bytes for a row to carry, and no threshold decides when a body is 'small
    enough' to keep — a document destination beside the CAS and the object store is what would make
    the lattice the content store. Asserted on the source, so an introduced fallback fails here
    rather than years later on disk.
    """
    import inspect

    src = inspect.getsource(ws_svc._store_content_in_s3)
    assert "put_bytes_encrypted" in src
    assert "put_text_direct" not in src, (
        "put_text_direct is cas=False — routing artifact content through it is what put 92% of a "
        "live lattice inside the documents"
    )
    assert "_MAX_INLINE_BYTES" not in src, "nothing is inline, so no size decides anything"

def test_an_addressed_body_never_reaches_the_document():
    """The property that matters: if the bytes have an address, the row carries none of them."""
    from mantle.db import doc_boundary

    doc = {"id": "a1", "content": "the body", "content_ref": _REF, "created_by": "u1"}
    doc_boundary.encrypt_artifact_content(doc)
    assert doc["content"] == ""
    assert doc["content_encrypted"] is False


def test_the_legacy_inline_path_survives_but_the_write_path_cannot_REACH_it():
    """`encrypt_artifact_content` still encrypts an UNADDRESSED body, and that is deliberate.

    Rows written before the content tier hold their bytes nowhere else, so clearing one here would
    destroy the only copy, and the inline encryption is the protection those rows have. Nothing
    arrives in that state now: `_store_content_in_s3` addresses every body it stores, so the branch
    is reachable only by a read-modify-write of existing data.

    Asserted as reachability rather than absence, because deleting the branch would be a data
    loss and keeping it must not become a second way to fill the lattice.
    """
    import inspect

    from mantle.db import doc_boundary

    legacy = {"id": "a2", "content": "body", "created_by": "u1", "origin_root": "u1"}
    with patch("mantle.services.content_crypto.encrypt_content", return_value=b"MEC1cipher"):
        doc_boundary.encrypt_artifact_content(legacy)
    assert legacy["content_encrypted"] is True, "a legacy body must still be encrypted at rest"

    write_path = inspect.getsource(ws_svc._store_content_in_s3)
    assert "cas=True" in write_path, (
        "every body the write path stores must be addressed — that is what makes the legacy "
        "branch unreachable rather than merely unused"
    )
