"""Artifact content crypto must never silently corrupt or silently downgrade.

Three invariants are pinned:

1. A failed decryption raises rather than returning ciphertext as plaintext.
2. `Artifact` models `content_encrypted`, so the flag survives the storage
   round trip.
3. A failed encryption raises rather than storing plaintext silently: a blob
   without the `MEC1` magic reads back as legacy plaintext, so silent
   degradation would otherwise be unrecoverable.

(1) and (2) together prevent a destruction chain: if a read failed to decrypt
and the flag were dropped, a subsequent save would encrypt the ciphertext a
second time, making the original plaintext unrecoverable.
`test_failed_decrypt_then_save_does_not_double_encrypt` exercises that chain
end to end.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mantle.db import backend as db_store
from mantle.db import doc_boundary
from mantle.entities.artifact import Artifact


# ---------------------------------------------------------------------------
# 2. The flag survives the round trip
# ---------------------------------------------------------------------------


def test_artifact_models_content_encrypted():
    """The content_encrypted flag survives to_dict/from_dict round trips."""
    art = Artifact(id="a1", content="ciphertext", created_by="owner", content_encrypted=True)

    assert art.content_encrypted is True
    assert art.to_dict().get("content_encrypted") is True, "flag must survive to_dict()"

    round_tripped = Artifact.from_dict(art.to_dict())
    assert round_tripped.content_encrypted is True, "flag must survive from_dict()"


def test_plain_artifact_does_not_carry_the_flag():
    """Ordinary (unencrypted) artifacts do not carry the content_encrypted flag."""
    art = Artifact(id="a1", content="hello", created_by="owner")

    assert art.content_encrypted is False
    assert "content_encrypted" not in art.to_dict()
    assert Artifact.from_dict(art.to_dict()).content_encrypted is False


# ---------------------------------------------------------------------------
# 1. A failed decrypt never yields ciphertext-as-content
# ---------------------------------------------------------------------------


def test_strict_decrypt_raises_rather_than_returning_ciphertext():
    raw = {"_key": "a1", "content": "Y2lwaGVydGV4dA==", "content_encrypted": True, "created_by": "o"}

    with patch("mantle.services.content_crypto.decrypt_content", side_effect=ValueError("no key")):
        with pytest.raises(doc_boundary.ContentDecryptionError):
            doc_boundary.decrypt_artifact_content(raw)


def test_non_strict_decrypt_drops_content_and_keeps_the_flag():
    """List paths must not fail the page — but must not serve ciphertext either."""
    raw = {"_key": "a1", "content": "Y2lwaGVydGV4dA==", "content_encrypted": True, "created_by": "o"}

    with patch("mantle.services.content_crypto.decrypt_content", side_effect=ValueError("no key")):
        doc_boundary.decrypt_artifact_content(raw, strict=False)

    assert raw["content"] is None, "ciphertext must never be surfaced as content"
    assert raw["content_encrypted"] is True, (
        "the flag must stay set so a write-back cannot double-encrypt"
    )


def test_successful_decrypt_clears_the_flag_positive_control():
    """The happy path decrypts successfully and clears the flag."""
    raw = {"_key": "a1", "content": "Y2lwaGVydGV4dA==", "content_encrypted": True, "created_by": "o"}

    with patch("mantle.services.content_crypto.decrypt_content", return_value=b"plaintext"):
        doc_boundary.decrypt_artifact_content(raw)

    assert raw["content"] == "plaintext"
    assert raw["content_encrypted"] is False


# ---------------------------------------------------------------------------
# 3. A failed encrypt never stores plaintext
# ---------------------------------------------------------------------------


def test_failed_encrypt_raises_rather_than_storing_plaintext():
    doc = {"_key": "a1", "content": "secret plaintext", "created_by": "owner"}

    with patch("mantle.services.content_crypto.encrypt_content", side_effect=ValueError("no key")):
        with pytest.raises(doc_boundary.ContentEncryptionError):
            doc_boundary.encrypt_artifact_content(doc)

    assert doc["content"] == "secret plaintext"
    assert not doc.get("content_encrypted"), (
        "must not claim encryption it did not perform"
    )


def test_successful_encrypt_sets_the_flag_positive_control():
    doc = {"_key": "a1", "content": "secret", "created_by": "owner"}

    with patch("mantle.services.content_crypto.encrypt_content", return_value=b"MEC1blob"):
        doc_boundary.encrypt_artifact_content(doc)

    assert doc["content_encrypted"] is True
    assert doc["content"] != "secret", "content must be replaced with the encoded blob"


def test_encrypt_is_idempotent():
    """Already-encrypted content must not be encrypted twice."""
    doc = {"_key": "a1", "content": "already", "created_by": "owner", "content_encrypted": True}

    with patch("mantle.services.content_crypto.encrypt_content") as enc:
        doc_boundary.encrypt_artifact_content(doc)

    enc.assert_not_called()
    assert doc["content"] == "already"


# ---------------------------------------------------------------------------
# The chain, end to end
# ---------------------------------------------------------------------------


def test_failed_decrypt_then_save_does_not_double_encrypt():
    """Read a doc whose decryption fails, build the entity, save it back, and
    assert the ciphertext is not encrypted a second time.
    """
    stored_ciphertext = "Y2lwaGVydGV4dA=="
    raw = {
        "_key": "a1",
        "content": stored_ciphertext,
        "content_encrypted": True,
        "created_by": "owner",
        "root_id": "a1",
    }

    # Step 1-2: read with a broken key, non-strict (the survivable path).
    with patch("mantle.services.content_crypto.decrypt_content", side_effect=ValueError("no key")):
        doc_boundary.decrypt_artifact_content(raw, strict=False)

    entity = Artifact.from_dict({**raw, "id": raw["_key"]})
    assert entity.content_encrypted is True, (
        "flag was dropped on the way into the entity — the next save re-encrypts"
    )

    # Step 3: the caller saves the entity back.
    with patch("mantle.services.content_crypto.encrypt_content") as enc:
        doc = db_store.to_lattice_doc(entity)

    # `assert not enc.called, "..."` — `enc.assert_not_called(), "..."` is a tuple expression, so
    # the message below was dead code rather than the failure explanation it looks like.
    assert not enc.called, (
        "content was encrypted a second time, which would make the original "
        "plaintext unrecoverable"
    )
    assert doc.get("content_encrypted") is True


# ---------------------------------------------------------------------------
# 4. A flagged doc whose blob is NOT encrypted is a hard error
#
# `content_crypto.decrypt_content` returns a blob without the MEC1 magic unchanged.
# That is a legitimate compatibility path for `content_service.get_bytes_decrypted` —
# S3 objects predating envelope encryption genuinely carry neither flag nor magic. It
# is a BYPASS on the artifact path, which only calls in when `content_encrypted` is
# already set: without `require_encrypted=True`, an attacker with store-write access
# could replace authenticated ciphertext with chosen plaintext and have it served as
# artifact content, with no tag left to fail.
#
# Nothing is mocked below: the real `content_crypto` runs, and the magic-check fires
# before any key lookup, which is exactly why the substitution would have been free.
# ---------------------------------------------------------------------------


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode("ascii")


CHOSEN = b"attacker-chosen plaintext"


def test_flagged_doc_with_an_unencrypted_blob_raises_rather_than_returning_it():
    raw = {"_key": "a1", "content": _b64(CHOSEN), "content_encrypted": True,
           "created_by": "o", "collection_id": "col-1"}

    with pytest.raises(doc_boundary.ContentDecryptionError):
        doc_boundary.decrypt_artifact_content(raw)

    assert raw["content"] != CHOSEN.decode(), (
        "unauthenticated bytes were surfaced as artifact content"
    )
    assert raw["content_encrypted"] is True, "the flag must survive a failed decrypt"


def test_flagged_doc_with_an_unencrypted_blob_is_dropped_on_the_non_strict_path():
    """List/stream paths must not fail the page — nor serve the substituted bytes."""
    raw = {"_key": "a1", "content": _b64(CHOSEN), "content_encrypted": True,
           "created_by": "o", "collection_id": "col-1"}

    doc_boundary.decrypt_artifact_content(raw, strict=False)

    assert raw["content"] is None
    assert raw["content_encrypted"] is True


def test_the_passthrough_itself_is_still_available_to_callers_that_need_it():
    """The control: `require_encrypted` is what closed it, not a blanket prohibition.

    Same bytes, same function — the flag on the CALL is the only difference. This is the
    behaviour `content_service.get_bytes_decrypted` still depends on for pre-envelope S3
    objects, so it must keep working while the artifact path refuses it.
    """
    from mantle.services import content_crypto

    assert content_crypto.decrypt_content("o", CHOSEN) == CHOSEN
    with pytest.raises(ValueError):
        content_crypto.decrypt_content("o", CHOSEN, require_encrypted=True)
