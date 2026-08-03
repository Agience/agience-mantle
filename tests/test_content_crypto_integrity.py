"""Artifact content crypto must never silently corrupt or silently downgrade.

Three defects that compose into permanent, invisible data loss on a LIVE system:

1. **Failed decryption returned ciphertext with HTTP 200.** `content` was left
   holding base64 ciphertext and handed to the caller as if it were plaintext.
2. **`Artifact` did not model `content_encrypted`.** The flag was dropped on the
   storage round trip.
3. **Failed encryption stored plaintext silently.** A blob without the `MEC1`
   magic reads back as "legacy plaintext", so the degradation was invisible
   forever after.

(1) + (2) chain into destruction: read fails to decrypt -> ciphertext in
`content` -> flag dropped -> caller saves -> `_encrypt_artifact_content` sees
content with no flag and encrypts the CIPHERTEXT again -> the original plaintext
is unrecoverable. `test_failed_decrypt_then_save_does_not_double_encrypt` is that
chain, run end to end.
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
    """THE LOAD-BEARING GUARD: without this the destruction chain reconnects."""
    art = Artifact(id="a1", content="ciphertext", created_by="owner", content_encrypted=True)

    assert art.content_encrypted is True
    assert art.to_dict().get("content_encrypted") is True, "flag must survive to_dict()"

    round_tripped = Artifact.from_dict(art.to_dict())
    assert round_tripped.content_encrypted is True, "flag must survive from_dict()"


def test_plain_artifact_does_not_carry_the_flag():
    """POSITIVE CONTROL: ordinary artifacts stay clean."""
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
    """POSITIVE CONTROL: the happy path still decrypts and clears the flag."""
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
    """THE DESTRUCTION CHAIN. This is the test that matters.

    Read a doc whose decryption fails, build the entity, save it back, and assert
    the ciphertext is not encrypted a second time. Before the fix each step was
    individually 'reasonable' and together they destroyed the artifact.
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

    enc.assert_not_called(), (
        "content was encrypted a SECOND time — the original plaintext is now "
        "unrecoverable, which is exactly the irreversible corruption this guards"
    )
    assert doc.get("content_encrypted") is True
