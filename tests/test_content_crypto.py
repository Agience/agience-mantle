"""Tests for per-principal content envelope encryption (services/content_crypto.py)."""
import pytest
from cryptography.exceptions import InvalidTag

import mantle.services.content_crypto as cc

# Deterministic fake master keys (no oracle/DB needed).
_KEYS = {"alice": b"\x01" * 32, "bob": b"\x02" * 32}
def _provider(pid):
    return _KEYS[pid]


def test_round_trip():
    blob = cc.encrypt_content("alice", b"hello world", master_key_provider=_provider)
    assert cc.is_encrypted(blob) and blob[:4] == b"MEC1"
    out = cc.decrypt_content("alice", blob, master_key_provider=_provider)
    assert out == b"hello world"


def test_accepts_str_plaintext():
    blob = cc.encrypt_content("alice", "unicode ✓ text", master_key_provider=_provider)
    assert cc.decrypt_content("alice", blob, master_key_provider=_provider) == "unicode ✓ text".encode()


def test_ciphertext_is_not_plaintext():
    blob = cc.encrypt_content("alice", b"secret-body", master_key_provider=_provider)
    assert b"secret-body" not in blob


def test_legacy_plaintext_passthrough():
    # A blob without the MEC1 magic is legacy plaintext — returned unchanged.
    assert cc.decrypt_content("alice", b"plain legacy content", master_key_provider=_provider) == b"plain legacy content"
    assert not cc.is_encrypted(b"plain legacy content")


def test_wrong_principal_fails_owner_binding():
    blob = cc.encrypt_content("alice", b"alice's data", master_key_provider=_provider)
    with pytest.raises(InvalidTag):
        cc.decrypt_content("bob", blob, master_key_provider=_provider)


def test_tamper_detection():
    blob = bytearray(cc.encrypt_content("alice", b"immutable", master_key_provider=_provider))
    blob[-1] ^= 0x01  # flip a ciphertext bit
    with pytest.raises(InvalidTag):
        cc.decrypt_content("alice", bytes(blob), master_key_provider=_provider)


def test_encrypt_requires_principal():
    with pytest.raises(ValueError):
        cc.encrypt_content("", b"x", master_key_provider=_provider)


def test_nonce_is_random_per_encrypt():
    a = cc.encrypt_content("alice", b"same", master_key_provider=_provider)
    b = cc.encrypt_content("alice", b"same", master_key_provider=_provider)
    assert a != b  # random nonce → distinct ciphertexts
    assert cc.decrypt_content("alice", a, master_key_provider=_provider) == b"same"
    assert cc.decrypt_content("alice", b, master_key_provider=_provider) == b"same"


# ---------------------------------------------------------------------------
# Scoped AAD binding (owner + collection) — and its mandatory dual-read
#
# The gap: the master key is per-PRINCIPAL and the AAD was the principal too, so
# a blob moved between two collections under the same origin-root principal
# authenticated fine. The scope is now bound as well.
#
# DATA SAFETY: every negative below holds the KEY correct and varies only the
# scope, and every legacy blob written under the old owner-only binding must
# still open. Nothing here may orphan ciphertext.
# ---------------------------------------------------------------------------


def _legacy_blob(principal: str, plaintext: bytes) -> bytes:
    """A blob in the PRE-CHANGE wire form: MEC1 ‖ nonce ‖ GCM(aad=principal_id).

    Built from the primitives rather than by calling `encrypt_content`, so this
    keeps proving that pre-existing ciphertext opens even after the writer has
    moved on entirely and no code path produces the old form any more.
    """
    import os

    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = cc._content_key(principal, _provider)
    nonce = os.urandom(12)
    return b"MEC1" + nonce + AESGCM(key).encrypt(nonce, plaintext, principal.encode("utf-8"))


def test_scoped_round_trip():
    blob = cc.encrypt_content("alice", b"scoped", collection_id="col-A",
                              master_key_provider=_provider)
    assert cc.decrypt_content("alice", blob, collection_id="col-A",
                              master_key_provider=_provider) == b"scoped"


def test_right_key_wrong_collection_fails():
    """The actual defect: same key, same owner, different collection → must fail.

    The dual-read fallback cannot rescue this one, because the blob was written
    under the scoped binding — the legacy AAD does not open it either.
    """
    blob = cc.encrypt_content("alice", b"col-A only", collection_id="col-A",
                              master_key_provider=_provider)
    # Sanity: the key is correct — the right scope opens it.
    assert cc.decrypt_content("alice", blob, collection_id="col-A",
                              master_key_provider=_provider) == b"col-A only"
    for wrong in ("col-B", "col-a", "col-A ", ""):
        with pytest.raises((InvalidTag, ValueError)):
            cc.decrypt_content("alice", blob, collection_id=wrong or None,
                               master_key_provider=_provider)


def test_scoped_aad_fields_cannot_be_repartitioned():
    """`principal|collection` would let ("a:b", "c") and ("a", "b:c") collide.

    The length prefix makes the split unambiguous, so a principal id containing
    the separator cannot forge another owner/scope pair.
    """
    a = cc._scoped_aad("alice:x", "col")
    b = cc._scoped_aad("alice", "x:col")
    assert a != b


def test_legacy_ciphertext_still_opens_dual_read():
    """PROOF: ciphertext written before the scoped binding is not orphaned."""
    blob = _legacy_blob("alice", b"written before the change")
    # No scope supplied (content_service path) — unchanged behaviour.
    assert cc.decrypt_content("alice", blob, master_key_provider=_provider) == \
        b"written before the change"
    # Scope supplied (doc_boundary path) — the scoped AAD misses, the fallback opens it.
    assert cc.decrypt_content("alice", blob, collection_id="col-A",
                              master_key_provider=_provider) == b"written before the change"


def test_legacy_fallback_is_counted_only_on_success():
    """`legacy_aad_reads` is the operator's signal for when the fallback can go."""
    cc.legacy_aad_reads = 0
    cc.decrypt_content("alice", _legacy_blob("alice", b"old"), collection_id="col-A",
                       master_key_provider=_provider)
    assert cc.legacy_aad_reads == 1
    # A new-form blob must NOT count — it opened on the scoped binding.
    cc.decrypt_content(
        "alice",
        cc.encrypt_content("alice", b"new", collection_id="col-A", master_key_provider=_provider),
        collection_id="col-A", master_key_provider=_provider)
    assert cc.legacy_aad_reads == 1
    # A genuine failure must NOT count either — it is not evidence of old ciphertext.
    with pytest.raises(InvalidTag):
        cc.decrypt_content("bob", _legacy_blob("alice", b"old"), collection_id="col-A",
                           master_key_provider=_provider)
    assert cc.legacy_aad_reads == 1


def test_a_legacy_blob_opens_under_any_scope():
    """The owner-only binding carries no scope, so the collection presented with it
    cannot matter — which is exactly what makes the fallback safe to keep and what
    stops a legacy blob being orphaned by the scope it is read under."""
    blob = _legacy_blob("alice", b"scopeless")
    for scope in ("col-NEW", "col-OTHER", None):
        assert cc.decrypt_content("alice", blob, collection_id=scope,
                                  master_key_provider=_provider) == b"scopeless"


def test_a_scoped_blob_does_not_travel_between_collections():
    """The fallback does NOT make a v2 blob portable, and no wording should suggest it.

    `MEC1` records the nonce and nothing else, so a blob written for `col-A` and read
    under `col-B` has no old scope to recover: the scoped AAD misses and the legacy
    owner-only AAD misses too. That is the binding holding — "authenticates only in the
    slot it was written for" cannot coexist with opening in another slot. Content
    changes collection by being re-encrypted on the write path.
    """
    blob = cc.encrypt_content("alice", b"written for col-A", collection_id="col-A",
                              master_key_provider=_provider)
    with pytest.raises(InvalidTag):
        cc.decrypt_content("alice", blob, collection_id="col-B",
                           master_key_provider=_provider)
    # ...and dropping the scope entirely does not reach the legacy binding either.
    with pytest.raises(InvalidTag):
        cc.decrypt_content("alice", blob, master_key_provider=_provider)


def test_wrong_owner_still_fails_under_the_scoped_binding():
    blob = cc.encrypt_content("alice", b"alice's", collection_id="col-A",
                              master_key_provider=_provider)
    with pytest.raises(InvalidTag):
        cc.decrypt_content("bob", blob, collection_id="col-A", master_key_provider=_provider)


def test_require_encrypted_refuses_the_plaintext_downgrade():
    """The `if not is_encrypted(blob): return bytes(blob)` branch, closed on demand.

    It is a legitimate mixed-content path for the S3 object store (objects predate
    envelope encryption and carry no flag). It is an unauthenticated DOWNGRADE for
    any caller that independently knows the blob is encrypted — an attacker with
    store-write access could swap authenticated ciphertext for chosen plaintext.
    """
    assert cc.decrypt_content("alice", b"legacy plaintext",
                              master_key_provider=_provider) == b"legacy plaintext"
    with pytest.raises(ValueError, match="not Mantle-encrypted"):
        cc.decrypt_content("alice", b"legacy plaintext", require_encrypted=True,
                           master_key_provider=_provider)
    # A real blob is unaffected by the flag.
    blob = cc.encrypt_content("alice", b"real", collection_id="col-A",
                              master_key_provider=_provider)
    assert cc.decrypt_content("alice", blob, collection_id="col-A", require_encrypted=True,
                              master_key_provider=_provider) == b"real"
