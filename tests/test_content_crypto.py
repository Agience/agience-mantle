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
