"""Unit tests for `search.mantle.cell` (Step 2.2b.i).

The cell module is the contained AES-256-GCM piece between OracleService
keys and storage. Tests cover:

- Round-trip: encrypt → decrypt yields the original plaintext
- Tampering detection: bit-flips anywhere in the blob raise CellTampered
- Wrong-key detection: decrypting with the wrong key raises CellTampered
- Nonce uniqueness: every call produces a different blob (same plaintext, same key)
- Key validation: non-32-byte keys are rejected
- Malformed-blob detection: short blobs raise CellMalformed
- Serialization: chunk lists round-trip through JSON
- Canonical encoding: same chunks always produce the same plaintext bytes
- Mutation helpers: upsert_chunk + remove_artifact_chunks
- pack_cell + unpack_cell convenience wrappers
"""

from __future__ import annotations

import json
import os

import pytest

from mantle.search.mantle.cell import (
    CellMalformed,
    CellTampered,
    artifact_ids,
    cell_aad,
    chunk_count,
    decrypt_cell,
    deserialize_chunks,
    encrypt_cell,
    pack_cell,
    remove_artifact_chunks,
    serialize_chunks,
    unpack_cell,
    upsert_chunk,
)
from .helpers import SelfContextVerifier, self_request


@pytest.fixture
def key() -> bytes:
    """Fresh 256-bit cell key for each test."""
    return os.urandom(32)


@pytest.fixture
def aad() -> str:
    """The canonical cell AAD (``collection:cluster``) — required on every call."""
    return cell_aad("col-A", "anchor-1")


@pytest.fixture
def chunks() -> list[dict]:
    return [
        {
            "artifact_id": "art-1",
            "chunk_id": 0,
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "tokens": 12,
        },
        {
            "artifact_id": "art-1",
            "chunk_id": 1,
            "embedding": [0.5, 0.6, 0.7, 0.8],
            "tokens": 9,
        },
        {
            "artifact_id": "art-2",
            "chunk_id": 0,
            "embedding": [0.9, 0.1, 0.1, 0.1],
            "tokens": 7,
        },
    ]


# ---------------------------------------------------------------------------
# AES-GCM primitives
# ---------------------------------------------------------------------------

class TestEncryptDecrypt:
    def test_round_trip_recovers_plaintext(self, key, aad):
        plaintext = b"hello MANTLE"
        blob = encrypt_cell(plaintext, key, aad)
        assert decrypt_cell(blob, key, aad) == plaintext

    def test_blob_layout_is_nonce_then_ciphertext_tag(self, key, aad):
        """Blob starts with a 12-byte nonce + at least 16 bytes (tag) after."""
        plaintext = b"x"
        blob = encrypt_cell(plaintext, key, aad)
        # 12-byte nonce + 1-byte ciphertext + 16-byte tag = 29 minimum
        assert len(blob) == 12 + 1 + 16

    def test_nonce_is_fresh_per_call(self, key, aad):
        """Same plaintext + same key → different blob each call (no nonce reuse)."""
        plaintext = b"identical"
        a = encrypt_cell(plaintext, key, aad)
        b = encrypt_cell(plaintext, key, aad)
        assert a != b
        # First 12 bytes (the nonce) differ
        assert a[:12] != b[:12]
        # But both decrypt back to the same plaintext.
        assert decrypt_cell(a, key, aad) == decrypt_cell(b, key, aad) == plaintext

    def test_wrong_key_raises_tampered(self, key, aad):
        plaintext = b"secret content"
        blob = encrypt_cell(plaintext, key, aad)
        wrong = os.urandom(32)
        with pytest.raises(CellTampered):
            decrypt_cell(blob, wrong, aad)

    def test_bitflip_in_ciphertext_raises_tampered(self, key, aad):
        plaintext = b"some payload that is long enough"
        blob = bytearray(encrypt_cell(plaintext, key, aad))
        # Flip a bit somewhere in the ciphertext (after the nonce, before the tag).
        blob[15] ^= 0x01
        with pytest.raises(CellTampered):
            decrypt_cell(bytes(blob), key, aad)

    def test_bitflip_in_nonce_raises_tampered(self, key, aad):
        plaintext = b"some payload"
        blob = bytearray(encrypt_cell(plaintext, key, aad))
        blob[0] ^= 0x80
        with pytest.raises(CellTampered):
            decrypt_cell(bytes(blob), key, aad)

    def test_bitflip_in_tag_raises_tampered(self, key, aad):
        plaintext = b"some payload"
        blob = bytearray(encrypt_cell(plaintext, key, aad))
        blob[-1] ^= 0x01
        with pytest.raises(CellTampered):
            decrypt_cell(bytes(blob), key, aad)

    def test_short_blob_raises_malformed(self, key, aad):
        with pytest.raises(CellMalformed):
            decrypt_cell(b"too short", key, aad)

    def test_empty_blob_raises_malformed(self, key, aad):
        with pytest.raises(CellMalformed):
            decrypt_cell(b"", key, aad)

    def test_wrong_key_size_rejected(self, aad):
        plaintext = b"x"
        with pytest.raises(ValueError, match="32 bytes"):
            encrypt_cell(plaintext, b"too-short", aad)
        with pytest.raises(ValueError, match="32 bytes"):
            decrypt_cell(b"\x00" * 28, b"too-short", aad)

    def test_empty_plaintext_round_trips(self, key, aad):
        blob = encrypt_cell(b"", key, aad)
        assert decrypt_cell(blob, key, aad) == b""


# ---------------------------------------------------------------------------
# Chunk serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self, chunks):
        blob = serialize_chunks(chunks)
        assert deserialize_chunks(blob) == chunks

    def test_canonical_encoding_is_stable(self, chunks):
        """Same input → exact same bytes (sorted keys, no whitespace).

        Re-orders keys in one record to verify the encoding is independent
        of insertion order.
        """
        a = serialize_chunks(chunks)
        scrambled = [
            {k: v for k, v in sorted(c.items(), reverse=True)}
            for c in chunks
        ]
        b = serialize_chunks(scrambled)
        assert a == b

    def test_no_whitespace(self, chunks):
        """Compact encoding — no spaces between keys/values."""
        blob = serialize_chunks(chunks)
        assert b": " not in blob
        assert b", " not in blob

    def test_invalid_json_raises_malformed(self):
        with pytest.raises(CellMalformed):
            deserialize_chunks(b"this is not json")

    def test_non_utf8_raises_malformed(self):
        with pytest.raises(CellMalformed):
            deserialize_chunks(b"\xff\xfe\x00not utf-8")

    def test_non_list_raises_malformed(self):
        with pytest.raises(CellMalformed):
            deserialize_chunks(b'{"this": "is an object"}')

    def test_empty_list_round_trips(self):
        assert deserialize_chunks(serialize_chunks([])) == []


# ---------------------------------------------------------------------------
# pack / unpack convenience wrappers
# ---------------------------------------------------------------------------

class TestPackUnpack:
    def test_round_trip(self, key, chunks, aad):
        blob = pack_cell(chunks, key, aad)
        assert unpack_cell(blob, key, aad) == chunks

    def test_unpack_with_wrong_key(self, key, chunks, aad):
        blob = pack_cell(chunks, key, aad)
        with pytest.raises(CellTampered):
            unpack_cell(blob, os.urandom(32), aad)


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

class TestUpsertChunk:
    def test_appends_new_chunk(self):
        chunks = [{"artifact_id": "art-1", "chunk_id": 0}]
        result = upsert_chunk(chunks, {"artifact_id": "art-2", "chunk_id": 0})
        assert len(result) == 2
        assert result[1]["artifact_id"] == "art-2"

    def test_replaces_existing_by_artifact_and_chunk_id(self):
        chunks = [
            {"artifact_id": "art-1", "chunk_id": 0, "embedding": [0.1]},
            {"artifact_id": "art-1", "chunk_id": 1, "embedding": [0.2]},
        ]
        upsert_chunk(chunks, {
            "artifact_id": "art-1", "chunk_id": 0, "embedding": [0.99]
        })
        # In-place replacement at index 0; index 1 untouched.
        assert chunks[0]["embedding"] == [0.99]
        assert chunks[1]["embedding"] == [0.2]

    def test_distinguishes_chunks_within_same_artifact(self):
        chunks = [{"artifact_id": "art-1", "chunk_id": 0, "embedding": [0.1]}]
        upsert_chunk(chunks, {"artifact_id": "art-1", "chunk_id": 1, "embedding": [0.2]})
        # Different chunk_id → append, don't replace.
        assert len(chunks) == 2

    def test_rejects_record_missing_keys(self):
        with pytest.raises(ValueError):
            upsert_chunk([], {"artifact_id": "art-1"})  # no chunk_id
        with pytest.raises(ValueError):
            upsert_chunk([], {"chunk_id": 0})  # no artifact_id


class TestRemoveArtifactChunks:
    def test_strips_all_chunks_for_artifact(self, chunks):
        result = remove_artifact_chunks(chunks, "art-1")
        assert artifact_ids(result) == {"art-2"}
        assert len(result) == 1

    def test_no_match_is_passthrough(self, chunks):
        result = remove_artifact_chunks(chunks, "art-ghost")
        assert result == chunks

    def test_does_not_mutate_input(self, chunks):
        original_len = len(chunks)
        remove_artifact_chunks(chunks, "art-1")
        assert len(chunks) == original_len


class TestStats:
    def test_chunk_count(self, chunks):
        assert chunk_count(chunks) == 3
        assert chunk_count([]) == 0

    def test_artifact_ids_distinct(self, chunks):
        assert artifact_ids(chunks) == {"art-1", "art-2"}

    def test_artifact_ids_skips_records_without_id(self):
        weird = [{"chunk_id": 0}, {"artifact_id": "x", "chunk_id": 0}]
        assert artifact_ids(weird) == {"x"}


# ---------------------------------------------------------------------------
# Realistic cell payload — embedding + metadata
# ---------------------------------------------------------------------------

class TestRealisticCell:
    def test_full_round_trip_with_embedding_and_metadata(self, key, aad):
        chunks = [
            {
                "artifact_id": "art-1",
                "chunk_id": 0,
                "embedding": [round(i * 0.01, 4) for i in range(1536)],  # ada-002 dim
                "metadata": {
                    "title": "Test artifact",
                    "tags": ["alpha", "beta"],
                },
            },
        ]
        blob = pack_cell(chunks, key, aad)
        recovered = unpack_cell(blob, key, aad)
        assert recovered == chunks
        # Embedding floats survive JSON round-trip exactly when they're
        # representable — here we used 4-decimal rounded floats so this holds.
        assert recovered[0]["embedding"][:3] == chunks[0]["embedding"][:3]

    def test_blob_hides_plaintext(self, key, aad):
        """Sanity: the encrypted blob doesn't leak the JSON plaintext."""
        chunks = [{"artifact_id": "secret", "chunk_id": 0, "embedding": [0.0]}]
        blob = pack_cell(chunks, key, aad)
        # The string "secret" must not appear as plaintext in the blob.
        assert b"secret" not in blob
        # And the JSON bytes should be unrecognizable as JSON.
        assert b'"artifact_id"' not in blob

    def test_size_overhead_is_bounded(self, key, aad):
        """Encrypted overhead = nonce(12) + tag(16) = 28 bytes regardless of payload."""
        plaintext_5b = serialize_chunks([{"artifact_id": "a", "chunk_id": 0}])
        blob = encrypt_cell(plaintext_5b, key, aad)
        assert len(blob) == len(plaintext_5b) + 28


# ---------------------------------------------------------------------------
# Integration with OracleService cell keys
# ---------------------------------------------------------------------------

class TestOracleIntegration:
    """Smoke test: keys derived by OracleService can encrypt/decrypt cells."""

    def test_oracle_derived_key_round_trips(self):
        from cryptography.fernet import Fernet
        from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService

        oracle = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())), grant_verifier=SelfContextVerifier())
        key = oracle.derive_cell_key("owner-1", "col-A", "anchor-1", self_request("owner-1", "update"))

        chunks = [{"artifact_id": "a", "chunk_id": 0, "embedding": [0.1]}]
        aad = cell_aad("col-A", "anchor-1")
        blob = pack_cell(chunks, key, aad)
        assert unpack_cell(blob, key, aad) == chunks

        # Different collection → different cell key → can't decrypt.
        wrong_key = oracle.derive_cell_key("owner-1", "col-B", "anchor-1", self_request("owner-1", "update"))
        with pytest.raises(CellTampered):
            unpack_cell(blob, wrong_key, aad)

    def test_oracle_keys_round_trip_per_collection(self):
        from cryptography.fernet import Fernet
        from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService

        oracle = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())), grant_verifier=SelfContextVerifier())
        # One cell per (owner, collection, anchor); re-derivation recovers the key.
        encrypted = {}
        for col in ("col-A", "col-B", "col-C"):
            k = oracle.derive_cell_key("owner-1", col, "anchor-1", self_request("owner-1", "update"))
            encrypted[col] = pack_cell(
                [{"artifact_id": col, "chunk_id": 0}], k, cell_aad(col, "anchor-1")
            )

        for col, blob in encrypted.items():
            recovered = unpack_cell(
                blob,
                oracle.derive_cell_key("owner-1", col, "anchor-1", self_request("owner-1", "update")),
                cell_aad(col, "anchor-1"),
            )
            assert recovered[0]["artifact_id"] == col


# ---------------------------------------------------------------------------
# AEAD associated-data BINDING (invariant #3)
#
# The gap this class closes: every pre-existing negative here varied the *key*
# (or the ciphertext) as well as the slot, so it fired for the wrong reason. A
# binding test must hold the key CORRECT and vary only the AAD — otherwise it
# proves nothing about associated data at all.
# ---------------------------------------------------------------------------

class TestAadBinding:
    """Right key + wrong AAD must fail, before any deserialization."""

    @pytest.mark.parametrize("wrong", [
        cell_aad("col-B", "anchor-1"),        # moved to another collection
        cell_aad("col-A", "anchor-2"),        # moved to another cluster/anchor
        cell_aad("col-B", "anchor-2"),        # both
        "col-A:anchor-1 ",                    # trailing byte
        "col-A",                              # the collection alone (pre-anchor AAD)
    ])
    def test_correct_key_wrong_aad_raises(self, key, chunks, aad, wrong):
        blob = pack_cell(chunks, key, aad)
        assert wrong != aad
        with pytest.raises(CellTampered):
            unpack_cell(blob, key, wrong)      # same key, different slot

    def test_correct_key_correct_aad_opens(self, key, chunks, aad):
        assert unpack_cell(pack_cell(chunks, key, aad), key, aad) == chunks

    def test_wrong_aad_fails_before_deserialization(self, monkeypatch, key, chunks, aad):
        """Order of operations: authenticate, THEN deserialize — never the reverse.

        `deserialize_chunks` is replaced with a landmine. If the wrong-AAD blob
        reached it, the test fails with the landmine's message instead of the
        expected CellTampered.
        """
        from mantle.search.mantle import cell as cell_mod

        blob = pack_cell(chunks, key, aad)

        def _landmine(_plaintext):
            raise AssertionError(
                "deserialize_chunks was reached on a blob that failed AEAD authentication"
            )

        monkeypatch.setattr(cell_mod, "deserialize_chunks", _landmine)
        with pytest.raises(CellTampered):
            cell_mod.unpack_cell(blob, key, cell_aad("col-B", "anchor-9"))

    def test_aad_is_never_none(self, key):
        """The old `collection_id=""` default silently produced an UNBOUND cell.

        Both entry points now refuse it, so there is no way to write a cell whose
        associated data is None.
        """
        for empty in ("", None):
            with pytest.raises(ValueError, match="required"):
                encrypt_cell(b"x", key, empty)   # type: ignore[arg-type]
            with pytest.raises(ValueError, match="required"):
                decrypt_cell(b"\x00" * 40, key, empty)   # type: ignore[arg-type]
        with pytest.raises(TypeError):
            encrypt_cell(b"x", key)              # type: ignore[call-arg]
        with pytest.raises(TypeError):
            decrypt_cell(b"\x00" * 40, key)      # type: ignore[call-arg]

    def test_cell_aad_requires_both_halves(self):
        """A half-empty AAD collides across slots — `cell_aad` refuses to build one."""
        with pytest.raises(ValueError, match="collection_id"):
            cell_aad("", "anchor-1")
        with pytest.raises(ValueError, match="cluster_id"):
            cell_aad("col-A", "")

    def test_aad_is_authenticated_not_encrypted(self, key, chunks, aad):
        """AAD is bound, not carried: the blob length is independent of it."""
        short = pack_cell(chunks, key, aad)
        long = pack_cell(chunks, key, cell_aad("col-A" * 20, "anchor-1" * 20))
        assert len(short) == len(long)


# silence unused-import warnings — ChunkRecord type alias is exported for callers
def test_module_export_surface():
    from mantle.search.mantle import cell

    expected = {
        "CellError", "CellMalformed", "CellTampered",
        "artifact_ids", "chunk_count",
        "decrypt_cell", "deserialize_chunks", "encrypt_cell",
        "pack_cell", "remove_artifact_chunks", "serialize_chunks",
        "unpack_cell", "upsert_chunk",
    }
    assert expected.issubset(set(cell.__all__))
    # Note: json is imported by the implementation; verify it's available.
    assert hasattr(cell, "json") or json is not None  # silences unused-import
