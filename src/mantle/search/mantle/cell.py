"""Cell encryption + serialization.

A MANTLE *cell* is the unit of encrypted storage that holds a set of indexed
artifact chunks for one ``(principal_id, collection_id, cluster_id)`` tuple.
This module is the contained, FAISS-free piece between :class:`OracleService`
(provides cell keys) and :class:`MantleIndexer` / :class:`MantleQueryEngine`
(produce / consume cells against S3).

Wire format (binary, AES-256-GCM):

    cell_blob = nonce (12 bytes) ‖ ciphertext ‖ tag (16 bytes)

The ``cryptography`` library's ``AESGCM.encrypt(...)`` returns
``ciphertext ‖ tag`` as a single byte string. We prepend the nonce so the
blob is fully self-contained — decryption only needs the cell key.

Cell plaintext is JSON-encoded UTF-8 bytes. Each cell holds a list of
*chunk records* (artifact_id, chunk_id, embedding, optional metadata).

This module does not know about S3, FAISS, or the OracleService — by design.
Callers wire those together. That keeps the crypto correctness reviewable in
isolation and lets the encrypted-search engine compose against any storage
backend (S3 today, IPFS / encrypted filesystem later).

See `.dev/features/mantle-mvp.md` § Layer 2b.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, List

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

_NONCE_BYTES = 12          # AES-GCM standard 96-bit nonce
_KEY_BYTES = 32            # 256-bit cell key (must match OracleService)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class CellError(Exception):
    """Base class for cell-level crypto errors."""


class CellTampered(CellError):
    """Raised when a cell fails GCM authentication (tampering or wrong key)."""


class CellMalformed(CellError):
    """Raised when a cell blob is too short or contains invalid JSON plaintext."""


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------

def _require_aad(collection_id: str) -> bytes:
    """The AAD bytes for a cell — REQUIRED, never ``None``.

    An empty (or non-string) value used to fall through to
    ``associated_data=None``, which silently produced an *unbound* cell: the blob
    would then open under any key derived for any slot, defeating invariant #3.
    That fallback is gone. The caller must name the context/anchor it is writing,
    and :func:`cell_aad` is the one formula that builds the value.
    """
    if not isinstance(collection_id, str) or not collection_id:
        raise ValueError(
            "collection_id (the cell AAD) is required and must be a non-empty string — "
            "build it with cell_aad(collection_id, cluster_id). An empty value would bind "
            "the blob to nothing, letting it be presented under the wrong key or moved to "
            "the wrong slot without failing authentication."
        )
    return collection_id.encode("utf-8")


def encrypt_cell(
    plaintext: bytes,
    key: bytes,
    collection_id: str,
) -> bytes:
    """Encrypt ``plaintext`` with a 256-bit AES-GCM ``key``.

    Returns ``nonce ‖ ciphertext ‖ tag``. A fresh nonce is drawn from
    ``os.urandom`` for every call — never reuse a (key, nonce) pair.

    ``collection_id`` is bound as AEAD associated data so that a cell blob
    cannot be decrypted under a key derived for a different collection.
    It is **required** and must be non-empty — see :func:`_require_aad`.
    """
    if len(key) != _KEY_BYTES:
        raise ValueError(f"cell key must be {_KEY_BYTES} bytes, got {len(key)}")
    aad = _require_aad(collection_id)
    aead = AESGCM(key)
    nonce = os.urandom(_NONCE_BYTES)
    ciphertext_and_tag = aead.encrypt(nonce, plaintext, associated_data=aad)
    return nonce + ciphertext_and_tag


def decrypt_cell(
    blob: bytes,
    key: bytes,
    collection_id: str,
) -> bytes:
    """Decrypt + authenticate a cell blob produced by :func:`encrypt_cell`.

    Raises :class:`CellTampered` if the GCM tag fails (wrong key, modified
    ciphertext, or the wrong AAD). Raises :class:`CellMalformed` if the blob is
    shorter than the nonce + tag overhead.

    ``collection_id`` is **required** and must match the value supplied at
    encryption time; a mismatch causes a :class:`CellTampered` error before any
    deserialization of cell contents occurs.
    """
    if len(key) != _KEY_BYTES:
        raise ValueError(f"cell key must be {_KEY_BYTES} bytes, got {len(key)}")
    aad = _require_aad(collection_id)
    if len(blob) < _NONCE_BYTES + 16:  # 16 bytes for the GCM tag
        raise CellMalformed(f"cell blob too short ({len(blob)} bytes)")

    nonce = blob[:_NONCE_BYTES]
    ciphertext_and_tag = blob[_NONCE_BYTES:]
    aead = AESGCM(key)
    try:
        return aead.decrypt(nonce, ciphertext_and_tag, associated_data=aad)
    except InvalidTag as exc:
        raise CellTampered(
            "cell GCM tag failed — wrong key, modified ciphertext, or wrong collection"
        ) from exc


# ---------------------------------------------------------------------------
# Serialization (chunk records → bytes → cell plaintext)
# ---------------------------------------------------------------------------

def cell_aad(collection_id: str, cluster_id: str) -> str:
    """AEAD associated-data string binding a cell to its ``(context, anchor)``:
    ``"collection:cluster"`` (canonical plan §5.1: AAD = ``context:anchor``).

    One formula — a blob cannot be decrypted under a key/slot derived for a
    different anchor. ``cluster_id`` is required (routing has no flat fallback,
    so there is no anchor-less cell). Pass the result as the ``collection_id``
    argument to :func:`encrypt_cell` / :func:`decrypt_cell`.

    BOTH halves are required. An empty half would make the AAD of one slot a
    prefix/suffix of another's ("" + ":" + anchor collides across every
    collection), which is the same unbound-blob failure ``_require_aad`` exists
    to prevent — so it raises here rather than producing a half-bound string.
    """
    if not isinstance(collection_id, str) or not collection_id:
        raise ValueError("cell_aad requires a non-empty collection_id")
    if not isinstance(cluster_id, str) or not cluster_id:
        raise ValueError("cell_aad requires a non-empty cluster_id")
    return f"{collection_id}:{cluster_id}"


def serialize_chunks(chunks: List[dict]) -> bytes:
    """Encode a list of chunk records as the cell plaintext.

    Each chunk record carries (at minimum) ``artifact_id``, ``chunk_id``,
    ``embedding`` (list of floats), and optional metadata. The JSON encoding
    is canonical (sorted keys, no whitespace) so the same input always
    serializes to the same bytes — useful when callers want to cache or
    fingerprint cell contents.
    """
    # `ensure_ascii=False` is required, not stylistic: the docstring above offers these bytes for
    # fingerprinting, and RFC 8785 canonical JSON emits raw UTF-8. With the default `ensure_ascii=True`
    # a non-ASCII chunk serializes to `\uXXXX` escapes, so the same logical input would produce
    # different bytes here than under the canonicaliser of record (`prism/canonical.py`) — i.e. two
    # different fingerprints for one input.
    # Reading is unaffected: `deserialize_chunks` uses `json.loads`, which accepts both spellings.
    return json.dumps(chunks, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def deserialize_chunks(plaintext: bytes) -> List[dict]:
    """Decode cell plaintext back into chunk records.

    Raises :class:`CellMalformed` if the plaintext isn't valid JSON or
    isn't a list. Individual chunk records aren't validated — that's the
    caller's concern (different cell schemas may evolve).
    """
    try:
        value = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CellMalformed(f"cell plaintext is not valid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise CellMalformed(
            f"cell plaintext must decode to a list, got {type(value).__name__}"
        )
    return value


# ---------------------------------------------------------------------------
# Convenience round-trip
# ---------------------------------------------------------------------------

def pack_cell(
    chunks: List[dict],
    key: bytes,
    collection_id: str,
) -> bytes:
    """Encode + encrypt in one call. Returns the encrypted blob ready for storage.

    ``collection_id`` is the required AAD — build it with :func:`cell_aad`."""
    return encrypt_cell(serialize_chunks(chunks), key, collection_id=collection_id)


def unpack_cell(
    blob: bytes,
    key: bytes,
    collection_id: str,
) -> List[dict]:
    """Decrypt + decode in one call. The inverse of :func:`pack_cell`."""
    return deserialize_chunks(decrypt_cell(blob, key, collection_id=collection_id))


# ---------------------------------------------------------------------------
# Mutation helpers (read-modify-write pattern)
# ---------------------------------------------------------------------------

def upsert_chunk(chunks: List[dict], record: dict) -> List[dict]:
    """Add or replace a chunk record in-place by ``(artifact_id, chunk_id)``.

    The indexer uses this when updating an artifact: load → upsert → write.
    Returns the same list reference for chaining.
    """
    artifact_id = record.get("artifact_id")
    chunk_id = record.get("chunk_id")
    if artifact_id is None or chunk_id is None:
        raise ValueError("chunk record must carry artifact_id and chunk_id")
    for i, existing in enumerate(chunks):
        if (
            existing.get("artifact_id") == artifact_id
            and existing.get("chunk_id") == chunk_id
        ):
            chunks[i] = record
            return chunks
    chunks.append(record)
    return chunks


def remove_artifact_chunks(chunks: List[dict], artifact_id: str) -> List[dict]:
    """Strip every chunk record belonging to ``artifact_id``.

    Returns a new list (does not mutate the input). Used by the indexer's
    artifact-removal path to evict from each affected cell.
    """
    return [c for c in chunks if c.get("artifact_id") != artifact_id]


def chunk_count(chunks: List[dict]) -> int:
    """Total chunk records in a cell — useful for stats / budget enforcement."""
    return len(chunks)


def artifact_ids(chunks: List[dict]) -> set[str]:
    """Distinct artifact IDs present in a cell."""
    return {c["artifact_id"] for c in chunks if "artifact_id" in c}


# Re-export for callers that import the module surface
__all__ = [
    "CellError",
    "CellMalformed",
    "CellTampered",
    "artifact_ids",
    "cell_aad",
    "chunk_count",
    "decrypt_cell",
    "deserialize_chunks",
    "encrypt_cell",
    "pack_cell",
    "remove_artifact_chunks",
    "serialize_chunks",
    "unpack_cell",
    "upsert_chunk",
]


# Type alias for clarity in callers
ChunkRecord = dict[str, Any]
