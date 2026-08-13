"""Encrypted posting-list manager for MANTLE-SSE.

A *posting list* is the encrypted unit of lexical storage: one blob per
``(principal_id, blind_token)``. Each posting list holds the entries for every
artifact whose tokenized text — under the blind token's implicit
``(field, term)`` — produces that token. The S3 backing store sees only
opaque hex-named blobs; plaintext terms never leave the indexer / the
narrowing.

Wire format (binary, AES-256-GCM, mirrors ``cell.py``):

    posting_blob = nonce (12 bytes) ‖ ciphertext ‖ tag (16 bytes)

Cell plaintext is canonical JSON (sorted keys, no whitespace) of either:

- a posting list — ``{"entries": [<PostingEntry>, ...]}``, where a ``PostingEntry`` is
  ``{"artifact_id", "collection_id", "field"}``
- an artifact manifest — ``{"tokens": [<blind_token>, ...]}``

Neither reader inspects the keys it is not asking for: :func:`deserialize_entries` hands an
entry back whole, and :func:`deserialize_manifest` reads ``tokens`` and ignores the rest. That
is what lets the entry and manifest shapes SHRINK — ``tf`` / ``dl`` / ``positions`` /
``field_dls`` are gone with BM25 — without a reindex. Blobs written under the old shape still
open, still match, and get smaller on their next write.

Per-token / per-manifest keys are derived deterministically from the
owner's SSE key via HKDF with distinct ``info`` prefixes
(``"posting:<token>"`` vs ``"manifest:<artifact_id>"``) so the two key
trees stay independent.

This module knows nothing about S3, the indexer, or the narrowing — by
design. The :class:`PostingStore` Protocol is the storage boundary; the
S3-backed implementation lives in ``mantle/search/mantle/sse/s3_stores.py``.

See ``.dev/features/mantle-sse-lexical-index.md`` § Posting List Contents
and § Deletion / Revocation.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Iterable, List, Optional, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crypto parameters
# ---------------------------------------------------------------------------

_NONCE_BYTES = 12          # AES-GCM standard 96-bit nonce
_KEY_BYTES = 32            # 256-bit AES-GCM key
_GCM_TAG_BYTES = 16
_OWNER_SSE_KEY_BYTES = 32  # matches OracleService.derive_sse_key

# HKDF salt — versioned so a future v2 key tree can coexist with v1-encrypted
# postings during a migration. Postings encrypted under different salts are
# independent.
_HKDF_SALT_V1 = b"agience-mantle-sse-posting-v1"

# Distinct info prefixes keep the posting-list and manifest key trees from
# colliding. Format: ``b"<prefix>:<id>"`` where id is ASCII (blind token hex
# or UUID). Both ids are fixed-length so length-prefixing isn't needed.
_INFO_PREFIX_POSTING = b"posting:"
_INFO_PREFIX_MANIFEST = b"manifest:"

# ---------------------------------------------------------------------------
# AEAD associated data — slot binding
#
# STATUS: wired. Every write path binds — ``sse/indexer.py`` passes
# :func:`posting_aad` on posting blobs and :func:`manifest_aad` on manifests, and
# ``sse/narrowing.py`` passes the same on the read side. The ``aad=`` parameters below
# still default to None
# because this module is a primitive: a caller that has no slot to name must be able
# to say so, and the default is what the pre-binding corpus was written with.
#
# What CAN be bound here, and what cannot:
#
#   posting list   (principal_id, blind_token)   ← bindable
#   manifest       (principal_id, artifact_id)   ← bindable
#   collection                                   ← NOT bindable at this layer
#
# A posting list is multi-collection BY CONSTRUCTION: its unique key is
# ``(artifact_id, collection_id)`` (see :func:`upsert_entry`), so one blob holds
# entries for every collection that contains the term. There is no single
# collection to bind, which is why cross-collection separation on the read path
# is a plaintext post-filter and not a tag check. Making it cryptographic would
# mean sharding posting lists per collection — a different index shape and a full
# reindex, not an AAD change.
#
# So the binding this layer can add is over the SLOT, which HKDF already binds
# into the key (``info="posting:<blind_token>"`` over a per-principal SSE key).
# The AAD is therefore defence in depth — it makes a slot mix-up fail on the tag
# even if a key-derivation bug ever handed back the wrong key — not new
# separation. That is worth having and worth being honest about.
_AAD_PREFIX_POSTING = b"sse-posting-aad-v1:"
_AAD_PREFIX_MANIFEST = b"sse-manifest-aad-v1:"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class PostingError(Exception):
    """Base class for posting-list crypto / format errors."""


class PostingTampered(PostingError):
    """Raised when a posting blob fails GCM authentication."""


class PostingMalformed(PostingError):
    """Raised when a posting blob is too short or contains invalid JSON."""


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_owner_sse_key(owner_sse_key: bytes) -> None:
    if not isinstance(owner_sse_key, (bytes, bytearray)):
        raise TypeError(
            f"owner_sse_key must be bytes, got {type(owner_sse_key).__name__}"
        )
    if len(owner_sse_key) != _OWNER_SSE_KEY_BYTES:
        raise ValueError(
            f"owner_sse_key must be {_OWNER_SSE_KEY_BYTES} bytes, "
            f"got {len(owner_sse_key)}"
        )


def _validate_aead_key(key: bytes) -> None:
    if len(key) != _KEY_BYTES:
        raise ValueError(f"AEAD key must be {_KEY_BYTES} bytes, got {len(key)}")


def _validate_blind_token(blind_token: str) -> None:
    if not blind_token:
        raise ValueError("blind_token is required")
    if len(blind_token) != 64:
        raise ValueError(
            f"blind_token must be 64 hex chars, got {len(blind_token)}"
        )
    # Cheap shape check — full hex validation is the indexer's concern.
    try:
        int(blind_token, 16)
    except ValueError as exc:
        raise ValueError(f"blind_token must be hex: {blind_token!r}") from exc


def _validate_artifact_id(artifact_id: str) -> None:
    if not artifact_id:
        raise ValueError("artifact_id is required")


# ---------------------------------------------------------------------------
# Per-posting-list / per-manifest key derivation
# ---------------------------------------------------------------------------

def derive_posting_key(owner_sse_key: bytes, blind_token: str) -> bytes:
    """Derive the AES-256-GCM key for one posting list.

    ``key = HKDF-Expand(owner_sse_key, info="posting:<blind_token>")``.

    Deterministic — re-derivation yields the same key. Per-blind-token
    independence means a key compromise in one posting list cannot decrypt
    another, even within the same owner.
    """
    _validate_owner_sse_key(owner_sse_key)
    _validate_blind_token(blind_token)
    return _hkdf(
        ikm=bytes(owner_sse_key),
        info=_INFO_PREFIX_POSTING + blind_token.encode("ascii"),
    )


def derive_manifest_key(owner_sse_key: bytes, artifact_id: str) -> bytes:
    """Derive the AES-256-GCM key for one artifact's blind-token manifest.

    ``key = HKDF-Expand(owner_sse_key, info="manifest:<artifact_id>")``.

    Manifests track which posting lists reference an artifact, so the
    deletion path can locate every posting list it must rewrite when the
    artifact is removed (per MANTLE-SSE spec § Deletion / Revocation).
    """
    _validate_owner_sse_key(owner_sse_key)
    _validate_artifact_id(artifact_id)
    return _hkdf(
        ikm=bytes(owner_sse_key),
        info=_INFO_PREFIX_MANIFEST + artifact_id.encode("ascii"),
    )


def _hkdf(*, ikm: bytes, info: bytes) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_BYTES,
        salt=_HKDF_SALT_V1,
        info=info,
    ).derive(ikm)


# ---------------------------------------------------------------------------
# Encrypt / decrypt primitives (mirror cell.py)
# ---------------------------------------------------------------------------

def posting_aad(principal_id: str, blind_token: str) -> bytes:
    """AEAD associated data binding a posting blob to its ``(principal, token)`` slot.

    Wired at both call sites — ``sse/indexer.py`` binds on write and read, ``sse/narrowing.py``
    binds on read. A blob written with an AAD does not open without it, so the two move
    together; see the note beside :data:`_AAD_PREFIX_POSTING` for what the binding does and
    does not separate.

    The corpus is read in DUAL-READ. :func:`decrypt_blob` tries the bound AAD first and
    falls back to the legacy unbound form, so blobs written before binding keep opening
    and each is bound on its next write. ``allow_unbound=True`` is the reader default and
    stays that way: with it, an unbound blob still opens; without it, an unbound blob is
    indistinguishable from a tampered one.

    Flipping ``allow_unbound=False`` is an OPERATIONAL decision, not a code change to make
    casually. It is correct only once a reindex has completed on the store being read —
    every posting list, manifest and stats blob rewritten — because from that moment any
    unbound blob really is anomalous. Flipped early it converts "written before binding"
    into :class:`PostingTampered`, i.e. a search that reports nothing found over an index
    that is intact. SSE data is rebuildable by reindex, so the migration is available; what
    is not available is knowing it finished from inside this module.

    The principal id is length-prefixed so ``(p, t)`` cannot be re-partitioned
    into a different pair with the same encoding.
    """
    if not principal_id:
        raise ValueError("posting_aad requires a non-empty principal_id")
    _validate_blind_token(blind_token)
    return _AAD_PREFIX_POSTING + (
        f"{len(principal_id.encode('utf-8'))}:{principal_id}:{blind_token}"
    ).encode("utf-8")


def manifest_aad(principal_id: str, artifact_id: str) -> bytes:
    """AEAD associated data binding a manifest blob to its ``(principal, artifact)``
    slot. Wired the same way and read under the same dual-read as :func:`posting_aad`."""
    if not principal_id:
        raise ValueError("manifest_aad requires a non-empty principal_id")
    _validate_artifact_id(artifact_id)
    return _AAD_PREFIX_MANIFEST + (
        f"{len(principal_id.encode('utf-8'))}:{principal_id}:{artifact_id}"
    ).encode("utf-8")


def encrypt_blob(plaintext: bytes, key: bytes, *,
                 aad: Optional[bytes] = None) -> bytes:
    """Encrypt ``plaintext`` under a 256-bit AES-GCM key.

    Returns ``nonce ‖ ciphertext ‖ tag``. A fresh 96-bit nonce is drawn from
    ``os.urandom`` for every call — never reuse a (key, nonce) pair.

    ``aad`` binds the blob to its slot — build it with :func:`posting_aad` /
    :func:`manifest_aad`, as every write path in ``sse/`` does. It defaults to ``None``
    because this is the primitive rather than a slot-aware caller; ``None`` is also the
    wire form the pre-binding corpus was written in, which is what
    :func:`decrypt_blob`'s dual-read still opens.
    """
    _validate_aead_key(key)
    aead = AESGCM(key)
    nonce = os.urandom(_NONCE_BYTES)
    return nonce + aead.encrypt(nonce, plaintext, associated_data=aad)


def decrypt_blob(blob: bytes, key: bytes, *,
                 aad: Optional[bytes] = None,
                 allow_unbound: bool = True) -> bytes:
    """Decrypt a blob produced by :func:`encrypt_blob`.

    Raises :class:`PostingTampered` on GCM tag failure (wrong key, modified
    ciphertext, or — once ``aad`` is supplied — the wrong slot). Raises
    :class:`PostingMalformed` if the blob is shorter than the nonce + tag
    overhead.

    **Dual-read.** When ``aad`` is supplied and does not authenticate, an
    unbound (``associated_data=None``) attempt follows, so blobs written before
    slot binding was enabled still open. Pass ``allow_unbound=False`` to drop
    that fallback once a corpus is known fully migrated — at which point the
    binding becomes enforcing rather than merely recorded.
    """
    _validate_aead_key(key)
    if len(blob) < _NONCE_BYTES + _GCM_TAG_BYTES:
        raise PostingMalformed(f"posting blob too short ({len(blob)} bytes)")

    nonce = blob[:_NONCE_BYTES]
    ciphertext_and_tag = blob[_NONCE_BYTES:]
    aead = AESGCM(key)
    try:
        return aead.decrypt(nonce, ciphertext_and_tag, associated_data=aad)
    except InvalidTag as exc:
        if aad is None or not allow_unbound:
            raise PostingTampered(
                "posting GCM tag failed — wrong key, modified ciphertext"
                + (", or wrong slot" if aad is not None else "")
            ) from exc
    try:
        # Legacy: written before slot binding. Decrypt-only — writes always bind.
        return aead.decrypt(nonce, ciphertext_and_tag, associated_data=None)
    except InvalidTag as exc:
        raise PostingTampered(
            "posting GCM tag failed under both the bound and the legacy unbound "
            "associated data — wrong key, modified ciphertext, or wrong slot"
        ) from exc


# ---------------------------------------------------------------------------
# Posting-list serialization
# ---------------------------------------------------------------------------

def serialize_entries(entries: List[dict]) -> bytes:
    """Encode a list of posting entries as canonical JSON bytes.

    Wraps the list in ``{"entries": [...]}`` so the outer envelope stays
    extensible (future fields like ``"version"`` or ``"compressed"`` can be
    added without rewriting every existing blob).
    """
    payload = {"entries": entries}
    # ensure_ascii=False per RFC 8785 — see the note in `search/mantle/cell.py`. Read back by
    # `deserialize_entries` via `json.loads`, which accepts both spellings, so existing blobs decode.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def deserialize_entries(plaintext: bytes) -> List[dict]:
    """Decode the canonical JSON form produced by :func:`serialize_entries`.

    Raises :class:`PostingMalformed` if the plaintext isn't valid JSON, the
    outer envelope isn't an object, or ``"entries"`` isn't a list.
    """
    try:
        payload = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostingMalformed(
            f"posting plaintext is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PostingMalformed(
            f"posting plaintext must decode to an object, "
            f"got {type(payload).__name__}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise PostingMalformed(
            f"posting payload missing list 'entries', got {type(entries).__name__}"
        )
    return entries


def pack_posting(entries: List[dict], key: bytes, *,
                 aad: Optional[bytes] = None) -> bytes:
    """Serialize + encrypt in one call. Returns the at-rest blob.

    ``aad``: build it with :func:`posting_aad`, as ``sse/indexer.py`` does on every
    write. Optional at this layer — the primitive does not require a slot — but a blob
    packed unbound is one more blob the dual-read has to carry."""
    return encrypt_blob(serialize_entries(entries), key, aad=aad)


def unpack_posting(blob: bytes, key: bytes, *,
                   aad: Optional[bytes] = None,
                   allow_unbound: bool = True) -> List[dict]:
    """Decrypt + deserialize. Inverse of :func:`pack_posting`.

    Authentication happens first: a wrong ``aad`` raises before
    :func:`deserialize_entries` ever sees a byte."""
    return deserialize_entries(
        decrypt_blob(blob, key, aad=aad, allow_unbound=allow_unbound))


# ---------------------------------------------------------------------------
# Manifest serialization (per-artifact tracker — the blind tokens an artifact wrote)
# ---------------------------------------------------------------------------

def serialize_manifest(blind_tokens: Iterable[str]) -> bytes:
    """Encode an artifact's manifest as canonical JSON.

    Wire shape::

        {"tokens": ["<bt1>", ...]}

    Tokens are de-duplicated and sorted so the on-disk representation is
    stable — useful for cache fingerprinting and for diffing manifests
    during incremental updates. This is the shape the doc spec ("§ Deletion / Revocation")
    describes: a flat list of blind tokens, and the one read revocation and re-index both need.

    ``field_dls`` USED TO RIDE HERE. It carried the per-field document length the artifact was
    indexed at, so the indexer could subtract the old document's contribution from the BM25
    corpus statistics before adding the new one. There are no corpus statistics, so there is
    nothing to subtract and nothing to carry. Its removal is pure subtraction on the wire:
    :func:`deserialize_manifest` reads ``tokens`` and ignores every other key, so a manifest
    written with ``field_dls`` still opens and still yields the same token list — it just
    stops carrying the field the next time it is written. No reindex.
    """
    deduped = sorted({str(t) for t in blind_tokens if t})
    payload = {"tokens": deduped}
    # ensure_ascii=False per RFC 8785 — see the note in `search/mantle/cell.py`. Read back by
    # `deserialize_manifest` via `json.loads`.
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def deserialize_manifest(plaintext: bytes) -> List[str]:
    """Decode a manifest produced by :func:`serialize_manifest`.

    Returns the list of blind tokens. Raises :class:`PostingMalformed` if the JSON is not an
    object or does not carry a ``tokens`` list.

    ANY OTHER KEY IS IGNORED, which is what makes dropping ``field_dls`` a format change
    needing no reindex: an older manifest carries it, this reads past it, and the token list —
    the only thing the indexer and the deletion path ever wanted — is unchanged.
    """
    try:
        payload = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PostingMalformed(
            f"manifest plaintext is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PostingMalformed(
            f"manifest plaintext must decode to an object, "
            f"got {type(payload).__name__}"
        )
    tokens_raw = payload.get("tokens")
    if not isinstance(tokens_raw, list):
        raise PostingMalformed(
            f"manifest payload missing list 'tokens', got {type(tokens_raw).__name__}"
        )
    return [str(t) for t in tokens_raw]


def pack_manifest(
    blind_tokens: Iterable[str],
    key: bytes,
    *,
    aad: Optional[bytes] = None,
) -> bytes:
    """Serialize + encrypt a manifest in one call.

    ``aad``: build it with :func:`manifest_aad`, as ``sse/indexer.py`` does on every
    write. Optional at this layer for the same reason as :func:`pack_posting`."""
    return encrypt_blob(serialize_manifest(blind_tokens), key, aad=aad)


def unpack_manifest(blob: bytes, key: bytes, *,
                    aad: Optional[bytes] = None,
                    allow_unbound: bool = True) -> List[str]:
    """Decrypt + deserialize a manifest. Inverse of :func:`pack_manifest`.

    Returns the token list. Authentication happens first: a wrong ``aad`` raises before
    :func:`deserialize_manifest` ever sees a byte.
    """
    return deserialize_manifest(
        decrypt_blob(blob, key, aad=aad, allow_unbound=allow_unbound))


# ---------------------------------------------------------------------------
# Mutation helpers (read-modify-write pattern)
# ---------------------------------------------------------------------------

def upsert_entry(entries: List[dict], record: dict) -> List[dict]:
    """Insert or replace a posting entry by ``(artifact_id, collection_id)``.

    The unique key is the ``(artifact_id, collection_id)`` pair: the field
    is implicit in the posting list's blind token (one list per
    ``(field, term)``), and the same artifact in the same collection
    contributes one entry per posting list. Re-indexing the same artifact
    overwrites in place.

    Returns the same list reference for chaining.
    """
    artifact_id = record.get("artifact_id")
    collection_id = record.get("collection_id")
    if not artifact_id or not collection_id:
        raise ValueError(
            "posting entry must carry non-empty artifact_id and collection_id"
        )
    for i, existing in enumerate(entries):
        if (
            existing.get("artifact_id") == artifact_id
            and existing.get("collection_id") == collection_id
        ):
            entries[i] = record
            return entries
    entries.append(record)
    return entries


def remove_artifact_entries(entries: List[dict], artifact_id: str) -> List[dict]:
    """Strip every entry for ``artifact_id`` (across all its collections).

    Returns a *new* list (does not mutate the input). Used by the deletion
    path when an artifact is fully revoked.
    """
    return [e for e in entries if e.get("artifact_id") != artifact_id]


def remove_artifact_collection_entries(
    entries: List[dict], artifact_id: str, collection_id: str,
) -> List[dict]:
    """Strip entries for one ``(artifact_id, collection_id)`` pair only.

    Used by partial revocation: an artifact is removed from one collection
    but remains in others. Returns a new list.
    """
    return [
        e for e in entries
        if not (
            e.get("artifact_id") == artifact_id
            and e.get("collection_id") == collection_id
        )
    ]


def entry_count(entries: List[dict]) -> int:
    """Total entries — useful for empty-posting-list detection."""
    return len(entries)


def artifact_ids_in_entries(entries: List[dict]) -> set[str]:
    """Distinct artifact IDs present in a posting list."""
    return {e["artifact_id"] for e in entries if "artifact_id" in e}


# ---------------------------------------------------------------------------
# Storage Protocol
# ---------------------------------------------------------------------------

class PostingStore(Protocol):
    """Encrypted posting-list + manifest storage.

    The indexer writes both posting lists and per-artifact manifests; the
    narrowing reads only posting lists; the deletion path reads manifests
    to find every posting list referencing the artifact, then rewrites each.

    Production S3 layout:
      ``{tenant}/{principal_id}/sse/posting/{blind_token}.enc``
      ``{tenant}/{principal_id}/sse/manifests/{artifact_id}.enc``

    **`get_posting` should be safe to call concurrently from several threads.** A recall needs
    one posting list per (term × field) and they are independent blobs, so nothing about them
    forces a serial read; the reader that fanned them out across a thread pool went with the
    BM25 path, and :class:`~.narrowing.TokenNarrower` issues them serially today. An
    implementation that shares mutable state across calls — a cursor, a buffer, a
    non-thread-safe client — must still guard it, because that is what makes fanning them out
    again a change to one caller rather than to every store.
    :class:`InMemoryPostingStore` holds an ``RLock`` for exactly this reason, and a boto3 client
    is already safe for concurrent operations.
    """

    # Posting-list operations
    def get_posting(self, principal_id: str, blind_token: str) -> Optional[bytes]:
        """Return the encrypted posting blob, or None. Concurrency-safe — see the class
        docstring."""

    def put_posting(self, principal_id: str, blind_token: str, blob: bytes) -> None:
        """Persist (or overwrite) the posting blob."""

    def delete_posting(self, principal_id: str, blind_token: str) -> None:
        """Remove the posting list. No-op if absent."""

    def list_tokens_for_owner(self, principal_id: str) -> List[str]:
        """Return every blind token with a stored posting list under
        ``principal_id``. Used by bulk re-key / migration paths."""

    # Manifest operations
    def get_manifest(self, principal_id: str, artifact_id: str) -> Optional[bytes]:
        """Return the encrypted manifest blob, or None."""

    def put_manifest(self, principal_id: str, artifact_id: str, blob: bytes) -> None:
        """Persist (or overwrite) the manifest blob."""

    def delete_manifest(self, principal_id: str, artifact_id: str) -> None:
        """Remove the manifest. No-op if absent."""


class InMemoryPostingStore:
    """Thread-safe dict-backed PostingStore. Test default; not durable."""

    def __init__(self) -> None:
        self._postings: dict[tuple[str, str], bytes] = {}
        self._manifests: dict[tuple[str, str], bytes] = {}
        self._lock = threading.RLock()

    # Posting-list operations
    def get_posting(self, principal_id: str, blind_token: str) -> Optional[bytes]:
        with self._lock:
            return self._postings.get((principal_id, blind_token))

    def put_posting(self, principal_id: str, blind_token: str, blob: bytes) -> None:
        with self._lock:
            self._postings[(principal_id, blind_token)] = blob

    def delete_posting(self, principal_id: str, blind_token: str) -> None:
        with self._lock:
            self._postings.pop((principal_id, blind_token), None)

    def list_tokens_for_owner(self, principal_id: str) -> List[str]:
        with self._lock:
            return [tok for (oid, tok) in self._postings if oid == principal_id]

    # Manifest operations
    def get_manifest(self, principal_id: str, artifact_id: str) -> Optional[bytes]:
        with self._lock:
            return self._manifests.get((principal_id, artifact_id))

    def put_manifest(self, principal_id: str, artifact_id: str, blob: bytes) -> None:
        with self._lock:
            self._manifests[(principal_id, artifact_id)] = blob

    def delete_manifest(self, principal_id: str, artifact_id: str) -> None:
        with self._lock:
            self._manifests.pop((principal_id, artifact_id), None)


__all__ = [
    # Errors
    "PostingError",
    "PostingMalformed",
    "PostingTampered",
    # Key derivation
    "derive_manifest_key",
    "derive_posting_key",
    # Slot binding (AEAD associated data) — wired on every write path
    "manifest_aad",
    "posting_aad",
    # Crypto primitives
    "decrypt_blob",
    "encrypt_blob",
    # Posting-list (de)serialization
    "deserialize_entries",
    "pack_posting",
    "serialize_entries",
    "unpack_posting",
    # Manifest (de)serialization
    "deserialize_manifest",
    "pack_manifest",
    "serialize_manifest",
    "unpack_manifest",
    # Mutation helpers
    "artifact_ids_in_entries",
    "entry_count",
    "remove_artifact_collection_entries",
    "remove_artifact_entries",
    "upsert_entry",
    # Storage
    "InMemoryPostingStore",
    "PostingStore",
]
