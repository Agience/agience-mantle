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
from typing import Any, Iterable, List, Optional, Protocol, Tuple

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

#: A third tree, ``ownerindex:``, was here. It keyed the owner-index accelerator — one blob per
#: principal holding that owner's whole ``token -> entries`` map — which existed because a probe on
#: the file store was an `open()` (4,520 of them for a ten-term query over 194 owners, mostly
#: misses). `SqlitePostingStore` makes a probe an indexed lookup on a primary key, so the
#: accelerator has nothing left to collapse and was removed rather than repaired, along with the
#: partial-index-read-as-complete failure that made a whole prior corpus unfindable.
#:
#: Its key derivation and AAD prefix are gone with it. An index tree written before the change still
#: holds `index.enc` blobs; nothing opens them, and `mantle.system.manage_sse_index` does not carry
#: them across.

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

#: Per-ENTRY binding — the slot plus the entry's own identity within it.
#:
#: The entry-level layout is what makes this binding possible. A whole-slot blob holds every
#: collection's entries, so there is no single collection to bind and collection separation can only
#: be a plaintext post-filter. An entry is one ``(artifact_id, collection_id)`` pair, so both go into
#: its AAD, and an entry moved between artifacts, between collections, or between tokens fails
#: authentication instead
#: of being re-filtered on read.
_AAD_PREFIX_ENTRY = b"sse-entry-aad-v1:"


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
    # UTF-8, not ASCII. An artifact id is any non-empty string — `_validate_artifact_id` says so
    # and enforces nothing else — so `.encode("ascii")` refused every id carrying a diacritic and
    # took the whole SSE arm down with it: `cn-archæological`, `cn-ardèche` and their neighbours
    # indexed `sse=failed`, and `stage.0.lexicon` is 1.84M ConceptNet entries. Measured on 71/home:
    # 100% of a reindex pass failed this way.
    #
    # The change cannot invalidate an existing key, which is why it is a fix rather than a
    # migration. UTF-8 and ASCII agree byte-for-byte on every id that encodes as ASCII, so every
    # key already derived re-derives identically; and an id that is not ASCII raised here instead
    # of deriving anything, so there is no stored manifest to orphan.
    #
    # The rest of this module already reached this conclusion for canonical JSON — see the
    # `ensure_ascii=False` / `.encode("utf-8")` pairs below and the RFC 8785 note in
    # `search/mantle/cell.py`. This derivation was simply missed.
    return _hkdf(
        ikm=bytes(owner_sse_key),
        info=_INFO_PREFIX_MANIFEST + artifact_id.encode("utf-8"),
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


def entry_aad(principal_id: str, blind_token: str,
              artifact_id: str, collection_id: str) -> bytes:
    """AEAD associated data binding ONE posting entry to its ``(principal, token, artifact,
    collection)`` slot.

    Every component is length-prefixed so no two different tuples can produce the same bytes — the
    same rule :func:`posting_aad` states for its pair, and it matters more here because there are
    four parts and two of them are caller-supplied ids that may contain any character.

    The collection is bound in here, which a whole-slot blob cannot do. See
    :data:`_AAD_PREFIX_ENTRY`: an entry re-filed under another collection fails the tag rather than
    depending on a plaintext post-filter on the read path.
    """
    if not principal_id:
        raise ValueError("entry_aad requires a non-empty principal_id")
    _validate_blind_token(blind_token)
    _validate_artifact_id(artifact_id)
    parts = (principal_id, blind_token, artifact_id, collection_id or "")
    body = ":".join("%d:%s" % (len(p.encode("utf-8")), p) for p in parts)
    return _AAD_PREFIX_ENTRY + body.encode("utf-8")


def pack_entry(entry: dict, key: bytes, *, aad: Optional[bytes] = None) -> bytes:
    """Serialize + encrypt ONE posting entry. Build ``aad`` with :func:`entry_aad`.

    One entry, one blob, which is what makes indexing a body affordable. The list form below seals
    every entry for a term together, so adding one artifact to a term means decrypting every entry
    already there, scanning them, and re-encrypting all of them: **O(artifacts already carrying that
    term)** per token, on a write that touches one artifact.

    That is what made write cost a function of corpus size, measured on 71/home before this
    changed: 14.6s for a POST carrying one name, 16.4s for 4 KB of prose, and 3.5s for 4 KB of
    ``'x '`` — "cost is terms, not bytes". A body contributes thousands of distinct stems, so
    indexing one was unaffordable by construction and `pipeline_unified._OFFER_FIELDS` excluded
    `content` for exactly that reason. With an entry as its own blob, adding an artifact to a term is
    one sealed write and one row, independent of how many artifacts already carry it.

    The envelope is the same ``{"entries": [...]}`` shape with a single element, so
    :func:`deserialize_entries` reads either form and one decoder covers both.
    """
    return encrypt_blob(serialize_entries([entry]), key, aad=aad)


def unpack_entry(blob: bytes, key: bytes, *, aad: Optional[bytes] = None,
                 allow_unbound: bool = True) -> dict:
    """Decrypt + deserialize ONE posting entry. Inverse of :func:`pack_entry`.

    Raises :class:`PostingMalformed` if the blob does not hold exactly one entry: a single-entry slot
    that decoded to two would mean two artifacts sharing one row, and to zero would mean a row whose
    key names an entry it does not carry. Both are index corruption rather than an empty answer.
    """
    entries = deserialize_entries(
        decrypt_blob(blob, key, aad=aad, allow_unbound=allow_unbound))
    if len(entries) != 1:
        raise PostingMalformed(
            f"a per-entry posting blob must hold exactly one entry, got {len(entries)}"
        )
    return entries[0]


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

    Any other key is ignored, which is what makes dropping ``field_dls`` a format change
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

    # ------------------------------------------------------------------
    # Entry operations — the write path
    # ------------------------------------------------------------------
    #
    # The mutation belongs to the store rather than to the caller, which is what keeps the write off
    # O(entries). A caller-side `get_posting` → decrypt every entry → scan → re-encrypt every entry →
    # `put_posting` is a read-modify-write over a blob whose size grows with the number of artifacts
    # carrying that term, and no store can make that cheap while the blob is the unit.
    #
    # These are entry-level operations rather than "one object per entry", so each backend picks its
    # own shape. Per-entry rows suit SQLite — an add is one upsert, a delete is one `DELETE … WHERE`,
    # a read is an indexed range scan on the key prefix. They would be pathological in S3: one object
    # per (owner × term × artifact) is an object explosion, and every probe would become a LIST
    # instead of a GET. So
    # `S3PostingStore` keeps one object per token internally and does the read-modify-write itself,
    # where the round trip dominates anyway. The protocol states the operation; the store states the
    # layout.

    def add_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                  collection_id: str, blob: bytes) -> None:
        """Insert or replace ONE sealed entry in a slot, keyed by ``(artifact_id, collection_id)``.

        That pair is the entry's unique key — the field is implicit in the blind token — so
        re-indexing the same artifact in the same collection replaces in place.
        """

    def get_entries(self, principal_id: str, blind_token: str) -> List[Tuple[str, str, bytes]]:
        """Every entry in one slot as ``(artifact_id, collection_id, sealed_blob)``. ``[]`` for a miss.

        The identity comes back with the bytes, which is what makes the per-entry AAD enforced
        rather than merely written. The reader cannot know an entry's ``(artifact, collection)``
        before opening it, so without this it would pass ``aad=None`` and write a binding nothing
        checks.

        With it, the reader builds :func:`entry_aad` from the ROW KEY and authenticates against
        it. The key is plaintext and mutable by anyone with write access to the store; the sealed
        entry names the slot it was minted for. Move a row to another artifact, another collection
        or another token and it fails the tag — the same shape as the master-key binding in
        `oracle._unframe_master_key`.

        Returning the identity reveals nothing new: those ids are already at rest in cleartext as
        the row key (and as an object key in S3, and as a path component in the retired file
        store).

        Order is unspecified deliberately: the reader turns these into a SET of artifact ids and no
        ranking reads a posting list, so imposing an order would be a cost with no consumer.
        """

    def delete_entries_for_artifact(self, principal_id: str, blind_token: str,
                                    artifact_id: str) -> int:
        """Remove every entry for ``artifact_id`` in one slot, across all its collections. Returns
        how many went. Used by the deletion path, which knows the artifact and not its collections."""

    def delete_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                     collection_id: str) -> bool:
        """Remove exactly one ``(artifact, collection)`` entry. Returns whether it was there.

        Distinct from the method above because partial revocation removes an artifact from ONE
        collection while it remains in others — dropping all of them would un-index it everywhere.
        """

    # ------------------------------------------------------------------
    # Legacy blob operations — read for migration, never written
    # ------------------------------------------------------------------
    #
    # An index written before the entry layout holds one blob per token carrying every entry, and
    # splitting one requires the owner's SSE key, which a store does not have and a blob-copying
    # migration cannot get. So the conversion happens where the keys already are: `narrowing` reads a
    # legacy blob when a slot has no entries, and `indexer` absorbs one into entries on the next
    # write to that slot and deletes it. The index write path writes no legacy blobs.
    #
    # This is the same dual-read discipline the AAD binding used, and for the same reason: the
    # alternative is a flag day on a store that is otherwise still perfectly readable.

    def get_posting(self, principal_id: str, blind_token: str) -> Optional[bytes]:
        """Return the legacy whole-slot blob, or None. Concurrency-safe — see the class
        docstring."""

    def put_posting(self, principal_id: str, blind_token: str, blob: bytes) -> None:
        """Persist a legacy whole-slot blob.

        Retained for tests and for a migration writing a fixture, rather than for the index write
        path. `SseIndexer` calls `add_entry`."""

    def delete_posting(self, principal_id: str, blind_token: str) -> None:
        """Remove the legacy blob. No-op if absent. Called once a slot's entries have been absorbed."""

    def list_tokens_for_owner(self, principal_id: str) -> List[str]:
        """Every blind token this owner has anything stored under — entries or a legacy blob.

        BOTH, because the callers are re-key and migration passes: one that saw only the new layout
        would silently skip every slot not yet converted. Used by bulk re-key and by
        `mantle.system.manage_sse_index`."""

    def list_owners(self) -> List[str]:
        """Every principal with anything stored here.

        Read off the STORE, not the artifact table: a rebuild must cover what the search will
        look at, and an owner whose artifacts were all deleted can still hold posting lists.
        Used by bulk re-key and by `mantle.system.manage_sse_index`.
        """

    # Manifest operations
    def get_manifest(self, principal_id: str, artifact_id: str) -> Optional[bytes]:
        """Return the encrypted manifest blob, or None."""

    def put_manifest(self, principal_id: str, artifact_id: str, blob: bytes) -> None:
        """Persist (or overwrite) the manifest blob."""

    def delete_manifest(self, principal_id: str, artifact_id: str) -> None:
        """Remove the manifest. No-op if absent."""

    # ------------------------------------------------------------------
    # Which analysis wrote this index
    # ------------------------------------------------------------------
    #
    # A blind token is an HMAC of an ANALYSED term, so the analysis is part of the index format in
    # the strongest sense available: the store holds hashes and cannot re-derive a term it has
    # never seen, so nothing can migrate an index in place. A client whose pipeline has moved reads
    # a store filed under the old terms and finds nothing, silently — the query is well-formed, the
    # store is healthy, and the answer is empty.
    #
    # These two make that condition a fact rather than an inference. Optional on purpose: a store
    # that predates them, or a third-party implementation, answers "unknown" through
    # `analyzer_generation_of` below rather than raising, because a stamp is a diagnostic and must
    # not be the thing that takes an index offline.

    def analyzer_generation(self) -> Optional[int]:
        """Which `tokenizer.ANALYZER` wrote this index, or ``None`` if it has never been stamped.

        ``None`` is not "generation 0". It means either an empty store, or one written before
        stamping existed — `analyzer_generation_of` is where those are told apart, because only the
        caller holding the store can ask whether it has any owners.
        """

    def record_analyzer_generation(self, generation: int) -> None:
        """Record which analysis is writing. Idempotent; last writer wins.

        Called on the write path rather than at open, so an empty store is not stamped with a
        generation it never used. A store that is written by two generations is already broken and
        the last stamp is the honest one: it names the analysis whose terms are now mixed in.
        """


def analyzer_generation_of(store: Any) -> Optional[int]:
    """The generation a store was written by, or ``None`` if it cannot say.

    ``getattr`` rather than a direct call: the two methods above are optional, and a store that
    does not implement them must read as "cannot say" rather than raise. A diagnostic that can
    break a working index is worse than the condition it diagnoses.
    """
    fn = getattr(store, "analyzer_generation", None)
    if not callable(fn):
        return None
    try:
        got = fn()
    except Exception:
        return None
    return int(got) if isinstance(got, int) else None


def stamp_analyzer_generation(store: Any, generation: int) -> None:
    """Record the writing generation, if the store can hold one. Never raises into the caller —
    failing to write a diagnostic must not fail the index write it accompanies."""
    fn = getattr(store, "record_analyzer_generation", None)
    if not callable(fn):
        return
    try:
        fn(int(generation))
    except Exception:
        pass


class InMemoryPostingStore:
    """Thread-safe dict-backed PostingStore. Test default; not durable."""

    def __init__(self) -> None:
        self._postings: dict[tuple[str, str], bytes] = {}
        #: ``(principal, token) -> {(artifact, collection): blob}``. Nested rather than flat so
        #: `get_entries` and `delete_entries_for_artifact` are both one dict lookup plus a walk of
        #: only that slot — the same access shape the SQLite primary key gives.
        self._entries: dict[tuple[str, str], dict[tuple[str, str], bytes]] = {}
        self._manifests: dict[tuple[str, str], bytes] = {}
        self._analyzer: Optional[int] = None
        self._lock = threading.RLock()

    # Entry operations
    def add_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                  collection_id: str, blob: bytes) -> None:
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError("InMemoryPostingStore.add_entry expects bytes")
        with self._lock:
            slot = self._entries.setdefault((principal_id, blind_token), {})
            slot[(artifact_id, collection_id or "")] = bytes(blob)

    def get_entries(self, principal_id: str, blind_token: str) -> List[Tuple[str, str, bytes]]:
        with self._lock:
            slot = self._entries.get((principal_id, blind_token), {})
            return [(aid, cid, blob) for (aid, cid), blob in slot.items()]

    def delete_entries_for_artifact(self, principal_id: str, blind_token: str,
                                    artifact_id: str) -> int:
        with self._lock:
            slot = self._entries.get((principal_id, blind_token))
            if not slot:
                return 0
            doomed = [k for k in slot if k[0] == artifact_id]
            for k in doomed:
                del slot[k]
            if not slot:
                del self._entries[(principal_id, blind_token)]
            return len(doomed)

    def delete_entry(self, principal_id: str, blind_token: str, artifact_id: str,
                     collection_id: str) -> bool:
        with self._lock:
            slot = self._entries.get((principal_id, blind_token))
            if not slot:
                return False
            gone = slot.pop((artifact_id, collection_id or ""), None) is not None
            if not slot:
                del self._entries[(principal_id, blind_token)]
            return gone

    # Legacy whole-slot blobs — read for migration, never written by the index path
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
        """Both layouts — see the Protocol. A lister that saw only one would make a re-key pass
        silently skip every slot in the other."""
        with self._lock:
            return sorted(
                {tok for (oid, tok) in self._postings if oid == principal_id}
                | {tok for (oid, tok) in self._entries if oid == principal_id}
            )

    def list_owners(self) -> List[str]:
        with self._lock:
            return sorted(
                {oid for (oid, _tok) in self._postings}
                | {oid for (oid, _tok) in self._entries}
                | {oid for (oid, _aid) in self._manifests}
            )

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

    # Analyzer generation
    def analyzer_generation(self) -> Optional[int]:
        with self._lock:
            return self._analyzer

    def record_analyzer_generation(self, generation: int) -> None:
        with self._lock:
            self._analyzer = int(generation)


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
