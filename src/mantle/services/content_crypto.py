"""Per-principal content envelope encryption.

Artifact content bytes are encrypted with a per-principal AES-256-GCM key derived
from the principal's MANTLE master key — the same KEK-wrapped per-principal key that
already protects the encrypted search index (``search/mantle/oracle.py``). No new key
system: content, search cells, and secrets all hang off one custody hierarchy, so a
compromised object store yields only ciphertext and no keys.

Wire format (versioned, self-describing):

    b"MEC1"  ‖  nonce(12)  ‖  AES-256-GCM(ciphertext ‖ tag)
    │ magic     │ random       │ AAD = principal_id (binds the blob to its owner)

**Backward compatible**: a blob WITHOUT the ``MEC1`` magic is treated as legacy
plaintext and returned as-is on decrypt — existing content stays readable and is
re-encrypted on its next write. No migration required.

Storage-agnostic: this module knows nothing about S3/the lattice. The content layer calls
``encrypt_content`` before a write and ``decrypt_content`` after a read; where the
ciphertext lives is the storage layer's concern (and invisible to API callers).
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_MAGIC = b"MEC1"                                   # Mantle Encrypted Content, v1
_NONCE_LEN = 12
_HKDF_SALT = b"agience-mantle-content-key-v1"      # versioned, distinct from cell/sse salts
_HKDF_INFO = b"content"

# principal_id -> 32-byte master key. Injectable so the crypto is unit-testable
# without the oracle/DB, and so the content layer can pass a cached-oracle provider.
MasterKeyProvider = Callable[[str], bytes]


def _default_master_key(principal_id: str, collection_id: Optional[str] = None) -> bytes:
    """The content master key for ``principal_id``, checked against the grant ledger.

    ⚠ THIS WAS THE TAUTOLOGICAL CHECK. It previously built
    ``KeyRequest(requester_id=principal_id, purpose=SELF)`` — requester and principal
    were *the same variable*, read off the very document being decrypted. The oracle
    dutifully compared them, found them equal (they could not be otherwise), and
    issued the key. Anyone who could reach ``decrypt_content(owner, blob)`` read the
    plaintext, because the only thing the "check" proved was that a value equals
    itself.

    The docstring that stood here was honest about the gap and named the fix:
    thread the acting principal from the routers down through the db layer. That is
    now done (:mod:`services.acting_principal`), so this asks the question that was
    previously unaskable — *does THIS caller hold a grant reaching this content?* —
    and content-key issuance is genuinely grant-gated rather than resting entirely
    on ``check_access`` at the router.

    ``GRANT``, not ``SELF``: a reader is usually NOT the artifact's ``created_by``.
    Shared content is the normal case, so the light cone has to decide, exactly as it
    does on the search path. ``collection_id`` narrows that decision to the artifact's
    own context when the caller knows it; without it the check is principal-scoped
    (the requester must reach at least one collection under the owner).

    ``action="read"``: holding a content key IS the ability to read content, so read
    is the accurate right to demand — and demanding ``update`` on the encrypt path
    would wrongly reject a legitimate *create*, whose grant carries ``create`` rather
    than ``update``. Write authorization remains ``check_access``'s job at the router,
    where the verb is known exactly.
    """
    # _build_oracle() returns a process-cached OracleService (or None pre-setup),
    # so this does NOT rebuild the oracle or lose the master-key cache per call.
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest
    from mantle.search.mantle.wiring import _build_oracle

    from mantle.services.acting_principal import require_acting_principal

    oracle = _build_oracle()
    if oracle is None:
        raise RuntimeError("key oracle unavailable — cannot encrypt/decrypt content")

    actor = require_acting_principal()
    return oracle.get_or_create_master_key(
        principal_id,
        KeyRequest(
            requester_id=actor.principal_id, purpose=KeyPurpose.GRANT,
            requester_type=actor.principal_type, action="read",
        ),
        collection_id=collection_id,
    )


def _provider_for(collection_id: Optional[str]) -> MasterKeyProvider:
    """Bind ``collection_id`` into the default provider.

    Keeps :data:`MasterKeyProvider` a one-argument ``(principal_id) -> bytes``
    callable so injected test//cache providers keep working unchanged, while still
    letting the real one narrow its grant check to a specific collection.
    """
    def _provider(principal_id: str) -> bytes:
        return _default_master_key(principal_id, collection_id)

    return _provider


def _content_key(principal_id: str, provider: MasterKeyProvider) -> bytes:
    master_key = provider(principal_id)
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=_HKDF_SALT, info=_HKDF_INFO)
    return hkdf.derive(master_key)


def is_encrypted(blob: bytes) -> bool:
    """True if *blob* is Mantle-encrypted content (has the MEC1 magic)."""
    return isinstance(blob, (bytes, bytearray)) and bytes(blob[:4]) == _MAGIC


def encrypt_content(
    principal_id: str,
    plaintext: bytes,
    *,
    collection_id: Optional[str] = None,
    master_key_provider: Optional[MasterKeyProvider] = None,
) -> bytes:
    """Encrypt *plaintext* for *principal_id*. Returns ``MEC1‖nonce‖ct``.

    ``collection_id`` narrows the grant check to the artifact's own context; omitting
    it falls back to a principal-scoped check. Requires an acting principal in scope
    (see :func:`_default_master_key`) unless a ``master_key_provider`` is injected.
    """
    if not principal_id:
        raise ValueError("principal_id is required to encrypt content")
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    key = _content_key(
        principal_id, master_key_provider or _provider_for(collection_id)
    )
    nonce = os.urandom(_NONCE_LEN)
    # AAD binds the ciphertext to its owner: a blob re-homed under another principal
    # won't authenticate.
    ct = AESGCM(key).encrypt(nonce, plaintext, principal_id.encode("utf-8"))
    return _MAGIC + nonce + ct


def decrypt_content(
    principal_id: str,
    blob: bytes,
    *,
    collection_id: Optional[str] = None,
    master_key_provider: Optional[MasterKeyProvider] = None,
) -> bytes:
    """Decrypt a Mantle-encrypted *blob* for *principal_id*.

    Legacy plaintext (no MEC1 magic) is returned unchanged (backward compatible).
    Raises ``cryptography.exceptions.InvalidTag`` if the ciphertext or owner is wrong,
    and ``GrantDenied`` / ``NoActingPrincipal`` if the caller may not hold this key.

    ⚠ The legacy-plaintext branch returns BEFORE any key is needed, so an unencrypted
    legacy blob is still readable without a grant. That is pre-existing behaviour and
    is not a new hole — such a row is plaintext in the store regardless — but it does
    mean this function is only an access control for rows that are actually encrypted.
    """
    if not is_encrypted(blob):
        return bytes(blob)
    if not principal_id:
        raise ValueError("principal_id is required to decrypt content")
    key = _content_key(
        principal_id, master_key_provider or _provider_for(collection_id)
    )
    nonce = blob[4:4 + _NONCE_LEN]
    ct = blob[4 + _NONCE_LEN:]
    return AESGCM(key).decrypt(nonce, ct, principal_id.encode("utf-8"))
