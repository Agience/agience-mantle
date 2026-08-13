"""Per-principal content envelope encryption.

Artifact content bytes are encrypted with a per-principal AES-256-GCM key derived
from the principal's MANTLE master key — the same KEK-wrapped per-principal key that
already protects the encrypted search index (``search/mantle/oracle.py``). No new key
system: content, search cells, and secrets all hang off one custody hierarchy, so a
compromised object store yields only ciphertext and no keys.

Wire format (versioned, self-describing):

    b"MEC1"  ‖  nonce(12)  ‖  AES-256-GCM(ciphertext ‖ tag)
    │ magic     │ random       │ AAD — see below

Associated data binds the ciphertext to its *slot*, not just its owner:

    scoped  (collection_id known)   AAD = b"mec1-aad-v2:<len>:<principal>:<scope>"
    legacy  (no collection_id)      AAD = principal_id

The scoped form closes the gap where a blob moved between two collections under
the same origin-root principal still authenticated. ``<len>`` is the byte length
of ``principal_id``, so the two fields cannot be re-partitioned by a principal id
that happens to contain the separator.

**Dual-read, mandatory** (see ``db/content_cache.py``'s
``legacy_key_for_collection`` — same house pattern): decrypt tries the scoped AAD
first and falls back to the legacy owner-only AAD. What that rescues is ciphertext
written under the OWNER-ONLY binding, which carries no scope and therefore opens in
any — including such a blob whose artifact has since moved collection. Writes always
use the new binding. Until a re-encrypt migration has run and the fallback is removed,
the scoped binding is *recorded* for the legacy corpus, not yet *enforced* on it;
:data:`legacy_aad_reads` counts how much corpus is still on the old form.

**A scoped blob does not travel.** ``MEC1`` records the nonce and nothing else, so a
blob written for ``(principal, col-A)`` presented under ``col-B`` has no old scope to
recover: the scoped AAD misses and the legacy AAD misses too, and the read raises
``InvalidTag``. That is the binding doing its job rather than a gap in it — "this
ciphertext authenticates only in the slot it was written for" is the whole property,
and it cannot hold while also opening in a slot it was not written for. Storing the
write scope inside the blob would make the AAD self-supplied and authenticate
everything. Moving content between collections is therefore a RE-ENCRYPT, on the write
path, under the new scope; the artifact API exposes no move today (``update_artifact``
rejects a target whose ``collection_id`` is not the workspace it was addressed
through), so nothing performs one.

**Backward compatible**: a blob WITHOUT the ``MEC1`` magic is treated as legacy
plaintext and returned as-is on decrypt — existing content stays readable and is
re-encrypted on its next write. No migration required. Callers that KNOW a blob
must be encrypted (the artifact path has a ``content_encrypted`` flag) should
pass ``require_encrypted=True`` to refuse that downgrade.

Storage-agnostic: this module knows nothing about S3/the lattice. The content layer calls
``encrypt_content`` before a write and ``decrypt_content`` after a read; where the
ciphertext lives is the storage layer's concern (and invisible to API callers).
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

_MAGIC = b"MEC1"                                   # Mantle Encrypted Content, v1
_NONCE_LEN = 12
_HKDF_SALT = b"agience-mantle-content-key-v1"      # versioned, distinct from cell/sse salts
_HKDF_INFO = b"content"

#: Prefix for the scoped (owner + collection) AAD form. Versioned so a v3 binding can
#: coexist with v2 ciphertext during a migration, exactly as `_HKDF_SALT` does for keys.
_AAD_V2_PREFIX = b"mec1-aad-v2:"

#: How many reads have fallen back to the legacy owner-only AAD. The one number an
#: operator needs before removing the fallback: while it climbs, un-migrated ciphertext
#: still exists and dropping the fallback would orphan it.
#: A plain int — increments are single bytecode ops under the GIL and this is a metric,
#: not a control. Reset it to sample a window.
legacy_aad_reads: int = 0

# principal_id -> 32-byte master key. Injectable so the crypto is unit-testable
# without the oracle/DB, and so the content layer can pass a cached-oracle provider.
MasterKeyProvider = Callable[[str], bytes]


def _default_master_key(principal_id: str, collection_id: Optional[str] = None,
                        *, may_create: bool = False, creator_id: Optional[str] = None) -> bytes:
    """The content master key for ``principal_id``, checked against the grant ledger.

    The acting principal is threaded from the routers down through the db layer
    (:mod:`services.acting_principal`), so this asks *does THIS caller hold a grant
    reaching this content?* and content-key issuance is grant-gated rather than resting
    entirely on ``check_access`` at the router.

    ``GRANT``, not ``SELF``: a reader is NOT required to be the artifact's ``created_by``.
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
            # Encrypt mints, decrypt does not: `action` stays "read" for the reason above; this
            # carries the other half. Minting on decrypt would hand back a fresh key that
            # decrypts nothing and reports "no results" instead of surfacing a missing key.
            may_create=may_create,
            # Only ever non-None on the encrypt path — see `KeyRequest.creator_id`.
            creator_id=creator_id,
        ),
        collection_id=collection_id,
    )


def _provider_for(collection_id: Optional[str], *, may_create: bool = False,
                  creator_id: Optional[str] = None) -> MasterKeyProvider:
    """Bind ``collection_id`` into the default provider.

    Keeps :data:`MasterKeyProvider` a one-argument ``(principal_id) -> bytes``
    callable so injected test//cache providers keep working unchanged, while still
    letting the real one narrow its grant check to a specific collection.
    """
    def _provider(principal_id: str) -> bytes:
        return _default_master_key(principal_id, collection_id, may_create=may_create,
                                   creator_id=creator_id)

    return _provider


def _content_key(principal_id: str, provider: MasterKeyProvider) -> bytes:
    master_key = provider(principal_id)
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=_HKDF_SALT, info=_HKDF_INFO)
    return hkdf.derive(master_key)


def _legacy_aad(principal_id: str) -> bytes:
    """The v1 binding: the owner and nothing else. Decrypt-only — never written now."""
    return principal_id.encode("utf-8")


def _scoped_aad(principal_id: str, collection_id: str) -> bytes:
    """The v2 binding: owner AND the content's scope (collection / origin root).

    Length-prefixed on ``principal_id`` so the two fields are unambiguous: without
    it a principal id containing the separator could be re-partitioned to forge a
    different (owner, scope) pair that encodes to the same bytes.
    """
    return _AAD_V2_PREFIX + (
        f"{len(principal_id.encode('utf-8'))}:{principal_id}:{collection_id}"
    ).encode("utf-8")


def is_encrypted(blob: bytes) -> bool:
    """True if *blob* is Mantle-encrypted content (has the MEC1 magic)."""
    return isinstance(blob, (bytes, bytearray)) and bytes(blob[:4]) == _MAGIC


def encrypt_content(
    principal_id: str,
    plaintext: bytes,
    *,
    collection_id: Optional[str] = None,
    creator_id: Optional[str] = None,
    master_key_provider: Optional[MasterKeyProvider] = None,
) -> bytes:
    """Encrypt *plaintext* for *principal_id*. Returns ``MEC1‖nonce‖ct``.

    ``collection_id`` narrows the grant check to the artifact's own context; omitting
    it falls back to a principal-scoped check. Requires an acting principal in scope
    (see :func:`_default_master_key`) unless a ``master_key_provider`` is injected.

    When ``collection_id`` is supplied it is ALSO bound into the AEAD associated data,
    so the blob authenticates only in the scope it was written for. Omitting it writes
    the owner-only binding — unchanged from before, and still readable either way.
    """
    if not principal_id:
        raise ValueError("principal_id is required to encrypt content")
    if isinstance(plaintext, str):
        plaintext = plaintext.encode("utf-8")
    key = _content_key(
        principal_id,
        master_key_provider or _provider_for(collection_id, may_create=True, creator_id=creator_id),
    )
    nonce = os.urandom(_NONCE_LEN)
    # AAD binds the ciphertext to its owner AND (when known) its collection scope:
    # a blob re-homed under another principal — or moved between two collections of
    # the SAME origin-root principal — won't authenticate under the new binding.
    aad = _scoped_aad(principal_id, collection_id) if collection_id else _legacy_aad(principal_id)
    ct = AESGCM(key).encrypt(nonce, plaintext, aad)
    return _MAGIC + nonce + ct


def decrypt_content(
    principal_id: str,
    blob: bytes,
    *,
    collection_id: Optional[str] = None,
    require_encrypted: bool = False,
    master_key_provider: Optional[MasterKeyProvider] = None,
) -> bytes:
    """Decrypt a Mantle-encrypted *blob* for *principal_id*.

    Legacy plaintext (no MEC1 magic) is returned unchanged (backward compatible) —
    unless ``require_encrypted=True``, which makes that path a hard error. Pass it
    from any caller that independently KNOWS the blob is encrypted (the artifact path
    has a ``content_encrypted`` flag); otherwise the magic-check is an unauthenticated
    downgrade an attacker with store-write access could use to substitute chosen
    plaintext for authenticated ciphertext.

    **Dual-read.** With a ``collection_id`` the scoped AAD is tried first, then the
    legacy owner-only AAD. That second attempt is what keeps ciphertext written under
    the owner-only binding readable, whatever scope it is presented under. It is
    decrypt-only: writes always use the scoped binding.

    It does NOT make a scoped blob portable. One written for another collection misses
    both bindings and raises — see the module docstring; content changes collection by
    being re-encrypted, not by being read somewhere else.

    Raises ``cryptography.exceptions.InvalidTag`` if neither binding authenticates,
    and ``GrantDenied`` / ``NoActingPrincipal`` if the caller may not hold this key.
    """
    if not is_encrypted(blob):
        if require_encrypted:
            raise ValueError(
                "blob is not Mantle-encrypted (no MEC1 magic) but the caller asserted it "
                "must be — refusing to return unauthenticated bytes as content"
            )
        return bytes(blob)
    if not principal_id:
        raise ValueError("principal_id is required to decrypt content")
    key = _content_key(
        principal_id, master_key_provider or _provider_for(collection_id)
    )
    nonce = blob[4:4 + _NONCE_LEN]
    ct = blob[4 + _NONCE_LEN:]
    global legacy_aad_reads
    aead = AESGCM(key)
    if not collection_id:
        return aead.decrypt(nonce, ct, _legacy_aad(principal_id))
    try:
        return aead.decrypt(nonce, ct, _scoped_aad(principal_id, collection_id))
    except InvalidTag:
        pass                           # fall through to the legacy owner-only binding
    plain = aead.decrypt(nonce, ct, _legacy_aad(principal_id))
    # Counted only on SUCCESS, so the number means "un-migrated ciphertext that is
    # really out there" and not "decrypt failures", which would make it useless as the
    # signal for when the fallback can be dropped.
    legacy_aad_reads += 1
    logger.debug(
        "content blob for principal %s opened only under the LEGACY owner-only AAD "
        "(collection %r) — it predates the scoped binding. Re-encrypt it to migrate.",
        principal_id, collection_id,
    )
    return plain
