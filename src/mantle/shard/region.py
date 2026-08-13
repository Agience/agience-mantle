"""Blinded region ids — deterministic for a person's own devices, opaque to everyone else.

## Why salt rather than randomise

    region_id = HMAC-SHA256(secret, "v2|<principal>|<collection>|<cluster>")

Properties, each of which the cleartext form lacked:

| property | why it matters |
|---|---|
| **deterministic** given the secret | devices agree; direct device-to-device sync survives |
| **opaque** without the secret | no oracle: you cannot compute a region you were not given |
| **unlinkable** | HMAC outputs share no prefix, so two regions of one person look unrelated |
| **concept-hiding** | the cluster is inside the MAC, so the region reveals no anchor |

SHA-256 is Merkle–Damgård, so a naive `sha256(secret + msg)` is length-extendable: an attacker who
learns one region id can forge others without the secret. HMAC is the construction that is actually
keyed: it is one line either way, and only one of them is correct.

`local_collection.py` notes that a random id per install would fragment one person's data into
per-device universes that can never merge. The same failure class is silent when it happens: a
component that mints a fresh secret instead of loading the shared one partitions the node while
every health metric stays green — see `content.py`, which guards against exactly this on the
content-key path.

So `principal_secret(create=False)` is the default, a read never creates, and an absent secret
yields the legacy id with `blinded=False` reported — never a freshly-minted salt, and never a
silent one.

## What this does not fix — the object layout

`{prefix}/{principal}/{collection}/{cluster}.cell` (`anchor_routing.parse_cell_key`) still carries
the same three fields in the storage key. Blinding the region id while the object key stays
cleartext moves the leak rather than closing it. Rewriting that layout is a data migration, and is
deliberately not done here — see `remaining_leak()` for the current inventory of what is closed
and what remains open.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

# Bump only on a derivation change. It is inside the MAC so a v1 and a v2 id can never collide,
# and a future v3 rollout can address both without ambiguity.
SCHEME = "v2"

SECRET_FILENAME = "principal.secret"
SECRET_BYTES = 32


def principal_secret(keys_dir, *, create: bool = False) -> Optional[bytes]:
    """The per-principal blinding secret, shared across this person's devices.

    Provisioning a new device means copying this file, not generating one — same as `content.key`.
    """
    p = Path(keys_dir) / SECRET_FILENAME
    if p.is_file():
        raw = p.read_bytes()
        if len(raw) != SECRET_BYTES:
            return None
        return raw
    if not create:
        return None
    Path(keys_dir).mkdir(parents=True, exist_ok=True)
    secret = os.urandom(SECRET_BYTES)
    p.write_bytes(secret)
    try:
        p.chmod(0o600)
    except (OSError, NotImplementedError):
        pass
    return secret


# ── the MAC, prepared once per secret ────────────────────────────────────────────────────────────
# `hmac.new()` re-derives the padded key blocks on every call, so one prepared object per secret,
# `.copy()`-ed per message, avoids that.
#
_PREPARED: dict = {}


def _prepared(secret: bytes):
    h = _PREPARED.get(secret)
    if h is None:
        h = hmac.new(secret, digestmod=hashlib.sha256)
        _PREPARED[secret] = h
    return h


def _encode(*parts: str) -> bytes:
    """Unambiguous encoding of an ordered tuple.

    Length-prefixing is unambiguous for any content, because the length says exactly how many bytes
    to consume — no separator can be confused with data."""
    out = [SCHEME.encode("utf-8")]
    for p in parts:
        b = p.encode("utf-8")
        out.append(b"%d:" % len(b))
        out.append(b)
    return b"\x00".join(out)


def provision_secret(keys_dir, secret_hex: str) -> bytes:
    """Install a principal's blinding secret from another device (the coordinated-rollout path).
    Idempotent when the bytes already match; raises rather than overwriting a different secret,
    because that would re-key every cell this node routes to."""
    secret = bytes.fromhex(secret_hex.strip())
    if len(secret) != SECRET_BYTES:
        raise ValueError("a principal secret is %d bytes; got %d" % (SECRET_BYTES, len(secret)))
    p = Path(keys_dir) / SECRET_FILENAME
    if p.is_file():
        existing = p.read_bytes()               # binary — never strip (see principal_secret)
        if existing and existing != secret:
            raise ValueError(
                "refusing to overwrite a DIFFERENT principal.secret — that would re-key every cell "
                "this node routes to. Remove it deliberately if a re-key is intended. "
                "(existing fp=%s, offered fp=%s)"
                % (_fingerprint(existing), _fingerprint(secret)))
        return secret
    Path(keys_dir).mkdir(parents=True, exist_ok=True)
    p.write_bytes(secret)
    try:
        p.chmod(0o600)
    except (OSError, NotImplementedError):
        pass
    return secret


def _fingerprint(secret: bytes) -> str:
    """A short, non-reversible id for a secret — safe to log and compare across nodes."""
    return hmac.new(secret, b"fingerprint", hashlib.sha256).hexdigest()[:16]


def secret_fingerprint(keys_dir) -> Optional[str]:
    """This node's blinding-secret fingerprint, or None if unprovisioned. Compare across a
    principal's devices + serving peers before enabling blinding — a mismatch means they will not
    route to each other, and catching that here beats discovering it as a silent convergence stall."""
    secret = principal_secret(keys_dir)
    return _fingerprint(secret) if secret else None


def _mac(secret: bytes, *parts: str) -> str:
    """Keyed derivation. HMAC because SHA-256 is Merkle–Damgård and `sha256(secret + msg)` is
    length-extendable — an attacker with one region id could forge others without the secret."""
    h = _prepared(secret).copy()
    h.update(_encode(*parts))
    return h.hexdigest()


def blind(secret: Optional[bytes], *parts: str) -> Tuple[str, bool]:
    """`(id, blinded)` for an ordered tuple of components.

    Components are length-prefixed before the MAC (see `_encode`), so no two distinct tuples can
    encode to the same bytes.

    Returns `blinded=False` and the legacy cleartext form when no secret exists, so a caller can
    always tell a protected id from an unprotected one. It is never guessed.
    """
    if not secret:
        return "/".join(parts), False
    return _mac(secret, *parts), True


def cell_region(principal: str, collection: str, cluster: str, *,
                secret: Optional[bytes] = None) -> Tuple[str, bool]:
    """`(region_id, blinded)` for one anchor cell — the blinded form of
    `anchor_routing.cell_region`.

    Same triple + same secret → same id everywhere, so an authored cell and a routed query still
    name the same shard. Without the secret this is byte-identical to the legacy id, so existing
    data stays addressable during migration."""
    return blind(secret, principal, collection, cluster)


def local_collection_id(principal: str, *, secret: Optional[bytes] = None) -> Tuple[str, bool]:
    """`(collection_id, blinded)`.

    Unsalted, this is `uuid5(_EMBER_NS, "local-collection:" + principal)` over a published namespace
    constant — so anyone who knows the email computes the id. Salted, it is a MAC under the person's
    own secret. The blinded form keeps UUID shape (RFC 4122 v4 bits set on MAC output) so every
    consumer expecting a UUID is unaffected."""
    from mantle.shard.local_collection import local_collection_id as _legacy
    if not principal:
        raise ValueError("a local collection needs a principal to belong to")
    if not secret:
        return _legacy(principal), False
    h = _prepared(secret).copy()
    h.update(_encode("local-collection", principal))
    return str(uuid.UUID(bytes=h.digest()[:16], version=4)), True


def remaining_leak() -> dict:
    """What is still exposed after blinding the region id. Reported, not forgotten.

    Blinding the region id while the storage key still spells out the same three fields moves the
    leak rather than closing it. Rewriting `{prefix}/{principal}/{collection}/{cluster}.cell` is a
    data migration and is deliberately left as an open item, tracked below rather than fixed here."""
    return {
        "closed": ["region_id: principal was cleartext",
                   "region_id: collection was publicly derivable from the principal",
                   "region_id: cluster named the concept region",
                   "region_id: shared principal/collection prefix made a person's regions groupable"],
        "open": [
            {"where": "anchor_routing.parse_cell_key / cell object keys",
             "leak": "{prefix}/{principal}/{collection}/{cluster}.cell still carries all three "
                     "fields in cleartext in the STORAGE KEY",
             "why_not_fixed": "changing the object layout is a data migration; the fleet is under "
                              "a freeze (QUIESCED-FOR-MIGRATION.md). John's call.",
             "note": "parse_cell_key is INVERTIBLE by design and cannot be, once keys are blinded — "
                     "a node computes forward from its own secret and never needs to invert."},
            {"where": "mesh directory / manifests",
             "leak": "a peer still learns WHICH region ids a node holds and how many",
             "mitigation": "with blinding those ids are opaque and unlinkable, so the count is no "
                           "longer attributable to a person or a concept"},
        ],
    }


__all__ = ["SCHEME", "SECRET_FILENAME", "SECRET_BYTES", "principal_secret", "provision_secret",
           "secret_fingerprint", "blind", "cell_region", "local_collection_id", "remaining_leak"]
