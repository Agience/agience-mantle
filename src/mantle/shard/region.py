"""Blinded region ids — deterministic for a person's own devices, opaque to everyone else.

## The leak this closes (AGENT-HOST-DESIGN.md Phase 0.2)

`anchor_routing.cell_region` built a region id as::

    f"{principal}/{collection}/{cluster}"        # anchor_routing.py:39

Three separate disclosures in one string, and the known-gap note in `local_collection.py` UNDERSTATES
it — it says *"anyone holding a principal id can compute their collection id"*, but in fact:

1. **The principal is in CLEARTEXT.** No computation required; the email is right there.
2. **The collection is publicly derivable** — `uuid5(_EMBER_NS, "local-collection:" + principal)`
   over a published namespace constant.
3. **The cluster names the CONCEPT region**, and anchor ids come from the shared AnchorSet.

Together: ask a peer "do you hold region X?" and you learn *does this person hold data about this
concept*. And because every one of a person's regions shares the `principal/collection/` prefix,
their whole footprint is groupable at a glance — you do not even need to ask about a concept to
count how much they have.

## Why salt rather than randomise

The determinism is REQUIRED and must survive (`local_collection.py`): the same person on a laptop
and a desktop must derive the SAME ids, or their devices fragment into universes that can never
merge — *"and the failure would be invisible: each device would work fine alone."* So the fix is a
**keyed** derivation, not a random one: a per-principal secret that a person's own devices share.

    region_id = HMAC-SHA256(secret, "v2|<principal>|<collection>|<cluster>")

Properties, each of which the cleartext form lacked:

| property | why it matters |
|---|---|
| **deterministic** given the secret | devices agree; direct device-to-device sync survives |
| **opaque** without the secret | no oracle: you cannot compute a region you were not given |
| **unlinkable** | HMAC outputs share no prefix, so two regions of one person look unrelated |
| **concept-hiding** | the cluster is INSIDE the MAC, so the region reveals no anchor |

## ⚠ HMAC, not `hash(secret + data)`

SHA-256 is Merkle–Damgård, so a naive `sha256(secret + msg)` is length-extendable: an attacker who
learns one region id can forge others without the secret. HMAC is the construction that is actually
keyed. This is not paranoia about a theoretical attack — it is one line either way, and only one of
them is correct.

## ⛔ A MISSING SECRET NEVER MINTS ONE

`local_collection.py` states the failure exactly: *"A random id per install would fragment one
person's data into per-device universes that can never merge."* And `content.py` records the same
class of bug actually happening — `_content_key` minted a fresh key on the READ path when the file
was absent, silently partitioning the node while every health metric stayed green.

So `principal_secret(create=False)` is the default, a read never creates, and an absent secret
yields the LEGACY id with `blinded=False` reported — never a freshly-minted salt, and never a
silent one.

## What this does NOT fix — the object layout

`{prefix}/{principal}/{collection}/{cluster}.cell` (`anchor_routing.parse_cell_key`) still carries
the same three fields in the STORAGE KEY. Blinding the region id while the object key stays
cleartext moves the leak rather than closing it. Rewriting that layout is a **data migration** and
the fleet is under a freeze, so it is deliberately not done here: see `remaining_leak()`, which
reports it as a live gap rather than letting it be forgotten.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

# Bump ONLY on a derivation change. It is inside the MAC so a v1 and a v2 id can never collide,
# and a future v3 rollout can address both without ambiguity.
SCHEME = "v2"

SECRET_FILENAME = "principal.secret"
SECRET_BYTES = 32


def principal_secret(keys_dir, *, create: bool = False) -> Optional[bytes]:
    """The per-principal blinding secret, shared across THIS person's devices.

    ⛔ `create=False` BY DEFAULT, AND THE READ PATH MUST NEVER PASS TRUE. Minting a secret on read
    would give each device a different one, so each would derive different region ids for the same
    data — the devices would still work alone and would silently never converge. That is the exact
    failure `local_collection.py` warns about and the one `content.py` actually suffered.

    Provisioning a new device means COPYING this file, not generating one — same as `content.key`.
    """
    p = Path(keys_dir) / SECRET_FILENAME
    if p.is_file():
        # ⛔ NEVER `.strip()` A BINARY SECRET. `os.urandom(SECRET_BYTES)` is arbitrary bytes, so
        # ~4.7% of secrets have a leading/trailing ASCII-whitespace byte (0x09/0x0a/0x0d/0x20/…).
        # Stripping silently CORRUPTS those on read-back, so the node derives wrong region ids and
        # silently partitions from its own other devices — the exact failure this module guards
        # against, and it surfaced as a flaky provisioning test (~1 in 20). Read exact bytes; a file
        # of the wrong length is not a usable secret and must not be half-used.
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
# ⚠ THE FIRST VERSION OF THIS CACHE WAS SLOWER THAN NO CACHE, and the first version of THIS COMMENT
# claimed a speedup that was never measured ("11.28 -> ~1 us"). Both are the same failure this repo
# keeps recording, committed twice in five minutes.
#   · Keying on `hashlib.sha256(secret).digest()` — so the key material would not also be a dict key
#     — spends a full SHA-256 per lookup to save one key-padding: 11.28 us -> 14.65 us. Removed.
#   · Keyed on the secret bytes directly (hashable, and the secret is already resident — a dict
#     entry leaks nothing further), MEASURED like-for-like on identical work:
#         prepared + copy + dict lookup   3.61 us
#         plain hmac.new                  4.30 us
#     ~16% faster. Real, worth keeping, and NOT the order of magnitude first claimed. An earlier
#     "naive 3.06 us" figure was apples-to-oranges — a short literal message, not `_encode` output.
_PREPARED: dict = {}


def _prepared(secret: bytes):
    h = _PREPARED.get(secret)
    if h is None:
        h = hmac.new(secret, digestmod=hashlib.sha256)
        _PREPARED[secret] = h
    return h


def _encode(*parts: str) -> bytes:
    """UNAMBIGUOUS encoding of an ordered tuple.

    ⛔ THE OBVIOUS VERSION IS WRONG AND THIS MODULE SHIPPED IT FIRST. Joining with a separator —
    `"|".join(parts)` — means `("a|b", "c")` and `("a", "b|c")` both encode to `a|b|c`, so two
    DIFFERENT regions derive the SAME id. Caught by the module's own encoding test, which is the
    only reason it is not still here. A collision like that is a correctness bug before it is a
    privacy one: two unrelated cells would silently share a shard.

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

    ⛔ THIS IS HOW A SECRET REACHES A SECOND DEVICE OR A SERVING PEER — never by minting a fresh one
    (that fragments routing; see `principal_secret`). One device generates via
    `principal_secret(create=True)`, reads `secret_fingerprint`, and installs the SAME bytes here.
    Refuses to overwrite a DIFFERENT existing secret, because silently replacing it would re-key
    every cell this node routes to. Idempotent when the bytes already match."""
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
    """A short, non-reversible id for a secret — safe to log and compare across nodes.

    ⚠ A fingerprint of the SECRET ITSELF would help an attacker confirm a guess, so this is
    HMAC(secret, "fingerprint") — it identifies the secret without exposing anything that helps
    recover it. [memory: content-encryption — *"fingerprint keys first"*, the check that would have
    caught the mesh `content.key` divergence before it silently partitioned the fleet.]"""
    return hmac.new(secret, b"fingerprint", hashlib.sha256).hexdigest()[:16]


def secret_fingerprint(keys_dir) -> Optional[str]:
    """This node's blinding-secret fingerprint, or None if unprovisioned. Compare across a
    principal's devices + serving peers BEFORE enabling blinding — a mismatch means they will not
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

    Components are LENGTH-PREFIXED before the MAC (see `_encode`), so no two distinct tuples can
    encode to the same bytes.

    Returns `blinded=False` and the LEGACY cleartext form when no secret exists, so a caller can
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

    Unsalted, this is `uuid5(_EMBER_NS, "local-collection:" + principal)` over a PUBLISHED namespace
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
    """⚠ WHAT IS STILL EXPOSED AFTER BLINDING THE REGION ID. Reported, not forgotten.

    Blinding the region id while the STORAGE KEY still spells out the same three fields moves the
    leak rather than closing it. Rewriting `{prefix}/{principal}/{collection}/{cluster}.cell` is a
    data migration, and the fleet is frozen, so it is a deliberate open item — not an oversight."""
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
