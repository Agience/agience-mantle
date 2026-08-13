"""Mantle mesh — content-addressed, signed shard manifests (MANTLE-MESH.md §2–4).

A **shard** is a set of anchor-region artifacts. The **manifest** is the verifiable
directory entry the DHT stores: it names the region, its version (freshness), the
authority that signed it, density stats, and a Merkle **content_root** over the
shard's items. Because the root is over the *item hashes* (of the already-encrypted
content bytes), any peer can serve a shard **blind** and any reader can verify it is
authentic **independent of who served it** — the property that lets the genesis seed
die and replicas take over.

Signing uses Ed25519 (small, permissive) — the mesh authority key, distinct from the
per-artifact envelope encryption. Verification needs only the authority's public key.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from prism.canonical import canonical_string as _jcs_string


def item_hash(content: bytes) -> str:
    """sha256 of an item's (encrypted) content bytes — the content address."""
    return hashlib.sha256(content).hexdigest()


def content_root(item_hashes: List[str]) -> str:
    """Deterministic Merkle-style root over the shard's leaf hashes.

    Leaves are **sorted** before folding, so item ordering never changes the root
    (a shard is a set, not a list). A full binary Merkle tree with inclusion proofs
    is a later refinement; a sorted hash-of-hashes is deterministic + verifiable and
    is enough for shard-level integrity.
    """
    h = hashlib.sha256()
    for ih in sorted(item_hashes):
        h.update(bytes.fromhex(ih))
    return h.hexdigest()


@dataclass
class ShardItem:
    """One item's row in a manifest — and, because the manifest names its signing `authority`,
    ONE OBSERVER'S ATTESTATION of that item.

    `origin` lets agreement count distinct ORIGINS, not distinct holders, because N replicas of
    one origin are one observation and not N. Empty means unattributed — this authority holds the
    bytes but cannot say who originated them, which is the honest state whenever origin cannot be
    determined.
    """

    id: str            # artifact id
    hash: str          # sha256 of the (encrypted) content bytes
    size: int
    origin: str = ""   # authority that ORIGINATED it ("self" => the signing authority); "" unknown
    channel: str = ""  # how THIS authority obtained it — a fact about acquisition, not a rank


@dataclass
class ShardManifest:
    region_id: str
    version: int               # monotonic; bumped on an authoritative update
    authority: str             # authority key id (who signs this region)
    density: int               # number of items
    content_root: str          # Merkle root over item hashes
    items: List[dict] = field(default_factory=list)   # list of ShardItem dicts
    signature: str = ""        # hex Ed25519 signature over canonical()

    # The bytes that are signed / verified — everything but the signature, canonical.
    def canonical(self) -> bytes:
        d = {
            "region_id": self.region_id,
            "version": self.version,
            "authority": self.authority,
            "density": self.density,
            "content_root": self.content_root,
            "items": self.items,
        }
        return _jcs_string(d).encode()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShardManifest":
        keys = ("region_id", "version", "authority", "density", "content_root", "items", "signature")
        return cls(**{k: d[k] for k in keys})


def build_manifest(region_id: str, version: int, authority: str,
                   items: Dict[str, bytes],
                   attest: Dict[str, dict] | None = None) -> ShardManifest:
    """Assemble an (unsigned) manifest from a shard's ``{artifact_id: content_bytes}``.

    ``attest`` is ``{artifact_id: {"origin": ..., "channel": ...}}`` — what THIS authority observed
    about each item. An item with no entry is published unattributed, which is a true statement
    about a node that does not know where the bytes came from, and is read as such by
    `prism.attestation`. It is not published as a weight, because this node measuring a weight
    every other node would measure identically is not an observation.
    """
    attest = attest or {}
    shard_items = []
    for aid, c in items.items():
        a = attest.get(aid) or {}
        shard_items.append(ShardItem(id=aid, hash=item_hash(c), size=len(c),
                                     origin=str(a.get("origin") or ""),
                                     channel=str(a.get("channel") or "")))
    return ShardManifest(
        region_id=region_id,
        version=version,
        authority=authority,
        density=len(shard_items),
        content_root=content_root([si.hash for si in shard_items]),
        items=[asdict(si) for si in shard_items],
    )


def sign_manifest(m: ShardManifest, priv: Ed25519PrivateKey) -> ShardManifest:
    m.signature = priv.sign(m.canonical()).hex()
    return m


def verify_manifest(m: ShardManifest, pub: Ed25519PublicKey) -> bool:
    """True iff the signature is valid AND the content_root matches the items —
    i.e. the manifest is authentic and internally consistent."""
    try:
        pub.verify(bytes.fromhex(m.signature), m.canonical())
    except (InvalidSignature, ValueError):
        return False
    return content_root([si["hash"] for si in m.items]) == m.content_root
