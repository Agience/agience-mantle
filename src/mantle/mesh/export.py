"""Export / import a VERIFIED shard — the mesh's server-independent content unit
(MANTLE-MESH.md "what to build first" #1).

A node exports its local shard as: (1) content bytes pushed to a content store
(Garage), addressed by sha256 so a replica serves them BLIND; (2) a signed
`ShardManifest` (manifest.py) whose Merkle `content_root` binds those hashes. Any
peer can then IMPORT the shard and verify — against the signed manifest, not the
server — that every byte is authentic. This is what lets the genesis seed die and
replicas take over: a replica's copy is exactly as trustworthy as the origin's.

Single-owner leaf: content is the owner's own plaintext (no light-cone). Envelope
encryption for meshed multi-tenant data is the deferred, security-critical layer;
the manifest math is identical either way (it hashes whatever bytes it is given).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .manifest import (
    ShardManifest,
    build_manifest,
    item_hash,
    sign_manifest,
    verify_manifest,
)

# Content is stored content-addressed under this prefix in the content store.
CONTENT_PREFIX = "cas/"


def _cas_key(h: str) -> str:
    return f"{CONTENT_PREFIX}{h}"


def export_shard(
    artifact_store,
    content_store,
    *,
    region_id: str,
    version: int,
    authority_id: str,
    priv: Ed25519PrivateKey,
    state: str = "committed",
    content_field: str = "content",
    limit: Optional[int] = None,
    skip_existing: bool = False,
) -> ShardManifest:
    """Push every artifact's content to the content store (content-addressed) and
    return a SIGNED manifest over the shard. Content-addressed writes are idempotent,
    so re-export overwrites identical bytes; `skip_existing` trades a HEAD per item to
    avoid re-PUTs (only worthwhile when most content is already uploaded)."""
    items: Dict[str, bytes] = {}
    for a in artifact_store.list_artifacts(state=state, limit=limit):
        aid = a["id"]
        body = (a.get(content_field) or "").encode("utf-8")
        h = item_hash(body)
        key = _cas_key(h)
        if not (skip_existing and content_store.exists(key)):
            content_store.put(key, body)
        items[aid] = body
    manifest = build_manifest(region_id, version, authority_id, items)
    return sign_manifest(manifest, priv)


def import_shard(
    manifest: ShardManifest,
    authority_pub: Ed25519PublicKey,
    content_store,
    into_artifact_store,
    *,
    hydrate: bool = True,
) -> Tuple[int, int]:
    """Verify the manifest (signature + content_root), then fetch each item's
    content from the content store BY HASH, verify the byte hash matches the
    manifest leaf, and (if hydrate) write the artifact into the target store.

    Returns (verified_items, imported_items). Raises on a tampered manifest or a
    content byte that fails its hash — a malicious peer cannot forge either."""
    if not verify_manifest(manifest, authority_pub):
        raise ValueError("shard manifest failed verification (bad signature or content_root)")

    verified = imported = 0
    for si in manifest.items:
        aid, h = si["id"], si["hash"]
        body = content_store.get(_cas_key(h))
        if item_hash(body) != h:
            raise ValueError(f"content for {aid} fails its hash — tampered or corrupt")
        verified += 1
        if hydrate:
            text = body.decode("utf-8", "ignore")
            into_artifact_store.put_artifact({
                "id": aid,
                "content_type": "text/plain",
                "state": "committed",
                # offer is not carried in the manifest leaf; a fuller manifest would
                # include it. For now content-first import; describe re-runs on demand.
                "context": text.split("\n", 1)[0][:200],
                "content": text,
                "created_by": manifest.authority,
            })
            imported += 1
    return verified, imported
