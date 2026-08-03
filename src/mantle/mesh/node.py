"""Mantle mesh node — the in-process shard store + verified sync (MANTLE-MESH.md §5).

A node holds some subset of anchor-region shards (a domain, a sparse set, or many)
plus the lightweight manifest map. It can:
  * ``put_shard``   — author/refresh a shard it is authoritative for (signs a manifest)
  * ``manifests``   — advertise its map (the DHT directory entries)
  * ``get_shard``   — serve a region's (encrypted) items + manifest, blind
  * ``import_shard``— accept a peer's shard ONLY if signature + every item hash +
                      content_root verify (tamper-evident, server-independent)
  * ``sync_from``   — pull regions that are missing or newer than what we hold

This is the transport-agnostic core; a libp2p/HTTP layer (the DHT + gossip) wraps it.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .manifest import ShardManifest, build_manifest, item_hash, sign_manifest, verify_manifest


class ShardVerifyError(ValueError):
    """Raised when an incoming shard fails signature / hash / content_root checks."""


class MeshNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        # region_id -> {"manifest": ShardManifest, "items": {artifact_id: content_bytes}}
        self._shards: Dict[str, dict] = {}

    # -- authoring (a node that is the source-of-record for a region) ----------
    def put_shard(self, region_id: str, items: Dict[str, bytes], *, version: int,
                  authority: str, priv: Ed25519PrivateKey,
                  consensus: Optional[Dict[str, float]] = None) -> ShardManifest:
        manifest = sign_manifest(
            build_manifest(region_id, version, authority, items, consensus), priv)
        self._shards[region_id] = {"manifest": manifest, "items": dict(items)}
        return manifest

    # -- serving (the map + the blind content) --------------------------------
    def manifests(self) -> List[ShardManifest]:
        return [s["manifest"] for s in self._shards.values()]

    def get_shard(self, region_id: str) -> Tuple[Optional[ShardManifest], Optional[Dict[str, bytes]]]:
        s = self._shards.get(region_id)
        if not s:
            return None, None
        return s["manifest"], dict(s["items"])

    def has_region(self, region_id: str) -> bool:
        return region_id in self._shards

    def version_of(self, region_id: str) -> int:
        s = self._shards.get(region_id)
        return s["manifest"].version if s else -1

    # -- accepting a shard (verify EVERYTHING before trusting the server) ------
    def import_shard(self, manifest: ShardManifest, items: Dict[str, bytes],
                     authority_pub: Ed25519PublicKey) -> None:
        if not verify_manifest(manifest, authority_pub):
            raise ShardVerifyError(f"{manifest.region_id}: bad signature or content_root")
        for si in manifest.items:
            content = items.get(si["id"])
            if content is None:
                raise ShardVerifyError(f"{manifest.region_id}: item {si['id']} missing")
            if item_hash(content) != si["hash"]:
                raise ShardVerifyError(f"{manifest.region_id}: item {si['id']} hash mismatch")
        self._shards[manifest.region_id] = {"manifest": manifest, "items": dict(items)}

    # -- sync: pull regions that are missing or newer than ours ---------------
    def sync_from(self, peer: "MeshNode", authority_pub: Ed25519PublicKey,
                  regions: Optional[Iterable[str]] = None) -> List[str]:
        """Pull from ``peer`` every region we lack or that has a higher version.
        Returns the region ids actually synced. Verification is per-shard: a bad
        shard raises and is skipped, it never corrupts what we already hold.

        ``regions`` restricts the pull to a specific set of region ids — this is
        how an **anchor-routed** query fetches only the cells it needs: route the
        query to its nearest anchors (``anchor_routing.route_query_regions``) and
        pass those region ids here, so a node syncs its working set, not the world.
        """
        want = set(regions) if regions is not None else None
        synced: List[str] = []
        for pm in peer.manifests():
            if want is not None and pm.region_id not in want:
                continue
            if pm.version > self.version_of(pm.region_id):
                manifest, items = peer.get_shard(pm.region_id)
                if manifest is None:
                    continue
                self.import_shard(manifest, items, authority_pub)   # raises on tamper
                synced.append(pm.region_id)
        return synced

    def summary(self) -> dict:
        return {
            "node_id": self.node_id,
            "regions": {r: {"version": s["manifest"].version, "density": s["manifest"].density}
                        for r, s in sorted(self._shards.items())},
        }
