"""Gossip region directory — resolve a routed region to the peer that holds it, best first.

The deployment-simple (stdlib-only) stand-in for the libp2p DHT of MANTLE-MESH.md §5.
A node folds peers' advertised manifests into ``region_id -> {peer -> ProviderInfo}``, so
a query routed by anchor (:func:`anchor_routing.route_query_regions`) resolves to *which*
peer to pull each cell from — preferring the **freshest, best-attributed, densest** source
(the "maps to where the source / denser information lives" property the mesh is for).

Directories **gossip** by :meth:`RegionDirectory.merge`: a node unions its neighbors'
views, so knowledge of who-holds-what spreads transitively with no central registry.
Trust is unchanged — the directory only says *where* to look; every pulled shard is still
verified against its ``content_root`` + authority signature, so a lying directory entry
can misdirect a fetch but never forge content (a bad shard just fails and we fail over).

Transport-agnostic: the core operates on ``(peer_id, manifests)`` and a ``get_shard``
callable, so it drives both in-process ``MeshNode`` sync and the HTTP wire layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .manifest import ShardManifest
from .node import MeshNode, ShardVerifyError


def attribution_of(manifest: ShardManifest) -> float:
    """What fraction of this replica's items name an ORIGIN — how well-provenanced the copy is.

    Attribution genuinely differs per replica, and it is the right preference: a copy that names
    origins is the copy a puller can compute agreement from, whereas an unattributed copy carries
    the bytes and nothing about who stands behind them. Fraction of the items, so it does not
    double-count density — which is already its own term in `rank`.
    """
    items = manifest.items or []
    if not items:
        return 0.0
    return sum(1 for i in items if i.get("origin")) / len(items)


@dataclass(frozen=True)
class ProviderInfo:
    """What a directory knows about one peer's copy of one region."""
    version: int
    density: int
    attribution: float

    # Best-first sort key: freshest wins; then best-attributed; then densest.
    def rank(self) -> Tuple[int, float, int]:
        return (self.version, self.attribution, self.density)


class RegionDirectory:
    """``region_id -> {peer_id -> ProviderInfo}`` with gossip merge + a pull plan."""

    def __init__(self) -> None:
        self._map: Dict[str, Dict[str, ProviderInfo]] = {}

    # ---------------------------------------------------------------- ingest
    def observe(self, peer_id: str, manifests: Sequence[ShardManifest]) -> None:
        """Fold a peer's advertised manifests into the map (newer version wins per peer)."""
        for m in manifests:
            info = ProviderInfo(version=int(m.version), density=int(m.density),
                                attribution=attribution_of(m))
            peers = self._map.setdefault(m.region_id, {})
            cur = peers.get(peer_id)
            if cur is None or info.version >= cur.version:
                peers[peer_id] = info

    def merge(self, other: "RegionDirectory") -> "RegionDirectory":
        """Gossip: union another directory into this one (newer version per (region, peer)
        wins). Returns self for chaining. This is how who-holds-what spreads transitively."""
        for region, peers in other._map.items():
            mine = self._map.setdefault(region, {})
            for peer_id, info in peers.items():
                cur = mine.get(peer_id)
                if cur is None or info.version >= cur.version:
                    mine[peer_id] = info
        return self

    # ---------------------------------------------------------------- query
    def providers(self, region_id: str) -> List[Tuple[str, ProviderInfo]]:
        """Peers holding ``region_id``, best source first (freshest/most-coherent/densest)."""
        peers = self._map.get(region_id, {})
        return sorted(peers.items(), key=lambda kv: kv[1].rank(), reverse=True)

    def plan(self, regions: Sequence[str]) -> "Dict[str, List[str]]":
        """For a routed region list, the ordered peer ids to try for each — the pull plan.
        A region with no known provider maps to ``[]`` (a coverage gap to surface, not hide)."""
        return {r: [peer for peer, _ in self.providers(r)] for r in regions}

    def regions(self) -> List[str]:
        return sorted(self._map)

    def peers(self) -> List[str]:
        return sorted({p for peers in self._map.values() for p in peers})

    # ---------------------------------------------------------------- wire form
    def to_dict(self) -> dict:
        return {r: {p: [i.version, i.density, i.attribution] for p, i in peers.items()}
                for r, peers in self._map.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "RegionDirectory":
        out = cls()
        for region, peers in d.items():
            out._map[region] = {p: ProviderInfo(int(v[0]), int(v[1]), float(v[2]))
                                for p, v in peers.items()}
        return out


# Something that returns (manifest, items) for a (peer_id, region_id), or (None, None).
GetShard = Callable[[str, str], Tuple[Optional[ShardManifest], Optional[Dict[str, bytes]]]]


def pull_regions(directory: RegionDirectory, regions: Sequence[str], into: MeshNode,
                 authority_pub, get_shard: GetShard) -> Dict[str, Optional[str]]:
    """Fetch each routed region from its **best available** provider, with failover.

    Tries providers best-first; verifies every shard (``import_shard`` raises on
    tamper/forgery) — a bad or unreachable provider is skipped and the next is tried, so
    the pull is **self-healing**. Returns ``region -> peer_id that satisfied it`` (``None``
    if no provider served a valid copy). This is anchor routing + directory + blind verify
    composed: pull only the query's working set, from wherever the good copy lives.
    """
    satisfied: Dict[str, Optional[str]] = {}
    for region, peers in directory.plan(regions).items():
        satisfied[region] = None
        for peer_id in peers:
            try:
                manifest, items = get_shard(peer_id, region)
                if manifest is None or items is None:
                    continue
                if manifest.version <= into.version_of(region):
                    satisfied[region] = peer_id  # already have this good/newer
                    break
                into.import_shard(manifest, items, authority_pub)  # raises on tamper
                satisfied[region] = peer_id
                break
            except (ShardVerifyError, OSError, ValueError):
                continue  # bad/unreachable provider — fail over to the next
    return satisfied


__all__ = ["ProviderInfo", "RegionDirectory", "attribution_of", "pull_regions"]
