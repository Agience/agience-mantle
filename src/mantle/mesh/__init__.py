"""The mesh — Ember's, because Ember is its only real consumer.

Content-addressed, Ed25519-signed shards; anchor-keyed regions; gossip freshness; verified,
selective sync. This is how a leaf holds its own shards and coordinates its freshness with
other nodes — which is the whole of what Ember is.

It briefly lived in Mantle and then in a package of its own. Both were wrong: Mantle's server
never imported it (only prototypes did), so there was exactly one consumer and it is this one.
The genuinely shared part — the anchor geometry — is in `mantle.search.anchors`, and the belief model
is in `prism.mass`; both really do have two consumers.
"""
from .anchor_routing import (
    cell_region, parse_cell_key, route_query_regions, route_write_region,
)
from .directory import ProviderInfo, RegionDirectory, coherence_of, pull_regions
from .manifest import (
    ShardItem, ShardManifest, build_manifest, content_root, item_hash, sign_manifest,
    verify_manifest,
)
from .node import MeshNode, ShardVerifyError

__all__ = [
    "ShardManifest", "ShardItem", "build_manifest", "content_root", "item_hash",
    "sign_manifest", "verify_manifest", "MeshNode", "ShardVerifyError",
    "cell_region", "parse_cell_key", "route_query_regions", "route_write_region",
    "RegionDirectory", "ProviderInfo", "coherence_of", "pull_regions",
]
