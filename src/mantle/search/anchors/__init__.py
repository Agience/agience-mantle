"""Anchors — the coordinate system, loaded and routed against.

The AnchorSet is the shared coordinate system / routing centroids / grounding
(see `.dev/features/mantle-canonical-architecture.md` §3–§4). A client seeds it;
this layer loads it, compares vectors against it, and names the cell each one
lands in.

**Mantle does not derive, grow, reconcile or crosswalk a coordinate system, and
that is deliberate.** An anchor id is content-addressed over
`(label, model_id, embedding)`, so a set fitted here would mint region ids no peer
computes — an index that looks healthy and shares with nobody. The whole surface is
therefore: seed a set (`python -m mantle.system.manage_anchors --action load`),
supply a query vector in that set's space, get ranked results.

INVARIANT (§1): this layer operates on plaintext vectors only. It never touches
cell keys, the light-cone, the oracle, or the ledger, and runs strictly before
partition/encryption (index) and routing (query). It cannot affect authorization.
"""

from .anchorset import (
    Anchor, AnchorSet, AnchorSetCorrupt, CANDIDATE, WORKING, CANONICAL,
    anchor_content_hash, anchor_id_for, anchorset_fingerprint, verify_anchor_id,
)
from .routing import (
    DEFAULT_NPROBE, QueryRoute, RouteDecision, route_query, route_query_scored,
    route_vector, route_vector_scored,
)
from .repo import AnchorRepo, StoreAnchorRepo, InMemoryAnchorRepo
from .store import (
    AnchorSetDiverged,
    AnchorSetNotProvisioned,
    get_anchor_repo,
    get_live_anchorset,
    indexed_geometry,
    live_fingerprint,
    record_indexed_geometry,
    require_live_anchorset,
    reset_anchorset,
    save_live_anchorset,
    set_anchor_repo,
)

__all__ = [
    "Anchor", "AnchorSet",
    "CANDIDATE", "WORKING", "CANONICAL",
    "DEFAULT_NPROBE",
    "route_query", "route_vector", "route_query_scored", "route_vector_scored",
    "RouteDecision", "QueryRoute",
    "AnchorSetNotProvisioned", "AnchorSetCorrupt", "AnchorSetDiverged",
    "anchor_content_hash", "anchor_id_for", "anchorset_fingerprint", "verify_anchor_id",
    "AnchorRepo", "StoreAnchorRepo", "InMemoryAnchorRepo",
    "get_anchor_repo", "set_anchor_repo",
    "get_live_anchorset", "require_live_anchorset",
    "save_live_anchorset", "reset_anchorset",
    "live_fingerprint", "indexed_geometry", "record_indexed_geometry",
]
