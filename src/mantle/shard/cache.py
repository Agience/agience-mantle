"""LocalCache — Ember's shards, and the hit/miss decision.

EVERYTHING IS AN ARTIFACT. `collection_id` is a collection *artifact's* id (a collection is an
artifact with the right edges), and `cluster` is an anchor's id — anchors are artifacts too.
A region id is therefore a path made entirely of artifact identities, which is what lets two
nodes that never met agree on what a cell is called.

This is the whole of Ember's identity: a cache. What makes it a *trustworthy* cache rather
than a degraded copy is that every shard is content-addressed and authority-signed, so a
cached shard verifies independently of who served it. What makes it a *safe* cache is blind
replication: items are opaque bytes, so Ember can hold ciphertext it cannot read.

The interesting property is that **hit/miss is computable, not guessed**. A query embeds into
the same anchor geometry the index uses, so `route_query_regions()` names exactly the cells
that could answer it. If we hold those cells, we can answer locally — with certainty, not
optimism. That is also what makes eviction principled: keep the regions your queries route to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from mantle.search.anchors.anchorset import AnchorSet
from prism.mass import Provenance, consensus_of
from mantle.mesh.node import MeshNode
from mantle.mesh.directory import RegionDirectory
from mantle.mesh.anchor_routing import route_query_regions, route_write_region


@dataclass
class Item:
    """One immutable version of an artifact.

    Artifacts are not edited in place — a change is a NEW version under a stable ``root_id``
    (Mantle's model: first version ``id == root_id``; edges point at the root). ``id`` is this
    version; ``root_id`` is the identity that survives across versions; exactly one version per
    root is HEAD (the cache tracks which). ``content`` is opaque bytes; ``mass`` is the provenance
    weight, carried so revision inertia can compare authorities.
    """
    id: str
    content: bytes
    region: str
    root_id: str = ""
    mass: float = 0.0

    def __post_init__(self) -> None:
        if not self.root_id:
            self.root_id = self.id          # first version: id == root_id (Mantle's rule)


@dataclass
class Routing:
    """The result of asking 'can I answer this locally?' — the honest answer, with detail."""
    regions: List[str]                 # cells that could answer, nearest anchor first
    held: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    @property
    def hit(self) -> bool:
        """A hit means we hold the NEAREST cell — the one a match would index into. Holding
        only outer probe cells is not a hit: the best answer would live in the one we lack."""
        return bool(self.regions) and self.regions[0] in self.held

    @property
    def complete(self) -> bool:
        """We hold every routed cell — the full working set for this query."""
        return not self.missing


class LocalCache:
    """Ember's local shard cache over a MeshNode."""

    def __init__(self, anchorset: AnchorSet, principal: str, collection_id: str,
                 node_id: str = "ember", nprobe: int = 8, secret=None) -> None:
        self.anchors = anchorset
        self.principal = principal
        self.collection_id = collection_id
        self.nprobe = nprobe
        # Per-principal blinding secret for region ids (Phase 0.2). None => legacy cleartext ids,
        # so an unprovisioned node still works; a provisioned one leaks nothing. Write and query
        # both carry it, so they agree on which shard a cell is.
        self.secret = secret
        self.node = MeshNode(node_id)
        self.directory = RegionDirectory()
        # region -> {item_id: Item}. Mirrors the node's shards with the decoded view Ember
        # uses for retrieval; the node holds the authoritative signed bytes.
        self._index: Dict[str, Dict[str, Item]] = {}
        self._vectors: Dict[str, np.ndarray] = {}   # item_id -> embedding
        # root_id -> the item_id of its HEAD version. Only the head answers a query; superseded
        # versions and non-head proposals are retained (nothing is destroyed) but not surfaced.
        self._head: Dict[str, str] = {}

    # ---------------------------------------------------------------- routing (the core)
    def route(self, query_vec: Sequence[float] | np.ndarray) -> Routing:
        """Which cells answer this query, and do we hold them? No network, no guessing."""
        regions = route_query_regions(self.anchors, query_vec, self.principal,
                                      self.collection_id, nprobe=self.nprobe, secret=self.secret)
        held = [r for r in regions if self.node.has_region(r)]
        missing = [r for r in regions if not self.node.has_region(r)]
        return Routing(regions=regions, held=held, missing=missing)

    def region_for(self, vec: Sequence[float] | np.ndarray) -> str:
        """The cell an item indexes into — writes and queries agree by construction."""
        return route_write_region(self.anchors, vec, self.principal, self.collection_id,
                                  secret=self.secret)

    # ---------------------------------------------------------------- authoring (local truth)
    def put(self, items: Sequence[tuple[str, bytes, Sequence[float]]], *, version: int,
            authority: str, priv, provenance: Optional["Provenance"] = None,
            evidence: Optional[dict] = None) -> List[str]:
        """Author local content into signed shards, grouped by the cell each item routes to.

        This is why Ember is a leaf and not merely a mirror: your own content is a region this
        node **originates**. It is authored, signed and content-addressed exactly like anything
        from the mesh, so it can be served to peers on the same terms.
        """
        rung = provenance or Provenance.UNKNOWN
        weight = consensus_of(rung)
        grouped: Dict[str, Dict[str, bytes]] = {}
        for item_id, content, vec in items:
            region = self.region_for(vec)
            grouped.setdefault(region, {})[item_id] = content
            # A fresh id is a NEW artifact: its first version, so id == root_id and it is head.
            self._index.setdefault(region, {})[item_id] = Item(item_id, content, region, mass=weight)
            self._vectors[item_id] = np.asarray(vec, dtype=np.float32).ravel()
            self._head.setdefault(item_id, item_id)
        # MASS travels with the shard. `ShardItem.consensus` is the provenance weight, so a
        # peer that pulls this region learns how much to believe it WITHOUT re-deriving anything
        # — the rung is not local knowledge, it is part of what the artifact is.
        # No provenance => UNKNOWN: weak, but above the ghost floor. Unlabeled is not fabricated.
        for region, payload in grouped.items():
            self.node.put_shard(region, payload, version=version, authority=authority, priv=priv,
                                consensus={item_id: weight for item_id in payload})
        return sorted(grouped)

    def revise(self, root_id: str, new_content: bytes, vec: Sequence[float], *,
               provenance: "Provenance", authority: str, priv, version: int = 2):
        """Author a NEW version of an existing artifact — never an in-place edit.

        Immutability, made real: the prior version is retained; a new version is committed under
        the same ``root_id``. Whether it becomes HEAD is decided by MASS, not by the fact of
        writing it — `mass.may_revise`: a version carrying >= the head's mass REPLACEs (becomes
        head, old one archived-but-kept); one carrying less PROPOSEs (committed as a non-head
        competing version). So a model's low-authority correction cannot silently overturn a
        human-validated head — it queues as a proposal. Returns the `Revision` decision.

        The new version routes by its OWN vector, so an edit that shifts meaning can land in a
        different cell than its predecessor — which is fine: identity is the root_id, not the cell.
        """
        from prism.mass import Revision, may_revise
        from mantle.mesh.manifest import item_hash

        if root_id not in self._head:
            raise KeyError(f"revise: unknown root_id {root_id!r} — nothing to revise")
        new_id = f"{root_id}~{item_hash(new_content)[:16]}"     # a distinct, content-derived version id
        rung = provenance or Provenance.UNKNOWN
        incoming = consensus_of(rung)

        head_id = self._head[root_id]
        head_item = self._find(head_id)
        decision = may_revise(head_item.mass if head_item else 0.0, incoming)

        region = self.region_for(vec)
        self._index.setdefault(region, {})[new_id] = Item(new_id, new_content, region,
                                                          root_id=root_id, mass=incoming)
        self._vectors[new_id] = np.asarray(vec, dtype=np.float32).ravel()
        self.node.put_shard(region, {new_id: new_content}, version=version,
                            authority=authority, priv=priv, consensus={new_id: incoming})
        if decision is Revision.REPLACE:
            self._head[root_id] = new_id                       # the new version takes head
        # PROPOSE: head unchanged; the new version is committed but not surfaced until promoted.
        return decision

    def _find(self, item_id: str) -> Optional[Item]:
        for region in self._index.values():
            if item_id in region:
                return region[item_id]
        return None

    # ---------------------------------------------------------------- persistence of derived state
    def export_views(self) -> dict:
        """The derived state a restart cannot rebuild from shards alone: the head map, and per
        item its root_id and (if readable) its retrieval vector. Paired with ShardStore.save_views."""
        items: Dict[str, dict] = {}
        for region in self._index.values():
            for item in region.values():
                vec = self._vectors.get(item.id)
                items[item.id] = {"root": item.root_id,
                                  "vec": vec.tolist() if vec is not None else None}
        return {"heads": dict(self._head), "items": items}

    def hydrate(self, views: dict) -> None:
        """Rebuild the readable view + head map from the loaded (already verified) node + the views
        sidecar. Content and mass come from the shards; root_id and vector from the sidecar.

        Fail-soft on a missing/partial sidecar: an item with no sidecar entry is its own root and
        its own head (a loose but honest reconstruction), and stays blind (no vector) if none was
        stored. So an older cache, or a corrupt sidecar, still loads — just without version
        lineage — rather than refusing to boot.
        """
        item_meta = views.get("items") or {}
        for region in self.node.summary()["regions"]:
            manifest, items = self.node.get_shard(region)
            if manifest is None:
                continue
            mass_of = {si["id"]: float(si.get("consensus", 0.0)) for si in manifest.items}
            for item_id, content in items.items():
                meta = item_meta.get(item_id) or {}
                root = meta.get("root") or item_id
                self._index.setdefault(region, {})[item_id] = Item(
                    item_id, content, region, root_id=root, mass=mass_of.get(item_id, 0.0))
                vec = meta.get("vec")
                if vec is not None:
                    self._vectors[item_id] = np.asarray(vec, dtype=np.float32).ravel()
                self._head.setdefault(root, item_id)   # provisional; overwritten by saved heads below
        for root, head in (views.get("heads") or {}).items():
            self._head[root] = head

    def adopt(self, region: str, items: Dict[str, bytes],
              vectors: Optional[Dict[str, Sequence[float]]] = None,
              manifest: Optional[object] = None) -> None:
        """Record the readable view of a shard we already imported+verified via the node.

        Split from `import_shard` on purpose: verification is the node's job and happens on
        opaque bytes; this only builds the retrieval view, and only for items we can actually
        read. A blind shard can be held and served with no vectors at all.

        ⛔ MASS TRAVELS WITH THE SHARD AND THIS USED TO THROW IT AWAY.
        `Item(item_id, content, region)` takes `mass=0.0` and `root_id=id` by default, and the
        caller (`boot._adopt_regions`) fetched the manifest, `None`-checked it, and discarded it —
        while `hydrate` reads the very same field (`ShardItem.consensus`) two methods below.
        Consequences, on the LIVE refill path (`ask()` -> `as_refill(on_import=_adopt_regions)`):

          * **A refilled `human_validated` item landed at mass 0.0.** `revise` then computes
            `may_revise(head_item.mass, incoming)`, and `may_revise` REPLACEs on
            `incoming >= current` — so *any* local revision, including one at `UNKNOWN`, silently
            took head from a validated artifact. That is precisely the guarantee `revise`'s own
            docstring asserts cannot be violated.
          * **Re-adopting an already-hydrated region reset every `root_id` to the item's own id**,
            so versions that had been superseded passed `search`'s head filter again and retired
            proposals became answerable.

        `manifest` is optional so existing callers keep working; when supplied, mass and lineage
        are preserved exactly as `hydrate` does it. `setdefault` on `_head` is kept — a genuinely
        new item is still head of its own root — but an item we already know the lineage of is no
        longer demoted to its own root."""
        mass_of = {}
        for si in (getattr(manifest, "items", None) or []):
            try:
                mass_of[si["id"]] = float(si.get("consensus", 0.0))
            except (TypeError, KeyError, ValueError):
                continue
        for item_id, content in items.items():
            prior = (self._index.get(region) or {}).get(item_id)
            root = prior.root_id if prior is not None else item_id
            self._index.setdefault(region, {})[item_id] = Item(
                item_id, content, region, root_id=root,
                mass=mass_of.get(item_id, prior.mass if prior is not None else 0.0))
            self._head.setdefault(item_id, item_id)   # a refilled item is head of its own root
        for item_id, vec in (vectors or {}).items():
            self._vectors[item_id] = np.asarray(vec, dtype=np.float32).ravel()

    # ---------------------------------------------------------------- retrieval (local only)
    def search(self, query_vec: Sequence[float] | np.ndarray, k: int = 5) -> List[tuple[Item, float]]:
        """Top-k readable items from the routed cells we hold. Pure local geometry."""
        from prism import vector as _vec
        r = self.route(query_vec)
        q = _vec.unit(np.asarray(query_vec, dtype=np.float32).ravel())
        scored: List[tuple[Item, float]] = []
        for region in r.held:
            for item in self._index.get(region, {}).values():
                if self._head.get(item.root_id, item.id) != item.id:
                    continue           # only the HEAD version answers; superseded/proposals don't
                v = self._vectors.get(item.id)
                if v is None:          # held blind — present, but not readable/searchable here
                    continue
                scored.append((item, float(np.dot(q, _vec.unit(v)))))   # cosine: unit(q)·unit(v)
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    # ---------------------------------------------------------------- introspection
    def summary(self) -> dict:
        return {
            "node": self.node.node_id,
            "regions": len(self.node.summary()["regions"]),
            "items": sum(len(v) for v in self._index.values()),
            "readable": len(self._vectors),
        }
