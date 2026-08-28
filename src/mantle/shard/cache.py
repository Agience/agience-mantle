"""LocalCache — Ember's shards, and the hit/miss decision.

Everything is an artifact. `collection_id` is a collection *artifact's* id (a collection is an
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from mantle.search.anchors.anchorset import AnchorSet
from prism.attestation import SELF_ORIGIN, AgreementRead, Ledger

#: The READ-TIME seam. Given a root and the revisions standing under it — each with the agreement
#: read over it — return the ids that ANSWER a query. mantle may not import the aperture (only
#: `ember/optics.py` reaches it), so the measurement arrives as an injection rather than a
#: dependency; a reader that has the aperture supplies one, a reader that does not gets every
#: revision and decides for itself.
Resolver = Callable[[str, List["Item"], Dict[str, Optional[AgreementRead]], Dict[str, Any]],
                    Sequence[str]]


def _unit(v: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector, zero-safe (a zero vector has no direction, so it stays zero).
    Inlined rather than `prism.vector.unit` — this is `search`'s one caller, scoring purely
    local, ephemeral top-k results that nothing else needs to reproduce."""
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n


@dataclass(frozen=True)
class Landed:
    """What `revise` returns: the id of the new version, its root, and how many versions now
    stand under that root. There is no verdict here on which version a reader will see — that is
    resolved at read time, not at write time."""
    id: str
    root_id: str
    versions: int
from prism.mass import Provenance
from mantle.mesh.node import MeshNode
from mantle.mesh.directory import RegionDirectory
from mantle.mesh.anchor_routing import route_query_regions, route_write_region


@dataclass
class Item:
    """One immutable version of an artifact.

    Artifacts are not edited in place — a change is a new version under a stable ``root_id``
    (Mantle's model: first version ``id == root_id``; edges point at the root). ``id`` is this
    version; ``root_id`` is the identity that survives across versions. ``content`` is opaque
    bytes.

    There is no `mass` field. Mass is a read over the attestations the node has verified
    (`LocalCache.agreement`), so it moves as peers attest rather than being a value stored on
    the item and asserted by whoever happens to hold the bytes.
    """
    id: str
    content: bytes
    region: str
    root_id: str = ""

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
                 node_id: str = "ember", nprobe: int = 8, secret=None,
                 resolve: Optional[Resolver] = None) -> None:
        """`resolve` is the READ-TIME head seam — see `_answering`. Optional, and its absence means
        every revision answers rather than a default being picked on the reader's behalf."""
        self.anchors = anchorset
        self.principal = principal
        self.collection_id = collection_id
        self.nprobe = nprobe
        # Per-principal blinding secret for region ids (Phase 0.2). None => legacy cleartext ids,
        # so an unprovisioned node still works; a provisioned one leaks nothing. Write and query
        # both carry it, so they agree on which shard a cell is.
        self.secret = secret
        self.node = MeshNode(node_id)
        # (see `_resolve` below — the read-time seam)
        self.directory = RegionDirectory()
        # region -> {item_id: Item}. Mirrors the node's shards with the decoded view Ember
        # uses for retrieval; the node holds the authoritative signed bytes.
        self._index: Dict[str, Dict[str, Item]] = {}
        self._vectors: Dict[str, np.ndarray] = {}   # item_id -> embedding
        # There is no stored head. Every revision commits and stands; which one answers a query
        # is decided when the query is made, by the resolver the reader supplies — head is
        # observer-relative, not a write-time verdict. mantle may not import the aperture
        # (one-aperture rule: only `ember/optics.py` reaches the aperture), so the measurement is
        # injected — a declared seam, exactly like the transport in `aria/identity.py`.
        #
        # With no resolver, every revision answers. That is the honest default: a store that
        # cannot measure which version is right must not pick one. Silently keeping the
        # most-attested, or the newest, would smuggle a decision back in under a different name.
        self._resolve: Optional[Resolver] = resolve
        # Counts published for observability: a resolver mechanism nobody can see fire is one
        # nobody can see fail. [[never-handroll-probes]]: a missing stat means add it, not probe
        # around it.
        self._resolution = {"roots_read": 0, "multi_revision": 0,
                            "grounded_out": 0, "narrowed": 0}
        # Every attestation this node has VERIFIED — its own, plus every peer manifest it imported.
        # Agreement (and therefore mass) is read out of here; it is never stored on an Item.
        self.ledger = Ledger()

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

        Originating is exactly what it attests. Content authored here carries
        `origin=SELF_ORIGIN`, which the ledger resolves to this authority at the boundary — a
        first-hand observation, and the only kind this node is entitled to publish. `channel` is
        the provenance label: how this node obtained the content, which is a fact about
        acquisition rather than a rank on a ladder.
        """
        rung = provenance or Provenance.UNKNOWN
        # What travels is what this node actually saw; agreement is computed from the set of
        # those observations rather than carried as a weight on the item itself.
        attest = {"origin": SELF_ORIGIN, "channel": rung.value}
        grouped: Dict[str, Dict[str, bytes]] = {}
        for item_id, content, vec in items:
            region = self.region_for(vec)
            grouped.setdefault(region, {})[item_id] = content
            # A fresh id is a NEW artifact: its first version, so id == root_id. Nothing is
            # marked head — there is no head to mark.
            self._index.setdefault(region, {})[item_id] = Item(item_id, content, region)
            self._vectors[item_id] = np.asarray(vec, dtype=np.float32).ravel()
        for region, payload in grouped.items():
            whole = self._region_payload(region, adding=payload)     # never drop the region's rest
            manifest = self.node.put_shard(region, whole, version=version, authority=authority,
                                           priv=priv,
                                           attest={item_id: attest for item_id in whole})
            self.ledger.observe(authority, manifest.items)
        return sorted(grouped)

    def revise(self, root_id: str, new_content: bytes, vec: Sequence[float], *,
               provenance: "Provenance", authority: str, priv, version: int = 2):
        """Author a new version of an existing artifact — never an in-place edit.

        Immutability, made real: the prior version is retained; a new version is committed under
        the same ``root_id``, and that is the whole of it — writing decides nothing about which
        version answers a query. Returns `Landed`: the id, its root, and how many versions now
        stand there.

        Head is not set at write time by an attestation count. Attestation count measures
        agreement — how many independent origins have observed a version — which is a reading of
        existence, not of validity; a correction starts life attested by exactly one origin, so a
        headcount rule would favor the standing version over the correction in the case that
        matters most. `prism.resolution.separated([incoming, current])` also declines to settle
        it from two counts alone: at n=2 the computed null is 1.0000, so no pair of counts is
        ever reported separated.

        Head is instead resolved at read time by the reader's own measurement (`_answering`), and
        every revision stands until then. Nothing is destroyed on a revision, and nothing is
        hidden either — a non-head version remains a version anyone can read.

        The new version routes by its own vector, so an edit that shifts meaning can land in a
        different cell than its predecessor — which is fine: identity is the root_id, not the cell.
        """
        from mantle.mesh.manifest import item_hash

        if not self._versions_of(root_id):
            raise KeyError(f"revise: unknown root_id {root_id!r} — nothing to revise")
        new_id = f"{root_id}~{item_hash(new_content)[:16]}"     # a distinct, content-derived version id
        rung = provenance or Provenance.UNKNOWN

        region = self.region_for(vec)
        self._index.setdefault(region, {})[new_id] = Item(new_id, new_content, region,
                                                          root_id=root_id)
        self._vectors[new_id] = np.asarray(vec, dtype=np.float32).ravel()
        manifest = self.node.put_shard(
            region, self._region_payload(region, adding={new_id: new_content}),
            version=version, authority=authority, priv=priv,
            attest={new_id: {"origin": SELF_ORIGIN, "channel": rung.value}})
        self.ledger.observe(authority, manifest.items)

        # No write-time decision: the revision commits and stands, alongside every other version
        # of this root, and which one answers is resolved when a query is made.
        return Landed(id=new_id, root_id=root_id, versions=len(self._versions_of(root_id)))

    def _versions_of(self, root_id: str) -> List["Item"]:
        """Every revision standing under this root. Derived from the items themselves — there is no
        separate lineage table to fall out of step with them."""
        out = []
        for region in self._index.values():
            out.extend(i for i in region.values() if i.root_id == root_id)
        return out

    def _answering(self, items: Iterable["Item"], frame: Optional[Dict[str, Any]] = None) -> set:
        """Head, resolved at read time. Which revisions answer this query?

        With no resolver injected: all of them. A store that cannot measure which version is
        right must not choose one — the two tempting defaults, most-attested and newest, are a
        headcount and a clock respectively, neither of which is a measure of validity.

        With a resolver: the reader's own measurement decides, per root, given the revisions and
        the agreement read over each. A resolver that returns nothing for a root is respected — a
        read that grounds out is a legitimate outcome, not a rejection
        [[one-resolution-not-thresholds]] — and a resolver that returns an id it was not offered
        is ignored rather than trusted.
        """
        items = list(items)
        if self._resolve is None:
            return {i.id for i in items}
        # The frame is the reader's own recall set: not the region and not the whole held set,
        # but the neighbourhood a query actually surfaced. That is what makes head genuinely
        # observer-relative — the same root may answer with different revisions to different
        # questions, because the frame it is read against differs. Vectors ride along because the
        # measurement is geometric and `Item` does not carry one.
        if frame is None:
            frame = {i.id: self._vectors[i.id] for i in items if i.id in self._vectors}
        by_root: Dict[str, List["Item"]] = {}
        for i in items:
            by_root.setdefault(i.root_id, []).append(i)
        answering = set()
        for root, revisions in by_root.items():
            offered = {i.id for i in revisions}
            reads = {i.id: self.agreement(i.id) for i in revisions}
            picked = set(self._resolve(root, revisions, reads, frame) or ()) & offered
            self._resolution["roots_read"] += 1
            if len(offered) > 1:
                # Only a multi-revision root is a real test: a root with one version cannot be
                # narrowed, so counting it as "grounded out" would report the mechanism working
                # when it was never asked anything.
                self._resolution["multi_revision"] += 1
                if picked == offered:
                    self._resolution["grounded_out"] += 1
                elif picked:
                    self._resolution["narrowed"] += 1
            answering |= picked
        return answering

    def _region_payload(self, region: str, *, adding: Optional[Dict[str, bytes]] = None) -> Dict[str, bytes]:
        """The complete contents of a region, plus whatever is being added.

        `MeshNode.put_shard` does `self._shards[region_id] = {"manifest": …, "items": dict(items)}`
        — it replaces the region wholesale, so a caller must supply the region's full contents on
        every write or persistence silently drops everything else already in that region.

        The merge belongs here rather than in `put_shard`: the manifest is signed over exactly
        the items it is given, so a node that merged internally would sign a set its caller never
        named. Making the full payload explicit keeps the signature meaning what it says.
        """
        payload = {i.id: i.content for i in (self._index.get(region) or {}).values()}
        payload.update(adding or {})
        return payload

    def agreement(self, item_id: str) -> Optional[AgreementRead]:
        """**Mass, as a read.** How many independent origins attest this item, and whether they
        agree on which bytes it is. `None` when nothing attests it — the ghost, derived.

        Deliberately not a stored weight on the item: a stored weight is a claim by whoever holds
        the bytes, whereas this is a measurement over what has actually been verified, so it moves
        when peers attest and cannot be asserted by a single node."""
        return self.ledger.read(item_id)

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
        # No `heads` key: a snapshot carrying a head would decide for its reader, which is exactly
        # what read-time resolution avoids. An older snapshot's `heads` is read back in `hydrate`
        # and ignored.
        return {"items": items}

    def hydrate(self, views: dict) -> None:
        """Rebuild the readable view + head map from the loaded (already verified) node + the views
        sidecar. Content and attestations come from the shards; root_id and vector from the sidecar.

        Fail-soft on a missing/partial sidecar: an item with no sidecar entry is its own root and
        its own head (a loose but honest reconstruction), and stays blind (no vector) if none was
        stored. So an older cache, or a corrupt sidecar, still loads — just without version
        lineage — rather than refusing to boot.

        Every held manifest is folded into the ledger as an attestation by ITS OWN signing
        authority, not by this node. A shard we merely hold is a shard someone else stood behind,
        and recording it under our own id would manufacture an origin.
        """
        item_meta = views.get("items") or {}
        for region in self.node.summary()["regions"]:
            manifest, items = self.node.get_shard(region)
            if manifest is None:
                continue
            self.ledger.observe(manifest.authority, manifest.items)
            for item_id, content in items.items():
                meta = item_meta.get(item_id) or {}
                root = meta.get("root") or item_id
                self._index.setdefault(region, {})[item_id] = Item(
                    item_id, content, region, root_id=root)
                vec = meta.get("vec")
                if vec is not None:
                    self._vectors[item_id] = np.asarray(vec, dtype=np.float32).ravel()
        # An older sidecar may carry `heads`; it is read and discarded on purpose, since honouring
        # it would silently reinstate a write-time head decision on a store that reads as
        # upgraded. The lineage (`root`) is what survives.
        _stale_heads = views.get("heads") or {}
        if _stale_heads:
            self._discarded_heads = len(_stale_heads)      # published, so the drop is observable

    def adopt(self, region: str, items: Dict[str, bytes],
              vectors: Optional[Dict[str, Sequence[float]]] = None,
              manifest: Optional[object] = None) -> None:
        """Record the readable view of a shard we already imported+verified via the node.

        Split from `import_shard` on purpose: verification is the node's job and happens on
        opaque bytes; this only builds the retrieval view, and only for items we can actually
        read. A blind shard can be held and served with no vectors at all.

        Mass is not a field that can be dropped or defaulted on a refill: it is a read over the
        attestations, and a refilled item's attestations arrive with its manifest rather than
        being reconstructed here. Likewise `root_id` is preserved for an item already known,
        rather than reset to the item's own id, so a superseded version stays superseded.

        `manifest` is optional so existing callers keep working; when supplied, the attestations and
        lineage are preserved exactly as `hydrate` does it — recorded under the manifest's OWN
        signing authority, since a shard we adopted is one someone else stood behind. An item whose
        lineage we already know is not demoted to its own root."""
        authority = getattr(manifest, "authority", None)
        if authority:
            self.ledger.observe(str(authority), getattr(manifest, "items", None) or [])
        for item_id, content in items.items():
            prior = (self._index.get(region) or {}).get(item_id)
            root = prior.root_id if prior is not None else item_id
            self._index.setdefault(region, {})[item_id] = Item(
                item_id, content, region, root_id=root)
        for item_id, vec in (vectors or {}).items():
            self._vectors[item_id] = np.asarray(vec, dtype=np.float32).ravel()

    # ---------------------------------------------------------------- retrieval (local only)
    def search(self, query_vec: Sequence[float] | np.ndarray, k: int = 5) -> List[tuple[Item, float]]:
        """Top-k readable items from the routed cells we hold. Pure local geometry."""
        r = self.route(query_vec)
        q = _unit(np.asarray(query_vec, dtype=np.float32).ravel())
        scored: List[tuple[Item, float]] = []
        for region in r.held:
            # Resolve the pool, then score, in that order: the pool is the frame the reader is
            # looking at, so it has to be assembled before anything is filtered out of it.
            # Resolving per region against a pre-filtered set would read each candidate against a
            # neighbourhood the query never saw.
            pool = list(self._index.get(region, {}).values())
            frame = {i.id: self._vectors[i.id] for i in pool if i.id in self._vectors}
            answering = self._answering(pool, frame)
            for item in pool:
                if item.id not in answering:
                    continue           # the reader's resolver did not return this revision
                v = self._vectors.get(item.id)
                if v is None:          # held blind — present, but not readable/searchable here
                    continue
                scored.append((item, float(np.dot(q, _unit(v)))))   # cosine: unit(q)·unit(v)
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    # ---------------------------------------------------------------- introspection
    def summary(self) -> dict:
        return {
            "node": self.node.node_id,
            "regions": len(self.node.summary()["regions"]),
            "items": sum(len(v) for v in self._index.values()),
            "readable": len(self._vectors),
            # What the READ-TIME head resolution actually did. `multi_revision` is the only count
            # that means anything — a single-version root cannot be narrowed — and `narrowed` is the
            # number of times a measurement changed what a query saw.
            "resolution": dict(self._resolution, resolver="wired" if self._resolve else "none"),
        }
