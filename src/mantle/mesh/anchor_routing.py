"""Anchor-region keying — make the mesh partition == the search partition.

MANTLE-MESH.md §1: a shard's region should be the **anchor cell** an item indexes
into, not a coarse ``context`` / ``(principal, collection)``. The cell store already
lays cells out by their routing anchor::

    {prefix}/{principal}/{collection}/{cluster}.cell     # cluster == the anchor id

so a shard *is* a cell. Keying the mesh this way means a query can be routed with the
very same geometry the index uses (:func:`search.anchors.routing.route_query`): the
query's ``nprobe`` nearest anchors name exactly the shards to pull — no more, no less.
A node then syncs its working set of cells instead of the whole corpus, and the blind
ciphertext of a cell is verified against its ``content_root`` like any other shard.

The routing here is pure geometry (the §1 invariant): it never decrypts a cell and
never touches keys / light-cone / oracle. It maps a query vector to region *ids*; the
mesh moves the opaque bytes those ids name.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mantle.search.anchors.anchorset import AnchorSet
from mantle.search.anchors.routing import route_query, route_vector
from .node import MeshNode

CELL_SUFFIX = ".cell"


# --------------------------------------------------------------------------- keys
def cell_region(principal: str, collection: str, cluster: str, *, secret=None) -> str:
    """The canonical mesh region id for one anchor cell.

    ``cluster`` is the routing anchor id (``route_vector``). Same triple → same
    region id everywhere, so an authored cell and a routed query name the same shard.
    Without a secret this returns the plain, unblinded id; `region.blind` is what
    reports `blinded=False` when blinding was requested but no secret exists, so
    degradation is never silent — this wrapper only returns the id."""
    if secret:
        from mantle.shard import region
        rid, _blinded = region.cell_region(principal, collection, cluster, secret=secret)
        return rid
    return f"{principal}/{collection}/{cluster}"


def parse_cell_key(key: str, prefix: str = "mantle-cells") -> Optional[Tuple[str, str, str]]:
    """``{prefix}/{principal}/{collection}/{cluster}.cell`` → ``(principal, collection,
    cluster)``; ``None`` for anything that isn't a cell key under ``prefix``.

    Defensive by design: a stray object under the prefix is skipped, not fatal.
    """
    p = prefix.strip("/")
    k = key
    if p:
        if not k.startswith(p + "/"):
            return None
        k = k[len(p) + 1:]
    if not k.endswith(CELL_SUFFIX):
        return None
    parts = k[: -len(CELL_SUFFIX)].split("/")
    if len(parts) != 3 or not all(parts):
        return None
    principal, collection, cluster = parts
    return principal, collection, cluster


# ------------------------------------------------------------------------ routing
def route_query_regions(anchorset: AnchorSet, query_vec: Sequence[float] | np.ndarray,
                        principal: str, collection: str, *, nprobe: int = 8, secret=None,
                        legacy_read: bool = True) -> List[str]:
    """The mesh region ids a query must pull, nearest anchor first.

    This is the whole point of anchor keying: ``route_query`` gives the ``nprobe``
    nearest anchor ids (the cells a match could live in); we name the corresponding
    shards. Pass the result to :meth:`MeshNode.sync_from`'s ``regions`` filter and a
    node fetches only those cells — the working set for this query — verified + blind.

    ``secret`` blinds the region ids (see :func:`cell_region`); it MUST be the same secret the
    write path used, or a query names a shard the write never created. ``legacy_read`` also
    includes each anchor's unblinded id, so a query still finds cells written before blinding
    was turned on; set it False once every write is blinded, to close the reader too. With no
    secret this is a no-op — there is only the legacy id."""
    anchor_ids = route_query(anchorset, query_vec, nprobe=nprobe)
    out: List[str] = []
    seen = set()
    for a in anchor_ids:
        for rid in (cell_region(principal, collection, a, secret=secret),
                    (cell_region(principal, collection, a) if (secret and legacy_read) else None)):
            if rid is not None and rid not in seen:
                seen.add(rid)
                out.append(rid)
    return out


def route_write_region(anchorset: AnchorSet, item_vec: Sequence[float] | np.ndarray,
                       principal: str, collection: str, *, secret=None) -> str:
    """The single region id an item indexes into — its nearest anchor's cell.

    The write-side mirror of :func:`route_query_regions`: an authoring node uses this
    to decide which shard a new item belongs to, so writes and queries agree on cells.
    ``secret`` blinds the id and MUST match the query side's.
    """
    return cell_region(principal, collection, route_vector(anchorset, item_vec), secret=secret)


# ------------------------------------------------------------- pack cells as shards
def build_node_from_cells(node_id: str, s3, bucket: str, prefix: str,
                          authority_priv, *, authority_id: str = "cell-authority",
                          version: int = 1, max_cells: Optional[int] = None) -> Tuple[MeshNode, int]:
    """Pack the encrypted cell blobs under ``prefix`` into signed shards, one shard
    **per anchor cell** (``{principal}/{collection}/{cluster}``).

    Blind: the ``.cell`` bytes are content-addressed as-is (ciphertext), never
    decrypted — a peer replicates and serves a cell without reading it. This is the
    faithful anchor-keyed layer: the encrypted IVF cell itself is the unit of the mesh.
    """
    regions: Dict[str, Dict[str, bytes]] = {}
    n = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parsed = parse_cell_key(key, prefix)
            if parsed is None:
                continue  # not a cell (manifest/stats/posting) — skip, blind
            principal, collection, cluster = parsed
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()  # ciphertext, opaque
            regions.setdefault(cell_region(principal, collection, cluster), {})[key] = body
            n += 1
            if max_cells and n >= max_cells:
                break
        if max_cells and n >= max_cells:
            break

    node = MeshNode(node_id)
    for region, items in regions.items():
        node.put_shard(region, items, version=version, authority=authority_id, priv=authority_priv)
    return node, n


__all__ = [
    "cell_region",
    "parse_cell_key",
    "route_query_regions",
    "route_write_region",
    "build_node_from_cells",
]
