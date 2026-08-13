"""Bridge Mantle's real artifacts into mesh shards (content-addressed, signed, verified).

Reads a node's visible artifacts over the Mantle API and groups them into shards by
``context`` — a stand-in for the density-equalized anchor-region partition until the
anchor router is wired in (MANTLE-MESH.md §1). The result is a ``MeshNode`` whose shards
carry **real node content**, so a peer can sync + verify it server-independently.

The mesh content-addresses whatever bytes it's given, so this works today over the
authorized-reader plaintext; wiring the encrypted IVF cell blobs (true blind ciphertext
replication) is the next step and only changes what bytes populate each item.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple
from urllib.request import Request, urlopen

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from prism.attestation import SELF_ORIGIN
from prism.mass import provenance_of

from .node import MeshNode


def fetch_visible(mantle_url: str, token: str) -> List[dict]:
    """GET /artifacts/visible with the operator's bearer token (light-cone filtered)."""
    req = Request(mantle_url.rstrip("/") + "/artifacts/visible",
                  headers={"authorization": "Bearer " + token})
    with urlopen(req, timeout=20) as r:
        return json.load(r)


def build_node_from_mantle(node_id: str, mantle_url: str, token: str,
                           authority_priv: Ed25519PrivateKey, *,
                           authority_id: str = "mantle-authority",
                           version: int = 1) -> Tuple[MeshNode, List[dict]]:
    """Fetch a node's visible artifacts and pack them into signed shards by region.

    Region = ``context``. A real anchor-region id replaces it once routing is wired.
    Each item's bytes = the artifact content; this authority's ATTESTATION of each item travels in
    the manifest, so a peer can compute agreement rather than inherit a number.

    What travels is what this bridge actually observed: it fetched these artifacts from this
    Mantle, so this authority ORIGINATES them into the mesh, and the provenance label rides as the
    CHANNEL — how they were obtained, not how much to believe them. A number every node would
    derive identically from the same table (a weight rather than an observation) tells a peer
    nothing it could not have computed itself, so none travels here.
    """
    artifacts = fetch_visible(mantle_url, token)
    regions: Dict[str, Dict[str, bytes]] = {}
    attest: Dict[str, Dict[str, dict]] = {}
    for a in artifacts:
        region = a.get("context") or "default"
        content = (a.get("content") or "").encode("utf-8")
        regions.setdefault(region, {})[a["id"]] = content
        attest.setdefault(region, {})[a["id"]] = {
            "origin": SELF_ORIGIN,
            "channel": provenance_of(a).value,
        }

    node = MeshNode(node_id)
    for region, items in regions.items():
        node.put_shard(region, items, version=version, authority=authority_id,
                       priv=authority_priv, attest=attest.get(region))
    return node, artifacts
