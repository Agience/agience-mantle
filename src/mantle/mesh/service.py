"""HTTP mesh node — serve/sync shards over the wire (stdlib only, no extra deps).

Deployment simplicity: one command stands up a node.

    # authoritative seed node (generates an authority key, seeds a demo region):
    python -m mesh.service --port 9701 --seed

    # a second node that syncs from it (paste the authority pubkey it printed):
    python -m mesh.service --port 9702 --sync http://localhost:9701 --authority <hex>

Endpoints:  GET /mesh/manifests  ·  GET /mesh/shard/{region}  ·  GET /mesh/health
Content is served/synced **blind** (opaque encrypted bytes); the peer verifies every
item hash + the content_root + the authority signature before trusting it.
"""
from __future__ import annotations

import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List
from urllib.request import urlopen

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .directory import RegionDirectory, pull_regions
from .manifest import ShardManifest
from .node import MeshNode


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def serve(node: MeshNode, port: int, host: str = "0.0.0.0",
          directory: "RegionDirectory | None" = None) -> ThreadingHTTPServer:
    """Start the mesh HTTP server in a background thread; return the server handle.

    ``directory`` (optional) is the node's gossiped view of the wider mesh, served at
    ``/mesh/directory`` so a fetching peer can discover cells held by nodes it hasn't
    talked to directly (transitive discovery — the stdlib stand-in for a DHT)."""
    class Handler(BaseHTTPRequestHandler):
        def _json(self, obj, code: int = 200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/mesh/manifests":
                self._json([m.to_dict() for m in node.manifests()])
            elif self.path.startswith("/mesh/shard/"):
                region = self.path[len("/mesh/shard/"):]
                manifest, items = node.get_shard(region)
                if manifest is None:
                    self._json({"error": "no such region"}, 404)
                    return
                self._json({"manifest": manifest.to_dict(),
                            "items": {k: _b64(v) for k, v in items.items()}})
            elif self.path == "/mesh/directory":
                self._json(directory.to_dict() if directory is not None else {})
            elif self.path == "/mesh/health":
                self._json(node.summary())
            else:
                self._json({"error": "not found"}, 404)

        def log_message(self, *args):  # quiet
            pass

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def fetch_manifests(base_url: str, timeout: int = 15) -> List[ShardManifest]:
    """GET a peer's advertised manifest map."""
    with urlopen(base_url.rstrip("/") + "/mesh/manifests", timeout=timeout) as r:
        return [ShardManifest.from_dict(d) for d in json.load(r)]


def fetch_shard(base_url: str, region_id: str, timeout: int = 30):
    """GET one region's (manifest, blind items) from a peer, or (None, None) on 404."""
    with urlopen(f"{base_url.rstrip('/')}/mesh/shard/{region_id}", timeout=timeout) as r:
        payload = json.load(r)
    if "manifest" not in payload:
        return None, None
    manifest = ShardManifest.from_dict(payload["manifest"])
    items = {k: _unb64(v) for k, v in payload["items"].items()}
    return manifest, items


def fetch_directory(base_url: str, timeout: int = 15) -> RegionDirectory:
    """GET a peer's gossiped directory view (empty if it serves none)."""
    try:
        with urlopen(base_url.rstrip("/") + "/mesh/directory", timeout=timeout) as r:
            return RegionDirectory.from_dict(json.load(r))
    except OSError:
        return RegionDirectory()


def gossip(base_urls: List[str]) -> RegionDirectory:
    """Build a directory from a set of seed peers: record each as a direct provider of
    the regions it advertises, and merge its gossiped view for transitive discovery.
    Peer ids in the returned directory ARE the base URLs (so :func:`http_shard_getter`
    can dial them)."""
    directory = RegionDirectory()
    for url in base_urls:
        base = url.rstrip("/")
        try:
            directory.observe(base, fetch_manifests(base))
        except OSError:
            continue  # unreachable seed — skip, the mesh routes around it
        directory.merge(fetch_directory(base))
    return directory


def http_shard_getter(timeout: int = 30):
    """A ``get_shard(peer_url, region)`` callable for :func:`directory.pull_regions`."""
    def _get(peer_url: str, region_id: str):
        return fetch_shard(peer_url, region_id, timeout=timeout)
    return _get


def pull_regions_http(directory: RegionDirectory, regions: List[str], node: MeshNode,
                      authority_pub: Ed25519PublicKey):
    """Anchor-routed multi-peer pull over HTTP: fetch each routed region from its best
    live provider in ``directory``, verified + blind, with self-healing failover."""
    return pull_regions(directory, regions, node, authority_pub, http_shard_getter())


def pull_from(base_url: str, node: MeshNode, authority_pub: Ed25519PublicKey) -> List[str]:
    """Sync from a peer's HTTP mesh endpoint: pull every missing/newer region,
    verify (signature + item hashes + content_root), import. Returns synced regions."""
    synced: List[str] = []
    for pm in fetch_manifests(base_url):
        if pm.version > node.version_of(pm.region_id):
            manifest, items = fetch_shard(base_url, pm.region_id)
            if manifest is None:
                continue
            node.import_shard(manifest, items, authority_pub)   # raises on tamper/forgery
            synced.append(pm.region_id)
    return synced


def _seed_demo(node: MeshNode) -> Ed25519PublicKey:
    """Author a demo region so a fresh seed node has something to serve."""
    priv = Ed25519PrivateKey.generate()
    node.put_shard(
        "geo-01",
        {"art-paris": b"ENC:city=paris", "art-tokyo": b"ENC:city=tokyo",
         "art-cairo": b"ENC:city=cairo"},
        version=1, authority="geo-authority", priv=priv,
        consensus={"art-paris": 0.98, "art-tokyo": 0.97, "art-cairo": 0.91},
    )
    return priv.public_key()


def main() -> None:
    from cryptography.hazmat.primitives import serialization

    ap = argparse.ArgumentParser(description="Agience Mantle mesh node")
    ap.add_argument("--port", type=int, default=9701)
    ap.add_argument("--node-id", default=None)
    ap.add_argument("--seed", action="store_true", help="author a demo region on startup")
    ap.add_argument("--sync", default=None, help="peer base URL to sync from (e.g. http://host:9701)")
    ap.add_argument("--authority", default=None, help="authority public key (hex) to verify synced shards")
    ap.add_argument("--gossip", default=None,
                    help="comma-separated peer URLs to gossip into this node's directory")
    ap.add_argument("--advertise", default=None,
                    help="this node's public base URL; adds self to the served directory")
    args = ap.parse_args()

    node = MeshNode(args.node_id or f"node-{args.port}")

    if args.seed:
        pub = _seed_demo(node)
        pub_hex = pub.public_bytes(serialization.Encoding.Raw,
                                   serialization.PublicFormat.Raw).hex()
        print(f"[seed] authored region geo-01. AUTHORITY PUBKEY (give to peers):\n  {pub_hex}")

    if args.sync:
        if not args.authority:
            raise SystemExit("--sync requires --authority <hex pubkey>")
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(args.authority))
        synced = pull_from(args.sync, node, pub)
        print(f"[sync] pulled {synced} from {args.sync}")

    # The node's gossiped view of the wider mesh (served at /mesh/directory).
    directory = gossip([u.strip() for u in args.gossip.split(",") if u.strip()]) if args.gossip \
        else RegionDirectory()
    if args.advertise:
        directory.observe(args.advertise.rstrip("/"), node.manifests())  # advertise our own cells
    if args.gossip:
        print(f"[gossip] directory knows {len(directory.peers())} peers, "
              f"{len(directory.regions())} regions")

    serve(node, args.port, directory=directory)
    print(f"[node {node.node_id}] serving on :{args.port}  "
          f"(GET /mesh/health · /mesh/directory)  ·  regions={list(node.summary()['regions'])}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
