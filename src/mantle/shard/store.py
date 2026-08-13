"""Persistence — a cache that forgets on restart is not a cache.

If `cache_dir` were configured but nothing wrote to it, every shard would die with the process,
making disconnected operation a fiction on the *second* boot: the first run answers locally, the
next one has nothing and must reach the network to say anything at all — which is exactly the
moment a leaf is supposed to be useful.

WHAT IS AND IS NOT PERSISTED
----------------------------
Shards are stored as their **signed manifest + opaque item bytes**, exactly as they arrived.
Nothing is re-derived, re-signed, or re-encrypted on the way to disk:

* **The manifest's signature is the point.** A shard loaded from disk is re-verified against the
  authority's key on read, so a tampered cache file is caught by the same check that catches a
  tampered peer. Disk is just another untrusted server. (This is why Ember can trust its own
  cache without trusting the filesystem — the trust comes from the signature, not the location.)
* **Item bytes stay opaque.** Ember holds ciphertext it cannot read; persistence must not
  assume otherwise, so bytes go to disk byte-for-byte.
* **Vectors are NOT persisted.** They are a derived retrieval view, cheap to recompute, and
  writing them would silently duplicate plaintext-adjacent data next to a cache that is
  otherwise blind. Storage is for what was given to us; derivations are rebuilt.

LAYOUT
------
    <cache_dir>/shards/<urlsafe(region_id)>.json      manifest + b64 items
    <cache_dir>/crosswalks/<crosswalk_id>.json        adopted/fitted cross-walk artifacts
    <cache_dir>/anchors.json                          the cached canonical AnchorSet

Region ids contain `/`, so they are urlsafe-b64'd into filenames rather than nested — a region
id is an opaque key, not a path, and treating it as one invites traversal.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from mantle.search.anchors.anchorset import AnchorSet, AnchorSetCorrupt
from mantle.mesh.node import MeshNode, ShardVerifyError
from mantle.mesh.manifest import ShardManifest


def _key(region_id: str) -> str:
    """Region id -> a flat, safe filename. Never a path: `..` and `/` must not escape."""
    return base64.urlsafe_b64encode(region_id.encode("utf-8")).decode("ascii").rstrip("=")


def _unkey(name: str) -> str:
    pad = "=" * (-len(name) % 4)
    return base64.urlsafe_b64decode(name + pad).decode("utf-8")


def _write_atomic(path: Path, payload: dict) -> None:
    """Write via a temp file + replace.

    A half-written shard is worse than a missing one: it survives restarts and fails
    verification forever, and the cache has no way to tell "corrupt" from "malicious". os.replace
    is atomic on POSIX and Windows, so a reader sees the old file or the new one, never a torn one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class ShardStore:
    """Signed shards on disk. Re-verified on load — disk is just another untrusted server."""

    def __init__(self, cache_dir: Path) -> None:
        self.root = Path(cache_dir)
        self.shards = self.root / "shards"
        self.crosswalks = self.root / "crosswalks"

    # ------------------------------------------------------------------ shards
    def put(self, manifest: ShardManifest, items: Dict[str, bytes]) -> None:
        _write_atomic(self.shards / f"{_key(manifest.region_id)}.json", {
            "manifest": manifest.to_dict(),
            "items": {k: base64.b64encode(v).decode("ascii") for k, v in items.items()},
        })

    def get(self, region_id: str) -> Optional[Tuple[ShardManifest, Dict[str, bytes]]]:
        p = self.shards / f"{_key(region_id)}.json"
        if not p.is_file():
            return None
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return (ShardManifest.from_dict(d["manifest"]),
                    {k: base64.b64decode(v) for k, v in d["items"].items()})
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Unreadable = absent. A corrupt cache entry must degrade to a miss (refetch),
            # never to a crash: the leaf's whole job is to keep working.
            return None

    def regions(self) -> Iterator[str]:
        if not self.shards.is_dir():
            return iter(())
        return (_unkey(p.stem) for p in self.shards.glob("*.json"))

    def evict(self, region_id: str) -> bool:
        p = self.shards / f"{_key(region_id)}.json"
        if p.is_file():
            p.unlink()
            return True
        return False

    # ------------------------------------------------------------------ whole-node
    def save_node(self, node: MeshNode) -> int:
        n = 0
        for region in node.summary()["regions"]:
            manifest, items = node.get_shard(region)
            if manifest is not None:
                self.put(manifest, items)
                n += 1
        return n

    def load_node(self, node: MeshNode, authority_pub) -> Tuple[int, int]:
        """Rehydrate a node from disk, **verifying every shard**. Returns (loaded, rejected).

        Verification is not ceremony. A cache file is as untrusted as a peer's response — same
        `import_shard`, same signature + content_root + per-item hash checks. A shard that fails
        is dropped rather than raising, so one bad file cannot stop a leaf from booting; the
        region simply becomes a miss and refills when a channel exists.
        """
        loaded = rejected = 0
        for region in list(self.regions()):
            got = self.get(region)
            if got is None:
                rejected += 1
                continue
            manifest, items = got
            try:
                node.import_shard(manifest, items, authority_pub)
                loaded += 1
            except ShardVerifyError:
                rejected += 1
        return loaded, rejected

    # ------------------------------------------------------------------ artifacts
    def put_artifact(self, artifact: dict) -> None:
        """Cache an artifact (e.g. a cross-walk) by its id. Everything is an artifact, so this
        is the one shape that stores all of them."""
        _write_atomic(self.crosswalks / f"{artifact['id']}.json", artifact)

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        p = self.crosswalks / f"{artifact_id}.json"
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return None

    def find_artifact(self, content_type: str) -> Optional[dict]:
        """First cached artifact of a type — used to look for a usable cross-walk before
        refitting. The caller still verifies it: cached is not the same as valid."""
        if not self.crosswalks.is_dir():
            return None
        for p in self.crosswalks.glob("*.json"):
            try:
                a = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError):
                continue
            if a.get("content_type") == content_type:
                return a
        return None

    # ------------------------------------------------------------------ anchors
    def save_anchors(self, anchors: AnchorSet) -> None:
        anchors.save(self.root / "anchors.json")

    def load_anchors(self) -> Optional[AnchorSet]:
        """The cached canonical set, or ``None`` when this shard has none.

        A CORRUPT set is not the same answer as an absent one and is not swallowed into it:
        `AnchorSetCorrupt` means the file states anchor ids its own contents do not produce, so a
        leaf that loaded it would route into regions no peer computes. Absent is a state an
        unprovisioned leaf is legitimately in; corrupt is a file to replace.
        """
        p = self.root / "anchors.json"
        if not p.is_file():
            return None
        try:
            return AnchorSet.load(p)
        except AnchorSetCorrupt:
            raise
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    # ------------------------------------------------------------------ derived views
    def save_views(self, views: dict) -> None:
        """Persist the DERIVED views the shards can't reconstruct on their own: the head map
        (which version of each root is current) and, per item, its root_id and retrieval vector.

        The shards hold content + mass (the manifest's consensus); they do NOT hold which version
        is HEAD (a local decision made at revise() time by `may_revise`) nor the vector (derived
        from the embed-text, not the content). So a restart that reloaded only shards would keep
        the bytes but lose the version lineage and the searchable view. This sidecar closes that.

        Vectors are stored ONLY for readable items — the same blind-safety line the rest of the
        cache holds: we persist a vector only where we already held one, never for ciphertext we
        cannot read.
        """
        _write_atomic(self.root / "views.json", views)

    def load_views(self) -> dict:
        """The head map + per-item root/vector sidecar, or an empty scaffold if absent (a first
        boot, or nothing to restore yet — both fine)."""
        p = self.root / "views.json"
        if not p.is_file():
            return {"heads": {}, "items": {}}
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return {"heads": dict(d.get("heads") or {}), "items": dict(d.get("items") or {})}
        except (json.JSONDecodeError, ValueError, TypeError):
            return {"heads": {}, "items": {}}   # corrupt sidecar => rebuild loose, never crash


__all__ = ["ShardStore"]
