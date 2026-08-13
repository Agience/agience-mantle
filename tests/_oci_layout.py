"""The one builder of a test OCI layout — shared by `test_oci_layout` and `test_oci_store`.

Not imported across test modules. `tests/` is a package and siblings import helpers relatively
(`from ._package_root import …`); reaching into another test module by bare name resolves against
sys.path, which in this workspace is where duplicate basenames can silently substitute for each
other. A helper module, imported the way the repo already imports helpers, has one identity.

One source, because the fixture is a claim about the format: two copies of this function would
drift, and the drift would show up as one test file proving the parser against a shape the other
file's tests do not use — coverage that reads as agreement and is not.

`org.opencontainers.image.ref.name` holds the bare tag (e.g. `edge`), not the qualified image
name — buildx puts the qualified name in `io.containerd.image.name` instead. `Image.ref_name` is
therefore a tag, which is what a registry tag actually is. `tests/fixtures/oci-layout-real/` is a
captured real build, and `test_oci_layout_real.py` reads it so this fixture cannot drift into a
model of itself.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Tuple


def put_blob(root: Path, data: bytes) -> str:
    """Write a blob at its TRUE address and return the digest. Nothing here fakes a hash — a
    fixture that could write a blob under the wrong name would make the verify tests meaningless."""
    hexd = hashlib.sha256(data).hexdigest()
    p = root / "blobs" / "sha256" / hexd
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return "sha256:" + hexd


def make_layout(root: Path, *, ref_name: str = "edge",
                layers: int = 2) -> Tuple[Path, str]:
    """A minimal but structurally real single-platform layout. Returns (root, manifest digest)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}), encoding="utf-8")

    config = json.dumps({"architecture": "amd64", "os": "linux"}).encode()
    cfg_digest = put_blob(root, config)
    layer_items = []
    for i in range(layers):
        body = b"layer-%d-" % i + bytes(200)
        d = put_blob(root, body)
        layer_items.append({"digest": d, "size": len(body),
                            "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"})

    manifest = json.dumps({
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.manifest.v1+json",
        "config": {"digest": cfg_digest, "size": len(config),
                   "mediaType": "application/vnd.oci.image.config.v1+json"},
        "layers": layer_items,
    }).encode()
    man_digest = put_blob(root, manifest)

    (root / "index.json").write_text(json.dumps({
        "schemaVersion": 2,
        # The annotation set and the `platform` key are what a real buildx index carries — copied
        # from `tests/fixtures/oci-layout-real/index.json` rather than imagined.
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [{"digest": man_digest, "size": len(manifest),
                       "mediaType": "application/vnd.oci.image.manifest.v1+json",
                       "annotations": {
                           "io.containerd.image.name": "docker.io/library/agience-mantle:" + ref_name,
                           "org.opencontainers.image.created": "2026-08-05T19:48:44Z",
                           "org.opencontainers.image.ref.name": ref_name},
                       "platform": {"architecture": "amd64", "os": "linux"}}],
    }), encoding="utf-8")
    return root, man_digest
