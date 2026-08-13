"""Read an OCI image layout from disk — the shape `docker buildx --output type=oci` writes.

    layout/
      oci-layout                 {"imageLayoutVersion": "1.0.0"}
      index.json                 manifests: [{digest, mediaType, annotations:{...ref.name}}]
      blobs/sha256/<hex>         every manifest, config and layer, one file per digest

This is the sovereign builder's output, and it is already content-addressed. The builder writes a
layout; this reads it; `oci.address` renames each digest to a content ref; the store keeps it. At no
point is the image re-derived, which is what makes the digest promoted to two boxes the same digest
rather than two builds that agree.

Every blob is verified against its claimed digest before it is accepted. A layout is a directory of
files. Its `index.json` is a claim about what those files hash to, and the filename is another
claim; neither is evidence. Ingesting on trust would let a corrupted or swapped file enter the store
under the name of a good one — and because the store verifies content against `ref` on read, that
blob would then fail to read with a corruption error pointing at the store, months later and nowhere
near the layout that caused it. Verify at the door: that is the one place where the bytes and the
claim are both present and the failure names the right thing.

The verify is the only place this module hashes. `address.py` must never hash — it renames a number
the caller already has. Here the number is being established against bytes for the first time,
which is a different act and belongs in a different module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Tuple

from mantle.oci.address import OciAddressError, parse_digest

#: Read in chunks: a layer is routinely hundreds of MB and this runs on a 2 GB Pi as readily as on
#: a build host. Reading a blob whole to check its hash would make ingest's memory a function of the
#: largest layer, which is exactly the shape that works in testing and fails on the real image.
_CHUNK = 1 << 20


class OciLayoutError(Exception):
    """A layout that cannot be trusted or cannot be read. Always names the blob and the rule."""


class Blob(NamedTuple):
    digest: str        # sha256:<hex>, VERIFIED against the bytes
    size: int
    path: Path
    media_type: str


class Image(NamedTuple):
    """One manifest from the layout's index, with every blob it transitively names."""
    digest: str        # the manifest's own digest — the promotable name of the image
    media_type: str
    ref_name: str      # the layout's `org.opencontainers.image.ref.name`, or "" — a TAG, not an id
    blobs: List[Blob]  # config + layers + the manifest itself


def _blob_path(root: Path, digest: str) -> Path:
    """`sha256:<hex>` -> `blobs/sha256/<hex>`, via the parser that a traversal path does not pass.

    `parse_digest` is what stands between `index.json` — an attacker-supplied file in the general
    case — and a path join. `blobs/sha256/../../../../etc/passwd` is a digest-shaped string only
    until something checks it.
    """
    return root / "blobs" / "sha256" / parse_digest(digest)


def verify_blob(path: Path, digest: str) -> int:
    """Hash the file and compare to the claim. Returns the size. Raises on mismatch."""
    want = parse_digest(digest)
    h = hashlib.sha256()
    size = 0
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)
    except FileNotFoundError:
        raise OciLayoutError(
            "the layout's index names %s but blobs/sha256/%s is not on disk — an incomplete layout, "
            "not a corrupt one, and ingesting the rest would store a manifest whose parts are "
            "missing" % (digest, want)) from None
    got = h.hexdigest()
    if got != want:
        raise OciLayoutError(
            "blob %s does not hash to its own name (got sha256:%s) — REFUSING to ingest. The store "
            "verifies content against its ref on read, so accepting this would turn a bad file here "
            "into an unreadable blob later, reported against the store rather than the layout"
            % (digest, got))
    return size


def read_layout(root: Path) -> List[Image]:
    """Every image in the layout, fully verified. Nothing is returned until everything checks out.

    All-or-nothing: a partially-ingested image is a manifest in the store naming blobs that are not
    there — which serves a 200 for the manifest and a 404 for a layer, i.e. an image that pulls
    halfway. So the whole layout is read atomically, raising on the first blob that fails and
    naming it, rather than ingesting part of one.
    """
    root = Path(root)
    marker = root / "oci-layout"
    if not marker.is_file():
        raise OciLayoutError(
            "%s has no `oci-layout` file — this is not an OCI layout. A `docker save` tarball is a "
            "DIFFERENT format and its digests are not the registry's" % root)
    try:
        version = json.loads(marker.read_text(encoding="utf-8")).get("imageLayoutVersion")
    except (json.JSONDecodeError, OSError) as exc:
        raise OciLayoutError("cannot read %s: %s" % (marker, exc)) from None
    if not str(version or "").startswith("1."):
        raise OciLayoutError(
            "unsupported imageLayoutVersion %r — refusing rather than guessing at a layout shape "
            "this code has not been read against" % (version,))

    index_path = root / "index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise OciLayoutError("cannot read %s: %s" % (index_path, exc)) from None

    manifests = index.get("manifests") or []
    if not manifests:
        raise OciLayoutError(
            "%s lists no manifests — an empty layout. Ingesting it would report success and add "
            "nothing, which is indistinguishable from a working build that produced no image"
            % index_path)

    images: List[Image] = []
    for entry in manifests:
        digest = entry.get("digest") or ""
        try:
            parse_digest(digest)
        except OciAddressError as exc:
            raise OciLayoutError("index.json: %s" % exc) from None
        mpath = _blob_path(root, digest)
        msize = verify_blob(mpath, digest)
        manifest = json.loads(mpath.read_text(encoding="utf-8"))

        blobs = [Blob(digest, msize, mpath, entry.get("mediaType")
                      or manifest.get("mediaType") or "application/octet-stream")]
        # config + layers. An index/manifest-list nests one level further; refused explicitly below
        # rather than silently ingesting only its own bytes and calling that an image.
        referenced = []
        if manifest.get("config"):
            referenced.append(manifest["config"])
        referenced.extend(manifest.get("layers") or [])
        if not referenced and manifest.get("manifests"):
            raise OciLayoutError(
                "%s is a manifest LIST (multi-platform). This reads single-platform images; "
                "ingesting the list alone would store an image whose platforms are all missing"
                % digest)
        for item in referenced:
            d = item.get("digest") or ""
            p = _blob_path(root, d)
            size = verify_blob(p, d)
            declared = item.get("size")
            if declared is not None and int(declared) != size:
                raise OciLayoutError(
                    "blob %s is %d bytes but the manifest declares %d — the content is what it "
                    "claims to be, so the MANIFEST is wrong, and a wrong size is what a puller "
                    "allocates against" % (d, size, int(declared)))
            blobs.append(Blob(d, size, p, item.get("mediaType") or "application/octet-stream"))

        images.append(Image(
            digest=digest,
            media_type=entry.get("mediaType") or manifest.get("mediaType") or "",
            ref_name=(entry.get("annotations") or {}).get(
                "org.opencontainers.image.ref.name", ""),
            blobs=blobs,
        ))
    return images


def blob_bytes(blob: Blob) -> Iterator[bytes]:
    """Stream a verified blob. Separate from `verify_blob` so nothing reads a blob it has not
    checked — the two are one call apart, and that is the only ordering that is safe."""
    with open(blob.path, "rb") as fh:
        while True:
            chunk = fh.read(_CHUNK)
            if not chunk:
                return
            yield chunk
