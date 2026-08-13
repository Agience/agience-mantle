"""Reading a build's OCI layout: a layout that lies about itself does not verify.

A layout is a directory of files plus an `index.json` making claims about what those files hash to.
The claims are not evidence. These tests build real layouts on disk — correct ones and each specific
kind of broken one — and assert which are accepted.

The failure this prevents is displaced in time, which is why it needs its own tests. The content
store verifies content against its ref on read. So a blob accepted here under the wrong name does
not fail here — it fails months later, on a pull, reported as store corruption, nowhere near the
layout that caused it. Verifying at the door is what keeps the error next to its cause.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mantle.oci.layout import (
    Blob, OciLayoutError, blob_bytes, read_layout, verify_blob,
)


from ._oci_layout import make_layout, put_blob as _put   # the one fixture builder


# ── the good case ────────────────────────────────────────────────────────────────────────────────

def test_a_real_layout_reads_with_every_blob_verified(tmp_path: Path):
    root, man = make_layout(tmp_path / "out")
    images = read_layout(root)
    assert len(images) == 1
    img = images[0]
    assert img.digest == man, "the manifest digest must come from the layout, not be re-derived"
    assert img.ref_name == "edge"   # a bare tag — verified against a real build, see test_oci_layout_real
    # manifest + config + 2 layers
    assert len(img.blobs) == 4
    assert all(b.digest.startswith("sha256:") for b in img.blobs)
    assert img.blobs[0].digest == man, "the manifest is itself a blob and must be stored as one"


def test_the_manifest_digest_is_the_promotable_name(tmp_path: Path):
    """The property the whole lane rests on: read it twice, get the same name; and that name is
    the sha256 of the manifest bytes, not of anything this code composed."""
    root, man = make_layout(tmp_path / "out")
    assert read_layout(root)[0].digest == read_layout(root)[0].digest == man
    body = (root / "blobs" / "sha256" / man.split(":")[1]).read_bytes()
    assert man == "sha256:" + hashlib.sha256(body).hexdigest()


def test_blob_bytes_streams_the_whole_blob(tmp_path: Path):
    root, _ = make_layout(tmp_path / "out")
    img = read_layout(root)[0]
    cfg = img.blobs[1]
    assert hashlib.sha256(b"".join(blob_bytes(cfg))).hexdigest() == cfg.digest.split(":")[1]


# ── invalid layouts, each named by what it prevents ─────────────────────────────────────────────

def test_a_blob_that_does_not_hash_to_its_name_is_refused(tmp_path: Path):
    """The one that matters: corrupt a layer in place, leaving every claim intact."""
    root, _ = make_layout(tmp_path / "out")
    img = read_layout(root)[0]
    victim = img.blobs[-1]
    victim.path.write_bytes(b"different bytes entirely")
    with pytest.raises(OciLayoutError, match="does not hash to its own name"):
        read_layout(root)


def test_a_missing_blob_is_refused_rather_than_partially_ingested(tmp_path: Path):
    """A manifest in the store naming blobs that are not there does not read as a layout — the
    alternative is an image that pulls halfway: 200 for the manifest, 404 for a layer."""
    root, _ = make_layout(tmp_path / "out")
    read_layout(root)[0].blobs[-1].path.unlink()
    with pytest.raises(OciLayoutError, match="is not on disk"):
        read_layout(root)


def test_a_wrong_declared_size_is_refused(tmp_path: Path):
    """The content is what it claims to be, so the manifest is wrong — and a wrong size is what a
    puller allocates against."""
    root, man = make_layout(tmp_path / "out")
    mpath = root / "blobs" / "sha256" / man.split(":")[1]
    doc = json.loads(mpath.read_bytes())
    doc["layers"][0]["size"] = 999999
    body = json.dumps(doc).encode()
    new = _put(root, body)                       # re-address it, so only the size claim is wrong
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    idx["manifests"][0]["digest"] = new
    (root / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    with pytest.raises(OciLayoutError, match="declares"):
        read_layout(root)


def test_a_traversal_digest_never_becomes_a_path(tmp_path: Path):
    """`index.json` is a file; in the general case it is attacker-supplied. A digest-shaped string
    is only digest-shaped until something checks it."""
    root, _ = make_layout(tmp_path / "out")
    idx = json.loads((root / "index.json").read_text(encoding="utf-8"))
    idx["manifests"][0]["digest"] = "sha256:../../../../etc/passwd"
    (root / "index.json").write_text(json.dumps(idx), encoding="utf-8")
    with pytest.raises(OciLayoutError):
        read_layout(root)


def test_a_docker_save_tarball_is_not_mistaken_for_a_layout(tmp_path: Path):
    """A different format whose digests are not the registry's — a docker-save tarball can carry
    an image with no registry digest at all."""
    d = tmp_path / "notalayout"
    d.mkdir()
    (d / "manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(OciLayoutError, match="not an OCI layout"):
        read_layout(d)


def test_an_empty_index_is_a_failure_not_a_no_op(tmp_path: Path):
    """An empty index is a failure, not a no-op: 'ingested successfully, added nothing' is
    indistinguishable from a working build."""
    root, _ = make_layout(tmp_path / "out")
    (root / "index.json").write_text(json.dumps({"manifests": []}), encoding="utf-8")
    with pytest.raises(OciLayoutError, match="no manifests"):
        read_layout(root)


def test_a_manifest_list_is_refused_explicitly(tmp_path: Path):
    """A manifest list does not read as an image: storing a multi-platform list's own bytes and
    calling that an image would leave every platform missing, while the manifest still serves
    200."""
    root = tmp_path / "out"
    root.mkdir()
    (root / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}), encoding="utf-8")
    inner = json.dumps({"schemaVersion": 2, "manifests": [
        {"digest": "sha256:" + "b" * 64, "size": 1,
         "mediaType": "application/vnd.oci.image.manifest.v1+json"}]}).encode()
    d = _put(root, inner)
    (root / "index.json").write_text(json.dumps({"manifests": [
        {"digest": d, "size": len(inner),
         "mediaType": "application/vnd.oci.image.index.v1+json"}]}), encoding="utf-8")
    with pytest.raises(OciLayoutError, match="manifest LIST"):
        read_layout(root)


def test_an_unknown_layout_version_is_refused_not_guessed(tmp_path: Path):
    root, _ = make_layout(tmp_path / "out")
    (root / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "2.0.0"}), encoding="utf-8")
    with pytest.raises(OciLayoutError, match="unsupported imageLayoutVersion"):
        read_layout(root)


def test_verify_blob_returns_the_measured_size_not_the_claimed_one(tmp_path: Path):
    data = b"x" * 4096
    p = tmp_path / "b"
    p.write_bytes(data)
    assert verify_blob(p, "sha256:" + hashlib.sha256(data).hexdigest()) == 4096
