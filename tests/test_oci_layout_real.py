"""The golden case: a layout produced by real `docker buildx --output type=oci`.

Every other OCI test runs against `_oci_layout.py`, a hand-written fixture built from the
specification. A fixture is a claim about the format, and a suite that only ever tests against its
own claim proves the code matches the fixture's model of the format, not the tool that actually
produces it — the `[[fixture-faithful-to-model-not-corpus]]` shape.

Captured from buildx v0.33.0 / docker 29.4.2: `org.opencontainers.image.ref.name` holds the bare tag
`edge`, not `agience-mantle:edge` — the hand-written fixture had assumed the qualified form. The
qualified name lives in a separate annotation, `io.containerd.image.name`; a registry tag is bare,
and this fixture is what lets a test say so against a real build rather than another guess.

The fixture in `fixtures/oci-layout-real/` is 1.5 KB of real build output: an index, a manifest, a
config and one layer, from `FROM scratch` + `COPY payload.txt`. `FROM scratch` deliberately, so
capturing it pulled no base image and the bytes depend on nothing that can change under us.

It is one builder at one version. A future buildx may write a different index shape; a golden
sample makes that visible immediately rather than silently, which is why it is captured rather than
regenerated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mantle.oci.address import digest_to_ref, ref_to_digest
from mantle.oci.layout import blob_bytes, read_layout

REAL = Path(__file__).resolve().parent / "fixtures" / "oci-layout-real"

pytestmark = pytest.mark.skipif(
    not (REAL / "index.json").is_file(),
    # Named, not silent: without this fixture the entire OCI suite is self-referential again.
    reason="the captured buildx layout is missing — every remaining OCI test then measures only "
           "the hand-written fixture")


def test_a_real_buildx_layout_reads():
    images = read_layout(REAL)
    assert len(images) == 1
    img = images[0]
    assert img.media_type == "application/vnd.oci.image.manifest.v1+json"
    # manifest + config + one layer
    assert len(img.blobs) == 3
    kinds = [b.media_type for b in img.blobs]
    assert kinds[0] == "application/vnd.oci.image.manifest.v1+json"
    assert kinds[1] == "application/vnd.oci.image.config.v1+json"
    assert kinds[2].startswith("application/vnd.oci.image.layer.v1.tar")


def test_the_ref_name_is_a_bare_tag():
    """The hand-written fixture claimed `agience-mantle:edge`; buildx writes `edge`. Pinned here so
    the fixture cannot drift back to the guess."""
    assert read_layout(REAL)[0].ref_name == "edge"
    index = json.loads((REAL / "index.json").read_text(encoding="utf-8"))
    ann = index["manifests"][0]["annotations"]
    assert ann["org.opencontainers.image.ref.name"] == "edge"
    assert ann["io.containerd.image.name"].endswith("agience-mantle:edge"), \
        "the qualified name lives in a DIFFERENT annotation — that is the distinction this pins"


def test_every_real_blob_verifies_against_its_own_name():
    """The verify path, run over bytes a build actually produced rather than bytes a test wrote."""
    for blob in read_layout(REAL)[0].blobs:
        body = b"".join(blob_bytes(blob))
        assert "sha256:" + hashlib.sha256(body).hexdigest() == blob.digest
        assert len(body) == blob.size


def test_the_real_digest_maps_to_a_content_ref_and_back():
    """The deploy claim, on a real image: the digest buildx assigned survives the trip into the
    lattice's address space and back unchanged."""
    d = read_layout(REAL)[0].digest
    assert ref_to_digest(digest_to_ref(d)) == d
    assert digest_to_ref(d) == "cas/" + d.split(":")[1]


def test_the_index_carries_a_platform_and_extra_annotations():
    """Keys the spec permits and a hand-written fixture would omit. The parser ignores what it does
    not use, so an index that carries more than it expects still reads."""
    entry = json.loads((REAL / "index.json").read_text(encoding="utf-8"))["manifests"][0]
    assert entry["platform"]["os"] == "linux"
    assert "org.opencontainers.image.created" in entry["annotations"]
    read_layout(REAL)          # must not raise on any of it
