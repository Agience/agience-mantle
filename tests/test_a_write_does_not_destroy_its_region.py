"""A write must not destroy the rest of its region.

`MeshNode.put_shard` replaces a region wholesale:

    self._shards[region_id] = {"manifest": …, "items": dict(items)}

so a caller in `LocalCache` must hand it the full contents of the region, not just the items it is
currently writing — passing only the new items would drop everything else in that region from
persistence. `revise` carries the same requirement: the previous version must remain in the shard,
not just in memory.

The fix belongs at the caller, not inside `put_shard`. The manifest is signed over exactly the
items it is given; a node that merged internally would be signing a set its caller never named.
`_region_payload` makes the full contents explicit, so the signature keeps meaning what it says.

These tests read from the node's shards (`_shard_items`), not the in-memory index: reads are
served from memory, so a region that lost content on disk would still look intact to every read
until a restart reloaded from the shard.
"""
from __future__ import annotations

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from mantle.search.anchors.anchorset import AnchorSet
from mantle.shard.cache import LocalCache
from prism.mass import Provenance

DIM = 8
PRINCIPAL = "0fa20e2c-0bb4-4fc8-94ea-f79e73659a64"
COLLECTION = "c7b1a2d3-4e5f-4a6b-8c9d-0e1f2a3b4c5d"


def _anchors() -> AnchorSet:
    a = AnchorSet(model_id="canonical-test", dim=DIM)
    for i, label in enumerate(["alpha", "beta"]):
        v = np.full(DIM, 0.05, dtype=np.float32)
        v[i] = 1.0
        a.add_text(label, v)
    return a


def _vec(i: int = 0):
    v = np.full(DIM, 0.05, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture()
def cache():
    return LocalCache(_anchors(), PRINCIPAL, COLLECTION, node_id="test-node")


def _shard_items(cache) -> set:
    """What actually persists: read from the node's shards, not the in-memory index — a region
    could lose content on disk while the in-memory index still reports it present."""
    out = set()
    for region in cache.node.summary()["regions"]:
        _, items = cache.node.get_shard(region)
        out |= set(items or {})
    return out


def test_a_SECOND_write_to_the_same_region_keeps_the_FIRST(cache):
    """Two separate writes routed to one region; both documents must persist.

    Pins against passing only the new items to `put_shard`. Asserted against the shard, not the
    in-memory index, because the index holds both the whole time and would report success either
    way."""
    priv = Ed25519PrivateKey.generate()
    cache.put([("doc-A", b"first", _vec(0))], authority="me", priv=priv, version=1)
    cache.put([("doc-B", b"second", _vec(0))], authority="me", priv=priv, version=1)

    same_region = len({i.region for i in (cache._find("doc-A"), cache._find("doc-B"))}) == 1
    assert same_region, "control: the two writes did not land in one region — the test proves nothing"
    assert _shard_items(cache) == {"doc-A", "doc-B"}, "a write destroyed its region's other content"


def test_a_REVISION_keeps_every_prior_version_IN_THE_SHARD(cache):
    """Immutability has to survive a restart, not just a read: the prior version must persist in
    the shard.

    Pins against `revise` writing `{new_id: content}` alone, which would let the previous version
    vanish from the shard while remaining in memory."""
    priv = Ed25519PrivateKey.generate()
    cache.put([("doc-1", b"v1", _vec(0))], authority="me", priv=priv, version=1)
    landed = cache.revise("doc-1", b"v2", _vec(0), provenance=Provenance.HYPOTHESIS,
                          authority="me", priv=priv)
    assert _shard_items(cache) == {"doc-1", landed.id}


def test_the_MANIFEST_describes_exactly_what_the_shard_holds(cache):
    """The manifest is signed over the items it is handed, so it must name the shard's true
    contents — otherwise a verifier and a reader disagree about what was stood behind. This is why
    the merge happens at the caller and not inside `put_shard`.

    Pins against merging inside `put_shard`, which would sign a set the caller never specified and
    leave the manifest describing fewer items than the shard holds."""
    priv = Ed25519PrivateKey.generate()
    cache.put([("doc-A", b"first", _vec(0))], authority="me", priv=priv, version=1)
    cache.put([("doc-B", b"second", _vec(0))], authority="me", priv=priv, version=1)
    for region in cache.node.summary()["regions"]:
        manifest, items = cache.node.get_shard(region)
        named = {getattr(it, "id", None) or it["id"] for it in manifest.items}
        assert named == set(items), (
            "the manifest names %s but the shard holds %s" % (sorted(named), sorted(items)))


def test_a_write_to_a_DIFFERENT_region_leaves_the_other_alone(cache):
    """The other direction: a write must not touch a region it was not asked about.

    Pins against `_region_payload` reading across regions instead of within one."""
    priv = Ed25519PrivateKey.generate()
    cache.put([("doc-A", b"first", _vec(0))], authority="me", priv=priv, version=1)
    before = {r: sorted(cache.node.get_shard(r)[1] or {}) for r in cache.node.summary()["regions"]}
    cache.put([("doc-Z", b"elsewhere", _vec(1))], authority="me", priv=priv, version=1)

    a_region = cache._find("doc-A").region
    z_region = cache._find("doc-Z").region
    assert a_region != z_region, "control: both landed in one region — this test proves nothing"
    assert sorted(cache.node.get_shard(a_region)[1] or {}) == before[a_region], (
        "writing doc-Z disturbed doc-A's region")
    assert _shard_items(cache) == {"doc-A", "doc-Z"}
