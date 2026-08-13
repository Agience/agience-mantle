"""The read-time head resolution is observable, because production traffic does not exercise it.

The live lattice holds zero multi-revision roots (no id carries the `root~hash` marker), and there
is no populated node cache — `/d/agience-data/ember/` holds one key file. `revise()` has not been
called in production, so the resolver has had nothing to resolve, and this suite exercises the
mechanism directly rather than relying on production traffic to test it.

A mechanism nobody can see fire is one nobody can see fail. [[never-handroll-probes]]: a missing
stat is added rather than probed around, so `summary()["resolution"]` publishes what the
resolution actually did.

`multi_revision` is the only count that means anything: a root with one version cannot be
narrowed, so counting it would report the mechanism working when it was never asked a question.
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


def _vec(i=0):
    v = np.full(DIM, 0.05, dtype=np.float32)
    v[i] = 1.0
    return v


def _cache(resolve=None):
    return LocalCache(_anchors(), PRINCIPAL, COLLECTION, node_id="test-node", resolve=resolve)


def test_an_UNWIRED_cache_says_so_rather_than_looking_healthy():
    """A summary that omits the resolver entirely reads the same as one with a resolver that never
    narrowed anything — "no resolver" and "a resolver that never narrowed anything" produce
    identical read behaviour, every revision answers — so only the stat can tell them apart.
    """
    c = _cache()
    assert c.summary()["resolution"]["resolver"] == "none"


def test_a_SINGLE_version_root_is_not_counted_as_a_resolution():
    """Every read passes single-version roots through the resolver; if those were tallied, a store
    that had never seen a revision would publish a healthy-looking `grounded_out`, with the
    mechanism untested and nothing to show it. The count is `len(offered) > 1`, not `roots_read`,
    so a single-version pass-through does not register as a resolution.
    """
    seen = []
    c = _cache(resolve=lambda root, revs, reads, frame: seen.append(root) or [r.id for r in revs])
    priv = Ed25519PrivateKey.generate()
    c.put([("doc-A", b"first", _vec(0))], authority="me", priv=priv, version=1)
    c.search(_vec(0), k=5)

    r = c.summary()["resolution"]
    assert seen, "the resolver was never consulted at all"
    assert r["roots_read"] >= 1
    assert r["multi_revision"] == 0, "a single-version root was counted as a resolution"
    assert r["grounded_out"] == 0 and r["narrowed"] == 0


def test_GROUNDING_OUT_and_NARROWING_are_counted_apart(monkeypatch):
    """The two outcomes must be distinguishable: a resolver that measured and let everything stand
    is not the same event as one that changed what a query saw. "Grounded out" is not a refusal —
    nothing was rejected, and the read had nowhere to stand, so every revision still answers. A
    single counter for both would make a resolver that never narrows anything indistinguishable
    from one that always does.
    """
    priv = Ed25519PrivateKey.generate()

    # (a) a resolver that lets everything stand
    ground = _cache(resolve=lambda root, revs, reads, frame: [r.id for r in revs])
    ground.put([("doc-1", b"v1", _vec(0))], authority="me", priv=priv, version=1)
    ground.revise("doc-1", b"v2", _vec(0), provenance=Provenance.HYPOTHESIS,
                  authority="me", priv=priv)
    ground.search(_vec(0), k=5)
    g = ground.summary()["resolution"]
    assert g["multi_revision"] >= 1
    assert g["grounded_out"] >= 1 and g["narrowed"] == 0

    # (b) a resolver that narrows to one
    narrow = _cache(resolve=lambda root, revs, reads, frame: [revs[0].id])
    narrow.put([("doc-1", b"v1", _vec(0))], authority="me", priv=priv, version=1)
    narrow.revise("doc-1", b"v2", _vec(0), provenance=Provenance.HYPOTHESIS,
                  authority="me", priv=priv)
    narrow.search(_vec(0), k=5)
    n = narrow.summary()["resolution"]
    assert n["multi_revision"] >= 1
    assert n["narrowed"] >= 1 and n["grounded_out"] == 0


def test_the_stat_reports_the_resolver_as_WIRED_when_one_is_injected():
    """The positive half of the unwired test above: "none" must not be what the field always says,
    or a hard-coded `resolver: "none"` would pass the unwired test forever while making a wired
    node indistinguishable from an unwired one in the published stat.
    """
    c = _cache(resolve=lambda root, revs, reads, frame: [r.id for r in revs])
    assert c.summary()["resolution"]["resolver"] == "wired"
