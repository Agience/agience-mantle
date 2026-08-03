"""Activate verb — the geometry core (activate_vector) + POST /artifacts/activate."""

import numpy as np
import pytest

from mantle.search.anchors import AnchorSet, store
from mantle.search.anchors.activate import activate_vector
from mantle.search.anchors.anchorset import l2norm
from mantle.search.anchors.repo import InMemoryAnchorRepo

D = 8


@pytest.fixture(autouse=True)
def _restore_repo():
    """Restore the default (the lattice) AnchorRepo after each test."""
    yield
    store.set_anchor_repo(None)


def _live_anchorset():
    store.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", D)
    base = l2norm(np.arange(1, D + 1, dtype=np.float32))
    aset.add_text("alpha", base)
    aset.add_text("beta", l2norm(np.ones(D, dtype=np.float32)))
    aset.add_text("gamma", l2norm(np.eye(D)[3]))
    store.save_live_anchorset(aset)
    return base


def test_activate_vector_geometry():
    base = _live_anchorset()
    a = activate_vector(l2norm(base + 0.01 * np.ones(D, dtype=np.float32)), top_anchors=2)
    assert a["model_id"] == "hf:test@1.0"
    assert a["anchors"][0]["label"] == "alpha"        # nearest grounded concept
    assert a["density"] is not None and a["density"]["layer"] in ("L0", "L1", "L2")
    assert isinstance(a["novel"], bool)


def test_activate_vector_no_anchorset():
    store.set_anchor_repo(InMemoryAnchorRepo())       # empty → no live AnchorSet
    a = activate_vector([0.1] * D)
    assert a == {"model_id": None, "anchors": [], "density": None, "novel": False}


@pytest.mark.asyncio
async def test_activate_route_is_gone(client):
    """FAILURE MODE: before 2026-07-30 this route returned 200, took a caller-supplied
    `embedding` + `model_id` as the carrier, and echoed the raw vector back — an
    embed/score service and a BYOK vector interface in one. Removed entirely rather than
    501, per the standing ruling on `/coherence` and `/embed`. `activate_vector` itself
    (the geometry core, tested above) is untouched; only the HTTP surface is gone."""
    resp = await client.post(
        "/artifacts/activate",
        json={"embedding": [0.1] * D, "act": False, "top_anchors": 2},
    )
    assert resp.status_code in (404, 405)
