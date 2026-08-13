"""The AnchorSet a store's cells were written under is recorded, and a later set is checked.

Every failure mode of a swapped set is silent: an anchor id IS a cluster id, so changing the set
changes every address the index is written at, and the queries that then miss still answer 200.
These tests hold the check and, just as importantly, the line it draws: a wider set within one
space warns, a different space refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantle.search.anchors import store as anchor_store
from mantle.search.anchors.anchorset import AnchorSet, anchorset_fingerprint
from mantle.search.anchors.repo import InMemoryAnchorRepo
from mantle.search.anchors.store import AnchorSetDiverged

MODEL = "hf:test@1.0"


def _set(n: int, dim: int = 8, model: str = MODEL, seed: int = 1) -> AnchorSet:
    rng = np.random.default_rng(seed)
    s = AnchorSet(model, dim)
    for i in range(n):
        s.add_text(f"a{i}", rng.standard_normal(dim))
    return s


def _install(aset: AnchorSet) -> None:
    repo = InMemoryAnchorRepo()
    repo.bulk_add(aset.anchors)
    anchor_store.set_anchor_repo(repo)


@pytest.fixture(autouse=True)
def _clean():
    anchor_store._warned_geometry = None
    yield
    anchor_store.set_anchor_repo(None)
    anchor_store._warned_geometry = None


# ── the fingerprint itself ──────────────────────────────────────────────────────────────────────

def test_the_fingerprint_is_the_sets_identity_and_ignores_insertion_order():
    a = _set(6)
    shuffled = AnchorSet(a.model_id, a.dim)
    for anchor in reversed(a.anchors):
        shuffled.add(anchor)
    assert anchorset_fingerprint(shuffled) == anchorset_fingerprint(a)
    assert anchorset_fingerprint(_set(7)) != anchorset_fingerprint(a)


def test_live_fingerprint_tracks_the_installed_set():
    assert anchor_store.live_fingerprint() is None      # nothing provisioned
    a = _set(5)
    _install(a)
    assert anchor_store.live_fingerprint() == anchorset_fingerprint(a)


# ── the gate ────────────────────────────────────────────────────────────────────────────────────

def test_an_unindexed_store_is_not_a_mismatch(monkeypatch):
    monkeypatch.setattr(anchor_store, "indexed_geometry", lambda: None)
    _install(_set(5))
    anchor_store.require_live_anchorset()               # must not raise


def test_the_same_set_passes(monkeypatch):
    a = _set(5)
    _install(a)
    monkeypatch.setattr(anchor_store, "indexed_geometry", lambda: {
        "fingerprint": anchorset_fingerprint(a), "model_id": MODEL, "dim": 8, "anchors": 5})
    anchor_store.require_live_anchorset()


def test_growth_within_one_space_warns_and_still_serves(monkeypatch, caplog):
    """A client seeds a wider set in the same space, so the fingerprint moves. Refusing here
    would take the arm down for cells that are still perfectly readable."""
    was = _set(5)
    _install(_set(9))                                   # same model + dim, more anchors
    monkeypatch.setattr(anchor_store, "indexed_geometry", lambda: {
        "fingerprint": anchorset_fingerprint(was), "model_id": MODEL, "dim": 8, "anchors": 5})

    with caplog.at_level("WARNING"):
        assert len(anchor_store.require_live_anchorset()) == 9
    assert "REINDEX" in caplog.text

    caplog.clear()
    with caplog.at_level("WARNING"):
        anchor_store.require_live_anchorset()
    assert caplog.text == "", "the warning must fire once per process, not once per call"


def test_a_different_width_refuses(monkeypatch):
    was = _set(5, dim=8)
    _install(_set(5, dim=32))
    monkeypatch.setattr(anchor_store, "indexed_geometry", lambda: {
        "fingerprint": anchorset_fingerprint(was), "model_id": MODEL, "dim": 8, "anchors": 5})
    with pytest.raises(AnchorSetDiverged) as e:
        anchor_store.require_live_anchorset()
    assert "no in-place re-cell" in str(e.value)


def test_a_different_model_refuses(monkeypatch):
    was = _set(5, model="hf:other@2.0")
    _install(_set(5))
    monkeypatch.setattr(anchor_store, "indexed_geometry", lambda: {
        "fingerprint": anchorset_fingerprint(was), "model_id": "hf:other@2.0",
        "dim": 8, "anchors": 5})
    with pytest.raises(AnchorSetDiverged):
        anchor_store.require_live_anchorset()


def test_the_indexer_records_the_geometry_it_writes_under(monkeypatch):
    """The record is claimed on the cell-write path — never on a read, or a query could make a
    store look indexed under whatever set happened to be loaded."""
    from cryptography.fernet import Fernet

    from mantle.search.mantle.indexer import MantleIndexer
    from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService
    from mantle.search.mantle.stores import InMemoryCellStore
    from .helpers import SelfContextVerifier, self_request

    a = _set(5)
    _install(a)
    recorded: list = []
    monkeypatch.setattr(anchor_store, "indexed_geometry", lambda: None)
    monkeypatch.setattr(anchor_store, "record_indexed_geometry", recorded.append)

    idx = MantleIndexer(
        OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())),
                      grant_verifier=SelfContextVerifier()),
        InMemoryCellStore(),
    )
    idx.index_artifact("p", "c", [{"artifact_id": "x", "chunk_id": "x0",
                                   "embedding": a.anchors[0].embedding.astype(float).tolist()}],
                       self_request("p", "update"))
    assert [anchorset_fingerprint(s) for s in recorded] == [anchorset_fingerprint(a)]
