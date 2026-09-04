"""The beacon cut is on the recall path — where the vector arm's result set stops.

``search/beacon/`` had no production importer: the cut was written, tested against its
oracle, and never asked anything. The live vector arm stopped instead at a z-score over a
noise floor with ``z = 3.5`` and a floor of five results, which is the tuned-constant
approach the reduced instrument exists to replace.

Grouped by the question each group answers:

- ``TestTheCutIsTaken`` — does the arm actually cut, and is the cut derived from the data?
- ``TestTheCutIsSelectable`` — can an operator turn it off, and is off honest?
- ``TestTheCutNeverCostsTheAnswer`` — does a cut that cannot be taken degrade to the horizon?
- ``TestTheCutDoesNotDragPrismIn`` — does putting beacon on the path add an install requirement?
- ``TestWhyNotTheFusedList`` — the reason the cut is here and not after RRF, measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from mantle.search.mantle.engine import CUT_BEACON, CUT_NONE, MantleQueryEngine
from mantle.search.mantle.indexer import MantleIndexer
from mantle.search.mantle.stores import InMemoryCellStore
from .helpers import make_oracle, req, self_request

DIM = 32


@pytest.fixture(autouse=True)
def _live_anchorset():
    from mantle.search.anchors import store
    from mantle.search.anchors.anchorset import AnchorSet
    from mantle.search.anchors.repo import InMemoryAnchorRepo

    store.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", DIM)
    aset.add_text("anchor-0", np.ones(DIM, dtype=np.float32))
    store.save_live_anchorset(aset)
    yield
    store.set_anchor_repo(None)


def _unit(v) -> list[float]:
    v = np.asarray(v, dtype=np.float64)
    return (v / np.linalg.norm(v)).tolist()


def _topic_corpus(n_on: int = 12, n_off: int = 28, seed: int = 5):
    """A retrieval-shaped candidate set: a coherent group around one direction, and a
    diffuse remainder. The query is the direction, and is not any stored vector — an
    exact-match query is a degenerate spectrum, not a search."""
    rng = np.random.default_rng(seed)
    topic = rng.standard_normal(DIM)
    topic /= np.linalg.norm(topic)
    chunks = []
    for i in range(n_on):
        chunks.append({
            "artifact_id": f"on-{i}", "chunk_id": 0,
            "embedding": _unit(topic + 0.25 * rng.standard_normal(DIM)),
        })
    for i in range(n_off):
        chunks.append({
            "artifact_id": f"off-{i}", "chunk_id": 0,
            "embedding": _unit(rng.standard_normal(DIM)),
        })
    return topic.tolist(), chunks


def _stack(cut: str):
    oracle = make_oracle()
    cells = InMemoryCellStore()
    return (
        MantleIndexer(oracle, cells),
        MantleQueryEngine(oracle, cells, cut=cut),
    )


def _seeded(cut: str, chunks):
    indexer, engine = _stack(cut)
    indexer.index_artifact(
        "owner-A", "col-1", chunks, self_request("owner-A", "update"),
    )
    return engine


class TestTheCutIsTaken:
    def test_the_arm_returns_fewer_than_the_horizon(self):
        """The horizon is what the caller will look at; the cut is how much of it belongs
        together. If the arm still returned exactly `top_k`, the constant would still be
        deciding and beacon would be decoration."""
        query, chunks = _topic_corpus()
        engine = _seeded(CUT_BEACON, chunks)
        hits = engine.search(query, [("owner-A", "col-1")], req(), top_k=len(chunks))
        assert 0 < len(hits) < len(chunks)

    def test_the_cut_keeps_the_coherent_group_and_drops_the_diffuse_tail(self):
        """What the screen is reading: the group that shares the query's structure. A cut
        that kept off-topic vectors while dropping on-topic ones would be reading noise."""
        query, chunks = _topic_corpus()
        engine = _seeded(CUT_BEACON, chunks)
        hits = engine.search(query, [("owner-A", "col-1")], req(), top_k=len(chunks))
        kept = {h.artifact_id for h in hits}
        assert all(a.startswith("on-") for a in kept)

    def test_the_kept_hits_stay_in_score_order(self):
        """Beacon decides membership; cosine decides rank. Reordering here would make the
        arm's contribution to RRF a function of the screen rather than of similarity."""
        query, chunks = _topic_corpus()
        engine = _seeded(CUT_BEACON, chunks)
        hits = engine.search(query, [("owner-A", "col-1")], req(), top_k=len(chunks))
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_the_cut_size_is_read_from_the_data_and_not_fixed(self):
        """The property that distinguishes a derived cut from a tuned one: two corpora with
        different structure must produce different result counts for the same call."""
        counts = set()
        for n_on, seed in ((6, 1), (12, 5), (20, 9)):
            query, chunks = _topic_corpus(n_on=n_on, n_off=40 - n_on, seed=seed)
            engine = _seeded(CUT_BEACON, chunks)
            counts.add(len(engine.search(
                query, [("owner-A", "col-1")], req(), top_k=len(chunks),
            )))
        assert len(counts) > 1, f"the cut returned the same size every time: {counts}"

    def test_the_callers_top_k_is_still_a_ceiling(self):
        """The cut narrows the horizon; it never widens it. A cut that could return more
        than the caller asked for would break every paginating consumer."""
        query, chunks = _topic_corpus()
        engine = _seeded(CUT_BEACON, chunks)
        assert len(engine.search(
            query, [("owner-A", "col-1")], req(), top_k=3,
        )) <= 3


class TestTheCutIsSelectable:
    def test_none_returns_the_whole_horizon(self):
        query, chunks = _topic_corpus()
        engine = _seeded(CUT_NONE, chunks)
        hits = engine.search(query, [("owner-A", "col-1")], req(), top_k=len(chunks))
        assert len(hits) == len(chunks)

    def test_the_env_var_selects_it(self, monkeypatch):
        monkeypatch.setenv("MANTLE_SEARCH_CUT", "none")
        assert MantleQueryEngine(make_oracle(), InMemoryCellStore())._cut == CUT_NONE

    def test_the_default_is_the_beacon_cut(self, monkeypatch):
        """Default ON is the point. A derived cut nobody enables is the same dark code the
        tuned constant was standing in for."""
        monkeypatch.delenv("MANTLE_SEARCH_CUT", raising=False)
        assert MantleQueryEngine(make_oracle(), InMemoryCellStore())._cut == CUT_BEACON

    def test_an_unknown_value_falls_back_to_the_cut_rather_than_guessing(self, monkeypatch):
        monkeypatch.setenv("MANTLE_SEARCH_CUT", "sometimes")
        assert MantleQueryEngine(make_oracle(), InMemoryCellStore())._cut == CUT_BEACON


class TestTheCutNeverCostsTheAnswer:
    def test_a_cut_that_raises_degrades_to_the_uncut_horizon(self, monkeypatch):
        """A cut is a refinement of an answer that already exists. Failing to take one must
        cost the refinement, never the results."""
        import mantle.search.beacon.cut as beacon_cut

        query, chunks = _topic_corpus()
        engine = _seeded(CUT_BEACON, chunks)
        monkeypatch.setattr(
            beacon_cut, "select",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no reading")),
        )
        hits = engine.search(query, [("owner-A", "col-1")], req(), top_k=len(chunks))
        assert len(hits) == len(chunks)

    @pytest.mark.parametrize("n", [1, 2])
    def test_a_pool_too_small_to_have_a_spectrum_is_returned_whole(self, n):
        """Two candidates have no spectrum. Reporting a cut there would be reporting a
        reading that never happened."""
        query, chunks = _topic_corpus()
        engine = _seeded(CUT_BEACON, chunks[:n])
        assert len(engine.search(
            query, [("owner-A", "col-1")], req(), top_k=10,
        )) == n


class TestTheCutDoesNotDragPrismIn:
    def test_the_cut_imports_without_prism(self, monkeypatch):
        """`beacon/instrument.py` imports `prism.rounding`. The cut is a different module and
        reaches only `beacon.engine`, so putting it on the recall path adds no install
        requirement — measured here by removing prism from the import system and importing the
        path the arm actually takes.
        """
        import builtins
        import sys

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "prism" or name.startswith("prism."):
                raise ImportError(f"prism is blocked for this test ({name})")
            return real_import(name, *args, **kwargs)

        for mod in [m for m in sys.modules if m.startswith("mantle.search.beacon")]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        for mod in [m for m in sys.modules if m == "prism" or m.startswith("prism.")]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setattr(builtins, "__import__", _blocked)

        # The control: the module that DOES take the prism edge must still fail, or the
        # blocker is not blocking and the assertion below proves nothing.
        with pytest.raises(ImportError):
            real_import("mantle.search.beacon.instrument")

        import importlib
        assert importlib.import_module("mantle.search.beacon.cut").select is not None


class TestWhyNotTheFusedList:
    def test_an_rrf_spectrum_has_no_silhouette_to_read(self):
        """The measured reason the cut sits on the arm, kept after fusion was removed.

        There is no fused list to point the cut at any more — MANTLE has one ranker, and the
        RRF implementation is deleted — so the hazard this measures is one a FUTURE fusion
        would walk back into rather than one presently reachable. It is kept because the
        argument is not obvious and the formula is three lines, so the spectrum is built here
        from the definition instead of imported from code that no longer exists.

        An RRF score is `Σ 1/(k + rank)`, so a single arm's consecutive ratios are
        `(k+r+1)/(k+r)` — strictly decreasing, largest at the very first pair. `gap_split`
        therefore always cuts after ONE item, no matter what the underlying scores were:
        the gaps belong to `k`, not to the data.
        """
        from mantle.search.beacon.cut import gap_split

        def _rrf(scores, k=60):
            """One arm's RRF spectrum: rank the scores, then score by `1/(k + rank)`."""
            ranked = sorted(scores, reverse=True)
            return [1.0 / (k + rank) for rank, _ in enumerate(ranked, start=1)]

        for scores in ([9.0, 8.9, 8.8, 1.0, 0.9], [9.0, 1.0, 0.9, 0.8, 0.7]):
            keep, _gap = gap_split(_rrf(scores))
            assert int(keep.sum()) == 1, (
                "an RRF spectrum produced a break that depends on the data — if this ever "
                "holds, revisit whether a cut could belong after a fusion"
            )

        # The same two spectra, read directly, DO carry different breaks — which is what
        # makes the arm the right place to take the cut.
        assert int(gap_split([9.0, 8.9, 8.8, 1.0, 0.9])[0].sum()) == 3
        assert int(gap_split([9.0, 1.0, 0.9, 0.8, 0.7])[0].sum()) == 1
