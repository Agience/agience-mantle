"""BEACON conformance — the reduced screen read must keep answering the same question.

There are two implementations of ONE question ("how many independent directions does this
frame have"), and they cannot import each other: `beam/optics.py` is the full aperture and
depends on a local entroptics checkout; `mantle/search/beacon` is the reduced,
dependency-free engine that ships to corporate installs, and mantle must stay beam-free
(the target DAG is `mantle => origin` only).

Two independent implementations of one measurement drift. This pins them the same way
`agience-beam/vectors/contract_vectors.json` pins prism-py/js/c parity: a shared vector
file, regenerated from seeds so no matrix is stored and both sides build byte-identical
input. A change to either engine that moves a number in that file is a DELIBERATE act and
must move the file in the same commit.

FAILURE MODE this exists to catch: the two engines silently disagreeing. Before 2026-07-30
`beacon` reproduced beam's MAD whitening but NOT its degradation flag, so a frame whose
scale is carried by one channel — a collapsed feature axis, where the spectrum read is not
the spectrum of the data — came back from beacon as a confident `k=1`, indistinguishable
from a genuine one-mode screen.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mantle.search import beacon


def _vectors_path() -> Path | None:
    """The shared vector file lives in agience-beam (one canonical copy, read as DATA).

    Reading a JSON file is not a code dependency, so the DAG is untouched. A standalone
    mantle checkout will not have the sibling repo — that skips, it does not fail.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "agience-beam" / "vectors" / "screen_read_vectors.json"
        if candidate.is_file():
            return candidate
    return None


def _build(v: dict) -> np.ndarray:
    """Rebuild a case's frame from its seed — the recipe is stated in the vector file."""
    rng = np.random.default_rng(v["seed"])
    if v.get("collapse"):
        m = np.zeros((v["N"], v["F"]))
        m[:, 0] = rng.normal(size=v["N"])
        return m
    m = rng.normal(size=(v["N"], v["F"]))
    for _ in range(v["planted"]):
        u = rng.normal(size=v["N"])
        w = rng.normal(size=v["F"])
        m += v["snr"] * np.outer(u / np.linalg.norm(u), w / np.linalg.norm(w))
    return m


_PATH = _vectors_path()
_DOC = json.loads(_PATH.read_text(encoding="utf-8")) if _PATH else {"vectors": []}
_VECTORS = _DOC.get("vectors", [])

pytestmark = pytest.mark.skipif(
    _PATH is None,
    reason="screen_read_vectors.json not found (standalone mantle checkout — no sibling agience-beam)",
)


@pytest.mark.parametrize("v", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_beacon_reproduces_the_pinned_read(v):
    read = beacon.signal_rank(_build(v))
    assert read.k == v["beacon"]["k"], (
        f"{v['name']}: beacon k moved {v['beacon']['k']} -> {read.k}. If that is intended, "
        "regenerate screen_read_vectors.json in the SAME commit."
    )
    assert read.live_channels == v["beacon"]["live_channels"]
    assert read.degraded == v["beacon"]["degraded"]


@pytest.mark.parametrize("v", _VECTORS, ids=[v["name"] for v in _VECTORS])
def test_planted_structure_is_recovered(v):
    """Ground truth, not just parity: where modes were planted well clear of the noise
    edge, the engine must find exactly that many. Agreeing with the other implementation
    is not enough if both are wrong."""
    if v["planted"] == 0 or v.get("collapse"):
        pytest.skip("no planted structure to recover")
    assert beacon.signal_rank(_build(v)).k == v["planted"]


def test_collapsed_axis_is_marked_degraded():
    """The case beam measured and beacon used to miss: a frame whose scale is carried by
    one channel. `whiten` zeroes channels with no spread, so the feature axis collapses and
    the spectrum read is not the spectrum of the data. It must NOT come back as a clean
    one-mode read."""
    case = next((v for v in _VECTORS if v.get("collapse")), None)
    assert case is not None, "the collapsed-axis vector must exist"
    read = beacon.signal_rank(_build(case))
    assert read.live_channels <= 1
    assert read.degraded is True


def test_every_read_names_its_instrument():
    """A reduced-install number must never be mistakable for a full-aperture one."""
    read = beacon.signal_rank(np.random.default_rng(0).normal(size=(60, 8)))
    assert read.instrument == beacon.ENGINE_ID
    assert read.instrument, "a read with no instrument has no provenance"


def test_exported_surface_is_only_the_beacon_cut():
    """This package is CONFIDENTIAL, so an export is DISCLOSED SURFACE, not just API.

    FAILURE MODE: it exported twelve symbols and exactly one pair was used outside the
    package. Anything added back here is trade-secret surface being published to every
    corporate install that imports it.
    """
    assert set(beacon.__all__) == {
        "DEFAULT_FAR", "ENGINE_ID", "BeaconEngineError", "RankResult", "signal_rank",
    }
