"""BEACON conformance — the reduced screen read must keep answering the same question.

There are two implementations of one question ("how many independent directions does this
frame have"), and they cannot import each other: `beam/optics.py` is the full aperture and
depends on a local checkout of the upstream instrument; `mantle/search/beacon` is the reduced,
dependency-free engine that ships to corporate installs, and mantle must stay beam-free
(the target DAG is `mantle => origin` only).

Two independent implementations of one measurement can drift apart silently. This pins them
the same way `prism.vectors/contract_vectors.json` pins prism-py/js/c parity: a shared vector
file, regenerated from seeds so no matrix is stored and both sides build byte-identical
input. A change to either engine that moves a number in that file is a deliberate act and
must move the file in the same commit.

The case that matters most: a frame whose scale is carried by one channel — a collapsed
feature axis, where the spectrum read is not the spectrum of the data — must come back from
beacon as degraded, not as a confident `k=1` indistinguishable from a genuine one-mode
screen.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantle.search import beacon

# The shared vectors come from the installed prism package. Absence of the data must not be
# mistaken for the two engines agreeing: `prism.vectors.load_vectors` raises `MissingVectors` on a
# missing file, and this module lets that propagate rather than skipping — a skipped module and a
# passing one look identical from outside, so a broken pin should error out of collection instead.
#
# This adds no new dependency edge: `mantle => prism` already exists (~28 modules import prism; it
# is declared under the `service` extra) and prism is the Apache-2.0 floor that imports nothing
# from this workspace. `agience-prism-py` is named in the `dev` extra so this test can reach the
# vectors directly. `mantle => beam` remains the edge the target DAG forbids, which is why the
# shared vector data lives in prism rather than beam.
from prism.vectors import load_vectors


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
    dead = int(v.get("dead_cols", 0))
    if dead:
        # Drawn first, killed after: the live 120 x (F - dead) block is byte-identical to the same
        # case built without the dead columns, which is what makes the invariance check meaningful.
        m[:, v["F"] - dead:] = 0.0
    return m


# No `or {}`, no default, no `skipif`. Absent vectors raise out of collection and this module
# ERRORS — the only outcome that cannot be mistaken for agreement between the two engines.
_DOC = load_vectors("screen_read_vectors")
_VECTORS = _DOC["vectors"]
assert _VECTORS, (
    "screen_read_vectors.json carries no cases — every parametrised test below would collect zero "
    "cases and this gate would report success while comparing the two engines on nothing")


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
    """The case beam measures and beacon must not miss: a frame whose scale is carried by
    one channel. `whiten` zeroes channels with no spread, so the feature axis collapses and
    the spectrum read is not the spectrum of the data. It must not come back as a clean
    one-mode read."""
    case = next((v for v in _VECTORS if v.get("collapse")), None)
    assert case is not None, "the collapsed-axis vector must exist"
    read = beacon.signal_rank(_build(case))
    assert read.live_channels <= 1
    assert read.degraded is True


def test_a_dead_channel_is_dropped_from_the_shape_as_well_as_the_energy():
    """The `dead_channels` case is a frame measured at 10 channels and stored at 16.

    A dead channel is the absence of a measurement, not a measurement of zero. It carries no energy
    so it contributes no singular value — but left in the shape it still counts toward `n`, the
    modes available, and the occupancy is a fraction of that. The same frame then reads more
    coherent purely because of the stride it was written at: 0.255912 stored wide against 0.409459
    stored tight, a factor of 1.600.

    So the pin is an invariance before it is a number. Both reads are taken here — the stored frame
    and the tight one — and they must be equal to the bit, which is also what makes this read agree
    with the aperture's `phi`. The pinned constant is asserted alongside it, because an invariance
    alone would still hold if both sides moved together.
    """
    case = next((v for v in _VECTORS if v.get("dead_cols")), None)
    assert case is not None, "the dead-channel vector must exist; the invariance is unpinned without it"

    stored = _build(case)
    tight = stored[:, :case["F"] - case["dead_cols"]]
    assert np.count_nonzero(stored[:, case["F"] - case["dead_cols"]:]) == 0,         "the case did not actually build dead columns, so it proves nothing"

    # `occupancy_fraction` is engine-internal: it is not part of the beacon cut, and
    # `test_exported_surface_is_only_the_beacon_cut` below is what keeps it that way. A gate may
    # reach past the published surface; a caller may not.
    from mantle.search.beacon import engine as _engine

    wide = _engine.occupancy_fraction(stored)
    narrow = _engine.occupancy_fraction(tight)
    assert wide == narrow, (
        "occupancy moved %.9f -> %.9f between the same frame stored wide and stored tight, so a "
        "dead channel is still being counted in the denominator" % (wide, narrow))
    assert wide == case["beacon"]["occupancy"], (
        "occupancy pinned at %.17g, measured %.17g. Both engines read this constant; if the change "
        "is intended, regenerate screen_read_vectors.json in the SAME commit."
        % (case["beacon"]["occupancy"], wide))


def test_every_read_names_its_instrument():
    """A reduced-install number must never be mistakable for a full-aperture one."""
    read = beacon.signal_rank(np.random.default_rng(0).normal(size=(60, 8)))
    assert read.instrument == beacon.ENGINE_ID
    assert read.instrument, "a read with no instrument has no provenance"


def test_exported_surface_is_only_the_beacon_cut():
    """An export from beacon is a promise: the published surface is exactly the beacon cut and
    nothing else. Beacon is Apache-2.0 and public — the reduced instrument that makes a free store
    genuinely useful — so a published export is something a third party may build on and cannot
    change without breaking them, which makes a narrow surface more important here, not less.

    `structure_rank` was retired from this surface: zero production callers anywhere in mantle,
    and the correlated-row path it existed to serve reaches `_permutation_core` directly through
    `instrument.py` now, not through this public wrapper. `ENGINE_ID_PERM` stays exported — it is
    still how a caller distinguishes a permutation-null read from a Tracy-Widom one, and
    `instrument.py`'s reads still carry it.
    """
    assert set(beacon.__all__) == {
        "DEFAULT_FAR", "ENGINE_ID", "ENGINE_ID_PERM", "BeaconEngineError", "RankResult",
        "signal_rank",
    }
