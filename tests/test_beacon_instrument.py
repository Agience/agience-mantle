# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# Beacon is the permissive half of the two-tier model, deliberately: mantle ships
# Apache so a store can be taken, built on and shipped by anyone, and beacon is the
# reduced instrument that makes such a store genuinely useful on its own.
# ---------------------------------------------------------------------------
"""Beacon fills the `Read` contract, deliberately leaves `Dynamics` and `Instrument` unfilled, and
diverges from the aperture in ways that are correct rather than fixable.

`ARCHITECTURE-TARGET.md` §3 states that the same crystal runs on a full node (the aperture) and on
a constrained store (beacon). These are the checks that fail if beacon stops being a complete,
honest embodiment of the one thing it does.

Beacon does the cut only — the projection and prediction flows live on the aperture. Section 1
asserts that per member, with a control that everything beacon does keep still works: without the
control, an implementation that raises for every member would satisfy the same assertion.

Failure modes, stated before the assertions:

  1. **A member is missing and `isinstance` says otherwise.** An implementation that fills exactly
     the declared members can pass `isinstance` and still raise at the first routed hop.
     `test_the_module_is_a_read_and_only_a_read` enumerates every member through
     `prism.instrument.require`, which is the door every real caller goes through.
  2. **`Dynamics` or `Instrument` gets filled.** Neither is owed, and for different reasons — a set
     has no lag; projection is the aperture's half of the product. Sections 1 and 8 assert both
     absences as gates, not as comments.
  3. **A divergence from the aperture gets "fixed" into agreement.** Two engines agreeing is not
     evidence either is correct, and two disagreeing is not evidence either is wrong; both need the
     domain argument. The divergences the header of `beacon/instrument.py` sets out are asserted
     here so a later change that quietly harmonises one is loud.
  4. **The embodiment surface grows a second rank rule.** `read_ordered` and the engine's own
     `signal_rank` / `_permutation_core` must resolve identically on every frame — a second
     implementation is how a module surface and an embodiment start naming different ranks while
     both look right.
  5. **Beacon acquires a heavy dependency.** It exists to keep mantle shippable while entroptics
     stays private. `test_the_instrument_imports_with_beam_entroptics_and_prism_blocked` blocks all
     three at the import system, control first.
  6. **A tolerance gets typed in.** Every level here descends from `engine.DEFAULT_FAR` or from the
     frame's own dtype; `test_no_level_survives_a_change_of_far` and the `_float_noise` checks make
     a hard-coded number fail.

What these checks cannot show: the frames below come from the same synthetic generator as
`screen_read_vectors.json` — `numpy.default_rng` with planted rank-1 modes. A green run proves the
reads agree with their definitions and with each other. It does not prove either engine is correct
on a real corpus.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from mantle.search.beacon import engine
from mantle.search.beacon import instrument as beacon_instrument
from mantle.search.beacon.instrument import _admits, _float_noise, _smallest_readable_rows
from prism.instrument import (DYNAMICS_MEMBERS, INSTRUMENT_MEMBERS, READ_MEMBERS, Dynamics,
                              Instrument, InstrumentRequired, Read, members_of, require)
from prism.vectors import load_vectors

# ── the frames: the SAME recipe the parity gate builds from ──────────────────────────────────
# Rebuilt from the shared vector file rather than invented here, so the divergences pinned below
# are pinned on the frames the two engines are already held to.
_VECTORS = load_vectors("screen_read_vectors")["vectors"]
assert _VECTORS, "screen_read_vectors.json carries no cases; every check below would be vacuous"


def _build(v: dict) -> np.ndarray:
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


_READABLE = [v for v in _VECTORS if not v.get("collapse")]
_IDS = [v["name"] for v in _VECTORS]


def _frame(planted: int = 3, *, N: int = 120, F: int = 16, snr: float = 60.0,
           seed: int = 13) -> np.ndarray:
    return _build({"seed": seed, "N": N, "F": F, "planted": planted, "snr": snr})


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · the contracts — filled, and one deliberately not
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_module_is_a_read_and_only_a_read() -> None:
    """An implementation can pass `isinstance` while a required member is missing, and then raise
    an `AttributeError` from inside a routed hop rather than at the door.

    Every member is fetched through `prism.instrument.require`, which is the one door a real caller
    uses, so a name that exists but is not callable fails here rather than at the measurement.

    The two negatives are asserted beside the positive on purpose. `isinstance(beacon, Instrument)`
    being False is what a host reads to know this store cannot project, and it is only trustworthy
    if it moves with the module — a stub that raised would make it True again."""
    assert isinstance(beacon_instrument, Read)
    assert not isinstance(beacon_instrument, Instrument)
    assert not isinstance(beacon_instrument, Dynamics)
    assert set(members_of(beacon_instrument, "read")) == set(READ_MEMBERS)
    assert members_of(beacon_instrument, "embodiment") == ()
    assert members_of(beacon_instrument, "dynamics") == ()
    for member in READ_MEMBERS:
        assert callable(require(beacon_instrument, member, contract="read", at="test"))


@pytest.mark.parametrize("member", INSTRUMENT_MEMBERS)
def test_every_instrument_member_refuses_by_name_and_the_read_still_works(member: str) -> None:
    """The removal of a member produces a named result, not a bare absence, with the control that
    makes it mean something.

    A beacon-backed crystal asking to `condense` goes through `prism.instrument.require`, and what
    it gets back is `InstrumentRequired`, carrying the contract, the member and the operation — a
    503 that says "this host is not equipped", never `None`, never a fabricated split, never an
    `AttributeError` from somewhere inside the flow.

    The control is the whole test. An implementation that raises `InstrumentRequired` for every
    member would also produce a named result for a missing one, so each case also proves that a
    `Read` member still resolves to a working callable through the same door — a partly-filled
    embodiment does what it can and names only what it cannot.

    The message must also name what would fill the slot: one that says only "missing" tells a host
    it is broken; one that names `ember.optics` tells it what to inject."""
    with pytest.raises(InstrumentRequired) as caught:
        require(beacon_instrument, member, contract="embodiment", at="crystal.condense")
    exc = caught.value
    assert exc.member == member and exc.contract == "embodiment"
    assert exc.at == "crystal.condense"
    assert exc.http_status == 503
    assert member in str(exc), "the refusal does not name the member a caller must discriminate on"

    # ── the control: everything beacon does fill still resolves and still measures ───────────────
    read = require(beacon_instrument, "read_ordered", contract="read", at="crystal.condense")
    assert callable(read)
    assert read(_frame(3)).k_signal >= 1, (
        "the `Read` half stopped working, so the refusal above proves only that this module "
        "refuses everything")


def test_the_two_unfilled_contracts_are_unfilled_for_different_reasons() -> None:
    """Both absences are statements, and they are not the same statement — asserted together so a
    reader cannot collapse them, because a reader who collapses them will eventually "fix" one.

    `Dynamics` — a question beacon's domain does not pose. Every member is about lag (`decay_profile`
    is C(τ), `embed` is a Takens delay, `fit_dynamics` is Koopman/DMD over consecutive frames) and a
    set of vectors has no lag. Filling these would mean fitting an operator to an axis whose order
    carries no information, and a fit always returns something: that is how noise gets reported as
    deterministic.

    `Instrument` — a question beacon's domain poses perfectly well, answered elsewhere in the
    product. The aperture projects a corpus matrix as readily as a signal frame; what decides it is
    that mantle is the Apache giveaway and the entroptics wrapper is the AGPL product.
    `absorb_transmit`'s own docstring names it: "the coupled band is the frame's projection onto
    the resolved subspace."

    Fails if somebody fills either one — most likely `Instrument`, since it is fillable and the
    argument against it is not about arithmetic."""
    for member in DYNAMICS_MEMBERS:
        assert not hasattr(beacon_instrument, member), (
            f"beacon grew `{member}`. A set of vectors has no lag; if this frame really does have "
            "an ordered axis, it is a signal and belongs on the aperture.")
    for member in INSTRUMENT_MEMBERS:
        assert not hasattr(beacon_instrument, member), (
            f"beacon grew `{member}`. Beacon does the CUT; projection is the aperture's half of "
            "the two-tier product, and a reduced instrument that quietly grows it is the AGPL "
            "wrapper given away by accident.")
    assert not hasattr(beacon_instrument, "MembraneScreen"), (
        "the membrane's object model came back. It is placement, coupling and rendering — the "
        "projection flow — and `prism.instrument.Screen` describes it for the aperture to fill.")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · one rank rule — the embodiment surface may not disagree with the engine
# ═════════════════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("v", _VECTORS, ids=_IDS)
def test_read_ordered_resolves_exactly_what_signal_rank_does(v: dict) -> None:
    """`read_ordered` and the engine call one extracted core (`engine._tw1_core`) rather than two
    separate Tracy-Widom implementations that could start naming different ranks while both look
    right — the same rule that makes `route_by_coupling` call `next_by_coupling` rather than
    re-implement the selection."""
    frame = _build(v)
    read = beacon_instrument.read_ordered(frame)
    if not read.screened:
        # An unreadable frame is reported rather than ranked, and the two surfaces are asserted to
        # agree that it is unreadable — otherwise this case would collect green while comparing
        # nothing.
        assert beacon_instrument.resolvable(frame) is None
        assert engine.signal_rank(frame).degraded is read.degraded
        return
    assert read.k_signal == engine.signal_rank(frame).k
    assert read.live_channels == engine.signal_rank(frame).live_channels
    assert read.degraded == engine.signal_rank(frame).degraded


def _permutation_k(M) -> int:
    """The retired `engine.structure_rank`'s own wrapping of `_permutation_core` — kept local
    since the public wrapper had zero production callers and was retired, while the underlying
    computation this test pins agreement against is unchanged and still live (`instrument.py`
    reaches `_permutation_core` directly on the correlated-rows path)."""
    core = engine._permutation_core(M)
    if not core.readable:
        return 1
    return max(1, core.k_tested + core.offset)


@pytest.mark.parametrize("v", _READABLE, ids=[v["name"] for v in _READABLE])
def test_the_permutation_path_resolves_exactly_what_structure_rank_did(v: dict) -> None:
    """Pins the same rule on the correlated-rows path — where it matters more, because the
    permutation rule carries a leading-run ordering and a mean-direction offset, and a
    re-implementation would drop one of them silently."""
    frame = _build(v)
    read = beacon_instrument.read_ordered(frame, null=beacon_instrument.correlated_null())
    assert read.k_signal == _permutation_k(frame)


@pytest.mark.parametrize("v", _VECTORS, ids=_IDS)
def test_resolvable_is_read_ordered_projected(v: dict) -> None:
    """`resolvable` is a projection of `read_ordered`, not a third read of its own — the contract
    says "nothing can fill it without being able to take the read"."""
    frame = _build(v)
    read = beacon_instrument.read_ordered(frame)
    assert beacon_instrument.resolvable(frame) == (read.k_signal if read.screened else None)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · The divergences from the aperture — pinned so neither side is "fixed" into agreement
# ═════════════════════════════════════════════════════════════════════════════════════════════
# These divergences are not in `screen_read_vectors.json`, which lives in `agience-prism` and is
# out of this lane's territory, so the pins are here instead — weaker, because a pin one engine
# holds alone cannot stop the other drifting. What follows is the argument each divergence carries.

@pytest.mark.parametrize("v", _VECTORS, ids=_IDS)
def test_coherence_is_never_measured_and_is_never_zero(v: dict) -> None:
    """Divergence 1 — the aperture (`ember.optics`) reports a lag-1 coherence z-score; beacon
    reports `None`, always.

    Pins against publishing `0.0` to make the field look filled. `0.0` is a real reading —
    "measured, and there is no lag-1 correlation" — and beacon has not measured it and cannot. Its
    domain is a set of vectors: axis 0 carries no order, so there is no lag-1 pair to take a
    statistic over. This is the same fact as `Dynamics` being unfilled, seen through one field.

    Holds on both settings of `with_screen`. The keyword exists on the aperture to skip the half
    that measures coherence; beacon has no such half, so the field is `None` because it is
    unmeasurable, not because it was skipped."""
    frame = _build(v)
    assert beacon_instrument.read_ordered(frame, with_screen=True).coherence is None
    assert beacon_instrument.read_ordered(frame, with_screen=False).coherence is None
    assert beacon_instrument.read_ordered(
        frame, null=beacon_instrument.correlated_null()).coherence is None


@pytest.mark.parametrize("v", _READABLE, ids=[v["name"] for v in _READABLE])
def test_the_point_read_sits_at_the_conservative_end_of_its_own_interval(v: dict) -> None:
    """Divergence 2 — the certified interval models a different error than the aperture's, and the
    consequence is visible: `k_lo == k_signal` here by construction.

    The aperture's `[k_lo, k_hi]` propagates a Weyl band on `‖Ĉ − C‖₂` — the error in the sample
    covariance as an estimate of a population covariance — so its point read sits inside its
    interval. Beacon's domain is a corpus, which is the population; there is no population
    covariance being estimated, so that band has nothing to bound here. What is genuinely uncertain
    is where the noise edge fell on this draw, and beacon models that exactly: the interval is the
    count at the edge's upper quantile `q(far)` and at its lower one `q(1 − far)`.

    Pins against reading the two `k_certain` flags as the same statement and "reconciling" them.
    They are different measurements and comparing them is a category error."""
    frame = _build(v)
    read = beacon_instrument.read_ordered(frame)
    assert read.screened
    assert read.k_lo == read.k_signal, (
        "beacon's point read IS the conservative end of its interval — it is taken at the edge's "
        "upper quantile. If this moved, the interval and the count are no longer drawn from one "
        "null and `k_certain` means nothing."
    )
    assert read.k_lo <= read.k_signal <= read.k_hi


def test_the_readable_boundary_is_recovered_from_the_gate_not_typed() -> None:
    """Divergence 3 — beacon reads a frame at 2 rows; the aperture needs 4.

    Pins against typing a row minimum, or copying the aperture's. The aperture's `MIN_ROWS` is
    recovered from `T(T−1)(T−2)(T−3)` — the dof of the disjoint term of its lag-1 permutation
    variance. Beacon takes no lag-1 statistic (divergence 1), so it has no such term; its boundary
    is the engine's own `min(N, F_live) >= 2`, which is what `johnstone` needs to describe a
    non-degenerate ensemble. The divergence is the same fact as `Dynamics` being unfilled.

    Asserted against the gate rather than against a number, so a change to the gate moves this."""
    m = beacon_instrument.MIN_ROWS
    assert m == _smallest_readable_rows()
    assert _admits(m, m + 1) and not _admits(m - 1, m), (
        "MIN_ROWS is not the boundary of `_admits` any more — it was typed, or the gate moved "
        "without it")
    rng = np.random.default_rng(3)
    assert beacon_instrument.read_ordered(rng.normal(size=(m, 8))).screened
    assert not beacon_instrument.read_ordered(rng.normal(size=(m - 1, 8))).screened


def test_an_unreadable_frame_defers_here_and_floors_at_one_in_the_engine() -> None:
    """Divergence 4's other half — two surfaces, one rule, different consumers.

    Pins against the two being "unified" so one consumer breaks silently. `engine.signal_rank`
    floors at `k = 1` on a degenerate frame because its caller projects onto the result and an
    empty subspace is worse than a coarse one. `resolvable` must return `None` there, because its
    caller can defer — an `adaptive_cut` that returns "keep all" instead turns the whole mode into
    a silent no-op reporting itself as a derived decision.

    Neither fabricates: both report the same frame, and `screened` / `degraded` is what tells them
    apart."""
    one_row = np.ones((1, 8))
    assert engine.signal_rank(one_row).k == 1
    assert beacon_instrument.resolvable(one_row) is None
    read = beacon_instrument.read_ordered(one_row)
    assert read.screened is False and read.k_signal == 0, (
        "k_signal past `screened=False` is NOT MEASURED, and 0 is how that is said. A 1 here would "
        "be a fabricated reading of a frame the instrument cannot read.")


@pytest.mark.parametrize("v", [v for v in _VECTORS if v.get("collapse")],
                         ids=[v["name"] for v in _VECTORS if v.get("collapse")])
def test_a_collapsed_axis_is_still_flagged_through_the_embodiment(v: dict) -> None:
    """The divergence already pinned in `screen_read_vectors.json` must survive the new surface.

    Pins against `read_ordered` dropping `degraded`, which would make a frame whose scale is
    carried by one channel indistinguishable from a genuine one-mode read. The aperture reports
    `scale_hazard=false` on this frame, and both are right: a dead channel is anomalous for a
    corpus and normal for a sparse ontology coordinate."""
    read = beacon_instrument.read_ordered(_build(v))
    assert read.degraded is True
    assert read.live_channels <= 1


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · The nulls — one decision, one place
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_null_carries_its_own_level_and_its_own_draw_count() -> None:
    """Pins against `far` reappearing as a keyword beside a null: `permutation(far=0.01)` could
    silently ignore a `far=0.05` passed alongside, so the read would report a level no cutoff was
    ever drawn at. One cutoff is one decision, so the provider owns both the threshold and the α
    it is drawn at."""
    default = beacon_instrument.correlated_null()
    assert default.far == engine.DEFAULT_FAR
    assert default.draws == engine._permutation_draws(engine.DEFAULT_FAR)
    sharp = beacon_instrument.correlated_null(far=0.01)
    assert sharp.far == 0.01 and sharp.draws == engine._permutation_draws(0.01)
    assert beacon_instrument.correlated_null(draws=200).draws == 200
    with pytest.raises(ValueError):
        beacon_instrument.correlated_null(far=1.5)
    with pytest.raises(ValueError):
        beacon_instrument.derived_null(far=0.0)


def test_no_level_survives_a_change_of_far() -> None:
    """Pins against a threshold typed somewhere inside the read, which would make sharpening `far`
    change nothing. Every level here descends from the one stated tolerance; a sharper level must
    raise the floor, and a raised floor can only ever resolve the same number of modes or fewer."""
    frame = _frame(3, snr=25.0)
    loose = beacon_instrument.read_ordered(frame, null=beacon_instrument.derived_null(far=0.20))
    tight = beacon_instrument.read_ordered(frame, null=beacon_instrument.derived_null(far=1e-4))
    assert tight.k_signal <= loose.k_signal
    assert tight.contrast < loose.contrast, (
        "the contrast is `s1 / floor`, so a sharper level must lower it. If it did not move, the "
        "floor was not drawn at the level that was asked for.")


def test_an_unknown_null_is_refused_rather_than_ignored() -> None:
    """An unrecognised provider does not fall back silently to a default: `read_ordered` raises
    rather than treating it as `None`, because a read cannot be taken against a cutoff whose level
    it cannot name."""
    with pytest.raises(engine.BeaconEngineError):
        beacon_instrument.read_ordered(_frame(1), null=object())


def test_the_permutation_interval_cannot_collapse_at_the_minimum_draw_budget() -> None:
    """The honest consequence of deriving `draws` as a minimum.

    At `B = 1/far − 1` the smallest attainable p-value is `far`, so every resolved component sits
    exactly on the level with nothing to spare: one surrogate the other way and it fails. The
    interval therefore cannot collapse, and `k_certain` is False — the correct report, not a
    defect. More draws buy resolution and the interval does collapse.

    Pins against making `k_certain` true at the minimum budget by widening `far` or by dropping the
    `±1/(1+B)` step. The band is a progress reading, and what closes it is more evidence, never a
    looser tolerance."""
    frame = _frame(3)
    minimum = beacon_instrument.read_ordered(frame, null=beacon_instrument.correlated_null())
    assert not minimum.k_certain
    richer = beacon_instrument.read_ordered(
        frame, null=beacon_instrument.correlated_null(draws=200))
    assert richer.k_certain, (
        "more draws did not tighten the interval, so the ±1/(1+B) step is not tracking the draw "
        "count and the interval is not measuring resolution")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 5 · The ordered-container contract
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_set_and_a_bare_iterator_are_refused() -> None:
    """Pins against a `set` being silently sequenced, or a single-shot iterator being consumed —
    either would mean the read cannot be shown the same input twice, so nobody can audit it.
    Beacon's domain is a set of vectors, so the row order carries no meaning to the read; the
    container must still have a reproducible one."""
    with pytest.raises(engine.BeaconEngineError):
        beacon_instrument.read_ordered({(1.0, 2.0), (3.0, 4.0)})
    with pytest.raises(engine.BeaconEngineError):
        beacon_instrument.read_ordered(iter([[1.0, 2.0], [3.0, 4.0]]))


def test_a_window_is_a_nested_subset_and_nothing_is_truncated_without_one() -> None:
    """Pins against a longer batch being quietly truncated to an adaptive window, which would make
    the read answer a different question than the one asked without the caller seeing it happen.
    `window=None` reads the whole frame; an explicit window reads the leading rows, which over a
    set of vectors is the one subset the caller can reproduce from the container it handed over."""
    frame = _frame(3, N=200)
    assert beacon_instrument.read_ordered(frame).n_rows == 200
    assert beacon_instrument.read_ordered(frame, window=50).n_rows == 50
    head = beacon_instrument.read_ordered(frame[:50])
    windowed = beacon_instrument.read_ordered(frame, window=50)
    assert (head.k_signal, head.contrast) == (windowed.k_signal, windowed.contrast)
    with pytest.raises(engine.BeaconEngineError):
        beacon_instrument.read_ordered(frame, window=0)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 6 · Scales — structure against how much of the corpus was looked at
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_scale_ladder_is_dyadic_nested_and_derived() -> None:
    """Pins against the ladder being a typed list of window sizes.

    It is successive halving down to the smallest readable frame — the coarsest non-trivial
    nesting, which makes it the minimum that says anything at all. A finer ladder resolves the
    trend better and costs one decomposition per extra rung, so it is compute and not modelling,
    and beacon does not guess a compute budget any more than it guesses a draw count. Each rung is
    a subset of the next, which is what makes the rungs comparable."""
    frame = _frame(3, N=120)
    profile = beacon_instrument.scales(frame)
    windows = [row["window"] for row in profile]
    assert windows == sorted(windows), "the profile must read ascending by aperture"
    assert windows[-1] == 120, "the widest rung must be the whole corpus"
    assert all(a == 2 * b or a == 2 * b + 1 for a, b in zip(windows[1:], windows[:-1])), (
        f"the ladder is not dyadic: {windows}")
    assert min(windows) >= beacon_instrument.MIN_ROWS
    assert all(row["n_rows"] == row["window"] for row in profile)


def test_scales_takes_the_callers_ladder_and_refuses_an_unreadable_frame() -> None:
    """Pins against an unreadable corpus coming back as an empty list, which would read as
    "measured, and there is no structure at any scale". `None` is the computed null, and it is a
    result."""
    frame = _frame(2, N=64)
    profile = beacon_instrument.scales(frame, windows=[16, 64])
    assert [row["window"] for row in profile] == [16, 64]
    assert beacon_instrument.scales(np.ones((1, 8))) is None


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 7 · The accumulator — a snapshot is not a stream
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_band_shrinks_as_evidence_accumulates() -> None:
    """Pins against a band that is constant, or that does not track `T` — either would stop it
    being a progress reading and leave loosening `far` as the only way to reach certification.

    The band is `(q(far) − q(1−far))·σ_J/μ`: the sampling scatter of the noise edge, relative to
    the edge's own centre, so it falls as `T^(−1/2)` as planes are pooled."""
    frame = _frame(3, N=240)
    acc = beacon_instrument.accumulator(16)
    bands = []
    for start in range(0, 240, 40):
        acc.add(frame[start:start + 40])
        bands.append(acc.band())
    assert all(b > a for a, b in zip(bands[1:], bands[:-1])), f"band did not fall: {bands}"
    assert bands[-1] < bands[0] / 2, (
        f"the band fell by less than the T^(-1/2) the model predicts over a 6x growth: {bands}")


def test_the_accumulated_read_reports_the_band_and_the_interval_beside_the_count() -> None:
    """Pins against reporting only `certified`, which would let a caller see that certification has
    not happened without seeing how far away it is — leaving loosening a tolerance as the only
    remaining way to make it happen. Reporting `band` and `interval` beside the count is required,
    not decorative."""
    frame = _frame(3, N=120)
    acc = beacon_instrument.accumulator(16)
    for start in range(0, 120, 20):
        acc.add(frame[start:start + 20])
    read = beacon_instrument.accumulated_read(acc)
    assert set(read) >= {"T", "F", "band", "k_signal", "interval", "certified"}
    assert read["T"] == 120 and read["F"] == 16 and read["planes"] == 6
    lo, hi = read["interval"]
    assert lo <= read["k_signal"] <= hi
    assert read["certified"] == (lo == hi == read["k_signal"])


def test_an_empty_accumulator_is_none_not_a_resolved_zero() -> None:
    """Pins against an accumulator that holds nothing reporting `k_signal = 0`, which would read as
    "measured, and nothing resolved". They are different statements and must not share a value."""
    assert beacon_instrument.accumulated_read(beacon_instrument.accumulator(8)) is None
    assert beacon_instrument.accumulated_read(None) is None


def test_merge_moves_only_the_pooled_covariance_and_refuses_a_width_mismatch() -> None:
    """Pins against `merge` needing the raw frames, which would force a peer contributing evidence
    to hand over its observations rather than just its pooled statistics. Only the pooled scatter
    and the column sums travel, and a merge across two coordinate systems does not fold into a
    spectrum of neither — it raises instead.

    Also pinned: `merge` returns a new accumulator. A merge that mutated its left operand would
    make "who has seen what" depend on call order."""
    frame = _frame(3, N=120)
    a, b = beacon_instrument.accumulator(16), beacon_instrument.accumulator(16)
    a.add(frame[:60])
    b.add(frame[60:])
    merged = a.merge(b)
    assert merged.T == 120 and merged.planes == 2
    assert a.T == 60 and b.T == 60, "merge mutated an operand"
    whole = beacon_instrument.accumulator(16)
    whole.add(frame[:60])
    whole.add(frame[60:])
    assert merged.spectral().resolved_modes == whole.spectral().resolved_modes
    with pytest.raises(engine.BeaconEngineError):
        a.merge(beacon_instrument.accumulator(8))
    with pytest.raises(engine.BeaconEngineError):
        a.add(np.ones((4, 8)))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 8 · The projection flow — beacon does not assert it, and that absence is the assertion
# ═════════════════════════════════════════════════════════════════════════════════════════════
# Beacon does not assert `absorb_transmit`, `next_by_coupling` or `MembraneScreen`: not that a
# split conserves energy exactly, that an unreadable frame propagates rather than being split,
# that routing picks the strongest coupling and terminates on its own, that the routing floor is
# derived rather than typed, that coupling is measured/bounded/symmetric, that certification is
# decided against a drawn null, or that a transfer balances 0 → 1 → 0. Those are projection —
# `ember/optics.py` holds the same arithmetic, and the aperture's own test suite in `agience-ember`
# holds the matching checks — which is the AGPL half of the product line. Beacon asserting a
# capability it does not ship would be the defect; the coverage of the arithmetic lives with the
# code that runs it.
#
# What beacon does assert here: the absence is a named result, per member, in section 1
# (`test_every_instrument_member_refuses_by_name_and_the_read_still_works`, with the control that
# the `Read` half still measures) and as a gate in
# `test_the_two_unfilled_contracts_are_unfilled_for_different_reasons`. `_float_noise` — the
# derived energy band those two members used — keeps its own sweep in
# `tests/test_rounding_law_is_single_sourced.py`, which is why it survives here too.


def test_the_routing_floor_is_still_derived_from_the_arithmetic_not_typed() -> None:
    """`_float_noise` is the one member of the old projection-flow surface that beacon still calls,
    so this is the one check from that surface that survives.

    Pins against `_float_noise` losing its last caller and becoming a constant nobody exercises,
    which would let it quietly stop tracking its inputs. A typed `1e-9` in this role runs
    backwards: most permissive on the smallest frames, where a spurious coupling is most likely.

    Error model: accumulation, not cancellation. Every term summed is a squared magnitude, hence
    non-negative, so partial sums increase monotonically and no catastrophic cancellation is
    possible. A term count would be the wrong model on a sum that can cancel."""
    small = np.ones((4, 16))
    large = np.ones((2048, 16))
    assert _float_noise(small, 1.0) < _float_noise(large, 1.0), (
        "the band does not grow with the number of products the norm performed")
    assert _float_noise(small, 100.0) == pytest.approx(100.0 * _float_noise(small, 1.0)), (
        "the band is not proportional to the energy it bounds, so it is not a relative bound")
    assert _float_noise(small.astype(np.float32), 1.0) > _float_noise(small, 1.0), (
        "a float32 frame must earn a wider band than a float64 one, from its own dtype")


def test_principal_directions_survives_the_removal_because_it_is_a_read() -> None:
    """`principal_directions` is a read, not projection, which is the line worth drawing: it says
    which directions the resolved modes span; `resolvable` says how many. One read, two
    projections of it — and `mantle.search.beacon.cut` is built on exactly that question. What
    beacon does not fill is the step that applies those directions as a projector to carry a
    signal through a membrane.

    Pins against the directional read being deleted as collateral of the projection absence, which
    would leave the cut without the basis it is built on."""
    frame = _frame(3)
    basis = beacon_instrument.principal_directions(frame)
    assert basis is not None
    assert basis.shape == (16, beacon_instrument.resolvable(frame))
    assert beacon_instrument.principal_directions(np.ones((1, 8))) is None, (
        "a frame that cannot carry a read must return the computed null, never a guessed basis")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 10 · The dependency floor — what makes beacon shippable at all
# ═════════════════════════════════════════════════════════════════════════════════════════════

_BLOCKED = ("beam", "entroptics", "prism")

#: The permitted prism modules, enumerated by exact name — narrower than a prefix match. What
#: beacon asks for is one module: `prism.rounding`, the single-sourced floating-point rounding
#: law, stdlib only, on prism's dependency-free base, no numpy and no extra. A derivation
#: duplicated elsewhere in this workspace drifted from the original in measurable cases, which is
#: why the rounding law is single-sourced here rather than reimplemented.
#:
#: The other four are measured, not chosen. Importing any submodule runs the package `__init__`
#: first, and `prism/__init__.py` eagerly imports `canonical`, `environment` and `errors`. The
#: full new-module closure of `import prism.rounding` is exactly:
#:
#:     prism · prism.canonical · prism.environment · prism.errors · prism.rounding
#:
#: all five on the dependency-free contract, all five held stdlib-only by prism's own
#: `tests/test_contract_install_is_pure.py`. Listing what is actually pulled rather than what was
#: asked for is the difference between a bounded edge and a claim about one. If prism's `__init__`
#: ever grows a heavier eager import, the probe below fails with the new module named — which makes
#: this list a live measurement of prism's own floor rather than a courtesy to it.
_PERMITTED = ("prism", "prism.canonical", "prism.environment", "prism.errors", "prism.rounding")

#: The control probes: modules that must still fail to import. `prism.conservation` is the pointed
#: one — it is where this same law is applied with numpy, the module a careless import would reach
#: for — and the other two prove the block covers the whole package minus the measured closure.
_MUST_REFUSE = ("beam", "entroptics", "prism.conservation", "prism.instrument", "prism.minting")

_PROBE = '''
import sys, json

for name in {blocked!r}:
    assert name not in sys.modules, (
        name + " was already imported before the blocker went up; a meta_path finder is only "
        "consulted for modules NOT in sys.modules, so this run would prove nothing")

MARK = "BLOCKED by the negative control"
PERMITTED = {permitted!r}


class Blocker:
    """Refuse the heavy edges at the import system. `find_spec`, because 3.12 ignores
    `find_module` — a finder that only defines the legacy hook is silently ignored and the whole
    check goes vacuous.

    `PERMITTED` is an EXACT-NAME allowance, never a prefix: `prism.rounding` passes and
    `prism.rounding_helpers`, `prism.conservation` and everything else under `prism` do not."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in PERMITTED:
            return None
        root = fullname.split(".")[0]
        if root in {blocked!r}:
            raise ImportError(MARK + ": " + fullname)
        return None


sys.meta_path.insert(0, Blocker())
v = {{}}

# ── 1. PROVE THE BLOCKER FIRES, before concluding anything from its silence ─────────────────
v["blocker_fires"] = {{}}
for name in {must_refuse!r}:
    try:
        __import__(name)
        v["blocker_fires"][name] = False
    except ImportError as exc:
        v["blocker_fires"][name] = MARK in str(exc)
    except BaseException as exc:
        v["blocker_fires"][name] = "raised " + type(exc).__name__

# ── 1b. AND THAT THE ONE PERMITTED MODULE IS GENUINELY REACHABLE AND STDLIB-FLOORED ─────────
# If this failed, section 2's success would be proving something else entirely.
try:
    from prism.rounding import accumulated_rounding, split_walk_rounding
    v["law_reachable"] = (accumulated_rounding(3, 2.0, 0.5) == 3.0
                          and split_walk_rounding(4, 1.0, 1.0, splits=1) == 16.0)
except BaseException as exc:
    v["law_reachable"] = False
    v["law_error"] = type(exc).__name__ + ": " + str(exc)

# ── 2. THE EMBODIMENT IMPORTS, AND DOES REAL WORK ──────────────────────────────────────────
try:
    import numpy as np
    import mantle.search.beacon.instrument as bi

    frame = np.random.default_rng(13).normal(size=(120, 16))
    for _ in range(3):
        u = np.random.default_rng(1).normal(size=120)
        w = np.random.default_rng(2).normal(size=16)
        frame = frame + 60.0 * np.outer(u / np.linalg.norm(u), w / np.linalg.norm(w))

    read = bi.read_ordered(frame)
    basis = bi.principal_directions(frame)
    acc = bi.accumulator(16)
    acc.add(frame)

    v["k_signal"] = int(read.k_signal)
    v["basis_shape"] = list(basis.shape)
    v["resolvable"] = int(bi.resolvable(frame))
    v["accumulated"] = bi.accumulated_read(acc)["k_signal"]
    v["scales"] = len(bi.scales(frame))
    v["works"] = True
except BaseException as exc:
    v["works"] = False
    v["error"] = type(exc).__name__ + ": " + str(exc)

v["leaked"] = sorted(n for n in sys.modules
                     if n.split(".")[0] in {blocked!r} and n not in PERMITTED)
print(json.dumps(v))
'''


def _probe() -> dict:
    prelude = f"import sys\nsys.path[:0] = {json.dumps(sys.path)}\n"
    proc = subprocess.run(
        [sys.executable, "-c", prelude + _PROBE.format(blocked=_BLOCKED, permitted=_PERMITTED,
                                                       must_refuse=_MUST_REFUSE)],
        capture_output=True, text=True, env=dict(os.environ), timeout=300)
    assert proc.returncode == 0, f"probe failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_blocker_actually_fires() -> None:
    """The precondition, promoted to a test. Everything below concludes from the absence of beam,
    entroptics and every prism module but one. If the blocker silently did nothing — the outcome
    for a finder written against the legacy `find_module` hook, which Python 3.12 ignores — those
    conclusions would be vacuous while the suite stayed green. A guard that cannot be shown to fire
    is indistinguishable from no guard.

    `prism.conservation` is in the control set on purpose. It is where the same rounding law is
    applied with numpy, so it is the module a careless import would reach for; if the allowance for
    `prism.rounding` ever widened into a prefix match, this is the probe that would go green and
    say so."""
    fires = _probe()["blocker_fires"]
    assert all(fires.values()), f"the blocker did not bite on every edge: {fires}"


def test_the_one_permitted_prism_module_is_reachable_and_needs_nothing() -> None:
    """The other side of the allowance, and it must fail if the law stops being free to reach.

    `prism.rounding` is imported with `beam`, `entroptics` and the rest of `prism` blocked, and
    then exercised — an import that resolves nothing proves only that a file parses. If this
    fails, the law has acquired a dependency and beacon would have to carry its own copy again."""
    v = _probe()
    assert v["law_reachable"], (
        f"prism.rounding was not reachable, or answered wrong, with everything else blocked: "
        f"{v.get('law_error')}")


def test_the_instrument_imports_with_beam_entroptics_and_prism_blocked() -> None:
    """Pins against beacon acquiring an edge to `beam`, `entroptics` or a prism module other than
    the one rounding law — directly, or transitively through something it imports — which would
    stop mantle being shippable while entroptics stays private. That is the entire reason this
    embodiment exists.

    Checked by removing them from the import system, not by reading import statements: a
    dependency claim checked by grep misses re-exports, lazy `__getattr__` paths that are lazy in
    name only, and anything a transitive module drags in.

    The allowance is one exact module name, narrower than a prefix match: `prism.rounding` is
    permitted by name and everything else under `prism` must still fail to import, with
    `prism.conservation` in the control set to prove it. Naming exactly one module and blocking
    the rest by measurement is stronger than blocking all of `prism` without saying which module
    would be acceptable if the question arose.

    The surface must work with them gone, not merely import — a split that moves the weight by
    breaking the feature is not a split. So the subprocess takes a read, reads a basis, projects it
    to a count, pools a plane and walks the scale ladder — every measurement beacon makes."""
    v = _probe()
    assert all(v["blocker_fires"].values()), "blocker did not fire; this result would mean nothing"
    assert v["works"], f"the embodiment broke without beam/entroptics/prism: {v.get('error')}"
    assert v["k_signal"] >= 1 and v["resolvable"] == v["k_signal"]
    assert v["basis_shape"] == [16, v["k_signal"]]
    assert v["accumulated"] >= 1 and v["scales"] >= 1
    assert not v["leaked"], f"a blocked package reached sys.modules anyway: {v['leaked']}"
