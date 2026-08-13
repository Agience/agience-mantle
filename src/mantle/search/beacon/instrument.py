# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc. Licensed under the Apache License, Version 2.0.
#
# Beacon is the permissive half of the two-tier model: mantle ships Apache so a
# store can be taken, built on, and shipped by anyone, and beacon is the reduced
# instrument that makes such a store useful on its own. The downstream consumer's
# `beacon_engine.py` carries a separate proprietary notice — a different tree, for
# the Foresight white-label pilot.
# ---------------------------------------------------------------------------

"""Beacon as an instrument — mantle's embodiment of `prism.instrument.Read`, on numpy alone.

`ARCHITECTURE-TARGET.md` §3 declares an instrument slot with two implementations,
and this is the second one::

    the aperture (ember)   domain: the signal — an ordered (T, F) frame
                            knows: ontology coordinates are sparse; evidence rows are correlated
                            needs: entroptics + numpy
    beacon (mantle)         domain: the corpus — a set of vectors
                            knows: its own corpus statistics
                            needs: numpy

The module itself is the embodiment: `import mantle.search.beacon.instrument as
beacon_instrument` and hand it to a host, exactly as a full node hands over
`ember.optics`. No adapter, no wrapper class — a contract that needed a shim to
fit the thing it describes would be a contract somebody invented.

    isinstance(beacon_instrument, prism.instrument.Read)        -> True   (6 members)
    isinstance(beacon_instrument, prism.instrument.Instrument)  -> False  (0 members)
    isinstance(beacon_instrument, prism.instrument.Dynamics)    -> False  (0 members)

## Instrument is unfilled: the cut is beacon's, the projection is the aperture's

`absorb_transmit`, `next_by_coupling`, and `membrane_screen` are not implemented
here, together with the `MembraneScreen` object model: they are projection —
carrying a signal through a membrane and accounting for what arrived — and beacon
measures only where a set stops. Beacon keeps the whole of the cut (`Read`, and
`mantle.search.beacon.cut`) and holds none of the projection: the giveaway under
the Apache/AGPL two-tier model is honest only if what it gives away is a complete
thing rather than a hobbled one.

The three names are genuinely absent, not stubbed to raise: `members_of(beacon,
"embodiment") == ()` and `isinstance(beacon, Instrument)` is False, and a caller
that reaches for one of them through `prism.instrument.require` gets
`InstrumentRequired`, naming the contract, the member, and the operation that
wanted it — the same mechanism that answers a caller asking for `Dynamics`. A stub
that raised on call would make `isinstance` true and `members_of` report a member
that cannot work.

## Dynamics is unfilled: a set has no lag

`decay_profile`, `resolution_limit`, `embed`, `fit_dynamics`, and `dynamics_state`
are not implemented here. Each is a statement about lag: `decay_profile` is C(τ),
`embed` is a Takens delay, `fit_dynamics` is a Koopman/DMD operator over
consecutive frames. §3 separates the two embodiments on exactly this axis — the
aperture's domain is an ordered (T, F) frame, beacon's is a set of vectors — and a
set has no lag.

`Dynamics` and `Instrument` are unfilled for different reasons: `Dynamics` is a
question beacon's domain does not pose (a set has no lag, so there is nothing to
measure), while `Instrument` is a question the domain poses perfectly well,
answered elsewhere in the product line (the aperture projects a corpus matrix as
readily as a signal frame). Both produce `isinstance(...) is False`. An embodiment
that fills neither is a complete embodiment of the cut, not a partial embodiment
of a fourteen-member protocol under which `isinstance(beacon, Read)` could never
have been true on its own.

The same domain argument has one visible consequence inside `Read` itself:
`SpectralRead.coherence` — the ordered-axis lag-1 z-score — is always `None` here,
never `0.0`. `None` means not measured; beacon cannot measure it for the same
reason it fills no `Dynamics` member, and publishing `0.0` would read as "measured:
no lag-1 correlation" for a statistic the domain cannot take.

## No tuned constants: every derivation names the error it models

A derivation can be exactly as wrong as a constant if it models the wrong error —
accumulation and cancellation are different errors, and mistaking one for the
other produces wrong cuts. So each quantity below is derived from
`engine.DEFAULT_FAR` (the one stated tolerance) or from the frame's own dtype, and
states which error it models:

    quantity            derived from                     error model it assumes
    ───────────────────────────────────────────────────────────────────────────
    noise floor         tw1_quantile(far)                the largest eigenvalue of an i.i.d.
                        (engine, unchanged)              Gaussian bulk fluctuates as TW1
    k_lo / k_hi         tw1_quantile(far) and            sampling scatter of the noise edge
    (tw1 path)          tw1_quantile(1 - far)            itself — where inside its own 1-2·far
                                                         range the edge fell on this draw.
                                                         Not the covariance estimation error the
                                                         aperture's Weyl band models — divergence 2.
    k_lo / k_hi         ±1/(1 + B)                       discreteness of a B-draw permutation
    (permutation path)                                   p-value — one surrogate either way is the
                                                         finest distinction B draws can resolve,
                                                         not a distributional bound.
    band()              (q(far) − q(1−far))·σ_J/μ        the same edge-scatter model, expressed
                                                         relative to the edge's own centre so it is
                                                         dimensionless and falls as T^(−1/2).
    draws               _permutation_draws(far)          an algebraic identity: a permutation
                        (engine, unchanged)              p-value cannot go below 1/(1+B).
    energy band         _float_noise                     forward error of floating-point
                                                         summation at the frame's own dtype —
                                                         accumulation, not cancellation, because
                                                         the terms summed are ‖·‖², all
                                                         non-negative, so no cancellation occurs.
    scales() ladder     successive halving               compute, not modelling — see `scales`.

## Where this embodiment diverges from the aperture, and why each is right

Two engines agreeing is not evidence either is correct, and disagreeing is not
evidence either is wrong — both need the domain argument, not just the numbers.

1. **`coherence` is always `None`.** A set has no ordered axis.

2. **The certified interval models a different error.** The aperture's
   `k_lo`/`k_hi` come from `entroptics.reads.resolved_dimension_interval`
   propagating a Weyl band `‖Ĉ − C‖₂ ≤ band` — the error in the sample covariance
   as an estimate of a population covariance. Beacon's domain is a corpus, so the
   set of vectors it holds *is* the population: there is no population covariance
   being estimated, and that band has nothing to bound. What is genuinely
   uncertain is where the noise edge fell on this draw, which beacon already
   models exactly (TW1), so the interval is
   `[#(S > floor at q(far)), #(S > floor at q(1−far))]`. Consequently `k_lo ==
   k_signal` here by construction, because beacon's point read is already taken
   at the conservative end of its own interval, where the aperture's point read
   sits inside its interval — the two `k_certain` flags mean different things and
   must not be compared.

3. **The minimum readable frame is 2 rows here, 4 there.** The aperture's
   `MIN_ROWS` is recovered from the falling factorial `T(T−1)(T−2)(T−3)` — the dof
   of the disjoint term of its lag-1 permutation variance. Beacon takes no lag-1
   statistic (divergence 1), so it has no such term; its own gate is the
   engine's, `min(N, F_live) >= 2`, which is what `johnstone` needs to describe a
   non-degenerate ensemble — the same fact as `Dynamics` being unfilled, seen
   from the shape end.

4. **`k >= 1` on a readable frame**, pinned in
   `prism.vectors/screen_read_vectors.json` (`noise_only`: beacon 1, aperture 0).
   This module keeps one rank rule and inherits it rather than introducing a
   second, so `accumulated_read` also floors at 1, and the contract's
   `k_signal == 0` (unmeasurable) case is reached here only through
   `screened=False`, never through a readable frame.

5. **Beacon fills no `Instrument` member; the aperture fills all three.** The
   product divergence: the aperture projects, beacon cuts. Pinned in
   `screen_read_vectors.json` as divergence 6.

All five divergences are pinned in `screen_read_vectors.json`, where both engines
read them, and asserted here with their domain argument: a shared data pin says
the numbers must not move, and a test says why they must not.

## Import boundary

Numpy only. This module may not import `beam` or `entroptics` — beacon is what
keeps mantle shippable while entroptics stays private, and
`tests/test_lexical_extra_is_numpy_free.py` holds the floor from the other side
(the semantic arm fails without numpy).

The one exception is `prism.rounding`, the single-sourced floating-point rounding
law, taken from prism's dependency-free base (stdlib only: no numpy, no
cryptography, no extra) rather than duplicated here. Importing it pulls exactly
five modules, because importing a submodule runs the package `__init__` first:

    prism · prism.canonical · prism.environment · prism.errors · prism.rounding

all five held stdlib-only by prism's own `test_contract_install_is_pure.py`.
`test_the_instrument_imports_with_beam_entroptics_and_prism_blocked` removes
`beam` and `entroptics` from the import system outright and permits prism modules
by exact name; `prism.conservation` — where the same law is applied with numpy —
is in the test's control set and must still fail to import here, so a second
prism edge, or a transitive pull through one, fails this test the day it lands.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The one edge out of this module. `prism.rounding` is the single-sourced floating-point rounding
# law: stdlib-only, on prism's dependency-free base, so this costs no numpy, no cryptography, and no
# extra. Its import closure is five contract modules, pinned by name in the import-boundary test
# below. See `_float_noise` for why the law does not stay duplicated here.
from prism.rounding import split_walk_rounding

from .engine import (DEFAULT_FAR, ENGINE_ID, ENGINE_ID_PERM, BeaconEngineError,
                     _as_matrix, _floor_sq, _permutation_core, _permutation_draws,
                     _tw1_core, johnstone, tw1_quantile)
from .engine import whiten as _whiten

__all__ = [
    # ── prism.instrument.Read (6) ──
    "correlated_null", "read_ordered", "resolvable", "scales", "accumulator", "accumulated_read",
    # ── the null providers a caller states an opinion with ──
    "derived_null", "CorrelatedNull", "DerivedNull",
    # ── the return shapes the contract's companion protocols describe ──
    "SpectralRead", "PooledScreen", "PooledSpectrum",
    # ── the directional companion of `resolvable`, and the readable-frame boundary ──
    "principal_directions", "MIN_ROWS",
]
# `decay_profile`, `resolution_limit`, `embed`, `fit_dynamics`, and `dynamics_state` are absent from
# this list and from this module — see the `Dynamics` section in the module docstring.
# `isinstance(this_module, Dynamics) is False` is asserted in the tests.
#
# `absorb_transmit`, `next_by_coupling`, `membrane_screen`, and `MembraneScreen` are absent for a
# different reason — see the `Instrument` section in the module docstring. A caller reaching for one
# of them through `prism.instrument.require` gets `InstrumentRequired` naming the member and the
# contract; a caller reaching for it as an attribute gets nothing, which is why the names are absent
# rather than stubbed to raise.


# ═══════════════════════════════════════════════════════════════════════════
# The readable-frame boundary — recovered from the engine's own gate
# ═══════════════════════════════════════════════════════════════════════════

def _admits(n_rows: int, n_features: int) -> bool:
    """The degeneracy gate, in one place: `min(N, F_live) >= 2`.

    It is the engine's own gate: below it, `signal_rank` and `structure_rank` both fall back to
    their degenerate default rather than taking a real read, and it is what `johnstone(N, F)` needs
    to describe a non-degenerate ensemble. Nothing else in this module compares a shape against an
    integer."""
    return min(int(n_rows), int(n_features)) >= 2


def _smallest_readable_rows() -> int:
    """The smallest row count `_admits` accepts, recovered from the gate rather than typed.

    Ascends from an empty frame with the feature axis held one wider than the row axis, so the rows
    are always the binding side, and returns the first count the gate admits. If the gate ever
    changes this follows it with no edit here.

    It is a report, not a second gate: nothing in this module compares against it. It is exported
    because a caller handed `None` by `resolvable` may legitimately want to know where the boundary
    fell — the same role `ember.optics.MIN_ROWS` plays, at a different value, for the reason set out
    as divergence 3 in the module docstring."""
    n = 0
    # No iteration bound: `min(n, n+1) = n` is strictly increasing, so this terminates by the
    # algebra.
    while not _admits(n, n + 1):
        n += 1
    return n


MIN_ROWS = _smallest_readable_rows()          # == 2, computed, not written down


# ═══════════════════════════════════════════════════════════════════════════
# The ordered-container contract — a set has no reproducible order to read
# ═══════════════════════════════════════════════════════════════════════════

_UNORDERED = (set, frozenset)


def _as_ordered_matrix(rows) -> np.ndarray:
    """Enforce the ordered-container contract, then materialise (T, F) float64 through the engine.

    The two senses of "ordered" are different and both apply here. Beacon's domain is a set of
    vectors, so the order of the rows carries no meaning to this instrument — that is why it fills
    no `Dynamics` member. But the container must still have a reproducible order, because a read a
    caller cannot show the same input to twice is a read nobody can audit. So a `set` has no
    reproducible row order and is not accepted (`Read.read_ordered`'s contract: an unordered
    container is not silently sequenced), and neither is a bare iterator, which is single-shot.

    Materialisation is `engine._as_matrix`, unchanged — including its gap policy (scattered
    non-finite cells are filled with the column mean rather than poisoning every mode of an SVD at
    once). One materialisation rule for the module and the embodiment.
    """
    if isinstance(rows, _UNORDERED):
        raise BeaconEngineError(
            f"{type(rows).__name__} has no reproducible row order. Beacon's domain is a SET of "
            "vectors, so the ORDER carries no meaning to the read — but a container whose order "
            "cannot be shown twice makes the read unauditable. Materialise it into a list or an "
            "ndarray in the order you mean."
        )
    if isinstance(rows, np.ndarray):
        W = rows
    elif isinstance(rows, Sequence):
        W = np.asarray([np.asarray(r, dtype=np.float64).ravel() for r in rows], dtype=np.float64)
    elif isinstance(rows, Iterator):
        raise BeaconEngineError(
            "a bare iterator is single-shot; its order cannot be reproduced or audited. "
            "Materialise it into a list in the order you mean."
        )
    else:
        W = rows
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    return np.ascontiguousarray(_as_matrix(W))


# ═══════════════════════════════════════════════════════════════════════════
# The nulls — the false-alarm level travels with the null, never beside it
# ═══════════════════════════════════════════════════════════════════════════
# This is a member and not a keyword, and `prism.instrument.Read.correlated_null` says why: the
# null is the domain's knowledge. One cutoff is one decision, so the provider owns both the
# threshold and the far it is drawn at, rather than being passed alongside a separate `far=`
# keyword that could disagree with it.
#
# Beacon already had both nulls, under its own names. `signal_rank` is the closed-form
# Tracy-Widom edge; `structure_rank` is the distribution-free permutation null, whose surrogate is
# entroptics' own `shuffle_in_time` reproduced argsort-and-all so the two embodiments draw from the
# same null. These two objects select between them, so `read_ordered` has one body and the null
# decides which edge it is read against, rather than two functions with two rank rules.


@dataclass(frozen=True)
class DerivedNull:
    """The closed-form Johnstone / Tracy-Widom edge, pinned to a stated false-alarm level.

    Correct where the rows are genuinely i.i.d.; optimistic where they are correlated by
    construction, which is what retrieved evidence always is — use `correlated_null()` there.
    """
    far: float
    kind: str = "tw1"
    instrument: str = ENGINE_ID


@dataclass(frozen=True)
class CorrelatedNull:
    """The distribution-free permutation null — the read for rows that share a generative cause.

    `draws` is compute, never a modelling choice, and it is stated as the minimum at which `far` is
    attainable (`engine._permutation_draws`), never a guessed budget.
    """
    far: float
    draws: int
    kind: str = "permutation"
    instrument: str = ENGINE_ID_PERM


def correlated_null(*, draws: Optional[int] = None, far: Optional[float] = None) -> CorrelatedNull:
    """The noise provider for correlated rows — what a corpus always is.

    `far=None` means the caller stated no level, so beacon uses its own published value
    (`engine.DEFAULT_FAR`) rather than restating one here — that constant is the single stated
    tolerance the whole engine descends from.

    `draws=None` means the caller stated no budget. It is then the smallest B at which a
    permutation p-value can reach the level at all: `p = (1 + #{surrogate >= observed}) / (1 + B)`
    has minimum `1/(1+B)`, so `B >= 1/far − 1` (Phipson & Smyth 2010). That inequality is the whole
    derivation and there is no choice in it. More draws buys power and costs one SVD each, which is
    a question for a measured resource envelope and for nothing else; nothing on this box publishes
    a per-draw cost or a wall-clock allowance, so no larger number is invented.
    """
    level = DEFAULT_FAR if far is None else float(far)
    if not (0.0 < level < 1.0):
        raise ValueError(f"far is a false-alarm probability and must lie in (0, 1); got {far!r}")
    b = _permutation_draws(level) if draws is None else max(1, int(draws))
    return CorrelatedNull(far=level, draws=b)


def derived_null(*, far: float) -> DerivedNull:
    """The closed-form edge, pinned to a stated level — the counterpart of `correlated_null`.

    Not a contract member. It exists so that a caller with an opinion about `far` on the
    Tracy-Widom path has somewhere to put it: with `far` gone from the read's signature (it never
    belonged beside a null), a null object is the only place a level can honestly live. The TW1
    quantile is inverted from the survival function for any level that is not tabulated, so an
    arbitrarily sharp `far` is still a derived edge and nothing is fitted.
    """
    level = float(far)
    if not (0.0 < level < 1.0):
        raise ValueError(f"far is a false-alarm probability and must lie in (0, 1); got {far!r}")
    return DerivedNull(far=level)


def _resolve_null(null) -> Tuple[str, float, Optional[int]]:
    """`(kind, far, draws)` for whatever the caller handed over. `None` -> beacon's own default."""
    if null is None:
        return "tw1", DEFAULT_FAR, None
    if isinstance(null, CorrelatedNull):
        return "permutation", float(null.far), int(null.draws)
    if isinstance(null, DerivedNull):
        return "tw1", float(null.far), None
    raise BeaconEngineError(
        "the null must be one of beacon's own providers — `correlated_null()` for rows correlated "
        f"by construction, `derived_null(far=...)` for an i.i.d. bulk — or None for beacon's "
        f"published default; got {type(null).__name__}. A read cannot be taken against a cutoff "
        "whose level it cannot name."
    )


# ═══════════════════════════════════════════════════════════════════════════
# The structure read — `prism.instrument.Read`
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SpectralRead:
    """What `read_ordered` returns — the read, and the continuous evidence behind its integer.

    `k_signal` alone is a count with nothing attached, and a count whose distance from the noise
    floor was discarded cannot say whether it was decided or coin-flipped. Every field here was
    measured to produce that integer and is reported rather than thrown away.

    Units are the null's, and they differ between the two paths because the decision differs.
    `k_margin_last` / `k_margin_next` / `contrast` report the distance from the cutoff in the units
    the comparison was actually made in: whitened singular values against the TW1 floor on the
    derived-null path, p-value against `far` on the permutation path. Restating one in the other's
    units would be a conversion nobody measured. `null_kind` says which you are holding.

    `coherence` is always `None` here, and that is the domain speaking, not an omission — a set has
    no lag-1 axis (see the module docstring). `None` means not measured; `0.0` would be a reading.
    """

    n_rows: int
    n_features: int
    live_channels: int
    k_signal: int
    k_lo: int
    k_hi: int
    k_margin_last: Optional[float]
    k_margin_next: Optional[float]
    contrast: float
    top_share: float
    coherence: Optional[float]          #: always None — beacon's domain has no ordered axis
    screened: bool                      #: False when the frame could not carry a read at all
    degraded: bool                      #: the collapsed-axis hazard (<= 1 live channel)
    far: float
    null_kind: str
    instrument: str

    @property
    def k_certain(self) -> bool:
        """True when the interval has collapsed onto the point read.

        Means something different from the aperture's flag of the same name — divergence 2 in the
        module docstring. Here it says: no mode lies between the noise edge's own lower and upper
        quantiles, so the count does not depend on where inside its sampling range the edge happened
        to fall. There it says: the Weyl band on the covariance estimate has collapsed. Do not
        compare them.
        """
        return self.k_lo == self.k_hi == self.k_signal

    def as_read(self) -> Dict[str, Any]:
        """The flat dict for a log line or an `Answer.read`. No rounding — this is a record a
        future reader re-judges from, and the margins are precisely the numbers that say whether a
        count was decided or coin-flipped."""
        return {
            "k_signal": self.k_signal, "k_lo": self.k_lo, "k_hi": self.k_hi,
            "k_certain": self.k_certain,
            "k_margin_last": self.k_margin_last, "k_margin_next": self.k_margin_next,
            "contrast": self.contrast, "top_share": self.top_share,
            "coherence": self.coherence, "screened": self.screened, "degraded": self.degraded,
            "n_rows": self.n_rows, "n_features": self.n_features,
            "live_channels": self.live_channels,
            "far": self.far, "null": self.null_kind, "instrument": self.instrument,
        }


def _unscreened(W: np.ndarray, live: int, degraded: bool, far: float, kind: str,
                instrument: str) -> SpectralRead:
    """The report for a frame that cannot carry a read at all.

    `k_signal = 0` here means not measured, and `screened=False` is what says so — it is not
    beacon changing its mind about `k >= 1`. The two surfaces answer different consumers:
    `engine.signal_rank` floors at 1 because its consumer projects onto the result and an empty
    subspace is worse than a coarse one, while this surface's consumer (`resolvable`, and through
    it a cut) must be able to defer. Both report the same frame; neither fabricates. A caller must
    gate on `screened`, never read `k_signal` past it."""
    return SpectralRead(
        n_rows=int(W.shape[0]), n_features=int(W.shape[1]) if W.ndim == 2 else 0,
        live_channels=int(live), k_signal=0, k_lo=0, k_hi=0,
        k_margin_last=None, k_margin_next=None, contrast=0.0, top_share=0.0,
        coherence=None, screened=False, degraded=bool(degraded), far=float(far),
        null_kind=kind, instrument=instrument)


def read_ordered(rows, *, null=None, seed: int = 0, window: Optional[int] = None,
                 with_screen: bool = True) -> SpectralRead:
    """Screen a (T, F) frame — how much of it is signal, against which null, at what scale.

    `null=None` runs on beacon's own derived default (the Tracy-Widom edge at `DEFAULT_FAR`), which
    is correct for an i.i.d. bulk and optimistic for correlated rows. A corpus is correlated by
    construction — documents share vocabulary, and a topic is a correlation among rows — so a
    corpus read should pass `correlated_null()`. Deterministic per `seed` (which only the
    permutation path consumes; the closed-form edge is exact and has nothing to sample).

    `window=None` reads the whole frame. Beacon never truncates a longer batch to an adaptive
    window, because that would answer a different question than the one asked without the caller
    being able to see it happened. An explicit `window=w` reads the first `w` rows — over a set of
    vectors "a window" can only mean a subset, and the leading `w` is the one subset the caller can
    reproduce from the container it handed over. `scales` is built from nested windows of exactly
    this kind.

    `with_screen` is accepted for signature parity and changes nothing here. The keyword exists on
    the aperture to skip entroptics' folded `Screen`, the half that measures `coherence`; beacon has
    no folded screen and cannot measure `coherence` in any case (see `Dynamics` in the module
    docstring), so the field the keyword gates reads `None` on both settings, never `0.0`. There is
    no second half here to skip.
    """
    W = _as_ordered_matrix(rows)
    if window is not None:
        w = int(window)
        if w < 1:
            raise BeaconEngineError(f"window must be at least one row; got {window!r}")
        W = W[:w]
    kind, far, draws = _resolve_null(null)
    return (_read_permutation(W, far, draws, seed) if kind == "permutation"
            else _read_tw1(W, far))


def _read_tw1(W: np.ndarray, far: float) -> SpectralRead:
    """The closed-form path — `engine._tw1_core`, with the interval and the margins kept."""
    core = _tw1_core(W, far)
    if not core.readable or not _admits(core.N, core.F):
        return _unscreened(W, core.live_channels, core.degraded, far, "tw1", ENGINE_ID)

    # The interval: the scatter of the edge, not the error of a covariance.
    # The floor is `sqrt(sigma^2 * (mu + q * sigma_J))`, and `q` is a quantile of the TW1 law that
    # the largest noise eigenvalue follows. The stated level `far` picks its upper quantile; the
    # matching lower one is `tw1_quantile(1 - far)`, inverted from the same survival function. The
    # two together bracket where the edge falls on `1 - 2*far` of its own draws, so:
    #     above the high floor -> resolved even if the edge landed high   -> k_lo (certainly above)
    #     above the low  floor -> resolved only if the edge landed low    -> k_hi (possibly above)
    # Error model: the sampling scatter of the noise edge. Beacon's domain is a corpus, which is
    # the population, so there is no population covariance being estimated, and the aperture's Weyl
    # band has nothing to bound here (divergence 2 in the module docstring).
    floor_hi = math.sqrt(max(0.0, _floor_sq(core.sigma2, core.N, core.F, far)))
    floor_lo = math.sqrt(max(0.0, _floor_sq(core.sigma2, core.N, core.F, 1.0 - far)))
    k_point = int(np.count_nonzero(core.sv > floor_hi))
    k_hi = int(np.count_nonzero(core.sv > floor_lo))
    # The `>= 1` rule (divergence 4) applies to the whole interval, not only its point: an interval
    # that could not contain the count it brackets would be a bound on nothing.
    k_signal, k_lo, k_hi = max(1, k_point), max(1, k_point), max(1, k_hi)

    # The margins are in the units the comparison was made in — whitened singular values against
    # the floor the count was taken at. `None` where the mode does not exist.
    m_last = float(core.sv[k_point - 1] - floor_hi) if 0 < k_point <= core.sv.size else None
    m_next = float(core.sv[k_point] - floor_hi) if 0 <= k_point < core.sv.size else None
    power = float(np.sum(core.sv ** 2))
    return SpectralRead(
        n_rows=core.N, n_features=int(core.A.shape[1]), live_channels=core.live_channels,
        k_signal=k_signal, k_lo=k_lo, k_hi=k_hi, k_margin_last=m_last, k_margin_next=m_next,
        contrast=float(core.sv[0] / floor_hi) if core.sv.size and floor_hi > 0.0 else 0.0,
        top_share=float(core.sv[0] ** 2 / power) if power > 0.0 else 0.0,
        coherence=None, screened=True, degraded=core.degraded, far=far,
        null_kind="tw1", instrument=ENGINE_ID)


def _read_permutation(W: np.ndarray, far: float, draws: Optional[int], seed: int) -> SpectralRead:
    """The distribution-free path — `engine._permutation_core`, the same rule `structure_rank` uses.

    Not a second implementation: `structure_rank` and this read call one extracted core, so there is
    one leading-run rule and one mean-direction offset. A parallel implementation is exactly how the
    module surface and the embodiment surface would start naming different ranks while both looked
    right; `tests/test_beacon_instrument.py` asserts the two agree on every frame it reads."""
    core = _permutation_core(W, far=far, draws=draws, seed=seed)
    if not core.readable:
        return _unscreened(W, core.live_channels, core.degraded, far, "permutation",
                           ENGINE_ID_PERM)
    k_point = max(1, core.k_tested + core.offset)

    # The interval: the discreteness of a B-draw p-value.
    # `p = (1 + #{surrogate >= observed}) / (1 + B)` moves in steps of exactly `1/(1+B)`: one
    # surrogate either way is the finest distinction B draws can resolve, and `_permutation_draws`
    # already derives B as the minimum at which `far` is attainable at all. So a component is
    # certainly resolved when it clears the level with a draw to spare, and possibly resolved when
    # it would clear it had one draw fallen the other way.
    # Error model: resolution, not a distributional bound. It does not claim a confidence level for
    # the p-value — it says how finely this many draws can distinguish anything at all.
    step = 1.0 / (1.0 + core.draws)
    k_lo = max(1, _leading_run(core.pvalue + step, far) + core.offset)
    k_hi = max(1, _leading_run(core.pvalue - step, far) + core.offset)

    # Margins in the units the decision was taken in: distance from the level, positive for a
    # component the evidence kept and negative for the first it stopped at.
    tested = core.k_tested
    m_last = float(far - core.pvalue[tested - 1]) if 0 < tested <= core.pvalue.size else None
    m_next = float(far - core.pvalue[tested]) if 0 <= tested < core.pvalue.size else None
    # The empirical edge: the largest leading singular value any surrogate produced. That is the
    # level-`1/(1+B)` cutoff — the finest this draw budget can resolve, and the same quantity B was
    # derived from, so no second level is introduced to state a contrast.
    crit = float(np.max(core.surrogate_top)) if core.surrogate_top.size else 0.0
    power = float(np.sum(core.s ** 2))
    return SpectralRead(
        n_rows=int(core.C.shape[0]), n_features=int(core.C.shape[1]),
        live_channels=core.live_channels, k_signal=k_point, k_lo=min(k_lo, k_point),
        k_hi=max(k_hi, k_point), k_margin_last=m_last, k_margin_next=m_next,
        contrast=float(core.s[0] / crit) if core.s.size and crit > 0.0 else 0.0,
        top_share=float(core.s[0] ** 2 / power) if power > 0.0 else 0.0,
        coherence=None, screened=True, degraded=core.degraded, far=far,
        null_kind="permutation", instrument=ENGINE_ID_PERM)


def _leading_run(pvalue: np.ndarray, far: float) -> int:
    """The leading run of components at or below `far` — the ordering rule, in one place.

    A component that fails is where the evidence stops; a later one clearing the level past it is
    the multiple-comparison artifact this ordering rule excludes by counting only the leading run."""
    failed = np.nonzero(pvalue > far)[0]
    return int(failed[0]) if failed.size else int(pvalue.shape[0])


def resolvable(rows, *, null=None, seed: int = 0,
               require_certain: bool = False) -> Optional[int]:
    """`k_signal` alone — the modes standing above the noise floor of the spectrum read.

    A projection of `read_ordered`, which is why it shares that contract rather than getting its
    own: nothing can fill it without being able to take the read.

    `None` is the computed null and it is a result. It means the frame cannot carry a read (too few
    rows, one live feature, all-zero) — a different statement from a resolved count of 1. A caller
    receiving `None` must defer, never substitute a keep-everything default.

    This is where `engine.signal_rank`'s floor-at-one does not apply, and the difference is the
    consumer, not the rule — see `_unscreened`. `signal_rank` answers a caller that will project
    onto the result; this answers a caller that can wait for more evidence.

    `require_certain=True` returns the count only when the interval has collapsed onto it. Read
    `SpectralRead.k_certain` before comparing that flag with the aperture's — divergence 2 in the
    module docstring.
    """
    rd = read_ordered(rows, null=null, seed=seed, with_screen=False)
    if not rd.screened:
        return None
    if require_certain and not rd.k_certain:
        return None
    return int(rd.k_signal)


def principal_directions(rows, *, null=None, seed: int = 0) -> Optional[np.ndarray]:
    """The resolved subspace basis — an `(F, k)` array of the directions `resolvable` counted.

    The directional companion to `resolvable`, read off the same spectrum: that says how many modes
    stand above the null, this says which directions they span. `None` when the frame cannot carry a
    read; an `(F, 0)` array when nothing resolves.

    Which matrix the directions are measured on is not the incident one. On the derived-null path
    they are the right singular vectors of the whitened matrix, because that is the only matrix on
    which the noise floor is a meaningful reference; on the permutation path they are those of the
    mean-centred matrix, because that is the matrix the null is drawn against. Either way they are
    returned as directions in the frame's own feature coordinates, so a caller that builds an
    orthogonal projector `B·pinv(B)` from them gets exact conservation whichever matrix measured the
    directions. That caller is not in this package: projection is the aperture's half of the product
    line (see `Instrument` in the module docstring). This function stays because it is the
    directional companion of `resolvable` — the same read, asked for *which* instead of *how many* —
    and `mantle.search.beacon.cut` is built on exactly that question.
    """
    W = _as_ordered_matrix(rows)
    kind, far, draws = _resolve_null(null)
    if kind == "permutation":
        core = _permutation_core(W, far=far, draws=draws, seed=seed)
        if not core.readable:
            return None
        M, k = core.C, max(1, core.k_tested + core.offset)
    else:
        core = _tw1_core(W, far)
        if not core.readable or not _admits(core.N, core.F):
            return None
        M = core.A
        k = max(1, int(np.count_nonzero(
            core.sv > math.sqrt(max(0.0, _floor_sq(core.sigma2, core.N, core.F, far))))))
    try:
        _u, _s, vt = np.linalg.svd(M, full_matrices=False)
    except Exception as exc:                 # LinAlgError (non-convergent SVD) and friends
        raise BeaconEngineError(
            f"basis read failed on a {M.shape} frame ({type(exc).__name__}: {exc})") from exc
    k = min(int(k), int(vt.shape[0]))
    return np.ascontiguousarray(vt[:k].T)


# ── Scales ────────────────────────────────────────────────────────────────────────────────────

def _window_ladder(n_rows: int) -> List[int]:
    """The derived observation-window ladder — nested subsets by successive halving.

    What "a window" means is the domain's, and `Read.scales` says so: trailing windows of the
    ordered axis for a signal, nested subsets for a corpus. Beacon is the corpus, so each rung is
    the leading `w` rows and every rung is a subset of the next — which is what makes the rungs
    comparable at all. A ladder of disjoint samples would measure sampling variance, not structure
    against aperture size.

    The ratio is the same kind of argument as `_permutation_draws`. Halving is the coarsest
    non-trivial nesting: a finer ladder resolves the trend better and costs one full decomposition
    per extra rung, which makes it compute and not modelling — beacon does not guess a compute
    budget any more than it guesses a draw count. Nothing on this box publishes a per-read cost or a
    wall-clock allowance, so the ladder is the minimum that says anything: the largest window, and
    each halving down to the smallest readable frame. A caller that has measured its own budget
    passes `windows` explicitly and gets a finer trend at the same cost per rung.
    """
    ladder: List[int] = []
    w = int(n_rows)
    while w >= MIN_ROWS:
        ladder.append(w)
        w //= 2
    return sorted(set(ladder))


def scales(rows, windows=None) -> Optional[List[Dict[str, Any]]]:
    """Structure vs observation window — the same read at several apertures.

    A caller can then see whether the structure is a property of the corpus or of how much of it was
    looked at. Returns one entry per rung, ascending by window size, each carrying the read taken
    over that nested subset::

        [{"window", "n_rows", "n_features", "live_channels", "k_signal", "k_lo", "k_hi",
          "k_certain", "contrast", "top_share", "screened", "degraded"}, ...]

    `windows=None` uses the derived ladder (`_window_ladder`). `None` is returned when the whole
    frame cannot carry a read — a corpus with no readable subset has no profile, and an empty list
    would be a claim that it was measured and found flat.

    Rungs that individually cannot carry a read are dropped rather than reported as zeros, and a
    caller comparing two profiles must compare rungs by `window`, not by position.
    """
    W = _as_ordered_matrix(rows)
    if not read_ordered(W).screened:
        return None
    ws = _window_ladder(int(W.shape[0])) if windows is None else \
        sorted({int(w) for w in windows if int(w) >= 1})
    out: List[Dict[str, Any]] = []
    for w in ws:
        if w > int(W.shape[0]):
            continue
        rd = read_ordered(W, window=w)
        if not rd.screened:
            continue
        out.append({"window": int(w), "n_rows": rd.n_rows, "n_features": rd.n_features,
                    "live_channels": rd.live_channels, "k_signal": rd.k_signal,
                    "k_lo": rd.k_lo, "k_hi": rd.k_hi, "k_certain": rd.k_certain,
                    "contrast": rd.contrast, "top_share": rd.top_share,
                    "screened": rd.screened, "degraded": rd.degraded})
    return out or None


# ── The accumulator — a snapshot is not a stream ─────────────────────────────────────────────
# The instrument accumulates. A single turn's frame is too short for an interval to collapse, and a
# count that cannot certify falls back on a statistic over a bare score column. `T` grows, the band
# shrinks, and the count becomes a measurement instead of a fit.


@dataclass(frozen=True)
class PooledSpectrum:
    """What `PooledScreen.spectral()` returns — the spectrum over everything pooled so far."""
    T: int
    F: int
    live_channels: int
    eigenvalues: np.ndarray        #: of the pooled centred scatter, descending
    singular_values: np.ndarray    #: sqrt of the above — the units the floor is compared in
    noise_floor: float             #: the floor at the pooled shape, at `far`
    noise_floor_lo: float          #: the same floor at `1 - far` — the edge's lower quantile
    resolved_modes: int
    top_share: float
    contrast: float
    band: float
    far: float
    instrument: str = ENGINE_ID


class PooledScreen:
    """A pooling screen — fed one intact plane per turn, read as one spectrum.

    Not row by row. `add` pools a whole `(T_p, F)` plane's second moment (`planeᵀ plane`), which is
    what keeps the within-plane correlation; flattening the planes and pooling rows would discard
    exactly the correlation the read is about.

    `merge` is the across-peers path: two nodes that have each been accumulating combine into one
    spectrum without exchanging a single raw frame — only the pooled scatter and the column sums
    travel. That is a peer contributing its evidence while its observations stay its own.

    Two things differ here from `engine.signal_rank`, both forced by pooling rather than chosen:

      1. **σ² comes from the mean row energy, not the median.** A median is not a poolable
         statistic — it cannot be recovered from a sum of second moments — while `trace(S)/T` is
         exactly the mean row energy and is. So the debias drops the Wilson-Hilferty `c_F` term,
         which exists only to correct a median of a chi²_F to its mean. The cost, in its named
         direction: this floor is not robust to a handful of bright rows the way `signal_rank`'s is.
         Bright rows raise the mean, which raises the floor, which makes the pooled read
         under-count — the conservative direction, and the safe one for a count whose whole purpose
         is to say when there is enough evidence.
      2. **`whiten=True` whitens each plane at add time**, for the same reason: the per-channel MAD
         is a median and cannot be pooled. A caller wanting one global whitening must whiten before
         adding, since the two are genuinely different normalisations.
    """

    __slots__ = ("F", "T", "planes", "_sum", "_scatter", "_whiten", "_far")

    def __init__(self, n_features: int, *, whiten: bool = False,
                 far: float = DEFAULT_FAR) -> None:
        f = int(n_features)
        if f < 1:
            raise BeaconEngineError(f"an accumulator needs at least one feature; got {n_features!r}")
        self.F = f
        self.T = 0
        self.planes = 0
        self._sum = np.zeros(f, dtype=np.float64)
        self._scatter = np.zeros((f, f), dtype=np.float64)
        self._whiten = bool(whiten)
        self._far = float(far)

    def add(self, plane) -> "PooledScreen":
        """Pool one intact `(T_p, F)` plane. `F` must be constant across planes by construction —
        that, and nothing else, is what a fixed coordinate basis is for."""
        P = _as_ordered_matrix(plane)
        if int(P.shape[1]) != self.F:
            raise BeaconEngineError(
                f"this accumulator pools {self.F}-feature planes and was handed {P.shape[1]}. A "
                "pooled spectrum over two coordinate systems is not a spectrum.")
        if self._whiten:
            P = _whiten(P)
        self.T += int(P.shape[0])
        self.planes += 1
        self._sum += P.sum(axis=0)
        self._scatter += P.T @ P
        return self

    def merge(self, other: "PooledScreen") -> "PooledScreen":
        """Combine two accumulations without exchanging a raw frame. Returns a new accumulator — a
        merge that mutated its left operand would make "who has seen what" depend on call order."""
        if not isinstance(other, PooledScreen):
            raise BeaconEngineError(
                f"can only merge another pooled screen; got {type(other).__name__}")
        if other.F != self.F:
            raise BeaconEngineError(
                f"cannot merge a {other.F}-feature accumulation into a {self.F}-feature one")
        out = PooledScreen(self.F, whiten=self._whiten, far=self._far)
        out.T = self.T + other.T
        out.planes = self.planes + other.planes
        out._sum = self._sum + other._sum
        out._scatter = self._scatter + other._scatter
        return out

    def _centred_scatter(self) -> np.ndarray:
        """`Σ (x − x̄)(x − x̄)ᵀ` from the pooled second moment and sum — the matrix a centred SVD
        would have decomposed, without ever holding the rows."""
        if self.T <= 0:
            return np.zeros((self.F, self.F))
        m = self._sum / float(self.T)
        return self._scatter - float(self.T) * np.outer(m, m)

    def band(self) -> float:
        """The current concentration band — a progress reading, not a verdict.

        Error model: the sampling scatter of the noise edge, the same one `read_ordered`'s interval
        uses, expressed relative to the edge's own centre so that it is dimensionless::

            band = (q(far) − q(1 − far)) · σ_J / μ

        `μ` and `σ_J` are Johnstone's finite-size centring and scaling at the pooled shape, so the
        band falls as `T^(−1/2)` — more evidence, a tighter statement about where the edge is. It is
        a progress reading: a wide band calls for more evidence, never for loosening `far`.
        """
        if self.T <= 0:
            return float("inf")
        live = self._live_channels()
        if not _admits(self.T, live):
            return float("inf")
        mu, sig_j = johnstone(int(self.T), int(live))
        if mu <= 0.0:
            return float("inf")
        width = tw1_quantile(self._far) - tw1_quantile(1.0 - self._far)
        return float(width * sig_j / mu)

    def _live_channels(self) -> int:
        """Channels the pooled spectrum is actually read over — those with non-zero pooled
        variance. Same meaning as `engine.live_width`, measured on the matrix this read
        decomposes; counting dead columns biases the floor in two opposing directions at once."""
        S = self._centred_scatter()
        return max(1, int(np.count_nonzero(np.abs(np.diag(S)) > 0.0)))

    def spectral(self) -> PooledSpectrum:
        """The pooled spectrum over everything added."""
        T, live = int(self.T), self._live_channels()
        S = self._centred_scatter()
        if T <= 0 or not _admits(T, live):
            zero = np.zeros(0)
            return PooledSpectrum(T=T, F=self.F, live_channels=live, eigenvalues=zero,
                                  singular_values=zero, noise_floor=float("inf"),
                                  noise_floor_lo=float("inf"), resolved_modes=0, top_share=0.0,
                                  contrast=0.0, band=self.band(), far=self._far)
        # A scatter matrix is symmetric PSD by construction, so `eigvalsh` is the right
        # decomposition and negative values are floating-point residue, not eigenvalues.
        ev = np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))[::-1]
        ev = np.maximum(ev, 0.0)
        sv = np.sqrt(ev)
        # σ² from the mean row energy — the only one a pooled scatter carries. See the class
        # docstring for the cost and its direction. The centring dof is `(T−1)/T`, exactly as
        # `engine._debias_denominator` applies it; the Wilson-Hilferty `c_F` term is absent because
        # it corrects a median, and there is no median here.
        sigma2 = float(np.trace(S)) / (float(live) * max(T - 1, 1)) + 1e-30
        floor = math.sqrt(max(0.0, _floor_sq(sigma2, T, live, self._far)))
        floor_lo = math.sqrt(max(0.0, _floor_sq(sigma2, T, live, 1.0 - self._far)))
        power = float(ev.sum())
        return PooledSpectrum(
            T=T, F=self.F, live_channels=live, eigenvalues=ev, singular_values=sv,
            noise_floor=floor, noise_floor_lo=floor_lo,
            resolved_modes=max(1, int(np.count_nonzero(sv > floor))),
            top_share=float(ev[0] / power) if power > 0.0 else 0.0,
            contrast=float(sv[0] / floor) if floor > 0.0 else 0.0,
            band=self.band(), far=self._far)


def screen_normalize(rows, mask=None):
    """Per-channel robust (MAD) whitening of the screen — beacon's fill of the `read` contract member.

    Required whenever a frame co-registers heterogeneous planes, because read raw the leading mode
    is the unit mismatch. `engine.whiten` supplies it, and `noise_floor` has always depended on it:
    measured on the whitened matrix, since the raw one is not a meaningful reference. The member
    exposes machinery that was already load-bearing here.

    This is an independent implementation of the same estimator as `entroptics.entropy.normalize` —
    the two repos may not import each other, so their agreement is evidence rather than a shared
    dependency. Checked across five frames (iid 120×16, heterogeneous units, planted rank-3, a dead
    channel, a short 8×16): the maximum difference is 0.0, exactly. A future divergence is therefore
    a real finding and belongs in the divergence section of `tests/test_beacon_instrument.py`, not a
    rounding difference to absorb.

    `mask` is accepted for contract compatibility and ignored — beacon has no masked path.
    entroptics marks masked cells NaN and excludes them from the statistics; `_as_matrix` here fills
    non-finite cells with the column mean instead. Passing a mask and having it silently honoured on
    one embodiment and dropped on the other would be a divergence hiding inside a shared name, so it
    is stated: a caller needing masked normalisation needs the aperture.
    """
    if mask is not None:
        raise BeaconEngineError(
            "beacon's screen_normalize has no masked path — entroptics excludes masked cells from "
            "the per-channel statistics and beacon fills non-finite cells with the column mean, so "
            "honouring a mask here would silently differ from the aperture. Use the aperture.")
    return _whiten(rows)


def accumulator(n_features: int, *, whiten: bool = False) -> PooledScreen:
    """An empty `PooledScreen` of fixed feature width — a factory, so the caller owns the object and
    its lifetime."""
    return PooledScreen(n_features, whiten=whiten)


def accumulated_read(acc) -> Optional[Dict[str, Any]]:
    """What a pooled accumulator currently resolves::

        {"planes", "T", "F", "band", "k_signal", "interval", "certified"}

    `certified` is True only when the interval has collapsed onto the count — the honest form of
    "how many modes are really there". Reporting `band` and `interval` beside it is required, not
    decorative: without them a caller can see only that certification has not happened, not how far
    away it is. The band shrinks as evidence accumulates, so a wide band calls for more evidence,
    never a smaller tolerance.

    `None` when the accumulator holds nothing — that is a different statement from a pooled read
    that resolved nothing, and the two must not share a value.
    """
    if acc is None:
        return None
    T = int(getattr(acc, "T", 0) or 0)
    if T <= 0:
        return None
    spec = acc.spectral()
    if spec.resolved_modes <= 0:
        # Pooled but not yet readable — the shape is still degenerate. Reported as a read that has
        # not happened, with the band that says how far off it is, never as a resolved zero.
        return {"planes": int(getattr(acc, "planes", 0)), "T": T, "F": int(acc.F),
                "band": spec.band, "k_signal": 0, "interval": (0, 0), "certified": False}
    lo = max(1, int(np.count_nonzero(spec.singular_values > spec.noise_floor)))
    hi = max(1, int(np.count_nonzero(spec.singular_values > spec.noise_floor_lo)))
    k = int(spec.resolved_modes)
    return {"planes": int(getattr(acc, "planes", 0)), "T": T, "F": int(acc.F), "band": spec.band,
            "k_signal": k, "interval": (lo, hi), "certified": lo == hi == k}


# ═══════════════════════════════════════════════════════════════════════════
# The energy band — arithmetic with no production caller in beacon
# ═══════════════════════════════════════════════════════════════════════════
# `_float_noise` has no production caller in beacon: beacon's routing and conservation code do not
# call it (see the module docstring for where that code lives). It stays because it is the subject of
# `tests/test_rounding_law_is_single_sourced.py`, which sweeps it against an independent body for bit
# equality, and because it is the reason `prism.rounding` is on the import-boundary test's allow-list,
# whose control set (`prism.conservation` must still fail to import) keeps that allowance from
# widening into a prefix. Deleting the function would retire both guards; a change that removes it
# must move those two tests in the same commit and say what now pins the prism edge.


def _float_noise(W: np.ndarray, energy: float, *, splits: int = 1) -> float:
    """The energy scale below which a difference is floating-point noise, not a measurement.

    An energy verdict may not be a typed level: this is the floor that decides whether a tekton
    couples at all — the routing decision itself — and [[capability-is-an-artifact-matched-by-
    propagation]] says that decision is measured. See the section comment above for why the law
    stays here despite having no caller in this module.

    This body is verbatim identical to `ember.optics._float_noise`, and `prism.conservation` states
    the same derivation a third time. Two implementations of one arithmetic required to agree
    exactly are two chances to disagree silently, so `tests/test_beacon_instrument.py` sweeps them
    against each other over a wide range of dtypes, shapes, magnitudes, and split counts:
    bit-identical, zero disagreements. The law lives once, in `prism.rounding`; this body is kept
    verbatim as an independent oracle against it — the arrangement that catches a divergence a
    single implementation cannot reveal on its own.

    Which error the law models, since a derivation can be as wrong as a constant: accumulation, not
    cancellation. The argument is specific rather than habitual: every term summed is a squared
    magnitude, hence non-negative, so the partial sums are monotonically increasing and no
    catastrophic cancellation is possible. The full argument is stated once, at `prism.rounding`,
    which is the only place it now needs stating.

    The edge this takes is precisely one module of prism's dependency-free base — no numpy, no
    cryptography, no extra, nothing this file could not already reach. That is why the law can live
    in prism and not stay duplicated here: `prism.conservation` needs numpy and sits behind `[wire]`,
    and a derivation behind an optional extra is a duplicate-drift risk: nothing forces the two
    copies to be checked against each other.
    `test_the_instrument_imports_with_beam_entroptics_and_prism_blocked` still blocks `beam` and
    `entroptics` outright and blocks every prism module except this one, so the edge is bounded by a
    measurement rather than by this sentence. What stays here is the one line prism cannot hold: `ε`
    read off the frame's own dtype, which is numpy vocabulary.
    """
    eps = float(np.finfo(W.dtype).eps) if np.issubdtype(W.dtype, np.floating) \
        else float(np.finfo(float).eps)
    return split_walk_rounding(int(W.size), energy, eps, splits=splits)

