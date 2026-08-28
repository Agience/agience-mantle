# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# Beacon is the permissive half of the two-tier model: mantle ships Apache so a
# store can be taken, built on and shipped by anyone, and beacon is the reduced
# instrument that makes such a store genuinely useful on its own. The downstream
# consumer's `beacon_engine.py` carries a proprietary-restriction notice instead —
# that is the Foresight white-label pilot, a different tree and a different
# arrangement.
# ---------------------------------------------------------------------------

"""Beacon matrix engine — the resolved-mode read. Self-contained, numpy only.

Given a matrix of row-vectors (embeddings, frames, whatever the pipeline already
has), the question this answers is how many independent directions it is really
made of. Everything beacon does downstream — the adaptive cut — is that number
applied to a different set of rows.

The answer is derived, not tuned. A matrix of pure noise still has a full singular
spectrum; what separates structure from noise is where the spectrum's top values sit
relative to the edge that noise alone would produce. That edge is a *distribution*,
not a constant, so the read is a hypothesis test at a stated false-alarm level:

    sigma^2   the de-biased per-cell noise variance, from the median row energy
              (robust — a few bright rows must not inflate the floor)
    mu, sig_J the finite-size Johnstone (2001) centring and scaling, so that
              (lambda_max - mu) / sig_J converges to Tracy-Widom_1
    q         the universal TW1 upper quantile at false-alarm level `far`
    floor     sqrt(sigma^2 * (mu + q * sig_J))
    signal rank  #(S > floor)

That derivation assumes an i.i.d. bulk: Tracy-Widom is the edge of an
i.i.d.-Gaussian ensemble. `_permutation_core` (used by `instrument.py`'s correlated-row
path, `ENGINE_ID_PERM`) answers the same question against a distribution-free permutation
null for rows that are correlated by construction, which needs no bulk model because it is
built from the caller's own matrix.

SIGNAL_RANK IS NOT A TOPIC-COUNT ESTIMATOR ON REAL TEXT. It is a noise-floor read, not a
semantic clustering criterion — do not borrow it for choosing an SVD rank meant to track
topics.

Public surface
--------------
    signal_rank(M, far=)      signal rank — the coherent dimension. The read.
    occupancy_fraction(M)            occupancy = 2^{H_sv}/n in (0,1] — the spectral fill
    shannon_bits(w)             H(w) in bits, the one definition
    whiten(M)                   per-channel robust MAD whitening
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ── the surface ─────────────────────────────────────────────────────────────
# `beacon.cut.select` is the one entry point Mantle itself takes (`search/mantle/engine.py`,
# lazily, at the moment a result set has to stop). `signal_rank` and `DEFAULT_FAR` are exported
# for callers outside this repo. `occupancy_fraction` is called from `cut.py`; `johnstone` and
# `tw1_quantile` are called from `instrument.py`. `whiten` and `tw1_sf` have no caller outside
# this module: `whiten` is used here at the floor computation and by `tests/test_beacon_live_width.py`
# (`instrument.py`'s `_whiten` is a separate private helper, not this one), and `tw1_sf` has none
# at all.
#
# CORRECTED 2026-08-25, and the correction is the point. This comment previously said `whiten`
# was called from `cut.py` and that `proximity.py` called `johnstone`/`tw1_quantile`. Measured:
# `cut.py` does not reference `whiten`, and `beacon/proximity.py` no longer exists in this repo —
# it moved upstream, to the instrument's own package. A note about who calls what goes stale
# silently, because nothing fails when a caller leaves.
#
# The export list is exactly the beacon cut and nothing else; the rest remain importable by
# path for tests inside this package, but they are not part of the contract and must not be
# relied on from outside. A published export is a promise: every name here is something a
# third party may build on and therefore something that cannot change without breaking them.
# Beacon is the permissive half of a two-tier model and is meant to be built on, which makes
# a narrow, deliberate surface more important, not less.
__all__ = [
    "BeaconEngineError", "DEFAULT_FAR", "ENGINE_ID", "ENGINE_ID_PERM",
    "RankResult", "signal_rank",
]

#: Which instrument produced a read. Every read names it, so a number from one engine can
#: never be mistaken for a number from another. Says nothing about what any other
#: implementation is called.
ENGINE_ID = "beacon.tw1"

#: The permutation-null read's instrument. A different null is a different instrument: the two
#: reads answer the same question against different assumptions about the rows, and a caller
#: handed a bare integer must still be able to tell which one it holds. See `_permutation_core`,
#: used by `instrument.py`'s correlated-row path.
ENGINE_ID_PERM = "beacon.perm"


class BeaconEngineError(RuntimeError):
    """A matrix read could not be performed. Never returned as a value.
    """


#: Default false-alarm level. The one input to the floor, and it is a policy choice
#: (how often are we willing to call noise "structure"?), not a tuned constant.
DEFAULT_FAR = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Tracy-Widom_1 (GOE / real matrices): universal edge quantiles + tail
# ═══════════════════════════════════════════════════════════════════════════

#: TW1 upper quantiles: P(TW1 <= q) = 1 - far. Universal constants of the TW1
#: distribution (Bejan 2005; Chiani 2014) — fixed properties of a limit law, not
#: values fit to this corpus. These are the only numeric constants in the floor.
_TW1_UPPER_Q: dict[float, float] = {0.10: 0.4501, 0.05: 0.9793,
                                    0.025: 1.3675, 0.01: 2.0234}

# TW1 survival via Chiani (2014)'s Gamma approximation, moment-matched to TW1's
# mean/variance/skewness (max CDF error ~7e-3). The TW1 CDF has no closed form; this
# gives p_k = P(TW1 > g_k) with no scipy dependency, deterministically.
_TW1_G_K = 46.44580      # Gamma shape    = 4 / skew^2
_TW1_G_TH = 0.1861300    # Gamma scale    = sqrt(var) * skew / 2
_TW1_G_LOC = -9.848007   # Gamma location = mean - shape * scale

_LANCZOS = (0.99999999999980993, 676.5203681218851, -1259.1392167224028,
            771.32342877765313, -176.61502916214059, 12.507343278686905,
            -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7)


def _gammaln(x: float) -> float:
    """log Gamma(x) for x > 0, Lanczos approximation (g = 7)."""
    x -= 1.0
    a = _LANCZOS[0]
    t = x + 7.5
    for i in range(1, 9):
        a += _LANCZOS[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _reg_gamma_upper(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x), x >= 0.

    Numerical Recipes: the series converges for x < a+1, the continued fraction
    above it. Using either outside its range loses precision silently."""
    if x <= 0.0:
        return 1.0
    gln = _gammaln(a)
    if x < a + 1.0:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(400):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    tiny = 1e-300
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 400):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def tw1_sf(g: float) -> float:
    """P(TW1 > g) — the Tracy-Widom_1 upper-tail probability."""
    return _reg_gamma_upper(_TW1_G_K, (g - _TW1_G_LOC) / _TW1_G_TH)


def _tw1_quantile_invert(far: float) -> float:
    """The q with tw1_sf(q) = far, by bisection on the monotone survival function.

    This is what lets `far` be sharpened arbitrarily (1e-5 and beyond) without a
    tabulated value — the edge stays derived rather than looked up."""
    lo, hi = -10.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tw1_sf(mid) > far:            # tail too heavy -> need a larger quantile
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tw1_quantile(far: float) -> float:
    """TW1 upper quantile q with P(TW1 <= q) = 1 - far.

    Exact tabulated values at the standard levels; any other level is inverted from
    the survival function."""
    if far in _TW1_UPPER_Q:
        return _TW1_UPPER_Q[far]
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    return _tw1_quantile_invert(far)


# ═══════════════════════════════════════════════════════════════════════════
# Finite-size Johnstone edge + de-biased per-cell noise variance
# ═══════════════════════════════════════════════════════════════════════════

def johnstone(N: int, F: int) -> tuple[float, float]:
    """Johnstone (2001) centring `mu` and scaling `sigma_J` for the largest eigenvalue
    (top singular value squared) of an N x F Gaussian matrix, so that
    ``(lambda_max - mu) / sigma_J -> Tracy-Widom_1``.

    The derived finite-size edge, not a fitted coefficient."""
    nn = math.sqrt(max(N - 1, 1))
    ff = math.sqrt(max(F, 1))
    a = nn + ff
    return a * a, a * (1.0 / nn + 1.0 / ff) ** (1.0 / 3.0)


def _debias_denominator(N: int, F: int) -> float:
    """The denominator `F * c_F * dof` turning a median row energy into sigma^2.

    Two corrections, both because the median is what makes the estimate robust:

      c_F       the sample median of ||row||^2 estimates the distribution median of a
                chi^2_F (= F * c_F), not its mean F — a Wilson-Hilferty small-F bias;
      (N-1)/N   per-channel centring deflates row energy. The mean-centring dof is
                applied as a conservative correction: it slightly over-corrects at
                small N, which raises the floor — the safe direction.
    """
    c_F = (1.0 - 2.0 / (9.0 * max(int(F), 1))) ** 3
    dof = max(int(N) - 1, 1) / int(N)
    return int(F) * c_F * dof


def _noise_sigma2(matrix: np.ndarray, N: int, F: int) -> float:
    """De-biased robust per-cell noise variance.

    The median row energy, not the mean: a handful of genuinely bright rows is exactly
    the signal we are trying to detect, and letting them raise the floor is how a
    detector talks itself out of its own findings."""
    row_energy = np.sum(np.abs(matrix) ** 2, axis=1)
    return float(np.median(row_energy)) / _debias_denominator(N, F) + 1e-30


def _floor_sq(sigma2: float, N: int, F: int, far: float) -> float:
    """The floor squared, in eigenvalue units: sigma^2 * (mu + q * sigma_J)."""
    mu, sig_J = johnstone(int(N), int(F))
    return sigma2 * (mu + tw1_quantile(far) * sig_J)


def live_width(A: np.ndarray) -> int:
    """The number of feature channels the spectrum is actually read over — columns that are
    not identically zero after whitening.

      * `_noise_sigma2` divides the median row energy by a denominator in F. Dead columns add
        no energy but do add F, so sigma^2 is under-estimated and the floor sinks: noise
        gets read as signal.
      * `johnstone(N, F)` returns the edge of a wider ensemble than was measured, which lifts
        the floor: signal gets read as noise.

    Zero live channels floors at 1: `johnstone` needs a non-degenerate shape, and a frame with
    nothing in it is reported through `degraded`, not through a fabricated width."""
    return max(1, int(np.count_nonzero(np.any(A != 0.0, axis=0))))


def _live_view(A: np.ndarray) -> np.ndarray:
    """`A` restricted to the rows and columns that carry anything — dead lines DROPPED, not zeroed.

    A dead channel is not a measurement of zero; it is the absence of a measurement, and the two
    must not read the same. Zeroing keeps it in every shape the read is sized by — the `F` the
    null is drawn at, the `n` an occupancy is a fraction of — so a frame stored wider than it was
    measured reads differently from the same frame stored tight. Dropping removes it from the
    shape as well as the energy, which is the only handling that leaves the read invariant to how
    the caller happened to lay the frame out.

    Rows are dropped on the same rule as columns: an all-zero row is an observation that was not
    taken, and it inflates `N` in exactly the way a dead column inflates `F`.

    A frame with nothing live returns empty rather than a fabricated line. Callers report that
    through `degraded`; they must not receive a shape that implies a measurement."""
    if A.ndim != 2 or A.size == 0:
        return A
    live_r = np.any(A != 0.0, axis=1)
    live_c = np.any(A != 0.0, axis=0)
    if live_r.all() and live_c.all():
        return A
    return A[np.ix_(live_r, live_c)]


# ═══════════════════════════════════════════════════════════════════════════
# Per-channel robust whitening — what makes the noise floor a clean reference
# ═══════════════════════════════════════════════════════════════════════════

#: MAD -> Gaussian sigma. Exact 1/Phi^{-1}(0.75). Derived, not fitted.
MAD_SCALE = 1.482602218505602

#: Asymptotic sampling variance of log(MAD-hat) from N Gaussian samples, ~ MAD_LOGVAR/N.
#: The analytic influence-function variance 1/(16 f(D)^2 D^2) at D = Phi^{-1}(3/4).
#: Derived like MAD_SCALE; sets the shrinkage below.
MAD_LOGVAR = 1.36046


def _shrink_mad(mad: np.ndarray, pos: np.ndarray, typical: float, N: int,
                eps: float) -> np.ndarray:
    """James-Stein shrinkage of each channel's MAD toward the pooled scale.

    A per-channel MAD estimated from N rows is noisy, with log-sampling-variance
    ~ MAD_LOGVAR/N. Left alone, that noise disperses the whitened matrix and inflates
    the floor — the detector talks itself out of its own findings on small N.

    Shrink each channel's log-MAD toward the pooled log-scale by the data-derived
    weight ``w = max(0, 1 - V_samp/V_obs)`` (empirical Bayes): homoscedastic channels
    (V_obs ~ V_samp — iid noise, few rows) collapse to one stable pooled scale, while
    genuinely different channels (V_obs >> V_samp) keep full per-channel whitening.
    Parameter-free."""
    if typical <= eps:
        return mad
    lm = np.log(np.maximum(mad, eps))
    lm0 = math.log(typical)
    if int(pos.sum()) > 1:
        V_obs = float(np.std(lm[pos])) ** 2
        w = 0.0 if V_obs <= 0.0 else max(0.0, min(1.0, 1.0 - (MAD_LOGVAR / max(N, 1)) / V_obs))
    else:
        w = 0.0
    return np.exp(lm0 + w * (lm - lm0))


def whiten(M) -> np.ndarray:
    """Per-channel robust (MAD) whitening — every channel to a common unit noise scale.

    Each channel's median is subtracted, then it is divided by a shrunk robust sigma
    (MAD * MAD_SCALE, pooled per :func:`_shrink_mad`). Channels with no spread are
    zeroed rather than amplified — dividing a dead channel by its ~0 scale is how noise
    becomes infinite signal.
    """
    A = _as_matrix(M)
    eps = 1e-12
    med = np.median(A, axis=0)
    centered = A - med[None, :]
    mad = np.median(np.abs(centered), axis=0) * MAD_SCALE
    pos = mad > eps
    typical = float(np.median(mad[pos])) if bool(pos.any()) else 1.0
    mad_eff = _shrink_mad(mad, pos, typical, int(A.shape[0]), eps)
    safe = mad_eff > eps
    return np.where(safe[None, :], centered / np.where(safe, mad_eff, 1.0)[None, :], 0.0)


# The corpus-A axes are nominal on both sides. The matrix is read at native resolution.
def _as_matrix(M) -> np.ndarray:
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2:
        raise BeaconEngineError(f"the input must be 2-D; got shape {A.shape}")
    if A.size == 0:
        raise BeaconEngineError("the input must be non-empty")
    if not np.all(np.isfinite(A)):
        # Scattered gaps are filled with the column mean (0 after centring) rather
        # than propagating NaN into an SVD, where it poisons every mode at once.
        A = np.where(np.isfinite(A), A, np.nan)
        col = np.nanmean(A, axis=0)
        col = np.where(np.isfinite(col), col, 0.0)
        A = np.where(np.isfinite(A), A, col)
    return A


@dataclass(frozen=True)
class RankResult:
    """signal rank, and everything a caller needs to know about how it was obtained.

    `k` alone is not a safe return value: a collapsed matrix and a genuinely one-mode
    matrix both read 1, and that confusion is the exact failure this engine's own error
    class was written about. So the read carries its `instrument` and its `degraded`.

    Behaves as an int where `signal_rank` is consumed (`int(read)`, comparisons), so a
    caller keeps working, with the hazard available to it rather than hidden behind the
    plain integer.
    """
    k: int
    live_channels: int
    degraded: bool
    instrument: str = ENGINE_ID

    def __int__(self) -> int:
        return int(self.k)

    def __index__(self) -> int:
        return int(self.k)


@dataclass(frozen=True)
class _TW1Core:
    """Everything the Tracy-Widom read measured, before it was collapsed to a count.

    Extracted rather than duplicated: `signal_rank` collapses this to `k`;
    `beacon.instrument.read_ordered` needs the spectrum, the floor and the noise variance to report
    the continuous evidence behind that integer. Two functions computing one floor two ways is how
    two callers start disagreeing about where noise stops — the same reason
    `route_by_coupling` calls `next_by_coupling` rather than re-implementing it. One rule.
    """
    A: np.ndarray          #: the whitened matrix the spectrum is read over
    N: int                 #: rows
    F: int                 #: the live feature width the null is sized by
    live_channels: int
    degraded: bool         #: the collapsed-axis hazard (<= 1 live channel)
    readable: bool         #: False when the shape is degenerate — `min(N, F) < 2`
    sv: np.ndarray         #: singular values, descending (empty when not readable)
    sigma2: float          #: the de-biased per-cell noise variance (0.0 when not readable)


def _tw1_core(M, far: float = DEFAULT_FAR) -> _TW1Core:
    """Whiten, decompose, and size the null by the live width — the one Tracy-Widom read."""
    A = whiten(M)                         # the floor is only meaningful on unit-noise channels
    # A channel `whiten` could not scale is returned as all-zeros; the live count is the
    # effective feature width the spectrum is actually read over, and the width the null is
    # sized by. See `live_width` for why counting dead columns biases the floor in two
    # opposing directions at once.
    live_channels = int(np.count_nonzero(np.any(A != 0.0, axis=0)))
    N, F = int(A.shape[0]), live_width(A)
    hazard = live_channels <= 1
    if min(N, F) < 2:                     # genuinely degenerate
        return _TW1Core(A=A, N=N, F=F, live_channels=live_channels, degraded=hazard,
                        readable=False, sv=np.zeros(0), sigma2=0.0)
    try:
        sv = np.linalg.svd(A, compute_uv=False)
    except BeaconEngineError:
        raise
    except Exception as exc:              # LinAlgError (non-convergent SVD) and friends
        raise BeaconEngineError(
            f"spectral read failed on a {A.shape} frame ({type(exc).__name__}: {exc})"
        ) from exc
    return _TW1Core(A=A, N=N, F=F, live_channels=live_channels, degraded=hazard,
                    readable=True, sv=sv, sigma2=_noise_sigma2(A, N, F))


def signal_rank(M, *, far: float = DEFAULT_FAR) -> "RankResult":
    """signal rank — how many singular values of `M` stand above the derived noise floor.

    The read. `k` is always >= 1: a matrix the engine can process has at least one
    direction, and returning 0 would leave every downstream subspace empty rather than
    coarse. The result's `degraded` flag is what tells a destroyed basis apart from a
    clean read of 1.
    """
    core = _tw1_core(M, far)
    if not core.readable:
        return RankResult(k=1, live_channels=core.live_channels, degraded=core.degraded)
    floor = math.sqrt(_floor_sq(core.sigma2, core.N, core.F, far))
    k = int(np.count_nonzero(core.sv > floor))
    return RankResult(k=max(1, k), live_channels=core.live_channels, degraded=core.degraded)


# ═══════════════════════════════════════════════════════════════════════════
# The permutation null — the read for rows that are correlated by construction
# ═══════════════════════════════════════════════════════════════════════════
# `signal_rank`'s Johnstone/Tracy-Widom floor is derived for an i.i.d.-Gaussian bulk.
# `beam.optics.read_ordered` states the domain rule in its own docstring: that edge is
# "correct for an i.i.d. bulk and optimistic for correlated rows. For retrieved evidence
# (correlated by construction) pass `correlated_null()`", the distribution-free permutation
# null the library directs correlated callers to. A term-document matrix is correlated by
# construction — documents share vocabulary and a topic is a correlation among rows — so
# applying the Tracy-Widom floor to one is a read taken outside the domain its edge was
# derived for. `_permutation_core` below is the read for that domain (reached via
# `instrument.py`'s correlated-row path, `ENGINE_ID_PERM`): the same question against a
# distribution-free permutation null, built from the caller's own matrix rather than a
# Gaussian model, so it needs no bulk assumption and collapses when the structure does.
#
# The substitute is not a new mechanism: a permutation null is the shuffled matrix, the
# same instrument the upstream library uses (`null_providers.shuffle_in_time`: "shuffle each
# column independently along the ordered axis, destroying ordered and cross-channel
# structure while preserving every channel's own marginal"). This function reproduces
# that shuffle, argsort trick included, written numpy-only because beacon is the
# dependency-free embodiment and imports no spectral library — so the two
# embodiments draw from the same null.
#
# `signal_rank` keeps its own job: it counts singular values above an i.i.d. noise floor,
# correctly, for rows that genuinely are i.i.d. — the closed-form edge is exact, O(1), and
# sharpens to any `far` without spending compute. Rows correlated by construction are a
# domain problem, so they get a second instrument with its own name, not a change of meaning
# under an existing one.


def _permutation_draws(far: float) -> int:
    """The smallest number of surrogate draws at which the level `far` is attainable.

    A permutation p-value is `(1 + #{surrogate >= observed}) / (1 + B)` (Phipson & Smyth 2010 —
    the +1 is what keeps it a valid p-value rather than an underestimate that can reach 0). Its
    smallest possible value is therefore `1/(1 + B)`, so a test at level `far` cannot pass unless
    `B >= 1/far - 1`. That inequality is the whole derivation; there is no choice in it, and at
    the default `far = 0.05` it gives B = 19.

    More draws is compute, not modelling, and this engine does not guess a budget. `beam.optics
    .correlated_null` sets out the same position at length: draws fix the finest resolvable p and
    cost CPU linearly, so how many a caller may afford is a question for a measured resource
    envelope and for nothing else, and nothing on this box publishes a per-draw cost or a
    wall-clock allowance. So the default is the only value that is derived rather than invented:
    the minimum at which the stated level can be reached. A caller that has measured its own
    budget passes `draws` and gets a more powerful test at the same level."""
    if not (0.0 < far < 1.0):
        raise ValueError(f"far must be in (0, 1); got {far}")
    return max(1, int(math.ceil(1.0 / far - 1.0)))


def _column_shuffle(A: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One surrogate draw: every column independently permuted down the rows.

    Preserves each channel's marginal exactly (a permutation moves values, it never alters them)
    and destroys every cross-row relationship — which is what a topic, a shared cause, or any
    other correlation among the rows lives in.

    One `argsort` draws all F independent permutations at once; this is the upstream
    `shuffle_in_time` verbatim in mechanism, so the two embodiments sample the same null."""
    n, F = int(A.shape[0]), int(A.shape[1])
    idx = np.argsort(rng.random((n, F)), axis=0)
    return np.take_along_axis(A, idx, axis=0)


@dataclass(frozen=True)
class _PermCore:
    """Everything the permutation read measured, before it was collapsed to a count.

    Extracted rather than duplicated, for the reason stated on `_TW1Core`:
    `beacon.instrument.read_ordered` must report the p-value each component's decision was taken
    on, and a second implementation of the leading-run rule is how the module surface and the
    embodiment surface would start naming different ranks while both looked right.
    """
    C: np.ndarray          #: the mean-centred matrix the null is drawn against
    live_channels: int
    degraded: bool
    readable: bool
    s: np.ndarray          #: the centred matrix's singular values, descending
    pvalue: np.ndarray     #: the permutation p-value per component
    surrogate_top: np.ndarray   #: each draw's leading singular value — the empirical edge
    draws: int             #: B
    offset: int            #: 1 when the untested mean direction is a real direction of `M`
    k_tested: int          #: the leading RUN of components with p <= far, before the offset


def _permutation_core(M, *, far: float = DEFAULT_FAR, draws: int | None = None,
                      seed: int = 0) -> _PermCore:
    """Centre, draw the marginal-preserving surrogates, and score every component."""
    A = _as_matrix(M)
    mean = A.mean(axis=0, keepdims=True)
    C = A - mean
    # The mean component is a rank-1 direction of `M` with singular value sqrt(N) * ||mean||.
    # Whether it is a direction or floating-point residue is numpy's own question, asked with
    # numpy's own answer: `matrix_rank` calls a singular value zero below `max(N, F) * eps *
    # s_max`. That tolerance is a property of the number system, not of this corpus: centring a
    # matrix by hand leaves a mean of ~1e-17, which a bare `> 0.0` test would count as a
    # direction that is not there.
    #
    # This is a rank tolerance — the representation of a zero singular value after the SVD has
    # rounded — not the accumulation bound `_float_noise` takes from `prism.rounding`. Nothing
    # accumulates here: there is no running total, only the question of whether one number is
    # distinguishable from zero at this matrix's scale. The two share the letter `eps` and answer
    # different questions, so merging them would be worse than keeping both.
    # `s_max` enters through `scale`, so the test is relative to the matrix's own magnitude.
    # A column that is constant centres to zero: it carries no structure a shuffle could destroy,
    # and it is not a channel the spectrum is read over. Same meaning as `live_width`, measured on
    # the matrix this read actually decomposes.
    live_channels = int(np.count_nonzero(np.any(C != 0.0, axis=0)))
    hazard = live_channels <= 1
    N = int(C.shape[0])
    empty = np.zeros(0)
    if min(N, live_channels) < 2:
        return _PermCore(C=C, live_channels=live_channels, degraded=hazard, readable=False,
                         s=empty, pvalue=empty, surrogate_top=empty, draws=0, offset=0, k_tested=0)
    B = _permutation_draws(far) if draws is None else max(1, int(draws))
    rng = np.random.default_rng(seed)
    try:
        s = np.linalg.svd(C, compute_uv=False)
        atleast = np.zeros(s.shape[0], dtype=np.int64)
        tops = np.zeros(B, dtype=np.float64)
        for b in range(B):
            sb = np.linalg.svd(_column_shuffle(C, rng), compute_uv=False)
            atleast += (sb >= s)
            tops[b] = float(sb[0]) if sb.size else 0.0
    except BeaconEngineError:
        raise
    except Exception as exc:              # LinAlgError (non-convergent SVD) and friends
        raise BeaconEngineError(
            f"permutation read failed on a {C.shape} frame ({type(exc).__name__}: {exc})"
        ) from exc
    mean_sv = math.sqrt(N) * float(np.linalg.norm(mean))
    scale = max(float(s[0]) if s.size else 0.0, mean_sv)
    offset = 1 if mean_sv > max(N, int(C.shape[1])) * np.finfo(np.float64).eps * scale else 0
    pvalue = (1.0 + atleast) / (1.0 + B)
    failed = np.nonzero(pvalue > far)[0]
    # The leading run, not the total count: a component that fails is where the evidence stops, and
    # a later one clearing the level past it is the multiple-comparison artifact this ordering
    # exists to prevent.
    k_tested = int(failed[0]) if failed.size else int(s.shape[0])
    return _PermCore(C=C, live_channels=live_channels, degraded=hazard, readable=True, s=s,
                     pvalue=pvalue, surrogate_top=tops, draws=B, offset=offset, k_tested=k_tested)


def shannon_bits(weights) -> float:
    """H(w) in bits over a non-negative weight array. H = -sum p log2 p, p = w / sum w.
    One definition, used by every entropy read here, so two callers cannot drift."""
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 1e-30:
        return 0.0
    p = w / total
    return float(-np.sum(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1.0)), 0.0)))


def occupancy_fraction(M) -> float:
    """occupancy = 2^{H_sv} / n in (0, 1] — the fraction of active modes of a matrix.

    The Shannon entropy of the singular spectrum, exponentiated: the effective number
    of modes carrying the energy, as a fraction of the modes available. Bounded by
    construction. occupancy -> 1 fully disordered (every mode active); occupancy -> 0 fully
    coherent (one mode carries everything).

    Distinct from `signal_rank`, and both are wanted: occupancy is a smooth measure of how
    spread the energy is on a heteroscedastic matrix, signal rank a hard count of how much
    of it clears noise.

    Dead lines are dropped, not counted. A channel that is identically zero carries no energy,
    so it contributes no singular value — but it still contributes to `n`, the count of modes
    AVAILABLE. Left in, it depresses the ratio, and a frame reads as more coherent purely
    because it was stored at a wider stride than it was measured at. That is the same bias
    `live_width` describes for the floor, arriving here through the denominator instead.

    This is also what makes the read agree with the aperture's `phi`, which restricts to its
    own live view before decomposing. Before the drop the two returned different numbers on any
    frame with a dead channel, silently, and this one feeds `derive_heads` -> the head count ->
    the cut."""
    A = _live_view(_as_matrix(M))
    if A.size == 0:
        return 0.0
    sv = np.linalg.svd(A, compute_uv=False)
    n = int(sv.shape[0])
    if n == 0:
        return 0.0
    return float(2.0 ** shannon_bits(sv ** 2) / n)
