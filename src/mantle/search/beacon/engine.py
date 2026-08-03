# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc. All rights reserved.
#
# CONFIDENTIAL — CONTAINS TRADE SECRETS OF IKAILO INC.
#
# This file and the BEACON algorithms it implements are the confidential and
# proprietary trade-secret property of Ikailo Inc., disclosed only to authorized
# parties under obligation of confidentiality. No part may be used, reproduced,
# modified, distributed, or disclosed without the prior written consent of
# Ikailo Inc. Unauthorized use or disclosure is prohibited and may violate
# trade-secret, copyright, and contract law.
# ---------------------------------------------------------------------------

"""BEACON matrix engine — the resolved-mode read. Self-contained, numpy only.

THE ONE QUESTION THIS ANSWERS: given a matrix (a matrix of row-vectors — embeddings,
frames, whatever the pipeline already has), **how many independent directions is it
really made of?** Everything BEACON does downstream — the adaptive cut, novelty,
grounding — is that number applied to a different set of rows.

The answer is derived, never tuned. A matrix of pure noise still has a full singular
spectrum; what separates structure from noise is where the spectrum's top values sit
relative to the edge that noise alone would produce. That edge is a *distribution*,
not a constant, so the read is a hypothesis test at a stated false-alarm level:

    sigma^2   the de-biased per-cell noise variance, from the MEDIAN row energy
              (robust — a few bright rows must not inflate the floor)
    mu, sig_J the finite-size Johnstone (2001) centring and scaling, so that
              (lambda_max - mu) / sig_J converges to Tracy-Widom_1
    q         the universal TW1 upper quantile at false-alarm level `far`
    floor     sqrt(sigma^2 * (mu + q * sig_J))
    signal rank  #(S > floor)

Public surface
--------------
    signal_rank(M, far=)      signal rank — the coherent dimension. THE read.
    noise_floor(M, far=)        the singular-value floor it thresholds
    component_significance(M)        per-mode (deviate, pvalue) — threshold-free evidence
    occupancy_fraction(M)            occupancy = 2^{H_sv}/n in (0,1] — the spectral fill
    shannon_bits(w)             H(w) in bits, the one definition
    spectrum_stats(M)             sigma_top / floor / signal rank / U,S,Vt in one pass
    whiten(M)                   per-channel robust MAD whitening
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# ── THE SURFACE ─────────────────────────────────────────────────────────────
# MEASURED 2026-07-30: of the twelve symbols this module used to export, exactly ONE
# (`signal_rank`, with `DEFAULT_FAR`) is imported anywhere outside this package —
# `mantle/search/anchors/density.py`. `spectrum_stats`, `noise_floor`, `component_significance`,
# `occupancy_fraction`, `whiten`, `johnstone`, `tw1_quantile`, `tw1_sf` and `ComponentSignificance`
# had ZERO external callers.
#
# This package is CONFIDENTIAL (see the header), so every exported name is disclosed
# surface, not merely public API. The export list is therefore exactly the beacon cut and
# nothing else; the rest remain importable by path for tests inside this package, but they
# are not part of the contract and must not be relied on from outside.
# [John, 2026-07-30: "only imports the methods that matter. the API exposure and code
#  exposure is limited to just what's required for the beacon cut."]
__all__ = [
    "BeaconEngineError", "DEFAULT_FAR", "ENGINE_ID",
    "RankResult", "signal_rank",
]

#: Which instrument produced a read. Every read names it, so a number from one engine can
#: never be mistaken for a number from another. [John, 2026-07-30: "every read names its
#: instrument."] Deliberately says nothing about what any other implementation is called.
ENGINE_ID = "beacon.tw1"


class BeaconEngineError(RuntimeError):
    """A matrix read could not be performed. Never returned as a value.

    ⛔ THE ENGINE NEVER DEGRADES TO A PLAUSIBLE NUMBER. `signal_rank` returning 1
    on failure would be indistinguishable from a genuine one-mode matrix, and that
    exact confusion once collapsed every coherent subspace in the pipeline while every
    health signal stayed green. A read either answers or raises.
    """


#: Default false-alarm level. The ONE input to the floor, and it is a policy choice
#: (how often are we willing to call noise "structure"?), not a tuned constant.
DEFAULT_FAR = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# Tracy-Widom_1 (GOE / real matrices): universal edge quantiles + tail
# ═══════════════════════════════════════════════════════════════════════════

#: TW1 UPPER quantiles: P(TW1 <= q) = 1 - far. UNIVERSAL constants of the TW1
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

    The DERIVED finite-size edge — no fitted coefficient. This is what replaced the
    empirical `0.18 / sqrt(max(N, F))` correction of the first implementation."""
    nn = math.sqrt(max(N - 1, 1))
    ff = math.sqrt(max(F, 1))
    a = nn + ff
    return a * a, a * (1.0 / nn + 1.0 / ff) ** (1.0 / 3.0)


def _debias_denominator(N: int, F: int) -> float:
    """The denominator `F * c_F * dof` turning a median row energy into sigma^2.

    Two corrections, both because the MEDIAN is what makes the estimate robust:

      c_F       the sample median of ||row||^2 estimates the DISTRIBUTION median of a
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

    The MEDIAN row energy, not the mean: a handful of genuinely bright rows is exactly
    the signal we are trying to detect, and letting them raise the floor is how a
    detector talks itself out of its own findings."""
    row_energy = np.sum(np.abs(matrix) ** 2, axis=1)
    return float(np.median(row_energy)) / _debias_denominator(N, F) + 1e-30


def _floor_sq(sigma2: float, N: int, F: int, far: float) -> float:
    """The floor SQUARED, in eigenvalue units: sigma^2 * (mu + q * sigma_J)."""
    mu, sig_J = johnstone(int(N), int(F))
    return sigma2 * (mu + tw1_quantile(far) * sig_J)


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

    A per-channel MAD estimated from N rows is NOISY, with log-sampling-variance
    ~ MAD_LOGVAR/N. Left alone, that noise disperses the whitened matrix and inflates
    the floor — the detector talks itself out of its own findings on small N.

    Shrink each channel's log-MAD toward the pooled log-scale by the DATA-DERIVED
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

    ⚠ THIS IS WHAT MAKES THE FLOOR MEANINGFUL. The Johnstone/TW1 edge is the edge of an
    iid Gaussian ensemble. Feeding it raw channels with wildly different scales tests
    the data against a null it does not resemble, and the answer is arbitrary: one
    high-variance dimension dominates the spectrum and reads as "structure" no matter
    what it contains. Whitening first is what makes "above the noise floor" mean
    "above THIS matrix's noise", rather than "in the biggest channel".

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


# EREA's axes are nominal on both sides. The matrix is read at native resolution.
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


def noise_floor(M, *, far: float = DEFAULT_FAR) -> float:
    """The singular-value noise floor for matrix `M`. `signal rank = #(S > floor)`.

    Measured on the WHITENED matrix — see :func:`whiten` for why the raw one is not a
    meaningful reference."""
    A = whiten(M)
    N, F = A.shape
    return math.sqrt(_floor_sq(_noise_sigma2(A, N, F), N, F, far))


@dataclass
class ComponentSignificance:
    """Per-singular-value evidence against the noise null, carrying NO threshold.

    The resolved dimension at any false-alarm level is `#(pvalue < far)` — the read
    reports evidence, the reader supplies the decision level."""
    deviate: np.ndarray   # g_k = (s_k^2 / sigma^2 - mu) / sigma_J, standardized TW1
    pvalue: np.ndarray    # p_k = P(TW1 > g_k), upper-tail probability


def component_significance(M, s: np.ndarray | None = None) -> ComponentSignificance:
    """Per-mode significance of the matrix's spectrum against the derived noise null.

    ⚠ `#(p_k < far)` and `signal_rank(far=...)` agree EXACTLY when the quantile is
    inverted from this same survival function. At a tabulated `far` the floor uses the
    exact table value while `p_k` uses the Chiani approximation, so a mode sitting
    within the ~7e-3 approximation gap can differ by one count. Documented rather than
    papered over: the two are different questions asked of the same evidence."""
    A = whiten(M)                         # same matrix the floor is measured on
    N, F = A.shape
    sv = np.linalg.svd(A, compute_uv=False) if s is None else np.asarray(s, dtype=float)
    if sv.size == 0:
        empty = sv[:0].copy()
        return ComponentSignificance(deviate=empty, pvalue=empty)
    mu, sig_J = johnstone(N, F)
    sigma2 = _noise_sigma2(A, N, F)
    g = (sv ** 2 / sigma2 - mu) / sig_J
    p = np.array([tw1_sf(float(gk)) for gk in g])
    return ComponentSignificance(deviate=g, pvalue=p)


@dataclass(frozen=True)
class RankResult:
    """signal rank, and everything a caller needs to know about how it was obtained.

    `k` alone is not a safe return value: a collapsed matrix and a genuinely one-mode
    matrix both read 1, and that confusion is the exact failure this engine's own error
    class was written about. So the read carries its `instrument` and its `degraded`.

    Behaves as an int in the places `signal_rank` used to (`int(read)`, comparisons),
    so an existing caller keeps working — but it can no longer do so WITHOUT the hazard
    being available to it.
    """
    k: int
    live_channels: int
    degraded: bool
    instrument: str = ENGINE_ID

    def __int__(self) -> int:
        return int(self.k)

    def __index__(self) -> int:
        return int(self.k)


def signal_rank(M, *, far: float = DEFAULT_FAR) -> "RankResult":
    """signal rank — how many singular values of `M` stand above the derived noise floor.

    THE read. `k` is always >= 1: a matrix the engine can process has at least one
    direction, and returning 0 would leave every downstream subspace empty rather than
    coarse.

    ⛔ Raises rather than falling back. A degenerate matrix (fewer than two rows or
    columns) is decided BEFORE any spectral work, so anything raised from here is a
    genuine fault, not a small input.

    ⚠ `degraded` — 2026-07-30. `whiten` zeroes channels with
    no spread rather than amplifying them, so a frame whose scale is carried by one channel
    arrives here with a COLLAPSED feature axis, and the spectrum of a collapsed matrix is
    not the spectrum of the data. Measured across 60 adversarial frames, an
    amplification test for this fires ZERO times — including on frames where the axis had
    collapsed to one live channel and the read came back confidently wrong. The collapse is
    therefore tested DIRECTLY (`live_channels <= 1`), which is the test that works. Without
    this flag a destroyed basis reads as a clean number."""
    A = whiten(M)                         # the floor is only meaningful on unit-noise channels
    N, F = A.shape
    # A channel `whiten` could not scale is returned as all-zeros; the live count is the
    # effective feature width the spectrum is actually read over.
    live_channels = int(np.count_nonzero(np.any(A != 0.0, axis=0)))
    hazard = live_channels <= 1
    if min(N, F) < 2:
        return RankResult(k=1, live_channels=live_channels, degraded=hazard)   # genuinely degenerate
    try:
        sv = np.linalg.svd(A, compute_uv=False)
        floor = math.sqrt(_floor_sq(_noise_sigma2(A, N, F), N, F, far))
        k = int(np.count_nonzero(sv > floor))
    except BeaconEngineError:
        raise
    except Exception as exc:              # LinAlgError (non-convergent SVD) and friends
        raise BeaconEngineError(
            f"spectral read failed on a {A.shape} frame ({type(exc).__name__}: {exc})"
        ) from exc
    return RankResult(k=max(1, k), live_channels=live_channels, degraded=hazard)


def shannon_bits(weights) -> float:
    """H(w) in bits over a non-negative weight array. H = -sum p log2 p, p = w / sum w.

    ONE definition, used by every entropy read here, so two callers cannot drift."""
    w = np.asarray(weights, dtype=np.float64)
    total = float(w.sum())
    if total <= 1e-30:
        return 0.0
    p = w / total
    return float(-np.sum(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1.0)), 0.0)))


def occupancy_fraction(M) -> float:
    """occupancy = 2^{H_sv} / n in (0, 1] — the fraction of ACTIVE modes of a matrix.

    The Shannon entropy of the singular spectrum, exponentiated: the effective number
    of modes carrying the energy, as a fraction of the modes available. Bounded by
    construction. occupancy -> 1 fully disordered (every mode active); occupancy -> 0 fully
    coherent (one mode carries everything).

    Distinct from `signal_rank`, and both are wanted: occupancy is a smooth measure of how
    SPREAD the energy is, signal rank a hard count of how much of it clears noise.

    ⚠ MEASURED ON THE RAW SPECTRUM, NOT THE WHITENED ONE — unlike every other read here.
    Whitening equalises each channel to unit noise, which is exactly right when the
    question is "does this stand above noise" and exactly wrong when the question is
    "how is the energy distributed": it redistributes the energy before you measure the
    distribution. Verified against the reference — whitening first shifts occupancy by ~0.12
    on a heteroscedastic matrix."""
    A = _as_matrix(M)
    sv = np.linalg.svd(A, compute_uv=False)
    n = int(sv.shape[0])
    if n == 0:
        return 0.0
    return float(2.0 ** shannon_bits(sv ** 2) / n)


def spectrum_stats(M, *, far: float = DEFAULT_FAR) -> dict[str, Any]:
    """sigma_top / noise_floor / signal rank / H_sv / occupancy plus the SVD factors, one pass.

    For callers that need the basis as well as the count and should not pay for two
    decompositions to get them.

    ⚠ EVERYTHING HERE IS THE WHITENED MATRIX, INCLUDING `occupancy` — so this `occupancy` is NOT
    `occupancy_fraction(M)`, which reads the raw spectrum on purpose. Named `occupancy_whitened`
    rather than `occupancy` so the two cannot be confused by a caller reaching for whichever
    is nearer."""
    A = whiten(M)
    N, F = A.shape
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    floor = math.sqrt(_floor_sq(_noise_sigma2(A, N, F), N, F, far))
    H_sv = shannon_bits(S ** 2)
    n = int(S.shape[0])
    return {
        "sigma_top": float(S[0]) if S.size else 0.0,
        "noise_floor": floor,
        "signal rank": max(1, int(np.count_nonzero(S > floor))) if min(N, F) >= 2 else 1,
        "H_sv": H_sv,
        "occupancy_whitened": float(2.0 ** H_sv / n) if n else 0.0,
        "U": U, "S": S, "Vt": Vt,
    }
