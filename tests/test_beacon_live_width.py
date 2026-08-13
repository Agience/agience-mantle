# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# BEACON is the permissive half of the two-tier model, deliberately: mantle ships
# Apache so a store can be taken, built on and shipped by anyone, and beacon is the
# reduced instrument that makes such a store genuinely useful on its own. mantle is
# Apache-2.0 with a public remote and `packages.find` has no exclude, so no
# proprietary-restriction notice applies here. The downstream consumer's
# `beacon_engine.py` keeps its own notice — that is the Foresight white-label pilot,
# a different tree and a different arrangement.
# ---------------------------------------------------------------------------
"""The null is sized by the channels that exist.

The Johnstone/TW1 edge and the de-biased noise variance are properties of an N x F iid
ensemble, so F decides where the floor sits. A dead channel is not a channel of that
ensemble: `whiten` returns any channel it could not scale as all-zeros, and `_as_matrix`
fills a wholly-absent column (every cell nonfinite) with 0.0. Neither counts toward F.

The failure modes these tests watch for, stated first so they can fail:
  - appending columns that carry no observation changes the resolved rank;
  - and, the control, appending columns that carry a real observation does not change it,
    which would mean the fix had become "ignore inconvenient columns".
"""
import math

import numpy as np
import pytest

from mantle.search.beacon.engine import (
    DEFAULT_FAR, _floor_sq, _noise_sigma2, live_width, signal_rank, whiten,
)


def noise_floor(M, *, far: float = DEFAULT_FAR) -> float:
    """Reproduces the retired `engine.noise_floor` — zero production callers, retired for that
    reason, kept local here since this file's own subject is exactly the live-width sizing this
    formula performs."""
    A = whiten(M)
    N, F = A.shape[0], live_width(A)
    return math.sqrt(_floor_sq(_noise_sigma2(A, N, F), N, F, far))


def _rank3(rng, N=120, F=24):
    """A matrix genuinely made of three directions, plus noise."""
    B = rng.standard_normal((3, F))
    C = rng.standard_normal((N, 3)) * np.array([6.0, 4.0, 3.0])
    return C @ B + 0.35 * rng.standard_normal((N, F))


def test_all_nan_columns_do_not_change_the_read():
    """A column with no finite cell anywhere is not a channel of the ensemble."""
    rng = np.random.default_rng(0)
    M = _rank3(rng)

    padded = np.concatenate([M, np.full((M.shape[0], 40), np.nan)], axis=1)
    assert signal_rank(padded).k == signal_rank(M).k
    assert noise_floor(padded) == pytest.approx(noise_floor(M), rel=1e-12)


def test_all_zero_columns_do_not_change_the_read():
    """`whiten` cannot scale a constant channel and returns it as zeros -- same situation,
    reached by a different route (and the one that actually occurs on real frames)."""
    rng = np.random.default_rng(1)
    M = _rank3(rng)

    padded = np.concatenate([M, np.zeros((M.shape[0], 40))], axis=1)
    assert live_width(whiten(padded)) == live_width(whiten(M))
    assert signal_rank(padded).k == signal_rank(M).k
    assert noise_floor(padded) == pytest.approx(noise_floor(M), rel=1e-12)


def test_real_columns_DO_change_the_read():
    """The control. Without it, 'size the null by whatever is convenient' passes every
    assertion above. Forty more channels of genuine noise is a genuinely wider ensemble,
    and the floor must move."""
    rng = np.random.default_rng(2)
    M = _rank3(rng)

    widened = np.concatenate([M, rng.standard_normal((M.shape[0], 40))], axis=1)
    assert live_width(whiten(widened)) > live_width(whiten(M))
    assert noise_floor(widened) != pytest.approx(noise_floor(M), rel=1e-9)


def test_noise_floor_must_be_sized_by_live_width_not_raw_width():
    """A rank-2 frame padded with 300 dead channels, showing why the noise floor must be sized
    by live width and not raw array width.

    `_noise_sigma2` divides the median row energy by a denominator in F. Dead columns add no
    energy, so counting them toward F sinks sigma^2 and the floor far enough that noise modes
    clear it. The floor is reconstructed both ways here on one frame: sized by raw array width
    it over-counts, sized by live width it recovers the planted rank of 2."""
    r = np.random.default_rng(5 * 7 + 10)
    B = r.standard_normal((2, 20))
    C = r.standard_normal((80, 2)) * 1.0
    M = C @ B + r.standard_normal((80, 20))
    padded = np.concatenate([M, np.zeros((80, 300))], axis=1)

    A = whiten(padded)
    N = A.shape[0]
    sv = np.linalg.svd(A, compute_uv=False)

    def _k(F):
        floor = math.sqrt(_floor_sq(_noise_sigma2(A, N, F), N, F, DEFAULT_FAR))
        return int(np.count_nonzero(sv > floor))

    assert live_width(A) == 20 and A.shape[1] == 320
    assert _k(A.shape[1]) > 2, "sizing by raw array width over-counts; if this stops being " \
                               "true the demonstration has gone stale, not the fix"
    assert _k(live_width(A)) == 2 == signal_rank(padded).k


def test_live_width_never_reports_zero():
    """`johnstone` needs a non-degenerate shape; an empty frame is reported through
    `degraded`, never through a fabricated width."""
    assert live_width(np.zeros((8, 16))) == 1
    read = signal_rank(np.zeros((8, 16)))
    assert read.degraded and read.live_channels == 0


def test_a_collapsed_axis_is_still_flagged():
    """The hazard flag reads the raw live count, not the floored width -- flooring it at 1
    for the null must not make a collapsed frame look healthy."""
    rng = np.random.default_rng(3)
    M = np.zeros((64, 12))
    M[:, 4] = rng.standard_normal(64)          # one live channel
    read = signal_rank(M)
    assert read.live_channels == 1 and read.degraded
