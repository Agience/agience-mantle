"""Density-zoom — frame-invariant novelty layer over the AnchorSet.

Canonical plan §6. An item's *coverage* by the anchor vocabulary — its
nearest-anchor cosine affinity — places it on a discrete layer:

    L2  common/dense    (well inside the covered manifold)
    L1  working
    L0  novel/sparse    (far from every anchor → an anchor candidate; RG-flow)

Thresholds are cosine-based, so the layer is frame-invariant (a gauge change /
different model leaves it unchanged). Non-authorizing: plaintext geometry only
(canonical plan §1).

⛔ THE NOVELTY CUT IS DERIVED, NOT PICKED. It used to be `percentile(nn, 10)`,
described as "data-driven — no magic constant". A quantile of real data is still a
constant somebody chose: nothing about the 10th percentile makes it the place where
covered stops and novel begins, and it declares a fixed 10% of the anchor vocabulary's
own spacing to be novel no matter how the vocabulary is actually shaped. The question
"is this vector closer to the anchors than chance would put it?" has an answer, so it
is answered — see `_chance_affinity`.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from mantle.search.beacon import DEFAULT_FAR, signal_rank

from .anchorset import L0, L1, L2, AnchorSet


#: Draws used to characterise the null. Enough that the far=0.05 quantile is stable to
#: ~0.003 in cosine, which is far finer than the layer boundary needs.
_NULL_DRAWS = 20000

#: Fixed. The cut must be a function of the anchor matrix and nothing else — a fresh
#: seed per process would move the L0/L1 boundary between restarts, so the same item
#: could be novel on one worker and covered on another.
_NULL_SEED = 0x8EAC0


def _chance_affinity(matrix: np.ndarray, far: float) -> float:
    """The nearest-anchor cosine an UNRELATED vector reaches by luck alone.

    The novelty cut: the `1 - far` quantile of `max_i (v · a_i)` for `v` drawn
    isotropically. Above it the affinity beats chance and the item is covered; below it
    the geometry genuinely does not place the item, and it is an anchor candidate.

    ⛔ MEASURED AGAINST THE REAL MATRIX, NOT DERIVED IN CLOSED FORM. Two closed forms
    were tried and both were wrong, in opposite directions:

      * scaling by `1/sqrt(signal_rank)` put the cut at cosine 0.95 on a 512-dim
        screen — "identical to an anchor, or else novel";
      * scaling by `1/sqrt(dim)` with a Bonferroni correction over the resolved rank
        let 18% of pure-chance vectors read as covered against a stated 5%.

    The reason no closed form lands is that the anchors are heavily CORRELATED. The
    effective number of independent chances a vector gets sits somewhere between the
    resolved rank and the anchor count, it depends on how the anchors are distributed
    within their span, and picking any single value for it is guessing. Sampling the
    actual matrix has no such freedom: it is the exact null for this vocabulary, with
    no distributional assumption to get wrong.

    Deterministic — fixed seed, fixed draw count, and it runs once per fit.
    """
    k, dim = matrix.shape
    if k < 1 or dim < 1:
        return 0.0
    rng = np.random.default_rng(_NULL_SEED)
    # float32 throughout: the draw is the whole cost of a fit, and the quantile is a
    # cosine boundary read to ~0.003 — single precision is three digits past that.
    v = rng.standard_normal((_NULL_DRAWS, dim), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    best = (v @ matrix.astype(np.float32, copy=False).T).max(axis=1)
    return float(min(1.0, np.quantile(best, 1.0 - far)))


class DensityZoom:
    """Maps a vector to its density layer over a fitted AnchorSet."""

    def __init__(self, anchorset: AnchorSet, *, far: float = DEFAULT_FAR) -> None:
        self.anchorset = anchorset
        self.far = far
        self._t_low, self._t_high = self._fit(anchorset, far)

    @staticmethod
    def _fit(anchorset: AnchorSet, far: float = DEFAULT_FAR) -> Tuple[float, float]:
        """Derive the (L0|L1, L1|L2) thresholds.

        `t_low` (novel) is the affinity chance alone produces against this vocabulary —
        see `_chance_affinity`. `t_high` (covered) stays the MEDIAN of the anchors'
        nearest-other-anchor cosines: unlike the old lower bound that is not an
        arbitrary quantile but a location statistic, and "as close to an anchor as
        anchors typically are to each other" is the definition of dense.
        """
        matrix = anchorset.matrix
        if matrix is None or len(anchorset) < 3:
            return 0.0, 1.0
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, -np.inf)
        nn = sims.max(axis=1)
        nn = nn[np.isfinite(nn)]
        if nn.size == 0:
            return 0.0, 1.0
        t_low = _chance_affinity(matrix, far)
        t_high = float(np.median(nn))
        if t_high <= t_low:
            t_high = t_low + 1e-3
        return t_low, t_high

    @property
    def signal_rank(self) -> int:
        """How many independent directions the anchor vocabulary actually spans.

        A DIAGNOSTIC, not part of the cut — `_chance_affinity` measures the null
        directly. It is what says whether the vocabulary has degenerated: 200 anchors
        resolving to 19 directions means the anchors are mostly restating each other,
        which is a signal to `grow`/`reconcile` the set rather than to reclassify
        anything. Returns 0 when the screen has no spectrum to read."""
        matrix = self.anchorset.matrix
        if matrix is None or len(self.anchorset) < 3:
            return 0
        try:
            read = signal_rank(matrix, far=self.far)
        except Exception:
            return 0
        if read.degraded:
            # The whitening collapsed the feature axis (live_channels <= 1), so the spectrum read is
            # not the spectrum of the anchors. Reporting `k` here would look like a healthy
            # low-rank vocabulary and trigger exactly the wrong `grow`/`reconcile` advice.
            return 0
        return int(read)

    def density(self, vec: Sequence[float] | np.ndarray) -> float:
        """Nearest-anchor cosine affinity — the coverage/density proxy in [-1, 1].
        Returns 0.0 when the vector can't be placed (dim mismatch / no anchors)."""
        near = self.anchorset.nearest(vec, k=1)
        return float(near[0][1]) if near else 0.0

    def layer(self, vec: Sequence[float] | np.ndarray) -> Tuple[str, float]:
        """Return ``(layer, density)`` for ``vec``."""
        a = self.density(vec)
        if a >= self._t_high:
            return L2, a
        if a >= self._t_low:
            return L1, a
        return L0, a
