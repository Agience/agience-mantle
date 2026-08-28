"""Cross-walk — the gauge between two derived vector SPACES.

WHY THIS EXISTS
---------------
Semantics here is CORPUS-DERIVED: every node computes its own reduced space (PPMI · SVD, rank
read by `signal_rank`) over its OWN documents. Two nodes holding different documents therefore
grow DIFFERENT bases. Comparing vectors across nodes — which is what mesh search IS — requires
a projection between those bases, and that projection is this file.

    same dim → orthogonal Procrustes — CLOSED FORM, a pure rotation, so the gauge is an ISOMETRY
    cross dim → rectangular least-squares — a genuine FIT, NOT an isometry, NOT a gauge

The same-dim branch is the universal shape rather than a trick: a rotation relating two frames
that describe one underlying thing. `Crosswalk.is_isometry` MEASURES that property (‖MᵀM − I‖)
rather than asserting it from `method`, because a stored matrix can be anything.

WHY THESE ARE *SPACES*, NOT MODELS
----------------------------------
The fields were once `source_model_id` / `target_model_id`. There are no models in this system —
no trained weights, anywhere, for any reason — and the math was never about models: it relates
one BASIS to another. A space's identity is whatever names the basis (a corpus digest, an anchor
set's id, a node id). The old names invited the reader to think a model was being aligned and to
reach for model-shaped guarantees that do not exist here. They survive as deprecated aliases so
nothing already written is orphaned; `crosswalk_artifact` reads the legacy artifact keys too.

WHY THERE IS NO `error_bound`
-----------------------------
There was a float called `error_bound`, and it was measured on the very pairs the fit saw. An
in-sample residual is not a bound on anything: a rectangular fit with fewer pairs than input
dimensions reproduces its training pairs *exactly* and generalises not at all, so the number was
smallest exactly when the fit was worst. It had already been read downstream as a fidelity
guarantee. It is now `Crosswalk.residual`, a :class:`FitResidual`, which

  * carries the in-sample number under a name that says in-sample,
  * carries a HELD-OUT number computed by k-fold refit — an honest generalisation estimate,
  * defines no ordering and no ``__float__``, so ``cw.residual < 0.05`` raises TypeError and a
    reader must state which of the two numbers it meant.

Numpy-only (no scipy). Non-authorizing: plaintext geometry (canonical plan §1).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from prism.rounding import accumulated_rounding as _accumulated_rounding

from .anchorset import l2norm

# Fewer than this many pairs and no fold can be held out while leaving a usable fit behind: a
# fold must contain at least one pair, and the training side must retain at least two or the
# held-out number measures the split rather than the walk.
_MIN_PAIRS_FOR_HOLDOUT = 4
_MAX_FOLDS = 5

PROCRUSTES = "procrustes"   # closed form, same dim, an isometry
LINEAR = "linear"           # rectangular least squares — a FIT


class DimensionMismatch(ValueError):
    """The two spaces do not have the same dimension.

    Typed so a caller can tell "these spaces are not comparable at this D" apart from every
    other ValueError, and refuse rather than reach for the rectangular FIT by accident.
    """


def null_residual(dim_out: int, n_pairs: int) -> Tuple[float, float]:
    """The residual expected when two spaces share NO structure, and its standard error.

    DERIVED, not tuned. The residual is ``mean(1 − cos)``. Two independent directions in
    ``dim_out`` dimensions have ``E[cos] = 0`` and ``sd[cos] = 1/sqrt(dim_out)``, so the null is
    exactly ``1.0``, with a standard error of ``1/sqrt(dim_out · n_pairs)`` over ``n_pairs``
    independent pairs.

    This is what makes "is this cross-walk telling me anything?" answerable without a threshold:
    compare the HELD-OUT residual against this computed null (see
    :attr:`FitResidual.z_below_null`), never against a number somebody typed in.
    """
    n = max(int(dim_out) * int(n_pairs), 1)
    return 1.0, 1.0 / math.sqrt(n)


@dataclass(frozen=True)
class FitResidual:
    """What a cross-walk fit COST — never a bound on what it will cost next.

    Deliberately NOT a float and deliberately unordered: no ``__float__``, no comparisons. Every
    reader has to name the number it wants. A bare float is exactly what let an in-sample
    measurement be read downstream as a guarantee.
    """

    in_sample: float
    """Mean ``1 − cos`` over the pairs the fit SAW. Descriptive only. It falls toward zero as the
    fit becomes MORE underdetermined, which is the opposite of what a quality number should do."""

    held_out: Optional[float]
    """Mean ``1 − cos`` over pairs the fit did NOT see, by k-fold refit. The honest number.
    ``None`` when it could not be computed — read :attr:`note` for why, and treat that as
    UNKNOWN rather than as fine."""

    n_pairs: int
    n_held_out: int
    dim_in: int
    dim_out: int
    closed_form: bool
    note: str = "ok"

    @property
    def underdetermined(self) -> Optional[bool]:
        """Did the fit have fewer constraints than parameters? ``None`` when ``n_pairs`` was not
        recorded (a walk rebuilt from an older artifact) — unknown is not a pass."""
        if not self.n_pairs:
            return None
        return self.n_pairs < self.dim_in

    @property
    def z_below_null(self) -> Optional[float]:
        """How many standard errors the HELD-OUT residual sits below the shared-nothing null.

        ``0`` means the two spaces are indistinguishable from unrelated — the walk still
        projects, and it carries no information. Large means real shared structure. ``None``
        when there is no held-out number to judge. No threshold is applied here on purpose: the
        caller decides what it needs, against a null this module COMPUTES."""
        if self.held_out is None:
            return None
        null, se = null_residual(self.dim_out, max(self.n_held_out, 1))
        if se <= 0.0:
            return None
        return (null - self.held_out) / se

    def summary(self) -> str:
        ho = "unknown" if self.held_out is None else f"{self.held_out:.4f}"
        return (f"in-sample={self.in_sample:.4f} held-out={ho} "
                f"(n={self.n_pairs}, held={self.n_held_out}, "
                f"{'closed-form' if self.closed_form else 'FITTED'}; {self.note})")


@dataclass
class Crosswalk:
    """A projection from ``source_space_id``'s basis into ``target_space_id``'s basis."""

    source_space_id: str
    target_space_id: str
    method: str            # PROCRUSTES (closed-form isometry) | LINEAR (rectangular FIT)
    matrix: np.ndarray     # (dim_in, dim_out)
    dim_in: int
    dim_out: int
    residual: FitResidual

    # ------------------------------------------------------------------ properties
    @property
    def closed_form(self) -> bool:
        """Was this SOLVED (Procrustes) or FITTED (least squares)? A fit is an approximation
        whose quality depends on the pairs it happened to see; a solve is not."""
        return self.method == PROCRUSTES

    @property
    def is_isometry(self) -> bool:
        """Does this transform actually preserve norms and distances? MEASURED, not asserted.

        A gauge relating two frames of one underlying thing must be an isometry; if ‖MᵀM − I‖ is
        not within float32 resolution then this matrix is not a gauge, whatever ``method``
        claims. The tolerance is DERIVED from the arithmetic — float32 eps accumulated over a
        ``dim_in``-long inner product — rather than chosen.
        """
        if self.dim_in != self.dim_out:
            return False
        m = np.asarray(self.matrix, dtype=np.float64)
        # The band comes from `prism.rounding`, the single home of `eps * total * n`, rather than
        # being written out again. ACCUMULATION is the model: each entry of `MᵀM` is a sum of
        # `dim_in` products, so `n_operations = dim_in`, and `total = 1.0` because the compared
        # quantity is a deviation from the identity. Bit-identical to `eps32 * dim_in`.
        tol = _accumulated_rounding(n_operations=max(self.dim_in, 1), total=1.0,
                                    eps=float(np.finfo(np.float32).eps))
        return bool(np.max(np.abs(m.T @ m - np.eye(self.dim_in))) <= tol)

    @property
    def underdetermined(self) -> Optional[bool]:
        return self.residual.underdetermined

    # --------------------------------------------------------- deprecated aliases
    @property
    def source_model_id(self) -> str:
        _warn_model_name("source_model_id", "source_space_id")
        return self.source_space_id

    @property
    def target_model_id(self) -> str:
        _warn_model_name("target_model_id", "target_space_id")
        return self.target_space_id

    @property
    def error_bound(self) -> float:
        """Removed, and it raises — loudly, because every silent alternative is worse.

        ``getattr(cw, "error_bound", None)`` still degrades to ``None`` for defensive callers,
        which is correct: the number they were reaching for never existed."""
        raise AttributeError(
            "Crosswalk.error_bound was an IN-SAMPLE residual read downstream as a fidelity "
            "guarantee, and is gone. Use `cw.residual.in_sample` for the same number under an "
            "honest name, or `cw.residual.held_out` for a generalisation estimate."
        )

    # ------------------------------------------------------------------ use
    def apply(self, vec: Sequence[float] | np.ndarray) -> np.ndarray:
        """Project + unit-normalize a single vector into the target space."""
        v = np.asarray(vec, dtype=np.float32).ravel()
        if v.shape[-1] != self.dim_in:
            raise DimensionMismatch(
                f"cross-walk expects {self.dim_in}-dim input, got {v.shape[-1]}"
            )
        return l2norm(v @ self.matrix)


def _warn_model_name(old: str, new: str) -> None:
    warnings.warn(
        f"{old!r} is deprecated: a cross-walk relates two derived SPACES, not models "
        f"(there are no models in this system). Use {new!r}.",
        DeprecationWarning,
        stacklevel=3,
    )


def _solve(A: np.ndarray, B: np.ndarray, method: str) -> np.ndarray:
    """The matrix, and nothing else. Shared by the fit and by every held-out refit."""
    if method == PROCRUSTES:
        if A.shape[1] != B.shape[1]:
            raise DimensionMismatch(
                f"procrustes requires equal dimensions, got {A.shape[1]} → {B.shape[1]}"
            )
        # Orthogonal R minimizing ||A·R − B|| : R = U·Vᵀ from SVD(Aᵀ·B).
        u, _s, vt = np.linalg.svd(A.T @ B)
        return (u @ vt).astype(np.float32)
    if method == LINEAR:
        return np.linalg.lstsq(A, B, rcond=None)[0].astype(np.float32)
    raise ValueError(f"unknown cross-walk method: {method!r}")


def _mean_residual(A: np.ndarray, B: np.ndarray, matrix: np.ndarray) -> float:
    return float(np.mean(1.0 - np.sum(l2norm(A @ matrix) * B, axis=1)))


def _held_out_residual(
    A: np.ndarray, B: np.ndarray, method: str
) -> Tuple[Optional[float], int, str]:
    """K-fold: refit WITHOUT each fold, score ON that fold. Returns (residual, n_scored, note).

    This is the number the old `error_bound` should have been. It can be worse than in-sample by
    orders of magnitude — that gap is the finding, not a bug.
    """
    n = A.shape[0]
    if n < _MIN_PAIRS_FOR_HOLDOUT:
        return None, 0, f"too few pairs to hold any out (n={n} < {_MIN_PAIRS_FOR_HOLDOUT})"
    k = min(_MAX_FOLDS, n)
    idx = np.arange(n)
    total, scored = 0.0, 0
    for fold in np.array_split(idx, k):
        train = np.setdiff1d(idx, fold, assume_unique=True)
        if train.size < 2:
            continue
        try:
            m = _solve(A[train], B[train], method)
        except np.linalg.LinAlgError as e:      # non-convergent SVD on a degenerate fold
            return None, 0, f"a fold did not solve: {e}"
        total += _mean_residual(A[fold], B[fold], m) * fold.size
        scored += int(fold.size)
    if not scored:
        return None, 0, "no fold left a trainable remainder"
    return total / scored, scored, "ok"


def _resolve_space_id(new: Optional[str], legacy: Optional[str], which: str) -> str:
    if new is not None and legacy is not None:
        raise ValueError(f"pass {which}_space_id or the deprecated {which}_model_id, not both")
    if new is not None:
        return new
    if legacy is not None:
        _warn_model_name(f"{which}_model_id", f"{which}_space_id")
        return legacy
    raise TypeError(f"fit_crosswalk() requires {which}_space_id")


def fit_crosswalk(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_space_id: Optional[str] = None,
    target_space_id: Optional[str] = None,
    method: str = "auto",
    holdout: bool = True,
    # deprecated, accepted so nothing already written breaks
    source_model_id: Optional[str] = None,
    target_model_id: Optional[str] = None,
) -> Crosswalk:
    """Fit a cross-walk from paired ``(source[i], target[i])`` vectors.

    ``method="auto"`` picks orthogonal Procrustes when the dims match — the closed-form gauge —
    and otherwise a rectangular least-squares map, which is a FIT and says so
    (``closed_form=False``, ``is_isometry=False``).

    The rectangular branch is kept. It would be removable if every node derived its basis at one
    fixed D, and nothing in this tree fixes D: an AnchorSet takes its dim
    from whatever artifact provisioned it (`repo.py`, `AnchorSet.load`), and the corpus-derived
    rank is read by `signal_rank`, which returns what the corpus supports rather than a constant.
    Asking for ``method="procrustes"`` across unequal dims raises :class:`DimensionMismatch`.

    ``holdout=False`` skips the k-fold refit when the caller cannot afford k extra solves; the
    result then reports ``held_out=None`` with the reason, which is UNKNOWN, not fine.
    """
    source_space_id = _resolve_space_id(source_space_id, source_model_id, "source")
    target_space_id = _resolve_space_id(target_space_id, target_model_id, "target")

    A = l2norm(np.asarray(source, dtype=np.float32))   # (n, d_in)
    B = l2norm(np.asarray(target, dtype=np.float32))   # (n, d_out)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise ValueError("source and target must be paired 2-D arrays")
    if A.shape[0] == 0:
        raise ValueError("cannot fit a cross-walk from zero pairs")
    d_in, d_out = A.shape[1], B.shape[1]

    if method == "auto":
        method = PROCRUSTES if d_in == d_out else LINEAR

    matrix = _solve(A, B, method)
    in_sample = _mean_residual(A, B, matrix)
    if holdout:
        held, n_held, note = _held_out_residual(A, B, method)
    else:
        held, n_held, note = None, 0, "holdout disabled by the caller"

    return Crosswalk(
        source_space_id=source_space_id,
        target_space_id=target_space_id,
        method=method,
        matrix=matrix,
        dim_in=d_in,
        dim_out=d_out,
        residual=FitResidual(
            in_sample=in_sample,
            held_out=held,
            n_pairs=int(A.shape[0]),
            n_held_out=n_held,
            dim_in=d_in,
            dim_out=d_out,
            closed_form=(method == PROCRUSTES),
            note=note,
        ),
    )


class CrosswalkRegistry:
    """In-memory registry of cross-walks, keyed by ``(source space, target space)``."""

    def __init__(self) -> None:
        self._walks: Dict[Tuple[str, str], Crosswalk] = {}

    def register(self, crosswalk: Crosswalk) -> Crosswalk:
        self._walks[(crosswalk.source_space_id, crosswalk.target_space_id)] = crosswalk
        return crosswalk

    def get(self, source_space_id: str, target_space_id: str) -> Optional[Crosswalk]:
        if source_space_id == target_space_id:
            return None
        return self._walks.get((source_space_id, target_space_id))

    def walk(
        self,
        vec: Sequence[float] | np.ndarray,
        source_space_id: str,
        target_space_id: str,
    ) -> np.ndarray:
        """Project ``vec`` from source into target space (identity when equal)."""
        if source_space_id == target_space_id:
            return l2norm(np.asarray(vec, dtype=np.float32).ravel())
        cw = self.get(source_space_id, target_space_id)
        if cw is None:
            raise ValueError(
                f"no registered cross-walk {source_space_id!r} → {target_space_id!r}"
            )
        return cw.apply(vec)


__all__ = [
    "Crosswalk", "CrosswalkRegistry", "FitResidual", "DimensionMismatch",
    "fit_crosswalk", "null_residual", "PROCRUSTES", "LINEAR",
]
