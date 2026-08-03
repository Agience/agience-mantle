"""Cross-walk / AlignmentRegistry — project one model's space into another.

Canonical plan §4.3; aligns to ``facet-commons/spec/embedding-registry.md`` §5.
A cross-walk is a learned linear projection ``source_model → target_model`` fit
from *paired* vectors (the same items — typically the anchor texts — embedded by
both models). It lets a query embedded by any model be reconciled against an
AnchorSet built in another model's space:

    same dim → orthogonal Procrustes (a pure rotation; the gauge is an isometry)
    cross dim → least-squares linear map (rectangular)

Numpy-only (no scipy). Non-authorizing: plaintext geometry (canonical plan §1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .anchorset import l2norm


@dataclass
class Crosswalk:
    """A projection from ``source_model_id`` space into ``target_model_id`` space.

    Field set mirrors the facet-commons cross-walk record (method/dims/error).
    """

    source_model_id: str
    target_model_id: str
    method: str            # "procrustes" (same-dim isometry) | "linear" (rectangular)
    matrix: np.ndarray     # (dim_in, dim_out)
    dim_in: int
    dim_out: int
    # ⚠ IN-SAMPLE ERROR. This is the mean cosine distance on the SAME pairs the map was fitted
    # to — it is training error, not a generalisation bound, and the name has been read as a
    # fidelity guarantee downstream ("reports exactly what this embedder loses"). It is smallest
    # precisely when the fit is worthless: with fewer pairs than input dimensions the system is
    # underdetermined and `lstsq` interpolates the fit set EXACTLY. Measured — 8 pairs, 64->32
    # dims: reported error_bound **1e-08**, held-out mean cosine distance **1.003** (1.0 = chance).
    # Read it together with `n_pairs`/`underdetermined` below, never alone.
    error_bound: float
    # How many pairs the fit saw. Defaulted so existing constructors (notably
    # `crosswalk_artifact.from_artifact`, which deserialises a stored walk) keep working; 0 there
    # means "not recorded", which `underdetermined` reports as unknown rather than as fine.
    n_pairs: int = 0

    @property
    def underdetermined(self) -> Optional[bool]:
        """Is `error_bound` meaningless because the fit had fewer constraints than parameters?

        None when `n_pairs` was not recorded — unknown is not a pass."""
        if not self.n_pairs:
            return None
        return self.n_pairs < self.dim_in

    def apply(self, vec: Sequence[float] | np.ndarray) -> np.ndarray:
        """Project + unit-normalize a single vector into the target space."""
        v = np.asarray(vec, dtype=np.float32).ravel()
        if v.shape[-1] != self.dim_in:
            raise ValueError(
                f"cross-walk expects {self.dim_in}-dim input, got {v.shape[-1]}"
            )
        return l2norm(v @ self.matrix)


def fit_crosswalk(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_model_id: str,
    target_model_id: str,
    method: str = "auto",
) -> Crosswalk:
    """Fit a cross-walk from paired ``(source[i], target[i])`` vectors.

    ``method="auto"`` picks orthogonal Procrustes when dims match (the §2 gauge
    is a rotation), else a rectangular least-squares linear map.
    """
    A = l2norm(np.asarray(source, dtype=np.float32))   # (n, d_in)
    B = l2norm(np.asarray(target, dtype=np.float32))   # (n, d_out)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] != B.shape[0]:
        raise ValueError("source and target must be paired 2-D arrays")
    # ⛔ A ZERO-PAIR FIT USED TO SUCCEED AND RETURN error_bound = NaN. It passed the shape check
    # above, SVD'd a zero matrix into a fabricated identity, and `np.mean` over an empty array
    # gave NaN — so the returned Crosswalk looked well-formed while EVERY threshold comparison
    # against it is False. Measured: `nan < 0.1` is False AND `nan > 0.1` is False, so a consumer
    # gating one way silently rejects and one gating the other way silently accepts. There is no
    # projection to be learned from zero examples; say so.
    if A.shape[0] == 0:
        raise ValueError("cannot fit a cross-walk from zero pairs")
    d_in, d_out = A.shape[1], B.shape[1]

    if method == "auto":
        method = "procrustes" if d_in == d_out else "linear"

    if method == "procrustes":
        if d_in != d_out:
            raise ValueError("procrustes requires equal dimensions")
        # Orthogonal R minimizing ||A·R − B|| : R = U·Vᵀ from SVD(Aᵀ·B).
        u, _s, vt = np.linalg.svd(A.T @ B)
        matrix = (u @ vt).astype(np.float32)
    elif method == "linear":
        matrix = np.linalg.lstsq(A, B, rcond=None)[0].astype(np.float32)
    else:
        raise ValueError(f"unknown cross-walk method: {method!r}")

    projected = l2norm(A @ matrix)
    error_bound = float(np.mean(1.0 - np.sum(projected * B, axis=1)))   # IN-SAMPLE; see the field
    return Crosswalk(
        n_pairs=int(A.shape[0]),
        source_model_id=source_model_id,
        target_model_id=target_model_id,
        method=method,
        matrix=matrix,
        dim_in=d_in,
        dim_out=d_out,
        error_bound=error_bound,
    )


class CrosswalkRegistry:
    """In-memory registry of cross-walks, keyed by ``(source, target)``."""

    def __init__(self) -> None:
        self._walks: Dict[Tuple[str, str], Crosswalk] = {}

    def register(self, crosswalk: Crosswalk) -> Crosswalk:
        self._walks[(crosswalk.source_model_id, crosswalk.target_model_id)] = crosswalk
        return crosswalk

    def get(self, source_model_id: str, target_model_id: str) -> Optional[Crosswalk]:
        if source_model_id == target_model_id:
            return None
        return self._walks.get((source_model_id, target_model_id))

    def walk(
        self,
        vec: Sequence[float] | np.ndarray,
        source_model_id: str,
        target_model_id: str,
    ) -> np.ndarray:
        """Project ``vec`` from source into target space (identity when equal)."""
        if source_model_id == target_model_id:
            return l2norm(np.asarray(vec, dtype=np.float32).ravel())
        cw = self.get(source_model_id, target_model_id)
        if cw is None:
            raise ValueError(
                f"no registered cross-walk {source_model_id!r} → {target_model_id!r}"
            )
        return cw.apply(vec)
