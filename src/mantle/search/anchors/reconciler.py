"""Reconciler — any source embedding → the native language of meaning.

See `.dev/features/mantle-canonical-architecture.md` §4. The native code is a
**sparse anchor-relative** representation: an item is its top-``m`` affinities to
the anchors. Model-unbiased (coordinate k means "closeness to anchor-concept k"
regardless of which model produced the raw vector) and dimension-agnostic (the
code lives in anchor space, not the source model's native dim) — which is why a
single index holds vectors from any embedder/modality.

INVARIANT (§1): geometry only — no keys/auth. Runs before partition/routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .anchorset import AnchorSet


@dataclass
class SparseCode:
    """A native-language vector: ``weights`` over the anchor ``indices``,
    L2-normalized over the active set. ``dim`` is the current anchor count K.

    THE THREE NUMBERS BELOW ARE THE MEASUREMENT THE NORMALISATION DESTROYS
    ---------------------------------------------------------------------
    ``weights`` is renormalised to unit L2 over the kept set. That is correct — it is what
    makes :meth:`dot` a cosine and makes two codes comparable — but it is also lossy in a
    way that is invisible afterwards: **a sharply-peaked code and a near-uniform one
    normalise to the same shell.** After the divide there is nothing in the record that says
    whether this item sat decisively on a handful of anchors or was weakly and evenly
    related to thirty-two of them.

    Three distinct things were being paid for and then discarded:

    ``pre_norm``
        The L2 norm of the KEPT affinities *before* renormalisation. The absolute scale of
        the item's relationship to the anchor vocabulary. A well-covered item and an item
        floating between anchors have very different pre-norms and identical weights.
    ``residual``
        The fraction of the total positive-affinity L2 mass that top-``m`` truncation CUT,
        in ``[0, 1]``. ``0.0`` means the top-``m`` captured everything; a large value means
        the item is broadly related to the vocabulary and the code is a narrow slice of what
        was actually measured. Without it, truncation is silent.
    ``negative_mass``
        The L2 norm of the ANTI-correlated affinities (cosine < ``min_affinity``, default
        0.0) that were filtered out. A negative affinity is evidence — "this item is
        specifically unlike anchor k" — not noise. It is not carried in ``weights`` (see the
        note in :meth:`Reconciler.to_native`), so it is carried as a scalar here.

    All three default to ``0.0`` so a :meth:`from_dict` over a record written before these
    existed loads cleanly. ``0.0`` reads as "not measured" for ``residual`` and
    ``negative_mass`` (both are "nothing was lost", the optimistic-but-honest default for an
    old record) and as "unknown" for ``pre_norm``.
    """

    indices: np.ndarray   # int positions into the AnchorSet
    weights: np.ndarray   # float32, unit-norm over the active set
    dim: int
    anchor_ids: List[str]
    pre_norm: float = 0.0       # ||kept affinities|| BEFORE the unit-norm divide
    residual: float = 0.0       # fraction of positive affinity mass cut by top-m, in [0,1]
    negative_mass: float = 0.0  # ||affinities below min_affinity|| — discarded anti-correlation

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def top_anchor_id(self) -> Optional[str]:
        """The routing anchor — the cell this item lands in (§5.1)."""
        return self.anchor_ids[0] if self.anchor_ids else None

    def dot(self, other: "SparseCode") -> float:
        """Cosine similarity between two native codes (both unit-norm)."""
        if len(self) == 0 or len(other) == 0:
            return 0.0
        lut = {int(i): float(w) for i, w in zip(self.indices, self.weights)}
        return float(sum(float(w) * lut.get(int(i), 0.0)
                         for i, w in zip(other.indices, other.weights)))

    def to_dict(self) -> dict:
        """JSON-serializable form for the cell chunk record (plain ints/floats)."""
        return {
            "i": [int(x) for x in self.indices],
            "w": [float(x) for x in self.weights],
            "ids": list(self.anchor_ids),
            "k": int(self.dim),
            # The evidence the unit-norm divide erases. Short keys: this rides in every
            # chunk record in the corpus, so three floats is the whole budget.
            "pn": float(self.pre_norm),
            "res": float(self.residual),
            "neg": float(self.negative_mass),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SparseCode":
        # Backward compatible by construction: every record written before `pn`/`res`/`neg`
        # existed simply lacks the keys and loads as 0.0. Nothing needs migrating, and a
        # zero here honestly means "this record predates the measurement", not "measured
        # zero" — a reader that needs to tell them apart should band on `pre_norm == 0.0`,
        # which a real code with any kept affinity can never produce.
        return cls(
            indices=np.asarray(d.get("i", []), dtype=int),
            weights=np.asarray(d.get("w", []), dtype=np.float32),
            dim=int(d.get("k", 0)),
            anchor_ids=list(d.get("ids", [])),
            pre_norm=float(d.get("pn", 0.0)),
            residual=float(d.get("res", 0.0)),
            negative_mass=float(d.get("neg", 0.0)),
        )


class Reconciler:
    """Project source embeddings into the native language over an AnchorSet."""

    def __init__(
        self,
        anchorset: AnchorSet,
        *,
        top_m: int = 32,
        min_affinity: float = 0.0,
        crosswalks=None,
    ) -> None:
        self.anchorset = anchorset
        self.top_m = int(top_m)
        self.min_affinity = float(min_affinity)
        self.crosswalks = crosswalks   # CrosswalkRegistry | None (AlignmentRegistry)

    def to_native(
        self,
        vec: Sequence[float] | np.ndarray,
        *,
        model_id: Optional[str] = None,
    ) -> SparseCode:
        """Reconcile one source vector to a sparse anchor-relative code.

        A ``model_id`` other than the AnchorSet's is projected into the AnchorSet
        space via the cross-walk registry (AlignmentRegistry, §4.3). Without a
        registered cross-walk we fail loudly rather than silently mis-project.
        """
        if model_id is not None and model_id != self.anchorset.model_id:
            if self.crosswalks is None:
                raise ValueError(
                    f"cross-walk required: {model_id!r} → {self.anchorset.model_id!r} "
                    "(no AlignmentRegistry configured)"
                )
            vec = self.crosswalks.walk(vec, model_id, self.anchorset.model_id)

        # ── WHY THE FULL AFFINITY VECTOR IS READ, NOT JUST THE TOP-m ──────────────────────
        # `residual` cannot be computed from the top-m alone: the mass that truncation cut is
        # by definition outside it. `nearest` already computes the affinity to EVERY anchor
        # (one `matrix @ q`) and then throws all but k away, so asking for all K costs one
        # full sort instead of an argpartition — not a second matmul. The anchor vocabulary
        # is thousands of entries, not millions; this is microseconds against an embedding.
        n_anchors = len(self.anchorset)
        scored = self.anchorset.nearest(vec, k=n_anchors) if n_anchors else []
        if not scored:
            return SparseCode(np.empty(0, dtype=int), np.empty(0, dtype=np.float32),
                              n_anchors, [])

        sims = np.asarray([s for _, s in scored], dtype=np.float64)   # descending

        # ── THE CLAMP: `max(s, 0.0)` IS NOT LOAD-BEARING, AND IT WAS NOT THE EROSION ──────
        # Checked every consumer of `weights`: `SparseCode.dot` is a plain sum of products
        # (sign-safe — it is a cosine, and a cosine is signed), `activate.activate_vector`
        # only rounds them for display, `top_anchor_id` reads `anchor_ids[0]`. Nothing
        # downstream requires a non-negative weight, so the sign is preserved here.
        #
        # But note what the clamp actually was: with `min_affinity` defaulting to 0.0, line
        # "s >= self.min_affinity" below has ALREADY dropped every negative affinity before
        # the clamp can see one — and no caller in the tree sets `min_affinity` at all
        # (`Reconciler(...)` in activate.py, pipeline_unified.py and the tests all take the
        # default). The clamp was therefore unreachable, and the real erasure of
        # anti-correlation is the FILTER, not the clamp. Removing the clamp changes no
        # current output; it stops the second erasure from being waiting there if anyone
        # ever passes a negative `min_affinity`.
        #
        # The filter itself is LEFT ALONE deliberately. Its threshold decides which anchors
        # appear in a stored code, so moving it rewrites the native code of every artifact
        # in the corpus. That is a data migration, not a code fix. What is fixed is that the
        # discarded anti-correlation is now MEASURED (`negative_mass`) instead of vanishing.
        below = sims[sims < self.min_affinity]
        negative_mass = float(np.linalg.norm(below)) if below.size else 0.0

        eligible = [(a, s) for a, s in scored if s >= self.min_affinity]
        if not eligible:
            # Everything was filtered. Nothing was TRUNCATED (residual 0.0), but the
            # anti-correlated mass we threw away is real and is reported.
            return SparseCode(np.empty(0, dtype=int), np.empty(0, dtype=np.float32),
                              n_anchors, [], pre_norm=0.0, residual=0.0,
                              negative_mass=negative_mass)

        total = float(np.linalg.norm([s for _, s in eligible]))
        near = eligible[:self.top_m]

        idx = np.fromiter((self.anchorset.position(a.anchor_id) for a, _ in near),
                          dtype=int, count=len(near))
        w = np.array([s for _, s in near], dtype=np.float32)

        # `pre_norm` is captured BEFORE the divide — line 113 used to compute exactly this
        # number, use it as a divisor, and then drop it on the floor. It is the only thing
        # that distinguishes a sharply-peaked code from a near-uniform one after
        # normalisation, and it cost nothing to keep.
        pre_norm = float(np.linalg.norm(w))
        if pre_norm > 0.0:
            w = w / pre_norm

        # Pythagorean residual over the affinity vector: the L2 mass sitting outside the
        # top-m, as a fraction of the eligible total. 0.0 = the code captured everything.
        # `max(0.0, ...)` guards float error only; total >= pre_norm holds by construction.
        residual = (float(np.sqrt(max(0.0, total * total - pre_norm * pre_norm)) / total)
                    if total > 0.0 else 0.0)

        return SparseCode(idx, w, n_anchors, [a.anchor_id for a, _ in near],
                          pre_norm=pre_norm, residual=residual,
                          negative_mass=negative_mass)
