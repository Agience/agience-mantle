"""Anchor growth — admit novel signals as new anchors (RG-flow).

Canonical plan §3/§6: the AnchorSet grows as the manifold grows. A signal in a
region the anchor vocabulary doesn't yet cover (density-zoom **L0 / novel**) is a
CANDIDATE for a new anchor. :func:`propose_anchor` admits such a signal to the
live AnchorSet and persists it; a signal already covered by an anchor is
rejected (no duplication). Promotion (CANDIDATE→WORKING→CANONICAL) and decay are
the full RG-flow — future.

Non-authorizing: plaintext geometry only (canonical plan §1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .anchorset import CANDIDATE, L0, Anchor
from .store import (
    get_anchor_repo,
    get_crosswalks,
    get_density_zoom,
    get_live_anchorset,
    reset_anchorset,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrowthDecision:
    """Why an anchor was or was not created — the continuous reading behind the boolean.

    THE ADMIT/REJECT IS AND MUST REMAIN DISCRETE. An anchor is created or it is not; there
    is no half-anchor, and every downstream consumer (the AnchorSet, the repo, the density
    refit) depends on that. What was wrong is that ``dz.layer(v)[0] != L0`` indexed past the
    float: :meth:`DensityZoom.layer` returns ``(layer, density)`` where ``density`` is the
    item's nearest-anchor COSINE, and the thresholds it is compared against
    (``_t_low`` = the 10th percentile of inter-anchor spacing, ``_t_high`` = the median) are
    themselves fitted from data and move every time the set grows.

    MEASURED (2026-07-21, the 5-anchor fixture from ``test_propose_anchor_admits_novel_
    rejects_covered``, all three previously indistinguishable as ``None``)::

        near-duplicate  layer=L2  density=0.9994  t_low=0.5167  novelty_margin=-0.4826
        halfway signal  layer=L1  density=0.7308  t_low=0.5167  novelty_margin=-0.2141
        no live set     layer=None                              reason="no_anchorset"

    The second is a genuinely ambiguous signal sitting between the cluster and empty space;
    the first is an item lying on top of an existing anchor; the third is not a geometric
    decision at all. One return value covered all three.

    THE THRESHOLDS ARE RECORDED FOR A REASON. In the same run, admitting ONE anchor moved
    the fitted ``t_low`` from 0.9980 to 0.5167 — the thresholds are refitted from the
    anchors' own spacing on every growth step, so a density logged without the threshold it
    was compared against is a number nobody can re-judge later. Both are kept.

    ``anchor`` is ``None`` on every reject path. ``reason`` names which one.
    """

    anchor: Optional[Anchor]
    reason: str                       # "admitted" | "not_novel" | "no_anchorset" | "dim_mismatch"
    layer: Optional[str] = None       # the density-zoom layer, when one was computed
    density: Optional[float] = None   # the continuous nearest-anchor cosine behind that layer
    t_low: Optional[float] = None     # the fitted L0|L1 threshold it was compared against
    t_high: Optional[float] = None    # the fitted L1|L2 threshold

    @property
    def admitted(self) -> bool:
        return self.anchor is not None

    @property
    def novelty_margin(self) -> Optional[float]:
        """``t_low - density``: how far INTO the novel region the signal sat. Positive means
        novel (admitted, if ``novel_only``); near zero either way means the decision turned
        on a threshold that is refitted on every growth step and should not be read as
        settled. ``None`` when no density was computed."""
        if self.density is None or self.t_low is None:
            return None
        return float(self.t_low) - float(self.density)

    def as_read(self) -> dict:
        """Flat, JSON-safe form for a log line or an audit record."""
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "layer": self.layer,
            "density": None if self.density is None else round(self.density, 6),
            "t_low": None if self.t_low is None else round(self.t_low, 6),
            "t_high": None if self.t_high is None else round(self.t_high, 6),
            "novelty_margin": (None if self.novelty_margin is None
                               else round(self.novelty_margin, 6)),
        }


def propose_anchor(
    label: str,
    vec: Sequence[float] | np.ndarray,
    *,
    model_id: Optional[str] = None,
    novel_only: bool = True,
) -> Optional[Anchor]:
    """Admit ``vec`` (labelled ``label``) as a new CANDIDATE anchor if it is
    novel — in a region the AnchorSet doesn't yet cover (density-zoom L0).

    Returns the new :class:`Anchor`, or ``None`` when there is no live AnchorSet,
    the dimension mismatches, or the signal is already covered (``novel_only``).
    A foreign ``model_id`` is projected via the cross-walk registry first.
    Persists the grown set (and refreshes the density caches so thresholds refit).

    Return type is unchanged (``Optional[Anchor]``) — ``manage_anchors.py`` and the tests
    both rely on it. Use :func:`propose_anchor_decided` when you need the continuous density
    that produced the decision; this is a thin wrapper over it.
    """
    return propose_anchor_decided(
        label, vec, model_id=model_id, novel_only=novel_only
    ).anchor


def propose_anchor_decided(
    label: str,
    vec: Sequence[float] | np.ndarray,
    *,
    model_id: Optional[str] = None,
    novel_only: bool = True,
) -> GrowthDecision:
    """:func:`propose_anchor`, with the reading that produced the decision.

    Same admit/reject behaviour, bit for bit — the boolean is deliberately untouched. What
    is added is the continuous density (and the fitted thresholds it was compared against)
    on BOTH paths, so a near-threshold rejection is auditable instead of being an
    indistinguishable ``None``.
    """
    aset = get_live_anchorset()
    if aset is None or len(aset) == 0:
        return GrowthDecision(None, "no_anchorset")

    v = np.asarray(vec, dtype=np.float32).ravel()
    if model_id and model_id != aset.model_id:
        v = np.asarray(
            get_crosswalks().walk(vec, model_id, aset.model_id), dtype=np.float32
        ).ravel()
    if v.shape[-1] != aset.dim:
        return GrowthDecision(None, "dim_mismatch")

    layer: Optional[str] = None
    density: Optional[float] = None
    t_low: Optional[float] = None
    t_high: Optional[float] = None

    if novel_only:
        dz = get_density_zoom()
        if dz is not None:
            # `layer()` has ALWAYS returned `(layer, density)`; the old call indexed `[0]`
            # and dropped the float on the floor. The cosine costs nothing extra — it is
            # what the layer was computed FROM.
            layer, density = dz.layer(v)
            # Private, but same package, and read-only: these are the fitted thresholds the
            # layer decision turned on. They are refitted from the anchors' own spacing on
            # every growth step, so recording the decision without them records a comparison
            # against a number nobody can reconstruct later. getattr keeps this from being a
            # hard coupling if DensityZoom's internals are renamed.
            t_low = getattr(dz, "_t_low", None)
            t_high = getattr(dz, "_t_high", None)
            if layer != L0:
                decision = GrowthDecision(None, "not_novel", layer, float(density),
                                          t_low, t_high)
                # Logged at DEBUG: a covered signal is the common case and this fires per
                # proposal. The margin is the number that matters — a rejection at
                # margin=-0.001 is a coin flip on a threshold that will move at the next
                # refit; one at -0.4 is an item sitting on an existing anchor.
                logger.debug(
                    "AnchorSet growth rejected %r: layer=%s density=%.4f "
                    "t_low=%s novelty_margin=%s",
                    label, layer, float(density), t_low, decision.novelty_margin,
                )
                return decision

    anchor = Anchor.make(label, v, aset.model_id, tier=CANDIDATE)
    get_anchor_repo().add(anchor)   # persist as a vnd.agience.anchor+json artifact
    reset_anchorset()               # density refits with the new anchor on next load
    logger.info(
        "Grew AnchorSet: +1 candidate anchor %r (layer=%s density=%s t_low=%s)",
        label, layer,
        None if density is None else round(float(density), 4), t_low,
    )
    return GrowthDecision(anchor, "admitted", layer,
                          None if density is None else float(density), t_low, t_high)
