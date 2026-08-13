"""Anchor routing — map vectors/queries to cells (cluster = routing anchor).

The AnchorSet is the partition (canonical plan §5.1): a chunk lands in the cell
of its nearest anchor; a query fans out to its ``nprobe`` nearest anchors. Pure
geometry — no keys/auth (the §1 invariant).

Every vector is anchor-routed; there is no other path. The AnchorSet is mandatory
(``store.require_live_anchorset`` returns the seeded set, or raises — nothing here derives one,
on first use or ever). There is no flat cell and no unrouted fallback — a vector that cannot be
placed against the anchors (empty set, dimension mismatch) is an error, not a second path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .anchorset import AnchorSet

__all__ = [
    "DEFAULT_NPROBE", "route_vector", "route_query",
    "RouteDecision", "QueryRoute", "route_vector_scored", "route_query_scored",
]


# An artifact lands in exactly one cell — the assignment is discrete — but the evidence for
# the pick is worth keeping alongside it. `route_vector`/`route_query` return just the
# winner; `route_vector_scored`/`route_query_scored` below are companion functions that also
# return the margin between the winner and the runner-up, so an assignment at 0.7001 vs.
# 0.7000 is distinguishable in the record from one at 0.91 vs. 0.12.
#
# The margin is the prerequisite for migrating an IC epoch affordably: when the anchor
# vocabulary shifts, a re-cell of the whole corpus is on the order of millions of artifacts.
# With the margin recorded at write time, the re-cell can be a filter — only artifacts whose
# winning margin is smaller than the epoch's perturbation can possibly change cell.
#
# `route_vector`/`route_query` keep their exact return types, since `mantle.mesh.anchor_routing`,
# `search.mantle.indexer` and `search.mantle.engine` all call them positionally and unpack the
# result directly.


@dataclass(frozen=True)
class RouteDecision:
    """The cell that was picked, plus the continuous evidence that picked it.

    anchor_id / score
        The winner and its cosine affinity.
    runner_up_id / runner_up_score
        The anchor that came second, and its cosine. ``None`` when the AnchorSet holds
        exactly one anchor — there is no contest, which is itself the honest reading.
    margin
        ``score - runner_up_score``, i.e. how decisively the winner won. ``float('inf')``
        when there is no runner-up (uncontested), never 0.0 — a one-anchor set is maximally
        decided, and reporting it as a tie would invert the meaning.
    """

    anchor_id: str
    score: float
    runner_up_id: Optional[str]
    runner_up_score: Optional[float]
    margin: float

    def as_read(self) -> dict:
        """Flat, JSON-safe form for a chunk record or a log line."""
        return {
            "anchor_id": self.anchor_id,
            "score": round(self.score, 6),
            "runner_up_id": self.runner_up_id,
            "runner_up_score": (None if self.runner_up_score is None
                                else round(self.runner_up_score, 6)),
            # inf is not JSON. An uncontested route reports its margin as None, which reads
            # as "no contest", not as "we did not measure it".
            "margin": (None if not np.isfinite(self.margin) else round(self.margin, 6)),
        }


@dataclass(frozen=True)
class QueryRoute:
    """The nprobe fan-out, with every probe's cosine retained.

    ``anchor_ids`` is exactly what :func:`route_query` returns, so a caller can migrate one
    field at a time. ``scores`` is positionally aligned with it. ``margin`` is the top-1 gap
    — the number that says whether the FIRST cell (the one a matching item would index into)
    was decided or coin-flipped.
    """

    anchor_ids: List[str]
    scores: List[float]
    margin: float

    @property
    def probes(self) -> List[Tuple[str, float]]:
        return list(zip(self.anchor_ids, self.scores))

    def as_read(self) -> dict:
        return {
            "anchor_ids": list(self.anchor_ids),
            "scores": [round(s, 6) for s in self.scores],
            "margin": (None if not np.isfinite(self.margin) else round(self.margin, 6)),
        }


def route_vector(anchorset: AnchorSet, vec: Sequence[float] | np.ndarray) -> str:
    """Cluster id (anchor_id) of the cell this vector indexes into — its nearest
    anchor.

    Raises :class:`ValueError` when the AnchorSet cannot place the vector (empty
    set or embedding/anchor dimension mismatch). There is no flat fallback.
    """
    near = anchorset.nearest(vec, k=1)
    if not near:
        raise ValueError(
            "route_vector: AnchorSet produced no nearest anchor "
            "(empty AnchorSet or embedding/anchor dimension mismatch)"
        )
    return near[0][0].anchor_id


#: How many cells a query opens. Cost, not meaning: every probe is one more cell to fetch and
#: decrypt, and the cosine over the union decides the answer either way. Eight is the width at
#: which a query sitting on a cell boundary still sees the neighbours it is between. A caller
#: bounding its own cost passes its own number — that is a claim about their machine, and it is
#: theirs to make.
DEFAULT_NPROBE = 8


def route_query(
    anchorset: AnchorSet,
    vec: Sequence[float] | np.ndarray,
    *,
    nprobe: int = DEFAULT_NPROBE,
) -> List[str]:
    """Candidate cluster ids (nearest anchors) a query must search. The nearest
    anchor — the cell a matching item would index into — is always first.

    Raises :class:`ValueError` when the AnchorSet cannot place the query (empty
    set or embedding/anchor dimension mismatch). There is no flat fallback.
    """
    return route_query_scored(anchorset, vec, nprobe=nprobe).anchor_ids


def route_vector_scored(
    anchorset: AnchorSet, vec: Sequence[float] | np.ndarray
) -> RouteDecision:
    """:func:`route_vector` with the evidence kept — the same cell, plus the margin.

    ``.anchor_id`` is byte-identical to what :func:`route_vector` returns for the same
    inputs: this asks for ``k=2`` instead of ``k=1``, and ``AnchorSet.nearest`` sorts
    descending, so the winner is the same element either way. The extra probe costs one more
    entry out of an argpartition that already scanned every anchor — the matmul, which is the
    whole cost, is unchanged.

    (``route_vector`` is not implemented in terms of this. It is the hot ingest path — one
    call per chunk across the corpus — and it should not pay even a two-element sort for
    evidence its caller has not asked for.)
    """
    near = anchorset.nearest(vec, k=2)
    if not near:
        raise ValueError(
            "route_vector_scored: AnchorSet produced no nearest anchor "
            "(empty AnchorSet or embedding/anchor dimension mismatch)"
        )
    top_anchor, top_score = near[0]
    if len(near) < 2:
        # A single-anchor set. Uncontested, not tied: margin is infinite.
        return RouteDecision(top_anchor.anchor_id, float(top_score), None, None, float("inf"))
    second_anchor, second_score = near[1]
    return RouteDecision(
        anchor_id=top_anchor.anchor_id,
        score=float(top_score),
        runner_up_id=second_anchor.anchor_id,
        runner_up_score=float(second_score),
        margin=float(top_score) - float(second_score),
    )


def route_query_scored(
    anchorset: AnchorSet,
    vec: Sequence[float] | np.ndarray,
    *,
    nprobe: int = DEFAULT_NPROBE,
) -> QueryRoute:
    """:func:`route_query` with every probe's cosine kept.

    ``.anchor_ids`` equals ``route_query(...)`` exactly. The scores are what tell a reader
    whether the fan-out actually spanned anything: eight probes at 0.71/0.70/0.70/… is a
    query sitting on a cell boundary, and eight probes at 0.94/0.31/… is a query that only
    needed one. Both look identical in the current return value.
    """
    near = anchorset.nearest(vec, k=max(1, int(nprobe)))
    if not near:
        raise ValueError(
            "route_query_scored: AnchorSet produced no nearest anchor "
            "(empty AnchorSet or embedding/anchor dimension mismatch)"
        )
    ids = [a.anchor_id for a, _ in near]
    scores = [float(s) for _, s in near]
    # Uncontested (a single candidate) is infinite margin, not zero — see RouteDecision.
    margin = (scores[0] - scores[1]) if len(scores) >= 2 else float("inf")
    return QueryRoute(anchor_ids=ids, scores=scores, margin=margin)
