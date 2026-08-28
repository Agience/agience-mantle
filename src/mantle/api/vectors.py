"""Supplied vectors — the ingress of the semantic arm, on both sides.

Mantle never embeds. It receives vectors that someone else already produced, which is
why the request carries a ``space_id`` alongside the numbers: two vectors are only
comparable when they live in the same space, and Mantle has no way to infer which
space a bare list of floats came from. There is one space per node — the one the seeded
AnchorSet names — and nothing here bridges two, so the name is the load-bearing part of
the payload: a vector without one is unusable, and that is why ``space_id`` is required
whenever ``vector`` is present.

Two directions, one contract. A WRITER supplies the vector of the text it is storing
(``POST /artifacts``); a READER supplies the vector it wants ranked against
(``POST /artifacts/recall``). Both are numbers computed elsewhere, so both are
validated here by :func:`validate_vector` and both fail the same way. They part company
only after validation: a stored vector keeps the writer's own space name as its
provenance, while a query vector has to be expressed in the coordinate system the
routing actually uses — see :func:`project_to_anchor_space`.

**Shape is validated; quality never is.** The checks below answer "can this be placed
at all" — finite numbers, a positive dimension, a non-zero norm, and (when an
AnchorSet is live) the dimension the anchors are expressed in. They do not answer
"is this a good vector", which is a question about someone else's model and none of
Mantle's business.

Normalization is not required of the caller: :func:`search.anchors.anchorset.l2norm`
unit-normalizes on the routing path, so any positive-norm vector routes identically
whether or not the writer normalized it. A zero vector is the one exception — it has
no direction, so no nearest anchor exists for it, and it is refused here rather than
failing inside the indexer where the writer would never see it.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

#: Upper bound on an accepted dimension. Not a tuning knob — a bound on how much
#: an unauthenticated-shaped payload can make this process allocate before the
#: light cone has said anything. Every embedding space in practical use is far
#: under it.
MAX_VECTOR_DIM = 16384


class VectorIngressError(ValueError):
    """A supplied vector cannot be placed. Routers surface this as a 4xx."""


class SuppliedVector(BaseModel):
    """One writer-supplied vector and the space it belongs to."""

    values: List[float] = Field(..., description="The vector's components, in space order.")
    space_id: str = Field(
        ...,
        description=(
            "Name of the embedding space these components live in. Required: it is what "
            "makes two vectors comparable. On a query it must equal the seeded AnchorSet's "
            "model_id — that is the one space this node ranks in."
        ),
    )


def validate_vector(values: Sequence[float], space_id: Optional[str]) -> SuppliedVector:
    """Return the validated vector, or raise :class:`VectorIngressError`.

    Shape only. The AnchorSet dimension is consulted when one is live, because a
    vector of the wrong width cannot be routed and there is no unrouted path — so
    the writer should learn that at the write, not from a background index job.
    When no AnchorSet is provisioned the dimension is accepted as-is; the vector arm
    already reports that deployment state on its own.

    That acceptance is the writer's, not the reader's. A stored vector on an unseeded node is
    provenance for data at rest and is placed when a set arrives, so accepting it costs the
    writer nothing; a QUERY vector on the same node can never be placed, and
    :func:`project_to_anchor_space` refuses it there rather than answering by recency.
    """
    if not space_id or not space_id.strip():
        raise VectorIngressError(
            "space_id is required when a vector is supplied — without it Mantle cannot "
            "tell whether this vector is comparable to any other"
        )

    dim = len(values)
    if dim == 0:
        raise VectorIngressError("vector must have at least one component")
    if dim > MAX_VECTOR_DIM:
        raise VectorIngressError(f"vector dimension {dim} exceeds the maximum of {MAX_VECTOR_DIM}")

    total = 0.0
    for i, raw in enumerate(values):
        try:
            component = float(raw)
        except (TypeError, ValueError):
            raise VectorIngressError(f"vector component {i} is not a number")
        if not math.isfinite(component):
            raise VectorIngressError(f"vector component {i} is not finite ({raw!r})")
        total += component * component

    if total <= 0.0:
        raise VectorIngressError(
            "vector has zero norm — it names no direction, so no nearest anchor exists for it"
        )

    expected = anchorset_dim()
    if expected is not None and dim != expected:
        raise VectorIngressError(
            f"vector dimension {dim} does not match this node's anchor geometry ({expected}). "
            "The AnchorSet is the one coordinate system, so a vector of another width names no "
            f"point in it. TO FIX: send {expected}-dimensional vectors from the model that "
            "produced the anchors, or seed the AnchorSet that belongs to this width "
            "(`python -m mantle.system.manage_anchors --action load --path anchors.json`)."
        )

    return SuppliedVector(values=[float(v) for v in values], space_id=space_id.strip())


def project_to_anchor_space(vector: SuppliedVector) -> List[float]:
    """The query vector, confirmed to be in the coordinate system the routed path reads.

    A query is placed by :func:`search.anchors.routing.route_query` against the live
    AnchorSet, and then compared by raw cosine to the vectors inside whichever cells
    that placement opened. Both steps are statements in the AnchorSet's basis, so a
    vector from another basis does not produce a worse answer — it produces a number
    that is not a similarity at all, while looking exactly like one.

    **The client's space IS the anchor space.** The client seeds the AnchorSet, so it owns
    both halves of the match: the file states its ``model_id`` and every query names the same
    one. Mantle projects between no two spaces — a projection needs the SAME texts embedded in
    both, and Mantle has no model with which to produce either half — so a name that does not
    match is refused here, by name, rather than silently ranked into nonsense.

    **NO AnchorSet is refused by the same rule, and it is one rule rather than two.** A space
    name is unusable when this node ranks in a different one, and it is unusable when this node
    ranks in none — the second is the case where EVERY name is unusable, not the case where any
    name will do. Passing the numbers through instead produced a 200 ordered by something the
    caller did not ask for — by query coverage for a text+vector recall, by recency for a
    vector-only one — and indistinguishable from a real answer, because that is the same body a
    query carrying no vector at all comes back as. A caller that supplied a vector could not
    tell that it had been ignored.

    The 400 covers a hybrid query too: a text+vector recall whose vector names a foreign space is
    refused, even though its text half would narrow perfectly well. Refusing a foreign space and
    accepting a nonexistent one would be the same request answered two ways. Dropping the
    ``vector`` field turns any such call into a text query, which works — see
    :meth:`search.mantle.sse.router_accessor.MantleSseSearchAccessor._by_coverage`, whose
    "a 400 would refuse a query that worked" holds precisely there: a caller that sends no
    vector is one that CANNOT embed, and its query really was answered. A caller that sends one
    can, and asked for it to be used.

    :func:`validate_vector` still accepts any width in this state, because it is also the
    WRITER's door: a stored vector is provenance for data at rest and is placed when a set
    arrives, so refusing it would refuse a write that a later seeding makes good.
    """
    try:
        from mantle.search.anchors.store import get_live_anchorset
        aset = get_live_anchorset()
    except ImportError:
        # No vector arm in this install (the lexical-only surface). Nothing here ranks, which
        # is the unseeded answer below by a different route.
        aset = None
    except Exception as exc:
        # A set that exists and could not be READ is not the same state as no set, and must
        # never be reported as one — `search/anchors/store.py` refuses to collapse
        # `AnchorSetCorrupt` into `None` for exactly this reason. Same door, same refusal to
        # answer with a corpus dump, different sentence.
        raise VectorIngressError(
            f"query vector names space {vector.space_id!r}, and this node's anchor geometry "
            f"could not be read, so the vector cannot be placed: {exc}. This is a fault in "
            f"this node, not in the request — inspect it with "
            f"`python -m mantle.system.manage_anchors --action inspect`."
        ) from exc
    if aset is None or len(aset) == 0:
        raise VectorIngressError(
            f"query vector names space {vector.space_id!r}, and this node has no AnchorSet — it "
            f"ranks in no space at all, so there is nothing for this vector to be placed "
            f"against. Answering anyway would return everything you can read in recency order, "
            f"which is not this query. "
            f"TO FIX: seed the AnchorSet for {vector.space_id!r} "
            f"(`python -m mantle.system.manage_anchors --action load --path anchors.json`) and "
            f"reindex; or send this recall as a text query, without `vector`, which narrows on "
            f"the query's terms and answers most-recently-updated first."
        )

    target = str(aset.model_id)
    if vector.space_id != target:
        raise VectorIngressError(
            f"query vector names space {vector.space_id!r}, and this node's AnchorSet is "
            f"expressed in {target!r}. Mantle ranks by cosine in the anchors' own basis and "
            f"projects between no two spaces, so a vector from another one would score as a "
            f"similarity without being one. "
            f"TO FIX: send this query with space_id={target!r} — embed it with the same model "
            f"that produced the anchors. If {vector.space_id!r} is the space you meant to serve, "
            f"seed the AnchorSet that belongs to it "
            f"(`python -m mantle.system.manage_anchors --action load --path anchors.json`) and "
            f"reindex; one node serves one space."
        )
    return list(vector.values)


def anchorset_dim() -> Optional[int]:
    """The live AnchorSet's dimension, or ``None`` when none is provisioned.

    Read through the same store the ingest arm reads, so the width accepted here and
    the width routing requires are one number rather than two that agree by habit.
    """
    try:
        from mantle.search.anchors.store import get_live_anchorset
        aset = get_live_anchorset()
    except Exception:
        return None
    if aset is None or len(aset) == 0:
        return None
    return int(aset.dim)


__all__ = [
    "MAX_VECTOR_DIM",
    "SuppliedVector",
    "VectorIngressError",
    "anchorset_dim",
    "project_to_anchor_space",
    "validate_vector",
]
