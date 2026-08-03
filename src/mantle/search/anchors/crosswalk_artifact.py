"""A cross-walk is an artifact — because everything is an artifact, and because refitting one
per device is waste the mesh exists to remove.

WHY THIS EXISTS
---------------
A cross-walk is *derived data with provenance*: it was fitted from a specific AnchorSet, by a
specific embedder, and it carries `error_bound` — a measured quality. That is an artifact's
shape, not a runtime detail.

The practical consequence is the interesting one. The fit is **deterministic given (embedder,
AnchorSet)**: least-squares and orthogonal Procrustes have no seed and no stochasticity. So
*every leaf running the same embedder against the same anchors computes the identical matrix.*
Today each one refits from scratch on boot. As an artifact it is fitted once, cached, signed,
and shared — which is precisely what the mesh is for. A leaf can then be aligned before it has
ever seen an anchor's label, by holding a cross-walk someone else fitted, and verify it rather
than trust it (`verify_crosswalk_artifact`).

THE TRAP THIS MODULE GUARDS
---------------------------
A cross-walk is only valid for the exact `(source_model_id, target_model_id, anchorset
fingerprint)` it was fitted against. Reusing one across a *changed* AnchorSet is silently
wrong: it still projects, still yields unit vectors, still routes — into cells chosen by
yesterday's geometry. Nothing raises. So the fingerprint is part of the identity, and the id is
derived from it: a stale cross-walk cannot be mistaken for a current one because it has a
different id.
"""
from __future__ import annotations

import base64
import hashlib
import uuid
from typing import Optional

import numpy as np

from mantle.search.anchors.anchorset import AnchorSet
from mantle.search.anchors.crosswalk import Crosswalk
from prism.mass import Provenance

_CW_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "crosswalks.agience.ai")

CROSSWALK_CONTENT_TYPE = "application/vnd.agience.crosswalk+json"


def anchorset_fingerprint(anchors: AnchorSet) -> str:
    """A content hash of the anchor set a cross-walk was fitted against.

    Over the *anchor ids*, sorted — ids are already content-addressed over
    (label, model_id, embedding), so this transitively covers every anchor's vector without
    re-hashing float arrays (whose bytes are platform-sensitive). Sorted because an AnchorSet
    is a set: insertion order must not change its identity.
    """
    h = hashlib.sha256()
    h.update(b"agience/anchorset/v1")
    h.update(anchors.model_id.encode("utf-8"))
    h.update(str(anchors.dim).encode("utf-8"))
    for aid in sorted(a.anchor_id for a in anchors.anchors):
        h.update(aid.encode("utf-8"))
    return h.hexdigest()


def crosswalk_id(source_model_id: str, target_model_id: str, fingerprint: str) -> str:
    """Deterministic id over exactly what makes a cross-walk valid.

    The fingerprint is in here on purpose: a cross-walk fitted against a *different* AnchorSet
    is a different object, and must not be able to masquerade as this one. Grow the anchors and
    the id changes, so a stale walk is a cache miss rather than a silent mis-projection.
    """
    return str(uuid.uuid5(_CW_NS, f"{source_model_id}|{target_model_id}|{fingerprint}"))


def to_artifact(cw: Crosswalk, anchors: AnchorSet, *, fitted_by: Optional[str] = None) -> dict:
    """Serialise a fitted cross-walk as an artifact, shaped for Mantle.

    The matrix rides base64'd in `context` rather than as `content` bytes so the whole record
    stays one self-describing JSON object — dtype and shape travel *with* the payload, because
    a float matrix without its dtype is a very convincing wrong answer.

    Provenance is `observed`, not `human_validated` or `hypothesis`: nobody asserted this and
    nobody guessed it. It is the deterministic output of a stated computation over stated
    inputs — reproducible by anyone holding the same anchors and embedder, which is exactly the
    claim `observed` makes.
    """
    m = np.ascontiguousarray(cw.matrix, dtype=np.float32)
    fp = anchorset_fingerprint(anchors)
    return {
        "id": crosswalk_id(cw.source_model_id, cw.target_model_id, fp),
        "content_type": CROSSWALK_CONTENT_TYPE,
        "name": f"{cw.source_model_id} -> {cw.target_model_id}",
        "context": {
            "provenance": Provenance.OBSERVED.value,
            "source_model_id": cw.source_model_id,
            "target_model_id": cw.target_model_id,
            "anchorset_fingerprint": fp,
            "anchorset_model_id": anchors.model_id,
            "n_anchors": len(anchors),
            "method": cw.method,
            "dim_in": int(cw.dim_in),
            "dim_out": int(cw.dim_out),
            # What the small embedder costs, carried WITH the walk: a consumer can decide
            # whether to accept it instead of discovering the loss at query time.
            "error_bound": float(cw.error_bound),
            "matrix_dtype": "float32",
            "matrix_shape": [int(m.shape[0]), int(m.shape[1])],
            "matrix_b64": base64.b64encode(m.tobytes()).decode("ascii"),
            "fitted_by": fitted_by or "ember",
        },
        "content": "",
    }


def from_artifact(artifact: dict) -> Crosswalk:
    """Rebuild a cross-walk from its artifact. Raises rather than guessing."""
    ctx = artifact.get("context") or {}
    if artifact.get("content_type") != CROSSWALK_CONTENT_TYPE:
        raise ValueError(f"not a cross-walk artifact: {artifact.get('content_type')!r}")
    try:
        shape = tuple(int(x) for x in ctx["matrix_shape"])
        raw = base64.b64decode(ctx["matrix_b64"])
        matrix = np.frombuffer(raw, dtype=np.dtype(ctx["matrix_dtype"])).reshape(shape)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"malformed cross-walk artifact: {e}") from e
    return Crosswalk(
        source_model_id=ctx["source_model_id"],
        target_model_id=ctx["target_model_id"],
        method=ctx["method"],
        matrix=np.array(matrix, dtype=np.float32),   # copy: frombuffer is read-only
        dim_in=int(ctx["dim_in"]),
        dim_out=int(ctx["dim_out"]),
        error_bound=float(ctx["error_bound"]),
    )


def verify_crosswalk_artifact(artifact: dict, anchors: AnchorSet,
                              embedder_model_id: str) -> tuple[bool, str]:
    """Is this cross-walk usable *here*, right now? Returns (ok, reason).

    Checked rather than trusted, because every failure mode is silent: a cross-walk fitted for
    another embedder, or against anchors that have since grown, still projects and still
    returns unit vectors. It just routes into cells chosen by the wrong geometry. Nothing
    raises, results look plausible, and the shards go to a private universe.
    """
    ctx = artifact.get("context") or {}
    if artifact.get("content_type") != CROSSWALK_CONTENT_TYPE:
        return False, f"wrong content_type: {artifact.get('content_type')!r}"
    if ctx.get("source_model_id") != embedder_model_id:
        return False, (f"fitted for embedder {ctx.get('source_model_id')!r}, "
                       f"this leaf runs {embedder_model_id!r}")
    if ctx.get("target_model_id") != anchors.model_id:
        return False, (f"targets {ctx.get('target_model_id')!r}, "
                       f"these anchors are {anchors.model_id!r}")
    fp = anchorset_fingerprint(anchors)
    if ctx.get("anchorset_fingerprint") != fp:
        return False, "anchor set has changed since this was fitted — refit (ids would drift)"
    if int(ctx.get("dim_out", -1)) != anchors.dim:
        return False, f"projects to {ctx.get('dim_out')} dims, anchors are {anchors.dim}"
    return True, "ok"


__all__ = [
    "CROSSWALK_CONTENT_TYPE", "anchorset_fingerprint", "crosswalk_id",
    "to_artifact", "from_artifact", "verify_crosswalk_artifact",
]
