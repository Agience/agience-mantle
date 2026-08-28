"""A cross-walk is an artifact — because everything is an artifact, and because refitting one
per device is waste the mesh exists to remove.

WHY THIS EXISTS
---------------
A cross-walk is *derived data with provenance*: it was fitted from a specific AnchorSet, over a
specific pair of spaces, and it carries a measured residual. That is an artifact's shape, not a
runtime detail.

The practical consequence is the interesting one. The fit is **deterministic given (source
space, AnchorSet)**: least-squares and orthogonal Procrustes have no seed and no stochasticity.
So *every leaf deriving the same source space against the same anchors computes the identical
matrix.* Today each one refits from scratch on boot. As an artifact it is fitted once, cached,
signed, and shared — which is precisely what the mesh is for. A leaf can then be aligned before
it has ever seen an anchor's label, by holding a cross-walk someone else fitted, and verify it
rather than trust it (`verify_crosswalk_artifact`).

THE TRAP THIS MODULE GUARDS
---------------------------
A cross-walk is only valid for the exact `(source_space_id, target_space_id, anchorset
fingerprint)` it was fitted against. Reusing one across a *changed* AnchorSet is silently wrong:
it still projects, still yields unit vectors, still routes — into cells chosen by yesterday's
geometry. Nothing raises. So the fingerprint is part of the identity, and the id is derived from
it: a stale cross-walk cannot be mistaken for a current one because it has a different id.

READING WHAT IS ALREADY ON DISK
-------------------------------
Artifacts written before the rename carry `source_model_id` / `target_model_id` and a bare
`error_bound` float. Both are READ here (see `_space_id` and `_residual`) so no stored walk is
orphaned; only the write side is current. The id is unaffected — `crosswalk_id` hashes the
VALUES, never the key names, so a re-read artifact keeps the id it was published under.
"""
from __future__ import annotations

import base64
import hashlib
import uuid
from typing import Optional

import numpy as np

from mantle.search.anchors.anchorset import AnchorSet
from mantle.search.anchors.crosswalk import Crosswalk, FitResidual, PROCRUSTES
from prism.mass import Provenance

_CW_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "crosswalks.agience.ai")

CROSSWALK_CONTENT_TYPE = "application/vnd.agience.crosswalk+json"

# current context key → the pre-rename key a stored artifact may carry instead
_LEGACY_KEYS = {
    "source_space_id": "source_model_id",
    "target_space_id": "target_model_id",
}


def _space_id(ctx: dict, key: str) -> Optional[str]:
    """Read a space id, accepting the pre-rename key. The current name wins."""
    v = ctx.get(key)
    return v if v is not None else ctx.get(_LEGACY_KEYS[key])


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


def crosswalk_id(source_space_id: str, target_space_id: str, fingerprint: str) -> str:
    """Deterministic id over exactly what makes a cross-walk valid.

    The fingerprint is in here on purpose: a cross-walk fitted against a *different* AnchorSet
    is a different object, and must not be able to masquerade as this one. Grow the anchors and
    the id changes, so a stale walk is a cache miss rather than a silent mis-projection.

    Hashes the values only, so renaming a field moves no id.
    """
    return str(uuid.uuid5(_CW_NS, f"{source_space_id}|{target_space_id}|{fingerprint}"))


def to_artifact(cw: Crosswalk, anchors: AnchorSet, *, fitted_by: Optional[str] = None) -> dict:
    """Serialise a fitted cross-walk as an artifact, shaped for Mantle.

    The matrix rides base64'd in `context` rather than as `content` bytes so the whole record
    stays one self-describing JSON object — dtype and shape travel *with* the payload, because
    a float matrix without its dtype is a very convincing wrong answer.

    Provenance is `observed`, not `human_validated` or `hypothesis`: nobody asserted this and
    nobody guessed it. It is the deterministic output of a stated computation over stated
    inputs — reproducible by anyone holding the same anchors and the same source space, which is
    exactly the claim `observed` makes.

    `error_bound` is not written: it is an in-sample residual under a name that promises a bound,
    and emitting it would carry that promise into every artifact a peer reads. The
    `residual` block carries the same in-sample number, the held-out one that actually answers
    "how much fidelity does this cost", and whether the walk was SOLVED or FITTED — so a
    consumer can decide before querying rather than discover the loss afterwards.
    """
    m = np.ascontiguousarray(cw.matrix, dtype=np.float32)
    fp = anchorset_fingerprint(anchors)
    r = cw.residual
    return {
        "id": crosswalk_id(cw.source_space_id, cw.target_space_id, fp),
        "content_type": CROSSWALK_CONTENT_TYPE,
        "name": f"{cw.source_space_id} -> {cw.target_space_id}",
        "context": {
            "provenance": Provenance.OBSERVED.value,
            "source_space_id": cw.source_space_id,
            "target_space_id": cw.target_space_id,
            "anchorset_fingerprint": fp,
            "anchorset_model_id": anchors.model_id,
            "n_anchors": len(anchors),
            "method": cw.method,
            "closed_form": bool(cw.closed_form),
            "is_isometry": bool(cw.is_isometry),
            "dim_in": int(cw.dim_in),
            "dim_out": int(cw.dim_out),
            "residual": {
                "in_sample": float(r.in_sample),
                "held_out": None if r.held_out is None else float(r.held_out),
                "n_pairs": int(r.n_pairs),
                "n_held_out": int(r.n_held_out),
                "note": r.note,
            },
            "matrix_dtype": "float32",
            "matrix_shape": [int(m.shape[0]), int(m.shape[1])],
            "matrix_b64": base64.b64encode(m.tobytes()).decode("ascii"),
            "fitted_by": fitted_by or "ember",
        },
        "content": "",
    }


def _residual(ctx: dict, dim_in: int, dim_out: int, method: str) -> FitResidual:
    """Rebuild the residual, from the current block or from a pre-rename `error_bound`.

    A legacy artifact recorded ONE number and did not say how many pairs produced it, so the
    rebuilt residual reports `n_pairs=0` — which makes `underdetermined` return None (unknown,
    not a pass) and leaves `held_out` None with the reason stated. That is the honest reading of
    a record that never held the information.
    """
    block = ctx.get("residual")
    if isinstance(block, dict):
        held = block.get("held_out")
        return FitResidual(
            in_sample=float(block.get("in_sample", 0.0)),
            held_out=None if held is None else float(held),
            n_pairs=int(block.get("n_pairs", 0)),
            n_held_out=int(block.get("n_held_out", 0)),
            dim_in=dim_in,
            dim_out=dim_out,
            closed_form=(method == PROCRUSTES),
            note=str(block.get("note", "ok")),
        )
    if "error_bound" in ctx:
        return FitResidual(
            in_sample=float(ctx["error_bound"]),
            held_out=None,
            n_pairs=0,
            n_held_out=0,
            dim_in=dim_in,
            dim_out=dim_out,
            closed_form=(method == PROCRUSTES),
            note="rebuilt from a pre-rename artifact: `error_bound` was IN-SAMPLE, and the "
                 "record does not say over how many pairs — refit to learn anything more",
        )
    raise ValueError("no residual block and no legacy error_bound")


def from_artifact(artifact: dict) -> Crosswalk:
    """Rebuild a cross-walk from its artifact. Raises rather than guessing."""
    ctx = artifact.get("context") or {}
    if artifact.get("content_type") != CROSSWALK_CONTENT_TYPE:
        raise ValueError(f"not a cross-walk artifact: {artifact.get('content_type')!r}")
    try:
        shape = tuple(int(x) for x in ctx["matrix_shape"])
        raw = base64.b64decode(ctx["matrix_b64"])
        matrix = np.frombuffer(raw, dtype=np.dtype(ctx["matrix_dtype"])).reshape(shape)
        method = ctx["method"]
        dim_in, dim_out = int(ctx["dim_in"]), int(ctx["dim_out"])
        source_space_id = _space_id(ctx, "source_space_id")
        target_space_id = _space_id(ctx, "target_space_id")
        if source_space_id is None or target_space_id is None:
            raise KeyError("source_space_id/target_space_id")
        residual = _residual(ctx, dim_in, dim_out, method)
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"malformed cross-walk artifact: {e}") from e
    return Crosswalk(
        source_space_id=source_space_id,
        target_space_id=target_space_id,
        method=method,
        matrix=np.array(matrix, dtype=np.float32),   # copy: frombuffer is read-only
        dim_in=dim_in,
        dim_out=dim_out,
        residual=residual,
    )


def verify_crosswalk_artifact(artifact: dict, anchors: AnchorSet,
                              local_space_id: str) -> tuple[bool, str]:
    """Is this cross-walk usable *here*, right now? Returns (ok, reason).

    ``local_space_id`` names the space this leaf's own vectors live in — whatever derived the
    basis. Checked rather than trusted, because every failure mode is silent: a cross-walk
    fitted from another space, or against anchors that have since grown, still projects and
    still returns unit vectors. It just routes into cells chosen by the wrong geometry. Nothing
    raises, results look plausible, and the shards go to a private universe.
    """
    ctx = artifact.get("context") or {}
    if artifact.get("content_type") != CROSSWALK_CONTENT_TYPE:
        return False, f"wrong content_type: {artifact.get('content_type')!r}"
    src, tgt = _space_id(ctx, "source_space_id"), _space_id(ctx, "target_space_id")
    if src != local_space_id:
        return False, f"fitted from space {src!r}, this leaf derives {local_space_id!r}"
    if tgt != anchors.model_id:
        return False, f"targets {tgt!r}, these anchors are {anchors.model_id!r}"
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
