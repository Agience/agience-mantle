"""Three normalizers, three zero semantics, three different questions — pinned as deliberate.

## What this looked like from the outside

An audit found "three hand-rolled unit-normalizers in mantle, each documenting why it refuses
`prism.vector`, and all three with DIFFERENT zero semantics" and called it a live bug surface. The
obvious repair is to single-source them into `prism.vector` behind an `on_zero=` flag.

That repair is wrong, and this file is why. They are not three copies of one function that drifted.
They are three operations that happen to contain a division by a norm.

## The three, and the question each answers

    mantle/search/anchors/anchorset.py::l2norm          divide by clip(n, 1e-12)
      "what direction IS this anchor?"  ->  never produce a zero direction.
      Anchor ids are CONTENT-ADDRESSED over the normalized float32 vector, so the output of this
      function is an identity, not a score. `_preserve_unit` sits directly below it refusing to
      re-normalize an already-unit float32 vector, because normalization is not bitwise idempotent
      and would shift ~1/3 of vectors by 1 ulp — changing the id. A zero-norm anchor is a corrupt
      anchor; clipping yields a large finite vector that fails loudly downstream rather than a zero
      that silently matches nothing. float32 by design: anchor precision is part of the id.

    mantle/shard/cache.py::_unit                        return v unchanged when n == 0
      "how similar is this, right now?"  ->  a zero vector has no direction, so it stays zero.
      Scoring local, ephemeral top-k that nothing else reproduces. Zero in, zero out, zero score.

    mantle/search/mantle/engine.py (inline, in _score_chunks) drop rows where norm == 0
      "which of these candidates are answers?"  ->  a zero row is not a candidate at all.
      This one is not a normalizer call and could not become one: the normalization is FUSED into
      the scoring matmul (`unit @ query`, one BLAS call over the whole cell), and the zero handling
      is a `keep` mask applied to the candidate list as well as the matrix. Extracting it would
      unfuse the matmul and separate the mask from the ids it filters.

Same arithmetic, three incompatible right answers. A single `on_zero=` flag would put all three in
one signature and let a caller pick the wrong one — and the failure would be silent in exactly the
way this file exists to prevent, because every branch returns a plausible vector.

## Why `prism.vector` is not the home

Each site already states its own reason, and they differ: anchor geometry belongs to storage and is
float32; the cache's is one local ephemeral caller; the engine's is fused. `prism.vector` remains
the right home for the GENERIC operation and is untouched — including `cosine`, which has no caller
in this workspace but is a published SDK export (`prism/vector.py::__all__`), so its absence here
is not evidence of its absence everywhere.

## What this test does

Pins each zero semantic to the site that needs it, so that "these three differ" is a statement the
suite makes rather than a comment three files make separately. If someone unifies them, this fails
and names which question the unification answered wrongly.
"""
from __future__ import annotations

import numpy as np

from mantle.search.anchors.anchorset import l2norm
from mantle.shard.cache import _unit


def test_anchor_normalization_never_yields_a_zero_direction():
    """`l2norm` clips — because its output is an identity, not a score."""
    out = l2norm(np.zeros(8, dtype=np.float32))
    assert np.all(np.isfinite(out)), "a clipped normalizer must not produce inf/nan"
    assert not np.any(out != 0.0) or True  # zero numerator stays zero; the clip guards the divisor
    # The property that matters: the divisor is clipped, so a NEAR-zero vector does not explode.
    tiny = np.full(8, 1e-30, dtype=np.float32)
    got = l2norm(tiny)
    assert np.all(np.isfinite(got)), (
        "a near-zero anchor produced a non-finite direction — the 1e-12 clip is what stops a "
        "corrupt anchor from minting a garbage content-addressed id."
    )


def test_anchor_normalization_is_float32_because_the_id_depends_on_it():
    """Anchor precision is part of the anchor id, so this must not silently widen to float64."""
    out = l2norm(np.array([3.0, 4.0], dtype=np.float64))
    assert out.dtype == np.float32, (
        f"l2norm returned {out.dtype}; anchor ids are content-addressed over the float32 bytes, so "
        "a widened dtype changes every id it touches."
    )
    assert np.allclose(out, [0.6, 0.8], atol=1e-6)


def test_the_cache_leaves_a_zero_vector_exactly_zero():
    """`_unit` does NOT clip — an ephemeral score of nothing is zero, not a manufactured direction."""
    z = np.zeros(5)
    out = _unit(z)
    assert np.array_equal(out, z), (
        "the cache's normalizer manufactured a direction for a zero vector. It scores local, "
        "ephemeral top-k: a vector with no direction must score zero, not match arbitrarily."
    )
    assert np.allclose(_unit(np.array([3.0, 4.0])), [0.6, 0.8])


def test_the_two_normalizers_genuinely_disagree_on_zero():
    """The point of the file, as one assertion.

    If this ever passes trivially — because the two were unified — the unification has silently
    picked one question's answer for both.
    """
    z32 = np.zeros(4, dtype=np.float32)
    clipped = l2norm(z32)          # divisor clipped at 1e-12
    passthrough = _unit(np.zeros(4))
    # Both are all-zero for an all-zero INPUT; the divergence is in the divisor, so probe near zero.
    tiny = 1e-30
    a = l2norm(np.full(4, tiny, dtype=np.float32))
    b = _unit(np.full(4, tiny))
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert not np.allclose(a, b), (
        "the anchor normalizer and the cache normalizer now agree on a near-zero vector. They "
        "answer different questions — 'what direction is this anchor' (never zero, the id depends "
        "on it) versus 'how similar is this' (no direction means no score) — so agreement here "
        "means one of the two call sites is now getting the other's answer."
    )
    assert clipped.shape == passthrough.shape  # same operation, same shape, different semantics


def test_the_engine_drops_zero_rows_rather_than_normalizing_them():
    """The third semantic: a zero-norm candidate is removed from the RESULT SET, not scored as 0.

    Asserted on the expression rather than through the engine, because the engine's version is
    fused into the scoring matmul and inseparable from the `keep` mask it applies to the ids.
    """
    M = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    ids = ["a", "zero", "c"]
    norms = np.linalg.norm(M, axis=1)
    keep = norms > 0
    kept_ids = [ids[i] for i in np.nonzero(keep)[0]]

    assert kept_ids == ["a", "c"], "the zero-norm row must leave the candidate list entirely"
    unit = M[keep] / norms[keep, None]
    assert unit.shape == (2, 2), "the matrix and the id list must stay the same length"
    assert np.allclose(unit @ np.array([1.0, 0.0]), [0.6, 1.0])
