# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# Beacon is the permissive half of the two-tier model, deliberately: mantle ships
# Apache so a store can be taken, built on and shipped by anyone, and beacon is the
# reduced instrument that makes such a store genuinely useful on its own.
#
# Everything in this file is Apache-2.0 and public. The downstream consumer's
# `adaptive_beacon.py` keeps its own trade-secret notice for the Foresight
# white-label pilot, a different tree and a different arrangement. This module
# holds the screen math; the consumer's glue (the embeddings, the RAG pipeline,
# the retrieval policy) stays on its own side. `tests/test_beacon_cut.py` verifies
# the two agree function by function.
# ---------------------------------------------------------------------------

"""The cut — beacon's silhouette: which of a candidate pool belong together, and where the set
stops.

    beacon (this package, Apache)   the cut — subspace membership, the parameter-free
                                    relative-gap break.
    the aperture (AGPL)             projection — absorb the band that couples here, transmit the
                                    residual, place surfaces on a membrane and read their coupling.
    the aperture (AGPL)             prediction — lag, delay embedding, the fitted propagator.

The cut is held bit-equal to the implementation it reproduces, over a shared corpus of inputs
(`tests/test_beacon_cut.py`), against an oracle kept verbatim and never allowed to call this
module.

Public surface
--------------
    primitives   gap_split · top_break
    the cut      derive_heads · head_screen · signal_power · select

`coherent_basis`, `in_subspace_fraction`, `anomaly`, `anomaly_rank`, `most_anomalous`,
`novelty_score`, and `subspace_coherence` were retired from this surface: zero production callers
anywhere in mantle, and the naming-collision avoidance they required (`anomaly_rank` rather than
`rank`, which collides with `signal_rank`/`structure_rank`; `subspace_coherence` rather than
`coherence`, which collides with `SpectralRead.coherence`) no longer applies to anything exported
here. See SEARCH-ARCHITECTURE.md.

═══════════════════════════════════════════════════════════════════════════════════════════════
No tuned constants — and what is still typed is named as a seam rather than defended
═══════════════════════════════════════════════════════════════════════════════════════════════

A derivation can be as wrong as a constant if it models the wrong error: a tie-break candidate of
`len(vals) * eps` models accumulation error — "the sum runs over n terms" — when the error present
in a relative-gap comparison is cancellation. So every number below states which error or which
geometry it comes from, and the ones that are still typed say so in those words.

    quantity              what it is                    status
    ─────────────────────────────────────────────────────────────────────────────────────────
    the cut itself        largest relative gap          derived — parameter-free. No MAD
                          (`gap_split`)                 multiple, no significance level, no
                                                        keep-fraction. Nothing here to tune.
    the salience floor    the median (`top_break`)      derived from the data. It is also what
                                                        keeps the relative gap from being
                                                        dominated by tiny tail denominators.
    the head count        `round(occupancy · N)`        derived — the active-mode count of the
                          (`derive_heads`)              singular spectrum. Not picked.
    the signal modes      `signal_rank` (`engine`)      derived — the Tracy-Widom edge at the
                                                        engine's one stated tolerance,
                                                        `DEFAULT_FAR`.
    `lo = _MIN_HEADS`     floor on the head count       derived — the engine's own readability
                          (`derive_heads`)              gate, seen from the head axis. Lands on
                                                        2. See `_smallest_readable_head_count`.
    `hi = 64`             ceiling on the head count     stated, not derived, and it binds. See
                                                        `derive_heads` — measured, and the
                                                        derivation that would replace it does not
                                                        reproduce, so it is not applied.
    `_UNIT_NORM_FLOOR`    1e-9, added to every norm     stated, not derived. A typed guard
                          before dividing               against dividing by a zero-length vector.
    `_RATIO_FLOOR`        1e-12, clip on the gap        stated, not derived. A typed guard on
                          denominator                   the denominator of a ratio.

The two floors are seams, named rather than fixed. The honest replacement for both is a bound read
off the frame's own dtype — the arrangement `instrument._float_noise` already uses, which asks
`prism.rounding` for the forward error of the summation actually performed. It is not applied here
because it does not reproduce: `‖v‖ + 1e-9` and `‖v‖ + tiny` are different denominators, so every
downstream number would move, and these functions back a pilot in active use. A "correct" bound
that shifts a live answer is worse than a stated one that carries its own reasoning.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .engine import _tw1_core, occupancy_fraction, signal_rank

__all__ = [
    # ── primitives ──
    "gap_split", "top_break",
    # ── the adaptive cut ──
    "derive_heads", "head_screen", "signal_power", "select",
]
# Deliberately not re-exported from `mantle.search.beacon`. The package `__all__` is a promise —
# every name in it is something a third party may build on and therefore something that cannot
# change without breaking them — and it is pinned by
# `tests/test_beacon_conformance.py::test_exported_surface_is_only_the_beacon_cut`. This module is
# reached by its own path, exactly as `beacon.instrument` is. Widening the package promise is a
# separate deliberate act with its own argument.


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# The two typed floors — named, so they cannot be mistaken for derivations
# ═══════════════════════════════════════════════════════════════════════════════════════════════

#: Added to every norm before dividing by it. Stated, not derived — a guard against dividing by a
#: zero-length vector, carried verbatim from the implementation this module reproduces, because
#: replacing it with a dtype-derived bound moves every number downstream of it and these functions
#: back a live pilot. What it would be: the forward error of the summation `‖·‖` actually
#: performed, read off the frame's own dtype (`prism.rounding`, the law `instrument._float_noise`
#: already uses), whose error model is accumulation — valid here for the same reason it is valid
#: there, because every term summed is a square and no cancellation can occur.
_UNIT_NORM_FLOOR = 1e-9

#: The smallest denominator a consecutive-ratio may be taken against in `gap_split`. Stated, not
#: derived, and the same seam as `_UNIT_NORM_FLOOR` with a different consequence: it decides the
#: size of the reported gap when the tail is numerically zero (a zero tail reports `1/1e-12`), not
#: where the cut falls. The cut position is unaffected because a zero tail is below every other
#: value by construction.
_RATIO_FLOOR = 1e-12


def _smallest_readable_head_count() -> int:
    """The fewest heads at which `signal_power`'s own read can be taken, recovered from the gate
    rather than typed.

    The head count is handed to `head_screen`, whose `(n, heads)` output goes straight to
    `signal_power`, which decomposes it and asks the engine for its resolved-mode count. The
    engine gives no reading below `min(N, F_live) >= 2`, so a screen with one head is a screen
    whose own read cannot be taken, and the returned "signal power" is then a single column's
    energy dressed as a spectral read. Flooring the head count at the smallest readable width is
    therefore the engine's gate, seen from the head axis, exactly as `instrument.MIN_ROWS` is that
    gate seen from the row axis.

    This asks `_tw1_core(...).readable` — the gate itself — rather than restating
    `min(N, F_live) >= 2` a third time, so a change to the gate moves this floor with no edit here.
    The probe holds the row axis one wider than the head axis, so the heads are always the binding
    side, and uses distinct column values so that nothing but the shape can make it unreadable (an
    `eye` probe would be unreadable for zero MAD instead, and would answer a different question).

    No iteration bound is needed: `min(h + 1, h) = h` is strictly increasing, so this terminates by
    the algebra.

    It reproduces exactly: the implementation this module came from ships a literal `2`, and this
    returns 2. `tests/test_beacon_cut.py` keeps that literal as a pinned oracle, so if the gate
    ever moves, the change is caught against the number a live pilot shipped with.
    """
    heads = 0
    while True:
        heads += 1
        probe = np.arange(1.0, float((heads + 1) * heads) + 1.0).reshape(heads + 1, heads)
        if _tw1_core(probe).readable:
            return heads


#: The floor on `derive_heads`, recovered from the engine's readability gate; computed, and equal
#: to 2.
_MIN_HEADS = _smallest_readable_head_count()


def _read_instrument():
    """The `Read` instrument this module's rank comes from, resolved through the registry when
    prism is on the path and falling back to beacon's own engine when it is not.

    `cut.py` carries no import-time edge to `prism` — `test_beacon_cut.py`'s subprocess proves the
    module works with the whole package blocked, stricter than `instrument.py`'s one-module
    allowance — so the import is attempted here, lazily, inside a function body, and a failure is
    the ordinary "prism is not installed" case rather than an error: `_resolved_rank` below falls
    back to calling `engine.signal_rank` directly, which is the exact call this module made before
    this seam existed.

    When prism *is* on the path, beacon registers itself as the process default the first time it
    is asked — not at import time, and not unconditionally, so a host that registered something
    fuller first (a richer, entroptics-backed `Read`) keeps its slot; `set_default_read` is one
    slot, last write wins, and this only writes when the slot is still empty. From then on,
    resolution goes through `prism.instrument.resolve_read`, which checks the explicit
    `instrument=` argument first (there is none here), then the process default, exactly as
    `resolve` already does for the embodiment slot.

    Returns `None` when prism cannot be imported at all — the caller's cue to fall back.
    """
    try:
        from prism import instrument as _pi
    except ImportError:
        return None
    if _pi.get_default_read() is None:
        from . import instrument as _beacon_read
        _pi.set_default_read(instrument=_beacon_read)
    return _pi


def _resolved_rank(M) -> int:
    """The coherent dimension of a frame — modes standing above the noise floor of whichever `Read`
    instrument is registered (beacon's own by default; a fuller one if a host registered one
    first).

    Always >= 1: a caller here builds a basis out of the answer immediately and has no way to defer
    a resolved count of zero the way `Read.resolvable`'s `None` lets a caller that can wait for
    more evidence. That floor is this module's own policy, applied to whatever the instrument
    reports — `Read.resolvable` returning `None` (an unreadable frame) floors here at 1 rather than
    propagating the deferral, because every function below needs a subspace to project onto and an
    empty one is worse than a coarse one.

    `engine.signal_rank` is beacon's own answer to this same question and is what runs when prism
    cannot be imported (see `_read_instrument`) or when beacon is the registered default — the two
    paths agree exactly, measured over 500 random frames spanning readable and near-degenerate
    shapes, because beacon's `Read.resolvable` is `engine.signal_rank`'s own floor-at-one
    computation with the floor removed, not a second implementation.
    """
    instrument = _read_instrument()
    if instrument is None:
        return int(signal_rank(M))
    k = instrument.resolve_read(None, "resolvable", at="beacon.cut._resolved_rank")(M)
    return 1 if k is None else int(k)


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# Primitives — the set geometry everything else is assembled from
# ═══════════════════════════════════════════════════════════════════════════════════════════════

def gap_split(scores) -> Tuple[np.ndarray, float]:
    """The cut. The natural break in a salience spectrum: split after the largest relative gap.

    Sort the (non-negative) scores descending and cut at the biggest multiplicative drop between
    consecutive values. Returns `(keep_mask, rel_gap)` — the mask over the original item order, and
    the size of the break (top/next ratio at the cut, >= 1) as a bare statistic the caller may
    judge for itself.

    Genuinely parameter-free: no fixed top-k, no MAD multiple, no significance level, no
    keep-fraction cap. There is nothing here to tune, and no per-corpus threshold to maintain as
    the corpus moves under it.

    Relative and not absolute, deliberately. An absolute gap is in the units of whatever produced
    the scores, so it needs a scale from somewhere; a ratio needs none, which is what makes the
    same rule work on a cosine spectrum, a power spectrum and a count.

    `n <= 1` keeps everything and reports `1.0`. One item has no consecutive pair, so there is no
    gap to find — `1.0` is the no-break ratio, not a measured break.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n <= 1:
        return np.ones(n, dtype=bool), 1.0
    order = np.argsort(s)[::-1]
    sd = s[order]
    ratios = sd[:-1] / np.clip(sd[1:], _RATIO_FLOOR, None)   # consecutive top/next = relative gaps
    cut = int(np.argmax(ratios))                             # the single biggest multiplicative drop
    keep = np.zeros(n, dtype=bool)
    keep[order[: cut + 1]] = True
    return keep, float(ratios[cut])


def top_break(scores) -> Tuple[np.ndarray, float]:
    """Isolate the top cluster of a non-negative salience spectrum, parameter-free.

    Restrict to the items strictly above the median — a floor derived from the data rather than
    typed — then `gap_split` that top region. Returns `(keep_mask, rel_gap)` over the original
    items.

    The median step exists because, although `gap_split` alone is already parameter-free, the
    largest relative gap over the whole spectrum is routinely found in the tail, where the
    denominator is nearly zero and a ratio of two pieces of noise beats every real break. The
    median is the coarsest data-derived way to keep the read inside the region that carries signal.
    It is a robustness step, not a threshold: it never decides how many are kept.

    When the top region has at most one item, exactly one is kept and the reported gap is `1.0`.
    That is the no-break report, and it is reachable on an all-equal spectrum (nothing is strictly
    above the median) — including the all-zero spectrum, which is what a screen with nothing to
    read produces. See `head_screen` for the one way that arises in practice.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    keep = np.zeros(n, dtype=bool)
    if n <= 1:
        keep[:] = True
        return keep, 1.0
    order = np.argsort(s)[::-1]
    idx = order[s[order] > np.median(s)]                     # high-salience items, descending
    if idx.size <= 1:
        keep[order[:1]] = True
        return keep, 1.0
    keep_local, rel_gap = gap_split(s[idx])
    keep[idx[keep_local]] = True
    return keep, rel_gap


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# The adaptive cut — a multi-head query-relative screen, read at its resolved modes
# ═══════════════════════════════════════════════════════════════════════════════════════════════

def head_screen(item_embs, query_emb, heads: int) -> np.ndarray:
    """The `(n, heads)` screen: each item's per-head cosine to the query, block by block.

    The feature axis is cut into `heads` contiguous blocks and each block is compared on its own,
    so an item that matches the query strongly on part of the coordinate and not at all elsewhere
    is visible as such rather than averaged into a single scalar. Negative cosines are clipped to
    zero: a direction pointing away from the query is not negative evidence, it is no evidence, and
    letting it subtract would let two irrelevant blocks cancel into a match.

    `heads > n_features` produces an all-zero screen: the block width is `n_features // heads`, so
    more heads than features makes it 0, every block is empty, and the returned matrix is all
    zeros. Downstream, that reaches `top_break` on an all-zero spectrum, which keeps exactly one
    item — the last by `argsort` — so the adaptive cut collapses to keeping one arbitrary candidate
    rather than reporting that it has nothing to read.

    This is reachable through `derive_heads` whenever `n_features < lo … hi`, i.e. for any embedding
    narrower than the derived head count. Every embedding model in the live pilot is 384–1536 wide,
    so it is not reachable there today. It is stated here, and asserted in
    `tests/test_beacon_cut.py`, because the arithmetic is reproduced exactly from the implementation
    this module came from, and a defect reproduced exactly is still a defect: repairing it would
    change what a live pilot produces and is its own deliberate act.
    """
    E = np.asarray(item_embs, dtype=np.float64)
    q = np.asarray(query_emb, dtype=np.float64)
    d = E.shape[1] // heads
    Eb = E[:, : heads * d].reshape(len(E), heads, d)
    qb = q[: heads * d].reshape(heads, d)
    Eb = Eb / (np.linalg.norm(Eb, axis=2, keepdims=True) + _UNIT_NORM_FLOOR)
    qb = qb / (np.linalg.norm(qb, axis=1, keepdims=True) + _UNIT_NORM_FLOOR)
    return np.clip(np.einsum("bhd,hd->bh", Eb, qb), 0.0, None)


def signal_power(W) -> np.ndarray:
    """Each row's energy in the signal modes of `W` — the leading K, K being the resolved-mode
    count of `W`'s own spectrum.

    Rows aligned with the coherent query-relevance structure carry the power; the off-query tail
    does not. This is the step that makes the cut a screen read rather than a re-ranking: a plain
    row-sum of `head_screen` would score an item that matches weakly everywhere the same as one
    that matches strongly along the structure the candidate set actually has.

    K is the engine's resolved-mode count — not a fraction of the modes and not a caller's guess.
    """
    W = np.asarray(W, dtype=np.float64)
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    k = max(1, min(_resolved_rank(W), len(S)))
    sig = np.zeros(len(S), dtype=bool)
    sig[:k] = True
    return ((U[:, sig] * S[sig]) ** 2).sum(1)


def derive_heads(item_embs, lo: int = _MIN_HEADS, hi: int = 64) -> int:
    """The head count, derived from the candidate set's own effective rank: `H = round(occupancy·N)`.

    `engine.occupancy_fraction` is `2^{H_sv} / N` — the active-mode count of the singular spectrum
    as a fraction of the modes available — so `occupancy · N = 2^{H_sv}` is the effective number of
    modes the set actually occupies. The screen is then cut into that many blocks: as many heads as
    the data has independent things to say. Nothing is picked.

    `lo` and `hi` are not the same kind of number, and only one of them is still typed. Both
    produce exactly the values the implementation this reproduces shipped with, and both are
    measured here rather than defended:

    `lo = _MIN_HEADS` is a floor, and it is derived: `_smallest_readable_head_count()`, the fewest
    heads at which `signal_power`'s own read can be taken at all, recovered by asking the engine's
    readability gate rather than restating it. It lands on 2. It binds on small pools — over 280
    draws (d ∈ 64…1536, n ∈ 2…400) the raw count falls below it in 77 of them, every one at
    `n <= 5` and never at `n >= 8`. What it buys is the screen's own premise: at `H = 1` there is
    one block, `head_screen` collapses to a single global cosine column, and the engine gives no
    reading while `signal_power` still returns a number.

    `hi = 64` is a ceiling, and it binds, and where it binds it decides. On near-isotropic candidate
    sets (no coherent structure) the raw count is ~0.93·N, so the ceiling bites from `n ≈ 70`
    upward — 12/12 draws at n=80, 12/12 at n=100, for every width from 64 to 1536. On structured
    retrieval-shaped sets it does not bind at all: the raw count at the pilot's default horizon of
    80 is 17–19, nowhere near 64. So it binds exactly in the regime where the set has no structure
    for the screen to read — but it is still a typed number changing a derived one, and calling
    that a "search bound" would be wrong.

    The derivation that would replace `hi`: the only ceiling the arithmetic itself requires is
    `H <= n_features`, below which `head_screen`'s block width goes to zero and the screen is all
    zeros (see `head_screen`). That bound is derived — it is the geometry, not a preference — and
    it is looser than 64 for every embedding width in use, so it would move the head count on
    exactly the isotropic sets where 64 binds today. It does not reproduce the values the
    implementation this module came from shipped with, so it is reported here rather than applied.
    """
    E = np.asarray(item_embs, dtype=np.float64)
    if E.ndim != 2 or min(E.shape) < 2:
        return lo
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + _UNIT_NORM_FLOOR)
    heads = round(float(occupancy_fraction(En)) * E.shape[0])     # occupancy · N = 2^{H_sv}
    # A tighter cap of `min(hi, n_features)` produces the identical head count at every embedding
    # width the pilot uses (384/768/1024/1536, N = 8…400), because `hi` already exceeds
    # `n_features` down to width 64 (see `head_screen`'s degeneracy note). It is not applied here
    # because `test_every_moved_function_is_bit_identical_to_the_body_it_came_from` checks this
    # function against the consumer's un-capped body and disagrees at narrow widths — 176 of 7100
    # comparisons. That bit-equality is the provenance guarantee for a function moved rather than
    # rewritten, so the tighter cap belongs in the consumer's body first.
    return int(max(lo, min(hi, heads)))


def select(item_embs, query_emb, heads: Optional[int] = None) -> np.ndarray:
    """The adaptive cut: which of `item_embs` to keep for `query_emb`. Returns their indices.

    `heads` is derived from the data when `None`; pass an int only to override. The lock is
    `top_break` on the per-item signal power — the top cluster above the largest relative gap — so
    there is no fixed k, no MAD multiple and no threshold. The screen decides how many to keep,
    which is the entire difference between an adaptive cut and a top-k with extra steps.

    `item_embs` is expected to be a bounded candidate pool (a top-B cosine horizon), not a whole
    corpus: the read is over the structure of the set it is given, so what is in the set is part of
    the question. Handing it everything asks a different question and gets a different answer.

    `n <= 1` returns every index. One candidate has no spectrum, so there is nothing to cut, and
    keeping it is not a fabricated decision — the alternative would be discarding the only evidence
    there is on the strength of a read that never happened.
    """
    E = np.asarray(item_embs, dtype=np.float64)
    if len(E) <= 1:
        return np.arange(len(E))
    H = derive_heads(E) if heads is None else heads
    power = signal_power(head_screen(E, query_emb, H))
    keep, _ = top_break(power)
    return np.where(keep)[0]
