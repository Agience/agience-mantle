# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------
"""Parity suite for `mantle/search/beacon/cut.py`, the RAG-facing screen math held in mantle,
where it is Apache-2.0 and public, rather than in the downstream consumer's trade-secret
`adaptive_beacon.py` (the Foresight white-label pilot). The giveaway is only honest if what is given away answers identically to
what the pilot runs, so every moved function is held bit-equal to an oracle kept verbatim in this
file.

The sections, in order:

  1. The parity sweep. Sweeps all thirteen moved functions against the oracle over a shared corpus
     of inputs, bit-equal — never `approx`, because comparing two numbers with a tolerance is how a
     divergence hides.
  2. No tuned constants. Asserts each derivation moves with its inputs — a derivation that returns
     the same number regardless is the old constant wearing a function — and pins the two numbers
     that are still typed as typed, with the measurement.
  3. Absence stays absence. An empty or single-candidate input produces a computed null, never a
     plausible number in its place.
  4. The measured degeneracy. Pins `heads > n_features`, a real defect reproduced deliberately
     rather than repaired, because repairing it would move a live pilot's answers. Two OTHER
     defects in the same file were repaired on 2026-08-22 — see the oracle header — and section 4a
     pins what that repair moved.
  5. The dependency floor. beacon acquires no edge to `beam` or `prism` through the
     new module.

What these checks cannot show: the oracle is a verbatim copy of the source implementation, so
agreement proves the move was faithful. It does not prove either is correct on a real corpus — a
copy and its original can be wrong together. Ground-truth checks are section 2's job, and they are
weaker than the parity sweep.

The oracle is a copy, not the live file: mantle's tests cannot reach into the Foresight tree, a
different checkout on a different arrangement. A comparison against the live `adaptive_beacon.py`
runs separately, through an in-memory shim, read-only.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from mantle.search.beacon import cut
from mantle.search.beacon.engine import _tw1_core, occupancy_fraction, signal_rank

# ═════════════════════════════════════════════════════════════════════════════════════════════
# 0 · The oracle — the source bodies, verbatim, never calling the module they check
# ═════════════════════════════════════════════════════════════════════════════════════════════
# These are not refactored and do not call `cut` — an oracle that calls the thing it checks proves
# only that a function equals itself. They are `backend/semantic/adaptive_beacon.py`, with exactly
# two mechanical substitutions, both renames that happened inside mantle and neither of which
# touches arithmetic:
#
#     _engine.resolved_rank(M)  ->  int(signal_rank(M))        (same body, same value)
#     _engine.fill_fraction(M)  ->  occupancy_fraction(M)      (same body, same value)
#
# If `cut.py` changes deliberately, these bodies change in the same commit and every moved number
# is explained. That is the arrangement, and it is the only one that catches a silent drift.

_ORACLE_NORM_FLOOR = 1e-9
_ORACLE_RATIO_FLOOR = 1e-12


def _o_resolved_rank(M) -> int:
    return int(signal_rank(M))


# ── `gap_split` and `top_break` ──────────────────────────────────────────────────────────────
#
# These two oracle bodies are maintained here rather than copied verbatim from the source, which is
# the arrangement this file states above: "if `cut.py` changes deliberately, these bodies change in
# the same commit and every moved number is
# explained."
#
# What was wrong. Both defects had one shape — the lock returned a cut where it had found nothing
# to cut at:
#
#   1. The median floor hid the break it was looking for. The break that ends a top cluster sits
#      BETWEEN the cluster and what follows, which is exactly the boundary the floor excludes. On
#      `[9,8,8,1,1,1,1,1]` the region is `[9,8,8]`, offering only the ratios 1.125 and 1.000, so it
#      cut after the 9 and kept one item of an obvious three; the real break, 8 -> 1 at a ratio of
#      8, was on the other side. Repaired by carrying one element below the floor into the region,
#      used to find the break and never itself kept.
#   2. A tied group was cut to one arbitrary member: all ratios are 1.0 and `argmax` lands on index
#      0. `[9,9,9,5,5,5,1,1]` kept 1 of 3; `[1]*9` kept 1 of 9; and the tie test needed a tolerance
#      derived from the dtype, because identical rows give powers differing in the last bits and a
#      ratio of `1 + 4e-15` is still greater than 1.
#
# What moved, and it is a live pilot's behaviour on these shapes:
#
#       spectrum                    before   after
#       [9, 9, 9, 5, 5, 5, 1, 1]         1       3
#       [9, 8, 8, 1, 1, 1, 1, 1]         1       3
#       [5, 5, 5, 5, 1, 1, 1, 1]         1       4
#       [7, 7, 1, 1]                     1       2
#       [1] * 9                          1       9
#       [10, 9, 1, 1e-3, 1e-9]           1       2
#
# On retrieval-shaped input it reaches about one spectrum in twenty (19 of 400 measured), and on
# every one of those it keeps MORE, never fewer — median 6 -> 8. That is the defect being undone:
# the old lock was cutting clusters short, and the recovered items sit between the first element
# and the real break. Whether they help is a question about a corpus and is not settled here; it is
# a recall-up change to a precision-oriented lock, and the labelled benches are where that gets
# adjudicated. The parity sweep below still passes bit-equal over its whole corpus, which is why
# the defects survived this long: the tests agreed because the inputs never reached them.
#
# `test_the_repair_is_what_changed_and_nothing_else` pins the before-and-after so the delta is
# checkable rather than asserted.


def _o_tie_tol(s):
    return 2.0 * max(1, s.size) * float(np.finfo(s.dtype).eps)


def _o_gap_split_at(s, tol):
    n = s.size
    if n <= 1:
        return np.ones(n, dtype=bool), 1.0
    order = np.argsort(s)[::-1]
    sd = s[order]
    ratios = sd[:-1] / np.clip(sd[1:], _ORACLE_RATIO_FLOOR, None)
    c = int(np.argmax(ratios))
    keep = np.zeros(n, dtype=bool)
    if ratios[c] <= 1.0 + tol:
        keep[:] = True
        return keep, 1.0
    keep[order[: c + 1]] = True
    return keep, float(ratios[c])


def _o_gap_split(scores):
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size <= 1:
        return np.ones(s.size, dtype=bool), 1.0
    return _o_gap_split_at(s, _o_tie_tol(s))


def _o_top_break(scores):
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    keep = np.zeros(n, dtype=bool)
    if n <= 1:
        keep[:] = True
        return keep, 1.0
    tol = _o_tie_tol(s)
    order = np.argsort(s)[::-1]
    med = float(np.median(s))
    n_above = int((s[order] > med * (1.0 + tol)).sum())
    if n_above == 0:
        keep[:] = True
        return keep, 1.0
    region = order[: min(n_above + 1, n)]
    keep_local, rel_gap = _o_gap_split_at(s[region], tol)
    n_keep = min(int(keep_local.sum()), n_above)
    keep[order[:n_keep]] = True
    return keep, rel_gap


# ── the pre-repair bodies, kept so the delta is measurable rather than described ──────────────

def _pre_repair_gap_split(scores):
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n <= 1:
        return np.ones(n, dtype=bool), 1.0
    order = np.argsort(s)[::-1]
    sd = s[order]
    ratios = sd[:-1] / np.clip(sd[1:], _ORACLE_RATIO_FLOOR, None)
    c = int(np.argmax(ratios))
    keep = np.zeros(n, dtype=bool)
    keep[order[: c + 1]] = True
    return keep, float(ratios[c])


def _pre_repair_top_break(scores):
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    keep = np.zeros(n, dtype=bool)
    if n <= 1:
        keep[:] = True
        return keep, 1.0
    order = np.argsort(s)[::-1]
    idx = order[s[order] > np.median(s)]
    if idx.size <= 1:
        keep[order[:1]] = True
        return keep, 1.0
    keep_local, rel_gap = _pre_repair_gap_split(s[idx])
    keep[idx[keep_local]] = True
    return keep, rel_gap


def _o_head_screen(item_embs, query_emb, heads):
    E = np.asarray(item_embs, dtype=np.float64)
    q = np.asarray(query_emb, dtype=np.float64)
    d = E.shape[1] // heads
    Eb = E[:, : heads * d].reshape(len(E), heads, d)
    qb = q[: heads * d].reshape(heads, d)
    Eb = Eb / (np.linalg.norm(Eb, axis=2, keepdims=True) + _ORACLE_NORM_FLOOR)
    qb = qb / (np.linalg.norm(qb, axis=1, keepdims=True) + _ORACLE_NORM_FLOOR)
    return np.clip(np.einsum("bhd,hd->bh", Eb, qb), 0.0, None)


def _o_signal_power(W):
    W = np.asarray(W, dtype=np.float64)
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    k = max(1, min(_o_resolved_rank(W), len(S)))
    sig = np.zeros(len(S), dtype=bool)
    sig[:k] = True
    return ((U[:, sig] * S[sig]) ** 2).sum(1)


def _o_derive_heads(item_embs, lo: int = 2, hi: int = 64) -> int:
    E = np.asarray(item_embs, dtype=np.float64)
    if E.ndim != 2 or min(E.shape) < 2:
        return lo
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + _ORACLE_NORM_FLOOR)
    heads = round(float(occupancy_fraction(En)) * E.shape[0])
    return int(max(lo, min(hi, heads)))


def _o_select(item_embs, query_emb, heads=None):
    E = np.asarray(item_embs, dtype=np.float64)
    if len(E) <= 1:
        return np.arange(len(E))
    H = _o_derive_heads(E) if heads is None else heads
    power = _o_signal_power(_o_head_screen(E, query_emb, H))
    keep, _ = _o_top_break(power)
    return np.where(keep)[0]


# ── bit equality, because two tolerances compared with a tolerance is the defect one level up ──

def _bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _identical(a, b) -> bool:
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
                and all(_identical(x, y) for x, y in zip(a, b)))
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
            return False
        if a.dtype.kind == "f":
            return all((np.isnan(x) and np.isnan(y)) or _bits(x) == _bits(y)
                       for x, y in zip(a.ravel(), b.ravel()))
        return bool(np.array_equal(a, b))
    if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        return int(a) == int(b)
    return bool((np.isnan(a) and np.isnan(b)) or _bits(a) == _bits(b))


# ── the shared corpus of inputs ────────────────────────────────────────────────────────────────
# The corners are the point: a sweep over well-conditioned random matrices proves only the easy
# half. These deliberately include the frames where a `+ 1e-9` guard, a median floor or a zero
# denominator decides the answer: all-zero, dead channel, duplicate row, 1e-8 and 1e9 magnitudes,
# and widths narrower than the head count.

def _frames():
    rng = np.random.default_rng(20260805)
    for d in (2, 3, 8, 16, 64, 384, 768):
        for n in (1, 2, 3, 5, 8, 13, 32, 80):
            yield rng.normal(size=(n, d))                                  # isotropic
            V = rng.normal(size=(min(4, max(1, n)), d))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            w = rng.dirichlet(np.ones(V.shape[0]) * 0.4, size=n)
            yield w @ V * 2.0 + rng.normal(size=(n, d)) * 0.7 / np.sqrt(d)  # structured
            yield rng.normal(size=(n, d)) * 1e-8                           # tiny
            yield rng.normal(size=(n, d)) * 1e9                            # huge
            yield np.zeros((n, d))                                         # all-zero
            dead = rng.normal(size=(n, d))
            dead[:, 0] = 0.0                                               # a dead channel
            yield dead
            if n >= 2:
                dup = rng.normal(size=(n, d))
                dup[1] = dup[0]                                            # an exact duplicate row
                yield dup


def _spectra():
    rng = np.random.default_rng(4242)
    out = [[], [1.0], [0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [5.0, 4.0, 1.0, 0.9],
           [1.0, 1.0, 1.0, 1.0], [1e-30, 1e-30, 1.0], [1e12, 1.0, 1e-12],
           [3.0, 3.0, 3.0, 1e-18], [0.0, 0.0, 0.0, 1.0], list(range(10))]
    for n in (2, 3, 5, 9, 16, 40):
        for _ in range(4):
            out.append(np.abs(rng.normal(size=n)).tolist())
            out.append((np.abs(rng.normal(size=n)) ** 4).tolist())
            out.append(np.zeros(n).tolist())
    return out


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · The parity sweep — bit-equal, every disagreement reported rather than tolerated
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_every_moved_function_is_bit_identical_to_the_body_it_came_from() -> None:
    """Fails if a body was "tidied" on the way in and now answers differently in some corner
    nobody exercises directly — so a public Apache beacon and the pilot that shipped the algorithm
    disagree about which items to keep, while both look right.

    Bit equality, never `approx`: these functions produce tolerances and cuts, so comparing two of
    them with a tolerance is the same defect one level up. Every disagreement is collected and
    reported, not counted — a count cannot be argued with."""
    bad: list[tuple] = []
    n = 0
    rng = np.random.default_rng(1)

    def check(name, mine, oracle, *args):
        nonlocal n
        n += 1
        try:
            got, err_a = mine(*args), None
        except BaseException as exc:                                     # noqa: BLE001
            got, err_a = None, type(exc).__name__
        try:
            want, err_b = oracle(*args), None
        except BaseException as exc:                                     # noqa: BLE001
            want, err_b = None, type(exc).__name__
        if err_a or err_b:
            if err_a != err_b:
                bad.append((name, "raised", err_a, err_b))
            return
        if not _identical(got, want):
            bad.append((name, getattr(args[0], "shape", args[0]), got, want))

    for E in _frames():
        d = E.shape[1]
        q = rng.normal(size=d)
        check("derive_heads", cut.derive_heads, _o_derive_heads, E)
        check("select", cut.select, _o_select, E, q)
        for heads in (1, 2, 7, 64):
            check("head_screen", cut.head_screen, _o_head_screen, E, q, heads)
            check("signal_power", cut.signal_power, _o_signal_power,
                  cut.head_screen(E, q, heads))

    for s in _spectra():
        check("gap_split", cut.gap_split, _o_gap_split, s)
        check("top_break", cut.top_break, _o_top_break, s)

    # Measured at 4020 after the novelty/drift retirement dropped five of the eight per-frame
    # checks (anomaly, anomaly_rank, most_anomalous, novelty_score, subspace_coherence, and the
    # in_subspace_fraction pair) — the floor below is set with margin under that count, not at it.
    assert n >= 3500, f"the sweep collapsed to {n} comparisons and would prove almost nothing"
    assert not bad, (
        f"{len(bad)} of {n} comparisons disagree with the body the function came from. ⛔ Do NOT "
        f"'fix' either side until the DOMAIN argument is settled. First five:\n"
        + "\n".join(repr(b) for b in bad[:5]))


def test_the_oracle_is_reachable_and_is_not_the_module_under_test() -> None:
    """The precondition, promoted to a test. If the oracle silently delegated to `cut`, the sweep
    above would compare a function with itself and pass forever. So: perturb the module's typed
    floor and watch the sweep's subject move while the oracle's does not."""
    E = np.zeros((4, 8))                          # every norm is 0, so the floor IS the denominator
    q = np.ones(8)
    assert _identical(cut.head_screen(E, q, 2), _o_head_screen(E, q, 2))
    original = cut._UNIT_NORM_FLOOR
    try:
        cut._UNIT_NORM_FLOOR = 1e-3
        moved = cut.head_screen(np.full((4, 8), 1e-4), q, 2)
        held = _o_head_screen(np.full((4, 8), 1e-4), q, 2)
        assert not _identical(moved, held), (
            "the oracle tracked a change made only to `cut`, so it is not independent of it and "
            "the parity sweep proves nothing")
    finally:
        cut._UNIT_NORM_FLOOR = original


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · No tuned constants — and the two that are still typed are pinned as typed
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_the_cut_is_parameter_free_and_the_break_is_where_the_data_puts_it() -> None:
    """Fails if `gap_split` acquires a keep-fraction, a MAD multiple or a significance level, and
    the "adaptive cut" becomes a top-k with extra steps.

    Asserted against moved data rather than a fixed expectation: put the break in three different
    places and the cut must follow it each time. A rule with a hidden constant would follow at most
    one of them."""
    for split_at in (1, 2, 5):
        scores = [100.0] * split_at + [1.0] * (8 - split_at)
        keep, gap = cut.gap_split(scores)
        assert int(keep.sum()) == split_at, f"the cut did not follow the break at {split_at}"
        assert gap == pytest.approx(100.0)
    keep, gap = cut.gap_split([1.0])
    assert keep.tolist() == [True] and gap == 1.0, "one item has no consecutive pair to break at"


def test_top_break_uses_the_median_as_a_derived_floor_not_a_threshold() -> None:
    """Fails if the median step is read as a cut and someone replaces it with a percentile "for
    tuning". It is a robustness step — it keeps the largest relative gap from being found between
    two pieces of tail noise — and it never decides how many are kept.

    Pinned by the case that distinguishes the two: a spectrum whose biggest multiplicative drop is
    in the tail, between two pieces of noise. `gap_split` alone follows it and keeps almost
    everything; `top_break` must find the real break at the head instead."""
    scores = [100.0, 90.0, 5.0, 4.0, 3.0, 1e-12]
    assert int(cut.gap_split(scores)[0].sum()) == 5, (
        "the fixture is wrong: the raw biggest relative gap must be in the tail for this to test "
        "anything")
    keep, _ = cut.top_break(scores)
    assert keep.tolist() == [True, True, False, False, False, False], (
        "top_break followed a gap between two pieces of tail noise instead of the real break")
    kept_values = np.asarray(scores)[keep]
    assert (kept_values > np.median(scores)).all(), (
        "top_break kept an item below the data-derived floor it is built on")


def test_derive_heads_tracks_the_spectrum_and_is_not_a_constant_in_disguise() -> None:
    """A derivation that returns the same number regardless is the old constant wearing a
    function. `H = round(occupancy · N)` is the active-mode count of the singular spectrum, so a
    concentrated set must earn fewer heads than a spread one at the same shape."""
    rng = np.random.default_rng(5)
    d, n = 384, 80
    spread = rng.normal(size=(n, d))
    V = rng.normal(size=(2, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    concentrated = rng.dirichlet(np.ones(2), size=n) @ V * 5.0 + rng.normal(size=(n, d)) * 0.01
    assert cut.derive_heads(concentrated) < cut.derive_heads(spread), (
        "a set whose energy sits in two directions earned as many heads as an isotropic one, so "
        "the count is not reading the spectrum")
    assert cut.derive_heads(np.ones((1, 8))) == 2, "a degenerate shape must fall to `lo`"


def test_the_head_count_bounds_are_stated_not_derived_and_the_ceiling_BINDS() -> None:
    """The two numbers in `derive_heads` that are not derived, pinned with their measurement so
    neither is mistaken for a compute bound: `lo=2, hi=64` might look like a search bound that
    does not change the answer inside its range, but one of them does.

      `lo`      is derived (`_smallest_readable_head_count`) and recovers exactly the `2` the
                source shipped with. It binds only at `n <= 5` (77 of 280 draws, d ∈ 64…1536),
                never at `n >= 8`. See the dedicated test below for the pinned oracle.
      `hi = 64` binds and decides. On near-isotropic sets the raw count is ~0.93·N, so it bites
                from n ≈ 70 upward (12/12 draws at n = 80). On structured retrieval sets at the
                pilot's default horizon of 80 the raw count is 17–19 and it never binds. The
                derivation that would replace it is `H <= n_features` (below which the screen is
                all-zero); that does not reproduce, so it is reported here rather than landed."""
    rng = np.random.default_rng(20260805)
    # `lo` binds where the raw count falls below it — a set whose energy is one direction, so
    # `occupancy · N` rounds to 1 — and it clamps upward.
    one_direction = np.repeat(rng.normal(size=(1, 64)), 5, axis=0) + 1e-9 * rng.normal(size=(5, 64))
    En_one = one_direction / (np.linalg.norm(one_direction, axis=1, keepdims=True)
                              + cut._UNIT_NORM_FLOOR)
    assert round(float(occupancy_fraction(En_one)) * 5) < 2, (
        "the fixture stopped being one-directional; this test can only pin the floor on a set "
        "whose raw count falls below it")
    assert cut.derive_heads(one_direction) == 2
    assert cut.derive_heads(one_direction, lo=5) == 5, "the floor is not the floor; `lo` was ignored"
    # `hi` binds on an isotropic set at n = 80, and it is the ceiling that decides.
    iso = rng.normal(size=(80, 512))
    En = iso / (np.linalg.norm(iso, axis=1, keepdims=True) + cut._UNIT_NORM_FLOOR)
    raw = round(float(occupancy_fraction(En)) * 80)
    assert raw > 64, (
        f"the fixture stopped being isotropic (raw={raw}); this test can only pin the ceiling on a "
        "set whose raw count exceeds it")
    assert cut.derive_heads(iso) == 64, "the ceiling did not bind where it was measured to bind"
    assert cut.derive_heads(iso, hi=128) == raw, (
        "raising the ceiling did not release the derived count, so 64 is not merely clamping — "
        "something else is deciding")


#: The value the pilot shipped with, kept as a pinned oracle. `derive_heads`'s floor is recovered
#: from the engine's readability gate rather than a typed literal, so a change to the gate moves it
#: with no edit — but a derivation is only an improvement if it lands on the same number. The
#: literal stays here, as data, so any change to the gate is checked against the value a live pilot
#: ran on.
_SHIPPED_LO = 2


def test_the_derived_head_floor_recovers_the_value_the_pilot_shipped_with() -> None:
    """A derivation is only an improvement if it reproduces the number it replaces. This is that
    proof, in both directions.

    Fails (mode 1) if `_MIN_HEADS` drifts off 2 and every small-pool cut a live pilot runs
    silently changes. Pinned against `_SHIPPED_LO`, a literal held as data.

    Fails (mode 2, the subtler one) if `_MIN_HEADS` is quietly turned back into a typed `2` that
    happens to agree, and the recovery stops tracking the gate it claims to come from. Pinned by
    asserting the boundary property: the floor is the first head count the engine will read, and
    one below it must not be readable. That cannot pass for a literal unless the literal happens
    to sit exactly on the gate, which is the only case where it would not matter."""
    assert cut._MIN_HEADS == _SHIPPED_LO, (
        f"the derived head floor moved to {cut._MIN_HEADS}; the implementation this came from "
        f"shipped {_SHIPPED_LO}, and a live pilot's small-pool cuts move with it")
    assert cut._MIN_HEADS == cut._smallest_readable_head_count()

    h = cut._MIN_HEADS
    readable = np.arange(1.0, float((h + 1) * h) + 1.0).reshape(h + 1, h)
    below = np.arange(1.0, float(h * (h - 1)) + 1.0).reshape(h, h - 1)
    assert _tw1_core(readable).readable, "the floor is not a width the engine will read"
    assert not _tw1_core(below).readable, (
        "one head below the floor is still readable, so the floor is not ON the gate — it was "
        "typed, or the gate moved without it")

    # And it is genuinely the default, not decoration.
    import inspect
    assert inspect.signature(cut.derive_heads).parameters["lo"].default == cut._MIN_HEADS


def test_the_two_typed_floors_are_named_and_are_load_bearing() -> None:
    """The seams, asserted so they are not mistaken for derivations.

    Fails if `1e-9` and `1e-12` are inlined again and stop being visible as typed numbers, so the
    next reader takes them for part of the geometry. They are named module constants, documented
    as stated values, and they bite — a zero-length vector reaches the first and a zero tail
    reaches the second, which is why replacing them with a dtype-derived bound would move live
    answers and is not done here."""
    assert cut._UNIT_NORM_FLOOR == 1e-9 and cut._RATIO_FLOOR == 1e-12
    # The norm floor is what makes a zero-norm row or query finite rather than NaN.
    assert np.isfinite(cut.head_screen(np.zeros((4, 8)), np.zeros(8), 2)).all()
    # The ratio floor decides the reported size of the gap on a zero tail, not where the cut falls.
    keep, gap = cut.gap_split([1.0, 0.0, 0.0])
    assert keep.tolist() == [True, False, False]
    assert gap == pytest.approx(1.0 / cut._RATIO_FLOOR), (
        "the reported gap on a zero tail is no longer 1/_RATIO_FLOOR, so the floor moved or the "
        "denominator is being clipped somewhere else")


def test_the_resolved_mode_count_is_the_engines_and_not_a_second_rule() -> None:
    """Fails if `cut` grows its own rank rule and the package starts naming two different ranks
    while both look right — the same defect `read_ordered` vs `signal_rank` is guarded for.

    `_resolved_rank` — what `signal_power` builds on — must be the engine's count on the matrix
    it was handed, capped by the number of singular values that exist."""
    rng = np.random.default_rng(9)
    for W in (rng.normal(size=(60, 16)), rng.normal(size=(8, 128)), np.zeros((5, 12))):
        S = np.linalg.svd(W, compute_uv=False)
        expected_k = max(1, min(int(signal_rank(W)), len(S)))
        assert cut._resolved_rank(W) == expected_k, (
            "cut is counting modes by a rule the engine does not share")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · Absences and edges — a computed null is a result, never a plausible number
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_a_single_candidate_is_not_cut() -> None:
    """Fails if `select` discards its only candidate on the strength of a read that never
    happened."""
    assert cut.select(np.zeros((0, 8)), np.ones(8)).tolist() == []
    assert cut.select(np.ones((1, 8)), np.ones(8)).tolist() == [0]


def test_there_is_no_consumer_retrieval_policy_creeping_in() -> None:
    """An absence asserted as a gate. `is_anomalous`, `assess_novelty` and friends carry a level
    (which this module derives, never types) or encode a retrieval policy, which does not live
    in mantle — the consumer's glue stays on its own side of the seam.

    Fails if somebody adds one and the level walks back in."""
    for banned in ("is_anomalous", "detect_anomaly", "is_novel", "has_drifted", "assess_novelty",
                   "assess_coherence", "beacon_select"):
        assert not hasattr(cut, banned), (
            f"`cut.{banned}` appeared. Either it carries a level (which this module derives, never "
            f"types) or it is the consumer's retrieval policy, which does not live in mantle.")


def test_the_public_surface_is_exactly_the_screen_math() -> None:
    """Fails if consumer glue creeps across the seam. `is_available`, `beacon_select`,
    `assess_novelty`, `assess_coherence` and `BeaconCutUnavailable` stay on the consumer's side:
    they supply embeddings, encode a retrieval policy (`max_keep`, `min_horizon`), shape an
    `info` dict for the consumer's telemetry, or
    hold a graceful-degradation contract that contradicts beacon's own idiom of erroring loudly.
    Anything added here is Apache-2.0 and public the day it lands."""
    assert set(cut.__all__) == {
        "gap_split", "top_break",
        "derive_heads", "head_screen", "signal_power", "select",
        # `screen_frame` is screen math and nothing else: it derives the head count and builds the
        # head screen, both already here, and returns the (T, F) array. It carries no retrieval
        # policy, no store access and no graceful-degradation contract — it raises like everything
        # else in this module.
        "screen_frame",
    }
    for name in cut.__all__:
        assert callable(getattr(cut, name)), f"`cut.{name}` is exported and is not callable"


def test_no_export_collides_with_the_packages_other_meanings() -> None:
    """`coherence` and `rank` are both spoken for elsewhere in this system — `coherence` is
    `SpectralRead.coherence`, the ordered-axis lag-1 z-score (always `None` for beacon); `rank`
    is a count of modes (`signal_rank`, `RankResult`). Neither is exported here, and this guards
    against either arriving under its bare name and being misread as the other meaning."""
    assert not hasattr(cut, "coherence"), (
        "`cut.coherence` collides with `SpectralRead.coherence`, which is the ORDERED-AXIS lag-1 "
        "z-score and is always None for beacon.")
    assert not hasattr(cut, "rank"), (
        "`cut.rank` collides with `signal_rank` / `RankResult`, both of which mean a COUNT OF "
        "MODES.")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · The measured degeneracy — reproduced exactly, and therefore stated exactly
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_more_heads_than_features_yields_an_all_zero_screen_and_keeps_one_item() -> None:
    """A defect pinned rather than repaired, because the pilot in production depends on this exact
    behavior.

    `head_screen`'s block width is `n_features // heads`. More heads than features makes it 0,
    every block is empty, and the screen is all zeros — after which `signal_power` is all zeros and
    `top_break` (nothing strictly above a median of 0) keeps exactly one item, the last by
    `argsort`. So the adaptive cut silently collapses to keeping one arbitrary candidate rather
    than reporting that it has no cut to make.

    Reachable through `derive_heads` for any embedding narrower than the derived head count. Every
    model in the live pilot is 384–1536 wide, so it is not reachable there. Repairing it —
    reporting no reading, or bounding `H <= n_features` — would change what that pilot produces, so
    it is a separate deliberate act.

    This test passes on the defect so it is not forgotten, and fails the day somebody repairs it —
    at which point this test moves in the same commit and the repair is explained rather than
    absorbed."""
    rng = np.random.default_rng(77)
    E = rng.normal(size=(20, 16))
    q = rng.normal(size=16)
    W = cut.head_screen(E, q, 64)
    assert W.shape == (20, 64) and not W.any(), (
        "the all-zero screen is gone — if `head_screen` now refuses or bounds the head count, "
        "that is the repair, and this test is what has to move with it")
    kept = cut.select(E, q, heads=64)
    # Repaired 2026-08-22: this asserted `[19]` — the last candidate by `argsort`, kept because an
    # all-zero spectrum has nothing strictly above its median. The lock now reports that it found
    # no break and keeps everything, so the degeneracy is still here (the screen is still all
    # zeros) without a confident cut being produced out of nothing. Bounding `H <= n_features` in
    # `head_screen` is a separate repair this test awaits.
    assert kept.tolist() == list(range(20)), (
        "the degenerate path no longer keeps everything; whatever changed, it changed what a live "
        "pilot would produce")
    # The control: with a head count the width can carry, the screen is real and the cut reads.
    good = cut.head_screen(E, q, 4)
    assert good.any() and good.shape == (20, 4)


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4a · What the 2026-08-22 repair moved, and what it did not
# ═════════════════════════════════════════════════════════════════════════════════════════════

_REPAIR_DELTA = {
    (9.0, 9.0, 9.0, 5.0, 5.0, 5.0, 1.0, 1.0): (1, 3),
    (9.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0): (1, 3),
    (5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0): (1, 4),
    (7.0, 7.0, 1.0, 1.0): (1, 2),
    (1.0,) * 9: (1, 9),
    (10.0, 9.0, 1.0, 1e-3, 1e-9): (1, 2),
}


def test_the_repair_is_what_changed_and_nothing_else() -> None:
    """The before-and-after, pinned on the shapes the repair targets.

    The pre-repair bodies are kept in this file so this is a measurement rather than a claim. On
    every one of these the old lock returned a cut it reported no evidence for."""
    for spectrum, (before, after) in _REPAIR_DELTA.items():
        a = np.array(spectrum, dtype=np.float64)
        assert int(_pre_repair_top_break(a)[0].sum()) == before, (spectrum, "before")
        assert int(cut.top_break(a)[0].sum()) == after, (spectrum, "after")


def test_the_repair_reaches_about_one_in_twenty_real_spectra_and_only_ever_keeps_MORE() -> None:
    """How far the repair reaches, measured rather than assumed — and it is not zero.

    It reaches roughly 5% of the synthetic pools below — a figure later shown to be an artefact of
    the generator, which plants near-identical rows and so manufactures the defect's own trigger.
    On real corpora it reads 0.0% (144 cached bge-m3 spectra) and 0.2% (832 lattice-path spectra).
    The assertion here is about this generator and is kept because it pins what the repair does; it
    is not a deployment estimate.

    The direction is one-way and is the defect being undone: on every differing draw the repaired
    lock keeps more, never fewer (19/19 over 400 draws, median 6 -> 8). That is what "the median
    floor hid the break" means in practice — the old lock cuts clusters short, and the items it now
    recovers are the ones between the first element and the real break.

    Whether the recovered items help is a question about a corpus and cannot be settled here.
    It is a recall-up change to a precision-oriented lock; the labelled benches are where that gets
    adjudicated, and this test only fixes what the change is."""
    rng = np.random.default_rng(4242)
    differing, more, fewer = 0, 0, 0
    for _ in range(400):
        n = int(rng.integers(8, 90))
        d = int(rng.choice([64, 128, 384, 768]))
        E = rng.normal(size=(n, d))
        E[: max(1, n // 4)] += 2.0 * rng.normal(size=(1, d))
        q = rng.normal(size=d)
        p = cut.signal_power(cut.head_screen(E, q, 8))
        after, before = cut.top_break(p)[0], _pre_repair_top_break(p)[0]
        if np.array_equal(after, before):
            continue
        differing += 1
        more += int(after.sum() > before.sum())
        fewer += int(after.sum() < before.sum())
    assert 0 < differing <= 40, f"{differing}/400 — the reach of the repair moved"
    assert fewer == 0, f"{fewer} draws kept FEWER after the repair; that is not this repair"
    assert more == differing


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 5 · The dependency floor — the new module does not widen it
# ═════════════════════════════════════════════════════════════════════════════════════════════

_PROBE = '''
import sys, json
BLOCKED = ("beam", "prism")
for name in BLOCKED:
    assert name not in sys.modules, name + " was imported before the blocker went up"
MARK = "BLOCKED by the negative control"


class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ImportError(MARK + ": " + fullname)
        return None


sys.meta_path.insert(0, Blocker())
v = {}
v["blocker_fires"] = {}
for name in BLOCKED:
    try:
        __import__(name)
        v["blocker_fires"][name] = False
    except ImportError as exc:
        v["blocker_fires"][name] = MARK in str(exc)
    except BaseException as exc:
        v["blocker_fires"][name] = "raised " + type(exc).__name__
try:
    import numpy as np
    from mantle.search.beacon import cut
    rng = np.random.default_rng(3)
    E = rng.normal(size=(40, 32))
    q = rng.normal(size=32)
    v["kept"] = int(len(cut.select(E, q)))
    v["heads"] = int(cut.derive_heads(E))
    v["works"] = True
except BaseException as exc:
    v["works"] = False
    v["error"] = type(exc).__name__ + ": " + str(exc)
v["leaked"] = sorted(n for n in sys.modules if n.split(".")[0] in BLOCKED)
print(json.dumps(v))
'''


def test_the_cut_imports_and_works_with_beam_and_prism_all_blocked() -> None:
    """Fails if the silhouette acquires an edge to `beam` or `prism` — directly or
    transitively — and mantle stops being shippable on its own. Keeping that
    edge closed is why the reduced instrument exists.

    Stricter than `instrument.py`'s probe: `instrument.py` is allowed one prism module
    (`prism.rounding`, the rounding law). `cut.py` asks for nothing outside numpy and
    `beacon.engine`, so all of prism is blocked here and the module must still take a real read.
    Control first — the blocker is proven to bite before its silence is read as evidence."""
    prelude = f"import sys\nsys.path[:0] = {json.dumps(sys.path)}\n"
    proc = subprocess.run([sys.executable, "-c", prelude + _PROBE],
                          capture_output=True, text=True, env=dict(os.environ), timeout=300)
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    v = json.loads(proc.stdout.strip().splitlines()[-1])
    assert all(v["blocker_fires"].values()), (
        f"the blocker did not bite on every edge: {v['blocker_fires']} — every conclusion below "
        "would be vacuous")
    assert v["works"], f"the cut broke without beam/prism: {v.get('error')}"
    assert v["heads"] >= 2 and 1 <= v["kept"] <= 40
    assert not v["leaked"], f"a blocked package reached sys.modules anyway: {v['leaked']}"


def test_the_cut_is_not_re_exported_from_the_package_promise() -> None:
    """The package `__all__` is a promise — every name in it is something a third party may build
    on and cannot be changed without breaking them. `cut` is reached by its own path, exactly as
    `instrument` is.

    Fails if thirteen names are added to `mantle.search.beacon.__all__` as a convenience and the
    published surface doubles by accident. Widening it is a separate deliberate act."""
    from mantle.search import beacon
    assert set(beacon.__all__).isdisjoint(set(cut.__all__))


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 6 · The shared corpus — the contract with the sibling implementation
# ═════════════════════════════════════════════════════════════════════════════════════════════
# `tests/lock_corpus.json` holds the same 6,400 cases as the sibling implementation's corpus. Both
# read it, so a change to the lock on either side fails on BOTH rather than only where somebody
# happened to write a test. That is the difference between a conformance test and a contract: the
# sibling's suite already measured this side, and nothing measured that one.
#
# It is a copy and not an import because the two trees do not depend on each other and must not
# start to: beacon is Apache and numpy-only so a store can ship standalone.
#
# It is not a byte-copy: this copy is LF (100,966 bytes), the sibling's is CRLF (107,351) — 6,400
# lines differing by line ending alone, so the two have never been byte-identical and no gate
# checks that. What has to match is the parsed cases, which is what `_lock_corpus()` reads and
# what the tests below compare.


def _lock_corpus():
    import json
    import pathlib
    return json.loads(
        (pathlib.Path(__file__).parent / "lock_corpus.json").read_text(encoding="utf8"))["cases"]


def test_the_shared_lock_corpus_reads_the_way_both_sides_recorded_it() -> None:
    """76 cases: every shape the 2026-08-22 defects lived in, plus 60 screen-shaped spectra."""
    for case in _lock_corpus():
        s = np.array(case["scores"], dtype=np.float64)
        for fn, name in ((cut.gap_split, "gap_split"), (cut.top_break, "top_break")):
            keep, gap = fn(s)
            assert [int(i) for i in np.where(keep)[0]] == case[name]["keep"], (case["name"], name)
            assert abs(gap - case[name]["rel_gap"]) <= 1e-12 * max(1.0, abs(case[name]["rel_gap"])), (
                case["name"], name, gap, case[name]["rel_gap"])


def test_the_corpus_covers_the_shapes_the_repair_moved() -> None:
    """A corpus of easy cases certifies nothing. These are the ones that moved on 2026-08-22."""
    names = {c["name"] for c in _lock_corpus()}
    assert {"plateau-3", "plateau-4", "flat", "cluster-with-inner-wobble",
            "tail-artifact", "near-tie", "all-zero", "empty"} <= names
