# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# Beacon is the permissive half of the two-tier model: mantle ships Apache so a
# store can be taken, built on and shipped by anyone, and beacon is the reduced
# instrument that makes such a store genuinely useful on its own. The downstream
# consumer's `beacon_engine.py` keeps a separate proprietary notice — that is the
# Foresight white-label pilot, a different tree and a different arrangement.
# ---------------------------------------------------------------------------
"""The rank of correlated rows is measured against a shuffle, not against a Gaussian.

`signal_rank` reads the Johnstone / Tracy-Widom edge, which is the edge of an i.i.d.-Gaussian
ensemble. `beam.optics.read_ordered` states the domain rule this engine is the reduced form of:
that edge is correct for an i.i.d. bulk and optimistic for correlated rows. For retrieved
evidence (correlated by construction) it directs callers to `correlated_null()`, the
distribution-free permutation null. `structure_rank` is that null, numpy-only.

What these tests pin, stated first so each can fail on its own:

  1. The read collapses on a shuffled matrix. A column-permuted matrix is a draw from the null,
     so a rule that still finds structure in it is reading something other than structure. This
     is the property `signal_rank` was reported to fail, and it is the gate that matters most.
  2. The read does not grow with the matrix, only with the structure. Corpora built at 60, 150,
     and 300 documents with the same three topics read the same rank — asserted directly, because
     "it is reading matrix dimensions" is the accusation.
  3. The read does not invent structure where there is none. A corpus drawn from one vocabulary
     pool has one direction and reads 1, not a coarse guess.
  4. The derived k does not walk past the separation peak. It is compared against a full k-sweep
     of the same separation function the probe used; landing far from the peak is the measured
     cost.
  5. The mean direction is counted rather than tested against a null that preserves it. A column
     permutation preserves every column mean exactly, so testing the rank-1 mean component
     against this null would give p ~ 1 by construction — a check that cannot fail. Uncentred, the
     rule returns 1 on every input, and test 5 pins the centring that avoids it.
  6. The draw count is derived, not a typed-in number. It follows from `far` alone, as the
     minimum at which a permutation p-value can reach that level, and test 7 re-derives it
     independently.

What this read is not: on real term-document matrices — corpus-A (BeIR) and this workspace's
markdown — neither `structure_rank` nor `signal_rank` returns the topic count, and the null is
not what separates them: on corpus-A 3x100 they read 81 and 23 against a true 3, and both fall
away
on a shuffled matrix (1 and 5). What inflates them is measured in
`test_uneven_document_mass_is_what_inflates_the_rank`: not private vocabulary (an even 40 private
terms per document takes df=1 to 92% of the vocabulary and leaves both reads at exactly 3), but
uneven document mass — the same private vocabulary spread lognormally takes both to ~25. That is
real structure, a shuffle really does destroy it, and it grows with the corpus because more
documents are more individually resolvable. Topic count is not the structural rank of a
term-document matrix, and no noise-floor rule of any shape will make it one.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantle.search import beacon
from mantle.search.beacon.engine import (
    ENGINE_ID, ENGINE_ID_PERM, DEFAULT_FAR, RankResult, _permutation_core, _permutation_draws,
    signal_rank,
)


def structure_rank(M, *, far: float = DEFAULT_FAR, draws: int | None = None,
                   seed: int = 0) -> "RankResult":
    """Reproduces `engine.structure_rank`'s wrapping of `_permutation_core`. The public wrapper
    was retired (zero production callers; `instrument.py`'s correlated-row path reaches
    `_permutation_core` directly) — what this file pins is the permutation-null behavior
    itself, which is unchanged and still live, so the wrapper is kept local to this file
    rather than deleted along with the tests."""
    core = _permutation_core(M, far=far, draws=draws, seed=seed)
    if not core.readable:
        return RankResult(k=1, live_channels=core.live_channels, degraded=core.degraded,
                          instrument=ENGINE_ID_PERM)
    return RankResult(k=max(1, core.k_tested + core.offset), live_channels=core.live_channels,
                      degraded=core.degraded, instrument=ENGINE_ID_PERM)


# ── the corpus generator ────────────────────────────────────────────────────────────────
# Same shape as the one behind ARCHITECTURE-TARGET.md §7 and SEMANTIC-PROBE.md: Zipf-sampled
# topic vocabularies over a Zipf-sampled shared pool, so the only structure planted is topic
# membership. Reproduced here rather than imported because the probe scripts live in `_scratch`
# and a test may not depend on a scratch directory.


def _zipf(n: int, s: float = 1.1) -> np.ndarray:
    p = 1.0 / np.arange(1, n + 1, dtype=float) ** s
    return p / p.sum()


def _corpus(counts, overlap, rng, *, vocab_per_topic=60, shared_n=40, doc_len=80):
    """`counts[t]` documents of topic t; `overlap` of each document drawn from the shared pool.

    `overlap = 1.0` plants no topic structure at all — every document is drawn from the one pool.
    """
    shared = np.array([f"s{i}" for i in range(shared_n)])
    topics = [np.array([f"t{t}_{i}" for i in range(vocab_per_topic)]) for t in range(len(counts))]
    p_s, p_t = _zipf(shared_n), _zipf(vocab_per_topic)
    docs, labels = [], []
    for t, n in enumerate(counts):
        for _ in range(n):
            k_s = int(doc_len * overlap)
            docs.append(list(rng.choice(shared, k_s, p=p_s))
                        + list(rng.choice(topics[t], doc_len - k_s, p=p_t)))
            labels.append(t)
    return docs, np.array(labels)


def _term_doc(docs) -> np.ndarray:
    vocab = sorted({t for d in docs for t in d})
    idx = {t: i for i, t in enumerate(vocab)}
    M = np.zeros((len(docs), len(vocab)))
    for r, d in enumerate(docs):
        for t in d:
            M[r, idx[t]] += 1.0
    return M


def _ppmi(C: np.ndarray) -> np.ndarray:
    total = C.sum()
    if total == 0:
        return C
    p = C / total
    pi, pj = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.log(p / (pi * pj))
    return np.nan_to_num(m, nan=0.0, neginf=0.0, posinf=0.0).clip(min=0.0)


def _matrix(counts, overlap, seed, **kw) -> tuple[np.ndarray, np.ndarray]:
    docs, labels = _corpus(counts, overlap, np.random.default_rng(seed), **kw)
    return _ppmi(_term_doc(docs)), labels


def _embed(M: np.ndarray, k: int) -> np.ndarray:
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    k = max(1, min(k, len(S)))
    V = U[:, :k] * S[:k]
    n = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.where(n == 0, 1.0, n)


def _separation(V: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-topic cosine minus mean across-topic cosine — the probe's own measure."""
    Sim = V @ V.T
    same = np.equal.outer(labels, labels)
    np.fill_diagonal(same, False)
    return float(Sim[same].mean() - Sim[~np.equal.outer(labels, labels)].mean())


def _column_shuffle(M: np.ndarray, seed: int) -> np.ndarray:
    """Permute each column down the rows: every marginal kept, every cross-row relation gone."""
    rng = np.random.default_rng(seed)
    return np.take_along_axis(M, np.argsort(rng.random(M.shape), axis=0), axis=0)


# ── 1. the gate that matters most ───────────────────────────────────────────────────────

@pytest.mark.parametrize("k_topics,per_topic", [(3, 20), (3, 50), (8, 20)])
def test_a_shuffled_matrix_reads_materially_lower(k_topics, per_topic):
    """The gate that matters most: a column-permuted matrix is a draw from the null itself, so no
    cross-row structure survives it. The read collapses to 1 — the mean direction alone, with
    nothing further to report."""
    M, _ = _matrix([per_topic] * k_topics, 0.5, seed=11)
    real = structure_rank(M).k
    shuffled = structure_rank(_column_shuffle(M, seed=12)).k
    assert shuffled == 1, f"a shuffled matrix read {shuffled} directions of structure"
    assert real >= k_topics, f"the real matrix read {real}, below its {k_topics} planted topics"
    assert real > shuffled


def test_the_collapse_is_not_an_artifact_of_the_seed():
    """Five independent shuffles, five independent read seeds. One collapse proves nothing."""
    M, _ = _matrix([50, 50, 50], 0.5, seed=11)
    got = [structure_rank(_column_shuffle(M, seed=s), seed=s).k for s in range(5)]
    assert got == [1] * 5, f"shuffled reads varied: {got}"


# ── 2. it must not read dimensions ──────────────────────────────────────────────────────

def test_rank_does_not_grow_with_corpus_size():
    """The "it is reading matrix dimensions" accusation, asserted directly. Three topics at 20,
    50 and 100 documents each — the matrix triples, the structure does not.

    It cannot: a column permutation preserves the Frobenius norm exactly, so the matrix and its
    surrogate carry the same total energy, and the read is a comparison of shapes."""
    got = [structure_rank(_matrix([n, n, n], 0.5, seed=11)[0]).k for n in (20, 50, 100)]
    assert got == [3, 3, 3], f"rank moved with corpus size: 20/50/100 docs per topic -> {got}"


# ── 3. ground truth, and the no-structure case ──────────────────────────────────────────

@pytest.mark.parametrize("k_topics", [2, 3, 8])
def test_the_planted_topic_count_is_recovered(k_topics):
    """k topics span k-1 dimensions once the mean is removed; the mean direction restores the
    one centring took. So a k-topic corpus reads k, and the arithmetic is derived end to end."""
    M, _ = _matrix([30] * k_topics, 0.5, seed=11)
    assert structure_rank(M).k == k_topics


def test_a_corpus_with_no_topic_structure_reads_one():
    """`overlap = 1.0`: every document drawn from the same pool, so there is nothing to find.
    The read returns 1 rather than a coarse guess."""
    M, _ = _matrix([50, 50, 50], 1.0, seed=11)
    assert structure_rank(M).k == 1


def test_pure_noise_reads_one():
    """The other no-structure case: an i.i.d. matrix with nothing planted in it."""
    M = np.random.default_rng(4).normal(size=(120, 40))
    assert structure_rank(M).k == 1


# ── 4. the derived k against the sweep ──────────────────────────────────────────────────

@pytest.mark.parametrize("k_topics,per_topic", [(3, 20), (3, 50), (8, 20)])
def test_the_derived_k_lands_on_the_separation_peak(k_topics, per_topic):
    """The probe's diagnosis was that the chooser walks past the peak — by 4x at 150 documents
    and 10x at 600. Sweep k with the probe's own separation function and check where this rule
    lands. Fails if the derived k costs more than 2% of the peak separation."""
    M, labels = _matrix([per_topic] * k_topics, 0.5, seed=11)
    sweep = {k: _separation(_embed(M, k), labels)
             for k in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)}
    peak_k = max(sweep, key=lambda k: sweep[k])
    derived = structure_rank(M).k
    assert derived == peak_k, f"derived k={derived}, sweep peaks at k={peak_k}"
    assert _separation(_embed(M, derived), labels) >= 0.98 * sweep[peak_k]


# ── 5. the mean direction, and why it is counted rather than tested ─────────────────────

def test_a_column_permutation_preserves_every_column_marginal_exactly():
    """The mechanical fact the centring follows from, asserted on the multiset itself rather than
    on the mean: a permutation moves values, it never alters them, so each column sorted is
    bit-identical. If this ever stops holding the surrogate is not marginal-preserving and the
    whole read is void. (The mean follows to float-summation order, hence the 1e-12.)"""
    M, _ = _matrix([20, 20, 20], 0.5, seed=11)
    S = _column_shuffle(M, seed=1)
    assert np.array_equal(np.sort(S, axis=0), np.sort(M, axis=0))
    assert np.allclose(S.mean(axis=0), M.mean(axis=0), rtol=1e-12, atol=1e-15)


def test_an_already_centred_matrix_is_not_given_a_mean_direction():
    """The `+1` is not a constant bolted on: it is the mean component of this matrix, counted
    only when the matrix has one. Centre the matrix by hand and the read drops by exactly one."""
    M, _ = _matrix([30, 30, 30], 0.5, seed=11)
    assert structure_rank(M).k == 3
    assert structure_rank(M - M.mean(axis=0, keepdims=True)).k == 2


def test_testing_the_mean_direction_would_be_a_check_that_cannot_fail():
    """State the failure mode, then show it. On a non-negative matrix the leading singular
    direction is the column mean, which the surrogate reproduces exactly — so its top singular
    value matches the matrix's to within rounding, and a test of it can only ever pass."""
    M, _ = _matrix([30, 30, 30], 0.5, seed=11)
    s_top = np.linalg.svd(M, compute_uv=False)[0]
    surrogate_top = np.linalg.svd(_column_shuffle(M, seed=7), compute_uv=False)[0]
    assert abs(surrogate_top - s_top) / s_top < 0.05
    # and the centred spectrum, which is what the read decomposes, is not reproduced
    C = M - M.mean(axis=0, keepdims=True)
    Cs = _column_shuffle(M, seed=7)
    Cs = Cs - Cs.mean(axis=0, keepdims=True)
    assert np.linalg.svd(C, compute_uv=False)[0] > 1.5 * np.linalg.svd(Cs, compute_uv=False)[0]


# ── 6. the documented limit: structure is not topics ────────────────────────────────────

def _with_private(counts, rng, *, per_doc, spread):
    """The same corpus, plus terms nobody else uses — evenly, or lognormally spread."""
    docs, labels = _corpus(counts, 0.5, rng)
    for i, d in enumerate(docs):
        n = per_doc if spread == 0 else int(np.clip(rng.lognormal(np.log(per_doc), spread), 0, 900))
        d.extend([f"p_{i}_{j}" for j in range(n)])
    return docs, labels


def test_uneven_document_mass_is_what_inflates_the_rank():
    """Pins why the inflated reads on real corpora are not the topic count.

    The obvious explanation for real corpora reading 30-90 where the truth is 3 is private
    vocabulary: 44-58% of corpus-A's and of this workspace's markdown vocabulary has document
    frequency 1, against ~0% in the generator §7 was validated on. That is not the explanation:
    forty private terms per document takes df=1 to 92% of the vocabulary and both reads stay at
    exactly 3 — because an even private allotment is as even as no allotment at all.

    Spread the same private vocabulary lognormally, so documents differ in mass the way real ones
    do, and both reads climb to ~25. Uneven document mass is structure, a shuffle does destroy it,
    and it grows with the corpus because more documents are more individually resolvable — which
    is why no noise-floor rule of any shape recovers topic rank on real text, and why swapping
    the null does not change that. Both reads are affected equally: this was never about the null.
    """
    even = _ppmi(_term_doc(_with_private([20, 20, 20], np.random.default_rng(11),
                                         per_doc=40, spread=0.0)[0]))
    uneven = _ppmi(_term_doc(_with_private([20, 20, 20], np.random.default_rng(11),
                                           per_doc=40, spread=1.0)[0]))
    assert ((_term_doc(_with_private([20, 20, 20], np.random.default_rng(11),
                                     per_doc=40, spread=0.0)[0]) > 0).sum(axis=0) == 1).mean() > 0.9
    assert structure_rank(even).k == 3, "an even private allotment is not structure"
    assert signal_rank(even).k == 3
    assert structure_rank(uneven).k > 10, "uneven document mass IS structure and must read as it"
    assert signal_rank(uneven).k > 10, "…and the other null reads it the same way"
    # …and it is structure by the shuffle's own account, which is what makes it real.
    assert structure_rank(_column_shuffle(uneven, seed=3)).k == 1


# ── 7. the draw count is derived ────────────────────────────────────────────────────────

@pytest.mark.parametrize("far", [0.10, 0.05, 0.025, 0.01])
def test_the_draw_count_is_the_minimum_at_which_far_is_attainable(far):
    """Re-derived here independently of the implementation: the smallest permutation p-value is
    1/(1 + draws), so a level `far` is unreachable below `1/far - 1` draws and reachable at it.
    Fails if the default is ever nudged to a round number."""
    B = _permutation_draws(far)
    assert 1.0 / (1.0 + B) <= far, "the default cannot reach its own level"
    assert 1.0 / (1.0 + (B - 1)) > far, f"draws={B} is larger than the minimum"


def test_the_default_far_gives_nineteen_draws():
    """The one arithmetic consequence worth writing down, so a change to either is visible."""
    assert DEFAULT_FAR == 0.05
    assert _permutation_draws(DEFAULT_FAR) == 19


def test_more_draws_does_not_move_the_answer_on_a_clear_read():
    """Draws buy resolution, not a different conclusion. A 10x budget on a well-planted corpus
    must return the same k, or the default is under-powered rather than merely coarse."""
    M, _ = _matrix([30, 30, 30], 0.5, seed=11)
    assert structure_rank(M, draws=190).k == structure_rank(M).k == 3


# ── 8. provenance, determinism, degenerate frames ───────────────────────────────────────

def test_the_read_names_its_own_instrument():
    """A permutation number and a Tracy-Widom number answer the same question against different
    assumptions. A caller holding a bare integer must still be able to tell them apart."""
    M, _ = _matrix([20, 20, 20], 0.5, seed=11)
    assert structure_rank(M).instrument == ENGINE_ID_PERM == "beacon.perm"
    assert signal_rank(M).instrument == ENGINE_ID == "beacon.tw1"
    assert ENGINE_ID != ENGINE_ID_PERM


def test_the_read_is_deterministic():
    M, _ = _matrix([20, 20, 20], 0.5, seed=11)
    assert structure_rank(M, seed=5).k == structure_rank(M, seed=5).k


def test_a_collapsed_axis_is_flagged_degraded():
    """The case the conformance gate exists for, on the new read: a frame whose spread is carried
    by one channel is not a clean one-mode matrix, and must not come back looking like one."""
    rng = np.random.default_rng(0)
    M = np.zeros((60, 8))
    M[:, 0] = rng.normal(size=60)
    read = structure_rank(M)
    assert read.live_channels <= 1
    assert read.degraded is True
    assert read.k == 1


def test_a_single_row_is_degenerate_not_an_answer():
    read = structure_rank(np.array([[1.0, 2.0, 3.0]]))
    assert read.k == 1


def test_it_refuses_what_it_cannot_read():
    with pytest.raises(beacon.BeaconEngineError):
        structure_rank(np.zeros((0, 3)))
    with pytest.raises(beacon.BeaconEngineError):
        structure_rank(np.zeros(5))
    with pytest.raises(ValueError):
        structure_rank(np.random.default_rng(0).normal(size=(30, 4)), far=0.0)


# ── 9. signal_rank is deliberately unchanged ────────────────────────────────────────────

def test_signal_rank_still_reads_the_tracy_widom_edge():
    """The two nulls are two instruments, not one instrument improved. `signal_rank` is correct
    at its own job — an i.i.d. bulk. The defect was one of domain, so nothing here changes what
    it means.

    This is the frame `test_beacon_conformance` pins; if the two ever disagree here, one of them
    moved."""
    M = np.random.default_rng(0).normal(size=(60, 8))
    read = signal_rank(M)
    assert read.k == 1 and read.instrument == ENGINE_ID
    B = np.random.default_rng(1).normal(size=(3, 12))
    C = np.random.default_rng(2).normal(size=(90, 3)) * np.array([6.0, 4.0, 3.0])
    planted = C @ B + 0.35 * np.random.default_rng(3).normal(size=(90, 12))
    assert signal_rank(planted).k == 3


def test_the_mean_direction_is_counted_on_a_mean_zero_frame_too():
    """The price of not claiming an absence. On a Gaussian frame the column mean is itself noise,
    but the surrogate reproduces it exactly, so the read has no evidence either way and counts it
    rather than asserting it away. On a wide planted frame that shows as k = planted+1.

    A caller reading `structure_rank` on mean-zero data should expect exactly one direction more
    than the rank it planted, and that one is untested."""
    B = np.random.default_rng(1).normal(size=(3, 120))
    C = np.random.default_rng(2).normal(size=(90, 3)) * np.array([6.0, 4.0, 3.0])
    wide = C @ B + 0.35 * np.random.default_rng(3).normal(size=(90, 120))
    assert signal_rank(wide).k == 3
    assert structure_rank(wide).k == 4
    assert structure_rank(_column_shuffle(wide, seed=3)).k == 1
