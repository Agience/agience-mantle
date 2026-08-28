# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# BEACON is the permissive half of the two-tier model, deliberately: mantle ships
# Apache so a store can be taken, built on and shipped by anyone, and beacon is the
# reduced instrument that makes such a store genuinely useful on its own. mantle is
# Apache-2.0 with a public remote and `packages.find` has no exclude, so no
# proprietary-restriction notice applies here. The downstream consumer's
# `beacon_engine.py` keeps its own notice — that is the Foresight white-label pilot,
# a different tree and a different arrangement.
# ---------------------------------------------------------------------------
"""The third rank criterion was built and measured, and does not ship. This is the measurement.

ARCHITECTURE-PLAN.md B3 needs an SVD rank for the semantic arm. Two reads are already
disqualified for it — `signal_rank` (Tracy-Widom edge) and `structure_rank` (permutation
parallel analysis) — because the structural rank of a term-document matrix is not the topic
count. B3 proposed a held-out predictive criterion: replace an in-sample property with a
held-out predictive one. A spectrum is in-sample in exactly that way — every direction it
reports is a direction of the matrix it was computed from.

So a self-supervised criterion was built: hold documents out of the PPMI/SVD fit, split each
held-out document's tokens into two independent halves, and keep the leading run of components
for which half 1 predicts half 2 better than break-even and better than chance pairing. No
labels, no queries, no trained model, no tuned constant — `far` is the module's one stated
false-alarm level and the null is the permutation null already in `structure_rank`.

It does not land on the separation peak, so it is not in `engine.py`. Measured against the
three corpora that carry a true topic count, by `_scratch/probes/run_heldout.py`. `sep@k` is
the probe's own separation function; the sweep peak is what a correct chooser would find:

    corpus                docs   true k   held-out k   sep@k    peak k   sep@peak
    corpus-A 3x20           60      3          7       0.231       5      0.243
    corpus-A 3x50          150      3         13       0.208       3      0.264
    corpus-A 3x100         300      3         16       0.180       3      0.287
    corpus-A 2 near x50    100      2          4       0.133       2      0.145
    OEWN 3 distant x50     150      3          1 ref.  0.001       3      0.223
    OEWN 3 distant x200    600      3          4       0.331       4      0.331
    OEWN 3 distant x500   1500      3          3       0.371       3      0.371
    OEWN 3 NEAR x50        150      3          1       0.000       5      0.100
    OEWN 8 topics x50      400      8          2       0.179       8      0.236
    OEWN 8 topics x200    1600      8          2       0.188       7      0.323
    repo md 3x10            30      3         15       0.076       7      0.223
    repo md 3x50           150      3         75       0.035       2      0.167

It overshoots on long documents and grows with corpus size (corpus-A 7 -> 13 -> 16, markdown
15 -> 75, all at a fixed true k of 3) and undershoots badly on short ones (2 against a true 8
on OEWN). Raising the fold count makes it worse, not better (markdown 3x50: 75 / 110 / 127 at
folds 2 / 4 / 8) — more training data resolves more genuinely-transferable non-topical
directions.

`test_beacon_structure_rank.py` measured that uneven document mass is what inflates both
noise-floor reads. Holding documents out does not escape it:
`test_uneven_document_mass_inflates_the_held_out_read_too` shows all three reads move together,
3/3/3 on an even allotment and 26/25/24 on a lognormal one, against a true 3. Uneven document
mass is real structure — it transfers, since both halves of a heavy document are heavy — so it
predicts held-out documents honestly. The defect is not in-sample-ness and not the null: topic
count is not the rank of anything in this matrix, and a criterion that measures a real property
of the matrix keeps finding that real property.

The failure modes these tests watch for, so they can fail:

  1. The criterion itself is broken rather than the finding. If it cannot recover a planted
     topic count when topic is the only structure present, the table above measures a bug.
     Test 1 pins that it can: 3 -> 3 and 8 -> 8.
  2. It finds structure in a true null. Test 2 shuffles the matrix — each column permuted down
     the rows — which preserves every term's marginal and destroys every cross-document
     relation. (Shuffling tokens within documents is not a null draw: it preserves document
     length and rare-word structure, and left top-3 spectral energy identical at 5.1% vs 5.1%.)
  3. The synthetic generator is treated as evidence. Test 5 pins that this criterion is stable
     with corpus size on the generator (3/3/3 at 60/150/300 documents) while the table above
     records 7/13/16 on real text at the same sizes — the generator plants topic as the only
     structure, so it cannot exhibit the failure or falsify the criterion.
  4. The number gets shipped anyway. Test 7 asserts `engine.py` exports no such chooser.

Re-validation is a re-run, by design. The three corpora above are labelled by accident of
provenance (BeIR source collection, OEWN `lexfile`, source repo). When a corpus labelled by
construction is available (the Agience Foundation corpus, ARCHITECTURE-PLAN.md I1), point
`_scratch/probes/run_heldout.py` at it: the criterion takes a count matrix and the validation
takes (matrix, labels), so re-validating against real labels is a corpus swap and nothing else.
If it lands on the peak there, this file is the thing to delete — after the run, not before it.
"""
from __future__ import annotations

import numpy as np
import pytest

from mantle.search.beacon import engine
from mantle.search.beacon.engine import (
    DEFAULT_FAR, _permutation_core, _permutation_draws, signal_rank,
)

# ── the criterion under test ────────────────────────────────────────────────────────────
# It lives here and not in `engine.py` because it did not earn a place there. It is kept
# runnable so the finding above is reproducible rather than merely asserted, and so the next
# reader can re-measure it in one command instead of rebuilding it from the prose.
# `_scratch/probes/heldout_rank.py` is the same code with the full derivation in its docstrings.


def _structure_rank_k(M, *, far: float = DEFAULT_FAR, draws: int | None = None,
                      seed: int = 0) -> int:
    """The permutation-null rank this file compares against — `engine.structure_rank`'s own
    wrapping of `_permutation_core`, reproduced here since the public wrapper was retired
    (zero production callers; `instrument.py` reaches `_permutation_core` directly). This
    file's measured comparisons are about the underlying null, not about the wrapper, so the
    retirement changes nothing this file is actually testing."""
    core = _permutation_core(M, far=far, draws=draws, seed=seed)
    if not core.readable:
        return 1
    return max(1, core.k_tested + core.offset)


def _ppmi_rows(C: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """PPMI of count rows against a given column distribution.

    PPMI is exactly a per-row transform once the column marginal is fixed —
    ``log(p_ij / (p_i. * p_.j)) == log((C_ij / n_i) / p_.j)`` — so a held-out document can be
    transformed with the training corpus's `pj` and no information crosses the fold boundary.
    That identity is what makes this read leak-free."""
    tot = C.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.log((C / np.where(tot == 0.0, 1.0, tot)) / pj[None, :])
    return np.nan_to_num(m, nan=0.0, neginf=0.0, posinf=0.0).clip(min=0.0)


def _held_out_rank(counts: np.ndarray, *, far: float = DEFAULT_FAR, folds: int = 2,
                   seed: int = 0) -> int | None:
    """Leading run of components that predict a held-out document better than chance.

    Returns the rank, or ``None`` when no read could be taken. ``gain_j > 0`` is break-even, not
    a threshold: adding component j changes held-out squared error by exactly ``-gain_j``, so
    the sign is the whole question. ``p_j <= far`` is the permutation null already in this
    module, at the draw count it already derives.

    `folds` is 2 — the minimum at which a document can be held out of a fit at all, and the only
    value at which train and test are exchangeable, so no size ratio is chosen."""
    C = np.asarray(counts, dtype=np.float64)
    n = int(C.shape[0])
    B = _permutation_draws(far)
    if n < 2 * folds:
        return None
    rng = np.random.default_rng(seed)
    assign = np.empty(n, dtype=np.int64)
    assign[rng.permutation(n)] = np.arange(n) % folds

    gains, nulls = [], []
    for f in range(folds):
        te, tr = np.nonzero(assign == f)[0], np.nonzero(assign != f)[0]
        if tr.size < 2 or te.size < 1:
            continue
        Ctr = C[tr]
        live = np.nonzero(Ctr.sum(axis=0) > 0.0)[0]
        if live.size < 2:
            continue
        Ctr = Ctr[:, live]
        pj = Ctr.sum(axis=0) / Ctr.sum()
        _, _, Vt = np.linalg.svd(_ppmi_rows(Ctr, pj), full_matrices=False)
        Cte = C[te][:, live]
        # a binomial half-split of the TOKENS: two independent samples of the same document
        C1 = rng.binomial(Cte.astype(np.int64), 0.5).astype(np.float64)
        C2 = Cte - C1
        ok = np.nonzero((C1.sum(axis=1) > 0.0) & (C2.sum(axis=1) > 0.0))[0]
        if ok.size < 2:
            continue
        A = _ppmi_rows(C1[ok], pj) @ Vt.T
        Bm = _ppmi_rows(C2[ok], pj) @ Vt.T
        penalty = 0.5 * ((A ** 2).sum(axis=0) + (Bm ** 2).sum(axis=0))
        gains.append(2.0 * (A * Bm).sum(axis=0) - penalty)
        # the null: half 2 from a DIFFERENT held-out document. `penalty` sums over documents and
        # is invariant to the pairing, so only the coupling term — the quantity in question — is
        # resampled.
        nulls.append(np.stack([2.0 * (A * Bm[rng.permutation(ok.size)]).sum(axis=0) - penalty
                               for _ in range(B)]))
    if not gains:
        return None
    r = min(g.shape[0] for g in gains)
    gain = np.sum([g[:r] for g in gains], axis=0)
    null = np.sum([nl[:, :r] for nl in nulls], axis=0)
    pvalue = (1.0 + (null >= gain[None, :]).sum(axis=0)) / (1.0 + B)
    failed = np.nonzero(~((gain > 0.0) & (pvalue <= far)))[0]
    return max(1, int(failed[0]) if failed.size else int(r))


# ── the corpus generator ────────────────────────────────────────────────────────────────
# The same shape as `test_beacon_structure_rank.py` and the probes, so the three criteria are
# compared on one corpus and any difference is the criterion.


def _zipf(n: int, s: float = 1.1) -> np.ndarray:
    p = 1.0 / np.arange(1, n + 1, dtype=float) ** s
    return p / p.sum()


def _corpus(counts, overlap, rng, *, vocab_per_topic=60, shared_n=40, doc_len=80):
    """`counts[t]` documents of topic t; `overlap` of each drawn from the shared pool.

    `overlap = 1.0` plants no topic structure — every document comes from the one pool."""
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


def _counts(counts, overlap, seed, **kw):
    docs, labels = _corpus(counts, overlap, np.random.default_rng(seed), **kw)
    return _term_doc(docs), labels


def _column_shuffle(M: np.ndarray, seed: int) -> np.ndarray:
    """Permute each column down the rows: every marginal kept, every cross-row relation gone."""
    rng = np.random.default_rng(seed)
    return np.take_along_axis(M, np.argsort(rng.random(M.shape), axis=0), axis=0)


# ── 1. the criterion is not broken ──────────────────────────────────────────────────────

@pytest.mark.parametrize("k_topics", [3, 8])
def test_it_recovers_the_planted_topic_count_when_topic_is_the_only_structure(k_topics):
    """So the table in the module docstring cannot be dismissed as a bug: on a corpus where
    topic membership is the only structure present, the criterion returns the planted count
    exactly. It is the corpus that defeats it, not the arithmetic."""
    C, _ = _counts([10] * k_topics, 0.5, seed=7)
    assert _held_out_rank(C) == k_topics


# ── 2. the true null ────────────────────────────────────────────────────────────────────

def test_a_matrix_shuffled_corpus_reads_one():
    """The negative control, on a true null. Each column permuted down the rows preserves every
    term's document frequency and total frequency exactly and destroys every cross-document
    relation. Nothing predicts a held-out document, so the read must collapse to 1."""
    C, _ = _counts([50, 50, 50], 0.5, seed=11)
    assert _held_out_rank(_column_shuffle(C, seed=12)) == 1


def test_the_collapse_is_not_an_artifact_of_one_shuffle():
    """One collapse proves nothing. Five independent shuffles, five independent read seeds."""
    C, _ = _counts([50, 50, 50], 0.5, seed=11)
    got = [_held_out_rank(_column_shuffle(C, seed=s), seed=s) for s in range(5)]
    assert got == [1] * 5, f"shuffled reads varied: {got}"


def test_a_corpus_with_no_topic_structure_reads_one():
    """`overlap = 1.0`: every document drawn from the same pool. There is nothing to find and
    the read must refuse rather than return a coarse guess."""
    C, _ = _counts([50, 50, 50], 1.0, seed=11)
    assert _held_out_rank(C) == 1


def test_a_corpus_too_small_to_hold_anything_out_is_unknown_not_one():
    """Absence of a read is not a read of 1. Below the size at which a document can be held
    out of a fit at all, the criterion returns None — unknown — and a caller that treats that as
    "one direction" has invented the direction. This is the distinction
    [[absence-is-not-an-affirmative-claim]] exists for."""
    C, _ = _counts([1, 1, 1], 0.5, seed=11)
    assert _held_out_rank(C) is None


# ── 3. the finding: it fails for the same reason the other two do ───────────────────────

def _with_private(counts, rng, *, per_doc, spread):
    """The same corpus, plus terms nobody else uses — evenly, or lognormally spread."""
    docs, labels = _corpus(counts, 0.5, rng)
    for i, d in enumerate(docs):
        n = per_doc if spread == 0 else int(np.clip(rng.lognormal(np.log(per_doc), spread), 0, 900))
        d.extend([f"p_{i}_{j}" for j in range(n)])
    return docs, labels


def test_uneven_document_mass_inflates_the_held_out_read_too():
    """`test_beacon_structure_rank.py::test_uneven_document_mass_is_what_inflates_the_rank` shows
    that neither noise-floor read is fooled by private vocabulary as such — forty private terms
    per document, evenly, leaves both at exactly 3 — but that the same vocabulary spread
    lognormally takes both to ~25. Real prose is lognormal.

    Holding documents out does not escape it. All three reads move together: 3/3/3 even,
    26/25/24 uneven, against a planted 3. Uneven document mass transfers — both halves of a
    heavy document are heavy — so it predicts held-out documents honestly, and a criterion that
    rewards honest prediction rewards it.

    The defect is not in-sample-ness and not the null: topic count is not the rank of anything
    in this matrix, so a criterion that measures a real property of the matrix keeps finding
    that real property. Fails if the held-out read ever stops tracking the other two here, which
    would mean it had found a different quantity and the table in the module docstring is owed a
    re-run."""
    even = _term_doc(_with_private([20, 20, 20], np.random.default_rng(11),
                                   per_doc=40, spread=0.0)[0])
    uneven = _term_doc(_with_private([20, 20, 20], np.random.default_rng(11),
                                     per_doc=40, spread=1.0)[0])
    assert signal_rank(_ppmi(even)).k == 3
    assert _structure_rank_k(_ppmi(even)) == 3
    assert _held_out_rank(even) == 3, "an even private allotment is not structure, for any read"

    assert signal_rank(_ppmi(uneven)).k > 10
    assert _structure_rank_k(_ppmi(uneven)) > 10
    assert _held_out_rank(uneven) > 10, (
        "holding documents out is supposed to be the escape from uneven document mass; "
        "measured, it is not")


# ── 4. what the synthetic generator can and cannot show ─────────────────────────────────

def test_the_read_is_stable_with_corpus_size_ON_THE_GENERATOR_ONLY():
    """A green run here is not evidence about real data, and that is the point of the test.

    Three topics at 20, 50 and 100 documents each, at `doc_len = 200` — matched to corpus-A's
    ~225 mean stems, so the comparison is against the real corpus rather than against a shorter
    one.
    On the generator the read is 3/3/3, perfectly stable. On real text at those exact sizes and
    that length it is 7/13/16, and on repo markdown 15/75, against the same true 3 — the table
    in this module's docstring.

    The generator plants topic as the only structure, so it cannot exhibit the failure or
    falsify the criterion — which is how agience-pharos/genesis/ARCHITECTURE-TARGET.md §7 came to report exact rank
    recovery for a rule that does not have it. Pinned here so a passing synthetic suite is not
    read as validation, including the tests above it, which are bug-checks and controls, not
    evidence that the criterion works on real data.

    Verified across four independent corpus seeds."""
    got = [_held_out_rank(_counts([n, n, n], 0.5, seed=11, doc_len=200)[0])
           for n in (20, 50, 100)]
    assert got == [3, 3, 3], f"the generator moved with corpus size: {got}"


def test_short_documents_are_refused_rather_than_guessed_at():
    """The other half of the length story, and the reason the test above pins a length.

    The same corpora at `doc_len = 80` — the synthetic length §7 was validated on — collapse to
    1 as the corpus grows: 3, 1, 1 at 20, 50 and 100 documents per topic. A half-split of an
    80-token document is 40 tokens over a 60-word topic vocabulary, and nothing survives it.

    That matches the real corpora exactly: OEWN glosses (~11 stems) refuse at 150 documents
    where corpus-A abstracts (~225 stems) do not, and corpus-A truncated to 80 stems refuses
    too. So
    refusal here is the criterion working — it is the one behaviour that held up throughout —
    and it is why the size test above states its document length instead of inheriting one."""
    got = [_held_out_rank(_counts([n, n, n], 0.5, seed=11, doc_len=80)[0])
           for n in (20, 50, 100)]
    assert got == [3, 1, 1], f"the short-document reads moved: {got}"


# ── 5. it did not ship ──────────────────────────────────────────────────────────────────

def test_no_held_out_rank_chooser_is_exported_from_the_engine():
    """The criterion was measured against three labelled corpora and did not land on the
    separation peak, so it is not in `engine.py` and not in the package's disclosed surface.

    If this test ever fails, one of two things happened: a validated criterion shipped — in
    which case delete this test and this file, after re-running
    `_scratch/probes/run_heldout.py` and recording the new table — or an unvalidated one did,
    which is what B2 and B3 exist to prevent. Two numbers already shipped in this codebase that
    looked principled and tracked the wrong thing; a third would be worse than an unbuilt arm."""
    exported = set(engine.__all__)
    assert not {n for n in exported if "held" in n.lower() or "predictive" in n.lower()}
    assert not hasattr(engine, "held_out_rank")
    # …and the read that does ship still says, in the module's own docstring, not to borrow it
    # for this. That sentence is the only guidance a caller gets, so it may not quietly go away.
    assert "SIGNAL_RANK IS NOT A TOPIC-COUNT ESTIMATOR ON REAL TEXT" in engine.__doc__
