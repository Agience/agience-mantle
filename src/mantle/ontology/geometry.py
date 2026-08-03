"""Ontology geometry — the model-free synset -> coordinate map that lets entroptics' continuous read
measure MEANING (not surface text like MinHash).

The construction (Steps 1-4, all deterministic, no learned parameters, no labelled data):

  1. IC(s)          information content by upward corpus-frequency propagation (Resnik). IC(root)=0,
                    specific synsets large. Pluggable source (Brown IC now; our own corpus later).
  2. vec_sparse(s)  the root->s hypernym path, edge parent p -> child c weighted by sqrt(IC(c)-IC(p)).
                    Telescoping gives ||vec||^2 = IC(s), <v(s1),v(s2)> = IC(LCS), so
                    ||v(s1)-v(s2)||^2 = IC(s1)+IC(s2)-2*IC(LCS) = Jiang-Conrath distance, EXACTLY
                    **for pairs that share an ancestor** — which is every noun pair (the noun
                    hierarchy has ONE root, `entity.n.01`, at IC 0) and most verb pairs. For
                    DISJOINT pairs, which only verbs can be (559 verb top nodes, 540 with IC > 0),
                    `jc_tree`'s `lcs_ic = 0.0` fallback assumes a universal IC-0 root that the verb
                    hierarchy does not have, and the two diverge by exactly IC(root1)+IC(root2) —
                    measured up to 19.12. See `jc_tree` for the numbers. This is a SCOPE LIMIT of
                    the claim, not an error in either quantity.
  3. dense(s)       signed feature-hashing of the sparse vector into a fixed D — inner-product
                    preserving (JL), so JC distance survives into D dims. numpy-authored, fixed seed.
  4. faithfulness   30-pair gate: Spearman(hashed L2^2, closed-form JC) — the whole verification.
                    No labels; just "does the hashed geometry track the closed-form metric."

  5. centering      OPT-IN (default OFF). Subtract the global mean token profile. A translation, so
                    pairwise distances — and therefore JC exactness — are untouched; what it changes
                    is the INNER PRODUCT, which is what cosine retrieval and the entroptics Screen
                    read. See `derive_centering_mean` and RESOLVED / limits below.

  6. fingerprint    The vector-space identity. Nodes whose basis fingerprints differ MUST refuse to
                    pool coordinates — `require_same_basis` raises. See `basis_fingerprint`.

⛔ THE FEATURE HASH (Step 3) IS LOAD-BEARING. DO NOT REPLACE IT WITH A NAMED-ANCHOR BASIS.
`LATTICE-2026-07-20.md` §11's kill list contains a stale row saying to delete `dense_vec` "replaced
by projection onto the named anchor basis". That row is refuted by §17.3 of its own document and by
`LATTICE-CONTRACT.md` RESOLVED-2. Measured: a named basis makes `IC(LCS(s, a_k))` produce sparse,
near-constant columns; entroptics normalizes per channel with James-Stein MAD shrinkage, and a
near-zero-MAD column explodes the noise floor to **1e10-1e11** (vs 1.09 for the hash). Three repairs
were tried — centering, antichain anchor selection, and `fold=False` — and none recovered it. The
greedy `(n-1)*IC` selection also returns a near-collinear CHAIN, not a basis, and forcing an
antichain yielded only 21 mutually incomparable regions. The hash spreads mass so that no column
degenerates; that IS its function. Leave Steps 1-3 alone.

HONEST LIMITS (measured, not aspirational — see the docstrings of the functions named):

  * **Nouns only.** Only the hypernym hierarchy carries meaning-geometry here. Adjectives have no
    hypernym tree in WordNet AT ALL — they are organised as `similar_to` clusters, which have never
    been ingested — so *"cheap lodging" -> "inexpensive accommodation" does not work*, and no amount
    of tuning this file will make it. Closing that gap needs `similar_to` +
    `derivationally_related_form` ingested as edges plus an `expand()` walk: a separate ~200-line
    work item, deliberately NOT this one. Verbs likewise need troponymy.

  * **OOV routes to the KEYED arm**, not to a char-trigram fallback inside the coordinate. A token
    with no synset has no meaning-geometry, and `_oov_vec` manufactures a surface-similarity signal
    that lives in the same D as real coordinates and is indistinguishable from one downstream. Use
    `oov="skip"` + `oov_tokens()` to hand those tokens to the lexical/keyed arm, which is the arm
    that can actually answer them.

⚠ MOVED HERE FROM `ember/ontology/geometry.py` — 2026-08-02, the chorus→ember DAG work. Behaviour
unchanged; only the address did.

It reads the STORE zero times in 1303 lines, and imports no beam — its only dependencies are numpy,
`prism.vector`, and the ontology driver, which is already mantle's. So it is not the runner's: it is
the measurement that sits directly on the driver, and it belongs beside it. Nine `chorus → ember`
import sites were personas reaching through the runner for exactly this.

⭐ THE LONGER-TERM HOME IS BEAM, and this is not it. A distance over an abstract IC + is-a graph is a
MEASUREMENT, and measurement is beam's. What stops that today is the 30 places this file looks the
driver up by importing it: beam may not import mantle (they are siblings), so geometry can only rise
once it RECEIVES the driver instead — the same dependency inversion `beam.reach` already uses for
`Keyring`/`Lightcone`. Until that rewrite, here is the correct side of the line: it is strictly closer
to its data than it was, and no DAG edge is invented to hold it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from prism import vector as _vec

# The hash width. This is a FIDELITY parameter, not a storage convenience.
#
# MEASURED (test_geometry_lattice.py::test_hashing_error_is_collision_noise_that_vanishes_with_D),
# Spearman(hashed L2^2, tree_JC) over 150 RANDOM DEEP noun pairs:
#
#     D= 256  rho=0.8849      <- bounded observer (Pi); see D_SMALL
#     D= 512  rho=0.9124      <- the OLD default
#     D=1024  rho=0.9453
#     D=2048  rho=0.9998      <- the default now
#     D=4096  rho=0.9998      <- saturated; past here you buy BLOB, not fidelity
#
# Why the default moved 512 -> 2048. The SPARSE coordinate is exact to machine epsilon (measured
# |sparse L2^2 - tree_JC| = 3.55e-15). At D=512 the HASHED coordinate was only 0.912 rank-faithful
# on deep pairs, so shipping "exact Jiang-Conrath" on top of that fold would have been a claim the
# implementation did not support. At D=2048 the claim survives the fold. The error was always pure
# collision noise — it shrinks monotonically in D — and 512 was simply never revisited against a
# deep-pair test: `faithfulness_check`'s DEFAULT_PAIRS are common, SHALLOW synsets with short
# hypernym paths and few occupied dimensions, so they under-report collisions by construction.
#
# Cost is 4x the per-vertex BLOB, which is the figure `LATTICE-2026-07-20.md` §5.5 already budgets:
# *"a Pi node uses D=256, a big node D=2048."* This is that sentence, honoured.
#
# D IS PART OF THE BASIS FINGERPRINT, so a D=256 Pi and a D=2048 node structurally refuse to pool
# coordinates rather than silently indexing into one space. That refusal is no longer hypothetical
# now that two widths are both in normal use — see test_pi_and_big_node_refuse_to_pool.
_D_DEFAULT = 2048
D_SMALL = 256      # bounded observer / Pi (§5.5, and Phase 4's bounded observer). Supported, not deprecated.
D_BIG = 2048       # a real node — the default
_HASH_SEED = "genesis-geom-v1"

# The identity of the CONSTRUCTION (Steps 1-3), not of this file. It is deliberately still "v1"
# even though centering and the fingerprint were added below, because with centering OFF this
# module emits coordinates that are byte-identical to the v1 space — same seed, same hash, same
# sqrt(dIC) path. Bumping it would make v1 and v2 nodes refuse to pool identical vectors. Centering
# is NOT hidden by that: it enters the fingerprint as its own field (`centering_id`), so a centered
# and an uncentered node still refuse each other. Bump this ONLY if Steps 1-3 change.
GEOMETRY_VERSION = "geom-v1"


# ── Step 1: information content ──────────────────────────────────────────────
# IC is no longer loaded from an external Brown-corpus file. It was materialized ONTO each `wn-*`
# artifact by scripts/enrich_wordnet.py (Resnik IC, derived once) and is served by the store-backed
# wn_store.Synset. This realizes the docstring's own plan: "swap to our-corpus counts later without
# touching anything downstream" — every function below threads `ic` through unchanged; it's just inert
# now (the value rides on the synset). This is what dropped the last nltk dependency.
def load_ic(kind: str = "ic-brown.dat"):
    """No-op sentinel. IC now lives on the synset (stored during enrichment); no IC file is loaded.
    Kept so the `(ic)` argument threaded through the geometry API stays undisturbed."""
    return None


# ⛔⛔ THE MAGNITUDE BOUND EXISTS BECAUSE `math.isfinite` WAS NOT ENOUGH, AND THE GAP WAS LIVE.
#
# nltk's `information_content` returns `_INF = 1e+300` — a MODULE-LEVEL FLOAT LITERAL, not
# `math.inf` — for any synset with zero corpus frequency. `1e300 == float("inf")` is **False** and
# `math.isfinite(1e300)` is **True**, so the isfinite guard below passed it through as an ordinary
# measurement. MEASURED over the full 117,659-synset corpus: **50,278 synsets (42.73%)** carried
# it, bit-identical on nodes 71 and 45, and it flowed into `jc_tree`, `sparse_vec` (as an edge
# weight of `sqrt(1e300)` = 1e150) and `dense_vec`.
#
# ⛔ `IC_ABSURD = 100.0` AND `IC_MEASURED_MAX = 14.709437882542113` STOOD HERE AND BOTH ARE GONE —
# 2026-08-01. They were defended as "derived, not chosen", and neither was:
#
#   * `IC_MEASURED_MAX` is the largest Resnik IC on **ic-brown.dat** — a corpus this runtime no
#     longer reads. `load_ic()` returns None and IC is the INTRINSIC measure now
#     (`wn_store.INTRINSIC_IC_FORMULA`), so the bound was quoting the provenance of a retired
#     measurement. MEASURED on this store: every one of 481,846 nouns carries intrinsic IC and the
#     largest value is exactly **1.0**, not 14.7.
#   * `IC_ABSURD` sat "~7x above that". Seven is the whole of the derivation. A margin chosen to
#     feel loose is a chosen number however carefully it is justified.
#
# THE BOUND IS THE FORMULA'S OWN RANGE, read from the corpus's record of which formula it used.
# `1 - log(desc+1)/log(N+1)` is in **[0, 1] by construction** — `desc >= 0` gives IC <= 1 and
# `desc <= N` gives IC >= 0 — so for an intrinsic corpus any |ic| > 1 is impossible, exactly, with
# no margin at all. A corpus enriched from an EXTERNAL frequency source records a different basis
# and has not told us its range, so there is NO bound to state and none is invented: such a value
# is left unbounded and only `math.isfinite` applies. Absence of a stated range is not a wide range
# ([[absence-is-not-an-affirmative-claim]]).
INTRINSIC_IC_FORMULA_RANGE = "1 - log(desc+1)/log(N+1) is in [0, 1] by construction"
_IC_BOUND: list = []          # [(wn_store generation, the upper bound or None)]


def ic_upper_bound() -> Optional[float]:
    """The largest |IC| this corpus's own IC formula can produce, or None if the corpus does not say.

        intrinsic  `1 - log(desc+1)/log(N+1)`      ->  1.0, exactly, by construction
        frequency  `-log(p)`, p >= 1/N             ->  log(N), exactly — a probability cannot be
                                                      smaller than one observation out of N
        no basis   the corpus states no formula    ->  None: there is no range to state, and an
                                                      unstated range is not a wide one

    ⚠ KEYED ON WHICH STORE, NOT ON THE GENERATION, AND THAT IS DELIBERATE. `ic_of` is the hottest
    function in this package — every node of every `tree_path`, millions of calls in one geometry
    derivation — so it cannot afford a store read per call. Keying on `wn_store.generation()` would
    look tighter and be far worse: `_gate` advances the generation on EVERY POLL for a store that
    cannot report a write mark, so on such a store this would take one `get_artifact` per `ic_of`.
    What this reads is WHICH FORMULA the corpus used, which changes only on a re-enrichment, and a
    re-enrichment goes through `wn_store.invalidate()` / `bind()` — both of which move `_SOURCE`,
    which is what this is keyed on."""
    try:
        from mantle.ontology import driver as wn
        key = id(wn._SOURCE)
    except Exception:
        return None
    if _IC_BOUND and _IC_BOUND[0][0] == key:
        return _IC_BOUND[0][1]
    bound = None
    try:
        basis = wn.ic_basis()
        src, n = basis.get("source"), basis.get("n")
        if src == wn.INTRINSIC_IC_SOURCE:
            bound = 1.0
        elif src and isinstance(n, (int, float)) and n > 1:
            bound = math.log(float(n))
    except Exception:
        bound = None
    _IC_BOUND[:] = [(key, bound)]
    return bound


class AbsurdIC(ValueError):
    """A stored `ic` that no real measurement could produce. Raised, never silently coerced."""


def ic_of(synset, ic=None, *, strict: bool = True) -> float:
    """Information content of a synset — read from the store-backed synset (stored, not recomputed).

    ⚠ AN ABSENT `ic` AND A ZERO `ic` STILL BOTH RETURN 0.0, DELIBERATELY: every caller here does
    arithmetic with the result, and `wn_store.Synset.has_ic()` is the way to tell them apart. What
    changed is that a value which is *present and impossible* no longer joins them. Absence is a
    known, counted state (`wn_store.ic_coverage()`); a sentinel is a wrong answer wearing the
    costume of a right one, and the two must not share a code path.

    `strict=True` (the default) RAISES `AbsurdIC` rather than returning a number. That is the
    "fail loudly" half of the contract: a caller that would rather exclude such synsets should do
    it EXPLICITLY — filter on `wn_store.Synset.has_ic()` or catch `AbsurdIC` — because the one
    behaviour this must never have again is flowing through as a plausible float.
    """
    # ⛔ THESE WERE `except Exception: return 0.0` AND `except (TypeError, ValueError): return 0.0`.
    # Neither could ever catch an ABSENT ic — `wn_store.Synset.ic()` handles absence itself and
    # never raises (`self._ic if self._ic is not None else 0.0`). So the only things they caught
    # were a wrong object (no `.ic`, or None) and a store returning a non-number, i.e. a programming
    # fault or a corrupt row — and they turned both into the one value that reads as a legitimate
    # measurement. `has_ic()`'s own docstring records where that lands: all-zero IC makes
    # `canonical_parent` pick the spanning tree ALPHABETICALLY, and every JC distance is then
    # computed over an arbitrary tree and returned as a measurement. Absence stays 0.0 (the
    # documented contract above); a fault is now loud.
    raw = synset.ic()
    if raw is None:
        return 0.0          # ABSENT — the documented contract above; `has_ic()` tells it from zero
    v = float(raw)          # anything non-numeric reaching here is a corrupt row, not an absence
    if not math.isfinite(v):
        # Same class as the absurd-value branch below, so it honours `strict` the same way.
        if strict:
            raise AbsurdIC(
                f"synset {getattr(synset, 'name', lambda: '?')()!r} has a non-finite ic={raw!r}. "
                f"This is a corrupt stored value, not an absent one — absence is 0.0 via "
                f"`Synset.ic()` and is counted by `wn_store.ic_coverage()`. Re-run enrichment."
            )
        return 0.0
    bound = ic_upper_bound()
    if bound is not None and abs(v) > bound:
        if strict:
            name = getattr(synset, "name", lambda: "?")()
            raise AbsurdIC(
                f"synset {name!r} has ic={v!r}, which this corpus's own IC formula cannot produce "
                f"({INTRINSIC_IC_FORMULA_RANGE}). 1e+300 is nltk's zero-frequency sentinel and is "
                f"FINITE, so it passes `math.isfinite` — it must never reach the coordinate. "
                f"Re-run enrichment: the correct handling is to record `ic_status` and leave `ic` "
                f"absent."
            )
        return 0.0
    return v


# ── the canonical spanning tree: ONE deterministic parent per synset ──────────
# WordNet is a DAG (a few synsets have multiple hypernyms). A single linear coordinate cannot
# realize DAG-JC exactly (the inner product SUMS over shared dims; DAG-JC needs the MAX-IC common
# subsumer). So we fix a canonical spanning TREE — max-IC parent, name as deterministic tiebreak —
# and the coordinate realizes TREE-JC exactly. WordNet is ~99% tree, so tree-JC ≈ DAG-JC, and a
# self-consistent exact metric is what E's locality-sensitivity actually wants.
def canonical_parent(synset, ic):
    parents = synset.hypernyms() + synset.instance_hypernyms()
    if not parents:
        return None
    return max(parents, key=lambda h: (ic_of(h, ic), h.name()))


def tree_path(synset, ic):
    """Root<-s along the canonical spanning tree: [s, parent, ..., root]."""
    out = [synset]
    cur, seen = synset, {synset.name()}
    while True:
        p = canonical_parent(cur, ic)
        if p is None or p.name() in seen:
            break
        out.append(p); seen.add(p.name()); cur = p
    return out


# ── Step 1b: Laplace smoothing of the zero-frequency hole, and IC's SECOND CHANNEL ──────────────
#
# THE HOLE. Resnik IC is `-log(count(s)/N)`. A synset with count 0 has NO IC — nltk returns its
# `_INF` sentinel (see `IC_ABSURD`) and `enrich_wordnet.measure_ic_detail` correctly records
# `ic_status="ic_zero_frequency"` and leaves `ic` GENUINELY ABSENT. Measured at corpus scale:
# **50,278 of 117,659 synsets (42.73%)** are in that state, and `ic_of` returns 0.0 for all of
# them. 0.0 is also the IC of the ROOT. So today the most specific corner of the ontology and the
# top of it are the same number, and every edge into a hole gets `max(IC(c)-IC(p), 0) == 0`, i.e.
# weight zero: the coordinate simply stops at the last attested ancestor.
#
# ⛔ WHY NOT BACK THE HOLE OFF TO 0.0 AND MOVE ON — MEASURED. That is what happens today, and it
# BREAKS MONOTONICITY (`IC(child) >= IC(parent)`), because a hole child sits at 0.0 under an
# attested parent at IC 8. Propagating that through the coordinate drove `||v1-v2||^2` vs `jc_tree`
# error to **27.83** — five orders of magnitude past the 3.55e-15 the sparse coordinate is supposed
# to hold. Monotonicity is not an aesthetic property here; it is the precondition for the
# telescoping identity in Step 2.
#
# THE FIX: LAPLACE, APPLIED **BEFORE** DESCENDANT-INCLUSIVE PROPAGATION.
# `ic-brown.dat` stores counts that are ALREADY descendant-inclusive (a hypernym subsumes its
# children's mass). Adding alpha to *that* number would not restore monotonicity — it would shift
# every node by the same constant and leave the hole a hole. So we de-propagate to per-synset OWN
# counts, add alpha there, and re-propagate:
#
#     own'(s)  = own(s) + alpha                          (every synset, holes included)
#     cum'(s)  = own'(s) + sum over tree-children cum'(c)
#     IC'(s)   = -log(cum'(s) / N')
#
# Monotonicity then holds BY CONSTRUCTION, not by luck: `cum'(parent) = own'(parent) + ... +
# cum'(child) >= cum'(child)` because `own' >= alpha > 0`, so `p'(parent) >= p'(child)` and
# `IC'(parent) <= IC'(child)` for every edge in the tree. No hole remains, because no `cum'` is
# zero. This is asserted over the whole noun vocabulary in `test_geometry_lattice.py`.
#
# ⛔ NOT GOOD-TURING. GT is the better estimator for held-out mass, and it is the wrong tool HERE:
# it needs the frequency-of-frequencies table (how many species were seen exactly once, twice, ...),
# which is NOT RECONSTRUCTIBLE from the `(count, sum)` pair a node keeps. Requiring it would mean
# shipping a third, order-DEPENDENT accumulator, and that costs the order-freeness that lets two
# observers merge their counts and converge without a coordinator. Laplace keeps the merge algebra
# trivial: counts add, and `alpha` is applied once at read time.
#
# WHY THE TREE AND NOT THE DAG. WordNet is a DAG, and nltk propagates a synset's count up EVERY
# hypernym path, so a parent's cumulative count can be less than the sum of its children's and the
# de-propagated `own` can come out NEGATIVE. Measured on ic-brown.dat's noun table: this happens on
# a minority of nodes and is an artefact of double-counting, not a real deficit. We clamp `own` at
# 0 and re-propagate over the CANONICAL SPANNING TREE — which is not a compromise but the correct
# choice, because the coordinate in Step 2 realizes TREE-JC exactly and nothing else. The count
# system and the metric it feeds are now the same system.
IC_LAPLACE_ALPHA = 1.0


def smooth_ic(own_counts: Dict[str, float], parents: Dict[str, Optional[str]], *,
              alpha: float = IC_LAPLACE_ALPHA) -> Dict[str, Tuple[float, float]]:
    """Laplace-smoothed Resnik IC and its standard error, from PER-SYNSET OWN counts.

    `own_counts` maps synset name -> its OWN corpus count (NOT descendant-inclusive). Names present
    in `parents` but absent here are treated as own count 0 — that is the hole, and it is exactly
    the case this function exists to fill. `parents` maps name -> canonical parent name, or None
    for a root; it must be the SAME spanning tree `canonical_parent` induces, or the IC this
    returns will not be the IC the coordinate telescopes over.

    Returns `{name: (ic, se)}` where

        ic = -log(cum'(s) / N')          nats, monotone non-decreasing down every tree edge
        se = sqrt((1 - p) / cum'(s))     nats, the SECOND CHANNEL (see `jc_tree_se`)

    `se` is the closed-form standard error of a `-log p` estimate under the binomial that generated
    the count: `Var(p_hat) = p(1-p)/N`, and the delta method through `-log` gives
    `se(IC) = sqrt((1-p)/(N p)) = sqrt((1-p)/cum')`. It is `~1/sqrt(n)` for small p — a CLOSED FORM,
    derived, not an entroptics read. A synset resting entirely on `alpha` (a pure hole) gets
    `cum' = alpha` and therefore the largest `se` in the vocabulary, which is the honest report:
    its IC is an interpolation, not a measurement, and the number says so.

    ⛔ `se` IS A SECOND CHANNEL. IT IS NEVER FOLDED INTO `ic`. See `jc_tree_se` for the measurement
    that forced that rule.
    """
    a = float(alpha)
    if a <= 0.0:
        raise ValueError(f"IC Laplace alpha must be > 0 (a hole needs mass); got {alpha!r}")

    children: Dict[str, List[str]] = {}
    for name, par in parents.items():
        if par is not None:
            children.setdefault(par, []).append(name)

    # Depth (root distance) gives a topological order without recursion — deepest first, so every
    # child's cum' is final before its parent reads it. Cycles cannot occur: `parents` comes from
    # `tree_path`, which refuses to revisit a name.
    depth: Dict[str, int] = {}

    def _depth(n: str) -> int:
        d, seen = 0, set()
        cur = n
        while True:
            if cur in depth:
                d += depth[cur]
                break
            p = parents.get(cur)
            if p is None or p in seen:
                break
            seen.add(cur)
            cur = p
            d += 1
        depth[n] = d
        return d

    order = sorted(parents, key=_depth, reverse=True)

    cum: Dict[str, float] = {}
    for n in order:
        cum[n] = max(float(own_counts.get(n, 0.0)), 0.0) + a + sum(
            cum.get(c, 0.0) for c in children.get(n, ()))

    # N' is the total smoothed mass = the sum over ROOTS of their cumulative counts. Equivalently
    # sum(own') over the whole vocabulary; taken at the roots so a multi-rooted hierarchy (verbs:
    # 559 top nodes) normalizes over its own forest rather than borrowing the nouns' N.
    total = sum(cum[n] for n, p in parents.items() if p is None)
    if total <= 0.0:
        raise ValueError("smooth_ic: total smoothed mass is zero; the tree has no roots")

    out: Dict[str, Tuple[float, float]] = {}
    for n, c in cum.items():
        p = c / total
        out[n] = (-math.log(p), math.sqrt(max(1.0 - p, 0.0) / c))
    return out


def ic_se_of(synset, *, default: Optional[float] = None) -> Optional[float]:
    """The SECOND CHANNEL: the standard error of this synset's IC, in nats, or `default` when the
    synset does not carry one.

    Returns `None` rather than 0.0 by default, and the distinction is the whole point: 0.0 means
    "this IC is exact", which is a strong and almost always false claim, whereas `None` means "this
    corpus was enriched before `ic_se` existed and the uncertainty is unknown". A caller that
    averages `se` over a vocabulary must not silently treat unmeasured synsets as perfectly
    measured ones — that is the same class of error as `ic_of` returning 0.0 for an absent IC, and
    it is why `wn_store.Synset.has_ic()` exists.
    """
    try:
        raw = synset.ic_se()
    except Exception:
        return default
    if raw is None:
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v) or v < 0.0:
        return default
    return v


def tree_lcs_ic(s1, s2, ic) -> float:
    """IC of the least common subsumer ON THE SPANNING TREE — the deepest node both tree-paths share.

    ⚠ THE `return 0.0` FALLBACK IS AN ASSUMPTION, NOT A NEUTRAL DEFAULT. It says "these two synsets
    share an ancestor of IC 0", i.e. it assumes a UNIVERSAL IC-0 ROOT. See `jc_tree`."""
    p2 = {s.name() for s in tree_path(s2, ic)}
    for node in tree_path(s1, ic):          # tree_path is deepest-first, so first shared = the LCS
        if node.name() in p2:
            return ic_of(node, ic)
    return 0.0


def jc_tree(s1, s2, ic) -> float:
    """Jiang-Conrath on the canonical spanning tree.

    ⚠ THE EXACTNESS CLAIM HAS A SCOPE LIMIT, AND IT WAS PREVIOUSLY UNSTATED. The module docstring
    says the coordinate realizes this metric EXACTLY. That is true **only for pairs that share an
    ancestor.** For DISJOINT pairs the coordinate and this function measure different things:

      * `tree_lcs_ic` returns 0.0 when the two tree-paths never meet, which ASSUMES a universal
        IC-0 root. NOUNS have exactly that: measured, the noun hierarchy has **1 top node**
        (`entity.n.01`) and its IC is 0, so the assumption is exactly right and no noun pair is
        ever disjoint.
      * **VERBS HAVE NO SINGLE ROOT.** Measured on the full corpus: the verb hierarchy has **559
        top nodes, 540 of them with IC > 0** (largest 14.7094). Two verbs under different top
        nodes share no ancestor at all.

    For such a pair this function returns `IC(s1) + IC(s2)`, while the coordinate measures
    `||v1||^2 + ||v2||^2` = `(IC(s1) - IC(r1)) + (IC(s2) - IC(r2))`, because each path vector
    telescopes only down to its OWN root. The gap is therefore exactly `IC(r1) + IC(r2)` — VERIFIED
    as a mechanism, not merely observed: over 2,878 disjoint verb pairs the largest deviation of
    the observed gap from `IC(r1)+IC(r2)` is **7.105427e-15**, i.e. machine epsilon.

    MEASURED |sparse L2^2 - jc_tree|, corrected corpus:

        noun pairs sharing an ancestor (n=20,000)   7.105427e-15   <- exact
        verb pairs sharing an ancestor (n=202)      3.552714e-15   <- exact
        verb pairs DISJOINT            (n=3,797)    up to 19.1159  <- NOT hash noise

    The worst measured case is `emigrate.v.01` vs `freeze.v.01`: `jc_tree` 24.6314, coordinate
    5.5154. Both numbers are self-consistent; they are answers to different questions.

    So: **the coordinate is exact for pairs sharing an ancestor, which is every noun pair and most
    verb pairs, and diverges by `IC(r1)+IC(r2)` for disjoint pairs, which only verbs can be.**
    Neither widening this to "always exact" nor narrowing it to "nouns only" is accurate. Closing
    it for real would mean either a synthetic IC-0 verb root or a per-pair root-aware correction —
    a separate decision, deliberately NOT made here."""
    return ic_of(s1, ic) + ic_of(s2, ic) - 2.0 * tree_lcs_ic(s1, s2, ic)


def tree_lcs(s1, s2, ic):
    """The least common subsumer ON THE SPANNING TREE — the synset, not its IC.

    `tree_lcs_ic` throws the node away and returns only the number, which is all `jc_tree` needs;
    the SECOND CHANNEL needs the node itself (to read its `se`), and re-walking both paths to get
    it back would be the same work twice.
    """
    p2 = {s.name(): s for s in tree_path(s2, ic)}
    for node in tree_path(s1, ic):
        if node.name() in p2:
            return node
    return None


def jc_tree_se(s1, s2, ic, *, default_se: Optional[float] = None) -> Optional[float]:
    """Standard error of `jc_tree(s1, s2, ic)`, in nats — the SECOND CHANNEL of the distance.

    `JC = IC(s1) + IC(s2) - 2*IC(LCS)`, so propagating the three standard errors in quadrature:

        se(JC) = sqrt( se(s1)^2 + se(s2)^2 + 4*se(LCS)^2 )

    Returns `None` if any of the three synsets carries no `ic_se` and no `default_se` is supplied.
    `None` propagates rather than defaulting to 0, for the reason in `ic_se_of`: an unmeasured
    uncertainty is not a zero uncertainty, and a retrieval layer that ranked on `JC ± 0` because
    the corpus predates `ic_se` would be reporting confidence it never earned.

    ⛔⛔ THIS IS A SECOND CHANNEL. IT MUST NEVER BE FOLDED BACK INTO THE COORDINATE. MEASURED:

        carry `se` alongside the point IC (this function)      |L2^2 - jc_tree| = 7.1e-15  EXACT
        resample IC per call within +/- 1 se, then coordinate  |L2^2 - jc_tree| = 5.99     DESTROYED

    The mechanism is not noise-averaging and will not go away with more samples. Step 2's identity
    is a TELESCOPING one: `<v(s1), v(s2)> = IC(LCS)` holds because the shared root->LCS prefix of
    the two paths is built from the SAME edge weights and cancels exactly. Draw IC independently on
    each call and the two prefixes are no longer the same numbers, so they stop cancelling, and the
    error is the accumulated prefix disagreement — which GROWS with path depth. Uncertainty about a
    coordinate is not the same object as a perturbation of it. Report it beside the value.
    """
    a = ic_se_of(s1, default=default_se)
    b = ic_se_of(s2, default=default_se)
    if a is None or b is None:
        return None
    lcs = tree_lcs(s1, s2, ic)
    c = 0.0 if lcs is None else ic_se_of(lcs, default=default_se)
    if c is None:
        return None
    return math.sqrt(a * a + b * b + 4.0 * c * c)


# ── Step 2: the sqrt(dIC) hypernym-path vector (sparse; L2^2 = tree Jiang-Conrath EXACTLY) ──
def _path_edges(synset, ic) -> List[Tuple[str, float]]:
    """Edges of the canonical tree path: child c -> parent p yields (c.name, IC(c)-IC(p))."""
    out: List[Tuple[str, float]] = []
    path = tree_path(synset, ic)
    for c, p in zip(path, path[1:]):
        out.append((c.name(), max(ic_of(c, ic) - ic_of(p, ic), 0.0)))
    return out


def sparse_vec(synset, ic) -> Dict[str, float]:
    """{node c : sqrt(IC(c) - IC(parent(c)))} over the path — the exact JC coordinate (pre-hash)."""
    return {name: math.sqrt(d) for name, d in _path_edges(synset, ic) if d > 0.0}


# ── Step 3: signed feature-hashing into a fixed dense D ───────────────────────
def _hash(node: str) -> Tuple[int, float]:
    h = int(hashlib.blake2b((_HASH_SEED + node).encode(), digest_size=8).hexdigest(), 16)
    return h, (1.0 if (h >> 40) & 1 else -1.0)


# The uncentered dense coordinate, keyed by (synset name, D, id(ic)). Bounded so a long-running
# node cannot grow it without limit; the working set of a conversation is tens of concepts, and the
# bound simply stops caching past it rather than evicting (an eviction policy here would be a
# premonition about which concepts matter). See `dense_vec` for why `id(ic)` is the right key.
_DENSE_CACHE: dict = {}
_DENSE_CACHE_MAX = 50_000
_DENSE_GEN: int = -1        # the `wn_store.generation()` the entries above were built under
_WN = None                  # the wn_store module, resolved once — this is read on every dense_vec


def _dense_gen() -> int:
    """The generation the dense coordinates are valid for — `wn_store.generation()`, which advances
    only when the store was actually written to.

    ⛔ `id(ic)` HAD STOPPED KEYING ON ANYTHING, AND THAT IS THE WHOLE BUG. The key below was built
    on a genuinely correct instinct, stated in its own comment: cache on the IDENTITY of what the
    value is derived from, so *"a genuinely new table is a new object, so the cache misses BY
    CONSTRUCTION rather than by a version number someone has to remember to bump."* But `load_ic()`
    returns **None** — IC was moved onto the synset and the file was dropped — so `id(ic)` is
    `id(None)`, the same integer for the life of the process. The key was carrying a guarantee it
    could no longer make, which is worse than not having it: MEASURED 2026-08-01, a repair rewrote
    141,102 stored `ic` values and every cached vector kept serving the old geometry.

    The identity that varies now is the store's. `wn_store` verifies each Synset against the
    artifact it was read from and advances its generation whenever the store moved under it, so
    keying here on that generation restores exactly the property the comment claimed — a changed
    corpus is a different key, and a stale vector is unreachable rather than merely unlikely.

    ⚠ IT IS DELIBERATELY COARSER THAN ONE SYNSET. A dense coordinate is not a function of its own
    synset alone: `sparse_vec` walks `tree_path`, so every ANCESTOR's IC is an input, and a parent
    being repaired changes a child's vector while the child's own artifact never moves. Attributing
    that exactly would mean re-walking the path — MEASURED 20 µs of the 37.6 µs a full rebuild
    costs, i.e. most of what the cache saves. So the whole cache is dropped when the store changes
    and refilled at 37.6 µs per concept over a working set of tens. That is a few milliseconds,
    once, caused by a real write — not a window, not a schedule, and not a guess about which
    ancestors mattered."""
    global _DENSE_GEN, _WN
    if _WN is None:
        try:
            from mantle.ontology import driver as _wn_mod
        except Exception:
            return _DENSE_GEN
        _WN = _wn_mod
    g = _WN.generation()
    if g != _DENSE_GEN:
        _DENSE_CACHE.clear()
        _DENSE_GEN = g
    return g


def dense_vec(synset, ic, D: int = _D_DEFAULT, center: Optional[np.ndarray] = None) -> np.ndarray:
    """Signed feature-hash of the sparse JC coordinate into D dims (JL: inner products preserved).

    `center` is the OPT-IN global mean token profile (default None = off; see
    `derive_centering_mean`). Subtracting it is a pure TRANSLATION of the whole coordinate set, so
    every pairwise distance ||v(s1) - v(s2)|| is EXACTLY invariant — the JC exactness claim in the
    module docstring survives centering unchanged, which is asserted in `test_geometry_lattice.py`.
    What centering moves is the inner product / cosine, i.e. exactly the quantity that cosine
    retrieval and the entroptics Screen read.

    ⚠ If you center, the vector is only comparable to other vectors centered by the SAME mean.
    That is why the mean's identity is a field of `basis_fingerprint` and not a local convenience.
    """
    # ── THE HASH IS A PURE FUNCTION OF THE COORDINATE — COMPUTE IT ONCE ─────────────────────────
    # `sparse_vec(synset, ic)` walks the synset's hypernym path and `_hash` is deterministic, so the
    # UNCENTERED vector depends on nothing but the synset and the IC table. It was rebuilt from
    # scratch for every concept on every turn: MEASURED 2026-07-31, `projection.frame` cost 42 ms
    # per answer almost entirely here, and a conversation re-derives the SAME handful of concepts
    # turn after turn.
    #
    # ⚠ KEYED ON THE IDENTITY OF WHAT THE COORDINATE IS BUILT FROM, NOT JUST THE SYNSET. IC is what
    # the coordinate is BUILT FROM — a rebuilt or reloaded IC is a different geometry, and serving a
    # stale vector against a new IC would be the silent-substitution failure this codebase keeps
    # finding. `id(ic)` was that guarantee while IC was a loaded table; it stopped being one when IC
    # moved onto the synset and `load_ic()` began returning `None` — see `_dense_gen`, which
    # restores it by keying on the generation `wn_store` last verified the corpus at. `id(ic)` is
    # kept beside it: it costs nothing and it is still exact for any caller that hands in a real
    # table.
    # Centering is applied AFTER the cache (it is a caller-supplied translation, not part of the
    # coordinate), so one cached vector serves centered and uncentered callers alike.
    _key = None
    try:
        _key = (getattr(synset, "name", lambda: None)(), int(D), id(ic), _dense_gen())
    except Exception:
        _key = None
    if _key is not None and _key[0] is not None:
        _hit = _DENSE_CACHE.get(_key)
        if _hit is not None:
            v = _hit.copy() if center is not None else _hit
            if center is None:
                return v
        else:
            v = np.zeros(D, dtype=np.float64)
            for node, w in sparse_vec(synset, ic).items():
                h, sgn = _hash(node)
                v[h % D] += sgn * w
            if len(_DENSE_CACHE) < _DENSE_CACHE_MAX:
                _DENSE_CACHE[_key] = v
            if center is None:
                return v
            v = v.copy()
        if center is not None:
            c = np.asarray(center, dtype=np.float64)
            if c.shape != (D,):
                raise ValueError(f"centering mean has shape {c.shape}, expected ({D},)")
            return v - c
        return v

    v = np.zeros(D, dtype=np.float64)
    for node, w in sparse_vec(synset, ic).items():
        h, sgn = _hash(node)
        v[h % D] += sgn * w
    if center is not None:
        c = np.asarray(center, dtype=np.float64)
        if c.shape != (D,):
            raise ValueError(f"centering mean has shape {c.shape}, expected ({D},)")
        v = v - c
    return v


# ── Step 5: centering — the global mean token profile ────────────────────────
# Measured (LATTICE-2026-07-20.md §17.4): subtracting the global mean token profile moves
# cross-domain separation from +0.505 to -0.332 — different domains become ANTI-correlated rather
# than merely less similar — at unchanged K_signal (18-22) and noise floor (1.09).
#
# It is "one line" of arithmetic and several of bookkeeping, and the bookkeeping is the part that
# matters: a mean recomputed per call, or computed over whatever corpus a node happened to hold, is
# a DIFFERENT translation on every node. Two such nodes emit coordinates that still look valid,
# still have the right norm, still matmul — and are silently in different spaces. That is the exact
# failure mode Prism's `model_id` was invented to stop. So the mean is (a) derived from a DEFINED,
# node-independent sample, (b) persisted rather than recomputed, and (c) identified by content hash
# inside the basis fingerprint.

_CENTER_MAGIC = "agience/geometry/centering/v1"


def derive_centering_mean(synsets: Iterable, ic=None, D: int = _D_DEFAULT) -> np.ndarray:
    """The global mean token profile over a DEFINED sample: mean of the row a monosemous token
    would emit, `log1p(IC(s)) * unit(dense_vec(s))`, over `synsets`.

    Constructed in the row space of `text_to_signal` on purpose — centering must subtract a mean of
    the same kind of object it is subtracted from, or it is just an arbitrary offset.

    The sample is the caller's, but it must be REPRODUCIBLE, and the canonical choice is the whole
    noun vocabulary of the IC source (`wn_store.all_synsets(NOUN)`) — which makes the mean a pure
    function of (IC source revision, D, geometry version) and therefore identical on every node
    that agrees on those. `centering_mean_id` pins whatever was actually used, so a hand-supplied
    or truncated sample cannot masquerade as the canonical one; it just gets a different id and
    stops pooling with nodes that used a different one.

    Uncentered vectors are NOT passed through here — this is only the offset. See `dense_vec`.
    """
    acc = np.zeros(D, dtype=np.float64)
    n = 0
    for s in synsets:
        v = dense_vec(s, ic, D)                       # uncentered by construction
        nv = _vec.norm(v)
        if not nv:
            continue
        acc += math.log1p(max(ic_of(s, ic), 0.0)) * (v / nv)
        n += 1
    if n == 0:
        raise ValueError("derive_centering_mean: sample produced no usable synset coordinates; "
                         "refusing to return a zero mean, which would silently mean 'no centering'")
    return acc / n


def centering_mean_id(center: Optional[np.ndarray]) -> str:
    """Content id of a centering mean — `"none"` when centering is off.

    Hashed over float32 big-endian bytes, not float64 native, so the id does not depend on the
    node's byte order or on float64 noise from a differently-ordered accumulation of the same
    sample. float32 is also the storage width (§2.2: D x float32 BLOB per vertex), so the id
    identifies what is actually written down.
    """
    if center is None:
        return "none"
    a = np.asarray(center, dtype=">f4")
    h = hashlib.blake2b(digest_size=16)
    h.update(_CENTER_MAGIC.encode())
    h.update(str(a.shape[0]).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def save_centering_mean(center: np.ndarray, path) -> str:
    """Persist a centering mean next to its id. Returns the id.

    Persisted rather than recomputed because recomputation is the bug: it is expensive (a full
    vocabulary walk), and two nodes that recompute over even slightly different vocabularies get
    two different translations and become silently incomparable.
    """
    a = np.asarray(center, dtype=">f4")
    mid = centering_mean_id(a)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"magic": _CENTER_MAGIC, "D": int(a.shape[0]), "id": mid,
                             "mean": [float(x) for x in a]}), encoding="utf-8")
    return mid


def load_centering_mean(path) -> np.ndarray:
    """Read a persisted centering mean, VERIFYING its recorded id against its bytes.

    The verification is not ceremony: a mean that was truncated, re-rounded, or written by a
    different D still loads as a perfectly usable float array and still centers — into the wrong
    space, with no error anywhere downstream."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("magic") != _CENTER_MAGIC:
        raise ValueError(f"not a centering-mean file: magic={d.get('magic')!r}")
    a = np.asarray(d["mean"], dtype=np.float64)
    if a.shape[0] != int(d["D"]):
        raise ValueError(f"centering mean length {a.shape[0]} != recorded D {d['D']}")
    got = centering_mean_id(a)
    if got != d.get("id"):
        raise ValueError(f"centering mean id mismatch: file says {d.get('id')}, bytes say {got}")
    return a


def default_centering(D: int = _D_DEFAULT) -> Optional[np.ndarray]:
    """The node's centering mean, or None.

    **Defaults to None — centering is OFF.** Phase 2 is formally gated on the Phase 0.A recall@k
    A/B, which has not run. Wiring, testing and proving centering is this unit's job; flipping the
    default is a separate, gated decision that belongs to whoever reads that A/B. Until then a node
    opts in explicitly by setting EMBER_GEOM_CENTERING to a saved mean's path.
    """
    p = os.getenv("EMBER_GEOM_CENTERING", "").strip()
    if not p:
        return None
    c = load_centering_mean(p)
    if c.shape[0] != D:
        raise ValueError(f"EMBER_GEOM_CENTERING mean is D={c.shape[0]}, this call asked for D={D}")
    return c


# ── the basis fingerprint: vector-space identity, and the refusal to pool across one ────────────
# Prism's `model_id` was a VECTOR-SPACE identity, not a model name (`hf:<path>@<ver>+seq<N>`),
# because two pods with different seq caps once reported the same id and Mantle indexed their
# outputs into ONE space. Nothing raised; retrieval just got quietly worse. The ontology coordinate
# has the same hazard with four knobs instead of one, so it gets the same discipline.

class BasisMismatch(Exception):
    """Two coordinate sets are not in the same vector space. Raised, never warned."""


@dataclass(frozen=True)
class BasisFingerprint:
    """Everything that has to agree before two nodes' coordinates may be compared or pooled.

    * `ic_revision`      — the IC source content revision. IC sets every edge weight, so a corpus
                           re-enrichment re-scales the whole coordinate.
    * `D`                — hash width. Different D = different space, trivially.
    * `geometry_version` — the Steps 1-3 construction (hash seed, sqrt(dIC), spanning-tree rule).
    * `centering_id`     — `"none"` or the mean's content id. A centered and an uncentered
                           coordinate have the same shape and norm-ish magnitude and will matmul
                           happily against each other while meaning nothing.
    """
    ic_revision: str
    D: int
    geometry_version: str
    centering_id: str

    def token(self) -> str:
        """Short stable hex token — what a node advertises and what rides on an exchanged BLOB."""
        h = hashlib.blake2b(digest_size=16)
        h.update(b"agience/geometry/basis/v1")
        for part in (self.ic_revision, str(self.D), self.geometry_version, self.centering_id):
            h.update(part.encode("utf-8")); h.update(b"\0")
        return h.hexdigest()

    def as_dict(self) -> Dict[str, object]:
        return {"ic_revision": self.ic_revision, "D": self.D,
                "geometry_version": self.geometry_version, "centering_id": self.centering_id,
                "token": self.token()}

    def __str__(self) -> str:
        return f"basis:{self.token()[:16]}(D={self.D},{self.geometry_version},c={self.centering_id[:8]})"


_IC_REV_CACHE: Dict[str, str] = {}


def ic_source_revision(*, override: Optional[str] = None, synsets: Optional[Iterable] = None) -> str:
    """Content revision of the IC source: blake2b over every (synset name, IC) pair, name-sorted.

    Not a version string a human maintains, because a human maintaining it is how it goes stale —
    and a stale IC revision is worse than none: it asserts sameness across a re-enrichment that
    re-scaled every edge weight in the geometry.

    `override` lets a node pin a revision it already knows (avoiding a vocabulary walk) and lets
    tests inject one. `synsets` supplies the vocabulary directly; otherwise it is read from
    `wn_store`. If neither is available this RAISES rather than returning "unknown" — an unknown
    revision that compares equal to another unknown revision is the confused-deputy bug, restated.
    """
    if override:
        return override
    if synsets is None:
        try:
            from mantle.ontology import driver as wn
            synsets = wn.all_synsets()
        except Exception as e:                                  # noqa: BLE001 - re-raised explicitly
            raise BasisMismatch(
                "cannot determine IC source revision (wn_store unavailable): "
                f"{e}. Refusing to fabricate one — pass override= if you know it."
            ) from e
    items = sorted((s.name(), ic_of(s, None)) for s in synsets)
    if not items:
        raise BasisMismatch("IC source is empty; refusing to fingerprint an empty basis")
    key = f"{len(items)}:{items[0][0]}:{items[-1][0]}"
    cached = _IC_REV_CACHE.get(key)
    if cached:
        return cached
    h = hashlib.blake2b(digest_size=16)
    h.update(b"agience/geometry/ic/v1")
    for name, v in items:
        h.update(name.encode("utf-8")); h.update(b"\0")
        h.update(np.float64(v).astype(">f8").tobytes())
    rev = h.hexdigest()
    _IC_REV_CACHE[key] = rev
    return rev


def basis_fingerprint(D: int = _D_DEFAULT, center: Optional[np.ndarray] = None, *,
                      ic_revision: Optional[str] = None,
                      synsets: Optional[Iterable] = None) -> BasisFingerprint:
    """The vector-space identity of coordinates this node would emit under these settings."""
    return BasisFingerprint(
        ic_revision=ic_source_revision(override=ic_revision, synsets=synsets),
        D=int(D),
        geometry_version=GEOMETRY_VERSION,
        centering_id=centering_mean_id(center),
    )


def basis_conflicts(a: BasisFingerprint, b: BasisFingerprint) -> List[str]:
    """The fields on which two bases disagree — empty list means compatible.

    Returns the fields rather than a bool because "these do not pool" is an operational report:
    a D mismatch is a config error, an `ic_revision` mismatch means one node re-enriched, and a
    `centering_id` mismatch means one node flipped the Phase-2 flag ahead of the other.
    """
    out: List[str] = []
    if a.ic_revision != b.ic_revision:
        out.append("ic_revision")
    if a.D != b.D:
        out.append("D")
    if a.geometry_version != b.geometry_version:
        out.append("geometry_version")
    if a.centering_id != b.centering_id:
        out.append("centering_id")
    return out


def require_same_basis(a: BasisFingerprint, b: BasisFingerprint) -> None:
    """Structural refusal: raises `BasisMismatch` if `a` and `b` are not the same space.

    This is a function that raises, not a comment saying "should check", because every failure it
    guards is a WRONG-ANSWER failure rather than an empty-answer one: mismatched coordinates have
    the right dtype and the right shape and produce a confident, meaningless ranking.
    """
    bad = basis_conflicts(a, b)
    if bad:
        raise BasisMismatch(
            f"refusing to pool coordinates across bases: {', '.join(bad)} differ "
            f"({a.as_dict()} vs {b.as_dict()})"
        )


def pool_coordinates(items: Sequence[Tuple[BasisFingerprint, np.ndarray]],
                     expect: Optional[BasisFingerprint] = None) -> np.ndarray:
    """Stack `(fingerprint, coords)` contributions into one [N, D] matrix, or refuse.

    The refusal lives HERE, at the only place coordinates from different observers actually meet,
    so it cannot be forgotten at a call site. An empty pool raises too: silently returning an empty
    matrix is how "nobody was compatible" becomes "no results", which reads as a normal miss.
    """
    if not items:
        raise BasisMismatch("pool_coordinates: nothing to pool")
    base = expect or items[0][0]
    mats = []
    for fp, arr in items:
        require_same_basis(base, fp)
        m = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        if m.shape[1] != base.D:
            raise BasisMismatch(f"coordinate width {m.shape[1]} != basis D {base.D}")
        mats.append(m)
    return np.vstack(mats)


# ── closed-form Jiang-Conrath (ground truth for the gate) ─────────────────────
def jc_exact(s1, s2, ic) -> float:
    commons = s1.common_hypernyms(s2)
    lcs_ic = max((ic_of(c, ic) for c in commons), default=0.0)   # most-informative common subsumer
    return ic_of(s1, ic) + ic_of(s2, ic) - 2.0 * lcs_ic


# ── Step 4: the faithfulness gate ─────────────────────────────────────────────
def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    """Spearman rank correlation, or **None when it is undefined**.

    ⛔ THIS REPORTED A PERFECT 1.0 FOR A GEOMETRY THAT ENCODED NOTHING.
    `ranks()` had no tie correction: `sorted(range(n), key=lambda i: x[i])` is STABLE, so for an
    all-equal series it returned the INDEX order. Measured: `ranks([0,0,0,0]) == [0,1,2,3]`.
    Both series therefore got the identical identity permutation and the correlation came out
    **exactly 1.0**.

    That is reachable, not theoretical. `ic_of` returns 0.0 from a bare `except`, and `wn_store`
    builds IC as `float(a.get("ic") or 0.0)` — so a corpus whose `wn-*` artifacts lack the `ic`
    enrichment gives IC 0 everywhere, hence `jc_tree` 0 for every pair, hence all-zero vectors and
    an all-constant `tree_j`/`hx`. `faithfulness_check` would then report
    `spearman_vs_tree_JC: 1.0` AND `prehash_max_abs_err: 0.0` — its own docstring calls the latter
    "the exactness claim" — for a measurement that never happened. Any gate reading
    `spearman >= threshold` passed on no evidence.

    Two fixes: MIDRANKS (ties share their average rank, which is the actual definition of
    Spearman), so a constant series now correctly has zero rank-variance; and an undefined
    correlation returns **None** rather than a number. None is the house convention for "not
    measured" (`invariant_holds`, `converged`) precisely so it cannot be mistaken for a verdict —
    and it fails a threshold comparison CLOSED, which 0.0 would not do in the `>= 0` case."""
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0            # midrank: every tied member shares the group average
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    n = len(a)
    if n < 2 or n != len(b):
        return None
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    if not va or not vb:
        return None                        # a constant series has no ranking to correlate
    return cov / (va * vb)


DEFAULT_PAIRS = [
    ("dog.n.01", "wolf.n.01"), ("dog.n.01", "cat.n.01"), ("dog.n.01", "domestic_cat.n.01"),
    ("car.n.01", "truck.n.01"), ("car.n.01", "bicycle.n.01"), ("car.n.01", "boat.n.01"),
    ("dog.n.01", "car.n.01"), ("dog.n.01", "tree.n.01"), ("dog.n.01", "justice.n.01"),
    ("copper.n.01", "gold.n.01"), ("copper.n.01", "metal.n.01"), ("river.n.01", "lake.n.01"),
    ("river.n.01", "mountain.n.01"), ("happiness.n.01", "joy.n.01"), ("happiness.n.01", "anger.n.01"),
    ("happiness.n.01", "dog.n.01"), ("apple.n.01", "banana.n.01"), ("apple.n.01", "fruit.n.01"),
    ("apple.n.01", "computer.n.01"), ("king.n.01", "queen.n.01"), ("king.n.01", "monarch.n.01"),
    ("king.n.01", "peasant.n.01"), ("doctor.n.01", "nurse.n.01"), ("doctor.n.01", "lawyer.n.01"),
    ("piano.n.01", "guitar.n.01"), ("piano.n.01", "violin.n.01"), ("piano.n.01", "hammer.n.01"),
    ("water.n.01", "ice.n.01"), ("sun.n.01", "star.n.01"), ("sun.n.01", "moon.n.01"),
]


def faithfulness_check(pairs=None, D: int = _D_DEFAULT) -> Dict[str, object]:
    """Steps 1-4 end to end. Reports:
      - prehash_max_abs_err : |sparse L2^2 - tree_JC| over ALL pairs (want 0 — the exactness claim)
      - spearman_vs_tree_JC : hashed L2^2 vs the metric we realize (want ~1.0; only hash noise short of it)
      - spearman_vs_DAG_JC  : hashed L2^2 vs DAG-JC (reference; the tree/DAG gap lives here)"""
    from mantle.ontology import driver as wn
    ic = load_ic()
    pairs = list(pairs or DEFAULT_PAIRS)
    tree_j: List[float] = []
    dag_j: List[float] = []
    hx: List[float] = []
    pre_worst = 0.0
    rows = []
    # ⚠ A SILENT SKIP IS NOW A REPORTED SKIP. MEASURED 2026-07-21: this check has been quietly
    # measuring **28 of its 30 DEFAULT_PAIRS**. `metal.n.01` and `monarch.n.01` are LEMMA ALIASES —
    # nltk's `wn.synset()` resolves them (to `metallic_element.n.01` and `sovereign.n.01`
    # respectively, verified), whereas `wn_store.synset()` is a STRICT canonical-name lookup and
    # raises. The bare `except Exception: continue` below then swallowed it, and `n_pairs` — the
    # only evidence — was a number nothing forced a caller to read.
    #
    # ⚠⚠ THE CORPUS IS **NOT** MISSING THESE SYNSETS. They are present under their canonical names.
    # This is a NAME-RESOLUTION difference between two lookups, not a coverage gap, and it must not
    # be reported or acted on as one.
    #
    # ⚠ FLAGGED, DELIBERATELY NOT FIXED HERE: the real repair is for `wn_store.synset()` to resolve
    # lemma aliases the way nltk does. That is a change to the lookup semantics of the store every
    # consumer of the ontology shares — `text_to_signal`, `oov_tokens`, the reasoning layer — and
    # its blast radius is wider than this gate. It belongs to whoever owns `wn_store`, with its own
    # measurement. What changes HERE is only that the skip is no longer invisible.
    skipped: List[Tuple[str, str, str]] = []
    for a, b in pairs:
        try:
            s1, s2 = wn.synset(a), wn.synset(b)
        except Exception as e:                      # noqa: BLE001 - recorded, not swallowed
            skipped.append((a, b, f"{type(e).__name__}: {e}"))
            continue
        jt, jd = jc_tree(s1, s2, ic), jc_exact(s1, s2, ic)
        d1s, d2s = sparse_vec(s1, ic), sparse_vec(s2, ic)
        keys = set(d1s) | set(d2s)
        l2_sparse = sum((d1s.get(k, 0.0) - d2s.get(k, 0.0)) ** 2 for k in keys)
        pre_worst = max(pre_worst, abs(l2_sparse - jt))
        v1, v2 = dense_vec(s1, ic, D), dense_vec(s2, ic, D)
        jh = float(np.sum((v1 - v2) ** 2))
        tree_j.append(jt); dag_j.append(jd); hx.append(jh)
        rows.append((a, b, round(jt, 3), round(jh, 3)))
    # ⛔ A CHECK THAT EVALUATED NOTHING USED TO LOOK LIKE A PASS.
    # `except Exception: continue` above drops any pair whose synset is missing, so an empty or
    # unloadable `wn_store` drops EVERY pair. That left `hx == []`, `pre_worst == 0.0` — the field
    # the docstring calls "the exactness claim" — and a spearman of 0.0, i.e. a clean-looking
    # exactness result and a number indistinguishable from a measured "no correlation".
    # `n_pairs` was the only signal and nothing forced a caller to read it. Now the metrics are
    # explicitly None when nothing was compared, and `measured` states it outright.
    def _r(v):
        return None if v is None else round(v, 4)
    measured = len(hx) >= 2
    # `n_requested` / `n_skipped` / `skipped` make the denominator explicit. `complete` is the
    # field a gate should read: "28 of 30, and here are the two and why" is a different fact from
    # "30 of 30", and before this they were the same dict.
    return {"D": D, "n_pairs": len(hx), "measured": measured,
            "n_requested": len(pairs), "n_skipped": len(skipped),
            "complete": bool(measured and not skipped),
            "skipped": skipped,
            "prehash_max_abs_err": round(pre_worst, 9) if hx else None,
            "spearman_vs_tree_JC": _r(_spearman(tree_j, hx)),
            "spearman_vs_DAG_JC": _r(_spearman(dag_j, hx)),
            "rows": rows}


# ── Step 5 (plumbing): text -> ordered [T, D] meaning-signal for the entroptics Screen ──────────
import re

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _oov_vec(token: str, D: int) -> np.ndarray:
    """Surface fallback for tokens not in WordNet: char-trigram signed hashing into the same D.

    ⚠ DEPRECATED under LATTICE — reachable only via `text_to_signal(oov="surface")`, which is still
    the default ONLY to avoid changing retrieval behaviour ahead of the Phase 0.A A/B.

    Why it is wrong: this manufactures a SURFACE-similarity vector and deposits it in the same D as
    the meaning coordinates, where nothing downstream can tell the two apart. `distinguish` and
    `distinguished` land near each other for reasons that have nothing to do with meaning, while
    `distinguish` and `discern` do not — the precise inversion the ontology coordinate exists to
    fix. §17.2 measured char-trigram hashing at K_signal 2-3 with cross-domain cosine +0.889; it is
    the encoding the ontology coordinate REPLACED, so re-injecting it per-token re-imports the
    failure at token granularity.

    An OOV token is not a meaning-geometry problem. It is a KEYED-arm query — an exact-match lookup
    the lexical arm answers well and this arm cannot answer at all. Use `oov="skip"` and route
    `oov_tokens(text)` to that arm.

    ⚠⚠ BUT IT IS CURRENTLY LOAD-BEARING FOR THE ENTROPTICS READ — MEASURED, 2026-07-20.
    Pooled screen over four disjoint-domain documents, via `entroptics.read()`:

        oov="surface"   D=256 floor=1.51   D=512 floor=3.20   D=2048 floor=2.65   (K_signal 5-14)
        oov="skip"      D=256 floor=7.97e3 D=512 floor=1.92e8 D=2048 floor=1.13e9

    Removing the surface rows moves the noise floor by EIGHT ORDERS OF MAGNITUDE — the same
    failure signature as §17.3's named-anchor basis. The reason is structural: a JC coordinate is
    SPARSE (one nonzero per hypernym-path edge, ~8-12 of D), whereas a char-trigram row is dense
    and normalized. The trigram rows were quietly populating channels the JC rows leave empty, and
    entroptics' per-channel whitening is what notices when they stop.

    So `oov="skip"` is architecturally right and NOT YET SAFE TO DEFAULT. It needs a
    channel-liveness screen (drop unoccupied channels before whitening) on the entroptics read
    path first. That is a dependency for whoever owns that path, not something to fix by putting
    surface noise back in the coordinate.
    """
    v = np.zeros(D, dtype=np.float64)
    t = "^" + token.lower() + "$"
    for i in range(len(t) - 2):
        h, sgn = _hash("tri:" + t[i:i + 3])
        v[h % D] += sgn
    return _vec.unit(v)


def oov_tokens(text: str) -> List[str]:
    """The tokens of `text` that have NO noun synset — i.e. the ones the semantic arm cannot answer
    and the KEYED/lexical arm must.

    This exists so "OOV routes to the keyed arm" is a call a caller can actually make, rather than
    a claim in a docstring. Order-preserving and deduplicated, first occurrence wins.

    ⛔ `max_tokens: int = 400` DELETED 2026-08-01 — see `text_to_signal`. It stopped scanning after
    the 400th token, so the OOV tokens of a longer document were silently not handed to the keyed
    arm. The two functions are a PAIR (`oov="skip"` + `oov_tokens()` is the documented routing), so
    a cap on one is a hole in the other.
    """
    from mantle.ontology import driver as wn
    out: List[str] = []
    seen = set()
    for m in _WORD.finditer(text):
        tok = m.group(0).lower()
        if tok in seen:
            continue
        if not wn.synsets(tok, pos=wn.NOUN):
            seen.add(tok)
            out.append(tok)
    return out


def text_to_signal(text: str, ic=None, D: int = _D_DEFAULT,
                   center: Optional[np.ndarray] = None, oov: str = "surface") -> np.ndarray:
    """text -> [T, D]: one ORDERED row per token; each row is the IC-weighted SUPERPOSITION of its
    noun senses' coordinates (ambiguity = smear), scaled by token salience (its IC).

    NOUNS ONLY — see the module docstring. Verbs, adverbs and above all ADJECTIVES carry no
    hypernym tree, so they contribute nothing here; *"cheap lodging" does not reach "inexpensive
    accommodation"*, and that is a missing-edges problem, not a tuning problem.

    `center` — OPT-IN global mean token profile, applied to each emitted row (default None = OFF;
    see `derive_centering_mean` / `default_centering`). Note this is a translation of the row set,
    so it is a NO-OP for `feature_covariance`/`fingerprint`/`e_distance`, which already subtract a
    per-document weighted mean. It matters for the per-token rows fed to an entroptics Screen and
    for any cosine/matmul retrieval over stored coordinates — which is where §17.4's +0.505 ->
    -0.332 was measured.

    `oov` — `"surface"` (default) emits a muted char-trigram row via the DEPRECATED `_oov_vec`;
    `"skip"` emits no row, which is the LATTICE behaviour: an OOV token is a keyed-arm query, and
    `oov_tokens()` hands it over. `"skip"` is not yet the default for the same reason centering is
    not — it changes retrieval behaviour and the Phase 0.A A/B has not run — AND for a second,
    measured reason: it moves the entroptics noise floor from ~3 to ~1e8 by leaving most channels
    unoccupied. See `_oov_vec`'s docstring for the numbers and the prerequisite.

    ⛔ `max_tokens: int = 400` DELETED — 2026-08-01. It truncated the emitted frame at 400 rows,
    and **`T` (rows) is the aperture's sample count**, so it was a cap on the EVIDENCE ITSELF —
    the one thing that must never be capped. `beam/optics.py` measures precisely what that costs:
    the Weyl band is `c·(sqrt(F/T) + F/T)`, so it shrinks only as `T` grows —

        one turn         T=34, F=195   band=16.26   interval [2,195]   never certifies
        pooled 20 turns  T=466                      band= 2.13         interval [10,59]

    — and a count that cannot certify falls back to an Otsu split on a bare score column, which is
    how "what is a dog" answered with ten stacked glosses. Four hundred was not derived from `D`,
    from the corpus, or from any measured envelope; it was a number that decided how much of a
    document the instrument was allowed to see ([[no-arbitrary-caps]]).

    There is no cap now. The bound on `T` is the TEXT: a document has as many tokens as it has. A
    caller that genuinely needs to bound its input bounds ITS OWN input at the call site (as
    `scripts/lattice_arms.py` already does with `text[:4000]`), where the truncation is VISIBLE to
    whoever reads the result; a caller bounded by machine size derives that bound from the MEASURED
    resource envelope ([[associative-reach-bounded-by-envelope]]), never from a constant buried in
    the transducer. ⚠ Note the old cap was also SILENT — the frame simply stopped, with nothing in
    the returned array to say it had been cut.
    """
    if oov not in ("surface", "skip"):
        raise ValueError(f"oov must be 'surface' or 'skip', got {oov!r}")
    from mantle.ontology import driver as wn
    ic = ic or load_ic()
    c = None
    if center is not None:
        c = np.asarray(center, dtype=np.float64)
        if c.shape != (D,):
            raise ValueError(f"centering mean has shape {c.shape}, expected ({D},)")
    rows = []
    for m in _WORD.finditer(text):
        tok = m.group(0).lower()
        senses = wn.synsets(tok, pos=wn.NOUN)
        if senses:
            ws = [ic_of(s, ic) for s in senses]
            # ⛔ WAS `tot = sum(ws) or 1.0`. When every sense of a token lacks stored IC the
            # weights are all zero, and `or 1.0` turned "I cannot weight these senses" into
            # "these senses weigh nothing": every contribution became (0/1.0)·vec, the row
            # collapsed to the zero vector, `if nv:` dropped it, and THE TOKEN VANISHED FROM
            # THE FRAME WITH NO RECORD. ~43% of synsets carry IC 0, so this was not rare.
            # Unweightable is not weightless: fall back to an explicit uniform split over the
            # token's own senses, which is the honest reading of "no sense is more specific".
            tot = sum(ws)
            if tot > 0.0:
                weights = [w / tot for w in ws]
            else:
                weights = [1.0 / len(senses)] * len(senses)
            v = np.zeros(D)
            for s, w in zip(senses, weights):
                v += w * dense_vec(s, ic, D)
            nv = _vec.norm(v)
            if nv:                            # unit meaning-direction × BOUNDED salience (log IC)
                row = math.log1p(max(ws)) * _vec.unit(v)
                rows.append(row if c is None else row - c)
        elif oov == "surface":
            # ⚠⚠ SEAM — UNRESOLVED, AND DELIBERATELY NOT DERIVED HERE. 2026-08-01.
            #
            # `0.3` is the amplitude of every surface/OOV row in the frame, and it is a TYPED
            # NUMBER that appears nowhere else in this module: `_oov_vec`'s docstring above spends
            # fifteen lines on whether these rows should exist at all — measuring that removing
            # them moves the entroptics noise floor by EIGHT ORDERS OF MAGNITUDE (D=256: 1.51 →
            # 7.97e3; D=2048: 2.65 → 1.13e9) — and never once mentions the factor that sets how
            # loud they are. A number admitted to control an eight-order effect has no stated
            # basis at all.
            #
            # WHAT IS ACTUALLY KNOWN (read off this function and `ic_of`, not asserted):
            #   · `_oov_vec` returns a UNIT vector, so this row's norm IS 0.3, exactly.
            #   · an IN-VOCAB row is `log1p(max IC) · unit(v)` with Resnik IC in
            #     [0, IC_MEASURED_MAX = 14.709], so its norm lies in [0, log1p(14.709)] = [0, 2.75].
            #   · so 0.3 is at the LOW end, and it is a mute: it places every OOV token at the
            #     salience of an IC of `exp(0.3) - 1 = 0.35` — the bottom 2.4% of the IC range.
            #     That is a strong claim about unknown words, made by an unexplained literal.
            #
            # WHY IT IS NOT DERIVED HERE — the two candidate readings disagree, and picking one
            # would be fitting:
            #   (a) SALIENCE. If this is a salience like the in-vocab rows', it must be the token's
            #       information content — and an OOV token has NO measured IC. Absence is not an
            #       affirmative claim; the honest amplitude is then not 0.3 but *unstated*, which
            #       means `oov="skip"`, which `_oov_vec` already argues for and already blocks on a
            #       channel-liveness screen. And note `fired_field` weights by `log(N/df)`, under
            #       which an unseen token is MAXIMALLY informative — the exact opposite of muted.
            #   (b) CHANNEL FILLER. If, as the measured 8-order result suggests, these rows exist
            #       only to occupy channels that a sparse JC coordinate leaves empty, then the right
            #       amplitude is whatever equalises per-channel occupancy before whitening — a
            #       quantity NOBODY HAS MEASURED. Deriving it needs the channel-liveness screen
            #       that `_oov_vec` names as the prerequisite.
            #
            # Writing `ln(1/eps)` for either reading would put an invented input inside a formula
            # and call it derived. It is left AS IT STANDS, flagged in the open, and reported as
            # the one constant in this pass that could not be resolved. Resolving it is downstream
            # of the channel-liveness screen, not of this line.
            row = 0.3 * _oov_vec(tok, D)      # ⚠ UNRESOLVED CONSTANT — see the block above
            rows.append(row if c is None else row - c)
    return np.array(rows) if rows else np.zeros((0, D))


def feature_covariance(frames: np.ndarray, forgetting: float = 1.0) -> np.ndarray:
    """The forgetting-weighted feature covariance = the entroptics Pxx the Aperture accumulates:
    recent frames weighted more (forgetting < 1 = leaky memory / light-cone taper). This D×D object
    is the point-in-time STATE — WHICH concept-directions co-occur — and it's sign-stable and lives
    in the fixed concept-coordinate space, so it's directly comparable across artifacts.

    NOTE on centering: this already subtracts a per-document weighted mean below, so applying
    `text_to_signal(center=...)` upstream is provably a NO-OP for this function and for everything
    built on it (`fingerprint`, `e_distance`). Centering is not useless — it is just not useful
    HERE. It acts on the per-token rows a Screen consumes and on cosine/matmul retrieval over
    stored per-vertex coordinates, which is where §17.4's separation gain was measured. Asserted in
    `test_geometry_lattice.py::test_centering_is_a_noop_for_feature_covariance`."""
    T = frames.shape[0]
    if T == 0:
        return np.zeros((frames.shape[1], frames.shape[1]))
    w = forgetting ** np.arange(T - 1, -1, -1.0)          # newest frame weight 1, older decays
    w = w / w.sum()
    mean = (w[:, None] * frames).sum(axis=0)
    X = frames - mean
    C = (X.T * w) @ X                                     # weighted feature covariance (D×D)
    return _vec.unit(C, axis=None)                        # Frobenius-normalized (zero-safe)


def fingerprint(text: str, ic=None, D: int = _D_DEFAULT, forgetting: float = 1.0) -> np.ndarray:
    """The doc's E-state fingerprint: the normalized forgetting-weighted feature covariance of its
    streamed meaning-signal. This is the content-address for associative recall against the ontology."""
    return feature_covariance(text_to_signal(text, ic=ic, D=D), forgetting)


def e_distance(text_a: str, text_b: str, ic=None, D: int = _D_DEFAULT, forgetting: float = 1.0):
    """E-distance = Frobenius distance between the two docs' state fingerprints (0 = same meaning-
    content). Continuous, order/forgetting-aware, in concept-space — the residual MinHash can't give."""
    ic = ic or load_ic()
    fa = fingerprint(text_a, ic, D, forgetting)
    fb = fingerprint(text_b, ic, D, forgetting)
    return float(_vec.distance(fa, fb))


if __name__ == "__main__":
    print(json.dumps(faithfulness_check(), indent=2))
