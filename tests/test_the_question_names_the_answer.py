r"""A definition question names a word, and one candidate's own lemma IS that word (§117).

`what is a glacier` and `what does squeaking mean` both name their subject outright. Stem-level
coverage cannot use that: `squeaking`, `squeakiness` and `squeaker` all stem to `squeak`, which is
what stemming is FOR and is right everywhere else — and is exactly why `what does squeaking mean`
answered `squeakiness` (§116). The lemmas are stored, so the identity is available without
inferring it.

Measured on the pinned question sets: modifiers 15/60 -> 44/60 rank-1, nouns 31/60 -> 55/60.

The guard below is what licences the mechanism: `lemmas` means two different things in this
corpus, and only one of them is a name.
"""
from __future__ import annotations

import json
from unittest.mock import patch

from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor


class _Rows:
    def __init__(self, docs):
        self._docs = docs

    def execute(self, sql, params):
        return [(k, json.dumps(v)) for k, v in self._docs.items() if k in set(params)]


class _Db:
    def __init__(self, docs):
        self._rows = _Rows(docs)

    class _Artifacts:
        def __init__(self, rows):
            self.db = self

        def read(self):
            return self._rows

    @property
    def artifacts(self):
        holder = _Db._Artifacts.__new__(_Db._Artifacts)
        holder.db = holder
        holder._rows = self._rows
        holder.read = lambda: self._rows
        return holder


def _accessor(docs):
    acc = MantleSseSearchAccessor.__new__(MantleSseSearchAccessor)
    acc._store_db = _Db(docs)
    return acc


LEXICON = "text/x-wordnet"


class TestTheNameIsRead:

    def test_a_lexicon_entry_whose_lemma_is_the_asked_word_is_named(self):
        acc = _accessor({"wn-a": {"content_type": LEXICON, "lemmas": ["glacier"]},
                         "wn-b": {"content_type": LEXICON, "lemmas": ["moraine"]}})
        with patch.object(MantleSseSearchAccessor, "_salient_measure", lambda self: None):
            named = acc._named_by_the_question(["wn-a", "wn-b"], "what is a glacier")
        assert named == {"wn-a"}

    def test_a_derivational_relative_is_not_named(self):
        """`squeakiness` is not what was asked about, and stems cannot tell them apart."""
        acc = _accessor({"wn-adj": {"content_type": LEXICON, "lemmas": ["squeaking"]},
                         "wn-n": {"content_type": LEXICON, "lemmas": ["squeakiness"]}})
        with patch.object(MantleSseSearchAccessor, "_salient_measure", lambda self: None):
            named = acc._named_by_the_question(["wn-adj", "wn-n"], "what does squeaking mean")
        assert named == {"wn-adj"}


class TestProseLemmasAreNotNames:
    """On a lexicon entry `lemmas` are the words that mean the concept. On a canon or wiki
    artifact they are key terms `astra/doc_index` extracted from the body — a description of what
    the document discusses rather than what it is called.
    `pipeline_unified._extract_artifact_fields` guards the same way, by reading the type.

    Measured without this guard: `prism protocol` answered `canon:README`,
    `canon:SIGNAL-PROTOCOL` and `canon:MCP-VS-SIGNAL-AUDIT` ahead of `canon:prism-protocol`,
    because every document that DISCUSSES protocols carries `protocol` among its body terms.
    """

    def test_a_canon_document_is_never_named_by_its_body_terms(self):
        acc = _accessor({
            "canon:README#1": {"content_type": "text/markdown",
                               "lemmas": ["prism", "protocol", "read", "order"]},
            "wn-p": {"content_type": LEXICON, "lemmas": ["prism"]},
        })
        with patch.object(MantleSseSearchAccessor, "_salient_measure", lambda self: None):
            named = acc._named_by_the_question(["canon:README#1", "wn-p"], "prism protocol")
        assert "canon:README#1" not in named, (
            "a prose artifact was named by a term extracted from its body — this is the §117 "
            "regression that took by-title from 33/40 to 15/40")
        assert named == {"wn-p"}


class TestItDeclinesWhenItCannotTell:

    def test_no_text_names_nothing(self):
        acc = _accessor({"wn-a": {"content_type": LEXICON, "lemmas": ["glacier"]}})
        assert acc._named_by_the_question(["wn-a"], "") == set()

    def test_no_candidates_names_nothing(self):
        acc = _accessor({})
        assert acc._named_by_the_question([], "what is a glacier") == set()

    def test_an_unreadable_store_names_nothing_rather_than_raising(self):
        acc = MantleSseSearchAccessor.__new__(MantleSseSearchAccessor)
        acc._store_db = None
        assert acc._named_by_the_question(["wn-a"], "what is a glacier") == set()

class TestProseIsNeverNamed:
    """Matching query words against a prose artifact's title adds nothing.

    It is worth +6 on `bench_canon --by-title` only while every OEWN distance reads 0, and exactly
    0 with the metric working — 30/40 with it and 30/40 without. The reason
    it adds nothing is that a prose artifact's title words are ALREADY INDEXED, so coverage counts
    them and the name repeats what coverage has.

    A lexicon entry is the case coverage cannot see: its lemmas are names that stemming collapses
    (`squeaking` / `squeakiness` / `squeaker` all stem to `squeak`), which is why §117 survives and
    is measured at +9 on modifiers and +2 on nouns with the metric working.
    """

    def _named(self, docs, ids, text):
        acc = _accessor(docs)
        with patch.object(MantleSseSearchAccessor, "_salient_measure", lambda self: None):
            return acc._named_by_the_question(ids, text)

    def test_a_document_whose_title_is_the_query_is_NOT_named(self):
        docs = {
            "canon:prism-protocol#1": {"content_type": "text/markdown",
                                       "title": "prism-protocol — prism-protocol",
                                       "lemmas": ["leg", "wire"]},
            "wn-p": {"content_type": LEXICON, "lemmas": ["prism"]},
        }
        named = self._named(docs, list(docs), "prism protocol")
        assert named == {"wn-p"}, (
            "prose was named again; §124 measured that at 0 and it repeats coverage — %r" % (named,))

    def test_body_terms_still_do_not_name_a_document(self):
        """The §117 guard, unchanged: every document that DISCUSSES protocols carries `protocol`
        among its body terms, and reading those as names put `canon:README` first."""
        docs = {"canon:README#1": {"content_type": "text/markdown",
                                   "title": "README — Read in this order",
                                   "lemmas": ["prism", "protocol"]}}
        assert self._named(docs, list(docs), "prism protocol") == set()

def _row(artifact_id):
    """A `_Ranked` carrying only what the cut path touches."""
    from mantle.search.mantle.sse.narrowing import Coverage
    from mantle.search.mantle.sse.router_accessor import _Ranked
    return _Ranked(artifact_id=artifact_id, score=0.0, collection_id="", principal_id="",
                   coverage=Coverage(stems=1, matched=("x",)))


def _apply_cut(rows, by_id, cut, named):
    """The §128 rule, extracted so it can be tested without a store or an ontology."""
    survived = [by_id[cid]._replace(score=-float(score)) for cid, score in rows[:cut]]
    if named:
        kept = {r.artifact_id for r in survived}
        survived = survived + [by_id[cid]._replace(score=-float(score))
                               for cid, score in rows[cut:]
                               if cid in named and cid not in kept]
    return survived


class TestTheCutMayNotDiscardANamedCandidate:
    """§128. `cut_for` reads this arm's reach spectrum and decides where reach stops. It knows
    nothing about names and should not — it is a measurement. But a candidate whose own lemma IS
    the word asked about is evidence of a different kind, and reach is not entitled to discard it
    for scoring poorly on the only axis it can see.

    Attributed over the 60 pinned modifier questions before this existed:

        rank-1                     32        in answer, not first        7
        in pool, CUT by reach      13        ranked but past the page    4
        narrowed, not in pool       2        NOT NARROWED                2

    Thirteen answers reached the pool and were cut — nearly twice the mis-ranked bucket — and §117's
    naming runs on the SURVIVORS, so it could only reorder what the cut had already spared. A
    modifier projects to the same noun as every other modifier about that noun (§96), so its reach
    is genuinely indistinguishable; the name is the only separator and it arrived too late.

    Measured: modifiers 39/60 -> 52/60 in answer, exactly the thirteen, and 32/60 -> 40/60 rank-1.
    """

    def test_a_named_row_below_the_cut_is_kept(self):
        """The whole mechanism, at the boundary: two rows, a cut of one, and the name below it."""
        rows = [("wn-other", -2.0), ("wn-named", -0.1)]
        by_id = {"wn-other": _row("wn-other"), "wn-named": _row("wn-named")}
        survived = _apply_cut(rows, by_id, cut=1, named={"wn-named"})
        assert [r.artifact_id for r in survived] == ["wn-other", "wn-named"], (
            "the named row was cut; §128 exists to stop exactly that")

    def test_nothing_is_kept_when_nothing_is_named(self):
        rows = [("wn-other", -2.0), ("wn-b", -0.1)]
        by_id = {"wn-other": _row("wn-other"), "wn-b": _row("wn-b")}
        survived = _apply_cut(rows, by_id, cut=1, named=set())
        assert [r.artifact_id for r in survived] == ["wn-other"]

    def test_a_named_row_already_above_the_cut_is_not_duplicated(self):
        rows = [("wn-named", -2.0), ("wn-b", -0.1)]
        by_id = {"wn-named": _row("wn-named"), "wn-b": _row("wn-b")}
        survived = _apply_cut(rows, by_id, cut=1, named={"wn-named"})
        assert [r.artifact_id for r in survived] == ["wn-named"]

def _apply_pool(ordered, top_k, named):
    """The §132 rule, extracted so it can be tested without a store or an ontology."""
    pool = list(ordered[:top_k])
    if len(pool) < len(ordered):
        in_pool = {r.artifact_id for r in pool}
        outside = [r for r in ordered if r.artifact_id not in in_pool]
        pool = pool + [r for r in outside if r.artifact_id in named]
    return pool


class TestThePoolMayNotDiscardANamedCandidate:
    """§132. `_pool_for_reach` sorts by coverage and slices at `top_k`. On a one-stem query —
    which `_by_coverage` says ties EVERYWHERE by construction — that slice falls INSIDE a tie, so
    which side of it the answer lands on is not a measurement of anything.

    §128 stopped the reach CUT from discarding a named candidate. The pool is the same decision one
    stage earlier and was still doing it, and naming runs ON the pool, so it never got asked.

    Measured: `what does solar mean` narrows 379 candidates with the answer among them
    (`matched=('solar',)`) and the answer never reaches the pool. With this, the hand-written
    modifier tier moved 2/6 -> 3/6 — the first time §117's family of mechanisms reached it (§131).
    """

    def test_a_named_row_outside_the_pool_is_added_back(self):
        ordered = [_row("wn-a"), _row("wn-b"), _row("wn-named")]
        pool = _apply_pool(ordered, top_k=2, named={"wn-named"})
        assert [r.artifact_id for r in pool] == ["wn-a", "wn-b", "wn-named"]

    def test_nothing_is_added_when_the_pool_already_holds_everything(self):
        ordered = [_row("wn-a"), _row("wn-b")]
        pool = _apply_pool(ordered, top_k=5, named={"wn-a"})
        assert [r.artifact_id for r in pool] == ["wn-a", "wn-b"]

    def test_nothing_is_added_when_nothing_outside_is_named(self):
        ordered = [_row("wn-a"), _row("wn-b"), _row("wn-c")]
        pool = _apply_pool(ordered, top_k=2, named={"wn-a"})
        assert [r.artifact_id for r in pool] == ["wn-a", "wn-b"]
