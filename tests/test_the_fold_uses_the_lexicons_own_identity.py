"""The Interlingual Index folds what the lemma key cannot.

Measured 2026-08-25 on 71/home: ILI is present on 558,560 of 676,225 synset rows — every OEWN row
(120,630) and all but 6 OMW rows (437,930) — giving 117,127 groups with more than one member,
440,902 collapsible rows, and 116,944 groups spanning more than one lexicon.

It reaches what `(pos, lemma set)` cannot. An OMW row is a TRANSLATION: `wn-omw-hr-08925093-n`
names its concept in Croatian, so the lemma key never matches its English twin and 437,936
translation rows fold onto nothing. The consequence was measured at the benchmark: 120 gold titles
were carried by 508 artifacts, and scoring by id put retrieval at MRR 0.323 when scoring by concept
put it at 0.739.

Princeton still carries no ILI (117,659 rows), which is why this is a FIRST key and not a
replacement.
"""
from __future__ import annotations

from mantle.search.ranking import _fold_key, _fold_sources


def _doc(**kw):
    import json
    return json.dumps(kw)


class TestTheKey:

    def test_a_well_formed_ili_is_the_key(self):
        assert _fold_key({"ili": "i37653", "lemmas": ["dog"], "pos": "n"}) == ("ili", "i37653")

    def test_the_malformed_ili_falls_through_to_the_lemma_key(self):
        """The literal `"in"` sits on exactly 3,216 rows — one bad parse, a language code reaching
        the field. Unvalidated it folds all 3,216 into a single "concept", which is the one way this
        key can be worse than no key."""
        assert _fold_key({"ili": "in", "lemmas": ["dog"], "pos": "n"}) == ("n", ("dog",))

    def test_a_princeton_row_keeps_the_lemma_key(self):
        assert _fold_key({"lemmas": ["dog", "Canis"], "pos": "n"}) == ("n", ("canis", "dog"))

    def test_a_row_that_names_nothing_still_folds_onto_nothing(self):
        assert _fold_key({"pos": "n"}) is None


class TestTheFold:

    def test_a_translation_folds_onto_what_it_translates(self):
        """The case the lemma key cannot reach: Croatian lemmas never match English ones."""
        ranked = [("wn-oewn-08944866-n", "text/x-wordnet", -9.0),
                  ("wn-omw-hr-08925093-n", "text/x-wordnet", -3.0)]
        docs = {"wn-oewn-08944866-n": _doc(ili="i123", lemmas=["Kyoto"], pos="n",
                                           operator="op.source.oewn"),
                "wn-omw-hr-08925093-n": _doc(ili="i123", lemmas=["Kioto"], pos="n",
                                             operator="op.source.omw")}
        out, folded = _fold_sources(ranked, docs)
        assert [r[0] for r in out] == ["wn-oewn-08944866-n"]
        assert folded == {"wn-oewn-08944866-n": ["wn-omw-hr-08925093-n"]}

    def test_two_translations_from_ONE_source_still_fold(self):
        """An ILI collision is not a collision — the index exists to say these are one concept.

        The different-source guard protects the LEMMA key, which collides 7,958 times within one
        lexicon on real senses. Applying it to an identity would leave 15 translations of one
        synset sitting in one answer.
        """
        ranked = [("wn-omw-hr-1-n", "text/x-wordnet", -9.0),
                  ("wn-omw-id-1-n", "text/x-wordnet", -3.0)]
        docs = {"wn-omw-hr-1-n": _doc(ili="i7", lemmas=["Kioto"], pos="n",
                                      operator="op.source.omw"),
                "wn-omw-id-1-n": _doc(ili="i7", lemmas=["Kyoto"], pos="n",
                                      operator="op.source.omw")}
        out, folded = _fold_sources(ranked, docs)
        assert [r[0] for r in out] == ["wn-omw-hr-1-n"]
        assert folded == {"wn-omw-hr-1-n": ["wn-omw-id-1-n"]}

    def test_two_senses_from_one_lexicon_are_NOT_folded_by_the_lemma_key(self):
        """Unchanged, and load-bearing: `able.a.01` and `able.s.02` are two real senses."""
        ranked = [("wn-able.a.01", "text/x-wordnet", -9.0),
                  ("wn-able.s.02", "text/x-wordnet", -3.0)]
        docs = {"wn-able.a.01": _doc(lemmas=["able"], pos="a", operator="op.source.wordnet"),
                "wn-able.s.02": _doc(lemmas=["able"], pos="s", operator="op.source.wordnet")}
        out, folded = _fold_sources(ranked, docs)
        assert len(out) == 2 and folded == {}

    def test_the_best_scoring_row_is_the_one_kept(self):
        """Best-first order in, so the first row seen is the keeper."""
        ranked = [("keeper", "text/x-wordnet", -9.0), ("dropped", "text/x-wordnet", -1.0)]
        docs = {"keeper": _doc(ili="i9", lemmas=["a"], pos="n", operator="op.source.oewn"),
                "dropped": _doc(ili="i9", lemmas=["b"], pos="n", operator="op.source.omw")}
        out, folded = _fold_sources(ranked, docs)
        assert [r[0] for r in out] == ["keeper"]
        assert folded["keeper"] == ["dropped"]
