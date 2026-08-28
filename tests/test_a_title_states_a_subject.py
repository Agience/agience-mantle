r"""A prose artifact anchors on its title, and a title is not made of alike words (§119).

`_salient_title` is not called. These tests pin its behaviour because the function is kept: it
prices at +2 on by-title while every OEWN distance reads 0, and -3 with the metric working.
Trimming a title's generic anchors helps only when
the anchors cannot be told apart by distance; when they can, the generic ones are already far and
dropping them removes real reach.

§110 keeps a document's body positions within one correlation length of a TITLE anchor. Every
anchor was treated alike, and measured on the live store that decided `prism protocol`:

    README - Read in this order        3 anchors, one of them `inch.n.01` from the word `in`
                                       124 body positions admitted
    prism-protocol - prism-protocol    2 anchors, both the subject
                                        12 body positions admitted

Per-position reach was equal to within 3% and the standardisation is size-aware by design (§118),
so README won on size alone. A generic anchor is genuinely near much of the taxonomy — `jc_tree`
already encodes information content — so the radius was never the problem. The problem is that a
word stating nothing became an anchor.

`salient_terms` already answers which words carry a piece of text, and this path already applies it
to the QUERY. A title is a piece of text.
"""
from __future__ import annotations

from unittest.mock import patch

import mantle.search.ranking as R


class _Matcher:
    """Stands in for the `match` seam. `salient_terms` keeps what carries the text's own mean
    information; here the corpus is stated outright so the rule is the only thing under test."""

    def __init__(self, salient):
        self._salient = set(salient)

    def salient_terms(self, stems, store=None):
        return [t for t in stems if t in self._salient]


class TestTheGenericWordsGo:

    def test_a_title_of_mostly_frame_words_keeps_only_its_subject(self):
        out = R._salient_title({"title": "README — Read in this order"}, None,
                               _Matcher({"readm", "read"}))
        assert out == "README Read", out
        assert "order" not in out and "in" not in out.split()

    def test_a_title_that_is_all_subject_is_untouched(self):
        title = "prism-protocol — prism protocol spec"
        out = R._salient_title({"title": title}, None,
                               _Matcher({"prism", "protocol", "spec"}))
        assert "prism" in out and "protocol" in out and "spec" in out


class TestItDeclinesWhenItCannotTell:

    def test_a_short_title_is_all_subject(self):
        """Under three stems there is nothing to separate — the same identifiability rule §103
        applies to a two-term query."""
        title = "prism-protocol"
        assert R._salient_title({"title": title}, None, _Matcher(set())) == title

    def test_an_unmeasurable_corpus_anchors_on_the_whole_title(self):
        class _Broken:
            def salient_terms(self, stems, store=None):
                raise RuntimeError("no corpus")
        title = "README — Read in this order"
        assert R._salient_title({"title": title}, None, _Broken()) == title

    def test_a_measure_that_keeps_nothing_anchors_on_the_whole_title(self):
        """Narrowing a title to nothing would leave the document with no anchor at all, and §110
        then returns every position — the opposite of what this is for."""
        title = "README — Read in this order"
        assert R._salient_title({"title": title}, None, _Matcher(set())) == title

    def test_an_empty_title_is_returned_as_is(self):
        assert R._salient_title({"title": ""}, None, _Matcher({"x"})) == ""


class TestTheSeamIsPassedNotLookedUp:
    """`_match` is a local inside `_reach_rank`, resolved from the seam per call, rather than a
    module global. Referred to by name it raises `NameError` into the function's own `except` and
    returns every title unchanged in every process. by-title does not move by one question, which
    is the only reason it was noticed."""

    def test_the_module_has_no_match_global_to_fall_back_on(self):
        assert not hasattr(R, "_match"), (
            "`_match` became a module global; the guard below no longer proves anything and "
            "`_salient_title` could silently look it up again")

    def test_the_matcher_argument_is_actually_used(self):
        seen = {}

        class _Recording(_Matcher):
            def salient_terms(self, stems, store=None):
                seen["called"] = True
                return super().salient_terms(stems, store)

        R._salient_title({"title": "README — Read in this order"}, None,
                         _Recording({"readm", "read"}))
        assert seen.get("called"), "the passed seam was never consulted"
