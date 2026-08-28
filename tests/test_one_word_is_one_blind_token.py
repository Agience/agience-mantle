"""Every spelling of one word reaches one blind token — which is the only way SSE can match at all.

## The mechanism this pins

The server holds `HMAC(key, "field:term")` and matches on byte equality of the term. It cannot
compare, stem, fold or relate anything: it has hashes. So two spellings of one word are two
unrelated postings unless the client collapses them BEFORE the HMAC, and collapsing them is the
entire job of the analysis pipeline. That is not a workaround — it is how SSE handles variants at
all, and the pipeline already did it three ways:

    lowercase             Running · running · RUNS   ->  run
    possessive stemmer    dog's · dogs · dog         ->  dog
    Porter                universities · university  ->  univers

The fold is the fourth member of the same chain, and its absence was not symmetric with the others.
`porter_stem` returns non-ASCII input untouched, so an accented word skipped the stemmer as well as
the fold: `cafes` reached `cafe` while `cafés` stayed `cafés`, so an accented word missed even its
own plural.

## What is asserted

Whole spelling families, not pairs. A pair can agree by accident; a family that collapses to exactly
one token is the property a posting list depends on. The blind token is asserted directly rather
than only the stem, because the stem is an intermediate and the token is what the server sees.
"""
from __future__ import annotations

import pytest

from mantle.search.mantle.sse.blind_tokens import blind_token
from mantle.search.mantle.sse.tokenizer import (
    ANALYZER, bigrams, fold_to_ascii, porter_stem, tokenize)

_KEY = bytes(range(32))
#: `VALID_FIELDS` is the short-code set; `c` is content.
_FIELD = "c"

#: Spelling families that name one thing. Each must collapse to exactly one term.
_FAMILIES = [
    ("cafe", ["cafe", "cafes", "Cafes", "café", "cafés", "Cafés", "CAFÉ", "café's"]),
    ("zurich", ["zurich", "Zurich", "Zürich", "ZÜRICH"]),
    ("naive", ["naive", "naïve", "Naïve"]),
    ("angstrom", ["angstrom", "Ångström", "ångström"]),
    ("resume", ["resume", "résumé", "Résumé", "resumes", "résumés"]),
    ("facade", ["facade", "façade", "facades", "façades"]),
]

#: ASCII input whose analysis must not move. The fold is the identity here, so any change in this
#: table means the fold reached further than combining marks.
_ASCII_UNCHANGED = [
    ("running", "run"), ("runs", "run"), ("dog's", "dog"), ("workers'", "worker"),
    ("universities", "univers"), ("state", "state"), ("hello", "hello"), ("x86", "x86"),
]


@pytest.mark.parametrize("name,family", _FAMILIES, ids=[n for n, _ in _FAMILIES])
def test_a_spelling_family_collapses_to_one_term(name, family):
    """One term for the whole family — asserted as a set, so a straggler is named in the failure."""
    terms = {tuple(tokenize(w)) for w in family}
    assert len(terms) == 1, (
        "%r analysed to %d different terms: %s. Each one is a separate posting list, so a query "
        "spelled one way reaches no document spelled another." % (name, len(terms), sorted(terms)))


@pytest.mark.parametrize("name,family", _FAMILIES, ids=[n for n, _ in _FAMILIES])
def test_a_spelling_family_reaches_one_blind_token(name, family):
    """The same property at the layer that matters. The stem is an intermediate; the token is what
    the server stores and compares, and it is what a recall failure is actually made of."""
    tokens = {blind_token(_KEY, _FIELD, t) for w in family for t in tokenize(w)}
    assert len(tokens) == 1, (
        "%r produced %d distinct blind tokens; the server sees them as unrelated terms." % (
            name, len(tokens)))


def test_an_accented_word_reaches_its_own_plural():
    """The asymmetry the fold removes, stated on its own.

    Without folding, `porter_stem` refuses non-ASCII input, so an accented word never reached the
    stemmer. That cost it more than its ASCII twin — it lost its own inflections too."""
    assert tokenize("café") == tokenize("cafés"), "an accented singular and plural are two terms"
    assert tokenize("café") == tokenize("cafes"), "accented and unaccented spellings are two terms"


@pytest.mark.parametrize("word,stem", _ASCII_UNCHANGED)
def test_ascii_analysis_does_not_move(word, stem):
    """Folding is the identity on ASCII, so every term that already worked is untouched. This is
    what bounds the reindex: it can only change documents that contain combining marks."""
    assert tokenize(word) == [stem]


def test_a_letter_unicode_does_not_decompose_is_left_alone():
    """`ø` and `ß` have no canonical decomposition, and neither does any non-Latin script. Folding
    them would be a guess at what English word they stand for. They index under themselves, which
    is what `porter_stem`'s non-ASCII branch has always done."""
    assert tokenize("Bjørn") == ["bjørn"]
    assert tokenize("Straße") == ["straße"]
    assert tokenize("Москва") == ["москва"]


def test_the_fold_is_unicodes_decomposition_and_not_a_table():
    """A transliteration table is a list somebody maintains and forgets, so this checks marks that
    appear in no fixture above."""
    assert fold_to_ascii("Ǎǔ Ḥḍ Ṣṭ") == "Au Hd St"
    assert fold_to_ascii("plain ascii") == "plain ascii"


def test_the_stem_invariant_bigrams_depends_on_now_holds():
    """`bigrams` joins stems with a space and states that no stem can contain one because stems are
    alphabetic after the pipeline. An unfolded token broke the spirit of that: it was not the
    alphabet the rest of the pipeline assumed. Folded, the claim is true of Latin input."""
    stems = tokenize("café société moderne")
    assert all(s.isascii() and s.isalpha() for s in stems), \
        "a stem left the pipeline outside the alphabet bigrams assumes: %s" % stems
    assert bigrams(stems) == ["cafe societ", "societ modern"]


def test_the_analyzer_generation_is_declared():
    """The analysis is part of the index format: the server holds HMACs and cannot re-derive a term
    it has never seen, so a store written by one generation cannot be queried by another.

    `ANALYZER` exists so that mismatch is a fact a caller can read instead of an unexplained recall
    failure. It is asserted rather than merely present, because a constant nobody checks drifts out
    of step with the pipeline it names — bump it in the same commit as any stage change, and
    rebuild with `search.init_search.reindex_all_artifacts`.
    """
    assert ANALYZER == 2, (
        "the analyzer generation moved. If a pipeline stage changed, that is correct and every "
        "existing MANTLE-SSE index must be rebuilt; update this assertion in the same commit.")
    assert porter_stem("café") == "café", (
        "porter_stem started folding. Folding is stage 2.5 and stemming is stage 4 — keeping them "
        "apart is what lets `test_the_two_porter_stemmers_agree` compare stemmers to stemmers.")


def test_the_comparison_would_notice_the_fold_being_removed():
    """Negative control. Re-running the pipeline without stage 2.5 must break the families above,
    or these tests would pass on an unfolded tokenizer and prove nothing."""
    def unfolded(text):
        from mantle.search.mantle.sse.tokenizer import split_words, strip_possessive
        out = []
        for raw in split_words(text):
            t = strip_possessive(raw.lower())
            if t:
                s = porter_stem(t)
                if s:
                    out.append(s)
        return out

    assert unfolded("café") != unfolded("cafe"), \
        "the pipeline collapses these even without the fold, so the fold is not what is being tested"
    assert unfolded("café") != unfolded("cafés"), (
        "an unfolded accented word already reached its own plural, so the asymmetry described in "
        "this module's header does not exist")
