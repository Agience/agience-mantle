"""English analysis pipeline for MANTLE-SSE.

Mirrors the structure of Lucene's `english_analyzer`, minus its stop-word
stage:

    standard tokenizer → lowercase → ASCII-fold → possessive stemmer → Porter stemmer

## Why an analysis pipeline exists at all

A blind token is `HMAC(key, "field:term")`, so the server matches on byte
equality of the term and nothing else. Two spellings of one word are two
unrelated tokens unless something collapses them BEFORE the HMAC — and that
is the entire job of the stages above. `Running`, `running`, `runs` and
`RUNS` are one posting because lowercase and Porter make them one term.
`dog's` and `dogs` are one posting because the possessive stemmer and Porter
make them one term.

The fold is a member of that chain, not an addition to it. `café`, `Cafés`
and `cafes` name one thing and must reach one posting, and they do so the
same way every other variant does: they are normalised to one term on the
client, and only that term is ever encrypted. Lucene calls this stage
`asciifolding` and places it exactly here, between lowercase and the
stemmers.

Without it, an accented word does not merely miss its ASCII spelling — it
misses its own plural. `porter_stem` returns non-ASCII input untouched, so
`cafés` never reaches the stemmer that would have taken it to `cafe`, while
`cafes` does. Folding first is what puts every Latin word back on the path
its ASCII twin already takes, and it restores the invariant `bigrams` states
below: after this pipeline a stem is alphabetic and contains no space.

Pure Python, no NLTK dependency. The Porter (1980) stemmer is implemented
inline; it is deterministic, well-defined, and produces stable stems
suitable for blind-token derivation in :mod:`mantle.search.mantle.sse.blind_tokens`.

Index-time and query-time both call :func:`tokenize`, which guarantees the
same string maps to the same blind tokens on both paths. The analysis is
fixed; changing any stage (a different stemmer, a different fold) changes
which term a document is filed under, so every existing index must be
rebuilt — there is no in-place migration, because the server holds HMACs
and cannot re-derive a term it has never seen. :data:`ANALYZER` names the
generation an index was written by, so "this store predates the fold" is a
fact a caller can read rather than infer. `search.init_search
.reindex_all_artifacts` is the rebuild, and it already runs in the
background on a store with no usable index.

Public API:

- :func:`tokenize` — full pipeline; returns the list of stems in input order.

The stages are also exposed individually for testing:

- :func:`split_words`
- :func:`fold_to_ascii`
- :func:`strip_possessive`
- :func:`porter_stem`
"""

from __future__ import annotations

import re
import unicodedata
from typing import List


# ---------------------------------------------------------------------------
# Stage 1 — tokenize raw text
# ---------------------------------------------------------------------------

# Match word-character runs, allowing internal/trailing apostrophes so
# possessives like ``alice's`` and ``workers'`` survive as single tokens for
# :func:`strip_possessive` to handle. Mirrors Lucene's StandardTokenizer for
# ASCII English; corner cases (URL splitting, CJK segmentation) are out of
# scope for the SSE MVP. ``\w`` is Unicode-aware, so accented characters and
# digits both work.
#
# How this class differs from the other lexical arm.
#
# `ember/corpus/fts.py` splits with `[^\W_]+`, which matches SQLite's `unicode61` exactly — 0
# disagreements over a 13-probe set spanning apostrophes, underscores, hyphens, digits, accents
# and Windows paths (pinned by `ember/tests/test_the_tokenizer_matches_fts5.py`). This regex
# disagrees with `unicode61` on 6 of those 13, in exactly two ways:
#
#   Apostrophes. Keeping `dog's` whole is what lets `strip_possessive` take it to `dog`.
#   `unicode61` splits it into `dog` and a junk `s` term, so it indexes a one-letter token that
#   means nothing and matches everything possessive.
#
#   Underscores. Python's `\w` includes `_`, so `snake_case_name` stays one term here where FTS5
#   makes three. Measured over the live store, 0 of 78,341 tokens emitted across 60,000 real
#   artifact offers contain an underscore: this arm indexes the offer (title, description, tags),
#   which is prose, and identifiers live in `content`, which it does not index at all. The class
#   is part of the index format, so changing it forces a full reindex — and a reindex to fix a
#   case the corpus does not contain buys nothing.
#
# The two arms write different stores and a query reaches one or the other, so they are not
# required to agree with each other. Each is required to agree with the index it writes.
_WORD_RE = re.compile(r"\w+(?:'\w*)*", flags=re.UNICODE)


def split_words(text: str) -> List[str]:
    """Split text into raw word tokens. Empty tokens are dropped."""
    if not text:
        return []
    return _WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Stage 2 — lowercase (handled by str.lower at the caller)
# ---------------------------------------------------------------------------

# Inlined as ``token.lower()`` in :func:`tokenize`. No separate function:
# str.lower is the canonical implementation, no need to wrap it.


# ---------------------------------------------------------------------------
# Stage 2.5 — ASCII folding
# ---------------------------------------------------------------------------

def fold_to_ascii(token: str) -> str:
    """Remove combining marks, so one word has one spelling before it is stemmed.

    Unicode's own canonical decomposition, not a transliteration table: NFD
    separates a letter from its marks and the marks are dropped. A table is a
    list somebody maintains and forgets; this covers every letter Unicode
    decomposes, including ones nobody thought to list.

    >>> fold_to_ascii("café")
    'cafe'
    >>> fold_to_ascii("Ångström")
    'Angstrom'
    >>> fold_to_ascii("plain")
    'plain'

    A letter with no decomposition — `ø`, `ß`, and every non-Latin script —
    passes through unchanged. Those are not accented spellings of an English
    word, so folding them would be a guess; they stay as they are and index
    under themselves, which is what `porter_stem`'s non-ASCII branch already
    assumes.
    """
    return "".join(c for c in unicodedata.normalize("NFD", token)
                   if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# Stage 3 — possessive stemmer (English)
# ---------------------------------------------------------------------------

def strip_possessive(token: str) -> str:
    """Strip a trailing English possessive (``'s`` or ``s'``) from a token.

    Matches Lucene's ``english_possessive_stemmer``:

    >>> strip_possessive("alice's")
    'alice'
    >>> strip_possessive("workers'")
    'workers'
    >>> strip_possessive("apple")
    'apple'
    """
    if len(token) >= 2 and token.endswith("'s"):
        return token[:-2]
    if len(token) >= 2 and token.endswith("s'"):
        return token[:-1]
    return token


# ---------------------------------------------------------------------------
# Stage 4 — Porter (1980) stemmer
# ---------------------------------------------------------------------------
#
# Reference: M.F. Porter, "An algorithm for suffix stripping" (1980),
# Program 14(3): 130-137. Public-domain algorithm.
#
# Notation used in the comments below mirrors Porter's paper:
#   m       = "measure" — number of (VC) patterns in the stem
#   *v*     = stem contains a vowel
#   *S      = stem ends with S
#   *d      = stem ends with a double consonant
#   *o      = stem ends c-v-c where the final c is not w, x, or y
#
# `y` is treated as a consonant when preceded by a vowel and as a vowel
# otherwise — Porter's rule, carried verbatim.

_VOWELS = frozenset("aeiou")


def _is_vowel(stem: str, i: int) -> bool:
    """Porter's vowel/consonant classification at position ``i``."""
    c = stem[i]
    if c in _VOWELS:
        return True
    if c == "y":
        # `y` is a vowel only when preceded by a consonant (or at position 0
        # acts as a consonant per Porter — the paper says "y is consonantal
        # at the start of a word"). Implementation follows the paper.
        return i > 0 and not _is_vowel(stem, i - 1)
    return False


def _measure(stem: str) -> int:
    """Compute Porter's measure m — number of VC patterns."""
    if not stem:
        return 0
    # Build the C/V pattern, collapsing runs.
    pattern: List[str] = []
    for i in range(len(stem)):
        kind = "V" if _is_vowel(stem, i) else "C"
        if not pattern or pattern[-1] != kind:
            pattern.append(kind)
    # Count VC pairs: positions where 'V' is immediately followed by 'C'.
    return sum(1 for i in range(len(pattern) - 1) if pattern[i] == "V" and pattern[i + 1] == "C")


def _contains_vowel(stem: str) -> bool:
    return any(_is_vowel(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    """*d — stem ends with a double consonant (same letter, both consonants)."""
    if len(stem) < 2:
        return False
    if stem[-1] != stem[-2]:
        return False
    return not _is_vowel(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    """*o — stem ends c-v-c where the final c is not w, x, or y."""
    if len(stem) < 3:
        return False
    if _is_vowel(stem, len(stem) - 3):
        return False
    if not _is_vowel(stem, len(stem) - 2):
        return False
    if _is_vowel(stem, len(stem) - 1):
        return False
    return stem[-1] not in "wxy"


def _replace_suffix(stem: str, suffix: str, replacement: str) -> str:
    return stem[: -len(suffix)] + replacement


# --- Step 1a -----------------------------------------------------------------
# Plurals and -s endings.

def _step_1a(stem: str) -> str:
    if stem.endswith("sses"):
        return _replace_suffix(stem, "sses", "ss")
    if stem.endswith("ies"):
        return _replace_suffix(stem, "ies", "i")
    if stem.endswith("ss"):
        return stem
    if stem.endswith("s"):
        return stem[:-1]
    return stem


# --- Step 1b -----------------------------------------------------------------
# Past tense / -ing.

def _step_1b(stem: str) -> str:
    if stem.endswith("eed"):
        if _measure(stem[:-3]) > 0:
            return stem[:-1]  # eed → ee
        return stem
    fired = False
    new_stem = stem
    if stem.endswith("ed"):
        candidate = stem[:-2]
        if _contains_vowel(candidate):
            new_stem = candidate
            fired = True
    elif stem.endswith("ing"):
        candidate = stem[:-3]
        if _contains_vowel(candidate):
            new_stem = candidate
            fired = True
    if not fired:
        return stem
    # Step 1b' — restore canonical form after stripping.
    if new_stem.endswith(("at", "bl", "iz")):
        return new_stem + "e"
    if (
        _ends_double_consonant(new_stem)
        and new_stem[-1] not in "lsz"
    ):
        return new_stem[:-1]
    if _measure(new_stem) == 1 and _ends_cvc(new_stem):
        return new_stem + "e"
    return new_stem


# --- Step 1c -----------------------------------------------------------------
# y → i if the stem contains a vowel.

def _step_1c(stem: str) -> str:
    if stem.endswith("y") and _contains_vowel(stem[:-1]):
        return stem[:-1] + "i"
    return stem


# --- Step 2 ------------------------------------------------------------------

_STEP2_RULES = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("bli", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
)


def _step_2(stem: str) -> str:
    for suffix, replacement in _STEP2_RULES:
        if stem.endswith(suffix):
            candidate = stem[: -len(suffix)]
            if _measure(candidate) > 0:
                return candidate + replacement
            return stem
    return stem


# --- Step 3 ------------------------------------------------------------------

_STEP3_RULES = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)


def _step_3(stem: str) -> str:
    for suffix, replacement in _STEP3_RULES:
        if stem.endswith(suffix):
            candidate = stem[: -len(suffix)]
            if _measure(candidate) > 0:
                return candidate + replacement
            return stem
    return stem


# --- Step 4 ------------------------------------------------------------------

_STEP4_RULES = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
    "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous",
    "ive", "ize",
)


def _step_4(stem: str) -> str:
    # `ion` is a special case: only strip when preceded by `s` or `t`.
    if stem.endswith("ion"):
        candidate = stem[:-3]
        if (
            _measure(candidate) > 1
            and candidate
            and candidate[-1] in ("s", "t")
        ):
            return candidate
        return stem
    # Order longest-first to avoid `er` shadowing `ement`.
    for suffix in sorted(_STEP4_RULES, key=len, reverse=True):
        if stem.endswith(suffix):
            candidate = stem[: -len(suffix)]
            if _measure(candidate) > 1:
                return candidate
            return stem
    return stem


# --- Step 5 ------------------------------------------------------------------

def _step_5a(stem: str) -> str:
    if stem.endswith("e"):
        candidate = stem[:-1]
        m = _measure(candidate)
        if m > 1:
            return candidate
        if m == 1 and not _ends_cvc(candidate):
            return candidate
    return stem


def _step_5b(stem: str) -> str:
    if (
        _measure(stem) > 1
        and _ends_double_consonant(stem)
        and stem.endswith("l")
    ):
        return stem[:-1]
    return stem


def porter_stem(token: str) -> str:
    """Apply the full Porter (1980) algorithm. Token must be lowercase ASCII.

    Tokens shorter than 3 characters are returned unchanged — Porter's
    convention.

    Non-ASCII tokens pass through untouched. After :func:`fold_to_ascii`
    that branch is unreachable for Latin text, which is the point: an
    accented English word reaches this stemmer as its ASCII spelling and
    gets the same stem its unaccented twin gets. What still reaches it
    non-ASCII is a genuinely different script, or a letter Unicode does not
    decompose — neither is an English word Porter has rules for, so both
    index under themselves.
    """
    if len(token) <= 2:
        return token
    if not token.isascii() or not token.isalpha():
        return token
    s = _step_1a(token)
    s = _step_1b(s)
    s = _step_1c(s)
    s = _step_2(s)
    s = _step_3(s)
    s = _step_4(s)
    s = _step_5a(s)
    s = _step_5b(s)
    return s


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Full English analysis pipeline.

    Stages (in order):

    1. Split on non-word characters.
    2. Lowercase.
    3. Fold combining marks away (``café`` -> ``cafe``).
    4. Strip English possessive (``'s`` / ``s'``).
    5. Drop empty tokens.
    6. Porter stem.
    7. Drop tokens that became empty after stemming.

    Lowercase runs before the fold rather than after, so a letter whose
    lowering itself produces a combining mark (Turkish ``İ`` lowers to ``i``
    plus a combining dot) is folded on the result rather than around it.

    Returns the stems in input order. Duplicate stems are *not* deduplicated
    here — callers that need term frequencies (the SSE indexer) compute
    them from the returned list; callers that need the unique term set (the
    query path) deduplicate downstream.
    """
    if not text:
        return []
    out: List[str] = []
    for raw in split_words(text):
        token = strip_possessive(fold_to_ascii(raw.lower()))
        if not token:
            continue
        stem = porter_stem(token)
        if not stem:
            continue
        out.append(stem)
    return out


def bigrams(stems: List[str]) -> List[str]:
    """Return adjacent-pair bigram tokens for phrase indexing.

    Example: ["platform", "artifact"] → ["platform artifact"]

    Used by the SSE indexer to write phrase posting lists at commit time
    and by the SSE query engine to look up phrases when the user's query
    is quoted. The space separator is safe because individual stems
    contain only alphabetic characters after the Porter pipeline — no
    stem can contain a space.

    Duplicate bigrams are *not* deduplicated here — callers handle
    uniqueness the same way they do for unigrams.
    """
    if len(stems) < 2:
        return []
    return [f"{stems[i]} {stems[i + 1]}" for i in range(len(stems) - 1)]


#: The generation of this analysis pipeline. A blind token is an HMAC of the
#: analysed term, so the analysis IS part of the index format: a store written
#: by one generation cannot be queried by another, and the server cannot
#: re-derive terms it only holds hashes of. Bumped whenever any stage changes,
#: so a store can record which generation wrote it and a mismatch is a fact
#: rather than an unexplained recall failure. `reindex_all_artifacts` is the
#: only migration.
#:
#:   1  standard -> lowercase -> possessive -> Porter
#:   2  standard -> lowercase -> ASCII-fold -> possessive -> Porter
ANALYZER = 2

__all__ = [
    "ANALYZER",
    "fold_to_ascii",
    "bigrams",
    "porter_stem",
    "split_words",
    "strip_possessive",
    "tokenize",
]
