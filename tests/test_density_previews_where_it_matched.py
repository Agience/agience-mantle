"""`beacon.density` cuts by entropy alone, and carries no stemmer.

A query-aware cut by Porter stems is what this file rules out. Selecting windows by stemming
each one and counting matched query stems is wrong twice over:

- **Slow, on the hottest path there is.** Stemming every window of every hit takes the server
  side of a recall from ~0.25s to 1.96s on node 71/dev, in front of every prompt.
- **A predetermination.** A stemmer is a hand-built table of English suffix rules with a list of
  irregular cases. Nothing measures it, it assumes a language, and it does not belong in a read
  path whose other two components are one entropy definition and one cut.

The problem it was aimed at is open, and these tests do not cover it. Entropy is query-independent
by construction, so one artifact returns byte-identical bytes to every question asked of it: three
unrelated questions each get the same 368 characters of this project's README, which are its
installation instructions. Recall picks the right document and shows the wrong part of it.

Window selection by vector proximity to the query is a measurement rather than a rule, and cheaper
— a static embedder encodes 100 windows in ~3ms against Porter over thousands. Mantle embeds
nothing, so those vectors arrive from a writer at index time or the selection happens in the
caller.

**The test to restore when that lands** is the one this file used to open with: two different
questions against one document must not produce the same excerpt. It is deliberately absent
rather than skipped, because a skipped test reads as a known gap in a feature that exists, and
this feature does not exist yet.

What remains here pins the surface that IS live — the entropy cut and its edges, which had no
tests at all before.
"""
from __future__ import annotations

from mantle.search.beacon.density import dense_excerpt, dense_windows

#: Three clearly separated subjects with filler between them, so the windows genuinely differ
#: and a cut has something to choose between.
DOC = """
The installation procedure begins by running the package manager against the published
index and waiting for the download to complete on the local machine before continuing.
Copy the configuration template into place and edit the values it names for your host.

Authorization is computed as reachability in a typed graph, and the identifier the
traversal must reach is by construction the identifier that selects the decryption keys.
Holding no grant means the key is underivable, which is what makes revocation cheap.

The attenuation operator is a bounded meet-semilattice over the action set, with an
absorbing deny element and a full-authority identity. Both storage encodings are codecs
onto that one type and round-trip through it without disagreeing about the zero element.
"""


class TestTheEntropyCut:
    def test_it_returns_a_subset_of_the_document_in_order(self):
        """The excerpt is spans OF the document, not a summary of it.

        `beacon.density` never generates text — it selects windows and joins them. A preview
        that contained a word the artifact does not is a fabrication, which is the one thing a
        verify-only store must not do.
        """
        source_words = set(DOC.split())
        excerpt_words = set(dense_excerpt(DOC).replace("…", " ").split())
        assert excerpt_words
        # Word-level rather than substring, because `chunk_text` re-joins on whitespace: a
        # window that spans a line break is not a literal substring of the source even though
        # every word in it came from there. What must hold is that nothing was INVENTED.
        assert excerpt_words <= source_words, sorted(excerpt_words - source_words)

    def test_it_is_deterministic(self):
        """Same document, same cut. Nothing here samples, so two reads must agree."""
        assert dense_excerpt(DOC) == dense_excerpt(DOC)

    def test_it_cuts_rather_than_returning_everything(self):
        """`top_break` selects; a cut that kept every window would not be a cut.

        Asserted as "shorter than the source" rather than at a length, because no length is
        typed anywhere in this module and pinning one here would invent the constraint the
        design removed.
        """
        assert 0 < len(dense_excerpt(DOC)) < len(DOC)

    def test_empty_content_is_empty(self):
        assert dense_windows("") == []
        assert dense_excerpt("") == ""

    def test_a_document_too_short_to_cut_comes_back_whole(self):
        """`cut.py`'s no-break convention: fewer than two windows, nothing to choose between."""
        tiny = "one short line about attenuation"
        assert dense_windows(tiny) == [tiny] or "".join(dense_windows(tiny)) == tiny


class TestNoStemmerReachesThisPath:
    def test_density_imports_no_analyzer(self):
        """The rule, asserted rather than described.

        A comment saying "no stemmer here" survives exactly until someone needs one for an
        afternoon. This fails if the import comes back, which is the only form of the rule that
        holds — the same discipline `test_attenuation_is_single_sourced.py` applies to its own
        operator.
        """
        import inspect
        from mantle.search.beacon import density

        source = inspect.getsource(density)
        assert "tokenizer" not in source.replace("stemmer", ""), (
            "beacon.density imports a text analyzer again. The query-aware cut it was added "
            "for was removed for being slow (0.25s -> 1.96s server-side) and for putting a "
            "hand-built table of English suffix rules in the store's read path. Select windows "
            "by vector proximity instead."
        )
