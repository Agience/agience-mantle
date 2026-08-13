"""`search/ingest/chunking.py` — the ingest-side invariants.

This module scores near-zero on authorization vocabulary, and its real exposure is elsewhere: a
standing rule against trained tokenizers, content loss during chunking, and hangs from degenerate
chunk-size/overlap arguments.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from mantle.search.ingest.chunking import (chunk_text, count_words,
                                    extract_text_from_context,
                                    should_chunk_content)

MODULE = (pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle"
          / "search" / "ingest" / "chunking.py")


# ── the standing rule ────────────────────────────────────────────────────────
def test_no_trained_tokenizer_is_imported():
    """[[no-trained-weights]]: a downloaded merge table is still trained weights — someone else's
    frequency statistics. Checked by AST rather than by importing, so a lazy in-function import is
    caught too."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.add((node.module or "").split(".")[0])
    banned = roots & {"tiktoken", "transformers", "tokenizers", "sentencepiece", "openai"}
    assert not banned, "a trained tokenizer came back into the ingest path: %s" % sorted(banned)


# ── no content may be lost ───────────────────────────────────────────────────
@pytest.mark.parametrize("n_words", [1, 5, 100, 751, 1500, 3000])
def test_every_word_survives_chunking(n_words):
    """A word in no chunk is unfindable, and the search reports success anyway."""
    words = ["w%d" % i for i in range(n_words)]
    chunks = chunk_text(" ".join(words))
    seen = " ".join(c["text"] for c in chunks).split()
    for w in words:
        assert w in seen, "%r fell out of every chunk (%d words -> %d chunks)" % (
            w, n_words, len(chunks))


def test_chunks_are_exact_substrings_of_the_input():
    """Downstream cache keys and index records assume this — the encode/decode round trip it
    replaced preserved it, so the replacement must too."""
    text = "alpha   beta\tgamma\ndelta " * 400
    for c in chunk_text(text):
        assert c["text"] in text


def test_chunk_spans_are_contiguous_and_advancing():
    chunks = chunk_text(" ".join("w%d" % i for i in range(2000)))
    assert chunks[0]["start_word"] == 0
    assert chunks[-1]["end_word"] == 2000
    for a, b in zip(chunks, chunks[1:]):
        assert b["start_word"] > a["start_word"], "stride stalled — this is the hang shape"
        assert b["start_word"] <= a["end_word"], "a gap between chunks means dropped words"


# ── the operator-settable hang ───────────────────────────────────────────────
@pytest.mark.parametrize("size,overlap", [(100, 100), (100, 200), (100, 10_000), (1, 1)])
def test_overlap_at_or_above_chunk_size_terminates(size, overlap):
    """Zero (or negative) stride would loop forever building chunks until memory went; both
    numbers are operator config. Degenerate config steps a full chunk instead."""
    chunks = chunk_text(" ".join("w%d" % i for i in range(500)), chunk_size=size, overlap=overlap)
    assert chunks and chunks[-1]["end_word"] == 500


# ── empty / degenerate input ─────────────────────────────────────────────────
@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_input_yields_no_chunks_rather_than_one_empty_chunk(text):
    """An empty chunk would be indexed as a real document with no content."""
    assert chunk_text(text) == []
    assert should_chunk_content(text) is False


def test_short_text_is_one_chunk_carrying_the_whole_input():
    out = chunk_text("just a few words here")
    assert len(out) == 1 and out[0]["text"] == "just a few words here"


def test_count_words_is_whitespace_delimited():
    assert count_words("a  b\tc\nd") == 4
    assert count_words("") == 0


# ── context extraction sits on the ingest path and must not raise ────────────
@pytest.mark.parametrize("bad", ["{not json", "", "   ", "null", "[1,2]", "12"])
def test_malformed_context_returns_empty_fields_instead_of_raising(bad):
    """Ingest processes whatever is in the store. One malformed context must not abort the run."""
    out = extract_text_from_context(bad)
    assert set(out) == {"title", "description", "tags_raw"}
    assert all(isinstance(v, str) for v in out.values())


def test_tags_are_joined_from_either_a_list_or_a_string():
    assert extract_text_from_context('{"tags": ["a", "b"]}')["tags_raw"] == "a, b"
    assert extract_text_from_context('{"tags": "a, b"}')["tags_raw"] == "a, b"
    assert extract_text_from_context('{"tags": ["a", null, "b"]}')["tags_raw"] == "a, b"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
