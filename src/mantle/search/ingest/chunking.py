from __future__ import annotations
# search/ingest/chunking.py
import logging
import re
from typing import Any, Dict, List

from origin import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ⛔ TIKTOKEN IS DELETED. [John, 2026-07-22: the no-models rule is UNIVERSAL]
#
# This module sized chunks with `tiktoken.get_encoding("cl100k_base")` — OpenAI's
# BPE merge table, downloaded from OpenAI's CDN on first use. A tokenizer's merge
# ranks are TRAINED artifacts (learned from a corpus, shipped as a rank file), so
# they fall under the same law as the remote providers removed from
# `embeddings.py`: a downloaded vocabulary is still trained weights — it is
# simply someone else's frequency statistics. It also made first-call chunking
# depend on an outbound fetch to openai.com.
#
# Chunks are now sized in whitespace-delimited WORDS, computed with stdlib only.
# Chunk texts are exact substrings of the input (as the encode/decode round trip
# was), so downstream cache keys and index records keep the same shape.
# ---------------------------------------------------------------------------

# Word-boundary scanner: a "word" is any maximal run of non-whitespace. Spans
# index into the ORIGINAL string so chunk texts are exact substrings.
_WORD_RE = re.compile(r"\S+")

# Config (`SEARCH_CHUNK_SIZE` / `SEARCH_CHUNK_OVERLAP`) is denominated in
# cl100k_base TOKENS (defaults 1000 / 200). Derivation of the word equivalents:
# cl100k_base averages ~4 characters/token on English prose (OpenAI's published
# rule of thumb), and English prose averages ~5.3 characters/word including the
# separating space — so 1 word ≈ 5.3/4 ≈ 4/3 tokens, i.e. words ≈ tokens × 3/4.
# 1000-token chunks → 750 words; 200-token overlap → 150 words. Same effective
# chunk mass, no downloaded vocabulary.
_WORDS_PER_TOKEN_NUM = 3
_WORDS_PER_TOKEN_DEN = 4


def _tokens_to_words(n_tokens: int) -> int:
    """Convert a token-denominated config value to words (see derivation above)."""
    return (n_tokens * _WORDS_PER_TOKEN_NUM) // _WORDS_PER_TOKEN_DEN


def count_words(text: str) -> int:
    """Count whitespace-delimited words in text (the deterministic chunk unit)."""
    return len(_WORD_RE.findall(text))


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Chunk text into overlapping segments by word count.

    ``chunk_size`` / ``overlap`` are in cl100k-token units (the config
    denomination, kept for setting compatibility) and are converted to words
    via the 3/4 derivation above.

    Returns list of chunks with:
    - chunk_id: sequential identifier
    - text: chunk content (exact substring of the input)
    - start_word: index of the chunk's first word in the original text
    - end_word: index one past the chunk's last word
    """
    if chunk_size is None:
        chunk_size = config.SEARCH_CHUNK_SIZE
    if overlap is None:
        overlap = config.SEARCH_CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    size_words = max(1, _tokens_to_words(chunk_size))
    overlap_words = max(0, _tokens_to_words(overlap))

    # Word spans into the original string — chunk texts are exact substrings.
    spans = [m.span() for m in _WORD_RE.finditer(text)]
    total_words = len(spans)

    if total_words <= size_words:
        # Single chunk
        return [
            {
                "chunk_id": 0,
                "text": text,
                "start_word": 0,
                "end_word": total_words,
            }
        ]

    # Overlap >= size would loop forever (a pre-existing hazard of the token
    # version too, with operator-settable config). Degenerate config steps a
    # full chunk — adjacent, no overlap — rather than hanging.
    step = size_words - overlap_words
    if step <= 0:
        step = size_words

    chunks = []
    chunk_id = 0
    start = 0

    while start < total_words:
        end = min(start + size_words, total_words)

        # Slice the original text from the first word's start to the last
        # word's end — interior whitespace is preserved verbatim.
        chunk_str = text[spans[start][0]:spans[end - 1][1]]

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_str,
                "start_word": start,
                "end_word": end,
            }
        )

        chunk_id += 1
        start += step

    logger.debug(f"Chunked {total_words} words into {len(chunks)} chunks")
    return chunks


def should_chunk_content(content: str) -> bool:
    """Determine if content should be chunked based on word count."""
    if not content or not content.strip():
        return False

    # Same unit conversion as chunk_text — content chunks iff it exceeds one
    # chunk's word budget.
    return count_words(content) > max(1, _tokens_to_words(config.SEARCH_CHUNK_SIZE))


def extract_text_from_context(context_str: str) -> Dict[str, str]:
    """
    Extract searchable text fields from artifact context JSON.

    Returns dict with:
    - title: artifact title
    - description: artifact description (PRIMARY search field)
    - tags_raw: comma-separated tags
    """
    import json

    result = {"title": "", "description": "", "tags_raw": ""}

    # ⛔ `.strip()` ASSUMED A STRING AND `context` IS OFTEN A DICT — 57 of 500 live vertices carry
    # one (measured 2026-07-31). This raised `AttributeError: 'dict' object has no attribute
    # 'strip'` on EVERY such artifact, straight through the parse below that was written to handle
    # dicts: the guard rejected the input before the code that understood it ever ran. Latent until
    # mantle was pointed at the real store and reindexed 1.3M chunks, at which point it fired
    # continuously. Emptiness is asked of the value AS IT IS, not of an assumed type.
    if context_str is None or (isinstance(context_str, str) and not context_str.strip()):
        return result
    if not context_str:                      # empty dict / empty list -- nothing to extract
        return result

    # ⛔ A CONTEXT THAT IS NOT JSON IS NOT A FAILURE — IT IS THE DESCRIPTION.
    # [John, 2026-07-31: "Do not add words in, do not take words out. prime directive. but more
    #  importantly - conservation of information. You cannot create or destroy information."]
    # MEASURED on the live shard (500 vertices): 395 carry no context, 57 carry a dict, and 48 carry
    # PLAIN PROSE — e.g. 'adds two numbers'. That prose is exactly the human-written description this
    # function exists to index. `json.loads` raised on every one of them, the except below logged a
    # warning, and the text was DISCARDED — thousands of warnings a minute on the running node, and
    # an index built without the descriptions it was supposed to carry. The information arrived and
    # did not leave. So a bare string is taken as what it plainly is.
    if isinstance(context_str, str):
        try:
            context = json.loads(context_str)
        except json.JSONDecodeError:
            result["description"] = context_str.strip()
            return result
    else:
        context = context_str
    if not isinstance(context, dict):
        # Valid JSON, but a scalar or list rather than an object (`"adds two numbers"` with quotes,
        # or `[...]`). Still content; render it rather than dropping it.
        result["description"] = (context_str.strip() if isinstance(context_str, str)
                                 else str(context))
        return result

    try:
        # Extract title
        result["title"] = context.get("title", "")

        # Extract description (PRIMARY search field)
        # Description is human-curated or AI-enhanced for optimal findability
        result["description"] = context.get("description", "")

        # Extract tags
        tags = context.get("tags", [])
        if isinstance(tags, list):
            result["tags_raw"] = ", ".join(str(t) for t in tags if t)
        elif isinstance(tags, str):
            result["tags_raw"] = tags

    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        logger.warning(f"Failed to extract text from context: {e}")

    return result
