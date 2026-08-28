from __future__ import annotations
# search/ingest/chunking.py
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#
# Chunks are sized in whitespace-delimited words, computed with stdlib only —
# not with a tokenizer's BPE merge table. A tokenizer's merge ranks are trained
# artifacts (learned from a corpus, shipped as a rank file), so a downloaded
# vocabulary falls under the same no-models rule as `search/embeddings.py`: it is
# someone else's frequency statistics, and using one would make first-call
# chunking depend on an outbound fetch.
#
# Chunk texts are exact substrings of the input, so downstream cache keys and
# index records keep the same shape.
# ---------------------------------------------------------------------------

# Word-boundary scanner: a "word" is any maximal run of non-whitespace. Spans
# index into the original string so chunk texts are exact substrings.
_WORD_RE = re.compile(r"\S+")

# Chunk geometry is a free parameter of this module, defaulted at the call
# signatures below rather than read from settings: it is a property of the index,
# and changing it without a reindex only makes the stored chunks disagree with the
# new ones. A caller that wants different geometry passes it.
#
# The two numbers are denominated in cl100k_base TOKENS (1000 / 200). Derivation
# of the word equivalents: cl100k_base averages ~4 characters/token on English
# prose (OpenAI's published rule of thumb), and English prose averages ~5.3
# characters/word including the separating space — so 1 word ≈ 5.3/4 ≈ 4/3 tokens,
# i.e. words ≈ tokens × 3/4. 1000-token chunks → 750 words; 200-token overlap →
# 150 words. Same effective chunk mass, no downloaded vocabulary.
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunk text into overlapping segments by word count.

    ``chunk_size`` / ``overlap`` are in cl100k-token units and are converted to
    words via the 3/4 derivation above.

    Returns list of chunks with:
    - chunk_id: sequential identifier
    - text: chunk content (exact substring of the input)
    - start_word: index of the chunk's first word in the original text
    - end_word: index one past the chunk's last word
    """
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

    # Overlap >= size would loop forever. Degenerate arguments step a full
    # chunk — adjacent, no overlap — rather than hanging.
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


def should_chunk_content(content: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bool:
    """Determine if content should be chunked based on word count.

    ``chunk_size`` defaults to the same value as :func:`chunk_text` so the two
    answer the same question; a caller passing one must pass both."""
    if not content or not content.strip():
        return False

    # Same unit conversion as chunk_text — content chunks iff it exceeds one
    # chunk's word budget.
    return count_words(content) > max(1, _tokens_to_words(chunk_size))


def extract_text_from_context(context_str) -> Dict[str, str]:
    """Legacy compatibility read of `context`. Returns `{"title": ..., "description": ...}`.

    `context` is not the offer. An artifact's offer — the thing a need is matched against — is its
    top-level `description`, and `pipeline_unified._extract_artifact_fields` reads that first. This
    exists for rows written before that was true, and returns empty strings for anything it cannot
    read as a structured object.

    ## A bare string is not an offer

    A `context` that does not parse as JSON is dropped rather than promoted whole to `description`.
    Two ingests write exactly such a string, and neither describes what the artifact is about:

        sage/canon.py           "canon knowledge: best-practices §intro"
        stage0_sources.py       "the concept 0: a ConceptNet 5.7 English term node"

    Those are provenance — where the row came from. Promoted to `description` they become the
    artifact's stated offer, so every canon document in the store offers "canon knowledge": on the
    live shard all 6,480 of them then position on `canon.n.01` and `cognition.n.01`, the same two
    nodes. A field that says the same thing about every member of a corpus cannot discriminate
    between them. The writers file provenance where provenance goes (`citation` / `source_path` /
    `via`).

    ## Tags are not read here either

    A tag, a collection, a group and an attribute are the same thing — an edge to another artifact —
    rather than a key in a dictionary, so group membership is read from the graph.
    """
    import json

    result = {"title": "", "description": ""}
    if not context_str:
        return result

    context = context_str
    if isinstance(context_str, str):
        try:
            context = json.loads(context_str)
        except (json.JSONDecodeError, ValueError):
            return result          # prose in `context` is provenance, not an offer
    if not isinstance(context, dict):
        return result              # a scalar or a list is not a structured offer either

    try:
        result["title"] = str(context.get("title") or "")
        result["description"] = str(context.get("description") or "")
    except (AttributeError, TypeError) as e:
        logger.warning("Failed to extract text from context: %s", e)
    return result
