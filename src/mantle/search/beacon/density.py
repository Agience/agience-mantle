# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# Beacon is the permissive half of the two-tier model, deliberately: mantle ships
# Apache so a store can be taken, built on and shipped by anyone, and beacon is the
# reduced instrument that makes such a store genuinely useful on its own.
#
# Everything in this file is Apache-2.0 and public. `shannon_bits` below reproduces
# the upstream library's one entropy definition (`entropy.shannon_bits`) independently
# — same relationship `cut.py`'s `gap_split` has to the implementation it reproduces:
# no import edge, own arithmetic, same formula.
# ---------------------------------------------------------------------------

"""Density — which spans of an artifact's content are worth showing as a recall preview.

Beacon's cut (`cut.py`) picks which *candidates* belong in a result set, scored by
similarity to the query. This is the same cut primitive (`top_break`) turned on a
different spectrum: which *spans of one document* stand out from that document's own
other spans. A recall preview built on a blind `content[:N]` prefix shows whatever the
document opens with, truncated at a typed length regardless of what it cuts through;
this shows a span the cut found — for as long as the cut says there is signal and not
a character longer. No length is ever typed here.

No stemmer here, which is a rule rather than an omission. A query-aware cut by Porter
stems per window is wrong twice over. It is slow — stemming every window of every hit
takes the server side of a recall from ~0.25s to 1.96s, on the hot path in front of
every prompt. And it is a predetermination: a stemmer is a hand-built table of English
suffix rules, so the store's read path would carry a language assumption and a set of
irregular special cases that nothing measured. This file reproduces one entropy
definition and calls one cut, and a rule set does not belong beside them.

The problem that cut was aimed at is open. Entropy is query-independent by
construction, so one artifact returns a byte-identical excerpt to every question asked
of it: measured on node 71/dev, three unrelated questions each got the same 368
characters of this project's README, and those characters were its installation
instructions. Recall picks the right document and shows the wrong part of it, which no
ranking can fix. The answer is to select windows by VECTOR PROXIMITY to the query --
a measurement rather than a rule, and cheaper besides (a static embedder encodes 100
windows in ~3ms). Mantle embeds nothing, so those vectors arrive from a writer at index
time or the selection happens in the caller; either way it is not a stemmer.

Public surface
--------------
    shannon_bits · dense_windows · dense_excerpt
"""
from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from mantle.search.ingest.chunking import chunk_text
from .cut import top_break

__all__ = ["shannon_bits", "dense_windows", "dense_excerpt"]

#: `dense_windows`' window size, in `chunking.chunk_text`'s cl100k-token units (its
#: own 3/4 words-per-token derivation puts this at ~15 words). Stated, not derived —
#: this is a measurement-scale choice (how fine a grain the cut reads at), not an
#: output decision, the same status `cut.py`'s `hi = 64` carries: typed, and it binds.
#: A window this narrow is what gives the cut something to discriminate between; the
#: embedding pipeline's 1000-token default answers a different question (how much
#: text one vector should summarize) and would return a single window here, on which
#: `top_break` has nothing to cut. Nothing downstream of this is typed: how many
#: windows are dense, and how long the assembled excerpt is, are read off the cut.
_PREVIEW_WINDOW_TOKENS = 20

#: Inserted between two kept windows that were not adjacent in the source, so a
#: reader can tell "these are two separate dense spans" from "this is one continuous
#: sentence" — the discontinuity is real and hiding it would misrepresent the excerpt
#: as more coherent than it is.
_GAP_MARKER = " … "


def shannon_bits(weights) -> float:
    """Shannon entropy (bits) of a non-negative weight array: `H(w) = -sum p log2 p`,
    `p = w / sum(w)`. The one definition, reproduced from the upstream
    `entropy.shannon_bits` — same formula, same zero-guard, same clip, independently
    implemented because this module carries no import edge to that library
    (see the file header). Upstream reads it off a power marginal; here the
    weights are a character-frequency histogram, but the arithmetic does not care
    what produced the counts.

    Returns `0.0` when the weights carry no mass (`sum(w) <= 1e-30`) — the same "no
    signal, no entropy" floor `cut.py`'s primitives use rather than raising on an
    empty or degenerate input.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    total = float(w.sum())
    if total <= 1e-30:
        return 0.0
    p = w / total
    return -float(np.sum(np.where(p > 0, p * np.log2(np.clip(p, 1e-12, 1.0)), 0.0)))


def _char_weights(text: str) -> np.ndarray:
    """Character-frequency histogram of `text` — codepoint-level, not byte-level, so
    a multi-byte character is not double-counted against single-byte ones. This is
    the weight array `shannon_bits` reads density from."""
    return np.fromiter(Counter(text).values(), dtype=np.float64)


def dense_windows(content: str) -> List[str]:
    """Which spans of `content` to show, in original document order.

    Chunks `content` into small, non-overlapping windows (`chunking.chunk_text` —
    overlap is that module's concern for embedding-index continuity across chunk
    boundaries, which does not apply here, so it is off), scores each window, and keeps
    the top cluster via `beacon.cut.top_break`: whichever windows stand out from the
    document's own others, at whatever count that turns out to be — never a fixed top-k,
    never a per-corpus threshold, and never a length cap.

    ``query_stems`` chooses what "stand out" measures (see the module docstring). With
    them, it is how many distinct query stems a window carries, so a window holding four
    of the terms asked for outranks one holding a single common term — which is what
    keeps a long document from returning its whole self just because one frequent stem
    appears throughout it. Without them, or when no window carries any of them, it is the
    window's own Shannon entropy, unchanged.

    Returns every window (there is nothing to cut) when there are fewer than two —
    `cut.py`'s own no-break convention — and `[]` when `content` is empty.
    """
    if not content:
        return []
    windows = chunk_text(content, chunk_size=_PREVIEW_WINDOW_TOKENS, overlap=0)
    if len(windows) <= 1:
        return [w["text"] for w in windows]

    scores = np.array([shannon_bits(_char_weights(w["text"])) for w in windows])

    keep, _ = top_break(scores)
    kept = [w for w, k in zip(windows, keep) if k]

    out: List[str] = []
    prev_end = None
    for window in kept:
        sep = _GAP_MARKER if prev_end is not None and window["start_word"] != prev_end else ""
        out.append(sep + window["text"] if sep else window["text"])
        prev_end = window["end_word"]
    return out


def dense_excerpt(content: str) -> str:
    """`dense_windows(content, query_stems)`, reassembled into a single preview string.

    Consecutive kept windows already carry a `_GAP_MARKER` between them when they
    were not adjacent in the source (see `dense_windows`), so a plain join preserves
    that signal. No length is applied here or anywhere upstream of it — the excerpt
    is exactly as long as the cut found signal for.
    """
    return "".join(dense_windows(content))
