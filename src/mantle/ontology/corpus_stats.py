"""Corpus token & document-frequency MEASUREMENT — the grounding primitive retrieval sits on.

These are the corpus's OWN measures of how much information a term carries (document frequency /
IDF), read straight off the store's FTS index (`mantle.db.lattice.fts`). They are pure MEASUREMENT
(no answering, no condensation), which is why they stay EMBER-side (Apache, grounding) even though
the BM25 retrieval TEKTON that consumes them moved to sage: the reach path (`ember.match.fired_field`)
and the recall path (sage's `content_search`) must agree about which words carry a question, so they
read ONE measure — this module — from the same store. [[ember-is-a-runner]] / [John, 2026-07-23:
"leave one path… no stop-list, nothing hand-authored."]

Extracted from the pre-move `ember/content_search.py` unchanged; `ember.match` imports `_salient`
here, and sage's moved `content_search` imports the whole cluster (`terms`/`_df`/`_salient`/…).
"""
from __future__ import annotations

import math
import re
from typing import List, Sequence

from mantle.db.lattice import fts as _fts   # THE index — one implementation, in the store


def terms(query: str) -> List[str]:
    """The query's tokens, verbatim — alphanumerics, lowercased. No filtering: the index's IDF is
    what separates informative words from filler, not a hand-authored list."""
    return [t for t in re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 1]


def _q(t: str) -> str:
    return '"%s"' % t.replace('"', "")


# ⛔ `_DF_CAP = 5000` REMOVED 2026-07-30. Its own comment claimed "the exact value changes nothing
# except how early the count halts". That was false: the query was `... MATCH ? LIMIT _DF_CAP+1`,
# so EVERY term past the cap reported exactly 5001 and all common words carried IDENTICAL IDF —
# indistinguishable from each other and from a genuinely mid-frequency term. The docstring of
# `_salient` below already recorded the damage ("eight terms tied at the cutoff and ALL of them
# were kept"), and `_salient` was reshaped to work around a saturated reading rather than fix it.
#
# df is now EXACT, read off `fts5vocab` — one row per term carrying `doc`, straight out of the
# index. No scan, no cap, nothing to tune.


def _df(conn, t: str) -> int | None:
    """The document frequency of a term — EXACT, from the index's own vocabulary table.

    Returns None when the store cannot report it (an older index with no vocab table). That is
    "unmeasurable", not "very common": a caller must say so rather than substitute a number."""
    return _fts.document_frequency(conn, t)


def _salient(conn, ts: Sequence[str]) -> List[str]:
    """The query's informative terms — those carrying at least the query's OWN MEAN information.

    ⛔ THIS TOOK "THE RARER HALF" BY COUNT AND IT DROPPED THE SUBJECT. MEASURED 2026-07-23 on the
    live corpus: for "what is a dog" the document frequencies are `what` 192, `dog` 242, `is` 5001,
    so `keep = len(ts) // 2` = **1**, the cutoff became `what`'s own df, and **`dog` was excluded
    from retrieval entirely**. The query "what is a dog" was retrieved on the word "what". It only
    looked fine because §13.14 made the need's own fired position a candidate by construction —
    one fix masking another.

    It failed in the opposite direction too — `_df` USED TO saturate at a cap, so every common word
    reports the same saturated 5001; on "the of and to an is are was" eight terms tied at the cutoff
    and ALL of them were kept, i.e. "the rarer half" returned everything.

    A FRACTION IS THE WRONG SHAPE — how many terms carry information is not a property of how many
    words were typed. Information is, and the corpus already measures it: `IDF = log(N/df)`. A term
    is salient here when it carries AT LEAST the mean information of this query's own terms. Nothing
    is chosen: the bar is the query's own mean, so it scales with the query and with the corpus.

    When every term carries the same information (all-common, all-rare, or one word) the mean equals
    each of them and everything is kept — which is the honest reading: the corpus reports that
    nothing here distinguishes anything. The `>= mean` comparison, not `>`, is what makes that so."""
    ts = list(ts)
    if len(ts) <= 1:
        return ts
    total = _corpus_rows(conn)
    if not total:
        return ts          # N unmeasurable → IDF unmeasurable → nothing distinguishes anything
    dfs = {t: _df(conn, t) for t in ts}
    if any(d is None for d in dfs.values()):
        return ts          # the index cannot report df → say so, do not guess
    idf = {t: math.log(max(1.0, total / max(1.0, float(dfs[t])))) for t in ts}
    # ⛔ `idf[t] >= mean - 1e-12` — A TYPED TOLERANCE, AND IT IS NOT NEEDED. The 1e-12 was patching
    # the rounding that `sum(...) / len(...)` introduces, so that the all-equal case (where every
    # term equals the mean and everything must be kept) is not lost to a last-bit error. Multiplying
    # through instead of dividing removes the division entirely: `idf[t] * n >= Σ idf` is the same
    # comparison with one fewer rounding, and the all-equal case then holds EXACTLY rather than
    # within a number someone chose. No epsilon, measured or otherwise.
    n = len(idf)
    total_idf = sum(idf.values())
    keep = [t for t in ts if idf[t] * n >= total_idf]
    return keep or ts


def _corpus_rows(conn) -> float:
    """How many rows the corpus holds — read off the store's MAINTAINED counter, never `count(*)`
    (which dereferences every record).

    ⛔ A missing counter used to fall back to `_DF_CAP`, i.e. it INVENTED a corpus size so the
    ratio "still orders terms correctly". A fabricated N is a fabricated IDF. Unmeasurable now
    reads 0.0 and `_salient` reports that nothing distinguishes anything."""
    try:
        row = conn.execute("SELECT n FROM counter WHERE name = 'vertex'").fetchone()
        if row and int(row[0]) > 0:
            return float(row[0])
    except Exception:
        pass
    return 0.0


def _match_expr(ts: Sequence[str]) -> str:
    """A safe FTS5 MATCH: every term quoted (so punctuation/operators can't reach the parser) and
    OR-ed, so BM25 ranks by how much of the query each document carries."""
    return " OR ".join(_q(t) for t in ts)


__all__ = ["terms", "_q", "_df", "_corpus_rows", "_salient", "_match_expr"]
