#!/usr/bin/env python3
"""UserPromptSubmit hook: recall relevant mantle artifacts for the prompt just submitted, and
print them to stdout so Claude Code adds them to context before Claude sees the prompt.

Best-effort: any failure (mantle down, no token, no hits) prints nothing and exits 0. A hook
that blocked the prompt because a local dev service was unreachable would be worse than one
that silently skipped the recall.
"""
from __future__ import annotations

import sys

from mantle_common import log_event, read_stdin_json, recall

#: Total characters of recalled content to inject, across ALL hits.
#:
#: A per-hit character cap is the wrong shape here. `recall`'s `content` is already an
#: entropy-density cut (`search/beacon/density.py`) -- the spans of the artifact that actually
#: carry signal, at whatever length that turns out to be -- so slicing it again mid-window throws
#: away the part the cut was chosen for and can leave a hit represented by half a sentence. What
#: has to be bounded is the total added to the prompt, so this drops WHOLE hits once the budget
#: is spent rather than damaging every one of them.
_TOTAL_CONTENT_BUDGET = 4000

#: Operators whose output is kept OUT of the automatic recall — by PROVENANCE, not by format.
#:
#: A session transcript is the largest thing in this store and the least specific. `recall`
#: orders by COVERAGE — the count of distinct query stems an artifact carries — so a 118 KB
#: transcript that mentions everything once outranks the 3 KB commit message that answers the
#: question, on almost every question. Measured 2026-08-13: three real engineering questions
#: returned 13 transcripts in 15 hits, while the actual answers sat in commit messages and the
#: README and never placed.
#:
#: ⭐ THIS IS A DEMOTION, NOT A DELETION, AND THE DISTINCTION IS THE WHOLE DESIGN. A transcript
#: is still information — often the only record of why something was done — so nothing here
#: removes one, retypes one, or stops it being found. What is excluded is the five slots
#: injected in front of every prompt, where a long unspecific document costs a short precise
#: one its place. Ask for transcripts and you get them; `recall` with no filter still returns
#: them; only the automatic path skips them.
#:
#: Keyed on the OPERATOR rather than a content type. An earlier cut minted
#: `text/vnd.agience.transcript` for this and that was a category error: a content type says
#: what the bytes are, and these bytes are markdown. What separates a transcript from a commit
#: message is what MADE it. Keying on provenance also means the next source of prose needs no
#: new MIME type — it names its operator, like everything else.
_EXCLUDED_OPERATORS = ("op-claude-code-transcript",)

#: Keep a hit only if it scores at least this fraction of the best hit in the same response.
#:
#: ⚠ THE FLOOR IS RELATIVE BECAUSE THE SCORE IS NOT A MEASURE. Under coverage ordering `score`
#: is an integer count of matched query stems, and `recall`'s own contract says it is
#: "comparable between the hits of one response and across no two" — so an absolute threshold
#: would mean something different for a three-word prompt than for a paragraph. A ratio within
#: one response is the only comparison the number supports.
#:
#: The effect worth having is that a query with ONE good answer injects one hit instead of
#: padding to five. Returning less is the point: five weak hits cost the same context as five
#: strong ones and teach the reader to ignore the block.
_MIN_SCORE_RATIO = 0.5


def _query_for(prompt: str) -> str:
    """The prompt, plus the operator exclusions.

    `!tags:` narrows before retrieval and conjoins with the prompt's own terms, so this is one
    query rather than a filter over results — the excluded artifacts never compete for a slot.
    The tag is the filterable mirror of `context.operator`; see `_EXCLUDED_OPERATORS`.
    """
    return " ".join([prompt] + [f"!tags:{op}" for op in _EXCLUDED_OPERATORS])


def _above_floor(hits: list) -> list:
    """Hits scoring at least `_MIN_SCORE_RATIO` of the best. All of them if scores are absent.

    A `None` score is what `recency` ordering returns — nothing measured the hits, so there is
    no spectrum to cut against and dropping any of them would be arbitrary.
    """
    scores = [h.get("score") for h in hits]
    if not scores or any(s is None for s in scores):
        return hits
    top = max(scores)
    if not top:
        return hits
    return [h for h in hits if h.get("score", 0) >= top * _MIN_SCORE_RATIO]


def main() -> int:
    payload = read_stdin_json()

    # Accept both spellings. The field name is the host's to choose and it is not worth a silent
    # no-op to get wrong: this hook's failure mode is indistinguishable from "nothing relevant
    # was found", so a mismatch here disables the whole recall leg without ever showing an error.
    prompt = ""
    for key in ("prompt", "user_input"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break
    if not prompt:
        log_event("recall", outcome="no-prompt", keys=sorted(payload.keys()))
        return 0

    # Ask for more than will be injected, so the floor has a spectrum to cut against rather
    # than a pre-truncated top five.
    raw = recall(_query_for(prompt), size=8)
    if not raw:
        log_event("recall", outcome="no-hits", prompt_chars=len(prompt), prompt=prompt)
        return 0

    hits = _above_floor(raw)[:5]
    dropped = len(raw) - len(hits)

    lines = ["[mantle recall - artifacts that may be relevant to this prompt]"]
    spent = 0
    for h in hits:
        content = (h.get("content") or "").strip()
        if content and spent + len(content) > _TOTAL_CONTENT_BUDGET:
            continue  # whole hits in or out; never a damaged excerpt
        title = h.get("title") or h.get("id")
        lines.append(f"- {title} (id: {h.get('id')}, score: {h.get('score')})")
        if content:
            lines.append(f"  {content}")
            spent += len(content)
    if len(lines) == 1:
        log_event("recall", outcome="all-over-budget", hits=len(hits))
        return 0
    print("\n".join(lines))
    # The PROMPT and the HITS themselves, not just counts. A count says the machinery ran; it
    # says nothing about whether what came back was worth injecting, which is the only question
    # left once the plumbing works. Judging that after the fact needs the query and the titles
    # it actually returned, side by side.
    # The prompt WHOLE, not a slice of it. A truncated query in an evidence log is the same
    # mistake as a truncated preview: the part you cut is exactly the part you would need to
    # explain why a given hit came back. The log already bounds itself by rotation.
    # `dropped` is recorded because a filter that silently removes most of its input reads as
    # "nothing else matched". Judging whether the floor is set right needs to see what it cut.
    log_event("recall", outcome="injected", hits=len(hits), chars=spent,
              prompt=prompt, dropped_below_floor=dropped,
              returned=[{"title": h.get("title"), "score": h.get("score")} for h in hits])
    return 0


if __name__ == "__main__":
    sys.exit(main())
