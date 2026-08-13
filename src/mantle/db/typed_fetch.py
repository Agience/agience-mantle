"""Typed artifact fetch — every row of one SMALL, SELECTIVE content_type, without streaming the corpus.

`list_artifacts(content_type=…)` with no limit streams all N rows and filters in PYTHON, because the
arcade applies caller filters client-side deliberately (any predicate beside the id range flips the
planner off the id index). At 5M rows that is hundreds of keyset pages. This asks the store for the
rows it actually wants, and reports whether the answer was EXHAUSTIVE rather than assuming it.

"""
from __future__ import annotations

from typing import Any, List

from mantle.db.constants import CT_FETCH_CAP as _CT_FETCH_CAP


def _typed(artifacts, name):
    """The typed lattice-store method `name`, or None on a store that does not have it.

    Absence means "this store predates the typed rewrite" — it never means "the answer is empty".
    Every caller below must therefore fall back to a path that can FAIL, never to a default."""
    fn = getattr(artifacts, name, None)
    return fn if callable(fn) else None

def list_by_content_type(artifacts, content_type: str) -> list:
    """EVERY artifact of one SMALL, SELECTIVE content_type, without streaming the whole corpus.

    WHY: `list_artifacts(content_type=…)` with no limit streams all N rows and filters in PYTHON
    (arcade.py applies caller filters client-side deliberately — any extra predicate beside the id
    range flips the planner off the id index). At 5M rows that is ~500 keyset pages of ~743ms each
    ≈ 370s, paid to reach the ~50 operator docs, the handful of s3sync cursors, or the taught
    triples. Those VALUES are selective, so one indexed query answers them in milliseconds.

    PRECONDITION: pass only a type expected to match few rows. A bulk value (text/markdown,
    text/x-wordnet, text/x-python-symbol) is NOT a candidate — matching most of the corpus is a scan
    whatever the index says. The cap enforces it rather than trusting the caller.

    `artifacts` is the ARTIFACT STORE (not the bundle). Result-preserving by construction: probe with
    LIMIT cap+1 and accept the fast answer ONLY if it came back short — a full page means the value
    is not selective (or the engine truncated), and the exhaustive stream runs instead. Never returns
    a silently short list. No COUNT: count(*) on a bulk value has to enumerate every match (that is
    the 120s timeout above), whereas a LIMITed fetch stops early.

    The typed path removes the ambiguity at the source: `list_by_content_type` returns
    `(docs, exhaustive)`, so "I did not finish" is not representable as "there is nothing"."""
    typed = _typed(artifacts, "list_by_content_type")
    if typed is not None:
        docs, exhaustive = typed(content_type, cap=_CT_FETCH_CAP)
        if exhaustive:
            return list(docs)
    # No typed accessor, or it did not come back exhaustive: stream the store's own list.
    return list(artifacts.list_artifacts(content_type=content_type))
