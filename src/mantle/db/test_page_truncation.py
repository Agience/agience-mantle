"""A truncated answer must not look like a complete one.

A plain list of rows can't distinguish "this is every artifact carrying this lemma" from "there
are hundreds more and you are seeing only the first `limit` of them" — `lookup_by_lemma` returning
exactly `limit` rows looks identical either way. `Page.truncated` is what makes the two cases
distinguishable.
"""
from __future__ import annotations

import pytest

from .vertex import Page


def _rows(n):
    return [{"id": "a%03d" % i, "lemmas": ["dog"], "content_type": "text/x-wordnet"}
            for i in range(n)]


# ── the type itself ─────────────────────────────────────────────────────────────
def test_page_is_a_real_list():
    """Backward compatibility is the reason this is a list subclass and not a tuple or a
    dataclass: every existing caller keeps working untouched."""
    p = Page(_rows(3), truncated=False, limit=12)
    assert isinstance(p, list)
    assert len(p) == 3
    assert p[0]["id"] == "a000"
    assert [r["id"] for r in p] == ["a000", "a001", "a002"]
    assert isinstance(p[:2], list)
    assert list(reversed(p))[0]["id"] == "a002"


def test_page_carries_its_verdict():
    p = Page(_rows(12), truncated=True, limit=12)
    assert p.truncated is True
    assert p.limit == 12


def test_complete_answer_is_not_truncated():
    p = Page(_rows(3), truncated=False, limit=12)
    assert p.truncated is False


def test_truncated_defaults_false_but_must_be_set_deliberately():
    """A bare `Page(rows)` claims completeness — so every construction site must pass the verdict
    it actually measured. This test documents that the default is an assertion, not a shrug."""
    p = Page(_rows(2))
    assert p.truncated is False
    assert p.limit is None


# ── the property that matters ───────────────────────────────────────────────────
def test_full_page_and_complete_answer_are_distinguishable():
    """The whole point. Both have len == limit; only one is the complete answer."""
    exactly_full = Page(_rows(12), truncated=False, limit=12)   # 12 matches, 12 returned
    truncated = Page(_rows(12), truncated=True, limit=12)       # >12 matches, 12 returned
    assert len(exactly_full) == len(truncated) == 12
    assert exactly_full.truncated != truncated.truncated, \
        "a full page and a complete answer MUST be distinguishable"


@pytest.mark.parametrize("n_matches,limit,expect_trunc", [
    (0, 12, False),
    (5, 12, False),
    (12, 12, False),      # exactly the limit, nothing beyond => complete
    (13, 12, True),       # one more exists => truncated
    (500, 12, True),
])
def test_truncation_boundary(n_matches, limit, expect_trunc):
    """Mirrors the fetch-limit+1 rule: fetch one extra row, and its presence is the verdict."""
    fetched = _rows(min(n_matches, limit + 1))
    truncated = len(fetched) > limit
    p = Page(fetched[:limit], truncated=truncated, limit=limit)
    assert p.truncated is expect_trunc
    assert len(p) == min(n_matches, limit)
