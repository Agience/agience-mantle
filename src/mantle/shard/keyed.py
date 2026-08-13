"""The keyed arm, across backends — a word/symbol -> the artifacts that carry it.

A dictionary lookup is keyed retrieval, not similarity, and its failure mode is distractor
density: `lemmas` is one namespace shared by every content type in the corpus, so
`lookup_by_lemma('spaceship', 200)` can return 200 rows and zero synsets — 6,063,979 `wiki-*`
rows crowding out 117,659 `wn-*` ones. A caller that wants senses must be able to say so.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple


def _supports_typed(store: Any, method: str) -> bool:
    """Does this backend accept `content_type=` on `method`?

    Asked of the SIGNATURE, once, rather than by calling and catching `TypeError` — a caught
    TypeError cannot be told apart from one raised *inside* a working method, which is how a
    capability probe turns into a swallowed bug."""
    fn = getattr(store, method, None)
    if fn is None:
        return False
    try:
        return "content_type" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def lookup_by_lemma(store: Any, word: str, *, limit: int = 12,
                    content_type: Optional[str] = None,
                    principal: Optional[str] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """`(rows, typed)` — artifacts carrying `word` as a lemma, of `content_type` if given.

    `typed` reports whether the type discrimination happened in the seek (True) or by
    post-filtering an over-fetched page (False). It is returned rather than logged because the
    caller is the only party that knows whether a thin result is worth re-asking."""
    from mantle.db.access import filter_visible
    # Over-fetch when scoping: a page consumed entirely by rows the caller may not see would
    # filter to [] and read as "not found" — the same structural limit as `typed=False`.
    want = limit if principal is None else min(max(int(limit) * 4, 24), int(limit) + 200)
    rows, typed = lookup_by_list_field(store, "lemmas", word, limit=want,
                                       content_type=content_type)
    if principal is not None:
        rows = filter_visible(rows, principal, store=store)[:limit]
    return rows, typed



def lookup_by_list_field(store: Any, field: str, value: str, *, limit: int = 20,
                         content_type: Optional[str] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """`(rows, typed)` — artifacts whose indexed list `field` contains `value`.

    Raises whatever the backend raises. In particular `mantle.db.ListIndexUnbuilt`
    propagates: a store whose keyed index has never been built must not answer `[]`, because
    `[]` from a keyed lookup asserts "this store does not contain that word"."""
    v = str(value).lower()
    if content_type is None:
        return list(store.lookup_by_list_field(field, v, limit=limit)), True
    if _supports_typed(store, "lookup_by_list_field"):
        rows = store.lookup_by_list_field(field, v, limit=limit, content_type=content_type)
        return list(rows), True
    # Degraded: discrimination after the fact. Over-fetch so the filter has room to work, then
    # keep only the requested type. Bounded deliberately — an unbounded over-fetch to "make sure"
    # is how a keyed lookup becomes a scan.
    over = min(int(limit) * 20, int(limit) + 400)
    rows = [a for a in store.lookup_by_list_field(field, v, limit=over)
            if a.get("content_type") == content_type]
    return rows[:limit], False
