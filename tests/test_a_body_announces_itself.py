"""The body is findable — as what it announces, never as itself.

The rule these tests pin: the lexical arm indexes the offer, and the raw body never reaches a
posting list. A body has an offer of its own — a title is what an artifact says about itself in a
line, and `_body_offer` is what a body says about itself when nobody supplied a line.

Two properties make that affordable, and this file leans on both:

* `sse/posting.py` holds one sealed blob per ``(artifact, collection)`` rather than one per term, so
  adding an artifact to a term is not O(entries already in the slot).
* `beacon.density` bounds the projection without a cap — `top_break` keeps whichever windows stand
  out from the document's own others, at whatever count that turns out to be, so the extent is read
  off the document rather than decreed for it.

The projection is query-independent, which is why it belongs at index time rather than in a preview.
Entropy does not know what was asked, so one artifact yields the same span to every question —
measured on 71/dev as the same 368 characters of this repo's README for three unrelated questions.
A preview should answer what was asked; an index write has no question in front of it, and an offer
must be the same offer whoever reads it.
"""
from __future__ import annotations

import pytest

from mantle.search.ingest.pipeline_unified import (
    _STATED_OFFER_FIELDS, _body_offer, _fields_to_index,
)


#: Prose whose middle carries the distinctive terms and whose edges are filler, so "the cut found the
#: dense part" is distinguishable from "the cut kept everything" and from "the cut kept the opening".
BODY = (
    "the the the and and of of to to a a in in it it is is that that was was for for on on "
    "with with as as by by at at from from but but not not are are this this have have had had "
    "Attenuation composes downward through a bounded meet-semilattice where deny absorbs "
    "and the join is deliberately unrepresentable, so authority narrows and never widens. "
    "the the the and and of of to to a a in in it it is is that that was was for for on on "
    "with with as as by by at at from from but but not not are are this this have have had had "
)


# ── the projection ───────────────────────────────────────────────────────────────────────────


def test_a_body_offer_is_shorter_than_the_body():
    """A projection, not the text. If this ever returns the body, the write cost the exclusion was
    protecting against is back — the entry layout fixed the O(slot) term, not "a body is thousands of
    distinct stems"."""
    offer = _body_offer(BODY)
    assert offer
    assert len(offer) < len(BODY), (
        "the body offer is the whole body — nothing was projected, and the cost argument in "
        "`_sse_index_artifact` applies again in full"
    )


def test_it_keeps_the_part_that_stands_out():
    """The cut's job. The distinctive middle must survive; the filler edges are what it is cutting
    against."""
    offer = _body_offer(BODY)
    assert "semilattice" in offer or "Attenuation" in offer, (
        f"the dense span was dropped in favour of filler: {offer!r}"
    )


def test_the_same_body_announces_the_same_thing_every_time():
    """Query-independence is the requirement at index time, where it is a defect in a preview.

    An offer that varied per reader would mean an artifact was findable by one caller's terms and not
    another's, from the same stored index — and nothing about the index write knows who will read it.
    """
    assert _body_offer(BODY) == _body_offer(BODY)


def test_a_short_body_announces_itself_entirely():
    """One window means nothing to cut against, and `dense_windows` returns everything — which is
    right rather than a degenerate case: a two-line body IS its own densest span."""
    short = "Attenuation composes downward and never widens."
    assert _body_offer(short).strip() == short


@pytest.mark.parametrize("empty", ["", "   ", "\n\t ", None])
def test_nothing_to_project_returns_nothing(empty):
    """`""`, never a guess. An empty projection means the stated offer alone is indexed, which is
    exactly what happened before this existed."""
    assert _body_offer(empty or "") == ""


def test_the_body_is_not_one_of_the_stated_offer_fields():
    """The split is still real. `content` is added as a projection by `_sse_index_artifact`, not by
    being declared something the artifact states about itself — so a future reader of
    `_STATED_OFFER_FIELDS` cannot conclude the raw body is indexed."""
    assert "content" not in _STATED_OFFER_FIELDS
    assert _STATED_OFFER_FIELDS == ("title", "description", "tags")


def test_no_length_is_typed_anywhere_in_the_projection():
    """The extent is read off the document rather than chosen for it.

    Two bodies of very different length must not produce offers of the same length — that is the
    signature of a cap. `beacon.density`'s only typed number is its measurement grain (the window
    size); how many windows are kept is `top_break`'s answer, and the assembled length follows from
    it.
    """
    long_body = BODY * 6
    short_offer, long_offer = _body_offer(BODY), _body_offer(long_body)
    assert short_offer and long_offer
    assert len(short_offer) != len(long_offer), (
        f"both bodies projected to {len(short_offer)} characters — that is a cap, however it is "
        f"spelled, and the whole point is that the cut decides"
    )


# ── and it is actually findable ──────────────────────────────────────────────────────────────


def test_the_arm_actually_passes_the_projection_to_the_index(tmp_path):
    """Pins the decision rather than the mechanism. `SseIndexer` indexes whatever `content` it is
    handed, so a test that calls `_body_offer` itself and passes the result to the indexer passes
    whether or not this arm hands it anything. `_fields_to_index` exists to make the choice
    reachable, and this asserts the arm makes it.
    """
    given = _fields_to_index({
        "title": "untitled note", "description": "", "tags": "", "content": BODY,
    })
    assert "content" in given, (
        "the arm dropped the body projection — a term only in the body is unfindable again"
    )
    assert given["content"] == _body_offer(BODY)
    assert given["content"] != BODY, "the RAW body reached the index"
    assert given["title"] == "untitled note", "the stated offer must still go through"


def test_a_body_with_nothing_to_project_indexes_the_stated_offer_alone(tmp_path):
    """The `""` case wired end to end: no projection means what happened before it existed."""
    assert _fields_to_index({"title": "just a title", "content": ""}) == {"title": "just a title"}
    assert _fields_to_index({"content": ""}) == {}, "nothing to index at all"


def test_a_term_only_in_the_body_becomes_findable(tmp_path):
    """A term carried only by the body is reachable through the ordinary narrowing.

    The read path probes the `content` field on every stem, and the projection is what writes it, so
    a term that appears in the body alone — not in the title, description or tags — finds the
    artifact.
    """
    from mantle.search.mantle.sse.indexer import SseIndexer
    from mantle.search.mantle.sse.narrowing import TokenNarrower
    from mantle.search.mantle.sse.sqlite_stores import SqlitePostingStore

    class _Oracle:
        def derive_sse_key(self, principal_id, request):     # noqa: ANN001
            return (principal_id.encode("utf-8") * 32)[:32]

    oracle = _Oracle()
    store = SqlitePostingStore(str(tmp_path / "sse.db"))
    try:
        offer = _body_offer(BODY)
        assert "semilattice" in offer, "fixture assumption: the term is in the projection"

        SseIndexer(oracle, store).index_artifact(
            "owner-1", "coll-1", "art-1",
            # A title that shares NOTHING with the query, so a hit can only come from the body.
            {"title": "untitled note", "content": offer},
            None,
        )
        narrower = TokenNarrower(oracle, store)
        found = dict(narrower.lookup_for("semilattice", None)([("owner-1", "coll-1")]))
        assert "art-1" in found, (
            "a term present only in the body is still unreachable — the projection is written but "
            "the read path is not finding it"
        )
        # And the control: a term in neither the title nor the projection still misses.
        assert dict(narrower.lookup_for("nothingmatchesthis", None)(
            [("owner-1", "coll-1")])) == {}
    finally:
        store.close()


def test_indexing_a_body_costs_terms_in_the_projection_not_in_the_body(tmp_path):
    """The cost argument, closed on its own terms.

    `_sse_index_artifact` records "cost is terms, not bytes": 4 KB of real prose took 16.4s while 4 KB
    of ``'x '`` took 3.5s, because the first carries thousands of distinct stems and the second one.
    So the measurement that matters is how many distinct slots a write touches — and with the
    projection that is a function of the CUT's extent, not of the body's length.

    Asserted by writing the same distinctive prose buried in six times as much filler: the filler adds
    length and almost no new terms, so a slot count that tracked the body would grow and one that
    tracks the projection should not grow anything like proportionally.
    """
    from mantle.search.mantle.sse.indexer import SseIndexer
    from mantle.search.mantle.sse.sqlite_stores import SqlitePostingStore

    class _Oracle:
        def derive_sse_key(self, principal_id, request):     # noqa: ANN001
            return (principal_id.encode("utf-8") * 32)[:32]

    def _slots(body, aid, path):
        store = SqlitePostingStore(str(path))
        try:
            SseIndexer(_Oracle(), store).index_artifact(
                "owner-1", "coll-1", aid, {"content": _body_offer(body)}, None)
            return len(store.list_tokens_for_owner("owner-1"))
        finally:
            store.close()

    small = _slots(BODY, "art-1", tmp_path / "a.db")
    large = _slots(BODY * 6, "art-2", tmp_path / "b.db")
    assert small > 0, "nothing indexed — the comparison would be vacuous"
    assert large < small * 3, (
        f"a body six times longer wrote {large} slots against {small} — the indexed extent is "
        f"tracking the body rather than the cut, which is the cost the exclusion existed to avoid"
    )
