"""`POST /artifacts/batch` reads each distinct id once.

The state this replaces. `_fetch_authorized_docs` looped over `artifact_ids` as given. The same
id three times therefore cost **six store operations** — the body is two per id, as the parameter's
own description says — and came back as three identical documents. Since the response is a page,
`total: 3` counted one artifact three times.

`max_length` bounded the cost of a HOSTILE list. It did nothing about a merely repetitive one,
which is the likelier shape: a caller assembling ids from several collections that share members.

`dict.fromkeys` rather than `set`: the page order is the caller's order, and a set would make the
response order arbitrary from one request to the next. The contract already allows fewer results
than ids — *"Ids you cannot read, and ids that do not exist, are SKIPPED … asking for 50 may yield
48"* — so one document per DISTINCT id needs no new caveat.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantle.routers import artifacts_router as ar


def _run(ids):
    """Call the real resolver with the store and the authorizer stubbed, counting the reads."""
    reads = []

    def _find(_db, aid):
        reads.append(aid)
        return {"id": aid}

    with (
        patch.object(ar, "_find_artifact", side_effect=_find),
        patch.object(ar, "check_access", return_value=None),
        patch.object(ar, "_normalize_artifact_doc", side_effect=lambda d: d),
    ):
        out = ar._fetch_authorized_docs(MagicMock(), MagicMock(), ids)
    return out, reads


def test_a_repeated_id_is_read_once_and_returned_once():
    out, reads = _run(["a", "a", "a"])
    assert reads == ["a"], "the store was read %d times for one distinct id: %r" % (len(reads), reads)
    assert [d["id"] for d in out] == ["a"], out


def test_the_callers_order_is_preserved():
    """A `set` would pass the test above and make the page order arbitrary between requests."""
    out, _ = _run(["c", "a", "b", "a", "c"])
    assert [d["id"] for d in out] == ["c", "a", "b"], out


def test_distinct_ids_are_all_still_fetched():
    """The positive control: dedup must not be indistinguishable from dropping work."""
    out, reads = _run(["a", "b", "c"])
    assert reads == ["a", "b", "c"], reads
    assert [d["id"] for d in out] == ["a", "b", "c"], out


def test_every_distinct_id_is_still_authorized_individually():
    """The batch route must not become an existence oracle — authorization stays per id."""
    checked = []
    with (
        patch.object(ar, "_find_artifact", side_effect=lambda _db, aid: {"id": aid}),
        patch.object(ar, "check_access", side_effect=lambda _a, aid, _act, _db: checked.append(aid)),
        patch.object(ar, "_normalize_artifact_doc", side_effect=lambda d: d),
    ):
        ar._fetch_authorized_docs(MagicMock(), MagicMock(), ["a", "b", "a"])
    assert checked == ["a", "b"], checked
