""": authorize the PAGE, and report `total: null`.

THE COST THAT WAS REMOVED. `_readable_members` ran over the whole member list before the page
was taken, and `check_access` is several queries PLUS an access-audit write per decision. One page
of an N-member container therefore performed N authorization decisions and N audit writes — 10,000
of each to return 100 rows from a 10,000-child container.

THE TRADE IS REAL AND WAS ACCEPTED, NOT OVERLOOKED. Authorizing after paging means a page can
come back short, and its length discloses by arithmetic how many members on THAT PAGE the caller
may not read. What bounds it is the ruling's other half: `total` is `null`, so the page length is
the only signal and it is bounded by `limit` rather than by the container.
"""
from __future__ import annotations

import ast
import io

from mantle.main import app
from mantle.routers import artifacts_router


def _list_children_src():
    tree = ast.parse(io.open(artifacts_router.__file__, encoding="utf-8").read())
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "list_children":
            return n
    raise AssertionError("list_children not found")


def test_the_page_is_taken_before_the_members_are_authorized():
    """The whole ruling in one assertion: the slice must precede the authorization call."""
    node = _list_children_src()
    slice_line = auth_line = None
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript) and isinstance(sub.slice, ast.Slice):
            if slice_line is None or sub.lineno < slice_line:
                slice_line = sub.lineno
        if (isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "offload_sync"
                and sub.args and getattr(sub.args[0], "id", "") == "_readable_members"):
            auth_line = sub.lineno
    assert slice_line and auth_line, (slice_line, auth_line)
    assert slice_line < auth_line, (
        "members are authorized at line %d before the page is sliced at %d — the cost is back to "
        "scaling with the container" % (auth_line, slice_line))


def test_total_is_reported_as_unknown():
    """`total` must be `None`. A number here is only obtainable by authorizing every member, which
    is exactly the cost the ruling removed."""
    node = _list_children_src()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and getattr(sub.func, "id", "") == "_page":
            kw = {k.arg: k.value for k in sub.keywords}
            assert isinstance(kw["total"], ast.Constant) and kw["total"].value is None, (
                "`total` is %s — authorizing every member to compute it is the defect"
                % ast.dump(kw["total"])[:60])
            return
    raise AssertionError("list_children no longer calls _page")


def test_the_page_is_filled_so_a_short_page_cannot_count_what_is_hidden():
    """THE SECURITY PROPERTY, and the reason "authorize the page" is not a one-line slice.

    Authorizing a single slice returns a SHORT page whose shortfall counts the members the caller
    may not read — the same existence oracle this route has always refused, arrived at by
    arithmetic, one page at a time. `tests/test_children_authorize_every_member.py` fails on that
    version, which is how it was caught rather than shipped.

    The loop fills to `limit + 1` READABLE rows. The extra row is what makes `has_more` mean
    "another member you may read exists" instead of "more members exist" — the weaker form answered `true` and then served an empty page, which is the same disclosure wearing a
    different code."""
    node = _list_children_src()
    loops = [n for n in ast.walk(node) if isinstance(n, ast.While)]
    assert loops, "the page is no longer filled — a single slice returns short pages"

    body = ast.unparse(loops[0])
    assert "_readable_members" in body, "the fill loop does not authorize"
    # `<= limit` is the extra row; `< limit` would stop one short and break `has_more`.
    assert "<= limit" in ast.unparse(loops[0].test), (
        "the loop fills to `limit`, not `limit + 1`, so `has_more` cannot tell whether another "
        "READABLE member exists: %s" % ast.unparse(loops[0].test))


def test_the_published_description_says_total_is_null_and_how_to_page():
    """The ruling's own build note: *"`total` is exact today, so the change belongs in the
    endpoint description as well as the code."*

    It was worse than missing. The published description asserted the OPPOSITE on three counts —
    that cost scales with the container, that every member is authorized before the page is taken,
    and that `total` is "the count you MAY read". All three were true when written and all three
    were made false by this change, so the contract a client reads was actively wrong."""
    d = app.openapi()["paths"]["/artifacts/{artifact_id}/children"]["get"]["description"]
    assert "null" in d and "total" in d, d[:200]
    assert "COST SCALES WITH THE CONTAINER" not in d, (
        "the superseded cost claim is back in the published contract")
    assert "has_more" in d, "the description does not tell a caller how to page without `total`"
