"""`list_artifacts(content_type=…)` must find every match a caller may read, not the ones that
happen to sort early.

The filter runs before paging, never after. Filtering within the hydrated page instead returns
whichever artifacts of that type happen to fall inside the first `limit` ids of a set sorted by id,
which at scale is almost none of them — measured on a live store of 10,393 authorized artifacts:

    list_artifacts(sensor+json, limit=3)       ->  0 of 4     total 10393
    list_artifacts(sensor+json, limit=1000)    ->  1 of 4     total 10393
    list_artifacts(host+json,   limit=50)      ->  0 of 1     total 10393
    list_artifacts(host+json,   offset=10300)  ->  host.71

The blast radius is the HTTP/MCP surface only — `list_visible`, and every caller reaching it
through `/artifacts` or the `list_artifacts` MCP tool. `agience-ember/src/ember/surface/console.py`
calls a function of the same name one layer down, which filters in SQL on `ix_v_ct` and is
unaffected: two functions sharing a name across a transport boundary is a mismatch worth checking
before assuming a shared blast radius.

`"host.71"` sorting at ~10,300 of 10,393 is why it was invisible over MCP: `h` orders after every
hex-leading UUID.

Filtering first does not mean hydrating first: `vertex.list_artifacts(content_type=…)` narrows on
the `ix_v_ct` index, so the candidate set is proportional to how many artifacts carry the type
rather than to how much the caller can see, and the fix is a set intersection.

`total` must be the filtered count, not the size of the caller's whole authorized set — a number
that reads as an answer to "how many matched" has to actually be one.
"""
from __future__ import annotations

import pytest

from mantle.db import open_lattice
from mantle.routers.artifacts_router import _ids_of_content_type

SENSOR = "application/vnd.agience.sensor+json"
HOST = "application/vnd.agience.host+json"


@pytest.fixture()
def lattice(tmp_path):
    return open_lattice(str(tmp_path / "lattice.db"), origin="test")


def _populate(lattice, noise=500):
    """A few typed artifacts buried in a lot of untyped ones, with ids that sort badly.

    The ids are chosen to reproduce the real failure: a hex-prefixed id sorts last because `h`
    orders after every hex character, and `zzz-host` does the same here. A test whose matching
    rows sort early would pass against a filter-after-paging implementation.
    """
    for i in range(noise):
        lattice.artifacts.put_artifact({"id": "%08x-noise" % i, "content_type": "text/plain"})
    lattice.artifacts.put_artifact({"id": "zzz-host", "content_type": HOST})
    lattice.artifacts.put_artifact({"id": "zzy-sensor-a", "content_type": SENSOR})
    lattice.artifacts.put_artifact({"id": "0001-sensor-b", "content_type": SENSOR})
    return noise


def test_the_index_lookup_finds_matches_wherever_they_sort(lattice):
    """The property the fix rests on: if this cannot find a late-sorting id, nothing downstream
    can either."""
    _populate(lattice)
    assert _ids_of_content_type(lattice, HOST) == {"zzz-host"}
    assert _ids_of_content_type(lattice, SENSOR) == {"zzy-sensor-a", "0001-sensor-b"}


def test_the_lookup_cost_does_not_scale_with_what_the_caller_can_see(lattice):
    """Why filter-before-page is affordable: the candidate set is proportional to how many
    artifacts carry the type, not to how many exist. Ten times the noise must not change the
    result or the work.
    """
    _populate(lattice, noise=2000)
    found = _ids_of_content_type(lattice, HOST)
    assert found == {"zzz-host"}, found


def test_archived_versions_are_not_listed_as_separate_artifacts(lattice):
    """A versioned artifact is one thing: it has a committed row and N archived snapshots sharing
    a root; listing them all would report the same property several times, once per past version.

    The sensor artifacts are exactly this shape — one per property, re-versioned on every real
    change — so this is not hypothetical for the caller that found the bug.
    """
    lattice.artifacts.put_artifact({"id": "s-1", "content_type": SENSOR, "state": "committed"})
    lattice.artifacts.put_artifact(
        {"id": "s-1@old", "root_id": "s-1", "content_type": SENSOR, "state": "archived"})
    assert _ids_of_content_type(lattice, SENSOR) == {"s-1"}


def test_a_type_nothing_carries_is_empty_not_everything(lattice):
    """The failure direction that would be worst: an empty narrow set must mean "no matches", not
    "no filter" — an intersection with the wrong empty value returns the caller's whole reach."""
    _populate(lattice)
    assert _ids_of_content_type(lattice, "application/vnd.nothing+json") == set()


def test_the_filter_is_applied_before_paging_in_the_router() -> None:
    """Asserted on the source, because the order is the bug: the tests above prove the lookup
    works, and this proves the router uses it in the right place. Reintroducing the old order
    would leave every assertion above passing.
    """
    import inspect
    import re

    from mantle.routers import artifacts_router

    src = inspect.getsource(artifacts_router.list_visible)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    narrow = code.index("_ids_of_content_type")
    page = code.index("sorted(authorized)[offset")
    assert narrow < page, (
        "the content_type narrowing happens AFTER the page is taken — that is the original defect: "
        "a caller gets whichever matches fall inside the first `limit` ids of everything it can read")
    assert re.search(r"authorized\s*=\s*authorized\s*&\s*typed", code), (
        "the narrowed set is not intersected into `authorized`, so `total` will still report the "
        "unfiltered count")


def test_has_more_is_computed_from_the_true_total() -> None:
    """A final page of exactly `limit` must not claim there is more.

    With the narrowing applied before the slice, `page` is a slice of the filtered set, so
    `has_more` must be computed from `total`, not from `len(page) == limit` — a full page is no
    longer proof that another one follows.

    The truncated-cone path genuinely has no `total` — counting it means building the set the
    truncation exists to avoid — so page length remains the signal there, and nowhere else.
    """
    import inspect
    import re

    from mantle.routers import artifacts_router

    code = "\n".join(ln for ln in inspect.getsource(artifacts_router.list_visible).splitlines()
                     if not ln.lstrip().startswith("#"))
    m = re.search(r"more\s*=\s*\(offset \+ limit < total\)\s*if total is not None", code)
    assert m, (
        "`has_more` is not computed from `total` — a final page of exactly `limit` items will "
        "report that another page follows:\n%s" % code[-600:])
    assert "len(page) == limit" in code, (
        "the page-length fallback is gone; the truncated-cone path has no `total` and needs it")


# ── the route itself, not just the helper ────────────────────────────────────────────────────────
#
# The tests above prove the parts: `_ids_of_content_type` finds late-sorting ids, and a source
# assertion pins the narrowing ahead of the page. Neither drives `list_visible`, so neither would
# notice if the two were wired together wrongly — that is proven at the route below
# (`list_artifacts(host+json, limit=50)` → 0 of 1 is the shape a wiring defect takes here).

def _drive(store_db, authorized, content_type, limit=5, offset=0):
    """Run the real handler with a known light cone and a stubbed hydrator.

    The hydration stub returns each id's own document, so what comes back is exactly what the
    paging-and-filtering logic selected — which is the thing under test.
    """
    import asyncio
    from unittest.mock import patch

    from mantle.routers import artifacts_router as ar
    from mantle.services.dependencies import AuthContext

    auth = AuthContext(principal_id="u", principal_type="user", user_id="u", bearer_grant=None)
    docs = {aid: {"id": aid, "content_type": ct} for aid, ct in authorized.items()}

    class _Resolver:
        def __init__(self, *a, **kw):
            pass

        def resolve(self, *a, **kw):
            return set(authorized)

    with patch("mantle.search.mantle.lightcone.LightConeResolver", _Resolver), \
            patch.object(ar, "_hydrate_batch", side_effect=lambda db, ids: {i: docs[i] for i in ids}), \
            patch.object(ar, "_ids_of_content_type",
                         side_effect=lambda db, ct: {i for i, c in authorized.items() if c == ct}):
        return asyncio.run(ar.list_visible(
            content_type=content_type, action="read", limit=limit, offset=offset,
            auth=auth, store_db=store_db))


def _cone(n_noise=400):
    """A light cone where the only typed artifacts sort last — the live shape.

    `host.71` sat at position ~10,300 of 10,393 because `"host.71"` begins with `h`, which orders
    after every hex-leading UUID. `zz*` reproduces that against `%08x` noise.
    """
    cone = {"%08x-noise" % i: "text/plain" for i in range(n_noise)}
    cone["zzz-host"] = HOST
    cone["zzy-sensor"] = SENSOR
    return cone


def test_the_route_finds_a_match_that_sorts_last() -> None:
    """The symptom, at the route: over 402 authorized ids with a match sorting at position 402,
    `limit=5` must still find it — filtering the first five by id instead would make that match
    unreachable at any sane page size."""
    body = _drive(None, _cone(), HOST, limit=5)
    ids = [d["id"] for d in body["items"]]
    assert ids == ["zzz-host"], (
        "asked for one content type over 402 authorized ids with limit=5 and got %r — the filter is "
        "being applied to the page instead of before it" % ids)


def test_the_route_reports_the_filtered_total() -> None:
    """`total` must be the filtered count — a number that reads as an answer to "how many matched"
    has to actually be one, not the size of the caller's whole light cone regardless of match."""
    body = _drive(None, _cone(), SENSOR, limit=5)
    assert body["total"] == 1, (
        "total is %r for a filter matching exactly one artifact — it is still counting the whole "
        "light cone" % body["total"])


def test_a_full_final_page_does_not_claim_there_is_more() -> None:
    """With the narrowing applied before paging, `page` is a slice of the filtered set — a full
    page of exactly `limit` filtered matches means there is no more, and `has_more=len(page) ==
    limit` would wrongly send a caller after a page that does not exist."""
    cone = {"a-%02d" % i: SENSOR for i in range(5)}          # exactly `limit` matches, no more
    body = _drive(None, cone, SENSOR, limit=5)
    assert len(body["items"]) == 5 and body["total"] == 5
    assert body["has_more"] is False, (
        "a final page of exactly `limit` items reports that another follows")


def test_an_unfiltered_listing_still_pages_normally() -> None:
    """The regression the narrowing could cause: with no `content_type`, nothing may be narrowed
    away — the whole light cone must still page normally."""
    cone = _cone(n_noise=10)
    body = _drive(None, cone, None, limit=4)
    assert len(body["items"]) == 4, [d["id"] for d in body["items"]]
    assert body["total"] == len(cone), (
        "an unfiltered listing reports %r of %d — the intersection ran when it should not have"
        % (body["total"], len(cone)))
    assert body["has_more"] is True
