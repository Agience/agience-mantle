"""`GET /artifacts/{id}/children` authorizes every member, not just the container.

The route used to check `read` on the container and then return each member's whole document —
**including its decrypted body**. `list_collection_artifacts` (`db/lattice_api.py:565-601`) builds
the full doc and calls `decrypt_artifact_content(strict=False)` at :593, and that decryption
succeeds for anyone holding the collection, because `content_key_scope` keys a member by its
COLLECTION (`db/doc_boundary.py:111-113`) while the read gate is per artifact.

So the two answers disagreed inside one second:

    GET /artifacts/C/children  →  200, member A's plaintext in the body
    GET /artifacts/A           →  404, for the same caller

Two member shapes reach that hole, and both are ordinary:

  * an explicit ``deny{read}`` naming the member — never consulted, because nothing asked;
  * a member linked in by ``_link_source_artifact`` with ``origin=False, propagate=[]``, so no
    grant reaches it at all. That edge shape exists precisely to carry no authority, and the
    route published its contents anyway.

MCP `get_children` calls `list_children` directly (`routers/mcp_router.py:452-458`), so it had the
same hole and is closed by the same filter — which is why these tests call the handler rather than
the HTTP surface.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mantle.routers import artifacts_router


CONTAINER = "col-1"
READABLE = "art-readable"
DENIED = "art-denied"


def _docs():
    """What the store hands the route: whole documents, bodies already decrypted."""
    return [
        {"id": READABLE, "content_type": "text/plain", "content": "public body",
         "collection_id": CONTAINER},
        {"id": DENIED, "content_type": "text/plain", "content": "SECRET BODY",
         "collection_id": CONTAINER},
    ]


@pytest.fixture()
def route(monkeypatch):
    """The handler with its store and enrichment stubbed — authorization is the subject here."""
    monkeypatch.setattr(artifacts_router.store, "list_collection_artifacts",
                        lambda db, aid, **kw: _docs())
    monkeypatch.setattr(artifacts_router, "_normalize_artifact_doc", lambda d: d)
    import mantle.services.collection_service as cser
    monkeypatch.setattr(cser, "attach_committed_collection_ids", lambda db, ents: None)
    return artifacts_router.list_children


def _gate(*allowed: str):
    """`check_access` that permits exactly *allowed* and 404s everything else, as the real one does."""
    def _check(auth, artifact_id, action, db, *a, **kw):
        if artifact_id not in allowed:
            raise HTTPException(status_code=404, detail="Not found")
        return True
    return _check


async def _call(route, gate):
    """The route's rows.

    `/children` returns `{items, total, has_more}` since 2026-08-25 (P-6/P-7). The envelope is
    asserted here rather than in each test, so every test below keeps reading a plain list while
    the shape itself stays covered — and a regression that dropped the envelope fails loudly here
    instead of silently changing what these tests iterate.
    """
    with patch.object(artifacts_router, "check_access", gate):
        body = await route(
            artifact_id=CONTAINER, request=MagicMock(), content_type=None,
            workspace_id=None, limit=100, offset=0,
            auth=MagicMock(), store_db=MagicMock(),
        )
    assert set(body) == {"items", "total", "has_more"}, "the list envelope changed: %r" % (body,)
    # `total` is `None` here by design. An authorized count is knowable only by authorizing every
    # member before the page is taken, and `check_access` is several queries plus an audit write per
    # decision, so reporting one would make a page of an N-member container perform N of each.
    #
    # The assertion is not weakened, it is RE-AIMED: an authorized count is exactly the
    # thing this route may no longer compute, so requiring one would require the defect back.
    assert body["total"] is None, (
        "`total` is %r; it must be null — a number here is only obtainable by authorizing "
        "every member, which is the cost H3 removed" % (body["total"],))
    return body["items"]


@pytest.mark.asyncio
async def test_a_member_the_caller_may_not_read_is_not_returned(route):
    """The whole finding, as the property it broke: reading the container is not reading the
    members, and a member whose own gate refuses must not appear — with or without its body."""
    children = await _call(route, _gate(CONTAINER, READABLE))

    ids = [c["id"] for c in children]
    assert READABLE in ids
    assert DENIED not in ids, "a member the caller cannot read was returned by the container route"


@pytest.mark.asyncio
async def test_the_secret_body_does_not_appear_anywhere_in_the_response(route):
    """Belt and braces on the same call: the defect was a CONTENT disclosure, so assert on the
    bytes rather than only on the id. A future enrichment that re-attaches a member's document by
    another route would pass the test above and fail this one."""
    children = await _call(route, _gate(CONTAINER, READABLE))
    assert "SECRET BODY" not in repr(children)


@pytest.mark.asyncio
async def test_a_container_whose_members_are_all_unreadable_returns_empty_not_403(route):
    """The empty-result contract: "nothing you may see" and "an empty container" are the same
    answer. A 403 here would report the members' existence, which is the oracle the batch route's
    `_fetch_authorized_docs` is written to avoid."""
    children = await _call(route, _gate(CONTAINER))
    assert children == []


@pytest.mark.asyncio
async def test_the_filter_runs_before_the_page_is_taken(monkeypatch, route):
    """Filtering after paging would return short pages, and the shortfall counts exactly how many
    members the caller may not see — the same oracle, arrived at by arithmetic. With one readable
    member out of two and a limit of one, a correct route returns that one member."""
    children = await _call(route, _gate(CONTAINER, READABLE))
    assert [c["id"] for c in children] == [READABLE]

    with patch.object(artifacts_router, "check_access", _gate(CONTAINER, READABLE)):
        paged = await route(
            artifact_id=CONTAINER, request=MagicMock(), content_type=None,
            workspace_id=None, limit=1, offset=0,
            auth=MagicMock(), store_db=MagicMock(),
        )
    assert [c["id"] for c in paged["items"]] == [READABLE], \
        "the readable member must survive a page of one; a post-paging filter would drop it"
    #: One readable member of two, and the page holds it — so the count is 1, not 2.
    #: `total` reports the AUTHORIZED set, never the container's real size; reporting 2
    #: here would leak the unreadable member by arithmetic, which is the oracle this
    #: whole file exists to refuse.
    # `total` is null, so what this test pins is the rows: the caller sees exactly the member it
    # may read and not the one it may not.
    assert paged["total"] is None, "an authorized count must not be computed"
    assert len(paged["items"]) == 1, "the caller must still see exactly what it may read"
    assert paged["has_more"] is False, "one authorized member, page of one: last page"
