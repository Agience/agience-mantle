"""A link is listed from both ends: `GET /artifacts/{id}/links` answers what an artifact links
(`out`) and what links to it (`in`), filtered by read, and a link confers nothing.

FR-043 part 1 (2026-11-16): an issue that blocks another was visible only from the blocking side.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi import HTTPException

from mantle.db import backend as store
from mantle.db import lattice_api as la
from mantle.db.edge import EdgesTruncated
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_typed_link_names_what_it_links import _auth, _create, _list
from .test_there_is_no_promotion_state import OTHER, USER, real_node


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


async def _links(db, artifact_id, user=USER, direction="both", relationship=None, limit=100, offset=0):
    with acting_as(user, principal_type="user"):
        return await artifacts.list_links(artifact_id=artifact_id, direction=direction,
                                          relationship=relationship, limit=limit, offset=offset,
                                          store_db=db, auth=_auth(user))


def _rows(page):
    return sorted((i.get("root_id") or i.get("id"), i["relationship"], i["direction"]) for i in page["items"])


def _grant_read(db, resource_id, user=OTHER):
    from mantle.attenuation import ACTIONS
    from mantle.entities.grant import Grant
    la.create_grant(db, Grant(id=str(uuid.uuid4()), resource_id=resource_id, grantee_type="user",
                              grantee_id=user, granted_by=USER, effect="allow", state=Grant.STATE_ACTIVE,
                              **{f"can_{a}": a == "read" for a in ACTIONS}))


async def _notes(db, n):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json",
                                    name="Issues").id
    return [(await _create(db, container_id=space, context=json.dumps({"title": f"issue {i}"}),
                           content="x"))["id"] for i in range(n)]


async def _link(db, holder, target, name):
    await _create(db, container_id=holder, source_artifact_id=target, relationship=name)


async def test_a_link_is_listed_from_both_ends_and_held_once(db):
    a, b = await _notes(db, 2)
    await _link(db, a, b, "blocks")
    assert _rows(await _links(db, b, direction="in")) == [(a, "blocks", "in")]
    assert _rows(await _links(db, a, direction="out")) == [(b, "blocks", "out")]
    assert _rows(await _links(db, a)) == [(b, "blocks", "out")], "listed once"
    named = [e for e in db.graph.edges_of(b, direction="in") if e["label"] != "contains"]
    assert len(named) == 1, "the store holds one edge"


async def test_a_source_the_reader_may_not_read_is_neither_shown_nor_counted(db):
    a, b = await _notes(db, 2)
    await _link(db, a, b, "blocks")
    _grant_read(db, b)
    page = await _links(db, b, user=OTHER, direction="in")
    assert page["items"] == [] and page["has_more"] is False and page["total"] is None


async def test_the_page_is_filled_with_readable_rows(db):
    a, c, b = await _notes(db, 3)
    await _link(db, a, b, "blocks")
    await _link(db, c, b, "blocks")
    _grant_read(db, b)
    _grant_read(db, c)
    page = await _links(db, b, user=OTHER, direction="in", limit=1)
    assert _rows(page) == [(c, "blocks", "in")]
    assert page["has_more"] is False, "the hidden source is not counted"


async def test_a_reader_without_read_gets_the_refusal_children_gives(db):
    a, b = await _notes(db, 2)
    await _link(db, a, b, "blocks")
    with pytest.raises(HTTPException) as links:
        await _links(db, b, user=OTHER)
    with pytest.raises(HTTPException) as children:
        await _list(db, b, user=OTHER, links="*")
    assert links.value.status_code == children.value.status_code


async def test_names_match_by_token_and_by_dotted_prefix(db):
    a, b = await _notes(db, 2)
    await _link(db, a, b, "step1.collection_id")
    assert _rows(await _links(db, b, direction="in", relationship="step1")) == [(a, "step1.collection_id", "in")]
    assert _rows(await _links(db, b, relationship="step1.collection_id")) == [(a, "step1.collection_id", "in")]
    assert _rows(await _links(db, b, relationship="step1.collection")) == []
    assert _rows(await _links(db, b, relationship="step2")) == []
    labels = {e["label"] for e in db.graph.edges_of(b, direction="in")}
    assert "step1.collection_id" not in labels, "the stored label is a token, never the name"


async def test_containment_and_context_are_not_links(db):
    a, b = await _notes(db, 2)
    store.add_artifact_to_collection(db, a, b)                              # containment
    store.add_artifact_to_collection(db, a, b, relationship="context")
    assert _rows(await _links(db, a)) == []
    assert _rows(await _links(db, b)) == []


async def test_an_archived_or_trashed_source_is_not_listed(db):
    a, c, b = await _notes(db, 3)
    await _link(db, a, b, "blocks")
    await _link(db, c, b, "blocks")
    assert store.archive_artifact(db, USER, a)
    assert store.archive_artifact(db, USER, c)
    from mantle.services import trash
    assert trash.mark(db, USER, c) is not None
    assert _rows(await _links(db, b, direction="in")) == []


async def test_children_links_and_out_links_agree(db):
    a, b, c, d = await _notes(db, 4)
    await _link(db, a, b, "blocks")
    await _link(db, a, c, "step1.collection_id")
    await _link(db, d, a, "blocks")                                        # inbound: not in `out`
    store.add_artifact_to_collection(db, a, d)                              # a member: not a link
    out = {(r[0], r[1]) for r in _rows(await _links(db, a, direction="out"))}
    assert out == {(i.get("root_id") or i.get("id"), i["relationship"])
                   for i in (await _list(db, a, links="*"))["items"]}
    assert out == {(b, "blocks"), (c, "step1.collection_id")}


async def test_more_edges_than_the_cap_refuse_and_never_clip(db, monkeypatch):
    a, b = await _notes(db, 2)
    await _link(db, a, b, "blocks")

    def _truncated(*_a, **_k):
        raise EdgesTruncated("more edges than the cap")
    monkeypatch.setattr(db.graph, "edges_of", _truncated)
    with pytest.raises(EdgesTruncated):
        await _links(db, a)
    with pytest.raises(EdgesTruncated):
        await _list(db, a, links="*")


async def test_a_link_confers_nothing(db):
    a, b = await _notes(db, 2)
    await _link(db, a, b, "blocks")
    _grant_read(db, a)
    assert _rows(await _links(db, a, user=OTHER)) == [], "reading A does not reach what A links"
    with pytest.raises(HTTPException):
        await _links(db, b, user=OTHER)
    edge = next(e for e in db.graph.edges_of(b, direction="in") if e["label"] != "contains")
    assert la._prop_mask(edge) == [], "the link's mask is blocked"
