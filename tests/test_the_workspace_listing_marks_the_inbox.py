"""Mantle lists a person's workspaces oldest first and marks their Inbox.

The Inbox is the oldest workspace the person made. A workspace someone shared with them is never
their Inbox, however old it is. The listing carries the mark, so no client works it out.
"""
from __future__ import annotations

import pytest

from mantle.services import grant_service
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext
from mantle.services.seed_provisioning import user_provisioning as up

from .test_there_is_no_promotion_state import OTHER, USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


async def _listing(db, user):
    from mantle.routers import artifacts_router as artifacts
    with acting_as(user, principal_type="user"):
        page = await artifacts.list_visible(content_type=WORKSPACE_CT, action="read", order="created",
                                            limit=100, offset=0, auth=_auth(user), store_db=db)
    return page["items"]


def _make(db, user, name):
    with acting_as(user, principal_type="user"):
        return ws.create_container(db, user, content_type=WORKSPACE_CT, name=name).id


async def test_the_listing_is_oldest_first_and_marks_nothing_inbox(db):
    first = _make(db, USER, "Inbox")
    second = _make(db, USER, "Projects")
    third = _make(db, USER, "Notes")
    items = await _listing(db, USER)
    assert [i["id"] for i in items] == [first, second, third]
    assert not [i for i in items if "inbox" in i]


async def test_a_shared_workspace_is_never_the_inbox(db):
    older = _make(db, OTHER, "Team")
    with acting_as(OTHER, principal_type="user"):
        _, token = grant_service.create_invite(db, user_id=OTHER, resource_id=older, role="viewer")
    grant_service.claim_invite(db, USER, token, "")
    own = _make(db, USER, "Inbox")
    assert up.inbox_of(db, USER) == own
    items = await _listing(db, USER)
    assert [i["id"] for i in items] == [older, own]
    assert not [i for i in items if "inbox" in i]


async def test_no_own_workspace_means_no_inbox(db):
    shared = _make(db, OTHER, "Team")
    with acting_as(OTHER, principal_type="user"):
        _, token = grant_service.create_invite(db, user_id=OTHER, resource_id=shared, role="viewer")
    grant_service.claim_invite(db, USER, token, "")
    assert up.inbox_of(db, USER) is None
    assert not [i for i in await _listing(db, USER) if "inbox" in i]


async def test_created_order_needs_a_content_type(db):
    from fastapi import HTTPException

    from mantle.routers import artifacts_router as artifacts
    with pytest.raises(HTTPException) as refused:
        await artifacts.list_visible(content_type=None, action="read", order="created", limit=10,
                                     offset=0, auth=_auth(USER), store_db=db)
    assert refused.value.status_code == 400
