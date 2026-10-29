"""A workspace shared by invite is listed for the reader who claimed it, and opens for them.

2026-10-23, a tester: a Viewer invite to a workspace was claimed (an active user grant with `read`),
and the invitee's app then said the workspace "hasn't been shared with you". The app lists the
workspaces it can open through `/artifacts/visible`, so the claimed one must be in that list.
"""
from __future__ import annotations

import pytest

from mantle.services import grant_service
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, check_access
from mantle.services import workspace_service as ws

from .test_there_is_no_promotion_state import OTHER, USER, real_node

WORKSPACE_CT = "application/vnd.agience.workspace+json"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


async def _visible(db, user):
    from mantle.routers import artifacts_router as artifacts
    with acting_as(user, principal_type="user"):
        page = await artifacts.list_visible(content_type=WORKSPACE_CT, action="read", limit=100,
                                            offset=0, auth=_auth(user), store_db=db)
    return {item["id"] for item in page["items"]}


async def test_a_claimed_viewer_invite_lists_and_opens_the_workspace(db):
    with acting_as(USER, principal_type="user"):
        shared = ws.create_container(db, USER, content_type=WORKSPACE_CT, name="QA Workspace").id
        _, token = grant_service.create_invite(db, user_id=USER, resource_id=shared, role="viewer")
    assert shared not in await _visible(db, OTHER)
    grant_service.claim_invite(db, OTHER, token, "")
    assert shared in await _visible(db, OTHER), "a claimed workspace must be listed for its reader"
    with acting_as(OTHER, principal_type="user"):
        check_access(_auth(OTHER), shared, "read", db)
