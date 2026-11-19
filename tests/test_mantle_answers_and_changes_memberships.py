"""POST /artifacts/memberships: Mantle says which containers hold all or some of the artifacts,
and adds or removes them, each change checked as its own call would be."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from mantle.routers import artifacts_router as ar
from mantle.services import workspace_service
from mantle.services.dependencies import AuthContext

_AUTH = AuthContext(principal_id="u", principal_type="user", user_id="u", bearer_grant=None)


def _drive(state: dict, **body):
    """Run the handler over an in-memory map of artifact -> containers holding it.

    Adding to "locked" is refused with 403; removing the last holder archives the artifact."""
    def fetch(_db, _auth, ids, include_archived=False):
        return [{"id": i, "root_id": i, "container_ids": sorted(state[i])} for i in ids if i in state]

    async def link(req, _auth, _db, report=None):
        if req.container_id == "locked":
            raise HTTPException(status_code=403, detail="No create on locked")
        state[req.source_artifact_id].add(req.container_id)

    def detach(_db, _uid, container_id, artifact_id, _auth):
        state[artifact_id].discard(container_id)
        return SimpleNamespace(state="archived" if not state[artifact_id] else "live")

    with patch.object(ar, "_fetch_authorized_docs", side_effect=fetch), \
            patch.object(ar, "_attach_memberships"), \
            patch.object(ar, "_default_create_artifact", side_effect=link), \
            patch.object(workspace_service, "remove_artifact_from_container", side_effect=detach):
        return asyncio.run(ar.change_memberships(
            ar.MembershipRequest(**body), auth=_AUTH, store_db=MagicMock()))


def test_with_no_change_it_answers_all_and_some():
    held = {"a": {"c1", "c2"}, "b": {"c1"}}
    out = _drive(held, artifact_ids=["a", "b"])
    assert out["all"] == ["c1"]
    assert out["some"] == ["c2"]
    assert out["failed"] == [] and out["archived"] == []


def test_adding_puts_every_artifact_in_and_skips_those_already_there():
    held = {"a": {"c1"}, "b": set()}
    out = _drive(held, artifact_ids=["a", "b"], add=["c1"])
    assert held == {"a": {"c1"}, "b": {"c1"}}
    assert out["all"] == ["c1"] and out["some"] == []


def test_removing_the_last_holder_reports_the_artifact_archived():
    held = {"a": {"c1"}, "b": {"c1", "c2"}}
    out = _drive(held, artifact_ids=["a", "b"], remove=["c1"])
    assert out["archived"] == ["a"]
    assert out["all"] == [] and out["some"] == ["c2"]


def test_a_refused_change_is_listed_and_the_rest_go_ahead():
    held = {"a": set(), "b": {"c1"}}
    out = _drive(held, artifact_ids=["a", "b"], add=["locked", "c2"], remove=["c1"])
    assert {(f["artifact_id"], f["container_id"]) for f in out["failed"]} == {("a", "locked"), ("b", "locked")}
    assert held == {"a": {"c2"}, "b": {"c2"}}
    assert out["all"] == ["c2"]


def test_an_artifact_the_caller_cannot_read_is_left_out():
    held = {"a": {"c1"}}
    out = _drive(held, artifact_ids=["a", "hidden"], add=["c2"])
    assert [i["id"] for i in out["items"]] == ["a"]
    assert held == {"a": {"c1", "c2"}}


def test_held_is_the_end_result_and_a_mixed_one_stays_unless_cleared():
    held = {"a": {"c1", "c2", "c3", "c5"}, "b": {"c1", "c5"}}
    out = _drive(held, artifact_ids=["a", "b"], held=["c1", "c4"], cleared=["c2"])
    assert held == {"a": {"c1", "c3", "c4"}, "b": {"c1", "c4"}}
    assert out["all"] == ["c1", "c4"] and out["some"] == ["c3"]


def test_held_empty_removes_every_container():
    held = {"a": {"c1", "c2"}}
    out = _drive(held, artifact_ids=["a"], held=[])
    assert held == {"a": set()} and out["archived"] == ["a"]
