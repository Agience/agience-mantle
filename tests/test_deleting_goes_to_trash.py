"""Delete moves an artifact to Trash; it is restorable there for 30 days and then deleted.

Runs on a real store with the real local index (the stack of `test_there_is_no_promotion_state`);
only key custody and the access check are stood in for.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from mantle.routers import artifacts_router as artifacts
from mantle.services import trash
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_there_is_no_promotion_state import (
    USER, OTHER, _auth, _container, _recall, _write_workspace_create, real_node,
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def _allowed(monkeypatch):
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)


def _note(db, monkeypatch, text="marram dune grass"):
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, text, monkeypatch)
    return inbox, root


async def _trash(db, root, user=USER):
    with acting_as(user, principal_type="user"):
        return await artifacts.trash_artifact(artifact_id=root, auth=_auth(user), store_db=db)


async def _list_trash(db, user=USER):
    with acting_as(user, principal_type="user"):
        return (await artifacts.list_trash(auth=_auth(user), store_db=db))["items"]


async def _children(db, container, state):
    with acting_as(USER, principal_type="user"):
        request = Request({"type": "http", "method": "GET", "headers": [], "path": "/",
                           "query_string": b""})
        page = await artifacts.list_children(
            artifact_id=container, request=request, state=state, content_type=None, limit=100, offset=0,
            workspace_id=None, auth=_auth(), store_db=db)
    return [c["root_id"] for c in page["items"]]


async def test_a_trashed_artifact_is_archived_marked_and_listed(db, monkeypatch):
    inbox, root = _note(db, monkeypatch)
    body = await _trash(db, root)

    assert body["state"] == "archived"
    assert body["trashed_by"] == USER
    trashed = datetime.fromisoformat(body["trashed_time"])
    assert datetime.fromisoformat(body["purge_after"]) - trashed == timedelta(days=30)

    assert [i["root_id"] for i in await _list_trash(db)] == [root]
    assert root not in _recall(db, "marram"), "a trashed artifact is still found by default"


async def test_trash_is_not_the_containers_archived_view(db, monkeypatch):
    inbox, root = _note(db, monkeypatch)
    _inbox2, kept = _note(db, monkeypatch, "sea holly")
    with acting_as(USER, principal_type="user"):
        await artifacts.update_artifact(artifact_id=kept,
                                        body=artifacts.UpdateArtifactRequest(state="archived"),
                                        auth=_auth(), store_db=db)
    await _trash(db, root)
    assert root not in await _children(db, inbox, "archived")
    assert root not in await _children(db, inbox, "live")


async def test_restore_takes_it_out_of_trash(db, monkeypatch):
    inbox, root = _note(db, monkeypatch)
    await _trash(db, root)
    with acting_as(USER, principal_type="user"):
        restored = await artifacts.update_artifact(
            artifact_id=root, body=artifacts.UpdateArtifactRequest(state="live"),
            auth=_auth(), store_db=db)
    assert restored["state"] == "live"
    assert "trashed_by" not in restored and "trashed_time" not in restored
    assert await _list_trash(db) == []
    assert root in await _children(db, inbox, "live")
    assert root in _recall(db, "marram")


async def test_an_archived_artifact_goes_to_trash_as_it_is(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)
    with acting_as(USER, principal_type="user"):
        await artifacts.update_artifact(artifact_id=root,
                                        body=artifacts.UpdateArtifactRequest(state="archived"),
                                        auth=_auth(), store_db=db)
    body = await _trash(db, root)
    assert body["state"] == "archived" and body["trashed_by"] == USER


async def test_trash_is_per_person(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)
    await _trash(db, root)
    assert await _list_trash(db, OTHER) == []


async def test_an_unknown_artifact_is_404(db):
    with pytest.raises(HTTPException) as e:
        await _trash(db, "no-such-artifact")
    assert e.value.status_code == 404


async def test_trash_needs_the_delete_right(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)

    def _refuse(auth, aid, action, store_db):
        assert action == "delete"
        raise HTTPException(status_code=404, detail="Not found")
    monkeypatch.setattr(artifacts, "check_access", _refuse)
    with pytest.raises(HTTPException):
        await _trash(db, root)
    assert db.artifacts.get_artifact(root)["state"] == "live"


async def test_the_index_follows_trash_and_restore(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)
    await _trash(db, root)
    assert [d["root_id"] for d in db.artifacts.lookup_by_list_field("trashed_by", USER)] == [root]
    with acting_as(USER, principal_type="user"):
        await artifacts.update_artifact(artifact_id=root,
                                        body=artifacts.UpdateArtifactRequest(state="live"),
                                        auth=_auth(), store_db=db)
    assert list(db.artifacts.lookup_by_list_field("trashed_by", USER)) == []


async def test_a_store_without_the_index_still_lists_trash(db, monkeypatch):
    from mantle.db.vertex import ListIndexUnbuilt
    _inbox, root = _note(db, monkeypatch)
    await _trash(db, root)

    def _unbuilt(*a, **k):
        raise ListIndexUnbuilt("not built")
    monkeypatch.setattr(type(db.artifacts), "lookup_by_list_field", _unbuilt)
    assert [i["root_id"] for i in await _list_trash(db)] == [root]


# ── the purge ────────────────────────────────────────────────────────────────────────────────

def _days_on(days):
    return datetime.now(timezone.utc) + timedelta(days=days)


async def test_nothing_is_purged_before_thirty_days(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)
    await _trash(db, root)
    assert trash.purge_expired(db, now=_days_on(29)) == {"deleted": 0, "refused": 0, "failed": 0}
    assert db.artifacts.versions_of(root)


async def test_after_thirty_days_it_is_deleted(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)
    await _trash(db, root)
    assert trash.purge_expired(db, now=_days_on(31))["deleted"] == 1
    assert not db.artifacts.versions_of(root), "a purged artifact left a version behind"
    assert await _list_trash(db) == []


async def test_the_purge_cannot_do_what_the_person_could_not(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)
    await _trash(db, root)

    def _no_longer_permitted(*a, **k):
        raise HTTPException(status_code=404, detail="Not found")
    monkeypatch.setattr(ws, "delete_artifact", _no_longer_permitted)
    assert trash.purge_expired(db, now=_days_on(31)) == {"deleted": 0, "refused": 1, "failed": 0}
    assert db.artifacts.versions_of(root)


async def test_a_restored_artifact_is_never_purged(db, monkeypatch):
    _inbox, root = _note(db, monkeypatch)
    await _trash(db, root)
    with acting_as(USER, principal_type="user"):
        await artifacts.update_artifact(artifact_id=root,
                                        body=artifacts.UpdateArtifactRequest(state="live"),
                                        auth=_auth(), store_db=db)
    assert trash.purge_expired(db, now=_days_on(31))["deleted"] == 0
    assert db.artifacts.get_artifact(root)["state"] == "live"


def test_the_trash_listing_is_not_taken_for_an_artifact_id():
    """`GET /artifacts/trash` is declared before `GET /artifacts/{artifact_id}`, which would
    otherwise answer it as the artifact named `trash`."""
    from mantle.main import app
    paths = [(r.path, sorted(getattr(r, "methods", []) or [])) for r in app.routes]
    trash_at = paths.index(("/artifacts/trash", ["GET"]))
    one_at = next(i for i, (p, m) in enumerate(paths) if p == "/artifacts/{artifact_id}" and "GET" in m)
    assert trash_at < one_at
