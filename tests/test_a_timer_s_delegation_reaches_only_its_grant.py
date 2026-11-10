"""A timer's delegation reaches only what its job grant names (John, 2026-11-10, BUG-024/BUG-035).

A timer runs on an Origin job grant: the delegations Origin mints for its firings, and every one
exchanged from them, carry the grant's reach (`job_reach`). Mantle holds such a delegation to that
reach as a light cone, exactly as a grant over those ids would reach, on the routes a timer's work
calls, for reading, writing, adding and running only, and never past what the person holds there.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as D
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_card_shared_alone_opens_for_its_reader import _create
from . import test_a_card_shared_alone_opens_for_its_reader as _card
from .test_there_is_no_promotion_state import OTHER, USER

db = _card.db

WORKSPACE = "application/vnd.agience.workspace+json"
COLLECTION = "application/vnd.agience.collection+json"


def _timer_auth(user, reach):
    return D.AuthContext(principal_id=user, principal_type="user", user_id=user,
                         actor="agience-server-lumen", delegation_aud="mantle", narrowed=True,
                         scope="persona:lumen", job_reach=tuple(reach))


def _allowed(db, auth, artifact_id, action):
    try:
        D.check_access(auth, artifact_id, action, db, consult_origin=False)
        return True
    except HTTPException:
        return False


async def _space(db):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type=WORKSPACE, name="Space").id
    runs = (await _create(db, container_id=space, content_type=COLLECTION, name="Runs"))["id"]
    timer = (await _create(db, container_id=space, content_type="text/markdown", content="t"))["id"]
    other = (await _create(db, container_id=space, content_type="text/markdown", content="o"))["id"]
    record = (await _create(db, container_id=runs, content_type="text/markdown", content="r"))["id"]
    return space, runs, timer, other, record


async def test_a_timer_reaches_its_items_and_what_they_hold_and_nothing_else(db):
    space, runs, timer, other, record = await _space(db)
    auth = _timer_auth(USER, (timer, runs))
    assert _allowed(db, auth, timer, "read") and _allowed(db, auth, timer, "update")
    assert _allowed(db, auth, runs, "create") and _allowed(db, auth, record, "read")
    assert not _allowed(db, auth, other, "read")                      # in the space, not reached
    assert not _allowed(db, auth, space, "read")                      # the space itself
    for action in ("delete", "evict", "share", "admin"):              # never these
        assert not _allowed(db, auth, timer, action), action
    # The person, on their own token, still reaches all of it.
    person = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    assert _allowed(db, person, other, "read") and _allowed(db, person, timer, "delete")


async def test_a_reach_never_passes_what_the_person_holds(db):
    _, _, timer, _, _ = await _space(db)
    assert not _allowed(db, _timer_auth(OTHER, (timer,)), timer, "read")
    assert not _allowed(db, _timer_auth(USER, ()), timer, "read")     # an empty reach: nothing


def _route(auth, method, path):
    try:
        D.enforce_job_reach_routes(auth, NS(method=method, url=NS(path=path)))
        return True
    except HTTPException:
        return False


def test_a_timer_s_delegation_reaches_only_the_routes_its_work_calls():
    auth = _timer_auth(USER, ("t",))
    for m, p in [("GET", "/artifacts/t"), ("PATCH", "/artifacts/t"), ("POST", "/artifacts"),
                 ("GET", "/artifacts/t/children"), ("POST", "/artifacts/t/op/invoke"),
                 ("GET", "/artifacts/platform-operators"), ("POST", "/artifacts/batch"),
                 # Crystal's gate before it runs the timer's target (BUG-041); `check_access`
                 # answers it, inside the reach.
                 ("GET", "/grants/my-access")]:
        assert _route(auth, m, p), (m, p)
    for m, p in [("GET", "/artifacts/visible"), ("POST", "/artifacts/recall"),
                 ("DELETE", "/artifacts/t"), ("POST", "/grants"),
                 ("GET", "/events"), ("POST", "/artifacts/search")]:
        assert not _route(auth, m, p), (m, p)
    plain = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    assert _route(plain, "GET", "/artifacts/visible")                 # no reach, no bound


def test_the_reach_is_read_from_origin_s_claim():
    assert D._job_reach_claim({}) is None
    assert D._job_reach_claim({"job_reach": ["a", "b"]}) == ("a", "b")
    for bad in ("a", [1], [""], None):
        assert D._job_reach_claim({"job_reach": bad}) == (), bad


def test_lumen_patches_only_on_a_timer_s_delegation():
    def server(reach):
        return D.AuthContext(principal_id=USER, principal_type="user", user_id=USER,
                             actor="agience-server-lumen", delegation_aud="agience-server-lumen",
                             narrowed=True, scope="persona:lumen", job_reach=reach)
    D.enforce_narrowed_bounds(server(("t",)), NS(method="PATCH", url=NS(path="/artifacts/abc")))
    for reach in (None, ("t",)):
        with pytest.raises(HTTPException):
            D.enforce_narrowed_bounds(server(reach), NS(method="DELETE", url=NS(path="/artifacts/abc")))
    with pytest.raises(HTTPException):                                # a person's Lumen delegation
        D.enforce_narrowed_bounds(server(None), NS(method="PATCH", url=NS(path="/artifacts/abc")))


async def test_a_timer_creates_nothing_at_the_top_level_and_archives_nothing(db):
    from fastapi.responses import Response
    from mantle.routers import artifacts_router as artifacts

    from .test_a_card_shared_alone_opens_for_its_reader import _req
    _, runs, timer, _, _ = await _space(db)
    auth = _timer_auth(USER, (timer, runs))
    with acting_as(USER, principal_type="user"):
        with pytest.raises(HTTPException) as top:
            await artifacts.create_artifact(
                request=_req(), response=Response(), auth=auth, store_db=db,
                body=artifacts.CreateArtifactRequest(content_type="text/markdown", content="x",
                                                     identity="hook:anything"))
        with pytest.raises(HTTPException) as archive:
            await artifacts.update_artifact(artifact_id=timer, auth=auth, store_db=db,
                                            body=artifacts.UpdateArtifactRequest(state="archived"))
    assert top.value.status_code == 403 and archive.value.status_code == 403


def test_sealing_a_row_names_it_so_a_writer_with_no_key_seals_to_its_drop(monkeypatch):
    """BUG-041 (2026-11-10): a timer's run, written back to the timer on its job delegation, was
    refused: the seal asked for a key with no row named, so the writer that may not have one could
    not fall back to the collection's drop key either."""
    from mantle.db import doc_boundary as B
    from mantle.services import content_crypto as cc

    named = []

    def encrypt(principal, plaintext, **kw):
        named.append(cc._KEY_ARTIFACT.get())
        return b"MEC3" + b"x" * 40

    monkeypatch.setattr(cc, "encrypt_content", encrypt)
    doc = {"id": "t-1", "root_id": "t-1", "created_by": "alice", "collection_id": "inbox-1",
           "content_type": "application/vnd.agience.timer+json",
           "context": {"title": "Digest", "timer_state": {"runs": 1}}}
    B.seal_content_fields(doc)
    assert named == ["t-1"]

