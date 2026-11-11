"""A timer's job delegation asks for a person's content key only to read an artifact its job grant
lists by id (John, 2026-11-10, Q9). Every other narrowed delegation still asks for none, and a
job's writes still go to the drop key; Origin decides the release on the same rule.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from mantle.services import content_crypto as cc
from mantle.services.acting_principal import KeyCustodyDenied, acting_from_auth


def _job(reach=("t-1", "target-1")):
    return NS(narrowed="lumen", job_reach=tuple(reach))


def test_it_reads_the_key_of_an_artifact_its_grant_lists():
    for artifact in ("t-1", "target-1"):
        with cc.content_key_artifact(artifact):
            cc._refuse_ember(_job())


@pytest.mark.parametrize("artifact", ["inbox-1", None])
def test_it_reads_no_other_key(artifact):
    with cc.content_key_artifact(artifact), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(_job())


def test_it_mints_and_writes_under_no_key():
    with cc.content_key_artifact("t-1"), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(_job(), may_create=True)


def test_a_narrowed_delegation_with_no_job_reads_nothing():
    with cc.content_key_artifact("t-1"), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(NS(narrowed="lumen", job_reach=()))


def test_the_acting_principal_carries_the_reach_origin_signed():
    auth = NS(principal_id="alice", principal_type="user", actor="agience-server-lumen",
              host_id=None, authority=None, scope="persona:lumen", narrowed_audience="lumen",
              job_reach=("t-1",))
    assert acting_from_auth(auth).job_reach == ("t-1",)
    assert acting_from_auth(NS(**{**vars(auth), "job_reach": None})).job_reach == ()


async def test_it_reads_the_key_of_an_item_held_in_a_collection_its_grant_lists(tmp_path, monkeypatch):
    """BUG-052 (John, 2026-11-11): a run that links a collection answers from the notes it holds,
    including one filed after the grant was approved. An item outside the collection stays shut."""
    import json

    from mantle.db import backend
    from mantle.services import workspace_service as ws
    from mantle.services.acting_principal import acting_as

    from .test_a_typed_link_names_what_it_links import COLLECTION, _create
    from .test_there_is_no_promotion_state import USER, real_node

    db = real_node(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, "_HANDLE", db)
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json",
                                    name="Acceptance space").id
    shelf = await _create(db, container_id=space, content_type=COLLECTION, context=json.dumps({"title": "Acceptance"}))
    held = await _create(db, container_id=shelf["id"], context=json.dumps({"title": "Smoke note"}), content="42")
    loose = await _create(db, container_id=space, context=json.dumps({"title": "Elsewhere"}), content="no")
    run = _job(reach=(shelf["id"],))
    with cc.content_key_artifact(held.get("root_id") or held["id"]):
        cc._refuse_ember(run)
    with cc.content_key_artifact(loose.get("root_id") or loose["id"]), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(run)
    with cc.content_key_artifact(held.get("root_id") or held["id"]), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(run, may_create=True)


async def test_an_item_only_linked_or_behind_an_edge_that_drops_read_stays_shut(tmp_path, monkeypatch):
    """The reach is the creation light cone: a link into the listed collection confers nothing,
    and a creation edge whose mask does not carry read ends the walk."""
    import json

    from mantle.db import backend, lattice_api
    from mantle.services import workspace_service as ws
    from mantle.services.acting_principal import acting_as

    from .test_a_typed_link_names_what_it_links import COLLECTION, _create
    from .test_there_is_no_promotion_state import USER, real_node

    db = real_node(tmp_path, monkeypatch)
    monkeypatch.setattr(backend, "_HANDLE", db)
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json",
                                    name="Acceptance space").id
        masked = ws.create_container(db, USER, content_type=COLLECTION, name="Masked").id
    shelf = await _create(db, container_id=space, content_type=COLLECTION, context=json.dumps({"title": "Acceptance"}))
    loose = await _create(db, container_id=space, context=json.dumps({"title": "Elsewhere"}), content="no")
    await _create(db, container_id=shelf["id"], source_artifact_id=loose["id"], relationship="evidence")
    lattice_api.add_artifact_to_collection(db, shelf["id"], masked, origin=True, propagate=["update"])
    run = _job(reach=(shelf["id"],))
    for artifact in (loose.get("root_id") or loose["id"], masked):
        with cc.content_key_artifact(artifact), pytest.raises(KeyCustodyDenied):
            cc._refuse_ember(run)
