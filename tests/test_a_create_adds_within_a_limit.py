"""A create can add to a field of an edged artifact, within that artifact's limit.

The created type's rule names the amount field and the edged artifact's `to` and `limit` fields;
the collection it is created in carries an edge to a tally (cap, used). The create adds the amount
to `used`, writes it back and stamps the tally's id, in one transaction; an add past the cap is
refused and writes nothing. Runs on a real node, as the numbering tests do.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from mantle.db import lattice_api as store
from mantle.services import type_offers
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE

from .test_there_is_no_promotion_state import OTHER, USER, real_node

TALLY = "application/vnd.example.tally+json"
ITEM = "application/vnd.example.item+json"
RULE = {"on_create": {"from_edge": "counted_by", "amount": "amount", "to": "used", "limit": "cap"}}


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = real_node(tmp_path, monkeypatch)
    type_offers.set_definitions({
        ITEM: {"content_type": ITEM, "fields": {"tally": RULE}},
        TALLY: {"content_type": TALLY, "on_create": {"does": "add"}},
    })
    yield database
    type_offers.set_definitions({})


def _container(db, name, user=USER):
    with acting_as(user, principal_type="user"):
        return ws.create_container(db, user, content_type=WORKSPACE_CONTENT_TYPE, name=name).id


def _create(db, collection, content_type, title, user=USER, **extra):
    ctx = {"content_type": content_type, "title": title, **extra}
    with acting_as(user, principal_type="user"):
        return ws.create_workspace_artifact(db, user, collection, context=json.dumps(ctx),
                                            content=title, content_type=content_type)


def _ctx(db, artifact_id, user=USER):
    with acting_as(user, principal_type="user"):
        entity = store.get_artifact(db, artifact_id)
    ctx = entity.context
    return json.loads(ctx) if isinstance(ctx, str) else dict(ctx)


def _counted(db, limit=100, used=0, user=USER, collection=None):
    col = collection or _container(db, "Runs")
    home = _container(db, "Theirs", user=user) if user != USER else col
    extra = {"used": used}
    if limit is not None:
        extra["cap"] = limit
    tally = _create(db, home, TALLY, "quota", user=user, **extra).root_id
    store.add_artifact_to_collection(db, col, tally, relationship="counted_by", origin=False)
    return col, tally


def _item(db, col, amount, title="a run"):
    return _create(db, col, ITEM, title, amount=amount)


def test_an_add_adds_to_the_field_and_names_the_edged_artifact(db):
    col, tally = _counted(db, limit=100, used=10)
    item = _item(db, col, 25)
    assert _ctx(db, item.id)["tally"] == tally
    assert _ctx(db, tally)["used"] == 35


def test_fractional_amounts_add_exactly(db):
    col, tally = _counted(db, limit=1)
    for _ in range(3):
        _item(db, col, 0.1)
    assert _ctx(db, tally)["used"] == 0.3


def test_an_add_that_reaches_the_limit_is_taken(db):
    col, tally = _counted(db, limit=5, used=3)
    _item(db, col, 2)
    assert _ctx(db, tally)["used"] == 5


def test_an_add_past_the_limit_is_refused_and_writes_nothing(db):
    col, tally = _counted(db, limit=5, used=3)
    before = len(store.list_collection_artifacts(db, col))
    with pytest.raises(HTTPException) as refused:
        _item(db, col, 3)
    assert refused.value.status_code == 409
    assert "cap" in refused.value.detail
    assert len(store.list_collection_artifacts(db, col)) == before
    assert _ctx(db, tally)["used"] == 3


def test_concurrent_adds_never_pass_the_limit(db):
    col, tally = _counted(db, limit=50)
    taken = []
    lock = threading.Lock()

    def one(i):
        try:
            _item(db, col, 1, title=f"run {i}")
        except HTTPException as refused:
            assert refused.status_code == 409
            return
        with lock:
            taken.append(i)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(80)))
    assert len(taken) == 50
    assert _ctx(db, tally)["used"] == 50


def test_an_edged_artifact_with_no_limit_refuses_the_add(db):
    col, tally = _counted(db, limit=None)
    with pytest.raises(HTTPException) as refused:
        _item(db, col, 1)
    assert refused.value.status_code == 409
    assert _ctx(db, tally)["used"] == 0


@pytest.mark.parametrize("amount", [-1, "lots", True, None, float("nan"), float("inf"), [1]])
def test_an_amount_that_is_not_a_number_of_zero_or_more_is_refused(db, amount):
    col, tally = _counted(db)
    with pytest.raises(HTTPException) as refused:
        _item(db, col, amount)
    assert refused.value.status_code == 422
    assert _ctx(db, tally)["used"] == 0


def test_a_creator_without_invoke_on_the_edged_artifact_is_refused(db):
    col = _container(db, "Runs")
    _, tally = _counted(db, user=OTHER, collection=col)
    with pytest.raises(HTTPException) as refused:
        _item(db, col, 1)
    assert refused.value.status_code == 403
    assert _ctx(db, tally, user=OTHER)["used"] == 0


def test_no_edged_artifact_refuses_the_add(db):
    col = _container(db, "Uncounted")
    with pytest.raises(HTTPException) as refused:
        _item(db, col, 1)
    assert refused.value.status_code == 409
    assert store.list_collection_artifacts(db, col) == []


def test_an_update_cannot_change_the_amount_or_the_stamp(db):
    col, tally = _counted(db)
    item = _item(db, col, 4)
    for change in ({"amount": 0}, {"tally": "elsewhere"}):
        with acting_as(USER, principal_type="user"):
            entity = store.get_artifact(db, item.id)
            ctx = json.loads(entity.context) if isinstance(entity.context, str) else dict(entity.context)
            ctx.update(change)
            entity.context = json.dumps(ctx)
            with pytest.raises(HTTPException) as refused:
                store.update_artifact(db, entity)
        assert refused.value.status_code == 409
    assert _ctx(db, item.id)["amount"] == 4
    assert _ctx(db, tally)["used"] == 4


@pytest.mark.parametrize("rule", [
    {"on_create": {"from_edge": "counted_by", "amount": "tally"}},
    {"on_create": {"from_edge": "counted_by", "amount": ""}},
    {"on_create": {"from_edge": "counted_by", "amount": 3}},
    {"on_create": {"from_edge": "counted_by", "amount": "amount", "to": "", "limit": "cap"}},
    {"on_create": {"from_edge": "counted_by", "amount": "amount", "to": "used", "limit": 5}},
    {"on_create": {"from_edge": "counted_by", "amount": "amount", "to": "cap", "limit": "cap"}},
])
def test_a_malformed_amount_rule_is_refused(rule):
    assert type_offers.validate_field_rules({"tally": rule})


def test_the_add_rule_and_verb_are_accepted():
    assert type_offers.validate_field_rules({"tally": RULE}) == []
    assert type_offers.validate_on_create({"does": "add"}) == []


def test_a_rule_that_names_no_limit_refuses_the_add(db):
    type_offers.set_definitions({
        ITEM: {"content_type": ITEM, "fields": {"tally": {"on_create": {
            "from_edge": "counted_by", "amount": "amount", "to": "used"}}}},
        TALLY: {"content_type": TALLY, "on_create": {"does": "add"}},
    })
    col, tally = _counted(db)
    with pytest.raises(HTTPException) as refused:
        _item(db, col, 1)
    assert refused.value.status_code == 409
    assert _ctx(db, tally)["used"] == 0
