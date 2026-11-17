"""A type can mark a field "set on create from an edge".

The type definition carries the rule; the collection an artifact is created in carries the
`numbered_by` edge to a provider (prefix, width, last). The create takes `last + 1`, writes it back
to the provider and stamps the field and its key, in one transaction. Runs on a real node: real
store, real light cone, real sealing; only key custody is stood in for.
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import HTTPException

from mantle.db import lattice_api as store
from mantle.services import on_create, type_offers
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE

from .test_there_is_no_promotion_state import OTHER, USER, real_node

PROVIDER = "application/vnd.agience.number-provider+json"
ITEM = "application/vnd.example.ticket+json"
CHILD = "application/vnd.example.urgent-ticket+json"
PLAIN = "application/vnd.example.note+json"
RULE = {"on_create": {"from_edge": "numbered_by", "key": "key"}}


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = real_node(tmp_path, monkeypatch)
    type_offers.set_definitions({
        ITEM: {"content_type": ITEM, "fields": {"number": RULE}},
        CHILD: {"content_type": CHILD, "inherits": [ITEM]},
        PLAIN: {"content_type": PLAIN},
        PROVIDER: {"content_type": PROVIDER, "on_create": {"does": "number"}},
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


def _provider(db, collection, prefix="BUG", width=0, last=113, user=USER):
    return _create(db, collection, PROVIDER, prefix, user=user, prefix=prefix, width=width,
                   last=last).root_id


def _edge(db, collection, provider):
    store.add_artifact_to_collection(db, collection, provider, relationship="numbered_by",
                                     origin=False)


def _ctx(db, artifact_id, user=USER):
    with acting_as(user, principal_type="user"):
        entity = store.get_artifact(db, artifact_id)
    ctx = entity.context
    return json.loads(ctx) if isinstance(ctx, str) else dict(ctx)


def _numbered(db, prefix="BUG", width=0, last=113):
    col = _container(db, "Backlog")
    provider = _provider(db, col, prefix=prefix, width=width, last=last)
    _edge(db, col, provider)
    return col, provider


# 1
def test_a_create_takes_the_next_number_and_its_key(db):
    col, provider = _numbered(db)
    item = _create(db, col, ITEM, "the login page hangs")
    ctx = _ctx(db, item.id)
    assert ctx["number"] == 114
    assert ctx["key"] == "BUG-114"
    assert _ctx(db, provider)["last"] == 114


def test_the_key_is_padded_to_the_providers_width(db):
    col, _ = _numbered(db, prefix="FR", width=4, last=6)
    assert _ctx(db, _create(db, col, ITEM, "padded").id)["key"] == "FR-0007"


# 2
def test_concurrent_creates_take_distinct_gap_free_numbers(db):
    col, provider = _numbered(db, last=0)
    count = 200
    numbers = []
    lock = threading.Lock()

    def one(i):
        item = _create(db, col, ITEM, f"item {i}")
        n = _ctx(db, item.id)["number"]
        with lock:
            numbers.append(n)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, range(count)))
    assert sorted(numbers) == list(range(1, count + 1))
    assert _ctx(db, provider)["last"] == count


# 3
def test_a_creator_without_invoke_on_the_provider_is_refused(db):
    col = _container(db, "Backlog")
    elsewhere = _container(db, "Theirs", user=OTHER)
    provider = _provider(db, elsewhere, user=OTHER)
    _edge(db, col, provider)
    before = len(store.list_collection_artifacts(db, col))
    with pytest.raises(HTTPException) as refused:
        _create(db, col, ITEM, "not mine to number")
    assert refused.value.status_code == 403
    assert len(store.list_collection_artifacts(db, col)) == before
    assert _ctx(db, provider, user=OTHER)["last"] == 113


# 4
def test_no_provider_refuses_the_create(db):
    col = _container(db, "Unnumbered")
    with pytest.raises(HTTPException) as refused:
        _create(db, col, ITEM, "nowhere to number from")
    assert refused.value.status_code == 409
    assert store.list_collection_artifacts(db, col) == []


def test_two_providers_at_one_level_refuse_the_create(db):
    col, _ = _numbered(db)
    _edge(db, col, _provider(db, col, prefix="TASK"))
    with pytest.raises(HTTPException) as refused:
        _create(db, col, ITEM, "which one")
    assert refused.value.status_code == 409


# 5
def test_the_create_commits_when_the_event_bus_fails(db, monkeypatch):
    from mantle.events import event_bus

    def broken(*a, **kw):
        raise RuntimeError("bus down")
    monkeypatch.setattr(event_bus, "emit_artifact_event_sync", broken)
    col, provider = _numbered(db)
    item = _create(db, col, ITEM, "announced or not")
    assert _ctx(db, item.id)["number"] == 114
    assert _ctx(db, provider)["last"] == 114


# 6
def test_the_same_create_again_keeps_its_number(db):
    col, provider = _numbered(db)
    item = _create(db, col, ITEM, "retried")
    with acting_as(USER, principal_type="user"):
        again = store.get_artifact(db, item.id)
        ctx = json.loads(again.context) if isinstance(again.context, str) else dict(again.context)
        ctx.pop("number")
        ctx.pop("key")
        again.context = json.dumps(ctx)
        store.create_artifact(db, again)
    assert _ctx(db, item.id)["number"] == 114
    assert _ctx(db, provider)["last"] == 114


# 7
def test_the_nearest_provider_wins(db):
    col, outer = _numbered(db)
    with acting_as(USER, principal_type="user"):
        sub = ws.create_workspace_artifact(
            db, USER, col, context=json.dumps({"content_type": WORKSPACE_CONTENT_TYPE,
                                               "title": "sub"}),
            content="", content_type=WORKSPACE_CONTENT_TYPE).root_id
    inner = _provider(db, sub, prefix="TASK", last=9)
    _edge(db, sub, inner)
    assert _ctx(db, _create(db, sub, ITEM, "inner").id)["key"] == "TASK-10"
    assert _ctx(db, outer)["last"] == 113


def test_a_subcollection_without_its_own_provider_uses_its_parents(db):
    col, outer = _numbered(db)
    with acting_as(USER, principal_type="user"):
        sub = ws.create_workspace_artifact(
            db, USER, col, context=json.dumps({"content_type": WORKSPACE_CONTENT_TYPE,
                                               "title": "sub"}),
            content="", content_type=WORKSPACE_CONTENT_TYPE).root_id
    assert _ctx(db, _create(db, sub, ITEM, "inherits the edge").id)["key"] == "BUG-114"
    assert _ctx(db, outer)["last"] == 114


# 8
def test_an_inheriting_type_gets_the_rule(db):
    col, _ = _numbered(db)
    assert _ctx(db, _create(db, col, CHILD, "urgent").id)["number"] == 114


# 9
def test_a_number_is_not_reused_after_a_delete(db):
    col, _ = _numbered(db)
    first = _create(db, col, ITEM, "gone soon")
    store.delete_artifact(db, first.id)
    assert _ctx(db, _create(db, col, ITEM, "next").id)["number"] == 115


# 10
def test_a_provider_from_another_node_refuses_here(db, monkeypatch):
    col, provider = _numbered(db)
    monkeypatch.setattr(db, "origin", "peer-node")
    with pytest.raises(HTTPException) as refused:
        _create(db, col, ITEM, "not home")
    assert refused.value.status_code == 409
    assert "home node" in refused.value.detail


# the interim guard (acceptance 4)
def _update_ctx(db, artifact_id, **changes):
    with acting_as(USER, principal_type="user"):
        entity = store.get_artifact(db, artifact_id)
        ctx = json.loads(entity.context) if isinstance(entity.context, str) else dict(entity.context)
        for k, v in changes.items():
            if v is None:
                ctx.pop(k, None)
            else:
                ctx[k] = v
        entity.context = json.dumps(ctx)
        store.update_artifact(db, entity)


def test_an_update_cannot_change_the_number_or_key(db):
    col, _ = _numbered(db)
    item = _create(db, col, ITEM, "fixed id")
    for change in ({"number": 7}, {"key": "BUG-7"}):
        with pytest.raises(HTTPException) as refused:
            _update_ctx(db, item.id, **change)
        assert refused.value.status_code == 409
    assert _ctx(db, item.id)["key"] == "BUG-114"


def test_an_update_that_leaves_the_number_out_keeps_it(db):
    col, _ = _numbered(db)
    item = _create(db, col, ITEM, "fixed id")
    _update_ctx(db, item.id, number=None, key=None, title="renamed")
    ctx = _ctx(db, item.id)
    assert (ctx["number"], ctx["key"], ctx["title"]) == (114, "BUG-114", "renamed")


# 11
def _imported(db, titles):
    col = _container(db, "Backlog")
    provider = _provider(db, col, last=0)
    _edge(db, col, provider)
    with acting_as(USER, principal_type="user"):
        items = [ws.create_workspace_artifact(
            db, USER, col, context=json.dumps({"content_type": PLAIN, "title": t}),
            content=t, content_type=PLAIN) for t in titles]
    # Typed after the import, as the Backlog's items were: retype in place.
    for it in items:
        with acting_as(USER, principal_type="user"):
            entity = store.get_artifact(db, it.id)
            ctx = json.loads(entity.context) if isinstance(entity.context, str) else entity.context
            ctx["content_type"] = ITEM
            entity.context = json.dumps(ctx)
            entity.content_type = ITEM
            store.update_artifact(db, entity)
    return col, provider, items


PATTERN = r"^(?P<prefix>BUG|TASK|FR)-(?P<n>\d+)\b"


def test_adopt_dry_run_writes_nothing(db):
    col, provider, items = _imported(db, ["BUG-3: one", "BUG-12: two", "TASK-4: theirs"])
    with acting_as(USER, principal_type="user"):
        report = on_create.adopt(db, provider, col, PATTERN)
    assert report["dry_run"] is True
    assert sorted(c["key"] for c in report["changes"]) == ["BUG-12", "BUG-3"]
    assert report["last"] == 12
    assert "number" not in _ctx(db, items[0].id)
    assert _ctx(db, provider)["last"] == 0


def test_adopt_sets_fields_strips_titles_and_sets_last(db):
    col, provider, items = _imported(db, ["BUG-3: one", "BUG-12: two", "TASK-4: theirs"])
    with acting_as(USER, principal_type="user"):
        on_create.adopt(db, provider, col, PATTERN, dry_run=False)
    one = _ctx(db, items[0].id)
    assert (one["number"], one["key"], one["title"]) == (3, "BUG-3", "one")
    assert "number" not in _ctx(db, items[2].id)
    assert _ctx(db, provider)["last"] == 12
    assert _ctx(db, _create(db, col, ITEM, "after the import").id)["key"] == "BUG-13"


def test_adopt_refuses_a_duplicate_number_and_writes_nothing(db):
    col, provider, items = _imported(db, ["BUG-3: one", "BUG-3: again"])
    with acting_as(USER, principal_type="user"), pytest.raises(HTTPException) as refused:
        on_create.adopt(db, provider, col, PATTERN, dry_run=False)
    assert refused.value.status_code == 409
    for it in items:
        assert it.root_id in refused.value.detail
        assert "number" not in _ctx(db, it.id)
    assert _ctx(db, provider)["last"] == 0


# 12
@pytest.mark.parametrize("fields", [
    "number",
    {"number": "next"},
    {"number": {"on_create": {"from_edge": "numbered_by", "op": "next"}}},
    {"number": {"on_create": {"from_edge": ""}}},
    {"number": {"on_create": {"from_edge": "numbered_by", "extra": 1}}},
    {"number": {"on_create": {"from_edge": "numbered_by", "key": "number"}}},
    {"by": {"on_update": {"stamp": "someone", "when": "state"}}},
    {"by": {"on_update": {"stamp": "caller", "when": "by"}}},
    {"by": {"on_update": {"stamp": "caller"}}},
])
def test_a_malformed_rule_is_refused(fields):
    assert type_offers.validate_field_rules(fields)


def test_publish_refuses_a_malformed_rule(db, monkeypatch):
    monkeypatch.setattr(type_offers, "_system_principal_id", lambda: "system")
    bad = {"content_type": "application/vnd.example.bad+json",
           "fields": {"number": {"on_create": {"from_edge": "numbered_by", "op": "previous"}}}}
    result = type_offers.publish(db, "test-server", [bad])
    assert "application/vnd.example.bad+json" in result["refused"]


@pytest.mark.parametrize("declared", ["number", {"does": "next"}, {"does": "number", "x": 1}])
def test_a_malformed_verb_declaration_is_refused(declared):
    assert type_offers.validate_on_create(declared)


def test_a_provider_whose_type_declares_no_verb_refuses_the_create(db):
    type_offers.set_definitions({ITEM: {"content_type": ITEM, "fields": {"number": RULE}}})
    col, provider = _numbered(db)
    with pytest.raises(HTTPException) as refused:
        _create(db, col, ITEM, "nothing declared")
    assert refused.value.status_code == 409
    assert _ctx(db, provider)["last"] == 113


def test_a_wellformed_rule_is_accepted():
    assert type_offers.validate_field_rules({"number": RULE}) == []


# 13
def test_on_create_names_no_work_item_kind_or_prefix():
    src = Path(on_create.__file__).read_text(encoding="utf-8")
    src += Path(type_offers.__file__).read_text(encoding="utf-8")
    for word in ("BUG", "TASK", "FR-", "backlog"):
        assert not re.search(r"\b" + re.escape(word), src, re.IGNORECASE), word


# 14
def test_a_type_without_rules_takes_no_numbering_step(db, monkeypatch):
    col = _container(db, "Plain")

    def never(*a, **kw):
        raise AssertionError("numbering ran for a type with no rules")
    monkeypatch.setattr(on_create, "apply_on_create", never)
    monkeypatch.setattr(on_create, "guard_update", lambda db, entity: None)
    note = _create(db, col, PLAIN, "just a note")
    assert "number" not in _ctx(db, note.id)
