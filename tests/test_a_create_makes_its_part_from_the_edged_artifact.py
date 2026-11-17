"""A create reaching an artifact whose type declares ``on_create: {"does": "create"}`` makes the
part that artifact describes, in the same transaction.

The item's collection is edged to an initializer. The initializer's context names what to create
(``creates``), where (``in``), the link from the item (``link``) and the part's starting fields.
The part is the initializer's own: its creator is the initializer, and no grant goes to the
person filing the item. Who may change the part is decided by grants on its home alone. A field
with ``on_update: {"stamp": ..., "when": ...}`` is written by Mantle when ``when`` changes.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi import HTTPException

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as store
from mantle.entities.grant import Grant
from mantle.services import on_create, type_offers
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import check_access
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE

from .test_there_is_no_promotion_state import OTHER, USER, real_node

INIT = "application/vnd.example.part-initializer+json"
PART = "application/vnd.example.decision+json"
ITEM = "application/vnd.example.request+json"
EDGE = "decided_by_part"
RULE = {"on_create": {"from_edge": EDGE}}
STAMPS = {
    "decided_by": {"on_update": {"stamp": "caller", "when": "state"}},
    "decided_at": {"on_update": {"stamp": "time", "when": "state"}},
}


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = real_node(tmp_path, monkeypatch)
    type_offers.set_definitions({
        ITEM: {"content_type": ITEM, "fields": {"part": RULE}},
        INIT: {"content_type": INIT, "on_create": {"does": "create"}},
        PART: {"content_type": PART, "fields": STAMPS},
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


def _grant(db, resource, grantee, actions):
    with acting_as(USER, principal_type="user"):
        store.create_grant(db, Grant(
            id=str(uuid.uuid4()), resource_id=resource, grantee_type="user", grantee_id=grantee,
            granted_by=USER, effect="allow", state=Grant.STATE_ACTIVE,
            **{f"can_{a}": a in actions for a in ACTIONS}))


def _setup(db, **spec):
    backlog = _container(db, "Requests")
    home = _container(db, "Decisions")
    full = {"creates": PART, "in": home, "link": "decision", "fields": {"state": "pending"}}
    full.update(spec)
    full = {k: v for k, v in full.items() if v is not None}
    init = _create(db, backlog, INIT, "decisions", **full).root_id
    store.add_artifact_to_collection(db, backlog, init, relationship=EDGE, origin=False)
    return backlog, home, init


def _members(db, collection):
    with acting_as(USER, principal_type="user"):
        return {a["root_id"] for a in store.list_collection_artifacts(db, collection)}


def _update(db, home, part, user, **changes):
    ctx = _ctx(db, part)
    ctx.update(changes)
    with acting_as(user, principal_type="user"):
        ws.update_artifact(db, user, home, part, context=json.dumps(ctx))


def _part_of(db, item):
    return _ctx(db, item.id)["part"]


def test_the_create_makes_the_part_in_its_home_owned_by_the_initializer(db):
    backlog, home, init = _setup(db)
    item = _create(db, backlog, ITEM, "ship it")
    part = _part_of(db, item)
    assert part in _members(db, home)
    with acting_as(USER, principal_type="user"):
        made = store.get_artifact(db, part)
    assert made.created_by == init
    ctx = _ctx(db, part)
    assert (ctx["content_type"], ctx["state"]) == (PART, "pending")


def _with_part_schema(part_def, base=None):
    defs = {ITEM: {"content_type": ITEM, "fields": {"part": RULE}},
            INIT: {"content_type": INIT, "on_create": {"does": "create"}},
            PART: {"content_type": PART, "fields": STAMPS, **part_def}}
    if base:
        defs[base[0]] = base[1]
    type_offers.set_definitions(defs)


def test_with_no_fields_the_part_starts_from_its_type_s_default(db):
    _with_part_schema({"context_schema": {"state": {"type": "enum", "default": "pending"},
                                          "reason": {"type": "string"}}})
    backlog, _, _ = _setup(db, fields=None)
    ctx = _ctx(db, _part_of(db, _create(db, backlog, ITEM, "ship it")))
    assert ctx["state"] == "pending" and "reason" not in ctx


def test_the_spec_s_fields_lie_over_the_type_s_defaults(db):
    _with_part_schema({"context_schema": {"state": {"default": "pending"},
                                          "reason": {"default": "none given"}}})
    backlog, _, _ = _setup(db, fields={"state": "declined"})
    ctx = _ctx(db, _part_of(db, _create(db, backlog, ITEM, "ship it")))
    assert (ctx["state"], ctx["reason"]) == ("declined", "none given")


def test_a_default_comes_from_the_nearest_type_that_declares_the_field(db):
    base = "application/vnd.example.base-decision+json"
    _with_part_schema({"inherits": [base], "context_schema": {"state": {"default": "pending"}}},
                      (base, {"content_type": base,
                              "context_schema": {"state": {"default": "approved"},
                                                 "reason": {"default": "inherited"}}}))
    backlog, _, _ = _setup(db, fields=None)
    ctx = _ctx(db, _part_of(db, _create(db, backlog, ITEM, "ship it")))
    assert (ctx["state"], ctx["reason"]) == ("pending", "inherited")


def test_with_no_default_and_no_fields_the_part_has_no_state(db):
    backlog, _, _ = _setup(db, fields=None)
    assert "state" not in _ctx(db, _part_of(db, _create(db, backlog, ITEM, "ship it")))


def test_the_item_links_to_its_part(db):
    backlog, _, _ = _setup(db)
    item = _create(db, backlog, ITEM, "ship it")
    with acting_as(USER, principal_type="user"):
        assert store.get_relationship_target(db, item.root_id, "decision") == _part_of(db, item)


def test_the_filer_holds_no_grant_on_the_part_and_cannot_change_it(db):
    backlog, home, _ = _setup(db)
    _grant(db, backlog, OTHER, {"read", "create", "add", "update", "invoke"})
    item = _create(db, backlog, ITEM, "ship it", user=OTHER)
    part = _part_of(db, item)
    with acting_as(USER, principal_type="user"):
        assert store.get_active_grants_for_principal_resource(db, OTHER, part) == []
    with pytest.raises(HTTPException) as refused:
        check_access(on_create._auth_of(OTHER), part, "update", db)
    assert refused.value.status_code in (403, 404)


def test_an_account_granted_update_on_the_home_decides_and_is_stamped(db):
    backlog, home, _ = _setup(db)
    part = _part_of(db, _create(db, backlog, ITEM, "ship it"))
    _grant(db, home, OTHER, {"read", "update"})
    check_access(on_create._auth_of(OTHER), part, "update", db)
    _update(db, home, part, OTHER, state="approved")
    ctx = _ctx(db, part)
    assert ctx["state"] == "approved"
    assert ctx["decided_by"] == OTHER
    assert ctx["decided_at"]


def test_a_stamp_moves_only_when_its_field_changes(db):
    backlog, home, _ = _setup(db)
    part = _part_of(db, _create(db, backlog, ITEM, "ship it"))
    _update(db, home, part, USER, state="approved")
    first = _ctx(db, part)
    _update(db, home, part, USER, note="looked again", decided_by="someone else")
    again = _ctx(db, part)
    assert (again["decided_by"], again["decided_at"]) == (first["decided_by"], first["decided_at"])
    assert again["decided_by"] == USER


def test_a_create_cannot_supply_a_stamped_field(db):
    _, home, _ = _setup(db)
    made = _create(db, home, PART, "forged", state="approved", decided_by="someone else",
                   decided_at="2000-01-01T00:00:00Z")
    ctx = _ctx(db, made.id)
    assert "decided_by" not in ctx and "decided_at" not in ctx


def _refused(db, backlog, status, user=USER):
    before = _members(db, backlog)
    with pytest.raises(HTTPException) as refused:
        _create(db, backlog, ITEM, "refused", user=user)
    assert refused.value.status_code == status
    assert _members(db, backlog) == before


@pytest.mark.parametrize("spec", [
    {"creates": None}, {"in": None}, {"link": None}, {"link": " "}, {"fields": "state"},
])
def test_a_malformed_spec_refuses_the_create(db, spec):
    backlog, home, _ = _setup(db, **spec)
    before = _members(db, home)
    _refused(db, backlog, 409)
    assert _members(db, home) == before


def test_an_initializer_whose_owner_cannot_create_in_the_home_refuses(db):
    backlog = _container(db, "Requests")
    elsewhere = _container(db, "Theirs", user=OTHER)
    init = _create(db, backlog, INIT, "decisions", creates=PART, **{"in": elsewhere},
                   link="decision").root_id
    store.add_artifact_to_collection(db, backlog, init, relationship=EDGE, origin=False)
    _refused(db, backlog, 409)
    with acting_as(OTHER, principal_type="user"):
        assert {a["root_id"] for a in store.list_collection_artifacts(db, elsewhere)} == set()


def test_no_initializer_refuses_the_create(db):
    _refused(db, _container(db, "Requests"), 409)


def test_two_initializers_refuse_the_create(db):
    backlog, home, _ = _setup(db)
    second = _create(db, backlog, INIT, "more", creates=PART, **{"in": home},
                     link="decision").root_id
    store.add_artifact_to_collection(db, backlog, second, relationship=EDGE, origin=False)
    _refused(db, backlog, 409)


def test_a_filer_without_invoke_on_the_initializer_is_refused(db):
    backlog, _, _ = _setup(db)
    _grant(db, backlog, OTHER, {"read", "create", "add", "update"})
    _refused(db, backlog, 403, user=OTHER)


def test_an_initializer_whose_type_declares_nothing_refuses(db):
    type_offers.set_definitions({ITEM: {"content_type": ITEM, "fields": {"part": RULE}},
                                 PART: {"content_type": PART, "fields": STAMPS}})
    backlog, home, _ = _setup(db)
    _refused(db, backlog, 409)
    assert _members(db, home) == set()


def test_the_verbs_are_the_declarations_vocabulary():
    assert set(on_create._VERBS) == set(type_offers.ON_CREATE_VERBS)


THIRD = "u-third"
FILING = {"create", "add", "invoke"}


def _can(db, who, artifact, action):
    try:
        check_access(on_create._auth_of(who), artifact, action, db)
    except HTTPException as refused:
        assert refused.status_code in (403, 404)
        return False
    return True


def test_a_filer_reads_its_own_part_and_no_other_and_updates_neither(db):
    backlog, home, _ = _setup(db, conducts=["read"])
    _grant(db, backlog, OTHER, FILING)
    _grant(db, backlog, THIRD, FILING)
    mine = _create(db, backlog, ITEM, "mine", user=OTHER)
    theirs = _create(db, backlog, ITEM, "theirs", user=THIRD)
    _grant(db, mine.root_id, OTHER, {"read"})
    _grant(db, theirs.root_id, THIRD, {"read"})
    my_part, their_part = _part_of(db, mine), _part_of(db, theirs)
    assert _can(db, OTHER, my_part, "read")
    assert _ctx(db, my_part, user=OTHER)["state"] == "pending"
    assert not _can(db, OTHER, their_part, "read")
    assert not _can(db, OTHER, my_part, "update")
    assert not _can(db, OTHER, their_part, "update")
    assert not _can(db, OTHER, home, "read")


def test_a_link_that_conducts_nothing_gives_the_item_s_reader_nothing(db):
    backlog, _, _ = _setup(db)
    item = _create(db, backlog, ITEM, "ship it")
    _grant(db, item.root_id, OTHER, {"read"})
    assert not _can(db, OTHER, _part_of(db, item), "read")


@pytest.mark.parametrize("spec", [
    {"conducts": "read"}, {"conducts": ["read", "fly"]},
])
def test_a_spec_that_conducts_badly_refuses(db, spec):
    backlog, home, _ = _setup(db, **spec)
    _refused(db, backlog, 409)
    assert _members(db, home) == set()


def test_a_field_the_made_type_marks_secret_refuses(db):
    type_offers.set_definitions({
        ITEM: {"content_type": ITEM, "fields": {"part": RULE}},
        INIT: {"content_type": INIT, "on_create": {"does": "create"}},
        PART: {"content_type": PART, "fields": {**STAMPS, "pin": {"secret": True}}},
    })
    backlog, home, _ = _setup(db, fields={"state": "pending", "pin": "1234"})
    _refused(db, backlog, 409)
    assert _members(db, home) == set()


def test_the_part_is_sealed_at_rest_like_any_create_in_its_home(db, tmp_path):
    # The initializer is readable configuration, so its spec is plain at rest; the part is not.
    spec_marker, own_marker = ("marker-" + uuid.uuid4().hex for _ in range(2))
    backlog, home, _ = _setup(db, fields={"state": "pending", "note": spec_marker})
    part = _part_of(db, _create(db, backlog, ITEM, "ship it"))
    assert _ctx(db, part)["note"] == spec_marker
    assert spec_marker not in json.dumps(db.artifacts.get_artifact(part), default=str)
    _update(db, home, part, USER, note=own_marker)
    assert _ctx(db, part)["note"] == own_marker
    assert own_marker not in json.dumps(db.artifacts.get_artifact(part), default=str)
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert own_marker.encode() not in path.read_bytes(), path


@pytest.fixture
def custody(tmp_path, monkeypatch):
    """A node whose keys go by the light cone, as in production: each collection's own key, the
    part's its own, and the oracle releasing a key only to a reach for read."""
    from mantle.search.mantle import oracle as o
    from mantle.search.mantle import wiring
    from mantle.services import content_crypto

    from .test_a_split_master_key_needs_two_custodians import _FakeKek

    real_encrypt, real_decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content
    database = real_node(tmp_path, monkeypatch)
    monkeypatch.setattr(content_crypto, "encrypt_content", real_encrypt)
    monkeypatch.setattr(content_crypto, "decrypt_content", real_decrypt)
    keys = o.OracleService(o.LatticeMasterKeyStore(_FakeKek(), lambda: database),
                           grant_verifier=o.LightConeGrantVerifier(database, ttl_s=0))
    monkeypatch.setattr(wiring, "_build_oracle", lambda: keys)
    type_offers.set_definitions({
        ITEM: {"content_type": ITEM, "fields": {"part": RULE}},
        INIT: {"content_type": INIT, "on_create": {"does": "create"}},
        PART: {"content_type": PART, "fields": STAMPS},
    })
    yield database, keys
    type_offers.set_definitions({})


def test_under_custody_the_filer_opens_its_own_part_s_key_and_no_other(custody):
    from mantle.search.mantle import oracle as o

    db, keys = custody
    backlog, home, _ = _setup(db, conducts=["read"])
    _grant(db, backlog, OTHER, FILING)
    _grant(db, backlog, THIRD, FILING)
    mine = _create(db, backlog, ITEM, "mine", user=OTHER)
    theirs = _create(db, backlog, ITEM, "theirs", user=THIRD)
    _grant(db, mine.root_id, OTHER, {"read"})
    _grant(db, theirs.root_id, THIRD, {"read"})
    my_part, their_part = _part_of(db, mine), _part_of(db, theirs)
    assert db.artifacts.get_artifact(my_part)["content_key_scope"] == my_part

    keys.evict()
    assert _ctx(db, my_part, user=OTHER)["state"] == "pending"
    # Holding its part's key, the filer still cannot change the part: only an update grant can.
    with pytest.raises(HTTPException) as refused:
        check_access(on_create._auth_of(OTHER), my_part, "update", db)
    assert refused.value.status_code in (403, 404)
    with acting_as(OTHER, principal_type="user"):
        for scope in (home, their_part):
            with pytest.raises(o.GrantDenied):
                keys.get_or_create_collection_key(scope, scope, o.KeyRequest(
                    requester_id=OTHER, purpose=o.KeyPurpose.GRANT, action="read"))
    unopened = _ctx(db, their_part, user=OTHER)              # it stays sealed to the filer
    assert "sealed" in unopened and "state" not in unopened

    decider = "u-decider"
    _grant(db, home, decider, {"read", "update"})
    keys.evict()
    _update(db, home, my_part, decider, state="approved")
    assert _ctx(db, my_part, user=decider)["state"] == "approved"
    assert _ctx(db, my_part, user=OTHER)["state"] == "approved"
    assert _ctx(db, their_part, user=decider)["state"] == "pending"
