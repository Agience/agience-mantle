"""An agent session's delegation reaches only what its approval names (FR-032).

Origin exchanges an approved session's actor token for a five-minute delegation: the person is
`sub`, the agent is `act`, the audience is Mantle, the scope is `mantle:agent`, and the approval's
reach (`job_reach`) and actions (`reach_actions`) ride in the token. Mantle holds the delegation to
that reach, for those actions only, on the routes an agent's work calls. A message a session files
is stamped at the level its token speaks at: `agent` for an acted token, `typed` for a person's own.
"""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from mantle.services import dependencies as D
from mantle.services import on_create, type_offers
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_card_shared_alone_opens_for_its_reader import _create
from . import test_a_card_shared_alone_opens_for_its_reader as _card
from .test_a_timer_s_delegation_reaches_only_its_grant import _allowed, _route, _space, _timer_auth
from .test_there_is_no_promotion_state import USER

db = _card.db

MESSAGE = "application/vnd.example.message+json"


def _agent_auth(user, reach, actions=("read", "create")):
    return D.AuthContext(principal_id=user, principal_type="user", user_id=user,
                         actor="agent-x", delegation_aud="mantle", narrowed=True,
                         scope=D.AGENT_SCOPE, job_reach=tuple(reach),
                         reach_actions=frozenset(actions))


async def test_an_agent_does_only_its_approval_s_actions_inside_its_reach(db):
    space, runs, timer, other, record = await _space(db)
    auth = _agent_auth(USER, (runs,))
    assert _allowed(db, auth, runs, "read") and _allowed(db, auth, runs, "create")
    assert _allowed(db, auth, record, "read")
    assert not _allowed(db, auth, record, "update")                   # not in the approval
    assert not _allowed(db, auth, other, "read")                      # outside the reach
    for action in ("delete", "evict", "share", "admin"):
        assert not _allowed(db, auth, runs, action), action
    none = _agent_auth(USER, (runs,), actions=())
    assert not _allowed(db, none, runs, "read")                       # no actions: nothing


def test_an_agent_reads_links_and_a_timer_does_not():
    agent, timer = _agent_auth(USER, ("t",)), _timer_auth(USER, ("t",))
    assert _route(agent, "GET", "/artifacts/t/links") and not _route(timer, "GET", "/artifacts/t/links")
    for m, p in [("GET", "/artifacts/t"), ("POST", "/artifacts"), ("GET", "/artifacts/t/children")]:
        assert _route(agent, m, p), (m, p)
    for m, p in [("GET", "/artifacts/visible"), ("DELETE", "/artifacts/t"), ("POST", "/grants")]:
        assert not _route(agent, m, p), (m, p)


def test_the_agent_scope_counts_only_for_mantle():
    auth = _agent_auth(USER, ("t",))
    D.enforce_narrowed_bounds(auth, NS(method="GET", url=NS(path="/artifacts/t/links")))
    with pytest.raises(HTTPException):
        D.enforce_narrowed_bounds(auth, NS(method="DELETE", url=NS(path="/artifacts/t")))
    elsewhere = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER,
                              actor="agent-x", delegation_aud="ember", narrowed=True,
                              scope=D.AGENT_SCOPE, job_reach=("t",))
    with pytest.raises(HTTPException):
        D.enforce_narrowed_bounds(elsewhere, NS(method="GET", url=NS(path="/artifacts/t")))


def test_the_actions_are_read_from_origin_s_claim():
    assert D._reach_actions_claim({}) is None
    assert D._reach_actions_claim({"reach_actions": ["read", "create"]}) == {"read", "create"}
    for bad in ("read", [1], None):
        assert D._reach_actions_claim({"reach_actions": bad}) == frozenset(), bad


def test_a_level_stamp_is_the_only_thing_its_rule_names():
    good = {"level": {"on_create": {"stamp": "level"}}}
    assert type_offers.validate_field_rules(good) == []
    for bad in ({"stamp": "caller"}, {"stamp": "level", "from_edge": "x"}):
        assert type_offers.validate_field_rules({"level": {"on_create": bad}}), bad


def test_the_level_is_typed_only_on_a_person_s_own_token():
    person = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    assert on_create.level_of(person) == "typed"
    assert on_create.level_of(_agent_auth(USER, ())) == "agent"
    assert on_create.level_of(None) == "agent"
    server = D.AuthContext(principal_id="s", principal_type="server", user_id=None)
    assert on_create.level_of(server) == "agent"


@pytest.fixture
def typed(db):
    type_offers.set_definitions({MESSAGE: {"content_type": MESSAGE,
                                           "fields": {"level": {"on_create": {"stamp": "level"}}}}})
    yield db
    type_offers.set_definitions({})


def _file(db, collection, auth, claimed=None):
    ctx = {"content_type": MESSAGE, "title": "m", **({"level": claimed} if claimed else {})}
    with acting_as(USER, principal_type="user"), on_create.creating_as(auth):
        entity = ws.create_workspace_artifact(db, USER, collection, context=json.dumps(ctx),
                                              content="m", content_type=MESSAGE)
    ctx = entity.context
    return (json.loads(ctx) if isinstance(ctx, str) else dict(ctx)).get("level")


async def test_a_message_is_stamped_with_the_level_its_token_speaks_at(typed):
    _, runs, _, _, _ = await _space(typed)
    person = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    assert _file(typed, runs, person) == "typed"
    assert _file(typed, runs, _agent_auth(USER, (runs,))) == "agent"
    assert _file(typed, runs, _agent_auth(USER, (runs,)), claimed="typed") == "agent"
    assert _file(typed, runs, None, claimed="typed") == "agent"       # no request: agent
