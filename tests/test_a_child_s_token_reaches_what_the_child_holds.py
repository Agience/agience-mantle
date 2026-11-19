"""A child's token reaches what the child holds, action by action (FR-007).

Grants are applied to artifacts: a workflow holds Invoke on its steps, a step that reads a document
holds Read on it. Origin exchanges the parent's token for the child's and lists what the child
holds in it: the ids (`job_reach`) and the actions on each (`reach_by`). Mantle holds the token to
exactly that, inside what the person holds, and refuses anything else.
"""
from __future__ import annotations

from mantle.services import dependencies as D

from . import test_a_card_shared_alone_opens_for_its_reader as _card
from .test_a_timer_s_delegation_reaches_only_its_grant import OTHER, _allowed, _space
from .test_there_is_no_promotion_state import USER

db = _card.db


def _child_auth(user, by):
    return D.AuthContext(principal_id=user, principal_type="user", user_id=user,
                         actor="crystal", delegation_aud="crystal", narrowed=True,
                         job_reach=tuple(sorted(by)),
                         reach_actions=frozenset(a for acts in by.values() for a in acts),
                         reach_by={k: frozenset(v) for k, v in by.items()})


async def test_each_id_conducts_only_the_actions_held_on_it(db):
    space, runs, timer, other, record = await _space(db)
    auth = _child_auth(USER, {timer: ["invoke", "read"], runs: ["read"], other: ["update"]})
    assert _allowed(db, auth, timer, "read") and _allowed(db, auth, runs, "read")
    assert _allowed(db, auth, record, "read")                         # held in what it reads
    assert not _allowed(db, auth, timer, "update")                    # held on another id only
    assert not _allowed(db, auth, runs, "create")                     # not held on runs
    assert not _allowed(db, auth, other, "read")                      # update only, not read
    assert not _allowed(db, auth, space, "read")                      # nothing held there


async def test_a_child_s_token_never_passes_the_person(db):
    _, runs, _, _, _ = await _space(db)
    assert not _allowed(db, _child_auth(OTHER, {runs: ["read"]}), runs, "read")
    assert not _allowed(db, _child_auth(USER, {}), runs, "read")      # holds nothing: nothing


def test_the_map_is_read_from_origin_s_claim():
    assert D._reach_by_claim({}) is None
    assert D._reach_by_claim({"reach_by": {"a": ["read"]}}) == {"a": frozenset({"read"})}
    for bad in ([], {"a": "read"}, {"a": [1]}, {"": ["read"]}, None):
        assert D._reach_by_claim({"reach_by": bad}) == {}, bad
