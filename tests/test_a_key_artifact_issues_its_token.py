"""A Key artifact issues its token: an agent's, or the person's own (FR-032).

A type whose field carries the ``token`` stamp makes each of its artifacts a bearer credential. A
person's own create issues ``<prefix><root id>.<secret>``, hands it back once, and stores only its
hash and its last four characters. The token speaks as that person, acted by the agent the Key
names (none: the person's own key), held to the ids and actions the Key names, until the Key ends or is archived, or the
collection it sits in is.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from mantle.db import lattice_api as la
from mantle.routers import credentials_router, mcp_router
from mantle.services import artifact_tokens, on_create, type_offers
from mantle.services import dependencies as D
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_card_shared_alone_opens_for_its_reader import _create
from . import test_a_card_shared_alone_opens_for_its_reader as _card
from .test_a_timer_s_delegation_reaches_only_its_grant import _allowed, _space
from .test_there_is_no_promotion_state import USER

db = _card.db

KEY = "application/vnd.example.key+json"
RULE = {"stamp": "token", "prefix": "agk_", "hint": "hint", "acts": "agent",
        "reach": "collections", "actions": "actions", "ends": "expires_at"}


def _later(**delta):
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat()


@pytest.fixture
def keyed(db):
    type_offers.set_definitions({KEY: {"content_type": KEY,
                                       "fields": {"token_hash": {"on_create": RULE}}}})
    yield db
    type_offers.set_definitions({})


def _status(call):
    try:
        call()
    except Exception as refused:                                    # noqa: BLE001
        return getattr(refused, "status_code", None) or getattr(refused, "status", None)
    return None


async def _key(db, collection, reach, **over):
    ctx = {"content_type": KEY, "title": "agent key", "agent": "agent-x",
           "collections": list(reach), "actions": ["read", "create"],
           "expires_at": _later(days=1), **over}
    return await _create(db, container_id=collection, content_type=KEY, name="agent key",
                         context=json.dumps(ctx), content="")


def _stored(db, artifact_id):
    ctx = db.artifacts.versions_of(artifact_id)[-1].get("context")
    return json.loads(ctx) if isinstance(ctx, str) else dict(ctx)


def test_a_token_rule_names_only_what_a_token_is():
    assert type_offers.validate_field_rules({"token_hash": {"on_create": RULE}}) == []
    for bad in ({**RULE, "prefix": "AGK"}, {**RULE, "extra": "x"}, {**RULE, "acts": ""},
                {**RULE, "hint": "token_hash"}, {"stamp": "token"}):
        assert type_offers.validate_field_rules({"token_hash": {"on_create": bad}}), bad
    type_offers.set_definitions({KEY: {"content_type": KEY,
                                       "fields": {"token_hash": {"on_create": RULE}}}})
    try:
        assert type_offers.token_rule(KEY) == ("token_hash", RULE)
        assert type_offers.token_rule("text/markdown") is None
    finally:
        type_offers.set_definitions({})


async def test_a_person_s_create_issues_the_token_once_and_stores_its_hash(keyed):
    _, runs, _, _, _ = await _space(keyed)
    out = await _key(keyed, runs, (runs,))
    token = out["issued"]["token_hash"]
    assert token.startswith("agk_") and artifact_tokens.looks_like(token)
    stored = _stored(keyed, out["id"])
    assert stored["token_hash"] == on_create.token_hash(token) != token
    assert stored["hint"] == token[-4:]
    assert token not in json.dumps(stored)
    reread = await _card._read(keyed, USER, out["id"])
    assert "issued" not in reread and token not in json.dumps(reread, default=str)


async def test_an_agent_cannot_mint_itself_a_key(keyed):
    _, runs, _, _, _ = await _space(keyed)
    agent = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER,
                          actor="agent-x", delegation_aud="mantle", narrowed=True,
                          scope=D.AGENT_SCOPE, job_reach=(runs,),
                          reach_actions=frozenset({"read", "create"}))
    ctx = {"content_type": KEY, "title": "k", "agent": "agent-x", "collections": [runs],
           "actions": ["read"], "expires_at": _later(days=1)}

    def mint():
        with acting_as(USER, principal_type="user"), on_create.creating_as(agent):
            ws.create_workspace_artifact(keyed, USER, runs, context=json.dumps(ctx), content="",
                                         content_type=KEY)
    assert _status(mint) == 403


async def test_a_key_says_what_it_is_or_nothing_is_created(keyed):
    _, runs, _, _, _ = await _space(keyed)
    for over in ({"agent": 7}, {"agent": "  "}, {"collections": []}, {"collections": "x"},
                 {"actions": ["read", "fly"]}, {"actions": []},
                 {"expires_at": _later(days=-1)}, {"expires_at": "soon"}):
        try:
            await _key(keyed, runs, (runs,), **over)
        except Exception as refused:                                # noqa: BLE001
            assert getattr(refused, "status_code", None) == 422, over
        else:
            raise AssertionError(f"created with {over}")


async def test_what_a_key_is_does_not_change(keyed):
    _, runs, _, other, _ = await _space(keyed)
    out = await _key(keyed, runs, (runs,))
    for field, value in (("collections", [other]), ("actions", ["read", "update"]),
                         ("agent", "agent-y"), ("expires_at", _later(days=9)),
                         ("token_hash", "0" * 64), ("hint", "abcd")):
        ctx = {**_stored(keyed, out["id"]), field: value}

        def change():
            with acting_as(USER, principal_type="user"):
                ws.update_artifact(keyed, USER, runs, out["id"], context=json.dumps(ctx))
        assert _status(change) == 409, field


async def test_the_token_speaks_as_the_person_acted_by_its_agent_inside_its_reach(keyed):
    _, runs, _, other, record = await _space(keyed)
    out = await _key(keyed, runs, (runs,))
    auth = D.resolve_auth(out["issued"]["token_hash"], keyed)
    assert (auth.user_id, auth.actor, auth.scope) == (USER, "agent-x", D.AGENT_SCOPE)
    assert auth.job_reach == (runs,) and auth.reach_actions == {"read", "create"}
    assert _allowed(keyed, auth, runs, "read") and _allowed(keyed, auth, record, "read")
    assert not _allowed(keyed, auth, other, "read")
    assert not _allowed(keyed, auth, record, "update")
    me = await credentials_router.credential_self(auth=auth)
    assert me["person"] == USER and me["actor"] == "agent-x" and me["artifact"] == out["id"]
    assert me["reach"] == [runs] and me["actions"] == ["create", "read"]
    person = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER)
    with pytest.raises(HTTPException) as refused:
        await credentials_router.credential_self(auth=person)
    assert refused.value.status_code == 404


async def test_a_key_naming_no_agent_is_the_person_s_own_held_to_its_reach(keyed):
    _, runs, _, other, record = await _space(keyed)
    ctx = {"content_type": KEY, "title": "my key", "collections": [runs], "actions": ["read"],
           "expires_at": _later(days=1)}
    out = await _create(keyed, container_id=runs, content_type=KEY, name="my key",
                        context=json.dumps(ctx), content="")
    auth = D.resolve_auth(out["issued"]["token_hash"], keyed)
    assert (auth.user_id, auth.actor, auth.token_artifact) == (USER, None, out["id"])
    assert _allowed(keyed, auth, record, "read") and not _allowed(keyed, auth, other, "read")
    assert not _allowed(keyed, auth, record, "update")
    me = await credentials_router.credential_self(auth=auth)
    assert me["person"] == USER and me["actor"] is None


async def test_the_token_reaches_mcp_and_not_its_whole_reach_listings(keyed):
    _, runs, _, _, _ = await _space(keyed)
    auth = D.resolve_auth((await _key(keyed, runs, (runs,)))["issued"]["token_hash"], keyed)
    D.enforce_narrowed_bounds(auth, NS(method="POST", url=NS(path="/mcp")))
    D.enforce_narrowed_bounds(auth, NS(method="GET", url=NS(path="/credentials/self")))
    for tool in ("list_artifacts", "recall"):
        with pytest.raises(HTTPException) as refused:
            await mcp_router._call_tool(tool, {}, auth=auth, store_db=keyed, request=None)
        assert refused.value.status_code == 403, tool


async def test_a_wrong_secret_or_an_ended_key_is_refused(keyed):
    _, runs, _, _, _ = await _space(keyed)
    out = await _key(keyed, runs, (runs,))
    token = out["issued"]["token_hash"]
    forged = token[:-6] + ("AAAAAA" if not token.endswith("AAAAAA") else "BBBBBB")
    for bad in (forged, f"agk_{runs}.{'x' * 43}", "zzz_" + token[4:]):
        with pytest.raises(HTTPException) as refused:
            D.resolve_auth(bad, keyed)
        assert refused.value.status_code == 401
    # Its end passes: refused.
    doc = keyed.artifacts.versions_of(out["id"])[-1]
    ctx = json.loads(doc["context"]) if isinstance(doc["context"], str) else dict(doc["context"])
    ctx["expires_at"] = _later(seconds=-1)
    doc["context"] = json.dumps(ctx) if isinstance(doc["context"], str) else ctx
    keyed.artifacts.put_artifact(doc)
    with pytest.raises(HTTPException) as refused:
        D.resolve_auth(token, keyed)
    assert refused.value.status_code == 401


async def test_archiving_the_key_or_its_collection_ends_the_token(keyed):
    space, runs, _, _, _ = await _space(keyed)
    out = await _key(keyed, runs, (runs,))
    token = out["issued"]["token_hash"]
    assert D.resolve_auth(token, keyed).token_artifact == out["id"]
    with acting_as(USER, principal_type="user"):
        assert la.archive_artifact(keyed, USER, out["id"])
    with pytest.raises(HTTPException):
        D.resolve_auth(token, keyed)

    again = await _key(keyed, runs, (runs,))
    token = again["issued"]["token_hash"]
    assert D.resolve_auth(token, keyed)
    with acting_as(USER, principal_type="user"):
        assert la.archive_artifact(keyed, USER, runs)
    with pytest.raises(HTTPException):
        D.resolve_auth(token, keyed)


async def _patch(db, auth, artifact_id, ctx):
    with acting_as(auth.user_id, principal_type="user"):
        return await _card.artifacts.update_artifact(
            artifact_id=artifact_id, auth=auth, store_db=db,
            body=_card.artifacts.UpdateArtifactRequest(context=json.dumps(ctx)))


async def test_a_key_created_blank_issues_on_the_first_update_that_says_what_it_is(keyed):
    _, runs, _, _, _ = await _space(keyed)
    blank = await _create(keyed, container_id=runs, content_type=KEY, name="key",
                          context=json.dumps({"content_type": KEY, "title": "key",
                                              "token_hash": "0" * 64, "hint": "abcd"}),
                          content="")
    assert "issued" not in blank
    assert "token_hash" not in _stored(keyed, blank["id"]) and "hint" not in _stored(keyed, blank["id"])

    forged = await _patch(keyed, _card._auth(USER), blank["id"],
                          {**_stored(keyed, blank["id"]), "token_hash": "1" * 64})
    assert "issued" not in forged and "token_hash" not in _stored(keyed, blank["id"])

    said = {**_stored(keyed, blank["id"]), "collections": [runs], "actions": ["read"],
            "expires_at": _later(days=1), "token_hash": "2" * 64}
    out = await _patch(keyed, _card._auth(USER), blank["id"], said)
    token = out["issued"]["token_hash"]
    assert _stored(keyed, blank["id"])["token_hash"] == on_create.token_hash(token)
    seen = artifact_tokens.auth_of(keyed, token)
    assert seen.user_id == USER and seen.actor is None and seen.job_reach == (runs,)

    with pytest.raises(HTTPException) as refused:
        await _patch(keyed, _card._auth(USER), blank["id"],
                     {**_stored(keyed, blank["id"]), "agent": "agent-x"})
    assert refused.value.status_code == 409
    assert artifact_tokens.auth_of(keyed, token).actor is None


async def test_only_its_creator_s_own_token_issues_a_blank_key(keyed):
    _, runs, _, _, _ = await _space(keyed)
    blank = await _create(keyed, container_id=runs, content_type=KEY, name="key",
                          context=json.dumps({"content_type": KEY, "title": "key"}), content="")
    la.create_grant(keyed, _card.Grant(
        id="g-editor", resource_id=runs, grantee_type="user", grantee_id=_card.OTHER,
        granted_by=USER, effect="allow", state=_card.Grant.STATE_ACTIVE,
        **{f"can_{a}": a in ("read", "update") for a in _card.ACTIONS}))
    said = {**_stored(keyed, blank["id"]), "collections": [runs], "actions": ["read"],
            "expires_at": _later(days=1)}
    agent = D.AuthContext(principal_id=USER, principal_type="user", user_id=USER,
                          actor="agent-x", scope=D.AGENT_SCOPE, job_reach=(runs,),
                          reach_actions=frozenset({"read", "update"}))
    for caller in (agent, _card._auth(_card.OTHER)):
        with pytest.raises(HTTPException) as refused:
            await _patch(keyed, caller, blank["id"], said)
        assert refused.value.status_code == 403, caller
        assert "credential" in str(refused.value.detail), refused.value.detail
    assert "token_hash" not in _stored(keyed, blank["id"])
