# ruff: noqa: F811 — the real-Origin fixtures are imported and named as test parameters
"""Ember's acts for a person, end to end under threshold custody, against a real Origin (review 10).

John, 2026-11-02: under custody Ember does everything except open keys. Origin accepts Ember's
narrowed delegation (`aud` and `act.sub` `ember`, Ember's scopes, five minutes) as the acting
credential for Ember's scoped acts only, and never to release a content key.

* remember: the person's memory collection is made at their sign-in, on their own token
  (`user_provisioning._ensure_ember_memory`), which gives it its key and its drop key. Ember then
  writes a memory into it as a Create sealed to that drop key, written without Read.
* share: none. Ember makes no grant (John, 2026-11-02, review 11); sharing happens only in the
  person's own session.
* the read check: a verdict (`check_access`, Origin's walk), never content.

In each, Origin is never asked for a key share on the Ember delegation, and Mantle opens none.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

pytest.importorskip("origin")

from mantle.db import lattice_api as la  # noqa: E402
from mantle.db import reach_journal  # noqa: E402
from mantle.routers import artifacts_router as artifacts  # noqa: E402
from mantle.services.acting_principal import ForwardedCredential, acting_as  # noqa: E402
from mantle.services.artifact_identity import derive_artifact_id  # noqa: E402
from mantle.services.dependencies import AuthContext, check_access  # noqa: E402

from .test_a_first_sign_in_is_co_signed_by_a_real_origin import (  # noqa: E402,F401
    custodian, node, origin_client)

EMBER_SCOPES = "ember:memory ember:read"
COLLECTION = "application/vnd.agience.collection+json"


@pytest.fixture
def threshold(monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")


def _ember_delegation(origin_client, monkeypatch, person_token, scope=EMBER_SCOPES):
    from origin import config
    monkeypatch.setattr(config, "NARROW_AUDIENCES", {"ember": EMBER_SCOPES.split()},
                        raising=False)
    # Narrowed in the person's Origin session, as the web app does (review 23).
    from jose import jwt as jose_jwt

    from origin.tests import operator_console_fixtures as oc
    oc.origin_page_session(origin_client, str(jose_jwt.get_unverified_claims(person_token)["sub"]))
    r = origin_client.post("/auth/me/authority/narrow", json={"audience": "ember", "scope": scope},
                           headers={"X-Origin-Session": "1",
                                    "Origin": config.APP_URI or config.ORIGIN_URI})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _as(person, credential, *, actor=None, scope=None):
    """What Mantle's `get_auth` makes of a token: the person, the actor and scope it names, and
    the token carried to show Origin (`acting_from_auth`)."""
    auth = AuthContext(principal_id=person, principal_type="user", user_id=person, actor=actor,
                       scope=scope, delegation_aud=actor)
    return auth, acting_as(person, principal_type="user", source="request", actor=actor,
                           scope=scope, credential=ForwardedCredential(credential),
                           narrowed=auth.narrowed_audience)


def _ember(person, delegation):
    from jose import jwt as jose_jwt
    claims = jose_jwt.get_unverified_claims(delegation)
    assert claims["aud"] == claims["act"]["sub"] == "ember"
    return _as(person, delegation, actor="ember", scope=claims.get("scope"))


async def _create(db, auth, ctx, **body):
    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts",
                   "query_string": b""})
    with ctx:
        return await artifacts.create_artifact(request=req, response=Response(), auth=auth,
                                               store_db=db,
                                               body=artifacts.CreateArtifactRequest(**body))


@pytest.fixture
def released(custodian, monkeypatch):
    """Every key-share release Mantle asks of Origin, with the credential it showed."""
    seen = []
    real = custodian.get_share

    def get_share(principal_id, *, credential, **scope):
        seen.append((principal_id, credential))
        return real(principal_id, credential=credential, **scope)

    monkeypatch.setattr(custodian, "get_share", get_share)
    return seen


@pytest.fixture
def alice(custodian, node, origin_client, monkeypatch, threshold, released):
    """Alice signs in on her own token: Mantle makes her Inbox, her card and her memory
    collection, and with it the collection's key and drop key."""
    from origin.tests import authority_fixtures as af

    from mantle.services.seed_provisioning import user_provisioning as up

    monkeypatch.delenv("AGIENCE_SEEDS_ROOT", raising=False)
    person, token = af.person("Alice", email=f"alice-{uuid.uuid4().hex[:6]}@x.test")
    with acting_as(person, principal_type="user", source="request",
                   credential=ForwardedCredential(token)):
        up.provision_user(node, person, email="alice@x.test", name="Alice")
    memory = derive_artifact_id(person, "ember:memory")
    assert la.get_artifact(node, memory) is not None
    released.clear()
    return person, token, _ember_delegation(origin_client, monkeypatch, token), memory


def _ember_released(released, delegation):
    return [p for p, cred in released if cred == delegation]


async def test_remember_writes_a_drop_under_custody_and_opens_no_key(alice, node, released):
    person, _token, delegation, memory = alice
    auth, ctx = _ember(person, delegation)
    mem = await _create(node, auth, ctx, container_id=memory, content="ships Friday",
                        content_type="text/markdown", name="ships")
    raw = node.artifacts.get_artifact(mem["id"])
    assert raw.get("content_dropped") is True and "ships Friday" not in json.dumps(raw)
    assert _ember_released(released, delegation) == [], "no key share asked on Ember's delegation"


async def test_ember_creates_no_memory_collection(alice, node):
    person, _token, delegation, _memory = alice
    auth, ctx = _ember(person, delegation)
    with pytest.raises(HTTPException) as refused:
        await _create(node, auth, ctx, identity="ember:other", name="x", content_type=COLLECTION)
    assert refused.value.status_code == 403


async def test_readable_by_is_a_verdict_and_opens_no_key(alice, node, released):
    person, _token, delegation, memory = alice
    auth, ctx = _ember(person, delegation)
    with ctx:
        check_access(auth, memory, "read", node)                 # Origin's verdict: allowed
    stranger = str(uuid.uuid4())
    auth, ctx = _ember(person, delegation)
    with ctx, pytest.raises(HTTPException) as refused:
        check_access(auth, stranger, "read", node)
    assert refused.value.status_code == 404
    assert _ember_released(released, delegation) == []


async def test_ember_makes_no_grant(alice, node, released):
    """Ember shares nothing: even a Read grant on a memory it wrote, to a named person, is refused
    on its delegation, and nothing is written (review 11)."""
    from origin.tests import authority_fixtures as af

    from mantle.routers import grants_router

    person, token, delegation, memory = alice
    auth, ctx = _ember(person, delegation)
    mem = await _create(node, auth, ctx, container_id=memory, content="for bob",
                        content_type="text/markdown", name="for bob")
    bob, _bob_token = af.person("Bob")
    until = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
    auth, ctx = _ember(person, delegation)
    body = grants_router.CreateGrantRequest(resource_id=mem["id"], grantee_type="user",
                                            grantee_id=bob, can_read=True, expires_at=until)
    with ctx, pytest.raises(HTTPException) as refused:
        await grants_router.create_grant_endpoint(body=body, auth=auth, store_db=node)
    assert refused.value.status_code == 403
    assert not la.get_active_grants_for_principal_resource(node, grantee_id=bob,
                                                           resource_id=mem["id"])
    assert _ember_released(released, delegation) == []


async def test_a_key_share_is_never_released_on_an_ember_delegation(alice, custodian):
    """Origin's own rule: even asked directly, it releases nothing on Ember's delegation."""
    person, _token, delegation, memory = alice
    from mantle.clients.origin_key_share_client import AuthorityRefused
    with pytest.raises(AuthorityRefused):
        custodian.get_share(f"ck:{memory}", credential=delegation, collection_id=memory)
