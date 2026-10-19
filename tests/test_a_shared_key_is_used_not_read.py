"""A shared credential is usable by those granted `invoke` on it without being readable by them.

2026-10-19 (John): a key and an authorizer's secret are one thing, and a platform resource — a model
connection — is an artifact with a grant or a public grant. A connection shared that way has to be
usable through its key; requiring `read` on the key to use it would hand the key to everyone it was
shared with. `POST /artifacts/{id}/use` releases a credential's content to a platform persona acting
for someone who holds `invoke` on it, and to nothing else.

Runs on a real store with real grants, real content encryption and the real access check.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from mantle.attenuation import ACTIONS
from mantle.entities.grant import Grant as GrantEntity
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import USER, real_node

COLLECTION = "application/vnd.agience.collection+json"
CREDENTIAL = "application/vnd.agience.credential+json"
TEAMMATE = "u-teammate"
STRANGER = "u-stranger"


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def _grant(db, resource_id, grantee, actions):
    from mantle.db import lattice_api as la
    la.create_grant(db, GrantEntity(
        id=str(uuid.uuid4()), resource_id=resource_id, grantee_type="user", grantee_id=grantee,
        granted_by=USER, effect="allow", state=GrantEntity.STATE_ACTIVE,
        **{f"can_{a}": (a in actions) for a in ACTIONS}))


def _as(user, *, actor=None, principal_type="user"):
    return AuthContext(principal_id=user, principal_type=principal_type, user_id=user, actor=actor)


async def _shared_key(db):
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type=COLLECTION, name="Model connections").id
        req = Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts", "query_string": b""})
        key = await artifacts.create_artifact(
            request=req, response=Response(), auth=_as(USER), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=shelf, content_type=CREDENTIAL,
                                                 context=json.dumps({"title": "Model key"}),
                                                 content="sk-team-secret"))
        plain = await artifacts.create_artifact(
            request=req, response=Response(), auth=_as(USER), store_db=db,
            body=artifacts.CreateArtifactRequest(container_id=shelf, context=json.dumps({"title": "notes"}),
                                                 content="not a key"))
    _grant(db, shelf, TEAMMATE, {"invoke"})
    return key["id"], plain["id"]


async def _use(db, auth, artifact_id):
    with acting_as(auth.principal_id, principal_type="user"):
        return await artifacts.use_credential(artifact_id=artifact_id, auth=auth, store_db=db)


async def _read(db, auth, artifact_id):
    with acting_as(auth.principal_id, principal_type="user"):
        return await artifacts.read_artifact(artifact_id=artifact_id, auth=auth, store_db=db)


async def test_a_persona_acting_for_someone_granted_invoke_gets_the_key_to_use(db):
    key, _ = await _shared_key(db)
    out = await _use(db, _as(TEAMMATE, actor="agience-server-lumen"), key)
    assert out["content"] == "sk-team-secret"


async def test_the_person_granted_invoke_cannot_read_the_key_themselves(db):
    key, _ = await _shared_key(db)
    with pytest.raises(HTTPException) as e:
        await _read(db, _as(TEAMMATE), key)
    assert e.value.status_code in (403, 404)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE), key)          # their own token, no persona acting
    assert e.value.status_code == 403


@pytest.mark.parametrize("auth", [
    AuthContext(principal_id=TEAMMATE, principal_type="mcp_client", user_id=TEAMMATE, actor="some-client"),
    AuthContext(principal_id=TEAMMATE, principal_type="user", user_id=TEAMMATE, actor="a-third-party-server"),
])
async def test_only_a_platform_persona_is_handed_a_key(db, auth):
    key, _ = await _shared_key(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, auth, key)
    assert e.value.status_code == 403


async def test_no_grant_no_key_and_only_credentials_are_released(db):
    key, plain = await _shared_key(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(STRANGER, actor="agience-server-lumen"), key)
    assert e.value.status_code in (403, 404)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE, actor="agience-server-lumen"), plain)
    assert e.value.status_code == 404, "use releases a credential, not any content invoke reaches"


# ── the key demand ────────────────────────────────────────────────────────────────────────────
# `real_node` hands every requester every key, so the tests above cannot see which right the
# content key was demanded under. These read it off the oracle's request directly.

def test_a_content_key_is_demanded_as_read_except_where_use_widens_it(monkeypatch):
    from mantle.search.mantle import wiring
    from mantle.services import content_crypto

    seen = []

    class _Oracle:
        def get_or_create_master_key(self, principal_id, request, collection_id=None):
            seen.append(request.action)
            return b"k" * 32
    monkeypatch.setattr(wiring, "_build_oracle", lambda: _Oracle())
    with acting_as(TEAMMATE, principal_type="user"):
        content_crypto._default_master_key("owner", "shelf")
        with content_crypto.content_key_action("invoke"):
            content_crypto._default_master_key("owner", "shelf")
        content_crypto._default_master_key("owner", "shelf")
    assert seen == ["read", "invoke", "read"]


async def test_the_use_route_decrypts_under_invoke_and_a_read_under_read(db, monkeypatch):
    from mantle.services import content_crypto

    key, _ = await _shared_key(db)
    seen = []
    real = artifacts._find_artifact

    def spy(store, artifact_id):
        seen.append(content_crypto._KEY_ACTION.get())
        return real(store, artifact_id)
    monkeypatch.setattr(artifacts, "_find_artifact", spy)
    await _use(db, _as(TEAMMATE, actor="agience-server-lumen"), key)
    await _read(db, _as(USER), key)
    assert seen == ["invoke", "read"]
