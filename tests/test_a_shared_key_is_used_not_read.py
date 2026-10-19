"""A shared secret is released only sealed to a proven persona, for a card that names it, to someone
granted `invoke` — never readable by them, and never in clear outside that persona.

2026-10-19 (John): a key and an authorizer's secret are one thing; a platform resource is an
artifact with a grant or a public grant; information is crypto-secured, visible but not readable,
and a secret is in clear only at the point of use, with proper authentication and delegation.

Runs on a real store with real grants, real content encryption and the real access check.
"""
from __future__ import annotations

import json
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
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
CONNECTION = "application/vnd.agience.llm-connection+json"
TEAMMATE = "u-teammate"
STRANGER = "u-stranger"
LUMEN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwk(key, kid):
    from prism.trust import sealing
    n = key.public_key().public_numbers()
    b = lambda i: sealing._b64(i.to_bytes((i.bit_length() + 7) // 8, "big"))
    return {"kty": "RSA", "kid": kid, "n": b(n.n), "e": b(n.e)}


class _Manifest:
    def get_jwks(self, name):
        if name == "lumen":
            return {"keys": [_jwk(LUMEN_KEY, "lumen-1")]}
        raise KeyError(name)


@pytest.fixture
def db(tmp_path, monkeypatch):
    import prism.trust.authority_trust as at
    monkeypatch.setattr(at, "get_authority_manifest", lambda: _Manifest())
    return real_node(tmp_path, monkeypatch)


def _grant(db, resource_id, grantee, actions):
    from mantle.db import lattice_api as la
    la.create_grant(db, GrantEntity(
        id=str(uuid.uuid4()), resource_id=resource_id, grantee_type="user", grantee_id=grantee,
        granted_by=USER, effect="allow", state=GrantEntity.STATE_ACTIVE,
        **{f"can_{a}": (a in actions) for a in ACTIONS}))


def _as(user, *, actor=None, signer=None, principal_type="user"):
    return AuthContext(principal_id=user, principal_type=principal_type, user_id=user, actor=actor,
                       actor_signer=signer)


LUMEN = dict(actor="agience-server-lumen", signer="lumen")


async def _create(db, **body):
    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts", "query_string": b""})
    with acting_as(USER, principal_type="user"):
        return await artifacts.create_artifact(request=req, response=Response(), auth=_as(USER), store_db=db,
                                               body=artifacts.CreateArtifactRequest(**body))


async def _shared(db):
    """A shelf holding a key, a connection naming it, another connection and a plain note; the
    teammate holds `invoke` on the shelf and nothing else."""
    with acting_as(USER, principal_type="user"):
        shelf = ws.create_container(db, USER, content_type=COLLECTION, name="Model connections").id
    key = await _create(db, container_id=shelf, content_type=CREDENTIAL,
                        context=json.dumps({"title": "Model key"}), content="sk-team-secret")
    conn = await _create(db, container_id=shelf, content_type=CONNECTION, context=json.dumps(
        {"title": "Team model", "provider": "anthropic", "credentials_ref": {"secret_id": key["id"]}}))
    other = await _create(db, container_id=shelf, content_type=CONNECTION, context=json.dumps(
        {"title": "Other model", "credentials_ref": {"secret_id": "some-other-key"}}))
    note = await _create(db, container_id=shelf, context=json.dumps({"title": "notes"}), content="x")
    _grant(db, shelf, TEAMMATE, {"invoke"})
    return key["id"], conn["id"], other["id"], note["id"]


async def _use(db, auth, key, card):
    with acting_as(auth.principal_id, principal_type="user"):
        return await artifacts.use_credential(artifact_id=key, body=artifacts.CredentialUseRequest(**{"for": card}),
                                              auth=auth, store_db=db)


def _open(out, key, card):
    from prism.trust import sealing
    return sealing.open_sealed(out["sealed"], LUMEN_KEY, kid="lumen-1", context=f"{key}|{card}").decode()


async def test_the_proven_persona_gets_the_secret_sealed_to_its_key_for_that_card(db):
    key, conn, _, _ = await _shared(db)
    out = await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert "sk-team-secret" not in json.dumps(out), "the answer carries ciphertext only"
    assert _open(out, key, conn) == "sk-team-secret"
    from prism.trust import sealing
    with pytest.raises(sealing.SealError):
        sealing.open_sealed(out["sealed"], LUMEN_KEY, kid="lumen-1", context=f"{key}|another-card")


async def test_the_person_granted_invoke_cannot_read_the_secret(db):
    key, _, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        with acting_as(TEAMMATE, principal_type="user"):
            await artifacts.read_artifact(artifact_id=key, auth=_as(TEAMMATE), store_db=db)
    assert e.value.status_code in (403, 404)


@pytest.mark.parametrize("auth", [
    _as(TEAMMATE),                                                          # their own sign-in
    _as(TEAMMATE, actor="agience-server-lumen", signer="chorus"),           # asserted, not proven
    _as(TEAMMATE, actor="agience-server-lumen"),                            # no signer at all
    _as(TEAMMATE, actor="agience-server-lumen", signer="iris"),             # signer is someone else
    _as(TEAMMATE, actor="some-client", principal_type="mcp_client"),
])
async def test_only_a_proven_persona_is_handed_a_secret(db, auth):
    key, conn, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, auth, key, conn)
    assert e.value.status_code == 403


async def test_a_secret_is_released_only_for_a_card_that_names_it_and_only_to_its_persona(db):
    key, conn, other, note = await _shared(db)
    for card in (other, note):
        with pytest.raises(HTTPException) as e:
            await _use(db, _as(TEAMMATE, **LUMEN), key, card)
        assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE, actor="agience-server-iris", signer="iris"), key, conn)
    assert e.value.status_code == 403, "an LLM connection's key is Lumen's to use, not Iris's"


async def test_no_grant_no_secret(db):
    key, conn, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(STRANGER, **LUMEN), key, conn)
    assert e.value.status_code in (403, 404)


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

    key, conn, _, _ = await _shared(db)
    seen = []
    real = artifacts._find_artifact

    def spy(store, artifact_id):
        seen.append(content_crypto._KEY_ACTION.get())
        return real(store, artifact_id)
    monkeypatch.setattr(artifacts, "_find_artifact", spy)
    await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    with acting_as(USER, principal_type="user"):
        await artifacts.read_artifact(artifact_id=key, auth=_as(USER), store_db=db)
    assert seen == ["invoke", "invoke", "read"]
