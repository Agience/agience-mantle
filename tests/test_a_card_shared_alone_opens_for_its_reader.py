"""A card shared on its own, short of its collection, opens for the person it was shared with — and
only that card.

2026-10-22, use case 07 on staging: a nurse invited to one care plan held an active read grant on
it, `/grants/my-access` said allowed, and `GET /artifacts/{id}` answered 404. Content keys are
checked per collection, and the key oracle refused a reader who reached the card but not the
collection, so the read decrypted nothing. Every share of a single card failed the same way.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from starlette.requests import Request

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.routers import artifacts_router as artifacts
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

from .test_there_is_no_promotion_state import OTHER, USER, real_node


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A node whose content keys are decided as in production: the light-cone verifier, unmemoized,
    and the content path asking the oracle. `real_node` hands every requester every key and keys
    content without the oracle, so it cannot see who is refused a key."""
    from cryptography.fernet import Fernet
    from mantle.search.mantle import wiring
    from mantle.search.mantle.oracle import FernetMasterKeyStore, LightConeGrantVerifier, OracleService
    from mantle.services import content_crypto

    encrypt, decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content
    database = real_node(tmp_path, monkeypatch)
    monkeypatch.setattr(content_crypto, "encrypt_content", encrypt)
    monkeypatch.setattr(content_crypto, "decrypt_content", decrypt)
    oracle = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())),
                           grant_verifier=LightConeGrantVerifier(database, ttl_s=0))
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)
    return database


def _auth(user):
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


def _req():
    return Request({"type": "http", "method": "GET", "headers": [], "path": "/", "query_string": b""})


async def _create(db, **body):
    with acting_as(USER, principal_type="user"):
        return await artifacts.create_artifact(request=_req(), response=Response(), auth=_auth(USER),
                                               store_db=db, body=artifacts.CreateArtifactRequest(**body))


async def _read(db, user, artifact_id):
    with acting_as(user, principal_type="user"):
        return await artifacts.read_artifact(artifact_id=artifact_id, auth=_auth(user),
                                             store_db=db)


def _share(db, resource_id):
    la.create_grant(db, Grant(id=str(uuid.uuid4()), resource_id=resource_id, grantee_type="user",
                              grantee_id=OTHER, granted_by=USER, effect="allow", state=Grant.STATE_ACTIVE,
                              **{f"can_{a}": a == "read" for a in ACTIONS}))


async def _record(db):
    with acting_as(USER, principal_type="user"):
        space = ws.create_container(db, USER, content_type="application/vnd.agience.workspace+json", name="Clinic").id
    rec = (await _create(db, container_id=space, content_type="application/vnd.agience.collection+json",
                         name="Record"))["id"]
    plan = await _create(db, container_id=rec, content="Start statin.", context=json.dumps({"title": "Care plan"}))
    note = await _create(db, container_id=rec, content="Impression: angina.", context=json.dumps({"title": "Visit"}))
    return rec, plan, note


async def test_a_card_shared_alone_opens_with_its_title_for_its_reader(db):
    _, plan, _ = await _record(db)
    _share(db, plan["id"])
    got = await _read(db, OTHER, plan["id"])
    assert got["content"] == "Start statin."
    assert json.loads(got["context"])["title"] == "Care plan", "its sealed title opens too"


async def test_its_siblings_stay_closed(db):
    _, plan, note = await _record(db)
    _share(db, plan["id"])
    with pytest.raises(HTTPException) as e:
        await _read(db, OTHER, note["id"])
    assert e.value.status_code == 404


async def test_a_reader_with_no_grant_still_gets_nothing(db):
    _, plan, _ = await _record(db)
    with pytest.raises(HTTPException) as e:
        await _read(db, OTHER, plan["id"])
    assert e.value.status_code == 404
