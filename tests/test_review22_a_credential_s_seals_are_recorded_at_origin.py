"""Review 22, F2 and F3 at Mantle.

F2: when a credential is written, Mantle sends Origin, on the writer's own credential, the SHA-256
of each persona's sealed copy and of the endpoint list (`OriginShareCustodian.record_credential_seals`),
so every verdict names the copy its writer made, and a copy Mantle seals later does not match.
F3: `/use` takes the persona's fresh nonce and forwards it to Origin for the verdict.

Runs on the real store of `test_a_shared_key_is_used_not_read`; Origin is a recording custodian.
"""
from __future__ import annotations

import json

import pytest

from mantle.routers import artifacts_router as artifacts
from mantle.services import credential_sealing
from mantle.services.acting_principal import ForwardedCredential, acting_as

from . import test_a_shared_key_is_used_not_read as base
from .test_a_shared_key_is_used_not_read import API, CREDENTIAL, USER, db  # noqa: F401

WRITER = "the-writer-s-own-token"
NONCE = "n" * 32


class _Origin:
    def __init__(self):
        self.seals, self.verdicts = [], []

    def record_credential_seals(self, credential_id, sealed_sha256, endpoints_sha256, *,
                                credential):
        self.seals.append((credential_id, sealed_sha256, endpoints_sha256, credential))

    def use_verdict(self, credential_id, card_id, nonce, *, credential):
        self.verdicts.append((credential_id, card_id, nonce, credential))
        return "verdict"


async def test_a_credential_s_seals_are_recorded_on_the_writer_s_credential(db, monkeypatch):  # noqa: F811
    from prism.trust import use_verdict

    origin = _Origin()
    monkeypatch.setattr(credential_sealing, "_seal_custodian", lambda: origin)
    with acting_as(USER, principal_type="user"):
        shelf = base.ws.create_container(db, USER, content_type=base.COLLECTION, name="Keys").id
    with acting_as(USER, principal_type="user", credential=ForwardedCredential(WRITER)):
        key = await base._create(db, container_id=shelf, content_type=CREDENTIAL, content="sk-x",
                                 context=json.dumps({"title": "k", "allowed_endpoints": [API]}))
    raw = db.artifacts.get_artifact(key["id"])
    (cid, sealed, endpoints, cred), = origin.seals
    assert (cid, cred) == (key["id"], WRITER)
    assert sealed == {p: use_verdict.sealed_digest(c) for p, c in raw["persona_sealed"].items()}
    assert endpoints == use_verdict.endpoints_digest([API])


async def test_use_forwards_the_persona_s_nonce(db, monkeypatch):  # noqa: F811
    origin = _Origin()
    monkeypatch.setattr(artifacts, "_use_verdict_custodian", lambda: origin)
    key, conn, _, _ = await base._shared(db)
    with acting_as(base.TEAMMATE, principal_type="user",
                   credential=ForwardedCredential("the-persona-s-delegation")):
        await artifacts.use_credential(
            artifact_id=key, auth=base._as(base.TEAMMATE, **base.LUMEN), store_db=db,
            body=artifacts.CredentialUseRequest(**{"for": conn, "nonce": NONCE}))
    assert origin.verdicts == [(key, conn, NONCE, "the-persona-s-delegation")]


def test_use_takes_no_request_without_a_nonce():
    with pytest.raises(Exception):
        artifacts.CredentialUseRequest(**{"for": "card-1"})


async def test_a_new_credential_s_seals_go_to_origin_after_its_creation_edge(db, monkeypatch):  # noqa: F811
    """Review 23, M2: Origin keeps a record only from the subject who added the credential's
    creation edge, so Mantle sends it once that edge is written, not before."""
    order = []
    origin = _Origin()
    real_record = origin.record_credential_seals

    def record(*a, **k):
        order.append("seals")
        return real_record(*a, **k)

    origin.record_credential_seals = record
    real_edge = base.ws.store.add_artifact_to_collection

    def edge(*a, **k):
        order.append("edge")
        return real_edge(*a, **k)

    monkeypatch.setattr(credential_sealing, "_seal_custodian", lambda: origin)
    monkeypatch.setattr(base.ws.store, "add_artifact_to_collection", edge)
    with acting_as(USER, principal_type="user"):
        shelf = base.ws.create_container(db, USER, content_type=base.COLLECTION, name="Keys").id
    order.clear()
    with acting_as(USER, principal_type="user", credential=ForwardedCredential(WRITER)):
        await base._create(db, container_id=shelf, content_type=CREDENTIAL, content="sk-x",
                           context=json.dumps({"title": "k", "allowed_endpoints": [API]}))
    assert order.index("edge") < order.index("seals"), order
