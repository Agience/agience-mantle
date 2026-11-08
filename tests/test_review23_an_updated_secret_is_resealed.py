"""Review 23, M3: a credential's secret changed by an update is resealed, and its new seals go to
Origin on the updater's own credential, so the old copy is never served again.

Runs on the real store of `test_a_shared_key_is_used_not_read`; Origin is a recording custodian.
"""
from __future__ import annotations

from mantle.services import credential_sealing
from mantle.services.acting_principal import ForwardedCredential, acting_as

from . import test_a_shared_key_is_used_not_read as base
from .test_a_shared_key_is_used_not_read import USER, db  # noqa: F401


class _Origin:
    def __init__(self):
        self.seals = []

    def record_credential_seals(self, credential_id, sealed_sha256, endpoints_sha256, *,
                                credential):
        self.seals.append((credential_id, sealed_sha256, endpoints_sha256, credential))

    def use_verdict(self, credential_id, card_id, nonce, *, credential):
        return "verdict"


async def test_an_updated_secret_is_resealed_and_recorded(db, monkeypatch):  # noqa: F811
    from prism.trust import use_verdict

    from mantle.routers import artifacts_router

    origin = _Origin()
    monkeypatch.setattr(credential_sealing, "_seal_custodian", lambda: origin)
    monkeypatch.setattr(artifacts_router, "_use_verdict_custodian", lambda: origin)
    key, conn, _, _ = await base._shared(db)
    before = db.artifacts.get_artifact(key)["persona_sealed"]
    shelf = db.artifacts.get_artifact(key)["collection_id"]
    with acting_as(USER, principal_type="user", credential=ForwardedCredential("writer")):
        base.ws.update_artifact(db, USER, shelf, key, content="sk-rotated")
    after = db.artifacts.get_artifact(key)["persona_sealed"]
    out = await base._use(db, base._as(base.TEAMMATE, **base.LUMEN), key, conn)
    assert before != after
    assert base._open(out, key, conn) == "sk-rotated"
    (cid, sealed, _endpoints, cred), = [s for s in origin.seals if s[0] == key][-1:]
    assert cred == "writer"
    assert sealed == {p: use_verdict.sealed_digest(c) for p, c in after.items()}


async def test_an_update_that_leaves_the_secret_reseals_nothing(db, monkeypatch):  # noqa: F811
    import json

    origin = _Origin()
    monkeypatch.setattr(credential_sealing, "_seal_custodian", lambda: origin)
    key, _conn, _, _ = await base._shared(db)
    before = db.artifacts.get_artifact(key)["persona_sealed"]
    shelf = db.artifacts.get_artifact(key)["collection_id"]
    origin.seals.clear()
    with acting_as(USER, principal_type="user", credential=ForwardedCredential("writer")):
        base.ws.update_artifact(db, USER, shelf, key, context=json.dumps({"title": "renamed"}))
    assert db.artifacts.get_artifact(key)["persona_sealed"] == before and origin.seals == []
