"""Mantle's `/use` returns Origin's signed verdict beside the sealed copy (review 21, F1).

Mantle's own checks of the caller's Invoke and the persona's Read stay, but they are Mantle's word,
and a persona opens nothing on Mantle's word alone. So `/use` asks Origin
(`POST /internal/authority/use-verdict`) on the persona's own delegation, the bearer it presented,
and returns the verdict Origin signed. No Origin, or a refusal, and no sealed copy is returned.

Runs on the real store and grants of `test_a_shared_key_is_used_not_read`; Origin is a recording
custodian, since what Origin decides is `agience-origin`'s own test.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from mantle.routers import artifacts_router as artifacts
from mantle.services.acting_principal import ForwardedCredential, acting_as

from . import test_a_shared_key_is_used_not_read as base
from .test_a_shared_key_is_used_not_read import LUMEN, TEAMMATE, _as, _shared, db  # noqa: F401

DELEGATION = "the-persona-s-own-delegation"


class _Origin:
    def __init__(self, refuse=None):
        self.asked, self.refuse = [], refuse

    def use_verdict(self, credential_id, card_id, nonce, *, credential):
        self.asked.append((credential_id, card_id, credential))  # the nonce is review 22's test
        if self.refuse is not None:
            raise self.refuse
        return "origin-signed-verdict"


async def _use(store, auth, key, card):
    with acting_as(auth.principal_id, principal_type="user",
                   credential=ForwardedCredential(DELEGATION)):
        return await artifacts.use_credential(
            artifact_id=key, body=artifacts.CredentialUseRequest(**{"for": card, "nonce": "n" * 32}), auth=auth,
            store_db=store)


async def test_the_sealed_copy_comes_with_origin_s_verdict(db, monkeypatch):  # noqa: F811 (the shared store fixture)
    origin = _Origin()
    monkeypatch.setattr(artifacts, "_use_verdict_custodian", lambda: origin)
    key, conn, _, _ = await _shared(db)
    out = await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert out["verdict"] == "origin-signed-verdict"
    assert origin.asked == [(key, conn, DELEGATION)], "asked on the persona's own delegation"
    assert base._open(out, key, conn) == "sk-team-secret"


async def test_no_verdict_no_sealed_copy(db, monkeypatch):  # noqa: F811 (the shared store fixture)
    from mantle.clients.origin_key_share_client import AuthorityRefused

    monkeypatch.setattr(artifacts, "_use_verdict_custodian",
                        lambda: _Origin(refuse=AuthorityRefused(403, "no Invoke")))
    key, conn, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert e.value.status_code == 403


async def test_no_origin_no_sealed_copy(db, monkeypatch):  # noqa: F811 (the shared store fixture)
    monkeypatch.setattr(artifacts, "_use_verdict_custodian", lambda: None)
    key, conn, _, _ = await _shared(db)
    with pytest.raises(HTTPException) as e:
        await _use(db, _as(TEAMMATE, **LUMEN), key, conn)
    assert e.value.status_code == 503


def test_the_client_asks_origin_on_a_token_bound_to_the_body(monkeypatch):
    """`OriginShareCustodian.use_verdict` posts the body, its digest bound in Mantle's token, and
    the persona's delegation as the acting credential."""
    import httpx

    from mantle.clients import origin_key_share_client as oc

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["acting"] = request.headers.get("x-acting-credential")
        return httpx.Response(200, json={"verdict": "v"})

    def sign(**kw):
        seen["signed"] = kw
        return "svc"

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", sign)
    c = oc.OriginShareCustodian("http://origin.test",
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.use_verdict("cred-1", "card-1", "n" * 32, credential=DELEGATION) == "v"
    assert seen["url"] == "http://origin.test/internal/authority/use-verdict"
    assert seen["auth"] == "Bearer svc" and seen["acting"] == DELEGATION
    claims = seen["signed"]["additional_claims"]
    assert seen["signed"]["audience"] == "origin"
    assert claims["htu"] == "/internal/authority/use-verdict" and claims["htm"] == "POST"
    assert claims["body_sha256"] == oc.entry_digest({"credential_id": "cred-1",
                                                     "card_id": "card-1", "nonce": "n" * 32})
