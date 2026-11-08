"""Review 25, I2 at Mantle: telling Origin a key is active carries the acting credential.

Origin activates a share only on the credential of the key's principal or of an Admin on what it
keys, as a replacement. The client sends it beside Mantle's bound token.
"""
from __future__ import annotations

import httpx

from mantle.clients import origin_key_share_client as oc


def test_activation_sends_the_acting_credential(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["acting"] = request.headers.get("x-acting-credential")
        return httpx.Response(200, json={"rekey_pending": False})

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", lambda **kw: "svc")
    c = oc.OriginShareCustodian("http://origin.test",
                                client=httpx.Client(transport=httpx.MockTransport(handler)))
    c.activate_share("ck:K", credential="the-admin-s-own")
    assert seen == {"path": "/internal/key-shares/ck:K/activate", "acting": "the-admin-s-own"}
