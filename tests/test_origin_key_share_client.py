"""`OriginShareCustodian` speaks Origin's `/internal/key-shares` and never reads a failure as absence.

Only a 404 whose body says `"absent": true` means "Origin holds no share", the one answer that lets
a key be minted; an unreachable Origin, a refusal, a bare 404 or any other status raises. Each call
signs a token bound to it.
"""
from __future__ import annotations

import httpx
import pytest

from mantle.clients import origin_key_share_client as oc


SIGNED = []


@pytest.fixture(autouse=True)
def _signed(monkeypatch):
    SIGNED.clear()

    def sign(audience="mantle", ttl_seconds=300, additional_claims=None):
        SIGNED.append({"aud": audience, "ttl": ttl_seconds, **(additional_claims or {})})
        return f"jwt-for-{audience}"

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt", sign)


def _client(handler):
    return oc.OriginShareCustodian("http://origin.test/",
                                   client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_a_share_is_fetched_with_mantle_s_service_token():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["path"] = req.url.raw_path.decode()
        return httpx.Response(200, json={"principal_id": "a/b", "x": 2, "y": "AA==", "kcv": "k"})

    assert _client(handler).get_share("a/b") == {"x": 2, "y": "AA==", "kcv": "k"}
    assert seen == {"auth": "Bearer jwt-for-origin", "path": "/internal/key-shares/a%2Fb"}


def test_each_token_is_bound_to_its_call():
    ok = lambda r: httpx.Response(200, json={"x": 2, "y": "AA==", "kcv": "k", "outcome": "stored"})  # noqa: E731
    c = _client(ok)
    c.get_share("p-1")
    c.get_share("p-1")
    c.put_share("p-2", {"x": 2, "y": "AA==", "kcv": "k"})
    g1, g2, p = SIGNED
    assert g1["ttl"] <= 60 and g1["aud"] == "origin"
    assert (g1["htm"], g1["htu"], g1["principal_id"]) == ("GET", "/internal/key-shares/p-1", "p-1")
    assert (p["htm"], p["htu"], p["principal_id"]) == ("PUT", "/internal/key-shares/p-2", "p-2")
    assert g1["jti"] and g1["jti"] != g2["jti"]


def test_only_an_explicit_absent_body_is_absent():
    assert _client(lambda r: httpx.Response(404, json={"absent": True})).get_share("p") is None
    for resp in (httpx.Response(404), httpx.Response(404, json={"detail": "Not Found"}),
                 httpx.Response(404, text="<html>not found</html>")):
        with pytest.raises(oc.CustodianUnavailable):
            _client(lambda r, x=resp: x).get_share("p")
    for status in (401, 403, 500, 503):
        with pytest.raises(oc.CustodianUnavailable):
            _client(lambda r, s=status: httpx.Response(s)).get_share("p")


def test_an_unreachable_origin_raises():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(oc.CustodianUnavailable):
        _client(handler).get_share("p")
    with pytest.raises(oc.CustodianUnavailable):
        _client(handler).put_share("p", {"x": 2, "y": "AA==", "kcv": "k"})


def test_a_put_names_what_it_supersedes_and_a_conflict_raises():
    bodies = []

    def handler(req):
        import json
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"outcome": "replaced"})

    assert _client(handler).put_share("p", {"x": 2, "y": "AA==", "kcv": "k"},
                                      supersedes="old") == "replaced"
    assert bodies == [{"x": 2, "y": "AA==", "kcv": "k", "supersedes": "old"}]
    with pytest.raises(oc.ShareConflict):
        _client(lambda r: httpx.Response(409)).put_share("p", {"x": 2, "y": "AA==", "kcv": "k"})


def test_no_origin_uri_is_refused_at_construction():
    with pytest.raises(ValueError, match="ORIGIN_URI"):
        oc.OriginShareCustodian("")
