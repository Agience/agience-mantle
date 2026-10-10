"""An unknown `kid` costs at most one JWKS fetch per issuer per `_JWKS_MIN_REFETCH_S`.

The `kid` is the token's own unverified header. Every miss forced a refetch, so a caller naming a
fresh `kid` per request turned each request into an outbound fetch to the issuer — and the fetch
ran under the verifier's one global lock, so every other token verification, for every issuer,
queued behind it.
"""

from __future__ import annotations

import base64
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from mantle.services import oidc
from mantle.services.oidc import OidcVerifier

ISS = "https://idp.example/"
AUD = "my-app"


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _keypair(kid):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption()).decode()
    n = key.public_key().public_numbers()
    jwk = {"kty": "RSA", "alg": "RS256", "use": "sig", "kid": kid,
           "n": _b64u(n.n.to_bytes((n.n.bit_length() + 7) // 8, "big")),
           "e": _b64u(n.e.to_bytes((n.e.bit_length() + 7) // 8, "big"))}
    return pem, jwk


def _token(pem, kid):
    now = int(time.time())
    return jwt.encode({"sub": "u", "iss": ISS, "aud": AUD, "iat": now, "exp": now + 3600},
                      pem, "RS256", headers={"kid": kid})


class _Resp:
    def __init__(self, keys):
        self._keys = keys

    def raise_for_status(self):
        pass

    def json(self):
        return {"keys": self._keys}


@pytest.fixture()
def idp(monkeypatch):
    """A JWKS endpoint serving `state["keys"]`, counting fetches and whether the lock was held."""
    state = {"keys": [], "fetches": 0, "held": [], "verifier": None}

    def fake_get(url, timeout=None):
        state["fetches"] += 1
        state["held"].append(state["verifier"]._lock.locked())
        return _Resp(list(state["keys"]))

    monkeypatch.setattr(oidc.httpx, "get", fake_get)
    state["verifier"] = OidcVerifier([{"issuer": ISS, "audience": AUD,
                                       "jwks_uri": "https://idp.example/jwks"}])
    return state


def test_a_stream_of_unknown_kids_fetches_once(idp):
    pem, jwk = _keypair("real")
    idp["keys"] = [jwk]
    v = idp["verifier"]
    assert v.verify(_token(pem, "real")) is not None
    for i in range(20):
        assert v.verify(_token(pem, f"forged-{i}")) is None
    assert idp["fetches"] == 1, f"{idp['fetches']} JWKS fetches for one real key and 20 forged kids"


def test_a_rotated_key_is_picked_up_once_the_spacing_has_passed(idp, monkeypatch):
    old_pem, old_jwk = _keypair("old")
    new_pem, new_jwk = _keypair("new")
    idp["keys"] = [old_jwk]
    v = idp["verifier"]
    assert v.verify(_token(old_pem, "old")) is not None

    idp["keys"] = [old_jwk, new_jwk]
    clock = time.time() + oidc._JWKS_MIN_REFETCH_S + 1
    monkeypatch.setattr(oidc.time, "time", lambda: clock)
    assert v.verify(_token(new_pem, "new")) is not None
    assert idp["fetches"] == 2


def test_the_fetch_runs_outside_the_verifier_lock(idp):
    pem, jwk = _keypair("real")
    idp["keys"] = [jwk]
    assert idp["verifier"].verify(_token(pem, "real")) is not None
    assert idp["held"] == [False], "the JWKS fetch ran holding the verifier's global lock"
