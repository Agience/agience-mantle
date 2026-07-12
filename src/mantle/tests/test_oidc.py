"""Tests for IdP-agnostic external-OIDC token verification."""
import base64
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from services.oidc import OidcVerifier


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _keypair_and_jwk(kid: str):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    nums = key.public_key().public_numbers()
    jwk = {
        "kty": "RSA", "alg": "RS256", "use": "sig", "kid": kid,
        "n": _b64u(nums.n.to_bytes((nums.n.bit_length() + 7) // 8, "big")),
        "e": _b64u(nums.e.to_bytes((nums.e.bit_length() + 7) // 8, "big")),
    }
    return priv_pem, jwk


def _token(priv, iss, aud, kid, sub="user-123", **extra):
    now = int(time.time())
    claims = {"sub": sub, "iss": iss, "aud": aud, "iat": now, "exp": now + 3600, **extra}
    return jwt.encode(claims, priv, "RS256", headers={"kid": kid})


def test_verifies_external_idp_token():
    priv, jwk = _keypair_and_jwk("entra-1")
    iss = "https://login.microsoftonline.com/tenant-abc/v2.0"
    aud = "my-app-client-id"
    v = OidcVerifier([{"issuer": iss, "audience": aud, "jwks": {"keys": [jwk]}}])

    assert v.is_trusted(iss)
    assert not v.is_trusted("https://attacker.example")

    claims = v.verify(_token(priv, iss, aud, "entra-1", sub="entra-user-xyz"))
    assert claims is not None
    assert claims["sub"] == "entra-user-xyz"
    assert claims["iss"] == iss


def test_rejects_wrong_audience():
    priv, jwk = _keypair_and_jwk("k1")
    iss = "https://idp.example/"
    v = OidcVerifier([{"issuer": iss, "audience": "right-aud", "jwks": {"keys": [jwk]}}])
    assert v.verify(_token(priv, iss, "wrong-aud", "k1")) is None


def test_rejects_untrusted_issuer():
    priv, jwk = _keypair_and_jwk("k1")
    v = OidcVerifier([{"issuer": "https://trusted", "audience": "a", "jwks": {"keys": [jwk]}}])
    # token from an issuer we don't trust
    assert v.verify(_token(priv, "https://untrusted", "a", "k1")) is None


def test_rejects_unknown_kid():
    priv, jwk = _keypair_and_jwk("k1")
    iss = "https://idp.example/"
    v = OidcVerifier([{"issuer": iss, "audience": "a", "jwks": {"keys": [jwk]}}])
    # signed but header kid doesn't match the published key
    assert v.verify(_token(priv, iss, "a", "other-kid")) is None


def test_rejects_bad_signature():
    priv_a, _ = _keypair_and_jwk("k1")
    _, jwk_b = _keypair_and_jwk("k1")  # different key, same kid
    iss = "https://idp.example/"
    v = OidcVerifier([{"issuer": iss, "audience": "a", "jwks": {"keys": [jwk_b]}}])
    assert v.verify(_token(priv_a, iss, "a", "k1")) is None


# -- multi-tenant identity ----------------------------------------------------

def _multi_tenant_verifier():
    return OidcVerifier([
        {"issuer": "https://login.microsoftonline.com/entra/v2.0", "audience": "app", "jwks": {"keys": []}},
        {"issuer": "https://acme.auth0.com/", "audience": "app", "jwks": {"keys": []}},
        {"issuer": "https://idp.rotates.example/", "audience": "app", "jwks": {"keys": []},
         "namespace": "tenant-pinned-key"},
    ])


def test_external_user_id_is_deterministic():
    v = _multi_tenant_verifier()
    iss = "https://acme.auth0.com/"
    a = v.external_user_id({"iss": iss, "sub": "user-42"})
    b = v.external_user_id({"iss": iss, "sub": "user-42"})
    assert a is not None and a == b


def test_same_sub_different_idp_yields_different_users():
    """The multi-tenant guarantee: a colliding `sub` across two IdPs must NOT
    map to the same Agience user."""
    v = _multi_tenant_verifier()
    entra = v.external_user_id({"iss": "https://login.microsoftonline.com/entra/v2.0", "sub": "12345"})
    auth0 = v.external_user_id({"iss": "https://acme.auth0.com/", "sub": "12345"})
    assert entra and auth0 and entra != auth0


def test_pinned_namespace_survives_issuer_change():
    """A pinned `namespace` keeps identity stable even if the issuer URL changes."""
    v1 = OidcVerifier([{"issuer": "https://old-url/", "audience": "a",
                        "jwks": {"keys": []}, "namespace": "stable-tenant"}])
    v2 = OidcVerifier([{"issuer": "https://new-url/", "audience": "a",
                        "jwks": {"keys": []}, "namespace": "stable-tenant"}])
    assert (v1.external_user_id({"iss": "https://old-url/", "sub": "u1"})
            == v2.external_user_id({"iss": "https://new-url/", "sub": "u1"}))


def test_external_user_id_none_for_untrusted():
    v = _multi_tenant_verifier()
    assert v.external_user_id({"iss": "https://stranger/", "sub": "u1"}) is None
    assert v.external_user_id({"iss": "https://acme.auth0.com/"}) is None  # no sub


def test_tenant_for():
    v = _multi_tenant_verifier()
    assert v.tenant_for("https://acme.auth0.com/") == "https://acme.auth0.com/"
    assert v.tenant_for("https://idp.rotates.example/") == "tenant-pinned-key"  # pinned
    assert v.tenant_for("https://stranger/") is None
