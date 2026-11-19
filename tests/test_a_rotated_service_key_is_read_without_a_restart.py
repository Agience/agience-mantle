"""A service key's name is its RFC 7638 thumbprint, and a rotated key is read from the manifest
without a restart.

A peer whose key was replaced signs with the new key; the authority manifest on disk is rewritten
with it. Mantle reads the manifest again on the first token that names a key it does not hold, or
whose signature its held key does not verify, and accepts the token. A verifier holding a manifest
whose JWK carries an older stated `kid` still finds the key by its thumbprint.
"""
from __future__ import annotations

import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from mantle.services import oidc
from mantle.services.jwk_thumbprint import names_key, rsa_public_jwk, thumbprint


def _key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _pem(key) -> str:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()


def _write(keys_dir, key, kid=None):
    jwk = {**rsa_public_jwk(key.public_key()), "alg": "RS256", "use": "sig"}
    jwk["kid"] = kid or thumbprint(jwk)
    (keys_dir / "authority.manifest.json").write_text(json.dumps(
        {"trust_anchors": {"peer": {"uri": "http://peer", "jwks": {"keys": [jwk]}}}}))
    return jwk["kid"]


def _token(key, kid):
    now = int(time.time())
    return jwt.encode({"iss": "peer", "sub": "peer", "aud": "mantle", "iat": now, "exp": now + 60},
                      _pem(key), algorithm="RS256", headers={"kid": kid})


@pytest.fixture
def keys_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    monkeypatch.delenv("AGIENCE_TRUSTED_ISSUERS", raising=False)
    yield tmp_path
    monkeypatch.undo()
    oidc.reset_oidc_verifier()


def _verifier():
    oidc.reset_oidc_verifier()
    return oidc.get_oidc_verifier()


def test_the_thumbprint_is_rfc_7638():
    # RFC 7638 section 3.1's example key and its thumbprint.
    jwk = {"kty": "RSA", "e": "AQAB", "alg": "RS256", "kid": "2011-04-29",
           "n": "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6"
                "tSoc_BJECPebWKRXjBZCiFV4n3oknjhMstn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5"
                "v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajrn1n91CbOpbISD08"
                "qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8a"
                "wapJzKnqDKgw"}
    assert thumbprint(jwk) == "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs"
    assert names_key(jwk, "2011-04-29") and names_key(jwk, thumbprint(jwk))
    assert not names_key(jwk, "") and not names_key(jwk, "other")


def test_a_rotated_key_is_read_from_the_manifest_without_a_restart(keys_dir, monkeypatch):
    old, new = _key(), _key()
    old_kid = _write(keys_dir, old)
    v = _verifier()
    assert v.verify(_token(old, old_kid), expected_audience="mantle")["sub"] == "peer"

    new_kid = _write(keys_dir, new)
    assert new_kid != old_kid, "a new key carries a new name"
    monkeypatch.setattr(oidc, "_MANIFEST_REREAD_S", 0.0)
    assert v.verify(_token(new, new_kid), expected_audience="mantle")["sub"] == "peer"
    assert v.verify(_token(old, old_kid), expected_audience="mantle") is None, "the old key is gone"


def test_a_rotated_key_under_the_same_stated_kid_is_read_too(keys_dir, monkeypatch):
    old, new = _key(), _key()
    _write(keys_dir, old, kid="peer-1")
    v = _verifier()
    assert v.verify(_token(old, "peer-1"), expected_audience="mantle") is not None
    _write(keys_dir, new, kid="peer-1")
    monkeypatch.setattr(oidc, "_MANIFEST_REREAD_S", 0.0)
    assert v.verify(_token(new, "peer-1"), expected_audience="mantle")["sub"] == "peer"


def test_a_thumbprint_kid_verifies_against_a_manifest_that_states_another(keys_dir):
    key = _key()
    _write(keys_dir, key, kid="peer-1")
    tp = thumbprint(rsa_public_jwk(key.public_key()))
    assert _verifier().verify(_token(key, tp), expected_audience="mantle")["sub"] == "peer"


def test_unknown_kids_read_the_manifest_at_most_once_per_window(keys_dir, monkeypatch):
    key = _key()
    _write(keys_dir, key)
    v = _verifier()
    reads = []
    real = oidc._read_authority_manifest
    monkeypatch.setattr(oidc, "_read_authority_manifest", lambda: reads.append(1) or real())
    for i in range(20):
        assert v.verify(_token(key, f"nobody-{i}"), expected_audience="mantle") is None
    assert len(reads) <= 1


def test_mantle_signs_with_the_kid_its_manifest_states_else_the_thumbprint(tmp_path, monkeypatch):
    from mantle.services import peer_signing
    key = _key()
    (tmp_path / "mantle.private.pem").write_text(_pem(key))
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    monkeypatch.setattr(peer_signing, "_priv_pem", None)
    monkeypatch.setattr(peer_signing, "_loaded_key", None)
    monkeypatch.setattr(peer_signing, "_loaded_kid", None)
    tp = thumbprint(rsa_public_jwk(key.public_key()))
    assert jwt.get_unverified_header(peer_signing.sign_service_jwt())["kid"] == tp

    jwk = {**rsa_public_jwk(key.public_key()), "kid": "mantle-1"}
    (tmp_path / "authority.manifest.json").write_text(json.dumps(
        {"trust_anchors": {"mantle": {"jwks": {"keys": [jwk]}}}}))
    monkeypatch.setattr(peer_signing, "_loaded_kid", None)
    assert jwt.get_unverified_header(peer_signing.sign_service_jwt())["kid"] == "mantle-1"
