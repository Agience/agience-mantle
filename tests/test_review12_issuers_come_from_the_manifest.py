"""Review 12, H1 and H2: Mantle's trust in an issuer roots in configuration, never in an artifact.

* The authority manifest's anchors always win: an issuer artifact for Origin's own issuer, or for
  any name the manifest pins, never adds a key, so a seeded copy cannot outlive a rotation and an
  admin cannot add a key for Origin.
* A stored artifact never adds an issuer of its own: an external IdP is trusted only when it is
  configured (`AGIENCE_TRUSTED_ISSUERS`), and its artifact only keeps it live (revocation narrows).
  Trust never rests on who created the artifact (§51).
* `POST /system/issuers` is gone; nothing seeds a copy of the manifest's anchors.
* A JWKS is fetched only from an address Mantle allowlists.
"""
from __future__ import annotations

import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwk, jwt

from mantle import config

ISS = "https://origin.example.test"


def _pair(kid):
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                          serialization.NoEncryption())
    pub = k.public_key().public_bytes(serialization.Encoding.PEM,
                                      serialization.PublicFormat.SubjectPublicKeyInfo)
    j = jwk.construct(pub, "RS256").to_dict()
    j = {a: (b.decode() if isinstance(b, bytes) else b) for a, b in j.items()}
    j.update({"kid": kid, "use": "sig"})
    return pem, j


def _manifest(tmp_path, monkeypatch, anchors):
    (tmp_path / "authority.manifest.json").write_text(json.dumps(
        {"issuer": ISS, "artifact_id": "x", "trust_anchors": anchors}))
    monkeypatch.setenv("KEYS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "AUTHORITY_ISSUER", ISS, raising=False)


def _token(pem, kid, iss=ISS, aud=ISS):
    now = time.time()
    return jwt.encode({"iss": iss, "sub": "11111111-1111-1111-1111-111111111111", "aud": aud,
                       "iat": now, "exp": now + 3600}, pem, algorithm="RS256",
                      headers={"kid": kid})


def test_a_seeded_copy_of_origin_s_anchor_never_outlives_a_rotation(tmp_path, monkeypatch):
    from mantle.services.oidc import OidcVerifier

    old_pem, old_jwk = _pair("origin-1")
    new_pem, new_jwk = _pair("origin-2")
    _manifest(tmp_path, monkeypatch, {"origin": {"jwks": {"keys": [new_jwk]}}})
    v = OidcVerifier(trusted_issuers=[])
    v._rebuild([{"issuer": ISS, "jwks": {"keys": [old_jwk]}, "role": "platform"}])
    assert v.verify(_token(old_pem, "origin-1"), expected_audience=ISS) is None
    assert v.verify(_token(new_pem, "origin-2"), expected_audience=ISS) is not None


def test_a_service_anchor_rotated_by_fingerprint_is_the_only_one_trusted(tmp_path, monkeypatch):
    """`rotate-keys --accept NAME:FP` rewrites the service's anchor; the old key fails at once."""
    from mantle.services.oidc import OidcVerifier

    old_pem, old_jwk = _pair("crystal-1")
    new_pem, new_jwk = _pair("crystal-2")
    _manifest(tmp_path, monkeypatch, {"crystal": {"jwks": {"keys": [new_jwk]}}})
    v = OidcVerifier(trusted_issuers=[])
    v._rebuild([{"issuer": "crystal", "jwks": {"keys": [old_jwk]}, "role": "platform"}])
    assert v.verify(_token(old_pem, "crystal-1", iss="crystal", aud="mantle"),
                    expected_audience="mantle") is None
    assert v.verify(_token(new_pem, "crystal-2", iss="crystal", aud="mantle"),
                    expected_audience="mantle") is not None


def test_an_artifact_never_adds_a_key_for_origin(tmp_path, monkeypatch):
    from mantle.services.oidc import OidcVerifier

    _pem, real = _pair("origin-1")
    atk_pem, atk = _pair("atk-1")
    _manifest(tmp_path, monkeypatch, {"origin": {"jwks": {"keys": [real]}}})
    v = OidcVerifier(trusted_issuers=[])
    v._rebuild([{"issuer": ISS, "jwks": {"keys": [atk]}, "role": "platform"},
                {"issuer": ISS, "jwks": {"keys": [atk]}, "role": "external",
                 "audience": ISS}])
    assert v.verify(_token(atk_pem, "atk-1"), expected_audience=ISS) is None
    assert not v.is_trusted(ISS)


def test_an_artifact_never_adds_an_issuer_configuration_did_not(tmp_path, monkeypatch):
    from mantle.services.oidc import OidcVerifier

    _manifest(tmp_path, monkeypatch, {})
    atk_pem, atk = _pair("idp-1")
    v = OidcVerifier(trusted_issuers=[])
    v._rebuild([{"issuer": "https://idp.attacker.test", "jwks": {"keys": [atk]},
                 "role": "external", "audience": "x"}])
    assert not v.is_trusted("https://idp.attacker.test")
    # Configured, and kept live by its artifact: trusted, with the configured keys.
    cfg = {"issuer": "https://idp.example.test", "jwks": {"keys": [atk]}, "audience": "x"}
    v2 = OidcVerifier(trusted_issuers=[cfg])
    v2._rebuild([{"issuer": "https://idp.example.test", "role": "external", "audience": "x"}])
    assert v2.is_trusted("https://idp.example.test")
    # Configured but revoked (no live artifact once the store has been read): not trusted.
    v2._rebuild([], from_store=True)
    assert not v2.is_trusted("https://idp.example.test")


def test_no_route_adds_an_issuer():
    from mantle.routers import system_router
    paths = {(m, r.path) for r in system_router.router.routes for m in getattr(r, "methods", ())}
    assert ("POST", "/system/issuers") not in paths and ("POST", "/issuers") not in paths


def test_creating_an_issuer_for_origin_or_a_pinned_name_is_refused(tmp_path, monkeypatch):
    from mantle.services import issuers

    _manifest(tmp_path, monkeypatch, {"crystal": {"jwks": {"keys": []}}})
    for iss in (ISS, "crystal"):
        with pytest.raises(ValueError, match="manifest"):
            issuers.create_issuer_artifact(object(), issuer=iss, authorized_by="x",
                                           jwks={"keys": []}, audience="a")


def test_a_jwks_is_fetched_only_from_an_allowlisted_address(monkeypatch):
    from mantle.services import oidc

    fetched = []
    monkeypatch.setattr(oidc.httpx, "get", lambda url, **k: fetched.append(url) or (_ for _ in ()).throw(RuntimeError("no")))
    v = oidc.OidcVerifier(trusted_issuers=[])
    monkeypatch.delenv("MANTLE_JWKS_FETCH_ALLOWLIST", raising=False)
    assert v._keys("https://idp.attacker.test",
                   {"issuer": "https://idp.attacker.test",
                    "jwks_uri": "http://169.254.169.254/latest"}, force=False) == []
    assert fetched == []
    monkeypatch.setenv("MANTLE_JWKS_FETCH_ALLOWLIST", "https://login.example.test")
    v._keys("https://login.example.test", {"issuer": "https://login.example.test",
                                           "jwks_uri": "https://login.example.test/keys"},
            force=True)
    assert fetched == ["https://login.example.test/keys"]
