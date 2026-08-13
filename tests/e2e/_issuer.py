"""A self-signed external OIDC issuer, for the multi-issuer / multi-tenant
dimension of the E2E suite.

The suite generates an RSA keypair in-process, exposes its public JWKS (which an
admin registers with Mantle via `POST /system/issuers`), and mints signed JWTs for
arbitrary subjects. Because Mantle namespaces external-IdP users by
`uuid5(tenant, sub)`, two issuers minting the same `sub` map to two DISTINCT
Mantle users — that is exactly the tenant-isolation lever we exercise.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt


def _kid() -> str:
    # Deterministic-enough; uniqueness across a run comes from the random key.
    return "e2e-" + uuid.uuid4().hex[:8]


@dataclass
class SelfSignedIssuer:
    """An in-process IdP: RSA key + issuer URL + audience + tenant namespace."""

    issuer: str
    audience: str
    namespace: str
    kid: str = field(default_factory=_kid)
    _private_pem: str = field(default="", repr=False)
    _public_jwk: dict = field(default_factory=dict)

    @classmethod
    def create(cls, *, issuer: str, audience: str, namespace: str) -> "SelfSignedIssuer":
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        priv_pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode()
        inst = cls(issuer=issuer, audience=audience, namespace=namespace, _private_pem=priv_pem)
        pub_pem = key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        # jose can build a JWK from a PEM.
        from jose import jwk as _jwk

        jwk_dict = _jwk.construct(pub_pem, "RS256").to_dict()
        jwk_dict["kid"] = inst.kid
        jwk_dict["use"] = "sig"
        jwk_dict["alg"] = "RS256"
        inst._public_jwk = jwk_dict
        return inst

    def jwks(self) -> dict:
        """Public JWKS to register with Mantle (inline, no fetch)."""
        return {"keys": [self._public_jwk]}

    def mint(self, sub: str, *, email: str = "", name: str = "", ttl: int = 3600,
             extra: dict | None = None) -> str:
        """Mint a signed JWT for `sub` as if this external IdP issued it."""
        now = int(time.time())
        claims = {
            "iss": self.issuer,
            "sub": sub,
            "aud": self.audience,
            "iat": now,
            "exp": now + ttl,
        }
        if email:
            claims["email"] = email
        if name:
            claims["name"] = name
        if extra:
            claims.update(extra)
        return jwt.encode(claims, self._private_pem, algorithm="RS256",
                          headers={"kid": self.kid})
