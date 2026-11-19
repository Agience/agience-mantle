"""A key's own name: the RFC 7638 JWK thumbprint.

The thumbprint is the base64url (unpadded) SHA-256 of the key's required public members, as JSON
with the members in lexicographic order and no whitespace. It names one key and changes when the
key does, so a `kid` taken from it tells a verifier holding an older key that it must look again.
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Dict, Optional

#: The required public members of each key type (RFC 7638 §3.2, RFC 8037 §2).
_REQUIRED = {"RSA": ("e", "kty", "n"), "EC": ("crv", "kty", "x", "y"), "OKP": ("crv", "kty", "x")}


def thumbprint(jwk: Dict[str, Any]) -> Optional[str]:
    """The RFC 7638 thumbprint of `jwk`, or None when it is not a key this can name."""
    names = _REQUIRED.get(str(jwk.get("kty") or ""))
    if not names or any(not isinstance(jwk.get(n), str) for n in names):
        return None
    canon = json.dumps({n: jwk[n] for n in names}, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canon.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _b64(i: int) -> str:
    return base64.urlsafe_b64encode(i.to_bytes((i.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()


def rsa_public_jwk(public_key) -> Dict[str, str]:
    """The public JWK (`kty`, `n`, `e`) of a `cryptography` RSA public key."""
    n = public_key.public_numbers()
    return {"kty": "RSA", "n": _b64(n.n), "e": _b64(n.e)}


def names_key(jwk: Dict[str, Any], kid: str) -> bool:
    """`kid` names `jwk`: its stated `kid`, or its thumbprint."""
    return bool(kid) and (jwk.get("kid") == kid or thumbprint(jwk) == kid)
