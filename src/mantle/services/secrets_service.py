"""
Secrets Service -- Generic encrypted credential storage.

Replaces the LLM-specific key storage that used to live in mantle's
`llm_service` (now at `chorus/verso/llm.py`) with a general-purpose secrets
keeper backed by `person.preferences.secrets` in the lattice.

Storage shape:
    person.preferences.secrets = [
        {
            "id":            "<uuid>",
            "type":          "llm_key" | "github_token" | "integration_key" | ...,
            "provider":      "openai" | "anthropic" | "github" | ...,
            "label":         "<user-friendly name>",
            "encrypted_value": "<Fernet token>",
            "created_time":  "<ISO-8601>",
            "is_default":    true | false,
        },
        ...
    ]

Encryption:
    Symmetric Fernet (AES-128-CBC + HMAC-SHA256) using the platform KEK, loaded at
    startup from ``KEYS_DIR/encryption.key`` via ``origin.key_manager`` (a Fernet key;
    generate with
    ``python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"``).
    The KEK is held exclusively by Core; servers receive secrets wrapped via JWE
    (RSA-OAEP-256 + AES-256-GCM) for the requesting server's registered public key.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from cryptography.fernet import Fernet
from mantle.db.store import Database

from prism.trust.key_manager import get_encryption_key
from mantle.db import identity_backend as identity_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Encryption bootstrap
# ---------------------------------------------------------------------------

_cipher: Optional[Fernet] = None

def _get_cipher() -> Optional[Fernet]:
    global _cipher
    if _cipher is not None:
        return _cipher
    key = get_encryption_key()
    if not key:
        raise RuntimeError(
            "PLATFORM_ENCRYPTION_KEY is not set. "
            "Generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    try:
        _cipher = Fernet(key.encode() if isinstance(key, str) else key)
        return _cipher
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Fernet cipher: {exc}") from exc


def encrypt_value(plaintext: str) -> str:
    """Encrypt a secret value for storage."""
    cipher = _get_cipher()
    return cipher.encrypt(plaintext.encode()).decode()


def decrypt_value(encrypted: str) -> str:
    """Decrypt a stored secret value."""
    cipher = _get_cipher()
    try:
        return cipher.decrypt(encrypted.encode()).decode()
    except Exception as exc:
        raise RuntimeError(f"Failed to decrypt secret value: {exc}") from exc


# ---------------------------------------------------------------------------
# SecretConfig value object
# ---------------------------------------------------------------------------

class SecretConfig:
    """In-memory representation of one stored secret (value never exposed)."""

    def __init__(
        self,
        id: str,
        type: str,
        provider: str,
        label: str,
        encrypted_value: str,
        created_time: str,
        is_default: bool = False,
        authorizer_id: str = "",
        expires_at: str = "",
    ) -> None:
        self.id = id
        self.type = type           # e.g. "llm_key", "github_token"
        self.provider = provider   # e.g. "openai", "github"
        self.label = label
        self.encrypted_value = encrypted_value
        self.created_time = created_time
        self.is_default = is_default
        self.authorizer_id = authorizer_id
        self.expires_at = expires_at  # ISO-8601 UTC expiry (for bearer tokens)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "type": self.type,
            "provider": self.provider,
            "label": self.label,
            "encrypted_value": self.encrypted_value,
            "created_time": self.created_time,
            "is_default": self.is_default,
            "authorizer_id": self.authorizer_id,
        }
        if self.expires_at:
            d["expires_at"] = self.expires_at
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SecretConfig":
        return cls(
            id=data["id"],
            type=data.get("type", "llm_key"),
            provider=data.get("provider", ""),
            label=data.get("label", ""),
            encrypted_value=data.get("encrypted_value", ""),
            created_time=data.get("created_time", ""),
            is_default=data.get("is_default", False),
            authorizer_id=data.get("authorizer_id", ""),
            expires_at=data.get("expires_at", ""),
        )



def _load_prefs(db: Database, user_id: str) -> dict:
    """
    Load user preferences.
    Returns the prefs dict.
    """
    person = identity_store.get_person_by_id(db, user_id)
    prefs: dict = (person.get("preferences") if person else {}) or {}
    return prefs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_secrets(
    db: Database,
    user_id: str,
    secret_type: Optional[str] = None,
    provider: Optional[str] = None,
    secret_id: Optional[str] = None,
    authorizer_id: Optional[str] = None,
) -> List[SecretConfig]:
    """
    List stored secrets for a user (encrypted values not decrypted here).

    Args:
        db: the lattice database handle
        user_id: Authenticated user ID
        secret_type: Filter by type, e.g. "llm_key" or "github_token"
        provider: Filter by provider, e.g. "openai" or "github"
        secret_id: Filter by exact secret ID
        authorizer_id: Filter by authorizer artifact delegation
    """
    prefs = _load_prefs(db, user_id)
    results = []
    for item in prefs.get("secrets", []):
        try:
            sec = SecretConfig.from_dict(item)
        except Exception:
            continue
        if secret_type and sec.type != secret_type:
            continue
        if provider and sec.provider != provider:
            continue
        if secret_id and sec.id != secret_id:
            continue
        if authorizer_id and sec.authorizer_id != authorizer_id:
            continue
        results.append(sec)
    return results


def add_secret(
    db: Database,
    user_id: str,
    secret_type: str,
    provider: str,
    label: str,
    value: str,
    is_default: bool = False,
    authorizer_id: str = "",
    expires_at: str = "",
) -> List[SecretConfig]:
    """
    Store a new secret (value is encrypted before storage).

    When is_default=True, existing defaults for the same (type, provider)
    are cleared first.

    Returns the updated full list of secrets.
    """
    prefs = _load_prefs(db, user_id)
    secrets = list(prefs.get("secrets", []))

    if is_default:
        for s in secrets:
            if s.get("type") == secret_type and s.get("provider") == provider:
                s["is_default"] = False

    new_secret = SecretConfig(
        id=str(_uuid.uuid4()),
        type=secret_type,
        provider=provider,
        label=label,
        encrypted_value=encrypt_value(value),
        created_time=datetime.now(timezone.utc).isoformat(),
        is_default=is_default,
        authorizer_id=authorizer_id,
        expires_at=expires_at,
    )
    secrets.append(new_secret.to_dict())
    prefs["secrets"] = secrets
    identity_store.update_person_preferences(db, user_id, prefs)

    return [SecretConfig.from_dict(s) for s in secrets]


def delete_secret(db: Database, user_id: str, secret_id: str) -> List[SecretConfig]:
    """Delete a secret by ID. Returns updated list."""
    prefs = _load_prefs(db, user_id)
    secrets = [s for s in prefs.get("secrets", []) if s.get("id") != secret_id]
    prefs["secrets"] = secrets
    identity_store.update_person_preferences(db, user_id, prefs)
    return [SecretConfig.from_dict(s) for s in secrets]


def set_default_secret(db: Database, user_id: str, secret_id: str) -> List[SecretConfig]:
    """
    Set a secret as the default for its (type, provider) combination.
    Clears the existing default for that combination first.
    """
    prefs = _load_prefs(db, user_id)
    secrets = list(prefs.get("secrets", []))

    # Find the target to determine its type+provider
    target_type: Optional[str] = None
    target_provider: Optional[str] = None
    for s in secrets:
        if s.get("id") == secret_id:
            target_type = s.get("type")
            target_provider = s.get("provider")
            break

    if target_type is None:
        return [SecretConfig.from_dict(s) for s in secrets]

    for s in secrets:
        if s.get("type") == target_type and s.get("provider") == target_provider:
            s["is_default"] = s.get("id") == secret_id

    prefs["secrets"] = secrets
    identity_store.update_person_preferences(db, user_id, prefs)
    return [SecretConfig.from_dict(s) for s in secrets]


def wrap_secret_for_server(plaintext: str, public_jwk: dict) -> dict:
    """JWE-wrap a plaintext secret for a specific server.

    Algorithm: RSA-OAEP-256 (key encryption) + AES-256-GCM (content encryption).
    Only the server holding the corresponding private key can decrypt.
    The PLATFORM_ENCRYPTION_KEY never leaves Core.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.backends import default_backend
    import os as _os
    import base64 as _b64

    def _b64url_decode(s: str) -> bytes:
        return _b64.urlsafe_b64decode(s + "=" * ((4 - len(s) % 4) % 4))

    def _b64url_encode(b: bytes) -> str:
        return _b64.urlsafe_b64encode(b).rstrip(b"=").decode()

    n = int.from_bytes(_b64url_decode(public_jwk["n"]), "big")
    e = int.from_bytes(_b64url_decode(public_jwk["e"]), "big")
    pub_key = RSAPublicNumbers(e, n).public_key(default_backend())

    session_key = _os.urandom(32)
    iv = _os.urandom(12)
    ct_with_tag = AESGCM(session_key).encrypt(iv, plaintext.encode("utf-8"), None)
    ciphertext = ct_with_tag[:-16]
    tag = ct_with_tag[-16:]

    encrypted_key = pub_key.encrypt(
        session_key,
        asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    return {
        "alg": "RSA-OAEP-256",
        "enc": "A256GCM",
        "ek": _b64url_encode(encrypted_key),
        "iv": _b64url_encode(iv),
        "ct": _b64url_encode(ciphertext),
        "tag": _b64url_encode(tag),
    }


def get_secret_value(
    db: Database,
    user_id: str,
    secret_type: str,
    provider: Optional[str] = None,
    secret_id: Optional[str] = None,
) -> Optional[str]:
    """
    Return the decrypted value of a secret.

    Lookup priority:
    1. Exact match by secret_id (if provided)
    2. Default secret for (type, provider)
    3. First secret matching (type, provider)
    """
    secrets = list_secrets(db, user_id, secret_type=secret_type, provider=provider)
    if not secrets:
        return None

    target: Optional[SecretConfig] = None

    if secret_id:
        for s in secrets:
            if s.id == secret_id:
                target = s
                break

    if target is None:
        for s in secrets:
            if s.is_default:
                target = s
                break

    if target is None:
        target = secrets[0]

    # Enforce expiry — an expired secret must not be handed out (bearer tokens etc.).
    if target.expires_at:
        try:
            exp = datetime.fromisoformat(str(target.expires_at).replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > exp:
                logger.info("secret %s is expired (expires_at=%s) — withholding", target.id, target.expires_at)
                return None
        except (ValueError, TypeError):
            logger.warning("secret %s has unparseable expires_at=%r — treating as non-expiring", target.id, target.expires_at)

    return decrypt_value(target.encrypted_value)


# ---------------------------------------------------------------------------
# Secrets-as-artifacts: material custody is LOCAL to Mantle (sovereign)
# ---------------------------------------------------------------------------
# A `vnd.agience.secret+json` artifact carries metadata only; its material lives
# in Mantle's own encrypted store (platform_settings, Fernet-encrypted) keyed by
# the artifact id — no Origin round-trip, so a standalone Origin-off Mantle works.
# `set_secret_material` stores it; `fetch_secret_material` is the type's `fetch`
# op handler — the dispatcher has already enforced the caller's `read` grant on
# the secret artifact, so Mantle decrypts + returns the material.
_SECRET_MATERIAL_PREFIX = "secret.material."

def set_secret_material(
    db: Database,
    owner_id: str,
    *,
    secret_id: str,
    value: str,
    secret_type: str,
    provider: str = "",
    label: str = "",
    authorizer_id: str = "",
) -> None:
    """Store/replace the material for a secret artifact in Mantle's own encrypted
    store (platform_settings), keyed by the secret artifact's id (so
    `fetch_secret_material` can resolve it). No Origin dependency."""
    del owner_id, provider, label, authorizer_id  # material is keyed by secret_id
    from mantle.services.platform_settings_service import settings as platform_settings

    platform_settings.set_setting(
        db,
        key=f"{_SECRET_MATERIAL_PREFIX}{secret_id}",
        value=value,
        category=secret_type or "secret",
        is_secret=True,
    )


def get_secret_material_by_id(secret_id: str) -> Optional[str]:
    """Resolve a secret artifact's material (plaintext) from Mantle's own encrypted store,
    keyed by the secret artifact id. Returns None if no material is stored.

    The Mantle-native counterpart to ``fetch_secret_material`` (the op-dispatch handler): callers
    that have already authorized the read out-of-band (e.g. the delegated ``/secrets/reveal``
    endpoint, which verifies the user owns the artifact) use this directly — no Chorus/crystal
    op-dispatcher, no Seraph type handler in the path."""
    from mantle.services.platform_settings_service import settings as platform_settings

    return platform_settings.get_secret(f"{_SECRET_MATERIAL_PREFIX}{secret_id}")


def fetch_secret_material(artifact: dict, body: dict, ctx: Any) -> dict:
    """`vnd.agience.secret+json` `fetch` operation — return decrypted material.

    Native dispatch handler: signature ``(artifact, body, ctx)``. The operation
    dispatcher has already enforced ``requires_grant: "read"`` on this secret
    artifact, so reaching here means the caller is authorized to use it. The
    material is resolved from Mantle's own encrypted store, keyed by the secret
    artifact id.
    """
    del body, ctx
    from mantle.services.platform_settings_service import settings as platform_settings

    secret_id = artifact.get("root_id") or artifact.get("_key") or artifact.get("id")
    if not secret_id:
        raise RuntimeError("secret artifact missing root_id")

    material = platform_settings.get_secret(f"{_SECRET_MATERIAL_PREFIX}{secret_id}")
    if material is None:
        raise RuntimeError(f"No material stored for secret artifact {secret_id}")
    return {"secret_id": secret_id, "material": material}

