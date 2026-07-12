import uuid

import pytest
from typing import AsyncGenerator
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport

import bcrypt as _bcrypt
import agience_core.config as _cfg
from entities.person import Person
from services.bootstrap_types import (
    ALL_PLATFORM_COLLECTION_SLUGS,
    AUTHORITY_ARTIFACT_SLUG,
    HOST_ARTIFACT_SLUG,
    AGENCY_ARTIFACT_SLUG,
    AGENT_ARTIFACT_SLUG_PREFIX,
    PLATFORM_AGENT_SLUGS,
)
from services.platform_topology import register_id
from services.dependencies import (
    get_auth,
    get_person,
    get_end_user_claims,
    AuthContext,
)
from services.dependencies import get_arango_db
from main import app
from unittest.mock import MagicMock
import main as _main_module

# ---------------------------------------------------------------------------
# Fast crypto: reduce bcrypt cost and PBKDF2 iterations so tests don't spend
# seconds on real key-stretching.  Applied before any test module imports.
# ---------------------------------------------------------------------------
_orig_gensalt = _bcrypt.gensalt
_bcrypt.gensalt = lambda rounds=4, prefix=b"2b": _orig_gensalt(rounds=4, prefix=prefix)
_cfg.PASSWORD_PBKDF2_ITERS = 1000

# Disable setup mode for all tests — the middleware blocks all non-setup routes
# when _setup_mode is True (the default at import time).
_main_module._setup_mode = False

_TEST_PLATFORM_IDS: dict[str, str] = {}

def _ensure_platform_registry():
    """Populate the platform topology registry with stable test UUIDs."""
    if not _TEST_PLATFORM_IDS:
        for slug in ALL_PLATFORM_COLLECTION_SLUGS:
            _TEST_PLATFORM_IDS[slug] = str(uuid.uuid4())
        for slug in [AUTHORITY_ARTIFACT_SLUG, HOST_ARTIFACT_SLUG, AGENCY_ARTIFACT_SLUG]:
            _TEST_PLATFORM_IDS[slug] = str(uuid.uuid4())
        for agent_slug in PLATFORM_AGENT_SLUGS:
            _TEST_PLATFORM_IDS[f"{AGENT_ARTIFACT_SLUG_PREFIX}{agent_slug}"] = str(uuid.uuid4())
    for slug, uid in _TEST_PLATFORM_IDS.items():
        register_id(slug, uid)

@pytest.fixture(autouse=True)
def _seed_platform_registry():
    """Ensure platform topology registry is populated before every test."""
    _ensure_platform_registry()

@pytest.fixture(autouse=True, scope="session")
def _init_test_encryption_key():
    """Initialize the encryption key with a test value so secrets_service works in tests."""
    from cryptography.fernet import Fernet
    import agience_core.key_manager as _km
    _km._encryption_key = Fernet.generate_key().decode()


@pytest.fixture(autouse=True, scope="session")
def _init_test_jwt_keys():
    """
    Generate an in-memory RSA key pair so create_jwt_token / verify_token work
    in tests without key files on disk.  Mirrors what the init container does at
    runtime so tests never need to call init_jwt_keys() directly.
    """
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import agience_core.key_manager as _km

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    _km._private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    _km._public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    _km._key_id = "test"


@pytest.fixture(scope="session")
def _test_trust_keys():
    """Generate origin/mantle/chorus keypairs once per test session (expensive)."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    return {
        name: rsa.generate_private_key(public_exponent=65537, key_size=2048)
        for name in ("origin", "mantle", "chorus")
    }


@pytest.fixture(autouse=True)
def _install_test_service_identity_and_authority(_test_trust_keys):
    """Phase C: install in-memory service identity (`mantle`) + authority manifest
    before every test. Function-scoped because individual test files
    (test_service_identity, test_authority_trust) reset module state to
    exercise their own setup paths — this fixture restores the canonical
    test identity afterwards.
    """
    import base64
    from agience_core import service_identity, authority_trust

    def _b64url(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    def _public_jwk(public_key, kid: str) -> dict:
        nums = public_key.public_numbers()
        n = nums.n.to_bytes((nums.n.bit_length() + 7) // 8, "big")
        e = nums.e.to_bytes((nums.e.bit_length() + 7) // 8, "big")
        return {"kty": "RSA", "alg": "RS256", "use": "sig", "kid": kid,
                "n": _b64url(n), "e": _b64url(e)}

    keys = _test_trust_keys

    service_identity.reset_service_identity_for_tests()
    service_identity._loaded = service_identity.ServiceIdentity(
        name="mantle", kid="mantle-1", private_key=keys["mantle"]
    )

    authority_trust.reset_authority_manifest_for_tests()
    raw = {
        "artifact_id": "test-authority",
        "content_type": "application/vnd.agience.authority+json",
        "schema_version": 1,
        "issuer": "https://platform.test",
        "trust_anchors": {
            name: {"uri": f"http://{name}:8080",
                   "jwks": {"keys": [_public_jwk(keys[name].public_key(), f"{name}-1")]}}
            for name in ("origin", "mantle", "chorus")
        },
        "bootstrap_token_hash": None,
    }
    authority_trust._manifest = authority_trust.AuthorityManifest(
        issuer=raw["issuer"],
        trust_anchors=raw["trust_anchors"],
        bootstrap_token_hash=raw["bootstrap_token_hash"],
        artifact_id=raw["artifact_id"],
        raw=raw,
    )

    # Mantle's token verification is decoupled from the trust library — its
    # OidcVerifier reads the authority manifest FILE itself. Write the same test
    # manifest to a temp KEYS_DIR so the verifier resolves the test anchors, and
    # reset the verifier singleton so it reloads them.
    import json as _json
    import os as _os
    import tempfile as _tempfile
    from pathlib import Path as _P
    keys_dir = _tempfile.mkdtemp(prefix="agience-test-keys-")
    _os.environ["KEYS_DIR"] = keys_dir
    (_P(keys_dir) / "authority.manifest.json").write_text(_json.dumps(raw))
    # Mantle's outbound signer (services.peer_signing) reads its OWN key + instance
    # namespace from the keyset files — write the test ones so signing works.
    from cryptography.hazmat.primitives import serialization as _ser
    (_P(keys_dir) / "mantle.private.pem").write_text(
        keys["mantle"].private_bytes(
            _ser.Encoding.PEM, _ser.PrivateFormat.TraditionalOpenSSL, _ser.NoEncryption()).decode())
    (_P(keys_dir) / "instance.uuid").write_text("11111111-1111-4111-8111-111111111111")
    try:
        from services.oidc import reset_oidc_verifier
        reset_oidc_verifier()
        import services.peer_signing as _ps
        _ps._priv_pem = None
    except Exception:
        pass
    yield

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

@pytest.fixture(autouse=True)
def override_dependencies():
    def _unified_auth():
        return AuthContext(
            principal_id="user-123",
            principal_type="user",
            user_id="user-123",
        )

    def _person():
        return Person(
            id="user-123",
            email="test@example.com",
            name="Test User",
            picture="https://example.com/avatar.png",
        )

    def _user_claims():
        return {"sub": "user-123", "client_id": "agience-frontend"}

    def _arango_db():
        yield MagicMock()

    app.dependency_overrides[get_auth] = _unified_auth
    app.dependency_overrides[get_person] = _person
    app.dependency_overrides[get_end_user_claims] = _user_claims
    app.dependency_overrides[get_arango_db] = _arango_db

    yield
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def _reset_runtime_types():
    """Self-registered (runtime) content types live in a process-global registry.
    Reset it around every test so one test's pushed/seeded types never leak into
    another (and so resolution sees only the core filesystem base by default)."""
    from services import types_service
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()
    types_service.set_lazy_type_loader(None)
    yield
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()
    types_service.set_lazy_type_loader(None)


@pytest.fixture
def mock_user():
    return Person(
        id="user-123",
        email="test@example.com",
        name="Test User",
        picture="https://example.com/avatar.png"
    )


# ---------------------------------------------------------------------------
# Standalone-Mantle collection guards
#
# Mantle was extracted from the monorepo as a pure database layer. Two classes of
# inherited tests don't apply to the standalone build; rather than edit/delete
# each file we quarantine them here so the suite stays collectable + green, with
# the reason documented in one place.
#
# (1) DEAD MACHINERY — tests importing modules that were removed when operation
#     dispatch moved OUT of Mantle into the agience-bridge gateway (Phase 2b).
#     They error on import and can never pass in Mantle by design; the gateway
#     owns + tests that behavior now (agience-bridge/tests).
#
# (2) SEED/TYPE TREE — tests that load `package/seeds` or `package/types`, which
#     belong to the application/Origin and are absent from standalone Mantle.
#     Skipped when the tree is absent; they run unchanged if it is mounted. This
#     converges with the queued "Mantle pure-DB / seeds→Origin" sprint.
# ---------------------------------------------------------------------------
from pathlib import Path as _Path  # noqa: E402

collect_ignore = [
    # Import removed dispatch machinery (moved to the gateway):
    "test_operator_and_secret_material.py",   # imports services.handler_registry (removed)
    "test_search_as_artifact_smoke.py",       # imports services.operation_dispatcher (removed)
    # Exercise routers Mantle no longer mounts (only secrets/downloads/artifacts/
    # gate/search/events remain). The behavior left Mantle entirely:
    "test_router_types.py",                   # types_router unmounted (types live in the gateway)
    "test_router_beacon.py",                  # beacon extracted to agience-beacon
]

_PKG_ROOT = _Path(__file__).resolve().parents[3] / "package"
_TREE_DEPENDENT = {
    "test_seed_exporter.py",
    "test_seed_platform_tree.py",
    "test_types_service.py",
    "test_user_provisioning.py",
    "test_seed_drift.py",                      # asserts seed files match manifest/persona lists
}


def pytest_collection_modifyitems(config, items):
    """Skip seed/type-tree tests when that tree isn't present (standalone Mantle)."""
    if (_PKG_ROOT / "seeds").is_dir() and (_PKG_ROOT / "types").is_dir():
        return
    skip = pytest.mark.skip(
        reason="requires package/seeds|types tree (moved to Origin in the repo split)"
    )
    for item in items:
        if _Path(str(item.fspath)).name in _TREE_DEPENDENT:
            item.add_marker(skip)


