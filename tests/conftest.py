# OpenBLAS sizes its thread pool once, at library load. `mantle/__init__.py` sets this
# environment variable, and every test module here that uses numpy imports mantle first,
# so the pin is already in effect by the time numpy loads. Setting it again at the top of
# this file keeps that guarantee independent of import order elsewhere in the suite.
import os as _os
import time as _time

_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
del _os

import itertools
import uuid

import pytest
from typing import AsyncGenerator
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport

import bcrypt as _bcrypt
import mantle.config as _cfg
from mantle.entities.person import Person
from mantle.services.bootstrap_types import (
    ALL_PLATFORM_COLLECTION_SLUGS,
    AUTHORITY_ARTIFACT_SLUG,
    HOST_ARTIFACT_SLUG,
    AGENCY_ARTIFACT_SLUG,
    AGENT_ARTIFACT_SLUG_PREFIX,
    PLATFORM_AGENT_SLUGS,
)
from mantle.services.platform_topology import register_id
from mantle.services.dependencies import (
    get_auth,
    get_person,
    get_end_user_claims,
    AuthContext,
)
from mantle.services.dependencies import get_store_db
from mantle.main import app
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fast crypto: reduce bcrypt cost and PBKDF2 iterations so tests don't spend
# seconds on real key-stretching.  Applied before any test module imports.
# ---------------------------------------------------------------------------
_orig_gensalt = _bcrypt.gensalt
_bcrypt.gensalt = lambda rounds=4, prefix=b"2b": _orig_gensalt(rounds=4, prefix=prefix)
_cfg.PASSWORD_PBKDF2_ITERS = 1000

#: Names the per-test keyset directories under `_test_keys_base`. A counter rather than a random
#: name so the entries stay short and strictly increasing.
_keyset_seq = itertools.count()

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


@pytest.fixture(autouse=True)
def _reset_acting_principal():
    """Clear the acting-principal contextvar around every test.

    Key issuance requires an authenticated acting principal (see
    ``services.acting_principal``). Tests declare one by building a KeyRequest via
    ``tests.helpers``, which sets the contextvar — and pytest runs tests in one
    thread, so without this reset that identity would persist into the NEXT test.

    That would be quietly corrosive: a test asserting a refusal could pass or fail
    depending on which test ran before it, and the fail-closed default — no
    principal means no key — would stop being what a fresh test actually starts
    from. Reset on the way in and out so neither ordering nor a raising test can
    leave an identity behind.
    """
    from mantle.services.acting_principal import _acting

    token = _acting.set(None)
    try:
        yield
    finally:
        _acting.reset(token)


@pytest.fixture(autouse=True, scope="session")
def _init_test_encryption_key():
    """Initialize the platform KEK with a test value, so the key oracle and the
    encrypted platform settings both have one in tests."""
    from cryptography.fernet import Fernet
    import prism.trust.key_manager as _km
    _km._encryption_key = Fernet.generate_key().decode()


@pytest.fixture(autouse=True, scope="session")
def _init_test_jwt_keys():
    """
    Generate an in-memory RSA key pair so create_jwt_token / verify_token work
    in tests without key files on disk.  Mirrors what the init container does at
    runtime so tests never need to call init_jwt_keys directly.
    """
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import prism.trust.key_manager as _km

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
    """Generate origin/mantle/peer keypairs once per test session (expensive)."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    return {
        name: rsa.generate_private_key(public_exponent=65537, key_size=2048)
        for name in ("origin", "mantle", "peer")
    }


@pytest.fixture(scope="session")
def _test_keys_base(tmp_path_factory):
    """A CLEAN parent directory for the per-test ``KEYS_DIR`` below.

    The keyset directory itself stays per-test — same isolation as before, nothing is shared —
    but it is created HERE rather than in the system temp directory, and this is a performance
    fix worth ~500x on Windows.

    ``tempfile.mkdtemp`` puts its directory in ``%TEMP%``, and the fixture below never removed
    it, so every suite run leaked one directory per test (~3000) into a directory shared with
    every other program on the machine. On NTFS the cost of creating an entry grows with the
    number of entries already present — 8.3 short-name generation has to scan for a free alias
    among same-prefixed siblings — so the leak made the NEXT run slower, and the run after that
    slower again. Measured on this machine once the leak had accumulated: ``mkdtemp`` in
    ``%TEMP%`` cost 0.345s and the first ``write_text`` into the result another 0.158s, against
    0.0007s for the same call in an empty parent.

    ``tmp_path_factory`` is pytest's own base, which pytest reaps (it keeps the last three
    sessions), and the fixture below removes each keyset on teardown, so this parent holds about
    one entry at a time and entry creation stays flat for the life of the suite.
    """
    return tmp_path_factory.mktemp("keysets")


@pytest.fixture(autouse=True)
def _install_test_service_identity_and_authority(_test_trust_keys, _test_keys_base):
    """Phase C: install in-memory service identity (`mantle`) + authority manifest
    before every test. Function-scoped because individual test files
    (test_service_identity, test_authority_trust) reset module state to
    exercise their own setup paths — this fixture restores the canonical
    test identity afterwards.
    """
    import base64
    from prism.trust import service_identity, authority_trust

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
            for name in ("origin", "mantle", "peer")
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
    import shutil as _shutil
    from pathlib import Path as _P
    # A fresh, empty directory per test — unchanged in kind from the `mkdtemp` this replaces —
    # but parented on `_test_keys_base` and removed again below, so it costs milliseconds and
    # leaves nothing behind. The name is a short counter rather than a random hex string because
    # both stay unique here and the short one gives NTFS nothing to disambiguate.
    keys_dir = _test_keys_base / str(next(_keyset_seq))
    keys_dir.mkdir()
    keys_dir = str(keys_dir)
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
        from mantle.services.oidc import reset_oidc_verifier
        reset_oidc_verifier()
        import mantle.services.peer_signing as _ps
        _ps._priv_pem = None
    except Exception:
        pass
    yield
    # Removed so the parent stays empty and entry creation stays O(1) — see `_test_keys_base`.
    # `ignore_errors` because a test may still hold an open handle to a file in here, and on
    # Windows that refuses the unlink; a directory left behind costs the next run nothing more
    # than the one entry it occupies.
    _shutil.rmtree(keys_dir, ignore_errors=True)

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

    def _store_db():
        yield MagicMock()

    app.dependency_overrides[get_auth] = _unified_auth
    app.dependency_overrides[get_person] = _person
    app.dependency_overrides[get_end_user_claims] = _user_claims
    app.dependency_overrides[get_store_db] = _store_db

    yield
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def _reset_runtime_types():
    """Self-registered (runtime) content types live in a process-global registry.
    Reset it around every test so one test's pushed/seeded types never leak into
    another (and so resolution sees only the core filesystem base by default)."""
    from mantle.services import types_service
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()
    types_service.set_lazy_type_loader(None)
    yield
    types_service.clear_runtime_types()
    types_service.invalidate_type_cache()
    types_service.set_lazy_type_loader(None)


@pytest.fixture(autouse=True)
def _reset_indexed_geometry():
    """A test starts from a store that has never been indexed under any AnchorSet.

    `anchors.store` records the fingerprint of the set a cell write happens under, and refuses a
    later set that names a different space. One deployment has one coordinate system, so that
    record is durable by design — but the suite swaps toy AnchorSets of different widths through
    one session lattice, which is a sequence no deployment performs. Clearing the record around
    each test keeps the gate answering about the test in front of it.
    """
    def _clear():
        try:
            from mantle.db import identity_backend as identity_store
            from mantle.search.anchors.store import _GEOMETRY_KEY
            from mantle.services.dependencies import get_store_db
            db_gen = get_store_db()
            db = next(db_gen)
            try:
                identity_store.delete_platform_setting(db, _GEOMETRY_KEY)
            finally:
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        except Exception:
            pass
    _clear()
    yield
    _clear()


# ---------------------------------------------------------------------------
# Standalone-Mantle collection guards
#
# Mantle is a pure database layer. Two classes of inherited tests don't apply to
# the standalone build; rather than edit or delete each file, they are quarantined
# here so the suite stays collectable and green, with the reason documented in one
# place.
#
# (1) dead machinery — tests importing modules that no longer exist in this package.
#     They error on import and cannot pass in Mantle; the gateway (agience-prism-py)
#     owns and tests that behavior now (agience-prism-py/tests).
#
# (2) seed/type tree — tests that load `package/seeds` or `package/types`, which
#     belong to the application/Origin and are absent from standalone Mantle.
#     Skipped when the tree is absent; they run unchanged if it is mounted.
# ---------------------------------------------------------------------------
from pathlib import Path as _Path  # noqa: E402

# `collect_ignore` hides a file from collection entirely: an ignored file is
# reported as neither run nor skipped, so an entry whose stated reason stops
# holding goes unnoticed until something checks it directly.
# `test_collect_ignore_entries_are_honest` enforces that every entry here is
# genuinely uncollectable rather than merely inconvenient.
collect_ignore: list = []

#: Entries expected to be uncollectable (import-time dead). Anything else in
#: `collect_ignore` must be justified as collect-but-fail, and nothing may be
#: ignored while passing — see the enforcing test.
_IGNORE_UNCOLLECTABLE: set = set()

# The seeds/types package tree is split across repos: seeds live at
# `agience-observe/package/seeds`; types live at `agience-crystal/src/types`, with
# a vendored copy at `agience-chorus/src/astra/web/vendor/package/types`.
# `_package_root` is the one place that resolves both locations, so the gate here
# and the tests below it agree on where the tree lives.
#
#     MANTLE_PACKAGE_ROOT=<workspace>/agience-observe/package python -m pytest -q tests
#
# points the gate at the observe package tree. Because that root holds seeds but not
# types, the seed-dependent tests run and the type-dependent ones still skip —
# reviving the type-dependent tests needs `_package_root` to resolve a second,
# separate root for `types/`.
from ._package_root import have_tree as _have_tree  # noqa: E402
from ._package_root import package_root as _package_root  # noqa: E402

_PKG_ROOT = _package_root()
_TREE_DEPENDENT = {
    "test_seed_exporter.py",
    "test_seed_platform_tree.py",
    "test_types_service.py",
    "test_user_provisioning.py",
    "test_seed_drift.py",                      # asserts seed files match manifest/persona lists
}

#: A roster of bare filenames matched against `item.fspath.name`, so a renamed file drops out of the
#: gate without anything failing — the marker simply stops being applied, and on a machine with the
#: tree mounted nothing looks different at all. Checked here, at collection, where a stale name is
#: an immediate loud error rather than a skip that quietly stops happening.
_TREE_DEPENDENT_MISSING = sorted(
    name for name in _TREE_DEPENDENT if not (_Path(__file__).parent / name).is_file()
)
assert not _TREE_DEPENDENT_MISSING, (
    "_TREE_DEPENDENT names test files that are not in tests/: %r — the skip gate silently stops "
    "covering a file it names by hand, so retarget the entry rather than deleting it."
    % _TREE_DEPENDENT_MISSING
)


# (3) The `/artifacts/{id}/op/{op}` dispatch route lives in the gateway
#     (agience-prism-py), not in Mantle. `db/test_op_dispatch_route_is_gone.py`
#     asserts the route stays absent from Mantle's own routes.


def pytest_collection_modifyitems(config, items):
    """Quarantine inherited tests that don't apply to standalone Mantle."""
    # (2) Seed/type-tree tests — only when that tree isn't present.
    # `_package_root.have_tree` is the one place that answers this, so the gate
    # and the tree-dependent tests always agree on whether the tree is mounted.
    if _have_tree():
        return
    skip = pytest.mark.skip(
        reason="requires a package root holding BOTH seeds/ and types/ — the tree is SPLIT: "
               "seeds at agience-observe/package/seeds, types at agience-crystal/src/types. "
               "NOT in agience-origin. See the note above."
    )
    for item in items:
        if _Path(str(item.fspath)).name in _TREE_DEPENDENT:
            item.add_marker(skip)


# ── the source-race re-verify ────────────────────────────────────────────────────────────────────
# 42 test files in this suite `ast.parse` source off disk, 4 of them across into sibling repos,
# and seven interactive sessions edit this workspace while a 6-9 minute run is in flight. Four full
# runs gave failure sets of 2 / 4 / 0 / 9, largely disjoint, all passing in isolation; two were
# traced to the minute to another session rewriting the exact file being parsed.
#
# Re-verify, not retry: a failed test is rerun once, and only
# when a source file was written while that test ran. When the tree is quiet nothing is rerun, so a
# genuinely flaky test still fails and stays visible — which is the whole difference between this
# and a blanket retry, and the reason a retry in a gate is a thing to add deliberately.
#
# Both outcomes are reported at the end of the run: a rescue is a line with a number attached, so
# the question *"is this race worth more effort"* becomes measurable instead of anecdotal.
from . import source_race as _race  # noqa: E402


def pytest_sessionstart(session):
    _race.mark_suite_start()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Owns the protocol, it does not wrap it — and the difference is the whole mechanism.

    The wrapper version ran the protocol, saw the failure, reran, and logged the passing reports.
    The run still ended `1 failed, 1 passed` and exited 1: by the time a wrapper resumes, the
    first failure has already been reported, and nothing can un-report it. The rescue printed and
    changed nothing — a mechanism that announces a result it did not produce.

    Running both attempts with `log=False` and reporting only the outcome that stands is what makes
    the rescue real. `logstart`/`logfinish` are emitted by hand because taking over the protocol
    means taking over the whole of it.
    """
    from _pytest.runner import runtestprotocol

    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    started = _time.time()
    reports = runtestprotocol(item, nextitem=nextitem, log=False)

    if any(r.failed for r in reports):
        moved = _race.the_tree_moved_during(started)
        if moved:
            # The tree moved while this test ran, so its failure may be a parse of a file that was
            # being rewritten. Ask once more, against what is on disk now.
            second = runtestprotocol(item, nextitem=nextitem, log=False)
            if any(r.failed for r in second):
                _race.CONFIRMED.append((item.nodeid, len(moved)))
            else:
                _race.RESCUES.append((item.nodeid, len(moved)))
            reports = second

    for r in reports:
        item.ihook.pytest_runtest_logreport(report=r)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """The number that decides whether this is worth keeping. Silence here would make a rescue
    indistinguishable from a test that simply passed, which is the failure mode of every retry."""
    if _race.RESCUES:
        terminalreporter.write_sep("-", "source-race re-verify")
        for nodeid, n in _race.RESCUES:
            terminalreporter.write_line(
                "RESCUED  %s — failed, %d source file(s) changed while it ran, passed on re-verify"
                % (nodeid, n))
    if _race.CONFIRMED:
        terminalreporter.write_sep("-", "source-race re-verify (failed twice)")
        for nodeid, n in _race.CONFIRMED:
            terminalreporter.write_line(
                "CONFIRMED  %s — %d source file(s) changed while it ran, and it failed again"
                % (nodeid, n))




@pytest.fixture(autouse=True)
def _content_tier_for_unit_tests(request, monkeypatch):
    """A working content store for the artifact write path, so a write with content can store it.

    Artifact bytes live in the CAS: `workspace_service._store_content_in_s3` puts them there and
    returns the `cas/<sha256>` address, and `db/doc_boundary` keeps the body out of the document
    and hydrates it back on read. A create or update carrying content therefore needs a content
    store and an acting principal to seal the envelope for — which unit tests built on MagicMock
    do not have. Before this, the failure was swallowed and the bytes stayed inline, so those
    tests passed over a write that silently did nothing.

    Patched at the artifact write path rather than at `content_service`. Patching the service
    globally reaches every test that exercises the real content routes — the upload path, the
    deferred mirror record, the tier itself — and replaces the thing under test. This substitutes
    only the step the artifact write makes, and addresses the bytes exactly as the real one does
    so an assertion on the ref sees a genuine content address.

    Tests that exercise `_store_content_in_s3` itself opt out with
    ``@pytest.mark.real_content_store``.
    """
    if request.node.get_closest_marker("real_content_store"):
        return
    import hashlib

    from mantle.services import workspace_service

    def _addressed(artifact_id, content, context_str, owner_id=None, collection_id=None):
        key = workspace_service._safe_content_key(
            _loads_or_empty(context_str), artifact_id)
        ref = ("cas/" + hashlib.sha256(content.encode("utf-8")).hexdigest()) if owner_id else None
        return key, content, ref

    def _loads_or_empty(raw):
        import json as _json
        try:
            parsed = _json.loads(raw) if raw else {}
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError):
            return {}

    monkeypatch.setattr(workspace_service, "_store_content_in_s3", _addressed)
