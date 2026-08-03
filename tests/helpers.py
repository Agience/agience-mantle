"""
Shared test helpers and mock factories for backend tests.
"""
import json
from mantle.entities.artifact import Artifact
from mantle.entities.person import Person


# Mock factories

def mock_artifact(
    id="c1",
    collection_id="w1",
    state="draft",
    context_dict=None,
    content="test content",
    root_id=None,
    created_by="user-123",
):
    """Create a mock Artifact entity."""
    context = json.dumps(context_dict or {})
    return Artifact(
        id=id,
        collection_id=collection_id,
        state=state,
        context=context,
        content=content,
        root_id=root_id,
        created_by=created_by,
    )


def mock_container(
    id="col1",
    name="Test Collection",
    description="Test collection",
    created_by="user-123",
    content_type="application/vnd.agience.collection+json",
):
    """Create a mock container (workspace or collection) artifact."""
    return Artifact(
        id=id,
        name=name,
        description=description,
        created_by=created_by,
        content_type=content_type,
        state=Artifact.STATE_COMMITTED,
    )


def mock_person(
    id="user-123",
    email="test@example.com",
    name="Test User",
    picture="https://example.com/avatar.png"
):
    """Create a mock Person entity."""
    return Person(
        id=id,
        email=email,
        name=name,
        picture=picture
    )


# Assertion helpers

def assert_artifact_matches(artifact, expected_dict):
    """Assert artifact matches expected values."""
    assert artifact.id == expected_dict.get("id")
    if "collection_id" in expected_dict:
        assert artifact.collection_id == expected_dict["collection_id"]
    if "state" in expected_dict:
        assert artifact.state == expected_dict["state"]
    if "content" in expected_dict:
        assert artifact.content == expected_dict["content"]
    if "name" in expected_dict:
        assert artifact.name == expected_dict["name"]
    if "description" in expected_dict:
        assert artifact.description == expected_dict["description"]
    if "created_by" in expected_dict:
        assert artifact.created_by == expected_dict["created_by"]


# HTTP response helpers

def assert_http_status(response, expected_status):
    """Assert HTTP response status code."""
    assert response.status_code == expected_status, \
        f"Expected {expected_status}, got {response.status_code}: {response.text}"


def assert_response_has_fields(response_dict, required_fields):
    """Assert response dict contains all required fields."""
    for field in required_fields:
        assert field in response_dict, f"Response missing field: {field}"


# Mock DB helpers

def mock_db_session():
    """Create a mock DB session for testing."""
    from unittest.mock import Mock
    session = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    return session


# MANTLE key-request helpers
#
# Key issuance is coupled to the grant check (canonical plan §5.3), so every
# oracle call needs a KeyRequest. These build the two legitimate shapes; there is
# deliberately NO "unchecked" helper, because no such request exists in the
# production code either.
#
# ⚠ THESE ALSO DECLARE THE TEST'S IDENTITY, AND THAT IS A TEST-ONLY AFFORDANCE.
# The oracle now requires `requester_id` to equal the AUTHENTICATED acting
# principal — that binding is the fix for the SELF skeleton key, where both sides
# of the comparison were caller-supplied. In production the identity comes from
# `get_auth` at the request boundary and a caller cannot choose it.
#
# A test has no router, so something has to stand in. Rather than make every test
# wrap itself in `acting_as(...)`, building a request declares "this is who is
# asking" — which mirrors production, where `requester_id` IS the authenticated
# principal by construction rather than by assertion.
#
# This does NOT soften the property under test:
#   * `conftest._reset_acting_principal` clears the identity before every test, so
#     it can never leak between tests and the fail-closed default is preserved;
#   * the grant verifier still runs on both arms, so declaring an identity gets you
#     a CHECK, not a key;
#   * `test_key_custody_bypasses.py` deliberately does NOT use these helpers for its
#     attack cases — it sets the attacker's identity explicitly and asserts refusal.

def _declare_test_identity(principal_id: str, principal_type: str) -> None:
    """Stand in for the request boundary: this test acts as ``principal_id``."""
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal
    set_acting_principal(
        ActingPrincipal(
            principal_id=principal_id, principal_type=principal_type, source="test",
        )
    )


def self_request(principal_id: str, action: str = "read"):
    """A SELF key request: the principal asking for its own key.

    ⚠ ``SELF`` no longer skips the grant verifier — it NARROWS the check by also
    requiring ``requester == principal``. An oracle used with this must therefore be
    wired with a verifier that authorizes the principal for its own contexts; see
    :class:`SelfContextVerifier`.
    """
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest
    _declare_test_identity(principal_id, "principal")
    return KeyRequest(
        requester_id=principal_id, purpose=KeyPurpose.SELF,
        requester_type="principal", action=action,
    )


def grant_request(requester_id: str, action: str = "read"):
    """A GRANT key request: a third party who must hold a grant to the context."""
    from mantle.search.mantle.oracle import KeyPurpose, KeyRequest
    _declare_test_identity(requester_id, "user")
    return KeyRequest(
        requester_id=requester_id, purpose=KeyPurpose.GRANT,
        requester_type="user", action=action,
    )


class SelfContextVerifier:
    """Authorizes a requester for the contexts of its OWN principal; denies others.

    Models the production arrangement in which a principal holds an owner/admin
    grant on its own collections (LATTICE §3: two collections, each with an
    owner/admin grant). Tests that exercise index/query mechanics rather than
    authorization use this so the mechanics are reachable, while a cross-principal
    request is still refused — a test double that authorized everything would make
    every fixture built on it blind to a reopened bypass.
    """

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action) -> bool:
        return requester_id == principal_id


class AllowListGrantVerifier:
    """Test GrantVerifier over an explicit allow-list of authorized contexts.

    ``contexts`` maps requester_id -> set of (principal_id, collection_id). Any
    pair not listed is denied, so a test that forgets to grant something gets a
    denial rather than accidental access.
    """

    def __init__(self, contexts: dict) -> None:
        self._contexts = {k: set(v) for k, v in contexts.items()}

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action) -> bool:
        pairs = self._contexts.get(requester_id, set())
        if collection_id:
            return (principal_id, collection_id) in pairs
        return any(p == principal_id for p, _ in pairs)


TEST_REQUESTER = "test-requester"


class SingleRequesterVerifier:
    """Authorizes ONE named requester for every context, plus each principal for its
    own contexts; denies everyone else.

    A test double for :class:`~search.mantle.oracle.GrantVerifier`. Deliberately
    named for what it does — it is not an "allow all" shim: an unnamed requester
    is DENIED, which is what makes the negative tests in
    ``test_oracle_grant_coupling.py`` meaningful against the same fixture.

    ⚠ The self-context arm exists because ``SELF`` no longer skips the verifier.
    A typical test INDEXES as the owner (``self_request(owner)``) and then QUERIES
    as a third party (``grant_request(TEST_REQUESTER)``) against the same oracle, so
    the double has to authorize both halves or the setup itself is refused. It
    models the production arrangement where a principal holds an owner/admin grant
    on its own collections (LATTICE §3).

    What it still refuses — and what the negative tests rely on — is the case that
    matters: a requester reaching a principal that is NEITHER itself NOR the one
    named requester.
    """

    def __init__(self, requester_id: str = TEST_REQUESTER) -> None:
        self._who = requester_id

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action) -> bool:
        return requester_id == self._who or requester_id == principal_id


def make_oracle(store=None, requester_id: str = TEST_REQUESTER):
    """An OracleService wired with a grant verifier, as production oracles are."""
    from cryptography.fernet import Fernet
    from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService
    if store is None:
        store = FernetMasterKeyStore(Fernet(Fernet.generate_key()))
    return OracleService(store, grant_verifier=SingleRequesterVerifier(requester_id))


def req(action: str = "read"):
    """The GRANT request matching :func:`test_oracle`'s authorized requester."""
    return grant_request(TEST_REQUESTER, action=action)
