"""
Shared test helpers for backend tests.
"""

# MANTLE key-request helpers
#
# Key issuance is coupled to the grant check (canonical plan §5.3), so every
# oracle call needs a KeyRequest. These build the two legitimate shapes; there is
# deliberately no "unchecked" helper, because no such request exists in the
# production code either.
#
#
# A test has no router, so something has to stand in. Rather than make every test
# wrap itself in `acting_as(...)`, building a request declares "this is who is
# asking" — which mirrors production, where `requester_id` IS the authenticated
# principal by construction rather than by assertion.
#
# This does not soften the property under test:
#   * `conftest._reset_acting_principal` clears the identity before every test, so
#     it can never leak between tests and the fail-closed default is preserved;
#   * the grant verifier still runs on both arms, so declaring an identity gets you
#     a check, not a key;
#   * `test_key_custody_bypasses.py` deliberately does not use these helpers for its
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
    """Authorizes a requester for the contexts of its own principal; denies others.

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
    is denied, which is what makes the negative tests in
    ``test_oracle_grant_coupling.py`` meaningful against the same fixture.

    What it still refuses — and what the negative tests rely on — is the case that
    matters: a requester reaching a principal that is neither itself nor the one
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
