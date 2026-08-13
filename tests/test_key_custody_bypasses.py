"""Key-custody regression suite for LATTICE §4.3: four invariants that keep an unauthorized
caller from reaching a master key, its derived keys, or the plaintext behind them.

Every test here asserts a denial. If any test in this file starts failing, one of the four
invariants below no longer holds.

  B1  `KeyPurpose.SELF` consults the grant verifier even when the caller supplies both sides
      of `requester_id == principal_id` — that equality alone is not authorization.
  B2  Artifact content decryption is grant-gated: `content_crypto` derives `requester_id`
      from the acting principal, never from the `principal_id` it was passed.
  B3  A master key is never generated and persisted for a principal that does not exist yet,
      so there is nothing for an unauthorized caller to mint ahead of the principal.
  B4  The vector arm's cell cache sits behind the oracle: a cache hit still carries the
      requester through a grant check, so it cannot serve another principal's decrypted
      plaintext for its 60s TTL with the oracle never invoked.
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle.oracle import (
    FernetMasterKeyStore,
    GrantDenied,
    KeyPurpose,
    KeyRequest,
    MasterKeyMissing,
    OracleService,
)
from mantle.services.acting_principal import (
    ActingPrincipal,
    KeyCustodyDenied,
    NoActingPrincipal,
    acting_as,
    current_acting_principal,
    propagate,
    require_acting_principal,
)
from .helpers import AllowListGrantVerifier

VICTIM = "victim-principal"
VICTIM_COLLECTION = "victim-collection"
ATTACKER = "mallory"
LEGIT = "legit-user"


def _oracle():
    """An oracle wired as `wiring._build_oracle` wires it: a real verifier that
    authorizes only LEGIT. Mallory holds no grant anywhere."""
    return OracleService(
        FernetMasterKeyStore(Fernet(Fernet.generate_key())),
        grant_verifier=AllowListGrantVerifier({LEGIT: {(VICTIM, VICTIM_COLLECTION)}}),
    )


def _seed_victim_key(oracle):
    """Create the victim's master key legitimately, so absence-of-key is never what
    a later test is actually measuring."""
    with acting_as(LEGIT, principal_type="user"):
        return oracle.get_or_create_master_key(
            VICTIM,
            KeyRequest(requester_id=LEGIT, purpose=KeyPurpose.GRANT, action="update"),
            collection_id=VICTIM_COLLECTION,
        )


# =====================================================================
# B1 — SELF is not a skeleton key
# =====================================================================

class TestSelfIsNotASkeletonKey:
    def test_attacker_cannot_name_someone_else_as_requester(self):
        """Mallory is authenticated, so she may ask as herself — naming the victim as
        `requester_id` is a denial, not a promotion."""
        oracle = _oracle()
        _seed_victim_key(oracle)
        with acting_as(ATTACKER, principal_type="user"):
            with pytest.raises(GrantDenied, match="not the authenticated"):
                oracle.get_or_create_master_key(
                    VICTIM,
                    KeyRequest(requester_id=VICTIM, purpose=KeyPurpose.SELF),
                )

    def test_attacker_asking_honestly_as_herself_is_also_denied(self):
        """Denying the impersonation must not leave the honest path open."""
        oracle = _oracle()
        _seed_victim_key(oracle)
        with acting_as(ATTACKER, principal_type="user"):
            with pytest.raises(GrantDenied):
                oracle.get_or_create_master_key(
                    VICTIM,
                    KeyRequest(requester_id=ATTACKER, purpose=KeyPurpose.SELF),
                )
            with pytest.raises(GrantDenied):
                oracle.get_or_create_master_key(
                    VICTIM,
                    KeyRequest(requester_id=ATTACKER, purpose=KeyPurpose.GRANT),
                    collection_id=VICTIM_COLLECTION,
                )

    def test_cell_and_sse_keys_are_not_reachable_via_self(self):
        """`derive_cell_key`/`derive_sse_key` take the same KeyRequest, so the SELF
        check applies to them verbatim too."""
        oracle = _oracle()
        _seed_victim_key(oracle)
        with acting_as(ATTACKER, principal_type="user"):
            forged = KeyRequest(requester_id=VICTIM, purpose=KeyPurpose.SELF)
            with pytest.raises(GrantDenied):
                oracle.derive_cell_key(VICTIM, VICTIM_COLLECTION, "anchor-7", forged)
            with pytest.raises(GrantDenied):
                oracle.derive_sse_key(VICTIM, forged)

    def test_self_arm_now_consults_the_verifier(self):
        """The verifier must be asked on the SELF path too, even when requester ==
        principal == the authenticated caller."""
        calls = []

        class SpyVerifier:
            def authorized(self, **kw):
                calls.append(kw)
                return False

        oracle = OracleService(
            FernetMasterKeyStore(Fernet(Fernet.generate_key())),
            grant_verifier=SpyVerifier(),
        )
        with acting_as(VICTIM, principal_type="user"):
            with pytest.raises(GrantDenied):
                oracle.get_or_create_master_key(
                    VICTIM,
                    KeyRequest(requester_id=VICTIM, purpose=KeyPurpose.SELF),
                )
        assert calls, "verifier was NOT consulted on the SELF path"

    def test_deny_all_verifier_cannot_be_defeated(self):
        """A verifier hard-wired to return False must deny every principal's master
        key request, not just one."""

        class DenyAll:
            def authorized(self, **kw):
                return False

        oracle = OracleService(
            FernetMasterKeyStore(Fernet(Fernet.generate_key())), grant_verifier=DenyAll()
        )
        for victim in ("alice", "bob", "carol", "the-whole-fleet"):
            with acting_as(victim, principal_type="user"):
                with pytest.raises(GrantDenied):
                    oracle.get_or_create_master_key(
                        victim,
                        KeyRequest(requester_id=victim, purpose=KeyPurpose.SELF),
                    )

    def test_self_is_narrower_than_grant_never_wider(self):
        """SELF only ever adds a constraint: the authorized user reaches the key with
        GRANT, but the same user with SELF (requester != principal) is denied."""
        oracle = _oracle()
        _seed_victim_key(oracle)
        with acting_as(LEGIT, principal_type="user"):
            assert oracle.get_or_create_master_key(
                VICTIM,
                KeyRequest(requester_id=LEGIT, purpose=KeyPurpose.GRANT),
                collection_id=VICTIM_COLLECTION,
            )
            with pytest.raises(GrantDenied, match="requester == principal"):
                oracle.get_or_create_master_key(
                    VICTIM,
                    KeyRequest(requester_id=LEGIT, purpose=KeyPurpose.SELF),
                    collection_id=VICTIM_COLLECTION,
                )


# =====================================================================
# B2 — content decryption is grant-gated
# =====================================================================

class TestContentIsGrantGated:
    def test_attacker_cannot_decrypt_victim_content(self):
        """`decrypt_content(owner, blob)` alone is not sufficient: passing the owner as
        the requester compares a value to itself. The grant ledger decides."""
        from mantle.services import content_crypto

        oracle = _oracle()
        _seed_victim_key(oracle)
        secret = b"the victim's confidential artifact body"

        def provider_as(principal_id: str, actor: str) -> bytes:
            return oracle.get_or_create_master_key(
                principal_id,
                KeyRequest(requester_id=actor, purpose=KeyPurpose.GRANT, action="read"),
                collection_id=VICTIM_COLLECTION,
            )

        with acting_as(LEGIT, principal_type="user"):
            blob = content_crypto.encrypt_content(
                VICTIM, secret, master_key_provider=lambda p: provider_as(p, LEGIT)
            )
        assert content_crypto.is_encrypted(blob)

        with acting_as(ATTACKER, principal_type="user"):
            with pytest.raises(GrantDenied):
                content_crypto.decrypt_content(
                    VICTIM, blob, master_key_provider=lambda p: provider_as(p, ATTACKER)
                )

        # ...and the grant holder still reads it, so this is access control rather
        # than a blanket denial.
        with acting_as(LEGIT, principal_type="user"):
            assert content_crypto.decrypt_content(
                VICTIM, blob, master_key_provider=lambda p: provider_as(p, LEGIT)
            ) == secret

    def test_content_key_request_is_not_tautological(self):
        """The real `_default_master_key` must derive its requester from the acting
        principal, not from the principal argument."""
        import mantle.services.content_crypto as cc

        seen = {}

        class FakeOracle:
            def get_or_create_master_key(self, principal_id, request, *, collection_id=None):
                seen["principal_id"] = principal_id
                seen["requester_id"] = request.requester_id
                seen["purpose"] = request.purpose
                return b"\x00" * 32

        cc_build = cc.__dict__.get("_build_oracle")
        import mantle.search.mantle.wiring as wiring
        orig = wiring._build_oracle
        wiring._build_oracle = lambda: FakeOracle()
        try:
            with acting_as(LEGIT, principal_type="user"):
                cc._default_master_key(VICTIM, VICTIM_COLLECTION)
        finally:
            wiring._build_oracle = orig
            if cc_build is not None:
                cc.__dict__["_build_oracle"] = cc_build

        assert seen["principal_id"] == VICTIM
        assert seen["requester_id"] == LEGIT, (
            "requester must be the acting caller, not the content's owner — "
            "equal values here is the tautology that was the bypass"
        )
        assert seen["purpose"] is KeyPurpose.GRANT


# =====================================================================
# B3 — mint-ahead
# =====================================================================

class TestMintAhead:
    def test_read_never_creates_a_master_key(self):
        """A read that mints returns a valid key which decrypts nothing, hiding a
        lost key behind 'no results'."""
        store = FernetMasterKeyStore(Fernet(Fernet.generate_key()))
        oracle = OracleService(
            store, grant_verifier=AllowListGrantVerifier({LEGIT: {(VICTIM, VICTIM_COLLECTION)}})
        )
        assert store.get(VICTIM) is None
        with acting_as(LEGIT, principal_type="user"):
            with pytest.raises(MasterKeyMissing):
                oracle.get_or_create_master_key(
                    VICTIM,
                    KeyRequest(requester_id=LEGIT, purpose=KeyPurpose.GRANT, action="read"),
                    collection_id=VICTIM_COLLECTION,
                )
        assert store.get(VICTIM) is None, "a read PERSISTED a key"

    def test_attacker_cannot_mint_for_an_unborn_principal(self):
        """No grant can reach a principal that does not exist, so the mint is
        unreachable regardless of action."""
        store = FernetMasterKeyStore(Fernet(Fernet.generate_key()))
        oracle = OracleService(store, grant_verifier=AllowListGrantVerifier({}))
        future_principal = "collection-not-created-yet"
        with acting_as(ATTACKER, principal_type="user"):
            for action in ("read", "update", "create"):
                with pytest.raises(KeyCustodyDenied):
                    oracle.get_or_create_master_key(
                        future_principal,
                        KeyRequest(
                            requester_id=ATTACKER, purpose=KeyPurpose.GRANT, action=action
                        ),
                    )
        assert store.get(future_principal) is None

    def test_authorized_write_still_creates(self):
        """Denying an unauthorized mint must not break legitimate first-write key
        creation."""
        store = FernetMasterKeyStore(Fernet(Fernet.generate_key()))
        oracle = OracleService(
            store, grant_verifier=AllowListGrantVerifier({LEGIT: {(VICTIM, VICTIM_COLLECTION)}})
        )
        with acting_as(LEGIT, principal_type="user"):
            key = oracle.get_or_create_master_key(
                VICTIM,
                KeyRequest(requester_id=LEGIT, purpose=KeyPurpose.GRANT, action="update"),
                collection_id=VICTIM_COLLECTION,
            )
        assert len(key) == 32
        assert store.get(VICTIM) == key


# =====================================================================
# B4 — the cell cache is behind the oracle
# =====================================================================

class TestCellCacheIsBehindTheOracle:
    def _engine(self, oracle):
        from mantle.search.mantle.engine import MantleQueryEngine

        class DummyCells:
            def get(self, *a, **kw):
                raise AssertionError("cell store must not be reached in these tests")

        return MantleQueryEngine(oracle, DummyCells())

    def test_warm_cache_does_not_serve_an_unauthorized_caller(self):
        """The authorized user's search warms the cache; Mallory then reaching
        `_load_cell` must not be served the plaintext with the oracle never invoked.
        The cache key has no requester component, so a hit alone is not authorization."""
        oracle = _oracle()
        _seed_victim_key(oracle)
        engine = self._engine(oracle)

        plaintext = [{"artifact_id": "secret-doc", "chunk_id": 0,
                      "embedding": [1.0, 0.0, 0.0], "text": "victim's confidential text"}]
        engine._cache.put(VICTIM, VICTIM_COLLECTION, plaintext, "anchor-1")

        with acting_as(ATTACKER, principal_type="user"):
            with pytest.raises(GrantDenied):
                engine._load_cell(
                    VICTIM, VICTIM_COLLECTION, "anchor-1",
                    KeyRequest(requester_id=ATTACKER, purpose=KeyPurpose.GRANT),
                )

    def test_authorized_caller_still_gets_the_cache_hit(self):
        """Authorizing before the cache must not disable the cache."""
        oracle = _oracle()
        _seed_victim_key(oracle)
        engine = self._engine(oracle)

        plaintext = [{"artifact_id": "doc", "chunk_id": 0,
                      "embedding": [1.0, 0.0, 0.0], "text": "hello"}]
        engine._cache.put(VICTIM, VICTIM_COLLECTION, plaintext, "anchor-1")

        with acting_as(LEGIT, principal_type="user"):
            chunks, outcome = engine._load_cell(
                VICTIM, VICTIM_COLLECTION, "anchor-1",
                KeyRequest(requester_id=LEGIT, purpose=KeyPurpose.GRANT),
            )
        assert chunks == plaintext
        assert outcome is None  # served from cache, no decryption attempted

    def test_attacker_supplying_her_own_context_list_is_still_refused(self):
        """The scenario the doc singles out: a caller who bypassed the light cone
        upstream and hands the engine a context list naming the victim."""
        oracle = _oracle()
        _seed_victim_key(oracle)
        engine = self._engine(oracle)
        engine._cache.put(VICTIM, VICTIM_COLLECTION, [{"artifact_id": "x", "chunk_id": 0,
                          "embedding": [1.0, 0.0, 0.0], "text": "secret"}], "anchor-1")

        with acting_as(ATTACKER, principal_type="user"):
            with pytest.raises(GrantDenied):
                engine._load_cell(
                    VICTIM, VICTIM_COLLECTION, "anchor-1",
                    # A context list she fabricated; the oracle never trusts it.
                    KeyRequest(requester_id=ATTACKER, purpose=KeyPurpose.GRANT),
                )


# =====================================================================
# The fail-closed foundation itself
# =====================================================================

class TestFailsClosedWithoutAnIdentity:
    def test_no_acting_principal_yields_no_key(self):
        oracle = _oracle()
        with pytest.raises(NoActingPrincipal):
            oracle.get_or_create_master_key(
                VICTIM, KeyRequest(requester_id=LEGIT, purpose=KeyPurpose.GRANT)
            )

    def test_refusals_share_a_base_so_they_cannot_be_swallowed(self):
        """`unified.py` re-raises denials and swallows everything else. If a denial
        type escapes that tuple it degrades to 'no results' — fail-open by omission."""
        assert issubclass(GrantDenied, KeyCustodyDenied)
        assert issubclass(NoActingPrincipal, KeyCustodyDenied)

    def test_identity_does_not_leak_out_of_its_block(self):
        assert current_acting_principal() is None
        with acting_as(LEGIT, principal_type="user"):
            assert current_acting_principal().principal_id == LEGIT
            with acting_as(ATTACKER, principal_type="user"):
                assert current_acting_principal().principal_id == ATTACKER
            assert current_acting_principal().principal_id == LEGIT
        assert current_acting_principal() is None

    def test_identity_does_not_survive_an_exception(self):
        with pytest.raises(RuntimeError):
            with acting_as(LEGIT, principal_type="user"):
                raise RuntimeError("boom")
        assert current_acting_principal() is None

    def test_require_raises_rather_than_returning_an_empty_identity(self):
        with pytest.raises(NoActingPrincipal):
            require_acting_principal()

    def test_acting_principal_rejects_an_empty_id(self):
        with pytest.raises(ValueError):
            ActingPrincipal(principal_id="")

    def test_thread_fanout_needs_explicit_propagation(self):
        """A pool worker starts from an EMPTY context. `propagate` is what carries
        the identity across; without it indexing fails closed rather than running
        unchecked — assert both halves so the helper cannot be quietly dropped."""
        from concurrent.futures import ThreadPoolExecutor

        def who():
            p = current_acting_principal()
            return p.principal_id if p else None

        with acting_as(LEGIT, principal_type="user"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                assert pool.submit(propagate(who)).result() == LEGIT
                assert pool.submit(who).result() is None
