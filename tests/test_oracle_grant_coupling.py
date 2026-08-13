"""Key issuance is coupled to the grant check: "if you don't have a grant, you simply cannot."

Canonical plan section 5.3 (single-node state): *couple key/anchor-position
issuance to the grant check immediately so the trust boundary matches the target.*

These tests assert a cryptographic-custody property, not a code path. The question
they answer is not "does the permission check run" but "can an unauthorized caller
obtain key material by any route the API allows". Every test here asserts a
failure for the unauthorized caller, and a matching success for the authorized one,
because a test that only proves the happy path would pass against an oracle that
issued keys to anyone.

Identity for a key request comes from the requester, never from the object being
read: ``get_or_create_master_key`` derives access from ``requester_id``, so there
is always a requester to check against a grant.
"""

from __future__ import annotations

import numpy as np
import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle.engine import MantleQueryEngine
from mantle.search.mantle.indexer import MantleIndexer
from mantle.search.mantle.oracle import (
    FernetMasterKeyStore,
    GrantDenied,
    KeyPurpose,
    KeyRequest,
    LightConeGrantVerifier,
    OracleService,
)
from mantle.search.mantle.stores import InMemoryCellStore

PRINCIPAL = "origin-root-1"
COLLECTION = "col-1"
ALICE = "user-alice"      # holds a grant
MALLORY = "user-mallory"  # holds none


class _Verifier:
    """Authorizes ALICE for (PRINCIPAL, COLLECTION), and any principal for its own
    contexts. Everyone else: denied.
    """

    def __init__(self):
        self.calls = []

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action):
        self.calls.append((requester_id, principal_id, collection_id, action))
        if requester_id == principal_id:
            return True                      # a principal's grant on its own context
        if requester_id != ALICE:
            return False
        if collection_id is not None and collection_id != COLLECTION:
            return False
        return principal_id == PRINCIPAL


@pytest.fixture
def verifier():
    return _Verifier()


@pytest.fixture
def oracle(verifier):
    store = FernetMasterKeyStore(Fernet(Fernet.generate_key()))
    return OracleService(store, grant_verifier=verifier)


def _declare(who: str, principal_type: str) -> None:
    """Stand in for the request boundary: this test acts as ``who``.

    The oracle requires ``requester_id`` to equal the authenticated acting principal
    — that binding is what keeps SELF issuance from being a skeleton key.

    This does not weaken anything these tests assert: the grant verifier still runs
    on every arm, so declaring an identity buys a check, not a key — MALLORY below
    declares her identity honestly and is still refused.
    """
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal
    set_acting_principal(
        ActingPrincipal(principal_id=who, principal_type=principal_type, source="test")
    )


def _grant(who, action="update"):
    """A third-party GRANT request.
    """
    _declare(who, "user")
    return KeyRequest(requester_id=who, purpose=KeyPurpose.GRANT, action=action)


def _self(who):
    _declare(who, "principal")
    return KeyRequest(requester_id=who, purpose=KeyPurpose.SELF,
                      requester_type="principal", action="update")


# ---------------------------------------------------------------------------
# The property: no grant -> no key
# ---------------------------------------------------------------------------

class TestNoGrantNoKey:
    def test_master_key_denied_without_grant(self, oracle):
        with pytest.raises(GrantDenied):
            oracle.get_or_create_master_key(PRINCIPAL, _grant(MALLORY),
                                            collection_id=COLLECTION)

    def test_cell_key_denied_without_grant(self, oracle):
        with pytest.raises(GrantDenied):
            oracle.derive_cell_key(PRINCIPAL, COLLECTION, "anchor-0", _grant(MALLORY))

    def test_sse_key_denied_without_grant(self, oracle):
        with pytest.raises(GrantDenied):
            oracle.derive_sse_key(PRINCIPAL, _grant(MALLORY))

    def test_grant_holder_still_works(self, oracle):
        """The control must not be a blanket denial -- the authorized path works."""
        mk = oracle.get_or_create_master_key(PRINCIPAL, _grant(ALICE),
                                             collection_id=COLLECTION)
        assert isinstance(mk, bytes) and len(mk) == 32
        assert len(oracle.derive_cell_key(PRINCIPAL, COLLECTION, "anchor-0", _grant(ALICE))) == 32
        assert len(oracle.derive_sse_key(PRINCIPAL, _grant(ALICE))) == 32

    def test_grant_is_scoped_to_the_context_not_just_the_requester(self, oracle):
        """Alice's grant reaches COLLECTION only -- not a sibling collection."""
        with pytest.raises(GrantDenied):
            oracle.derive_cell_key(PRINCIPAL, "col-someone-elses", "anchor-0",
                                   _grant(ALICE))

    def test_denial_leaks_no_key_material(self, oracle):
        """A denied request must not create or persist a key as a side effect.

        Otherwise an unauthorized caller could still cause a principal's key to be
        generated -- which, on a principal whose real key merely failed to load,
        is the key-destruction path documented in the latticeMasterKeyStore.get.
        """
        with pytest.raises(GrantDenied):
            oracle.get_or_create_master_key(PRINCIPAL, _grant(MALLORY))
        assert oracle._store.storage == {}, "a denied request persisted a master key"
        assert oracle._cache == {}, "a denied request populated the key cache"


# ---------------------------------------------------------------------------
# The cache must not become a bypass
# ---------------------------------------------------------------------------

class TestCacheIsNotABypass:
    def test_unauthorized_caller_cannot_ride_on_a_warm_cache(self, oracle, verifier):
        """Caller A must not benefit from caller B's authorized fetch.

        This is the specific hazard of a process-lifetime cache keyed by principal
        alone: if the cache were consulted before the grant check, Alice's
        legitimate request would warm the entry and Mallory's would then be served
        from it without ever being checked -- an authorization bypass with a
        process-lifetime TTL and no log line.
        """
        # Alice legitimately warms the cache.
        alice_key = oracle.get_or_create_master_key(PRINCIPAL, _grant(ALICE),
                                                    collection_id=COLLECTION)
        assert PRINCIPAL in oracle._cache, "precondition: the cache is warm"

        # Mallory must still be refused.
        with pytest.raises(GrantDenied):
            oracle.get_or_create_master_key(PRINCIPAL, _grant(MALLORY),
                                            collection_id=COLLECTION)

        # ...and Alice is unaffected.
        assert oracle.get_or_create_master_key(PRINCIPAL, _grant(ALICE),
                                               collection_id=COLLECTION) == alice_key

    def test_every_call_is_checked_even_when_cached(self, oracle, verifier):
        """The verifier is consulted on every call, not only on a cache miss.

        A check that runs once per principal per process is not a check; it is a
        one-time initialization that a later revocation cannot affect.
        """
        oracle.get_or_create_master_key(PRINCIPAL, _grant(ALICE), collection_id=COLLECTION)
        n = len(verifier.calls)
        oracle.get_or_create_master_key(PRINCIPAL, _grant(ALICE), collection_id=COLLECTION)
        assert len(verifier.calls) == n + 1, "cached path skipped the grant check"

    def test_a_verdict_change_reaches_a_caller_whose_key_is_already_cached(self, oracle, verifier):
        """The MASTER-KEY cache does not shortcut the verifier: a new verdict lands at once.

        Named for what it proves. The verifier here is a memo-free double, so flipping its
        answer IS the new verdict — which makes this a statement about `oracle._cache`
        (the key cache), not about revocation propagating through the real verifier's
        memoized light-cone decision. That memo is the thing a revocation actually has to
        get past, and stubbing the component holding it is precisely how a test can pass
        while the window it is named for stays open.

        The revocation property lives in `tests/test_grant_cache_invalidation.py`, against
        a real store and the production-default `LightConeGrantVerifier`.
        """
        oracle.get_or_create_master_key(PRINCIPAL, _grant(ALICE), collection_id=COLLECTION)
        verifier.authorized = lambda **kw: False
        with pytest.raises(GrantDenied):
            oracle.get_or_create_master_key(PRINCIPAL, _grant(ALICE),
                                            collection_id=COLLECTION)


# ---------------------------------------------------------------------------
# SELF issuance is verified, not assumed
# ---------------------------------------------------------------------------

class TestSelfIssuance:
    def test_self_issuance_cannot_reach_another_principal(self, oracle):
        """SELF is not a bypass flag: the oracle checks the identity itself."""
        with pytest.raises(GrantDenied):
            oracle.get_or_create_master_key(PRINCIPAL, _self(MALLORY))

    def test_principal_may_obtain_its_own_key(self, oracle):
        assert len(oracle.get_or_create_master_key(PRINCIPAL, _self(PRINCIPAL))) == 32

    def test_self_and_grant_paths_agree_on_the_key(self, oracle):
        """Ingest (SELF) and query (GRANT) must derive the same key.

        If they diverged, cells written at index time would be undecryptable at
        query time -- the silent 'search returns nothing' failure mode.
        """
        write = oracle.derive_cell_key(PRINCIPAL, COLLECTION, "anchor-0", _self(PRINCIPAL))
        read = oracle.derive_cell_key(PRINCIPAL, COLLECTION, "anchor-0", _grant(ALICE))
        assert write == read


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------

class TestFailsClosed:
    def test_oracle_without_a_verifier_refuses_to_issue(self):
        """An oracle configured without a verifier must refuse to issue keys, not fall back to
        "we could not check, so we allowed it"."""
        bare = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())))
        with pytest.raises(GrantDenied):
            bare.get_or_create_master_key(PRINCIPAL, _grant(ALICE))

    def test_a_request_is_required_not_optional(self, oracle):
        """Omitting the requester must be a hard error.

        An optional requester would let a call site silently issue a key without one,
        defeating the coupling between key issuance and the grant check.
        """
        with pytest.raises(TypeError):
            oracle.get_or_create_master_key(PRINCIPAL)          # type: ignore[call-arg]
        with pytest.raises(TypeError):
            oracle.derive_cell_key(PRINCIPAL, COLLECTION, "a")  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            oracle.derive_sse_key(PRINCIPAL)                    # type: ignore[call-arg]

    def test_a_non_keyrequest_is_rejected(self, oracle):
        """A bare string must not be accepted as an identity assertion."""
        with pytest.raises(TypeError):
            oracle.get_or_create_master_key(PRINCIPAL, ALICE)   # type: ignore[arg-type]

    def test_anonymous_request_cannot_be_constructed(self):
        with pytest.raises(ValueError):
            KeyRequest(requester_id="", purpose=KeyPurpose.GRANT)


# ---------------------------------------------------------------------------
# End-to-end: the query path and the ingest path
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _live_anchorset():
    from mantle.search.anchors import store
    from mantle.search.anchors.anchorset import AnchorSet
    from mantle.search.anchors.repo import InMemoryAnchorRepo

    store.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", 16)
    aset.add_text("anchor-0", np.ones(16, dtype=np.float32))
    store.save_live_anchorset(aset)
    yield
    store.set_anchor_repo(None)


def _chunk(artifact_id, chunk_id=0, seed=1, dim=16):
    rng = np.random.default_rng(seed)
    return {"artifact_id": artifact_id, "chunk_id": chunk_id,
            "embedding": rng.standard_normal(dim).tolist()}


class TestEndToEnd:
    def test_ingest_then_authorized_query_works(self, oracle):
        """The grant property must not break the legitimate paths.

        Ingest writes under SELF (the context principal writing its own cells);
        an authorized query reads the same cells under GRANT.
        """
        cells = InMemoryCellStore()
        indexer = MantleIndexer(oracle, cells)
        engine = MantleQueryEngine(oracle, cells)

        touched = indexer.index_artifact(PRINCIPAL, COLLECTION, [_chunk("art-1")],
                                         _self(PRINCIPAL))
        assert touched == 1, "ingest must still work"

        hits = engine.search(_chunk("art-1")["embedding"], [(PRINCIPAL, COLLECTION)],
                             _grant(ALICE))
        assert [h.artifact_id for h in hits] == ["art-1"]

    def test_unauthorized_query_cannot_decrypt_the_cells(self, oracle):
        """Mallory can name the context, and still gets no plaintext.

        The context list is an input here -- Mallory supplies the very
        (principal, collection) pair the data lives in, simulating a caller who
        bypassed or forged the light-cone resolution upstream. The custody
        boundary must hold on its own, without relying on the caller having built
        an honest context list.
        """
        cells = InMemoryCellStore()
        indexer = MantleIndexer(oracle, cells)
        engine = MantleQueryEngine(oracle, cells)
        indexer.index_artifact(PRINCIPAL, COLLECTION, [_chunk("art-1")], _self(PRINCIPAL))

        with pytest.raises(GrantDenied):
            engine.search(_chunk("art-1")["embedding"], [(PRINCIPAL, COLLECTION)],
                          _grant(MALLORY))

    def test_unauthorized_ingest_cannot_write_into_another_principal(self, oracle):
        cells = InMemoryCellStore()
        indexer = MantleIndexer(oracle, cells)
        with pytest.raises(GrantDenied):
            indexer.index_artifact(PRINCIPAL, COLLECTION, [_chunk("art-1")],
                                   _self(MALLORY))


# ---------------------------------------------------------------------------
# The real verifier delegates to the light cone (it does not reimplement it)
# ---------------------------------------------------------------------------

class TestLightConeGrantVerifier:
    def test_authorizes_only_contexts_the_light_cone_returns(self, monkeypatch):
        import mantle.search.mantle.lightcone as lc

        monkeypatch.setattr(
            lc, "resolve_authorized_contexts",
            lambda db, who, *, lightcone, action, principal_type="user": (
                [(PRINCIPAL, COLLECTION)] if who == ALICE else []
            ),
        )
        v = LightConeGrantVerifier(object(), resolver=object())

        assert v.authorized(requester_id=ALICE, requester_type="user",
                            principal_id=PRINCIPAL, collection_id=COLLECTION,
                            action="read")
        assert not v.authorized(requester_id=MALLORY, requester_type="user",
                                principal_id=PRINCIPAL, collection_id=COLLECTION,
                                action="read")

    def test_a_principal_with_no_grants_is_denied_everything(self, monkeypatch):
        import mantle.search.mantle.lightcone as lc

        monkeypatch.setattr(lc, "resolve_authorized_contexts",
                            lambda db, who, *, lightcone, action, principal_type="user": [])
        v = LightConeGrantVerifier(object(), resolver=object())
        assert not v.authorized(requester_id=MALLORY, requester_type="user",
                                principal_id=PRINCIPAL, collection_id=COLLECTION,
                                action="read")
        assert not v.authorized(requester_id=MALLORY, requester_type="user",
                                principal_id=PRINCIPAL, collection_id=None,
                                action="read")

    def test_memoization_does_not_outlive_its_ttl(self, monkeypatch):
        """The authorization cache must expire; key material may be cached longer."""
        import mantle.search.mantle.lightcone as lc

        calls = []

        def _resolve(db, who, *, lightcone, action, principal_type="user"):
            calls.append(who)
            return [(PRINCIPAL, COLLECTION)]

        monkeypatch.setattr(lc, "resolve_authorized_contexts", _resolve)
        v = LightConeGrantVerifier(object(), resolver=object(), ttl_s=0)
        for _ in range(3):
            v.authorized(requester_id=ALICE, requester_type="user",
                         principal_id=PRINCIPAL, collection_id=COLLECTION,
                         action="read")
        assert len(calls) == 3, "a zero TTL must not memoize the decision"
