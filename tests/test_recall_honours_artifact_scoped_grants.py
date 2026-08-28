"""An artifact-scoped grant must not read back as recall over its collection.

The escalation this file pins down
==================================

Bob keeps two artifacts in ``col-bob`` and shares exactly ONE of them with
Alice. ``grants_router`` supports artifact-scoped grants and
``services.dependencies.check_access`` honours them — ``POST /artifacts/recall``
was the one path that did not.

The mechanism was a lossy projection, not a missing check.
``resolve_authorized_contexts`` took the light cone's artifact-granular id set,
read each artifact's ``collection_id``, and returned only the deduped
``(cell_principal, collection_id)`` pairs. After that line, "Alice may read
art-shared" and "Alice may read everything in col-bob" were the same value:

* the vector arm scored every chunk in every decrypted cell — a cell is the unit
  of ENCRYPTION, so holding its key decrypts artifacts nobody granted;
* the lexical arm filtered posting entries on ``collection_id`` alone.

The collection-shaped cut is not wrong, it is just not sufficient: keys are
derived per ``(principal, collection)``, so it is the only cut expressible as
key custody, and the oracle is right to authorize Alice for that key. The fix
keeps that cut and adds the artifact set back as a second, finer MEET —
see :class:`~mantle.search.mantle.lightcone.AuthorizedScope`.

The tests below run the production objects at three levels — each arm on its
own, and the whole accessor beneath ``POST /artifacts/recall`` — so that a
regression in either arm is caught where it happens rather than only in the
fused total. The two grant shapes are both covered: an artifact-scoped grant
must narrow, and a collection-scoped grant must NOT (its containment expansion
already puts every child id in the same set, so one set serves both cases and
nothing is special-cased).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pytest
from cryptography.fernet import Fernet

from mantle.search.mantle import MantleIndexer, MantleQueryEngine, OracleService
from mantle.search.mantle.oracle import FernetMasterKeyStore, KeyPurpose, KeyRequest
from mantle.search.mantle.sse.indexer import SseIndexer
from mantle.search.mantle.sse.sqlite_stores import SqlitePostingStore
from mantle.search.mantle.sse.narrowing import TokenNarrower
#: Only the pre-existing surface is imported at module scope, on purpose: the
#: escalation below is expressed entirely in terms of `POST /artifacts/recall`'s
#: own API, so it stays a runnable reproduction against a tree without the fix
#: rather than an ImportError. `resolve_authorized_scope` is imported inside the
#: tests that are specifically about the new return shape.
from mantle.search.mantle.sse.router_accessor import MantleSseSearchAccessor
from mantle.search.mantle.stores import InMemoryCellStore


# ---------------------------------------------------------------------------
# The scenario
# ---------------------------------------------------------------------------

ALICE = "user-alice"

#: Bob's collection. It is self-rooted (no origin parent in the fake lattice
#: below), so `resolve_cell_principal` returns the collection id itself — which
#: is exactly what the production index path wrote the cells under.
COLLECTION = "col-bob"
CELL_PRINCIPAL = COLLECTION

SHARED = "art-shared"       # the ONE artifact Bob shares with Alice
PRIVATE = "art-private"     # Bob's other artifact, in the same collection

#: A term present in BOTH artifacts. If it were only in the shared one, the test
#: would pass for the wrong reason.
TERM = "quasar"

_FIELDS = {
    SHARED: {
        "title": "quasar redshift survey",
        "content": "the quasar catalogue Bob agreed to share with Alice",
    },
    PRIVATE: {
        "title": "quasar acquisition memo",
        "content": "the quasar deal Bob did not share with anyone",
    },
}

DIM = 16


def _act(principal_id: str, principal_type: str = "user") -> None:
    """Stand in for the request boundary — the oracle binds requester to actor."""
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal

    set_acting_principal(
        ActingPrincipal(principal_id=principal_id, principal_type=principal_type,
                        source="artifact-scope-test")
    )


def _write_request():
    """The indexing request. A read action never mints a master key, so the
    corpus has to be written under a write — otherwise every query below would
    see `MasterKeyMissing` and pass by looking at an empty index."""
    _act(CELL_PRINCIPAL, "principal")
    return KeyRequest(requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
                      requester_type="principal", action="update")


def _read_as_alice():
    _act(ALICE, "user")
    return KeyRequest(requester_id=ALICE, purpose=KeyPurpose.GRANT,
                      requester_type="user", action="read")


class _OracleAsTheLightConeWouldDecide:
    """The grant verifier, decided the way ``LightConeGrantVerifier`` decides it.

    This is the load-bearing detail of the whole file: Alice IS authorized for
    the ``(col-bob, col-bob)`` key, because the verifier answers at collection
    granularity and her grant on ``art-shared`` genuinely reaches into that
    collection. A verifier that denied her here would hide the bug rather than
    reproduce it — the leak lives strictly downstream of correct key custody.
    """

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action) -> bool:
        if requester_id == principal_id:          # the principal's own cells
            return True
        return (requester_id == ALICE
                and principal_id == CELL_PRINCIPAL
                and collection_id in (None, COLLECTION))


# ---------------------------------------------------------------------------
# Lattice / light-cone doubles
# ---------------------------------------------------------------------------

class _FakeArtifacts:
    def __init__(self, docs: dict) -> None:
        self._docs = docs

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        return self._docs.get(artifact_id)


class _FakeGraph:
    """No origin edges: every node is its own origin root."""

    def edges_of(self, node, label=None, direction="out", limit=None) -> list:
        return []


class _FakeStoreDB:
    def __init__(self) -> None:
        self.artifacts = _FakeArtifacts({
            # The collection is an artifact too, and self-references its own id.
            COLLECTION: {"id": COLLECTION, "root_id": COLLECTION, "context": {}},
            SHARED: {"id": SHARED, "root_id": SHARED, "collection_id": COLLECTION,
                     "context": {"title": _FIELDS[SHARED]["title"]},
                     "content": _FIELDS[SHARED]["content"], "state": "committed"},
            PRIVATE: {"id": PRIVATE, "root_id": PRIVATE, "collection_id": COLLECTION,
                      "context": {"title": _FIELDS[PRIVATE]["title"]},
                      "content": _FIELDS[PRIVATE]["content"], "state": "committed"},
        })
        self.graph = _FakeGraph()


class _GrantedResource:
    """A grant as `mask_of` reads one: a resource, an effect, and the action flags."""

    def __init__(self, resource_id):
        self.resource_id = resource_id
        self.effect = "allow"
        for _a in ("create", "read", "update", "delete", "evict", "invoke", "add", "share",
                   "admin"):
            setattr(self, "can_" + _a, True)


class _FakeLightCone:
    """Returns the id set the real resolver would return for a given grant shape.

    ``resolve()`` is artifact-granular by contract: direct grants PLUS everything
    reachable through action-permitted containment and context edges. So an
    artifact-scoped grant on ``art-shared`` yields ``{art-shared}``, while a
    collection-scoped grant on ``col-bob`` yields the collection id and — via
    ``list_origin_descendants`` — every artifact filed under it. Both shapes are
    exercised below.
    """

    def __init__(self, authorized: set) -> None:
        self._authorized = set(authorized)

    def resolve(self, principal_id, action="read", *, principal_type="user") -> set:
        return set(self._authorized) if principal_id == ALICE else set()

    def _grants_for(self, principal_id: str, principal_type: str = "user"):
        """The same reach, expressed as grants.

        `resolve_authorized_scope` authorizes each candidate by walking up from it and testing the
        resources on the way, so a double saying "these ids are authorized" says it by granting
        them. Both shapes this file exercises survive that translation intact: the artifact-scoped
        set grants only `art-shared`, whose CUSTODY still widens to `col-bob` because keys are
        derived per collection — which is the escalation this file exists to pin — while the
        artifact meet still admits `art-shared` alone.
        """
        return [_GrantedResource(a) for a in self.resolve(principal_id,
                                                          principal_type=principal_type)]


ARTIFACT_SCOPED = {SHARED}
COLLECTION_SCOPED = {COLLECTION, SHARED, PRIVATE}


# ---------------------------------------------------------------------------
# The stack (production objects; only the lattice + light cone are doubles)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _live_anchorset():
    """One anchor, so every dim-16 vector routes to the same cell — which is the
    adversarial routing for this test: both artifacts land in ONE cell, so
    decrypting it hands the query every chunk Bob owns."""
    from mantle.search.anchors import store
    from mantle.search.anchors.anchorset import AnchorSet
    from mantle.search.anchors.repo import InMemoryAnchorRepo

    store.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", DIM)
    aset.add_text("anchor-0", np.ones(DIM, dtype=np.float32))
    store.save_live_anchorset(aset)
    yield
    store.set_anchor_repo(None)


def _vec(seed: int) -> list:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DIM).tolist()


#: One query vector, and two near-identical document vectors, so BOTH artifacts
#: are genuine vector-arm matches. Any filtering that happens is authorization,
#: not similarity.
QUERY_VEC = _vec(7)
_DOC_VECS = {SHARED: _vec(7), PRIVATE: _vec(7)}


@pytest.fixture
def stack(tmp_path):
    """Real SSE + MANTLE indexes over Bob's two artifacts, plus both engines."""
    oracle = OracleService(
        FernetMasterKeyStore(Fernet(Fernet.generate_key())),
        grant_verifier=_OracleAsTheLightConeWouldDecide(),
    )
    root = os.path.join(str(tmp_path), "sse-index")
    postings = SqlitePostingStore(os.path.join(root, "mantle-sse.db"))
    cells = InMemoryCellStore()

    sse_indexer = SseIndexer(oracle, postings)
    vec_indexer = MantleIndexer(oracle, cells)
    for artifact_id, fields in _FIELDS.items():
        sse_indexer.index_artifact(
            CELL_PRINCIPAL, COLLECTION, artifact_id, fields, _write_request(),
        )
        vec_indexer.index_artifact(
            CELL_PRINCIPAL, COLLECTION,
            [{"artifact_id": artifact_id, "chunk_id": 0,
              "embedding": _DOC_VECS[artifact_id], "text": fields["content"]}],
            _write_request(),
        )

    return {
        "vector": MantleQueryEngine(oracle, cells),
        "narrower": TokenNarrower(oracle, postings),
    }


def _reach(stack, text: str) -> set:
    """The narrowing's OWN answer over Bob's collection, un-met.

    The lexical side has no artifact-granular cut of its own any more, and that is the point
    of the tests below: the narrowing reports what carries the terms, and
    `resolve_authorized_scope` is the single site where that meets what the light cone
    authorized. A second opinion about it inside the narrowing is exactly what would let the
    two disagree.
    """
    lookup = stack["narrower"].lookup_for(text, _read_as_alice())
    return set() if lookup is None else set(lookup([(CELL_PRINCIPAL, COLLECTION)]))


def _accessor(stack, authorized: set) -> MantleSseSearchAccessor:
    """The production composition: the postings narrow, the cells rank.

    Both are real objects over the real encrypted indexes built above — only the lattice and
    the light cone are doubles — so what the assertions below measure is the meet of the
    light cone against retrieval, not a stub's opinion of it.
    """
    return MantleSseSearchAccessor(
        _FakeLightCone(authorized),
        store_db=_FakeStoreDB(),
        embeddings=lambda texts: [list(QUERY_VEC) for _ in texts],
        narrower=stack["narrower"],
        ranker=stack["vector"],
    )


def _query(size: int = 20, user_id: str = ALICE):
    """A recall request, with the acting principal bound as the router would.

    The oracle refuses to issue a key to a `requester_id` that is not the
    authenticated actor, so the identity has to be in scope before the accessor
    builds its own KeyRequest from `query.user_id`.
    """
    from mantle.search.types import SearchQuery

    _act(user_id, "user")
    return SearchQuery(query_text=TERM, user_id=user_id, size=size)


# ---------------------------------------------------------------------------
# The corpus is real — a test that filtered nothing would still have to pass
# these, or the ones below prove nothing.
# ---------------------------------------------------------------------------

class TestTheCorpusActuallyMatchesBothArtifacts:
    def test_both_artifacts_are_lexical_matches_for_the_term(self, stack):
        assert _reach(stack, TERM) == {SHARED, PRIVATE}

    def test_both_artifacts_are_vector_matches_for_the_query(self, stack):
        hits = stack["vector"].search(
            QUERY_VEC, [(CELL_PRINCIPAL, COLLECTION)], _read_as_alice(),
        )
        assert {h.artifact_id for h in hits} == {SHARED, PRIVATE}

    def test_alice_really_does_hold_the_collections_key(self, stack):
        """The oracle authorizes her — so nothing below can pass by key denial."""
        stack["vector"]._oracle.authorize(
            CELL_PRINCIPAL, COLLECTION, _read_as_alice(),
        )


# ---------------------------------------------------------------------------
# THE ESCALATION — this is the test that failed before the fix
# ---------------------------------------------------------------------------

class TestArtifactScopedGrantDoesNotEscalate:
    def test_recall_returns_only_the_shared_artifact(self, stack):
        """Bob shared ONE artifact. Alice recalls a term present in both.

        Before the fix this returned both: the light cone's ``{art-shared}`` was
        projected to ``(col-bob, col-bob)`` and the artifact granularity was
        gone by the time either arm ran.
        """
        result = _accessor(stack, ARTIFACT_SCOPED).search(_query())
        assert {h.doc_id for h in result.hits} == {SHARED}, (
            "an artifact-scoped grant escalated to whole-collection recall"
        )
        assert result.total == 1

    def test_candidates_returns_only_the_shared_artifact(self, stack):
        """`candidates()` is documented as the chokepoint every search flavor
        ranks within, so a leak here is a leak in every flavor at once."""
        out = _accessor(stack, ARTIFACT_SCOPED).candidates(_query())
        assert {c["artifact_id"] for c in out["candidates"]} == {SHARED}

    def test_the_lexical_meet_narrows_to_the_grant(self, stack):
        """Where the lexical half of the escalation is now closed.

        The narrowing itself is un-met — it reaches BOTH artifacts, because both carry the
        term in a collection Alice holds the key for — and `resolve_authorized_scope` is the
        one place that answer meets the light cone's. The `authorized_artifacts` cut this used
        to exercise inside the lexical arm is gone with the arm; the meet replaced it, and a
        meet cannot admit an id the resolver did not produce.
        """
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        assert _reach(stack, TERM) == {SHARED, PRIVATE}

        lookup = stack["narrower"].lookup_for(TERM, _read_as_alice())
        _act(ALICE, "user")
        scope = resolve_authorized_scope(
            _FakeStoreDB(), ALICE,
            lightcone=_FakeLightCone(ARTIFACT_SCOPED),
            token_lookup=lambda pairs: lookup(pairs).keys(),
        )
        assert scope.artifact_ids == frozenset({SHARED})

    def test_the_vector_arm_alone_narrows_to_the_grant(self, stack):
        """The cell decrypts either way — that is the point. Authorization has
        to happen on the chunks, not on the ability to read them."""
        hits = stack["vector"].search(
            QUERY_VEC, [(CELL_PRINCIPAL, COLLECTION)], _read_as_alice(),
            authorized_artifacts=frozenset({SHARED}),
        )
        assert {h.artifact_id for h in hits} == {SHARED}

    def test_a_phrase_query_narrows_too(self, stack):
        """The bigram gate is a second entry-selecting site in the lexical arm, and it goes
        through the same meet: what it admits is still only a candidate, and the light cone
        still decides."""
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        lookup = stack["narrower"].lookup_for('"quasar redshift"', _read_as_alice())
        assert set(lookup([(CELL_PRINCIPAL, COLLECTION)])) == {SHARED}, (
            "the phrase clears the gate in the shared artifact — otherwise the assertion "
            "below would be vacuous"
        )
        # A light cone that authorizes ONLY the private artifact. The gate admits the shared
        # one; the meet keeps neither, because the gate's candidate is not authorized and the
        # authorized id is not a candidate.
        _act(ALICE, "user")
        scope = resolve_authorized_scope(
            _FakeStoreDB(), ALICE,
            lightcone=_FakeLightCone({PRIVATE}),
            token_lookup=lambda pairs: lookup(pairs).keys(),
        )
        assert scope.artifact_ids == frozenset(), (
            "the phrase gate admitted an artifact outside the grant"
        )

    def test_an_empty_authorized_set_is_not_read_as_unspecified(self, stack):
        """`None` means "no artifact information"; an empty set means "nothing is
        authorized". Conflating them would make a revoked principal see the whole
        collection — the same bug with a different trigger.

        The lexical half of this is now structural rather than checked: the narrowing has no
        `authorized_artifacts` parameter to conflate, and an empty light cone means
        `resolve_authorized_scope` returns before a posting list is opened. The vector arm
        still takes the set, so it still has to tell the two apart.
        """
        assert stack["vector"].search(
            QUERY_VEC, [(CELL_PRINCIPAL, COLLECTION)], _read_as_alice(),
            authorized_artifacts=frozenset(),
        ) == []
        assert _accessor(stack, set()).search(_query()).hits == []


# ---------------------------------------------------------------------------
# The other grant shape must be untouched
# ---------------------------------------------------------------------------

class TestCollectionScopedGrantStillSeesEverything:
    def test_recall_returns_both_artifacts(self, stack):
        """A grant on the collection expands, through containment, to every id
        under it — so the SAME meet is a no-op here and no case is special."""
        result = _accessor(stack, COLLECTION_SCOPED).search(_query())
        assert {h.doc_id for h in result.hits} == {SHARED, PRIVATE}

    def test_candidates_returns_both_artifacts(self, stack):
        out = _accessor(stack, COLLECTION_SCOPED).candidates(_query())
        assert {c["artifact_id"] for c in out["candidates"]} == {SHARED, PRIVATE}

    def test_an_unrelated_principal_gets_nothing(self, stack):
        acc = _accessor(stack, COLLECTION_SCOPED)
        result = acc.search(_query(user_id="user-mallory"))
        assert result.hits == []


# ---------------------------------------------------------------------------
# The resolver keeps both granularities
# ---------------------------------------------------------------------------

class TestResolveAuthorizedScope:
    def test_it_keeps_the_artifact_ids_alongside_the_pairs(self):
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        scope = resolve_authorized_scope(
            _FakeStoreDB(), ALICE, lightcone=_FakeLightCone(ARTIFACT_SCOPED),
        )
        assert scope.contexts == [(CELL_PRINCIPAL, COLLECTION)]
        assert scope.artifact_ids == frozenset({SHARED}), (
            "the artifact set was collapsed into the collection pair — the "
            "escalation is exactly this loss of granularity"
        )

    def test_a_collection_grant_carries_its_descendants(self):
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        scope = resolve_authorized_scope(
            _FakeStoreDB(), ALICE, lightcone=_FakeLightCone(COLLECTION_SCOPED),
        )
        assert scope.contexts == [(CELL_PRINCIPAL, COLLECTION)]
        assert scope.artifact_ids == frozenset(COLLECTION_SCOPED)

    def test_the_coarse_wrapper_still_answers_key_custody(self):
        """`oracle.LightConeGrantVerifier` asks a collection-shaped question and
        must keep getting a collection-shaped answer."""
        from mantle.search.mantle.lightcone import resolve_authorized_contexts

        assert resolve_authorized_contexts(
            _FakeStoreDB(), ALICE, lightcone=_FakeLightCone(ARTIFACT_SCOPED),
        ) == [(CELL_PRINCIPAL, COLLECTION)]

    def test_no_grants_yields_an_empty_scope(self):
        from mantle.search.mantle.lightcone import resolve_authorized_scope

        scope = resolve_authorized_scope(
            _FakeStoreDB(), "user-mallory", lightcone=_FakeLightCone(ARTIFACT_SCOPED),
        )
        assert scope.contexts == []
        assert scope.artifact_ids == frozenset()
