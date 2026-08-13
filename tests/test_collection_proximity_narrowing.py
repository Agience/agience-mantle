# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
# ---------------------------------------------------------------------------

"""Collection proximity: a digest narrows a light cone, and can do nothing else.

The counterpart of `test_blind_token_narrowing.py` and of
`test_field_filters_narrow_recall.py`'s `test_that_answer_is_identical_to_a_filter_matching_
nothing`, for the third narrowing. Same claims, same order of importance:

1. **A proximity match can only narrow.** `resolve_authorized_scope` meets the lookup's answer
   into `artifact_ids` with `&`, so a collection the light cone did not authorize contributes
   nothing however near it is. The narrower genuinely returns such ids here — `col-secret` sits
   under the SAME cell principal, so Alice holds its key and really can open and compare its
   digest — so the meet does visible work rather than being handed a subset already.

2. **The narrowing is not an existence oracle.** A query digest that matches an unreadable
   collection exactly, and one that matches nothing at all, must be indistinguishable in EVERY
   observable. If they differed, a caller could learn that a collection resembling theirs
   exists somewhere they cannot read.

3. **Narrowing carries what distance does not.** `rows` is a property the store already knows,
   so the 4-row collection and the 12-row one never meet in one gallery — and
   `spectral_distance` is used exactly as shipped, with nothing added to compensate for the
   short-record attractor.

4. **Absent, it changes nothing.** The proximity narrowing rides the resolver's existing
   `token_lookup` slot, so `None` is byte-identical to a resolve that never went near it.

The stack is the production one. Only the lattice and the light cone are doubles: the digests
are real reads off real frames built by the real tokenizer, sealed with a real `OracleService`
key into a real `InMemoryPostingStore`, and read back through the real `DigestSlot`.
"""
from __future__ import annotations

from typing import Optional

import pytest
from cryptography.fernet import Fernet

from mantle.search.field_filters import compile_filters
from mantle.search.ingest import collection_frame as CF
from mantle.search.mantle import collection_proximity as CP
from mantle.search.mantle.lightcone import resolve_authorized_scope
from mantle.search.mantle.oracle import (
    FernetMasterKeyStore,
    KeyPurpose,
    KeyRequest,
    OracleService,
)
from mantle.search.mantle.sse.posting import InMemoryPostingStore
from mantle.search.query_parser import parse_query

import numpy as np

#: `digest_collection`/`CollectionProximityNarrower` take an injected proximity instrument now
#: — mantle never imports entroptics, so the real one (`entroptics.proximity`) lives in and is
#: tested by `agience-entroptics`'s own suite. This file is about the narrowing/authorization
#: plumbing built around an injected instrument, not the instrument's own math, so a minimal
#: stand-in is enough — deterministic, real (not a fake number), and correct by construction
#: rather than optimised, since the probe's exactness is not this file's claim to prove.
_ENGINE_ID = "test.stub.svd"


def _read(matrix):
    A = np.asarray(matrix, dtype=np.float64)
    centred = A - np.median(A, axis=0)
    return np.linalg.svd(centred, compute_uv=False)


def _common_prefix(x, y):
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    n = min(a.size, b.size)
    return a[:n], b[:n]


def _spectral_distance(x, y):
    a, b = _common_prefix(x, y)
    return float(np.linalg.norm(a - b))


class _Hit:
    def __init__(self, index, distance):
        self.index = index
        self.distance = distance


class _BruteForceProbe:
    """A correct-by-construction stand-in for `entroptics.proximity.SpectrumProbe`: same
    `within`/`nearest` interface, a full scan instead of the real probe's searchsorted-based
    pruning. This file's claims are about narrowing and authorization, not about the probe's
    own exactness (which is entroptics' own test's job), so a brute-force reference that is
    trivially correct is the right double here."""

    def __init__(self, spectra):
        self._gallery = [np.asarray(s, dtype=np.float64).ravel() for s in spectra]

    def within(self, query, radius):
        out = [_Hit(i, _spectral_distance(query, g)) for i, g in enumerate(self._gallery)]
        out = [h for h in out if h.distance <= radius]
        out.sort(key=lambda h: (h.distance, h.index))
        return out

    def nearest(self, query, k=1):
        out = [_Hit(i, _spectral_distance(query, g)) for i, g in enumerate(self._gallery)]
        out.sort(key=lambda h: (h.distance, h.index))
        return out[:k]


ALICE = "user-alice"

#: One cell principal over several collections — the ordinary shape when collections share an
#: origin root. It is what makes the security case non-vacuous: Alice legitimately holds this
#: principal's SSE key, so she can decrypt every digest below, including the one she may not
#: act on. Nothing about `col-secret` is hidden by a key she lacks.
OWNER = "col-root"

COL_A = "col-a"
COL_B = "col-b"
COL_SECRET = "col-secret"
COL_SMALL = "col-small"

#: The collection whose digest is the QUERY. Alice holds it; it is a genuine collection of her
#: own that happens to be built from the same material as `col-secret`.
COL_QUERY = "col-query"
#: A second query collection resembling nothing in the store.
COL_ALIEN = "col-alien"

_A_WORDS = "quarterly budget report ledger reconciliation"
_B_WORDS = "cover plate image catalogue plate scan"
_SECRET_WORDS = "acquisition record merger diligence memorandum"
_ALIEN_WORDS = "hydrothermal basalt olivine spectroscopy"


def _texts(words: str, n: int, *, salt: str) -> list:
    """`n` artifacts whose counts differ from each other in a fixed, reproducible way."""
    out = []
    parts = words.split()
    for i in range(n):
        repeated = " ".join(" ".join([w] * (1 + ((i + j) % 4)))
                            for j, w in enumerate(parts))
        out.append(f"{repeated} {salt}{i}")
    return out


#: `col-query` and `col-secret` are built from the SAME texts, so their frames are identical
#: and their digests sit at distance exactly 0. That is what lets a radius of 0 mean "this
#: collection and nothing else" without any number being chosen for the corpus.
_ROWS = 12
_CORPUS = {
    COL_A:      _texts(_A_WORDS, _ROWS, salt="a"),
    COL_B:      _texts(_B_WORDS, _ROWS, salt="b"),
    COL_SECRET: _texts(_SECRET_WORDS, _ROWS, salt="s"),
    COL_SMALL:  _texts(_SECRET_WORDS, 4, salt="s"),
    COL_QUERY:  _texts(_SECRET_WORDS, _ROWS, salt="s"),
    COL_ALIEN:  _texts(_ALIEN_WORDS, _ROWS, salt="z"),
}

#: Every collection Alice's light cone reaches. `col-secret` and `col-small` are NOT in it.
AUTHORIZED_COLLECTIONS = (COL_A, COL_B, COL_QUERY)
#: Every pair a narrower could be handed if the contexts were wrong — used to show the index
#: really can tell the secret apart.
ALL_PAIRS = tuple((OWNER, c) for c in _CORPUS)


def _members(collection_id: str) -> list:
    return [f"{collection_id}:art-{i:03d}" for i in range(len(_CORPUS[collection_id]))]


def _fields(collection_id: str) -> list:
    return [(a, {"content": t})
            for a, t in zip(_members(collection_id), _CORPUS[collection_id])]


AUTHORIZED_ARTIFACTS = {a for c in AUTHORIZED_COLLECTIONS for a in _members(c)}


# ---------------------------------------------------------------------------
# Doubles: the lattice and the light cone only
# ---------------------------------------------------------------------------


def _act(principal_id: str, principal_type: str = "user") -> None:
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal

    set_acting_principal(
        ActingPrincipal(principal_id=principal_id, principal_type=principal_type,
                        source="collection-proximity-test")
    )


class _AliceHoldsTheOwnerKey:
    """Key custody as `LightConeGrantVerifier` decides it — per `(principal, collection)`.

    Alice holds this principal's key outright. Everything below therefore happens strictly
    downstream of CORRECT custody: she can open `col-secret`'s digest, and the only thing
    keeping its members out of her results is the meet.
    """

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action) -> bool:
        if requester_id == principal_id:
            return True
        return requester_id == ALICE and principal_id == OWNER


class _FakeArtifacts:
    def __init__(self, docs: dict) -> None:
        self._docs = docs

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        return self._docs.get(artifact_id)


class _FakeGraph:
    """Every collection hangs off `OWNER` by an origin containment edge, so
    `resolve_cell_principal` walks each of them to the one principal."""

    def edges_of(self, node, label=None, direction="out", limit=None) -> list:
        if direction == "in" and node in _CORPUS:
            return [{"src": OWNER, "dst": node, "label": "contains",
                     "is_origin": True, "propagate": None, "props": {"is_origin": True}}]
        return []


def _doc(artifact_id: str, collection_id: str, content_type: str) -> dict:
    return {
        "id": artifact_id,
        "root_id": artifact_id,
        "collection_id": collection_id,
        "content_type": content_type,
        "created_by": "user-bob",
        "created_time": "2025-01-01T00:00:00Z",
        "modified_time": "2025-01-01T00:00:00Z",
        "state": "committed",
        "context": {"title": artifact_id, "tags": [], "description": "a filed artifact"},
    }


#: Half of `col-a` is a pdf, so a field filter has something of its own to remove and the
#: three-way composition below is not answered by any one cut alone.
def _content_type(artifact_id: str) -> str:
    return "application/pdf" if artifact_id.endswith(("0", "2", "4", "6", "8")) \
        else "image/png"


class _FakeStoreDB:
    def __init__(self) -> None:
        docs = {}
        for collection_id in _CORPUS:
            docs[collection_id] = {"id": collection_id, "root_id": collection_id,
                                   "context": {}}
            for artifact_id in _members(collection_id):
                docs[artifact_id] = _doc(artifact_id, collection_id,
                                         _content_type(artifact_id))
        docs[OWNER] = {"id": OWNER, "root_id": OWNER, "context": {}}
        self.artifacts = _FakeArtifacts(docs)
        self.graph = _FakeGraph()


class _FakeLightCone:
    def __init__(self, authorized: set) -> None:
        self._authorized = set(authorized)

    def resolve(self, principal_id, action="read", *, principal_type="user") -> set:
        return set(self._authorized) if principal_id == ALICE else set()


# ---------------------------------------------------------------------------
# The real digests
# ---------------------------------------------------------------------------


def _write_request():
    _act(OWNER, "principal")
    return KeyRequest(requester_id=OWNER, purpose=KeyPurpose.SELF,
                      requester_type="principal", action="update")


def _read_request():
    _act(ALICE, "user")
    return KeyRequest(requester_id=ALICE, purpose=KeyPurpose.GRANT,
                      requester_type="user", action="read")


@pytest.fixture
def stack():
    oracle = OracleService(
        FernetMasterKeyStore(Fernet(Fernet.generate_key())),
        grant_verifier=_AliceHoldsTheOwnerKey(),
    )
    postings = InMemoryPostingStore()
    slot = CP.DigestSlot(oracle, postings)
    digests = {}
    for collection_id in _CORPUS:
        digest = CF.digest_collection(
            _fields(collection_id), exhaustive=True, collection_id=collection_id,
            read=_read, engine_id=_ENGINE_ID,
        )
        digests[collection_id] = digest
        slot.put(OWNER, digest, _write_request())
    return {
        "oracle": oracle,
        "postings": postings,
        "slot": slot,
        "digests": digests,
        "narrower": CP.CollectionProximityNarrower(
            oracle, postings, _members, probe_factory=_BruteForceProbe,
        ),
    }


def _lookup(stack, query_collection, *, radius=None, nearest=None, properties=None):
    return stack["narrower"].lookup_for(
        stack["digests"][query_collection], _read_request(),
        radius=radius, nearest=nearest, properties=properties,
    )


def _raw(stack, query_collection, *, pairs=ALL_PAIRS, radius=None, nearest=None,
         properties=None):
    """The narrower's OWN answer, un-met — what the light cone is protecting against."""
    return stack["narrower"].ids_near(
        stack["digests"][query_collection], pairs, _read_request(),
        radius=radius, nearest=nearest, properties=properties,
    )


def _scope(stack, *, authorized=None, query=None, radius=None, nearest=None,
           properties=None, filters=None, lookup=None):
    """`resolve_authorized_scope` with whichever cuts this case is exercising."""
    if lookup is None and query is not None:
        lookup = _lookup(stack, query, radius=radius, nearest=nearest,
                         properties=properties)
    predicate = None
    if filters is not None:
        predicate = compile_filters(parse_query(filters).filters)
    _act(ALICE, "user")
    return resolve_authorized_scope(
        _FakeStoreDB(), ALICE,
        lightcone=_FakeLightCone(AUTHORIZED_ARTIFACTS if authorized is None else authorized),
        artifact_predicate=predicate,
        token_lookup=lookup,
    )


# ---------------------------------------------------------------------------
# The corpus is real. Without these, everything below could pass vacuously.
# ---------------------------------------------------------------------------


class TestTheGalleryWouldOtherwiseReturnTheSecret:
    def test_the_query_collection_and_the_secret_are_at_distance_exactly_zero(self, stack):
        """Built from the same texts, so the read is the same read — which is what lets a
        radius of 0 mean "this collection and nothing else" without a number being picked."""
        q = stack["digests"][COL_QUERY].read
        assert _spectral_distance(q, stack["digests"][COL_SECRET].read) == 0.0
        for other in (COL_A, COL_B, COL_ALIEN):
            assert _spectral_distance(q, stack["digests"][other].read) > 0.0

    def test_the_secrets_digest_is_readable_with_the_key_alice_holds(self, stack):
        """Not an absent record, and not one behind a key she lacks."""
        got = stack["slot"].get(OWNER, COL_SECRET, _read_request())
        assert got == stack["digests"][COL_SECRET]

    def test_the_narrower_reaches_the_secret_when_handed_its_pair(self, stack):
        assert _raw(stack, COL_QUERY, radius=0.0) == set(_members(COL_SECRET)) | set(
            _members(COL_QUERY))

    def test_a_query_resembling_nothing_reaches_nothing(self, stack):
        assert _raw(stack, COL_ALIEN, radius=0.0) == set(_members(COL_ALIEN))

    def test_the_three_cuts_each_remove_something_the_others_keep(self, stack):
        """The corpus property the composition test rests on."""
        everything = _scope(stack).artifact_ids
        assert everything == frozenset(AUTHORIZED_ARTIFACTS)
        by_filter = _scope(stack, filters="type:application/pdf").artifact_ids
        by_proximity = _scope(stack, query=COL_A, radius=0.0).artifact_ids
        assert by_filter < everything
        assert by_proximity < everything
        assert by_filter != by_proximity


# ---------------------------------------------------------------------------
# THE SECURITY PROPERTY
# ---------------------------------------------------------------------------


class TestAProximityMatchOnlyEverNarrows:
    def test_a_digest_naming_an_unreadable_collection_matches_nothing_of_its(self, stack):
        """`col-query` is Alice's own, so it survives; not one member of `col-secret` does,
        though the two are at distance exactly 0 and her key opens both digests."""
        ids = _scope(stack, query=COL_QUERY, radius=0.0).artifact_ids
        assert ids == frozenset(_members(COL_QUERY))
        assert not (ids & set(_members(COL_SECRET)))

    def test_that_answer_is_identical_to_a_digest_matching_nothing(self, stack):
        """The property, stated as an indistinguishability.

        A query digest matching a real, indexed collection Alice may not read — one whose
        digest she can and does decrypt — must leave by the same door as a query matching no
        collection at all. Here both are asked about `col-secret`'s slice of the answer: if
        they differed in ANY observable, the narrowing would be an oracle for the existence of
        a collection resembling the caller's, outside the light cone.

        `test_the_secrets_digest_is_readable_with_the_key_alice_holds` is the other half: the
        gallery really can tell these two queries apart. Only the meet cannot.
        """
        # a query that matches col-secret exactly, restricted to what the secret would add
        unreadable = _scope(stack, authorized=set(_members(COL_A)), query=COL_QUERY,
                            radius=0.0)
        # a query that matches nothing at all
        nonexistent = _scope(stack, authorized=set(_members(COL_A)), query=COL_ALIEN,
                             radius=0.0)

        assert unreadable == nonexistent
        assert unreadable.artifact_ids == nonexistent.artifact_ids == frozenset()
        assert unreadable.contexts == nonexistent.contexts == [(OWNER, COL_A)]
        assert unreadable.updated_at == nonexistent.updated_at == {}

    def test_the_difference_the_gallery_can_see_is_the_difference_the_meet_erases(self, stack):
        """Stated as one assertion so the two halves cannot drift apart."""
        assert _raw(stack, COL_QUERY, radius=0.0) != _raw(stack, COL_ALIEN, radius=0.0)
        assert _scope(stack, authorized=set(_members(COL_A)), query=COL_QUERY, radius=0.0) \
            == _scope(stack, authorized=set(_members(COL_A)), query=COL_ALIEN, radius=0.0)

    def test_a_digest_cannot_widen_past_the_light_cone(self, stack):
        """A digest TRUE of the secret and of nothing Alice holds still returns nothing. There
        is no query for which proximity adds an id the resolver did not."""
        scope = _scope(stack, authorized=set(_members(COL_B)), query=COL_QUERY, radius=0.0)
        assert scope.artifact_ids == frozenset()

    def test_the_secrets_own_pair_never_reaches_the_narrower(self, stack):
        """The second, independent cut: the callback is handed the resolver's CONTEXTS, so a
        collection outside the light cone contributes no pair and its digest is never opened
        at all. The meet is what makes the property hold even if this were ever wrong."""
        seen = []

        def _spy(pairs):
            seen.append(list(pairs))
            return set()

        _scope(stack, lookup=_spy)
        assert seen, "the lookup must have been called"
        assert {c for _p, c in seen[0]} == set(AUTHORIZED_COLLECTIONS)
        assert COL_SECRET not in {c for _p, c in seen[0]}

    def test_a_narrowed_scope_is_always_a_subset_of_the_unnarrowed_one(self, stack):
        """Swept over every query and both probe shapes rather than asserted on one case."""
        unnarrowed = _scope(stack).artifact_ids
        for query in _CORPUS:
            for kwargs in ({"radius": 0.0}, {"radius": 1e9}, {"nearest": 1},
                           {"nearest": 5}):
                got = _scope(stack, query=query, **kwargs).artifact_ids
                assert got <= unnarrowed, (query, kwargs)

    def test_a_lookup_that_returns_the_whole_store_still_narrows(self, stack):
        """The meet does not trust its input. A lookup claiming every id in existence —
        including ids no light cone produced — cannot add one."""
        everything = {a for c in _CORPUS for a in _members(c)} | {"art-nowhere"}
        assert _scope(stack, lookup=lambda pairs: everything).artifact_ids == frozenset(
            AUTHORIZED_ARTIFACTS)

    def test_a_lookup_that_raises_narrows_to_nothing(self, stack):
        """Fail closed for recall, and never a way to make the resolve fail."""
        def _boom(pairs):
            raise RuntimeError("digest store unavailable")

        assert _scope(stack, lookup=_boom).artifact_ids == frozenset()

    def test_an_unreadable_digest_is_the_same_answer_as_an_absent_one(self, stack):
        """A corrupt digest must not be distinguishable from a collection that has none —
        the same trade `TokenNarrower._entries` takes for an unreadable posting list."""
        key = stack["oracle"].derive_sse_key(OWNER, _read_request())
        token = CP.digest_slot_token(key, COL_B)
        stack["postings"].put_posting(OWNER, token, b"garbage that will not open")
        assert stack["slot"].get(OWNER, COL_B, _read_request()) is None
        assert stack["slot"].get(OWNER, "col-never-digested", _read_request()) is None

    def test_the_contexts_are_untouched_by_the_cut(self, stack):
        """Key custody is not a function of what the caller searched for. The pairs are the
        same with the narrowing on and off, and the narrowing is handed exactly those pairs."""
        assert _scope(stack, query=COL_ALIEN, radius=0.0).contexts == _scope(stack).contexts


# ---------------------------------------------------------------------------
# Narrowing carries what distance does not
# ---------------------------------------------------------------------------


class TestNarrowingCarriesWhatDistanceDoesNot:
    def test_rows_is_read_off_the_record_of_the_frame_that_was_read(self, stack):
        gallery = stack["narrower"].candidates(
            stack["digests"][COL_QUERY], ALL_PAIRS, _read_request())
        by_id = {props.collection_id: props for props, _d in gallery}
        assert by_id[COL_SMALL].rows == 4
        assert by_id[COL_SECRET].rows == _ROWS
        assert by_id[COL_A].principal_id == OWNER

    def test_a_size_narrowing_removes_the_short_collection_before_it_is_ranked(self, stack):
        """The short-record attractor, prevented by never making the comparison.

        `col-small` is four rows of the SAME material as the twelve-row `col-secret`, which is
        exactly the shape that wins on a common prefix. With `same_rows` it is not in the
        gallery at all, so no distance to it is ever computed."""
        wide = stack["narrower"].candidates(
            stack["digests"][COL_QUERY], ALL_PAIRS, _read_request())
        narrow = stack["narrower"].candidates(
            stack["digests"][COL_QUERY], ALL_PAIRS, _read_request(),
            properties=CP.same_rows(_ROWS))
        assert COL_SMALL in {p.collection_id for p, _d in wide}
        assert COL_SMALL not in {p.collection_id for p, _d in narrow}
        assert {p.collection_id for p, _d in narrow} < {p.collection_id for p, _d in wide}

    def test_equal_length_records_are_compared_on_every_mode_either_of_them_has(self, stack):
        """Why equality is the narrowing and not a band: with equal `rows` the common prefix
        truncates nothing, so the attractor cannot arise rather than being corrected for."""
        q = stack["digests"][COL_QUERY]
        for props, digest in stack["narrower"].candidates(
                q, ALL_PAIRS, _read_request(), properties=CP.same_rows(q.rows)):
            a, b = _common_prefix(q.read, digest.read)
            assert a.size == b.size == len(q.read) == len(digest.read)

    # `test_the_distance_itself_was_not_touched` used to AST-inspect
    # `mantle.search.beacon.proximity.spectral_distance` here — no floor, no normalisation, no
    # weight added. That module now lives in `agience-entroptics`
    # (`src/entroptics/proximity.py`), reachable from mantle only through the
    # `read=`/`engine_id=`/`probe_factory=` seam, so the guard belongs in entroptics' own test
    # suite rather than this one.

    def test_a_property_narrowing_can_only_remove(self, stack):
        """Swept: for every predicate, the narrowed gallery is a subset of the wide one."""
        q = stack["digests"][COL_QUERY]
        wide = {p.collection_id for p, _d in
                stack["narrower"].candidates(q, ALL_PAIRS, _read_request())}
        for predicate in (CP.same_rows(_ROWS), CP.same_rows(4), CP.same_rows(999),
                          lambda props: props.collection_id != COL_A,
                          lambda props: False, lambda props: True):
            got = {p.collection_id for p, _d in
                   stack["narrower"].candidates(q, ALL_PAIRS, _read_request(),
                                                properties=predicate)}
            assert got <= wide, predicate

    def test_none_applies_no_narrowing_at_all(self, stack):
        q = stack["digests"][COL_QUERY]
        assert stack["narrower"].candidates(q, ALL_PAIRS, _read_request()) == \
            stack["narrower"].candidates(q, ALL_PAIRS, _read_request(),
                                         properties=lambda props: True)


# ---------------------------------------------------------------------------
# Composition: three cuts, one intersection
# ---------------------------------------------------------------------------


class TestTheThreeCutsMeetAsOneIntersection:
    def test_all_three_active_is_the_intersection_of_all_three(self, stack):
        alone = _scope(stack).artifact_ids
        by_filter = _scope(stack, filters="type:application/pdf").artifact_ids
        by_proximity = _scope(stack, query=COL_A, radius=0.0).artifact_ids
        both = _scope(stack, filters="type:application/pdf", query=COL_A,
                      radius=0.0).artifact_ids
        assert both == by_filter & by_proximity
        assert both <= alone

    def test_no_ordering_of_the_cuts_changes_the_answer(self, stack):
        """Intersection is commutative, and this is the check that the code implements an
        intersection rather than two sequential filters that could disagree."""
        a = _scope(stack, filters="type:application/pdf", query=COL_B, radius=1e9)
        b = _scope(stack, query=COL_B, radius=1e9, filters="type:application/pdf")
        assert a.artifact_ids == b.artifact_ids

    def test_absent_it_returns_the_resolvers_set_verbatim(self, stack):
        """`token_lookup=None` is the default and this narrowing rides that same slot."""
        assert _scope(stack).artifact_ids == frozenset(AUTHORIZED_ARTIFACTS)
        assert _scope(stack, lookup=None).artifact_ids == frozenset(AUTHORIZED_ARTIFACTS)

    def test_it_composes_through_the_same_slot_the_token_narrowing_uses(self, stack):
        """The structural claim behind "one place where narrowings meet": the compiled
        proximity lookup has the same call signature the resolver's `TokenLookup` names, so it
        needed no new parameter and no second meeting point."""
        lookup = _lookup(stack, COL_A, radius=0.0)
        assert callable(lookup)
        assert isinstance(lookup([(OWNER, COL_A)]), set)


# ---------------------------------------------------------------------------
# The query is the caller's, including its radius
# ---------------------------------------------------------------------------


class TestTheProbeShapeIsTheCallers:
    def test_exactly_one_of_radius_and_nearest_must_be_stated(self, stack):
        for kwargs in ({}, {"radius": 0.5, "nearest": 1}):
            with pytest.raises(CP.ProximityQueryError):
                _lookup(stack, COL_A, **kwargs)

    def test_neither_has_a_default(self, stack):
        """A default radius would be a number nobody derived, in the one module whose whole
        claim is that it contains none."""
        import inspect

        sig = inspect.signature(CP.CollectionProximityNarrower.lookup_for)
        assert sig.parameters["radius"].default is None
        assert sig.parameters["nearest"].default is None

    @pytest.mark.parametrize("kwargs", [{"radius": -1.0}, {"nearest": 0}, {"nearest": -3}])
    def test_a_malformed_probe_raises_rather_than_returning_nothing(self, stack, kwargs):
        """An empty answer means "nothing was near". A malformed query has not asked that."""
        with pytest.raises(CP.ProximityQueryError):
            _lookup(stack, COL_A, **kwargs)

    def test_nearest_is_exact_and_returns_the_collection_it_came_from(self, stack):
        ids = _raw(stack, COL_QUERY, pairs=[(OWNER, c) for c in (COL_A, COL_B, COL_SECRET)],
                   nearest=1)
        assert ids == set(_members(COL_SECRET))


# ---------------------------------------------------------------------------
# Comparability, determinism, and the storage slot
# ---------------------------------------------------------------------------


class TestTheRecordSaysWhatItWasReadBy:
    def test_a_digest_from_another_instrument_is_dropped_rather_than_compared(self, stack):
        """`proximity` is explicit that records digested against different references are not
        comparable. Re-digest, do not mix."""
        import dataclasses

        stale = dataclasses.replace(stack["digests"][COL_A], engine_id="beacon.mp.excess")
        stack["slot"].put(OWNER, stale, _write_request())
        got = {p.collection_id for p, _d in stack["narrower"].candidates(
            stack["digests"][COL_QUERY], ALL_PAIRS, _read_request())}
        assert COL_A not in got
        assert COL_SECRET in got

    def test_a_digest_from_another_frame_convention_is_dropped_too(self, stack):
        import dataclasses

        stale = dataclasses.replace(stack["digests"][COL_B], frame_id="counts.v0")
        stack["slot"].put(OWNER, stale, _write_request())
        got = {p.collection_id for p, _d in stack["narrower"].candidates(
            stack["digests"][COL_QUERY], ALL_PAIRS, _read_request())}
        assert COL_B not in got

    def test_the_digest_slot_cannot_collide_with_any_indexed_term(self, stack):
        """The slot is blinded from a term carrying a NUL byte and colons. The tokenizer emits
        `\\w+` runs, joined by a single space for bigrams, so no stem and no phrase can reach
        it."""
        from mantle.search.mantle.sse.blind_tokens import blind_token
        from mantle.search.mantle.sse.tokenizer import bigrams, tokenize

        key = stack["oracle"].derive_sse_key(OWNER, _read_request())
        slot = CP.digest_slot_token(key, COL_A)
        stems = tokenize(" ".join(_CORPUS[COL_A]) + " proximity digest col a")
        for term in list(stems) + bigrams(stems):
            for field in "tdgc":
                assert blind_token(key, field, term) != slot

    def test_a_digest_written_for_one_collection_does_not_open_as_another(self, stack):
        assert stack["slot"].get(OWNER, COL_A, _read_request()).collection_id == COL_A
        assert stack["slot"].get(OWNER, COL_B, _read_request()).collection_id == COL_B

    def test_the_answer_is_identical_on_repeat(self, stack):
        """Deterministic to the bit: no RNG in the read, a stable argsort in the probe."""
        first = _scope(stack, query=COL_A, radius=1e9)
        second = _scope(stack, query=COL_A, radius=1e9)
        assert first == second
        assert _raw(stack, COL_A, radius=1e9) == _raw(stack, COL_A, radius=1e9)

    def test_a_collection_with_no_digest_is_simply_not_a_candidate(self, stack):
        pairs = ALL_PAIRS + ((OWNER, "col-never-digested"),)
        got = {p.collection_id for p, _d in stack["narrower"].candidates(
            stack["digests"][COL_QUERY], pairs, _read_request())}
        assert "col-never-digested" not in got

    def test_an_empty_pair_list_yields_nothing_rather_than_raising(self, stack):
        assert _raw(stack, COL_A, pairs=(), radius=0.0) == set()
        assert _raw(stack, COL_A, pairs=[("", "")], radius=0.0) == set()

    def test_a_member_enumeration_that_fails_costs_recall_and_never_authority(self, stack):
        def _boom(collection_id):
            raise RuntimeError("lattice unavailable")

        narrower = CP.CollectionProximityNarrower(
            stack["oracle"], stack["postings"], _boom, probe_factory=_BruteForceProbe)
        assert narrower.ids_near(stack["digests"][COL_QUERY], ALL_PAIRS, _read_request(),
                                 radius=1e9) == set()
