"""Blind-token narrowing: a token set narrows a light cone, and can do nothing else.

What this file pins
===================

:mod:`mantle.search.mantle.sse.narrowing` answers MEMBERSHIP off the encrypted posting
lists — which artifacts carry these stems — and, as a by-product of the same lookups, COVERAGE:
how many of the query's stems each one carried. This file is the proof of both, and of the
line between them: the first decides what comes back, the second decides only what order.

Four claims, in order of how much they matter:

1. **A token match can only narrow.** `resolve_authorized_scope` meets the lookup's answer
   into `artifact_ids` with `&`, so an id the light cone did not authorize cannot survive.
   The narrower genuinely returns such ids here — `art-secret` is in the same collection,
   under the same owner key, carrying the same terms — so the meet is doing visible work
   rather than being handed a subset already.

2. **The narrowing is not an existence oracle.** A token naming content outside the light
   cone and a token naming nothing must be indistinguishable in EVERY observable. This is
   the counterpart of `test_field_filters_narrow_recall.py`'s
   `test_that_answer_is_identical_to_a_filter_matching_nothing`, and the same argument: the
   two answers have to leave by the same door.

3. **The COUNTS are not an existence oracle either.** Making the order depend on match counts
   adds a second observable to the one claim 2 covers, and a count that moved because an
   unreadable artifact matched would report that artifact's existence just as surely as an id
   would. `TestTheCountsAreNotAnExistenceOracleEITHER` measures that it does not — including
   against a corpus built without the secret at all, which is what would catch a count computed
   over the index rather than over one artifact.

4. **Absent, it changes nothing.** `token_lookup=None` is the default and returns the
   resolver's set verbatim. The 75 tests in `test_field_filters_narrow_recall.py` are the
   real proof of that; the test here states it directly for the same inputs.

The stack is the production one. Only the lattice and the light cone are doubles, as in
`test_field_filters_narrow_recall.py`: the index is a real `SseIndexer` over real encrypted
posting lists, the keys come from a real `OracleService`, and the narrower reads what the
indexer wrote. No numpy and no vector arm — membership needs neither, which is the point.

The corpus is built so all three cuts are necessary. Within Alice's light cone the field
filter `type:application/pdf` selects {pdf, note} and the token `catalogue` selects
{pdf, image}; only their meet is {pdf}. A corpus where any one cut already gave the answer
would let the other two be broken without the suite noticing.
"""

from __future__ import annotations

import itertools
from typing import Optional

import pytest
from cryptography.fernet import Fernet

from mantle.search.field_filters import compile_filters
from mantle.search.mantle.lightcone import resolve_authorized_scope
from mantle.search.mantle.oracle import (
    FernetMasterKeyStore,
    KeyPurpose,
    KeyRequest,
    OracleService,
)
from mantle.search.mantle.sse.indexer import SseIndexer
from mantle.search.mantle.sse.narrowing import Coverage, TokenNarrower, phrase_stems
from mantle.search.mantle.sse.posting import InMemoryPostingStore
from mantle.search.query_parser import parse_query

ALICE = "user-alice"
COLLECTION = "col-1"
CELL_PRINCIPAL = COLLECTION          # self-rooted, as in the production index path

PDF = "art-pdf"
NOTE = "art-note"
IMAGE = "art-image"
#: Indexed, in the same collection, under the same owner key, carrying every term the
#: authorized artifacts carry — and NOT in Alice's light cone. The security property.
SECRET = "art-secret"

#: In every artifact including the secret, so narrowing on it narrows nothing and the meet is
#: what removes the secret.
TERM = "quasar"
#: In `art-secret` alone. A real token for real, indexed, unreadable content.
SECRET_TERM = "acquisition"
#: In no artifact at all.
ABSENT_TERM = "zzznosuchword"

_TEXT = {
    PDF:    "quasar quarterly budget catalogue",
    NOTE:   "quasar loose budget entries quarterly review",
    IMAGE:  "quasar catalogue cover plate",
    SECRET: "quasar catalogue acquisition record",
}

_CONTENT_TYPE = {
    PDF:    "application/pdf",
    NOTE:   "application/pdf",
    IMAGE:  "image/png",
    SECRET: "application/pdf",
}

AUTHORIZED = {PDF, NOTE, IMAGE}
#: `type:application/pdf` ∩ light cone.
PDFS_ALICE_CAN_READ = {PDF, NOTE}
#: token `catalogue` ∩ light cone.
CATALOGUE_ALICE_CAN_READ = {PDF, IMAGE}


# ---------------------------------------------------------------------------
# Doubles: the lattice and the light cone only
# ---------------------------------------------------------------------------


def _act(principal_id: str, principal_type: str = "user") -> None:
    from mantle.services.acting_principal import ActingPrincipal, set_acting_principal

    set_acting_principal(
        ActingPrincipal(principal_id=principal_id, principal_type=principal_type,
                        source="token-narrowing-test")
    )


class _AliceHoldsTheCollectionKey:
    """Key custody, decided the way `LightConeGrantVerifier` decides it — per collection.

    Alice legitimately holds this key, and that is the interesting case: everything below
    happens strictly downstream of correct custody. She can open the very posting lists that
    contain the secret's entries, so nothing about the secret is hidden by a key she lacks.
    """

    def authorized(self, *, requester_id, requester_type, principal_id,
                   collection_id, action) -> bool:
        if requester_id == principal_id:
            return True
        return (requester_id == ALICE and principal_id == CELL_PRINCIPAL
                and collection_id in (None, COLLECTION))


class _FakeArtifacts:
    def __init__(self, docs: dict) -> None:
        self._docs = docs

    def get_artifact(self, artifact_id: str) -> Optional[dict]:
        return self._docs.get(artifact_id)


class _FakeGraph:
    def edges_of(self, node, label=None, direction="out", limit=None) -> list:
        return []


def _doc(artifact_id: str) -> dict:
    return {
        "id": artifact_id,
        "root_id": artifact_id,
        "collection_id": COLLECTION,
        "content_type": _CONTENT_TYPE[artifact_id],
        "created_by": "user-bob",
        "created_time": "2025-01-01T00:00:00Z",
        "modified_time": "2025-01-01T00:00:00Z",
        "state": "committed",
        "context": {"title": artifact_id, "tags": [], "description": "a filed artifact"},
        "content": _TEXT[artifact_id],
    }


class _FakeStoreDB:
    def __init__(self) -> None:
        docs = {a: _doc(a) for a in _TEXT}
        docs[COLLECTION] = {"id": COLLECTION, "root_id": COLLECTION, "context": {}}
        self.artifacts = _FakeArtifacts(docs)
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
    def __init__(self, authorized: set) -> None:
        self._authorized = set(authorized)

    def resolve(self, principal_id, action="read", *, principal_type="user") -> set:
        return set(self._authorized) if principal_id == ALICE else set()

    def _grants_for(self, principal_id: str, principal_type: str = "user"):
        """The same reach this double already describes, expressed as grants.

        `resolve_authorized_scope` authorizes each candidate by walking up from it and testing the
        resources on the way against the caller's grants, so a double saying "these ids are
        authorized" says it by granting them — each id its own granted resource, which is what an
        artifact-scoped grant looks like. Derived from `resolve` rather than restated, so whatever
        that method gates on carries here too.
        """
        return [_GrantedResource(a) for a in self.resolve(principal_id,
                                                          principal_type=principal_type)]


# ---------------------------------------------------------------------------
# The real index
# ---------------------------------------------------------------------------


def _write_request():
    """A write: a read action never mints a master key, so the corpus must be indexed under
    one or every lookup below would see an empty index and pass for the wrong reason."""
    _act(CELL_PRINCIPAL, "principal")
    return KeyRequest(requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
                      requester_type="principal", action="update")


def _read_request():
    """Alice's own read request — the one `router_accessor._key_request` builds."""
    _act(ALICE, "user")
    return KeyRequest(requester_id=ALICE, purpose=KeyPurpose.GRANT,
                      requester_type="user", action="read")


@pytest.fixture
def stack():
    oracle = OracleService(
        FernetMasterKeyStore(Fernet(Fernet.generate_key())),
        grant_verifier=_AliceHoldsTheCollectionKey(),
    )
    postings = InMemoryPostingStore()
    indexer = SseIndexer(oracle, postings)
    for artifact_id, text in _TEXT.items():
        indexer.index_artifact(
            CELL_PRINCIPAL, COLLECTION, artifact_id, {"content": text},
            _write_request(),
        )
    return {"oracle": oracle, "postings": postings,
            "narrower": TokenNarrower(oracle, postings)}


def _scope(stack, *, authorized=AUTHORIZED, query=None, filters=None, lookup=None):
    """`resolve_authorized_scope` with whichever cuts this case is exercising."""
    if lookup is None and query is not None:
        lookup = stack["narrower"].lookup_for(query, _read_request())
    predicate = None
    if filters is not None:
        predicate = compile_filters(parse_query(filters).filters)
    _act(ALICE, "user")
    return resolve_authorized_scope(
        _FakeStoreDB(), ALICE,
        lightcone=_FakeLightCone(authorized),
        artifact_predicate=predicate,
        token_lookup=lookup,
    )


def _cover(stack, query, *, contexts=((CELL_PRINCIPAL, COLLECTION),)):
    """The narrower's OWN answer, un-met: ``{artifact_id: Coverage}``."""
    lookup = stack["narrower"].lookup_for(query, _read_request())
    return {} if lookup is None else dict(lookup(list(contexts)))


def _raw(stack, query, *, contexts=((CELL_PRINCIPAL, COLLECTION),)):
    """Just the ids of that answer — what the light cone is protecting against.

    Iterating the mapping is iterating its keys, and its keys ARE the membership answer; the
    counts beside them decide order and never membership.
    """
    return set(_cover(stack, query, contexts=contexts))


# ---------------------------------------------------------------------------
# The corpus is real. Without these, everything below could pass vacuously.
# ---------------------------------------------------------------------------


class TestTheIndexWouldOtherwiseReturnEverything:
    def test_the_narrower_reaches_every_artifact_carrying_the_term(self, stack):
        assert _raw(stack, TERM) == {PDF, NOTE, IMAGE, SECRET}, (
            "every artifact carries the term, including the one outside the light cone; if it "
            "did not, the meet below would be narrowing a set that was already narrow"
        )

    def test_the_secret_is_indexed_and_reachable_with_the_key_alice_holds(self, stack):
        """Not an absent document, and not one behind a key she lacks.

        Alice can open the very posting list the secret's entry sits in — she holds the
        collection key — so the only thing keeping it out of her results is the meet.
        """
        assert _raw(stack, SECRET_TERM) == {SECRET}

    def test_a_token_in_nothing_reaches_nothing(self, stack):
        assert _raw(stack, ABSENT_TERM) == set()

    def test_the_three_cuts_each_remove_something_the_others_keep(self, stack):
        """The corpus property the composition test rests on."""
        assert _scope(stack).artifact_ids == frozenset(AUTHORIZED)
        assert _scope(stack, filters="type:application/pdf").artifact_ids == frozenset(
            PDFS_ALICE_CAN_READ)
        assert _scope(stack, query="catalogue").artifact_ids == frozenset(
            CATALOGUE_ALICE_CAN_READ)
        assert PDFS_ALICE_CAN_READ != CATALOGUE_ALICE_CAN_READ != AUTHORIZED


# ---------------------------------------------------------------------------
# THE SECURITY PROPERTY
# ---------------------------------------------------------------------------


class TestATokenOnlyEverNarrows:
    def test_a_token_naming_an_unreadable_artifact_matches_nothing(self, stack):
        assert _scope(stack, query=SECRET_TERM).artifact_ids == frozenset()

    def test_that_answer_is_identical_to_a_token_matching_nothing(self, stack):
        """The property, stated as an indistinguishability.

        `acquisition` occurs in a real, indexed artifact Alice may not read — one whose
        posting list she can and does decrypt. `zzznosuchword` occurs nowhere. If the two
        answers differed in ANY observable — the artifact ids, the contexts, the type of the
        result, or an exception raised in one case and not the other — the narrowing would be
        an oracle for the existence of content outside the light cone.

        `test_the_secret_is_indexed_and_reachable_with_the_key_alice_holds` above is the
        other half: the index really can tell these two tokens apart. Only the meet cannot.
        """
        unreadable = _scope(stack, query=SECRET_TERM)
        nonexistent = _scope(stack, query=ABSENT_TERM)

        assert unreadable == nonexistent
        assert unreadable.artifact_ids == nonexistent.artifact_ids == frozenset()
        assert unreadable.contexts == nonexistent.contexts == [(CELL_PRINCIPAL, COLLECTION)]

    def test_the_difference_the_index_can_see_is_the_difference_the_meet_erases(self, stack):
        """Stated as one assertion so the two halves cannot drift apart."""
        assert _raw(stack, SECRET_TERM) != _raw(stack, ABSENT_TERM)
        assert _scope(stack, query=SECRET_TERM) == _scope(stack, query=ABSENT_TERM)

    def test_a_token_cannot_widen_past_the_light_cone(self, stack):
        """A token TRUE of the secret and of nothing Alice holds still returns nothing.
        There is no query text that adds an id the resolver did not."""
        assert _scope(stack, authorized={NOTE}, query="catalogue").artifact_ids == frozenset()

    def test_a_narrowed_scope_is_always_a_subset_of_the_unnarrowed_one(self, stack):
        """Swept over every query in the corpus rather than asserted on one case."""
        unnarrowed = _scope(stack).artifact_ids
        for text in (TERM, SECRET_TERM, ABSENT_TERM, "catalogue", "budget",
                     "quarterly", '"quarterly budget"', "catalogue acquisition"):
            assert _scope(stack, query=text).artifact_ids <= unnarrowed, text

    def test_a_lookup_that_returns_the_whole_store_still_narrows(self, stack):
        """The meet does not trust its input. A lookup claiming every id in existence —
        including ids no light cone produced — cannot add one."""
        everything = set(_TEXT) | {"art-does-not-exist", "art-someone-elses"}
        scope = _scope(stack, lookup=lambda pairs: everything)
        assert scope.artifact_ids == frozenset(AUTHORIZED)

    def test_a_lookup_that_raises_narrows_to_nothing(self, stack):
        """Fail closed for recall, and never a way to make the resolve fail. A lookup that
        raises has proved nothing about any artifact, so it narrows to nothing rather than
        to everything."""
        def _boom(pairs):
            raise RuntimeError("store unavailable")

        scope = _scope(stack, lookup=_boom)
        assert scope.artifact_ids == frozenset()
        assert scope.contexts == [(CELL_PRINCIPAL, COLLECTION)]


# ---------------------------------------------------------------------------
# COMPOSITION — light cone ∩ field filter ∩ token set
# ---------------------------------------------------------------------------


class TestTheThreeCutsCompose:
    def test_all_three_active_is_the_intersection_of_all_three(self, stack):
        scope = _scope(stack, filters="type:application/pdf", query="catalogue")
        assert scope.artifact_ids == frozenset(
            AUTHORIZED & PDFS_ALICE_CAN_READ & CATALOGUE_ALICE_CAN_READ)
        assert scope.artifact_ids == frozenset({PDF})

    def test_no_ordering_of_the_three_changes_the_answer(self, stack):
        """Intersection is commutative, but only if all three cuts really are sets meeting
        one set — which is the claim. A cut applied as anything else (a re-query, a
        post-filter over a different universe) would not survive reordering."""
        cone = set(_scope(stack).artifact_ids)
        by_filter = set(_scope(stack, filters="type:application/pdf").artifact_ids)
        by_token = set(_scope(stack, query="catalogue").artifact_ids)
        together = set(
            _scope(stack, filters="type:application/pdf", query="catalogue").artifact_ids)

        for order in itertools.permutations((cone, by_filter, by_token)):
            folded = set(order[0])
            for cut in order[1:]:
                folded &= cut
            assert folded == together, order

    def test_the_contexts_are_untouched_by_either_cut(self, stack):
        """Both are recall answers; the pairs are a key-custody answer. Narrowing custody on
        what a caller searched for would make the key a function of the query."""
        plain = _scope(stack).contexts
        for kwargs in ({"filters": "type:application/pdf"},
                       {"query": "catalogue"},
                       {"filters": "type:application/pdf", "query": SECRET_TERM},
                       {"query": ABSENT_TERM}):
            assert _scope(stack, **kwargs).contexts == plain, kwargs


# ---------------------------------------------------------------------------
# THE BIGRAM GATE
# ---------------------------------------------------------------------------


class TestAQuotedPhraseRequiresAdjacency:
    """`indexer` writes a posting entry per adjacent stem pair, so adjacency is already a
    membership question. Both artifacts below carry both stems; only one carries the pair."""

    def test_both_artifacts_carry_both_stems(self, stack):
        assert {PDF, NOTE} <= _raw(stack, "quarterly")
        assert {PDF, NOTE} <= _raw(stack, "budget")

    def test_the_phrase_selects_only_the_one_where_they_are_adjacent(self, stack):
        assert _raw(stack, '"quarterly budget"') == {PDF}

    def test_the_same_words_unquoted_do_not_gate(self, stack):
        assert {PDF, NOTE} <= _raw(stack, "quarterly budget")

    def test_a_phrase_in_no_document_matches_nothing(self, stack):
        assert _raw(stack, '"budget quarterly"') == set()

    def test_a_phrase_crossing_into_the_secret_still_only_narrows(self, stack):
        """`catalogue acquisition` IS adjacent — in the artifact outside the light cone."""
        assert _raw(stack, '"catalogue acquisition"') == {SECRET}
        assert _scope(stack, query='"catalogue acquisition"').artifact_ids == frozenset()

    def test_quotes_are_read_the_same_way_the_ranker_reads_them(self, stack):
        """`phrase_stems` is the analysis both paths share. A narrowing that disagreed with
        the ranker about what a phrase IS would gate on one query and score another."""
        assert phrase_stems('"quarterly budget"') == (["quarterli", "budget"], True)
        assert phrase_stems("quarterly budget") == (["quarterli", "budget"], False)
        assert phrase_stems('""') == ([], False)


# ---------------------------------------------------------------------------
# OFF THE LIVE PATH — absent, it changes nothing
# ---------------------------------------------------------------------------


class TestTheCallbackDefaultsToAbsent:
    def test_an_omitted_callback_and_an_explicit_none_are_one_answer(self, stack):
        db, cone = _FakeStoreDB(), _FakeLightCone(AUTHORIZED)
        _act(ALICE, "user")
        omitted = resolve_authorized_scope(db, ALICE, lightcone=cone)
        _act(ALICE, "user")
        explicit = resolve_authorized_scope(db, ALICE, lightcone=cone, token_lookup=None)

        assert omitted == explicit
        assert omitted.artifact_ids == frozenset(AUTHORIZED)
        assert omitted.contexts == [(CELL_PRINCIPAL, COLLECTION)]

    def test_absent_it_returns_the_resolvers_set_verbatim(self, stack):
        """The behaviour-neutrality claim, for every combination of the OTHER cut."""
        db, cone = _FakeStoreDB(), _FakeLightCone(AUTHORIZED)
        pdf_only = compile_filters(parse_query("type:application/pdf").filters)
        for predicate, expected in ((None, AUTHORIZED), (pdf_only, PDFS_ALICE_CAN_READ)):
            _act(ALICE, "user")
            scope = resolve_authorized_scope(
                db, ALICE, lightcone=cone, artifact_predicate=predicate)
            assert scope.artifact_ids == frozenset(expected)
            assert scope.contexts == [(CELL_PRINCIPAL, COLLECTION)]

    def test_a_query_with_no_stems_compiles_to_no_callback(self, stack):
        """`None` is "narrow nothing", not "narrow to nothing". A query that analyzes to no
        stems has nothing to look up, and answering it with an empty set would be a silent
        shrug — the same distinction `artifact_predicate=None` draws."""
        narrower = stack["narrower"]
        for text in ("", "   ", '""', "-- ... --", None):
            assert narrower.lookup_for(text, _read_request()) is None, text

    def test_a_narrower_with_no_searchable_field_compiles_to_no_callback(self, stack):
        narrower = TokenNarrower(stack["oracle"], stack["postings"], fields=["nonesuch"])
        assert narrower.lookup_for(TERM, _read_request()) is None

    def test_the_lookup_is_called_once_with_the_resolvers_own_contexts(self, stack):
        """It is handed the pairs, not a caller's idea of them — which is what makes it read
        only indexes this principal already holds keys for."""
        seen = []

        def _spy(pairs):
            seen.append(list(pairs))
            return set(AUTHORIZED)

        scope = _scope(stack, lookup=_spy)
        assert seen == [[(CELL_PRINCIPAL, COLLECTION)]]
        assert scope.artifact_ids == frozenset(AUTHORIZED)


# ---------------------------------------------------------------------------
# THE NARROWER ITSELF
# ---------------------------------------------------------------------------


class TestTheLookupReadsOnlyWhatItIsGiven:
    def test_an_unauthorized_collection_yields_nothing(self, stack):
        """One posting list spans every collection its owner indexed the term in, so the
        collection cut is applied to ENTRIES and not merely to which blobs are opened."""
        assert _raw(stack, TERM, contexts=[(CELL_PRINCIPAL, "col-elsewhere")]) == set()

    def test_no_contexts_yields_nothing(self, stack):
        assert _raw(stack, TERM, contexts=[]) == set()

    def test_a_never_indexed_principal_yields_nothing_rather_than_raising(self, stack):
        """A principal never written to has no master key and therefore no index. That is an
        empty answer, not an error — the `MasterKeyMissing` catch is what makes it one, and it
        is the same catch the write path's provider raise is defined for."""
        _act("user-nobody", "principal")
        self_request = KeyRequest(requester_id="user-nobody", purpose=KeyPurpose.SELF,
                                  requester_type="principal", action="read")
        stems, _ = phrase_stems(TERM)
        assert stack["narrower"].ids_for_stems(
            stems, [("user-nobody", "col-nobody")], self_request) == {}

    def test_a_refused_key_narrows_to_nothing_at_the_resolver(self, stack):
        """A key the custodian refuses is not caught here — `sse/query` does not catch it
        either, and a second opinion about custody is exactly what this arm must not hold. It
        reaches `resolve_authorized_scope`, whose catch-all narrows to nothing, so a refusal
        costs recall and is still not distinguishable from a token matching nothing."""
        lookup = stack["narrower"].lookup_for(TERM, _read_request())
        with pytest.raises(Exception):
            lookup([("user-nobody", "col-nobody")])
        assert _scope(stack, lookup=lambda pairs: lookup(
            [("user-nobody", "col-nobody")])).artifact_ids == frozenset()

    def test_stems_are_met_by_union_by_default(self, stack):
        """The union is the candidate set BM25 scores today: a document carrying one query
        term is a document the ranked path returns."""
        assert _raw(stack, "catalogue plate") == (
            _raw(stack, "catalogue") | _raw(stack, "plate"))

    def test_the_conjunction_is_available_and_is_a_subset_of_the_union(self, stack):
        all_stems = TokenNarrower(
            stack["oracle"], stack["postings"], require_all_stems=True)
        stems, _ = phrase_stems("catalogue plate")
        conjunction = all_stems.ids_for_stems(
            stems, [(CELL_PRINCIPAL, COLLECTION)], _read_request())
        assert set(conjunction) == {IMAGE}
        assert set(conjunction) < _raw(stack, "catalogue plate")

    def test_a_field_restriction_narrows_further(self, stack):
        """Only `content` was indexed here, so restricting to `title` finds nothing. A field
        is a place to look, not a weight — membership has no weights to read."""
        titles_only = TokenNarrower(stack["oracle"], stack["postings"], fields=["title"])
        stems, _ = phrase_stems(TERM)
        assert titles_only.ids_for_stems(
            stems, [(CELL_PRINCIPAL, COLLECTION)], _read_request()) == {}


# ---------------------------------------------------------------------------
# COVERAGE — the counts, and why they cannot leak
# ---------------------------------------------------------------------------


class TestCoverageCountsWhatTheNarrowingLookedUp:
    """The counts are a by-product of the membership pass, and say only what it found.

    They are an ORDINAL and not a metric: nothing here is normalised, weighted by how rare a
    term is, or scaled by how often it occurs or how long a field is. Each test below states
    which of those a count is NOT, by constructing the corpus condition that would move a
    weighted number and requiring that this one does not move.
    """

    def test_a_stem_count_is_the_number_of_distinct_query_stems_matched(self, stack):
        cover = _cover(stack, "catalogue plate")
        # `catalogue` is in PDF and IMAGE; `plate` is in IMAGE alone.
        assert cover[IMAGE].stems == 2
        assert cover[PDF].stems == 1

    def test_it_is_bounded_by_the_number_of_stems_in_the_query(self, stack):
        for text, n in ((TERM, 1), ("catalogue plate", 2),
                        ("quasar catalogue budget quarterly", 4)):
            cover = _cover(stack, text)
            assert cover, text
            assert all(0 < c.stems <= n for c in cover.values()), (text, cover)

    def test_repeating_a_term_in_the_DOCUMENT_does_not_raise_its_count(self, stack):
        """Term frequency is not in this number. `art-repeat` carries the term four times and
        `art-once` carries it once; a tf-weighted score would separate them."""
        indexer = SseIndexer(stack["oracle"], stack["postings"])
        indexer.index_artifact(CELL_PRINCIPAL, COLLECTION, "art-once",
                               {"content": "tantalum"}, _write_request())
        indexer.index_artifact(CELL_PRINCIPAL, COLLECTION, "art-repeat",
                               {"content": "tantalum tantalum tantalum tantalum"},
                               _write_request())
        cover = _cover(stack, "tantalum")
        # The COUNTS, not the whole tuple: `Coverage` also names WHICH stems matched, and this
        # test is about term frequency not raising a count. Both artifacts match the same single
        # stem, so `matched` is equal here too — but asserting it would restate the fixture
        # rather than the claim.
        assert ((cover["art-once"].stems, cover["art-once"].bigrams)
                == (cover["art-repeat"].stems, cover["art-repeat"].bigrams)
                == (1, 0))

    def test_a_rare_term_and_a_common_one_count_the_same(self, stack):
        """Document frequency is not in this number either. `quasar` is in every artifact and
        `plate` is in one; an IDF-weighted score would rank the rare one far above the common
        one, and here a match is a match."""
        assert _cover(stack, TERM)[IMAGE].stems == _cover(stack, "plate")[IMAGE].stems == 1

    def test_a_quoted_phrase_carries_its_bigram_count(self, stack):
        cover = _cover(stack, '"quarterly budget"')
        # Counts only: `Coverage` also names the stems, and this asserts the BIGRAM count.
        assert set(cover) == {PDF}
        assert (cover[PDF].stems, cover[PDF].bigrams) == (2, 1)

    def test_an_unquoted_query_issues_no_bigram_lookups_and_counts_none(self, stack):
        """Stated so the zero is read as "not looked up" rather than "looked up and absent".
        `quarterly budget` IS adjacent in the PDF; the unquoted form does not ask."""
        assert all(c.bigrams == 0 for c in _cover(stack, "quarterly budget").values())

    def test_the_counts_never_change_which_artifacts_are_reached(self, stack):
        """Coverage rides beside membership, never into it. The key set of the map is exactly
        the set the narrowing returned before there were any counts to return."""
        for text in (TERM, "catalogue", "catalogue plate", '"quarterly budget"'):
            assert set(_cover(stack, text)) == _raw(stack, text), text


class TestTheCountsAreNotAnExistenceOracleEITHER:
    """The sharpest risk in making the order depend on match counts.

    The set-level property is `TestATokenOnlyEverNarrows` above: an unreadable artifact and a
    nonexistent one leave by the same door. A COUNT is a second observable, and it would be a
    new leak if any of it survived the meet — a response whose ordering, score or total moved
    because `art-secret` matched three stems would report that `art-secret` exists.

    It cannot, and the reason is structural rather than checked: the counts are read only for
    ids in the SURVIVING set (see `_by_coverage`), and that set is `authorized & reached`. An id
    the light cone refused is absent from it, so its count is never looked up. These tests are
    the measurement of that.
    """

    def test_the_secret_has_a_real_and_larger_count_in_the_raw_answer(self, stack):
        """Non-vacuity: there IS a difference for the meet to erase. `art-secret` matches both
        stems of `catalogue acquisition`; nothing Alice can read matches more than one."""
        raw = _cover(stack, "catalogue acquisition")
        assert raw[SECRET].stems == 2
        assert max(c.stems for a, c in raw.items() if a != SECRET) == 1

    def test_no_surviving_artifact_takes_a_count_from_the_secret(self, stack):
        """The counts are per-artifact, so the secret's cannot be attributed to another id."""
        scope = _scope(stack, query="catalogue acquisition")
        cover = _cover(stack, "catalogue acquisition")
        assert SECRET not in scope.artifact_ids
        assert all(cover[a].stems == 1 for a in scope.artifact_ids)

    def test_a_query_matching_only_the_secret_orders_nothing(self, stack):
        """Whatever count the secret earned, there is no surviving id to attach it to."""
        assert _scope(stack, query=SECRET_TERM).artifact_ids == frozenset()

    def test_the_surviving_counts_are_identical_with_and_without_the_secret_indexed(
        self, stack, tmp_path,
    ):
        """The strongest form: build the SAME corpus with the secret omitted entirely, and
        require every count Alice can observe to be byte-identical.

        If any coverage number were computed over the whole index rather than per artifact —
        a document frequency, a corpus size, an average — this is the test that would move.
        Nothing Alice can read may depend on whether `art-secret` was ever written.
        """
        without = OracleService(
            FernetMasterKeyStore(Fernet(Fernet.generate_key())),
            grant_verifier=_AliceHoldsTheCollectionKey(),
        )
        postings = InMemoryPostingStore()
        indexer = SseIndexer(without, postings)
        for artifact_id, text in _TEXT.items():
            if artifact_id == SECRET:
                continue
            indexer.index_artifact(CELL_PRINCIPAL, COLLECTION, artifact_id,
                                   {"content": text}, _write_request())
        no_secret = {"oracle": without, "postings": postings,
                     "narrower": TokenNarrower(without, postings)}

        for text in (TERM, "catalogue", "catalogue plate", "catalogue acquisition",
                     '"quarterly budget"', ABSENT_TERM):
            with_secret = {a: c for a, c in _cover(stack, text).items() if a in AUTHORIZED}
            assert with_secret == _cover(no_secret, text), text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class TestSalienceDecidesWhichStemsAreLookedUp:
    """The narrowing looks up the stems that carry the question, not every stem typed.

    The coverage ordering downstream counts stems WITHOUT weighting — deliberately, so that it
    stays an ordinal on the query rather than a metric on the data (`_by_coverage`). That leaves
    a function word worth exactly as much as the subject, and on the live corpus it decided
    answers: `recall("what is a glacier")` narrowed on `what`/`is`/`glacier`, canon documents
    titled "What this is (and is not)" matched two of the three, and they filled the reach arm's
    pool before any glacier synset — which matched one — could enter it.

    So the filter belongs above the count. `salient` is injected rather than imported because it
    is the corpus's own measure and lives behind the `match` seam, which this layer may not
    import.
    """

    def test_a_stem_the_measure_drops_is_never_looked_up(self, stack):
        """Only the kept stem's artifacts enter the narrowing.

        Keeping the SECRET's stem rather than the common one is what makes this discriminating:
        every artifact carries `TERM`, so a filter that kept only that would return the same four
        ids whether or not it ran. `SECRET_TERM` is in one artifact, so if the filter reaches the
        lookup the answer collapses to that one, and if it does not the unfiltered pair returns
        all four.
        """
        seen = []

        # The measure sees STEMS, not the words typed — `acquisition` reaches it as `acquisit`.
        # Deriving the expected pair with `phrase_stems` rather than writing it out keeps this
        # test from encoding the stemmer's output as a constant.
        subject_stems, _ = phrase_stems(TERM)
        secret_stems, _ = phrase_stems(SECRET_TERM)

        def only_the_secret(stems):
            seen.append(list(stems))
            return [s for s in stems if s in secret_stems]

        lookup = stack["narrower"].lookup_for(
            "%s %s" % (SECRET_TERM, TERM), _read_request(), salient=only_the_secret,
        )
        found = lookup(((CELL_PRINCIPAL, COLLECTION),))
        assert seen and set(seen[0]) == set(subject_stems) | set(secret_stems), (
            "the measure must see every stem before choosing which carry the question"
        )
        assert set(found) == {SECRET}, (
            "dropping the common stem must leave only the artifact carrying the kept one; "
            "all four coming back means the filter never reached the lookup"
        )

    def test_no_measure_keeps_every_stem(self, stack):
        """`salient=None` is 'not measurable here', and must behave exactly as before."""
        both = stack["narrower"].lookup_for(
            "%s %s" % (SECRET_TERM, TERM), _read_request(),
        )
        assert SECRET in both(((CELL_PRINCIPAL, COLLECTION),))

    def test_a_measure_that_raises_keeps_every_stem(self, stack):
        """A measure that cannot run measures nothing; it must not narrow the query to nothing."""
        def broken(_stems):
            raise RuntimeError("no index to measure against")

        lookup = stack["narrower"].lookup_for(
            "%s %s" % (SECRET_TERM, TERM), _read_request(), salient=broken,
        )
        assert SECRET in lookup(((CELL_PRINCIPAL, COLLECTION),))

    def test_a_measure_that_keeps_nothing_keeps_every_stem(self, stack):
        """An empty answer is 'nothing distinguishes anything', not 'look nothing up'."""
        lookup = stack["narrower"].lookup_for(
            "%s %s" % (SECRET_TERM, TERM), _read_request(), salient=lambda _s: [],
        )
        assert SECRET in lookup(((CELL_PRINCIPAL, COLLECTION),))

    def test_a_phrase_is_never_filtered(self, stack):
        """A phrase's gate is a membership question over consecutive bigrams.

        Dropping one of its stems would break an adjacency the caller explicitly asked for, and
        quoting the words IS the claim that they belong together — which is the one claim a
        salience measure is not entitled to overrule.
        """
        called = []
        stack["narrower"].lookup_for(
            '"%s %s"' % (SECRET_TERM, TERM), _read_request(),
            salient=lambda s: called.append(list(s)) or [],
        )
        assert called == [], "a phrase must not be handed to the measure at all"
