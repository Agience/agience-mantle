"""Edge relation typing — the information-centric `relation` kind on edges.

Phase 0 of the Information Gauge DB build. Pure-logic tests for `derive_relation`
(the mapping from an edge's existing signals to its relation kind) and the enum
contract; the live backfill is exercised at startup (`_backfill_edge_fields`).
"""
from mantle.entities.relation import EDGE_RELATIONS, Relation, derive_relation


def test_relation_vocabulary_is_information_centric():
    # No physics names; the five information-centric kinds, valence borrowed elsewhere.
    assert {r.value for r in Relation} == {
        "grant", "temporal", "semantic", "lifecycle", "derivation"
    }


def test_edges_collection_only_stores_grant_and_derivation():
    assert EDGE_RELATIONS == {"grant", "derivation"}


def test_operator_edges_are_derivations():
    assert derive_relation(origin=False, relationship="operator") == "derivation"
    assert derive_relation(origin=True, relationship="operator") == "derivation"


def test_containment_edges_are_grants():
    # origin containment and plain links both sit in the access/grant topology.
    assert derive_relation(origin=True, relationship=None) == "grant"
    assert derive_relation(origin=False, relationship=None) == "grant"
    assert derive_relation(origin=True, relationship="reference") == "grant"
