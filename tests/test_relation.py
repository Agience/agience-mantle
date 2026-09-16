"""Edge relations are observed, never enumerated — and the vocabulary is open.

These tests pin the property that matters: the store does not decide what kinds of relation exist.
A test that asserted a fixed set of kinds would be the forcing itself, written down and made
permanent, so what is asserted here is the absence of one.
"""
import mantle.entities.relation as relation_module
from mantle.entities.relation import derive_relation, observed_relation


def test_what_was_observed_is_returned_unchanged():
    """Whatever the writer recorded is what comes back. No mapping, no normalisation."""
    for word in ("operator", "reference", "cites", "rebuts", "was-measured-by", "小分類", "🜃"):
        assert observed_relation(origin=False, relationship=word) == word
        assert observed_relation(origin=True, relationship=word) == word


def test_the_vocabulary_is_open():
    """An unseen word is carried, not rejected and not mapped to a nearest member.

    This is the whole property. A closed vocabulary fails in both directions — a corpus whose
    relations do not fit has them flattened at write time, where nothing downstream can recover the
    distinction; a corpus with fewer kinds carries a vocabulary asserting structure it lacks.
    """
    invented = "a-relation-nobody-declared-in-advance"
    assert observed_relation(origin=False, relationship=invented) == invented


def test_nothing_observed_is_reported_as_nothing():
    """`None` is a fact about the edge, not a gap to be filled with a default."""
    assert observed_relation(origin=True, relationship=None) is None
    assert observed_relation(origin=False, relationship=None) is None


def test_containment_does_not_derive_a_kind():
    """`origin` is part of what was observed and is not used to classify.

    An edge's containment role is real and the store records it in its own column. Turning it into
    a relation *kind* would be this module classifying, and the same edge would then carry two
    disagreeing accounts of what it is.
    """
    assert observed_relation(origin=True, relationship=None) == observed_relation(
        origin=False, relationship=None
    )
    assert observed_relation(origin=True, relationship="cites") == observed_relation(
        origin=False, relationship="cites"
    )


def test_there_is_no_enumeration_to_import():
    """The closed vocabulary is gone, and its absence is asserted rather than assumed.

    `Relation` and `EDGE_RELATIONS` were a five-member enum and a two-member frozenset over it.
    Either one reappearing would restore a fixed set of possible relations, so their absence is
    checked here — a module attribute is easy to add back by habit, and this is what notices.
    """
    assert not hasattr(relation_module, "Relation")
    assert not hasattr(relation_module, "EDGE_RELATIONS")


def test_the_previous_name_still_resolves():
    """`derive_relation` is retained as an alias so existing imports keep working."""
    assert derive_relation is observed_relation
