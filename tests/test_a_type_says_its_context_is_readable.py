"""A type definition says whether its artifacts' context is readable configuration."""
from __future__ import annotations

from mantle.db import doc_boundary as B
from mantle.services import type_offers

ISSUER = "application/vnd.agience.issuer+json"
NOTE = "application/vnd.agience.note+json"
KIND = "application/vnd.agience.trusted-issuer+json"


def _keys(ct):
    return sorted(B.sealable_keys({"content_type": ct}, {"content_type": ct, "title": "t", "issuer": "x"}))


def test_a_flagged_type_seals_its_title_alone():
    type_offers.set_definitions({ISSUER: {"content_type": ISSUER, "readable_context": True}})
    assert type_offers.readable_context(ISSUER)
    assert _keys(ISSUER) == ["title"]


def test_an_unflagged_type_seals_its_whole_context():
    type_offers.set_definitions({})
    assert not type_offers.readable_context(ISSUER)
    assert _keys(ISSUER) == ["issuer", "title"]
    assert _keys(NOTE) == ["issuer", "title"]


def test_the_flag_is_inherited_and_a_nearer_type_may_withdraw_it():
    type_offers.set_definitions({
        ISSUER: {"content_type": ISSUER, "readable_context": True},
        KIND: {"content_type": KIND, "inherits": [ISSUER]},
        NOTE: {"content_type": NOTE, "inherits": [ISSUER], "readable_context": False},
    })
    assert _keys(KIND) == ["title"]
    assert _keys(NOTE) == ["issuer", "title"]


def test_a_type_definition_is_readable_before_any_definition_is_known():
    type_offers.set_definitions({})
    ct = "application/vnd.agience.type+json"
    assert _keys(ct) == ["title"]
