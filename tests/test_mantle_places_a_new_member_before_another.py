"""Mantle computes where a new member sits: just before a named member, or at the end."""
from __future__ import annotations

import random
from unittest.mock import MagicMock, patch

from mantle.db import lattice_api as api


def _edges(*pairs):
    return [{"dst": dst, "props": {"order_key": key}} for key, dst in pairs]


def test_a_key_before_a_member_sorts_between_it_and_the_one_before():
    edges = _edges(("U", "r-1"), ("V", "r-2"), ("W", "r-3"))
    with patch.object(api, "_membership_edges", return_value=edges):
        key = api.key_before(MagicMock(), "col", "r-2")
    assert "U" < key < "V"


def test_a_key_before_the_first_member_sorts_first():
    edges = _edges(("U", "r-1"), ("V", "r-2"))
    with patch.object(api, "_membership_edges", return_value=edges):
        key = api.key_before(MagicMock(), "col", "r-1")
    assert key < "U"


def test_no_member_named_or_a_stranger_places_it_last():
    edges = _edges(("U", "r-1"), ("V", "r-2"))
    with patch.object(api, "_membership_edges", return_value=edges):
        assert api.key_before(MagicMock(), "col", None) > "V"
        assert api.key_before(MagicMock(), "col", "not-a-member") > "V"


def test_an_empty_collection_starts_at_U():
    with patch.object(api, "_membership_edges", return_value=[]):
        assert api.key_before(MagicMock(), "col", None) == "U"


def test_repeated_insertions_anywhere_keep_the_order():
    rng = random.Random(7)
    keys = ["U"]
    for _ in range(2000):
        i = rng.randint(0, len(keys))
        a = keys[i - 1] if i else None
        b = keys[i] if i < len(keys) else None
        m = api.mid_key(a, b)
        assert (a is None or a < m) and (b is None or m < b), (a, b, m)
        keys.insert(i, m)
