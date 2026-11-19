"""One call moves members just before another, and no other member's key changes."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mantle.db import lattice_api as api
from mantle.routers.artifacts_router import ReorderRequest


class _Container:
    """A membership held as {root: order_key}, read and written through the lattice seams."""

    def __init__(self, *roots):
        self.keys = {}
        prev = None
        for r in roots:
            prev = api.after_key(prev)
            self.keys[r] = prev

    def order(self):
        return [r for r, _ in sorted(self.keys.items(), key=lambda kv: kv[1])]

    def edges(self, _db, _cid):
        return [{"dst": r, "props": {"order_key": k}} for r, k in self.keys.items()]

    def find(self, _db, _cid, root):
        return {"dst": root} if root in self.keys else None

    def set_key(self, _db, _cid, root, key):
        if root not in self.keys:
            return False
        self.keys[root] = key
        return True

    def move(self, roots, before):
        with patch.object(api, "_membership_edges", self.edges), \
             patch.object(api, "_find_membership", self.find), \
             patch.object(api, "set_edge_order_key", self.set_key):
            return api.move_members_before(MagicMock(), "col", roots, before)


def test_one_member_moves_before_another_and_the_rest_keep_their_keys():
    c = _Container("a", "b", "c", "d")
    before = dict(c.keys)
    assert c.move(["d"], "b") == 1
    assert c.order() == ["a", "d", "b", "c"]
    assert {r: k for r, k in c.keys.items() if r != "d"} == {
        r: k for r, k in before.items() if r != "d"}


def test_a_group_moves_together_in_the_order_given():
    c = _Container("a", "b", "c", "d", "e")
    assert c.move(["e", "b"], "a") == 2
    assert c.order() == ["e", "b", "a", "c", "d"]


def test_no_target_moves_it_last():
    c = _Container("a", "b", "c")
    assert c.move(["a"], None) == 1
    assert c.order() == ["b", "c", "a"]


def test_a_target_outside_the_container_moves_nothing():
    c = _Container("a", "b")
    assert c.move(["a"], "stranger") == 0
    assert c.order() == ["a", "b"]


def test_a_stranger_in_the_moved_ids_is_not_counted():
    c = _Container("a", "b")
    assert c.move(["stranger", "b"], "a") == 1
    assert c.order() == ["b", "a"]


def test_the_request_takes_one_shape():
    assert ReorderRequest(move_ids=["x"], before_id="y").move_ids == ["x"]
    assert ReorderRequest(ordered_ids=["x", "y"]).ordered_ids == ["x", "y"]
    with pytest.raises(ValidationError):
        ReorderRequest()
    with pytest.raises(ValidationError):
        ReorderRequest(ordered_ids=["x"], move_ids=["x"])
    with pytest.raises(ValidationError):
        ReorderRequest(ordered_ids=["x"], before_id="y")
    with pytest.raises(ValidationError):
        ReorderRequest(move_ids=[])
