"""What a PARTIAL `ordered_ids` actually does to the members it does not name.

This pins current behaviour and verifies the sentence the route now publishes. It is not
asserting a fix — refusing a partial membership would be the stricter contract and is recorded as a
proposal, because unlike the all-or-nothing check (which refuses ids that could not be applied at
all) a partial membership DOES apply exactly what was asked. It is the unnamed remainder that is
unspecified.

MEASURED. `reorder_collection_artifacts` assigns monotonically increasing `order_key`s **only along
the ids given**, starting at `after_key(None)` — which is `"U"` — and touches nothing else. A member
that is not named keeps the key it already had, and the listing sorts by `order_key`, so the
arrangement is whatever merging the two key sets produces.

So a partial list does not mean *"move these and leave the rest alone"*, and it does not mean
*"these first, the rest after"*. It means *"give these a fresh ascending run and let the result
interleave"* — deterministic, and not predictable without reading the key arithmetic.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from mantle.db import lattice_api as api


def test_the_run_starts_at_U():
    """The documented starting point. If this moves, the route's docstring is wrong."""
    assert api.after_key(None) == "U"


def test_the_keys_are_strictly_increasing_along_the_sequence():
    a = api.after_key(None)
    b = api.after_key(a)
    c = api.after_key(b)
    assert a < b < c, (a, b, c)


def test_only_the_named_ids_are_touched():
    db = MagicMock()
    with patch.object(api, "set_edge_order_key", return_value=True) as setter:
        updated = api.reorder_collection_artifacts(db, "col-1", ["r-2", "r-5"])
    touched = [call.args[2] for call in setter.call_args_list]
    assert touched == ["r-2", "r-5"], (
        "reorder touched %r — a member the caller did not name had its order_key rewritten"
        % touched)
    assert updated == 2


def test_an_id_that_is_not_a_member_is_not_counted():
    """`set_edge_order_key` returns False when there is no membership edge; that is what the
    route's all-or-nothing check reads to refuse a reorder it could not apply whole."""
    db = MagicMock()
    with patch.object(api, "set_edge_order_key", side_effect=[True, False]):
        updated = api.reorder_collection_artifacts(db, "col-1", ["r-2", "not-a-member"])
    assert updated == 1


def test_an_empty_sequence_touches_nothing():
    db = MagicMock()
    with patch.object(api, "set_edge_order_key") as setter:
        assert api.reorder_collection_artifacts(db, "col-1", []) == 0
    setter.assert_not_called()
