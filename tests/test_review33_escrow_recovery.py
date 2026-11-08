"""Review 33, F1 (prepared, pending John's decision): recovery mode (offline, with the escrow
identity) combines this node's share with the escrow share and does not ask Origin at all, so it
runs with Origin down and for a split charged root whose grant has ended, where Origin mints no
delegation. Outside recovery, a refusal from Origin still never falls through to escrow."""
from __future__ import annotations

from . import test_split_master_keys_migration as m

node = m.node
_as_the_platform = m._as_the_platform


def test_recovery_with_origin_down_and_no_delegation_combines_from_escrow(node):
    m._split(node)
    node.origin.down = True          # Origin cannot answer: no release...
    node.origin.background = False   # ...and no delegation either
    counts = m._combine(node, node.make(recovery_identity=node.identity))
    assert counts["combined"] == 3 and counts["failed"] == 0
    assert m._whole_keys(node) == m.KEYS


def test_recovery_where_origin_refuses_the_delegation_combines_from_escrow(node):
    m._split(node)
    node.origin.background = False   # a charged root after activation: no live grant
    counts = m._combine(node, node.make(recovery_identity=node.identity))
    assert counts["combined"] == 3 and counts["failed"] == 0
    assert m._whole_keys(node) == m.KEYS


def test_recovery_asks_origin_nothing(node):
    m._split(node)
    asked = []
    node.origin.get_share = lambda *a, **k: asked.append("share")
    node.origin.background_delegation = lambda *a, **k: asked.append("mint")
    assert m._combine(node, node.make(recovery_identity=node.identity))["combined"] == 3
    assert asked == []


def test_outside_recovery_a_refusal_never_reaches_escrow(node):
    m._split(node)
    node.origin.background = False
    counts = m._combine(node)
    assert counts["combined"] == 0 and counts["failed"] == 3
