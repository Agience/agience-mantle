"""An edge publishes only if BOTH its endpoints do.

Why this test exists [John's ruling, 2026-08-25]. `sync.py`'s leaf assembly filtered VERTICES by
`_is_replicated` and shipped EDGES unconditionally. Excluding a content type therefore sent the edge
and withheld its vertex, and the peer ended up holding a membership edge to an artifact that is not
there — which is exactly `contains_edges_to_missing_vertex`, the invariant
`agience-cloud/deploy/data_integrity_check.py` pins at 0. The rule would have manufactured, on every
peer, the defect the integrity gate exists to catch.

It was not hypothetical when this landed: measured on 71/home, **18 `contains` edges pointed at
`application/vnd.agience.shard-done+json` vertices**, a type already in `_OP_EXCLUDE`. Switching the
mesh on would have published 18 dangling edges on the first round.

The tests below pin the three reasons an endpoint is withheld, because they are three different
mistakes and only the first is the one the ruling was about.
"""
from __future__ import annotations

from mantle.mesh.sync import _OP_EXCLUDE, _withheld_endpoints


class _Vertices:
    """A vertex accessor with exactly the surface `_withheld_endpoints` uses."""

    def __init__(self, rows):
        self._rows = dict(rows)

    def get_many(self, ids):
        return {i: self._rows[i] for i in ids if i in self._rows}


_REPLICATED = "text/markdown"
_EXCLUDED = sorted(_OP_EXCLUDE)[0]


def _edge(f, t):
    return {"f": f, "t": t, "label": "contains", "props": None}


# ── reason 1: the content type does not replicate — the ruling's case ────────────────────────────
def test_an_edge_to_a_non_replicating_type_is_withheld():
    v = _Vertices({"col": {"content_type": _REPLICATED},
                   "op": {"content_type": _EXCLUDED}})
    held, exhaustive = _withheld_endpoints(v, [_edge("col", "op")], set())
    assert exhaustive
    assert held == {"op"}, "the excluded endpoint must be withheld, so its edge is dropped"


def test_an_edge_between_two_replicating_endpoints_is_kept():
    """The other direction, so the filter is not simply always-on."""
    v = _Vertices({"a": {"content_type": _REPLICATED},
                   "b": {"content_type": _REPLICATED}})
    held, exhaustive = _withheld_endpoints(v, [_edge("a", "b")], set())
    assert exhaustive
    assert held == set()


def test_the_source_side_is_checked_too_not_just_the_destination():
    """The measured case was destination-side; a container of an excluded type is the same defect."""
    v = _Vertices({"op": {"content_type": _EXCLUDED},
                   "b": {"content_type": _REPLICATED}})
    held, _ = _withheld_endpoints(v, [_edge("op", "b")], set())
    assert held == {"op"}


# ── reason 2: grant-gated — edges were leaking the EXISTENCE of withheld ids ─────────────────────
def test_an_edge_to_a_grant_gated_id_is_withheld():
    v = _Vertices({"a": {"content_type": _REPLICATED},
                   "secret": {"content_type": _REPLICATED}})
    held, _ = _withheld_endpoints(v, [_edge("a", "secret")], {"secret"})
    assert held == {"secret"}, "an edge naming a withheld id discloses that the id exists"


# ── reason 3: the endpoint is not in this store at all ──────────────────────────────────────────
def test_an_edge_to_an_absent_vertex_is_withheld():
    """Publishing an edge to a vertex we do not have propagates a dangling edge rather than
    creating one, which is no better."""
    v = _Vertices({"a": {"content_type": _REPLICATED}})
    held, _ = _withheld_endpoints(v, [_edge("a", "ghost")], set())
    assert held == {"ghost"}


# ── the contract: non-exhaustive makes the caller refuse, it never silently ships ────────────────
def test_it_reports_non_exhaustive_rather_than_guessing_when_it_cannot_look_up():
    """`_private_set` has the same contract and the publish path honours it by holding the leaf.
    Failing closed matters more here than anywhere: the alternative is shipping edges we could not
    verify."""
    held, exhaustive = _withheld_endpoints(None, [_edge("a", "b")], set())
    assert not exhaustive
    assert held == set()


def test_over_the_cap_is_non_exhaustive_rather_than_truncated():
    v = _Vertices({})
    erecs = [_edge("f%d" % i, "t%d" % i) for i in range(10)]
    held, exhaustive = _withheld_endpoints(v, erecs, set(), cap=5)
    assert not exhaustive, "a truncated lookup must never read as a resolved one"


def test_no_edges_is_exhaustive_and_empty():
    """An empty leaf is a real answer, not an unresolved one — it must not hold the publish."""
    held, exhaustive = _withheld_endpoints(None, [], set())
    assert exhaustive and held == set()
