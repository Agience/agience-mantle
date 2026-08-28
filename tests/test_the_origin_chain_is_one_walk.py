"""Where a grant can sit to reach an artifact — one walk, honouring attenuation each hop.

## Why this has its own tests

`origin_chain` is the single implementation of authorization traversal. `check_access` reads it to
decide whether a caller may touch an artifact, and `oracle.LightConeGrantVerifier` reads it to
decide whether to issue a key for a collection. Both used to walk separately, and the second walked
the other way — enumerating every descendant — which is why a collection with more members than
`edges_of` will return raised there while the first answered in milliseconds.

It answers WHERE to look, not whether the answer is yes. A user's grants are a ledger lookup and a
grant key's are a bundle resolved at authentication, so the caller supplies its own notion of
"holds a grant on this resource" and the walk supplies the resources, nearest first.

## Attenuation

Each containment edge carries a propagate mask. A grant above an edge whose mask does not carry the
action does not reach through it, and the walk stops there — the same rule
`list_origin_descendants` applies to prune a subtree on the way down, so the two directions cannot
disagree about which edges conduct.
"""
from __future__ import annotations

import pytest

from mantle.db import open_lattice
from mantle.db.lattice_api import OriginChainUnterminated, origin_chain


@pytest.fixture()
def lattice(tmp_path):
    return open_lattice(str(tmp_path / "lattice.db"), origin="test")


def _put(lattice, aid, **doc):
    lattice.artifacts.put_artifact({"id": aid, **doc})
    return aid


def _contain(lattice, parent, child, *, propagate=None, is_origin=True):
    """A containment edge as `add_artifact_to_collection` writes one.

    The mask goes through `_ser_propagate`, which is what the store does: a list is stored as JSON
    and NULL is stored as NULL. Handing the column a raw list writes something `_prop_mask` cannot
    read back, so the double has to serialise it the same way the writer does.
    """
    from mantle.db.lattice_api import _ser_propagate

    lattice.graph.add_edge(parent, child, "contains",
                           {"root_id": child, "is_origin": bool(is_origin),
                            "propagate": _ser_propagate(propagate)})


def test_the_artifact_comes_first(lattice):
    """A grant on the artifact itself is the nearest place one can sit, so it is asked first."""
    _put(lattice, "a")
    assert list(origin_chain(lattice, "a", "read")) == ["a"]


def test_the_root_is_asked_before_any_ancestor(lattice):
    """Grants are held on roots, so a version's lineage is nearer than its container."""
    _put(lattice, "root")
    _put(lattice, "v2", root_id="root")
    assert list(origin_chain(lattice, "v2", "read", root_id="root")) == ["v2", "root"]


def test_the_walk_climbs_to_every_container(lattice):
    _put(lattice, "top")
    _put(lattice, "mid")
    _put(lattice, "leaf")
    _contain(lattice, "top", "mid")
    _contain(lattice, "mid", "leaf")

    assert list(origin_chain(lattice, "leaf", "read")) == ["leaf", "mid", "top"]


def test_an_edge_that_does_not_carry_the_action_stops_the_walk(lattice):
    """The attenuation. `mid` conducts `read` but not `update`, so a grant on `top` reaches the
    leaf for one action and not the other — from the same edges."""
    _put(lattice, "top")
    _put(lattice, "mid")
    _put(lattice, "leaf")
    _contain(lattice, "top", "mid")
    _contain(lattice, "mid", "leaf", propagate=["read"])

    assert list(origin_chain(lattice, "leaf", "read")) == ["leaf", "mid", "top"]

    # `mid` itself drops out for `update`, and that is the point rather than an off-by-one: the
    # edge that does not carry the action is the one BETWEEN `mid` and `leaf`, so a grant on `mid`
    # does not reach `leaf` either. `check_access` broke before consulting the parent for the same
    # reason. The walk stops at the edge, not past it.
    assert list(origin_chain(lattice, "leaf", "update")) == ["leaf"], (
        "an edge that does not carry `update` must stop the walk AT it — the resource on the far "
        "side of that edge cannot reach through it")


def test_a_null_mask_carries_everything(lattice):
    """NULL is 'unrestricted', which is what every edge the API writes carries."""
    _put(lattice, "top")
    _put(lattice, "leaf")
    _contain(lattice, "top", "leaf", propagate=None)

    for action in ("read", "update", "delete", "invoke"):
        assert list(origin_chain(lattice, "leaf", action)) == ["leaf", "top"]


def test_an_edge_that_is_not_the_origin_is_not_climbed(lattice):
    """Secondary membership — the same artifact appearing in another container — does not confer
    a second parent. `is_origin` marks the one canonical one, and it is the only one authority
    travels down."""
    _put(lattice, "elsewhere")
    _put(lattice, "leaf")
    _contain(lattice, "elsewhere", "leaf", is_origin=False)

    assert list(origin_chain(lattice, "leaf", "read")) == ["leaf"]


def test_a_cycle_does_not_repeat_a_resource(lattice):
    """A malformed lattice must not make the walk ask the same question forever."""
    _put(lattice, "a")
    _put(lattice, "b")
    _contain(lattice, "a", "b")
    _contain(lattice, "b", "a")

    chain = list(origin_chain(lattice, "a", "read"))
    assert chain == ["a", "b"], chain


def test_a_chain_past_its_ceiling_raises_rather_than_truncating(lattice):
    """A partial chain answers a NARROWER authorization question than the one asked, and nothing
    above it can tell that it was cut short."""
    for i in range(6):
        _put(lattice, "n%d" % i)
    for i in range(5):
        _contain(lattice, "n%d" % (i + 1), "n%d" % i)

    with pytest.raises(OriginChainUnterminated):
        list(origin_chain(lattice, "n0", "read", ceiling=3))


def test_an_artifact_that_is_not_here_still_yields_itself(lattice):
    """A grant can name a resource this store has never seen — the walk reports what it can reach
    rather than refusing to start."""
    assert list(origin_chain(lattice, "never-created", "read")) == ["never-created"]


def test_the_root_is_read_when_the_caller_does_not_supply_one(lattice):
    """Callers pass the root because they already hold the document. The fallback exists for a
    caller holding only an id, and it must find the same chain."""
    _put(lattice, "root")
    _put(lattice, "v2", root_id="root")

    assert list(origin_chain(lattice, "v2", "read")) == ["v2", "root"]
