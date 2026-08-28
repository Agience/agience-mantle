"""The context lattice — nesting, composition, termination, the bound, and the ceiling (D16).

Context is an artifact with edges to its own context, and indexing, recall, propagation and
authorization are traversals of that one graph. Four properties carry the security content
and each has a test here that can fail:

1. **A context edge narrows; it is never a source of reach.** The walk may not leave the set
   of ids its caller already holds authority over, so unioning its result into that set
   cannot widen it — at any depth, under any mask, for any action. This is the one that
   matters most, and it is the one the first version of this walk got backwards: it unioned
   an unconfined walk into the resolver's answer, and a grant on `org` alone came back
   holding `{"org", "project", "doc-1"}`. §7 sweeps the invariant over grant/edge
   configurations rather than asserting it on the one case that was found.
2. **Authority strictly attenuates along a context chain.** A narrowing anywhere on the path
   survives to the end, and composition never widens.
3. **The recursion terminates.** A context's context is recursive, so a cycle must stop, and
   a deep or wide lattice must stop at a stated bound rather than run to exhaustion.
4. **Only the defining edge propagates, and only what it says propagates.** Naming someone
   else's artifact as sitting in your context confers nothing, exactly as a non-origin
   containment link confers nothing — and an edge written with no mask now confers nothing
   either.

`tests/test_attenuation_algebra.py` proves the operator over its whole domain and
`tests/test_attenuation_is_single_sourced.py` proves there is only one of it; this file is
about what the traversal does with it.

A note on `within=cs.UNCONFINED` below: most tests here are about the TRAVERSAL — what the
edges reach, what the masks prune, where the recursion stops — so they opt out of the
identity ceiling explicitly and read the structure. The ceiling itself is §7's subject. There
is no default for the argument, which is why every call has to say which it wants.
"""
from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest

from mantle.attenuation import ACTIONS, DENY, NOTHING, TOP, Mask
from mantle.db import backend as db_store
from mantle.db.edge import CONTEXT_LABEL, DEFAULT_CONTEXT_PROPAGATE
from mantle.db.lattice_api import LatticeDatabase
from mantle.entities.artifact import Artifact, COLLECTION_CONTENT_TYPE
from mantle.entities.context import (
    CONTEXT_CONTENT_TYPE,
    Context,
    is_context,
    is_context_node,
    new_context,
)
from mantle.entities.grant import Grant as GrantEntity, mask_of
from mantle.search.mantle import LightConeResolver
from mantle.services import context_service as cs


@pytest.fixture()
def db(tmp_path):
    """A real lattice. Edges and grants need no content key, so nothing is stubbed —
    the traversal runs against the store it will run against in production."""
    return LatticeDatabase(str(tmp_path / "ctx.db"), origin="test-context")


def _chain(db, *nodes, propagate=None, is_origin=True):
    """`a → b → c …` as context edges — each node is the context of the next.

    `propagate` is one value per hop, or None for an unrestricted chain — spelled `TOP` rather
    than left to the writer's default, because that default is the empty mask and a chain
    transmitting nothing would make every traversal test below pass for the wrong reason. Per-hop
    rather than one-for-all, because every interesting property here is about a chain whose hops
    differ.
    """
    hops = list(zip(nodes, nodes[1:]))
    masks = list(propagate) if propagate is not None else [TOP] * len(hops)
    assert len(masks) == len(hops), "give one propagate value per hop"
    for (src, dst), mask in zip(hops, masks):
        cs.set_context(db, dst, src, propagate=mask, is_origin=is_origin)


# ═════════════════════════════════════════════════════════════════════════════
# 1 · the entity — a role over the artifact, not a new kind of thing
# ═════════════════════════════════════════════════════════════════════════════

def test_a_context_is_an_artifact():
    """The alias is the design. If `Context` ever stops being `Artifact`, a context loses
    version lineage, provenance and grantability and becomes a second entity to keep in
    sync."""
    assert Context is Artifact
    assert CONTEXT_CONTENT_TYPE == "application/vnd.agience.context+json"


def test_new_context_declares_the_role_and_nothing_more():
    ctx = new_context("Acme", description="the org", created_by="u-1")
    assert ctx.content_type == CONTEXT_CONTENT_TYPE
    assert ctx.name == "Acme" and ctx.description == "the org"
    assert isinstance(ctx, Artifact)
    # Nesting is an edge, never a scalar field: putting it back on `collection_id` would put
    # the composable relation back into the handle this decision exists to replace.
    assert ctx.collection_id == ""


def test_the_two_context_predicates_answer_different_questions():
    ctx = new_context("Acme")
    coll = Artifact(id="c-1", content_type=COLLECTION_CONTENT_TYPE)

    assert is_context(ctx) and is_context_node(ctx)
    # A collection is ALREADY a context node — that is what makes this additive — but it has
    # not declared the role.
    assert is_context_node(coll) and not is_context(coll)
    assert not is_context(Artifact(id="a-1", content_type="text/markdown"))
    assert not is_context_node(Artifact(id="a-1", content_type="text/markdown"))


def test_the_predicates_read_every_shape_an_artifact_arrives_in():
    """Entity, raw lattice doc, and a bare content type. A predicate that worked on one of
    the three would be silently wrong on the other two."""
    for shape in (new_context("A"),
                  {"id": "x", "content_type": CONTEXT_CONTENT_TYPE},
                  CONTEXT_CONTENT_TYPE):
        assert is_context(shape), shape
    assert not is_context(None)
    assert not is_context({"id": "x"})


# ═════════════════════════════════════════════════════════════════════════════
# 2 · the edge — a distinct label on the existing store
# ═════════════════════════════════════════════════════════════════════════════

def test_a_context_edge_is_written_with_the_promoted_columns(db):
    cs.set_context(db, "project", "org", propagate=["read"], is_origin=True, order_key="U")
    rows = db.graph.context_edges("org", direction="out")
    assert len(rows) == 1
    row = rows[0]
    assert (row["src"], row["dst"], row["label"]) == ("org", "project", CONTEXT_LABEL)
    assert row["is_origin"] == 1
    assert row["order_key"] == "U"
    # Encoded exactly as `lattice_api._ser_propagate` writes the column, so one decoder reads
    # every edge in the store regardless of which writer produced it.
    assert Mask.from_propagate(row["propagate"]).actions == {"read"}


def test_an_edge_written_with_no_mask_propagates_nothing(db):
    """A `None` default decodes to `TOP`, i.e. *everything flows*.

    A default that confers maximum authority is the wrong default for a security-relevant edge:
    it makes the careless write the dangerous one. The empty mask is not a new
    encoding — `'[]'` is what `artifacts_router` already writes on a non-lineage link and
    what the one decoder already reads as `attenuation.NOTHING`, a permitted path that
    transmits nothing.
    """
    assert DEFAULT_CONTEXT_PROPAGATE == ()

    cs.set_context(db, "project", "org")                      # no mask stated
    row = db.graph.context_edges("org", direction="out")[0]
    assert row["propagate"] == "[]"
    assert Mask.from_propagate(row["propagate"]) is NOTHING

    # And it is not merely stored narrow — it walks narrow, at every action.
    for action in ACTIONS:
        assert cs.reach(db, ["org"], action, within=cs.UNCONFINED).ids == set(), action

    # The store's own writer defaults the same way, so the service and the store cannot
    # disagree about what an unstated mask means.
    db.graph.add_context_edge("org", "direct")
    direct = [e for e in db.graph.context_edges("org", direction="out")
              if e["dst"] == "direct"][0]
    assert Mask.from_propagate(direct["propagate"]) is NOTHING


def test_stating_the_mask_is_what_opens_the_edge(db):
    """The positive control for the test above: without it, every assertion there would pass
    against a writer that could not open an edge at all."""
    cs.set_context(db, "project", "org", propagate=["read"])
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED).ids == {"project"}
    assert cs.reach(db, ["org"], "update", within=cs.UNCONFINED).ids == set()

    cs.set_context(db, "wide", "org", propagate=None)         # the column's "unrestricted"
    assert cs.reach(db, ["org"], "admin", within=cs.UNCONFINED).ids == {"wide"}


def test_re_adding_a_context_edge_leaves_exactly_one_row(db):
    """Idempotency is the store's, via `edge_key = blake2b(src ‖ dst ‖ label)`. Asserted for
    this label because mesh segments are replayed by design and the guarantee is only useful
    if it holds for every label, not for the ones that happened to be tested."""
    before = db.graph.count_edges()
    for _ in range(4):
        cs.set_context(db, "project", "org", propagate=["read"])
    assert db.graph.count_edges() - before == 1
    assert len(db.graph.context_edges("org", direction="out")) == 1


def test_a_context_edge_and_a_containment_edge_coexist_on_one_pair(db):
    """Different label, different `edge_key`, different row — which is what makes the whole
    decision additive: no existing walk sees a context edge, because every existing walk
    names its label."""
    from mantle.db import lattice_api as api

    api.add_artifact_to_collection(db, "org", "project")
    cs.set_context(db, "project", "org")

    labels = {e["label"] for e in db.graph.edges_of("org", direction="out")}
    assert labels == {"contains", CONTEXT_LABEL}
    assert cs.members_of(db, "org") == ["project"]

    cs.clear_context(db, "project", "org")
    assert cs.members_of(db, "org") == []
    assert [e["label"] for e in db.graph.edges_of("org", direction="out")] == ["contains"]


def test_the_edge_reads_both_ways(db):
    cs.set_context(db, "project", "org")
    cs.set_context(db, "project", "eu-region", is_origin=False)

    assert cs.members_of(db, "org") == ["project"]
    assert sorted(cs.contexts_of(db, "project")) == ["eu-region", "org"]
    # `origin_only` is the authority-bearing subset; a reference is still navigable.
    assert cs.contexts_of(db, "project", origin_only=True) == ["org"]


def test_a_mask_can_be_handed_to_the_writer_directly(db):
    """`Mask.to_propagate()` is the single-sourced encoder; the writer accepts it so a caller
    holding an authority value never has to reach for a second encoding of it."""
    cs.set_context(db, "b", "a", propagate=Mask.of(("read", "update")))
    row = db.graph.context_edges("a", direction="out")[0]
    assert Mask.from_propagate(row["propagate"]).actions == {"read", "update"}

    cs.set_context(db, "c", "a", propagate=TOP)
    unrestricted = [e for e in db.graph.context_edges("a", direction="out")
                    if e["dst"] == "c"][0]
    assert unrestricted["propagate"] is None      # TOP is what a null column means
    assert Mask.from_propagate(unrestricted["propagate"]) is TOP


# ═════════════════════════════════════════════════════════════════════════════
# 3 · nesting and multi-parent
# ═════════════════════════════════════════════════════════════════════════════

def test_contexts_nest(db):
    _chain(db, "org", "project", "doc")
    got = cs.reach(db, ["org"], "read", within=cs.UNCONFINED)
    assert got.ids == {"project", "doc"}        # seeds excluded, same as the origin BFS
    assert not got.truncated and got.limit is None


def test_a_node_may_sit_in_several_contexts(db):
    cs.set_context(db, "doc", "project", propagate=TOP)
    cs.set_context(db, "doc", "legal", propagate=TOP)
    assert cs.reach(db, ["project"], "read", within=cs.UNCONFINED).ids == {"doc"}
    assert cs.reach(db, ["legal"], "read", within=cs.UNCONFINED).ids == {"doc"}
    assert sorted(cs.contexts_of(db, "doc")) == ["legal", "project"]


def test_two_parents_are_alternatives_not_co_requirements(db):
    """A join over paths of meets along paths. One blocked route must not close an open one —
    "attenuation" said carelessly suggests every extra parent narrows, and it does the
    opposite."""
    cs.set_context(db, "doc", "project", propagate=[])          # nothing passes
    cs.set_context(db, "doc", "legal", propagate=["read"])      # read passes
    assert cs.reach(db, ["project", "legal"], "read", within=cs.UNCONFINED).ids == {"doc"}
    assert cs.reach(db, ["project"], "read", within=cs.UNCONFINED).ids == set()


def test_reach_from_several_seeds_unions(db):
    _chain(db, "org-a", "team-a")
    _chain(db, "org-b", "team-b")
    got = cs.reach(db, ["org-a", "org-b"], "read", within=cs.UNCONFINED)
    assert got.ids == {"team-a", "team-b"}


def test_containment_and_context_compose(db):
    """The alternation is the point. A nested context's CONTENTS must be reached, not just the
    context node — otherwise the two lattices sit side by side instead of being one."""
    from mantle.db import lattice_api as api

    cs.set_context(db, "project", "org", propagate=TOP)
    api.add_artifact_to_collection(db, "project", "doc-1")      # containment below a context
    cs.set_context(db, "sub", "doc-1", propagate=TOP)           # and a context below that

    got = cs.reach(db, ["org"], "read", within=cs.UNCONFINED,
                   expand_containment=db_store.list_origin_descendants)
    assert got.ids == {"project", "doc-1", "sub"}

    # Without the expander the walk follows context edges alone, and says so by reaching less.
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED).ids == {"project"}


# ═════════════════════════════════════════════════════════════════════════════
# 4 · attenuation — the action ceiling
# ═════════════════════════════════════════════════════════════════════════════

def test_authority_attenuates_along_a_context_chain(db):
    """Every hop is a meet, so the composed authority is below every mask on the path."""
    _chain(db, "org", "project", "doc", propagate=[["read", "update"], ["read"]])
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED).ids == {"project", "doc"}
    # `update` reaches the project and stops: the second hop does not carry it.
    assert cs.reach(db, ["org"], "update", within=cs.UNCONFINED).ids == {"project"}
    assert cs.reach(db, ["org"], "delete", within=cs.UNCONFINED).ids == set()


def test_a_narrowing_anywhere_on_the_path_survives_to_the_end(db):
    """The regression this file exists for. A restrictive edge early in a chain cannot be
    re-widened by a permissive edge after it — composition only removes."""
    _chain(db, "a", "b", "c", "d", propagate=[["read"], TOP, TOP])
    assert cs.reach(db, ["a"], "read", within=cs.UNCONFINED).ids == {"b", "c", "d"}
    # Two unrestricted (TOP) hops follow the narrow one. If the meet were re-derived per edge
    # instead of composed along the path, `update` would reappear at `c`.
    assert cs.reach(db, ["a"], "update", within=cs.UNCONFINED).ids == set()


def test_an_empty_mask_absorbs_the_rest_of_the_path(db):
    """`propagate=[]` is a permitted path that transmits nothing — `attenuation.NOTHING`.
    Everything behind it is unreachable, at any depth."""
    _chain(db, "a", "b", propagate=[[]])
    _chain(db, "b", "c", "d")                     # unrestricted, but behind the closed edge
    for action in ("read", "update", "admin"):
        assert cs.reach(db, ["a"], action, within=cs.UNCONFINED).ids == set(), action
    assert NOTHING.allows("read") is False


def test_the_seed_ceiling_is_a_real_ceiling(db):
    """The authority a caller holds bounds the whole walk, because the first thing every path
    does is meet with it."""
    _chain(db, "org", "project")                  # unrestricted edges
    assert cs.reach(db, ["org"], "update", within=cs.UNCONFINED,
                    authority=Mask.of(("read",))).ids == set()
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED,
                    authority=Mask.of(("read",))).ids == {"project"}
    # DENY is the absorbing zero: it seeds nothing, whatever the edges say.
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED, authority=DENY).ids == set()


def test_composition_never_widens_for_any_action(db):
    """The law, swept over the whole action vocabulary rather than asserted on one verb: what
    the chain reaches for an action is exactly what EVERY edge on it allows."""
    _chain(db, "a", "b", "c", propagate=[["read", "update", "admin"], ["read", "update"]])
    path = [Mask.of(("read", "update", "admin")), Mask.of(("read", "update"))]
    for action in ACTIONS:
        expected = {"b", "c"} if all(m.allows(action) for m in path) else (
            {"b"} if path[0].allows(action) else set())
        assert cs.reach(db, ["a"], action, within=cs.UNCONFINED).ids == expected, action


def test_a_non_origin_context_edge_confers_nothing(db):
    """A reference, not an authority path. Without this, writing an edge that names someone
    else's artifact as sitting in your context would be a way to acquire authority over it —
    the exact reason the origin BFS refuses a non-origin containment link.

    `propagate=TOP` deliberately: `is_origin` must do this on its own, or the assertion would
    only be re-testing the writer's narrow default."""
    cs.set_context(db, "someone-elses-doc", "my-context", is_origin=False, propagate=TOP)
    assert cs.reach(db, ["my-context"], "read", within=cs.UNCONFINED).ids == set()
    # Still visible as structure — it is a real relation, it just carries no authority.
    assert cs.members_of(db, "my-context") == ["someone-elses-doc"]


def test_an_unknown_action_reaches_nothing_and_touches_no_store(db):
    """An unmapped verb is a denial, never a hole opened by a typo."""
    _chain(db, "org", "project")
    assert cs.reach(db, ["org"], "no-such-action", within=cs.UNCONFINED) == cs.EMPTY_REACH
    assert cs.reach(db, [], "read", within=cs.UNCONFINED) == cs.EMPTY_REACH


# ═════════════════════════════════════════════════════════════════════════════
# 5 · confinement — the IDENTITY ceiling, and the reason this module exists
# ═════════════════════════════════════════════════════════════════════════════

def test_within_is_required_and_has_no_default():
    """No call site gets a confinement decision by not making one. A default in either
    direction would be wrong: unconfined-by-default is the bug this file documents, and
    confined-by-default would silently answer a navigation question with an authorization
    one."""
    import inspect

    param = inspect.signature(cs.reach).parameters["within"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):
        cs.reach(object(), ["a"], "read")          # type: ignore[call-arg]


def test_a_context_edge_cannot_admit_a_node_outside_the_universe(db):
    """THE narrowing. `is_origin` and `propagate` guard the EDGE; this guards the
    DESTINATION, which is the thing being acquired. Without it, whoever can write an edge
    names any id at all as sitting in a context they hold and the walk confers authority
    over it."""
    _chain(db, "org", "project", "doc")            # unrestricted, origin, wide open

    # Unconfined, the walk reaches both — that is what the edges say structurally.
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED).ids == {"project", "doc"}

    # Confined, it admits only ids the universe already contains — the excluded one closes
    # the path and everything behind it with it.
    assert cs.reach(db, ["org"], "read", within={"org"}).ids == set()
    assert cs.reach(db, ["org"], "read", within={"org", "project"}).ids == {"project"}
    assert cs.reach(db, ["org"], "read",
                    within={"org", "project", "doc"}).ids == {"project", "doc"}

    # `ids` excludes the seeds, so when the universe IS the seed set — the resolver's case —
    # there is nothing left for the walk to hand back. The union that follows it is a no-op,
    # which is the whole point: it cannot widen what it was given.
    universe = {"org", "project", "doc"}
    assert cs.reach(db, universe, "read", within=universe).ids == set()


def test_confinement_stops_the_chain_rather_than_skipping_the_gap(db):
    """A node the caller does not hold is not a stepping stone. Authority passes only through
    a node you hold authority over, so an excluded middle closes everything behind it — it
    does not get hopped over to reach a descendant that happens to be in the universe."""
    _chain(db, "org", "project", "doc")
    assert cs.reach(db, ["org"], "read", within={"org", "doc"}).ids == set()


def test_containment_expansion_is_confined_too(db):
    """The second half of the alternation is the half the original bug actually leaked
    through: it pulled the WHOLE containment cone in below every newly-reached context node,
    with no membership test at all."""
    from mantle.db import lattice_api as api

    cs.set_context(db, "project", "org", propagate=TOP)
    api.add_artifact_to_collection(db, "project", "doc-1")

    wide = cs.reach(db, ["org"], "read", within=cs.UNCONFINED,
                    expand_containment=db_store.list_origin_descendants)
    assert wide.ids == {"project", "doc-1"}

    confined = cs.reach(db, ["org"], "read", within={"org", "project"},
                        expand_containment=db_store.list_origin_descendants)
    assert confined.ids == {"project"}          # `doc-1` is outside the universe


def test_an_empty_universe_is_not_an_absent_one(db):
    """`within=set()` means *nothing is authorized*; `within=UNCONFINED` means *do not
    ceiling this walk*. They are opposite answers, which is why the opt-out is a named
    sentinel rather than a falsy value that an empty set could be mistaken for."""
    _chain(db, "org", "project")
    assert cs.reach(db, ["org"], "read", within=set()).ids == set()
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED).ids == {"project"}
    assert cs.UNCONFINED is not None and bool(cs.UNCONFINED) is True


def test_a_bare_id_is_refused_rather_than_iterated_into_characters(db):
    """`within="org"` would confine the walk to a universe of single letters — an
    empty-in-practice ceiling that looks like it worked. A silently wrong authority universe
    is exactly the failure this argument exists to prevent, so it is loud."""
    _chain(db, "org", "project")
    with pytest.raises(TypeError, match="not one id"):
        cs.reach(db, ["org"], "read", within="org")


def test_a_seed_the_caller_does_not_hold_is_not_walked_from(db):
    """Seeds are the caller's assertion, but they are not exempt: a seed outside the universe
    stays in the cycle guard and yields no reach. Otherwise "confined" would mean "confined
    except at the one place the caller chose"."""
    _chain(db, "org", "project")
    assert cs.reach(db, ["org"], "read", within={"project"}).ids == set()


def test_confinement_composes_with_the_action_ceiling_rather_than_replacing_it(db):
    """Two independent ceilings — identities and actions — and neither substitutes for the
    other. A node inside the universe still needs the mask, and a node the mask allows still
    needs to be in the universe."""
    _chain(db, "org", "project", propagate=[["read"]])
    universe = {"org", "project"}
    assert cs.reach(db, ["org"], "read", within=universe).ids == {"project"}
    assert cs.reach(db, ["org"], "update", within=universe).ids == set()
    assert cs.reach(db, ["org"], "read", within={"org"}).ids == set()


# ═════════════════════════════════════════════════════════════════════════════
# 6 · termination and the bound
# ═════════════════════════════════════════════════════════════════════════════

def test_a_context_cycle_terminates(db):
    """A context's context is recursive. `seen` is the guard, reused from the origin BFS."""
    _chain(db, "a", "b", "c")
    cs.set_context(db, "a", "c", propagate=TOP)   # close the loop: c → a
    got = cs.reach(db, ["a"], "read", within=cs.UNCONFINED)
    assert got.ids == {"b", "c"}                  # the seed is not re-admitted
    assert not got.truncated


def test_a_self_referential_context_terminates(db):
    cs.set_context(db, "a", "a", propagate=TOP)
    assert cs.reach(db, ["a"], "read", within=cs.UNCONFINED).ids == set()
    assert cs.context_root(db, "a") == "a"


def test_the_depth_bound_stops_the_walk_and_says_so(db):
    _chain(db, *[f"n{i}" for i in range(12)])
    got = cs.reach(db, ["n0"], "read", within=cs.UNCONFINED, max_depth=4)
    assert got.truncated and got.limit == "depth"
    assert got.depth == 4
    assert got.ids == {"n1", "n2", "n3", "n4"}
    # Truncation only ever REMOVES nodes, so it under-reaches. Fail-closed for authority.
    assert got.ids < cs.reach(db, ["n0"], "read", within=cs.UNCONFINED).ids


def test_the_node_bound_stops_the_walk_and_says_so(db):
    for i in range(20):
        cs.set_context(db, f"leaf-{i}", "org", propagate=TOP)
    got = cs.reach(db, ["org"], "read", within=cs.UNCONFINED, max_nodes=5)
    assert got.truncated and got.limit == "nodes"
    assert len(got.ids) == 5
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED).truncated is False


def test_an_untruncated_walk_never_claims_it_was_cut(db):
    """The flag is part of the result, not a diagnostic: a caller that cannot tell "I stopped
    early" from "there is nothing more" has been handed two different answers as one."""
    _chain(db, "a", "b")
    got = cs.reach(db, ["a"], "read", within=cs.UNCONFINED, max_depth=cs.DEFAULT_MAX_DEPTH)
    assert (got.truncated, got.limit) == (False, None)


def test_the_bound_is_reported_when_it_bites(db, caplog):
    _chain(db, *[f"n{i}" for i in range(6)])
    with caplog.at_level("WARNING", logger="mantle.services.context_service"):
        cs.reach(db, ["n0"], "read", within=cs.UNCONFINED, max_depth=2)
    said = [r.getMessage() for r in caplog.records]
    assert any("truncated" in m and "depth" in m for m in said), said


def test_a_walk_that_can_admit_nothing_does_not_claim_truncation(db):
    """The short-circuit for `within ⊆ seeds` must return the same answer the loop would —
    empty and COMPLETE. Reporting it as truncated would say "there may be more" about a
    universe that provably has none."""
    _chain(db, "a", "b", "c")
    got = cs.reach(db, ["a", "b", "c"], "read", within={"a", "b", "c"})
    assert got == cs.EMPTY_REACH
    assert not got.truncated and got.limit is None


def test_context_root_climbs_the_defining_chain_only(db):
    _chain(db, "org", "division", "project")
    cs.set_context(db, "project", "a-reference", is_origin=False)
    assert cs.context_root(db, "project") == "org"
    assert cs.context_root(db, "org") == "org"


# ═════════════════════════════════════════════════════════════════════════════
# 7 · the resolver — the walk may not widen what grants authorize
# ═════════════════════════════════════════════════════════════════════════════

def _seed_grant(db, user_id, resource_id, **flags):
    from mantle.db import lattice_api as api
    db.artifacts.put_artifact({"id": resource_id, "state": "committed",
                               "content_type": COLLECTION_CONTENT_TYPE,
                               "origin_root": resource_id})
    api.upsert_user_collection_grant(db, user_id=user_id, collection_id=resource_id,
                                     granted_by="test", **flags)


def _grants_alone(db, user_id, action):
    """What the principal reaches with the context lattice taken away entirely: the grants
    themselves plus the origin-containment cone below them.

    Computed from the same two primitives `resolve` uses, so this is the *definition* of the
    upper bound rather than a hand-maintained expectation that could drift away from it.
    """
    grants = db_store.get_active_grants_for_grantee(
        db, grantee_id=user_id, grantee_type="user")
    ids = [g.resource_id for g in grants if g.resource_id and mask_of(g).allows(action)]
    if not ids:
        return set()
    return set(ids) | set(db_store.list_origin_descendants(db, ids, action))


def test_a_context_edge_does_not_widen_what_a_grant_reaches(db):
    """The bug, as a test that now fails if it comes back.

    This assertion used to read `{"org", "project", "doc-1"}` off a grant on `"org"` alone,
    and was written as if that were the feature: "context becomes a unit of sharing". It is
    not. `project` sits in `org` by a CONTEXT edge, and `doc-1` was dragged in by the
    containment cone below `project` — two ids no grant reached and
    `services.dependencies.check_access` would refuse to read. A search that returns them
    hands `oracle.LightConeGrantVerifier` a content key for an artifact the read gate 404s.

    Context becoming a real unit of sharing needs `check_access` to walk the context lattice
    too. Until it does, the narrower of two disagreeing answers is the only safe one.
    """
    from mantle.db import lattice_api as api

    _seed_grant(db, "alice", "org", can_read=True)
    _chain(db, "org", "project")                   # unrestricted context edge
    api.add_artifact_to_collection(db, "project", "doc-1")

    reached = LightConeResolver(db).resolve("alice", "read")
    assert reached == {"org"}
    assert reached == _grants_alone(db, "alice", "read")

    # The structure is still there and still readable — it just confers nothing.
    assert cs.members_of(db, "org") == ["project"]
    assert cs.reach(db, ["org"], "read", within=cs.UNCONFINED).ids == {"project"}


def test_a_grant_reaches_its_containment_cone_exactly_as_before(db):
    """The positive control, and the statement that nothing about containment changed: a
    grant still propagates down origin `contains` edges, which is the reach `check_access`
    honours."""
    from mantle.db import lattice_api as api

    _seed_grant(db, "alice", "org", can_read=True)
    api.add_artifact_to_collection(db, "org", "doc-1")
    api.add_artifact_to_collection(db, "doc-1", "doc-1-a")

    assert LightConeResolver(db).resolve("alice", "read") == {"org", "doc-1", "doc-1-a"}


def test_the_resolver_confines_the_context_walk_to_the_grant_derived_set(db):
    """Not asserted from the outside only: the universe the resolver HANDS the walk must be
    exactly what grants authorize. A resolver that passed `UNCONFINED` and then filtered
    would be relying on the filter alone."""
    seen = {}

    def fake_reach(_db, seeds, action, **kw):
        seen["seeds"] = set(seeds)
        seen["within"] = kw.get("within")
        # A walk that lies and returns an id outside the universe still must not widen the
        # answer — the resolver intersects rather than trusts.
        return cs.ContextReach(frozenset({"smuggled"}), 1, False, None)

    from mantle.db import lattice_api as api
    _seed_grant(db, "alice", "org", can_read=True)
    api.add_artifact_to_collection(db, "org", "doc-1")
    _chain(db, "org", "project")

    with patch.object(cs, "reach", side_effect=fake_reach):
        reached = LightConeResolver(db).resolve("alice", "read")

    assert seen["within"] == {"org", "doc-1"}
    assert seen["seeds"] == {"org", "doc-1"}
    assert reached == {"org", "doc-1"}
    assert "smuggled" not in reached


def test_the_resolver_attenuates_through_the_context_lattice(db):
    """A context edge's mask still narrows; what it can never do is add. Both actions come
    back as the grant-derived set, and the `update` case shows the mask is not the only thing
    holding the line — even the `read` hop the edge permits confers nothing."""
    _seed_grant(db, "alice", "org", can_read=True, can_update=True)
    _chain(db, "org", "project", "doc", propagate=[["read", "update"], ["read"]])

    for action in ("read", "update"):
        assert LightConeResolver(db).resolve("alice", action) == {"org"}
        assert (LightConeResolver(db).resolve("alice", action)
                == _grants_alone(db, "alice", action))


def test_a_deny_grant_reaches_nothing_through_the_context_lattice_either(db):
    """S1's guarantee has to hold on the new edges too, or the context lattice reintroduces
    what standardizing the operator removed."""
    _chain(db, "org", "project", "doc")
    deny = GrantEntity(resource_id="org", grantee_type="user", grantee_id="alice",
                       granted_by="test", effect=GrantEntity.EFFECT_DENY, can_read=True)
    with patch.object(db_store, "get_active_grants_for_grantee", return_value=[deny]):
        assert LightConeResolver(db).resolve("alice", "read") == set()

    # The positive control: the same lattice, the same seed, an allow effect. It reaches its
    # own resource and stops — the context chain below `org` adds nothing.
    allow = GrantEntity(resource_id="org", grantee_type="user", grantee_id="alice",
                        granted_by="test", effect=GrantEntity.EFFECT_ALLOW, can_read=True)
    with patch.object(db_store, "get_active_grants_for_grantee", return_value=[allow]):
        assert LightConeResolver(db).resolve("alice", "read") == {"org"}


def test_the_resolver_threads_its_bound_into_the_context_walk(db):
    """The bound is per-node and threaded (pinned directly in
    `tests/test_lightcone_resolver.py`). Here it is the consequence that is asserted: because the
    walk is confined, the answer does not depend on how deep it was allowed to go, and truncation
    cannot change a set the walk cannot add to."""
    _seed_grant(db, "alice", "n0", can_read=True)
    _chain(db, *[f"n{i}" for i in range(10)])

    assert LightConeResolver(db, max_depth=2).resolve("alice", "read") == {"n0"}
    assert LightConeResolver(db).resolve("alice", "read") == {"n0"}


# ── the invariant, swept ─────────────────────────────────────────────────────
#
# One case is an anecdote. The property is: for EVERY principal, over every shape a context
# edge can take, `resolve` ⊆ what grants alone authorize. The sweep below builds a small
# lattice per configuration and checks it for every action in the vocabulary, so a future
# change that re-opens the widening fails here even if it does so through a mask shape or an
# action nobody thought to write a case for.

#: Every shape the `propagate` column legally holds — including the two that are easy to get
#: wrong: `None` (the column's *unrestricted*, and the old default) and the substrate's
#: compact `"r"`, which the decoder reads as carrying nothing.
_PROPAGATE_SHAPES = (None, [], ["read"], ["read", "update"], TOP, Mask.of(("admin",)), "r")


def _widening_lattice(path, hop1, hop2, is_origin):
    """A grant on `org`, a containment child below it, and a context chain out to two ids the
    grant does not reach. `outsider` is the id an attacker would be naming."""
    from mantle.db import lattice_api as api

    d = LatticeDatabase(str(path), origin="test-context")
    _seed_grant(d, "alice", "org", can_read=True, can_update=True, can_admin=True)
    api.add_artifact_to_collection(d, "org", "held-doc")        # inside the grant's cone
    cs.set_context(d, "outsider", "org", propagate=hop1, is_origin=is_origin)
    cs.set_context(d, "outsider-child", "outsider", propagate=hop2, is_origin=is_origin)
    api.add_artifact_to_collection(d, "outsider", "outsider-doc")
    # A context edge onto something already held — the legitimate, non-widening case.
    cs.set_context(d, "held-doc", "org", propagate=hop1, is_origin=is_origin)
    return d


def test_context_augmented_reach_is_never_a_superset_of_grant_reach(tmp_path):
    """The invariant, over grant/edge configurations and the whole action vocabulary.

    `resolve(p, a) ⊆ grants_alone(p, a)` — never a superset, at any mask, at any depth,
    origin or not. Equality is the expected outcome today (see the module docstring on
    `check_access`); the assertion is the SUBSET, because that is the property that must
    survive `check_access` learning the context lattice.
    """
    combos = list(itertools.product(_PROPAGATE_SHAPES, _PROPAGATE_SHAPES, (True, False)))
    assert len(combos) == 98

    for i, (hop1, hop2, is_origin) in enumerate(combos):
        d = _widening_lattice(tmp_path / f"sweep-{i}.db", hop1, hop2, is_origin)
        for action in ACTIONS:
            bound = _grants_alone(d, "alice", action)
            got = LightConeResolver(d).resolve("alice", action)
            assert got <= bound, (
                f"the context lattice widened authorization: hop1={hop1!r} "
                f"hop2={hop2!r} is_origin={is_origin} action={action!r} "
                f"added {sorted(got - bound)}"
            )
            # Named explicitly, because these are the ids a context edge would be used to
            # acquire and a subset assertion against an empty bound would pass vacuously.
            for stolen in ("outsider", "outsider-child", "outsider-doc"):
                assert stolen not in got, (hop1, hop2, is_origin, action, stolen)


def test_the_sweep_would_catch_a_widening_walk(tmp_path):
    """The sweep's own control. If confinement were removed, the configuration above reaches
    ids the grant does not — so the assertion is load-bearing rather than vacuously true.

    Asserted through `UNCONFINED`, which is exactly what the resolver used to pass by having
    no such argument at all.
    """
    d = _widening_lattice(tmp_path / "control.db", TOP, TOP, True)
    bound = _grants_alone(d, "alice", "read")
    assert bound == {"org", "held-doc"}

    unconfined = cs.reach(d, bound, "read", within=cs.UNCONFINED,
                          authority=Mask.of(("read",)),
                          expand_containment=db_store.list_origin_descendants)
    assert not unconfined.ids <= bound
    assert {"outsider", "outsider-child", "outsider-doc"} <= set(unconfined.ids)


# ═════════════════════════════════════════════════════════════════════════════
# 8 · events fan out — the amplifying direction, kept apart
# ═════════════════════════════════════════════════════════════════════════════

def test_fan_out_runs_upward_and_takes_no_action(db):
    """Delivery is amplifying: one write notifies many containers. It has no `action`
    parameter because there is nothing to attenuate — visibility is the separate question,
    answered by `reach` per subscriber. A mask on this path would make fan-out a way to widen
    authority."""
    _chain(db, "org", "project", "doc")
    got = cs.fan_out_targets(db, "doc")
    assert got.ids == {"project", "org"}
    assert not got.truncated
    import inspect
    assert "action" not in inspect.signature(cs.fan_out_targets).parameters
    # And no `within` either: confinement is an authorization ceiling, and delivery is not an
    # authorization decision. Putting one here would silently turn "who is told" into "who
    # may see", which is the conflation this whole module keeps apart.
    assert "within" not in inspect.signature(cs.fan_out_targets).parameters


def test_fan_out_reaches_a_non_origin_container_too(db):
    """Deliberately wider than `reach`: a reference context is still somewhere the change is
    worth announcing, and announcing is not authorizing."""
    cs.set_context(db, "doc", "project")
    cs.set_context(db, "doc", "a-reference", is_origin=False)
    assert cs.fan_out_targets(db, "doc").ids == {"project", "a-reference"}
    assert cs.reach(db, ["a-reference"], "read", within=cs.UNCONFINED).ids == set()


def test_fan_out_terminates_on_a_cycle_and_honours_its_bound(db):
    _chain(db, "a", "b", "c")
    cs.set_context(db, "a", "c", propagate=TOP)
    assert cs.fan_out_targets(db, "c").ids == {"b", "a"}

    _chain(db, *[f"n{i}" for i in range(8)])
    capped = cs.fan_out_targets(db, "n7", max_depth=3)
    assert capped.truncated and capped.limit == "depth"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
