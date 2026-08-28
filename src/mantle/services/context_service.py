"""The context lattice — one bounded walk that authorization and recall both take.

`entities/context.py` says what a context node is. This module says how the lattice made of
them is written and walked, and it is where D16's actual claim lives:

> Authorization and selection are the same traversal. Not "compute the authorized set, then
> filter the results" — one walk, attenuated at every hop, that cannot reach what it may not
> reach.

A context edge NARROWS. It is never a source of reach
=====================================================
This is the module's first claim, ahead of everything else it says, because the first version
of this walk got it backwards and the bug is worth naming: :func:`reach` unioned its result
into the caller's authorized set, admitted an edge's `dst` with no membership test against
anything a grant had conferred, and then pulled the whole containment cone in below it. A grant
on `org` alone came back holding `{"org", "project", "doc-1"}` — two ids no grant had reached.
That is a context edge *manufacturing* authority, which is the one thing an attenuator may not
do. It was inert only because nothing wrote a context edge yet.

The rule now, and it is structural rather than asserted:

    **A walk may not leave the set of ids the caller already holds authority over.**

:func:`reach` therefore takes a **required** `within` argument — the authority universe — and
admits a node only if it is in it. Two consequences fall straight out and neither needs its own
check:

* *Authority is only ever passed through a node you hold authority over.* Every node on the
  frontier is a seed the caller vouched for, or a node already admitted, and admitted means in
  `within`. There is no third way onto the frontier.
* *The result is bounded above by `within`,* so `caller_set | reach(...).ids ⊆ caller_set`
  whenever the caller passes its own authorized set — union with the walk cannot widen it, at
  any depth, under any edge, for any mask.

`within=`:data:`UNCONFINED` is the explicit, named opt-out, for a caller that wants the
*structure* of the lattice (navigation, display, a test of the traversal itself) rather than an
authorization answer. It is spelled out at the call site precisely because it is the unsafe
one; there is no default, so no call site gets it by not thinking about it. The writer half of
the same posture is `db.edge.DEFAULT_CONTEXT_PROPAGATE` — an edge written with no mask
now transmits nothing rather than everything.

Where that leaves the resolver today
------------------------------------
`services.dependencies.check_access` is the gate in front of every artifact read, and it walks
**origin containment only** — it has never heard of a context edge. So the authority universe
`search.mantle.lightcone.resolve` may hand this walk is exactly its grant-derived set, and the
context walk consequently contributes **nothing** to the authorized set it returns. That is not
a defect in this module, it is the honest consequence of two traversals answering "what may this
principal reach": the narrower one is the answer. Making context a real unit of sharing (D16)
requires `check_access` to walk the context lattice too, and until it does, widening `resolve`
alone would hand `oracle.LightConeGrantVerifier` a content key for an artifact the read gate
would then refuse — the same class of failure as audit finding S1.

Three things travel along a context edge, and they are three different operations
=================================================================================
Conflating them is the trap the attenuation work flags, so they are separate functions here
and each says which it is:

============  =============  ==========================================================
Authority     **attenuates**  :func:`reach` — the `propagate` mask, composed with the one
                              meet in :mod:`mantle.attenuation`. Monotone, never widens.
Recall reach  **attenuates**  :func:`reach` again, and deliberately the *same* call: a
                              query may only reach what authority reached, so giving recall
                              its own traversal is how the two come to disagree.
Events        **fans out**    :func:`fan_out_targets` — amplifying, the opposite operation.
                              Event *visibility* attenuates and asks :func:`reach`; event
                              *delivery* does not and must not. Reaching for the meet on a
                              delivery path would turn fan-out into a way to widen authority.
============  =============  ==========================================================

What composition means on this walk
===================================
Each hop composes the authority held so far with the edge's `propagate` mask, using
`Mask.__and__` — the single-sourced meet, with `DENY` absorbing. Nothing here re-implements
intersection; `tests/test_attenuation_is_single_sourced.py` is what keeps that true.

Composition along a path is therefore a **meet**, and a node with several parent contexts is
reached by a **join over paths of meets along paths**: two contexts are alternatives, not
co-requirements, so one blocked path does not close a second open one. That is the ordinary
reading of a lattice and it is also what a grant already means — but it is worth stating,
because "attenuation" said carelessly suggests every additional parent narrows, and it does
the opposite.

One consequence makes the walk cheap. For a single action, `(m & e).allows(a)` is
`m.allows(a) and e.allows(a)`, so the accumulated mask cannot change a later prune decision
that the edge does not already make. Every node on the frontier passed the test, so
first-arrival is the whole answer and no node needs revisiting under a wider mask. The
composed mask is still carried and still consulted — it is the thing being proved, and it is
what lets this same walk answer a multi-action question later — but it costs one interned
lookup per hop and no fixpoint.

Termination and bounds
======================
A context's context is recursive, so the walk needs a root and a cycle guard. Both already
exist in this store's idiom and both are reused: `seen` (as `list_origin_descendants` and
`GraphStore.descendants` use it) makes a cycle terminate rather than hang, and
:func:`context_root` walks the origin chain the way `lattice_api.get_origin_root` does.

`resolve()` has been able to say it "runs to exhaustion" because the containment lattice is
shallow and engine-driven. The context lattice is deeper by construction, so this walk is
bounded: :data:`DEFAULT_MAX_DEPTH` hops and :data:`DEFAULT_MAX_NODES` nodes. **A bound that
truncates says so** — :class:`ContextReach` carries `truncated` and names which limit was
hit, and the walk logs a warning naming it. Truncation only ever removes nodes, so it
under-reaches: fail-closed for authority, visibly incomplete for recall, never widening.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set

from mantle.attenuation import ACTIONS, TOP, Mask
from mantle.db.edge import CONTEXT_LABEL, DEFAULT_CONTEXT_PROPAGATE, EdgesTruncated
from mantle.entities.context import (
    CONTEXT_CONTENT_TYPE,
    Context,
    is_context,
    is_context_node,
    new_context,
)

log = logging.getLogger(__name__)

__all__ = [
    "CONTEXT_CONTENT_TYPE",
    "CONTEXT_LABEL",
    "Context",
    "ContextReach",
    "DEFAULT_CONTEXT_PROPAGATE",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_NODES",
    "UNCONFINED",
    "clear_context",
    "context_root",
    "contexts_of",
    "fan_out_targets",
    "is_context",
    "is_context_node",
    "members_of",
    "new_context",
    "reach",
    "set_context",
]


#: Hops from a seed. A hand-authored nesting (org → division → project → workspace) is a
#: handful deep; this leaves an order of magnitude of headroom and still stops a pathological
#: chain from being walked one node at a time. It is a cost ceiling, not a policy cut — which
#: is why hitting it is reported rather than absorbed.
DEFAULT_MAX_DEPTH = 64

#: Distinct nodes admitted beyond the seeds. The value is `db.constants.CT_FETCH_CAP`'s,
#: and deliberately so: that cap answers the same question for a typed content-type fetch —
#: how much will this pull before it stops claiming to be exhaustive — and two different
#: numbers for one policy would be two things to reason about.
DEFAULT_MAX_NODES = 10_000

class _Unconfined:
    """The type of :data:`UNCONFINED`. A singleton value, not a set — so it can never be
    mistaken for an authority universe that happens to be empty, and so `within=UNCONFINED`
    reads as the deliberate choice it is at every call site."""

    __slots__ = ()

    def __repr__(self) -> str:                              # pragma: no cover - display only
        return "UNCONFINED"


#: Opt out of confinement: walk the context lattice as *structure*, admitting whatever the edges
#: reach. **The result is then not an authorization answer** — it is navigation, display, or a
#: test of the traversal itself, and a caller that unions it into an authorized set has widened
#: that set by exactly the amount this module exists to prevent.
#:
#: It is a required argument with no default precisely because it is the unsafe direction. The
#: safe one cannot be reached by omission either — a caller must name the universe it holds — so
#: neither choice is made by accident.
UNCONFINED = _Unconfined()

#: What `expand_containment` must look like. `lattice_api.list_origin_descendants` is the
#: implementation; it is a parameter rather than a direct call so that a caller can walk the
#: context lattice alone, and so tests can watch exactly when containment is consulted.
ContainmentExpander = Callable[[Any, List[str], str], Iterable[str]]


class ContextReach(NamedTuple):
    """What a walk reached, and whether it is the whole answer.

    `truncated` and `limit` are not diagnostics — they are part of the result. A caller that
    drops them turns "I stopped early" into "there is nothing more", and those are different
    answers in the same way `count_edges_by_label`'s `None` differs from `0`.
    """

    #: Nodes reached, seeds EXCLUDED — the same contract as `list_origin_descendants`.
    ids: frozenset
    #: Hops actually taken.
    depth: int
    #: True when a bound stopped the walk with a frontier still in hand. Conservative: it
    #: means the answer is not PROVEN complete, not that something was certainly missed.
    #: Erring the other way would let a walk that finished by luck claim a completeness it
    #: cannot demonstrate.
    truncated: bool
    #: Which bound: ``"depth"``, ``"nodes"``, or None.
    limit: Optional[str]


EMPTY_REACH = ContextReach(frozenset(), 0, False, None)


# ─────────────────────────────────────────────────────────────────────────────
# writing the lattice
# ─────────────────────────────────────────────────────────────────────────────

def set_context(db: Any, node_id: str, context_id: str, *,
                propagate: Any = DEFAULT_CONTEXT_PROPAGATE, is_origin: bool = True,
                order_key: Optional[str] = None) -> bool:
    """Record that `node_id` sits in the context `context_id`.

    Add, not set: a node may sit in several contexts, and that is the composability the whole
    decision is for. Idempotent by the store's `edge_key` primary key, so a replay leaves one
    row.

    `is_origin=False` writes a *reference* — the node is in that context for navigation and
    naming, and no authority flows through it. :func:`reach` refuses to walk it, for the same
    reason `list_origin_descendants` refuses to walk a non-origin containment link: otherwise
    naming someone else's artifact as sitting in your context would be a way to acquire
    authority over it.

    `propagate` defaults to :data:`~mantle.db.edge.DEFAULT_CONTEXT_PROPAGATE` — the
    empty mask. An edge written with no mask is a membership fact and nothing more; a writer
    that means authority to flow says which actions. The default is mirrored from the store's
    writer rather than restated, so the two cannot drift into disagreeing about what an
    unstated mask means.
    """
    try:
        db.graph.add_context_edge(context_id, node_id, propagate=propagate,
                                  is_origin=is_origin, order_key=order_key)
        return True
    except Exception:                      # same contract as `add_artifact_to_collection`
        log.warning("context edge %s -> %s was not written", context_id, node_id,
                    exc_info=True)
        return False


def clear_context(db: Any, node_id: str, context_id: str) -> bool:
    """Drop one context edge. Idempotent; a missing edge is already the requested state."""
    try:
        db.graph.delete_context_edge(context_id, node_id)
        return True
    except Exception:
        log.warning("context edge %s -> %s was not removed", context_id, node_id,
                    exc_info=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# reading it
# ─────────────────────────────────────────────────────────────────────────────

def _edges(db: Any, node_id: str, direction: str) -> List[Dict[str, Any]]:
    """Context-edge rows on one side of a node, or none if the store cannot answer.

    Prefers the typed accessor and falls back to a labelled `edges_of`, so a graph store that
    predates :data:`CONTEXT_LABEL` still reads correctly instead of raising.

    A read that fails yields nothing rather than propagating. That is the contract
    `list_origin_descendants` already keeps on this path, and it is the fail-closed direction:
    an unreadable edge withholds reach, it never grants it.
    """
    graph = getattr(db, "graph", None)
    if graph is None:
        return []
    try:
        typed = getattr(graph, "context_edges", None)
        if typed is not None:
            return list(typed(node_id, direction=direction) or [])
        return list(graph.edges_of(node_id, label=CONTEXT_LABEL, direction=direction) or [])
    except EdgesTruncated:
        # The fail-closed reading above ("an unreadable edge withholds reach") is right for a row
        # that will not open and wrong for a read that stopped early: the edges ARE readable and
        # there are more of them. Yielding `[]` would withhold reach the graph does grant, which
        # is a wrong answer rather than a conservative one.
        raise
    except Exception:
        return []


def _prop(edge: Dict[str, Any], key: str) -> Any:
    """Edge attribute, promoted column first then the props blob — `lattice_api._eprop`'s rule,
    because an edge written through the wire path lands on the other side of the row."""
    got = edge.get(key)
    if got is None:
        got = (edge.get("props") or {}).get(key)
    return got


def contexts_of(db: Any, node_id: str, *, origin_only: bool = False) -> List[str]:
    """The contexts `node_id` sits in — the inbound read of the edge.

    `origin_only` narrows to the defining contexts, i.e. the ones authority actually flows
    through. The default is the full set because navigation and display want the references
    too; every authorization caller passes True or, better, asks :func:`reach`.
    """
    out: List[str] = []
    for e in _edges(db, node_id, "in"):
        if origin_only and not _prop(e, "is_origin"):
            continue
        src = e.get("src")
        if src and src not in out:
            out.append(src)
    return out


def members_of(db: Any, context_id: str, *, origin_only: bool = False) -> List[str]:
    """What sits in this context — the outbound read. One hop, not the transitive set;
    :func:`reach` is the transitive, attenuated answer."""
    out: List[str] = []
    for e in _edges(db, context_id, "out"):
        if origin_only and not _prop(e, "is_origin"):
            continue
        dst = e.get("dst")
        if dst and dst not in out:
            out.append(dst)
    return out


def context_root(db: Any, node_id: str, *, max_depth: int = DEFAULT_MAX_DEPTH) -> str:
    """The top of this node's defining context chain — the recursion's root.

    Deliberately shaped like `lattice_api.get_origin_root`, including its `visited` guard:
    a context whose context is (transitively) itself must terminate at the node it re-enters,
    not loop. Where several origin contexts exist the first is taken and the walk stops
    climbing that fork — a multi-parent lattice has no single root, and inventing a tiebreak
    would be inventing information. Callers that need every ancestor want
    :func:`fan_out_targets`, which returns the set rather than a point.
    """
    current = str(node_id)
    visited = {current}
    for _ in range(max(0, int(max_depth))):
        parents = contexts_of(db, current, origin_only=True)
        nxt = next((p for p in parents if p and p not in visited), None)
        if nxt is None:
            return current
        visited.add(nxt)
        current = nxt
    return current


# ─────────────────────────────────────────────────────────────────────────────
# the walk — authority and recall, attenuated
# ─────────────────────────────────────────────────────────────────────────────

def reach(db: Any, seeds: Iterable[str], action: str, *,
          within: Any,
          authority: Optional[Mask] = None,
          max_depth: int = DEFAULT_MAX_DEPTH,
          max_nodes: int = DEFAULT_MAX_NODES,
          expand_containment: Optional[ContainmentExpander] = None) -> ContextReach:
    """Every node reachable from *seeds* for *action* through the context lattice, **without
    ever leaving *within***.

    The walk alternates two steps per hop, because the two lattices only compose if it does:

    1. **context edges** out of the frontier, each composing the authority held so far with
       the edge's `propagate` mask through the one meet, pruning where the composed authority
       stops allowing *action*, and skipping non-origin edges outright;
    2. **containment** below whatever step 1 just found, via *expand_containment*, so a
       nested context's contents are reached and not just the context node itself. Omit the
       expander and the walk follows context edges alone.

    *within* — **required, no default** — is the set of ids the caller already holds authority
    over, and it is a hard ceiling on the *identities* the walk may admit, exactly as
    *authority* is a hard ceiling on the *actions* it may carry. Both steps above check it, so
    neither a context edge nor a containment expansion below one can name a node the caller did
    not already reach. Pass :data:`UNCONFINED` to walk the lattice as structure instead; see
    that constant, and the module docstring, for why that answer is not an authorization one.

    Because the frontier only ever holds a seed or an admitted node, and an admitted node is in
    *within*, "authority is only passed through a node you hold authority over" is a property of
    the loop rather than a rule it enforces. Seeds outside *within* are held in the cycle guard
    but never walked from — a caller that vouches for a seed it does not hold gets no reach out
    of it.

    *authority* is the ceiling the seeds are held under — pass the grant's mask, and nothing
    below can exceed it, because the first thing every path does is meet with it. It defaults
    to :data:`~mantle.attenuation.TOP`, the identity, so a caller that has already checked its
    seeds is not made to state a ceiling twice. It has a default and *within* does not, because
    an over-wide action ceiling is caught by the per-edge meet below it and an over-wide identity
    universe is caught by nothing.

    Returns a :class:`ContextReach`. Seeds are excluded from `ids`, matching
    `list_origin_descendants`, so the two results union cleanly — and when *within* is the
    caller's own authorized set, that union is provably the set it started with.
    """
    if action not in ACTIONS:
        # An unmapped verb is a denial, never a hole opened by a typo — `Mask.allows` takes
        # the same position and this returns before touching the store rather than walking a
        # lattice whose every prune would be False anyway.
        return EMPTY_REACH

    ceiling = TOP if authority is None else authority
    if not ceiling.allows(action):
        return EMPTY_REACH

    universe: Optional[Set[str]] = None
    if within is not UNCONFINED:
        if isinstance(within, str):
            # A bare id iterates into its characters, which would confine the walk to a
            # universe of single letters — an empty-in-practice ceiling that looks like it
            # worked. Loud, because a silently wrong authority universe is the failure this
            # argument exists to prevent.
            raise TypeError(
                "`within` is a collection of ids, not one id: %r would confine the walk to "
                "its characters. Pass {%r}." % (within, within))
        universe = {str(x) for x in (within or ()) if x}
        if not universe:
            # An empty universe authorizes nothing, so there is no node this walk could
            # legally admit. Distinct from `UNCONFINED`, which is why the opt-out is a
            # sentinel and not `None`: `within=set()` must mean "nothing", never "anything".
            return EMPTY_REACH

    seen: Set[str] = {str(s) for s in seeds if s}
    if not seen:
        return EMPTY_REACH

    # Nothing outside `universe` is admissible and everything inside it is already in `seen`,
    # so no hop could add a node. Returning here is the same answer the loop would reach, at
    # the cost of one set comparison instead of an edge read per seed. It is also the ordinary
    # case for `lightcone.resolve` today, which passes its own authorized set as both — see the
    # module docstring on why that is the honest universe until `check_access` walks context.
    if universe is not None and universe <= seen:
        return EMPTY_REACH

    walkable = seen if universe is None else (seen & universe)
    if not walkable:
        return EMPTY_REACH

    frontier: Dict[str, Mask] = {node: ceiling for node in walkable}
    reached: Set[str] = set()
    max_depth = max(0, int(max_depth))
    max_nodes = max(0, int(max_nodes))
    depth = 0
    limit: Optional[str] = None

    #: What a containment-expanded node's authority is *known* to be. `list_origin_descendants`
    #: returns ids, having already pruned on the action, so this is exactly what it told us —
    #: no more. It is not a widening: see the module docstring on why the carried mask cannot
    #: change a later prune the edge does not already make.
    via_containment = Mask.of((action,))

    while frontier and limit is None:
        if depth >= max_depth:
            limit = "depth"
            break
        depth += 1
        nxt: Dict[str, Mask] = {}

        for node, held in frontier.items():
            for e in _edges(db, node, "out"):
                if not _prop(e, "is_origin"):
                    continue               # a reference, not an authority path
                composed = held & Mask.from_propagate(_prop(e, "propagate"))
                if not composed.allows(action):
                    continue               # prune: this edge, and everything behind it
                dst = e.get("dst")
                if not dst or dst in seen:
                    continue               # the cycle guard, and the re-entry guard
                if universe is not None and dst not in universe:
                    # THE narrowing. Without it a context edge is a source of reach: whoever
                    # can write one names any id as sitting in a context they hold and the walk
                    # confers authority over it. `is_origin` and `propagate` both guard the
                    # EDGE; this guards the DESTINATION, which is the thing being acquired.
                    continue
                if len(reached) >= max_nodes:
                    limit = "nodes"
                    break
                seen.add(dst)
                reached.add(dst)
                nxt[dst] = composed
            if limit is not None:
                break

        if limit is None and expand_containment is not None and nxt:
            for dst in _containment(db, list(nxt), action, expand_containment):
                if dst in seen:
                    continue
                if universe is not None and dst not in universe:
                    continue               # the same ceiling, one lattice over
                if len(reached) >= max_nodes:
                    limit = "nodes"
                    break
                seen.add(dst)
                reached.add(dst)
                nxt[dst] = via_containment

        frontier = nxt

    truncated = limit is not None
    if truncated:
        log.warning(
            "context walk truncated at the %s bound (depth=%d, reached=%d, action=%r): the "
            "result is incomplete, not empty — raise max_%s if this lattice is legitimately "
            "this large", limit, depth, len(reached), action,
            "depth" if limit == "depth" else "nodes")
    return ContextReach(frozenset(reached), depth, truncated, limit)


def _containment(db: Any, roots: List[str], action: str,
                 expander: ContainmentExpander) -> Set[str]:
    """The containment cone below *roots*, through the caller's expander.

    Wrapped so a store that cannot answer withholds reach instead of aborting the whole walk —
    the same fail-closed posture `_edges` takes, and the same one the origin BFS already takes
    per node.
    """
    try:
        return {str(x) for x in (expander(db, roots, action) or ()) if x}
    except Exception:
        log.warning("containment expansion failed below %d context node(s)", len(roots),
                    exc_info=True)
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# events — the amplifying direction
# ─────────────────────────────────────────────────────────────────────────────

def fan_out_targets(db: Any, node_id: str, *,
                    max_depth: int = DEFAULT_MAX_DEPTH,
                    max_nodes: int = DEFAULT_MAX_NODES) -> ContextReach:
    """The contexts a change at `node_id` should be announced to — child → container, upward.

    **This is not attenuation and must not be given a mask.** Delivery fans out: one write
    notifies many subscribers, which is the amplifying operation, and composing a `propagate`
    mask into it would make the fan-out path a way to widen authority rather than a way to
    reach subscribers. It takes no `action` argument for exactly that reason — there is no
    action to attenuate.

    What a subscriber may then *see* is a separate question with a separate answer:
    :func:`reach`, or the light cone, asked per subscriber. Delivery decides who is told
    something changed; visibility decides what they are told. Keeping the two functions apart
    is the whole safeguard.

    Provided now so the event bus can consume container/context propagation when it lands.
    Nothing here depends on that work existing.
    """
    seen: Set[str] = {str(node_id)}
    frontier: List[str] = [str(node_id)]
    out: Set[str] = set()
    max_depth = max(0, int(max_depth))
    max_nodes = max(0, int(max_nodes))
    depth = 0
    limit: Optional[str] = None

    while frontier and limit is None:
        if depth >= max_depth:
            limit = "depth"
            break
        depth += 1
        nxt: List[str] = []
        for node in frontier:
            for container in contexts_of(db, node):
                if container in seen:
                    continue               # a context cycle terminates here too
                if len(out) >= max_nodes:
                    limit = "nodes"
                    break
                seen.add(container)
                out.add(container)
                nxt.append(container)
            if limit is not None:
                break
        frontier = nxt

    if limit is not None:
        log.warning("context fan-out truncated at the %s bound from %r (depth=%d, targets=%d)",
                    limit, node_id, depth, len(out))
    return ContextReach(frozenset(out), depth, limit is not None, limit)
