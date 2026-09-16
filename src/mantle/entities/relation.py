"""What an edge records about itself — observed, never enumerated.

An edge carries what was **observed** when it was written, verbatim. It does not carry a type drawn
from a list, because there is no list: the kinds of relation a corpus contains are a property of the
corpus, discovered by reading it, and they are unbounded.

Why there is no vocabulary here
-------------------------------
A fixed set of relation kinds is a decision made in advance, on other data, about what relationships
are possible — which is the same objection as a hand-set threshold, at the scale of an ontology. It
fails in both directions. A corpus whose relations do not fit the list has them flattened into the
nearest member, and the distinction is lost at write time where nothing can recover it. A corpus
with fewer distinct relations than the list carries a vocabulary asserting structure it does not
have.

Nothing in a trained network has an edge-type table either. What such a system calls a relation is a
direction that emerged, and how many of them there are is bounded by nothing declared in advance. An
edge store that asserts a closed vocabulary cannot represent that, and a store built to hold what a
corpus actually contains must not.

So this module classifies nothing. `observed_relation` returns what the writer saw, and a reader
that wants to know what *kind* of relation an edge is measures it from the records at either end.

What the store does hold
------------------------
Two open, uninterpreted strings, and both are the writer's:

- `label` — on the edge row, carried by `LatticeGraphStore`. Already open, already unbounded; a
  corpus build routinely writes tens of distinct labels, and `count_edges_by_label` indexes whatever
  is there without knowing the set in advance.
- `relationship` — what the writer recorded about *this* edge's role, or `None`.

Neither is a type, neither is validated against a set, and no read in this package branches on the
value of either.
"""
from __future__ import annotations


def observed_relation(*, origin: bool, relationship: str | None) -> str | None:
    """What was observed about this edge, returned unchanged.

    `relationship` is the writer's own word for what this edge records, and it is passed through
    verbatim — not mapped, not normalised, not checked against a set. `None` means nothing was
    observed, which is a fact about the edge and is reported as itself rather than filled in with a
    default.

    `origin` is accepted because callers have it to hand and an edge's containment role is part of
    what was observed. It is deliberately not used to *derive* a kind: deriving one would be this
    module classifying, which is the thing it does not do.
    """
    return relationship


#: Retained so callers written against the previous name keep working. It is the same function:
#: nothing is derived, and the word is kept only to avoid breaking an import.
derive_relation = observed_relation
