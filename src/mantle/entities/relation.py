"""Edge relation kinds — the information-centric typing of graph edges.

Build-side vocabulary of the Information Gauge DB design (see
``agience-pharos/features/INFORMATION-GAUGE-DB-IMPLEMENTATION.md`` §0/§3). An
edge is named for *what it does to information*, never by a physics analogy. Five
kinds exist across the field, but only two are carried by the ``edges``
(``collection_artifacts``) collection; the rest are represented in their own
subsystems:

- ``grant``      — access / authority: the structural, grant-propagating edge.        (edges collection)
- ``derivation`` — a content transform (an operator produced this), identity kept.     (edges collection)
- ``temporal``   — order / causal precedence.  Represented by the ``root_id`` version
  chain + ``created_time`` — not a stored edge.
- ``semantic``   — ontological position (anchor-relative embedding affinity).
  Represented in the AnchorSet / vector index — not a stored edge.
- ``lifecycle``  — a state transition (e.g. commit).  Represented in the commit
  records (``entities/commit.py``) + the artifact ``state`` — not a stored edge.

Phase 0 records ``relation`` on the edges that exist and leaves the vocabulary in
place for the rest; typing them is a later formalization, not a schema change here.
"""
from __future__ import annotations

from enum import Enum


class Relation(str, Enum):
    """The information-centric kind of a relation between measurements."""

    GRANT = "grant"            # WHO   — access / authority (the confining, grant-propagating edge)
    TEMPORAL = "temporal"      # WHEN  — order / causal precedence / version chain
    SEMANTIC = "semantic"      # WHERE — ontological position (anchor-relative affinity)
    LIFECYCLE = "lifecycle"    # WHAT  — a state transition (e.g. commit)
    DERIVATION = "derivation"  # HOW   — a content transform (operator), identity preserved


# The kinds that the `edges` (collection_artifacts) collection actually stores.
EDGE_RELATIONS = frozenset({Relation.GRANT.value, Relation.DERIVATION.value})


def derive_relation(*, origin: bool, relationship: str | None) -> str:
    """Best-effort ``relation`` for a ``collection_artifacts`` edge from its
    existing signals. An operator-produced edge is a ``derivation``; every other
    edge in this collection is part of the access/containment topology, so it is a
    ``grant`` (origin edges propagate grants; plain links still sit in the access
    structure). ``origin`` is accepted for future refinement and forward-compat."""
    if relationship == "operator":
        return Relation.DERIVATION.value
    return Relation.GRANT.value
