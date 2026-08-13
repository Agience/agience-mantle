# entities/context.py
"""Context as a first-class artifact — the node the context lattice is built out of.

A context is not a new kind of thing. ``Collection = Artifact`` is already literally true
(`entities/collection.py`), so a context is a **role an artifact plays**, discriminated by
`content_type` exactly as a workspace or a collection is. Declaring a role rather than a new
entity is what buys identity, version lineage, provenance and grantability for free: a
context can be revised, cited and *granted on* because an artifact can, and context becomes
a unit of sharing without a second permission system.

Why the role has to exist at all
--------------------------------
Context is expressed three unrelated ways: a collection's scalar ``origin_root``, the
``collection_id ‖ cluster_id`` pair a vector cell key is derived from, and
``artifact.context``, a free-form JSON blob. A scalar handle cannot **compose** — it has no
parent, cannot nest, cannot be shared between two collections, and cannot itself be the
subject of a grant. A context *artifact* does all four. That is the whole content of D16, and
it is why "context is itself an artifact" is the load-bearing sentence rather than a
notational preference.

The relation that makes contexts nest is `db.edge.CONTEXT_LABEL`; the bounded,
attenuating walk over it is :mod:`mantle.services.context_service`.

What the role does NOT buy
--------------------------
Grantability, not authority. A context is a unit of *sharing* because a grant can name it —
the grant is what confers, and the edge only shapes what a grant already conferred. A context
edge is never a source of reach on its own: `context_service.reach` may not leave the set of
ids its caller already holds, and an edge written with no `propagate` mask now transmits
nothing (`db.edge.DEFAULT_CONTEXT_PROPAGATE`). Read the "A context edge NARROWS"
section of :mod:`mantle.services.context_service` before writing anything that walks these
edges; the first version of that walk unioned rather than narrowed, and a grant on one node
came back holding two its grants had never reached.

What this module does NOT change
--------------------------------
**Cell-key derivation.** ``HKDF(master, collection_id ‖ cluster_id)`` is untouched and no
ciphertext on disk moves. It needs no change to start composing, either: a collection id
already *is* an artifact id, so under this model it is already a context-node id, and
indexing keeps working unmodified. Re-rooting the key on a context node is a separate,
deliberate migration with the shape `scripts/mantle_cas_rekey.py` already has — additive
first, keys unchanged.

Two questions, two predicates
-----------------------------
:func:`is_context` asks whether an artifact **declares** the role. :func:`is_context_node`
asks whether it **already acts** as one, which the containers do — a workspace and a
collection are context nodes that predate the vocabulary. The two are kept apart because
answering the second with the first would say the existing corpus has no context at all,
and answering the first with the second would let any container claim a declared role it
never took.
"""

from typing import Any, Optional

from .artifact import (
    Artifact,
    COLLECTION_CONTENT_TYPE,
    WORKSPACE_CONTENT_TYPE,
)

#: The declared role. Registered like every other Agience artifact type; nothing about the
#: store, the index or the key path needs to know it exists for a context artifact to be
#: created, versioned, granted on or searched.
CONTEXT_CONTENT_TYPE = "application/vnd.agience.context+json"

#: A context IS an artifact — same table, same entity, same lineage. The alias exists for the
#: same reason `Collection = Artifact` does: so call sites can name the role they mean without
#: implying a second storage shape.
Context = Artifact

#: Every content type that acts as a context node today. The containers are in it because they
#: already were one: `collection_id` is an artifact id, which is what makes this decision
#: additive rather than a migration.
CONTEXT_NODE_CONTENT_TYPES = frozenset({
    CONTEXT_CONTENT_TYPE,
    COLLECTION_CONTENT_TYPE,
    WORKSPACE_CONTENT_TYPE,
})

__all__ = [
    "CONTEXT_CONTENT_TYPE",
    "CONTEXT_NODE_CONTENT_TYPES",
    "Context",
    "content_type_of",
    "is_context",
    "is_context_node",
    "new_context",
]


def content_type_of(subject: Any) -> Optional[str]:
    """The `content_type` of *subject*, whatever shape it arrived in.

    Duck-typed for the reason :func:`entities.grant.mask_of` is: artifacts reach these
    predicates as entities from the router layer, as raw lattice docs from `db`, and
    as plain strings from callers that hold only the type. A predicate that worked on one of
    the three would be silently wrong on the other two.
    """
    if subject is None:
        return None
    if isinstance(subject, str):
        return subject
    if isinstance(subject, dict):
        got = subject.get("content_type")
    else:
        got = getattr(subject, "content_type", None)
    return got if isinstance(got, str) else None


def is_context(subject: Any) -> bool:
    """Does *subject* declare the context role?

    Strict: exactly :data:`CONTEXT_CONTENT_TYPE`. A collection is not one of these, even
    though it is a context node — see the module docstring for why the two questions stay
    separate.
    """
    return content_type_of(subject) == CONTEXT_CONTENT_TYPE


def is_context_node(subject: Any) -> bool:
    """Does *subject* already act as a context — a declared context, or a container?

    This is the predicate traversal and indexing want. It is deliberately wider than
    :func:`is_context`, because the corpus that exists was written before the role did and
    every collection in it is a context node whether or not anyone said so.
    """
    return content_type_of(subject) in CONTEXT_NODE_CONTENT_TYPES


def new_context(
    name: str,
    *,
    description: Optional[str] = None,
    collection_id: str = "",
    created_by: Optional[str] = None,
    context_id: Optional[str] = None,
    **fields: Any,
) -> Artifact:
    """An unsaved context artifact.

    A constructor rather than a subclass: there is no behaviour to add, only a
    `content_type` to get right, and a subclass would invite a second `from_dict` that could
    disagree with :class:`Artifact`'s about the stored columns.

    Nesting is NOT expressed here. `collection_id` is containment — where the context
    document itself lives — and a context's *context* is an edge, written by
    :func:`services.context_service.set_context`. Conflating the two would put the
    composable relation back into a scalar field, which is the thing this type exists to
    stop.
    """
    return Artifact(
        id=context_id,
        collection_id=collection_id,
        content_type=CONTEXT_CONTENT_TYPE,
        name=name,
        description=description,
        created_by=created_by,
        **fields,
    )
