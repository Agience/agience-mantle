"""Derive an artifact id from a caller-chosen natural key, so a rewrite is not a new artifact.

**The problem this exists to remove.** ``POST /artifacts`` assigns a fresh ``uuid4`` and
``CreateArtifactRequest`` carries no ``id``, so a client that stores the same thing twice — the
same file, the same session, the same note — has exactly one way to make the second write land
on the first artifact: remember the id it was given. Remembering it means keeping a map from
"the thing" to "the id" somewhere outside this store, and that map is a second source of truth
about identity. It fails the way second sources of truth fail: a write whose reply is lost
still SUCCEEDS here, the client records nothing, and the next write of the same thing creates a
second root that nothing will ever reconcile. Measured on the dogfooding node before this
landed: one README as two artifacts three minutes apart, one session as five.

**The fix is to make the id a function of the thing rather than of the write.** The caller
names what the artifact is *of* — ``file:c:/repo/README.md``, ``session:7c7bcb7b`` — and the id
follows from that name. Two writes of one thing are then one artifact whatever happened to the
network in between, and no client needs to remember anything.

This is the pattern the lattice already uses everywhere identity has to survive a restart or be
computed by two parties independently: ``services/oidc.external_user_id`` is
``uuid5(_USER_NS, f"{tenant}\\n{sub}")``, ``search/anchors/anchorset`` makes an anchor's id
``uuid5(_ANCHOR_NS, sha256(label ‖ model_id ‖ embedding))``, and ``prism/schema`` derives a
schema id the same way. Nothing new is being invented here; a derivation is being extended to
the one place that was still remembering instead.

**The principal is inside the derivation, and that is what makes it safe.** A globally derived
id would let two principals compute the same id for their own unrelated ``README.md``, and the
second one's create would name an artifact the first one owns — which is a collision the caller
cannot see and this store must not resolve by guessing. Binding the derivation to the acting
principal makes the collision impossible to construct rather than something to detect: an
identity names *this principal's* artifact for a thing, always. Two principals converging on
one artifact is what a grant is for, and it stays a deliberate act rather than an accident of
two people choosing the same filename.
"""
from __future__ import annotations

import uuid
from typing import Optional

__all__ = ["ARTIFACT_IDENTITY_NS", "derive_artifact_id"]

#: The namespace every derived artifact id hangs off. A named constant rather than
#: ``NAMESPACE_URL`` directly, so ids derived here can never collide with ids derived for a
#: different purpose from the same inputs.
ARTIFACT_IDENTITY_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "artifact-identity.mantle.agience.ai")


def derive_artifact_id(principal_id: str, identity: str) -> str:
    """The id of *this principal's* artifact for the thing called ``identity``.

    ``identity`` is opaque to Mantle and is meant to be. The store has no idea what a repository
    is, what a session is, or which of a caller's concerns deserve their own artifact — the
    caller is the only party that knows when two writes are about the same thing, so the caller
    names it. What Mantle contributes is that the same name always produces the same id, on
    every node, with no state to consult and nothing to keep in sync.

    The two parts are joined with ``\\n`` for the reason ``oidc.external_user_id`` joins its two
    the same way: the separator has to be a character the parts cannot contain, or
    ``("ab", "c")`` and ``("a", "bc")`` derive one id. A principal id is a UUID and a newline is
    not in it, so the join is unambiguous on the left; an identity carrying a newline can only
    collide with another identity from the same principal, which is that caller's own namespace
    to organise.

    Raises ``ValueError`` on an empty part rather than deriving from ``""`` — an empty identity
    would give every artifact a caller wrote without one the same id, which is the loudest
    possible version of the bug this module exists to prevent.
    """
    if not principal_id:
        raise ValueError("derive_artifact_id needs a principal id")
    if not identity or not identity.strip():
        raise ValueError(
            "derive_artifact_id needs a non-empty identity — an empty one would derive a "
            "single id for every artifact this principal writes"
        )
    return str(uuid.uuid5(ARTIFACT_IDENTITY_NS, "%s\n%s" % (principal_id, identity)))


def derived_id_for(principal_id: str, identity: Optional[str]) -> Optional[str]:
    """:func:`derive_artifact_id`, or ``None`` when no identity was supplied.

    The call-site shape: ``identity`` is optional on the write path, and a caller that omits it
    gets the unchanged ``uuid4`` behaviour. Keeping the ``None`` check here means the router
    reads as one expression rather than a branch that could drift from the one below it.
    """
    if identity is None:
        return None
    return derive_artifact_id(principal_id, identity)
