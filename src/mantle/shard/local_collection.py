"""Ember's local collection — an artifact, because everything is an artifact.

A collection is not a namespace string. Mantle is explicit: *"Container-as-artifact: a
workspace IS a collection IS an artifact"* (`entities/collection.py` — `Collection = Artifact`,
discriminated by `content_type`, membership carried by **edges**). So the `collection_id` in a
region path is a real artifact's id, and a leaf that invented a name like `"default"` would
compute region ids corresponding to no artifact: plausible, routable, and shareable with
nobody. It would look like it worked.

So Ember authors its local collection the same way it authors anything else — as a signed,
content-addressed artifact it originates and can push. The leaf does not borrow an identity
from config; it makes one.

KNOWN GAP — ENUMERABLE (see memory: mesh-metadata-leak)
-------------------------------------------------------
The namespace here is **public**, so anyone holding a principal id can compute their collection
id, derive their region ids, and ask peers "do you hold region X?" — an oracle for *does this
person have data about this concept*. The determinism below is required (devices must agree), so
the fix is to **salt, not remove**: derive from a per-principal secret that a user's own devices
share. Settle this before multi-tenant grants/sharing testing — region ids are baked into every
shard, so re-keying later is a migration of everything.

WHY THE ID IS DETERMINISTIC
---------------------------
`uuid5(principal)`, not a random uuid4. The same person on a laptop and a desktop derives the
**same** local collection id, so both leaves compute **identical region ids** and can sync to
each other directly — your own devices are peers in your own mesh, with no cloud round-trip and
nothing to reconcile. A random id per install would fragment one person's data into per-device
universes that can never merge, and the failure would be invisible: each device would work fine
alone.
"""
from __future__ import annotations

import uuid
from typing import Optional

from prism.mass import Provenance

# Fixed namespace so the derivation is stable across versions and machines. Changing this
# string re-homes every leaf's local collection — treat it as a wire constant.
_EMBER_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "ember.agience.ai")

# Mantle's discriminator. A collection artifact is an ordinary artifact wearing this type.
# ⚠ ONE DEFINITION, IN THE STORE THAT OWNS THE TYPE (2026-07-31). This was a second literal of
# mantle's own collection content type; two copies of a content type is two ways for a
# collection to stop being findable.
from mantle.db.lattice.constants import COLLECTION_CONTENT_TYPE  # noqa: F401


def local_collection_id(principal: str) -> str:
    """The deterministic id of ``principal``'s local collection artifact.

    Same principal → same id → same region ids → their devices can sync to each other.
    """
    if not principal:
        raise ValueError("a local collection needs a principal to belong to")
    return str(uuid.uuid5(_EMBER_NS, f"local-collection:{principal}"))


def local_collection_artifact(principal: str, name: Optional[str] = None) -> dict:
    """The collection artifact itself, ready to author into a shard or push to Mantle.

    Shaped as Mantle expects (`id` / `content_type` / `context`), so it needs no translation:
    the leaf and the server are describing the same object, not two views of one.

    Provenance is `human_validated`: a person deliberately made this container. That is a real
    claim about origin — unlike the *contents*, which are weighed on their own rungs. A
    collection's mass says who put it there, never whether what's inside is true.
    """
    return {
        "id": local_collection_id(principal),
        "content_type": COLLECTION_CONTENT_TYPE,
        "name": name or "Local",
        "context": {
            "name": name or "Local",
            # A person made this container on purpose; that IS the provenance of the container.
            "provenance": Provenance.HUMAN_VALIDATED.value,
            "origin": "ember",
            "principal": principal,
        },
        "content": "",
    }


__all__ = ["COLLECTION_CONTENT_TYPE", "local_collection_id", "local_collection_artifact"]
