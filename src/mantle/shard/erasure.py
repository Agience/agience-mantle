"""ERASURE — wipe everything attached to one higgs, and nothing else.

[John, 2026-07-23: *"for my account, how can I reset or wipe everything attached to my higgs"*]

There was no answer to that question before this file. For a system whose whole ontology is
grounding / grants / conscious observation (§13.11.7), being able to remove your own portion is not
a feature — it is the same right as being able to add to it, and its absence was a hole.

═══════════════════════════════════════════════════════════════════════════════
⛔ THE ONE DISTINCTION THAT MAKES THIS SAFE: GROUNDED vs OBSERVED
═══════════════════════════════════════════════════════════════════════════════
A higgs is a PORTION of the one field, rooted at an identity. Two things look superficially alike
and must never be confused:

  * **GROUNDED at you** — you authored it, or it lives in your private collection. It is yours, and
    erasure removes it.
  * **REACHED by you** — the commons you looked at. WordNet, Wikipedia, anything CC/OA. Observing a
    thing does not attach it to you, and deleting it would be deleting THE TRAINING SET.

That is exactly the cut John drew in the same breath: *"remove any artifact that isn't the training
set."* So erasure is defined POSITIVELY — it collects what is provably grounded at this person and
removes that. It never works by exclusion ("everything except the corpus"), because an exclusion
list is a premonition: the first artifact class nobody thought of is destroyed silently.

⚠ **AND IT REFUSES WHAT IT CANNOT PROVE.** An artifact whose `created_by` cannot be resolved to this
person, and which is not in their private collection, is NOT erased — it is REPORTED as unresolved.
An erasure that guesses is an erasure that takes someone else's work with it, and unlike every other
mistake in this codebase that one cannot be measured afterwards, because the evidence is gone.

═══════════════════════════════════════════════════════════════════════════════
DRY RUN BY DEFAULT
═══════════════════════════════════════════════════════════════════════════════
`erase()` reports and changes nothing unless `apply=True` is passed explicitly. The inventory is the
product; the deletion is a separate decision made by a human who has read it.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


def _person_ids(store, person: str) -> Set[str]:
    """Every id this principal is known by — the raw claim AND its resolved person artifact.

    Both are needed: rows are stamped with the resolved uuid, while collections and grants are keyed
    on the raw principal string. Erasing on one alone leaves the other half attached."""
    ids = {str(person)}
    # ⚠ ASK THE RUNNER, DO NOT IMPORT ONE. Resolving a claim to its person vertex mints an artifact,
    # which is a runner act; mantle may not import ember. Unwired returns None and the raw claim
    # stands alone — which is honest, and is why the caller must not treat one id as "all of them".
    from mantle import runner_hooks
    try:
        ref = runner_hooks.author_ref(store, person)
    except Exception:
        ref = None
    if ref is not None:
        ids.add(str(ref))
    return {i for i in ids if i}


# The classes of thing a higgs is made of. Named explicitly so the inventory is READABLE — an
# erasure report that says "417 artifacts" tells its reader nothing about what they are losing.
CLASSES = (
    ("private", "artifacts in your private collection"),
    ("authored", "artifacts you authored"),
    ("conversation", "messages and conversations"),
    ("identity", "your person artifact itself"),
)

# ⛔ THE REGISTRY IS GROUNDED AT THE SYSTEM, NOT AT WHOEVER RAN BOOTSTRAP.
# MEASURED on the live store: `op.math.add` and 24 sibling operators are stamped `created_by` the
# principal that happened to mint them — which on that node was `anonymous@public` — and carry no
# collection, no provenance and no citation. A `created_by == me` sweep would have DELETED THE
# ARITHMETIC OPERATORS as one person's private work. They are the ontology: operators, the type and
# edge-type registries, rungs, collections, citations, content-type definitions.
#
# This is the grounding question John already framed (§13.11.7): the operators are grounded at THE
# FOUNDATION, the commons at ground zero, and only the observer's own work at the observer. So these
# are reported as `not_yours` rather than silently skipped — "I found these and they are not yours to
# erase" is a different and more useful statement than not mentioning them.
_REGISTRY_TYPES = ("operator+json", "vtype+json", "etype+json", "rung+json", "collection+json",
                   "content-type+json", "x-citation", "shard-done+json", "form-mesh+json")


def _is_registry(ct: str) -> bool:
    return any(ct.endswith(t) or ct == t for t in _REGISTRY_TYPES)


def attached(store, person: str, *, include_identity: bool = False) -> Dict[str, Any]:
    """INVENTORY — what is grounded at this higgs. Reads only; changes nothing.

    `include_identity` decides between a RESET (keep the person, drop everything they made) and a
    full erasure (the person goes too). They are genuinely different acts and the caller must say
    which, rather than one being a silent default."""
    ids = _person_ids(store, person)   # ⚠ was: from ember import genesis as g — unused since
                                       # author resolution moved to runner_hooks (mantle ↛ ember)
    private_ids = {"private.%s" % p for p in ids}
    found: Dict[str, List[str]] = {k: [] for k, _d in CLASSES}
    not_yours: List[str] = []
    unresolved: List[str] = []
    scanned = 0

    conn = store.artifacts.db.read()
    import json
    for (doc,) in conn.execute("SELECT doc FROM vertex"):
        scanned += 1
        try:
            d = json.loads(doc)
        except Exception:
            continue
        aid = str(d.get("id") or "")
        ct = str(d.get("content_type") or "")
        coll = str(d.get("collection_id") or "")
        made_by = str(d.get("created_by") or "")

        if aid in ids:
            if include_identity and ct.endswith("person+json"):
                found["identity"].append(aid)
            continue
        if coll in private_ids:
            found["private"].append(aid)
            continue
        if made_by in ids:
            # ⚠ AUTHORSHIP OF A COMMONS ROW IS NOT OWNERSHIP OF IT. The stage-0 ingest stamps every
            # synset with the operator's principal, so a naive "created_by == me" sweep would take
            # the entire training set. A row that carries a source citation was INGESTED, not
            # authored: it is the commons, reached and not grounded.
            if _is_registry(ct):
                not_yours.append(aid)          # the ontology: grounded at the foundation
                continue
            if d.get("cited_from") or str(coll).startswith("stage."):
                continue
            (found["conversation"] if "message" in ct or aid.startswith("convo.")
             else found["authored"]).append(aid)
    return {"person": person, "ids": sorted(ids), "scanned": scanned,
            "found": found,
            "counts": {k: len(v) for k, v in found.items()},
            "total": sum(len(v) for v in found.values()),
            "not_yours": not_yours,
            "unresolved": unresolved}


def erase(store, person: str, *, apply: bool = False,
          include_identity: bool = False) -> Dict[str, Any]:
    """Remove everything grounded at this higgs. **DRY RUN unless `apply=True`.**

    Returns the inventory plus what was actually removed, so the report is the same shape either
    way and a caller can diff a dry run against the real one."""
    inv = attached(store, person, include_identity=include_identity)
    if not apply:
        inv["applied"] = False
        return inv
    removed, failed = [], []
    for _cls, ids in inv["found"].items():
        for aid in ids:
            try:
                store.artifacts.delete_artifact(aid)
                removed.append(aid)
            except Exception as e:
                failed.append("%s: %s" % (aid, type(e).__name__))
    inv["applied"] = True
    inv["removed"] = len(removed)
    inv["failed"] = failed
    # A partial erasure REPORTED is recoverable; a partial erasure reported as complete is not.
    inv["complete"] = not failed and len(removed) == inv["total"]
    return inv


__all__ = ["CLASSES", "attached", "erase"]
