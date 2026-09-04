"""The context an artifact is minted WITH — the thin arm, always present, standalone.

"Context" already means two things in this store, and this module is the third writer to touch
the word, so it says which one it is before anything else:

    doc['context']            This module. The descriptor recorded at mint time.
                              `search/ingest/pipeline_unified` calls it "conceptually the offer".
    context EDGES             `services/context_service.py`. A traversal structure that ATTENUATES
                              authorization — "a context edge NARROWS; it is never a source of
                              reach". Not this.

They are related — the structural screen below is a natural set of context edges — but they are
written by different code for different reasons, and conflating them is how an offer blob would end
up conferring authority.

Why this exists
===============

Measured on 71/home: every content namespace carries an empty `doc['context']` — `canon:` 0%,
`wiki-` 0%, all three lexicons 0%, `cn-` 0%, over 3,000 rows sampled each. Only `collection:` rows
and the Claude Code hooks' writes have one.

The cause is bulk ingest: 2.9 million corpus vertices went straight into the lattice and never
through `pipeline_unified.index_artifact`. A reindex back-fills the search index, but nothing
back-fills context, because context is not derivable from the row — it is what was true at the
moment of the write, and afterwards it is gone.

The consequences are not subtle and all three were measured:

    no context  ->  no offer, so the lexical arm indexes a fallback heading
    no anchors  ->  `diagram`/`colimit` cannot run (outgoing edges: {})
    no vector   ->  the semantic arm is inert, by contract

Why it is here and not in a tekton
==================================

Because mantle must be useful alone. `search/beacon/__init__.py` states the same rule: beacon is the
permissive half of the two-tier model, so that a store can be taken, built on and shipped by anyone,
with beacon the reduced instrument that makes such a store useful on its own.

Mantle is Apache; chorus is AGPL. Putting an ingest tekton on the critical path of a correct mint
would make an Apache store depend on an AGPL service at the one place that can never be repaired
afterwards. So context minting takes the shape this store already uses twice — `beacon` for the
spectral read, coverage/`_knee` for ranking:

    thin arm, ships here      this module. Complete, not provisional.
    sharp arm, when present   a host fills the `context` seam (astra), and enriches in place.

`search/ranking` states the same rule: a thin arm that is not less correct, and a sharper one when it
is present.

The split, and the one question that decides it
===============================================

**Can the store know this without knowing what the content MEANS?**

DB-LEVEL — recorded here, unconditionally, no domain knowledge:

    minting      when, which acting principal, under which scope
    placement    the collection it landed in, and its origin root
    addressing   the declared content_type, the plaintext SHA-256, the byte length
    screen       what else was present at the mint: the ids co-written in this same call,
                 and the siblings already in the target collection

DOMAIN — behind the seam, never here: format parsing (PDF/HTML), chunking, `lex:en` anchors,
vectors, connector semantics, licences and upstream revision ids, entity extraction.

The screen is worth recording rather than assumed to be: holding a chunk out of a retrieval and
answering only from its neighbours scores MRR **0.328** against **0.973** with it present — and today
that comes from `citation.section` alone, a scrap the canon ingest happened to keep.

What this module will NOT do
============================

* **It does not invent.** A facet it cannot observe is ABSENT, never defaulted. A placeholder would
  make "not recorded" indistinguishable from "recorded as empty", which is the /distinction this
  workspace runs on, pushed down to the data layer.
* **It does not import chorus, astra or ember.** The seam is a name resolved through
  `prism.runner`, exactly as `search/ranking._registered_seam` resolves `match`. A store that
  registered nothing imports nothing.
* **It does not decide authorization.** Nothing here is read by a grant check. See the header.
"""
from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

#: The seam a host fills to enrich a mint. Same registry and same mechanism as `match` / `optics`.
CONTEXT_SEAM = "context"

#: Where enrichment lands. Separate from the db-level facets, and that separation is load-bearing:
#: a store with no seam must be distinguishable from one whose seam ran and found nothing.
ENRICHED_KEY = "enriched"

#: Marks which arm wrote the db-level half, so a reader can tell a thin mint from an enriched one
#: without inferring it from which keys happen to be present.
ARM_KEY = "minted_by"
ARM_THIN = "mantle.mint_context"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _acting() -> Dict[str, Any]:
    """Who is minting, from the ambient acting principal. `{}` when there is none.

    Deliberately does NOT raise. Key custody raises on a missing principal because issuing key
    material without one is unsafe; recording who minted is a description, and a mint that cannot
    name its principal should still record everything else it knows rather than fail closed on the
    least dangerous field.
    """
    try:
        from mantle.services.acting_principal import current_acting_principal
        p = current_acting_principal()
    except Exception:  # noqa: BLE001 — a description must not take the write down
        return {}
    if p is None:
        return {}
    out: Dict[str, Any] = {}
    for attr in ("principal_id", "principal_type", "actor", "scope", "authority", "source"):
        v = getattr(p, attr, None)
        if v:
            out[attr] = str(v)
    return out


def _origin_root(db: Any, collection_id: Optional[str]) -> Optional[str]:
    """The collection's origin root — the sub-tree the artifact actually belongs to.

    `None` rather than a guess when it cannot be resolved. `search/mantle/principal` refuses to
    substitute `collection_id` for an unresolved root because the two derive different keys; the
    same refusal applies to describing placement, for the weaker reason that a wrong root here is a
    false statement about where the artifact lives.
    """
    if not collection_id or db is None:
        return None
    try:
        from mantle.db import backend as db_store
        return db_store.get_origin_root(db, collection_id) or None
    except Exception:  # noqa: BLE001
        return None


def _siblings(db: Any, collection_id: Optional[str], *, limit: int) -> Optional[Sequence[str]]:
    """Ids already in the target collection — the part of the screen the store can see.

    Bounded, and the bound is REPORTED (`screen.siblings_capped`) rather than silently applied: a
    truncated screen that does not say it was truncated is a measurement claiming to be complete.
    """
    if not collection_id or db is None:
        return None
    # This must be a bounded peek, and the first two drafts were not.
    #
    # Draft 1 called a `list_collection_member_ids` that does not exist; the `except` turned that
    # into a silently absent screen — a mint that looked complete and recorded half of what it
    # claimed. Draft 2 used `lattice_api._membership_edges`, which is `edges_of` at its default
    # limit, and that put an O(collection) read on the store's write path. Measured 2026-08-24 on
    # 71/home:
    #
    #     stage.0.lexicon   1,846,820 edges   41.48s   -> 0 ids (it hits the cap and refuses)
    #     stage.1.grammar     188,322 edges    4.30s   -> 65 ids
    #
    # A 41-second write into a large collection is the exact failure `pipeline_unified` names about
    # indexing the body: "write cost a function of the corpus, which is the one thing this system
    # cannot have." A description of the write may never be more expensive than the write.
    #
    # `edges_of` already has the right door and says so: "``partial_ok=True`` is for callers that
    # genuinely want a bounded peek and have said so." `edge(src, label)` is indexed, so a bounded
    # peek is a limit on an index walk regardless of how large the collection is. A truncated answer
    # is the answer this caller wants, and `siblings_capped` reports it.
    try:
        edges = db.graph.edges_of(collection_id, direction="out",
                                  limit=max(1, int(limit)), partial_ok=True)
    except Exception:  # noqa: BLE001 — a store that cannot answer records no screen, honestly
        return None
    out = []
    for e in edges or []:
        dst = e.get("dst") if isinstance(e, dict) else None
        if dst:
            out.append(str(dst))
    return out


def mint(
    db: Any = None,
    *,
    artifact_id: str,
    content_type: Optional[str] = None,
    content: Optional[str] = None,
    collection_id: Optional[str] = None,
    co_written: Optional[Iterable[str]] = None,
    caller: Optional[Mapping[str, Any]] = None,
    sibling_limit: int = 64,
) -> Dict[str, Any]:
    """The db-level context for one mint. Complete on its own; enriched in place if a seam is filled.

    `caller` is whatever the writer already knows and the store cannot — a source path, a connector,
    a session id. It is recorded verbatim under `caller` and never merged into the db-level facets,
    so a reader can always tell what the store OBSERVED from what a caller ASSERTED.

    Every facet is omitted when unobservable. There are no defaults.
    """
    # Idempotent, so the mint can be wired at every door without double-stamping.
    # The router covers every external writer, but an in-process caller reaches the service
    # functions directly — and that is not a hypothetical hole, it is how 2.9M vertices got into
    # this store with no context at all (`index_namespace.py`: "bulk-ingested straight into the
    # lattice"). So the service primitives stamp too, and a context already carrying this arm's
    # mark passes through untouched rather than being re-derived with a later timestamp.
    if isinstance(caller, Mapping) and caller.get(ARM_KEY) == ARM_THIN:
        return dict(caller)

    ctx: Dict[str, Any] = {ARM_KEY: ARM_THIN}

    minted = {"at": _now()}
    minted.update(_acting())
    ctx["minted"] = minted

    placement: Dict[str, Any] = {}
    if collection_id:
        placement["collection_id"] = str(collection_id)
        root = _origin_root(db, collection_id)
        if root:
            placement["origin_root"] = root
    if placement:
        ctx["placement"] = placement

    addressing: Dict[str, Any] = {}
    if content_type:
        addressing["content_type"] = str(content_type)
    if content is not None:
        raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        # The hash is of the PLAINTEXT, matching how the store addresses content
        # (`cas/<sha256(plaintext)>`), so a context read and a CAS read agree on what
        # "this content" means.
        addressing["sha256"] = hashlib.sha256(raw).hexdigest()
        addressing["bytes"] = len(raw)
    if addressing:
        ctx["addressing"] = addressing

    # ── the structural screen ──────────────────────────────────────────────────────────────────
    # "Everything on the screen when the artifact was created" is, for a STORE, the other rows of
    # this same write and the collection it landed in. Both are in hand at mint time and neither
    # needs to know what the content means.
    screen: Dict[str, Any] = {}
    co = [str(x) for x in (co_written or []) if x and str(x) != str(artifact_id)]
    if co:
        screen["co_written"] = co
    sibs = _siblings(db, collection_id, limit=sibling_limit)
    if sibs is not None:
        capped = len(sibs) > sibling_limit
        kept = [s for s in sibs[:sibling_limit] if str(s) != str(artifact_id)]
        if kept:
            screen["siblings"] = kept
        if capped:
            screen["siblings_capped"] = sibling_limit
    if screen:
        ctx["screen"] = screen

    if caller:
        ctx["caller"] = dict(caller)

    _stamp_rung(ctx)

    enriched = _enrich(ctx, artifact_id=artifact_id, content_type=content_type,
                       content=content, db=db)
    if enriched:
        ctx[ENRICHED_KEY] = enriched
    return ctx


#: The operator's way to name a host that fills the `context` seam, exactly as
#: `MANTLE_ONTOLOGY_HOST` names the one that fills `match`. Same shape, same reason.
_CONTEXT_HOST_ENV = "MANTLE_CONTEXT_HOST"
#: Import-once state, so a node whose host module is missing logs once rather than per write.
_HOST_TRIED: Optional[str] = None


def _load_context_host() -> None:
    """Import the operator-named host module, at most once per process.

    A re-mint comes back with `enriched: 0%` unless something registers a host: the seam resolves
    to nothing in the mantle server when nothing in that process has ever registered one — and
    nothing can from inside mantle itself, since mantle may not import a tekton, and a tekton may
    not import another tekton (`agience-chorus`: "Tektons never import one another"), so neither
    side can do the wiring.

    The operator does it, which is the same answer `search/ranking` already reached:
    "`MANTLE_ONTOLOGY_HOST=ember` is the operator's way to put one there, and importing what it
    names is the entire mechanism."

    Failure is logged and swallowed. A node configured to load a context host it does not have must
    keep minting thin-and-complete, not refuse writes.
    """
    global _HOST_TRIED
    target = (os.getenv(_CONTEXT_HOST_ENV) or "").strip()
    if not target or _HOST_TRIED == target:
        return
    _HOST_TRIED = target
    try:
        import importlib
        importlib.import_module(target)
        from prism.runner import register_seam
        register_seam(CONTEXT_SEAM, target)
        logger.info("%s=%s imported; the context seam is filled", _CONTEXT_HOST_ENV, target)
    except Exception as exc:  # noqa: BLE001 — a bad module name must not break writes
        logger.warning("%s=%s could not be loaded (%s: %s); mints stay thin",
                       _CONTEXT_HOST_ENV, target, type(exc).__name__, exc)



def _stamp_rung(ctx: Dict[str, Any]) -> None:
    """Stamp the provenance rung this write earned, server-side, from the principal.

    The store has no mass. Measured on 71/home: not one vertex carries `mass > 0` — of 2.17M rows,
    only the 5,484 colimit objects have the field at all, and every one reads `0.0`. So
    `colimit._mass` sums zeros and "fewer objects, more mass each" is arithmetic over nothing.

    Carrying a rung is a write-time property: `prism.mass.stamp()` supplies the mechanism, and a
    write path has to call it. This is the write path that does, which is why the rung arrives here
    rather than being reconstructed later — the same shape as the context finding this module exists
    for, one field over.

    The rung is derived here rather than accepted from a caller. Provenance is a claim about origin
    and claims need an authority, so the rung is derived server-side from the authenticated context
    and a client's self-declared rung is ignored: a raw client `POST` is forced to `UNKNOWN`. This
    function runs inside the store, holding the authenticated acting principal, which is the only
    place that derivation can be made.

    `prism.mass.authorize_and_stamp` is *"the one call a server write path makes"*. This is that
    path, so this is that call.

    It does NOT invent mass. It records the channel the write earned; corroboration accumulates
    within a band and never re-rungs. A store that has verified nothing still reads low, correctly.
    """
    try:
        from prism.mass import authorize_and_stamp
    except Exception:  # noqa: BLE001 — a store without prism.mass records no rung, honestly
        return
    minted = ctx.get("minted") or {}
    ptype = str(minted.get("principal_type") or "unknown")
    # Delegation is a fact about the principal, not a flag a caller sets: an acting principal
    # carrying an `actor` distinct from itself is acting FOR someone.
    actor = minted.get("actor")
    pid = minted.get("principal_id")
    is_delegated = bool(actor and pid and str(actor) != str(pid))
    try:
        stamped = authorize_and_stamp(dict(ctx), principal_type=ptype, is_delegated=is_delegated)
        if isinstance(stamped, Mapping):
            for k, v in stamped.items():
                if k not in ctx:
                    ctx[k] = v
    except Exception as exc:  # noqa: BLE001
        logger.warning("rung stamping failed (%s: %s); the mint records no rung rather than a "
                       "guessed one", type(exc).__name__, exc)

def _enrich(base: Mapping[str, Any], **kw: Any) -> Optional[Dict[str, Any]]:
    """The sharp arm, if a host registered one. `None` when none did — which is not a failure.

    Resolved by NAME through `prism.runner`, so mantle's static import graph names no tekton. A
    seam that raises is logged once and skipped: an enrichment service being down must not fail a
    write whose db-level context is already complete.
    """
    _load_context_host()
    try:
        from prism.runner import registered_seams
        target = (registered_seams() or {}).get(CONTEXT_SEAM)
    except Exception:  # noqa: BLE001
        return None
    if not target:
        return None
    try:
        import importlib
        mod = importlib.import_module(target)
        fn = getattr(mod, "enrich", None)
        if fn is None:
            return None
        got = fn(dict(base), **kw)
        return dict(got) if isinstance(got, Mapping) and got else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("context seam %s did not enrich (%s: %s); the thin mint stands",
                       target, type(exc).__name__, exc)
        return None


def facets(ctx: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    """Which db-level facets a context actually carries. The gate's unit of measurement.

    Reads what is THERE. It does not judge whether a facet ought to have been observable for this
    artifact — that is the caller's question, and a checker that guessed at it would report a
    correctly-absent facet as a defect.
    """
    c = ctx or {}
    scr = c.get("screen") or {}
    return {
        "minted": bool((c.get("minted") or {}).get("at")),
        "placement": bool(c.get("placement")),
        "addressing": bool((c.get("addressing") or {}).get("sha256")),
        "screen": bool(scr.get("co_written") or scr.get("siblings")),
        "enriched": bool(c.get(ENRICHED_KEY)),
    }
