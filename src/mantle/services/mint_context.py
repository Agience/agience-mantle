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
import json
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


# ── who owns which top-level key ──────────────────────────────────────────────────────────────
#
# A stored context has two authors. The store writes the facets below — the mint's record, the
# rung it derived, the content address and the upload bookkeeping — and a client writes the rest.
# A client never writes a server-owned key: every door that accepts a client's context removes
# them (`client_part`), and every write keeps the stored server-owned facets and replaces only the
# client's keys (`with_client_keys`, `merge_caller_update`).
#
# Whether a context was minted is never inferred from its JSON. A client can send any key it
# likes, `minted_by` included, so a mint the router has already made reaches the service as a
# `Minted` — an object, which no request body can construct.
#: What the mint writes: how the artifact was made, and the rung it earned. Always the mint's.
MINT_KEYS = frozenset({
    ARM_KEY, "minted", "placement", "addressing", "screen", ENRICHED_KEY, "caller",
    "provenance", "provenance_history", "provenance_evidence",
})
#: What the store's own write paths record about the bytes and their upload. A client never
#: writes these; an in-process writer (the upload path, the content routes) does. `access`,
#: `content_source` and `uri` are among them because the download-URL answer reads them: a
#: client-written `access: "public"` with a `uri` would send every reader to an address the
#: client chose, as though it were the artifact's own download.
BOOKKEEPING_KEYS = frozenset({
    "content_key", "content_cas_ref", "content_sha256", "content_bytes", "upload", "processing",
    "storage", "access", "content_source", "uri",
})
SERVER_OWNED_KEYS = MINT_KEYS | BOOKKEEPING_KEYS
#: The one server-owned key a client may SEND on a create: `provenance` there is the claim
#: `_stamp_rung` authorizes against the principal (`prism.mass.authorize_and_stamp`). It is not
#: stored as sent: the top-level `provenance` is the derived rung, and a claim the principal had
#: not earned is kept in `provenance_history`.
CLAIMABLE_ON_CREATE = frozenset({"provenance"})


class Minted:
    """A context `mint` made in this process, handed from the router to a service by reference.

    A typed parameter rather than ambient request state: the handoff is visible in every
    signature it crosses, it cannot leak into a nested write the request happens to make, and a
    test can pass one directly. JSON cannot construct it, which is the whole point — a string or a
    dict arriving in a request body is a client's claim, however it is spelled."""

    __slots__ = ("_ctx",)

    def __init__(self, ctx: Mapping[str, Any]) -> None:
        self._ctx = dict(ctx)

    def context(self) -> Dict[str, Any]:
        return dict(self._ctx)


def parse_context(value: Any) -> Dict[str, Any]:
    """A context as a dict: a dict as given, JSON parsed, anything else kept whole under
    `_opaque` rather than dropped (it is the writer's statement about its own artifact)."""
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {"_opaque": value}
    return dict(parsed) if isinstance(parsed, dict) else {"_opaque": value}


def client_part(value: Any, *, allow: Iterable[str] = ()) -> Dict[str, Any]:
    """The keys of a client-supplied context a client may write: every server-owned key removed,
    except the ones named in ``allow``."""
    keep = frozenset(allow)
    return {k: v for k, v in parse_context(value).items()
            if k not in SERVER_OWNED_KEYS or k in keep}


# ── the client's keys: stored once, at the top level ────────────────────────────────────────
#
# A stored context holds the client's keys at the top level, beside the server-owned facets, and
# nowhere else. Every write of a client key goes through `with_client_keys`, which puts them
# there and removes any `caller`:
#
#     create (router)              `mint`                       client keys from the request
#     create (in-process member)   `workspace_service._mint_member_context`
#     PATCH, identity update,      `merge_caller_update`        the request's context
#       identity upsert overwrite                                (`update_keys`)
#     every in-process context     `normalised`                 upload status (incl.
#       write                                                    `context_patch`), bindings,
#                                                                card actions, PUT /content
#
# `caller` is reserved: the store reads it and never writes it, and a client key of that name is
# removed like any other server-owned key. A stored context may hold client keys under `caller`
# with none at the top level, or some in each; `client_view` reads the top level, and `caller`
# for a key the top level does not hold. An update is read the same way against the context it
# updates (`update_keys`): its `caller` supplies only a key that neither its own top level nor
# the stored top level holds.


def client_view(value: Any) -> Dict[str, Any]:
    """The client's current keys in a stored context: the top level's, and `caller`'s for a key
    the top level does not hold, the server-owned keys removed."""
    ctx = parse_context(value)
    caller = ctx.get("caller")
    nested = client_part(caller) if isinstance(caller, Mapping) else {}
    return {**nested, **client_part(ctx)}


def with_client_keys(stored: Any, keys: Mapping[str, Any]) -> str:
    """``stored`` with its client keys replaced by ``keys``: the stored server-owned facets as
    they were, less any `caller`, and ``keys`` (less any server-owned ones) at the top level."""
    out = {k: v for k, v in parse_context(stored).items()
           if k in SERVER_OWNED_KEYS and k != "caller"}
    out.update(client_part(keys))
    return json.dumps(out)


def normalised(value: Any) -> Any:
    """A context written in-process, with its client keys at the top level and no `caller`. What
    the writer set at the top level wins over a `caller` it did not touch (`client_view`).

    A context that is not a JSON object (an opaque string) has no keys to place and is returned
    as given, and so is one already in this shape, so a write that changed nothing stays a
    write that changed nothing."""
    ctx = parse_context(value)
    if set(ctx) == {"_opaque"}:
        return value
    out = with_client_keys(ctx, client_view(ctx))
    if isinstance(value, str) and json.loads(out) == ctx:
        return value
    return out


def update_keys(update: Any, stored: Any = None) -> Dict[str, Any]:
    """The client keys a context update states, read against the context it updates (``stored``).

    ======================================  ==================================================
    the update's top level holds a client   its top-level client keys, and each client key of
    key                                     its `caller` that neither its top level nor
                                            ``stored``'s top level holds
    it holds none                           the client keys of its `caller`
    ======================================  ==================================================

    A `caller` that is not a JSON object states no keys, and a key the update states nowhere is
    not stated. So a key ``stored`` holds at the top level is decided by the update's top level
    alone, and `caller` supplies a key only where ``stored`` has none at the top level: on a
    stored context, a key it holds under `caller` alone."""
    upd = parse_context(update)
    top = client_part(upd)
    caller = upd.get("caller")
    nested = client_part(caller) if isinstance(caller, Mapping) else {}
    if not top:
        return nested
    held = client_part(stored)
    return {**{k: v for k, v in nested.items() if k not in held}, **top}


def merge_caller_update(stored: Any, update: Any) -> str:
    """A client's context update applied to a stored context.

    The stored server-owned facets stay exactly as they were, and the client's keys are replaced
    wholesale by the ones the update states (`update_keys`, read against ``stored``): a key it
    does not state is removed.
    They are stored at the top level, with no `caller` (`with_client_keys`).

    Nothing is re-minted and no rung is re-derived: an update records what the client changed
    and nothing it asked for about how the artifact was made. The rung derived at create stands,
    with a claim the principal had not earned in `provenance_history`."""
    return with_client_keys(stored, update_keys(update, stored))


def with_content_type(stored: Any, content_type: str) -> Any:
    """``stored`` with its client `content_type` key set to ``content_type``, when it holds one.

    A write that applies an artifact's content type as its own field keeps the context's key in
    agreement with it: the content routes read the key (`workspace_service._store_content_in_s3`
    types the stored object by it). A context with no such key, or with this value already, is
    returned as given; one that changes is stored with its client keys at the top level
    (`with_client_keys`)."""
    ctx = parse_context(stored)
    keys = client_view(ctx)
    if "content_type" not in keys or keys["content_type"] == content_type:
        return stored
    keys["content_type"] = content_type
    return with_client_keys(ctx, keys)


def readdressed(stored: Any, content: Any = None, *, sha256: Optional[str] = None,
                nbytes: Optional[int] = None) -> str:
    """A stored context after its content was replaced.

    `addressing` describes the bytes the mint saw, so a content overwrite recomputes its
    `sha256` and `bytes` — otherwise one context would name two contents, the mint's and the one
    `content_cas_ref` points at. From the new plaintext (``content``, text or bytes) when the
    writer holds it; otherwise from what the writer measured (``sha256``, ``nbytes``), and a
    measure it does not have is removed rather than left describing the old bytes.

    `enriched` is DROPPED rather than recomputed: it is the sharp arm's description of the old
    bytes, recomputing it needs the seam (`_enrich`) and a request that may not have one, and an
    absent `enriched` reads as "no enrichment for these bytes", which is true. A context the store
    never minted has no `addressing` and is returned as it was, less any `enriched`.

    The client keys are stored at the top level with no `caller` (`with_client_keys`), as on every
    write."""
    ctx = parse_context(stored)
    ctx.pop(ENRICHED_KEY, None)
    addressing = ctx.get("addressing")
    if isinstance(addressing, dict):
        addressing = dict(addressing)
        if content is not None:
            raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
            sha256, nbytes = hashlib.sha256(raw).hexdigest(), len(raw)
        for key, value in (("sha256", sha256), ("bytes", nbytes)):
            if value is None:
                addressing.pop(key, None)
            else:
                addressing[key] = value
        ctx["addressing"] = addressing
    return with_client_keys(ctx, client_view(ctx))


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
    except Exception:  # noqa: BLE001 — a store that cannot answer records no screen at all
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
    a session id. Its keys are written at the top level beside the facets, less the server-owned
    ones (`client_part`). Its `provenance` is the claim the rung is derived against
    (`_stamp_rung`), not a key stored as sent. No client key ever replaces a db-level facet, so a
    reader can always tell what the store OBSERVED from what a caller ASSERTED.

    Every facet is omitted when unobservable. There are no defaults.
    """
    # Every call mints. The router mints each create and hands the result to the service as a
    # `Minted`; a service primitive reached in-process without one mints here — which is what
    # keeps a bulk ingest that skips the router from landing with no context at all. `caller`'s
    # keys are recorded as given, minus the server-owned keys (`client_part`): a dict that carries
    # `minted_by` is a client's claim, not a mint.
    claim = client_part(caller, allow=CLAIMABLE_ON_CREATE).get("provenance") if caller else None

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
        ctx.update(client_part(caller))

    _stamp_rung(ctx, claim)

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



def _stamp_rung(ctx: Dict[str, Any], claim: Any = None) -> None:
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
    path, so this is that call. It reads the claim from a context's top-level `provenance`, so
    ``claim`` (the caller's, if it sent one) is handed to it there, and the principal is the
    authenticated acting principal recorded under `minted`. The earned channel lands at the top
    level (`provenance`), with the claim kept in `provenance_history` when the two differ.

    It does NOT invent mass. It records the channel the write earned; corroboration accumulates
    within a band and never re-rungs. A store that has verified nothing still reads low, correctly.
    """
    try:
        from prism.mass import authorize_and_stamp
    except Exception:  # noqa: BLE001 — a store without prism.mass records no rung at all
        return
    minted = ctx.get("minted") or {}
    ptype = str(minted.get("principal_type") or "unknown")
    # Delegation is a fact about the principal, not a flag a caller sets: an acting principal
    # carrying an `actor` distinct from itself is acting FOR someone.
    actor = minted.get("actor")
    pid = minted.get("principal_id")
    is_delegated = bool(actor and pid and str(actor) != str(pid))
    try:
        stamped = authorize_and_stamp({"provenance": claim} if claim else {},
                                      principal_type=ptype, is_delegated=is_delegated)
        if isinstance(stamped, Mapping):
            for k in ("provenance", "provenance_history"):
                if k in stamped:
                    ctx[k] = stamped[k]
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
