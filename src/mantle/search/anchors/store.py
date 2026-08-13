"""Live AnchorSet store — load from the artifact-backed repo + process cache.

The AnchorSet is a **collection of anchor artifacts** (see :mod:`.repo`). This
module is the process-level cache around it:

- :func:`require_live_anchorset` — mandatory accessor for the index/query path; returns the
  seeded set or raises.

The AnchorSet is loaded by a direct, non-authorizing read (canonical plan §1:
public geometry — no cell keys, no light-cone, no ledger).

**This module loads; it never derives.** A node that has not been seeded holds no AnchorSet,
and there is no code path here or anywhere else in Mantle that would give it one:
:func:`get_live_anchorset` answers ``None`` and :func:`require_live_anchorset` raises. That is
the semantic arm off — writes index lexically, recall answers lexically — until a client loads
a set with ``python -m mantle.system.manage_anchors --action load --path anchors.json``.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

from .anchorset import AnchorSet, AnchorSetCorrupt, anchorset_fingerprint
from .repo import AnchorRepo

logger = logging.getLogger(__name__)

#: Where the geometry this store's cells were written under is recorded. One row, one JSON
#: object — the same `platform_settings` surface `search.reindex.completed_at` uses, for the same
#: reason: it is a small durable fact about this deployment, not content.
_GEOMETRY_KEY = "search.anchorset.indexed_geometry"
_GEOMETRY_CATEGORY = "search"


class AnchorSetDiverged(RuntimeError):
    """The live AnchorSet is not the coordinate system this store's cells were written under.

    Raised only when the two cannot be reconciled at all — a different model space or a
    different width. See :func:`_gate_geometry` for why growth within one space warns instead.
    """


class AnchorSetNotProvisioned(RuntimeError):
    """The AnchorSet is absent and cannot be manufactured locally.

    Lives here rather than beside the seed corpus because this is where the failure happens:
    the routed path asked for the set and there is none.

    This is the state of a node nobody has provisioned, so it is the FIRST state, not a rare
    one. It is not a fault in the caller and retrying cannot clear it: only an operator
    loading anchor artifacts can. The whole semantic arm is behind it — indexing skips the
    vector arm, and recall answers lexical-only — so callers that can survive without the arm
    (`ingest.pipeline_unified`, `init_search.reindex_all_artifacts`) catch this by name, log
    once, and continue.
    """


_lock = threading.RLock()
_cache: Optional[AnchorSet] = None
_loaded = False
_repo_override: Optional[AnchorRepo] = None   # tests inject an InMemoryAnchorRepo
_fingerprint: Optional[str] = None            # fingerprint of the cached set (computed once)
_warned_geometry: Optional[str] = None        # fingerprint already warned about, once per process


def get_anchor_repo() -> AnchorRepo:
    """The active :class:`AnchorRepo` — an injected one (tests) or the
    production the lattice-backed repo over the current request DB handle."""
    if _repo_override is not None:
        return _repo_override
    from mantle.services.dependencies import get_store_db
    from .repo import StoreAnchorRepo
    return StoreAnchorRepo(next(get_store_db()))


def set_anchor_repo(repo: Optional[AnchorRepo]) -> None:
    """Inject the AnchorRepo (tests). Pass ``None`` to restore the default
    (the lattice) repo. Resets the AnchorSet-derived caches."""
    global _repo_override
    with _lock:
        _repo_override = repo
    reset_anchorset()


def get_live_anchorset() -> Optional[AnchorSet]:
    """Return the cached live AnchorSet, loading from the repo once.

    Read-only accessor for inspection. Returns ``None`` until a set is seeded. Index and query
    callers MUST use :func:`require_live_anchorset` instead — the routed path needs the set to
    exist and never has a flat fallback.
    """
    global _cache, _loaded
    if _loaded:
        return _cache
    with _lock:
        if _loaded:
            return _cache
        try:
            _cache = get_anchor_repo().load()
        except AnchorSetCorrupt:
            # NOT swallowed into `None`. `None` means "nobody provisioned this node", and
            # `require_live_anchorset` says so at length — which is the wrong answer, and an
            # actively misleading one, for a node somebody DID provision, incorrectly. A set
            # whose ids do not follow from its contents has no reading under which the semantic
            # arm is safe to run, and one command fixes it.
            _loaded = False
            raise
        except Exception:
            logger.warning("Failed to load AnchorSet from repo", exc_info=True)
            _cache = None
        if _cache is not None:
            logger.info(
                "Loaded live AnchorSet: %d anchors (%s)", len(_cache), _cache.model_id
            )
        _loaded = True
        return _cache


def save_live_anchorset(anchorset: AnchorSet) -> None:
    """Persist every anchor of ``anchorset`` through the repo, then refresh the cache.

    The whole-set path, and the only one: a set arrives whole, from a file the client authored.
    """
    get_anchor_repo().bulk_add(anchorset.anchors)
    reset_anchorset()
    logger.info("Saved live AnchorSet: %d anchors", len(anchorset))


# ── the fingerprint gate ────────────────────────────────────────────────────────────────────────
#
# An anchor id is a cluster id, so the AnchorSet is not a parameter of the index — it is the
# address space the index is written in. Swap the set under an indexed store and every failure
# mode is silent: queries route to regions the writer never produced, cells miss, recall returns
# nothing, and every call still answers 200. These three functions are the check that says so.

def live_fingerprint() -> Optional[str]:
    """:func:`anchorset_fingerprint` of the live set — computed once per load, ``None`` if none.

    Cached with the set: it is a hash over every anchor id, so recomputing it per request would
    be a six-figure loop on the query path for a value that cannot change without a reload.
    """
    global _fingerprint
    aset = get_live_anchorset()
    if aset is None or len(aset) == 0:
        return None
    if _fingerprint is None:
        with _lock:
            if _fingerprint is None:
                _fingerprint = anchorset_fingerprint(aset)
    return _fingerprint


def indexed_geometry() -> Optional[dict]:
    """The geometry this store's cells were written under, or ``None`` if nothing was recorded.

    Read from the store, not from :mod:`platform_settings_service`'s process cache. The record is
    a fact about the cells in *this* store, and a cache is a claim about whichever store filled
    it — believing one store's cache about another store's cells is the same species of mistake
    this gate exists to catch.
    """
    try:
        from mantle.db import identity_backend as identity_store
        from mantle.services.dependencies import get_store_db
        db_gen = get_store_db()
        db = next(db_gen)
        try:
            row = identity_store.get_platform_setting(db, _GEOMETRY_KEY)
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    except Exception:
        return None
    raw = (row or {}).get("value")
    if not raw:
        return None
    try:
        rec = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def _gate_geometry(aset: AnchorSet) -> None:
    """Compare the live AnchorSet against the geometry the cells were written under.

    REFUSE vs WARN. Changing the AnchorSet after indexing orphans every already-written cell, and
    there is no re-cell migration tool in the tree, so the temptation is to refuse any change. It
    is the wrong line: a client seeds a wider set than the one this node was indexed under, and a
    mesh peer syncing anchor artifacts in changes the set with no local operator action at all.
    Both change the fingerprint by construction. A hard refusal on any change would take the arm
    down for the cells that are still perfectly readable, converting stale coverage into no
    service.

    So the line is drawn where the old cells actually become unreachable rather than merely
    incomplete:

    same model_id and dim
        The vocabulary moved within one space. Every previously written cluster id is still a
        valid address, vectors still arrive in the same space, and what is lost is coverage of
        whatever the new anchors now attract. WARN, loudly, once, with both counts — the
        operator's move is a reindex, and they need to know it is owed, not to be locked out.

    a different model_id or dim
        The two sets do not name the same space. No cell written under the old one can be read
        under the new one, and no vector arriving now belongs to the geometry that wrote them.
        REFUSE — there is no partially-correct reading, and continuing writes a second private
        universe on top of the first.

    A store with nothing recorded is a store that has never indexed under any geometry, which is
    not a mismatch; :func:`record_indexed_geometry` claims it on the first cell write.
    """
    global _warned_geometry
    rec = indexed_geometry()
    if not rec:
        return
    fp = live_fingerprint()
    if fp is None or rec.get("fingerprint") == fp:
        return

    was_model, was_dim = rec.get("model_id"), rec.get("dim")
    if was_model != aset.model_id or int(was_dim or -1) != aset.dim:
        raise AnchorSetDiverged(
            f"this store's cells were written under AnchorSet {rec.get('fingerprint')} "
            f"(model {was_model!r}, dim {was_dim}, {rec.get('anchors')} anchors), and the live "
            f"set is {fp} (model {aset.model_id!r}, dim {aset.dim}, {len(aset)} anchors). Those "
            "are different spaces, so every existing cell is unreadable under the live geometry "
            "and every cell written now would be unreadable under the old — the semantic arm "
            "refuses rather than build a second private universe on top of the first. "
            "TO FIX: restore the AnchorSet this store was indexed under, or drop the cells and "
            "reindex against the new one. There is no in-place re-cell."
        )

    if _warned_geometry != fp:
        with _lock:
            _warned_geometry = fp
        logger.warning(
            "AnchorSet changed under an indexed store: cells were written under %s (%s anchors), "
            "the live set is %s (%d anchors), same model %s / dim %d. Already-written cells keep "
            "their cluster ids and stay readable, but nothing has been re-celled, so recall "
            "misses whatever the new anchors now attract. REINDEX to close the gap.",
            rec.get("fingerprint"), rec.get("anchors"), fp, len(aset), aset.model_id, aset.dim,
        )


def record_indexed_geometry(aset: AnchorSet) -> None:
    """Claim (or update) the geometry this store's cells are written under.

    Called from the cell-write path, never from a read: a query must not be able to make a store
    look like it was indexed under whatever set happens to be loaded. Cheap enough for that path —
    a cached fingerprint against one primary-key row read, and a write only when they differ.
    """
    fp = live_fingerprint()
    if fp is None:
        return
    rec = indexed_geometry()
    if rec and rec.get("fingerprint") == fp:
        return
    payload = json.dumps({
        "fingerprint": fp,
        "model_id": aset.model_id,
        "dim": aset.dim,
        "anchors": len(aset),
    }, separators=(",", ":"))
    try:
        from mantle.db import identity_backend as identity_store
        from mantle.services.dependencies import get_store_db
        db_gen = get_store_db()
        db = next(db_gen)
        try:
            identity_store.set_platform_setting(
                db, key=_GEOMETRY_KEY, value=payload, category=_GEOMETRY_CATEGORY,
            )
        finally:
            try:
                next(db_gen)
            except StopIteration:
                pass
    except Exception:
        # Loud, not fatal. Failing the write would take the arm down over a marker; staying
        # silent would leave the node permanently unable to notice its own geometry changing,
        # which is the failure this whole gate exists to remove. So it says so and indexes on.
        logger.warning(
            "Could not record the indexing geometry (%s): this node's cells are being written "
            "under AnchorSet %s and a later change to the set will not be detectable here.",
            _GEOMETRY_KEY, fp, exc_info=True,
        )
        return
    logger.info("Recorded indexing geometry: AnchorSet %s (%d anchors, model %s, dim %d)",
                fp, len(aset), aset.model_id, aset.dim)


def require_live_anchorset() -> AnchorSet:
    """Return the live AnchorSet. Never returns ``None`` — the routed path is the only path.

    Reads the process cache, then the anchor repo. **Raises when the set is absent**: it cannot be
    manufactured here.

    On a node nobody has provisioned this raises every time, which is why the raise carries the
    operator's whole procedure rather than a name. Everything semantic sits on the other side of
    it: :mod:`mantle.search.mantle.indexer` (no cell is written) and
    :mod:`mantle.search.mantle.engine` (no cell is read).
    """
    aset = get_live_anchorset()
    if aset is not None and len(aset) > 0:
        _gate_geometry(aset)
        return aset
    with _lock:
        aset = get_live_anchorset()
        if aset is not None and len(aset) > 0:
            _gate_geometry(aset)
            return aset
        raise AnchorSetNotProvisioned(
            "No AnchorSet is present, so the semantic arm cannot run: artifact writes index "
            "lexically only and recall answers lexically only. This is the state of a node "
            "nobody has seeded yet, not a fault in this call — nothing here derives an "
            "AnchorSet and no later boot will. Anchors fitted to whatever corpus this node "
            "happens to hold would mint region ids no peer computes, so the index would look "
            "healthy and share with nobody; and anchors are vectors, which the no-models rule "
            "says this process never produces. "
            "TO FIX: seed the set your client authored with "
            "`python -m mantle.system.manage_anchors --action load --path anchors.json`, which "
            "preserves every anchor id and verifies each one against its own content. Do not "
            "post anchors through POST /artifacts: an anchor's id IS its cluster id and that "
            "path assigns a fresh uuid4, which routes this node into regions no peer computes "
            "while every call still answers 200. Read the current state with "
            "`python -m mantle.system.manage_anchors --action inspect`, and reindex afterwards "
            "so already-stored artifacts reach the vector cells. See README.md, "
            "'Semantic recall is inert until you seed an AnchorSet'."
        )


def reset_anchorset() -> None:
    """Drop the cached set and its fingerprint (admin reload / tests / after a set is loaded)."""
    global _cache, _loaded, _fingerprint
    with _lock:
        _cache = None
        _loaded = False
        _fingerprint = None
