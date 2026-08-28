"""AnchorRepo — the AnchorSet's persistence as artifacts.

An anchor **is an artifact** (`vnd.agience.anchor+json`); the **AnchorSet is a
collection** of them (slug ``agience-anchorset``). The geometry layer loads
anchors by a **direct, non-authorizing read of the lattice** (canonical plan §1: no cell
keys, no light-cone, no oracle — anchors are public geometry), builds the
in-memory :class:`AnchorSet`, and caches it. There is no JSON-file store.

Two implementations:
- :class:`StoreAnchorRepo` — production; backs onto the artifact store.
- :class:`InMemoryAnchorRepo` — tests; keeps the geometry suite db-free.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import List, Optional, Protocol

from .anchorset import Anchor, AnchorSet, AnchorSetCorrupt, verify_anchor_id

logger = logging.getLogger(__name__)

# Provenance for platform-created anchor artifacts. `created_by` is provenance
# only — it carries no access or ownership.
_ANCHOR_CREATED_BY = "agience-mantle"

# Fallback namespace for deriving the AnchorSet collection id when the instance
# has no signing identity (pure standalone). The id only needs to be stable
# within a deployment — it's persisted on first create and reloaded thereafter.
_ANCHORSET_FALLBACK_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "anchorset.mantle.agience")


class AnchorRepo(Protocol):
    """Persistence boundary for the AnchorSet."""

    def load(self) -> Optional[AnchorSet]:
        """Build the live AnchorSet from stored anchors, or ``None`` if empty."""

    def add(self, anchor: Anchor) -> None:
        """Persist one anchor artifact (idempotent on the anchor's id)."""

    def bulk_add(self, anchors: List[Anchor]) -> None:
        """Persist many anchors. Raises if any anchor could not be written."""

    def count(self) -> int:
        """Number of stored anchors."""


#: How many offending anchors an assembly failure names before summarising the rest.
_REPORTED_OFFENDERS = 5

#: Anchors written between progress callbacks during a bulk load.
_PROGRESS_EVERY = 500


def _build_anchorset(anchors: List[Anchor]) -> Optional[AnchorSet]:
    """Assemble an :class:`AnchorSet` from anchors (the first one fixes model/dim).

    A mixed-width or mixed-model set is REFUSED, not thinned. The width and the model come from
    whichever anchor happened to be read first, so "skip the ones that disagree" resolves a file
    holding two vocabularies into whichever of them the store listed first — a different
    AnchorSet on a node whose listing order differs, with a different fingerprint, addressing a
    different set of cells, and no error anywhere. There is no reading under which some of a
    canonical set is the canonical set.
    """
    if not anchors:
        return None
    aset = AnchorSet(model_id=anchors[0].model_id, dim=anchors[0].embedding.shape[-1])
    bad: List[str] = []
    for a in anchors:
        try:
            aset.add(a)
        except ValueError as e:
            bad.append(f"{a.label!r}: {e}")
    if bad:
        raise AnchorSetCorrupt(
            f"{len(bad)} of {len(anchors)} stored anchors do not belong to one set "
            f"(model {aset.model_id!r}, dim {aset.dim}). "
            + " | ".join(bad[:_REPORTED_OFFENDERS])
            + (f" | ... and {len(bad) - _REPORTED_OFFENDERS} more"
               if len(bad) > _REPORTED_OFFENDERS else "")
            + " -- one collection is one space. TO FIX: reset the anchorset collection and load "
              "a single set with `python -m mantle.system.manage_anchors --action load`."
        )
    return aset if len(aset) else None


# ---------------------------------------------------------------------------
# In-memory (tests)
# ---------------------------------------------------------------------------

class InMemoryAnchorRepo:
    """Dict-backed AnchorRepo — keeps the geometry tests db-free."""

    def __init__(self) -> None:
        self._anchors: dict[str, Anchor] = {}

    def load(self) -> Optional[AnchorSet]:
        return _build_anchorset(list(self._anchors.values()))

    def add(self, anchor: Anchor) -> None:
        self._anchors[anchor.anchor_id] = anchor  # idempotent on id

    def bulk_add(self, anchors: List[Anchor], *, progress=None) -> None:
        for i, a in enumerate(anchors, start=1):
            self.add(a)
            if progress is not None:
                progress(i, len(anchors))

    def count(self) -> int:
        return len(self._anchors)


# ---------------------------------------------------------------------------
# the lattice (production)
# ---------------------------------------------------------------------------

class StoreAnchorRepo:
    """Backs the AnchorSet onto the artifact store: anchors are
    ``vnd.agience.anchor+json`` artifacts in the ``agience-anchorset`` collection.

    Loading is a direct read (non-authorizing) — the geometry layer never goes
    through the light-cone or the oracle.
    """

    def __init__(self, db) -> None:
        self._db = db
        self._cid: Optional[str] = None   # memoized AnchorSet collection id

    def _collection_id(self) -> Optional[str]:
        """The AnchorSet collection id for READING — resolved, never created.

        The registry lookup alone is not a resolution. `get_id_optional` answers only from this
        process's topology registry, so a fresh CLI run whose registry has not loaded the slug gets
        `None` and `load()` reports an empty AnchorSet, while the service whose registry did load it
        routes on the same collection: `count()` saying 0, a bulk load saying 37 and `/status`
        saying 21 are three processes reading one store, none of them lying.

        The derived id is the tie-breaker, because every process computes it from the same inputs —
        `uuid5(instance_namespace, "agience", slug)` — and it is what `_ensure_collection_id` falls
        back to when it writes. A read resolves the same way the write path does.

        This resolves without creating. `_ensure_collection_id` is the write path's helper and it
        registers, persists and mints a collection if none exists, so calling it from `load()` would
        make a read create an empty AnchorSet as a side effect. This derives the same id and returns
        it only if that collection exists, and `None` otherwise, which is the answer for "nothing
        has written an AnchorSet here yet".
        """
        from mantle.db import backend as db_store
        from mantle.services.bootstrap_types import ANCHORSET_COLLECTION_SLUG
        from mantle.services.platform_topology import get_id_optional

        cid = get_id_optional(ANCHORSET_COLLECTION_SLUG)
        if cid and db_store.get_collection_by_id(self._db, cid) is not None:
            return cid
        try:
            # The same two imports `_ensure_collection_id` derives with, so the read and the
            # write cannot disagree about which collection the slug names.
            from mantle.services.peer_signing import get_instance_namespace
            from mantle.services.seed_provisioning.loader import derive_uuid

            ns = get_instance_namespace() or _ANCHORSET_FALLBACK_NS
            derived = derive_uuid(ns, "agience", ANCHORSET_COLLECTION_SLUG)
            if derived and db_store.get_collection_by_id(self._db, derived) is not None:
                return derived
        except Exception:
            logger.debug("AnchorRepo: could not derive the AnchorSet collection id",
                         exc_info=True)
        return None

    def _ensure_collection_id(self) -> str:
        """Return the AnchorSet collection id, CREATING the collection (and
        registering its slug) on first use.

        The AnchorSet *is* a collection, so the geometry layer provisions its own
        home lazily — no platform seed required. This is generic, public,
        non-authorizing geometry infrastructure (the shared coordinate system for
        vector search), which is why the database engine may create it itself,
        unlike platform *content* (agents, personas, LLM connections) that stays
        an application/Origin concern. Mirrors the runtime create-if-missing +
        persist-slug pattern used for the People collection.
        """
        if self._cid:
            return self._cid

        from mantle.services.bootstrap_types import ANCHORSET_COLLECTION_SLUG
        from mantle.services.platform_topology import get_id_optional, register_id
        from mantle.db import backend as db_store

        cid = get_id_optional(ANCHORSET_COLLECTION_SLUG)
        if cid and db_store.get_collection_by_id(self._db, cid) is not None:
            self._cid = cid
            return cid

        # Derive a stable id (same convention as the platform seed run, so a
        # later platform seed is idempotent), register the slug, create the
        # collection if missing, and persist the mapping for future boots.
        from mantle.services.peer_signing import get_instance_namespace
        from mantle.services.seed_provisioning.loader import derive_uuid, _persist_seed_ids

        ns = get_instance_namespace() or _ANCHORSET_FALLBACK_NS
        cid = cid or derive_uuid(ns, "agience", ANCHORSET_COLLECTION_SLUG)
        register_id(ANCHORSET_COLLECTION_SLUG, cid)
        register_id(f"agience/{ANCHORSET_COLLECTION_SLUG}", cid)

        if db_store.get_collection_by_id(self._db, cid) is None:
            from datetime import datetime, timezone
            from mantle.entities.collection import (
                Collection as CollectionEntity,
                COLLECTION_CONTENT_TYPE,
            )
            now = datetime.now(timezone.utc).isoformat()
            db_store.create_collection(self._db, CollectionEntity(
                id=cid,
                name="AnchorSet",
                description=(
                    "MANTLE geometry anchors — the shared coordinate system for "
                    "vector search. Public, non-authorizing geometry."
                ),
                created_by=_ANCHOR_CREATED_BY,
                content_type=COLLECTION_CONTENT_TYPE,
                state=CollectionEntity.STATE_COMMITTED,
                created_time=now, modified_time=now,
            ))
            _persist_seed_ids(self._db, {ANCHORSET_COLLECTION_SLUG: cid})
            logger.info("Created AnchorSet collection %s at runtime", cid)
        self._cid = cid
        return cid

    def clear(self) -> int:
        """Remove every anchor artifact from the set. Returns how many were removed.

        A set that only grows is a trap: `bulk_add` adds, so a seed you regret is permanent short
        of store surgery. An anchor id is the cluster id — it names the cell path, the HKDF `info`,
        the AEAD associated data and the mesh region — so a set carrying regions nobody wanted
        mis-routes every query afterwards while every call answers 200. Loading a corrected
        16-anchor set over an existing 21 produces 37, with the superseded regions still routing.

        Deletes rather than archives, and it is the one place in this repo where that is right: an
        archived artifact still occupies its cluster id, and the id is the thing being reclaimed.
        """
        # `_ensure_collection_id` rather than `_collection_id`. The latter is a registry lookup only
        # (`get_id_optional`) and answers None in any process whose topology registry has not loaded
        # the slug, so `count()` reports 0 in a fresh CLI run while the collection sits right there,
        # and a clear keyed on it would remove nothing and report success. The resolving path falls
        # back to the stable derived id (`uuid5` of the instance namespace),
        # which is the same collection every process computes.
        cid = self._ensure_collection_id()
        if not cid:
            return 0
        from mantle.db import backend as db_store
        from mantle.services.bootstrap_types import ANCHOR_CONTENT_TYPE
        try:
            docs = db_store.list_collection_artifacts(self._db, cid)
        except Exception:
            logger.warning("AnchorRepo: failed listing AnchorSet %s for clear", cid, exc_info=True)
            return 0
        removed = 0
        for d in docs:
            if d.get("content_type") != ANCHOR_CONTENT_TYPE:
                continue
            aid = d.get("id") or d.get("_key")
            if not aid:
                continue
            try:
                db_store.delete_artifact(self._db, str(aid))
                removed += 1
            except Exception:
                logger.warning("AnchorRepo: could not remove anchor %s", aid, exc_info=True)
        return removed

    def load(self) -> Optional[AnchorSet]:
        cid = self._collection_id()
        if not cid:
            return None
        from mantle.db import backend as db_store
        from mantle.services.bootstrap_types import ANCHOR_CONTENT_TYPE
        try:
            docs = db_store.list_collection_artifacts(self._db, cid)
        except Exception:
            logger.warning("AnchorRepo: failed listing AnchorSet %s", cid, exc_info=True)
            return None
        anchors: List[Anchor] = []
        unreadable: List[str] = []
        for d in docs:
            if d.get("content_type") != ANCHOR_CONTENT_TYPE:
                continue
            ctx = d.get("context")
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except (TypeError, ValueError):
                    continue
            if not isinstance(ctx, dict):
                continue
            aid = d.get("root_id") or d.get("id") or d.get("_key")
            if not aid:
                continue
            try:
                anchors.append(Anchor.from_context(str(aid), ctx))
            except Exception:
                logger.debug("AnchorRepo: skipped malformed anchor doc %s", aid)
        # The id an anchor artifact carries is the CLUSTER id — it names the cell path, the HKDF
        # `info`, the AEAD associated data and the mesh region. Provisioning through
        # POST /artifacts mints a fresh uuid4 for the artifact, so an anchor loaded here can
        # perfectly well state an id its own content does not produce, and every downstream use
        # accepts an arbitrary string. That is the whole silent failure: the node routes
        # confidently into regions no peer computes, queries miss, sync transfers nothing, and
        # every call returns 200. Nothing else in the read path can tell, so it is told here.
        for a in anchors:
            reason = verify_anchor_id(a.anchor_id, a.label, a.model_id, a.embedding)
            if reason is not None:
                unreadable.append(reason)
        if unreadable:
            raise AnchorSetCorrupt(
                f"{len(unreadable)} of {len(anchors)} anchors in collection {cid} carry an id "
                "their content does not produce, so this node's cells would be addressed by ids "
                "no holder of the same canonical set computes. "
                + " | ".join(unreadable[:_REPORTED_OFFENDERS])
                + (f" | ... and {len(unreadable) - _REPORTED_OFFENDERS} more"
                   if len(unreadable) > _REPORTED_OFFENDERS else "")
                + " -- TO FIX: load the canonical set with "
                "`python -m mantle.system.manage_anchors --action load --path PATH`, which "
                "preserves ids; POST /artifacts assigns a fresh uuid4 and cannot."
            )
        return _build_anchorset(anchors)

    def _artifact_for(self, anchor: Anchor, cid: str):
        from mantle.entities.artifact import Artifact
        from mantle.services.bootstrap_types import ANCHOR_CONTENT_TYPE
        return Artifact(
            id=anchor.anchor_id,
            root_id=anchor.anchor_id,
            collection_id=cid,
            content="",
            context=json.dumps(anchor.to_context(), separators=(",", ":")),
            state=Artifact.STATE_COMMITTED,
            created_by=_ANCHOR_CREATED_BY,
            name=anchor.label,
            content_type=ANCHOR_CONTENT_TYPE,
        )

    def add(self, anchor: Anchor) -> None:
        cid = self._ensure_collection_id()
        from mantle.db import backend as db_store

        # Idempotent: deterministic id means re-adding the same anchor is a no-op
        # on the doc; we still ensure the membership edge.
        exists = False
        try:
            exists = db_store.get_artifact(self._db, anchor.anchor_id) is not None
        except Exception:
            exists = False
        if not exists:
            db_store.create_artifact(self._db, self._artifact_for(anchor, cid))
        db_store.add_artifact_to_collection(self._db, cid, anchor.anchor_id, origin=True)

    def bulk_add(self, anchors: List[Anchor], *, progress=None) -> None:
        """Persist many anchors — one batched existence probe, then only the missing writes.

        A canonical set is six figures of anchors, so the existence check is `get_raw_artifacts`
        over the whole id list rather than a read per anchor; re-running a load then costs one
        probe and no writes. Failures are collected and raised together: a half-written AnchorSet
        is the silent state this module exists to prevent, and a per-anchor warning in a log is
        indistinguishable from success to whoever ran the load.

        ``progress`` is an optional ``(done, total) -> None`` callback for large sets.
        """
        if not anchors:
            return
        cid = self._ensure_collection_id()
        from mantle.db import backend as db_store

        try:
            present = set(db_store.get_raw_artifacts(self._db, [a.anchor_id for a in anchors]) or {})
        except Exception:
            logger.debug("AnchorRepo: batched existence probe unavailable; falling back per anchor",
                         exc_info=True)
            present = None

        failed: List[str] = []
        total = len(anchors)
        for i, a in enumerate(anchors, start=1):
            try:
                if present is None:
                    self.add(a)
                else:
                    if a.anchor_id not in present:
                        db_store.create_artifact(self._db, self._artifact_for(a, cid))
                    db_store.add_artifact_to_collection(self._db, cid, a.anchor_id, origin=True)
            except Exception as e:
                failed.append(f"{a.label!r} ({a.anchor_id[:12]}): {e}")
            if progress is not None and (i % _PROGRESS_EVERY == 0 or i == total):
                progress(i, total)
        if failed:
            raise AnchorSetCorrupt(
                f"{len(failed)} of {total} anchors could not be written, so this node holds a "
                "PARTIAL AnchorSet — a different set from the canonical one, with a different "
                "fingerprint and a different set of cells. "
                + " | ".join(failed[:_REPORTED_OFFENDERS])
                + (f" | ... and {len(failed) - _REPORTED_OFFENDERS} more"
                   if len(failed) > _REPORTED_OFFENDERS else "")
            )

    def count(self) -> int:
        aset = self.load()
        return len(aset) if aset else 0
