"""AnchorRepo — the AnchorSet's persistence as artifacts.

An anchor **is an artifact** (`vnd.agience.anchor+json`); the **AnchorSet is a
collection** of them (slug ``agience-anchorset``). The geometry layer loads
anchors by a **direct, non-authorizing the lattice read** (canonical plan §1: no cell
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

from .anchorset import Anchor, AnchorSet

logger = logging.getLogger(__name__)

# Provenance for platform-created anchor artifacts. `created_by` is provenance
# only (no access, no ownership — Phase 1 decoupled cell keys from it).
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
        """Persist many anchors (best-effort per anchor)."""

    def count(self) -> int:
        """Number of stored anchors."""


def _build_anchorset(anchors: List[Anchor]) -> Optional[AnchorSet]:
    """Assemble an :class:`AnchorSet` from anchors (first one fixes model/dim)."""
    if not anchors:
        return None
    aset = AnchorSet(model_id=anchors[0].model_id, dim=anchors[0].embedding.shape[-1])
    for a in anchors:
        try:
            aset.add(a)
        except ValueError:
            # dim/model mismatch — a foreign-model anchor slipped in; skip it
            # rather than corrupt the set (cross-walks bridge models elsewhere).
            logger.debug("AnchorRepo: skipped anchor %s (model/dim mismatch)", a.label)
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

    def bulk_add(self, anchors: List[Anchor]) -> None:
        for a in anchors:
            self.add(a)

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
        from mantle.services.bootstrap_types import ANCHORSET_COLLECTION_SLUG
        from mantle.services.platform_topology import get_id_optional
        return get_id_optional(ANCHORSET_COLLECTION_SLUG)

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
        return _build_anchorset(anchors)

    def add(self, anchor: Anchor) -> None:
        cid = self._ensure_collection_id()
        from mantle.db import backend as db_store
        from mantle.entities.artifact import Artifact
        from mantle.services.bootstrap_types import ANCHOR_CONTENT_TYPE

        # Idempotent: deterministic id means re-adding the same anchor is a no-op
        # on the doc; we still ensure the membership edge.
        exists = False
        try:
            exists = db_store.get_artifact(self._db, anchor.anchor_id) is not None
        except Exception:
            exists = False
        if not exists:
            artifact = Artifact(
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
            db_store.create_artifact(self._db, artifact)
        db_store.add_artifact_to_collection(self._db, cid, anchor.anchor_id, origin=True)

    def bulk_add(self, anchors: List[Anchor]) -> None:
        for a in anchors:
            try:
                self.add(a)
            except Exception:
                logger.warning("AnchorRepo: failed adding anchor %s", a.label, exc_info=True)

    def count(self) -> int:
        aset = self.load()
        return len(aset) if aset else 0
