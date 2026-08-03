# db/store.py
# The ArtifactStore / GraphStore seam (Step 0).
#
# All artifact document CRUD and all edge/traversal graph work go through these
# two interfaces so the backend is a
# swappable detail. This module is DELIBERATELY import-light — stdlib + typing
# only — so a leaf node (Ember) can embed a backend in-process without pulling
# the FastAPI app. Mantle-the-service adopts the same seam during cutover.
#
# The artifact document shape mirrors entities/artifact.py (id, root_id,
# collection_id, context, content, state, content_type, created_by/time, ...).
# `context` is the OFFER field — what the artifact advertises — and is what
# retrieval indexes on; `content` is the payload retrieval returns.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional

# ⭐ THE ONE PLACE THE DRIVER TYPE IS NAMED.
#
# `Database` is the handle type threaded through the service and router layers. It
# was `store.database.StandardDatabase`, imported directly at ~380 call sites —
# which meant every signature in the codebase was typed against a specific product's
# driver, and swapping the backend would have been a 380-site edit rather than a
# one-line one.
#
# Aliased here, in the seam, because the seam is what those call sites are really
# depending on: "a handle to the artifact store". Importing this module costs
# nothing at runtime and a leaf node (Ember) can embed a backend without any
# driver installed.
#
# When a different backend replaces the lattice, this alias changes and the call sites do
# not.
#
# 2026-07-22: the TYPE_CHECKING import of `store.database.StandardDatabase` is
# gone with the python-store dependency (dead since the store flip). `Database`
# is now `Any` for the type checker too — the honest type until the lattice
# backend's handle type is named here. No call-site signature changes: every
# consumer already annotated against this alias, which is the point of the seam.
Database = Any


class ContentStore(ABC):
    """Opaque byte storage (any S3-compatible endpoint for a node that wants S3 semantics, or
    filesystem for the minimal node). Mantle encrypts BEFORE calling put; the
    store only ever sees ciphertext (label-blind — the mesh property)."""

    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class ArtifactStore(ABC):
    """Document CRUD over artifacts (and, during Mantle cutover, the other
    document collections). Backend-agnostic. Documents are plain dicts in the
    Artifact.to_dict() shape."""

    @abstractmethod
    def ensure_schema(self) -> None: ...

    @abstractmethod
    def put_artifact(self, doc: Dict[str, Any], *, stamp_rev: bool = True) -> Dict[str, Any]:
        """Idempotent upsert keyed on `id`. Returns the stored doc.

        `stamp_rev=True` stamps a fresh `_rev` for the mesh UPDATE change-feed. Mesh CONSUME
        passes `stamp_rev=False` to PRESERVE the origin's `_rev`, so a replicated doc keeps the
        rev it was published with and does not echo around the mesh forever.

        ⚠ THE KEYWORD IS NOT OPTIONAL IN PRACTICE — it was simply missing from this declaration.
        Every production caller passes it (`worker.py`, and ~10 sites in `mesh/sync.py`), and both
        real backends accept it, so the ABC was the only thing that was wrong. A new backend
        written honestly against this signature would raise `TypeError` on the consume path — and
        `reconcile_merkle` SWALLOWS that into `applied: 0`, i.e. replication silently applies
        nothing while reporting a clean zero. That is not hypothetical: it is exactly the outage
        the sqlite backend shipped with until 2026-07-20."""

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def list_artifacts(
        self,
        *,
        state: Optional[str] = None,
        content_type: Optional[str] = None,
        collection_id: Optional[str] = None,
        created_by: Optional[str] = None,
        limit: Optional[int] = None,
        skip: int = 0,
    ) -> Iterator[Dict[str, Any]]: ...

    @abstractmethod
    def count(self, *, state: Optional[str] = None) -> int: ...

    @abstractmethod
    def delete_artifact(self, artifact_id: str) -> None: ...

    @abstractmethod
    def put_many(self, docs: Iterable[Dict[str, Any]], *, batch: int = 500,
                 stamp_rev: bool = True) -> int:
        """Bulk upsert. `stamp_rev` as in `put_artifact`; `batch` sizes the per-transaction chunk.

        ⚠ **RETURNS THE NUMBER *HANDLED*, NOT THE NUMBER *WRITTEN*.** This used to say "the
        number stored", and the difference is load-bearing: `mesh.sync._apply_artifacts` uses
        this return as its ONLY guard against advancing the consume cursor —

            written = store.artifacts.put_many(batch, batch=500, stamp_rev=False)
            if written < len(batch): raise RuntimeError("partial apply: ...")

        — and a segment recorded as applied advances `last_key` behind a monotone `StartAfter`
        marker, so anything not written is GONE. "Handled" therefore has a precise meaning that
        every implementation must honour:

          * a doc WRITTEN                      -> counts
          * a doc correctly REJECTED by LWW    -> counts (declining to overwrite a newer local
            row IS the right outcome; it is handled, not lost)
          * a doc that ERRORED                 -> MUST NOT COUNT

        An implementation that returns `len(docs)` unconditionally silently disables the mesh's
        only data-loss guard from inside the store layer.

        ⚠ This declaration previously omitted BOTH keywords while every production call site
        passed them — see `put_artifact` for what that cost. The legacy adapter carried a comment
        saying "Widen `store.py` first" since the trap was found; this is that widening."""


class GraphStore(ABC):
    """Typed edges + traversals. Artifacts are also projected as lightweight
    vertices keyed on `id`; traversals return ids and callers hydrate full
    records from the ArtifactStore (mirrors the legacy graph store's `edges -> artifacts/{id}`).

    `label` is the edge kind — e.g. 'offers' (operator -> content-type it
    describes), 'lineage' (version/derivation), 'collection' (membership)."""

    @abstractmethod
    def ensure_schema(self) -> None: ...

    @abstractmethod
    def add_edge(
        self, from_id: str, to_id: str, label: str, props: Optional[Dict[str, Any]] = None
    ) -> None: ...

    @abstractmethod
    def add_edges(self, edges: Iterable[Any], *, batch: int = 500) -> int:
        """Bulk edge upsert. `edges` is an iterable of `(from_id, to_id, label, props)`.

        ⚠ DECLARED BECAUSE IT IS THE PATH PRODUCTION ACTUALLY USES — `mesh/sync.py`,
        `ingest.py` and three sites in `genesis.py` all write edges through this, while the
        singular `add_edge` above described the path nothing uses. The legacy graph store even
        implements the declared singular form by delegating to this undeclared bulk one.

        **MUST BE IDEMPOTENT.** Mesh segments are replayed — consume is retried on any held
        cursor — so re-adding an existing edge must UPDATE it in place, not append. The legacy
        backend enforces this with a UNIQUE `edge_key`; a backend that issues a plain INSERT
        accumulates duplicate edges without bound on every replay. The two shipped backends
        already disagree on exactly this point, which is the kind of divergence a declared seam
        exists to prevent."""

    @abstractmethod
    def neighbors(
        self, node_id: str, label: Optional[str] = None, *, direction: str = "out"
    ) -> List[str]:
        """Return neighbor ids. direction in {'out','in','both'}."""

    @abstractmethod
    def descendants(
        self, root_id: str, label: str, *, direction: str = "out"
    ) -> List[str]:
        """Transitive reachable ids (the PRUNE/BFS analogue — the light-cone
        primitive; on a single-owner leaf there is no authorization prune).

        ⛔ `max_depth=64` REMOVED 2026-07-30. `seen` includes the root and every vertex is
        admitted once over a finite graph, so the BFS always drains — the cap was a bare claim
        that nothing 65 hops away matters. This is the primitive every light-cone sits on, and
        the same lattice was described as 4, 10, 25, 32 and 64 deep across five files.
        """
