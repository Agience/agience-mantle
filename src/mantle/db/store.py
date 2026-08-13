# db/store.py
# The ArtifactStore / GraphStore seam (Step 0).
#
# All artifact document CRUD and all edge/traversal graph work go through these
# two interfaces so the backend is a
# swappable detail. This module is deliberately import-light — stdlib + typing
# only — so a leaf node (Ember) can embed a backend in-process without pulling
# the FastAPI app. Mantle-the-service uses the same seam.
#
# The artifact document shape mirrors entities/artifact.py (id, root_id,
# collection_id, context, content, state, content_type, created_by/time, ...).
# `context` is the OFFER field — what the artifact advertises — and is what
# retrieval indexes on; `content` is the payload retrieval returns.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Iterator, List, Optional

#
# `Database` is the handle type threaded through the service and router layers.
# Aliasing it here, in the seam, lets every call site depend on "a handle to the
# artifact store" rather than a specific backend's driver type — importing this
# module costs nothing at runtime, and a leaf node (Ember) can embed a backend
# without any driver installed.
#
# `Database` is `Any` because no concrete handle type is named here yet; when one
# is, this alias changes and the call sites do not.
Database = Any


class ContentStore(ABC):
    """Opaque byte storage (any S3-compatible endpoint for a node that wants S3 semantics, or
    filesystem for the minimal node). Mantle encrypts before calling put; the
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

        `stamp_rev=True` stamps a fresh `_rev` for the mesh update change-feed. Mesh consume
        passes `stamp_rev=False` to preserve the origin's `_rev`, so a replicated doc keeps the
        rev it was published with and does not echo around the mesh forever."""

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def get_many(self, artifact_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Plural `get_artifact` — `{id: doc}` for the ids that exist.

        The read a batching caller needs. Without it a page of `n` ids costs `n` round
        trips, and a caller that wants one query has to hand-roll a `WHERE id IN (...)`
        against the backend's own tables — the probe this seam exists to make unnecessary.
        The plural belongs here, on the interface, so every backend answers it the same way.

        Ids are deduplicated: a repeated id costs one read and yields one entry. An id with
        no row is ABSENT from the mapping rather than mapped to `None` or `{}`, so "not
        stored" and "stored empty" stay different answers.

        An implementation must not put the caller's whole id list into one statement. Every
        backend has a bind-parameter or statement-size ceiling — SQLite's is 999 host
        parameters on builds before 3.32 and 32766 after, i.e. a property of whichever
        library the process happens to link, not of the request. So the read is chunked
        internally and `get_many` of 5,000 ids is a supported call, never a ceiling the
        caller discovers at runtime."""

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

            written = store.artifacts.put_many(batch, batch=500, stamp_rev=False)
            if written < len(batch): raise RuntimeError("partial apply: ...")

        — and a segment recorded as applied advances `last_key` behind a monotone `StartAfter`
        marker, so anything not written is gone. "Handled" therefore has a precise meaning that
        every implementation must honour:

          * a doc written                      -> counts
          * a doc correctly rejected by LWW    -> counts (declining to overwrite a newer local
            row is the right outcome; it is handled, not lost)
          * a doc that errored                 -> must not count

        An implementation that returns `len(docs)` unconditionally silently disables the mesh's
        only data-loss guard from inside the store layer."""


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

        It must be idempotent: mesh segments are replayed — consume is retried on any held
        cursor — so re-adding an existing edge has to update it in place, not append. The legacy
        backend enforces this with a unique `edge_key`; a backend that issues a plain INSERT
        accumulates duplicate edges without bound on every replay. The two shipped backends
        disagree on exactly this point, which is the kind of divergence a declared seam exists
        to prevent."""

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
        """
