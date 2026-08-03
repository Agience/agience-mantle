"""`mantle.db.lattice` — the lattice store (LATTICE Phase 1).

A hardened SQLite store behind the `mantle.db.store` ABCs. It promotes
`agience-ember/src/ember/sqlite_store.py` and deletes the three things that made
that seed unsafe:

  * `_SqliteConnShim`, which pattern-matched the legacy graph engine SQL strings and **returned
    `[]` for anything unrecognised** — six live wrong-answer defects (contract §5).
    Replaced by typed methods, where a drifted call site is an AttributeError.
  * `_rev = time.time_ns()`, a clock that is not injective (2000 calls -> 1 distinct
    value on Windows' 15.6ms tick). Replaced by `(_origin, _seq)` proper time.
  * `count(*)` and `LIMIT ? OFFSET ?`. Replaced by incremental counters and keyset
    pagination.

Typical wiring — ONE file, ONE connection, so an artifact and its edges commit
atomically:

    from mantle.db.lattice import open_lattice
    store = open_lattice("/var/lib/ember/lattice.db", origin="node-71")
    store.artifacts.put_artifact({"id": "a1", "content_type": "text/markdown"})
    store.graph.add_edges([("a1", "a2", "lineage", {"is_origin": 1})])

`origin` is this observer's stable identity and MUST be pinned across restarts.
A node that changes its origin forks its own proper time, and every peer then sees
two unrelated, permanently-unordered event streams.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional

from .constants import (CAS_PREFIX, CRUDEASIO, DEFAULT_LEAVES, EDGE_FORCES,
                        NEWER, NULL_AUDIT_FIELDS, OLDER, SAME, UNORDERED,
                        VERSION_ORDERS, NullAuditField, compare_version,
                        edge_key, is_missing, leaf_of, row_hash)
from .edge import LatticeGraphStore
from .schema import ALL_DDL, ensure_schema
from .seq import (LatticeConn, SeqAllocator, allocator_for, check_tail,
                  counter_of, high_water, max_observed, seq_accounting)
from .vertex import LatticeArtifactStore, ListIndexUnbuilt

__all__ = [
    "CAS_PREFIX", "CRUDEASIO", "DEFAULT_LEAVES", "EDGE_FORCES",
    "NEWER", "OLDER", "SAME", "UNORDERED", "VERSION_ORDERS",
    "NULL_AUDIT_FIELDS", "NullAuditField", "is_missing",
    "compare_version", "edge_key", "leaf_of", "row_hash",
    "LatticeArtifactStore", "LatticeGraphStore", "LatticeConn", "SeqAllocator",
    "ListIndexUnbuilt",
    "allocator_for", "check_tail", "high_water", "max_observed",
    "seq_accounting",
    "ensure_schema", "ALL_DDL", "Lattice", "open_lattice",
]


class Lattice(NamedTuple):
    """The two stores plus the connection they share."""
    artifacts: LatticeArtifactStore
    graph: LatticeGraphStore
    db: LatticeConn
    origin: str


def open_lattice(path: str, *, origin: str,
                 leaves: Optional[int] = None) -> Lattice:
    """Open (or create) a lattice at `path`.

    Both stores share ONE `LatticeConn` and ONE `SeqAllocator`, which matters for
    two reasons: vertices and edges commit in a single transaction, and they draw
    from a SINGLE proper-time sequence — one counter per OBSERVER, spanning both
    tables (contract §4 RESOLVED-5). Two allocators over one origin hand out the
    same `_seq` twice, which destroys the uniqueness of `(_origin, _seq)`.

    THE MERKLE LEAF COUNT IS RESOLVED ONCE AND SHARED. `leaves=None` (the normal
    case) reads the store's recorded operating resolution (`merkle.leaves`), or
    falls back to `DEFAULT_LEAVES` for a store that has never recorded one — the
    DERIVED `natural_leaves(corpus)` is then reached by `reshard()`, keeping both
    tables' `_leaf` in lockstep. An explicit `leaves=` overrides (tests / tools).

    Sharing is now automatic: `allocator_for` reuses the registered allocator, so
    constructing the two stores directly on one `LatticeConn` is equally safe.
    This helper remains the recommended entry point."""
    db = LatticeConn(path)
    with db.write() as cur:            # ensure the schema exists so the meta read below is valid
        ensure_schema(cur)
    if leaves:
        resolved = int(leaves)
    else:
        row = db.read().execute("SELECT v FROM meta WHERE k = 'merkle.leaves'").fetchone()
        resolved = int(row["v"]) if row and row["v"] is not None else DEFAULT_LEAVES
    arts = LatticeArtifactStore(db, origin=origin, leaves=resolved)
    graph = LatticeGraphStore(db, origin=origin, leaves=resolved)
    return Lattice(artifacts=arts, graph=graph, db=db, origin=origin)
