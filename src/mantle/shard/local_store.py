"""Ember's local store handle — the SQLite lattice + FS content, reached through Mantle.

**Mantle is the one data-access path.** `open_store()` returns the lattice store
(`mantle.db` + `FileContentCache`) via `sqlite_store.open_sqlite_store()`. There is no
other backend: SQLite + `FsContentStore` is the only one.

The store is the durable truth (artifacts, offers, edges). Ember's anchor/IVF index is a derived,
rebuildable cache on top of it — delete the index, rebuild from the store.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:                       # names for annotations only — never imported at runtime
    from mantle.db.store import ArtifactStore, GraphStore, ContentStore
else:
    ArtifactStore = GraphStore = ContentStore = Any


@dataclass
class LocalStore:
    """The three store faces of the local lattice shard, bound and schema-ready."""
    artifacts: ArtifactStore
    graph: GraphStore
    content: ContentStore
    conn: Optional[Any] = None           # always None on the lattice backend — see `ready()`
    keys_dir: Optional[Path] = None      # for the content-encryption key (content.py)
    # The READ-seam tier (mantle.db.content_tier.TieredContentStore): FileContentCache in
    # front of the S3/CDN mirror, plaintext surface. Deliberately a SEPARATE handle from `content`:
    # the legacy write path (content.put_content) hands `content` CIPHERTEXT, and the tier's
    # plaintext-verifying surface would loudly refuse it. Readers (resolve_text) prefer this when
    # present; writers keep the surface they have. None ⇒ local-only (air-gapped is first-class).
    content_tier: Optional[Any] = None

    def ready(self) -> bool:
        """Is the store actually usable? The lattice is a local file opened in-process — there is no
        server to ping — so ask the store: a keyed read that can raise. A store whose file is gone,
        locked or corrupt says so rather than returning a blind `True` (see contract §5 on
        unmeasured-rendered-as-healthy)."""
        try:
            self.artifacts.get_artifact("__ready_probe__")   # keyed lookup: no scan, no count(*)
            return True
        except Exception:
            return False


def open_store(keys_dir: Optional[Path] = None, *, ensure_schema: bool = True) -> LocalStore:
    """Open the local store — the SQLite lattice + FS content, via Mantle (the one data path).

    SQLite + `FsContentStore` is the sole backend, so this is a thin alias for
    `sqlite_store.open_sqlite_store()`. The store resolves its own keys dir from the environment
    and ensures its own schema.

    Same defect class as `wn_store._arts()` binding an ambient store: a parameter that is accepted
    and discarded is worse than no parameter, because the call site reads as if it were honoured.
    So `open_store()` refuses a `keys_dir` argument outright: no caller passes it — every one
    passes only `ensure_schema=` — so refusing costs nothing and closes the trap. To point at a
    different store, set `EMBER_SQLITE_DIR` / `EMBER_SQLITE_DB`, which is what actually decides.

    `ensure_schema` is likewise not consulted (the lattice always ensures its schema), but it is
    still accepted because real callers pass it and its stated meaning is what happens anyway."""
    from mantle.shard.sqlite_store import open_sqlite_store
    if keys_dir is not None:
        raise TypeError(
            "open_store() does not take a keys_dir/path — accepting and silently ignoring it "
            "would let a caller believe they have an isolated store when they are reading the "
            "process-default one. Set EMBER_SQLITE_DIR / EMBER_SQLITE_DB (and "
            "EMBER_STORE_KEYS_DIR) instead; got %r"
            % (keys_dir,))
    return open_sqlite_store()
