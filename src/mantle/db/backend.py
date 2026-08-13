"""The store backend — one import point for the routers/services.

The lattice is the store: one SQLite file + FS-CAS content, opened in-process, zero external DB
processes. This module delegates every attribute to `db.lattice_api`; the envelope crypto +
change-event chokepoint lives in `db.doc_boundary`.
"""
from __future__ import annotations

from mantle.db import lattice_api as _impl
from mantle.db.constants import state_of

_HANDLE = None                            # lattice mode: one store handle per process


def store_handle():
    """The process-wide lattice handle (opened once; SQLite schema created on open)."""
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = _impl.open_database()    # MANTLE_LATTICE_PATH / MANTLE_ORIGIN env
    return _HANDLE


def init_store():
    """Startup store initialization — the boot chokepoint. Opens the process handle;
    there is nothing else to create (the schema rides the open)."""
    return store_handle()


_CONTENT = None                           # the tiered content handle, opened once per process


def content_handle():
    """The process-wide content handle — local CAS first, an S3/CDN mirror behind it when set.

    A mantle shard is a local CAS file store that can also be backed by S3, local or global;
    S3 is just a CDN. `TieredContentStore` reads a local `FileContentCache` first, consults a
    remote only on a miss and verifies it against the ref, and treats `remote=None` as a
    first-class configuration rather than a degraded one.

    Returns None when this node genuinely has no local content tier — an index-only node is a legal
    shape. Callers must treat None as "this node cannot serve content", never as "not found":
    the two are different answers and only one of them is about the request.

    A missing `content.key` is not a missing tier: `_open_lattice_content` returns None when the
    key is absent, which here would read as "index-only node" for what is actually a mis-provisioned
    one. The key is checked first and refused loudly, because a missing key is a silent partition
    (see `content_encryption`) and the whole point of this layer is that it fails where the cause is.
    """
    global _CONTENT
    if _CONTENT is not None:
        return _CONTENT

    import os
    from pathlib import Path

    # The default comes from `lattice_api`, which takes it from `config` — spelling it again here
    # would let the content tier and the store it belongs to point at two different directories.
    db_path = Path(os.path.abspath(os.path.expanduser(
        os.getenv("MANTLE_LATTICE_PATH", str(_impl.DEFAULT_LATTICE_PATH)))))
    root = db_path.parent
    keys_dir = Path(os.getenv("KEYS_DIR") or (root / "keys"))

    keyfile = keys_dir / "content.key"
    if not keyfile.exists():
        raise RuntimeError(
            "no content.key under %s — this node cannot encrypt or read stored content. That is a "
            "PROVISIONING fault, not an empty store: reading on would treat every blob as missing "
            "and report 'no results' for content that is present and unreadable." % keys_dir)

    from mantle.shard.sqlite_store import _open_content_tier, _open_lattice_content
    cache = _open_lattice_content(str(root), str(keys_dir), str(db_path))
    _CONTENT = _open_content_tier(cache, str(keys_dir))
    return _CONTENT


def get_raw_artifact(db, artifact_id: str):
    """Raw artifact doc by id (the one raw-doc read shape for call sites)."""
    return db.artifacts.get_artifact(artifact_id)


def get_raw_artifacts(db, artifact_ids):
    """Raw artifact docs by id — `{id: doc}`, the plural of `get_raw_artifact`.

    One chunked read for the whole set (see `ArtifactStore.get_many`), so a page of ids is one
    store call rather than one per id. Ids are deduplicated; ids with no row are absent from the
    mapping, so a miss stays a missing key rather than an empty doc."""
    return db.artifacts.get_many(artifact_ids)


def find_newest_by_root(db, root_id: str):
    """Newest non-archived version row for a root (proper-time order)."""
    rows = [v for v in (db.artifacts.versions_of(root_id) or [])
            if state_of(v) != "archived"]
    return rows[-1] if rows else None          # proper-time order: last = newest


def find_newest_by_roots(db, root_ids):
    """`{root_id: newest non-archived version row}` — the plural of `find_newest_by_root`.

    Every named lineage in one chunked read (`ArtifactStore.versions_of_many`) rather than one
    read per root. A root with no versions, and a root whose every version is archived, are both
    absent from the mapping — the same answer `find_newest_by_root` gives as `None`."""
    out = {}
    for root_id, rows in (db.artifacts.versions_of_many(root_ids) or {}).items():
        live = [v for v in rows if state_of(v) != "archived"]
        if live:
            out[root_id] = live[-1]            # proper-time order: last = newest
    return out


def find_version_in_state(db, root_id: str, state: str):
    """Newest version row of ``root_id`` whose state is ``state``, else the row AT ``root_id``.

    The read recall hydrates through, and the reason it is not `get_artifact`.

    Both search arms key the index on ``root_id`` (`search/ingest/pipeline_unified`), so a
    ranked hit names a LINEAGE, not a version. Hydrating it with a direct lookup on that id is
    right only while the row at the root is the version that was indexed — true for every
    artifact where ``id == root_id``, which is every top-level one. It stops being true the
    moment a committed collection member is edited: `workspace_service._ensure_draft` writes
    the new content to a NEW id under the same root and leaves the committed row alone, so the
    draft segment holds the root, the direct lookup returns the committed row, and recall
    answers with the new version's tokens and the old version's bytes.

    Resolving within the searched STATE rather than to a global head is what keeps this from
    widening anything: each state is a separately keyed index tree, so the caller has already
    named which segment they are reading, and this returns that segment's version of a lineage
    the light cone has already authorized. Grants are held on the root — `add_artifact_to_collection`
    edges the root, and the light cone narrows on it — so every version this can return belongs
    to a lineage the caller was already cleared for. A global head resolution would not have
    that property against the archived segment, where the newest live row is precisely the one
    the caller did not ask for.

    Falls back to the row at ``root_id`` so a store with no version rows behaves exactly as it
    did before — the absent-lineage case and the ``id == root_id`` case are the same answer.
    """
    rows = [v for v in (db.artifacts.versions_of(root_id) or [])
            if state_of(v) == state]
    if rows:
        return rows[-1]                        # proper-time order: last = newest
    return db.artifacts.get_artifact(root_id)


def check_store_health() -> dict:
    """Health for `/status` — reads the maintained counter (no `count(*)` on any path)."""
    try:
        db = store_handle()
        return {"store": "lattice", "store_status": True,
                "vertices": db.artifacts.count()}
    except Exception as e:
        return {"store": "lattice", "store_status": False, "error": str(e)}


def __getattr__(name: str):
    """PEP 562 delegation — the full `db.lattice_api` surface, no name list to fall out of date."""
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
