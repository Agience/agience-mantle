"""The store backend — ONE import point for the routers/services.

THE LATTICE IS THE STORE: one SQLite file + FS-CAS content, opened in-process, zero external DB
processes. This module delegates every attribute to `db.lattice_api`; the envelope crypto +
change-event chokepoint lives in `db.doc_boundary`.

⛔ A `MANTLE_DB` BACKEND SELECTOR LIVED HERE and it is gone [John, 2026-07-23: "leave one path. the
only path. No constants, no fitting. no forcing."]. It read an env var, accepted exactly one value,
and raised on anything else — a choice with one option is not a choice, it is a branch that has to
be read, tested and explained forever. Verified before removing it: nothing in any repo sets
`MANTLE_DB` (`agience-bundle/backup.sh` sets `MANTLE_DB_PATH`, which is a different variable and is
still read). An env file that still sets it now has no effect, which is the accurate outcome — it
names a backend that does not exist.
"""
from __future__ import annotations

from mantle.db import lattice_api as _impl

_HANDLE = None                            # lattice mode: ONE store handle per process


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


def get_raw_artifact(db, artifact_id: str):
    """Raw artifact doc by id (the one raw-doc read shape for call sites)."""
    return db.artifacts.get_artifact(artifact_id)


def find_newest_by_root(db, root_id: str):
    """Newest non-archived version row for a root (proper-time order)."""
    rows = [v for v in (db.artifacts.versions_of(root_id) or [])
            if v.get("state") != "archived"]
    return rows[-1] if rows else None          # proper-time order: last = newest


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
