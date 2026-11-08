"""The lock Mantle's server holds for as long as it runs (review 32).

A file beside the lattice (`<lattice>.server.lock`), locked exclusively by the server process at
startup and released by the OS when the process ends, however it ends. A pass that must not run
beside a live server (step 21's rotation of charged roots' keys, where a server holding the old
key in its cache could write under it after the rotation commits) asks `held_elsewhere`: it takes
the lock without waiting and lets it go at once, and a lock it cannot take is another process's.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

#: The one handle this process holds, so a second `hold` in the same process is a no-op.
_HELD: dict = {}


def default_path() -> str:
    from mantle.db import lattice_api

    lattice = os.path.abspath(os.path.expanduser(
        os.getenv("MANTLE_LATTICE_PATH", str(lattice_api.DEFAULT_LATTICE_PATH))))
    return lattice + ".server.lock"


def _lock(fh) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fh) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _open(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fh = open(path, "a+b")
    if os.path.getsize(path) == 0:
        fh.write(b"\0")
        fh.flush()
    return fh


def hold(path: Optional[str] = None):
    """Take the server's lock for the life of this process. Raises `RuntimeError` when another
    process holds it: a second server on the same store."""
    path = path or default_path()
    if path in _HELD:
        return _HELD[path]
    fh = _open(path)
    if not _lock(fh):
        fh.close()
        raise RuntimeError(f"another Mantle server holds {path}")
    _HELD[path] = fh
    return fh


def take_for_server(path: Optional[str] = None):
    """The server's start (review 33): hold the lock for the life of the process. On the
    production plane (`AGIENCE_PLANE=production`) a lock that cannot be taken refuses the start
    (`RuntimeError`): a second server on the same store would let step 21 rotate beside a live
    one. Elsewhere it is logged, and None returned. The lock is an OS file lock, reliable only on
    a local filesystem, so the store lives on one."""
    try:
        return hold(path)
    except Exception as exc:  # noqa: BLE001 — refused on production, logged elsewhere
        plane = (os.getenv("AGIENCE_PLANE") or "").strip().lower()
        if plane == "production":
            raise RuntimeError(f"Mantle does not start without its server lock: {exc}") from exc
        logger.error("the server lock was not taken (%s); this plane starts anyway", exc)
        return None


def release(fh) -> None:
    for p, held in list(_HELD.items()):
        if held is fh:
            del _HELD[p]
    _unlock(fh)
    fh.close()


def held_elsewhere(path: Optional[str] = None) -> bool:
    """Does another process hold the server's lock now? This process's own hold is not counted."""
    path = path or default_path()
    if path in _HELD:
        return False
    try:
        fh = _open(path)
    except OSError:
        logger.warning("the server lock %s cannot be opened; counted as held", path)
        return True
    try:
        if not _lock(fh):
            return True
        _unlock(fh)
        return False
    finally:
        fh.close()
