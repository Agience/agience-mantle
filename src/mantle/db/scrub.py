"""Leave no freed bytes in the store file (reviews 7 and 26).

SQLite keeps a replaced or deleted row's old bytes in its file, and in the WAL, until the page is
reused. Two things remove them:

  * `secure_delete` is ON for every connection to the store (`schema.PRAGMAS`), so the page image
    SQLite writes after a replace or delete has the freed cell zeroed. That covers every place a
    key row (or any row) is replaced or deleted, with no call site to remember.
  * The old page image is still in the main file until a checkpoint copies the new one over it,
    and earlier WAL frames can still hold it until the WAL is reset. A truncating checkpoint after
    the replace (`settle_replaced`) does both.

Measured on a real store file (review 26): with neither, the whole KEK-wrapped key the owner-call
split replaced was in the file; with the checkpoint alone it was still in the freed cell; with
`secure_delete` alone it was still in the main file's old page; with both it was gone.

`settle_replaced` runs on the request path (the split on a person's call), so it waits for readers
at most `SETTLE_WAIT_MS`. A reader holding a snapshot longer than that leaves the checkpoint
undone; that is recorded (`SCRUB_DUE_ID`, counted by `split_master_keys.exposure_counts`) and
cleared by the next settle that completes, since the page images written since are already
zeroed. `scrub_freed` (a `VACUUM`, which rewrites the whole file) is for the passes, which run with
the node quiet: on a live store of many gigabytes it is far too slow for a person's call.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

#: How long a settle waits for readers to finish before it records itself as due.
SETTLE_WAIT_MS = 2000

#: The per-box row that records a settle still due. Never replicated (`mesh.sync._OP_EXCLUDE`).
SCRUB_DUE_ID = "custody:scrub-due"
SCRUB_DUE_CT = "application/vnd.agience.custody-scrub-due+json"


def scrub_freed(db) -> None:
    """Rebuild the file and truncate the WAL: for a pass, under the store's write lock, once its
    writes are done."""
    lattice = db.artifacts.db
    with lattice._wlock:
        c = lattice.conn()
        c.execute("PRAGMA secure_delete=ON")
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        c.execute("VACUUM")
        c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    _clear_due(db)


def _lattice(db):
    lattice = getattr(getattr(db, "artifacts", None), "db", None)
    return lattice if lattice is not None and hasattr(lattice, "conn") else None


def settle_replaced(db) -> bool:
    """After a key row was replaced or deleted: fold the WAL into the file and truncate it, so the
    old row is in neither. True when it completed; False when it is recorded as due."""
    from mantle.db.schema import wal_checkpoint

    lattice = _lattice(db)
    if lattice is None:                     # no SQLite file behind this store: nothing to settle
        return True
    busy, why = 1, "busy"
    try:
        with lattice._wlock:
            if getattr(lattice, "_depth", lambda: 0)() > 0:
                why = "inside a write transaction"
            else:
                c = lattice.conn()
                prev = int(c.execute("PRAGMA busy_timeout").fetchone()[0])
                c.execute("PRAGMA busy_timeout=%d" % SETTLE_WAIT_MS)
                try:
                    busy, _pages, _done = wal_checkpoint(c, "TRUNCATE")
                finally:
                    c.execute("PRAGMA busy_timeout=%d" % prev)
    except Exception as exc:  # noqa: BLE001 — recorded as due, never fails the caller
        busy, why = 1, f"{type(exc).__name__}: {exc}"
    if not busy:
        _clear_due(db)
        return True
    logger.error("ALARM: a replaced key row is not yet settled out of the store file (%s); the "
                 "next settle or `mantle-wal-checkpoint` with readers stopped clears it", why)
    _mark_due(db, why)
    return False


def due(db) -> int:
    """How many settles are recorded as due (0 when none)."""
    try:
        doc = db.artifacts.get_artifact(SCRUB_DUE_ID)
    except Exception:  # noqa: BLE001
        return 0
    return int((doc or {}).get("count") or 0)


def _mark_due(db, why: str) -> None:
    try:
        doc = {"id": SCRUB_DUE_ID, "content_type": SCRUB_DUE_CT, "count": due(db) + 1,
               "last": datetime.now(timezone.utc).isoformat(timespec="seconds"), "why": why[:200]}
        if hasattr(db.artifacts, "page_by_origin"):
            # A per-box row, pinned outside every observer's sequence as `mesh.sync._put_op` pins
            # its operational rows, so it never enters the publish feed.
            doc.update({"_origin": "_local:" + SCRUB_DUE_ID, "_seq": 1})
        db.artifacts.put_artifact(doc, stamp_rev=False)
    except Exception:  # noqa: BLE001
        logger.warning("could not record a settle as due", exc_info=True)


def _clear_due(db) -> None:
    try:
        if db.artifacts.get_artifact(SCRUB_DUE_ID) is not None:
            db.artifacts.delete_artifact(SCRUB_DUE_ID)
    except Exception:  # noqa: BLE001
        logger.warning("could not clear the settle due record", exc_info=True)
