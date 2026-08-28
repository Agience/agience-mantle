"""Access-audit **force** — every authorization decision leaves a trace.

Observation records a witness. Each access check (allow OR deny) appends an
append-only edge:

    (principals/{principal_id}) -[access_events {action, result, ts, ctx}]-> (artifacts/{artifact_id})

Because it is emitted inside the authorization layer (``dependencies.check_access``),
auditability is a **property of access itself** — you cannot touch an artifact without
it being witnessed. Writes already carry commit provenance; this adds read / attempt
provenance, so an artifact's full lineage — created, changed, **observed, and denied** —
lives in the graph and is queryable per-artifact (INBOUND to ``artifacts/{id}``).

Design goals:
- **Non-blocking**: ``record_access()`` enqueues into a bounded, thread-safe in-memory
  buffer and returns immediately. It never raises and never slows the request path
  (auth checks run in FastAPI's threadpool; the buffer is thread-safe).
- **Batched + durable-enough**: a background asyncio task flushes the buffer into the
  ``access_events`` edge collection every ``_FLUSH_INTERVAL_S`` (or when a batch fills),
  and a final synchronous drain runs on shutdown.
- **Append-only**: no update/delete surface.
- **Backpressure, never silent**: if the sink can't keep up and the buffer exceeds
  ``_MAX_BUFFER``, the oldest events are dropped and a counter is logged — liveness over
  completeness under overload, surfaced rather than hidden.
- **Retention is opt-in**: the log is kept indefinitely unless ``MANTLE_AUDIT_RETENTION_DAYS``
  sets an age bound, in which case the flusher sweeps on a slow cadence. The reasoning for
  the default, and the fact that events outlive the artifacts they name, is in
  ``db/audit.py``.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from mantle.db.store import Database

logger = logging.getLogger(__name__)

ACCESS_EVENTS_COLLECTION = "access_events"

_FLUSH_INTERVAL_S = 1.0
_BATCH = 500
_MAX_BUFFER = 20000

#: How often the retention sweep runs while an age bound is configured. Retention is measured
#: in days, so an hourly pass makes the boundary exact to well within its own unit while the
#: delete stays a rare event on the write lock. With no bound set the sweep is never entered.
_PRUNE_INTERVAL_S = 3600.0

_buffer: "deque[dict]" = deque()
_lock = threading.Lock()
_dropped_total = 0
_lost_total = 0        # events drained but never written (a failed flush) — see `stats()`
_task: Optional[asyncio.Task] = None
_stopping = False


def record_access(
    *,
    principal_id: Optional[str],
    artifact_id: str,
    action: str,
    result: str,                      # "allowed" | "denied"
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Enqueue an access event. Fire-and-forget: never raises, never blocks."""
    global _dropped_total
    if not artifact_id:
        return
    evt = {
        "_from": f"principals/{principal_id or 'anonymous'}",
        "_to": f"artifacts/{artifact_id}",
        "principal_id": principal_id,
        "artifact_id": artifact_id,
        "action": action,
        "result": result,
        "ts": datetime.now(timezone.utc).isoformat(),
        "ctx": context or {},
    }
    with _lock:
        if len(_buffer) >= _MAX_BUFFER:
            _buffer.popleft()
            _dropped_total += 1
            if _dropped_total % 1000 == 1:
                logger.warning(
                    "audit backpressure: buffer full, dropping oldest events (total dropped=%d)",
                    _dropped_total,
                )
        _buffer.append(evt)


def _drain(n: int) -> list:
    out: list = []
    with _lock:
        for _ in range(min(n, len(_buffer))):
            out.append(_buffer.popleft())
    return out


def stats() -> dict:
    """Published counters for health monitoring — the sanctioned way to read this service.

    `dropped_total` and `lost_total` are absolute counts of access events this
    process failed to record:

      * `dropped_total` — refused at the door: the buffer was full (backpressure).
      * `lost_total`    — accepted, drained, then the insert failed. Strictly worse: the caller was
                          told the event was recorded.

    Either being non-zero means the audit trail is incomplete, as a number rather than
    a guess from a log line.
    """
    return {
        "pending": pending(),
        "dropped_total": _dropped_total,
        "lost_total": _lost_total,
    }


def pending() -> int:
    with _lock:
        return len(_buffer)


def flush_once(db: Database) -> int:
    """Batch-insert one drained batch of access edges. Returns count written.

    A failed insert is counted, not just logged: `stats()["lost_total"]`. Losing
    access events without a number attached is the one outcome this service must
    never allow.
    """
    global _lost_total
    batch = _drain(_BATCH)
    if not batch:
        return 0
    try:
        from mantle.db import audit as _lattice_audit
        return _lattice_audit.append_access_events(db.conn, batch)
    except Exception:
        _lost_total += len(batch)
        logger.warning("audit flush failed; %d events lost (lost_total=%d)",
                       len(batch), _lost_total, exc_info=True)
        return 0


def prune_once(db: Database) -> int:
    """Apply the configured retention bound. Returns rows deleted; 0 when retention is
    unlimited, which is the default and issues no statement.

    Best-effort like the flush path: a failed prune is a log line, never a raise into the
    caller — an audit trail that grows is a worse outcome than one that fails to shrink,
    but neither may take down the worker."""
    try:
        from mantle.db import audit as _lattice_audit
        return _lattice_audit.prune_access_events(db.conn)
    except Exception:
        logger.warning("audit retention prune failed", exc_info=True)
        return 0


async def _flusher(db: Database) -> None:
    logger.info("audit access-log flusher started (interval=%.1fs, batch=%d)", _FLUSH_INTERVAL_S, _BATCH)
    # Elapsed against the loop clock, not wall time: only the cadence depends on it, and a
    # monotonic clock cannot be walked backwards by an NTP correction into never sweeping.
    next_prune = asyncio.get_running_loop().time()
    while not _stopping:
        try:
            # Keep draining while batches are full (catch up under load), then sleep.
            while flush_once(db) >= _BATCH:
                await asyncio.sleep(0)
            now = asyncio.get_running_loop().time()
            if now >= next_prune:
                next_prune = now + _PRUNE_INTERVAL_S
                prune_once(db)
        except Exception:
            logger.debug("audit flusher iteration error", exc_info=True)
        await asyncio.sleep(_FLUSH_INTERVAL_S)


def start_audit_worker(db: Database) -> None:
    """Start the background flush task (idempotent). Call from the app lifespan."""
    global _task, _stopping
    _stopping = False
    if _task is None or _task.done():
        _task = asyncio.create_task(_flusher(db))


async def stop_audit_worker(db: Database, *, drain: bool = True) -> None:
    """Stop the flusher and (optionally) synchronously drain remaining events."""
    global _stopping, _task
    _stopping = True
    if _task is not None:
        _task.cancel()
        try:
            await _task
        except (asyncio.CancelledError, Exception):
            pass
        _task = None
    if drain:
        # Final best-effort drain so shutdown doesn't lose buffered events.
        while flush_once(db) > 0:
            pass


def get_artifact_access_log(
    db: Database,
    artifact_id: str,
    *,
    limit: int = 100,
    offset: int = 0,
    result: Optional[str] = None,     # "allowed" | "denied"
    since: Optional[str] = None,      # inclusive ISO-8601 lower bound on `ts`
    until: Optional[str] = None,      # inclusive ISO-8601 upper bound on `ts`
) -> list:
    """Access history witnessing an artifact, newest first.

    Reads the persisted collection — events still buffered (< the flush interval)
    are not yet visible, which is acceptable for an audit log (near-real-time).
    """
    from mantle.db import audit as _lattice_audit
    try:
        return _lattice_audit.access_log_of(
            db.conn, artifact_id, limit=limit, offset=offset, result=result,
            since=since, until=until)
    except Exception:
        logger.warning("access-log query failed for %s", artifact_id, exc_info=True)
        return []


def _reset_for_tests() -> None:
    global _buffer, _dropped_total, _lost_total, _task, _stopping
    with _lock:
        _buffer = deque()
    _dropped_total = 0
    _lost_total = 0
    _task = None
    _stopping = False
