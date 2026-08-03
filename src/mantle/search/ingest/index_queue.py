"""Asynchronous indexing queue for unified search pipelines.

Provides a lightweight in-process worker pool that accepts indexing callables and
executes them concurrently on background threads. This keeps API handlers responsive
while the index finishes writing, and exposes simple status metrics so the UI
can surface pending work to users if needed.

"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class IndexTask:
    """Callable payload queued for background execution."""

    action: Callable[[], bool]
    description: str = ""
    tenant_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    #: The caller's context, captured at ENQUEUE time and re-entered in the worker
    #: thread — see :meth:`IndexQueue._execute_task`. Carries the acting principal
    #: (``services.acting_principal``), which indexing needs in order to obtain a
    #: cell key.
    #:
    #: ⚠ REQUIRED BECAUSE ThreadPoolExecutor DOES NOT PROPAGATE CONTEXTVARS. Unlike
    #: an asyncio task, a pool worker starts from an EMPTY context, so without this
    #: the identity of whoever queued the work is silently lost and indexing fails
    #: closed with ``NoActingPrincipal`` — correct, but useless.
    #:
    #: Capturing the enqueuer's context (rather than substituting a system identity)
    #: is also the more accurate answer: a queued index write really is done on
    #: behalf of the user whose write triggered it, so it should be checked against
    #: THAT user's grants. Genuinely system-initiated work enqueues from inside
    #: ``system_acting_context()`` and this carries the system principal instead.
    context: Optional[contextvars.Context] = None


class IndexQueue:
    """Thread pool for concurrent search indexing operations."""

    def __init__(self, max_workers: int = 4) -> None:
        #: ``None`` is the shutdown sentinel. The coordinator blocks in an untimed
        #: ``get()``, so ``stop()`` must PUT something to wake it — see ``stop()``.
        self._queue: "queue.Queue[Optional[IndexTask]]" = queue.Queue()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._coordinator: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        # ⛔ RLock, NOT Lock — `get_status()` DEADLOCKS ON ITSELF WITH A PLAIN LOCK.
        # It acquires this lock and then calls `is_running()`, which acquires it again. With a
        # non-reentrant Lock the calling thread blocks forever **while still holding it**, so
        # every subsequent `enqueue()` blocks too and indexing dies process-wide from a single
        # status read. Demonstrated: `get_status()` never returned after 5s on a fresh queue.
        # Currently latent only because nothing in-repo calls the exported `get_status` yet —
        # i.e. it is live ammunition in the public API rather than an active outage.
        self._lock = threading.RLock()
        #: Signalled when ``_inflight`` reaches zero. `flush()` waits on this instead of
        #: re-asking; a Condition over the SAME lock that already guards the counters, so
        #: there is no second lock to order against (RLock is re-entrant, as above).
        self._idle = threading.Condition(self._lock)
        #: Tasks accepted but not yet finished — enqueued, minus completed/failed/dropped.
        #: This is what `flush()` actually means. `self._queue.empty()` is NOT: the queue
        #: goes empty the moment the coordinator hands work to the executor, while that
        #: work is still running.
        self._inflight = 0
        self._max_workers = max_workers

        self._pending_per_tenant: Dict[str, int] = defaultdict(int)
        self._total_enqueued = 0
        self._total_processed = 0
        self._total_failed = 0

    # --- lifecycle -----------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._executor and self._coordinator and self._coordinator.is_alive():
                return
            self._stop_event.clear()
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="IndexWorker")
            self._coordinator = threading.Thread(target=self._coordinator_loop, daemon=True)
            self._coordinator.start()
            logger.info(f"IndexQueue worker pool started with {self._max_workers} threads")

    def stop(self, *, drain: bool = True, timeout: float = 5.0) -> None:
        if drain:
            self.flush(timeout=timeout)
        self._stop_event.set()
        # The coordinator is parked in an untimed `get()`, so setting the event is not
        # enough to wake it — nothing polls the event any more. Push the sentinel.
        self._queue.put(None)

        executor = None
        coordinator = None
        with self._lock:
            executor = self._executor
            coordinator = self._coordinator
            self._executor = None
            self._coordinator = None
        
        if coordinator:
            coordinator.join(timeout=timeout)
            if coordinator.is_alive():
                logger.warning("IndexQueue coordinator did not stop within timeout")
        
        if executor:
            executor.shutdown(wait=True, cancel_futures=not drain)
        
        logger.info("IndexQueue worker pool stopped")

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._executor and self._coordinator and self._coordinator.is_alive())

    # --- enqueue -------------------------------------------------------
    def enqueue(self, task: IndexTask) -> None:
        with self._lock:
            if task.tenant_id:
                self._pending_per_tenant[task.tenant_id] += 1
            self._total_enqueued += 1
            self._inflight += 1
        self._queue.put(task)
        logger.debug(f"Enqueued task: {task.description} (queue size: {self._queue.qsize()})")

    # --- status --------------------------------------------------------
    def get_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            pending_total = self._queue.qsize()
            status = {
                "pending_total": pending_total,
                "total_enqueued": self._total_enqueued,
                "total_processed": self._total_processed,
                "total_failed": self._total_failed,
                "worker_alive": self.is_running(),
            }
            if tenant_id:
                status["pending_for_tenant"] = self._pending_per_tenant.get(tenant_id, 0)
            else:
                status["pending_per_tenant"] = dict(self._pending_per_tenant)
            return status

    def flush(self, timeout: float = 5.0) -> None:
        """Block until all accepted work has FINISHED, or the timeout elapses.

        ⛔ THIS USED TO POLL `self._queue.empty()` EVERY 50 ms, AND IT ANSWERED THE WRONG
        QUESTION. The queue drains the instant the coordinator hands a task to the
        executor, so `empty()` went True while the task was still executing: `flush()`
        returned "done" over work in flight, and `stop(drain=True)` relied on that. The
        counter it should have been reading — tasks accepted minus tasks finished — is
        `_inflight`, and a Condition over it is both the correct question and a real
        blocking primitive, so the 50 ms poll disappears rather than being retuned.
        """
        with self._idle:
            if not self._idle.wait_for(lambda: self._inflight == 0, timeout=timeout):
                logger.warning("IndexQueue flush timed out with %s tasks unfinished", self._inflight)

    def _release(self, count: int = 1) -> None:
        """Mark ``count`` accepted tasks as no longer in flight and wake `flush()`."""
        with self._idle:
            self._inflight = max(0, self._inflight - count)
            if self._inflight == 0:
                self._idle.notify_all()

    # --- coordinator loop ----------------------------------------------
    def _coordinator_loop(self) -> None:
        """Coordinator thread that dispatches tasks to the worker pool.

        Blocks in `Queue.get()` until there IS work, drains everything already queued
        behind it, and submits the batch. Idle costs nothing: no timeout, no wakeup, no
        loop iteration until something is enqueued.

        ⛔ THE `get(timeout=0.2)` THIS REPLACED WAS NOT A LATENCY KNOB. `get()` returns
        the moment an item lands either way; the timeout existed ONLY so the loop could
        re-check `_stop_event`, which cost ~5 wakeups/second forever to notice an event
        that fires at most once. `stop()` now PUTS a `None` sentinel, so the check happens
        when there is something to check. Latency is unchanged (it was never the timeout
        that delivered a task) and shutdown gets *faster*: no up-to-200 ms wait to notice.

        The futures list this used to keep is gone. Nothing ever read it; it existed only
        to be periodically pruned of the completed futures that it was itself the sole
        reason to retain. Failures are already caught and counted in `_execute_task`, so a
        dropped future discards nothing.
        """
        while not self._stop_event.is_set():
            # Blocks here — indefinitely — until work arrives or `stop()` pushes None.
            first = self._queue.get()
            if first is None:
                self._queue.task_done()
                if self._stop_event.is_set():
                    break
                # A sentinel left over from a previous stop()/start() cycle. Not ours.
                continue

            tasks_to_submit = [first]
            stopping = False
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    self._queue.task_done()
                    stopping = self._stop_event.is_set()
                    break
                tasks_to_submit.append(nxt)

            if self._executor:
                for task in tasks_to_submit:
                    try:
                        self._executor.submit(self._execute_task, task)
                    except Exception as e:
                        logger.error(f"Failed to submit task to executor: {e}")
                        self._queue.task_done()
                        self._release()
            else:
                logger.warning("Executor not available, skipping tasks")
                for _ in tasks_to_submit:
                    self._queue.task_done()
                self._release(len(tasks_to_submit))

            if stopping:
                break

    def _execute_task(self, task: IndexTask) -> None:
        """Execute a single indexing task (runs in worker thread)."""
        from datetime import datetime, timezone
        start_time = time.time()
        logger.debug(f"[{datetime.now(timezone.utc).isoformat()}] 🏁 Starting task: {task.description}")
        try:
            # Re-enter the enqueuer's context so the acting principal (and anything
            # else contextual) is in scope on this pool thread. `copy_context()` at
            # enqueue returned a private copy per task, so running it here mutates
            # nothing the caller can observe.
            if task.context is not None:
                task.context.run(task.action)
            else:
                task.action()
            with self._lock:
                self._total_processed += 1
            elapsed = time.time() - start_time
            logger.debug(f" Task completed in {elapsed:.2f}s: {task.description}")
        except Exception:  # pragma: no cover - logged for observability
            with self._lock:
                self._total_failed += 1
            elapsed = time.time() - start_time
            logger.exception(f" Task failed after {elapsed:.2f}s: {task.description or task.action}")
        finally:
            if task.tenant_id:
                with self._lock:
                    remaining = self._pending_per_tenant.get(task.tenant_id, 0)
                    self._pending_per_tenant[task.tenant_id] = max(0, remaining - 1)
            self._queue.task_done()
            # Last thing done: this is what `flush()` is waiting on.
            self._release()


# Global singleton used by API process
_index_queue: Optional[IndexQueue] = None


def _get_queue() -> IndexQueue:
    """Lazy initialization of index queue with config."""
    global _index_queue
    if _index_queue is None:
        from origin import config
        _index_queue = IndexQueue(max_workers=config.INDEX_QUEUE_MAX_WORKERS)
    return _index_queue


def start_worker() -> None:
    _get_queue().start()


def stop_worker(*, drain: bool = True, timeout: float = 5.0) -> None:
    _get_queue().stop(drain=drain, timeout=timeout)


def enqueue(action: Callable[[], bool], *, description: str = "", tenant_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
    queue = _get_queue()
    if not queue.is_running():
        raise RuntimeError(f"IndexQueue worker not running - cannot enqueue task: {description}")
    # Capture the CALLER's context here, in the enqueuing thread — this is the only
    # moment the acting principal is still in scope. See `IndexTask.context`.
    task = IndexTask(
        action=action,
        description=description,
        tenant_id=tenant_id,
        metadata=metadata or {},
        context=contextvars.copy_context(),
    )
    queue.enqueue(task)
    # Return immediately - task will be processed asynchronously in background threads


def flush(timeout: float = 5.0) -> None:
    _get_queue().flush(timeout=timeout)


def get_status(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    return _get_queue().get_status(tenant_id=tenant_id)


def is_running() -> bool:
    return _get_queue().is_running()
