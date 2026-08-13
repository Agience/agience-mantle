"""The drain for the mirror leg this node still owes — the other half of `_record_mirror_pending`.

`PUT /artifacts/{id}/content` writes the bytes locally and then, on a node that has one, to the
object store. When that second leg fails in a way a retry could still fix, the route records the
obligation as an `op.content.mirror` task on the store's own work pool (`db/schema.py`'s `task`
sidecar). That record is durable, idempotent by `content_key`, and visible as a first-class
artifact — and until this module existed, nothing ever acted on it.

This is what acts on it. Three pieces, in the order they matter:

  * :func:`mirror_one` — the OPERATOR BODY. Read the CAS ref, put it to the object store under
    the task's `content_key`. It writes no bytes of its own: the object it puts is the exact
    `MEC1‖nonce‖ct` envelope the upload produced, read back out of the same
    `TieredContentStore` the upload wrote, and pushed through the same
    `content_service.put_bytes_encrypted` mirror leg that failed. One mirror-write path, used
    twice, so the retry cannot drift from the original.
  * :func:`drain_mirror_pending` — one bounded PASS. Claim, run, settle. Synchronous and
    self-contained: it is the whole drain, and every runner below is a schedule around it.
  * :func:`start_mirror_drain` — the RUNNER, an asyncio task owned by the app lifespan. It is
    started only on a node that has an object store, and it never polls (see below).

**The air-gap invariant is enforced twice, and neither is a policy check.** A node with no object
store never reaches `on_mirror_deferred` at all — `put_bytes_encrypted` returns before it — so its
queue is empty forever. On top of that, :func:`start_mirror_drain` does not create a task, and
:func:`drain_mirror_pending` returns without touching the store, when `edge_store_configured()` is
false. An air-gapped node spends nothing here: no task, no timer, no read.

**No polling, and therefore no poll interval.** The runner sleeps until the earliest
`next_retry_at` among its own pending rows — the queue's own schedule, read from the queue — and
when there are none it blocks on an event that `_record_mirror_pending` sets. So the drain's
cadence is derived from the work rather than from a number chosen here, and an idle node's drain
costs one blocked coroutine.

**What this drain will not claim, and what will not claim it.** Both directions are settled by
NAMES rather than by policy, and neither costs a check at run time.

*Outward:* the drain narrows its window to `operator = 'op.content.mirror'` in SQL
(`pending_window(..., operator=...)`), so it never claims, and never has to put back, work it
cannot do. That stays true however many operators later share this content type.

*Inward:* `ember/runtime/pool.py::claim` selects on `(ct, status)` alone — it cannot filter on an
operator, and mantle cannot teach it to. So these tasks are simply not written under the content
type it claims. `ember`'s pool is `application/vnd.agience.task+json`; this one is
:data:`MIRROR_TASK_CT`, which nothing else selects on. An ember worker sharing this lattice reads
its own queue and never sees a row of ours, so it cannot claim one, cannot fail to invoke
`op.content.mirror`, and cannot dead-letter at its own `MAX_ATTEMPTS` an obligation it never
looked at the bytes for. What used to be a deployment invariant — *run the drain where an ember
pool worker is not* — is a naming fact instead, and a naming fact is enforced by the query
planner rather than by an operator remembering it.

MIGRATION
---------
Rows written before the rename are on disk under the shared content type, and they are the
obligations this whole module exists to keep. :func:`adopt_shared_pool_tasks` moves the live ones
across at the head of every pass; see its docstring for what "live" means and why the terminal
ones stay where they are.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: This work pool's content type — MANTLE'S, and no other service's.
#:
#: Named for the work rather than for the mechanism (`mirror-task`, not `task`), which is the
#: same rule every other per-box type in this tree is named under: `mesh-cursor`,
#: `content-cursor`, `s3sync-cursor`, `shard-done`. A type called `task` says only that a row is
#: work, which is exactly the claim that made it selectable by somebody else's worker.
#:
#: The rename IS the fix for that. `ember/runtime/pool.py::claim` selects on `(ct, status)` and
#: has no operator predicate, so the one thing that decides whether an ember worker takes a
#: mantle row is which content type the row carries. Under this one it takes none.
MIRROR_TASK_CT = "application/vnd.agience.mirror-task+json"

#: The SHARED pool these tasks used to be written under — `ember`'s, and every other operator's.
#: It appears here for exactly one reason: rows already on disk under it. Nothing is WRITTEN under
#: it any more; see :func:`adopt_shared_pool_tasks`.
SHARED_TASK_CT = "application/vnd.agience.task+json"

#: Distinct from `op.content.promote` (`shard/content_tier.py`), which walks the `content_ref`
#: COLUMN and targets the OVH origin. This one walks nothing: it names one already-identified
#: object and targets the MinIO/S3 edge that `services/content_service.py` writes. Two different
#: mirrors reached from two different records; giving them one operator name would let a drain
#: claim work it cannot do.
MIRROR_OPERATOR = "op.content.mirror"


def mirror_task_key(content_key: str) -> str:
    """The task's natural key: **the `content_key`**, under the operator that owes it.

    The work is "the edge bucket is missing an object at this key", so the key names the object,
    not the request that discovered it missing. That gives idempotence for free through
    `put_artifact`'s upsert-by-id:

      * a second failed upload for the same artifact writes the same row — one task, never two;
      * a failed upload of DIFFERENT bytes writes that same row too, refreshed, which is right:
        the earlier ref stopped being the artifact's content the moment `_record_content_ref`
        overwrote the context. A second row would oblige the mirror to hold a version no reader
        will ever ask for.

    The CAS ref is deliberately NOT part of the key, and the reason is stronger than preference:
    the ref addresses the ENVELOPE, whose nonce is fresh on every encryption, so two uploads of
    byte-identical plaintext produce two different refs. Keyed on the ref, every retry of the same
    content would stack another task.
    """
    return MIRROR_OPERATOR + ":" + content_key


def mirror_task_id(content_key: str) -> str:
    """`task-<operator>:<digest>` — the shape `pool.enqueue` builds, over this key rather than
    over a hash of the whole argument blob (which would change with every ref and defeat the point
    of having a natural key at all).

    Digested rather than spelled out because a content key contains `/` (`artifacts/x.content`),
    and an id carrying a slash does not survive `GET /artifacts/{id}`: the extra segment stops the
    route matching and the read 404s. A row whose entire purpose is to be visible must be
    addressable. The construction is `_snapshot_prior`'s — blake2b-64 over the deciding string —
    so a derived id is derived the same way everywhere in this store. The readable form is not
    lost: it is on the row as `task_key`, and `arguments.content_key` carries it verbatim.
    """
    return "task-%s:%s" % (MIRROR_OPERATOR,
                           hashlib.blake2b(content_key.encode("utf-8"),
                                           digest_size=8).hexdigest())


# ── outcomes ─────────────────────────────────────────────────────────────────────────────────
#
# Three, and the middle one is the one this module exists to get right.
#
#   DONE      the object is in the store. Terminal, and the only success.
#   RETRY     repeating this, unchanged, could plausibly get a different answer. Back to
#             `pending` behind a derived `next_retry_at`.
#   DEAD      it could not. Terminal, never re-run, and carrying the reason on the row.
#
# What separates RETRY from DEAD is `content_service.mirror_failure_is_transient` for anything the
# store said, and determinacy for everything else: an obligation whose own premises are false (the
# artifact is gone, it never had content, it points somewhere else, the bytes are no longer on
# this node) cannot become true by being asked again.
_DONE, _RETRY, _DEAD = "done", "pending", "dead"

#: Task ids this PROCESS is attempting right now. Read only by `_reclaim_own`, which must not
#: mistake a live attempt for the leftovers of a dead one. A plain `set` is enough: every mutation
#: is a single add or discard, which the GIL makes atomic, and the only reader tolerates either
#: answer being one moment stale (it reclaims on the next pass instead).
_in_flight: set = set()


class _Permanent(Exception):
    """An obligation that is false, not merely unmet. Dead-letters on the first attempt.

    Raised for the determinate failures only. A store that refused is ALSO permanent, but it is
    not raised here — it arrives as the store's own exception and is classified by
    `mirror_failure_is_transient`, which reads HTTP rather than a list this module curates.
    """


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(when: datetime) -> str:
    return when.isoformat()


def _worker_id(store_db) -> str:
    """This drain's identity: the node's own origin plus the process that is running it.

    The origin because that is what the store already calls this node, and the pid because a
    multi-worker `mantle-serve` runs one drain per process and two of them must be
    distinguishable — `settle` compares on `claimed_by`, so a shared id would let one process
    settle another's claim, which is the exact thing that comparison exists to prevent.
    """
    origin = getattr(getattr(store_db, "artifacts", None), "origin", None) or "node"
    return "mirror-drain:%s:%d" % (origin, os.getpid())


# ── backoff, derived ─────────────────────────────────────────────────────────────────────────

def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """The delay the STORE asked for, when it asked for one.

    RFC 9110 §10.2.3 makes `Retry-After` the origin's own statement about when it will be able to
    answer, and RFC 6585 §4 pairs it with 429 specifically. When the store has told us, there is
    nothing to derive: any number computed here would be a guess overriding a fact. Both the
    delta-seconds and the HTTP-date forms are accepted, because the RFC defines both.
    """
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return None
    headers = (resp.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
    if not isinstance(headers, dict):
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except (TypeError, ValueError):
        pass
    try:
        from email.utils import parsedate_to_datetime
        when = parsedate_to_datetime(str(raw))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - _now()).total_seconds())
    except Exception:
        return None


def _client_patience_seconds() -> float:
    """How long this node has already declared it is willing to wait for this store.

    Read off the edge client's own botocore config — the connect and read timeouts that every
    `put_object` on this node is already signed up to spend. It is the right floor for a retry
    delay for a reason that is not a preference: you cannot learn anything new about whether the
    store is reachable faster than the time you were prepared to wait for it to say so, so a
    retry sooner than that is a request whose answer you already have.

    Zero when it cannot be read (a stubbed client in a test, an unusual transport). Zero is
    honest — it removes the floor and leaves the measured cost of the attempt as the whole
    derivation — rather than a substituted constant.
    """
    from mantle.services import content_service
    try:
        cfg = content_service._s3_edge_internal.meta.config
        return max(float(cfg.connect_timeout or 0.0), float(cfg.read_timeout or 0.0))
    except Exception:
        return 0.0


def _next_retry_at(when: datetime, *, attempts: int, elapsed_s: float,
                   exc: Optional[BaseException] = None) -> str:
    """When this task becomes eligible again — derived, never chosen.

    Three real quantities, in the order of how much they know:

      1. **What the store said.** A `Retry-After` is the origin's own timing and wins outright.
      2. **What the attempt cost.** `elapsed_s` is the measured duration of the write that just
         failed: a store that hangs for two minutes before failing has told us how long a round
         with it takes, and coming back faster than that would just buy another two minutes.
      3. **What this node was already willing to spend.** The edge client's configured
         connect/read timeout (:func:`_client_patience_seconds`) — the floor under (2), because a
         connection REFUSED costs nearly nothing to discover and the near-zero delay that would
         fall out of (2) alone is a hot loop, not a backoff.

    The greater of (2) and (3) is one attempt's worth of waiting. It is multiplied by `attempts`,
    the count of consecutive failures on this row, which is the only evidence available about how
    long the outage has lasted — so the delay grows with the outage and shrinks back to one round
    the moment a fresh upload refreshes the row (`_record_mirror_pending` writes no `attempts`,
    which is what makes new bytes a new beginning rather than an inherited backlog).

    There is no cap and no floor beyond the above. A cap would be a chosen constant, and it is not
    needed: nothing here retries forever except a genuinely transient failure, which is precisely
    the case where continuing to try is correct.
    """
    if exc is not None:
        told = _retry_after_seconds(exc)
        if told is not None:
            return _iso(when + timedelta(seconds=told))
    one_round = max(float(elapsed_s), _client_patience_seconds())
    return _iso(when + timedelta(seconds=one_round * max(1, int(attempts))))


# ── the operator body ────────────────────────────────────────────────────────────────────────

def _context_of(doc: Dict[str, Any]) -> Dict[str, Any]:
    raw = doc.get("context")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def mirror_one(store_db, task: Dict[str, Any]) -> Dict[str, Any]:
    """Redo the mirror leg one task owes. Returns `{"outcome", "reason", "elapsed_s", "exc"}`.

    **The artifact authorizes the work; the task only points at it.** `arguments` arrives from a
    row, and a row of this content type is something a caller can create through
    `POST /artifacts` — so nothing in it is trusted as an instruction. The task names an
    artifact; that artifact's own recorded context is then the sole authority for WHICH bytes are
    read and WHICH key they are written to, and the task's arguments are only checked against it.
    A forged task therefore cannot make this drain copy an arbitrary CAS object to an arbitrary
    object-store key: it can only ever ask for a re-mirror of content the named artifact already
    records at the key that artifact already owns, which is a write the upload route would have
    made anyway. The shape of the ref is checked too (`is_cas_ref`), so a ref cannot address a
    path.

    **It writes through the existing mirror leg, not a second one.** `put_bytes_encrypted` with
    `owner_id=None, cas=False` IS that leg with both of the other tiers switched off: no
    envelope is formed (the bytes read out of the CAS are already the envelope the upload made,
    and re-encrypting them would produce an object no reader could open), no local CAS write (the
    local copy is the thing being read), and the object-store call, the bucket check and the
    presence memo are the same three lines the upload ran. A second `put_object` here would be a
    second place for the bucket, the key and the content type to be decided.
    """
    from mantle.services import content_service as cs

    args = task.get("arguments")
    if not isinstance(args, dict):
        raise _Permanent("the task carries no arguments")
    artifact_id = args.get("artifact_id")
    content_key = args.get("content_key")
    ref = args.get("content_ref")
    if not (isinstance(artifact_id, str) and isinstance(content_key, str)):
        raise _Permanent("the task names no artifact or no content key")
    if not cs.is_cas_ref(ref):
        raise _Permanent("content_ref %r is not a CAS address" % (ref,))

    # The route's own resolution, borrowed rather than restated: `artifact_id` may be a root id,
    # and a drain that resolved it differently from the route that wrote the row would complete a
    # task about a different artifact than the one the row is about.
    from mantle.routers.artifacts_router import _find_artifact
    doc = _find_artifact(store_db, artifact_id)
    if doc is None:
        raise _Permanent("artifact %s no longer exists" % artifact_id)

    ctx = _context_of(doc)
    if ctx.get("content_key") != content_key:
        # Determinate, and this is the security refusal: the key is not this artifact's key.
        raise _Permanent(
            "artifact %s does not record content_key %r (it records %r)"
            % (artifact_id, content_key, ctx.get("content_key")))
    if ctx.get("content_cas_ref") != ref:
        # SUPERSEDED — the artifact's content is no longer these bytes. Not raised as permanent on
        # sight, because it is momentarily indistinguishable from a live obligation: a fresh
        # failed upload refreshes this row with its new ref BEFORE `_record_content_ref` rewrites
        # the artifact's pointer, so a claim landing inside that one store write sees exactly this
        # mismatch on a task that is perfectly good. The caller resolves it by evidence rather
        # than by a timer — see `_settle_after`.
        return {"outcome": "superseded", "reason":
                "artifact %s now points at %r, not %r"
                % (artifact_id, ctx.get("content_cas_ref"), ref)}

    tier = cs.local_content_tier()
    if tier is None:
        # No local content tier AT ALL (the keys volume is not mounted). A configuration this node
        # can regain, and nothing has been lost — the bytes are still on the disk. Transient.
        return {"outcome": "retry", "reason": "this node has no local content tier to read from",
                "elapsed_s": 0.0}
    if not cs.local_content_has(ref):
        # The tier is open and does not hold the object. There is nothing on this node to mirror
        # and no second copy to recover it from — that is what the mirror was FOR. Determinate.
        raise _Permanent("%s is no longer in this node's local CAS; the bytes to mirror are gone"
                         % ref)

    started = time.monotonic()
    try:
        blob = tier.get(ref, collection=doc.get("collection_id"))
    except Exception as exc:
        # The tier says it holds the object and could not produce it: an I/O fault, not an
        # absence. Retryable — the presence check above already separated "gone" from "unreadable".
        return {"outcome": "retry", "elapsed_s": time.monotonic() - started,
                "reason": "local CAS read failed (%s: %s)" % (type(exc).__name__, str(exc)[:160])}

    ctype = ctx.get("content_type") or doc.get("content_type") or "application/octet-stream"
    try:
        cs.put_bytes_encrypted(content_key, blob, ctype, None, cas=False)
    except Exception as exc:
        elapsed = time.monotonic() - started
        if cs.mirror_failure_is_transient(exc):
            return {"outcome": "retry", "elapsed_s": elapsed, "exc": exc,
                    "reason": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
        raise _Permanent("the object store refused the write (%s: %s)"
                         % (type(exc).__name__, str(exc)[:160]))
    return {"outcome": "done", "elapsed_s": time.monotonic() - started,
            "reason": "mirrored %s to %s" % (ref, content_key)}


# ── one pass ─────────────────────────────────────────────────────────────────────────────────

def _settle_after(tasks, task_id: str, *, worker_id: str, task: Dict[str, Any],
                  result: Dict[str, Any], when: datetime) -> str:
    """Turn one attempt's result into the row's next state, atomically against the claim.

    Every exit goes through `settle`, which compares on `claimed_by` — so a row that was rewritten
    to `pending` underneath this attempt (a fresh failed upload for the same `content_key`, which
    clears `claimed_by`) is NOT overwritten, and this returns `"lost"`. The in-flight attempt was
    for superseded bytes and it loses, which is correct; what matters is that it finds out.
    """
    attempts = int(task.get("attempts") or 0) + 1
    outcome = result.get("outcome")

    if outcome == "done":
        ok = tasks.settle(task_id, worker_id=worker_id, to_status=_DONE,
                          completed_at=_iso(when),
                          fields={"attempts": attempts, "last_error": None,
                                  "result": result.get("reason")})
        return _DONE if ok else "lost"

    if outcome == "superseded":
        # Evidence, not a timer. The window in which a good task can look superseded is one store
        # write wide (`_record_content_ref`), and it is closed by the time this row is eligible
        # again. So a first sighting is retried — the cheapest way to find out — and a SECOND
        # sighting, of a row that has already been round the loop once, is the artifact's settled
        # state: this obligation names bytes nothing will ever ask for.
        if int(task.get("attempts") or 0) >= 1:
            ok = tasks.settle(task_id, worker_id=worker_id, to_status=_DEAD,
                              completed_at=_iso(when),
                              fields={"attempts": attempts, "dead_reason": "superseded",
                                      "last_error": result.get("reason")})
            return _DEAD if ok else "lost"
        ok = tasks.settle(task_id, worker_id=worker_id, to_status=_RETRY,
                          next_retry_at=_next_retry_at(when, attempts=attempts, elapsed_s=0.0),
                          fields={"attempts": attempts, "last_error": result.get("reason")})
        return _RETRY if ok else "lost"

    if outcome == "dead":
        ok = tasks.settle(task_id, worker_id=worker_id, to_status=_DEAD,
                          completed_at=_iso(when),
                          fields={"attempts": attempts, "dead_reason": "permanent",
                                  "last_error": result.get("reason")})
        return _DEAD if ok else "lost"

    ok = tasks.settle(
        task_id, worker_id=worker_id, to_status=_RETRY,
        next_retry_at=_next_retry_at(when, attempts=attempts,
                                     elapsed_s=float(result.get("elapsed_s") or 0.0),
                                     exc=result.get("exc")),
        fields={"attempts": attempts, "last_error": result.get("reason")})
    return _RETRY if ok else "lost"


def drain_mirror_pending(store_db, *, worker_id: Optional[str] = None,
                         limit: Optional[int] = None) -> Dict[str, Any]:
    """One bounded pass: claim what is eligible, mirror it, settle it. Never raises.

    Returns a count per outcome plus `next_eligible` — the earliest `next_retry_at` still standing
    on this operator's pending rows, which is what the runner sleeps until. `{"skipped": ...}`
    when this node has no object store.

    Bounded by `pending_window`'s own default rather than by a number chosen here; each task is
    attempted at most once per pass, so a failure that computes a near-zero backoff cannot spin
    inside a pass.
    """
    from mantle.services import content_service as cs

    if not cs.edge_store_configured():
        # The air-gap invariant. A node with no object store owes no mirror, has no queue, and
        # must not pay even a read to discover that. This is the same predicate
        # `put_bytes_encrypted` uses to decide there is nothing to defer.
        return {"skipped": "no object store is configured on this node"}

    tasks = getattr(store_db, "artifacts", None)
    if tasks is None or not hasattr(tasks, "pending_window"):
        return {"skipped": "the store has no work pool"}

    me = worker_id or _worker_id(store_db)
    when = _now()
    # Before anything is read under this operator's own content type: rows written under the
    # shared one, which nothing else will ever discharge. Ordered ahead of `_reclaim_own` so a
    # row that was claimed when a previous incarnation died is reclaimable in the same pass it
    # is adopted in.
    adopted = adopt_shared_pool_tasks(store_db)
    out: Dict[str, Any] = {"claimed": 0, _DONE: 0, _RETRY: 0, _DEAD: 0, "lost": 0,
                           "worker": me, "adopted": adopted,
                           "reclaimed": _reclaim_own(tasks, me)}
    window_kwargs: Dict[str, Any] = {"now_iso": _iso(when), "operator": MIRROR_OPERATOR}
    if limit is not None:
        window_kwargs["limit"] = int(limit)
    try:
        window = tasks.pending_window(MIRROR_TASK_CT, **window_kwargs)
    except Exception:
        logger.warning("the mirror drain could not read its queue", exc_info=True)
        return {"skipped": "the work pool could not be read"}

    for row in window:
        tid = row.get("id")
        if not tid or not tasks.try_claim(tid, worker_id=me, now_iso=_iso(when)):
            continue                       # another drain won it; `try_claim` is the compare-and-set
        out["claimed"] += 1
        _in_flight.add(tid)
        try:
            _one(store_db, tasks, tid, me, out)
        finally:
            _in_flight.discard(tid)

    out["next_eligible"] = _earliest_retry(tasks)
    return out


def _one(store_db, tasks, tid: str, me: str, out: Dict[str, Any]) -> None:
    """One task this caller has already won: run it, settle it, count it.

    Split from the loop so the `_in_flight` bookkeeping around it has a single exit, and so the
    claim is accounted for on every path including the ones that raise.
    """
    task = tasks.get_artifact(tid)
    if task is None:
        # `try_claim` won the CAS but the document is gone: the sidecar and the document store
        # disagree. Same reading `pool.claim` gives it — reported, not presented as success.
        logger.error("claimed mirror task %s has no artifact — the task index and the "
                     "document store disagree", tid)
        out["lost"] += 1
        return
    try:
        result = mirror_one(store_db, task)
    except _Permanent as exc:
        result = {"outcome": "dead", "reason": str(exc)}
    except Exception as exc:
        logger.warning("mirror task %s raised", tid, exc_info=True)
        result = {"outcome": "retry", "elapsed_s": 0.0,
                  "reason": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
    state = _settle_after(tasks, tid, worker_id=me, task=task, result=result, when=_now())
    out[state] = out.get(state, 0) + 1
    if state == _DEAD:
        logger.error("mirror task %s is DEAD and will not be retried: %s. The content is "
                     "still local and readable on this node; it is not reachable from a peer.",
                     tid, result.get("reason"))
    elif state == "lost":
        logger.info("mirror task %s was reclaimed or refreshed while this drain worked on it; "
                    "the newer record stands and this attempt is discarded", tid)


def adopt_shared_pool_tasks(store_db) -> int:
    """Move this operator's LIVE rows off :data:`SHARED_TASK_CT` onto :data:`MIRROR_TASK_CT`.
    Returns how many moved.

    The rename is only half a fix while the rows it was made for are still under the old name.
    A `pending` row there is an obligation nothing will now discharge — this drain no longer looks
    at that content type — while remaining exactly what an ember worker still claims and
    dead-letters. So it is moved, at the head of every pass, before the window is read.

    **The move is a re-put of the row's own doc through `_put_op`, and that is the whole
    mechanism.** `ct` is a projected column of the doc, so writing the doc back with a different
    `content_type` is what moves it: `_recount` swaps `c_ct(old) -> c_ct(new)` and `_sync_task`
    swaps `c_task_status(old, status) -> c_task_status(new, status)` and rewrites `task.ct`, all
    in the one transaction the put already opens. Nothing else changes — same id, same
    `task_key`, same `status`, same `attempts`, same `next_retry_at`, same claim. A row claimed
    when the process died is still claimed after the move, which is what leaves it visible to
    :func:`_reclaim_own`.

    **LIVE means `pending` or `claimed`, and the two accessors used to find them are the two this
    drain already uses.** `pending_window` and `claimed` are index seeks into one `(ct, status)`
    bucket each, and both buckets are the live queue — small by construction. There is no walk of
    the shared pool's `done`/`failed`/`dead` history, which is the part that grows without bound
    on a node that also runs an ember worker, and which is why this can afford to run every pass
    instead of once behind a flag that would have to be remembered and reset.

    **Terminal rows stay where they are, and that is not an omission.** A `done` row is a record
    of a mirror that happened; a `dead` row is an obligation already determined not to be worth
    re-attempting. Moving either changes no outcome, and REVIVING a dead one would re-run work
    whose premises were found false. The one case that matters — a row an ember worker
    dead-lettered without looking at the bytes — is recovered by the thing that recovers it
    anyway: `mirror_task_id` is derived from the `content_key`, so the next failed upload of that
    content rewrites THAT SAME ROW, under the new content type, `pending`, with a fresh attempt
    count. The obligation comes back the moment it is owed again.

    Never raises. A store that cannot answer is reported and the pass continues to the work it
    can see; the rows are durable and the next pass tries again.
    """
    from mantle.mesh.sync import _put_op

    tasks = getattr(store_db, "artifacts", None)
    if tasks is None:
        return 0
    ids = []
    for read in (lambda: tasks.pending_window(SHARED_TASK_CT, operator=MIRROR_OPERATOR),
                 lambda: tasks.claimed(SHARED_TASK_CT)):
        try:
            rows = read()
        except Exception:
            logger.warning("could not read the shared work pool while adopting mirror tasks",
                           exc_info=True)
            continue
        for r in rows:
            rid = r.get("id")
            # `pending_window` narrowed in SQL; `claimed` cannot, so the operator is checked here.
            if rid and rid not in ids and r.get("operator", MIRROR_OPERATOR) == MIRROR_OPERATOR:
                ids.append(rid)

    moved = 0
    for rid in ids:
        try:
            doc = tasks.get_artifact(rid)
            if doc is None or doc.get("operator") != MIRROR_OPERATOR:
                continue
            d = dict(doc)
            d["content_type"] = MIRROR_TASK_CT
            _put_op(store_db, d)
            moved += 1
        except Exception:
            logger.warning("could not adopt mirror task %s onto %s", rid, MIRROR_TASK_CT,
                           exc_info=True)
    if moved:
        logger.info("adopted %d mirror task(s) from %s onto %s", moved, SHARED_TASK_CT,
                    MIRROR_TASK_CT)
    return moved


def _reclaim_own(tasks, me: str) -> int:
    """Return to `pending` any mirror task still claimed under THIS worker's own id.

    A claim only outlives the attempt that made it if the attempt did not finish — the process was
    killed, or shut down, mid-write. Nothing else in mantle reclaims a stale lease (ember's
    `reclaim_stale` runs only where an ember pool runs), so without this a node that restarts while
    a mirror write is in flight strands that obligation in `claimed` forever: `pending_window` does
    not see it, and only a fresh failed upload for the same key would ever rewrite it.

    Scoped to `claimed_by == me` — `<origin>:<pid>` — because that is the one claim this process
    can prove is not live: it is the id THIS process will use, and no other live process on this
    node can hold it. A broader sweep would be this drain reclaiming a sibling worker's in-flight
    lease, which is the fleet-wide policy decision `release` exists for and this is not.

    `_in_flight` subtracts the claims this process is holding RIGHT NOW, which is what keeps the
    proof true when two passes overlap inside one process (a manual call alongside the runner's
    loop). Without it the newer pass would reclaim the older one's live lease — recoverable,
    because the put is idempotent by key and the loser's `settle` reports `False`, but it would
    be this function creating exactly the race it exists to clean up after.
    """
    reclaimed = 0
    try:
        rows = tasks.claimed(MIRROR_TASK_CT)
    except Exception:
        return 0
    for r in rows:
        if r.get("id") in _in_flight:
            continue
        if r.get("claimed_by") == me and r.get("operator") == MIRROR_OPERATOR:
            try:
                if tasks.release(r["id"], to_status=_RETRY):
                    reclaimed += 1
            except Exception:
                logger.debug("could not reclaim %s", r.get("id"), exc_info=True)
    if reclaimed:
        logger.info("mirror drain reclaimed %d task(s) left claimed by a previous incarnation of "
                    "%s", reclaimed, me)
    return reclaimed


def _earliest_retry(tasks) -> Optional[str]:
    """The earliest `next_retry_at` standing on this operator's pending rows, or None if any row
    is eligible now (NULL sorts first as "now") or there are no rows at all.

    Read from the same bounded window the drain claims from, so it costs nothing extra and cannot
    disagree with what the next pass will see.
    """
    try:
        rows = tasks.pending_window(MIRROR_TASK_CT, operator=MIRROR_OPERATOR)
    except Exception:
        return None
    stamps = [r.get("next_retry_at") for r in rows]
    if not stamps:
        return None
    if any(s is None for s in stamps):
        return _iso(_now())
    return min(str(s) for s in stamps)


# ── the runner ───────────────────────────────────────────────────────────────────────────────

_task: Optional["asyncio.Task"] = None
_wake: Optional[asyncio.Event] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_stopping = False


def notify_pending() -> None:
    """Tell a sleeping drain that a new obligation exists. Called from `_record_mirror_pending`.

    Thread-safe and best-effort in both directions: it runs on the request's worker thread, so it
    hops to the loop with `call_soon_threadsafe`, and it does nothing at all when no drain is
    running (an air-gapped node, a CLI, a test). A missed wake costs latency and never
    correctness — the row is already durable, and the next pass finds it.
    """
    loop, wake = _loop, _wake
    if loop is None or wake is None or loop.is_closed():
        return
    try:
        loop.call_soon_threadsafe(wake.set)
    except RuntimeError:
        pass


async def _run(store_db) -> None:
    """Pass, then sleep exactly as long as the queue says to. No interval lives here.

    `drain_mirror_pending` does not raise, so the guard below is for the impossible case, and it
    resolves to "wait for new work" rather than to a retry timer on purpose: a pass that raised
    could not read or write the store at all, which no interval repairs, and a timer over it would
    be a hot loop against a broken store. The obligation is durable either way — the next upload
    wakes this, and the next boot runs a pass unconditionally.
    """
    while not _stopping:
        try:
            report = await asyncio.to_thread(drain_mirror_pending, store_db)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("the mirror drain pass failed", exc_info=True)
            report = {}
        if report.get(_DONE) or report.get(_DEAD) or report.get(_RETRY):
            logger.info("mirror drain: %s", {k: v for k, v in report.items() if v})
        delay = _sleep_for(report.get("next_eligible"))
        if _wake is None:
            return
        _wake.clear()
        try:
            # `delay is None` means the queue is empty: there is nothing to be scheduled FOR, so
            # the drain waits for work to appear instead of waking to look for it.
            await asyncio.wait_for(_wake.wait(), timeout=delay) if delay is not None \
                else await _wake.wait()
        except asyncio.TimeoutError:
            pass


def _sleep_for(next_eligible: Optional[str]) -> Optional[float]:
    """Seconds until `next_eligible`, or None when there is nothing pending."""
    if not next_eligible:
        return None
    try:
        when = datetime.fromisoformat(str(next_eligible))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - _now()).total_seconds())


def start_mirror_drain(store_db) -> bool:
    """Start the drain for this process, or decline and say why. Idempotent.

    Declines on a node with no object store, and that is the air-gap invariant costing exactly one
    predicate at boot: no task is created, so nothing is scheduled, nothing wakes, and nothing is
    read. A node that gains an object store later gains its drain on the next restart, which is
    when its clients are rebuilt anyway (`reinit_edge_clients`).
    """
    global _task, _wake, _loop, _stopping
    if _task is not None and not _task.done():
        return True
    from mantle.services import content_service as cs
    if not cs.edge_store_configured():
        logger.debug("no object store on this node, so no mirror drain — nothing can be owed")
        return False
    _stopping = False
    _wake = asyncio.Event()
    _loop = asyncio.get_running_loop()
    _task = asyncio.create_task(_run(store_db))
    logger.info("mirror drain started (%s)", _worker_id(store_db))
    return True


async def stop_mirror_drain() -> None:
    """Cancel the drain and forget it. Safe when it was never started."""
    global _task, _wake, _loop, _stopping
    _stopping = True
    task, _task = _task, None
    _wake, _loop = None, None
    if task is None:
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass
