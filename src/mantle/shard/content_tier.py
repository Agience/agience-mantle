"""The content promote drain — the one path from the lattice tier (`store.content_tier`) up to the S3 mirror.

The tier itself lives in mantle (`mantle.db.content_tier.TieredContentStore`):
`FileContentCache` (local, encrypted, verify-on-read) in front of the generic S3/CDN mirror
(`s3_content.S3ContentStore`, immutable `cas/` headers). This module owns what a node does with it:

  • write   → local only (`tier.put`, via the store) — ingest never blocks on the WAN.
  • promote → this drain copies local content up to the mirror (`tier.promote_one`: verified local
              read, re-encrypted under the shared cipher so the mirror is fleet-readable;
              content-addressed → skip-if-exists → idempotent and write-once, which is what makes
              the mirror's `Cache-Control: immutable` true). op.content.promote.
  • evict   → `tier.evict_local` after a confirmed promote, plus the operator-floored
              `tier.evict_for_space` (EMBER_CACHE_MIN_FREE_GB; unset ⇒ nothing evicted) — the
              working-set bound. Never deletes a copy not proven remote.
  • read    → not here: `content.resolve_text` reads through the tier (local → CDN pull-through
              with sha256 verify). The drain is the write side of the same one path.

The walk is `(_origin, _seq)` only: injective, gap-free, clock-free (see `_seq_page`).

Config: <keys_dir>/ovh.access_key + ovh.secret_key select the mirror (absent → air-gapped, drain
no-ops); EMBER_OVH_ENDPOINT/REGION/BUCKET point it; CONTENT_READ_URL_BASE (in mantle's
`s3_content`) puts a CDN in front of reads.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def open_ovh_store(keys_dir: Optional[Path]):
    """Open the content mirror (the tier's remote): the OVH S3 store when creds are present, else a
    directory mirror when `EMBER_CONTENT_DIR` is set (an on-prem NAS/mounted volume — same `cas/`
    interface, no cloud, no creds), else None (air-gapped, local-only). One selector, two backends."""
    # 1) OVH S3 mirror (cloud) — creds in the keys dir select it.
    if keys_dir:
        kd = Path(keys_dir)
        ak, sk = kd / "ovh.access_key", kd / "ovh.secret_key"
        if ak.exists() and sk.exists():
            try:                               # the generic S3 mirror
                from mantle.db.s3_content import S3ContentStore
            except ImportError:
                from db.s3_content import S3ContentStore
            # The connection pool is a resource-envelope concern, derived in one place from the
            # measured ceiling (cgroup-aware — on these pods `nproc`/`free` report the host).
            from prism import envelope as _res
            return S3ContentStore(
                endpoint_url=os.getenv("EMBER_OVH_ENDPOINT", "https://s3.ca-east-tor.io.cloud.ovh.net"),
                region=os.getenv("EMBER_OVH_REGION", "ca-east-tor"),
                access_key=ak.read_text(encoding="utf-8").strip(),
                secret_key=sk.read_text(encoding="utf-8").strip(),
                bucket=os.getenv("EMBER_OVH_BUCKET", "agience-genesis-content"),
                max_pool_connections=_res.s3_pool())
    # 2) SMB mirror (on-prem, userspace — no CIFS mount, no sudo) — a gated account on the NAS share.
    #    The path for a WSL ember that can't mount: it talks SMB directly. EMBER_SMB_SERVER selects it.
    if os.getenv("EMBER_SMB_SERVER"):
        try:
            from mantle.db.s3_content import SMBContentStore
        except ImportError:
            from db.s3_content import SMBContentStore
        return SMBContentStore(
            os.getenv("EMBER_SMB_SERVER"), os.getenv("EMBER_SMB_SHARE", "shared"),
            os.getenv("EMBER_SMB_USER"), os.getenv("EMBER_SMB_PASS"),
            prefix=os.getenv("EMBER_SMB_PREFIX", "agience-genesis/_content"),
            port=os.getenv("EMBER_SMB_PORT", "445"))
    # 3) directory mirror (on-prem) — a shared folder reachable synchronously (a CIFS mount); point
    #    EMBER_CONTENT_DIR at it. A batch relay cannot serve read-through misses.
    content_dir = os.getenv("EMBER_CONTENT_DIR")
    if content_dir:
        try:
            from mantle.db.s3_content import DirContentStore
        except ImportError:
            from db.s3_content import DirContentStore
        return DirContentStore(content_dir)
    return None


# How many consecutive cycles stuck at the same cursor position with errors before the drain gives
# up on those refs and advances past them (recording them). 3 cycles is ~4-5 minutes at the
# measured ~90s cycle: long enough that a transient outage is retried several times, short enough
# that a permanently-broken ref cannot cost a night.
_STUCK_MAX = int(os.getenv("EMBER_PROMOTE_STUCK_MAX", "3"))

#: How many failed refs one drain round collects detail for. A memory bound on the round: `errors`
#: is counted in full and is always exact, so nothing about the drain's outcome changes with this
#: value — only how much of the tail is available to the cursor's published sample and to
#: quarantine. A different value would be right if quarantine ever needed to be complete rather
#: than a sample, at which point the failed refs belong in their own rows, not in a bounded list
#: on a cursor artifact.
_ERR_SAMPLE = 200

#: Upper bound on the persisted quarantine list carried on the cursor artifact. Same character:
#: `quarantined_total` is published alongside it, so overflow is visible rather than silent.
_QUARANTINE_MAX = 1000


# Shared with stats.py rather than restated, since both modules must agree on whether a store
# exposes a typed method — two copies of that rule is how they could disagree about what an empty
# result meant.
from mantle.db.lattice_api import typed_method as _typed                                             # noqa: E402


def _backfill_page(artifacts, after_id: str, n: int):
    """One page of the one-shot id backfill: `WHERE id >:cur ORDER BY id LIMIT:n`.

    **Returns `(rows, ok)`, and `ok` is the entire point of this function existing.**

    The rule it enforces is LATTICE-IMPLEMENTATION §1.4, the same one the mesh consume path
    enforces: **a consume cursor may never advance past a segment that did not apply.** Retiring
    the backfill is advancing a cursor — past the whole corpus. Empty-because-finished and
    empty-because-broken are different facts and a single `[]` cannot carry both, so the failure is
    returned separately and the caller holds both `last_id` and `backfill_done` when it arrives.

    One path: typed `page_by_id` only. A store without the typed walk gets `([], False)` — no
    answer, not an empty corpus."""
    page = _typed(artifacts, "page_by_id")
    if page is None:
        return [], False
    try:
        return list(page(after=after_id, limit=int(n))), True
    except Exception:
        return [], False


def _seq_page(artifacts, after_seq: int, n: int):
    """One page of the primary walk under `(_origin, _seq)`: `WHERE _origin =:me AND _seq >:cur
    ORDER BY _seq LIMIT:n`. Returns `(rows, ok)`.

    `_seq` is allocated from the store, one value per write, gap-free and injective, so there are
    no duplicate groups to drain and a strict `>` is simply correct. It is allocated inside the
    writing transaction under `BEGIN IMMEDIATE`, and the counter it comes from commits with the
    row — a `_seq` is visible exactly when it is allocated, and allocations are serialised. There
    is no commit-lag window to hold a cursor back from, so there is no horizon and no clock to
    compare against; neither this walk nor its cursor reads a clock."""
    page = _typed(artifacts, "page_by_origin")
    if page is None:
        return [], False
    try:
        return list(page(after_seq=int(after_seq), limit=int(n))), True
    except Exception:
        return [], False


_BACKLOG_FIELDS = ("backlog", "backlog_at", "backlog_phase")


def _fresh_backlog(store, cid: str) -> Dict[str, Any]:
    """The backlog fields as they stand right now, for the drain's cursor write to carry forward.

    These belong to the health loop's 900s pass; this module only has to avoid destroying them.
    Read immediately before the write, never from the copy taken at the top of a ~75s drain cycle —
    health can land its measurement inside that window, and carrying the stale copy wipes it."""
    try:
        doc = store.artifacts.get_artifact(cid) or {}
        return {k: doc[k] for k in _BACKLOG_FIELDS if k in doc}
    except Exception:
        return {}          # never let a read failure here abort a completed page of promotion


def promote_local_content(store, *, max_refs: int = 2000, page: int = 200,
                          workers: int = None, evict: bool = None) -> Dict[str, Any]:
    """Async promote: copy this box's local ciphertext up to the OVH origin so S3 is authoritative.
    Content-addressed → idempotent (skip refs already in OVH). Cursor-resumable via a checkpoint artifact.

    Parallel: each ref's exists→get→put is a WAN round-trip; serial promotion tops out at a few refs/s
    and falls hopelessly behind ingest. The page fans out across a thread pool (boto3 is thread-safe
    per call), so throughput scales with `workers` — set high to hide WAN latency.

    Evict (default on): once a ref is confirmed in S3, delete the local copy. S3 is authoritative, so
    the local content cache becomes a working set — cold just-ingested content is offloaded (bounded disk on
    the 100 GB ingest boxes), and hot content re-caches for free on the next read (TieredContentStore.get
    pulls-through on a miss). Disable with EMBER_PROMOTE_NO_EVICT=1 (e.g. a box that must retain content).
    Never deletes a ref that wasn't confirmed in S3 first — a promote failure can't lose content."""
    from concurrent.futures import ThreadPoolExecutor
    from prism import grounding as genesis   # contract, not the runner: only the provenance
    # rung + citation id + timestamp are used here. mantle may reach prism; it may not reach ember.
    # One path: the lattice tier (mantle `TieredContentStore`) on `store.content_tier` — see
    # LocalStore: writers hand `store.content` ciphertext, which the tier's verifying surface
    # refuses, so the two are separate handles. Promotion goes through `tier.promote_one`
    # (verified local read, re-encrypted under the shared cipher so the mirror is fleet-readable)
    # and eviction through `tier.evict_local` (only ever deletes a remote-confirmed copy).
    # No tier, or a tier with no mirror, is a no-op with a reason — an air-gapped box is a
    # first-class configuration, not an error.
    mtier = getattr(store, "content_tier", None)
    if mtier is None or getattr(mtier, "remote", None) is None:
        return {"promoted": 0, "reason": "no content tier with an S3 mirror configured"}
    if workers is None:
        # Envelope-derived, env still wins — same source as the pool that serves it, so the two
        # cannot drift apart (a fan-out larger than its pool is just threads queueing).
        from prism import envelope as _res
        workers = _res.promote_workers()
    if evict is None:
        evict = os.getenv("EMBER_PROMOTE_NO_EVICT", "") in ("", "0", "false")

    def _one(pair):
        """`promote_one` → optional evict, for one (ref, collection). Returns
        ('put'|'skip'|'err', ref, reason).
        remote-confirmed rule — a promote failure cannot lose content, enforced one layer down."""
        ref, coll = pair
        try:
            outcome = mtier.promote_one(ref, collection=coll)
            if evict:
                try:
                    mtier.evict_local(ref)
                except Exception:
                    pass
            return (("put" if outcome == "put" else "skip"), ref, None)
        except Exception as exc:
            return ("err", ref, "%s: %s" % (type(exc).__name__, str(exc)[:160]))

    cid = "content.promote.cursor"
    cur = store.artifacts.get_artifact(cid) or {}
    # Keyset paging, not SKIP: `WHERE content_ref IS NOT NULL SKIP {offset} LIMIT {page}` is
    # O(offset) and this cursor grows monotonically into the millions, so every drain would pay
    # that cost; `content_ref IS NOT NULL` also cannot be index-served (LSM skips nulls), so that
    # shape forces a full scan under every SKIP.
    # `WHERE id >:cur ORDER BY id` rides the id unique index and stays flat regardless of depth
    # (~743ms at depth 5M). content_ref is filtered in Python: adding it to the WHERE flips the
    # planner off the id index.
    last_id = str(cur.get("last_id", "") or "")
    promoted = skipped = scanned = errors = 0
    failed: List[Dict[str, Any]] = []
    exhausted = False
    # ── Why the drain has two walks ─────────────────────────────────────────────────────────────
    # Primary: walk `_seq`, monotone in write order (see `_seq_page` for why a strict `>` is
    # correct here). An id-ordered walk cannot serve as the primary walk on its own, because ids
    # come from the source (`wiki-en-44245722`), not from insertion order: ingesting shard 21 after
    # shard 6 writes rows that sort behind a cursor that has already passed them, and a
    # forward-only id walk misses those permanently.
    #
    # Backfill: an LSM index skips nulls, so a `_seq`-ordered walk never returns a row whose `_seq`
    # is absent. Any row written before `_seq` existed would otherwise be invisible to the drain.
    # So the id walk is kept as a one-time backfill: it runs to exhaustion once, sets
    # `id_backfill_done`, and is never re-armed.
    #
    # Two properties worth knowing before changing this:
    #
    # (a) `seq_cur` stays at its persisted value through the backfill, so when the seq walk takes
    #     over it resumes from there and walks the corpus once from that point rather than from the
    #     max `_seq` seen during the backfill — seeding from the max would skip any row written
    #     while the backfill was running. One extra pass is the cost of not having a skip window;
    #     after it, steady state is O(new).
    #
    # (b) A row with no `_seq` at all is invisible to the seq walk regardless of when it arrives;
    #     the backfill above is the only walk that reaches it.
    #
    # A cursor artifact carrying the fields `rev`/`rev_id`/`grp_done` (not part of the current
    # cursor shape) is read but not written back — those fields are simply dropped.
    typed_walk = _typed(store.artifacts, "page_by_origin") is not None
    seq_cur = int(cur.get("seq") or 0)
    backfill_done = bool(cur.get("id_backfill_done"))
    walk_error = None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while scanned < max_refs:
            if not backfill_done:
                rows, ok = _backfill_page(store.artifacts, last_id, page)
                if not ok:
                    walk_error = "backfill page unreadable"
                    break
                if not rows:
                    backfill_done = True             # permanently; not re-armed on a timer
                    last_id = ""
                    continue
            elif not typed_walk:
                # The (_origin, _seq) walk is the only primary walk. A store without it cannot be
                # drained, and that is reported rather than guessed at with raw SQL — an
                # unanswerable store must not read as an exhausted one.
                walk_error = "store lacks the typed (_origin, _seq) walk"
                break
            else:
                rows, ok = _seq_page(store.artifacts, seq_cur, page)
                if not ok:
                    walk_error = "seq page unreadable"
                    break
                if not rows:
                    exhausted = True                 # caught up: steady state, not an error.
                    break
                # Strict `>` is correct: `_seq` is injective, so there is nothing to resume
                # mid-group and no group to be done with.
                seq_cur = max(seq_cur, int(rows[-1].get("_seq") or seq_cur))
            scanned += len(rows)
            # The lattice page walks return the full `doc`, so the collection each ref keys
            # under travels WITH the row — `promote_one` needs it for the verified local read.
            work = pool.map(_one, [
                (r.get("content_ref"), (r.get("doc") or {}).get("collection_id"))
                for r in rows if r.get("content_ref")])
            # Errors are counted and returned. `_one` returns "err" on any exception from
            # remote.exists / remote.put / local.get.
            for outcome, _ref, _why in work:                   # fanned out concurrently above
                if outcome == "put":
                    promoted += 1
                elif outcome == "skip":
                    skipped += 1
                elif outcome == "err":
                    errors += 1
                    if len(failed) < _ERR_SAMPLE:    # bounded: a sample is enough to diagnose
                        failed.append({"ref": _ref, "why": _why})
            # Only the backfill walk advances the id cursor; the seq walk advances `seq_cur`
            # (done above, before the refs are dispatched).
            if not backfill_done:
                last_id = str(rows[-1].get("id") or last_id)
            if len(rows) < page:
                # A short page ends this pass. It does not reset anything: the backfill completes
                # exactly once (handled above), and the seq walk simply resumes from `seq_cur`
                # next cycle. Nothing re-scans.
                exhausted = True
                break
    #
    # A retry that has failed identically N times is not a retry, it is a loop. After _STUCK_MAX
    # rounds stuck at the same position we advance past the poison refs — but never silently: the
    # refs and their reasons are persisted to the cursor so the content is accounted for, not lost
    # in a counter. A box must not be retired on a drain that quarantined refs.
    _page_start = str(cur.get("last_id", "") or "")
    _stuck_rounds = (int(cur.get("stuck_rounds") or 0) + 1
                     if (errors and str(cur.get("stuck_at") or "") == _page_start) else
                     (1 if errors else 0))
    _force = bool(errors) and _stuck_rounds >= _STUCK_MAX
    _quarantine = list(cur.get("quarantine") or [])
    if _force:
        for f in failed:
            if (len(_quarantine) < _QUARANTINE_MAX
                    and f.get("ref") not in {q.get("ref") for q in _quarantine}):
                _quarantine.append({**f, "at": __import__("time").time()})
    store.artifacts.put_artifact({
        "id": cid, "content_type": "application/vnd.agience.content-cursor+json",
        "state": "committed",
        "last_id": (last_id if (not errors or _force) else cur.get("last_id", "")),
        # What failed, and why — a bounded sample, so "errors: 5" is diagnosable from the published
        # cursor instead of requiring an ssh session and a hand-written probe.
        "last_errors": failed[:20],
        "stuck_at": (_page_start if errors and not _force else None),
        "stuck_rounds": (0 if _force else _stuck_rounds),
        "quarantine": _quarantine,
        "quarantined_total": len(_quarantine),
        **({} if (errors and not _force) else {"seq": seq_cur}),
        # A page the store could not answer is reported, not folded into `errors` (which counts
        # per-ref promotion failures) and not left implicit in an unchanged cursor.
        "walk_error": walk_error,
        "id_backfill_done": backfill_done,
        "promoted_total": (int(cur.get("promoted_total") or 0) + promoted),
        # Monotone counter so the rate is derivable. `promoted_total` alone is a poor progress
        # signal because most pages legitimately promote 0 (the ref is already in S3), so the
        # number sits still while the drain is working hard. `scanned_total` always advances while
        # the drain runs, so stats._rate_from_history can difference it into refs/min.
        "scanned_total": (int(cur.get("scanned_total") or 0) + scanned),
        "swept_at": (__import__("time").time() if (exhausted and last_id == "")
                     else cur.get("swept_at")),
        **{k: v for k, v in _fresh_backlog(store, cid).items()},
        "exhausted": bool(exhausted and backfill_done and not errors),
        "exhausted_at": (__import__("time").time()
                         if (exhausted and backfill_done and not errors) else cur.get("exhausted_at")),
        # `offer`, not `content` — corrected 2026-08-26. A constant label describing what this
        # cursor IS is a title, and `offer` is the one naming field [John, 2026-08-25]. `content`
        # is what `artifacts_holding_inline_plaintext` counts, pinned at 0, so a fixed label parked
        # there is a row of that population for no benefit — nothing reads this string back.
        # 0 rows of this content-type exist on 71/home, so this corrects a latent path before it
        # first runs rather than repairing anything.
        "offer": "local content-cache -> OVH promote cursor", "provenance": genesis.P_HUMAN,
        "cited_from": genesis.CITE_GENESIS, "updated": genesis._now()})
    # `errors` is reported, never swallowed: a caller must be able to tell "nothing to
    # promote here" from "every promotion failed" apart.
    # Evict read-through re-caches alongside the walk, so local disk is bounded under read traffic
    # and not only under ingest. Bounded work per cycle; never blocks the drain.
    try:
        # The tier re-caches read-throughs internally (there is no ring to drain). The working-set
        # bound is `evict_for_space`, gated on an operator-owned floor: EMBER_CACHE_MIN_FREE_GB.
        # Unset ⇒ nothing evicted beyond promote's own — the honest default. The floor is the
        # operator's disk-budget fact, deliberately never a constant baked in here (no arbitrary
        # caps); the mechanism itself refuses to evict any object not proven remote.
        _floor = (os.getenv("EMBER_CACHE_MIN_FREE_GB") or "").strip()
        _rc = (mtier.evict_for_space(min_free_bytes=int(float(_floor) * 2 ** 30), limit=2000)
               if _floor else {"evicted": 0})
    except Exception:
        _rc = {"evicted": 0}
    return {"promoted": promoted, "already_in_origin": skipped, "scanned": scanned,
            "recache_evicted": _rc.get("evicted", 0), "recache_pending": _rc.get("pending", 0),
            "errors": errors,
            "exhausted": exhausted, "last_id": last_id, "evict": evict, "workers": workers,
            "last_errors": failed[:5], "quarantined_total": len(_quarantine),
            "forced_past_errors": _force,
            # "the store would not answer" must not look like "there was nothing to do".
            "walk_error": walk_error, "walk": "seq", "seq": seq_cur}
