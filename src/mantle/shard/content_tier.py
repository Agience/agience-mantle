"""The content promote drain — ONE PATH: the lattice tier (`store.content_tier`) up to the S3 mirror.

The tier itself lives in mantle (`mantle.db.lattice.content_tier.TieredContentStore`):
`FileContentCache` (local, encrypted, verify-on-read) in front of the generic S3/CDN mirror
(`s3_content.S3ContentStore`, immutable `cas/` headers). This module owns what a NODE does with it:

  • WRITE  → local only (`tier.put`, via the store) — ingest never blocks on the WAN.
  • PROMOTE→ this drain copies local content UP to the mirror (`tier.promote_one`: verified local
             read, re-encrypted under the SHARED cipher so the mirror is fleet-readable;
             content-addressed → skip-if-exists → idempotent AND write-once, which is what makes
             the mirror's `Cache-Control: immutable` true). op.content.promote.
  • EVICT  → `tier.evict_local` after a confirmed promote, plus the operator-floored
             `tier.evict_for_space` (EMBER_CACHE_MIN_FREE_GB; unset ⇒ nothing evicted) — the
             working-set bound. Never deletes a copy not proven remote.
  • READ   → not here: `content.resolve_text` reads through the tier (local → CDN pull-through
             with sha256 verify). The drain is the write side of the same one path.

The Garage-era two-tier class that used to live here (local Garage daemon + `_rev` walk + re-cache
ring) was deleted 2026-07-22 with the fleet off ArcadeDB/Garage — per its own quarantine notes.
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
    """Open the content MIRROR (the tier's remote): the OVH S3 store when creds are present, else a
    DIRECTORY mirror when `EMBER_CONTENT_DIR` is set (an on-prem NAS/mounted volume — same `cas/`
    interface, no cloud, no creds), else None (air-gapped, local-only). One selector, two backends."""
    # 1) OVH S3 mirror (cloud) — creds in the keys dir select it.
    if keys_dir:
        kd = Path(keys_dir)
        ak, sk = kd / "ovh.access_key", kd / "ovh.secret_key"
        if ak.exists() and sk.exists():
            try:                               # the generic S3 mirror — no legacy-backend coupling
                from mantle.db.lattice.s3_content import S3ContentStore
            except ImportError:
                from db.lattice.s3_content import S3ContentStore
            # The connection pool is a RESOURCE-ENVELOPE concern, derived in one place from the
            # measured ceiling (cgroup-aware — on these pods `nproc`/`free` report the HOST).
            from prism import envelope as _res
            return S3ContentStore(
                endpoint_url=os.getenv("EMBER_OVH_ENDPOINT", "https://s3.ca-east-tor.io.cloud.ovh.net"),
                region=os.getenv("EMBER_OVH_REGION", "ca-east-tor"),
                access_key=ak.read_text(encoding="utf-8").strip(),
                secret_key=sk.read_text(encoding="utf-8").strip(),
                bucket=os.getenv("EMBER_OVH_BUCKET", "agience-genesis-content"),
                max_pool_connections=_res.s3_pool())
    # 2) SMB mirror (on-prem, USERSPACE — no CIFS mount, no sudo) — a GATED account on the NAS share.
    #    The path for a WSL ember that can't mount: it talks SMB directly. EMBER_SMB_SERVER selects it.
    if os.getenv("EMBER_SMB_SERVER"):
        try:
            from mantle.db.lattice.s3_content import SMBContentStore
        except ImportError:
            from db.lattice.s3_content import SMBContentStore
        return SMBContentStore(
            os.getenv("EMBER_SMB_SERVER"), os.getenv("EMBER_SMB_SHARE", "shared"),
            os.getenv("EMBER_SMB_USER"), os.getenv("EMBER_SMB_PASS"),
            prefix=os.getenv("EMBER_SMB_PREFIX", "agience-genesis/_content"),
            port=os.getenv("EMBER_SMB_PORT", "445"))
    # 3) DIRECTORY mirror (on-prem) — a shared folder reachable synchronously (a CIFS mount); point
    #    EMBER_CONTENT_DIR at it. A batch relay CANNOT serve read-through misses.
    content_dir = os.getenv("EMBER_CONTENT_DIR")
    if content_dir:
        try:
            from mantle.db.lattice.s3_content import DirContentStore
        except ImportError:
            from db.lattice.s3_content import DirContentStore
        return DirContentStore(content_dir)
    return None


# How many consecutive cycles stuck at the SAME cursor position with errors before the drain gives
# up on those refs and advances past them (recording them). 3 cycles is ~4-5 minutes at the
# measured ~90s cycle: long enough that a transient outage is retried several times, short enough
# that a permanently-broken ref cannot cost a night.
_STUCK_MAX = int(os.getenv("EMBER_PROMOTE_STUCK_MAX", "3"))


# Shared with stats.py rather than restated. Both modules have to make the SAME judgement about
# whether a store exposes a typed method, and two copies of that rule is exactly how the shim's
# six call sites came to disagree about what an empty result meant.
from mantle.db.lattice_api import typed_method as _typed                                             # noqa: E402


def _backfill_page(artifacts, after_id: str, n: int):
    """One page of the ONE-SHOT id backfill: `WHERE id > :cur ORDER BY id LIMIT :n`.

    **Returns `(rows, ok)`, and `ok` is the entire point of this function existing.**

    ⛔⛔ THIS WAS A CURSOR-DISCIPLINE VIOLATION AND IT RETIRED THE BACKFILL WITHOUT RUNNING IT.
    The call site was a bare `rows = artifacts.c.query(...)` followed by
    `if not rows: backfill_done = True`. On `_SqliteConnShim` that statement matched no pattern
    and returned `[]` — every time, for ever — so on the first drain cycle of a sqlite node the
    backfill was marked complete having read zero rows, `id_backfill_done` was persisted True, and
    nothing in the tree ever writes it back to False (the file's own comment at the `_rev` walk
    says so). The one-shot walk that exists to cover rows the primary walk cannot see was
    permanently skipped, silently, on a node that reported a healthy drain.

    The rule it broke is LATTICE-IMPLEMENTATION §1.4, the same one the mesh consume path enforces:
    **a consume cursor may never advance past a segment that did not apply.** Retiring the backfill
    IS advancing a cursor — past the whole corpus. Empty-because-finished and empty-because-broken
    are different facts and a single `[]` cannot carry both, so the failure is returned separately
    and the caller holds both `last_id` and `backfill_done` when it arrives.

    ONE PATH: typed `page_by_id` only. The raw-SQL fallback died with the shim/ArcadeDB — a store
    without the typed walk gets `([], False)`: no answer, and NOT an empty corpus."""
    page = _typed(artifacts, "page_by_id")
    if page is None:
        return [], False
    try:
        return list(page(after=after_id, limit=int(n))), True
    except Exception:
        return [], False


def _seq_page(artifacts, after_seq: int, n: int):
    """One page of the PRIMARY walk under `(_origin, _seq)`: `WHERE _origin = :me AND _seq > :cur
    ORDER BY _seq LIMIT :n`. Returns `(rows, ok)`.

    **This replaces BOTH `_rev` queries, and one of them ceases to exist.** The old walk needed two
    statements: an equality drain of the CURRENT revision group (`WHERE _rev = :r ORDER BY id`,
    filtered and ordered in Python) and then a strict advance (`WHERE _rev > :r AND _rev <= :ceil`).
    The group drain existed for exactly one reason, recorded at the site: `_rev` was
    `time.time_ns()` stamped in a tight loop, and **2000 successive calls on this hardware yielded
    ONE distinct value**, so a 500-row `put_many` shared a single `_rev` and a bare `_rev > R`
    cursor excluded the 300 rows the first page did not return. Permanently.

    `_seq` is allocated from the store, one value per write, gap-free and injective — so there are
    no duplicate groups to drain, `grp_done` / `rev_id` describe a state that cannot occur, and a
    strict `>` is simply correct. The ~25 lines of group machinery are deleted rather than ported.

    The `_rev` CEILING goes too. `_REV_SLACK_NS` held the horizon 5 minutes back because `_rev` was
    stamped when a doc was PREPARED while the row became visible only at COMMIT, so a slow batch
    could land rows below an advanced cursor. `_seq` is allocated INSIDE the writing transaction
    under `BEGIN IMMEDIATE`, and the counter it comes from commits with the row — a `_seq` is
    visible exactly when it is allocated, and allocations are serialised. There is no commit-lag
    window to hold back from, so there is no horizon and no clock to compare against. That also
    removes the two failure modes `_rev_ceiling` documents (a slow peer's rows falling below our
    cursor; an NTP step backwards reporting `exhausted`), because neither this walk nor its cursor
    reads a clock any more.

    ⚠ ONE THING THIS DOES NOT COVER, AND IT MUST BE SAID RATHER THAN PAPERED OVER.
    `page_by_origin` defaults to the LOCAL origin, so this walk visits rows this observer authored.
    Rows replicated from a peer carry the PEER's `(_origin, _seq)` — preserved deliberately, that is
    what `stamp_rev=False` means — and are invisible to it. The id backfill covers them on its
    single pass; a peer row arriving AFTER that pass is not covered.
    This is not a regression: the `_rev` walk had the same hole in a worse form, since a peer 20
    minutes slow wrote rows below our cursor and the site's own comment records that as a silent
    permanent skip. Under `(_origin, _seq)` the gap is structural and statable instead of
    clock-dependent — we cover origin X exactly iff we hold a cursor for X.
    Closing it properly needs either a per-origin cursor map (which needs an origins enumeration
    the lattice store does not yet expose) or, better and per this file's own precedent for the
    missing-`_rev` hole, closing it AT THE SOURCE in `mesh/sync._apply_artifacts` — the consume
    path already knows exactly which rows it just admitted. Flagged for unit E; not invented here."""
    page = _typed(artifacts, "page_by_origin")
    if page is None:
        return [], False
    try:
        return list(page(after_seq=int(after_seq), limit=int(n))), True
    except Exception:
        return [], False


_BACKLOG_FIELDS = ("backlog", "backlog_at", "backlog_phase")


def _fresh_backlog(store, cid: str) -> Dict[str, Any]:
    """The backlog fields as they stand RIGHT NOW, for the drain's cursor write to carry forward.

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
    """Async promote: copy this box's local ciphertext UP to the OVH origin so S3 is authoritative.
    Content-addressed → idempotent (skip refs already in OVH). Cursor-resumable via a checkpoint artifact.

    PARALLEL: each ref's exists→get→put is a WAN round-trip; serial promotion tops out at a few refs/s
    and falls hopelessly behind ingest. We fan the page out across a thread pool (boto3 is thread-safe
    per call), so throughput scales with `workers` — set high to hide WAN latency.

    EVICT (default on): once a ref is confirmed in S3, delete the LOCAL copy. S3 is authoritative, so
    the local content cache becomes a WORKING SET — cold just-ingested content is offloaded (bounded disk on
    the 100 GB ingest boxes), and hot content re-caches for free on the next read (TieredContentStore.get
    pulls-through on a miss). Disable with EMBER_PROMOTE_NO_EVICT=1 (e.g. a box that must retain content).
    Never deletes a ref that wasn't confirmed in S3 first — a promote failure can't lose content."""
    from concurrent.futures import ThreadPoolExecutor
    from prism import grounding as genesis   # ⚠ CONTRACT, NOT THE RUNNER: only the provenance
    # rung + citation id + timestamp are used here, and those moved to prism on 2026-07-31.
    # mantle may reach prism; it may not reach ember.
    # ONE PATH: the lattice tier (mantle `TieredContentStore`) on `store.content_tier` — see
    # LocalStore: writers hand `store.content` ciphertext, which the tier's verifying surface
    # refuses, so the two are separate handles. Promotion goes through `tier.promote_one`
    # (verified local read, re-encrypted under the SHARED cipher so the mirror is fleet-readable)
    # and eviction through `tier.evict_local` (only ever deletes a remote-CONFIRMED copy).
    # No tier, or a tier with no mirror, is a NO-OP with a reason — an air-gapped box is a
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

        ⛔ THE REASON IS NEVER DISCARDED. `except Exception: return "err"` once counted failures
        and threw away what they WERE, so a permanently-broken ref and a transient 503 were the
        same number on the dashboard — and the cursor-hold below treats those two cases
        oppositely. The skip branch inside `promote_one` IS the exists-check, so the WAN cost
        profile is one round-trip for settled refs; eviction defers to `evict_local`'s
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
    # KEYSET, NOT SKIP. This was `... WHERE content_ref IS NOT NULL SKIP {offset} LIMIT {page}` and
    # it was the single most expensive query in the fleet -- a double fault:
    #   * SKIP is O(offset): measured 142s at depth 5M, and this cursor grows monotonically into
    #     the millions, so every drain paid that cost.
    #   * `content_ref IS NOT NULL` cannot be index-served at all (LSM skips nulls), so the SKIP
    #     was applied over a FULL SCAN.
    # worker.py drives up to 200 drains per cycle, so each cycle stacked full scans. The client
    # gave up at its 60s read timeout while ArcadeDB kept executing server-side -- observed on T5
    # as request threads pegged at 99.9% with GC thrash and zero progress for hours.
    # `WHERE id > :cur ORDER BY id` rides the id UNIQUE index and is FLAT (~743ms at depth 5M).
    # content_ref is filtered in Python: adding it to the WHERE flips the planner off the id index.
    last_id = str(cur.get("last_id", "") or "")
    promoted = skipped = scanned = errors = 0
    failed: List[Dict[str, Any]] = []
    exhausted = False
    # ── WHY TWO CURSORS ─────────────────────────────────────────────────────────────────────────
    # PRIMARY: walk `_rev`, which is monotone in WRITE order. The old walk was `id`-ordered, and ids
    # come from the SOURCE (`wiki-en-44245722`), NOT from insertion — so ingesting shard 21 after
    # shard 6 writes rows that sort BEHIND a cursor which has already passed them. A forward-only id
    # walk misses those permanently, and that is the only reason the old code reset to "" and
    # re-scanned the entire corpus every time it finished: a full re-scan was the sole way to find
    # work it had stepped over. `_rev` cannot be stepped over, so the re-sweep — and the arbitrary
    # 6h timer I had put on it — are both gone rather than tuned.
    #
    # BACKFILL: an LSM index SKIPS NULLS, so a `_rev`-ordered walk never returns a row whose `_rev`
    # is absent. Any legacy row written before `_rev` existed would be invisible to the drain
    # FOREVER — strictly worse than the timer it replaces. So the id walk is kept as a ONE-TIME
    # backfill: it runs to exhaustion once, sets `id_backfill_done`, and is never re-armed.
    #
    # TWO PROPERTIES OF THIS WORTH KNOWING BEFORE CHANGING IT:
    #
    # (a) `rev_cur` stays 0 through the backfill, so when the `_rev` walk takes over it starts from
    #     zero and re-walks the corpus ONCE. That is deliberate, not an oversight: seeding `rev_cur`
    #     from the max `_rev` seen during the backfill would skip any row written WHILE the backfill
    #     was running. One extra pass is the cost of not having a skip window. After it, steady
    #     state is O(new).
    #
    # (b) The residual risk this used to carry — a row arriving AFTER the backfill with NO `_rev`,
    #     invisible to both walks — is now CLOSED AT ITS SOURCE, not here. `sync._apply_artifacts`
    #     stamps a `_rev` on any consumed doc that lacks one (a doc with no origin revision has
    #     nothing to preserve), so the only entry point for an unstamped row is sealed.
    #     The fix once written down here was "re-arm the backfill on a consume event". That was the
    #     WRONG fix and is deliberately not implemented: re-arming costs a full id walk of the whole
    #     corpus on every consume, which is precisely the scan the `_rev` walk exists to eliminate.
    #     Closing the hole where rows enter costs nothing at all.
    #
    # ONE PATH (2026-07-22): the primary walk is `seq_cur` alone — the `_rev` walk and its
    # `rev`/`rev_id`/`grp_done` cursor fields were deleted with the fleet off ArcadeDB (per that
    # code's own quarantine note). A cursor doc that still carries them is simply ignored, and the
    # seq write below never persists them. See `_seq_page` for why (a) survives unchanged and (b)
    # is closed at the source rather than here.
    typed_walk = _typed(store.artifacts, "page_by_origin") is not None
    seq_cur = int(cur.get("seq") or 0)
    backfill_done = bool(cur.get("id_backfill_done"))
    walk_error = None

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while scanned < max_refs:
            if not backfill_done:
                rows, ok = _backfill_page(store.artifacts, last_id, page)
                if not ok:
                    # ⛔ THE READ FAILED. Hold EVERYTHING: the id cursor, and above all
                    # `backfill_done`. An unanswered query is not an exhausted corpus, and
                    # `id_backfill_done` is a ONE-WAY latch that nothing ever clears.
                    walk_error = "backfill page unreadable"
                    break
                if not rows:
                    backfill_done = True             # permanently; not re-armed on a timer
                    last_id = ""
                    continue
            elif not typed_walk:
                # ONE PATH: `(_origin, _seq)` is the only primary walk. A store without it cannot
                # be drained — REPORTED, never guessed at with raw SQL (the raw path died with the
                # shim, and an unanswerable store must not read as an exhausted one).
                walk_error = "store lacks the typed (_origin, _seq) walk"
                break
            else:
                rows, ok = _seq_page(store.artifacts, seq_cur, page)
                if not ok:
                    walk_error = "seq page unreadable"
                    break
                if not rows:
                    exhausted = True                 # caught up. STEADY STATE, not an error.
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
            # ERRORS ARE COUNTED AND RETURNED. `_one` returns "err" on ANY exception from
            # remote.exists / remote.put / local.get, and this loop used to DISCARD that outcome
            # entirely — so a node whose S3 credentials had expired, or whose origin returned 503
            # for every PUT, reported `{"promoted": 0, "already_in_origin": 0, "scanned": 2000}`.
            # That is byte-identical to the healthy "this page has no content_ref" case, so the
            # drain loop treated total failure as ordinary progress and advanced the cursor past
            # every failed ref — content never promoted, disk never bounded, nothing logged.
            for outcome, _ref, _why in work:                   # fanned out concurrently above
                if outcome == "put":
                    promoted += 1
                elif outcome == "skip":
                    skipped += 1
                elif outcome == "err":
                    errors += 1
                    if len(failed) < 200:            # bounded: a sample is enough to diagnose
                        failed.append({"ref": _ref, "why": _why})
            # Only the BACKFILL walk advances the id cursor; the seq walk advances `seq_cur`
            # (done above, before the refs are dispatched).
            if not backfill_done:
                last_id = str(rows[-1].get("id") or last_id)
            if len(rows) < page:
                # A short page ends THIS pass. It does not reset anything: the backfill completes
                # exactly once (handled above), and the seq walk simply resumes from `seq_cur`
                # next cycle. Nothing re-scans.
                exhausted = True
                break
    # ⛔⛔ HOLDING THE CURSOR ON *ANY* ERROR IS A LIVELOCK, AND IT STOPPED THE FLEET.
    # "A cursor may never advance past work that did not apply" is right for a TRANSIENT failure
    # (an OVH 503 window) and catastrophic for a PERMANENT one. Measured on TU and T5 2026-07-20,
    # four consecutive cycles each, byte-identical:
    #     {"promoted":0,"scanned":20000,"already":19995,"errors":5,"last_id":"wiki-simple-41172"}
    # Five refs fail every time, so the cursor is pinned, so the SAME 20,000 rows are re-walked
    # forever and the drain never reaches the rest of the corpus. Both boxes had been "draining"
    # for hours while promoting exactly zero. This is my own fix from earlier today, applied
    # without asking what happens when the error never clears.
    #
    # A retry that has failed identically N times is not a retry, it is a loop. After _STUCK_MAX
    # rounds stuck at the same position we advance PAST the poison refs — but never silently: the
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
            if len(_quarantine) < 1000 and f.get("ref") not in {q.get("ref") for q in _quarantine}:
                _quarantine.append({**f, "at": __import__("time").time()})
    store.artifacts.put_artifact({
        "id": cid, "content_type": "application/vnd.agience.content-cursor+json",
        "state": "committed",
        "last_id": (last_id if (not errors or _force) else cur.get("last_id", "")),
        # What failed, and WHY — a bounded sample, so "errors: 5" is diagnosable from the published
        # cursor instead of requiring an ssh session and a hand-written probe.
        "last_errors": failed[:20],
        "stuck_at": (_page_start if errors and not _force else None),
        "stuck_rounds": (0 if _force else _stuck_rounds),
        "quarantine": _quarantine,
        "quarantined_total": len(_quarantine),
        # ⛔ HOLD THE CURSOR WHEN A PAGE HAD ERRORS.
        # Counting errors is not retrying them. The previous code reported `errors: 200` honestly
        # and then persisted the advanced cursor anyway, so an OVH 503 window meant the drain
        # resumed PAST every ref that failed — never promoted, never evicted, and on a box whose
        # local cache is the only copy, permanently at risk. Same rule the mesh consume path
        # already enforces: a cursor may never advance past work that did not apply.
        # `rev`/`rev_id`/`grp_done` are never written: they describe a duplicate-revision-group
        # state that gap-free `_seq` cannot produce, and persisting dead fields is how a reader
        # (stats.drain_cursor) ends up rendering a position nothing is walking.
        **({} if (errors and not _force) else {"seq": seq_cur}),
        # A page the store could not answer is REPORTED, not folded into `errors` (which counts
        # per-ref promotion failures) and not left implicit in an unchanged cursor.
        "walk_error": walk_error,
        "id_backfill_done": backfill_done,
        # ⛔ `dict.get(k, 0)` RETURNS None WHEN THE KEY IS PRESENT-BUT-NULL, and `None + int` raises.
        # That TypeError propagates out of put_artifact and out of promote_local_content, so the
        # cursor write is lost and an entire completed page of promotion is discarded and re-done
        # every cycle — a silent crash loop that makes no progress forever. The line immediately
        # below already guards this correctly with `int(... or 0)`; the inconsistency was the tell.
        "promoted_total": (int(cur.get("promoted_total") or 0) + promoted),
        # MONOTONE COUNTER so the rate is derivable. `promoted_total` alone is useless as a
        # progress signal because most pages legitimately promote 0 (the ref is already in S3) —
        # the number sits still while the drain is working hard, which is exactly the "nothing ever
        # moves" complaint. `scanned_total` always advances while the drain runs, so
        # stats._rate_from_history can difference it into refs/min.
        "scanned_total": (int(cur.get("scanned_total") or 0) + scanned),
        "swept_at": (__import__("time").time() if (exhausted and last_id == "")
                     else cur.get("swept_at")),
        # ⛔ THE RETIREMENT SIGNAL — AND IT WAS COMPUTED HERE EVERY CYCLE AND THEN THROWN AWAY.
        # `exhausted` means the `_rev` walk reached the slack horizon with nothing left to promote.
        # That IS "this box has nothing further to drain", it is FREE (the walk already knows), and
        # it is the only statement of it that is both cheap and true. It was returned to the caller
        # and never persisted, so `stats` had no way to publish it and fell back on differencing a
        # count — see stats._finish for what that cost.
        # Qualified by `backfill_done`, because a short page during the ONE-TIME id backfill also
        # sets `exhausted` and means only "this page was short", not "the corpus is drained".
        # Qualified by `not errors`, because a page where every promotion failed also ends the pass;
        # treating that as drained is how a box gets retired holding the only copy of its content.
        # Self-clearing: new content moves the horizon and the next cycle writes False.
        # ⛔ THE DRAIN WAS WIPING THE BACKLOG THE HEALTH LOOP HAD JUST MEASURED.
        # `put_artifact` replaces the whole document, and `backlog` / `backlog_at` are written by
        # health-loop's 900s pass — NOT by this function. So every drain cycle (~90s) dropped them,
        # and the field survived for at most ~90 of every 900 seconds. THAT is why every box read
        # "backlog not measured" on the status page while its own health log showed the count
        # succeeding in 1.6-3.9s. The count was never the problem; this write was.
        # Exact mirror of the clobber already fixed in the other direction (health rewinding this
        # cursor's `last_id`/`promoted_total`). Two writers, one document, neither preserving the
        # other's fields. Carry them forward explicitly.
        #
        # ⛔ AND CARRY THEM FROM A FRESH READ, NOT FROM `cur`. `cur` was read at the TOP of this
        # function and a drain cycle runs ~75s (measured on TU). Health writes the backlog inside
        # that window, so carrying from `cur` carries a stale ABSENCE and wipes it just the same —
        # which is exactly what happened on the first attempt at this fix. This is the identical
        # read-modify-write race that health-loop's own "DO NOT WRITE BACK THE DOC WE READ 25s AGO"
        # comment describes; I reproduced it in the mirror direction.
        **{k: v for k, v in _fresh_backlog(store, cid).items()},
        "exhausted": bool(exhausted and backfill_done and not errors),
        "exhausted_at": (__import__("time").time()
                         if (exhausted and backfill_done and not errors) else cur.get("exhausted_at")),
        "content": "local content-cache -> OVH promote cursor", "provenance": genesis.P_HUMAN,
        "cited_from": genesis.CITE_GENESIS, "updated": genesis._now()})
    # `errors` is REPORTED, never swallowed: a caller must be able to tell "nothing to
    # promote here" from "every promotion failed" -- those looked identical before.
    # Evict read-through re-caches alongside the walk, so local disk is bounded under read traffic
    # and not only under ingest. Bounded work per cycle; never blocks the drain.
    try:
        # The tier re-caches read-throughs internally (there is no ring to drain). The working-set
        # bound is `evict_for_space`, gated on an OPERATOR-owned floor: EMBER_CACHE_MIN_FREE_GB.
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
