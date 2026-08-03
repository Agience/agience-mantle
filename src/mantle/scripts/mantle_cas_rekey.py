#!/usr/bin/env python3
"""Migrate a local CAS from PER-COLLECTION at-rest keys to the ONE shared key (mantle §1).

    python -m mantle.scripts.mantle_cas_rekey --cas <dir> --keys-dir <dir> --db <lattice.db>
    python -m mantle.scripts.mantle_cas_rekey ... --dry-run        # report only, touch nothing

⛔ WHY A SCRIPT AT ALL, WHEN THE CACHE MIGRATES ITSELF ON READ.

`FileContentCache.get()` rewrites a legacy object under the shared key the moment it is read, so a
node converges on its own with no downtime and no re-fetch. That is the *safe* path and it is
enough for correctness. This script exists for the two things lazy migration cannot give you:

  1. **An answer to "is this store fully migrated?"** -- the lazy path only touches what someone
     happens to read. Cold objects stay legacy indefinitely, and you cannot retire the
     `legacy_key_for_collection` wiring until you know none are left.
  2. **A count of what CANNOT be opened by either key** — objects that are neither shared-keyed nor
     legacy-keyed under any collection this store knows about. Those are the real casualties of the
     pre-§1 destruction (EREA measured 6 across 39 roots), and they need a re-fetch from the durable
     tier, not a re-key. Lazily, each one surfaces as a single confusing read error months apart.

⛔ THIS SCRIPT NEVER DELETES. Not a legacy object, not an unreadable one, not a stray file. The bug
it exists to clean up after was a delete-on-a-guess; the fix does not get to make the same move.
Rewrites are atomic (tmp + os.replace), so a kill mid-run leaves every object readable under one key
or the other, and re-running resumes.

⚠ VERIFICATION IS NOT OPTIONAL AND HAPPENS BEFORE THE REWRITE. Every object is decrypted AND
sha256-checked against its own ref before anything is written back. Re-keying unverified bytes would
launder corruption into a valid-looking object under the CURRENT key, where it would decrypt cleanly
and be trusted forever — strictly worse than leaving it broken.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

COLLECTION_CT = "application/vnd.agience.collection+json"


def _legacy_roots(db_path: str, *, extra_cts=(), extra_roots=()) -> set:
    """The set of `origin_root` values the OLD scheme could have keyed on. Read-only.

    ⛔ THIS USED TO KEY OFF ONE HARDCODED CONTENT TYPE, AND THAT WAS A REAL DEFECT.

    It read `WHERE ct = 'application/vnd.agience.collection+json'` and built
    `collection_id -> origin_root`. **Any consumer that names its own collection type got an EMPTY
    map** — which is every consumer of a general-purpose store, and is the entire point of shipping
    one. MEASURED by EREA (2026-07-29) on a reconstructed 39-collection / 487-blob corpus whose
    collections are `application/vnd.erea.project+json`: 0 keys derived, **487/487 objects
    classified `unreadable`**, reported as "needs a re-fetch from the durable tier" — which on a
    `remote=None` node reads as total data loss. A false alarm that could have provoked a
    destructive recovery.

    ⭐ THE FIX IS TO READ WHAT THE DERIVATION ACTUALLY CONSUMES. `collection_key` takes an
    `origin_root`, not a collection artifact — so the content type was never load-bearing, only
    incidental to how the roots happened to be discovered. `origin_root` is a REAL COLUMN
    (`schema.py`) with its own index `ix_v_root`, so `SELECT DISTINCT` is an index scan, not a
    corpus dereference. It is also strictly WIDER than the old query: it covers roots whose
    collection artifact was archived, is of an unknown type, or never existed.

    ⚠ THE COLUMN IS A MIGRATION, SO IT MAY BE ABSENT OR NULL. `schema._VERTEX_ADDED_COLUMNS` adds
    `origin_root` to stores that predate it, and MEASURED on node 45 some stores lack it entirely;
    rows written before the backfill carry NULL. So a missing column falls back to the old
    doc-scan, and both sources are UNIONed rather than either being trusted alone.
    """
    roots = set(str(r) for r in extra_roots if r)
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        try:
            # Indexed (ix_v_root). Never a `json_extract` predicate — that is unindexed and scans
            # the whole store regardless of any LIMIT.
            for (r,) in con.execute(
                    "SELECT DISTINCT origin_root FROM vertex WHERE origin_root IS NOT NULL"):
                if r:
                    roots.add(str(r))
        except sqlite3.OperationalError as e:
            if "no such column" not in str(e).lower():
                raise
            print("  note: this store predates the `origin_root` column (%s) — falling back to a "
                  "collection-artifact scan." % e)
        # UNION the doc-level value for the known + caller-supplied collection types. Cheap: `ct`
        # is indexed and these values are selective. Covers a store whose column is un-backfilled.
        for ct in (COLLECTION_CT,) + tuple(extra_cts):
            for (doc,) in con.execute("SELECT doc FROM vertex WHERE ct = ?", (ct,)):
                try:
                    d = json.loads(doc)
                except Exception:
                    continue
                if d.get("origin_root"):
                    roots.add(str(d["origin_root"]))
    finally:
        con.close()
    return roots


def _walk(cas: str):
    """Every CAS object under the two-level fan-out. Skips mkstemp leftovers by name length."""
    for dirpath, _dirs, files in os.walk(cas):
        for fn in files:
            if len(fn) == 64:
                yield "cas/" + fn, os.path.join(dirpath, fn)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cas", required=True, help="the CAS directory")
    ap.add_argument("--keys-dir", required=True, help="dir holding content.key")
    ap.add_argument("--db", required=True, help="lattice sqlite db (for the legacy collection map)")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--limit", type=int, default=0, help="stop after N objects (0 = all)")
    ap.add_argument("--collection-ct", action="append", default=[], metavar="CT",
                    help="an additional collection content type to harvest origin_roots from "
                         "(repeatable). Rarely needed — origin_root is read from its own indexed "
                         "column — but available for a store whose column is un-backfilled.")
    ap.add_argument("--legacy-root", action="append", default=[], metavar="ROOT",
                    help="an origin_root to try explicitly (repeatable). The escape hatch when "
                         "discovery cannot see a root at all.")
    args = ap.parse_args(argv)

    from mantle.db.lattice.content_cache import (CacheCorrupt, ContentKeyMismatch,
                                                 collection_key, shared_content_key)
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    keyfile = Path(args.keys_dir) / "content.key"
    if not keyfile.exists():
        print("FATAL: no content.key in %s -- refusing to guess a root secret." % args.keys_dir,
              file=sys.stderr)
        return 2
    root_secret = hashlib.blake2b(keyfile.read_bytes().strip(), digest_size=32).digest()
    shared = shared_content_key(root_secret)

    try:
        roots = _legacy_roots(args.db, extra_cts=tuple(args.collection_ct),
                              extra_roots=tuple(args.legacy_root))
    except Exception as e:
        print("FATAL: cannot read origin_roots from %s (%s: %s)" % (args.db, type(e).__name__, e),
              file=sys.stderr)
        return 2
    legacy_keys = {r: collection_key(root_secret, r) for r in sorted(roots)}
    print("cas=%s  objects keyed by: shared + %d legacy origin_root key(s)%s"
          % (args.cas, len(legacy_keys), "  [DRY RUN]" if args.dry_run else ""))

    # ⛔ NO KEYS + OBJECTS PRESENT = A DISCOVERY FAULT, AND IT MUST NOT BE REPORTED AS DATA LOSS.
    # With an empty legacy set every object fails both the shared key and the (empty) legacy set,
    # and the old code classified all of them `unreadable` — "needs a re-fetch from the durable
    # tier", which on a `remote=None` node reads as total loss and invites a destructive recovery.
    # Refusing here is the difference between "I could not look" and "it is gone". (EREA, 487/487.)
    if not legacy_keys and any(True for _ in _walk(args.cas)):
        print(
            "\nFATAL: the CAS holds objects but NO legacy origin_root was discovered, so nothing\n"
            "could be tried but the shared key. REFUSING to classify: with no keys to test, every\n"
            "object would be reported unreadable, which is indistinguishable from real data loss\n"
            "and is almost certainly wrong.\n\n"
            "  * If this store is already fully migrated, there is nothing to do — the objects\n"
            "    open under the shared key and a re-key pass is unnecessary.\n"
            "  * If it holds pre-§1 objects, discovery failed. `origin_root` is read from its own\n"
            "    indexed column; a store predating that column, or one never backfilled, needs\n"
            "    --collection-ct <your collection type> or --legacy-root <root> to name them.\n\n"
            "Nothing was read, written or deleted.", file=sys.stderr)
        return 2

    counts = {"already_shared": 0, "rekeyed": 0, "unreadable": 0, "corrupt": 0, "failed": 0}
    unreadable = []
    for n, (ref, path) in enumerate(_walk(args.cas)):
        if args.limit and n >= args.limit:
            print("... stopped at --limit %d (NOT a full pass; the counts below are partial)"
                  % args.limit)
            break
        aad = ref.encode("utf-8")
        try:
            with open(path, "rb") as fh:
                blob = fh.read()
        except OSError as e:
            counts["failed"] += 1
            print("  ! unreadable file %s (%s)" % (ref, e))
            continue
        if len(blob) < 13:
            counts["corrupt"] += 1        # provably not a ciphertext; still NOT deleted here
            unreadable.append((ref, "too short to hold a nonce"))
            continue
        # 1) already migrated?
        try:
            AESGCM(shared).decrypt(blob[:12], blob[12:], aad)
            counts["already_shared"] += 1
            continue
        except InvalidTag:
            pass
        # 2) openable under some legacy origin_root key? try them ALL — the CAS does not record
        #    which root wrote an object, and after §1's damage the writer may not be the root that
        #    references it. (EREA measured exactly this: the only blobs that failed were the
        #    cross-collection references to shared refs.)
        plain = None
        for k in legacy_keys.values():
            try:
                plain = AESGCM(k).decrypt(blob[:12], blob[12:], aad)
                break
            except InvalidTag:
                continue
        if plain is None:
            counts["unreadable"] += 1
            unreadable.append((ref, "opens under neither the shared key nor any of the %d legacy "
                                    "origin_root key(s)" % len(legacy_keys)))
            continue
        # 3) VERIFY BEFORE REWRITING — see the module docstring.
        if hashlib.sha256(plain).hexdigest() != ref[4:]:
            counts["corrupt"] += 1
            unreadable.append((ref, "decrypts but does not hash to its own address"))
            continue
        if args.dry_run:
            counts["rekeyed"] += 1
            continue
        nonce = os.urandom(12)
        try:
            import tempfile
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(nonce + AESGCM(shared).encrypt(nonce, plain, aad))
                os.replace(tmp, path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            counts["rekeyed"] += 1
        except Exception as e:
            counts["failed"] += 1
            print("  ! rewrite failed for %s (%s: %s) -- object left under its legacy key"
                  % (ref, type(e).__name__, e))

    total = sum(counts.values())
    print("\n%d object(s) examined" % total)
    for k in ("already_shared", "rekeyed", "unreadable", "corrupt", "failed"):
        print("  %-15s %d" % (k, counts[k]))
    if unreadable:
        # ⛔ STATE THE POSSIBILITIES; DO NOT ASSERT LOSS. The earlier wording ("these need a
        # re-fetch from the durable tier") named ONE cause as if it were established. It is not
        # distinguishable from here — a key the run never had produces byte-identical evidence to a
        # genuinely destroyed object, and on a `remote=None` node the confident version reads as
        # total data loss and invites a destructive recovery.
        pct = (100.0 * len(unreadable) / total) if total else 0.0
        print("\nWARNING: %d of %d object(s) (%.0f%%) opened under NO key this run. "
              "NOTHING WAS DELETED." % (len(unreadable), total, pct))
        if pct >= 50.0:
            print("  ⚠ THAT PROPORTION POINTS AT THIS RUN, NOT AT THE DATA. A key you did not have\n"
                  "    fails exactly like a destroyed object. Before treating ANY of these as lost,\n"
                  "    confirm the legacy key set is complete (%d root(s) discovered) — see\n"
                  "    --collection-ct / --legacy-root." % len(legacy_keys))
        print("  Each is one of: (a) written under an origin_root not in the %d discovered, "
              "(b) altered on disk, or (c) destroyed by the pre-§1 defect and in need of "
              "a re-fetch. This run cannot tell them apart." % len(legacy_keys))
        for ref, why in unreadable[:40]:
            print("    %s  -- %s" % (ref, why))
        if len(unreadable) > 40:
            print("    ... and %d more" % (len(unreadable) - 40))
    # A store is fully migrated only when nothing needed a key that is not the shared one.
    if counts["rekeyed"] == 0 and not unreadable and not args.dry_run:
        print("\nFULLY MIGRATED: every object opens under the shared key. The "
              "`legacy_key_for_collection` wiring can be dropped for this store.")
    return 1 if (unreadable or counts["failed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
