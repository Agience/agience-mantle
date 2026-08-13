#!/usr/bin/env python3
"""Migrate a local CAS from per-collection at-rest keys to the one shared key (mantle §1).

    python -m mantle.scripts.mantle_cas_rekey --cas <dir> --keys-dir <dir> --db <lattice.db>
    python -m mantle.scripts.mantle_cas_rekey ... --dry-run        # report only, touch nothing

`FileContentCache.get()` rewrites a legacy object under the shared key the moment it is read, so a
node converges on its own with no downtime and no re-fetch. That is the *safe* path and it is
enough for correctness. This script exists for the two things lazy migration cannot give you:

  1. **An answer to "is this store fully migrated?"** -- the lazy path only touches what someone
     happens to read. Cold objects stay legacy indefinitely, and you cannot retire the
     `legacy_key_for_collection` wiring until you know none are left.
  2. **A count of what cannot be opened by either key** — objects that are neither shared-keyed nor
     legacy-keyed under any collection this store knows about. Those are objects whose original
     bytes are unrecoverable under any key, and they need a re-fetch from the durable tier, not a
     re-key. Lazily, each one surfaces as a single confusing read error months apart.

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

#: How many unreadable refs to name individually before summarising the rest. A presentation
#: bound on one terminal report, not a judgement about the data: the count and the percentage
#: above it are always exact and always complete, and nothing branches on this number. Read
#: once, the tail count derives from the same slice that gets printed, so the two always agree.
#: A different value would be right if this report were being machine-read rather than
#: eyeballed, in which case the answer is a file, not a bigger cap.
_LIST_CAP = 40


def _legacy_roots(db_path: str, *, extra_cts=(), extra_roots=()) -> set:
    """The set of `origin_root` values the per-collection key scheme could have keyed on. Read-only.

    It reads `WHERE ct = 'application/vnd.agience.collection+json'` and builds
    `collection_id -> origin_root`. A consumer that names its own collection type gets an empty
    map from this alone — true of every consumer of a general-purpose store, which is the entire
    point of shipping one. That is why `extra_cts` and `extra_roots` exist: without them, a store
    whose collections use a different content type has every object classified `unreadable`,
    indistinguishable from total data loss on a `remote=None` node.

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

    # The wire format is imported, never restated. This script reads and rewrites the very blobs
    # `content_cache` writes, so a second copy of the nonce/tag lengths here is a copy that can
    # silently disagree with the writer.
    from mantle.db.content_cache import (CacheCorrupt, ContentKeyMismatch,
                                                 collection_key, shared_content_key,
                                                 _MIN_BLOB_BYTES, _NONCE_BYTES)
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

    if not legacy_keys and any(True for _ in _walk(args.cas)):
        print(
            "\nFATAL: the CAS holds objects but NO legacy origin_root was discovered, so nothing\n"
            "could be tried but the shared key. REFUSING to classify: with no keys to test, every\n"
            "object would be reported unreadable, which is indistinguishable from real data loss\n"
            "and is almost certainly wrong.\n\n"
            "  * If this store is already fully migrated, there is nothing to do — the objects\n"
            "    open under the shared key and a re-key pass is unnecessary.\n"
            "  * If it holds objects keyed under a legacy origin_root, discovery failed.\n"
            "    `origin_root` is read from its own indexed column; a store with no such column,\n"
            "    or one never backfilled, needs --collection-ct <your collection type> or\n"
            "    --legacy-root <root> to name them.\n\n"
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
        if len(blob) < _MIN_BLOB_BYTES:
            counts["corrupt"] += 1        # provably not a ciphertext; still NOT deleted here
            unreadable.append((ref, "too short to hold a nonce+tag (%d < %d bytes)"
                                    % (len(blob), _MIN_BLOB_BYTES)))
            continue
        # 1) already migrated?
        try:
            AESGCM(shared).decrypt(blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], aad)
            counts["already_shared"] += 1
            continue
        except InvalidTag:
            pass
        # 2) openable under some legacy origin_root key? try them all — the CAS does not record
        #    which root wrote an object, and the writer may not be the root that references it, so
        #    the cross-collection references to shared refs need every key tried.
        plain = None
        for k in legacy_keys.values():
            try:
                plain = AESGCM(k).decrypt(blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], aad)
                break
            except InvalidTag:
                continue
        if plain is None:
            counts["unreadable"] += 1
            unreadable.append((ref, "opens under neither the shared key nor any of the %d legacy "
                                    "origin_root key(s)" % len(legacy_keys)))
            continue
        # 3) verify before rewriting — see the module docstring.
        if hashlib.sha256(plain).hexdigest() != ref[4:]:
            counts["corrupt"] += 1
            unreadable.append((ref, "decrypts but does not hash to its own address"))
            continue
        if args.dry_run:
            counts["rekeyed"] += 1
            continue
        nonce = os.urandom(_NONCE_BYTES)
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
        pct = (100.0 * len(unreadable) / total) if total else 0.0
        print("\nWARNING: %d of %d object(s) (%.0f%%) opened under NO key this run. "
              "NOTHING WAS DELETED." % (len(unreadable), total, pct))
        # No threshold gates this warning. The null this rate has to be read against is computed,
        # not typed: if the discovered key set is complete then every object opens under some key,
        # so the expected unreadable count is exactly zero — `if unreadable:` above already tests
        # against that null, and there is no second, larger rate at which the key set becomes newly
        # suspect. A run missing one of two roots is just as incomplete as one missing both, so the
        # advice below (confirm the key set before calling anything lost) applies at any nonzero
        # rate. `pct` is reported alongside it so the reader has the number and the context together.
        print("  ⚠ THIS MAY POINT AT THIS RUN, NOT AT THE DATA. A key you did not have fails\n"
              "    exactly like a destroyed object, and that is true of the first object as much\n"
              "    as the thousandth. Before treating ANY of these as lost, confirm the legacy\n"
              "    key set is complete (%d root(s) discovered) — see --collection-ct /\n"
              "    --legacy-root." % len(legacy_keys))
        print("  Each is one of: (a) written under an origin_root not in the %d discovered, "
              "(b) altered on disk, or (c) unrecoverable under any known key and in need of "
              "a re-fetch. This run cannot tell them apart." % len(legacy_keys))
        shown = unreadable[:_LIST_CAP]
        for ref, why in shown:
            print("    %s  -- %s" % (ref, why))
        if len(unreadable) > len(shown):
            print("    ... and %d more" % (len(unreadable) - len(shown)))
    # A store is fully migrated only when nothing needed a key that is not the shared one.
    if counts["rekeyed"] == 0 and not unreadable and not args.dry_run:
        print("\nFULLY MIGRATED: every object opens under the shared key. The "
              "`legacy_key_for_collection` wiring can be dropped for this store.")
    return 1 if (unreadable or counts["failed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
