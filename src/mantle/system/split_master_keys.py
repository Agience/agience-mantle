"""Split every whole master key into threshold shares, or combine them back.

WHY. Under ``MANTLE_MASTER_KEY_CUSTODY=threshold`` each principal's master key is held as 2-of-3
Shamir shares: share 1 in this node's lattice under the KEK, share 2 at Origin, share 3 sealed to
the operator's offline age recipient in the durable object store
(`search/mantle/threshold_custody.py`). Keys minted before that are stored whole, wrapped by the
KEK (`oracle.LatticeMasterKeyStore`). This pass splits each one. The key itself does not change,
so no content, cell or index entry is re-encrypted.

For each ``master-key:<principal>`` row stored whole:

  1. unwrap it, split it, and check that every pair of shares recombines to it;
  2. (after all keys are split) seal every escrow share into ONE export file for the operator;
  3. under the custody lock, re-read the row and go on only if its stored form is the one read in
     step 1; write the sealed escrow share to the durable store; store share 2 at Origin, fetch it
     back, and check that it recombines with share 1 to the key;
  4. only then replace the row with the share form, and read the key back through the threshold
     store; a key that does not read back is restored to its whole form and reported.

The custody lock (`threshold_custody.CustodyLock`, a file beside the lattice) is the one the server
takes around every key-row write, so this pass and a live server cannot interleave on a row.
Idempotent: a row already in share form is left as it is, and a re-run after a partial one splits
only what is still whole. A key whose Origin share is already stored (a re-split after a rollback)
replaces it, since Origin accepts a share of the same key.

`--combine` is the reverse, for a rollback to a release that reads only whole keys: each split key
is re-read under the lock, combined (shares 1 and 2, or 1 and 3 in recovery mode) and written back
whole under the KEK, then read back. A ``pending`` row (a key minted under threshold custody whose
write never completed, so never served) is dropped, if it is still the row first read. Origin's
shares are left in place, and a later split replaces them; escrow objects are never replaced, so
a later split writes its own beside them.

`--export-escrow PATH` copies every split key's sealed escrow share from the durable store into
one file, the operator's offline copy. Each entry opens with the escrow identity.

    python -m mantle.system.split_master_keys --dry-run
    python -m mantle.system.split_master_keys --escrow-out /path/master-key-escrow.age
    python -m mantle.system.split_master_keys --combine
    python -m mantle.system.split_master_keys --export-escrow /path/master-key-escrow-sealed.json

The escrow recipient is read from ``MANTLE_ESCROW_RECIPIENT_FILE`` (or ``--recipient-file``): a
public key. This pass never reads the identity, except as the recovery mode of the threshold store
when ``--combine`` runs with ``MANTLE_MASTER_KEY_RECOVERY=escrow``.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import json
import logging
import os
import sys
from datetime import datetime, timezone

logger = logging.getLogger("split_master_keys")

_PREFIX = "master-key:"


def _principal(doc: dict) -> str:
    return str(doc.get("id", ""))[len(_PREFIX):]


def _rows(db, content_type: str) -> list:
    return [d for d in db.artifacts.list_artifacts(content_type=content_type)
            if isinstance(d, dict) and str(d.get("id", "")).startswith(_PREFIX)]


def _unchanged(db, doc: dict) -> bool:
    from mantle.search.mantle.threshold_custody import stored_form
    return stored_form(db.artifacts.get_artifact(doc["id"])) == stored_form(doc)


def split_master_keys(db, kek, custodian, recipient: str, escrow_store, lock, *,
                      dry_run: bool = False, write_escrow=None) -> dict:
    """Split every whole key. `write_escrow(entries)` receives the escrow shares, in the clear,
    before any row is replaced; it must seal and persist them. Returns the counts."""
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore
    from mantle.search.mantle.threshold_custody import (
        MANTLE_X, ThresholdMasterKeyStore, _unframe_share, escrow_name, split_key,
    )
    from prism.trust import shamir

    whole_rows = [d for d in _rows(db, LatticeMasterKeyStore.CONTENT_TYPE) if d.get("token")]
    counts = {"whole": len(whole_rows), "already_split": len(_rows(db, SHARE_CONTENT_TYPE)),
              "split": 0, "changed": 0, "failed": 0}
    if dry_run or not whole_rows:
        return counts

    whole = LatticeMasterKeyStore(kek, lambda: db)
    threshold = ThresholdMasterKeyStore(kek, lambda: db, custodian, recipient, escrow_store, lock)

    planned = []
    for doc in whole_rows:
        pid = _principal(doc)
        try:
            key = whole.get(pid)
            row, origin_share, escrow, sealed = split_key(pid, key, kek, recipient)
            x, y = _unframe_share(pid, kek.unwrap(row["share"]))
            if x != MANTLE_X or shamir.combine(
                    [(x, y), (escrow["x"], base64.b64decode(escrow["y"]))]) != key:
                raise RuntimeError("the wrapped share does not recombine with the escrow share")
        except Exception as exc:  # noqa: BLE001 — per key; the next key
            counts["failed"] += 1
            logger.error("  cannot split %s: %s", pid, exc)
            continue
        planned.append((doc, pid, key, row, origin_share, escrow, sealed))

    if planned and write_escrow is not None:
        write_escrow([p[5] for p in planned])

    for doc, pid, key, row, origin_share, _escrow, sealed in planned:
        with lock():
            if not _unchanged(db, doc):
                counts["changed"] += 1
                logger.warning("  %s changed since it was read; left for a re-run", pid)
                continue
            try:
                escrow_store.put_if_absent(escrow_name(row), sealed)
                custodian.put_share(pid, origin_share)
                theirs = custodian.get_share(pid)
                mine = _unframe_share(pid, kek.unwrap(row["share"]))[1]
                if theirs is None or not hmac.compare_digest(
                        shamir.combine([(MANTLE_X, mine),
                                        (int(theirs["x"]), base64.b64decode(theirs["y"]))]), key):
                    raise RuntimeError("Origin's stored share does not recombine to the key")
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                logger.error("  cannot store the other shares of %s: %s (left whole)", pid, exc)
                continue
            db.artifacts.put_artifact(row)
            try:
                ok = threshold.get(pid) == key
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                db.artifacts.put_artifact(doc)
                counts["failed"] += 1
                logger.error("  %s did not read back through the threshold store; restored whole",
                             pid)
                continue
        counts["split"] += 1
    return counts


def combine_master_keys(db, kek, threshold_store, lock, *, dry_run: bool = False) -> dict:
    """Write every split key back whole under the KEK. Returns the counts."""
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore

    rows = _rows(db, SHARE_CONTENT_TYPE)
    pending = [d for d in rows if d.get("state") == "pending"]
    counts = {"split": len(rows) - len(pending), "pending": len(pending), "combined": 0,
              "pending_dropped": 0, "changed": 0, "failed": 0}
    if dry_run:
        return counts
    whole = LatticeMasterKeyStore(kek, lambda: db)
    for doc in rows:
        pid = _principal(doc)
        with lock():
            if not _unchanged(db, doc):
                counts["changed"] += 1
                logger.warning("  %s changed since it was read; left for a re-run", pid)
                continue
            if doc.get("state") == "pending":
                db.artifacts.delete_artifact(doc["id"])
                counts["pending_dropped"] += 1
                continue
            try:
                key = threshold_store.get(pid)
                if key is None:
                    raise RuntimeError("no key combined")
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                logger.error("  cannot combine %s: %s (left split)", pid, exc)
                continue
            whole.put(pid, key)
            try:
                ok = whole.get(pid) == key
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                db.artifacts.put_artifact(doc)
                counts["failed"] += 1
                logger.error("  %s did not read back whole; restored its shares", pid)
                continue
        counts["combined"] += 1
    return counts


def sealed_escrow_rows(db, escrow_store) -> list:
    """Every active split key's sealed escrow share, as `{principal_id, kcv, escrow}` (escrow is
    the base64 of an age file, or None when the durable store holds none for it)."""
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE
    from mantle.search.mantle.threshold_custody import escrow_name

    out = []
    for d in _rows(db, SHARE_CONTENT_TYPE):
        if d.get("state") == "pending":
            continue
        sealed = escrow_store.get(escrow_name(d))
        out.append({"principal_id": _principal(d), "kcv": d.get("kcv"),
                    "escrow": base64.b64encode(sealed).decode() if sealed else None})
    return out


def escrow_writer(path: str, recipient: str):
    """A `write_escrow` that seals all entries into one age file at `path`, refusing to replace a
    file that exists (an earlier export is the only copy of its shares)."""
    from mantle.search.mantle.threshold_custody import seal

    def write(entries: list) -> None:
        doc = {"format": "agience-mantle-escrow-v1",
               "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "shares": entries}
        sealed = seal(json.dumps(doc, sort_keys=True).encode(), recipient)
        with open(path, "xb") as fh:
            fh.write(sealed)
        logger.info("escrow shares for %d key(s) sealed to %s", len(entries), path)

    return write


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--combine", action="store_true",
                    help="the reverse: write split keys back whole under the KEK, before a rollback")
    ap.add_argument("--escrow-out", help="the sealed export file for this run's escrow shares")
    ap.add_argument("--export-escrow", help="copy every sealed escrow share into this file")
    ap.add_argument("--recipient-file", default=os.getenv("MANTLE_ESCROW_RECIPIENT_FILE", ""),
                    help="the escrow age recipient (public key); default MANTLE_ESCROW_RECIPIENT_FILE")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    # Arguments are settled before any key is loaded, so a mistyped run touches nothing.
    from mantle.search.mantle import threshold_custody as tc
    recipient = None
    if not (args.combine or args.export_escrow):
        if not args.recipient_file:
            ap.error("the escrow recipient is required: MANTLE_ESCROW_RECIPIENT_FILE or "
                     "--recipient-file")
        recipient = tc.recipient_from_file(args.recipient_file)
        if not args.dry_run and not args.escrow_out:
            ap.error("--escrow-out is required: it is the one export of this run's escrow shares")
        if args.escrow_out and os.path.exists(args.escrow_out):
            ap.error(f"{args.escrow_out} exists; an earlier export is the only copy of its shares")
    if args.export_escrow and os.path.exists(args.export_escrow):
        ap.error(f"{args.export_escrow} exists")

    from prism.trust.key_manager import init_encryption_key, init_nonce_secret
    init_encryption_key()
    init_nonce_secret()
    from mantle.services import peer_signing
    peer_signing.init()
    from mantle.search.mantle.key_provider import build_key_provider
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    kek = build_key_provider()
    lock = tc.CustodyLock(tc.default_lock_path())

    def escrow_store():
        try:
            return tc.durable_escrow_store()
        except ValueError as exc:
            ap.error(str(exc))

    if args.export_escrow:
        entries = sealed_escrow_rows(db, escrow_store())
        with open(args.export_escrow, "x", encoding="utf-8") as fh:
            json.dump({"format": "agience-mantle-escrow-sealed-v1", "entries": entries}, fh)
        missing = sum(1 for e in entries if not e["escrow"])
        logger.info("sealed escrow shares of %d key(s) written to %s (%d with none stored)",
                    len(entries) - missing, args.export_escrow, missing)
        return 1 if missing else 0

    if args.combine:
        if not os.getenv("MANTLE_ESCROW_RECIPIENT_FILE") and args.recipient_file:
            os.environ["MANTLE_ESCROW_RECIPIENT_FILE"] = args.recipient_file
        counts = combine_master_keys(db, kek, tc.build_threshold_store(kek, lambda: db), lock,
                                     dry_run=args.dry_run)
        logger.info("%s: %s", "would combine" if args.dry_run else "combined", counts)
        return 1 if counts["failed"] or counts["changed"] else 0

    from mantle.clients.origin_key_share_client import OriginShareCustodian
    counts = split_master_keys(
        db, kek, None if args.dry_run else OriginShareCustodian(), recipient,
        None if args.dry_run else escrow_store(), lock, dry_run=args.dry_run,
        write_escrow=escrow_writer(args.escrow_out, recipient) if args.escrow_out else None)
    logger.info("%s: %s", "would split" if args.dry_run else "split", counts)
    return 1 if counts["failed"] or counts["changed"] else 0


if __name__ == "__main__":
    sys.exit(main())
