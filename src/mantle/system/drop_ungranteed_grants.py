"""Drop the grants that name no grantee.

WHY. A grant reaches the principal it names. One with an empty grantee reaches no one, and Origin
refuses it as malformed (`grantee_id must be a non-empty string`), so it can never be in Origin's
replica either. 2026-11-09, measured when Origin's ledger was seeded: two in production, written by
a direct grant whose request named no grantee; that path now refuses one (400), and the store
refuses to write a grant without a grantee (`lattice_api.GrantWithoutGrantee`).

Idempotent, and no reverse: there is nothing such a grant did that undoing this would restore.

    python -m mantle.system.drop_ungranteed_grants --dry-run
    python -m mantle.system.drop_ungranteed_grants
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("drop_ungranteed_grants")


def ungranteed(db) -> list:
    """The ids of every grant, archived ones included, that names no grantee."""
    from mantle.db.reach_journal import GRANT_CT
    return sorted(str(d.get("id")) for d in db.artifacts.list_artifacts(content_type=GRANT_CT,
                                                                         include_archived=True)
                  if not str(d.get("grantee_id") or "").strip())


def drop_ungranteed_grants(db, *, dry_run: bool = False) -> dict:
    found = ungranteed(db)
    counts = {"found": len(found), "dropped": 0}
    if dry_run:
        return counts
    for gid in found:
        db.artifacts.delete_artifact(gid)
        counts["dropped"] += 1
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    counts = drop_ungranteed_grants(db, dry_run=args.dry_run)
    logger.info("%s: %s", "would drop" if args.dry_run else "ungranteed grants", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
