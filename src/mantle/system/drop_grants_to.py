"""Drop every grant to one account (archived ones included).

For an account the operator has deactivated and means to retire: the release tool's
`platform_release person drop-grants <who>` runs this only after Origin shows the account
deactivated. 2026-11-10: the throwaway accounts a night of release smoke tests made held 49 grants
that never reached Origin's replica (they were made before live co-signing worked); John approved
dropping them.

A plain store delete, as `drop_ungranteed_grants` is: these grants are not in Origin's replica, so
there is nothing there to withdraw. Idempotent: a second run finds nothing.

    python -m mantle.system.drop_grants_to --grantee <account id> --dry-run
    python -m mantle.system.drop_grants_to --grantee <account id>
"""
from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("drop_grants_to")


def grants_to(db, grantee: str) -> list:
    """The ids of every grant to `grantee`, archived ones included."""
    from mantle.db.reach_journal import GRANT_CT
    return sorted(str(d.get("id")) for d in db.artifacts.list_artifacts(content_type=GRANT_CT,
                                                                         include_archived=True)
                  if str(d.get("grantee_id") or "") == grantee)


def drop_grants_to(db, grantee: str, *, dry_run: bool = False) -> dict:
    if not grantee.strip():
        raise ValueError("name the account whose grants to drop")
    found = grants_to(db, grantee)
    counts = {"grantee": grantee, "found": len(found), "dropped": 0}
    if dry_run:
        return counts
    for gid in found:
        db.artifacts.delete_artifact(gid)
        counts["dropped"] += 1
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grantee", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from mantle.services.dependencies import get_store_db

    counts = drop_grants_to(next(get_store_db()), args.grantee, dry_run=args.dry_run)
    print(f"grants to {counts['grantee']}: {counts['found']} found, {counts['dropped']} dropped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
