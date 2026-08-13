#!/usr/bin/env python3
"""
manage_seed.py -- Platform inbox-seed collection management tool.

Run from the mantle/ directory so that imports resolve correctly.

Actions
-------
seed         Ensure the platform "inbox seeds" collection exists (created empty if missing).
             Called automatically at every backend startup -- idempotent.

grant-write  Give a user write access to the inbox-seed collection so they can
             author / edit seed content via the normal UI.  Idempotent.

revoke       Remove a user's grant on the inbox-seed collection.

migrate      Back-fill existing users: issue read grants to the inbox-seed collection
             and grant access to other platform-owned collections.
             Idempotent -- safe to re-run.

Options
-------
--user        User ID (required for grant-write and revoke).
              Find your user_id from the JWT sub claim or the people table.
--dry-run     Print what would happen without making changes.

Workflow (fresh install)
------------------------
  1. Start backend -- seed collection is created automatically (empty).
  2. Grant write access to the admin user:
       python manage_seed.py --action grant-write --user <your-user-id>
  3. Open the app, find "Agience Inbox Seeds" in your collections, and add artifacts.
    4. New user sign-ups will automatically receive the platform collection grants.
  5. For existing users, run migrate:
       python manage_seed.py --action migrate
  6. Revoke admin write access when done:
       python manage_seed.py --action revoke --user <your-user-id>

Examples
--------
  python manage_seed.py --action seed
  python manage_seed.py --action grant-write --user <your-user-id>
  python manage_seed.py --action revoke --user <your-user-id>
  python manage_seed.py --action migrate
  python manage_seed.py --action migrate --dry-run
"""

import argparse
import logging

from mantle.db.store import Database
from mantle import config
from mantle.config import AGIENCE_PLATFORM_USER_ID
import os
from pathlib import Path

from mantle.services.seed_provisioning import provision_user, seed_from_artifacts
from mantle.services.bootstrap_types import (
    INBOX_SEEDS_COLLECTION_SLUG,
    START_HERE_COLLECTION_SLUG,
    PLATFORM_ARTIFACTS_COLLECTION_SLUG,
    ALL_SERVERS_COLLECTION_SLUG,
    ALL_TOOLS_COLLECTION_SLUG,
)
from mantle.services.platform_topology import get_id, pre_resolve_platform_ids
from mantle.services.platform_settings_service import settings as platform_settings


def _platform_seeds_root() -> Path:
    base = os.getenv("AGIENCE_SEEDS_ROOT")
    if base:
        return Path(base) / "platform"
    return Path(__file__).resolve().parents[2] / "package" / "seeds" / "platform"

_ADMIN_WRITE_SLUGS = [
    INBOX_SEEDS_COLLECTION_SLUG,
    START_HERE_COLLECTION_SLUG,
    PLATFORM_ARTIFACTS_COLLECTION_SLUG,
    ALL_SERVERS_COLLECTION_SLUG,
    ALL_TOOLS_COLLECTION_SLUG,
]

_SLUG_NAMES = {
    INBOX_SEEDS_COLLECTION_SLUG: "Agience Inbox Seeds",
    START_HERE_COLLECTION_SLUG: "Start Here",
    PLATFORM_ARTIFACTS_COLLECTION_SLUG: "Platform Artifacts",
    ALL_SERVERS_COLLECTION_SLUG: "All Servers",
    ALL_TOOLS_COLLECTION_SLUG: "All Tools",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("manage_seed")


# â"â" DB connections â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"

def connect() -> Database:
    """Open THE store — the standalone lattice (MANTLE_LATTICE_PATH / MANTLE_ORIGIN env)."""
    from mantle.db import backend
    logger.info("Opening the lattice store...")
    db = backend.store_handle()
    logger.info("Open.")
    return db


# â"â" Actions â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"â"

def action_seed(db: Database, dry_run: bool) -> None:
    """Apply the declarative platform seed tree (package/seeds/platform). Idempotent."""
    if dry_run:
        logger.info("[DRY-RUN] Would apply %s via the loader.", _platform_seeds_root())
        return
    report = seed_from_artifacts(db, _platform_seeds_root())
    logger.info("Platform seed: %s", report.summary())
    for err in report.errors:
        logger.warning("  %s", err)


def action_populate(db: Database, dry_run: bool) -> None:
    """Alias for `seed` — the declarative loader applies the full platform tree."""
    action_seed(db, dry_run)


def action_grant_write(db: Database, user_id: str, dry_run: bool) -> None:
    """Grant admin write access to all inbox-seed collections (parent + all sub-collections)."""
    from mantle.db.backend import upsert_user_collection_grant

    if dry_run:
        for slug in _ADMIN_WRITE_SLUGS:
            logger.info("[DRY-RUN] Would grant user %s write access to %s", user_id, _SLUG_NAMES.get(slug, slug))
        return

    for slug in _ADMIN_WRITE_SLUGS:
        col_id = get_id(slug)
        if not col_id:
            logger.warning("Skipping grant for unresolved slug: %s", slug)
            continue
        grant, changed = upsert_user_collection_grant(
            db,
            user_id=user_id,
            collection_id=col_id,
            granted_by=AGIENCE_PLATFORM_USER_ID,
            can_read=True,
            can_update=True,
            name=f"Admin write grant -- {_SLUG_NAMES.get(slug, slug)}",
        )
        if changed:
            logger.info("Granted write access: user=%s  collection=%s (%s)  grant_id=%s", user_id, col_id, _SLUG_NAMES.get(slug, slug), grant.id)


def action_revoke(db: Database, user_id: str, dry_run: bool) -> None:
    """Revoke all active grants for a user on the inbox-seed collection."""
    from mantle.db.backend import get_active_grants_for_principal_resource, update_grant
    from mantle.services import grant_key_service
    from datetime import datetime, timezone

    col_id = get_id(INBOX_SEEDS_COLLECTION_SLUG)
    grants = get_active_grants_for_principal_resource(
        db, grantee_id=user_id, resource_id=col_id
    )
    if not grants:
        logger.info("No active grant for user=%s  collection=%s -- nothing to revoke.", user_id, col_id)
        return

    for g in grants:
        if dry_run:
            logger.info("[DRY-RUN] Would revoke grant %s (user=%s  collection=%s)", g.id, user_id, col_id)
        else:
            now = datetime.now(timezone.utc).isoformat()
            g.state = "revoked"
            g.modified_time = now
            update_grant(db, g)
            # The ledger now says `revoked`; the light-cone memo does not. Until it is
            # dropped, every key derivation and cell decryption that consults it keeps
            # answering with the pre-revocation verdict — the direction that matters.
            # AFTER the write, so a concurrent request cannot refill the entry from the
            # pre-change ledger. `invalidate_for` decides which principal id to name
            # (a bearer key's memo is keyed on the root grant, not `grantee_id`).
            #
            # Best-effort, as on every other revocation path: a stale memo is a delay
            # bounded by its TTL, and failing the revoke over it would leave the ledger
            # edit unreported.
            try:
                grant_key_service.invalidate_for(db, g)
            except Exception:
                logger.warning("grant-cache invalidation failed for grant %s", g.id,
                               exc_info=True)
            logger.info("Revoked grant %s (user=%s  collection=%s)", g.id, user_id, col_id)


def action_migrate(store_db: Database, dry_run: bool) -> None:
    """
    Back-fill all existing users:
    1. Issue a read grant to the inbox-seed collection (idempotent).
    2. Grant access to other platform-owned collections (for example the current host collection).
    """
    from mantle.db.identity_backend import list_all_people

    people = list_all_people(store_db)
    logger.info("Found %d users to migrate.", len(people))
    success = errors = 0

    for person in people:
        person_id = person.get("id") or person.get("_key")
        label = person.get("email", "") or person_id
        if dry_run:
            logger.info("  [DRY-RUN] %s -- would provision user.", label)
            continue
        try:
            provision_user(store_db, person_id)
            success += 1
        except Exception as exc:
            logger.warning("  Migration failed for %s: %s", label, exc)
            errors += 1

    logger.info("Migration complete: %d succeeded, %d errors.", success, errors)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Manage Agience platform inbox-seed collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--action",
        choices=["seed", "populate", "grant-write", "revoke", "migrate"],
        required=True,
        help="What to do.",
    )
    parser.add_argument(
        "--user",
        help="User ID (required for grant-write and revoke).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing anything.",
    )
    args = parser.parse_args()

    if args.action in ("grant-write", "revoke") and not args.user:
        parser.error(f"--user is required for --action {args.action}")

    db = connect()
    platform_settings.load_all(db)
    pre_resolve_platform_ids(db)

    # CLI: no request context, so declare the identity explicitly. Every action
    # below writes or reads artifacts, and artifact content encryption now requires
    # an acting principal the grant ledger can check.
    from mantle.services.system_identity import system_acting_context

    with system_acting_context(scope="platform.manage-seed"):
        if args.action == "seed":
            action_seed(db, args.dry_run)
        elif args.action == "populate":
            action_populate(db, args.dry_run)
        elif args.action == "grant-write":
            action_grant_write(db, args.user, args.dry_run)
        elif args.action == "revoke":
            action_revoke(db, args.user, args.dry_run)
        elif args.action == "migrate":
            action_migrate(db, args.dry_run)


if __name__ == "__main__":
    main()

