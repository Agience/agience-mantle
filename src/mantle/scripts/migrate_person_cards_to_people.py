"""One-shot migration: move existing Person cards out of user Inboxes into the
platform **People** collection (and ensure each owner can edit their own card).

Why: Person profile artifacts used to be created inside the user's Inbox. They
now live in the runtime-created **People** directory (admin-managed; each member
reaches only their own card via an owner grant). New cards are created there and
existing ones self-heal on the owner's next login — this script applies that move
proactively across the whole instance so a prod deployment doesn't wait for logins.

Idempotent: a card already in People (with its owner grant) is skipped.

Usage (from repo root, with Mantle's env in scope so it hits the same Arango):

    python src/mantle/scripts/migrate_person_cards_to_people.py            # apply
    python src/mantle/scripts/migrate_person_cards_to_people.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("migrate_person_cards_to_people")

_MANTLE_DIR = Path(__file__).resolve().parent.parent   # <repo>/src/mantle
_SRC_DIR = _MANTLE_DIR.parent                           # <repo>/src (for `platform`)
for _p in (str(_MANTLE_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("MANTLE_SKIP_LIFESPAN_INIT", "1")


def _parse_context(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Move Person cards into the People collection")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args()

    from db.arango import (
        COLLECTION_ARTIFACTS,
        add_artifact_to_collection,
        get_artifact,
        get_edge,
        query_documents,
        remove_artifact_from_collection,
        update_artifact,
    )
    from entities.artifact import Artifact as ArtifactEntity
    from services.dependencies import get_arango_db
    from services.platform_settings_service import settings
    from services.seed_provisioning import user_provisioning as up

    db_gen = get_arango_db()
    db = next(db_gen)
    try:
        settings.load_all(db)  # operator-grant resolution inside _ensure_people_collection

        people_id = up._ensure_people_collection(db) if not args.dry_run else \
            up.derive_uuid(up.get_instance_namespace(), "agience", up.PEOPLE_COLLECTION_SLUG)
        log.info("People collection id: %s", people_id)

        persons = list(query_documents(
            db, ArtifactEntity, COLLECTION_ARTIFACTS,
            {"content_type": up.PERSON_CONTENT_TYPE},
        ))
        log.info("Found %d Person card(s)", len(persons))

        moved = skipped = 0
        for person in persons:
            pid = str(getattr(person, "id", "") or "")
            if not pid:
                continue
            already = get_edge(db, people_id, pid) is not None
            old_col = str(getattr(person, "collection_id", "") or "")
            needs_move = not already or (old_col and old_col != people_id)
            if not needs_move:
                skipped += 1
                continue
            ctxd = _parse_context(getattr(person, "context", ""))
            user_id = str((ctxd.get("identity") or {}).get("agience_root_id") or getattr(person, "created_by", "") or "")
            if args.dry_run:
                log.info("WOULD move %s (owner=%s) from %s -> People", pid, user_id, old_col or "(none)")
                moved += 1
                continue
            if not already:
                add_artifact_to_collection(db, people_id, pid, origin=True)
            if old_col and old_col != people_id:
                remove_artifact_from_collection(db, old_col, pid)
                entity = get_artifact(db, pid)
                if entity is not None and getattr(entity, "collection_id", None) != people_id:
                    entity.collection_id = people_id
                    update_artifact(db, entity)
            if user_id:
                up._ensure_person_owner_grant(db, pid, user_id)
            log.info("Moved %s (owner=%s) -> People", pid, user_id)
            moved += 1

        log.info("Done. moved=%d skipped=%d total=%d%s",
                 moved, skipped, len(persons), " (dry-run)" if args.dry_run else "")
        return 0
    finally:
        try:
            next(db_gen)
        except StopIteration:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
