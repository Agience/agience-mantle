"""Make the old People directory the platform's sign-up collection.

WHY. A person's card is their own (John, 2026-11-01), and the platform's user list is Origin's
own records (2026-11-02). The People directory provisioning wrote before then holds one card per
member, in a collection the platform owns, with an owner grant each member issued themselves.
John, 2026-11-02: it becomes the platform's sign-up collection. This pass renames it and, on each
member's old card, revokes the member's own grants, every grant the member issued on it to
someone else, and its invites, so the platform owns those cards and no one holds anything on
them through the member. The member is named by the card's id, which provisioning derived from
theirs (`person_artifact_id`), never by its editable context. Each member gets their own card at their next sign-in
(`user_provisioning._ensure_person_artifact`), at an id of its own; nothing adds a card here.

A revocation is a narrowing, so Origin takes it from the platform's system principal, which this
pass acts as. Nothing is deleted: the cards, their edges and the revoked grants all stay, and the
platform's own grants on the collection (the operator's, the system principal's) are untouched.

Resumable and idempotent: a revoked grant is skipped and a renamed collection is left as it is,
so an interrupted run is finished by running it again and a finished run does nothing.

    python -m mantle.system.people_to_signups --dry-run
    python -m mantle.system.people_to_signups
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("people_to_signups")

NAME = "Sign-ups"
DESCRIPTION = ("The platform's record of who signed up before 2026-11-02. It belongs to the "
               "platform; members hold nothing here, and each has a card of their own.")


def people_collection_id(db) -> Optional[str]:
    """The People directory's id: the persisted slug, else its derivation, if it exists."""
    from mantle.db import lattice_api as la
    from mantle.services.bootstrap_types import PEOPLE_COLLECTION_SLUG
    from mantle.services.platform_topology import get_id_optional
    from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace

    pid = get_id_optional(PEOPLE_COLLECTION_SLUG) or derive_uuid(
        get_instance_namespace(), "agience", PEOPLE_COLLECTION_SLUG)
    return pid if la.get_collection_by_id(db, pid) is not None else None


def _member_of(db, card) -> Optional[str]:
    """Whose old card this is, from its id: provisioning derived it from the member's id
    (`person_artifact_id`). The card's context and `created_by` are only candidates, checked
    against that derivation, as is every grantee and issuer of a grant on it: an edited context
    names no one."""
    from mantle.db import lattice_api as la
    from mantle.services.seed_provisioning.user_provisioning import person_artifact_id

    candidates = {str(getattr(card, "created_by", "") or "")}
    raw = getattr(card, "context", "") or ""
    try:
        ctx = json.loads(raw) if isinstance(raw, str) and raw else (raw or {})
        ident = ctx.get("identity") if isinstance(ctx, dict) else None
        if isinstance(ident, dict) and ident.get("agience_root_id"):
            candidates.add(str(ident["agience_root_id"]))
    except (TypeError, ValueError):
        pass
    for g in la.get_grants_for_collection(db, card.id):
        candidates.update({str(g.grantee_id or ""), str(getattr(g, "granted_by", "") or "")})
    for c in sorted(x for x in candidates if x):
        if person_artifact_id(c) == card.id:
            return c
    return None


def _cards(db, people_id: str) -> list:
    from mantle.db import lattice_api as la

    out = []
    for row in la.list_collection_artifacts(db, people_id) or []:
        rid = str((row.get("root_id") or row.get("id") or "") if isinstance(row, dict) else "")
        card = la.get_artifact(db, rid) if rid else None
        if card is not None and getattr(card, "collection_id", None) == people_id:
            out.append(card)
    return out


def _member_grants(db, card, member: str) -> list:
    """Every grant the member holds on their old card, every grant they issued on it to anyone
    else, and every invite on it: what ends with their hold on the card."""
    from mantle.db import lattice_api as la

    return [g for g in la.get_grants_for_collection(db, card.id)
            if (getattr(g, "grantee_type", "") == "user" and g.grantee_id == member)
            or getattr(g, "granted_by", None) == member
            or getattr(g, "grantee_type", "") == "invite"]


def people_to_signups(db, *, people_id: Optional[str] = None, dry_run: bool = False) -> dict:
    """Rename the People directory; on each old card revoke the member's grants, the grants they
    issued on it, and its invites. Returns the counts; `dry_run` counts what it would do and
    writes nothing."""
    from mantle.db import lattice_api as la
    from mantle.entities.grant import Grant
    from mantle.services.system_identity import system_acting_context

    people_id = people_id or people_collection_id(db)
    counts = {"collection": people_id, "renamed": 0, "cards": 0, "revoked": 0,
              "already_revoked": 0}
    if not people_id:
        return counts
    with system_acting_context(scope="platform.people-to-signups"):
        collection = la.get_collection_by_id(db, people_id)
        if collection is not None and collection.name != NAME:
            counts["renamed"] = 1
            if not dry_run:
                collection.name, collection.description = NAME, DESCRIPTION
                collection.modified_time = datetime.now(timezone.utc).isoformat()
                la.update_collection(db, collection)
        for card in _cards(db, people_id):
            member = _member_of(db, card)
            if not member:
                logger.warning("  %s names no member by its id; left alone", card.id)
                continue
            counts["cards"] += 1
            for g in _member_grants(db, card, member):
                if g.state == Grant.STATE_REVOKED or g.state == "deleted":
                    counts["already_revoked"] += 1
                    continue
                counts["revoked"] += 1
                if dry_run:
                    continue
                g.state = Grant.STATE_REVOKED
                g.revoked_by = "platform.people-to-signups"
                g.revoked_at = datetime.now(timezone.utc).isoformat()
                la.update_grant(db, g)
            if not dry_run:
                try:
                    from mantle.search.mantle.wiring import invalidate_grant_cache
                    invalidate_grant_cache(member)
                except Exception:  # noqa: BLE001 — the memo lapses within its TTL
                    logger.debug("grant cache not invalidated for %s", member, exc_info=True)
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from mantle.services.authority_ledger import install_for_cli
    install_for_cli()
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    counts = people_to_signups(db, dry_run=args.dry_run)
    logger.info("%s: %s", "would change" if args.dry_run else "changed", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
