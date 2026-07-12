"""Per-user first-login provisioning — the single declarative path.

On user create / first login the platform:
  1. ensures the user has an "Inbox" workspace (owner-grant side effects live in
     `workspace_service.create_workspace`),
  2. applies the declarative ``package/seeds/user`` grant artifacts with the
     user's context (``{{user.id}}``), and — when the user is the designated
     platform admin (``platform.operator_id``) — also the ``package/seeds/admin``
     grant set, and
  3. materializes the curated platform seed artifacts into that workspace,
     preserving any existing ordering.

Grants are uniform: same loader, same grant format. The only special thing is
that the one designated admin user receives the (fuller) admin grant set. Steps
1 and 3 loop over live DB state (not static data), so they stay as thin runtime
glue here; step 2 is pure declarative seeding via the loader.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from arango.database import StandardDatabase

from db.arango import (
    add_artifact_to_collection as db_add_artifact_to_collection,
    create_artifact as db_create_artifact,
    create_collection as db_create_collection,
    create_grant as db_create_grant,
    get_active_grants_for_principal_resource as db_get_grants_for_principal_resource,
    get_artifact as db_get_artifact,
    get_collection_by_id as db_get_collection_by_id,
    get_edge as db_get_edge,
    list_collection_artifacts as db_list_collection_artifacts,
    remove_artifact_from_collection as db_remove_artifact_from_collection,
    update_artifact as db_update_artifact,
    upsert_user_collection_grant as db_upsert_user_collection_grant,
)
from entities.artifact import Artifact as ArtifactEntity
from entities.collection import Collection as CollectionEntity, COLLECTION_CONTENT_TYPE
from entities.grant import Grant as GrantEntity
from agience_core.config import AGIENCE_PLATFORM_USER_ID
from services.bootstrap_types import INBOX_MATERIALIZATION_SLUGS, PEOPLE_COLLECTION_SLUG
from services.platform_topology import get_id_optional, register_id
from .loader import (
    UserContext,
    derive_uuid,
    get_instance_namespace,
    seed_from_artifacts,
    _persist_seed_ids,
)

logger = logging.getLogger(__name__)

PERSON_CONTENT_TYPE = "application/vnd.agience.person+json"
# Leads collection slug whose members are linked to a user when they sign up
# (email match). Deployment-configurable; defaults to the website contact slug.
_LEADS_COLLECTION_SLUG = os.getenv("LEADS_COLLECTION_SLUG", "contact")


def _parse_json_obj(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            val = json.loads(raw)
            return val if isinstance(val, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def person_artifact_id(user_id: str) -> str:
    """Deterministic id for a user's Person artifact. NOT == user_id: the
    operator's personal collection already uses id == user_id, so the Person
    artifact carries its identity via context.identity.agience_root_id instead."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agience://person/{user_id}"))


def _seeds_base() -> Path:
    env = os.getenv("AGIENCE_SEEDS_ROOT")
    if env:
        return Path(env)
    # BASE_DIR is /app in Docker and the repo root in local dev, so the
    # seed tree resolves correctly in both without an env override.
    from agience_core import config
    return config.BASE_DIR / "package" / "seeds"


def _is_platform_admin(arango_db: StandardDatabase, user_id: str) -> bool:
    """The designated platform admin is the user in ``platform.operator_id``."""
    from services.platform_settings_service import settings

    return bool(user_id) and settings.get("platform.operator_id") == user_id


def provision_user(
    arango_db: StandardDatabase,
    user_id: str,
    *,
    email: Optional[str] = None,
    name: Optional[str] = None,
    tenant: Optional[str] = None,
    seeds_base: Optional[Path] = None,
) -> None:
    """Provision a user on first login: ensure Inbox workspace, apply the user
    grant seeds (plus the admin grant set if this is the designated admin), and
    materialize curated seed artifacts. Idempotent.

    ``tenant`` (the external IdP / issuer the user authenticated through) is
    recorded on the Person card so users are traceable to their tenant in a
    multi-tenant deployment. ``None`` for platform/Origin users.
    """
    if not user_id:
        logger.warning("provision_user called with empty user_id — skipping")
        return

    base = seeds_base or _seeds_base()

    # Resolve the user's email/name from the identity record when callers don't
    # supply them (most do not) — needed to mint the Person profile and to match
    # pre-signup leads by email. Non-fatal: degrades to no email.
    if email is None or name is None:
        try:
            from services import person_service
            person = person_service.get_user_by_id(arango_db, user_id)
            if person is not None:
                email = email or getattr(person, "email", None)
                name = name or getattr(person, "name", None)
        except Exception:
            logger.debug("provision_user (%s): could not resolve person email/name", user_id, exc_info=True)

    inbox_id = _ensure_inbox_workspace(arango_db, user_id)
    ctx = UserContext(id=user_id, email=email, name=name, inbox_id=inbox_id, tenant=tenant)

    _apply_grant_set(arango_db, base / "user", ctx, user_id)
    if _is_platform_admin(arango_db, user_id):
        _apply_grant_set(arango_db, base / "admin", ctx, user_id)

    if inbox_id:
        _materialize_inbox(arango_db, user_id, inbox_id)

    # Identity linkage: give the user a Person artifact and attach any pre-signup
    # leads (contact form, newsletter) that share their email — so the lead's
    # history rolls up to the user. Both idempotent + non-fatal.
    created_person = False
    try:
        created_person = _ensure_person_artifact(arango_db, ctx)
    except Exception:
        logger.warning("provision_user (%s): Person artifact creation failed", user_id, exc_info=True)
    try:
        _convert_leads_for_person(arango_db, user_id, email)
    except Exception:
        logger.warning("provision_user (%s): lead conversion failed", user_id, exc_info=True)
    if created_person:  # genuine first login → one-time welcome
        try:
            _send_welcome_email(arango_db, ctx)
        except Exception:
            logger.warning("provision_user (%s): welcome email failed", user_id, exc_info=True)


def _ensure_people_collection(arango_db: StandardDatabase) -> str:
    """Idempotently ensure the platform **People** collection exists, creating it
    at RUNTIME. Returns its deterministic id.

    People is deliberately NOT a user-readable platform-registry seed (it's absent
    from ``ALL_PLATFORM_COLLECTION_SLUGS`` / the seed tree). It's created here the
    same way an app provisions its bound data collection — deterministic id, then
    create-if-missing + persist the slug→id mapping. Access model: the platform/
    admin manages the directory (the operator gets a management grant); members
    never get a grant on the collection itself — each reaches only their OWN card
    via that card's owner grant ([[feedback_authority_roots_to_person]]).
    """
    people_id = derive_uuid(get_instance_namespace(), "agience", PEOPLE_COLLECTION_SLUG)
    # Register for this session so get_id(<slug>) / ref lookups resolve it.
    register_id(PEOPLE_COLLECTION_SLUG, people_id)
    register_id(f"agience/{PEOPLE_COLLECTION_SLUG}", people_id)

    if db_get_collection_by_id(arango_db, people_id) is None:
        now = datetime.now(timezone.utc).isoformat()
        db_create_collection(arango_db, CollectionEntity(
            id=people_id,
            name="People",
            description=(
                "Platform people directory — each member's profile lives here. "
                "Members manage only their own card; the platform/admin manages all."
            ),
            created_by=AGIENCE_PLATFORM_USER_ID,
            content_type=COLLECTION_CONTENT_TYPE,
            state=CollectionEntity.STATE_COMMITTED,
            created_time=now, modified_time=now,
        ))
        _persist_seed_ids(arango_db, {PEOPLE_COLLECTION_SLUG: people_id})
        logger.info("Created People collection %s at runtime", people_id)

    # Platform/admin manageability (idempotent): the operator manages the directory.
    from services.platform_settings_service import settings
    operator_id = settings.get("platform.operator_id")
    if operator_id:
        try:
            db_upsert_user_collection_grant(
                arango_db, user_id=operator_id, collection_id=people_id,
                granted_by=AGIENCE_PLATFORM_USER_ID, name="People (admin management)",
                can_read=True, can_create=True, can_update=True, can_delete=True,
                can_evict=True, can_add=True, can_invoke=True, can_admin=True,
            )
        except Exception:
            logger.debug("People: operator grant upsert failed", exc_info=True)
    return people_id


def _ensure_person_owner_grant(arango_db: StandardDatabase, person_id: str, user_id: str) -> None:
    """Grant the user read+update on their OWN Person card (idempotent).

    People is a platform collection — members get only ``read`` on it, so the
    owner needs an explicit grant to EDIT their own profile. Authority roots to
    the person whose card it is (granted_by == the user). See
    ``feedback_authority_roots_to_person``.
    """
    existing = db_get_grants_for_principal_resource(arango_db, user_id, person_id)
    if any(getattr(g, "can_update", False) for g in existing):
        return
    db_create_grant(arango_db, GrantEntity(
        resource_id=person_id, grantee_type="user", grantee_id=user_id,
        granted_by=user_id, can_read=True, can_update=True,
        name="Own profile (read + edit)",
    ))


def _migrate_person_home(
    arango_db: StandardDatabase, ctx: UserContext, person_id: str, people_id: Optional[str]
) -> None:
    """One-time self-heal: move an existing Person card out of the legacy Inbox
    home into the People collection. No-op once migrated. Safe to run every login.
    """
    if not people_id:
        return
    if not db_get_edge(arango_db, people_id, person_id):
        db_add_artifact_to_collection(arango_db, people_id, person_id, origin=True)
    if ctx.inbox_id and db_get_edge(arango_db, ctx.inbox_id, person_id):
        db_remove_artifact_from_collection(arango_db, ctx.inbox_id, person_id)
        entity = db_get_artifact(arango_db, person_id)
        if entity is not None and getattr(entity, "collection_id", None) != people_id:
            entity.collection_id = people_id
            db_update_artifact(arango_db, entity)
        logger.info("Migrated Person %s out of inbox into People", person_id)


def _ensure_person_artifact(arango_db: StandardDatabase, ctx: UserContext) -> bool:
    """Create the user's Person artifact on first login (idempotent).

    The card is homed in the platform **People** collection (the directory /
    data home for the People app) — NOT the user's Inbox. It carries the stable
    user id in ``context.identity.agience_root_id`` and gets an owner grant so
    the user can view + edit their own profile. Pre-existing cards are migrated
    off the Inbox. Returns True only when it creates the card this call.
    """
    pid = person_artifact_id(ctx.id)
    # People is created at runtime (not seeded) — ensure it before homing the card.
    people_id = _ensure_people_collection(arango_db)

    if db_get_artifact(arango_db, pid):
        # Already exists — self-heal its home + owner grant, then report no-create.
        _migrate_person_home(arango_db, ctx, pid, people_id)
        _ensure_person_owner_grant(arango_db, pid, ctx.id)
        return False

    display_name = (ctx.name or "").strip()
    if not display_name:
        display_name = (ctx.email or "").split("@")[0] if ctx.email else "Member"
    identity = {"agience_root_id": ctx.id}
    if ctx.tenant:
        # Which IdP / tenant this member authenticated through (multi-tenant).
        identity["tenant"] = ctx.tenant
    context = {
        "content_type": PERSON_CONTENT_TYPE,
        "identity": identity,
        "display_name": display_name,
        "email": ctx.email or "",
    }
    now = datetime.now(timezone.utc).isoformat()
    db_create_artifact(arango_db, ArtifactEntity(
        id=pid, root_id=pid, collection_id=people_id,
        state=ArtifactEntity.STATE_COMMITTED,
        context=json.dumps(context, separators=(",", ":")),
        content="", content_type=PERSON_CONTENT_TYPE,
        name=display_name, description="Platform member profile.",
        created_by=ctx.id, created_time=now,
    ))
    if not db_get_edge(arango_db, people_id, pid):
        # origin edge: the operator's People grant propagates to cards (admin sees
        # all); members have no People grant, so nothing propagates to them — they
        # reach only their own card via its owner grant.
        db_add_artifact_to_collection(arango_db, people_id, pid, origin=True)
    _ensure_person_owner_grant(arango_db, pid, ctx.id)
    logger.info("Created Person artifact %s for user %s (People)", pid, ctx.id)
    return True


def _send_welcome_email(arango_db: StandardDatabase, ctx: UserContext) -> None:
    """Send a one-time welcome email to a newly-provisioned user.

    Sent AS the platform operator (who holds read on the platform email secrets,
    so the authorizer resolves) via Mantle's chorus_client, which mints the
    operator delegation. Recipient is the new user. Non-fatal.
    """
    if not ctx.email:
        return
    from services.platform_settings_service import settings
    from services import chorus_client, server_registry

    operator_id = settings.get("platform.operator_id")
    iris_id = server_registry.resolve_name_to_id("iris")
    if not operator_id or not iris_id:
        return

    greeting = ctx.name or "there"
    body_html = (
        f"<p>Hi {greeting},</p>"
        "<p>Welcome to Agience — your account is ready. Provenance-first, "
        "governable AI, built in from the start.</p>"
        "<p>Jump in at <a href=\"https://my.agience.ai\">my.agience.ai</a>.</p>"
        "<p>— The Agience team</p>"
    )
    chorus_client.call_tool(
        iris_id, "send_email",
        {"to": ctx.email, "subject": "Welcome to Agience", "body_html": body_html},
        user_id=str(operator_id),
    )
    logger.info("Sent welcome email to %s", ctx.email)


def _convert_leads_for_person(arango_db: StandardDatabase, user_id: str, email: Optional[str]) -> int:
    """Link unclaimed leads matching the user's email to their new Person.

    Sets context.person_id + status=converted on each matching lead artifact in
    the Leads collection. Idempotent (skips already-linked leads). Returns count.
    """
    target = (email or "").strip().lower()
    if not target:
        return 0
    col_id = get_id_optional(_LEADS_COLLECTION_SLUG)
    if not col_id:
        return 0

    try:
        rows = db_list_collection_artifacts(arango_db, col_id) or []
    except Exception:
        logger.exception("convert_leads: failed listing leads collection %s", col_id)
        return 0

    converted = 0
    for row in rows:
        rid = str(
            (row.get("root_id") or row.get("id") or "") if isinstance(row, dict) else ""
        ).strip()
        if not rid:
            continue
        entity = db_get_artifact(arango_db, rid)
        if entity is None:
            continue
        ctxd = _parse_json_obj(getattr(entity, "context", ""))
        if ctxd.get("type") != "lead" or ctxd.get("person_id"):
            continue
        content = _parse_json_obj(getattr(entity, "content", ""))
        lead_email = str(content.get("email") or ctxd.get("email") or "").strip().lower()
        if lead_email != target:
            continue
        ctxd["person_id"] = user_id
        ctxd["status"] = "converted"
        ctxd["converted_at"] = datetime.now(timezone.utc).isoformat()
        entity.context = json.dumps(ctxd, separators=(",", ":"))
        db_update_artifact(arango_db, entity)
        converted += 1

    if converted:
        logger.info("convert_leads: linked %d lead(s) to person %s", converted, user_id)
    return converted


def _apply_grant_set(
    arango_db: StandardDatabase, root: Path, ctx: UserContext, user_id: str
) -> None:
    report = seed_from_artifacts(arango_db, root, user=ctx)
    for err in report.errors:
        logger.warning("provision_user (%s): %s", user_id, err)


def _ensure_inbox_workspace(arango_db: StandardDatabase, user_id: str) -> Optional[str]:
    """Return the user's primary (oldest) workspace id, creating an "Inbox"
    workspace on first login. ``create_workspace`` issues the owner grant."""
    from services import workspace_service

    existing = workspace_service.list_workspaces(arango_db, user_id)
    if existing:
        primary = min(existing, key=lambda w: getattr(w, "created_time", "") or "")
        return primary.id
    new_ws = workspace_service.create_workspace(arango_db, user_id, "Inbox")
    return new_ws.id


def _materialize_inbox(arango_db: StandardDatabase, user_id: str, inbox_workspace_id: str) -> None:
    """Link curated platform seed artifacts into the user's Inbox workspace.

    Skips artifacts already linked so user/operator reordering (``order_key``) is
    never clobbered on re-run.
    """
    seen: set[str] = set()
    for slug in INBOX_MATERIALIZATION_SLUGS:
        col_id = get_id_optional(slug)
        if not col_id:
            continue
        try:
            artifacts = db_list_collection_artifacts(arango_db, col_id)
        except Exception:
            logger.exception("Failed loading seed artifacts from collection %s", slug)
            continue

        for artifact in artifacts or []:
            root_id = str(
                (artifact.get("root_id", "") if isinstance(artifact, dict)
                 else getattr(artifact, "root_id", "")) or ""
            ).strip()
            if not root_id or root_id in seen:
                continue
            seen.add(root_id)

            # Skip if already linked — avoids resetting order_key on every login.
            if db_get_edge(arango_db, inbox_workspace_id, root_id):
                continue
            try:
                db_add_artifact_to_collection(arango_db, inbox_workspace_id, root_id)
            except Exception:
                logger.exception(
                    "Failed importing seed artifact root %s into inbox workspace %s",
                    root_id, inbox_workspace_id,
                )
