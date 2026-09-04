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

from mantle.db.store import Database

from mantle.db.backend import (
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
from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.entities.collection import Collection as CollectionEntity, COLLECTION_CONTENT_TYPE
from mantle.entities.grant import Grant as GrantEntity
from mantle.config import AGIENCE_PLATFORM_USER_ID
from mantle.services.bootstrap_types import (
    AUTHORITY_COLLECTION_SLUG,
    INBOX_MATERIALIZATION_SLUGS,
    PEOPLE_COLLECTION_SLUG,
)
from mantle.services.platform_topology import get_id_optional, register_id
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


def inbox_workspace_id(user_id: str) -> str:
    """Deterministic id for the workspace first login creates, alongside
    :func:`person_artifact_id`.

    `_materialize_inbox` has a PARAMETER of this name, which shadows this function
    inside that body. Nothing there calls it, and the symmetry with
    :func:`person_artifact_id` is worth more than renaming a used signature.

    Provisioning runs as a side effect of a read, so two requests on a first login
    arrive together as a matter of course. Deriving the id is what makes them
    converge: both racers address the same row instead of each minting a UUID and
    leaving one workspace owned and orphaned.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agience://workspace/inbox/{user_id}"))


def _seeds_base() -> Optional[Path]:
    """Root of the platform seed corpus, or None when Mantle runs bare.

    Mantle bundles no seed content: it is a bare encrypted data plane. A seed corpus is an
    install-package artifact supplied by the deployment rather than shipped in the Mantle image,
    provided at deploy time by pointing ``AGIENCE_SEEDS_ROOT`` at the mounted seeds. Unset means no
    seed application: the runtime provisioners (People/Authorities collections, Person cards,
    issuers) still run, and only the declarative grant seeds are skipped."""
    env = os.getenv("AGIENCE_SEEDS_ROOT")
    return Path(env) if env else None


def _is_platform_admin(store_db: Database, user_id: str) -> bool:
    """The designated platform admin is the user in ``platform.operator_id``."""
    from mantle.services.platform_settings_service import settings

    return bool(user_id) and settings.get("platform.operator_id") == user_id


def provision_user(
    store_db: Database,
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
            from mantle.services import person_service
            # No subject token: provisioning is a bootstrap path, run under no user's request —
            # it is often what runs BEFORE that user has ever presented a token here. So this
            # stays the unscoped call, and nothing is minted to make it look otherwise. It is
            # already non-fatal: an Origin that refuses degrades to "no email", which the code
            # below is written to survive.
            person = person_service.get_user_by_id(store_db, user_id)
            if person is not None:
                email = email or getattr(person, "email", None)
                name = name or getattr(person, "name", None)
        except Exception:
            logger.debug("provision_user (%s): could not resolve person email/name", user_id, exc_info=True)

    inbox_id = _ensure_inbox_workspace(store_db, user_id)
    _ensure_observations_container(store_db, user_id)
    ctx = UserContext(id=user_id, email=email, name=name, inbox_id=inbox_id, tenant=tenant)

    # Declarative grant seeds are applied ONLY when the install package supplies a
    # seed corpus (AGIENCE_SEEDS_ROOT). Bare Mantle has none — skip cleanly.
    if base is not None:
        _apply_grant_set(store_db, base / "user", ctx, user_id)
        if _is_platform_admin(store_db, user_id):
            _apply_grant_set(store_db, base / "admin", ctx, user_id)

    if inbox_id:
        _materialize_inbox(store_db, user_id, inbox_id)

    # Identity linkage: give the user a Person artifact and attach any pre-signup
    # leads (contact form, newsletter) that share their email — so the lead's
    # history rolls up to the user. Both idempotent + non-fatal.
    try:
        _ensure_person_artifact(store_db, ctx)
    except Exception:
        logger.warning("provision_user (%s): Person artifact creation failed", user_id, exc_info=True)
    try:
        _convert_leads_for_person(store_db, user_id, email)
    except Exception:
        logger.warning("provision_user (%s): lead conversion failed", user_id, exc_info=True)


def _ensure_people_collection(store_db: Database) -> str:
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

    if db_get_collection_by_id(store_db, people_id) is None:
        now = datetime.now(timezone.utc).isoformat()
        db_create_collection(store_db, CollectionEntity(
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
        _persist_seed_ids(store_db, {PEOPLE_COLLECTION_SLUG: people_id})
        logger.info("Created People collection %s at runtime", people_id)

    # Platform/admin manageability (idempotent): the operator manages the directory.
    from mantle.services.platform_settings_service import settings
    operator_id = settings.get("platform.operator_id")
    if operator_id:
        try:
            db_upsert_user_collection_grant(
                store_db, user_id=operator_id, collection_id=people_id,
                granted_by=AGIENCE_PLATFORM_USER_ID, name="People (admin management)",
                can_read=True, can_create=True, can_update=True, can_delete=True,
                can_evict=True, can_add=True, can_invoke=True, can_admin=True,
            )
        except Exception:
            logger.debug("People: operator grant upsert failed", exc_info=True)
    return people_id


def ensure_authority_collection(store_db: Database) -> str:
    """Idempotently ensure the platform **authority** collection exists (runtime),
    returning its deterministic id.

    This is the canonical resource platform-admin roots to: a user holding an
    active ``can_admin``/``can_update`` grant on it IS a platform admin (see
    ``services.dependencies.require_platform_admin``). It MUST exist unconditionally
    (not only when a seed corpus is loaded) so multi-admin management works on any
    node — created here the same deterministic-id, create-if-missing way as People.
    """
    authority_id = derive_uuid(get_instance_namespace(), "agience", AUTHORITY_COLLECTION_SLUG)
    register_id(AUTHORITY_COLLECTION_SLUG, authority_id)
    register_id(f"agience/{AUTHORITY_COLLECTION_SLUG}", authority_id)

    if db_get_collection_by_id(store_db, authority_id) is None:
        now = datetime.now(timezone.utc).isoformat()
        db_create_collection(store_db, CollectionEntity(
            id=authority_id,
            name="Authorities",
            description="Platform authority root — platform-admin grants live here.",
            created_by=AGIENCE_PLATFORM_USER_ID,
            content_type=COLLECTION_CONTENT_TYPE,
            state=CollectionEntity.STATE_COMMITTED,
            created_time=now, modified_time=now,
        ))
        _persist_seed_ids(store_db, {AUTHORITY_COLLECTION_SLUG: authority_id})
        logger.info("Created Authorities collection %s at runtime", authority_id)
    return authority_id


def _ensure_person_owner_grant(store_db: Database, person_id: str, user_id: str) -> None:
    """Grant the user read+update+delete on their OWN Person card (idempotent).

    People is a platform collection — members get only ``read`` on it, so the
    owner needs an explicit grant to EDIT their own profile. Authority roots to
    the person whose card it is (granted_by == the user). See
    ``feedback_authority_roots_to_person``.

    ``can_delete`` is part of owning it. This card is the artifact that holds a
    member's plaintext email and display name, and ``DELETE /artifacts/{id}``
    resolves through the same grant flags as every other artifact — without the
    flag the one person the data is about is the one principal who cannot remove
    it. The idempotence check therefore tests for delete too, so a card issued
    with the narrower pair is widened on the owner's next login rather than
    staying undeletable for the life of the account.
    """
    existing = db_get_grants_for_principal_resource(store_db, user_id, person_id)
    if any(getattr(g, "can_update", False) and getattr(g, "can_delete", False)
           for g in existing):
        return
    db_create_grant(store_db, GrantEntity(
        resource_id=person_id, grantee_type="user", grantee_id=user_id,
        granted_by=user_id, can_read=True, can_update=True, can_delete=True,
        name="Own profile (read + edit + delete)",
    ))


def _migrate_person_home(
    store_db: Database, ctx: UserContext, person_id: str, people_id: Optional[str]
) -> None:
    """One-time self-heal: move an existing Person card out of the Inbox
    home into the People collection. No-op once migrated. Safe to run every login.
    """
    if not people_id:
        return
    if not db_get_edge(store_db, people_id, person_id):
        db_add_artifact_to_collection(store_db, people_id, person_id, origin=True)
    if ctx.inbox_id and db_get_edge(store_db, ctx.inbox_id, person_id):
        db_remove_artifact_from_collection(store_db, ctx.inbox_id, person_id)
        entity = db_get_artifact(store_db, person_id)
        if entity is not None and getattr(entity, "collection_id", None) != people_id:
            entity.collection_id = people_id
            db_update_artifact(store_db, entity)
        logger.info("Migrated Person %s out of inbox into People", person_id)


def _ensure_person_artifact(store_db: Database, ctx: UserContext) -> bool:
    """Create the user's Person artifact on first login (idempotent).

    The card is homed in the platform **People** collection (the directory /
    data home for the People app) — NOT the user's Inbox. It carries the stable
    user id in ``context.identity.agience_root_id`` and gets an owner grant so
    the user can view + edit their own profile. Pre-existing cards are migrated
    off the Inbox. Returns True only when it creates the card this call.
    """
    pid = person_artifact_id(ctx.id)
    # People + Authorities are created at runtime (not seeded) — ensure both.
    people_id = _ensure_people_collection(store_db)
    ensure_authority_collection(store_db)

    if db_get_artifact(store_db, pid):
        # Already exists — self-heal its home + owner grant, then report no-create.
        _migrate_person_home(store_db, ctx, pid, people_id)
        _ensure_person_owner_grant(store_db, pid, ctx.id)
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
    db_create_artifact(store_db, ArtifactEntity(
        id=pid, root_id=pid, collection_id=people_id,
        state=ArtifactEntity.STATE_COMMITTED,
        context=json.dumps(context, separators=(",", ":")),
        content="", content_type=PERSON_CONTENT_TYPE,
        name=display_name, description="Platform member profile.",
        created_by=ctx.id, created_time=now,
    ))
    if not db_get_edge(store_db, people_id, pid):
        # origin edge: the operator's People grant propagates to cards (admin sees
        # all); members have no People grant, so nothing propagates to them — they
        # reach only their own card via its owner grant.
        db_add_artifact_to_collection(store_db, people_id, pid, origin=True)
    _ensure_person_owner_grant(store_db, pid, ctx.id)
    logger.info("Created Person artifact %s for user %s (People)", pid, ctx.id)
    return True


def _convert_leads_for_person(store_db: Database, user_id: str, email: Optional[str]) -> int:
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
        rows = db_list_collection_artifacts(store_db, col_id) or []
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
        entity = db_get_artifact(store_db, rid)
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
        db_update_artifact(store_db, entity)
        converted += 1

    if converted:
        logger.info("convert_leads: linked %d lead(s) to person %s", converted, user_id)
    return converted


def _apply_grant_set(
    store_db: Database, root: Path, ctx: UserContext, user_id: str
) -> None:
    report = seed_from_artifacts(store_db, root, user=ctx)
    for err in report.errors:
        logger.warning("provision_user (%s): %s", user_id, err)


def _ensure_observations_container(store_db: Database, user_id: str) -> Optional[str]:
    """Ensure the user owns the container their observation events are addressed to.

    Provisioned here because the read path does not write. `events/observation.py` records an
    event for every MCP tool call and addresses it to this container — which is what keeps one
    principal's queries out of the event feed of everyone holding a grant on whatever those
    queries matched. That addressing needs the container to exist and to carry the owner grant
    `create_container` issues; obtaining it lazily during a read would put a container create
    inside a user-facing tool call, and announce it on the change feed as a side effect of looking.

    Idempotent, and non-fatal: a user whose container is missing simply records no observations
    until this runs again. The audit gap is the honest failure — a failed provisioning must not
    fail a login.
    """
    from mantle.events import observation

    return observation.ensure_observations_container(store_db, user_id)


def _ensure_inbox_workspace(store_db: Database, user_id: str) -> Optional[str]:
    """Return the user's primary (oldest) workspace id, creating an "Inbox"
    workspace on first login. ``create_workspace`` issues the owner grant."""
    from mantle.services import workspace_service

    existing = workspace_service.list_workspaces(store_db, user_id)
    if existing:
        primary = min(existing, key=lambda w: getattr(w, "created_time", "") or "")
        return primary.id
    # The id is DERIVED, not minted. This block is check-then-act and nothing makes it
    # atomic, so two requests on a first login both see the empty list and both create.
    # With a fresh UUID each that left a second workspace owned and orphaned, resolved
    # afterwards by `min(created_time)` and reported to nobody.
    #
    # A pinned id makes the two writes address the SAME row, so the duplicate cannot
    # exist. Measured 2026-08-26: `create_collection` is `put_artifact` — the same
    # call as `update_collection` — so the second write UPSERTS rather than raising.
    # That is last-writer-wins, which is safe only because both racers write identical
    # content here; it is NOT an arbiter, so do not build create-then-catch on it.
    #
    # Existing users are untouched: this runs only when the user has no workspace at
    # all, so an Inbox already carrying a random UUID keeps it and needs no migration.
    new_ws = workspace_service.create_workspace(
        store_db, user_id, "Inbox", artifact_id=inbox_workspace_id(user_id),
    )
    return new_ws.id


def _materialize_inbox(store_db: Database, user_id: str, inbox_workspace_id: str) -> None:
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
            artifacts = db_list_collection_artifacts(store_db, col_id)
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
            if db_get_edge(store_db, inbox_workspace_id, root_id):
                continue
            try:
                db_add_artifact_to_collection(store_db, inbox_workspace_id, root_id)
            except Exception:
                logger.exception(
                    "Failed importing seed artifact root %s into inbox workspace %s",
                    root_id, inbox_workspace_id,
                )
