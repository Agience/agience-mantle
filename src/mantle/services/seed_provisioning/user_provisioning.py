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

import contextlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from mantle.db.store import Database

from mantle.db.backend import (
    add_artifact_to_collection as db_add_artifact_to_collection,
    create_collection as db_create_collection,
    create_grant as db_create_grant,
    get_active_grants_for_principal_resource as db_get_grants_for_principal_resource,
    get_artifact as db_get_artifact,
    get_collection_by_id as db_get_collection_by_id,
    get_edge as db_get_edge,
    list_collection_artifacts as db_list_collection_artifacts,
    update_artifact as db_update_artifact,
)
from mantle.entities.collection import Collection as CollectionEntity, COLLECTION_CONTENT_TYPE
from mantle.entities.grant import Grant as GrantEntity
from mantle.config import AGIENCE_PLATFORM_USER_ID
from mantle.services import mint_context as _mc
from mantle.services.bootstrap_types import (
    AUTHORITY_COLLECTION_SLUG,
    INBOX_MATERIALIZATION_SLUGS,
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
    seed application: the runtime provisioners (the Authorities collection, Person cards,
    issuers) still run, and only the declarative grant seeds are skipped."""
    env = os.getenv("AGIENCE_SEEDS_ROOT")
    return Path(env) if env else None


def _is_platform_admin(store_db: Database, user_id: str) -> bool:
    """The designated platform admin is the user in ``platform.operator_id``."""
    from mantle.services.platform_settings_service import settings

    return bool(user_id) and settings.get("platform.operator_id") == user_id


def _resolve_profile(user_id: str, email: Optional[str], name: Optional[str]):
    """`(email, name)`, filled from Origin's person record when either is missing, on the
    person's own access token only (review 21, F4): the first sign-in request carries it. A
    script, a delegation or a grant key carries none, and then Origin is asked nothing and the
    profile has no address."""
    if email is not None and name is not None:
        return email, name
    try:
        from mantle.services import person_service
        token = person_service.own_subject_token(user_id)
        if token is None:
            return email, name
        person = person_service.get_user_by_id(None, user_id, subject_token=token)
        if person is not None:
            email = email or getattr(person, "email", None)
            name = name or getattr(person, "name", None)
    except Exception:
        logger.debug("provision_user (%s): could not resolve person email/name", user_id,
                     exc_info=True)
    return email, name


def provision_user(
    store_db: Database,
    user_id: str,
    **kwargs,
) -> None:
    """Provision a user on first login (`_provision_user`), acting as that user.

    Everything it writes is theirs, and their Inbox's name is sealed under their key, so it runs
    as them whoever called it: the first-login request (already them), or an admin script
    (`manage_seed`, `manage_bootstrap`) that has no principal of its own. 2026-10-29, sealing
    everything: run bare, the Inbox write was refused key material."""
    if not user_id:
        logger.warning("provision_user called with empty user_id — skipping")
        return
    from mantle.services.acting_principal import acting_as
    with acting_as(str(user_id), principal_type="user"):
        _provision_user(store_db, user_id, **kwargs)


def _provision_user(
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

    # The user's email and name from Origin's record when the caller did not supply them, needed
    # to mint the Person profile and to match pre-signup leads by email. Non-fatal.
    email, name = _resolve_profile(user_id, email, name)

    inbox_id = _ensure_inbox_workspace(store_db, user_id)
    _ensure_observations_container(store_db, user_id)
    _ensure_ember_memory(store_db, user_id)
    ctx = UserContext(id=user_id, email=email, name=name, inbox_id=inbox_id, tenant=tenant)

    # Declarative grant seeds are applied ONLY when the install package supplies a
    # seed corpus (AGIENCE_SEEDS_ROOT). Bare Mantle has none — skip cleanly.
    if base is not None:
        _apply_grant_set(store_db, base / "user", ctx, user_id)
        if _is_platform_admin(store_db, user_id):
            _apply_grant_set(store_db, base / "admin", ctx, user_id)

    if inbox_id:
        _materialize_inbox(store_db, user_id, inbox_id)

    # The platform's own operators, one grant each (`platform_operators`). Non-fatal: an account
    # without them still signs in, and the release migration grants every account again.
    try:
        from mantle.services.seed_provisioning.platform_operators import grant_to_account
        grant_to_account(store_db, user_id)
    except Exception:
        logger.warning("provision_user (%s): platform operators not granted", user_id, exc_info=True)

    # Identity linkage: give the user a Person artifact and attach any pre-signup
    # leads (contact form, newsletter) that share their email — so the lead's
    # history rolls up to the user. Both idempotent + non-fatal.
    try:
        _ensure_person_artifact(store_db, ctx)
    except Exception:
        # Under threshold custody a refusal here leaves a member without a linked card: it is
        # not swallowed there, so the sign-in says so.
        from mantle.db import reach_journal as _rj
        logger.error("provision_user (%s): Person artifact creation failed", user_id,
                     exc_info=True)
        if _rj.required():
            raise
    try:
        _convert_leads_for_person(store_db, user_id, email)
    except Exception:
        logger.warning("provision_user (%s): lead conversion failed", user_id, exc_info=True)


@contextlib.contextmanager
def _as_the_platform():
    """Write one of the platform's own collections as the platform's id, its `created_by`, so the
    key oracle's creator rule mints its key. Provisioning runs as the member signing in, who holds
    no key to a platform row. The platform system principal is granted the collection once it
    exists (`ensure_platform_principal_grant`), and reads it from then on."""
    from mantle.services.acting_principal import acting_as
    with acting_as(AGIENCE_PLATFORM_USER_ID, principal_type="service", scope="platform.provision",
                   source="system"):
        yield


#: The ledger's grantee type for any principal that acts by id, a service included
#: (`search.mantle.lightcone.ledger_grantee_type`): a grant typed "service" is found by no lookup.
_PRINCIPAL_GRANTEE = "user"


def _platform_holds(store_db: Database, collection_id: str) -> bool:
    from mantle.services.peer_signing import get_system_principal_id
    system = get_system_principal_id()
    return bool(system) and any(
        getattr(g, "grantee_type", None) == _PRINCIPAL_GRANTEE and getattr(g, "can_read", False)
        and getattr(g, "can_update", False)
        for g in db_get_grants_for_principal_resource(store_db, system, collection_id))


def ensure_platform_principal_grant(store_db: Database, collection_id: str) -> bool:
    """Grant the platform system principal read and update on a collection the platform owns
    (idempotent). True when a grant was made.

    The platform's collections are created by `AGIENCE_PLATFORM_USER_ID`, a name no request ever
    acts as, so nothing held their key: work the platform does on its own rows (sealing them, the
    rollback's unsealing) runs as the system principal (`system_acting_context`) and was refused.
    Without it the platform's own rows can be sealed by no actor and stay in plain text."""
    from mantle.services.peer_signing import get_system_principal_id
    system = get_system_principal_id()
    if not system or not collection_id:
        return False
    if _platform_holds(store_db, collection_id):
        return False
    db_create_grant(store_db, GrantEntity(
        resource_id=collection_id, grantee_type=_PRINCIPAL_GRANTEE, grantee_id=system,
        granted_by=AGIENCE_PLATFORM_USER_ID, can_read=True, can_update=True,
        name="Platform rows (read + update)",
    ))
    return True


def ensure_authority_collection(store_db: Database) -> str:
    """Idempotently ensure the platform **authority** collection exists (runtime),
    returning its deterministic id.

    This is the canonical resource platform-admin roots to: a user holding an
    active ``can_admin``/``can_update`` grant on it IS a platform admin (see
    ``services.dependencies.require_platform_admin``). It MUST exist unconditionally
    (not only when a seed corpus is loaded) so multi-admin management works on any
    node — created here by deterministic id, if missing.
    """
    authority_id = derive_uuid(get_instance_namespace(), "agience", AUTHORITY_COLLECTION_SLUG)
    register_id(AUTHORITY_COLLECTION_SLUG, authority_id)
    register_id(f"agience/{AUTHORITY_COLLECTION_SLUG}", authority_id)

    if db_get_collection_by_id(store_db, authority_id) is None:
        now = datetime.now(timezone.utc).isoformat()
        with _as_the_platform():
            db_create_collection(store_db, CollectionEntity(
                id=authority_id,
                name="Authorities",
                description="Platform authority root — platform-admin grants live here.",
                created_by=AGIENCE_PLATFORM_USER_ID,
                content_type=COLLECTION_CONTENT_TYPE,
                state=CollectionEntity.STATE_LIVE,
                created_time=now, modified_time=now,
            ))
        _persist_seed_ids(store_db, {AUTHORITY_COLLECTION_SLUG: authority_id})
        logger.info("Created Authorities collection %s at runtime", authority_id)
    ensure_platform_principal_grant(store_db, authority_id)
    return authority_id


def own_card_id(store_db: Database, user_id: str) -> str:
    """The id of the person's own card. `person_artifact_id`, unless a card written in the People
    directory before 2026-11-01 holds that id: that card is the platform's now
    (`mantle.system.people_to_signups`), and the person's own card has an id of its own."""
    pid = person_artifact_id(user_id)
    old = db_get_artifact(store_db, pid)
    if old is not None and getattr(old, "collection_id", None) not in (None, "", pid):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"agience://person/{user_id}/own"))
    return pid


def _ensure_person_artifact(store_db: Database, ctx: UserContext) -> bool:
    """Create the person's own card on first sign-in (idempotent). True only when it creates it.

    The card belongs to the person (John, 2026-11-01): it is its own root, created as the person
    on their own credential, with their own creation grant, in one batch Origin issues
    (`workspace_service.create_container`). It is placed in no collection and no one else is
    granted anything on it. Sharing it is the person's own act: they place it, as a link, in a
    collection where they hold Add. It carries the stable user id in
    ``context.identity.agience_root_id``.

    A card written in the People directory before 2026-11-01 is the platform's
    (`mantle.system.people_to_signups`): it is left alone, nothing is granted on it, and the
    person's own card is made beside it at an id of its own (`own_card_id`).

    It ensures no platform collection: a member's sign-in holds no credential that may create one
    (the Authorities collection is the bootstrap's, `manage_bootstrap`).
    """
    pid = own_card_id(store_db, ctx.id)
    if db_get_artifact(store_db, pid):
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
    # Provisioning acts as the person (`provision_user`), on the credential their sign-in carries.
    from mantle.services import workspace_service
    workspace_service.create_container(
        store_db, ctx.id, content_type=PERSON_CONTENT_TYPE, name=display_name,
        context=json.dumps(context, separators=(",", ":")),
        description="Platform member profile.", artifact_id=pid,
    )
    logger.info("Created Person artifact %s for user %s (own root)", pid, ctx.id)
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
        # The lead's own keys, wherever the stored context holds them (`mint_context.client_view`).
        stored = getattr(entity, "context", "")
        ctxd = _mc.client_view(stored)
        if ctxd.get("type") != "lead" or ctxd.get("person_id"):
            continue
        content = _parse_json_obj(getattr(entity, "content", ""))
        lead_email = str(content.get("email") or ctxd.get("email") or "").strip().lower()
        if lead_email != target:
            continue
        ctxd["person_id"] = user_id
        ctxd["status"] = "converted"
        ctxd["converted_at"] = datetime.now(timezone.utc).isoformat()
        entity.context = _mc.with_client_keys(stored, ctxd)
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


#: The natural key Ember's memory collection is derived from, per person (`ember.mantle_api`).
EMBER_MEMORY_IDENTITY = "ember:memory"


def _ensure_ember_memory(store_db: Database, user_id: str) -> Optional[str]:
    """The person's own memory collection for Ember, made at their sign-in on their own token.

    Ember opens no person's key (John, 2026-11-02, review 10), so it can neither make this
    collection nor read it: it only writes memories into it, sealed to the collection's drop key.
    Made here, as the person, the collection gets its key and its drop key from their own open, and
    is theirs by their creation grant. Idempotent; non-fatal (Ember then says the collection comes
    at the next sign-in)."""
    from mantle.db.constants import COLLECTION_CONTENT_TYPE as _COLLECTION
    from mantle.services import workspace_service
    from mantle.services.artifact_identity import derive_artifact_id

    mem_id = derive_artifact_id(str(user_id), EMBER_MEMORY_IDENTITY)
    # Made only on the person's own token: never on a narrowed delegation (Ember's, a persona's),
    # which Origin refuses to make it anyway (review 11).
    from mantle.services.acting_principal import current_acting_principal
    acting = current_acting_principal()
    if acting is not None and getattr(acting, "narrowed", None):
        return None
    try:
        if db_get_artifact(store_db, mem_id) is not None:
            return mem_id
        workspace_service.create_container(
            store_db, str(user_id), content_type=_COLLECTION, name="Memory",
            description="What you asked Ember to remember. Yours alone until you share it.",
            artifact_id=mem_id)
        return mem_id
    except Exception:  # noqa: BLE001 — a missing memory collection must not fail a sign-in
        logger.warning("provision_user (%s): Ember's memory collection not made", user_id,
                       exc_info=True)
        return None


def ensure_ember_memory_for(store_db: Database, auth, known: set) -> Optional[str]:
    """Make the person's memory collection at a sign-in that is not their first (review 11):
    first-login provisioning makes it for a new person, and a person who already held grants
    before it existed gets it here. Only on their own token, and only when their reach does not
    already hold it."""
    from mantle.services.artifact_identity import derive_artifact_id
    from mantle.services.dependencies import is_narrowed

    user_id = getattr(auth, "user_id", None)
    if not user_id or is_narrowed(auth) or getattr(auth, "principal_type", "user") != "user":
        return None
    mem_id = derive_artifact_id(str(user_id), EMBER_MEMORY_IDENTITY)
    if mem_id in (known or set()):
        return None
    try:
        if db_get_artifact(store_db, mem_id) is not None:
            return None                       # there already; nothing new to resolve
    except Exception:                         # noqa: BLE001 — fall through to the ensure
        pass
    return _ensure_ember_memory(store_db, str(user_id))


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


def provision_existing_accounts(store_db: Database, base: Optional[Path]) -> Dict[str, int]:
    """After a seed: give every account the store knows the user grant set and the platform
    operators, for each `(account, resource)` pair that has never had a grant record.

    An account that signed in before the seed found nothing to be granted, and first sign-in runs
    once (71/test, 2026-11-12). A pair with any record, active or revoked, is left as it is, so a
    grant once withheld stays withheld."""
    from mantle.db import lattice_api as la
    from mantle.services.peer_signing import get_system_principal_id
    from mantle.services.seed_provisioning.loader import UserContext, only_where_never_granted
    from mantle.services.seed_provisioning.platform_operators import grant_to_account

    docs = list(la._grant_docs(store_db))
    pairs = frozenset((str(d.get("grantee_id")), str(d.get("resource_id"))) for d in docs)
    accounts = {str(d["grantee_id"]) for d in docs
                if d.get("grantee_type") == "user" and d.get("grantee_id")}
    accounts -= {get_system_principal_id(), ""}
    out = {"accounts": len(accounts), "operators": 0}
    with only_where_never_granted(pairs):
        for a in sorted(accounts):
            try:
                if base is not None and (base / "user").is_dir():
                    _apply_grant_set(store_db, base / "user", UserContext(id=a), a)
                out["operators"] += grant_to_account(store_db, a, skip=pairs)
            except Exception:  # noqa: BLE001 — one account must not stop the rest
                logger.warning("provision_existing_accounts (%s): not provisioned", a, exc_info=True)
    return out
