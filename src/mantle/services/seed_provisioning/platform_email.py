"""Platform outbound-email provisioner.

Provisions the composable platform email sender from the deployment's Gmail OAuth
env (``EMAIL_PROVIDER=gmail`` + ``GMAIL_OAUTH_CLIENT_ID`` / ``_CLIENT_SECRET`` /
``_REFRESH_TOKEN`` + ``PLATFORM_EMAIL_ADDRESS``) as a graph of canonical artifacts:

    operator  (vnd.agience.operator+json)  "Platform Email Sender"
      └─(rel: authorizer)→ authorizer (vnd.agience.authorizer+json)  Gmail OAuth client
                              ├─ client_secret_artifact_id → secret (vnd.agience.secret+json)
                              └─ refresh_token_artifact_id → secret (vnd.agience.secret+json)

Invoking the operator runs ``iris:send_email`` with the authorizer attached by
edge; ``send_email`` reads the authorizer config, resolves the two secrets via
their ``fetch`` op (under the caller's delegation — for platform sends that's the
operator, who holds read on the secrets), performs the OAuth exchange, and sends.
The credentials are never returned to the caller. There is NO token-minting op on
any user-facing artifact (the banned ``invoke → provide_access_token`` is not used).

All artifacts are created_by + granted to the platform operator (a real user, so
the secret material lives in a real vault). Deterministic ids (uuid5 by slug) so
the graph is stable and re-runs are idempotent. The only out-of-band input is the
secret MATERIAL (always re-injected from env so it self-heals); structure is
created once. Types stay declarative; this provisions the one platform instance.

Idempotent, non-fatal, runs every startup. No-op until the operator is designated
and ``EMAIL_PROVIDER=gmail`` with creds is configured. Rotate creds by updating
the env and restarting (material is re-injected each boot).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from mantle.db.store import Database

from mantle.db.backend import (
    add_artifact_to_collection as db_add_edge,
    create_artifact as db_create_artifact,
    get_artifact as db_get_artifact,
    update_artifact as db_update_artifact,
    upsert_user_collection_grant as db_upsert_user_collection_grant,
)
from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.services import secrets_service
from mantle.services.platform_topology import register_id
from mantle.services.seed_provisioning.loader import derive_uuid, get_instance_namespace

logger = logging.getLogger(__name__)

_NS = "platform"
_OPERATOR_SLUG = "platform-email-sender"
_AUTHORIZER_SLUG = "platform-email-authorizer"
_CLIENT_SECRET_SLUG = "platform-email-client-secret"
_REFRESH_TOKEN_SLUG = "platform-email-refresh-token"

_OPERATOR_CT = "application/vnd.agience.operator+json"
_AUTHORIZER_CT = "application/vnd.agience.authorizer+json"
_SECRET_CT = "application/vnd.agience.secret+json"

_GMAIL_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_GMAIL_SEND_SCOPE = "https://www.googleapis.com/auth/gmail.send"


def _operator_id(db: Database) -> str:
    """The designated platform admin — created_by + grantee of the email graph.
    Sovereign-capable resolution (Mantle setting → env → Origin fallback)."""
    from mantle.services.operator import resolve_operator_id

    return resolve_operator_id(db)


def _ensure_artifact(
    db: Database, artifact_id: str, *, content_type: str, name: str,
    description: str, context: dict, content: str, owner: str,
) -> bool:
    """Upsert a committed artifact, CONVERGING its content/context to the current
    values (so config like the authorizer's client_id updates when env changes —
    "agnostic to what exists"). Returns True if it already existed."""
    now = datetime.now(timezone.utc).isoformat()
    ctx_json = json.dumps(context, separators=(",", ":"))
    existing = db_get_artifact(db, artifact_id)
    if existing is not None:
        # Reconcile content/context/labels to current; preserve identity + provenance.
        existing.context = ctx_json
        existing.content = content
        existing.name = name
        existing.description = description
        existing.modified_by = owner
        existing.modified_time = now
        db_update_artifact(db, existing)
        return True
    db_create_artifact(db, ArtifactEntity(
        id=artifact_id, root_id=artifact_id,
        state=ArtifactEntity.STATE_COMMITTED,
        context=ctx_json,
        content=content,
        content_type=content_type,
        name=name, description=description,
        created_by=owner, created_time=now,
    ))
    return False


def ensure_platform_email_sender(db: Database) -> bool:
    """Provision the platform email operator graph + inject secret material.

    Idempotent and order-agnostic — safe to run any time, regardless of what is
    already provisioned. Returns ``True`` when the work is done (or intentionally
    not needed: no/other email provider, creds absent), and ``False`` only for a
    TRANSIENT prerequisite (operator/Origin not resolvable yet, or an error), so
    the caller can self-heal by retrying until it converges.
    """
    provider = (os.getenv("EMAIL_PROVIDER", "") or "").strip().lower()
    if provider != "gmail":
        return True  # email off / other provider — nothing to provision

    client_id = (os.getenv("GMAIL_OAUTH_CLIENT_ID", "") or "").strip()
    client_secret = (os.getenv("GMAIL_OAUTH_CLIENT_SECRET", "") or "").strip()
    refresh_token = (os.getenv("GMAIL_OAUTH_REFRESH_TOKEN", "") or "").strip()
    sender = (os.getenv("PLATFORM_EMAIL_ADDRESS", "") or "").strip()
    if not (client_id and client_secret and refresh_token):
        logger.info("platform_email: GMAIL_OAUTH_* not fully configured — skipping")
        return True  # config-driven; won't change without a restart, so don't retry

    operator = _operator_id(db)
    if not operator:
        logger.info(
            "platform_email: operator not yet resolvable (Origin unreachable or "
            "operator undesignated) — will retry until it converges"
        )
        return False  # transient — Origin/operator may come up; self-heal retries

    ns = get_instance_namespace()
    op_id = derive_uuid(ns, _NS, _OPERATOR_SLUG)
    authz_id = derive_uuid(ns, _NS, _AUTHORIZER_SLUG)
    cs_id = derive_uuid(ns, _NS, _CLIENT_SECRET_SLUG)
    rt_id = derive_uuid(ns, _NS, _REFRESH_TOKEN_SLUG)
    for slug, uid in ((_OPERATOR_SLUG, op_id), (_AUTHORIZER_SLUG, authz_id),
                      (_CLIENT_SECRET_SLUG, cs_id), (_REFRESH_TOKEN_SLUG, rt_id)):
        register_id(slug, uid)

    try:
        # 1. Secret artifacts (metadata) + their material (always re-injected so
        #    rotated creds self-heal). Material lives in the operator's vault,
        #    keyed by the secret artifact id — fetch_secret_material resolves it.
        _ensure_artifact(
            db, cs_id, content_type=_SECRET_CT, name="Platform Email — Gmail client secret",
            description="OAuth2 client secret for the platform Gmail sender.",
            context={"content_type": _SECRET_CT, "type": "oauth_client_secret", "provider": "google"},
            content="", owner=operator,
        )
        _ensure_artifact(
            db, rt_id, content_type=_SECRET_CT, name="Platform Email — Gmail refresh token",
            description="OAuth2 refresh token for the platform Gmail sender.",
            context={"content_type": _SECRET_CT, "type": "oauth_refresh_token", "provider": "google"},
            content="", owner=operator,
        )
        secrets_service.set_secret_material(
            db, operator, secret_id=cs_id, value=client_secret,
            secret_type="oauth_client_secret", provider="google",
            label="Platform Email — Gmail client secret",
        )
        secrets_service.set_secret_material(
            db, operator, secret_id=rt_id, value=refresh_token,
            secret_type="oauth_refresh_token", provider="google",
            label="Platform Email — Gmail refresh token",
        )

        # 2. Authorizer — config only; references the two secret artifacts by id.
        authz_config = {
            "provider": "google",
            "client_id": client_id,
            "token_endpoint": _GMAIL_TOKEN_ENDPOINT,
            "scopes": _GMAIL_SEND_SCOPE,
            "sender_address": sender,
            "client_secret_artifact_id": cs_id,
            "refresh_token_artifact_id": rt_id,
        }
        _ensure_artifact(
            db, authz_id, content_type=_AUTHORIZER_CT, name="Platform Email Authorizer",
            description="Gmail OAuth2 client for the platform email sender.",
            context={"content_type": _AUTHORIZER_CT, "provider": "google", "sender_address": sender},
            content=json.dumps(authz_config, separators=(",", ":")),
            owner=operator,
        )

        # 3. Operator — invoke → iris:send_email, authorizer attached by edge.
        op_context = {
            "content_type": _OPERATOR_CT,
            "title": "Platform Email Sender",
            "operator": {
                "server": "iris",
                "tool": "send_email",
                "arguments": {
                    "to": "$.body.params.to",
                    "subject": "$.body.params.subject",
                    "body_html": "$.body.params.body_html",
                    "authorizer_artifact_id": "@relationship.authorizer",
                },
            },
        }
        op_existed = _ensure_artifact(
            db, op_id, content_type=_OPERATOR_CT, name="Platform Email Sender",
            description="Send outbound email via the platform Gmail account. Invoke op/invoke with {params:{to,subject,body_html}}.",
            context=op_context, content="", owner=operator,
        )

        # 4. Typed edge operator --authorizer--> authorizer (resolved by
        #    @relationship.authorizer at invoke time). Idempotent at the db layer.
        if not op_existed:
            db_add_edge(db, op_id, authz_id, "a0", origin=False, relationship="authorizer")

        # 5. Grants: invoke the operator, read the authorizer + its secrets (so
        #    send_email resolves them server-side under the caller's delegation).
        #    Issued to BOTH the operator (direct human sends) AND the platform
        #    system principal (webhook/background sends act AS this principal).
        #    Both are `granted_by` the operator — that provenance is what roots
        #    the system principal's authority to a person.
        from mantle.services.peer_signing import get_system_principal_id

        grantees = [operator]
        system_principal = get_system_principal_id()
        if system_principal:
            grantees.append(system_principal)
            register_id("platform-system-principal", system_principal)

        for grantee in grantees:
            for resource, invoke in ((op_id, True), (authz_id, False), (cs_id, False), (rt_id, False)):
                db_upsert_user_collection_grant(
                    db, user_id=grantee, collection_id=resource, granted_by=operator,
                    can_read=True, can_invoke=invoke, name="Platform email sender",
                )

        logger.info("platform_email: provisioned email operator %s (sender=%s, owner=%s)",
                    op_id, sender or "(unset)", operator)
        return True
    except Exception:
        logger.warning("platform_email: provisioning failed (will retry)", exc_info=True)
        return False  # transient (e.g. Origin vault unreachable) — self-heal retries
