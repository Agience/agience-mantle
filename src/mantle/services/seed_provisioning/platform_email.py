"""Platform outbound-email provisioner.

Provisions the composable platform email sender from the deployment's Gmail OAuth
env (``EMAIL_PROVIDER=gmail`` + ``GMAIL_OAUTH_CLIENT_ID`` / ``_CLIENT_SECRET`` /
``_REFRESH_TOKEN`` + ``PLATFORM_EMAIL_ADDRESS``) as a graph of canonical artifacts:

    operator  (vnd.agience.operator+json)  "Platform Email Sender"
      └─(rel: authorizer)→ authorizer (vnd.agience.authorizer+json)  Gmail OAuth client
                              ├─ client_secret_artifact_id → credential (vnd.agience.credential+json)
                              └─ refresh_token_artifact_id → credential (vnd.agience.credential+json)

A credential is an ordinary artifact and nothing more. Its value is the artifact's
``content``, so the write goes through the same envelope every other artifact's content
does (``db/doc_boundary.encrypt_artifact_content`` → ``services/content_crypto``:
AES-256-GCM under a key rooted at the artifact's ``origin_root``, which for a top-level
artifact is its own id, with the (principal, scope) pair bound in as AAD). So each
credential gets its own key, exactly as wide as the grant it came from — one leaked grant
reaches one credential. Reading it back is the ordinary artifact read,
and the light cone is the only thing that decides who may do it — there is no second
store, no second cipher, and no bespoke check. ``context`` is plaintext, so it carries the
label/provider/kind and nothing else.

Invoking the operator runs ``iris:send_email`` with the authorizer attached by
edge; ``send_email`` reads the authorizer config, reads the two credential artifacts
under the caller's delegation (for platform sends that is the operator, who holds read
on them), performs the OAuth exchange, and sends. The credentials are never returned to
the caller. There is NO token-minting op on any user-facing artifact (the banned
``invoke → provide_access_token`` is not used).

All artifacts are created_by + granted to the platform operator. Deterministic ids (uuid5
by slug) so the graph is stable and re-runs are idempotent. The credential VALUE is
re-injected from env on every run so rotated creds self-heal; structure is created once.
Types stay declarative; this provisions the one platform instance.

Idempotent, non-fatal, runs every startup. No-op until the operator is designated
and ``EMAIL_PROVIDER=gmail`` with creds is configured. Rotate creds by updating
the env and restarting (the value is re-written each boot).
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
from mantle.services.acting_principal import acting_as
from mantle.services.bootstrap_types import CREDENTIAL_CONTENT_TYPE
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
_CREDENTIAL_CT = CREDENTIAL_CONTENT_TYPE

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


def _credential_content(value: str) -> str:
    """The credential artifact's body — a JSON object, matching the ``+json`` suffix.

    An object rather than the bare token so the shape can carry a second field later
    without every reader having to tell the two encodings apart. This string becomes
    ``Artifact.content``, which the write boundary encrypts; nothing else in the artifact
    holds the value.
    """
    return json.dumps({"value": value}, separators=(",", ":"))


def ensure_platform_email_sender(db: Database) -> bool:
    """Provision the platform email operator graph + write the credential values.

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

    # Every write below is made AS the operator, because the operator is the artifacts'
    # `created_by` and therefore the principal their content keys root at. Encrypting an
    # artifact's content requires the writer to hold that key, and the creator holds it for
    # the artifact it is creating (`oracle._authorize`'s create base case) — so acting as
    # anyone else here would be asking for a key nobody has yet issued a grant for. The id
    # is resolved server-side by `_operator_id`, never supplied by a caller.
    try:
        with acting_as(operator, principal_type="user", scope="platform.email-provisioning"):
            _provision(db, operator, op_id, authz_id, cs_id, rt_id,
                       client_id=client_id, client_secret=client_secret,
                       refresh_token=refresh_token, sender=sender)
        logger.info("platform_email: provisioned email operator %s (sender=%s, owner=%s)",
                    op_id, sender or "(unset)", operator)
        return True
    except Exception:
        logger.warning("platform_email: provisioning failed (will retry)", exc_info=True)
        return False  # transient (e.g. the key oracle not yet up) — self-heal retries


def _provision(
    db: Database, operator: str, op_id: str, authz_id: str, cs_id: str, rt_id: str,
    *, client_id: str, client_secret: str, refresh_token: str, sender: str,
) -> None:
    """Write the graph. Runs inside the operator's acting context (see the caller)."""
    # 1. Credential artifacts. The value IS the content, so the envelope encrypts it
    #    at rest; it is re-written every run so rotated creds self-heal. `context` is
    #    plaintext and carries only what is safe to read without the grant.
    _ensure_artifact(
        db, cs_id, content_type=_CREDENTIAL_CT, name="Platform Email — Gmail client secret",
        description="OAuth2 client secret for the platform Gmail sender.",
        context={"content_type": _CREDENTIAL_CT, "kind": "oauth_client_secret",
                 "provider": "google", "label": "Platform Email — Gmail client secret",
                 "is_default": True},
        content=_credential_content(client_secret), owner=operator,
    )
    _ensure_artifact(
        db, rt_id, content_type=_CREDENTIAL_CT, name="Platform Email — Gmail refresh token",
        description="OAuth2 refresh token for the platform Gmail sender.",
        context={"content_type": _CREDENTIAL_CT, "kind": "oauth_refresh_token",
                 "provider": "google", "label": "Platform Email — Gmail refresh token",
                 "is_default": True},
        content=_credential_content(refresh_token), owner=operator,
    )

    # 2. Authorizer — config only; references the two credential artifacts by id.
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

    # 5. Grants: invoke the operator, read the authorizer + its credentials. Read on a
    #    credential is the WHOLE authorization story — the light cone gates the artifact
    #    read and the content key alike, so there is nothing else to check.
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
