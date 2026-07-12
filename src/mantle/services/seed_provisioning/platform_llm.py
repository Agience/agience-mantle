"""Platform default LLM connection provisioner.

Provisions a platform-default ``vnd.agience.llm-connection+json`` artifact from
the deployment's LLM env (``LLM_PROVIDER`` / ``LLM_API_KEY`` + optional
``LLM_MODEL`` / ``LLM_ENDPOINT``) as a graph of canonical artifacts — the LLM
counterpart of the platform email sender (`platform_email.py`), provisioned the
same way:

    connection (vnd.agience.llm-connection+json)  "Platform LLM"
      └─ credentials_ref.secret_id → secret (vnd.agience.secret+json)  API key

The API key is custodied in Origin's vault (via ``set_secret_material``), NOT in
the connection or process config. At invoke time Verso resolves it server-side:
``credentials_ref.resolution == "platform_secret"`` → Verso obtains a
``platform-llm`` system delegation and fetches the secret's material via the
secret artifact's ``fetch`` op (the same proven path Iris uses for email
secrets). The key never returns to the invoking user.

Both artifacts are created_by + granted to the platform operator (a real user,
so the material lives in a real vault) AND the operator-rooted system principal
(so Verso can resolve the key for any user's chat turn without that user holding
read on the platform key). Deterministic ids (uuid5 by slug) so the graph is
stable, re-runs are idempotent, and Aria can address the connection without a
slug→id lookup. The only out-of-band input is the key MATERIAL (always
re-injected from env so it self-heals); structure is created once.

Idempotent, non-fatal, runs every startup. No-op until the operator is
designated and ``LLM_API_KEY`` is configured. Rotate the key by updating the env
and restarting (material is re-injected each boot).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from arango.database import StandardDatabase

from db.arango import (
    create_artifact as db_create_artifact,
    get_artifact as db_get_artifact,
    update_artifact as db_update_artifact,
    upsert_user_collection_grant as db_upsert_user_collection_grant,
)
from entities.artifact import Artifact as ArtifactEntity
from services import secrets_service
from services.platform_topology import register_id
from services.seed_provisioning.loader import derive_uuid, get_instance_namespace

logger = logging.getLogger(__name__)

_NS = "platform"
_CONNECTION_SLUG = "platform-llm-connection"
_SECRET_SLUG = "platform-llm-api-key"

_CONNECTION_CT = "application/vnd.agience.llm-connection+json"
_SECRET_CT = "application/vnd.agience.secret+json"

# Sensible per-provider default model when LLM_MODEL is unset. Keep current.
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "mistral": "mistral-small-latest",
}


def _operator_id(db: StandardDatabase) -> str:
    """The designated platform admin — created_by + grantee of the LLM graph."""
    from services.platform_settings_service import settings as platform_settings

    op = platform_settings.get("platform.operator_id")
    if op:
        return str(op)
    try:
        from clients.origin_client import get_origin_client
        return str(get_origin_client().get_operator_id() or "")
    except Exception:
        logger.debug("platform_llm: operator_id unavailable (non-fatal)", exc_info=True)
        return ""


def _ensure_artifact(
    db: StandardDatabase, artifact_id: str, *, content_type: str, name: str,
    description: str, context: dict, content: str, owner: str,
) -> bool:
    """Upsert a committed artifact, CONVERGING its content/context to the current
    values (so config like the model/provider updates when env changes). Returns
    True if it already existed."""
    now = datetime.now(timezone.utc).isoformat()
    ctx_json = json.dumps(context, separators=(",", ":"))
    existing = db_get_artifact(db, artifact_id)
    if existing is not None:
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


def ensure_platform_llm_connection(db: StandardDatabase) -> bool:
    """Provision the platform default LLM connection + inject key material.

    Idempotent and order-agnostic. Returns ``True`` when the work is done (or
    intentionally not needed: no ``LLM_API_KEY`` configured), and ``False`` only
    for a TRANSIENT prerequisite (operator/Origin not resolvable yet, or an
    error), so the caller can self-heal by retrying until it converges.
    """
    api_key = (os.getenv("LLM_API_KEY", "") or "").strip()
    if not api_key:
        logger.info("platform_llm: LLM_API_KEY not configured — skipping")
        return True  # config-driven; won't change without a restart, so don't retry

    provider = (os.getenv("LLM_PROVIDER", "anthropic") or "anthropic").strip().lower()
    model = (os.getenv("LLM_MODEL", "") or "").strip() or _DEFAULT_MODELS.get(provider, "")
    endpoint = (os.getenv("LLM_ENDPOINT", "") or "").strip()

    operator = _operator_id(db)
    if not operator:
        logger.info(
            "platform_llm: operator not yet resolvable (Origin unreachable or "
            "operator undesignated) — will retry until it converges"
        )
        return False  # transient — Origin/operator may come up; self-heal retries

    ns = get_instance_namespace()
    conn_id = derive_uuid(ns, _NS, _CONNECTION_SLUG)
    secret_id = derive_uuid(ns, _NS, _SECRET_SLUG)
    for slug, uid in ((_CONNECTION_SLUG, conn_id), (_SECRET_SLUG, secret_id)):
        register_id(slug, uid)

    try:
        # 1. Secret artifact (metadata) + its material (always re-injected so a
        #    rotated key self-heals). Material lives in the operator's vault,
        #    keyed by the secret artifact id — fetch_secret_material resolves it.
        _ensure_artifact(
            db, secret_id, content_type=_SECRET_CT, name="Platform LLM — API key",
            description=f"API key for the platform default LLM connection ({provider}).",
            context={"content_type": _SECRET_CT, "type": "llm_key", "provider": provider},
            content="", owner=operator,
        )
        secrets_service.set_secret_material(
            db, operator, secret_id=secret_id, value=api_key,
            secret_type="llm_key", provider=provider,
            label="Platform LLM — API key",
        )

        # 2. Connection — config in context; credentials_ref points at the secret
        #    artifact (resolution=platform_secret → Verso fetches it server-side
        #    under a platform-llm system delegation; never config-read, never
        #    returned to the user).
        conn_context = {
            "content_type": _CONNECTION_CT,
            "title": "Platform LLM",
            "provider": provider,
            "model": model,
            "is_platform_default": True,
            "tier": "free",
            "capabilities": {"chat": True},
            "credentials_ref": {
                "secret_id": secret_id,
                "secret_type": "llm_key",
                "provider": provider,
                "resolution": "platform_secret",
            },
        }
        if endpoint:
            conn_context["endpoint"] = endpoint
        _ensure_artifact(
            db, conn_id, content_type=_CONNECTION_CT, name="Platform LLM",
            description="Platform default LLM connection. Backs chat when no other "
                        "connection is supplied. Key custodied in the vault.",
            context=conn_context, content="", owner=operator,
        )

        # 3. Grants: read the secret + read/invoke the connection. Issued to BOTH
        #    the operator (direct use) AND the operator-rooted system principal
        #    (so Verso resolves the key for ANY user's chat turn — the user never
        #    holds read on the platform key). Both `granted_by` the operator —
        #    that provenance roots the system principal's authority to a person.
        from services.peer_signing import get_system_principal_id

        grantees = [operator]
        system_principal = get_system_principal_id()
        if system_principal:
            grantees.append(system_principal)
            register_id("platform-system-principal", system_principal)

        for grantee in grantees:
            db_upsert_user_collection_grant(
                db, user_id=grantee, collection_id=secret_id, granted_by=operator,
                can_read=True, can_invoke=False, name="Platform LLM key",
            )
            db_upsert_user_collection_grant(
                db, user_id=grantee, collection_id=conn_id, granted_by=operator,
                can_read=True, can_invoke=True, name="Platform LLM connection",
            )

        logger.info("platform_llm: provisioned LLM connection %s (provider=%s, model=%s, owner=%s)",
                    conn_id, provider, model or "(default)", operator)
        return True
    except Exception:
        logger.warning("platform_llm: provisioning failed (will retry)", exc_info=True)
        return False  # transient (e.g. Origin vault unreachable) — self-heal retries
