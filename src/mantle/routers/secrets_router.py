"""
Secrets Router -- generic encrypted credential management.

Endpoints:
    GET    /secrets             List stored secrets (filter by ?type= and/or ?provider=)
    POST   /secrets             Store a new secret
    DELETE /secrets/{id}        Delete a secret
    POST   /secrets/{id}/set-default  Mark a secret as the default for its (type, provider)
    POST   /secrets/fetch            Return a secret wrapped as JWE for a delegated server

Secret types:
    llm_key         -- BYOK keys for Anthropic, Google, Azure, etc.
    github_token    -- GitHub OAuth / PAT for Copilot and GitHub API access
    integration_key -- Any other third-party integration credential
"""

import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel
from mantle.db.store import Database

from mantle.services.dependencies import get_store_db
from mantle.services.dependencies import get_auth, AuthContext
from mantle.services import secrets_service, auth_service
from mantle.db.backend import (
    get_server_jwk,
    create_artifact as db_create_artifact,
    get_artifact as db_get_artifact,
    list_committed_artifacts_by_context_content_type,
)
from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.api.secrets import SecretResponse, SecretFetchRequest, SecretFetchResponse, SecretCreateRequest

router = APIRouter(prefix="/secrets", tags=["Secrets"])

# A secret lives in the artifact model like everything else: a `vnd.agience.secret+json`
# artifact owned by the user (metadata only) whose material is held in Mantle's own encrypted
# store (Fernet), keyed by the artifact id. Creation + delegated reveal below are BOTH
# Mantle-native — no Chorus/crystal op-dispatcher, no Seraph type handler in the path.
SECRET_CT = "application/vnd.agience.secret+json"

# WHO THIS SERVICE IS, for the `aud` check on delegated reveals. It matches the `audience` Origin
# stamps in its system-delegation purpose table (`origin/routers/auth_router.py`), which is the one
# place that decides who a token is minted FOR. Named here rather than inlined so the two ends are
# greppable together — they are a contract, and a contract with only one end written down is how
# `aud` came to be compared against the presenter instead of the recipient.
_SELF_AUDIENCE = "mantle"


def _to_response(s: secrets_service.SecretConfig, include_encrypted: bool = False) -> SecretResponse:
    resp = SecretResponse(
        id=s.id,
        type=s.type,
        provider=s.provider,
        label=s.label,
        created_time=s.created_time,
        is_default=s.is_default,
        authorizer_id=s.authorizer_id or None,
        expires_at=s.expires_at or None,
    )
    if include_encrypted:
        resp.encrypted_value = s.encrypted_value
    return resp


@router.get("", response_model=List[SecretResponse])
def list_secrets(
    type: Optional[str] = Query(None, description="Filter by secret type"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    id: Optional[str] = Query(None, description="Filter by exact secret ID"),
    authorizer_id: Optional[str] = Query(None, description="Filter by authorizer artifact ID"),
    include_encrypted: bool = Query(False, description="Include encrypted_value in response"),
    db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """List stored secrets. Encrypted values only returned when include_encrypted=True."""
    secrets = secrets_service.list_secrets(
        db, auth.user_id,
        secret_type=type,
        provider=provider,
        secret_id=id,
        authorizer_id=authorizer_id,
    )
    return [_to_response(s, include_encrypted=include_encrypted) for s in secrets]


@router.post("", response_model=List[SecretResponse], status_code=status.HTTP_201_CREATED)
def add_secret(
    payload: SecretCreateRequest,
    db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Store a new secret (value encrypted before storage). Returns all secrets."""
    secrets = secrets_service.add_secret(
        db,
        auth.user_id,
        secret_type=payload.type,
        provider=payload.provider,
        label=payload.label,
        value=payload.value,
        is_default=payload.is_default,
        authorizer_id=payload.authorizer_id or "",
        expires_at=payload.expires_at or "",
    )
    return [_to_response(s) for s in secrets]


@router.delete("/{secret_id}", response_model=List[SecretResponse])
def delete_secret(
    secret_id: str,
    db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Delete a stored secret. Returns remaining secrets."""
    secrets = secrets_service.delete_secret(db, auth.user_id, secret_id)
    return [_to_response(s) for s in secrets]


@router.post("/{secret_id}/set-default", response_model=List[SecretResponse])
def set_default_secret(
    secret_id: str,
    db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Mark a secret as the default for its (type, provider) combination."""
    secrets = secrets_service.set_default_secret(db, auth.user_id, secret_id)
    return [_to_response(s) for s in secrets]


@router.post("/fetch", response_model=SecretFetchResponse)
def fetch_secret_for_server(
    payload: SecretFetchRequest,
    request: Request,
    db: Database = Depends(get_store_db),
):
    """Return a secret wrapped as JWE for the requesting server.

    Auth: The caller must present a **delegation JWT** (RFC 8693) issued by
    Core.  ``sub`` is the user whose secret is being fetched; ``act.sub`` is
    the ``client_id`` of the server requesting the secret.  The JWE envelope
    is encrypted with the server's registered RSA public key so only that
    server can decrypt it.  Plaintext never leaves Core unencrypted.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = auth_header[7:].strip()
    claims = auth_service.verify_token(token)
    if not claims or claims.get("principal_type") != "delegation":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Delegation JWT required")

    user_id: str = claims["sub"]
    server_client_id: str = claims.get("act", {}).get("sub", "")
    if not server_client_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="act.sub missing from delegation JWT")

    # The aud claim must match act.sub — proving this token was issued TO the
    # server that is presenting it, not re-used from a different delegation.
    if claims.get("aud") != server_client_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token audience does not match presenter")

    public_jwk = get_server_jwk(db, server_client_id)
    if not public_jwk:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No public key registered for server")

    secrets = secrets_service.list_secrets(
        db, user_id,
        secret_type=payload.type,
        provider=payload.provider,
        secret_id=payload.secret_id,
        authorizer_id=payload.authorizer_id,
    )
    if not secrets:
        raise HTTPException(status_code=404, detail="Secret not found")

    secret = secrets[0]
    plaintext = secrets_service.decrypt_value(secret.encrypted_value)
    jwe = secrets_service.wrap_secret_for_server(plaintext, public_jwk)

    return SecretFetchResponse(id=secret.id, type=secret.type, jwe=jwe)


# ---------------------------------------------------------------------------
# Secret-as-artifact: create + delegated reveal (Mantle-native, Chorus-free)
# ---------------------------------------------------------------------------
class SecretArtifactCreateRequest(BaseModel):
    """Store a user's secret (e.g. a BYOK Anthropic key) as an owned secret ARTIFACT."""
    value: str
    provider: str = ""
    type: str = "llm_key"
    label: str = ""


class SecretArtifactResponse(BaseModel):
    secret_id: str
    content_type: str = SECRET_CT
    provider: str = ""
    type: str = "llm_key"


@router.post("/artifact", response_model=SecretArtifactResponse,
             status_code=status.HTTP_201_CREATED)
def create_secret_artifact(
    payload: SecretArtifactCreateRequest,
    db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """Create a ``vnd.agience.secret+json`` artifact owned by the caller and stash its material
    in Mantle's encrypted vault (keyed by the artifact id). The key is NEVER written to the
    artifact's ``content`` — only metadata (type/provider) lives there. Mirrors the platform-LLM
    provisioner's secret pattern, but user-owned. Everything is an artifact, secrets included."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    if not payload.value:
        raise HTTPException(status_code=400, detail="secret value is required")

    provider = (payload.provider or "").strip().lower()
    secret_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    context = {"content_type": SECRET_CT, "type": payload.type, "provider": provider}
    db_create_artifact(db, ArtifactEntity(
        id=secret_id, root_id=secret_id,
        state=ArtifactEntity.STATE_COMMITTED,
        context=json.dumps(context, separators=(",", ":")),
        content="",
        content_type=SECRET_CT,
        name=payload.label or f"{provider or 'API'} key",
        description="User-owned secret (BYOK). Material custodied in Mantle's vault.",
        created_by=auth.user_id, created_time=now,
    ))
    secrets_service.set_secret_material(
        db, auth.user_id, secret_id=secret_id, value=payload.value,
        secret_type=payload.type, provider=provider, label=payload.label,
    )
    return SecretArtifactResponse(secret_id=secret_id, provider=provider, type=payload.type)


class SecretArtifactListItem(BaseModel):
    id: str
    provider: str = ""
    type: str = "llm_key"
    label: str = ""
    created_time: str = ""


@router.get("/artifact", response_model=List[SecretArtifactListItem])
def list_secret_artifacts(
    type: Optional[str] = Query(None, description="Filter by secret type"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    db: Database = Depends(get_store_db),
    auth: AuthContext = Depends(get_auth),
):
    """List the caller's own secret ARTIFACTS — metadata only, never material. Backs the Facet
    Keys UI. Material is revealed only to a delegated service via /secrets/reveal."""
    if not auth.user_id:
        raise HTTPException(status_code=401, detail="User identification required")
    out: List[SecretArtifactListItem] = []
    for a in list_committed_artifacts_by_context_content_type(db, SECRET_CT, created_by=auth.user_id):
        try:
            ctx = json.loads(a.context) if isinstance(a.context, str) and a.context else (a.context or {})
        except (ValueError, TypeError):
            ctx = {}
        a_type = ctx.get("type", "llm_key")
        a_provider = (ctx.get("provider") or "")
        if type and a_type != type:
            continue
        if provider and a_provider.lower() != provider.lower():
            continue
        out.append(SecretArtifactListItem(
            id=a.id, provider=a_provider, type=a_type,
            label=a.name or "", created_time=a.created_time or ""))
    out.sort(key=lambda s: s.created_time, reverse=True)
    return out


class SecretRevealRequest(BaseModel):
    """Reveal the material of a secret ARTIFACT the delegating user owns. Supply either a
    concrete ``secret_id`` or a ``(type, provider)`` selector (newest match wins)."""
    secret_id: str = ""
    provider: str = ""
    type: str = "llm_key"


class SecretRevealResponse(BaseModel):
    secret_id: str
    material: str


@router.post("/reveal", response_model=SecretRevealResponse)
def reveal_secret_artifact(
    payload: SecretRevealRequest,
    request: Request,
    db: Database = Depends(get_store_db),
):
    """Return a secret artifact's material to a **delegated platform service** acting for the
    owning user. Auth: a delegation JWT (RFC 8693) — ``sub`` = the user, ``act.sub`` = the
    service. We only ever reveal a secret the delegating user OWNS (``created_by == sub``); the
    delegation proves the service is acting for that user, and only recognized platform services
    can obtain one from Origin. Mantle-native: no JWE, no server-JWK, no Chorus op-dispatch."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth_header[7:].strip()
    claims = auth_service.verify_token(token)
    if not claims or claims.get("principal_type") != "delegation":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Delegation JWT required")

    user_id: str = claims.get("sub", "")
    server_client_id: str = claims.get("act", {}).get("sub", "")
    if not user_id or not server_client_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="delegation missing sub or act.sub")
    # ── `aud` NAMES THE RECIPIENT (RFC 8693 / RFC 7519 §4.1.3) ───────────────────────────────────
    # ⛔ THIS READ `claims.get("aud") != server_client_id` — the audience had to equal the PRESENTING
    # SERVICE. That inverts what `aud` means: it is the party the token was minted FOR, and this
    # route is that party. Origin stamps `aud` from its purpose table (`audience: "mantle"`), so the
    # two could never agree and **every delegated reveal returned 403** — which is why three
    # registered seraph tools and ophan's Stripe secret resolution all sat on a path that could not
    # succeed. Fixed per John, 2026-07-30: "follow RFC".
    #
    # The replay protection the old line was reaching for is still here, just expressed correctly:
    # `aud` proves the token was issued for MANTLE (so a token minted for another service cannot be
    # replayed here), and `act.sub` names the acting service, which the ownership check below binds
    # to the delegating user. Overloading one claim to do both jobs is what made it do neither.
    audience = claims.get("aud")
    accepted = {a.strip() for a in (audience if isinstance(audience, list) else [audience or ""])}
    if _SELF_AUDIENCE not in accepted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="token audience %r was not minted for this service (%r)"
                   % (audience, _SELF_AUDIENCE))

    provider = (payload.provider or "").strip().lower()

    # Resolve which secret artifact — by id, or by (type, provider) among the user's own secrets.
    if payload.secret_id:
        art = db_get_artifact(db, payload.secret_id)
        if art is None or art.created_by != user_id or art.content_type != SECRET_CT:
            raise HTTPException(status_code=404, detail="secret not found")
        secret_id = art.id
    else:
        candidates = list_committed_artifacts_by_context_content_type(
            db, SECRET_CT, created_by=user_id)
        chosen = None
        for a in candidates:
            try:
                ctx = json.loads(a.context) if isinstance(a.context, str) and a.context else (a.context or {})
            except (ValueError, TypeError):
                ctx = {}
            if payload.type and ctx.get("type") != payload.type:
                continue
            if provider and (ctx.get("provider") or "").lower() != provider:
                continue
            if chosen is None or (a.created_time or "") > (chosen.created_time or ""):
                chosen = a           # newest match wins
        if chosen is None:
            raise HTTPException(status_code=404, detail="no matching secret for this user")
        secret_id = chosen.id

    material = secrets_service.get_secret_material_by_id(secret_id)
    if material is None:
        raise HTTPException(status_code=404, detail="secret has no stored material")
    return SecretRevealResponse(secret_id=secret_id, material=material)
