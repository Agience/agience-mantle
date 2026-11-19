"""What the caller's credential is.

``GET /credentials/self`` answers who an artifact token (`services.artifact_tokens`) speaks for:
the person, the agent acting (none for the person's own key), the ids it reaches, what it may do
there, when it ends and the Key artifact that holds it. Origin's token exchange reads it to verify an agent's key as the actor
token; an agent reads it to know its own bounds. Any other credential is 404: it is not an
artifact's token, and its bounds are read where it was issued.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException

from mantle.api.errors import ERROR_DESCRIPTIONS
from mantle.services.dependencies import AuthContext, get_auth

router = APIRouter(prefix="/credentials", tags=["Credentials"])


@router.get("/self", summary="What this artifact token is",
            responses={c: {"description": ERROR_DESCRIPTIONS[c]} for c in (401, 404)})
async def credential_self(auth: AuthContext = Depends(get_auth)) -> Dict[str, Any]:
    """The bounds of the artifact token the caller presented; 404 for any other credential."""
    if not auth.token_artifact:
        raise HTTPException(status_code=404, detail="this credential is not an artifact's token")
    return {"person": auth.user_id, "actor": auth.actor, "artifact": auth.token_artifact,
            "reach": list(auth.job_reach or ()), "actions": sorted(auth.reach_actions or ()),
            "ends": auth.token_ends, "scope": auth.scope}
