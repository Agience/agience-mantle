"""A shared credential is the persona's own resource (John, 2026-11-01).

A credential's secret is sealed, when it is written, to the enrolled key of each persona that uses
secrets (`USING_PERSONAS`), bound to the credential's id, and kept beside it (`persona_sealed`).
At use, a caller holding Invoke asks the persona to act: Mantle hands that persona the secret
sealed to it, without opening it. Neither the caller nor Mantle's request path ever holds the
credential's content key, and no action but Read releases one (architecture §51).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: The personas that use secrets, as `artifacts_router._SECRET_USES` names them.
USING_PERSONAS = ("lumen", "iris")


def _jwk(persona: str) -> Optional[dict]:
    from prism.trust.authority_trust import get_authority_manifest

    try:
        return (get_authority_manifest().get_jwks(persona).get("keys") or [])[0]
    except Exception:  # noqa: BLE001 — not enrolled: nothing to seal to
        return None


def _seal_custodian():
    """The Origin client that keeps a credential's seal digests: the ledger's own when it is
    installed, else one on `ORIGIN_URI`, else None."""
    from mantle.routers.artifacts_router import _use_verdict_custodian
    return _use_verdict_custodian()


def record_at_origin(artifact_id: str, sealed: Optional[Dict[str, dict]],
                     endpoints: Optional[list]) -> bool:
    """Send Origin, on the writer's own credential (the request's), the SHA-256 of each persona's
    sealed copy and of the endpoint list (review 22, F2): every use verdict carries the persona's,
    and the persona opens only the copy whose digest it names. Without Origin, or without the
    writer's credential, nothing is recorded, and the credential gets no verdict until its Admin
    writes it again. Never raises: a write is not undone by Origin's answer."""
    from prism.trust import use_verdict

    from mantle.services.acting_principal import current_acting_principal

    if not sealed:
        return False
    acting = current_acting_principal()
    held = getattr(getattr(acting, "credential", None), "token", "") if acting else ""
    custodian = _seal_custodian()
    if not held or custodian is None or not hasattr(custodian, "record_credential_seals"):
        logger.warning("credential %s: its seals are not recorded at Origin (no Origin, or no "
                       "writer's credential); it gets no use verdict until written again",
                       artifact_id)
        return False
    try:
        custodian.record_credential_seals(
            artifact_id, {p: use_verdict.sealed_digest(c) for p, c in sealed.items()},
            use_verdict.endpoints_digest(list(endpoints or [])), credential=held)
    except Exception as exc:  # noqa: BLE001
        logger.error("credential %s: Origin did not record its seals (%s); it gets no use "
                     "verdict until written again", artifact_id, exc)
        return False
    return True


def seal_for_personas(artifact_id: str, secret: str) -> Dict[str, dict]:
    """The secret sealed to each enrolled persona's key, bound to this credential."""
    from prism.trust import sealing

    out: Dict[str, dict] = {}
    for persona in USING_PERSONAS:
        jwk = _jwk(persona)
        if jwk is None:
            continue
        kid = str(jwk.get("kid") or f"{persona}-1")
        out[persona] = sealing.seal(secret.encode("utf-8"), sealing.public_key_from_jwk(jwk),
                                    kid=kid, context=str(artifact_id))
    if not out:
        logger.warning("credential %s: no persona has an enrolled key; it cannot be used",
                       artifact_id)
    return out
