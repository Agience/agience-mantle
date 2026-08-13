"""The platform system principal — Mantle-as-a-service, not Mantle-as-a-store.

``services/acting_principal.py`` is on the embeddable distribution surface: an embedding
consumer installs Mantle as a standalone artifact/collection store with access grants, and
``db/doc_boundary.py`` imports ``KeyCustodyDenied`` from it. That surface
depends on stdlib + cryptography only.

``system_acting_context`` cannot honour that: it resolves the platform's system identity via
``services.peer_signing`` and the platform operator via ``services.operator`` — both belong to
the deployed service, not to the store. Keeping it in a separate module makes the boundary
structural: ``acting_principal`` is provably stdlib-only, and everything requiring the platform
lives here, behind the ``service`` extra (``pip install agience-mantle[service]``).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from .acting_principal import ActingPrincipal, SystemPrincipalUnavailable, acting_as

__all__ = ["system_acting_context"]


@contextmanager
def system_acting_context(
    *, scope: Optional[str] = None, db=None
) -> Iterator[ActingPrincipal]:
    """Run system-initiated work under the platform system principal.

    For the bulk reindex, the index-queue worker, collection auto-index on create,
    and importers — every write path that has no request context.

    The subject is the system principal; ``actor`` is the platform operator, to whom
    that principal's authority roots, and ``host_id`` says where it ran — the three
    §4B requires provenance to record.

    Raises :class:`SystemPrincipalUnavailable` when the instance namespace is not
    resolvable. Fails closed: no placeholder identity is substituted.
    """
    from mantle.services.peer_signing import get_host_id, get_system_principal_id

    principal_id = get_system_principal_id()
    if not principal_id:
        raise SystemPrincipalUnavailable(
            "instance namespace unavailable; cannot derive the platform system "
            "principal, so system-initiated work cannot be authorized"
        )

    # Best-effort context. Neither is an authorization input — they are recorded for
    # audit — so an unresolvable operator or host must not block system work that
    # already has a resolved subject.
    operator_id = None
    try:
        from mantle.services.operator import resolve_operator_id

        operator_id = resolve_operator_id(db) or None
    except Exception:  # pragma: no cover - diagnostics only
        operator_id = None

    try:
        host_id = get_host_id() or None
    except Exception:  # pragma: no cover - diagnostics only
        host_id = None

    with acting_as(
        principal_id,
        principal_type="service",
        actor=operator_id,
        host_id=host_id,
        scope=scope,
        source="system",
    ) as principal:
        yield principal
