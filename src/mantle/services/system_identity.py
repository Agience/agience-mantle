"""The PLATFORM system principal — Mantle-as-a-service, not Mantle-as-a-store.

⛔ WHY THIS IS ITS OWN MODULE, SPLIT OUT OF ``acting_principal``.

``services/acting_principal.py`` is on the EMBEDDABLE distribution surface: an embedding
consumer (EREA is the first) installs Mantle as a standalone artifact/collection store with
access grants, and ``db/doc_boundary.py`` imports ``KeyCustodyDenied`` from it. That surface
must depend on **stdlib + cryptography only**.

``system_acting_context`` cannot honour that. It resolves the platform's system identity via
``origin.service_identity`` and the platform operator via ``services.operator`` — both belong to
the *deployed service*, not to the store. The import was function-local, so it never fired for an
embedding consumer, but a latent cross-package import inside a shipped file is a trap: it makes
the package's real dependency set a property of which code paths happen to run.

Splitting it makes the boundary structural instead of incidental. ``acting_principal`` is now
provably stdlib-only, and everything requiring the platform lives here, behind the ``service``
extra (``pip install agience-mantle[service]``).

⚠ THE EXCEPTION TYPES STAY IN ``acting_principal``. ``SystemPrincipalUnavailable`` subclasses
``NoActingPrincipal`` subclasses ``KeyCustodyDenied``, and callers catch the base structurally
rather than by enumeration. Splitting the hierarchy across modules to match the dependency split
would break ``except KeyCustodyDenied`` for anyone importing only one half. The exceptions are
pure stdlib, so they cost the embeddable surface nothing.
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

    ⚠ THIS IS A PRINCIPAL, NOT A BYPASS. It resolves the SAME
    ``get_system_principal_id()`` that Origin uses as the ``sub`` of a system
    delegation and that Mantle already issues issuer-artifact grants to, so system
    work is checked against the same light cone as everyone else and must hold real
    grants on what it touches. A trusted-caller flag is exactly what
    ``KeyPurpose.SELF`` decayed into; this is deliberately not one.

    The subject is the system principal; ``actor`` is the platform operator, to whom
    that principal's authority roots, and ``host_id`` says where it ran — the three
    §4B requires provenance to record.

    Raises :class:`SystemPrincipalUnavailable` when the instance namespace is not
    resolvable. Fails closed: no placeholder identity is substituted.
    """
    try:
        from prism.trust.service_identity import get_host_id, get_system_principal_id
    except ImportError as e:                       # pragma: no cover - install-shape diagnostic
        raise SystemPrincipalUnavailable(
            "the platform system principal needs `origin` (agience-origin), which is not "
            "installed. This is the SERVICE surface, not the embeddable store: "
            "`pip install agience-mantle[service]`. The store itself (artifacts, collections, "
            "grants, content) never needs it — if you reached this from an embedding "
            "consumer, something called a platform code path by mistake. (%s)" % e
        ) from e

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
