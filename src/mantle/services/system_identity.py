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

import contextvars
import logging
from contextlib import ExitStack, contextmanager
from typing import Iterator, Optional

from .acting_principal import ActingPrincipal, SystemPrincipalUnavailable, acting_as

__all__ = ["system_acting_context", "seed_pass"]

logger = logging.getLogger(__name__)

#: The seed credential of the pass in progress (`seed_pass`), shared by every seed scope in it.
_pass_credential: contextvars.ContextVar = contextvars.ContextVar("seed_pass_credential",
                                                                  default=None)


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

    # A seed pass that grants runs on the operator's seed credential, minted at Origin for that
    # pass (`POST /oracle/seed-credential`) and handed to it in `MANTLE_SEED_CREDENTIAL_FILE`. That
    # is a deliberate act: background work, with no such credential, may only narrow reach.
    # One credential serves one pass: a seed scope outside a `seed_pass` is a pass of its own.
    with ExitStack() as stack:
        seed = None
        if scope and scope.endswith("-seed"):
            seed = _pass_credential.get() or stack.enter_context(seed_pass())
        with acting_as(
            principal_id,
            principal_type="service",
            actor=operator_id,
            host_id=host_id,
            scope=scope,
            source="seed" if seed else "system",
            credential=seed,
        ) as principal:
            yield principal


SEED_CREDENTIAL_SETTING = "MANTLE_SEED_CREDENTIAL_FILE"


@contextmanager
def seed_pass(custodian=None):
    """One seed pass. The operator's seed credential is read once from
    `MANTLE_SEED_CREDENTIAL_FILE` and used by every seed scope inside; when the pass ends, done or
    failed, Origin is told (`seed-done`, which revokes the credential) and the file is deleted. A
    credential therefore serves one pass, and nothing of it is left on disk after."""
    import os

    if _pass_credential.get() is not None:              # already inside a pass: the same one
        yield _pass_credential.get()
        return
    path = (os.getenv(SEED_CREDENTIAL_SETTING) or "").strip()
    cred = seed_credential()
    if cred is None:
        yield None
        return
    token = _pass_credential.set(cred)
    # The pass's plan (review 26, open item c): the operator planned this pass
    # (`python -m mantle.system.seed_plan`) and minted the credential with the plan's digests;
    # with `MANTLE_SEED_PLAN_FILE` the pass derives its new grant ids from the plan and submits
    # only planned entries. A plan that cannot be read stops the pass before anything is made.
    from mantle.services import seed_plan as _plan
    planned = ExitStack()
    try:
        plan = _plan.plan_from_env()
        if plan is not None:
            planned.enter_context(_plan.enforcing(plan))
        with planned:
            yield cred
    finally:
        _pass_credential.reset(token)
        try:
            if custodian is None:                       # the ledger's own Origin client
                from mantle.db import reach_journal
                custodian = getattr(reach_journal._authority, "custodian", None)
            if custodian is None:
                from mantle.clients.origin_key_share_client import OriginShareCustodian
                custodian = OriginShareCustodian()
            custodian.seed_done(cred.token)
        except Exception:  # noqa: BLE001 — it still expires within ten minutes
            logger.warning("seed pass: Origin was not told the pass is over; the seed credential "
                           "lapses at its expiry", exc_info=True)
        try:
            os.remove(path)
        except OSError:
            logger.warning("seed pass: could not delete %s", path, exc_info=True)


def seed_credential():
    """The operator's seed credential for this pass, from `MANTLE_SEED_CREDENTIAL_FILE`, or None.
    Read for the pass and kept nowhere else."""
    import os

    from mantle.services.acting_principal import ForwardedCredential

    path = (os.getenv(SEED_CREDENTIAL_SETTING) or "").strip()
    if not path or not os.path.exists(path):        # none handed over, or its pass is over
        return None
    with open(path, encoding="utf-8") as fh:
        token = fh.read().strip()
    return ForwardedCredential(token) if token else None
