"""The acting principal — the delegation chain, carried below the router.

Key issuance is grant-gated in ``search/mantle/oracle.py``, and this is how the oracle
learns who is asking.

Identity is delegated (§4B)::

    subject ──delegates──▶ actor (operator/server) ──runs on──▶ host
                                    │
                                    └──▶ authority (issuer) vouches for all of it

That is the shape of Origin's delegation JWT — ``sub`` / ``act.sub`` / ``host_id`` /
``iss`` (``origin/services/auth_service.py:97-133``) — which ``resolve_auth`` already
unpacks into ``AuthContext`` (``services/dependencies.py:311-318``). Nothing new is
invented; the existing chain is carried the rest of the way down.

Fail closed: this is deliberately not a token — contrast a contextvar that carries a
raw bearer, as an application tier typically does. It holds an already-verified chain, so
nothing below the router re-parses or re-trusts a credential. Verification stays at
the boundary; this is only the result of it.

Usage: on the request path, the auth dependency sets it — nothing else to do.

On a background / system path, declare it explicitly at the entry point::

    from mantle.services.system_identity import system_acting_context

    with system_acting_context(scope="platform.reindex"):
        reindex_everything()

``acting_as`` / ``system_acting_context`` are context managers that restore the
previous value in a ``finally``, so they nest correctly and cannot strand an identity
on the thread or task that runs next.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

__all__ = [
    "ActingPrincipal",
    "KeyCustodyDenied",
    "NoActingPrincipal",
    "SystemPrincipalUnavailable",
    "acting_as",
    "acting_from_auth",
    "current_acting_principal",
    "propagate",
    "require_acting_principal",
    "set_acting_principal",
    "reset_acting_principal",
]
# `system_acting_context` lives in `services.system_identity`, not here, so this
# module stays stdlib-only for the embeddable store surface. See `__getattr__` at the bottom.


class KeyCustodyDenied(PermissionError):
    """Base for every refusal to issue key material. ``GrantDenied`` subclasses it.

        except (KeyCustodyDenied, SystemicKeyFailure):
            raise                       # "not authorized" must not become "no results"
        except Exception:
            ...                         # degrade to SSE-only

    A shared base means any new refusal type is caught by the first clause
    automatically, without a hand-listed tuple that a new subclass could silently
    fall through into the second — turning "not authorized" into a quietly narrower
    result set.

    Deliberately narrower than ``PermissionError`` itself: catching that would also
    trap an unrelated OS/storage permission error and convert a degradable fault
    into a hard search failure.
    """


class NoActingPrincipal(KeyCustodyDenied):
    """No authenticated caller is in scope, so no key may be issued.

    A ``PermissionError`` rather than a ``LookupError`` because that is what it
    means: not "a value was missing" but "this call is unauthenticated". Callers
    that already map ``PermissionError`` to 403 get the right behaviour for free.
    """


class SystemPrincipalUnavailable(NoActingPrincipal):
    """The platform system principal could not be derived, so system work cannot run.

    Subclasses :class:`NoActingPrincipal` so every fail-closed handler catches it.
    Raised rather than substituting a placeholder id: a fabricated system identity
    would acquire whatever grants happened to match it, which is the failure mode
    ``issuers.py`` already fails closed against.
    """


@dataclass(frozen=True)
class ActingPrincipal:
    """The verified delegation chain in whose name the current work is being done.

    Field names mirror Origin's delegation claims so the mapping stays obvious:
    ``principal_id``←``sub``, ``actor``←``act.sub``, ``host_id``←``host_id``,
    ``authority``←``iss``.
    """

    #: ``sub`` — the principal the work is done ON BEHALF OF. This is what the grant
    #: verifier resolves a light cone for. For system work it is the platform system
    #: principal, whose authority roots to the operator.
    principal_id: str

    #: ``user | server | mcp_client | grant_key | service | delegation``.
    principal_type: str = "user"

    #: ``act.sub`` — the operator/server actually running, when this is a delegated
    #: call. ``None`` for a direct call. Carried for provenance and audit (§4B); it
    #: is NOT consulted for the authorization decision, which is made against the
    #: subject's grants.
    actor: Optional[str] = None

    #: ``host_id`` — where it ran.
    host_id: Optional[str] = None

    #: ``iss`` — the authority that vouched for the chain.
    authority: Optional[str] = None
    #: The delegation's ``scope`` claim, when present. Recorded for audit.

    scope: Optional[str] = None

    #: Where this identity came from — ``request`` | ``system`` | ``test``.
    #: Diagnostic only; never consulted for an authorization decision, because an
    #: identity that granted itself more authority by claiming a source would be the
    #: same self-asserted-operand defect this module exists to remove.
    source: str = "request"

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("ActingPrincipal.principal_id is required")


_acting: contextvars.ContextVar[Optional[ActingPrincipal]] = contextvars.ContextVar(
    "agience_mantle_acting_principal", default=None
)


def set_acting_principal(principal: ActingPrincipal) -> contextvars.Token:
    """Set the acting principal, returning a token for :func:`reset_acting_principal`.

    Prefer :func:`acting_as` — it cannot leak. This lower-level form exists for the
    FastAPI dependency and ASGI middleware, where set and reset straddle an ``await``.
    """
    if not isinstance(principal, ActingPrincipal):
        raise TypeError("set_acting_principal requires an ActingPrincipal")
    return _acting.set(principal)


def reset_acting_principal(token: contextvars.Token) -> None:
    """Restore whatever was in scope before the matching :func:`set_acting_principal`."""
    _acting.reset(token)


def current_acting_principal() -> Optional[ActingPrincipal]:
    """The acting principal, or ``None`` if this call is unauthenticated.

    Only for diagnostics and logging. Anything making an authorization decision must
    use :func:`require_acting_principal`, so that "absent" cannot be mistaken for
    "allowed" by a falsy check.
    """
    return _acting.get()


def require_acting_principal() -> ActingPrincipal:
    """The acting principal, or raise :class:`NoActingPrincipal`.

    The fail-closed accessor. Absence is an error, never an empty identity.
    """
    principal = _acting.get()
    if principal is None:
        raise NoActingPrincipal(
            "no acting principal is in scope; refusing to issue key material. "
            "Request paths get one from the auth dependency; background work must "
            "declare one explicitly (see system_acting_context)."
        )
    return principal


def acting_from_auth(auth) -> ActingPrincipal:
    """Build an :class:`ActingPrincipal` from an ``AuthContext``.

    Reads the delegation chain ``resolve_auth`` already unpacked. Kept here rather
    than in ``dependencies`` so this module has no import-time dependency on FastAPI
    — background code and tests import it freely.
    """
    principal_type = getattr(auth, "principal_type", "user") or "user"
    actor = getattr(auth, "actor", None)

    # A grant key acts as ITSELF — it is its own principal, not a stand-in for whoever
    # issued it. A grant key holds its permissions directly, so there is no owning
    # user to promote: `principal_id` is the root grant's id and the light cone
    # resolves against the key's own grants.
    #
    # This is the honest shape for a detached credential — it keeps a leaked key's
    # actions attributable to the key in the audit log rather than to a person who
    # was not present.
    principal_id = getattr(auth, "principal_id", "") or getattr(auth, "user_id", "") or ""

    if not principal_id:
        raise NoActingPrincipal(
            "AuthContext carries no principal_id; refusing to construct an "
            "acting principal for an unidentified caller"
        )
    return ActingPrincipal(
        principal_id=str(principal_id),
        principal_type=principal_type,
        actor=actor,
        host_id=getattr(auth, "host_id", None),
        authority=getattr(auth, "authority", None),
        source="request",
    )


@contextmanager
def acting_as(
    principal_id: str,
    *,
    principal_type: str = "service",
    actor: Optional[str] = None,
    host_id: Optional[str] = None,
    authority: Optional[str] = None,
    scope: Optional[str] = None,
    source: str = "system",
) -> Iterator[ActingPrincipal]:
    """Run a block as ``principal_id``, restoring the previous identity on exit.

    Restores in a ``finally``, so an exception inside the block cannot strand the
    identity for whatever runs next on this thread or task.
    """
    principal = ActingPrincipal(
        principal_id=principal_id,
        principal_type=principal_type,
        actor=actor,
        host_id=host_id,
        authority=authority,
        scope=scope,
        source=source,
    )
    token = _acting.set(principal)
    try:
        yield principal
    finally:
        _acting.reset(token)


def propagate(fn):
    """Wrap *fn* so it later runs inside a COPY OF THE CURRENT context.

    For handing work to a ``ThreadPoolExecutor`` or a bare ``threading.Thread``,
    neither of which propagates contextvars — a pool worker starts from an EMPTY
    context, so the acting principal silently vanishes at the fan-out and indexing
    fails closed with ``NoActingPrincipal``.

    Capture happens when ``propagate`` is CALLED, not when the wrapper runs, so it
    must be called on the submitting thread::

        with ThreadPoolExecutor(max_workers=n) as pool:
            pool.submit(propagate(work), item)      # captured here, on this thread
            pool.submit(work, item)                 # not propagated — starts from an empty context

    Each call returns a private copy, so wrapped callables are independent and safe
    to submit concurrently.
    """
    ctx = contextvars.copy_context()

    def _in_context(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return _in_context


def __getattr__(name: str):
    """`system_acting_context` lives in `services.system_identity`, not here.

    This hook makes the failure name the fix instead of surfacing as a bare
    AttributeError."""
    if name == "system_acting_context":
        raise ImportError(
            "system_acting_context has MOVED to `services.system_identity` (or "
            "`mantle.services.system_identity`).\n\n"
            "It resolves the PLATFORM system identity via `origin`, so it cannot live in "
            "`acting_principal` — that module is on the embeddable distribution surface and must "
            "stay stdlib-only (an embedding consumer installs the store without `origin`).\n\n"
            "    from services.system_identity import system_acting_context\n\n"
            "The exception types (KeyCustodyDenied / NoActingPrincipal / SystemPrincipalUnavailable) "
            "are unchanged and still live here."
        )
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
