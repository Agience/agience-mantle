"""The acting principal — the delegation chain, carried below the router.

WHY THIS EXISTS
---------------
Key issuance is grant-gated in ``search/mantle/oracle.py``, but until this module
existed the oracle had no way to learn WHO was asking. Every layer below the HTTP
router re-derived a "principal" from the *data* rather than from the *caller*:

* content encryption used ``doc["created_by"]`` — read off the very document being
  decrypted, then passed as BOTH sides of ``requester_id == principal_id``;
* ingest used ``resolve_cell_principal(collection)`` — the container's origin root.

Both are lineage, not identity. A check comparing a value to itself is not a check,
and that is what made ``KeyPurpose.SELF`` an unauthenticated skeleton key: the caller
supplied both operands, so the grant verifier was never consulted at all.

IDENTITY IS DELEGATED (§4B, John, 2026-07-21)
---------------------------------------------
This does not carry a bare principal id, because a bare id cannot answer the
questions an audit actually asks. It carries the chain Origin already mints::

    subject ──delegates──▶ actor (operator/server) ──runs on──▶ host
                                    │
                                    └──▶ authority (issuer) vouches for all of it

That is exactly the shape of Origin's delegation JWT — ``sub`` / ``act.sub`` /
``host_id`` / ``iss`` (``origin/services/auth_service.py:97-133``) — and
``resolve_auth`` already unpacks all four into ``AuthContext``
(``services/dependencies.py:311-318``). Before this module they were extracted at the
router and then dropped. Nothing new is invented here; the existing chain is simply
carried the rest of the way down.

FAIL CLOSED
-----------
Ambient authority is a real smell: code can forget to set it, and ambient state that
defaults to "allowed" is worse than no check. So the default is NOT a permissive
identity — it is *absent*, and ``require_acting_principal()`` RAISES. No acting
principal means no key. A background job that forgets to declare its identity fails
loudly at the point of key issuance rather than quietly acquiring one.

Deliberately NOT a token (contrast ``agience-chorus``'s contextvar, which carries a
raw bearer): this holds an already-verified chain, so nothing below the router
re-parses or re-trusts a credential. Verification stays at the boundary; this is only
the result of it.

USAGE
-----
Request path — set by the auth dependency, nothing else to do.

Background / system path — declare it EXPLICITLY at the entry point::

    from mantle.services.system_identity import system_acting_context

    with system_acting_context(scope="platform.reindex"):
        reindex_everything()

``acting_as`` / ``system_acting_context`` are context managers that restore the
previous value in a ``finally``, so they nest correctly and cannot strand an identity
on the thread or task that runs next.

⛔ THIS MODULE IS STDLIB-ONLY, AND THAT IS A CONTRACT, NOT AN ACCIDENT.
It is on Mantle's EMBEDDABLE distribution surface — ``db/doc_boundary.py`` imports
``KeyCustodyDenied`` from here, so anyone embedding the store (artifacts, collections,
grants, content) imports this module. ``system_acting_context`` used to live here and
reached for ``origin.service_identity``; it now lives in ``services/system_identity.py``,
behind the ``service`` extra. Keep platform imports out of this file — enforced by
``tests/test_embeddable_surface.py``.
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
# NOTE: `system_acting_context` is NOT here — it moved to `services.system_identity` so this
# module stays stdlib-only for the embeddable store surface. See `__getattr__` at the bottom.


class KeyCustodyDenied(PermissionError):
    """Base for every refusal to issue key material. ``GrantDenied`` subclasses it.

    ⚠ EXISTS SO REFUSALS ARE CAUGHT STRUCTURALLY, NOT BY ENUMERATION. The unified
    accessor deliberately re-raises authorization failures and swallows everything
    else, so a flaky vector arm degrades to SSE-only::

        except (KeyCustodyDenied, SystemicKeyFailure):
            raise                       # "not authorized" must not become "no results"
        except Exception:
            ...                         # degrade to SSE-only

    With a hand-listed tuple, any NEW refusal type is silently swallowed by the
    blanket clause below it — "you are not authorized" quietly becomes a narrower
    result set, which is the fail-open shape this work exists to remove. A shared
    base means a new refusal is caught the day it is written, without anyone
    remembering to update the tuple.

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

    #: ``user | api_key | server | mcp_client | grant_key | service | delegation``.
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

    #: The delegation's ``scope`` claim, when present. Recorded for audit. ⚠ NOT yet
    #: enforced as a ceiling — ``resolve_auth`` does not extract ``scope`` today, so
    #: bounding is by the subject's grant set. Narrowing by scope is a separate
    #: change and must not be assumed here.
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
    principal_id = getattr(auth, "principal_id", "") or getattr(auth, "user_id", "") or ""
    if not principal_id:
        raise NoActingPrincipal(
            "AuthContext carries no principal_id; refusing to construct an "
            "acting principal for an unidentified caller"
        )
    return ActingPrincipal(
        principal_id=str(principal_id),
        principal_type=getattr(auth, "principal_type", "user") or "user",
        actor=getattr(auth, "actor", None),
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
            pool.submit(propagate(work), item)      # ✅ captured here, on this thread
            pool.submit(work, item)                 # ⛔ worker gets an empty context

    Each call returns a private copy, so wrapped callables are independent and safe
    to submit concurrently.
    """
    ctx = contextvars.copy_context()

    def _in_context(*args, **kwargs):
        return ctx.run(fn, *args, **kwargs)

    return _in_context


def __getattr__(name: str):
    """`system_acting_context` MOVED to `services.system_identity` — fail loudly, not silently.

    ⛔ DELIBERATELY NOT A RE-EXPORT. A module-level `from .system_identity import …` would put
    this module's dependency set back where it was: importing `acting_principal` would drag in a
    module that reaches for `origin`, which is exactly the coupling the split removed. The point
    was to make this file provably stdlib-only, and a convenience alias would undo that.

    So callers update their import. This hook exists only so the failure names the fix instead of
    surfacing as a bare AttributeError."""
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
