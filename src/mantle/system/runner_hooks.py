"""What the STORE asks of a RUNNER, when one is present — injected, never imported.

The runner tells the store what it can do by wiring these hooks; the store only asks when it
already has an answer to hand. The sites that need it are the mesh daemon invoking operators, and
erasure resolving an author claim to its person vertex.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

#: (store, operator_id, args) -> result dict. Wired by the runner.
_INVOKE: Optional[Callable[..., Any]] = None

#: (store, author_claim) -> the person artifact's vertex id, minting it if absent.
_AUTHOR_REF: Optional[Callable[..., Any]] = None

#: The operator ids that ingest from a source — the daemon schedules these.
_SOURCE_INGESTERS: Optional[Any] = None

_warned: set = set()


def set_hooks(*, invoke=None, author_ref=None, source_ingesters=None) -> None:
    """Wire the runner's capabilities. Called once at boot; passing None leaves a hook alone.
    Idempotent by design — a process that boots the runner twice must not end up with half a set."""
    global _INVOKE, _AUTHOR_REF, _SOURCE_INGESTERS
    if invoke is not None:
        _INVOKE = invoke
    if author_ref is not None:
        _AUTHOR_REF = author_ref
    if source_ingesters is not None:
        _SOURCE_INGESTERS = source_ingesters


def _once(key: str, msg: str) -> None:
    """Say it the first time and then stop — a per-row warning in a drain loop is noise that
    trains people to ignore the log, which is worse than not logging at all."""
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def invoke(store, operator_id: str, args: dict):
    """Invoke an operator through the wired runner, or None if there is none."""
    if _INVOKE is None:
        _once("invoke", "mantle: no runner wired, so operator %r was not invoked. This is expected "
                        "for an embedded store; wire it with runner_hooks.set_hooks(invoke=...) if "
                        "the caller needs operators." % operator_id)
        return None
    return _INVOKE(store, operator_id, args)


def author_ref(store, author: str):
    """Resolve an author claim to its person vertex, or None if no runner is wired.
    Returning None rather than a fallback keeps a caller from citing a vertex that was never minted."""
    if _AUTHOR_REF is None:
        _once("author_ref", "mantle: no runner wired, so an author claim was not resolved to a "
                            "person vertex. Callers must treat this as unresolved, never as the "
                            "claim itself.")
        return None
    return _AUTHOR_REF(store, author)


def source_ingesters():
    """The source-ingest operator ids the runner knows about; empty when unwired."""
    return _SOURCE_INGESTERS or ()
