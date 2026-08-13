"""The SSE arm's key contract — the seam between encrypted lexical search and key custody.

`sse/` implements blind tokens, encrypted posting lists, and the narrowing that reads them —
deterministic, fast, and
encrypted, with no model and no embedding. That makes it the retrieval story for an embedding
consumer of the store. It is blind: the index holds HMAC'd tokens and encrypted postings, never
plaintext terms, so the server cannot read the vocabulary of a collection or confirm whether any
given term appears in it.

Every `sse/` module besides `router_accessor.py` — the vector-arm and custody integration —
depends on nothing but stdlib + `cryptography`. That includes `indexer.py` and
`narrowing.py`, which need a key and take :class:`SseKeyProvider` from here rather than `..oracle`,
the module carrying the grant verifier, the lattice-backed master key store, and the CRUDEASIO
mint/read policy. None of that is retrieval; all of it is custody.

Dependency direction: this module is the interface and knows nothing of grants, the lattice, or
Fernet. `oracle.OracleService` is one implementation of it; an embedding consumer supplies
another. The implementation depends on the interface, never the reverse — so nothing here
imports `..oracle`.

The refusals a provider raises — `MasterKeyUnavailable` and `MasterKeyMissing` — are exactly one
class object each and live in :mod:`..custody`, which defines them and nothing else. Duplicating
them here would give the same name two types, and every `except` clause naming one would silently
stop matching the other; defining them in `..oracle` instead, next to the raises, made
`narrowing.py` import the whole custodian to catch one name. A module that holds only the two names is what lets
both sides name the same class without either reaching the other.

Stdlib only. No `cryptography`, no sibling package.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "SseKeyProvider",
]


@runtime_checkable
class SseKeyProvider(Protocol):
    """Derives the per-principal SSE key that blinds tokens and encrypts posting lists.

    The whole interface `sse/` needs from key custody. `OracleService` satisfies it in the deployed
    platform — gating on the CRUDEASIO grant light cone, minting only under write actions, reading
    the master key from the lattice — and an embedding consumer supplies its own.

    `request` is opaque here on purpose — it carries the purpose/action/context the provider's
    policy reads (`oracle.KeyRequest` in the platform). Typing it concretely would drag the custody
    model back across this seam, which is the coupling this interface exists to cut.
    """

    def derive_sse_key(self, principal_id: str, request: Any) -> bytes:
        ...
