"""The refusals a key custodian raises — one class object per name, and nothing else.

`oracle.py` raises these; `sse/narrowing.py` and `search/ingest/pipeline_unified.py` catch them.
An ``except`` clause matches on the CLASS OBJECT, so the raise and every catch must name the
same class — a second, faithfully copied definition is a different type and every clause
naming it silently stops matching, on the failure path, which is the worst place to find out.
That constraint says the names must live in exactly one module. It does not say which one,
and putting them with the RAISE made the answer `oracle`, which is the custody
implementation: the grant verifier, the lattice-backed master key store, the CRUDEASIO mint
policy, Fernet. The lexical arm needs one name out of that module and consequently imported
all of it, inverting the direction `sse/keys.py` declares — the implementation depends on the
interface, never the reverse.

So the definitions sit HERE instead: beside `oracle.py` rather than inside it, at the level
both the custodian and the arm hang off, importing nothing. `oracle` imports it, `sse` imports
it, neither reaches the other, and there is still exactly one class object per name.

Zero imports, deliberately — the same property `sse/keys.py` holds and for the same reason.
A module two packages depend on to break a dependency cannot itself carry one, and a
refusal is a name, not a behaviour: there is nothing here to need a dependency for.

:class:`MasterKeyUnavailable` subclasses ``RuntimeError`` and NOT
``services.acting_principal.KeyCustodyDenied``. That base is for a refusal to ISSUE key
material — an authorization answer, mapped to 403 — and these two are not that. They say the
key could not be read, or was never minted; nobody was denied anything. Rooting them at
`KeyCustodyDenied` would put them inside every fail-closed handler that catches it and turn a
storage fault into an authorization verdict.
"""

from __future__ import annotations

__all__ = [
    "MasterKeyMissing",
    "MasterKeyUnavailable",
]


class MasterKeyUnavailable(RuntimeError):
    """A principal's master key cannot be used right now.

    A key exists (or may exist) but could not be read or unwrapped — deliberately distinct
    from "no key yet", because a caller must never treat it as first use. The recovery for
    first use is to generate and persist a new key, which overwrites the only copy of the old
    one. Failing a request is recoverable; that is not.
    """


class MasterKeyMissing(MasterKeyUnavailable):
    """No master key exists for this principal, and this request may not create one.

    Subclasses :class:`MasterKeyUnavailable` so existing handling — which already treats "the key is
    not usable" as a hard stop — applies unchanged. The distinction is *why*: absent rather than
    unreadable, and absence is the one case a read may legitimately answer as empty (there is
    nothing indexed under a key that was never minted). See `sse/narrowing.py`'s narrow catch.
    """
