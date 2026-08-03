"""The SSE arm's KEY CONTRACT — the one seam between encrypted lexical search and key custody.

⛔ WHY THIS MODULE EXISTS: TO MAKE THE LEXICAL ARM SHIPPABLE.

`sse/` is blind tokens, encrypted posting lists and BM25 — deterministic, fast, and encrypted, with
no model and no embedding. That makes it the retrieval story for an EMBEDDING consumer of the store
(EREA §5, 2026-07-28: they are retiring their vector index rather than porting it, because the
alternative — `db/lattice/fts.py` — is FTS5 *contentless* but still holds PLAINTEXT term postings,
which discloses the vocabulary of every collection and confirms whether any given term appears).

Ten of the eleven `sse/` modules were already free of everything but stdlib + `cryptography`. The
lexical arm was pinned service-side by ONE import — `indexer.py` and `query.py` reaching for
`..oracle`, a 703-line module carrying the grant verifier, the lattice-backed master key store and
the CRUDEASIO mint/read policy. None of that is retrieval. All of it is custody.

⚠ MEASURED, NOT ASSUMED. The entire coupling was: `OracleService` as a type annotation, ONE method
call (`derive_sse_key`), `KeyRequest` passed straight through, and `MasterKeyMissing` caught in one
place. That is an INTERFACE, and it was being satisfied by dragging in an implementation.

DIRECTION OF THE DEPENDENCY. `oracle.py` imports FROM here and re-exports, rather than this module
importing from `oracle`. The implementation depends on the interface, never the reverse — and it is
why the exceptions had to MOVE rather than be duplicated: `pipeline_unified.py` and
`test_key_custody_bypasses.py` both `except MasterKeyMissing` off the oracle import, so a second
class of the same name would silently stop matching. There is exactly ONE class; `oracle` re-exports
the same object.

⚠ THE EXCEPTIONS ARE PART OF THE CONTRACT, NOT INCIDENTAL. `query.py` distinguishes
`MasterKeyMissing` ("no key exists yet — an honest empty result") from `MasterKeyUnavailable` ("the
key exists and we cannot read it — a hard stop"). Collapsing them would turn a broken key volume
into "no results", which is the silent-`[]` failure class this codebase keeps killing. Any provider
substituted here MUST preserve that distinction.

STDLIB ONLY. No `cryptography`, no sibling package. Deliberately: this is the file that decides
whether the lexical arm can travel.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "SseKeyProvider",
    "MasterKeyUnavailable",
    "MasterKeyMissing",
]


class MasterKeyUnavailable(RuntimeError):
    """A principal's master key cannot be used right now.

    ⛔ A HARD STOP, NEVER AN EMPTY RESULT. The key material exists as far as anyone knows; what
    failed is reading or unwrapping it (an unmounted keys volume, a rotated wrapping key, a store
    that will not answer). Degrading this to "no results" would report a broken deployment as a
    successful search that found nothing — indistinguishable, to the caller, from a corpus that
    genuinely lacks the term.
    """


class MasterKeyMissing(MasterKeyUnavailable):
    """No master key EXISTS for this principal, and this request may not create one.

    Subclasses :class:`MasterKeyUnavailable` so existing handling — which already treats "the key is
    not usable" as a hard stop — applies unchanged. The distinction is *why*: absent rather than
    unreadable, and absence is the one case a READ may legitimately answer as empty (there is
    nothing indexed under a key that was never minted). See `query.py`'s narrow catch.
    """


@runtime_checkable
class SseKeyProvider(Protocol):
    """Derives the per-principal SSE key that blinds tokens and encrypts posting lists.

    The WHOLE interface `sse/` needs from key custody. `OracleService` satisfies it in the deployed
    platform — gating on the CRUDEASIO grant light cone, minting only under write actions, reading
    the master key from the lattice — and an embedding consumer supplies its own.

    ⚠ WHAT A SUBSTITUTE MUST STILL DO, because `sse/` cannot check it:
      * AUTHORIZE. This method is where the grant is enforced. Returning a key for a principal the
        caller may not act as hands them a readable index. `sse/` treats any returned key as
        already-authorized — it has no grant, no light cone, and no way to second-guess one.
      * DERIVE, NOT INVENT. The same (principal, request) must yield the same key forever, or
        previously-written postings become unreadable — the index is not re-derivable from
        ciphertext.
      * DISTINGUISH ABSENT FROM UNREADABLE. Raise :class:`MasterKeyMissing` when no key exists and
        this request may not mint one; raise :class:`MasterKeyUnavailable` (or a subclass that is
        NOT `MasterKeyMissing`) when a key exists but cannot be read. `query.py` answers empty for
        the first and propagates the second.

    `request` is opaque here on purpose — it carries the purpose/action/context the PROVIDER's
    policy reads (`oracle.KeyRequest` in the platform). Typing it concretely would drag the custody
    model back across this seam, which is the coupling the module exists to cut.
    """

    def derive_sse_key(self, principal_id: str, request: Any) -> bytes:
        ...
