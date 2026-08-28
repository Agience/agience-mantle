"""A brand-new install must be able to store the first thing it is given.

IT COULD NOT, reproduced 2026-08-17 by following the README quickstart on a clean install:
`pip install` → `mantle-init-keys` → `mantle-serve` → `mantle-token` → `create_artifact` answered

    ContentEncryptionError: content encryption unavailable for 'artifacts/….content';
    refusing to persist plaintext to the object store

underneath which was `GrantDenied: … holds no 'read' grant reaching principal … / collection …`.
Every other test in this suite passed while that was true, because every one of them arrives at a
store where somebody already holds a grant. The failure is invisible to a seeded corpus and is the
first thing a new user meets.

The cause was a correct fix landing on an unguarded ordering. `LightConeGrantVerifier.authorized`
has a self-custody base case whose own comment states this exact failure as its reason for
existing: "a brand-new principal could not write a first top-level artifact on any store, because
the check would be asking someone to be granted access to themselves". It sat BELOW the
collection-scoped branch, so it applied only when no scope was named — and the C10 repair (binding
the content AAD to its collection) began naming one on every content write. The base case became
unreachable on the one path that needed it.

C10 is right and is untouched. It decides which BLOB opens; this decides whose KEY it is. A master
key is per-principal by construction, so on a caller's OWN key there is nothing for a collection
scope to narrow — refusing it there denies a key the caller already holds, in a way no grant can
repair.
"""
from __future__ import annotations

import pytest

from mantle.search.mantle.oracle import LightConeGrantVerifier


ALICE = "user-alice"
BOB = "user-bob"


class _G:
    """A grant as `mask_of` reads one."""

    def __init__(self, resource_id, **flags):
        self.resource_id = resource_id
        self.effect = "allow"
        for a in ("create", "read", "update", "delete", "evict", "invoke", "add", "share", "admin"):
            setattr(self, "can_" + a, flags.get("can_" + a, False))


def _verifier(pairs=(), grants=(), roots=None):
    """A real verifier whose reach is exactly what is described here — `()` is a virgin store.

    Two seams, because the verifier asks two different questions. A request naming a COLLECTION is
    answered by walking up from it and testing each resource against the requester's grants, so
    that case is described by `grants` plus a chain. A request naming no collection falls to the
    principal-scoped tail, which still reads resolved pairs.
    """
    v = LightConeGrantVerifier.__new__(LightConeGrantVerifier)
    v._contexts = lambda requester_id, requester_type, action: set(pairs)   # noqa: SLF001
    v._grants = lambda requester_id, requester_type: list(grants)           # noqa: SLF001
    # A flat store: every collection is its own root and has no parent, so the chain is the
    # collection itself. Tests that need a container describe one explicitly.
    v._chain = lambda collection_id, action: iter([collection_id])          # noqa: SLF001
    # A resource's origin root is its cell principal. The unscoped branch asks whether any resource
    # the requester holds sits under the principal being asked about, so a test describes reach
    # there by naming roots. Unnamed, a resource is its own root — a flat store.
    _roots = dict(roots or {})
    v._root_of = lambda resource_id: _roots.get(resource_id, resource_id)   # noqa: SLF001
    return v


def _ask(v, requester, principal, collection):
    return v.authorized(requester_id=requester, requester_type="user",
                        principal_id=principal, collection_id=collection, action="read")


# ── the regression ───────────────────────────────────────────────────────────────────────────


def test_a_principal_holds_its_own_key_even_with_a_scope_named():
    """Self-custody on an empty ledger, at the smallest scale that shows it.

    Alice asks for Alice's key, scoped to the artifact she is creating, on an empty ledger. This is
    the call `content_service.put_bytes_encrypted` has made on every content write since C10, and it
    answered False — asking Alice to have been granted access to herself.
    """
    assert _ask(_verifier(), ALICE, ALICE, "the-artifact-being-created") is True


def test_the_same_question_without_a_scope_always_worked():
    """The control that localises it: identical question, no `collection_id`. This path was never
    broken, which is why the failure appeared only once C10 started naming a scope."""
    assert _ask(_verifier(), ALICE, ALICE, None) is True


@pytest.mark.parametrize("scope", ["some-collection", None])
def test_the_scope_still_refuses_somebody_else_s_key(scope):
    """What self-custody must not hand back.

    C10's finding was "reach any one collection under an owner, get the key to all of them", and
    self-custody sitting above the scope check must not re-open it: Bob asking for Alice's key is
    refused
    either way, because the base case is guarded by `requester_id == principal_id`.
    """
    assert _ask(_verifier(), BOB, ALICE, scope) is False


def test_a_reachable_pair_is_still_what_authorizes_another_principal_s_key():
    """And the positive control for the scoped branch: Bob CAN reach Alice's key when the light cone
    actually resolves that pair. The branch still decides everything it decided before."""
    v = _verifier(grants=[_G("shared-collection", can_read=True)])
    assert _ask(v, BOB, ALICE, "shared-collection") is True
    assert _ask(v, BOB, ALICE, "a-different-collection") is False, (
        "the scope must still narrow — this is the C10 property"
    )


def test_an_unscoped_request_for_another_principal_still_needs_reach():
    """The principal-scoped tail (the SSE key spans a whole corpus): authorized iff the requester
    reaches at least one collection under that principal."""
    reaching = _verifier(grants=[_G("c1", can_read=True)], roots={"c1": ALICE})
    assert _ask(reaching, BOB, ALICE, None) is True

    elsewhere = _verifier(grants=[_G("c1", can_read=True)], roots={"c1": "someone-else"})
    assert _ask(elsewhere, BOB, ALICE, None) is False


# ── and the round trip, because a write that cannot be read back is not a write ───────────────


def test_content_seals_and_opens_under_the_same_principal():
    """The seal and the open derive from the same principal.

    Re-keying container content to the container rather than the creator makes the write succeed and
    the read 404: `doc_boundary.decrypt_artifact_content` derives from
    `content_key_principal or created_by`, so the two sides disagreed about whose key it was, and
    `_find_artifact` swallowed the decryption failure into "not found" — a create that reports
    success and cannot be read.

    So the principal is the creator on both sides, and the scope binds the AAD on both sides.
    """
    from mantle.services import content_crypto

    key = lambda principal: b"k" * 32                                  # noqa: E731
    sealed = content_crypto.encrypt_content(
        ALICE, b"the body", collection_id="artifact-1", master_key_provider=key)
    assert content_crypto.decrypt_content(
        ALICE, sealed, collection_id="artifact-1", master_key_provider=key) == b"the body"

    # And C10's binding still holds: the same bytes do not open under another scope.
    with pytest.raises(Exception):
        content_crypto.decrypt_content(
            ALICE, sealed, collection_id="artifact-2", master_key_provider=key)
