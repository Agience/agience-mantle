"""`resolve_text` must not hand a sealed envelope back as if it were text.

WHAT THIS COST, measured on 71/home 2026-08-27. A capture's body is sealed under its collection's
origin root (`doc_boundary.content_key_principal`), so opening it needs an acting principal. The
describer (`sage/describe._split`) calls `resolve_text` and has none, so the MEC1 envelope came
back unopened — and `.decode("utf-8", "ignore")` turned 154,865 bytes of AES-GCM ciphertext into a
lossy pseudo-string that was 69% printable. Nothing raised.

The rest followed on its own: `doc_index.extract_terms` found no terms in ciphertext, the
describer took its documented fallback (`lemmas = [... or "document"]`, deliberate so that
"describe is always terminating"), and `describe_dark` — which skips any artifact that already has
lemmas — never looked at it again. 874 capture artifacts carry the literal `['document']` and 473
carry `['module']`; every one is a body nobody could read, recorded as a body nobody needed to.

`_split`'s own docstring records fixing this exact SHAPE once before, for a different cause: a path
line made `ast.parse` fail, extraction returned empty, and the file was "still keyed from its stem,
so `describe_dark` skips it forever with no error at any layer."

So the rule: a caller that cannot open the envelope must be TOLD, not handed ciphertext. Ciphertext
that decodes to something string-shaped is the worst possible return value — it is indistinguishable
from an empty document to every caller downstream.
"""
from __future__ import annotations

import pytest

from mantle.services import content_crypto
from mantle.shard import content as C


class _SealedTier:
    """A content tier that returns what the real one returns for a sealed body: the envelope."""

    def __init__(self, blob):
        self.blob = blob

    def get(self, ref, collection=None):
        return self.blob


class _Bundle:
    def __init__(self, tier):
        self.content_tier = tier
        self.content = None
        self.keys_dir = None


def _sealed_blob() -> bytes:
    """MEC1 magic + plausible body. Built from the module's own constant, never a literal."""
    return b"MEC1" + bytes(range(12)) + b"\x9f\x21\x00\xd3" * 40


def test_the_fixture_really_is_an_envelope():
    """The premise, asserted directly, so the test below cannot pass for another reason."""
    assert content_crypto.is_encrypted(_sealed_blob())


def test_a_sealed_body_raises_instead_of_returning_ciphertext():
    blob = _sealed_blob()
    bundle = _Bundle(_SealedTier(blob))
    artifact = {"id": "a1", "content_ref": "cas/deadbeef", "collection_id": "c1"}
    with pytest.raises(C.ContentStillSealed):
        C.resolve_text(bundle, artifact)


def test_the_error_names_the_remedy():
    """A custody failure must say what to do about it. `op.describe.*` is an OPERATOR, so the fix
    is to INVOKE it as one — with a principal holding a read grant — not to run it as a bare
    function and not to escalate to the system principal, which would authorize a user's content
    as the platform (see `ember/custody.py`)."""
    bundle = _Bundle(_SealedTier(_sealed_blob()))
    try:
        C.resolve_text(bundle, {"id": "a1", "content_ref": "cas/x", "collection_id": "c1"})
    except C.ContentStillSealed as exc:
        msg = str(exc)
    else:
        pytest.fail("sealed content did not raise")
    assert "principal" in msg.lower()
    assert "cas/x" in msg or "a1" in msg


def test_plaintext_still_comes_back_unchanged():
    """The repair must not break the ordinary path."""
    bundle = _Bundle(_SealedTier(b"# A title\n\nSome prose."))
    got = C.resolve_text(bundle, {"id": "a1", "content_ref": "cas/x", "collection_id": "c1"})
    assert got == "# A title\n\nSome prose."


def test_non_utf8_plaintext_is_still_tolerated():
    """`ignore` was doing a second, legitimate job: real content is not always clean UTF-8. Only
    the SEALED case changes; a stray byte in genuine plaintext must still come through."""
    bundle = _Bundle(_SealedTier(b"caf\xe9 au lait"))
    got = C.resolve_text(bundle, {"id": "a1", "content_ref": "cas/x", "collection_id": "c1"})
    assert "au lait" in got
