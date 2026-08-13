"""The digest ⟷ content-ref mapping — the number must survive the trip unchanged, both ways.

Everything the deploy lane claims rests on one property: the digest a box is told to run is the
digest that was built, with nothing re-deriving it in between. If this mapping is even slightly
lossy — case, prefix, algorithm — then "both boxes provably run the same code" becomes "both boxes
ran something we called the same name", which is the state the digest-not-a-tag rule exists to
escape.

Every rejection test names the input it refuses. These are not tidiness: `digest_to_ref` turns a
string taken straight off an HTTP path into a key in the content store.
"""
from __future__ import annotations

import hashlib

import pytest

from mantle.oci.address import (
    OciAddressError, digest_to_ref, ref_to_digest, parse_digest, is_digest,
)
from mantle.shard.content import CAS_PREFIX, content_ref

HEX = "a" * 64


# ── the property that matters ────────────────────────────────────────────────────────────────────

def test_the_store_address_and_the_oci_digest_are_the_same_number():
    """Computed from real bytes through both systems, not asserted on a constant.

    `content_ref` is what the store uses; `sha256:<hex>` is what OCI uses. If these ever diverge,
    a blob would be stored under one name and requested under another, and the registry would
    answer 404 for content it holds.
    """
    blob = b"a layer's worth of bytes"
    store_side = content_ref(blob)                                  # cas/<hex>
    oci_side = "sha256:" + hashlib.sha256(blob).hexdigest()          # sha256:<hex>
    assert digest_to_ref(oci_side) == store_side
    assert ref_to_digest(store_side) == oci_side


# Named ids: without them pytest builds a test id out of the bytes themselves, and this repo's
# conftest puts ids into the environment — a 10 KB blob then fails with "the environment variable is
# longer than 32767 characters", a limit on the test id rather than on the code under test.
@pytest.mark.parametrize("blob", [b"", b"x", b"\x00\xff" * 5000, bytes(range(256))],
                         ids=["empty", "one-byte", "10k-binary", "every-byte-value"])
def test_round_trip_is_exact_for_arbitrary_bytes(blob: bytes):
    ref = content_ref(blob)
    assert digest_to_ref(ref_to_digest(ref)) == ref


def test_nothing_here_hashes_anything():
    """The mapping must be a rename, never a re-derivation.

    A module that computed a digest would be a second opinion about what an image is called. The
    test is structural because the failure is not visible in any single call's output.
    """
    # An AST walk, not a text search: a text search matches this module's own prose — which says
    # "sha256(plaintext)" while explaining that it must never call it — and would fail on the
    # explanation rather than on the code.
    import ast
    import mantle.oci.address as mod
    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())

    imported = {n.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for n in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert "hashlib" not in imported, "address.py imports hashlib — it must not be able to hash"

    called = {node.func.attr if isinstance(node.func, ast.Attribute) else
              getattr(node.func, "id", "") for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not (called & {"sha256", "new", "digest", "hexdigest"}), \
        "address.py computes a digest (%s) — it must come from the caller" % sorted(
            called & {"sha256", "new", "digest", "hexdigest"})


# ── the refusals ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad,why", [
    ("", "empty"),
    (HEX, "no algorithm prefix — a bare hex string is not a digest"),
    ("sha256:", "no hex at all"),
    ("sha256:" + "a" * 63, "63 characters — one short, and a truncated digest addresses nothing"),
    ("sha256:" + "a" * 65, "65 characters"),
    ("sha256:" + "A" * 64, "UPPERCASE — two names for one blob would defeat dedupe"),
    ("sha256:" + "g" * 64, "not hex"),
    ("sha256:../../etc/passwd" + "a" * 42, "traversal, and this becomes a store key"),
    ("sha512:" + "a" * 128, "an algorithm this store cannot VERIFY content against"),
    ("sha256:" + " " * 64, "whitespace"),
])
def test_a_digest_that_cannot_be_verified_is_refused(bad: str, why: str):
    with pytest.raises(OciAddressError):
        digest_to_ref(bad)
    assert is_digest(bad) is False, "is_digest accepted %r (%s)" % (bad, why)


def test_the_inverse_is_typed_and_refuses_a_foreign_string():
    """Stripping the prefix if present would be lossy. A string without the prefix is not a ref,
    and guessing would mint a digest for content whose address nobody established."""
    with pytest.raises(OciAddressError):
        ref_to_digest(HEX)                       # no `cas/`
    with pytest.raises(OciAddressError):
        ref_to_digest("sha256:" + HEX)           # the other space — a real confusion, not a typo
    with pytest.raises(OciAddressError):
        ref_to_digest(CAS_PREFIX + "nothex")


def test_a_tag_is_not_mistaken_for_a_digest():
    """`GET /v2/<name>/manifests/<reference>` takes either, and they resolve differently — a digest
    is a keyed content lookup, a tag needs a name lookup in the repository."""
    for tag in ("latest", "stable", "v1.2.3", "edge", "sha256", "sha256-ish"):
        assert is_digest(tag) is False
    assert is_digest("sha256:" + HEX) is True


def test_parse_digest_returns_the_hex_and_only_the_hex():
    assert parse_digest("sha256:" + HEX) == HEX


def test_the_error_says_which_rule_failed():
    """A refusal that does not name the rule sends someone to read this file to find out."""
    with pytest.raises(OciAddressError, match="algorithm"):
        digest_to_ref("sha512:" + "a" * 128)
    with pytest.raises(OciAddressError, match="64 lowercase hex"):
        digest_to_ref("sha256:zz")
