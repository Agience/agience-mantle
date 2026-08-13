"""The one conversion between an OCI digest and a lattice content ref.

    sha256:<64 hex>   ⟷   cas/<64 hex>

It is the same number, and that is the whole property. `shard/content.content_ref` is
`"cas/" + sha256(plaintext).hexdigest()`; an OCI digest is `"sha256:" + the same hex over the same
bytes. Nothing here hashes anything — if this module ever computed a digest it would have become a
second opinion about what an image is called, and "both boxes provably run the same code" rests on
there being exactly one.

Every rejection below is a security boundary, not tidiness. This function turns a string taken
off an HTTP path into a filesystem-ish key in the content store. `sha256:../../etc/passwd` and
`sha256:` + 64 spaces must not become refs; an algorithm this store cannot verify must not be
accepted and silently treated as sha256. The store verifies content against `ref` on read, so a
malformed ref cannot yield wrong bytes — but it can yield a confusing miss instead of a clear
refusal, and the refusal is what tells an operator their registry is being probed.
"""
from __future__ import annotations

import re

#: `shard/content.CAS_PREFIX`, imported rather than repeated — the two must never drift, and a
#: literal here would be a typed-in value standing for a decision made somewhere else.
from mantle.shard.content import CAS_PREFIX

#: sha256 only, deliberately. The store's content address is sha256 (`content_ref` computes it and
#: `TieredContentStore` verifies against it), so accepting `sha512:…` would mean accepting a name
#: this store cannot check. The OCI spec permits other algorithms; supporting one we cannot verify
#: is worse than refusing it, because the verify-on-read guarantee would silently not apply.
_ALGO = "sha256"
_HEX = re.compile(r"\A[0-9a-f]{64}\Z")     # lowercase only: OCI requires it, and `Aa` vs `aa` would
                                           # otherwise be two names for one blob and defeat dedupe.


class OciAddressError(ValueError):
    """A digest that cannot be turned into a content ref. Always says which rule failed."""


def parse_digest(digest: str) -> str:
    """`sha256:<hex>` -> the hex. Raises OciAddressError on anything else."""
    if not isinstance(digest, str):
        raise OciAddressError("digest must be a string, got %s" % type(digest).__name__)
    algo, sep, hexpart = digest.partition(":")
    if not sep:
        raise OciAddressError(
            "digest %r has no algorithm prefix — expected '%s:<64 hex>'" % (digest, _ALGO))
    if algo != _ALGO:
        raise OciAddressError(
            "digest algorithm %r is not supported: this store addresses and VERIFIES content by "
            "%s, so accepting %r would mean accepting a name it cannot check" % (algo, _ALGO, algo))
    if not _HEX.match(hexpart):
        raise OciAddressError(
            "digest %r is not 64 lowercase hex characters — refusing to use it as a content key"
            % (digest,))
    return hexpart


def is_digest(reference: str) -> bool:
    """Is this reference a digest rather than a tag? Never raises — callers branch on it.

    `GET /v2/<name>/manifests/<reference>` takes either, and the two resolve differently: a digest
    is a keyed content lookup, a tag is a name that has to be looked up in the repository.
    """
    try:
        parse_digest(reference)
        return True
    except OciAddressError:
        return False


def digest_to_ref(digest: str) -> str:
    """`sha256:<hex>` -> `cas/<hex>`. The only way a digest becomes a store key."""
    return CAS_PREFIX + parse_digest(digest)


def ref_to_digest(ref: str) -> str:
    """`cas/<hex>` -> `sha256:<hex>`. The inverse, and it is exact in both directions.

    Typed, because the inverse of a prefix-strip is not total. A ref that does not carry the
    prefix is not "a ref without a prefix" — it is not a ref, and guessing would mint a digest for
    content whose address nobody established. This function raises instead: the caller knows which
    space its string came from and this function does not.
    """
    if not isinstance(ref, str) or not ref.startswith(CAS_PREFIX):
        raise OciAddressError(
            "%r is not a content ref (expected the %r prefix) — refusing to invent a digest for it"
            % (ref, CAS_PREFIX))
    hexpart = ref[len(CAS_PREFIX):]
    if not _HEX.match(hexpart):
        raise OciAddressError("content ref %r does not carry 64 lowercase hex characters" % (ref,))
    return "%s:%s" % (_ALGO, hexpart)
