"""Ingest an image into the lattice's content store, and read it back out by digest.

The blobs go in under the names they already have. `shard.content.put_content` addresses content
at `cas/sha256(plaintext)` — which is the OCI digest with a different prefix — so ingesting an image
is not a conversion, it is a move. `oci.address` does the renaming and nothing hashes twice.

Idempotent for free, as a property of content addressing rather than a feature: `put_content`
skips a ref the store already holds. Re-ingesting an unchanged image writes nothing;
ingesting a rebuild writes only the layers that actually changed; and the same image ingested
independently into two separate stores arrives at the identical digest in both — which is what makes
"foundation and foresight run the same artifact" provable rather than asserted, without either node
reading the other's store.

`put_content` / `get_content` are a matched pair and both are used. `put_content` hands the store
Fernet ciphertext addressed by the plaintext's hash; `get_content` decrypts on the way out. Reaching
past them — to `content_store.put` directly, or to the tiered reader that verifies plaintext — mixes
two encryption models on one object, which `sqlite_store` warns about in the store bundle's own
comments. One pair in, the same pair out.

The store is not a place to put an unverified blob. Everything here takes blobs that
`oci.layout.read_layout` has already hashed and checked. That ordering is the security property:
this module never sees a digest it has not been given evidence for.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple

from mantle.oci.address import digest_to_ref
from mantle.oci.layout import Blob, Image, blob_bytes
from mantle.shard.content import get_content, put_content


class IngestedBlob(NamedTuple):
    digest: str
    ref: str
    size: int
    media_type: str
    stored: bool        # False = the store already held it. NOT an error — the normal case.


class ImageRecord(NamedTuple):
    """What an image artifact carries. The manifest BYTES live in the CAS like any other blob;
    this is the index into them, and it is small enough to be an artifact's context."""
    repository: str
    digest: str          # the manifest's digest — the promotable name
    media_type: str
    tag: str             # may be "" — a tag is a name someone chose, never an identity
    blobs: List[IngestedBlob]

    @property
    def manifest_ref(self) -> str:
        return digest_to_ref(self.digest)

    def as_context(self) -> Dict[str, object]:
        """The artifact context. Digests, not refs — an artifact records what the image IS called,
        and `oci.address` derives the store key from that. Recording the ref instead would put the
        store's internal spelling into a record that outlives it."""
        return {
            "oci_repository": self.repository,
            "oci_digest": self.digest,
            "oci_media_type": self.media_type,
            "oci_tag": self.tag,
            "oci_blobs": [{"digest": b.digest, "size": b.size, "mediaType": b.media_type}
                          for b in self.blobs],
        }


def ingest_image(content_store, keys_dir, image: Image, *, repository: str) -> ImageRecord:
    """Put every blob of a VERIFIED image into the content store. Returns what landed.

    The manifest goes in last. Its blobs are what it points at, so writing it first would leave a
    window — however short, and longer on a slow disk — in which the store holds a manifest naming
    content it does not have. A puller arriving in that window gets 200 for the manifest and 404 for
    a layer, which reads as a corrupt image rather than as an ingest still running.
    """
    manifest_blob = image.blobs[0]
    if manifest_blob.digest != image.digest:
        raise ValueError(
            "the first blob must be the manifest itself (%s), got %s — read_layout builds this "
            "list and something has changed its order" % (image.digest, manifest_blob.digest))

    out: List[IngestedBlob] = []
    for blob in list(image.blobs[1:]) + [manifest_blob]:
        out.append(_put(content_store, keys_dir, blob))

    # Reported in the layout's order — manifest first — so the record reads the way the image does.
    out = [out[-1]] + out[:-1]
    return ImageRecord(repository=repository, digest=image.digest,
                       media_type=image.media_type, tag=image.ref_name, blobs=out)


def _put(content_store, keys_dir, blob: Blob) -> IngestedBlob:
    ref = digest_to_ref(blob.digest)
    already = _has(content_store, ref)
    if already:
        # Skipped, and SAID so rather than silently. "0 blobs written" on a rebuild is the expected
        # answer for an unchanged image and an alarming one for a changed image — the caller can
        # only tell those apart if this is reported.
        return IngestedBlob(blob.digest, ref, blob.size, blob.media_type, stored=False)
    body = b"".join(blob_bytes(blob))
    got_ref, size = put_content(content_store, keys_dir, body)
    if got_ref != ref:
        # Cannot happen unless the two address spaces have diverged — which is exactly the failure
        # that would otherwise be invisible, so it is checked rather than assumed.
        raise ValueError(
            "the store addressed this blob as %s but its OCI digest is %s — the content address "
            "and the digest have diverged, and promotion by digest is no longer meaningful"
            % (got_ref, ref))
    return IngestedBlob(blob.digest, ref, size, blob.media_type, stored=True)


def _has(content_store, ref: str) -> bool:
    try:
        return bool(content_store.exists(ref))
    except Exception:
        return False


def read_blob(content_store, keys_dir, digest: str) -> bytes:
    """A blob by its OCI digest — the `/v2/<name>/blobs/<digest>` path.

    Keyed, O(1), no scan: the digest is the key. The alternative — an artifact per blob found by
    `json_extract(doc,'$.digest') = ?` — is unindexed and reads every record in the store regardless
    of LIMIT, which does not scale to a large corpus.
    """
    return get_content(content_store, keys_dir, digest_to_ref(digest))
