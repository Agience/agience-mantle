"""The registry IS the lattice — an image is a collection, a blob is content.

There is no registry service here, and that is the point. A `registry:2` container beside the
store would be a side-car holding a second copy of the one thing mantle already is: a
content-addressed store. An OCI image is not *like* what mantle keeps — it IS what mantle keeps:

    OCI                         mantle
    ─────────────────────────   ────────────────────────────────────────────────
    blob, addressed sha256:…    content, addressed `cas/<sha256 of the plaintext>`
    image manifest              an artifact whose content_ref IS the manifest blob
    repository                  a collection
    tag                         a name on an artifact inside that collection

The address is the same number. `shard/content.content_ref` is `"cas/" + sha256(plaintext).hex`
and an OCI digest is `"sha256:" + the same hex over the same bytes. So a blob is not *translated*
into the store, it is already there under the name the registry protocol asks for — which is what
makes a digest promoted from here identical to the digest anywhere else. `promotion by digest`
across two boxes is only meaningful if nothing re-derives the number, and nothing here does.

Not an artifact per blob, looked up by a `digest` field: finding it would mean
`json_extract(doc,'$.digest') = ?`, which is unindexed — it scans every record in the store
regardless of LIMIT, and on the 29 GB seed lattice that is the query shape that has already zombied
a node. The CAS lookup is keyed and O(1). A registry that cannot serve a blob without a table scan
is not a registry.

Not the S3 content path either: `services/content_service.put_bytes_encrypted` writes to the
S3 edge bucket; `fsr` has no S3 and answers "Encrypted search is not available" for the same reason.
The CAS tier is local-first and legal air-gapped (`TieredContentStore(cache, None)`), so a node can
serve its own images with nothing external — the same claim the store itself makes.
"""
from __future__ import annotations

from mantle.oci.address import (           # noqa: F401
    OciAddressError,
    digest_to_ref,
    ref_to_digest,
    parse_digest,
    is_digest,
)
