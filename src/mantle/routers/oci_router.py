"""`/v2` — the OCI distribution surface. The registry IS the lattice.

`mantle/oci/` — `address.py`, `layout.py`, `store.py` — supplies the storage, the addressing and
the verification; `tests/test_oci_store.py::test_two_separate_stores_arrive_at_the_same_digest`
carries the claim that two stores which never see each other agree on the name. This router is the
twenty lines of HTTP that let anything ask, and nothing more.

## Read-only, and by digest only

No `PUT`, no `POST`, no upload session. Push is a separate act with its own hazards — write auth,
ref-update races, garbage collection — and it is the phase where "authorization is the encryption"
has to earn its keep, because a push is a grant. It is not in this file.

And no tags, which is a property of this system rather than a limitation of this file.
Two independent reasons, either sufficient:

1. **The fleet already refuses tags.** `scripts/mantle_common.sh` rejects anything that is not
   `…@sha256:…` — ":stable resolves to different code on different days, so two boxes deployed from
   one tag a week apart do not match while every report says they do". A registry that served tags
   would be the one component in the promotion path willing to answer a question the rest of it
   treats as unanswerable.
2. **The lookup does not exist and must not be invented here.** A tag lives in an image artifact's
   context (`oci_tag`, written by `store.ImageRecord.as_context`). Finding one means
   `json_extract(doc,'$.oci_tag') = ?`, which is unindexed — `mantle/oci/__init__.py` says it
   outright: that shape "scans every record in the store regardless of LIMIT, and on the 29 GB seed
   lattice that is the query shape that has already zombied a node. A registry that cannot serve a
   blob without a table scan is not a registry."

A tag reference therefore returns `MANIFEST_UNKNOWN` **with the reason in the message**, not a bare
404 — a client that asked for `:latest` should learn why this registry has no opinion about it.

## Authorization — what this does and, more importantly, what it does not

Every route requires an authenticated caller (`get_auth`), so anonymous is 401, the same answer
`/mcp` gives. That is the floor, not a claim of fine-grained control:

**A blob is read out of the CAS by digest, and the CAS read consults no grant.** `oci.store.read_blob`
is keyed and O(1) precisely because the digest IS the key; it decrypts with the node's own content
key. So **any authenticated caller who knows a digest can read that blob.** Knowing a
content-address is not nothing — you cannot enumerate from here, there is no catalogue route and no
tag list — but it is not authorization either, and the difference is worth stating rather than
implying.

Per-image grant scoping is the honest next step and belongs with the artifact record, not with the
byte read. Until it exists, the deployment rule is: **do not put a blob in a node's CAS that its
authenticated callers may not read.** Anonymous pull — the thing that would make this a public
mirror anyone can fork from — is a further, deliberate act, and is not enabled by this file.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse

from mantle.db.backend import content_handle
from mantle.oci.address import OciAddressError, parse_digest
from mantle.oci.store import read_blob
from mantle.services.dependencies import AuthContext, get_auth, offload_sync


def _absent_exceptions() -> tuple:
    """The exception types that mean ABSENT, across both content-store shapes.

    NOT a single class, because there is no single store. `FsContentStore` raises
    `FileNotFoundError`; `TieredContentStore` raises `CacheMiss`. `CacheCorrupt` is deliberately not
    here — corrupt means present and unreadable, which must stay a 500 naming `content.key`, because
    routing it to 404 sends an investigation looking for a blob that is there.

    Imported defensively: this router must work in a minimal environment where the cache module is
    not on the path, and a missing import must not make every read a 500.
    """
    types = [FileNotFoundError, KeyError]
    try:
        from mantle.db.content_cache import CacheMiss
        types.append(CacheMiss)
    except Exception:                                  # noqa: BLE001 — absence is a legal env
        pass
    return tuple(types)


_ABSENT = _absent_exceptions()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["OCI"])

#: The header every conformant client checks on the version endpoint before it does anything else.
API_VERSION_HEADER = {"Docker-Distribution-Api-Version": "registry/2.0"}


def _oci_error(status: int, code: str, message: str, detail=None) -> JSONResponse:
    """The distribution-spec error envelope.

    A bare status code is a legal answer and a bad one: `docker pull` renders `message` to the
    operator, so this is the only place a reason reaches the person who ran the command. Every
    refusal in this file therefore carries one that names the actual cause.
    """
    body = {"errors": [{"code": code, "message": message, "detail": detail or {}}]}
    return JSONResponse(status_code=status, content=body, headers=API_VERSION_HEADER)


def _keys_dir() -> Path:
    """The same derivation `db.backend.content_handle` uses, and it must stay the same.

    Two opinions about where the keys are is how a node comes up healthy and reads nothing: content
    is encrypted under `content.key`, so a reader pointed at the wrong directory finds every blob
    missing rather than unreadable, and reports 404 for content that is present.
    """
    lattice = os.getenv("MANTLE_LATTICE_PATH")
    if os.getenv("KEYS_DIR"):
        return Path(os.environ["KEYS_DIR"])
    if lattice:
        return Path(lattice).resolve().parent / "keys"
    from mantle.db import lattice_api
    return Path(lattice_api.DEFAULT_LATTICE_PATH).parent / "keys"


@router.get("/", include_in_schema=False)
@router.get("", include_in_schema=False)
async def version_check(auth: AuthContext = Depends(get_auth)) -> Response:
    """`GET /v2/` — the endpoint every client hits first to decide the API version.

    Authenticated like everything else here. A 401 from THIS route is how a client learns it needs
    credentials, which is the documented negotiation: an anonymous probe gets 401, not a 404, and
    the difference is what tells it a registry is present at all.
    """
    return JSONResponse(status_code=200, content={}, headers=API_VERSION_HEADER)


def _resolve_reference(reference: str) -> str:
    """A reference must be a digest here. Returns the FULL `sha256:<hex>`, validated.

    `parse_digest` returns the bare HEX — it is the store-key half of the address, and
    `digest_to_ref` is what prepends `cas/`. `read_blob` takes the full digest and calls
    `digest_to_ref` itself, so handing it the hex would make every read fail with "no algorithm
    prefix" and this router would answer 404 for content it holds. Validate with `parse_digest`,
    return the canonical full form.
    """
    try:
        return "sha256:%s" % parse_digest(reference)
    except OciAddressError:
        raise OciAddressError(
            "this registry serves images by DIGEST only; %r is a tag. The fleet's own deploy path "
            "refuses tags for the same reason (a tag resolves to different code on different days), "
            "and resolving one here would need an unindexed scan of the whole store. Ask for "
            "sha256:<hex>." % reference)


async def _serve(digest_ref: str, *, head_only: bool, kind: str, repository: str = "") -> Response:
    """The one read path for both manifests and blobs — they differ only in what a client does next.

    In the distribution spec a manifest IS a blob; `read_layout` stores it as blob zero and
    `ImageRecord.digest` is its digest. Serving them through one function is not a shortcut, it is
    the shape of the thing — and it means a manifest cannot be served from a different code path
    that verifies differently.
    """
    try:
        digest = _resolve_reference(digest_ref)
    except OciAddressError as exc:
        code = "MANIFEST_UNKNOWN" if kind == "manifest" else "BLOB_UNKNOWN"
        return _oci_error(404, code, str(exc), {"reference": digest_ref})

    content_store = content_handle()
    if content_store is None:
        # An index-only node is a legal shape (`content_handle` says so). It is not a 404: the blob
        # is not missing, this node cannot serve content at all, and a client told "not found" would
        # conclude the image does not exist anywhere.
        return _oci_error(
            503, "UNAVAILABLE",
            "this node has no local content tier, so it cannot serve blobs. That is a node shape, "
            "not a missing image — ask a node that holds content.")

    try:
        body = await offload_sync(read_blob, content_store, str(_keys_dir()), digest,
                                  collection=repository or None)
    except _ABSENT:
        # `FileNotFoundError` alone would be wrong: `FsContentStore` — what the unit tests build —
        # raises `FileNotFoundError` for a missing blob, but a deployed node's `TieredContentStore`
        # raises `CacheMiss`, which is not a subclass of it, so an absent blob would fall through to
        # the generic handler and answer 500 instead of 404 on every real node — inverting the
        # "absent vs unreadable" distinction this router makes a point of preserving, in exactly the
        # deployment it matters in.
        #
        # `test_v2_serves_from_a_real_tiered_store.py` is the test that reads through the store
        # shape a node actually has; a doubles-based suite would stay green against this regression.
        return _oci_error(404, "BLOB_UNKNOWN", "this node does not hold %s" % digest,
                          {"digest": digest})
    except Exception as exc:
        # NOT flattened into a 404. A decryption failure, a wrong content key or an unreadable CAS
        # are all "present and unreadable", and reporting them as "not found" is what sends an
        # investigation to the wrong node. `content_encryption` calls a wrong key a silent partition;
        # this is where it stops being silent.
        logger.warning("oci: %s read failed for %s", kind, digest, exc_info=True)
        return _oci_error(
            500, "BLOB_UNKNOWN",
            "this node holds %s but could not read it — a wrong content.key presents exactly this "
            "way. Fingerprint the key before concluding the blob is absent." % digest,
            {"digest": digest, "error": type(exc).__name__})

    headers = dict(API_VERSION_HEADER)
    # The digest the client asked for, echoed as the spec requires. Clients verify against it, which
    # is the same check `oci.layout` runs on ingest — the content is verified at both ends and by
    # its own name in between.
    headers["Docker-Content-Digest"] = digest
    headers["Content-Length"] = str(len(body))
    headers["Etag"] = '"%s"' % digest
    # Immutable by construction: the name IS the hash, so this can never go stale.
    headers["Cache-Control"] = "max-age=31536000, immutable"

    media = ("application/vnd.oci.image.manifest.v1+json" if kind == "manifest"
             else "application/octet-stream")
    if head_only:
        # A HEAD carries the headers and no body — and `Content-Length` above is the real length,
        # not zero, because that is the number a client uses to decide whether to fetch.
        return Response(status_code=200, headers=headers, media_type=media)
    return Response(content=body, status_code=200, headers=headers, media_type=media)


@router.get("/{name:path}/manifests/{reference}", include_in_schema=False)
async def get_manifest(name: str, reference: str, request: Request,
                       auth: AuthContext = Depends(get_auth)) -> Response:
    """`GET /v2/<name>/manifests/<digest>`.

    `name` is passed down as the `collection`, and that is not a scope. `FileContentCache.path_of`
    derives the object path from the ref alone, and `content_cache.py` states the reason at length:
    per-collection keying is deliberately not used for writes, because "addressing objects globally
    while keying them per collection would let one shared ref overwrite another root's copy".

    What `collection` actually does on a read is select the legacy per-collection key to try if the
    object predates the shared-key scheme (`_decrypt` -> `was_legacy`). So passing the true
    repository is the correct value and matters for migration-era objects; it is not
    authorization, and it does not partition storage. `name` is not a lookup key, and this router
    performs no lookup by it.
    """
    return await _serve(reference, head_only=False, kind="manifest", repository=name)


@router.head("/{name:path}/manifests/{reference}", include_in_schema=False)
async def head_manifest(name: str, reference: str, request: Request,
                        auth: AuthContext = Depends(get_auth)) -> Response:
    return await _serve(reference, head_only=True, kind="manifest", repository=name)


@router.get("/{name:path}/blobs/{digest}", include_in_schema=False)
async def get_blob(name: str, digest: str, request: Request,
                   auth: AuthContext = Depends(get_auth)) -> Response:
    return await _serve(digest, head_only=False, kind="blob", repository=name)


@router.head("/{name:path}/blobs/{digest}", include_in_schema=False)
async def head_blob(name: str, digest: str, request: Request,
                    auth: AuthContext = Depends(get_auth)) -> Response:
    return await _serve(digest, head_only=True, kind="blob", repository=name)
