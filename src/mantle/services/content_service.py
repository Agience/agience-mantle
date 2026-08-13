from urllib.parse import urlparse
import hashlib
import os
import boto3
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
from botocore.config import Config
from botocore.exceptions import ClientError
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64

from mantle import config
#: `cas/` — the one content-address prefix, imported rather than respelled.
from mantle.db.constants import CAS_PREFIX

logger = logging.getLogger(__name__)


class ContentUrlSigningError(RuntimeError):
    """Raised when a signed content URL cannot be produced.

    Deliberately fatal: the signature is the access control, so a caller must
    never silently receive an unsigned link in its place.
    """


# S3 supports single PUT up to 5GB; files above that require multipart.
# This threshold is 100MB — anything larger goes multipart.
SINGLE_PUT_MAX = 100 * 1024 * 1024  # 100 MiB


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Edge store: browser-visible origin. MinIO in local/self-host deployments.
_EDGE_BUCKET = os.getenv("CONTENT_EDGE_BUCKET") or config.CONTENT_BUCKET
_EDGE_ACCESS_KEY_ID = os.getenv("CONTENT_EDGE_ACCESS_KEY_ID") or os.getenv("CONTENT_ROOT_USER")
_EDGE_SECRET_ACCESS_KEY = os.getenv("CONTENT_EDGE_SECRET_ACCESS_KEY") or os.getenv("CONTENT_ROOT_PASSWORD")
_EDGE_REGION = os.getenv("CONTENT_EDGE_REGION") or "us-east-1"
_EDGE_ENDPOINT_URL_INTERNAL = (
    os.getenv("CONTENT_EDGE_ENDPOINT_URL_INTERNAL")
    or os.getenv("AWS_ENDPOINT_URL_INTERNAL")
    or os.getenv("AWS_ENDPOINT_URL")
)
_EDGE_ENDPOINT_URL_PUBLIC = (
    os.getenv("CONTENT_EDGE_ENDPOINT_URL_PUBLIC")
    or os.getenv("AWS_ENDPOINT_URL_PUBLIC")
    or os.getenv("AWS_ENDPOINT_URL")
    or _EDGE_ENDPOINT_URL_INTERNAL
)
# Optional override for presigned URLs consumed by MCP servers.
# When servers run in a different network context than the backend, presigned
# URLs need a hostname reachable from the server's network.
_EDGE_ENDPOINT_URL_SERVER = os.getenv("CONTENT_EDGE_ENDPOINT_URL")

# Durable store: hidden persistence layer. AWS S3 in cloud deployments.
_DURABLE_BUCKET = (os.getenv("CONTENT_DURABLE_BUCKET") or "").strip()
_DURABLE_ACCESS_KEY_ID = (
    os.getenv("CONTENT_DURABLE_ACCESS_KEY_ID")
    or os.getenv("CONTENT_DURABLE_AWS_ACCESS_KEY_ID")
)
_DURABLE_SECRET_ACCESS_KEY = (
    os.getenv("CONTENT_DURABLE_SECRET_ACCESS_KEY")
    or os.getenv("CONTENT_DURABLE_AWS_SECRET_ACCESS_KEY")
)
_DURABLE_REGION = (
    os.getenv("CONTENT_DURABLE_REGION")
    or os.getenv("CONTENT_DURABLE_AWS_REGION")
    or "ca-central-1"
)
_DURABLE_ENDPOINT_URL = (
    os.getenv("CONTENT_DURABLE_ENDPOINT_URL")
    or os.getenv("CONTENT_DURABLE_AWS_ENDPOINT_URL")
    or None
)
_EVICT_EDGE_AFTER_DURABLE_SYNC = _env_flag("CONTENT_EDGE_EVICT_AFTER_DURABLE_SYNC", default=False)


def get_content_storage_mode() -> str:
    """Return the effective storage topology used for content artifacts."""
    cloudfront_enabled = bool(
        os.getenv("CLOUDFRONT_KEY_ID")
        and (os.getenv("CLOUDFRONT_PRIVATE_KEY") or os.getenv("CLOUDFRONT_PRIVATE_KEY_PATH"))
    )
    if cloudfront_enabled and _durable_store_enabled():
        return "cloudfront-s3"
    if _durable_store_enabled():
        return "minio-s3-backed"
    return "minio-only"


def _make_s3_client(
    *,
    access_key_id: Optional[str],
    secret_access_key: Optional[str],
    region_name: str,
    endpoint_url: Optional[str] = None,
):
    config = Config(s3={"addressing_style": "path"}) if endpoint_url else None
    kwargs = {
        "service_name": "s3",
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_access_key,
        "region_name": region_name,
    }
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if config is not None:
        kwargs["config"] = config
    return boto3.client(**kwargs)


_s3_edge_internal = _make_s3_client(
    access_key_id=_EDGE_ACCESS_KEY_ID,
    secret_access_key=_EDGE_SECRET_ACCESS_KEY,
    region_name=_EDGE_REGION,
    endpoint_url=_EDGE_ENDPOINT_URL_INTERNAL,
)
_s3_edge_public = _make_s3_client(
    access_key_id=_EDGE_ACCESS_KEY_ID,
    secret_access_key=_EDGE_SECRET_ACCESS_KEY,
    region_name=_EDGE_REGION,
    endpoint_url=_EDGE_ENDPOINT_URL_PUBLIC,
)
_s3_edge_server = (
    _make_s3_client(
        access_key_id=_EDGE_ACCESS_KEY_ID,
        secret_access_key=_EDGE_SECRET_ACCESS_KEY,
        region_name=_EDGE_REGION,
        endpoint_url=_EDGE_ENDPOINT_URL_SERVER,
    )
    if _EDGE_ENDPOINT_URL_SERVER
    else None
)
_s3_durable = (
    _make_s3_client(
        access_key_id=_DURABLE_ACCESS_KEY_ID,
        secret_access_key=_DURABLE_SECRET_ACCESS_KEY,
        region_name=_DURABLE_REGION,
        endpoint_url=_DURABLE_ENDPOINT_URL,
    )
    if _DURABLE_BUCKET
    else None
)

_BUCKET_CHECKED: bool = False
_BUCKET_WARNING_EMITTED: bool = False
_CORS_APPLIED: bool = False
_CORS_UNSUPPORTED: bool = False


def reinit_edge_clients() -> None:
    """Re-create edge S3 clients using credentials from key_manager and current config.

    Called at startup after key initialization, and again after platform settings are
    loaded from the DB, so the endpoint URL and credentials always reflect the live
    config rather than the module-level env-var snapshot.
    """
    global _s3_edge_internal, _s3_edge_public, _s3_edge_server, _BUCKET_CHECKED, _CORS_APPLIED, _CORS_UNSUPPORTED
    try:
        from prism.trust.key_manager import get_minio_pass
        secret_key = get_minio_pass()
    except RuntimeError:
        return
    access_key = os.getenv("MINIO_ROOT_USER") or os.getenv("CONTENT_ROOT_USER") or "agience"

    # Use explicit env-var overrides first; fall back to config.CONTENT_URI so that
    # the endpoint is always populated after load_settings_from_db() runs.
    endpoint_internal = (
        _EDGE_ENDPOINT_URL_INTERNAL
        or config.CONTENT_URI
        or None
    )
    endpoint_public = (
        _EDGE_ENDPOINT_URL_PUBLIC
        or config.CONTENT_URI
        or None
    )

    _s3_edge_internal = _make_s3_client(
        access_key_id=access_key,
        secret_access_key=secret_key,
        region_name=_EDGE_REGION,
        endpoint_url=endpoint_internal,
    )
    _s3_edge_public = _make_s3_client(
        access_key_id=access_key,
        secret_access_key=secret_key,
        region_name=_EDGE_REGION,
        endpoint_url=endpoint_public,
    )
    # Server-facing client for presigned URLs consumed by MCP servers in a
    # different network context.  Only created when CONTENT_EDGE_ENDPOINT_URL
    # is set.
    if _EDGE_ENDPOINT_URL_SERVER:
        _s3_edge_server = _make_s3_client(
            access_key_id=access_key,
            secret_access_key=secret_key,
            region_name=_EDGE_REGION,
            endpoint_url=_EDGE_ENDPOINT_URL_SERVER,
        )
    else:
        _s3_edge_server = None
    # Reset so the next operation re-checks (and creates if needed) the bucket
    # with the freshly configured clients.
    _BUCKET_CHECKED = False
    _CORS_APPLIED = False
    _CORS_UNSUPPORTED = False
    # A presence memo holds against one endpoint only; it says nothing about a different one.
    forget_edge_presence()
    logger.info("Edge S3 clients reinitialized (endpoint: %s)", endpoint_internal)


def _apply_bucket_cors() -> None:
    """Apply a permissive CORS policy so browsers can PUT presigned upload requests directly."""
    global _CORS_APPLIED, _CORS_UNSUPPORTED
    if _CORS_APPLIED or _CORS_UNSUPPORTED:
        return
    try:
        _s3_edge_internal.put_bucket_cors(
            Bucket=_EDGE_BUCKET,
            CORSConfiguration={
                "CORSRules": [
                    {
                        "AllowedOrigins": ["*"],
                        "AllowedMethods": ["GET", "PUT", "POST", "DELETE", "HEAD"],
                        "AllowedHeaders": ["*"],
                        "ExposeHeaders": ["ETag", "Content-Length", "Content-Type"],
                        "MaxAgeSeconds": 3600,
                    }
                ]
            },
        )
        _CORS_APPLIED = True
        logger.info("Applied CORS policy to content bucket '%s'", _EDGE_BUCKET)
    except ClientError as exc:
        error_code = (exc.response or {}).get("Error", {}).get("Code")
        if error_code == "NotImplemented":
            _CORS_UNSUPPORTED = True
            logger.info(
                "Skipping bucket CORS apply for '%s': storage endpoint does not support PutBucketCors",
                _EDGE_BUCKET,
            )
            return
        logger.warning("Could not apply CORS policy to content bucket '%s': %s", _EDGE_BUCKET, exc)
    except Exception as exc:
        logger.warning("Could not apply CORS policy to content bucket '%s': %s", _EDGE_BUCKET, exc)


def _ensure_bucket_exists_once() -> None:
    """Ensure the edge content bucket exists (creating it if needed), then apply CORS."""
    global _BUCKET_CHECKED, _BUCKET_WARNING_EMITTED
    if _BUCKET_CHECKED:
        return
    if not edge_store_configured():
        # No object store is a configuration, not a fault (`db/content_tier.py`: `remote=None` is
        # first-class). Probing for a bucket on a node that deliberately has none produced a
        # "Unable to locate credentials" warning at every boot, which is noise about a decision.
        _BUCKET_CHECKED = True
        return

    try:
        _s3_edge_internal.head_bucket(Bucket=_EDGE_BUCKET)
        _BUCKET_CHECKED = True
        _apply_bucket_cors()
        return
    except Exception:
        pass

    try:
        kwargs: dict = {"Bucket": _EDGE_BUCKET}
        if _EDGE_REGION and _EDGE_REGION != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": _EDGE_REGION}
        _s3_edge_internal.create_bucket(**kwargs)
        logger.info("Created content bucket '%s'", _EDGE_BUCKET)
        _apply_bucket_cors()
    except Exception as exc:
        if not _BUCKET_WARNING_EMITTED:
            logger.warning(
                "Could not access or create content bucket '%s': %s",
                _EDGE_BUCKET,
                exc,
            )
            _BUCKET_WARNING_EMITTED = True
    finally:
        _BUCKET_CHECKED = True


def ensure_content_bucket() -> None:
    """Public entry point: (re-)check and create the content bucket.

    Call this after reinit_edge_clients() to eagerly provision the bucket
    at setup / startup rather than waiting for the first upload.
    """
    global _BUCKET_CHECKED, _BUCKET_WARNING_EMITTED
    _BUCKET_CHECKED = False
    _BUCKET_WARNING_EMITTED = False
    _ensure_bucket_exists_once()


# ── the two tiers this module writes through ─────────────────────────────────────────────────
#
# ONE content path, two tiers — the shape `db/content_tier.py` already implements for ingest and
# `shard/content_tier.py` drains: the node's own encrypted CAS in front of an object store, with
# `remote=None` (no object store at all) a first-class configuration rather than a degraded one.
# This module is the REST leg of that same path; it does not build a second one.
#
#   local  — `db.backend.content_handle()`: the process-wide `TieredContentStore` over
#            `<store root>/cas` (`AGIENCE_BASE_DIR/.data/cas` unless `MANTLE_LATTICE_PATH` moves
#            the store). Verify-on-read, atomic write, AES-256-GCM at rest under the node's
#            `shared_content_key`, AAD = the ref.
#   object — the edge bucket above (MinIO/S3), and the durable bucket behind it. Optional.
#
# What each tier holds is the SAME envelope: `put_bytes_encrypted` produces `MEC1‖nonce‖ct` under
# the owner's per-principal key (`services/content_crypto.py`) and BOTH tiers store exactly those
# bytes. The local tier then encrypts that ciphertext again at rest under the node key, and
# `TieredContentStore.promote_one` re-encrypts it under the shared fleet cipher on its way to a
# mirror — so the envelope is never opened, never re-formed, and never weakened by the tiering.
# A node with no object store loses no property of the envelope; it loses a copy.


def edge_store_configured() -> bool:
    """Does this node actually have an object store to write to?

    The clients above are built unconditionally — boto3 constructs a client without asking for
    credentials and only fails at call time, with `NoCredentialsError` — so their existence is not
    the answer. The request signer's RESOLVED credentials are: they are what the next `put_object`
    would sign with, and they cover the ambient chain (env vars, profile, instance role) as well as
    the explicit keys wired at import and by `reinit_edge_clients`.

    `None` means every object-store call on this node raises. That is a CONFIGURATION, not a
    failure — the same reading `db/content_tier.py` gives `remote=None` — so callers skip the tier
    quietly instead of warning on every write.
    """
    try:
        return _s3_edge_internal._request_signer._credentials is not None
    except AttributeError:                # a stubbed/faked client in tests: assume it is usable
        return True


_LOCAL_TIER_ABSENT_LOGGED = False


def _lattice_root_and_keys() -> Tuple[Path, Path]:
    """`(<store root>, <keys dir>)`, resolved exactly as `db.backend.content_handle` resolves them.

    Both defaults come from `config` (`DEFAULT_LATTICE_PATH` is re-exported through
    `db.lattice_api`, which is where `db.backend` reads it), so this never recomputes what
    `AGIENCE_BASE_DIR` means — it reads the value the rest of the node is using. A node that moves
    its store with `MANTLE_LATTICE_PATH` moves its CAS with it, in one place.
    """
    from mantle.db import lattice_api as _lattice
    db_path = Path(os.path.abspath(os.path.expanduser(
        os.getenv("MANTLE_LATTICE_PATH", str(_lattice.DEFAULT_LATTICE_PATH)))))
    root = db_path.parent
    return root, Path(os.getenv("KEYS_DIR") or (root / "keys"))


def _bootstrap_local_content() -> None:
    """Make the local CAS openable: the `cas/` directory, and a `content.key` to derive its
    at-rest key from.

    THE WRITE PATH ONLY, and the rule is `shard/content.py`'s, reused rather than restated: a write
    on a provisioned node may bootstrap the first content key (`_content_key(create=True)` —
    exclusive-create, so concurrent writers cannot mint two), and NOTHING may create the keys
    directory. An absent keys dir means the volume is not mounted, and minting a key into a fresh
    directory is how a node silently partitions itself from content it already holds. Reads never
    come through here: a read that finds no key has a configuration fault and must say so.
    """
    root, keys_dir = _lattice_root_and_keys()
    if not keys_dir.is_dir():
        return
    from mantle.shard.content import _content_key
    _content_key(keys_dir, create=True)
    (root / "cas").mkdir(parents=True, exist_ok=True)


def local_content_tier(*, bootstrap: bool = False):
    """This node's `TieredContentStore`, or None when it genuinely has no local content tier.

    The SAME handle the ingest path writes and `shard/content_tier.promote_local_content` drains —
    `db.backend.content_handle()`, opened once per process. `bootstrap=True` is passed by the write
    path only (see :func:`_bootstrap_local_content`).

    None is returned rather than raised for the one case this module has to survive: a node whose
    keys volume is not mounted still has an object store to fall back to, and the caller decides
    whether losing the local tier is fatal. It is logged once, not once per write.
    """
    global _LOCAL_TIER_ABSENT_LOGGED
    import mantle.db.backend as _backend
    if bootstrap:
        try:
            _bootstrap_local_content()
        except Exception:
            logger.debug("local content tier could not be bootstrapped", exc_info=True)
    try:
        tier = _backend.content_handle()
    except Exception as exc:
        tier = None
        if not _LOCAL_TIER_ABSENT_LOGGED:
            logger.info("no local content tier on this node (%s) — content goes to the object "
                        "store only", exc)
            _LOCAL_TIER_ABSENT_LOGGED = True
    return tier


def local_content_has(ref: str) -> bool:
    """Is `ref` already in this node's local CAS? Used to make a re-upload a no-op rather than a
    duplicate. False on any failure — an unanswerable tier must not read as a hit."""
    if not is_cas_ref(ref):
        return False
    tier = local_content_tier()
    cache = getattr(tier, "cache", None) if tier is not None else None
    try:
        return cache is not None and ref in cache
    except Exception:
        return False


def is_cas_ref(ref) -> bool:
    """`cas/<64 lowercase hex>` and nothing else.

    A ref reaches this module out of an artifact's context, which a caller can write, so it is
    VALIDATED and not trusted — the same discipline as `workspace_service._safe_content_key`. This
    is the shape check; the envelope is the authorization (see :func:`get_bytes_decrypted`).
    """
    if not isinstance(ref, str) or not ref.startswith(CAS_PREFIX):
        return False
    h = ref[len(CAS_PREFIX):]
    return len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def cas_ref_for(blob: bytes) -> str:
    """The content address of `blob` — `cas/sha256(blob)`, the corpus's one addressing scheme."""
    return CAS_PREFIX + hashlib.sha256(blob).hexdigest()


class ContentStoreUnavailable(RuntimeError):
    """No tier could take (or answer for) this content: no local CAS *and* no object store.

    Distinct from a missing object: this says the node cannot serve content at all, which is a
    provisioning fault, and it names both tiers so the remedy is readable from the message.
    """


class ContentEncryptionError(RuntimeError):
    """Raised when S3 object content cannot be encrypted for storage."""


class ContentDecryptionError(RuntimeError):
    """Raised when a stored S3 object cannot be decrypted for a read."""


def mirror_failure_is_transient(exc: BaseException) -> bool:
    """Would repeating this mirror write, unchanged, plausibly get a different answer?

    The classification is HTTP's own, not a list of error strings this module curates:

      * **No answer at all** — anything that is not a `ClientError` carries no `response`, because
        botocore never got one: the endpoint did not resolve, the connection was refused, the
        socket timed out or was reset. Nothing about the request was rejected, because nothing
        about it was read. Transient.
      * **An answer** — a `ClientError` carries the store's own HTTP status. RFC 9110 §15.6 makes
        5xx "the server failed to fulfil an apparently valid request", which is the definition of
        retryable; §15.5 makes 4xx a fault in the request, and this request is byte-identical on
        every retry (the body is content-addressed), so an identical answer is the only one it can
        get. `AccessDenied`, `InvalidAccessKeyId`, `SignatureDoesNotMatch`, `NoSuchBucket` all land
        there and none of them is fixed by waiting.
      * **408 and 429** are the two 4xx codes whose own definitions license a repeat (§15.5.9 "the
        client MAY repeat the request"; RFC 6585 §4 pairs 429 with `Retry-After`). They are read
        out of the protocol, not chosen as a policy.

    Undecidable cases resolve to transient. The cost of a task nobody can complete is one visible
    row carrying the reason it fails; the cost of dropping the obligation is content that is
    silently unreachable from every peer, which is the whole gap this record exists to close.

    Note that boto3 has already applied its own retry policy before raising, so what reaches here
    is a failure that survived the in-request retries. That is exactly why it needs a durable
    record rather than another immediate attempt.
    """
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return True
    status = (resp.get("ResponseMetadata") or {}).get("HTTPStatusCode")
    try:
        status = int(status)
    except (TypeError, ValueError):
        return True
    return status >= 500 or status in (408, 429)


def put_bytes_encrypted(content_key: str, data: bytes, content_type: str,
                        owner_id: Optional[str], *,
                        collection_id: Optional[str] = None,
                        cas: bool = True,
                        on_mirror_deferred=None) -> Optional[str]:
    """Write raw bytes envelope-encrypted for *owner_id*, to every tier this node has.

    Returns the local CAS ref (``cas/<sha256 of the envelope>``) when the bytes landed locally,
    else None. The caller records it: it is how a read finds the local copy, and it is the only
    thing the two tiers do not share (the object store is addressed by *content_key*).

    **Local first, mirror second — writes do not block on the WAN.** This is the tier's own
    discipline ("ingest never blocks on the WAN", `shard/content_tier.py`), now applied to the REST
    leg. A node with no object store writes only locally and that is a complete, successful write,
    not a degraded one; there is no promotion queue standing behind it, so nothing accumulates.
    A node with an object store still writes to it, exactly as before.

    **The envelope is unchanged and identical on both tiers.** Both hold the same
    ``MEC1‖nonce‖ct`` bytes under the owner's per-principal key. The local tier encrypts that
    ciphertext AGAIN at rest under the node key with the ref as AAD, so the local copy is strictly
    better protected than the object-store copy, never worse. If encryption fails, this raises and
    NOTHING is written to either tier: failing a write is recoverable — the caller retries once the
    oracle is back — while silently persisting plaintext is not.

    **The local tier requires an owner.** With no *owner_id* there is no envelope, and an
    unenveloped object in a globally content-addressed CAS is reachable by anyone who can name its
    address. That case (legacy, and vanishing) keeps its old behaviour: the object store only.

    ``cas=False`` says the caller already holds the authoritative copy elsewhere and will not
    record a ref — :func:`put_text_direct`, whose bytes also live envelope-encrypted inside the
    artifact document. Writing a CAS object nobody references would be a leak, not a tier, so that
    caller mirrors to the object store only. It is the same one path either way; the flag says who
    owns the reference.

    Buffered, not streamed: `data` is already whole in memory, the AEAD is one-shot, and the CAS
    write is one atomic file. The bound is therefore AES-GCM's own — a plaintext over
    ``2**31 - 1`` bytes raises ``OverflowError`` out of the cipher, which the caller surfaces as a
    413 rather than as a server fault. No limit is invented here.

    ``on_mirror_deferred`` is how a caller learns that the mirror leg is still OWED, which is the
    one thing the return value cannot say: a ref means "the bytes are local", on a node with no
    object store and on a node whose mirror is down alike. It is called ``(ref, exc)`` when, and
    only when, the bytes landed locally, an object store IS configured, its write raised, and
    :func:`mirror_failure_is_transient` says a retry could still land it. A node with no object
    store never reaches the call — the early return above is the guard, so "nothing to mirror"
    cannot be mistaken for "the mirror failed" by any predicate someone forgets to write. The
    callback is advisory bookkeeping over a write that has already succeeded, so it is never
    allowed to turn that success into a failure; omitting it changes nothing.

    Raises `ContentStoreUnavailable` when neither tier could take the bytes.
    """
    body = data
    if owner_id:
        try:
            from mantle.services import content_crypto
            body = content_crypto.encrypt_content(owner_id, data)
        except OverflowError:
            raise                      # the cipher's own size bound — surfaced, not relabelled
        except Exception as exc:
            logger.error(
                "content encryption failed for %s — refusing to store plaintext",
                content_key, exc_info=True,
            )
            raise ContentEncryptionError(
                f"content encryption unavailable for {content_key!r}; refusing to "
                f"persist plaintext to the object store"
            ) from exc

    ref: Optional[str] = None
    local_error: Optional[Exception] = None
    if cas and owner_id:
        tier = local_content_tier(bootstrap=True)
        if tier is not None:
            candidate = cas_ref_for(body)
            try:
                tier.put(candidate, body, collection=collection_id)
                ref = candidate
            except Exception as exc:               # a local failure must not hide a usable mirror
                local_error = exc
                logger.warning("local CAS write failed for %s (%s: %s)",
                               content_key, type(exc).__name__, exc)

    if not edge_store_configured():
        if ref is None:
            raise ContentStoreUnavailable(
                f"nothing on this node can store {content_key!r}: no object store is configured "
                f"(no credentials resolve for the edge bucket) and the local content tier is "
                f"unavailable"
                + (f" ({type(local_error).__name__}: {local_error})" if local_error else
                   " — no keys directory, so no content key and no local CAS")
            )
        return ref                     # air-gapped node: local IS the store. Quiet, and complete.

    _ensure_bucket_exists_once()
    try:
        _s3_edge_internal.put_object(Bucket=_EDGE_BUCKET, Key=content_key, Body=body,
                                     ContentType=content_type)
    except Exception as exc:
        if ref is None:
            raise                      # no local copy: the caller must hear that nothing landed
        # The bytes ARE stored, verified and readable, on this node. Failing the request now would
        # discard a durable write to report an unreachable mirror — the exact coupling the tier
        # exists to break. Loud in the log, successful to the caller.
        logger.warning("mirror write failed for %s; the local copy at %s is authoritative",
                       content_key, ref, exc_info=True)
        if not mirror_failure_is_transient(exc):
            # Named as its own outcome. The bytes are just as unreachable from a peer either way,
            # but only one of the two is waiting on the network — this one is waiting on a person,
            # and queueing a retry for it would just re-ask a question already answered.
            logger.error("the object store REFUSED %s (%s: %s) — a retry of the identical write "
                         "gets the identical answer, so no mirror retry is recorded; the local "
                         "copy at %s is authoritative and this content is not reachable from a "
                         "peer until the store's configuration is fixed",
                         content_key, type(exc).__name__, exc, ref)
        elif on_mirror_deferred is not None:
            try:
                on_mirror_deferred(ref, exc)
            except Exception:
                # The write succeeded and is durable. Losing the note that the mirror is owed is
                # bad; turning a stored, verified, readable object into a 500 to report that we
                # could not write the note is worse.
                logger.warning("could not record that %s still owes the mirror a copy — the bytes "
                               "are stored at %s", content_key, ref, exc_info=True)
        return ref
    _remember_edge_presence(content_key)   # a write we just made is a presence we already know
    return ref


def get_bytes_decrypted(content_key: str, owner_id: Optional[str], *,
                        cas_ref: Optional[str] = None,
                        collection_id: Optional[str] = None) -> bytes:
    """Read the bytes back, decrypting for *owner_id*. Local CAS first, object store behind it.

    *cas_ref* is what :func:`put_bytes_encrypted` returned, carried on the artifact. When it names
    an object this node holds, the read is served locally — verified against its own content
    address on the way out by `FileContentCache`, and, on a node that has a mirror, pulled through
    and sha256-verified by `TieredContentStore` when the local copy is missing. A node with a local
    copy and an unreachable mirror never touches the mirror at all and answers normally.

    **The local leg demands an envelope.** `cas_ref` arrives from caller-writable context, and the
    CAS is one global address space shared with everything else this node has ever stored — so a
    ref that opens to something this route did not write must not be served. `require_encrypted`
    refuses any object without the ``MEC1`` magic (every corpus object written by ingest, for
    instance), and the envelope's own key and AAD refuse anything enveloped for another principal.
    Shape is checked too, so a ref cannot address a path. Together that is the light cone: naming
    someone else's address yields an error, never their bytes.

    On the object-store leg, legacy plaintext objects (no ``MEC1`` magic) still pass through
    unchanged — those genuinely predate envelope encryption and that is not a failure path.

    If decryption fails, this raises: the caller gets an error instead of ciphertext dressed as
    plaintext.
    """
    if cas_ref and owner_id and is_cas_ref(cas_ref):
        tier = local_content_tier()
        if tier is not None:
            try:
                blob = tier.get(cas_ref, collection=collection_id)
            except Exception as exc:
                # Absent locally and no mirror to pull from, or a mirror that did not answer. The
                # object store below may still hold it under `content_key`, so this is a miss to
                # fall through on — but a decrypt failure below is NOT, and does not reach here.
                logger.info("local CAS read missed for %s (%s: %s)",
                            cas_ref, type(exc).__name__, exc)
            else:
                return _decrypt_envelope(cas_ref, blob, owner_id, require_encrypted=True)

    if not content_key or not edge_store_configured():
        # `content_key` is the object store's whole address; without one there is nothing to ask
        # for, and asking anyway would turn a "this node holds no copy" into a parameter error.
        raise ContentStoreUnavailable(
            f"{content_key or cas_ref!r} is not in this node's local content tier and there is no "
            f"object-store address to look up"
            + ("" if content_key else " (the artifact records no content_key)")
        )
    response = _s3_edge_internal.get_object(Bucket=_EDGE_BUCKET, Key=content_key)
    raw = response["Body"].read()
    if owner_id:
        return _decrypt_envelope(content_key, raw, owner_id)
    return raw


def _decrypt_envelope(name: str, raw: bytes, owner_id: str, *,
                      require_encrypted: bool = False) -> bytes:
    """Open the per-principal envelope, or raise. One implementation for both tiers, so neither
    can drift into returning ciphertext the other would refuse."""
    try:
        from mantle.services import content_crypto
        return content_crypto.decrypt_content(owner_id, raw, require_encrypted=require_encrypted)
    except Exception as exc:
        logger.error("failed to decrypt stored content %s", name, exc_info=True)
        raise ContentDecryptionError(
            f"stored object {name!r} could not be decrypted; refusing to "
            f"return ciphertext as content"
        ) from exc


def put_text_direct(content_key: str, text: str, content_type: str = "text/plain", *, owner_id: Optional[str] = None) -> None:
    """Write a text/bytes payload directly to the edge store (server-side, no presigned URL).

    Used when the backend itself is the uploader — e.g. auto-migrating inline
    artifact content to S3 on create/update. Envelope-encrypted per ``owner_id``.

    ``cas=False``: this content's authoritative copy is the artifact document itself, which
    `db/doc_boundary.py` envelope-encrypts on the same wire format. The caller records no CAS ref,
    so writing one would leave an object nothing points at. It raises on a node with no object
    store, exactly as it did before, and its caller keeps the content inline — the degradation this
    path has always had, and the reason it needs no local tier of its own.
    """
    data = text.encode("utf-8") if isinstance(text, str) else text
    put_bytes_encrypted(content_key, data, content_type, owner_id, cas=False)


def get_text_direct(content_key: str, *, owner_id: Optional[str] = None) -> str:
    """Fetch a text object directly from the edge store (server-side, no presigned URL).

    Used by agents and MCP server tools that need the raw content of an artifact
    (operator config JSON, authorizer config, etc.) when artifact.content is empty
    because it was stored in S3. Decrypts per ``owner_id``; legacy plaintext passes through.
    """
    return get_bytes_decrypted(content_key, owner_id).decode("utf-8")


def _durable_store_enabled() -> bool:
    return bool(_DURABLE_BUCKET and _s3_durable is not None)


def _copy_between_clients(src_client, src_bucket: str, dst_client, dst_bucket: str, key: str) -> None:
    source = src_client.get_object(Bucket=src_bucket, Key=key)
    body = source["Body"]
    try:
        put_kwargs = {
            "Bucket": dst_bucket,
            "Key": key,
            "Body": body,
        }
        content_type = source.get("ContentType")
        cache_control = source.get("CacheControl")
        if content_type:
            put_kwargs["ContentType"] = content_type
        if cache_control:
            put_kwargs["CacheControl"] = cache_control
        dst_client.put_object(**put_kwargs)
    finally:
        try:
            body.close()
        except Exception:
            pass


# ── edge-object presence memo ────────────────────────────────────────────────
#
# Every presign HEADs the edge bucket before signing, which doubles the S3 round trips on the
# download path. The bucket-existence check beside it (`_BUCKET_CHECKED`) is already memoized;
# this is the same memo one level down, per object key.
#
# Only PRESENCE is remembered. An absence is what triggers hydration from durable storage, so
# caching it would pin a key as missing for as long as the entry lived and defeat the hydration
# it exists to trigger.
#
# Bounded two ways, because a cached "present" for an object that has since been deleted signs a
# URL that 404s:
#   * TTL — an entry expires, so the wrongness is a short window rather than a wedged path.
#   * Size — an LRU cap, so a long-running process cannot grow this without limit.
# Every path in this module that removes an edge object also forgets it here, so the TTL is a
# backstop for deletions that happen OUTSIDE this process, not the primary mechanism.
_EDGE_PRESENT_TTL_S: float = 300.0
_EDGE_PRESENT_MAX: int = 4096
_edge_present: "OrderedDict[str, float]" = OrderedDict()
_edge_present_lock = threading.Lock()


def _edge_presence_cached(key: str) -> bool:
    now = time.monotonic()
    with _edge_present_lock:
        expires_at = _edge_present.get(key)
        if expires_at is None:
            return False
        if expires_at <= now:
            _edge_present.pop(key, None)
            return False
        _edge_present.move_to_end(key)
        return True


def _remember_edge_presence(key: str) -> None:
    with _edge_present_lock:
        _edge_present[key] = time.monotonic() + _EDGE_PRESENT_TTL_S
        _edge_present.move_to_end(key)
        while len(_edge_present) > _EDGE_PRESENT_MAX:
            _edge_present.popitem(last=False)


def forget_edge_presence(key: Optional[str] = None) -> None:
    """Drop the memo for one key, or all of it when *key* is None.

    Called wherever an edge object stops being present — deletion, eviction after a durable
    sync — and wherever the clients themselves are rebuilt, since a memo taken against one
    endpoint says nothing about another."""
    with _edge_present_lock:
        if key is None:
            _edge_present.clear()
        else:
            _edge_present.pop(key, None)


def ensure_edge_object_present(key: str) -> bool:
    if _edge_presence_cached(key):
        return True

    try:
        _s3_edge_internal.head_object(Bucket=_EDGE_BUCKET, Key=key)
        _remember_edge_presence(key)
        return True
    except Exception:
        pass

    if not _durable_store_enabled():
        return False

    try:
        _ensure_bucket_exists_once()
        _copy_between_clients(_s3_durable, _DURABLE_BUCKET, _s3_edge_internal, _EDGE_BUCKET, key)
        logger.info("Hydrated edge content from durable store for key=%s", key)
        _remember_edge_presence(key)
        return True
    except Exception as exc:
        logger.warning("Failed to hydrate edge content for key=%s: %s", key, exc)
        return False


def persist_object_to_durable(key: str) -> bool:
    if not _durable_store_enabled():
        return False

    _copy_between_clients(_s3_edge_internal, _EDGE_BUCKET, _s3_durable, _DURABLE_BUCKET, key)
    logger.info("Persisted content to durable store for key=%s", key)
    if _EVICT_EDGE_AFTER_DURABLE_SYNC:
        _s3_edge_internal.delete_object(Bucket=_EDGE_BUCKET, Key=key)
        forget_edge_presence(key)
        logger.info("Evicted edge content after durable sync for key=%s", key)
    return True


def build_public_content_url(file_id: str, filename: str, tenant_domain: Optional[str] = None) -> str:
    """Build a public content URL for agience-hosted public files.

    Always derives the base from CONTENT_URI -- supports both single-domain
    path-prefix deployments (https://domain.com/content) and subdomain
    deployments (https://content.domain.com). The tenant_domain parameter
    is deprecated and ignored.
    """
    safe_name = filename.replace("/", "_")
    base = config.CONTENT_URI.rstrip("/")
    return f"{base}/files/{file_id}/{safe_name}"


def generate_signed_url(
    key: str,
    expires_in: Optional[int] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
    as_attachment: bool = False,
    server_facing: bool = False,
) -> str:
    """Generate a signed content URL from the edge origin, hydrating from durable storage when needed.

    When *server_facing* is True the URL is generated using the server-facing
    S3 client so that MCP servers in a different network context can reach the
    content store.  Browser callers should use the default (public) URL.
    """
    if expires_in is None:
        expires_in = config.CONTENT_DOWNLOAD_URL_EXPIRY
    effective_content_type = content_type
    cloudfront_key_id = os.getenv("CLOUDFRONT_KEY_ID")
    cloudfront_private_key = os.getenv("CLOUDFRONT_PRIVATE_KEY")
    cloudfront_private_key_path = os.getenv("CLOUDFRONT_PRIVATE_KEY_PATH")

    if cloudfront_key_id and (cloudfront_private_key or cloudfront_private_key_path):
        try:
            return _generate_cloudfront_signed_url(
                key,
                expires_in,
                cloudfront_key_id,
                cloudfront_private_key,
                cloudfront_private_key_path,
                filename,
                effective_content_type,
            )
        except Exception as exc:
            logger.warning("Error generating CloudFront signed URL for %s: %s", key, exc)

    try:
        if not ensure_edge_object_present(key):
            logger.info("Edge object missing for key=%s and could not be hydrated before signing", key)
        params = {
            "Bucket": _EDGE_BUCKET,
            "Key": key,
        }

        if filename and as_attachment:
            params["ResponseContentDisposition"] = f'attachment; filename="{filename}"'
        if effective_content_type:
            if effective_content_type.startswith("text/") and "charset" not in effective_content_type.lower():
                params["ResponseContentType"] = f"{effective_content_type}; charset=utf-8"
            else:
                params["ResponseContentType"] = effective_content_type

        s3_client = _s3_edge_public
        if server_facing:
            s3_client = _s3_edge_server or _s3_edge_internal
        return s3_client.generate_presigned_url(
            "get_object",
            Params=params,
            ExpiresIn=expires_in,
        )
    except Exception as exc:
        logger.error("Error generating edge presigned URL for %s: %s", key, exc)
        raise ContentUrlSigningError(
            f"could not generate a signed content URL for {key}"
        ) from exc


def _generate_cloudfront_signed_url(
    key: str,
    expires_in: int,
    key_id: str,
    private_key_str: Optional[str] = None,
    private_key_path: Optional[str] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """Generate CloudFront signed URL using RSA-SHA1 signature."""
    if private_key_str:
        private_key_bytes = private_key_str.encode("utf-8")
    elif private_key_path:
        with open(private_key_path, "rb") as f:
            private_key_bytes = f.read()
    else:
        raise ValueError("Either CLOUDFRONT_PRIVATE_KEY or CLOUDFRONT_PRIVATE_KEY_PATH must be set")

    from cryptography.hazmat.backends import default_backend

    private_key = serialization.load_pem_private_key(
        private_key_bytes,
        password=None,
        backend=default_backend(),
    )

    u = urlparse(config.CONTENT_URI)
    scheme = u.scheme or "https"
    host = u.netloc or u.path
    resource_url = f"{scheme}://{host}/{key}"

    expire_time = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    expire_timestamp = int(expire_time.timestamp())

    policy = {
        "Statement": [{
            "Resource": resource_url,
            "Condition": {
                "DateLessThan": {
                    "AWS:EpochTime": expire_timestamp,
                }
            },
        }]
    }

    import json

    policy_json = json.dumps(policy, separators=(",", ":"))

    signature = private_key.sign(  # type: ignore[union-attr]
        policy_json.encode("utf-8"),
        padding.PKCS1v15(),  # type: ignore[call-arg]
        hashes.SHA1(),  # type: ignore[call-arg]
    )

    signature_b64 = base64.b64encode(signature).decode("utf-8")
    signature_b64 = signature_b64.replace("+", "-").replace("=", "_").replace("/", "~")

    policy_b64 = base64.b64encode(policy_json.encode("utf-8")).decode("utf-8")
    policy_b64 = policy_b64.replace("+", "-").replace("=", "_").replace("/", "~")

    return f"{resource_url}?Policy={policy_b64}&Signature={signature_b64}&Key-Pair-Id={key_id}"


def presign_put_or_multipart(key: str, content_type: str, size: int):
    _ensure_bucket_exists_once()
    if size <= SINGLE_PUT_MAX:
        url = _s3_edge_public.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": _EDGE_BUCKET,
                "Key": key,
                "ContentType": content_type,
                "CacheControl": "private, max-age=31536000, immutable",
            },
            ExpiresIn=config.CONTENT_UPLOAD_URL_EXPIRY,
        )
        return {"mode": "put", "url": url}
    resp = _s3_edge_internal.create_multipart_upload(
        Bucket=_EDGE_BUCKET,
        Key=key,
        ContentType=content_type,
        CacheControl="private, max-age=31536000, immutable",
    )
    return {"mode": "multipart", "uploadId": resp["UploadId"]}


def generate_multipart_part_url(key: str, upload_id: str, part_number: int, expires_in: Optional[int] = None) -> str:
    """Generate presigned URL for uploading a single part in multipart upload."""
    if expires_in is None:
        expires_in = config.CONTENT_MULTIPART_PART_URL_EXPIRY

    _ensure_bucket_exists_once()

    return _s3_edge_public.generate_presigned_url(
        "upload_part",
        Params={
            "Bucket": _EDGE_BUCKET,
            "Key": key,
            "UploadId": upload_id,
            "PartNumber": part_number,
        },
        ExpiresIn=expires_in,
    )


def complete_multipart(key: str, upload_id: str, parts: list[dict]):
    parts_sorted = sorted(parts, key=lambda p: p["PartNumber"])
    return _s3_edge_internal.complete_multipart_upload(
        Bucket=_EDGE_BUCKET,
        Key=key,
        MultipartUpload={"Parts": parts_sorted},
        UploadId=upload_id,
    )


def head_object(key: str):
    try:
        return _s3_edge_internal.head_object(Bucket=_EDGE_BUCKET, Key=key)
    except Exception:
        return None


def delete_object(key: str) -> bool:
    """Delete an object from edge storage and durable storage when configured."""
    deleted = False
    forget_edge_presence(key)          # before the call: a failed delete may still have landed
    try:
        _s3_edge_internal.delete_object(Bucket=_EDGE_BUCKET, Key=key)
        deleted = True
    except Exception as exc:
        logger.warning("Error deleting edge object %s: %s", key, exc)

    if _durable_store_enabled():
        try:
            _s3_durable.delete_object(Bucket=_DURABLE_BUCKET, Key=key)
            deleted = True
        except Exception as exc:
            logger.warning("Error deleting durable object %s: %s", key, exc)

    return deleted
