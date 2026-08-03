"""`S3ContentStore` — a generic S3-compatible content store: the durable MIRROR / CDN backing tier.

Opaque, content-addressed bytes over any S3-compatible endpoint (AWS S3, a CDN origin, or a
self-hosted object store during migration). Ciphertext-only from the store's view — `content.py`
Fernet-encrypts before anything reaches here, so the object bytes are opaque. This is the BACKING
tier behind the local `FileContentCache`; the hot read path never requires it (the air-gap invariant
— a node answers from its local cache with the backing unreachable).

Extracted 2026-07-22 from a vendor-named store class and made BRAND-NEUTRAL — same boto3 client,
S3-generic name, no coupling to the old adapter. The rename is the point: any S3-compatible endpoint
serves this tier, so the code must not name one.
"""
from __future__ import annotations

import os

from ..store import ContentStore

# cas/ objects are WRITE-ONCE at a content address (promotion is skip-if-exists), so this header
# is simply TRUE — and it is the whole CDN story: an immutable public object needs no
# invalidation, ever, and any dumb HTTP edge cache serves it optimally. Scoped to cas/ because
# only the CAS discipline guarantees write-once; other keys (cursors, manifests) mutate.
_IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"


def cache_control_for(key: str):
    """The Cache-Control header a key has EARNED, or None. Only cas/ addresses are provably
    immutable; claiming immutability for a mutable key would pin its first version into every
    edge cache for a year."""
    return _IMMUTABLE_CACHE_CONTROL if key.startswith("cas/") else None


def _http_get(url: str, timeout: float = 30.0) -> bytes:
    """One plain HTTPS GET. No auth, no SDK — the read side of a public-ciphertext bucket/CDN
    is deliberately dumb (the tier above verifies sha256 against the ref, so a lying edge is
    caught there, not here)."""
    from urllib.request import Request, urlopen
    with urlopen(Request(url, headers={"User-Agent": "agience-lattice"}), timeout=timeout) as r:
        if r.status != 200:
            raise OSError("GET %s -> %s" % (url, r.status))
        return r.read()


class S3ContentStore(ContentStore):
    """S3-backed opaque byte store over any S3-compatible endpoint. Ciphertext-only from the store's view.

    READ/WRITE SPLIT (the CDN seam): writes always use the S3 API; reads MAY come from a plain
    HTTPS base (`read_url_base` param or CONTENT_READ_URL_BASE env) — a CDN or public-bucket URL
    in front of the same objects. A failed CDN read falls back to the S3 API and is COUNTED
    (`stats["cdn_fallback"]`), never silent: "the CDN serves nothing" must be a visible fact,
    not a latency mystery."""

    def __init__(self, endpoint_url: str, access_key: str, secret_key: str,
                 bucket: str, region: str = "auto", *, max_pool_connections: int = None,
                 read_url_base: str = None):
        import boto3  # local import so importing this module never requires boto3
        from botocore.config import Config
        # ⛔ THE CONNECTION POOL — NOT THE THREAD COUNT — IS THE THROUGHPUT CEILING. boto3 defaults to
        # `max_pool_connections=10`; the promote fan-out shares one client, so most workers block on
        # the pool. Size it to the fan-out (same env var, so the two cannot drift), floored at boto3's
        # default so this can never make things worse.
        if max_pool_connections is None:
            max_pool_connections = max(10, int(os.getenv("EMBER_PROMOTE_WORKERS", "64")) + 8)
        self.bucket = bucket
        # Env default so an existing deployment gains the CDN read path by setting ONE variable,
        # with zero call-site changes. Explicit param wins.
        self.read_url_base = (read_url_base or os.getenv("CONTENT_READ_URL_BASE") or "").rstrip("/") or None
        self.stats = {"cdn_hit": 0, "cdn_fallback": 0}
        self._s3 = boto3.client(
            "s3", endpoint_url=endpoint_url, region_name=region,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            config=Config(max_pool_connections=max_pool_connections,
                          retries={"max_attempts": 3, "mode": "standard"}),
        )

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        cc = cache_control_for(key)
        if cc:
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=data,
                                ContentType=content_type, CacheControl=cc)
        else:
            self._s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        if self.read_url_base:
            try:
                data = _http_get("%s/%s" % (self.read_url_base, key))
                self.stats["cdn_hit"] += 1
                return data
            except Exception:
                # Fall back to the authoritative S3 API — counted, so a dead CDN shows up in the
                # published stats instead of masquerading as ordinary (slower) reads.
                self.stats["cdn_fallback"] += 1
        return self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=key)


class DirContentStore(ContentStore):
    """A DIRECTORY-backed content mirror — the same opaque `cas/<sha>` interface as S3ContentStore,
    but the shared store is a filesystem directory (a CIFS-mounted NAS share, a mounted volume).

    For an ON-PREM fleet with no cloud bucket: content promotes to the shared directory and reads
    pull-through from it, exactly as the S3 mirror does, with no boto/creds. Ciphertext-only from
    the store's view (the tier above re-encrypts under the shared cipher and verifies sha256 against
    the ref, so an untrusted filesystem is caught there).

    ⚠ Reads are SYNCHRONOUS filesystem reads, so `root` MUST be reachable from the reading node at
    read time (mount the NAS in WSL, don't rely on an async copy — a batch relay cannot serve a
    read-through miss)."""

    def __init__(self, root):
        from pathlib import Path
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket = str(self.root)

    def _p(self, key: str):
        return self.root / key.replace("/", os.sep)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)     # atomic; cas/ is write-once so any racer writes byte-identical content

    def get(self, key: str) -> bytes:
        p = self._p(key)
        if not p.exists():
            raise OSError("no such content key: %s" % key)
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def delete(self, key: str) -> None:
        try:
            self._p(key).unlink()
        except FileNotFoundError:
            pass


class SMBContentStore(ContentStore):
    """A USERSPACE-SMB content mirror — the same `cas/` interface, backed by an SMB share (a NAS)
    reached DIRECTLY over the network via `smbprotocol`: **no CIFS mount, no sudo**. For a WSL ember
    that cannot mount the NAS. Scoped to ONE share with a GATED, least-privilege account (never the
    admin), so the fleet touches only that share. Ciphertext-only from the store's view (the tier
    re-encrypts under the shared cipher and verifies sha256 against the ref).

    Config (ember/content_tier.open_ovh_store): EMBER_SMB_SERVER / EMBER_SMB_SHARE / EMBER_SMB_USER /
    EMBER_SMB_PASS / EMBER_SMB_PREFIX. `pip install smbprotocol` (userspace, no root)."""

    def __init__(self, server, share, username, password, *,
                 prefix="agience-genesis/_content", port=445):
        import smbclient                        # from the smbprotocol package; userspace SMB
        self._smb = smbclient
        smbclient.register_session(server, username=username, password=password, port=int(port))
        self.server, self.share = server, share
        self.prefix = (prefix or "").strip("/").replace("/", "\\")
        self.bucket = "\\\\%s\\%s" % (server, share)

    def _p(self, key: str) -> str:
        k = key.replace("/", "\\")
        parts = [p for p in ("\\\\%s\\%s" % (self.server, self.share), self.prefix, k) if p]
        return "\\".join(parts)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._p(key)
        parent = p.rsplit("\\", 1)[0]
        try:
            self._smb.makedirs(parent, exist_ok=True)
        except Exception:
            pass
        with self._smb.open_file(p, mode="wb") as f:   # cas/ is write-once; a racer writes same bytes
            f.write(data)

    def get(self, key: str) -> bytes:
        try:
            with self._smb.open_file(self._p(key), mode="rb") as f:
                return f.read()
        except Exception as e:
            raise OSError("no such content key: %s (%s)" % (key, e))

    def exists(self, key: str) -> bool:
        try:
            return bool(self._smb.path.exists(self._p(key)))
        except Exception:
            return False

    def delete(self, key: str) -> None:
        try:
            self._smb.remove(self._p(key))
        except Exception:
            pass
