"""`FileContentCache` — the local, encrypted, content-addressed middle tier.

    target architecture:   SQLite  +  file-content-cache  +  s3-content
                                      ^^^^^^^^^^^^^^^^^^
                                      this module

──────────────────────────────────────────────────────────────────────────────────────────────
The contract

  Identity        `ref = "cas/" + sha256(plaintext)`. Already the corpus's content address, so the
                  cache invents no key space. Key-independent by construction: it survives
                  re-keying, because it names the plaintext, not the ciphertext.

  Verify on read  Every `get()` re-hashes the decrypted plaintext and compares it to the ref.
                  Disk is just another untrusted server. A cached object that no longer hashes
                  is reported corrupt and evicted — never returned, never repaired in place.

  Encrypted at rest, one key per node
                  `shared_content_key = HKDF(root, info="filecache|v1|shared")`, each object
                  AES-256-GCM with `AAD = ref`. One immutable object, one address, one key.

"""
from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import threading
from typing import Callable, Dict, Optional, Tuple

_log = logging.getLogger(__name__)


def _min_free_bytes() -> int:
    """The cache free-space floor in bytes, from ``MANTLE_CACHE_MIN_FREE_GB`` (default 0 =
    unbounded).

    A lean hot-cache node (a Pi, a made-lean workstation) sets this so a growing corpus can never
    take the volume to 0. A warehouse node leaves it unset and keeps the unbounded behavior — this
    is strictly opt-in, so it changes nothing unless configured.

    Both the current name and the deprecated ``EMBER_CACHE_MIN_FREE_GB`` are honored, so a node
    still using the old name keeps its floor enforced rather than losing it silently to a rename."""
    for name in ("MANTLE_CACHE_MIN_FREE_GB", "EMBER_CACHE_MIN_FREE_GB"):
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        try:
            gb = float(raw)
        except ValueError:
            _log.warning("content cache: %s=%r is not a number — ignoring", name, raw)
            continue
        if name.startswith("EMBER_"):
            _log.warning("content cache: %s is DEPRECATED; rename it to MANTLE_CACHE_MIN_FREE_GB. "
                         "Still honored, so the disk floor is not silently lost.", name)
        return int(gb * (1024 ** 3)) if gb > 0 else 0
    return 0

CAS_PREFIX = "cas/"
_KDF_INFO = b"filecache|v1|"
# Distinct info string, so the shared key can never collide with a collection key: `_KDF_INFO` is
# always followed by a non-empty origin_root, and "shared" is not a valid principal id.
_KDF_SHARED_INFO = b"filecache|v1|shared"

# ── AES-GCM wire format ──────────────────────────────────────────────────────
# SPECIFICATION, not a tuned bound. Every stored object is `nonce || ciphertext || tag`.
# NIST SP 800-38D §5.2.1.1 makes 96 bits the one nonce length GCM does not have to re-derive,
# and `cryptography`'s AESGCM appends the full 128-bit tag. These two ARE the format; the
# minimum-length floor below is arithmetic over them, never a number chosen for a corpus.
_NONCE_BYTES = 12
_TAG_BYTES = 16
#: The shortest byte string that could possibly BE a ciphertext (empty plaintext: nonce + tag).
#: Anything shorter is provably corrupt without holding any key.
_MIN_BLOB_BYTES = _NONCE_BYTES + _TAG_BYTES


class ContentCacheError(Exception):
    """Base. Every failure mode below is distinct, and none of them is an empty return."""


class CacheMiss(ContentCacheError):
    """The object is not present locally. Not an empty result — see the contract."""


class CacheCorrupt(ContentCacheError):
    """Proven corruption: the object decrypted (so AES-GCM authenticated the bytes) and the
    plaintext then failed its own sha256, or it is too short to be a ciphertext at all."""
class ContentKeyMismatch(CacheCorrupt):
    """The object could not be decrypted — wrong key or altered bytes, and the two are
    indistinguishable from a failed AES-GCM tag alone.

    Unlike `CacheCorrupt`, this is never proof of corruption, so `get()` never evicts on it: an
    object shared across containment roots keyed differently could otherwise be destroyed by
    reading it under the wrong key, with the damage compounding on every subsequent read once the
    bytes were gone.

    Still never returned as data — the caller gets an exception either way. The only thing that
    changes is that the bytes survive to be read again once the key situation is fixed."""


def _hash_of(ref: str) -> str:
    if not ref.startswith(CAS_PREFIX):
        raise ValueError("not a CAS ref (expected %r prefix): %r" % (CAS_PREFIX, ref))
    h = ref[len(CAS_PREFIX):]
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError("CAS ref is not a sha256 hex digest: %r" % ref)
    return h


def collection_key(root_secret: bytes, origin_root: str) -> bytes:
    """Derive a collection's at-rest key from its immutable origin root (P9.3).

    `origin_root` is the collection's origin-root principal id — the one field that, by design,
    never moves. Deriving from a grant, an owner or `created_by` would re-key the whole collection
    every time authority changed, orphaning every object already written."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    if not origin_root:
        raise ValueError("collection_key requires a non-empty origin_root — deriving from an "
                         "empty principal would give every collection the SAME key")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_KDF_INFO + origin_root.encode("utf-8")).derive(root_secret)


def shared_content_key(root_secret: bytes) -> bytes:
    """The at-rest key for local content — one key for the whole node, from the same root secret.

    `ref = cas/sha256(plaintext)` addresses one immutable, write-once object globally. Encrypting
    it under a per-collection key would make one address mean N different ciphertexts, only one of
    which opens — so a second containment root storing bytes the first already had would find the
    ref present, fail to verify it, evict it, and rewrite under its own key, destroying the first
    root's only copy. A single shared key removes the collision: one immutable object, one address,
    one key.

    Secondary and weaker, but confirming: even a per-collection key would not survive promotion.
    `content_tier`'s durable tier already stores the same bytes under the shared fleet cipher
    (`content.key`), so a local per-collection scope would evaporate one `promote_one` later."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    if not root_secret:
        raise ValueError("shared_content_key requires a non-empty root secret")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_KDF_SHARED_INFO).derive(root_secret)


class FileContentCache:
    """Local encrypted CAS. Thread-safe. Read-verifying. Loud on every failure mode."""

    def __init__(self, root: str, *, key: Optional[bytes] = None,
                 legacy_key_for_collection: Optional[Callable[[str], bytes]] = None,
                 is_durable: Optional[Callable[[str], bool]] = None):
        """`key` is the at-rest key (32 bytes) — normally `shared_content_key(root_secret)`.

        Injected rather than derived here so this module never holds the root secret and never
        decides who may have a key. Issuance is gated elsewhere; this object only encrypts and
        decrypts with a key it is handed.

        `legacy_key_for_collection(collection_id) -> 32 bytes` is the migration fallback, tried only
        when the shared key fails to open an object. Objects it opens are re-encrypted under `key`
        in place (see `get`), so a corpus still holding objects under the older per-collection keying
        converges without a re-fetch and without downtime. Pass None once a store is known fully
        migrated.

        `is_durable(ref) -> bool` proves an object exists in the durable tier. Required for the
        disk-floor eviction in `_ensure_free` to run at all — without it nothing is evicted, because
        this class cannot otherwise tell a re-fetchable blob from the only copy. `TieredContentStore`
        wires it from its remote."""
        if key is None:
            raise ValueError(
                "FileContentCache requires `key` (the shared at-rest key, normally "
                "shared_content_key(root_secret)). The per-collection-keyed construction "
                "(`key_for_collection=`) is not used for writes: addressing objects globally while "
                "keying them per collection would let one shared ref overwrite another root's copy "
                "on write and again on read (§1: one immutable object, one address, one key). "
                "Pass `legacy_key_for_collection=` to migrate "
                "an existing corpus — it is a decrypt-only fallback, never used to write.")
        if len(key) != 32:
            raise ValueError("at-rest key must be 32 bytes, got %d" % len(key))
        self.root = root
        self._key = key
        self._legacy_key_for = legacy_key_for_collection
        self._is_durable = is_durable
        self._lock = threading.Lock()
        self.stats: Dict[str, int] = {"hit": 0, "miss": 0, "corrupt": 0, "key_mismatch": 0,
                                      "put": 0, "evicted": 0, "rekeyed": 0, "rekey_failed": 0}
        self.tiers: Dict[str, int] = {}
        os.makedirs(self.root, exist_ok=True)

    # ── layout ───────────────────────────────────────────────────────────────
    def path_of(self, ref: str) -> str:
        """`<root>/<aa>/<bb>/<sha256>`. Two levels of fan-out because a single directory with
        6.1M entries is pathological on every filesystem that matters, and the hash is uniform so
        the split is balanced for free."""
        h = _hash_of(ref)
        return os.path.join(self.root, h[:2], h[2:4], h)

    def __contains__(self, ref: str) -> bool:
        try:
            return os.path.exists(self.path_of(ref))
        except ValueError:
            return False

    # ── read ─────────────────────────────────────────────────────────────────
    def get(self, ref: str, *, collection: str) -> bytes:
        """Plaintext, or raise. Never returns b"" and never returns unverified bytes."""
        p = self.path_of(ref)
        if not os.path.exists(p):
            with self._lock:
                self.stats["miss"] += 1
            raise CacheMiss("not cached locally: %s" % ref)
        with open(p, "rb") as fh:
            blob = fh.read()
        try:
            plain, was_legacy = self._decrypt(blob, ref, collection)
        except ContentKeyMismatch as e:
            with self._lock:
                self.stats["key_mismatch"] += 1
            _log.error("content cache: %s", e)
            raise
        except CacheCorrupt:
            self._evict(p)
            with self._lock:
                self.stats["corrupt"] += 1
            raise
        if hashlib.sha256(plain).hexdigest() != _hash_of(ref):
            self._evict(p)
            with self._lock:
                self.stats["corrupt"] += 1
            raise CacheCorrupt(
                "cached object does not hash to its own content address (%s) — EVICTED. This is "
                "corruption, not a miss: the bytes were present and wrong." % ref)
        if was_legacy:
            # Lazy migration. The object is verified (it decrypted and hashed), so rewriting it
            # under the shared key is safe here and nowhere else. Ordered after the sha256 check on
            # purpose: re-encrypting unverified bytes would launder corruption into a valid-looking
            # object under the current key. Best-effort — a failed rewrite must not fail the read,
            # which already has correct plaintext in hand; the object simply migrates on a later
            # read (or via `scripts/mantle_cas_rekey.py`). Nothing is ever deleted by this path.
            self._rewrite_shared(p, ref, plain)
        with self._lock:
            self.stats["hit"] += 1
        return plain

    def _rewrite_shared(self, path: str, ref: str, plain: bytes) -> bool:
        """Re-encrypt a verified plaintext under the shared key, atomically, in place."""
        try:
            self._atomic_write(path, self._encrypt(plain, ref))
        except Exception as e:
            with self._lock:
                self.stats["rekey_failed"] += 1
            _log.warning("content cache: could not re-key %s under the shared key (%s: %s) — the "
                         "object is intact and still readable via the legacy key",
                         ref, type(e).__name__, e)
            return False
        with self._lock:
            self.stats["rekeyed"] += 1
        return True

    # ── write ────────────────────────────────────────────────────────────────
    def put(self, ref: str, plaintext: bytes, *, collection: str,
            tier: str = "unknown") -> bool:
        """Store `plaintext` under `ref`. Returns True if written, False if already present.

        Idempotent and resumable: an object already present and still hashing is left alone, so
        an interrupted population resumes without re-fetching and without rewriting."""
        if hashlib.sha256(plaintext).hexdigest() != _hash_of(ref):
            raise CacheCorrupt(
                "refusing to store: plaintext does not hash to %s. A short read stored here "
                "would be indistinguishable from the real object forever." % ref)
        p = self.path_of(ref)
        if os.path.exists(p):
            try:
                self.get(ref, collection=collection)
                return False                      # present and verified — resumable no-op
            except ContentKeyMismatch:
                raise
            except ContentCacheError:
                pass                              # proven corrupt: fall through and rewrite
        # Bounded cache: keep a free-space floor so a growing corpus can never fill the volume.
        # Opt-in (EMBER_CACHE_MIN_FREE_GB); no-op unless a lean node configures it. Eviction is LRU
        # and content is durable in S3 / re-ingestable, so it costs a re-fetch, never loss.
        _floor = _min_free_bytes()
        if _floor:
            self._ensure_free(_floor + len(plaintext))
        self._atomic_write(p, self._encrypt(plaintext, ref))
        with self._lock:
            self.stats["put"] += 1
            self.tiers[tier] = self.tiers.get(tier, 0) + 1
        return True

    def _atomic_write(self, path: str, blob: bytes) -> None:
        """Atomic publish: a reader must never observe a partially-written object. Without this a
        kill mid-write leaves a short file that fails its hash on the next read and is evicted —
        recoverable, but it would count as corruption when it was only interruption.

        Shared with the re-key path, where atomicity matters even more: a torn rewrite there would
        destroy a readable legacy object to produce an unreadable shared-key one."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(blob)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ── bounded cache: LRU eviction to a free-space floor ──────────────────────
    def _ensure_free(self, need_free_bytes: int) -> int:
        """Evict least-recently-used blobs until at least `need_free_bytes` are free on the cache
        volume. Returns the count evicted. Runs only when the volume is below the floor; then frees
        a little extra (to ~1.25× the floor) so subsequent writes do not immediately re-scan.

        LRU by access time (falls back to mtime). Only evicts an object `_is_durable` confirms
        exists in the remote tier; without that oracle nothing is evicted, because this cache
        cannot otherwise tell a re-fetchable blob from the only copy."""
        try:
            if shutil.disk_usage(self.root).free >= need_free_bytes:
                return 0
        except OSError:
            return 0
        if self._is_durable is None:
            _log.warning(
                "content cache: %s is below its free-space floor, but NOTHING WAS EVICTED — this "
                "cache has no durability oracle, so it cannot tell a re-fetchable blob from the "
                "only copy. The floor is UNENFORCED. Wire a remote tier (TieredContentStore does "
                "this automatically) or accept an unbounded cache; deleting on a guess is not an "
                "option.", self.root)
            return 0
        target = int(need_free_bytes * 1.25)
        entries = []
        try:
            for aa in os.scandir(self.root):
                if not aa.is_dir():
                    continue
                for bb in os.scandir(aa.path):
                    if not bb.is_dir():
                        continue
                    for f in os.scandir(bb.path):
                        try:
                            st = f.stat()
                            entries.append((st.st_atime or st.st_mtime, f.path))
                        except OSError:
                            continue
        except OSError:
            pass
        entries.sort(key=lambda e: e[0])          # oldest access first
        evicted = 0
        for _t, path in entries:
            try:
                if shutil.disk_usage(self.root).free >= target:
                    break
            except OSError:
                break
            fn = os.path.basename(path)
            if len(fn) != 64:
                continue                          # mkstemp leftovers etc. — not CAS objects
            try:
                if not self._is_durable(CAS_PREFIX + fn):
                    continue                      # the only copy stays, always
            except Exception:
                continue                          # can't prove it's durable → keep it
            try:
                os.unlink(path)
                evicted += 1
            except OSError:
                continue
        if evicted:
            with self._lock:
                self.stats["evicted"] = self.stats.get("evicted", 0) + evicted
        return evicted

    # ── crypto ───────────────────────────────────────────────────────────────
    def _encrypt(self, plain: bytes, ref: str) -> bytes:
        """Always under the shared key. The legacy key is decrypt-only — writing under it again
        would re-create the collision the per-collection scheme was vulnerable to."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(_NONCE_BYTES)
        # AAD = the ref. Binds ciphertext to its content address so an object cannot be moved
        # between refs on disk without the decrypt failing.
        return nonce + AESGCM(self._key).encrypt(nonce, plain, ref.encode("utf-8"))

    def _decrypt(self, blob: bytes, ref: str, collection: str) -> Tuple[bytes, bool]:
        """`(plaintext, was_legacy)`. `was_legacy=True` means the object opened only under the
        older per-collection key and should be rewritten under the shared key."""
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if len(blob) < _MIN_BLOB_BYTES:
            # Provable without any key: no ciphertext can be shorter than nonce + tag, so this is
            # the one decrypt-side failure that is proof and stays evictable as `CacheCorrupt`.
            # `_MIN_BLOB_BYTES` is arithmetic over the two format constants above rather than a
            # typed number, so it cannot drift from them and admit a blob too short to be a real
            # ciphertext. A blob that merely fails `InvalidTag` later is ambiguous — that could be
            # a missing key rather than proof of corruption — which is why this floor is checked
            # first, before any key is tried.
            raise CacheCorrupt(
                "cached object is %d bytes, shorter than the %d-byte nonce+tag floor: %s"
                % (len(blob), _MIN_BLOB_BYTES, ref))
        aad = ref.encode("utf-8")
        try:
            return AESGCM(self._key).decrypt(blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], aad), False
        except InvalidTag:
            pass
        # ── legacy fallback: an object written under this collection's own (older, per-collection) key ──
        if self._legacy_key_for is not None:
            try:
                legacy = self._legacy_key_for(collection)
            except Exception:
                legacy = None                     # underivable (no origin_root) — nothing to try
            if legacy is not None:
                try:
                    return AESGCM(legacy).decrypt(
                        blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], aad), True
                except InvalidTag:
                    pass
        raise ContentKeyMismatch(
            "AES-GCM tag mismatch for %s (collection %r) under the shared at-rest key%s — either a "
            "wrong/rotated key or the object was altered on disk. THE TWO ARE INDISTINGUISHABLE "
            "HERE, so the object is left in place: never returned as data, never destroyed on a "
            "guess." % (ref, collection,
                        " or this collection's legacy key" if self._legacy_key_for else ""))

    def _evict(self, path: str) -> None:
        try:
            os.unlink(path)
            with self._lock:
                self.stats["evicted"] += 1
        except OSError:
            pass

    # ── reporting ────────────────────────────────────────────────────────────
    def report(self) -> Dict[str, object]:
        """Hit / miss / corrupt / key_mismatch counts, `reads` and `hit_rate` derived from them,
        and `tiers` — a count of `put()` calls by tier, so a population run's provenance is
        recorded per run rather than only visible in aggregate."""
        with self._lock:
            s = dict(self.stats)
            t = dict(self.tiers)
        # `key_mismatch` counts as a read — it is a served request that returned no data. Leaving
        # it out would inflate `hit_rate` by shrinking the denominator, which is the exact shape of
        # metric that reads healthy on a node answering nothing.
        reads = s["hit"] + s["miss"] + s["corrupt"] + s["key_mismatch"]
        s["reads"] = reads
        s["hit_rate"] = round(s["hit"] / reads, 4) if reads else None
        s["tiers"] = t
        return s
