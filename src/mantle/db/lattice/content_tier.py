"""`TieredContentStore` — ONE content substrate, tiered: local encrypted CAS ⊕ S3 ciphertext.

    target architecture:   SQLite  +  file-content-cache  +  s3-content
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      this module ties the two content tiers together

This is the unification the supplant plan (§0.1) names: `FileContentCache` (the local,
per-observer, AES-GCM collection-keyed, verify-on-read tier — `content_cache.py`) in front of a
generic ciphertext backing store (`S3ContentStore` — the durable mirror / wide-area CDN tier).
It supersedes ember's earlier `content_tier.TieredContentStore`, whose local tier was a separate
object-store DAEMON; here the local tier is the lattice's own encrypted filesystem cache, so a node
needs ZERO external processes to read its working set. That daemon is the reason this module exists.

THE TWO ENCRYPTION MODELS MEET HERE, AND THE BOUNDARY IS EXPLICIT:

  local  = `FileContentCache`: PLAINTEXT in/out; encrypts at rest itself (AES-256-GCM, AAD=ref)
           under ONE node-wide key, `shared_content_key(root_secret)`.
  remote = opaque CIPHERTEXT under the SHARED content cipher (MultiFernet `content.key`), the
           same bytes every node can decrypt — that is what makes one S3 origin serve the fleet.

  ⚠ THE TWO TIERS NOW AGREE ON SCOPE, WHICH THEY DID NOT BEFORE (EREA §1, fixed 2026-07-29). The
  local tier used to key per COLLECTION while the remote keyed fleet-wide — the same immutable
  object at the same address under two different key scopes. `promote_one` below is where the
  contradiction was visible in one line (`cache.get(collection=…)` → `remote.put(_encrypt(…))`),
  and its cost was that a ref shared by two containment roots destroyed one root's copy. Both
  tiers are now scoped to the whole node/fleet; the ciphers still differ (AES-GCM+AAD locally,
  Fernet remotely) because only the local tier can bind AAD=ref, and that binding is worth keeping.

So a read-through miss is:  remote.get → shared-cipher decrypt → **sha256 verify against the ref**
→ best-effort re-cache locally → return plaintext. The verify is not optional: the remote is a
bucket or a CDN edge — an untrusted server exactly like the local disk — and `ref` names the
plaintext, so the one check covers both tiers and both ciphers.

WHY THIS IS A CDN SUBSTRATE FOR FREE. Every remote object is (a) ciphertext — zero-knowledge to
whoever serves it — and (b) write-once at a content address (`cas/sha256(plaintext)`; promotion is
skip-if-exists). Immutable public objects are the exact case HTTP CDNs optimize: no invalidation,
infinite TTL (`s3_content.py` stamps `Cache-Control: immutable` on cas/ puts), any dumb edge cache
works. Authorization stays where it already lives — in key distribution, not in the transport
("storage gets ciphertext; grants are keys, not metadata").

THE AIR-GAP INVARIANT HOLDS. The remote tier is reached ONLY on a local miss. A node whose
working set is cached answers with the backing unreachable (F3 / P7-S3.8); `remote=None` is a
legal, first-class configuration (an air-gapped box), not a degraded one.

FAILURE MODES ARE DISTINCT AND LOUD — same contract as `content_cache.py`:
  * a MISS raises; it never returns b"" (the silent-`[]` class of defect);
  * CORRUPT is not MISS, remote or local; a present-but-undecryptable object is reported as
    corruption (a missing shared key is a SILENT PARTITION and must never read as a cache miss);
  * `ContentKeyMissing` (node-wide config fault) propagates UNWRAPPED, so callers that swallow
    per-object failures cannot also swallow "this node has no key" — the resolve_text precedent.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time as _time
from pathlib import Path
from typing import Callable, Dict, Optional

from .content_cache import (CacheCorrupt, CacheMiss, ContentCacheError,
                            FileContentCache, _hash_of)


class ContentKeyMissing(ContentCacheError):
    """No usable shared `content.key` — a CONFIGURATION fault, never a missing artifact.

    Mirrors ember `content.ContentKeyMissing` (same name deliberately: propagation checks are
    by class NAME so either class survives the ember→mantle flip). Raised instead of minting:
    a generated key would silently partition this node's content from the fleet."""


class RemoteMiss(ContentCacheError):
    """Neither tier holds the object (or the backing did not answer — the message says which).
    NOT an empty result; the caller decides what absence means."""


class RemoteCorrupt(ContentCacheError):
    """The remote returned bytes that do not decrypt, or decrypt to the WRONG plaintext
    (sha256 ≠ ref). Never cached, never returned — a CDN edge or bucket is just another
    untrusted server."""


# ── the shared content cipher ────────────────────────────────────────────────────────────────
_KEY_CACHE: dict = {}
_KEY_TTL = 60.0


def content_cipher(keys_dir) -> "MultiFernet":
    """READ-ONLY MultiFernet over `<keys_dir>/content.key` (+ `content.key.*` decrypt fallbacks).

    The same key convention as ember `content._content_key` — primary encrypts, extras are
    decrypt-only fallbacks for peers' historically-diverged keys — with ONE deliberate
    difference: **this loader never mints.** Bootstrapping the first key is a WRITE-path
    decision (ember `put_content(create=True)` is the only caller allowed); a standalone-Mantle
    read path that finds no key has a configuration fault and says so. No mkdir either — an
    absent keys dir means the volume is not mounted (the silent-partition precedent)."""
    from cryptography.fernet import Fernet, MultiFernet
    kd = Path(keys_dir)
    ck = str(kd.resolve()) if kd.exists() else str(kd)
    now = _time.time()
    ent = _KEY_CACHE.get(ck)
    if ent is not None and now - ent[0] <= _KEY_TTL:
        return ent[1]
    kp = kd / "content.key"
    if not kp.exists():
        raise ContentKeyMissing(
            "no content.key in %s — refusing to mint one on a read path. A generated key would "
            "silently partition this node's content from the fleet. Mount the keys volume, or "
            "bootstrap via the write path (ember content.put_content)." % kd)
    keys = [Fernet(kp.read_bytes().strip())]                 # primary (encrypt) first
    for extra in sorted(kd.glob("content.key.*")):           # decrypt-only fallbacks
        try:
            keys.append(Fernet(extra.read_bytes().strip()))
        except Exception:
            pass
    mf = MultiFernet(keys)
    _KEY_CACHE[ck] = (now, mf)
    return mf


class TieredContentStore:
    """`FileContentCache` (local plaintext tier) in front of a ciphertext backing store.

    Surface is PLAINTEXT + collection-aware, like the cache: `get(ref, collection=…)`. This is
    what lets ember's `resolve_text` collapse its FileContentCache/ContentStore fork into one
    branch at the flip."""

    def __init__(self, cache: Optional[FileContentCache], remote=None, *,
                 decrypt: Optional[Callable[[bytes], bytes]] = None,
                 encrypt: Optional[Callable[[bytes], bytes]] = None):
        """`remote` is any `db.store.ContentStore` holding SHARED-cipher ciphertext (normally
        `S3ContentStore`). `decrypt`/`encrypt` are the shared content cipher's callables —
        injected, like the cache's `key_for_collection`, so this module never loads a key itself
        (`content_cipher(keys_dir).decrypt/.encrypt` is the standard wiring).

        At least one tier is required: a content store with no tier at all is a configuration
        fault, and refusing here is cheaper than every read raising something misleading later."""
        if cache is None and remote is None:
            raise ValueError("TieredContentStore needs at least one tier (cache and remote are "
                             "both None) — a store with no tier cannot serve anything, and "
                             "constructing one is always a wiring mistake.")
        if remote is not None and decrypt is None:
            raise ValueError("a remote tier holds SHARED-cipher ciphertext; constructing it "
                             "without `decrypt` builds a store whose every remote read fails. "
                             "Pass content_cipher(keys_dir).decrypt (or equivalent).")
        self.cache = cache
        self.remote = remote
        self._decrypt = decrypt
        self._encrypt = encrypt
        # ⛔ GIVE THE CACHE ITS DURABILITY ORACLE. `FileContentCache._ensure_free` deletes blobs to
        # hold a disk floor and has no other way to tell a re-fetchable object from the only copy;
        # without this it refuses to evict at all (and says so). This is the same "nothing is
        # deleted that is not PROVEN remote" rule `evict_local`/`evict_for_space` already enforce —
        # the write-path evictor simply had no way to ask until now. With `remote=None` it stays
        # None, so an air-gapped node never evicts: correct, and the floor goes unenforced loudly.
        if cache is not None and remote is not None and getattr(cache, "_is_durable", None) is None:
            cache._is_durable = self._remote_has

        self._lock = threading.Lock()
        self.stats: Dict[str, int] = {
            "remote_hit": 0, "remote_miss": 0, "remote_corrupt": 0,
            "recache_failed": 0, "promoted": 0, "promote_skipped": 0, "evicted": 0}

    def _count(self, k: str) -> None:
        with self._lock:
            self.stats[k] += 1

    def _remote_has(self, ref: str) -> bool:
        """Durability oracle for the cache's disk-floor evictor. An unreachable backing must read
        as NOT durable — 'I could not ask' is never 'yes, it is safe to delete'."""
        try:
            return bool(self.remote.exists(ref))
        except Exception:
            return False

    # ── read ─────────────────────────────────────────────────────────────────
    def get(self, ref: str, *, collection: str) -> bytes:
        """Plaintext, or raise. Local first; the remote is touched ONLY on a local miss/corrupt."""
        local_err: Optional[ContentCacheError] = None
        if self.cache is not None:
            try:
                return self.cache.get(ref, collection=collection)
            except (CacheMiss, CacheCorrupt) as e:
                # A corrupt local object was evicted and counted by the cache; the durable
                # origin is the repair path. If there is no remote, the ORIGINAL error is
                # re-raised below — corrupt must not be laundered into miss by the tiering.
                local_err = e
        if self.remote is None:
            raise local_err if local_err is not None else CacheMiss(
                "no local cache configured and no remote tier: %s" % ref)
        try:
            blob = self.remote.get(ref)
        except Exception as e:
            self._count("remote_miss")
            # boto3 raises the same ClientError family for absent-key and unreachable-backing;
            # the message carries which. Distinct-exception refinement can come with a measured
            # need — what must not happen is either case returning b"".
            raise RemoteMiss("remote tier did not yield %s (%s: %s)"
                             % (ref, type(e).__name__, e)) from e
        try:
            plain = self._decrypt(blob)
        except Exception as e:
            if type(e).__name__ == "ContentKeyMissing":
                raise                    # node-wide config fault — NEVER wrapped (see module doc)
            self._count("remote_corrupt")
            raise RemoteCorrupt(
                "remote object %s does not decrypt under the shared content cipher (%s) — a "
                "wrong or missing fallback key, or altered bytes. Never returned as data."
                % (ref, type(e).__name__)) from None
        # ⚠ VERIFY AGAINST THE REF BEFORE CACHING OR RETURNING. The ref names the plaintext, so
        # this one check makes a CDN edge exactly as trustworthy as the origin — which is what
        # lets the read endpoint be a dumb public cache at all.
        if hashlib.sha256(plain).hexdigest() != _hash_of(ref):
            self._count("remote_corrupt")
            raise RemoteCorrupt(
                "remote object %s decrypts but does not hash to its own address — wrong object "
                "behind the ref (a poisoned edge cache, or a corrupted origin). EVICT NOTHING "
                "LOCALLY; never returned, never cached." % ref)
        self._count("remote_hit")
        if self.cache is not None:
            try:
                self.cache.put(ref, plain, collection=collection, tier="s3")
            except Exception:
                # Best-effort by design: an unkeyable collection (no origin_root) can still be
                # SERVED from the remote — but the failure is counted, because "every read is a
                # WAN read" must be visible, not folded into a healthy-looking hit rate.
                self._count("recache_failed")
        return plain

    # ── write ────────────────────────────────────────────────────────────────
    def put(self, ref: str, plaintext: bytes, *, collection: str) -> bool:
        """Store locally. Writes NEVER block on the WAN — promotion to the durable tier is
        `promote_one`'s job (the async drain), same discipline as the earlier tier. With no
        local cache configured (migration tooling), writes go through to the remote directly."""
        if self.cache is not None:
            return self.cache.put(ref, plaintext, collection=collection, tier="local-write")
        if hashlib.sha256(plaintext).hexdigest() != _hash_of(ref):
            raise CacheCorrupt("refusing to store: plaintext does not hash to %s" % ref)
        if self._encrypt is None:
            raise ValueError("cache-less write-through needs `encrypt` (the shared cipher); "
                             "storing plaintext in the remote would publish it.")
        if not self.remote.exists(ref):          # content-addressed → idempotent
            self.remote.put(ref, self._encrypt(plaintext))
            return True
        return False

    def exists(self, ref: str, *, collection: str = None) -> bool:
        if self.cache is not None and ref in self.cache:
            return True
        if self.remote is not None:
            try:
                return self.remote.exists(ref)
            except Exception:
                return False             # unreachable backing ≠ absent, but exists() cannot say
                                         # more; get() is the loud path and reports the cause
        return False

    # ── promotion / eviction (the drain's building blocks) ───────────────────
    def promote_one(self, ref: str, *, collection: str) -> str:
        """Copy one local object UP to the durable tier: 'put' | 'skip'. Idempotent (skip-if-
        exists — which is also the write-once discipline that makes `Cache-Control: immutable`
        true). Raises loudly if the local object is unreadable: promoting bytes we cannot verify
        would put garbage behind a valid-looking address, fleet-wide."""
        if self.remote is None:
            raise ValueError("promote_one with no remote tier configured")
        if self._encrypt is None:
            raise ValueError("promote_one needs `encrypt` (the shared cipher) — the remote "
                             "holds ciphertext only")
        if self.remote.exists(ref):
            self._count("promote_skipped")
            return "skip"
        plain = self.cache.get(ref, collection=collection)   # verified read, loud on failure
        self.remote.put(ref, self._encrypt(plain))
        self._count("promoted")
        return "put"

    def evict_local(self, ref: str) -> bool:
        """Delete the LOCAL copy — only ever after the object is CONFIRMED in the remote.
        Never deletes what has not been proven durable elsewhere (a promote failure can't lose
        content); with no remote there is nothing to prove against, so nothing is evicted."""
        if self.cache is None or self.remote is None:
            return False
        try:
            if not self.remote.exists(ref):
                return False
        except Exception:
            return False                 # can't prove it's remote → keep the local copy
        try:
            os.unlink(self.cache.path_of(ref))
        except OSError:
            return False
        self._count("evicted")
        return True

    def evict_for_space(self, *, min_free_bytes: int, limit: int = 5000) -> Dict[str, object]:
        """Evict remote-CONFIRMED local objects until the cache volume has ≥ `min_free_bytes` free.

        MECHANISM ONLY — the floor is the CALLER's fact (an operator's disk budget, or a measured
        envelope), deliberately not chosen here: a constant baked into this module would be exactly
        the arbitrary cap the project forbids. Un-called, nothing is evicted (the honest default).

        Selection is walk-order among confirmed-remote objects. Under the uniform 2-level hash
        layout that is uniform-RANDOM eviction — an honest working-set policy when re-caching on
        demand is cheap (the next read pulls the object straight back through the CDN tier).
        True LRU needs an access-order signal the CAS layout does not record (atime is unreliable
        on the deployed filesystems); that refinement is a FLAGGED SEAM, not silently approximated
        here with mtime (which records POPULATION order, not use, and would evict the oldest-
        ingested hot object first — worse than random, dressed as better).

        The evict_local rule holds per object: nothing is deleted that is not PROVEN remote."""
        import shutil
        if self.cache is None or self.remote is None:
            return {"evicted": 0, "errors": 0, "free": None,
                    "reason": "needs both tiers (never evict the only copy)"}
        free = shutil.disk_usage(self.cache.root).free
        out: Dict[str, object] = {"evicted": 0, "errors": 0, "free_before": free}
        if free < min_free_bytes:
            done = False
            for dirpath, _dirs, files in os.walk(self.cache.root):
                if done:
                    break
                for fn in files:
                    if len(fn) != 64:            # mkstemp leftovers etc. — not CAS objects
                        continue
                    if int(out["evicted"]) + int(out["errors"]) >= limit:
                        done = True              # bounded work per call, like the drain's pages
                        break
                    ref = "cas/" + fn
                    try:
                        if not self.remote.exists(ref):
                            continue             # the only copy stays, always
                        os.unlink(os.path.join(dirpath, fn))
                        self._count("evicted")
                        out["evicted"] = int(out["evicted"]) + 1
                        if int(out["evicted"]) % 128 == 0:
                            if shutil.disk_usage(self.cache.root).free >= min_free_bytes:
                                done = True
                                break
                    except Exception:
                        out["errors"] = int(out["errors"]) + 1
        out["free"] = shutil.disk_usage(self.cache.root).free
        return out

    # ── reporting ────────────────────────────────────────────────────────────
    def report(self) -> Dict[str, object]:
        with self._lock:
            s = dict(self.stats)
        s["local"] = self.cache.report() if self.cache is not None else None
        s["remote_configured"] = self.remote is not None
        return s
