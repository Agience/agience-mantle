"""`FileContentCache` — the LOCAL, ENCRYPTED, CONTENT-ADDRESSED middle tier.

    target architecture:   SQLite  +  file-content-cache  +  s3-content
                                      ^^^^^^^^^^^^^^^^^^
                                      this module

⛔ WHY THIS EXISTS, AND WHY IT IS NOT A CACHE IN THE ORDINARY SENSE.

LATTICE-OUTSTANDING §12.2/D4 states the gap plainly: the middle tier "does not exist and nobody
owns it", and "S3 cannot be decommissioned until it exists." It is not an optimization — it is
the component that makes the decommission criterion satisfiable at all:

    F3 / P7-S3.8:  "with EVERY external store unreachable, can new Ember open this store
                    and answer?"  —  "If anything still reaches for them, consolidation failed."

While the only copy of an artifact's content is remote, that question has one answer and it is
no. So this tier is a *durability* boundary, not a latency one, and it is sized and verified
accordingly: every object is checked on read, and a miss is loud.

──────────────────────────────────────────────────────────────────────────────────────────────
THE CONTRACT

  IDENTITY        `ref = "cas/" + sha256(PLAINTEXT)`. Already the corpus's content address, so
                  the cache does not invent a key space. Key-independent by construction: it
                  survives re-keying, because it names the plaintext, not the ciphertext.

  VERIFY ON READ  Every `get()` re-hashes the decrypted plaintext and compares it to the ref.
                  **Disk is just another untrusted server.** A cached object that no longer
                  hashes is reported CORRUPT and evicted — never returned, never repaired in
                  place.

  ENCRYPTED AT REST, ONE KEY PER NODE
                  `shared_content_key = HKDF(root, info="filecache|v1|shared")`, and each object is
                  AES-256-GCM with `AAD = ref`. **One immutable object, one address, one key.**

                  ⚠ THIS WAS PER-COLLECTION UNTIL §1 (EREA, 2026-07-28) AND MUST NOT GO BACK.
                  Keying per collection while addressing globally meant one `cas/sha256(plaintext)`
                  had N ciphertexts, only one of which opened — so a second containment root storing
                  bytes the first already had evicted the incumbent and rewrote it under its own
                  key. Silent, permanent, and it compounded on every subsequent READ. See
                  `shared_content_key` for the full argument, including why the isolation given up
                  was nominal in every deployment that exists.

                  ⚠ `collection_key` SURVIVES and is still exported — as the legacy decrypt
                  fallback here, and because the COMMS PLANE derives per-group keys from it
                  (`iris/comms/wiring.py`, `ember/reach.py`). That isolation is untouched.

                  ⚠ AUTHORIZATION WAS NEVER IN THE BLOB KEY, so nothing weakened. Access is the
                  CRUDEASIO light cone over the artifacts that REFERENCE a ref; storage holds
                  ciphertext and answers no questions about who may read it. P9.3 still holds where
                  it applies: keys root at an IMMUTABLE identity, never at a grant, because "grants
                  mutate, and re-keying on every revoke would ORPHAN CELLS".

                  ⚠ AAD = the ref binds ciphertext to content address, so a blob cannot be moved
                  between refs undetected. Without it, swapping two files on disk is invisible
                  until the sha256 check — and the sha256 check is exactly what an attacker who
                  can write the cache directory would target. This is why the local tier keeps
                  AES-GCM rather than adopting the remote tier's Fernet, which has no AAD.

  A MISS IS A MISS `get()` raises `CacheMiss`. It NEVER returns b"", never returns a partial
                  read, and never falls back to a preview. §6A rules the 300-char inline preview
                  a HARD OUT: an offer built from it "would describe the lead paragraph, not the
                  artifact", and because the offer IS the index target that truncation becomes a
                  permanent ceiling on recall. An empty-string return here is precisely the
                  silent-`[]` failure class this refactor exists to kill.

  DESTRUCTION NEEDS PROOF
                  Eviction is irreversible, so only a PROVEN-bad object is evicted: one that
                  decrypted (AES-GCM authenticated the bytes) and then failed its own sha256.
                  A failed *decrypt* proves nothing — `ContentKeyMismatch` is raised, reported
                  loudly, and the object is LEFT IN PLACE. Wrong key and altered bytes are
                  indistinguishable at that point, and one of the two readings is recoverable.

  STATS           `hit` / `miss` / `corrupt` / `key_mismatch` are counted DISTINCTLY. Collapsing
                  corrupt into miss is how a failing disk reads as a cold cache forever; collapsing
                  `key_mismatch` is how a node that answers every read over the WAN still reports a
                  healthy local hit rate.

  POPULATION      Resumable and idempotent: `put()` is a no-op when the object is present and
                  still hashes. Population records the TIER each object actually came from, so
                  "the remote tier served 0 of N" is a recorded measurement rather than a lost detail.

⚠ PLACEMENT. This sits in `mantle/db/lattice/` beside the store rather than in `ember/` for one
practical reason: `ember/__init__.py` imports `core`, which is not installed on the migration
host, so anything there cannot be imported by the tooling that needs this. The lattice package
imports cleanly on every node, and the content tier travels with the store it belongs to.
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
    """The cache free-space FLOOR in bytes, from ``MANTLE_CACHE_MIN_FREE_GB`` (default 0 = unbounded).

    A lean hot-cache node (a Pi, a made-lean workstation) sets this so a growing corpus can never
    take the volume to 0 (the 2026-07-25 incident: an unbounded content cache filled a node's disk).
    A warehouse node leaves it unset and keeps the historical unbounded behavior — this is strictly
    opt-in, so it changes nothing unless configured.

    ⚠ BOTH NAMES ARE READ, AND THAT IS NOT TIDINESS. The store is being extracted as a standalone
    distribution, so its environment moves to a neutral `MANTLE_*` namespace — but deployed nodes
    still set the ember-era name. A hard rename would make `_min_free_bytes()` return 0 on every one
    of them: the floor silently unenforced, the cache unbounded, and the exact disk-fill incident
    this setting exists to prevent, reintroduced by a rename with no error anywhere."""
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


class ContentCacheError(Exception):
    """Base. Every failure mode below is DISTINCT and none of them is an empty return."""


class CacheMiss(ContentCacheError):
    """The object is not present locally. NOT an empty result — see the contract."""


class CacheCorrupt(ContentCacheError):
    """PROVEN corruption: the object decrypted (so AES-GCM authenticated the bytes) and the
    plaintext then failed its own sha256, or it is too short to be a ciphertext at all.

    ⚠ A DECRYPT FAILURE IS NOT A MISS — but it is not this either; see `ContentKeyMismatch`, which
    subclasses this so every existing `except CacheCorrupt` keeps working unchanged. MultiFernet-
    style key sets make a missing key a SILENT PARTITION rather than a loud failure: the object is
    there, the key is not, and the caller sees nothing. Reporting it as a miss would send the
    operator looking for a fetch bug when the real fault is key distribution.

    ⛔ ONLY THIS BASE CASE EVICTS. Eviction is irreversible, so it requires PROOF."""


class ContentKeyMismatch(CacheCorrupt):
    """The object could not be decrypted — WRONG KEY *or* altered bytes, and we cannot tell which.

    ⛔ NEVER EVICTED. `_decrypt` has always documented this failure as ambiguous; the code then took
    the irreversible action anyway, on `get()` — the one operation callers assume is safe. A failed
    AEAD decrypt is not proof of corruption, and destruction requires proof.

    MEASURED (EREA, first external consumer, 2026-07-28): 39 containment roots, 491 objects, refs
    byte-identical across projects. Verify pass 1 → "2 CORRUPT"; pass 2 → "6 not local, 0 corrupt".
    Reading destroyed them, and the damage COMPOUNDED on each subsequent read.

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
    """Derive a collection's at-rest key from its IMMUTABLE ORIGIN ROOT (P9.3).

    `origin_root` is the collection's origin-root principal id — the one field that, by design,
    NEVER MOVES. Deriving from a grant, an owner or `created_by` would re-key the whole
    collection every time authority changed, orphaning every object already written.

    ⚠ THIS IS NO LONGER THE CONTENT CACHE'S AT-REST KEY — see `shared_content_key`. It is kept,
    exported and tested for two live reasons, so do not "clean it up":
      1. the LEGACY DECRYPT FALLBACK that migrates already-written objects (see `_decrypt`);
      2. **the comms plane's per-GROUP keys** — `ember/reach.py::EmberKeyring.group_key` derives
         group keys from exactly this function. (It was cited here as
         `iris/comms/wiring.py:MantleKeyring.group_key` until 2026-07-30; behaviour and method name
         are unchanged, but that module is gone and there is now ONE implementation.) That
         isolation is real and is NOT what §1 traded away; only the content cache's use of it
         changed."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    if not origin_root:
        raise ValueError("collection_key requires a non-empty origin_root — deriving from an "
                         "empty principal would give every collection the SAME key")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_KDF_INFO + origin_root.encode("utf-8")).derive(root_secret)


def shared_content_key(root_secret: bytes) -> bytes:
    """THE at-rest key for local content — ONE key for the whole node, from the same root secret.

    ⛔ WHY THIS REPLACED PER-COLLECTION KEYING (EREA §1, 2026-07-28).

    `ref = cas/sha256(plaintext)` addresses one immutable, write-once object GLOBALLY. Encrypting it
    under a per-collection key made one address mean N different ciphertexts, only one of which
    opened — so a second containment root storing bytes the first already had would find the ref
    present, fail to verify it, EVICT IT, and rewrite under its own key. The first root's only copy
    was gone, silently, and a later read from it evicted the replacement in turn. Measured on EREA's
    corpus: 39 roots, 491 objects, 2 shared refs → 6 artifacts unreadable across two verify passes.

    ⚠ THE ISOLATION GIVEN UP HERE WAS NOMINAL, AND THAT IS THE ACTUAL ARGUMENT. Every deployment
    derives EVERY collection key from one root secret (`ember/sqlite_store.py:_open_lattice_content`
    reads `content.key`, then the full `collection → origin_root` map, and derives on demand), so
    any process that can open the store already holds all of them. It was never defense in depth.
    It would be real only where an oracle ISSUES per-collection keys and the node lacks the root
    secret — a mode `__init__`'s key injection still permits but no deployment uses.

    Secondary and weaker, but confirming: it did not survive promotion anyway. `content_tier`'s
    durable tier already stores the same bytes under the SHARED fleet cipher (`content.key`), so
    the local per-collection scope evaporated one `promote_one` later.

    ⚠ SAME ROOT SECRET, SO THERE IS NO NEW KEY TO DISTRIBUTE — this is a derivation change, not a
    key-management change. Nodes that already have `content.key` need nothing new.

    ⚠ AES-GCM WITH `AAD = ref` IS DELIBERATELY RETAINED. EREA asked for "the same shared content
    cipher the remote tier already uses", which is Fernet — and Fernet has NO AAD. Adopting it
    literally would drop the binding of ciphertext to content address, which exists precisely
    because "the sha256 check is exactly what an attacker who can write the cache directory would
    target". The requirement was one object / one key / one copy; that is what changed. The cipher
    did not need to."""
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
        """`key` is THE at-rest key (32 bytes) — normally `shared_content_key(root_secret)`.

        Injected rather than derived here so this module never holds the root secret and never
        decides who may have a key. Issuance is gated elsewhere; this object only encrypts and
        decrypts with a key it is handed.

        `legacy_key_for_collection(collection_id) -> 32 bytes` is the MIGRATION fallback, tried only
        when the shared key fails to open an object. Objects it opens are re-encrypted under `key`
        in place (see `get`), so a corpus written before §1 converges without a re-fetch and without
        downtime. Pass None once a store is known fully migrated.

        `is_durable(ref) -> bool` proves an object exists in the durable tier. Required for the
        disk-floor eviction in `_ensure_free` to run at all — without it nothing is evicted, because
        this class cannot otherwise tell a re-fetchable blob from the only copy. `TieredContentStore`
        wires it from its remote."""
        if key is None:
            raise ValueError(
                "FileContentCache requires `key` (the shared at-rest key, normally "
                "shared_content_key(root_secret)). The per-collection-keyed construction "
                "(`key_for_collection=`) is GONE: it addressed objects globally while keying them "
                "per collection, so one shared ref destroyed another root's copy on write and "
                "again on read (EREA §1). Pass `legacy_key_for_collection=` to migrate an existing "
                "corpus — it is a decrypt-only fallback, never used to write.")
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
        """Plaintext, or raise. NEVER returns b"" and never returns unverified bytes."""
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
            # ⛔ LOUD, BUT NOT DESTRUCTIVE. Ordered BEFORE the `CacheCorrupt` arm because it is a
            # subclass. Counted distinctly because the alternative is invisible: a key-mismatched
            # object makes every read fall through to the remote tier, which reads as a perfectly
            # healthy `remote_hit` rate while the node silently pays a WAN round-trip per read.
            with self._lock:
                self.stats["key_mismatch"] += 1
            _log.error("content cache: %s", e)
            raise
        except CacheCorrupt:
            self._evict(p)
            with self._lock:
                self.stats["corrupt"] += 1
            raise
        # ⚠ VERIFY ON READ, ALWAYS — the object is content-addressed, so this is the one check
        # that makes the local copy as trustworthy as the origin. Disk is just another untrusted
        # server: bit-rot, a truncated write, or anyone with write access to the cache directory.
        if hashlib.sha256(plain).hexdigest() != _hash_of(ref):
            self._evict(p)
            with self._lock:
                self.stats["corrupt"] += 1
            raise CacheCorrupt(
                "cached object does not hash to its own content address (%s) — EVICTED. This is "
                "corruption, not a miss: the bytes were present and wrong." % ref)
        if was_legacy:
            # LAZY MIGRATION. The object is verified (it decrypted AND hashed), so rewriting it
            # under the shared key is safe here and nowhere else. Ordered AFTER the sha256 check on
            # purpose: re-encrypting unverified bytes would launder corruption into a valid-looking
            # object under the current key. Best-effort — a failed rewrite must not fail the READ,
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

        ⛔ THE PLAINTEXT IS VERIFIED AGAINST THE REF BEFORE IT IS STORED. A short or truncated
        read must FAIL LOUDLY (§6A) — storing it would put a silently-truncated object behind a
        valid-looking address, and every later reader would trust it because the address matched
        the file name rather than the bytes.

        Idempotent and resumable: an object already present AND still hashing is left alone, so
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
                # ⛔ REFUSE TO OVERWRITE WHAT WE CANNOT VERIFY. This is the write half of the same
                # rule: the incumbent object may be perfectly good under a key we do not hold, and
                # rewriting it under ours makes its rightful holder's copy unreadable forever. That
                # is precisely how one shared ref used to destroy another root's content — silently,
                # with nothing logged on the loser's side. Failing here is loud and reversible.
                raise
            except ContentCacheError:
                pass                              # PROVEN corrupt: fall through and rewrite
        # Bounded cache: keep a free-space floor so a growing corpus can never fill the volume.
        # OPT-IN (EMBER_CACHE_MIN_FREE_GB); no-op unless a lean node configures it. Eviction is LRU
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
        volume. Returns the count evicted. Runs only when the volume is BELOW the floor; then frees
        a little extra (to ~1.25× the floor) so subsequent writes do not immediately re-scan.

        LRU by access time (falls back to mtime).

        ⛔ EVERY EVICTION REQUIRES `is_durable(ref)`. This method used to justify itself as "safe by
        tier design: a blob is re-fetchable from the S3 remote or re-ingestable" — but it never
        CHECKED, and that justification is false on a `remote=None` node, which `content_tier`
        declares first-class ("an air-gapped box, not a degraded one"). There, crossing the floor
        deleted content that existed nowhere else. `evict_local` and `evict_for_space` both already
        enforce "nothing is deleted that is not PROVEN remote"; this path disagreed with them, and
        it is the one that runs AUTOMATICALLY, on the write path, unattended.

        ⚠ WITH NO DURABILITY ORACLE, NOTHING IS EVICTED AND THE FLOOR GOES UNENFORCED — loudly. That
        is the honest outcome, not a regression: you cannot both bound the disk and keep the only
        copy, and silently choosing to delete data is the wrong way to resolve it. The operator gets
        a warning naming the choice; the alternative is the 2026-07-25 incident's mirror image,
        where a "bounded cache" quietly ate the corpus."""
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
                continue                          # can't PROVE it's durable → keep it
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
        """ALWAYS under the shared key. The legacy key is decrypt-only — writing under it again
        would re-create the very divergence §1 removed."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        nonce = os.urandom(12)
        # AAD = the ref. Binds ciphertext to its content address so an object cannot be moved
        # between refs on disk without the decrypt failing.
        return nonce + AESGCM(self._key).encrypt(nonce, plain, ref.encode("utf-8"))

    def _decrypt(self, blob: bytes, ref: str, collection: str) -> Tuple[bytes, bool]:
        """`(plaintext, was_legacy)`. `was_legacy=True` means the object opened only under the
        pre-§1 per-collection key and should be rewritten under the shared key."""
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        if len(blob) < 13:
            # Provable without any key: no ciphertext can be shorter than nonce + tag. This is the
            # one decrypt-side failure that IS proof, so it stays evictable `CacheCorrupt`.
            raise CacheCorrupt("cached object is too short to contain a nonce: %s" % ref)
        aad = ref.encode("utf-8")
        try:
            return AESGCM(self._key).decrypt(blob[:12], blob[12:], aad), False
        except InvalidTag:
            pass
        # ── legacy fallback: a pre-§1 object written under this collection's own key ──
        if self._legacy_key_for is not None:
            try:
                legacy = self._legacy_key_for(collection)
            except Exception:
                legacy = None                     # underivable (no origin_root) — nothing to try
            if legacy is not None:
                try:
                    return AESGCM(legacy).decrypt(blob[:12], blob[12:], aad), True
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
        """hit / miss / corrupt DISTINCTLY, plus where objects actually came from.

        ⚠ The tier split is load-bearing for the decommission decision, not colour. §13.5 grades
        a remote tier serving 0/N as `low` severity on the reasoning that it is a per-box hot cache in
        front of the durable origin — and says that reasoning "should be confirmed before it is
        relied on; if that tier were ever intended as a replica, 0/60 is a finding, not a cache
        miss." Recording the split per population run is what makes that confirmable later."""
        with self._lock:
            s = dict(self.stats)
            t = dict(self.tiers)
        # `key_mismatch` counts as a READ — it is a served request that returned no data. Leaving it
        # out would inflate `hit_rate` by shrinking the denominator, which is the exact shape of
        # metric that reads healthy on a node answering nothing (see [[metrics-are-not-capability]]).
        reads = s["hit"] + s["miss"] + s["corrupt"] + s["key_mismatch"]
        s["reads"] = reads
        s["hit_rate"] = round(s["hit"] / reads, 4) if reads else None
        s["tiers"] = t
        return s
