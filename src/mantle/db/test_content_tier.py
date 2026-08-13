"""Unit tests for the tiered content substrate (`content_tier.py` + the `s3_content.py` seam).

Each test pins a contract clause from the module docstrings — the loud-miss / distinct-corrupt /
verify-on-pull rules — or guards against a failure mode (unbounded re-cache, silent CDN death,
promote-of-unreadable). Run:

    python -m pytest src/mantle/db/test_content_tier.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest

#: `<repo>/src/mantle/db/` → three levels up is `<repo>/src`, which is what makes the fully
#: qualified `mantle.*` imports below resolve in an uninstalled checkout. Asserted, not trusted: a
#: `sys.path.insert` of the wrong directory is a silent no-op.
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
assert os.path.basename(_SRC) == "src" and os.path.isdir(os.path.join(_SRC, "mantle", "db")), (
    "path depth is wrong: expected <repo>/src, resolved %s — fix the depth" % _SRC)
sys.path.insert(0, _SRC)

from mantle.db import s3_content as s3c                           # noqa: E402
from mantle.db.content_cache import (CacheCorrupt, CacheMiss,     # noqa: E402
                                             ContentKeyMismatch,
                                             FileContentCache)
from mantle.db.content_tier import (ContentKeyMissing, RemoteCorrupt,  # noqa: E402
                                            RemoteMiss, TieredContentStore,
                                            content_cipher)
from mantle.db.s3_content import cache_control_for                # noqa: E402

import hashlib


def _ref(plain: bytes) -> str:
    return "cas/" + hashlib.sha256(plain).hexdigest()


class FakeRemote:
    """dict-backed ContentStore double. Counts gets so 'remote touched exactly once' is provable."""

    def __init__(self):
        self.objects = {}
        self.gets = 0

    def put(self, key, data, content_type="application/octet-stream"):
        self.objects[key] = data

    def get(self, key):
        self.gets += 1
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    def exists(self, key):
        return key in self.objects

    def delete(self, key):
        self.objects.pop(key, None)


@pytest.fixture()
def cipher():
    from cryptography.fernet import Fernet
    return Fernet(Fernet.generate_key())


def _collection_key(collection: str) -> bytes:
    """Distinct per collection — the legacy (pre-§1) at-rest derivation.

    It still earns its keep after §1: it is what the migration fallback must open."""
    return hashlib.blake2b(collection.encode("utf-8"), digest_size=32).digest()


SHARED_KEY = b"s" * 32          # stands in for shared_content_key(root_secret)


@pytest.fixture()
def cache(tmp_path):
    return FileContentCache(str(tmp_path / "cas"), key=SHARED_KEY)


@pytest.fixture()
def legacy_cache(tmp_path):
    """A cache holding pre-§1 objects: written per-collection, read under the shared key.

    `legacy_key_for_collection` is decrypt-only — this fixture writes through a second cache
    configured the old way, so the migration path is exercised against real legacy bytes rather
    than a mock."""
    def legacy_key_for(collection):
        if collection == "unkeyed":
            raise KeyError("no origin_root for %r" % collection)
        return _collection_key(collection)
    root = str(tmp_path / "cas")
    return (FileContentCache(root, key=SHARED_KEY, legacy_key_for_collection=legacy_key_for),
            _LegacyWriter(root, legacy_key_for))


class _LegacyWriter:
    """Writes objects exactly as the pre-§1 cache did: AES-GCM, AAD=ref, per-collection key."""

    def __init__(self, root, key_for):
        self.root, self.key_for = root, key_for

    def write(self, ref: str, plain: bytes, collection: str) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        h = ref[len("cas/"):]
        p = os.path.join(self.root, h[:2], h[2:4], h)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        nonce = os.urandom(12)
        blob = nonce + AESGCM(self.key_for(collection)).encrypt(nonce, plain, ref.encode("utf-8"))
        with open(p, "wb") as fh:
            fh.write(blob)


@pytest.fixture()
def tier(cache, cipher):
    return TieredContentStore(cache, FakeRemote(),
                              decrypt=cipher.decrypt, encrypt=cipher.encrypt)


# ── wiring refusals ──────────────────────────────────────────────────────────
def test_no_tier_at_all_refuses():
    with pytest.raises(ValueError):
        TieredContentStore(None, None)


def test_remote_without_decrypt_refuses(cache):
    with pytest.raises(ValueError):
        TieredContentStore(cache, FakeRemote())


# ── read path ────────────────────────────────────────────────────────────────
def test_local_hit_never_touches_remote(tier):
    plain = b"the local working set answers with the backing unreachable"
    ref = _ref(plain)
    tier.put(ref, plain, collection="c1")
    assert tier.get(ref, collection="c1") == plain
    assert tier.remote.gets == 0                       # the air-gap invariant, measured


def test_read_through_verifies_recaches_then_serves_locally(tier, cipher):
    plain = b"pulled once from the durable origin, local ever after"
    ref = _ref(plain)
    tier.remote.put(ref, cipher.encrypt(plain))        # only the remote holds it
    assert tier.get(ref, collection="c1") == plain     # miss -> pull -> verify -> re-cache
    assert tier.get(ref, collection="c1") == plain     # now a local hit
    assert tier.remote.gets == 1
    assert tier.stats["remote_hit"] == 1
    assert tier.cache.stats["hit"] == 1


def test_absent_everywhere_raises_never_empty(tier):
    with pytest.raises(RemoteMiss):
        tier.get("cas/" + "0" * 64, collection="c1")


def test_wrong_object_behind_ref_is_remote_corrupt_and_not_cached(tier, cipher):
    plain, other = b"the real object", b"a different object entirely"
    ref = _ref(plain)
    tier.remote.put(ref, cipher.encrypt(other))        # poisoned edge / corrupted origin
    with pytest.raises(RemoteCorrupt):
        tier.get(ref, collection="c1")
    assert ref not in tier.cache                       # never cached, never returned
    assert tier.stats["remote_corrupt"] == 1


def test_undecryptable_remote_is_corrupt_not_miss(tier):
    ref = "cas/" + "a" * 64
    tier.remote.put(ref, b"not fernet ciphertext at all")
    with pytest.raises(RemoteCorrupt):
        tier.get(ref, collection="c1")


def test_content_key_missing_propagates_unwrapped(cache):
    def decrypt(_):
        raise ContentKeyMissing("no content.key — node-wide fault")
    t = TieredContentStore(cache, FakeRemote(), decrypt=decrypt)
    ref = "cas/" + "b" * 64
    t.remote.put(ref, b"whatever")
    with pytest.raises(ContentKeyMissing):             # not RemoteCorrupt — callers must be able
        t.get(ref, collection="c1")                    # to tell config fault from bad object


def test_local_corrupt_with_no_remote_stays_corrupt(cache):
    plain = b"corrupt must not be laundered into miss by the tiering"
    ref = _ref(plain)
    cache.put(ref, plain, collection="c1")
    with open(cache.path_of(ref), "wb") as fh:         # rot the at-rest bytes
        fh.write(b"garbage")
    t = TieredContentStore(cache, None)
    with pytest.raises(CacheCorrupt):
        t.get(ref, collection="c1")


def test_an_unkeyable_collection_is_no_longer_a_thing(tier, cipher):
    """A collection with no `origin_root` caches like any other. The at-rest key does not depend
    on the collection, so there is nothing left to be underivable and no read is forced to the
    WAN."""
    plain = b"a collection with no origin_root now caches like any other"
    ref = _ref(plain)
    tier.remote.put(ref, cipher.encrypt(plain))
    assert tier.get(ref, collection="unkeyed") == plain
    assert tier.stats["recache_failed"] == 0
    assert tier.get(ref, collection="unkeyed") == plain
    assert tier.remote.gets == 1                       # ...and the second read is LOCAL


def test_recache_failure_is_still_counted_when_the_write_fails(tier, cipher, monkeypatch):
    """The `recache_failed` stat outlives its original trigger. Read-through re-caching is
    best-effort by design, so when it fails the node silently pays a WAN round-trip per read —
    'every read is a WAN read' must stay visible rather than folded into a healthy hit rate."""
    plain = b"the durable tier still serves; the local write is what failed"
    ref = _ref(plain)
    tier.remote.put(ref, cipher.encrypt(plain))

    def boom(*a, **k):
        raise OSError("cache volume is read-only")
    monkeypatch.setattr(tier.cache, "put", boom)
    assert tier.get(ref, collection="c1") == plain     # served anyway — a write fault is not a miss
    assert tier.stats["recache_failed"] == 1


# ── §1: one immutable object, one address, one key ───────────────────────────
def test_shared_ref_is_readable_by_every_collection(cache):
    """One immutable object at one address, referenced by many roots, readable by all. §1.

    `content_ref` exists because it is the same content: `cas/sha256(plaintext)` names a write-once
    blob, so two artifacts sharing a ref is the design working, not a collision. Per-collection
    keying cannot honor that — one object, N keys, only one of which opens — so the shared key is
    what makes it dedup instead."""
    plain = b"byte-identical across containment roots (a consumer: guidance.md, 4 projects)"
    ref = _ref(plain)
    assert cache.put(ref, plain, collection="c1") is True
    for c in ("c2", "c3", "c4"):
        assert cache.put(ref, plain, collection=c) is False    # dedup: present, verified, no-op
    for c in ("c1", "c2", "c3", "c4"):
        assert cache.get(ref, collection=c) == plain
    assert cache.report()["put"] == 1                          # ONE copy on disk, not four


def test_repeated_verify_is_idempotent_across_collections(cache):
    """A verify sweep must be idempotent: reading is not a write, so a pass that evicts must not
    make a later pass find rows missing that the first pass never touched."""
    plain = b"verify pass 1 evicted; verify pass 2 found nothing left to evict"
    ref = _ref(plain)
    cache.put(ref, plain, collection="c1")
    for _ in range(2):
        for c in ("c1", "c2"):
            assert cache.get(ref, collection=c) == plain
    assert cache.report()["evicted"] == 0


# ── §2: destroying data requires proof, and a failed decrypt is not proof ────
# A rotated `content.key`, an unmounted keys volume, or any wiring error can still present the
# wrong key, and `get()` — the one operation callers assume is safe — must not delete on it.
# Induced the way it happens in production: a second cache over the same directory whose node key
# differs.


def _rekeyed_cache(cache, key=b"z" * 32):
    """The same CAS directory, opened under a DIFFERENT node key (a rotated content.key)."""
    return FileContentCache(cache.root, key=key)


def test_wrong_key_read_does_not_destroy_the_object(cache):
    plain = b"a failed decrypt is not proof of corruption"
    ref = _ref(plain)
    cache.put(ref, plain, collection="c1")
    wrong = _rekeyed_cache(cache)
    with pytest.raises(ContentKeyMismatch):
        wrong.get(ref, collection="c1")                  # loud, never returned as data
    assert os.path.exists(cache.path_of(ref)), "read under a wrong key DESTROYED the object"
    assert cache.get(ref, collection="c1") == plain      # intact once the right key is back


def test_put_refuses_to_overwrite_what_it_cannot_verify(cache):
    """Refusing is loud and reversible; rewriting is silent and permanent."""
    plain = b"the incumbent may be perfectly good under a key we do not hold"
    ref = _ref(plain)
    cache.put(ref, plain, collection="c1")
    wrong = _rekeyed_cache(cache)
    with pytest.raises(ContentKeyMismatch):
        wrong.put(ref, plain, collection="c1")
    assert cache.get(ref, collection="c1") == plain      # the rightful holder is unharmed


def test_key_mismatch_is_counted_distinctly_and_does_not_inflate_hit_rate(cache):
    """`key_mismatch` is its own counter, and it counts as a read.

    Not decoration. Leaving the object in place means every read of it now falls through to the
    remote tier — correct, but it turns a local read into a WAN round-trip. Folded into `miss` that
    is indistinguishable from a cold cache; left out of `reads` entirely it would shrink the
    denominator and inflate `hit_rate`, so a node answering nothing locally would report perfect
    health. Both are the metric-that-cannot-fail shape."""
    plain = b"a WAN round-trip per read must be visible, not folded into a healthy hit rate"
    ref = _ref(plain)
    cache.put(ref, plain, collection="c1")
    wrong = _rekeyed_cache(cache)
    with pytest.raises(ContentKeyMismatch):
        wrong.get(ref, collection="c1")
    r = wrong.report()
    assert r["key_mismatch"] == 1
    assert r["corrupt"] == 0, "a wrong key is not proven corruption"
    assert r["miss"] == 0, "and it is not a miss either — the object is right there"
    assert r["evicted"] == 0
    assert r["reads"] == 1 and r["hit_rate"] == 0.0      # honest: one read, nothing served


def test_proven_corruption_still_evicts(cache):
    """The counterweight: `get()` must not become a no-op. An object that decrypts (so AES-GCM
    authenticated the bytes) and then fails its own sha256 is proven bad, and still goes."""
    plain = b"present and wrong is not the same as present and unreadable"
    ref = _ref(plain)
    cache.put(ref, plain, collection="c1")
    # Re-encrypt different bytes under the right key at the same ref: decrypts cleanly, wrong hash.
    cache._atomic_write(cache.path_of(ref), cache._encrypt(b"different bytes entirely", ref))
    with pytest.raises(CacheCorrupt) as ei:
        cache.get(ref, collection="c1")
    assert not isinstance(ei.value, ContentKeyMismatch)
    assert not os.path.exists(cache.path_of(ref))        # proven corrupt -> evicted
    assert cache.report()["corrupt"] == 1


# ── §1 migration: a pre-existing corpus converges without a re-fetch ─────────
def test_legacy_object_is_read_and_rekeyed_in_place(legacy_cache):
    """A corpus written before §1 must keep answering, and migrate itself as it is read.

    Nothing is deleted and nothing is re-fetched: the object is opened under its old per-collection
    key, verified, then rewritten under the shared key. This is what makes the cutover safe on a
    populated node — the alternative (re-key everything up front) would mass-evict on first read."""
    cache, legacy = legacy_cache
    plain = b"written under the pre-S1 per-collection key"
    ref = _ref(plain)
    legacy.write(ref, plain, "c1")
    assert cache.get(ref, collection="c1") == plain      # opened via the fallback
    assert cache.report()["rekeyed"] == 1
    # Now readable without the fallback — proof it was rewritten, not just decrypted.
    migrated = FileContentCache(cache.root, key=SHARED_KEY)
    assert migrated.get(ref, collection="c1") == plain
    assert migrated.report()["rekeyed"] == 0


def test_legacy_shared_ref_converges_for_every_root(legacy_cache):
    """The migration converges the per-collection keys §1 removed: two roots, one ref, written
    under two different legacy keys. Whichever is read first wins the rewrite, and the other then
    opens it under the shared key, so the ref ends up readable by both roots."""
    cache, legacy = legacy_cache
    plain = b"one ref, two roots, two legacy keys"
    ref = _ref(plain)
    legacy.write(ref, plain, "c2")                       # last writer under the legacy scheme
    assert cache.get(ref, collection="c2") == plain      # opens via c2's legacy key, rekeys
    assert cache.get(ref, collection="c1") == plain      # c1 could not read this under the legacy scheme


def test_migration_never_rekeys_unverified_bytes(legacy_cache):
    """Ordering guard: the re-key happens after the sha256 check, never before. Rewriting
    unverified bytes would launder corruption into a valid-looking object under the current key —
    it would then decrypt perfectly and be trusted forever."""
    cache, legacy = legacy_cache
    ref = _ref(b"the real object")
    legacy.write(ref, b"NOT the object this ref names", "c1")   # legacy-keyed but wrong content
    with pytest.raises(CacheCorrupt):
        cache.get(ref, collection="c1")
    assert cache.report()["rekeyed"] == 0
    assert cache.report()["corrupt"] == 1


# ── write / promote / evict ──────────────────────────────────────────────────
def test_put_is_local_only_promotion_is_explicit(tier):
    plain = b"writes never block on the WAN"
    ref = _ref(plain)
    tier.put(ref, plain, collection="c1")
    assert not tier.remote.exists(ref)                 # nothing implicit went up
    assert tier.promote_one(ref, collection="c1") == "put"
    assert tier.remote.exists(ref)
    assert tier.promote_one(ref, collection="c1") == "skip"   # idempotent = write-once = immutable


def test_promoted_ciphertext_round_trips(tier, cipher):
    plain = b"what goes up must decrypt fleet-wide"
    ref = _ref(plain)
    tier.put(ref, plain, collection="c1")
    tier.promote_one(ref, collection="c1")
    assert cipher.decrypt(tier.remote.objects[ref]) == plain


def test_promote_of_unreadable_local_raises(tier):
    with pytest.raises(CacheMiss):                     # never put garbage behind a valid address
        tier.promote_one("cas/" + "c" * 64, collection="c1")


def test_evict_only_after_remote_confirmed(tier):
    plain = b"never delete the only copy"
    ref = _ref(plain)
    tier.put(ref, plain, collection="c1")
    assert tier.evict_local(ref) is False              # not in remote yet -> refused
    tier.promote_one(ref, collection="c1")
    assert tier.evict_local(ref) is True
    assert ref not in tier.cache
    assert tier.get(ref, collection="c1") == plain     # …and it re-caches from the remote


def test_cacheless_write_through(cipher):
    t = TieredContentStore(None, FakeRemote(),
                           decrypt=cipher.decrypt, encrypt=cipher.encrypt)
    plain = b"migration tooling writes straight through"
    ref = _ref(plain)
    assert t.put(ref, plain, collection="c1") is True
    assert t.put(ref, plain, collection="c1") is False          # idempotent
    assert t.get(ref, collection="c1") == plain                 # pull + verify, no cache


def test_evict_for_space_spares_the_only_copy(tier):
    kept = b"never promoted - the local copy is the ONLY copy"
    gone = b"promoted - safe to evict, re-caches on demand"
    kept_ref, gone_ref = _ref(kept), _ref(gone)
    tier.put(kept_ref, kept, collection="c1")
    tier.put(gone_ref, gone, collection="c1")
    tier.promote_one(gone_ref, collection="c1")
    out = tier.evict_for_space(min_free_bytes=2 ** 62)   # unsatisfiable floor: evict all eligible
    assert out["evicted"] == 1
    assert kept_ref in tier.cache                        # the only copy survives, always
    assert gone_ref not in tier.cache
    assert tier.get(gone_ref, collection="c1") == gone   # ...and comes back through the remote


# ── the disk-floor evictor must obey the same rule as every other eviction ───
def test_disk_floor_never_evicts_without_a_durability_oracle(cache, monkeypatch, caplog):
    """With no oracle the floor goes unenforced and says so. You cannot both bound the disk and keep
    the only copy; silently choosing to delete data is the wrong way to resolve that."""
    plain = b"the only copy - no remote tier to re-fetch from"
    ref = _ref(plain)
    cache.put(ref, plain, collection="c1")
    assert cache._is_durable is None                      # standalone cache: nothing to ask
    monkeypatch.setenv("MANTLE_CACHE_MIN_FREE_GB", "999999")   # unsatisfiable floor
    with caplog.at_level("WARNING"):
        cache.put(_ref(b"another"), b"another", collection="c1")
    assert ref in cache, "the disk floor destroyed the only copy"
    assert cache.report()["evicted"] == 0
    assert "NOTHING WAS EVICTED" in caplog.text           # unenforced, and loud about it


def test_disk_floor_spares_the_only_copy_even_with_an_oracle(tier, monkeypatch):
    """With a remote wired, the floor evicts — but still only what is proven durable."""
    kept, gone = b"never promoted - the only copy", b"promoted - safe to evict"
    kept_ref, gone_ref = _ref(kept), _ref(gone)
    tier.put(kept_ref, kept, collection="c1")
    tier.put(gone_ref, gone, collection="c1")
    tier.promote_one(gone_ref, collection="c1")
    assert tier.cache._is_durable is not None             # wired by TieredContentStore
    monkeypatch.setenv("MANTLE_CACHE_MIN_FREE_GB", "999999")
    tier.put(_ref(b"trigger"), b"trigger", collection="c1")
    assert kept_ref in tier.cache, "the only copy was evicted to hold a disk floor"
    assert gone_ref not in tier.cache                     # durable -> costs a re-fetch, never loss


def test_deprecated_env_name_is_still_honored(monkeypatch, caplog):
    """A hard rename would return 0 on every deployed node — floor unenforced, cache unbounded,
    and a disk-fill risk introduced silently, with no error raised anywhere."""
    from mantle.db.content_cache import _min_free_bytes
    monkeypatch.delenv("MANTLE_CACHE_MIN_FREE_GB", raising=False)
    monkeypatch.setenv("EMBER_CACHE_MIN_FREE_GB", "2")
    with caplog.at_level("WARNING"):
        assert _min_free_bytes() == 2 * 1024 ** 3
    assert "DEPRECATED" in caplog.text
    monkeypatch.setenv("MANTLE_CACHE_MIN_FREE_GB", "5")   # the new name wins when both are set
    assert _min_free_bytes() == 5 * 1024 ** 3


def test_evict_for_space_noop_when_floor_met(tier):
    plain = b"nothing to do when the volume is comfortable"
    ref = _ref(plain)
    tier.put(ref, plain, collection="c1")
    tier.promote_one(ref, collection="c1")
    out = tier.evict_for_space(min_free_bytes=1)         # any real volume has >1 byte free
    assert out["evicted"] == 0
    assert ref in tier.cache


# ── the shared-cipher loader ─────────────────────────────────────────────────
def test_content_cipher_reads_never_mint(tmp_path):
    with pytest.raises(ContentKeyMissing):
        content_cipher(tmp_path)                       # empty dir: refuse, do not fabricate
    assert not (tmp_path / "content.key").exists()     # and nothing was written


def test_content_cipher_primary_plus_fallbacks(tmp_path):
    from cryptography.fernet import Fernet
    old, new = Fernet.generate_key(), Fernet.generate_key()
    (tmp_path / "content.key").write_bytes(new)
    (tmp_path / "content.key.old").write_bytes(old)
    mf = content_cipher(tmp_path)
    assert mf.decrypt(Fernet(old).encrypt(b"peer segment under the OLD key")) \
        == b"peer segment under the OLD key"           # decrypt-only fallback works
    assert Fernet(new).decrypt(mf.encrypt(b"x")) == b"x"        # primary encrypts


# ── the s3 seam ──────────────────────────────────────────────────────────────
def test_cache_control_only_for_cas_keys():
    assert cache_control_for("cas/" + "d" * 64) == "public, max-age=31536000, immutable"
    assert cache_control_for("mesh/segments/node45/000001.seg") is None
    assert cache_control_for("content.promote.cursor") is None


def _bare_s3(read_base, s3_stub):
    """An S3ContentStore without boto3: __init__ needs a client, the read seam does not."""
    from mantle.db.s3_content import S3ContentStore
    o = S3ContentStore.__new__(S3ContentStore)
    o.bucket = "b"
    o.read_url_base = read_base
    o.stats = {"cdn_hit": 0, "cdn_fallback": 0}
    o._s3 = s3_stub
    return o


class _S3Stub:
    def __init__(self):
        self.gets = 0

    def get_object(self, Bucket, Key):
        self.gets += 1
        import io
        return {"Body": io.BytesIO(b"from-s3-api")}


def test_cdn_read_hit_skips_s3_api(monkeypatch):
    monkeypatch.setattr(s3c, "_http_get", lambda url, timeout=30.0: b"from-cdn")
    o = _bare_s3("https://cdn.example", _S3Stub())
    assert o.get("cas/" + "e" * 64) == b"from-cdn"
    assert o.stats == {"cdn_hit": 1, "cdn_fallback": 0}
    assert o._s3.gets == 0


def test_cdn_failure_falls_back_to_s3_and_is_counted(monkeypatch):
    def boom(url, timeout=30.0):
        raise OSError("edge down")
    monkeypatch.setattr(s3c, "_http_get", boom)
    o = _bare_s3("https://cdn.example", _S3Stub())
    assert o.get("cas/" + "f" * 64) == b"from-s3-api"
    assert o.stats == {"cdn_hit": 0, "cdn_fallback": 1}   # a dead CDN is a visible fact
    assert o._s3.gets == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
