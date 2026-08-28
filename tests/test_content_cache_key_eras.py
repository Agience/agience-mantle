"""A key rotation must not orphan the objects already on disk.

Measured 2026-08-25 on 71/home: `keys/content.key` was replaced on 2026-08-24 at 11:05, and all
313,982 CAS-backed artifacts stopped being readable. Nothing was corrupt — 197 blobs sampled across
12 random shards split 138 under the previous key and 59 under the current one, with NONE failing
both. A rotation rewrites no objects, so the store immediately holds two populations that differ
only in which key opens them, and with a single key configured the older one is indistinguishable
from corruption.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from mantle.db.content_cache import (CacheMiss, ContentKeyMismatch, FileContentCache,
                                     shared_content_key)

_OLD = shared_content_key(b"the key era that wrote the corpus")
_NEW = shared_content_key(b"the key era after the rotation")


@pytest.fixture
def root():
    with tempfile.TemporaryDirectory() as d:
        yield os.path.join(d, "cas")


def _written_under(root, key, payload=b"the body of an article"):
    """Store an object under `key` and return its ref — i.e. simulate the earlier era.

    The ref IS the address of the plaintext, so it is the same in every era; only the bytes on
    disk differ. That is exactly why a rotation is invisible until something tries to read.
    """
    import hashlib

    ref = "cas/" + hashlib.sha256(payload).hexdigest()
    FileContentCache(root, key=key).put(ref, payload, collection="c")
    return ref, payload


class TestAPreviousEraStillOpens:

    def test_the_current_key_alone_cannot_read_the_old_era(self, root):
        """The failure this reproduces: a whole corpus reading as corrupt after a rotation."""
        ref, _ = _written_under(root, _OLD)
        after = FileContentCache(root, key=_NEW)
        with pytest.raises(ContentKeyMismatch):
            after.get(ref, collection=None)

    def test_a_previous_key_opens_it(self, root):
        ref, payload = _written_under(root, _OLD)
        after = FileContentCache(root, key=_NEW, previous_keys=[_OLD])
        assert after.get(ref, collection=None) == payload

    def test_the_current_era_still_opens_alongside(self, root):
        """Both populations are live; a fallback that broke the current one would be no fix."""
        ref, payload = _written_under(root, _NEW)
        cache = FileContentCache(root, key=_NEW, previous_keys=[_OLD])
        assert cache.get(ref, collection=None) == payload

    def test_a_previous_key_read_is_counted(self, root):
        """The size of the un-migrated population must be visible, not inferred."""
        ref, _ = _written_under(root, _OLD)
        cache = FileContentCache(root, key=_NEW, previous_keys=[_OLD])
        cache.get(ref, collection=None)
        assert cache.report()["previous_key_reads"] == 1

    def test_reading_does_not_rewrite_by_default(self, root):
        """`rekey_previous` defaults False: a read must not become a 1.47 GB rewrite by accident."""
        ref, _ = _written_under(root, _OLD)
        cache = FileContentCache(root, key=_NEW, previous_keys=[_OLD])
        cache.get(ref, collection=None)
        assert cache.report()["rekeyed"] == 0
        # still only openable by the old era — nothing was converged behind the reader's back
        assert FileContentCache(root, key=_NEW).__contains__(ref)
        with pytest.raises(ContentKeyMismatch):
            FileContentCache(root, key=_NEW).get(ref, collection=None)

    def test_an_object_under_no_known_era_still_refuses(self, root):
        """The refusal must survive: an unknown key is not laundered into a hit."""
        ref, _ = _written_under(root, shared_content_key(b"an era nobody configured"))
        cache = FileContentCache(root, key=_NEW, previous_keys=[_OLD])
        with pytest.raises(ContentKeyMismatch):
            cache.get(ref, collection=None)

    def test_a_missing_object_is_still_a_miss_not_a_key_problem(self, root):
        cache = FileContentCache(root, key=_NEW, previous_keys=[_OLD])
        with pytest.raises(CacheMiss):
            cache.get("cas/" + "0" * 64, collection=None)

    def test_a_wrong_length_previous_key_is_refused_at_construction(self, root):
        """A 32-byte contract, checked where it is cheap to check."""
        with pytest.raises(ValueError):
            FileContentCache(root, key=_NEW, previous_keys=[b"too short"])
