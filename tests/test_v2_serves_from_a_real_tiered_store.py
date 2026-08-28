"""`/v2` reading through the store a DEPLOYED node actually has — not a stand-in.

This is the seam that was broken and the one no other test covered. `test_oci_router.py` replaces
`read_blob` with a double, so it measures the router's shape — status codes, headers, refusals — and
says nothing about whether the read underneath works. `test_oci_store.py` builds an `FsContentStore`
directly, so it measures the store against the fixture's idea of a store.

Between those two sat the actual defect: a node's `content_handle()` returns a `TieredContentStore`
whose `get` is `get(ref, *, collection)` and returns PLAINTEXT, while `get_content` was written for
`FsContentStore.get(ref)` returning ciphertext. **`/v2` would have answered 500 on every pull from a
real node**, and both suites would have stayed green. It was found by running the ingest end to end,
not by reading either function.

So this test does the one thing neither of those does: a real tiered store, a real ingest, and the
router's own read path — no doubles below the HTTP layer.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mantle.main import app
from mantle.routers import oci_router as mod
from mantle.shard.content import put_content
from mantle.shard.sqlite_store import FsContentStore

client = TestClient(app)

BODY = b'{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json"}'
DIGEST = "sha256:" + hashlib.sha256(BODY).hexdigest()
REPO = "agience/agience-mantle"


@pytest.fixture
def tiered_node(tmp_path: Path, monkeypatch):
    """A real `TieredContentStore` over a real CAS, holding BODY under DIGEST in REPO's collection.

    Built the way `db.backend.content_handle` builds one, so the object under test is the shape a
    deployed node has — the whole point of this file. The content key is minted the way any node's
    first key is: by writing one byte through `put_content`.
    """
    from mantle.shard.sqlite_store import _open_content_tier, _open_lattice_content

    root = tmp_path / "store"
    keys = root / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    put_content(FsContentStore(str(root / "cas")), keys, b"seed")

    cache = _open_lattice_content(str(root), str(keys), str(root / "lattice.db"))
    tiered = _open_content_tier(cache, str(keys))
    assert type(tiered).__name__ == "TieredContentStore", \
        "this test is only meaningful against the tiered store; got %s" % type(tiered).__name__

    # Written through the same call the ingest uses, into the same collection a puller will name.
    put_content(tiered, str(keys), BODY, collection=REPO)

    monkeypatch.setattr(mod, "content_handle", lambda: tiered)
    monkeypatch.setenv("KEYS_DIR", str(keys))
    return tiered


def test_a_blob_is_served_from_a_real_tiered_store(tiered_node):
    """The regression, pinned. Before the read side was fixed this raised
    `TypeError: get() missing 1 required keyword-only argument: 'collection'` and the router turned
    it into a 500 — a pull that fails on every node while every test passes."""
    r = client.get("/v2/%s/blobs/%s" % (REPO, DIGEST))
    assert r.status_code == 200, r.text
    assert r.content == BODY
    assert "sha256:" + hashlib.sha256(r.content).hexdigest() == DIGEST
    assert r.headers["Docker-Content-Digest"] == DIGEST


def test_the_manifest_path_reads_the_same_way(tiered_node):
    """A manifest IS a blob; one read path. If these ever diverge, one of them is verifying
    differently from the other."""
    r = client.get("/v2/%s/manifests/%s" % (REPO, DIGEST))
    assert r.status_code == 200, r.text
    assert r.content == BODY


def test_the_collection_is_not_a_storage_scope_and_this_pins_that(tiered_node):
    """This test asserted the opposite and was wrong. Kept, inverted, as the record.

    It read: *"a wrong repository must miss — reading across scopes to be forgiving would be
    authorization-by-accident"*, and it failed, because content **is** served under a repository the
    caller did not write it under.

    That is deliberate, and `content_cache.py` explains it: objects are addressed GLOBALLY by ref
    (`path_of` uses the hash alone), and per-collection KEYING is not used for writes because
    "addressing objects globally while keying them per collection would let one shared ref overwrite
    another root's copy". `collection` selects the LEGACY per-collection key on read, nothing more.

    So the honest assertion is the one below, and the consequence worth recording is that **`/v2`
    offers no per-repository isolation**. `oci_router`'s module docstring already says the CAS read
    consults no grant; this is the same boundary seen from the collection side, and it is why the
    deployment rule there ("do not put a blob in a node's CAS that its authenticated callers may not
    read") is the real control rather than the repository string.
    """
    assert client.get("/v2/%s/blobs/%s" % (REPO, DIGEST)).status_code == 200

    other = client.get("/v2/agience-mantle/blobs/%s" % DIGEST)
    assert other.status_code == 200, (
        "a differently-named repository stopped resolving — if storage became collection-scoped, "
        "this test and the docstrings that cite it need rewriting, not deleting")


def test_a_digest_this_store_does_not_hold_is_a_clean_404(tiered_node):
    """Still a 404 and not a 500, against the real store: 'absent' and 'unreadable' stay different
    answers, which is the distinction that sends an investigation to the right node."""
    absent = "sha256:" + "0" * 64
    r = client.get("/v2/%s/blobs/%s" % (REPO, absent))
    assert r.status_code == 404, r.text
    assert r.json()["errors"][0]["code"] == "BLOB_UNKNOWN"


class TestACorruptLocalObjectIsNotReportedAsAMiss:
    """A local object that will not decrypt must not surface as "not in this tier".

    Measured 2026-08-25 on 71/home: all 313,982 CAS-backed artifacts failed to hydrate, and the
    error the caller saw was `ContentStoreUnavailable: ... is not in this node's local content
    tier and there is no object-store address to look up (the artifact records no content_key)`.
    Every clause of that was misleading. The blob was present (`ref in cache` was True, the file
    was on disk), the tier root was correct, and the real error four layers down was
    `ContentKeyMismatch: AES-GCM tag mismatch` — the content key had been replaced and the CAS was
    written under the previous one.

    `TieredContentStore.get` already carries the rule — "corrupt must not be laundered into miss by
    the tiering" — and applied it only when `self.remote is None`. With a remote configured, the
    local `ContentKeyMismatch` was discarded and a `RemoteMiss` raised in its place, which sent the
    diagnosis after the object store instead of after the key.
    """

    @staticmethod
    def _tier(local_exc, remote_exc=RuntimeError("no such key")):
        from mantle.db.content_tier import TieredContentStore

        class _Cache:
            root = "/tmp/cas"

            def get(self, ref, *, collection=None):
                raise local_exc

            def __contains__(self, ref):
                return True                      # present on disk, which is the whole point

        class _Remote:
            def get(self, ref):
                raise remote_exc

        tier = TieredContentStore.__new__(TieredContentStore)
        tier.cache = _Cache()
        tier.remote = _Remote()
        tier._count = lambda *_a, **_k: None
        return tier

    def test_a_key_mismatch_survives_the_remote_miss(self):
        from mantle.db.content_cache import ContentKeyMismatch
        import pytest

        tier = self._tier(ContentKeyMismatch("AES-GCM tag mismatch for cas/abc"))
        with pytest.raises(ContentKeyMismatch) as excinfo:
            tier.get("cas/abc", collection="stage.1.grammar")
        assert "tag mismatch" in str(excinfo.value), (
            "the caller must be told the key does not fit, not that the object is absent"
        )

    def test_a_plain_miss_still_defers_to_the_remote(self):
        """Genuinely absent locally and absent remotely is a remote miss — that report is right."""
        from mantle.db.content_cache import CacheMiss
        from mantle.db.content_tier import RemoteMiss
        import pytest

        tier = self._tier(CacheMiss("not here"))
        with pytest.raises(RemoteMiss):
            tier.get("cas/abc", collection="c")

    def test_a_corrupt_object_with_no_remote_is_unchanged(self):
        """The pre-existing guard, still holding: no remote, so the local error is re-raised."""
        from mantle.db.content_cache import ContentKeyMismatch
        import pytest

        tier = self._tier(ContentKeyMismatch("tag mismatch"))
        tier.remote = None
        with pytest.raises(ContentKeyMismatch):
            tier.get("cas/abc", collection="c")
