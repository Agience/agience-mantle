"""An image into the lattice and back out — against a real content store, not a mock.

The test that carries the deploy claim is `test_two_separate_stores_arrive_at_the_same_digest`.
Foundation and foresight are fully separate — separate stores, separate keys, separate containers
— and each ingests the image independently. "Both boxes run the same artifact" is therefore only
true if two stores that never see each other agree on the name. That is what content addressing
buys, and it is worth measuring rather than believing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.fernet import InvalidToken

from mantle.oci.layout import read_layout
from mantle.oci.store import ImageRecord, ingest_image, read_blob
from mantle.shard.sqlite_store import FsContentStore

from ._oci_layout import make_layout   # the one fixture builder


def _store(tmp: Path, name: str = "s"):
    """A content store and its keys dir. `put_content` mints the first content key deliberately."""
    root = tmp / name
    keys = root / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    return FsContentStore(str(root / "cas")), keys


# ── in and out ───────────────────────────────────────────────────────────────────────────────────

def test_an_image_goes_in_and_every_blob_reads_back_identical(tmp_path: Path):
    root, man = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    cs, keys = _store(tmp_path)

    rec = ingest_image(cs, keys, image, repository="agience-mantle")
    assert rec.digest == man
    assert len(rec.blobs) == 4 and all(b.stored for b in rec.blobs)

    for blob in image.blobs:
        got = read_blob(cs, keys, blob.digest)
        assert hashlib.sha256(got).hexdigest() == blob.digest.split(":")[1], \
            "blob %s did not survive the round trip" % blob.digest


def test_the_manifest_is_stored_as_a_blob_like_any_other(tmp_path: Path):
    """It is content with a digest; nothing about it needs a second mechanism."""
    root, man = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    cs, keys = _store(tmp_path)
    ingest_image(cs, keys, image, repository="agience-mantle")

    body = read_blob(cs, keys, man)
    assert json.loads(body)["schemaVersion"] == 2
    assert "sha256:" + hashlib.sha256(body).hexdigest() == man


def test_ingest_is_idempotent_and_says_what_it_skipped(tmp_path: Path):
    """Re-ingesting writes nothing. Reported, because '0 written' is expected for an unchanged
    image and alarming for a changed one — the caller can only tell those apart if it is said."""
    root, _ = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    cs, keys = _store(tmp_path)

    first = ingest_image(cs, keys, image, repository="agience-mantle")
    second = ingest_image(cs, keys, image, repository="agience-mantle")
    assert all(b.stored for b in first.blobs)
    assert not any(b.stored for b in second.blobs)
    assert first.digest == second.digest


def test_a_rebuild_only_writes_the_layers_that_changed(tmp_path: Path):
    """The dedupe that makes promoting a rebuild cheap — and it is a property of the address,
    not of a cache."""
    root_a, _ = make_layout(tmp_path / "a", layers=2)
    image_a = read_layout(root_a)[0]
    cs, keys = _store(tmp_path)
    ingest_image(cs, keys, image_a, repository="agience-mantle")

    # A second image sharing this one's layers but not its manifest.
    root_b, _ = make_layout(tmp_path / "b", layers=2, ref_name="agience-mantle:next")
    image_b = read_layout(root_b)[0]
    rec = ingest_image(cs, keys, image_b, repository="agience-mantle")
    shared = [b for b in rec.blobs if not b.stored]
    assert shared, "identical layers were re-written — content addressing is not deduping"


# ── the claim the deploy makes ───────────────────────────────────────────────────────────────────

def test_two_separate_stores_arrive_at_the_same_digest(tmp_path: Path):
    """Foundation and foresight, proven rather than asserted.

    Two stores, separate roots and separate content keys, neither able to see the other. Ingest the
    same layout into both and the promotable name must be identical — otherwise "both boxes run the
    same artifact" is a claim nothing can check.
    """
    root, man = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]

    foundation, f_keys = _store(tmp_path, "foundation")
    foresight, s_keys = _store(tmp_path, "foresight")
    a = ingest_image(foundation, f_keys, image, repository="agience-mantle")
    b = ingest_image(foresight, s_keys, image, repository="agience-mantle")

    assert a.digest == b.digest == man
    assert [x.digest for x in a.blobs] == [x.digest for x in b.blobs]

    # And the stores really are separate: different keys, so the bytes at rest differ.
    assert (f_keys / "content.key").read_bytes() != (s_keys / "content.key").read_bytes(), \
        "the two stores share a content key — they are not separate and the test proves nothing"
    assert read_blob(foundation, f_keys, man) == read_blob(foresight, s_keys, man)


def test_a_store_cannot_read_the_other_stores_bytes(tmp_path: Path):
    """The separation is real, not just two directories. A content key is what divides them.

    Two things this asserts that a bare `pytest.raises(Exception)` could not. First, the blob is
    genuinely there under foundation's own key — otherwise a `KeyError` from a byte range nobody
    stored satisfies the refusal without decryption ever being attempted, and the test goes on
    passing after the separation stops holding. Second, the refusal is `InvalidToken` specifically,
    which is what Fernet raises for a key that does not open the ciphertext — the failure that
    means "wrong key", not "wrong path".
    """
    root, man = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    foundation, f_keys = _store(tmp_path, "foundation")
    _foresight, s_keys = _store(tmp_path, "foresight")
    ingest_image(foundation, f_keys, image, repository="agience-mantle")
    # Mint the other node's key by writing something under it, then try it against foundation.
    ingest_image(_foresight, s_keys, image, repository="agience-mantle")

    # Control: the bytes ARE in foundation, and foundation's own key opens them.
    assert read_blob(foundation, f_keys, man)

    with pytest.raises(InvalidToken):
        read_blob(foundation, s_keys, man)


# ── ordering and refusals ────────────────────────────────────────────────────────────────────────

def test_the_manifest_is_written_last(tmp_path: Path, monkeypatch):
    """Refuses a window in which the store holds a manifest naming content it does not have —
    200 for the manifest, 404 for a layer, which reads as a corrupt image rather than a running
    ingest."""
    root, man = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    cs, keys = _store(tmp_path)

    order = []
    import mantle.oci.store as mod
    real = mod.put_content
    # `**kw` rather than a fixed signature: `put_content` gained a `collection` keyword on
    # 2026-08-24 (repository == collection), and a double that pins the old shape fails on the
    # change instead of measuring the ordering it exists to measure. Forwarded verbatim so the
    # real call still receives it.
    monkeypatch.setattr(mod, "put_content",
                        lambda s, k, d, **kw: (order.append(hashlib.sha256(d).hexdigest()),
                                               real(s, k, d, **kw))[1])
    ingest_image(cs, keys, image, repository="agience-mantle")
    assert order[-1] == man.split(":")[1], "the manifest was not written last"


def test_the_repository_is_the_collection_the_blobs_land_in(tmp_path: Path, monkeypatch):
    """The ruling, asserted [John, 2026-08-24]. `oci/__init__.py` has always said "repository ==
    a collection"; until now nothing supplied it, and on a real node — whose store SCOPES writes —
    the ingest could not run at all.

    Checked at the seam rather than through a scoping store, because `FsContentStore` (what these
    tests use) accepts no collection and would swallow the distinction. What matters is that the
    value reaching `put_content` is the repository the caller named.
    """
    root, _man = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    cs, keys = _store(tmp_path)

    seen = []
    import mantle.oci.store as mod
    real = mod.put_content
    monkeypatch.setattr(mod, "put_content",
                        lambda s, k, d, **kw: (seen.append(kw.get("collection")), real(s, k, d))[1])
    ingest_image(cs, keys, image, repository="agience-mantle")

    assert seen and set(seen) == {"agience-mantle"}, \
        "blobs landed in %s, not in the repository's collection" % set(seen)


def test_the_record_carries_digests_not_store_refs(tmp_path: Path):
    """An artifact records what the image IS called. The store's internal spelling would outlive
    its usefulness and tie the record to one backend."""
    root, man = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    cs, keys = _store(tmp_path)
    ctx = ingest_image(cs, keys, image, repository="agience-mantle").as_context()

    assert ctx["oci_digest"] == man
    assert ctx["oci_repository"] == "agience-mantle"
    assert ctx["oci_tag"] == "edge"
    assert all(b["digest"].startswith("sha256:") for b in ctx["oci_blobs"])
    assert "cas/" not in json.dumps(ctx), "a store ref leaked into the artifact record"


def test_a_reordered_blob_list_is_refused(tmp_path: Path):
    """The manifest must be first — `read_layout` builds it that way, and if something changes
    that, writing order silently stops meaning what it says."""
    root, _ = make_layout(tmp_path / "layout")
    image = read_layout(root)[0]
    cs, keys = _store(tmp_path)
    scrambled = image._replace(blobs=list(reversed(image.blobs)))
    with pytest.raises(ValueError, match="must be the manifest itself"):
        ingest_image(cs, keys, scrambled, repository="agience-mantle")
