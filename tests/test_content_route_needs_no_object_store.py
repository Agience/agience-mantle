"""`PUT /artifacts/{id}/content` on a node with no object store.

The route's byte path is `services/content_service.py`, and it used to be one call to
`put_object`: a node with no bucket credentials failed EVERY upload, and the read that followed
found nothing. This file holds the shape that replaced it — the tier `db/content_tier.py` already
implements for ingest, reached through `db.backend.content_handle()`:

  * write   → the node's own encrypted CAS first, the object store after it, and a node with no
              object store is a complete configuration rather than a degraded one;
  * read    → local first, the object store behind it;
  * neither → a named failure, not a `NoCredentialsError` behind a generic 500.

The envelope does not change to make that work. Both tiers hold the same `MEC1‖nonce‖ct` bytes
under the owner's per-principal key (`services/content_crypto.py`); the local tier then encrypts
that ciphertext AGAIN at rest under the node key. `tests/test_content_handle.py` holds the
neighbouring claim for the ingest pair — a node with no S3 still has a content handle.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mantle.services import content_service as cs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def envelope(monkeypatch):
    """The REAL `content_crypto` envelope, with the key oracle replaced by a fixed provider.

    Not a fake envelope: `encrypt_content`/`decrypt_content` run unmodified, so the MEC1 wire
    format, the AAD binding and the per-principal key separation are the ones the node uses. Only
    the master-key SOURCE is injected — the oracle needs a store, a grant ledger and an acting
    principal, none of which this file is about.
    """
    from mantle.services import content_crypto

    import hashlib

    real_encrypt, real_decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content

    def _provider(principal_id: str) -> bytes:
        # Per-principal and distinct, which is the only property these tests lean on: two owners
        # must not open each other's blobs.
        return hashlib.sha256(("master:" + principal_id).encode("utf-8")).digest()

    monkeypatch.setattr(content_crypto, "encrypt_content",
                        lambda pid, pt, **kw: real_encrypt(pid, pt, master_key_provider=_provider, **kw))
    monkeypatch.setattr(content_crypto, "decrypt_content",
                        lambda pid, blob, **kw: real_decrypt(pid, blob, master_key_provider=_provider, **kw))
    return _provider


@pytest.fixture()
def node(tmp_path: Path, monkeypatch):
    """A provisioned node with NO object store: a store path and a keys directory, nothing else.

    Deliberately no `content.key` and no `cas/` directory — `mantle-init-keys` writes neither, so
    this is the state a fresh standalone node is actually in, and the write path has to reach a
    working local tier from here.
    """
    root = tmp_path / "var"
    (root / "keys").mkdir(parents=True)
    monkeypatch.setenv("MANTLE_LATTICE_PATH", str(root / ".data" / "lattice.db"))
    (root / ".data").mkdir()
    monkeypatch.setenv("KEYS_DIR", str(root / "keys"))

    import mantle.db.backend as backend
    backend._CONTENT = None                    # the handle is process-wide; isolate the test
    cs._LOCAL_TIER_ABSENT_LOGGED = False
    with patch.object(cs, "edge_store_configured", return_value=False):
        yield root / ".data", root / "keys"
    backend._CONTENT = None


def _cas_path(store_root: Path, ref: str) -> Path:
    h = ref[len("cas/"):]
    return store_root / "cas" / h[:2] / h[2:4] / h


# ---------------------------------------------------------------------------
# The write works, and the read finds it
# ---------------------------------------------------------------------------

def test_a_node_with_no_object_store_stores_content_and_serves_it_back(node, envelope):
    """The defect, stated as the property it broke: an upload succeeds and the download returns
    the same bytes, on a node that has no bucket and no credentials."""
    store_root, _keys = node
    payload = bytes(range(256)) * 40

    ref = cs.put_bytes_encrypted("artifacts/a-1.content", payload, "application/octet-stream",
                                 "owner-a")

    assert cs.is_cas_ref(ref), "the write must report where it put the bytes"
    assert cs.get_bytes_decrypted("artifacts/a-1.content", "owner-a", cas_ref=ref) == payload


def test_the_bytes_land_in_the_local_cas_beside_the_store(node, envelope):
    """`<store root>/cas/<aa>/<bb>/<sha256>` — the same directory the ingest path writes and
    `shard/content_tier.promote_local_content` drains, not a second one. The store root is
    `AGIENCE_BASE_DIR/.data` unless `MANTLE_LATTICE_PATH` moves the store, and it is read through
    config rather than recomputed."""
    store_root, _keys = node
    ref = cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")
    assert _cas_path(store_root, ref).is_file()


def test_the_write_path_provisions_what_it_needs_and_never_the_keys_volume(node, envelope):
    """A fresh node has no `content.key` and no `cas/`. A write may mint the first content key —
    the rule `shard/content.py` already applies to `put_content` — but an absent keys DIRECTORY
    means the volume is not mounted, and that is never created."""
    store_root, keys_dir = node
    assert not (keys_dir / "content.key").exists()

    cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")

    assert (keys_dir / "content.key").is_file()
    assert (store_root / "cas").is_dir()


def test_an_unmounted_keys_volume_is_never_created(tmp_path: Path, monkeypatch, envelope):
    store_root = tmp_path / "data"
    store_root.mkdir()
    monkeypatch.setenv("MANTLE_LATTICE_PATH", str(store_root / "lattice.db"))
    monkeypatch.setenv("KEYS_DIR", str(tmp_path / "not-mounted"))
    import mantle.db.backend as backend
    backend._CONTENT = None
    try:
        with patch.object(cs, "edge_store_configured", return_value=False):
            with pytest.raises(cs.ContentStoreUnavailable):
                cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")
    finally:
        backend._CONTENT = None
    assert not (tmp_path / "not-mounted").exists(), \
        "minting a key into a directory that should have been mounted partitions the node silently"


# ---------------------------------------------------------------------------
# The envelope is unchanged, and it is the same on both legs
# ---------------------------------------------------------------------------

def test_both_tiers_hold_the_identical_envelope(node, envelope):
    """The object store's bytes and the CAS object's address are the same `MEC1` blob. The local
    leg is not a re-encryption under a different key and not a plaintext copy."""
    store_root, _keys = node
    fake = MagicMock()
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        ref = cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")

    sent = fake.put_object.call_args.kwargs["Body"]
    assert sent[:4] == b"MEC1", "the object store still receives the per-principal envelope"
    assert ref == cs.cas_ref_for(sent), "the CAS addresses exactly the bytes the object store got"


def test_nothing_reaches_the_disk_in_the_clear(node, envelope):
    """At rest the local object is the envelope encrypted AGAIN under the node key (AES-GCM,
    AAD = the ref), so neither the plaintext nor the envelope's own magic is on disk."""
    store_root, _keys = node
    payload = b"the quick brown fox jumps over the lazy dog"
    ref = cs.put_bytes_encrypted("artifacts/a-1.content", payload, "text/plain", "owner-a")

    on_disk = _cas_path(store_root, ref).read_bytes()
    assert payload not in on_disk
    assert b"MEC1" not in on_disk
    assert len(on_disk) > len(payload)


def test_content_without_an_owner_never_enters_the_content_addressed_store(node, envelope):
    """No owner means no envelope, and an unenveloped object in a globally addressed CAS is
    readable by anyone who can name its address. That case keeps the old behaviour."""
    fake = MagicMock()
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        ref = cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", None)
    assert ref is None
    assert fake.put_object.call_args.kwargs["Body"] == b"body"


# ---------------------------------------------------------------------------
# The light cone: naming an address is not authorization to read it
# ---------------------------------------------------------------------------

def test_a_corpus_object_is_never_served_as_this_routes_content(node, envelope):
    """The CAS is ONE address space, shared with everything this node has ever stored, and the ref
    arrives from caller-writable artifact context. An object with no envelope — every object the
    ingest path wrote — must not be readable by naming its address.

    The counterpart of `workspace_service._safe_content_key`, one level down: that binds the object
    store's key to its artifact, this refuses a content address that opens to something this route
    did not write."""
    tier = cs.local_content_tier(bootstrap=True)
    corpus = b"an article body the caller has no grant for"
    ref = cs.cas_ref_for(corpus)
    tier.put(ref, corpus, collection=None)

    with pytest.raises(cs.ContentDecryptionError):
        cs.get_bytes_decrypted("artifacts/attacker.content", "owner-a", cas_ref=ref)


def test_content_enveloped_for_another_principal_is_refused(node, envelope):
    """Naming someone else's address yields an error, never their bytes: the envelope's key and
    AAD decide, and they were not written for this principal."""
    ref = cs.put_bytes_encrypted("artifacts/a-1.content", b"alice's private bytes",
                                 "text/plain", "owner-a")
    with pytest.raises(cs.ContentDecryptionError):
        cs.get_bytes_decrypted("artifacts/b-1.content", "owner-b", cas_ref=ref)


def test_a_ref_that_is_not_a_content_address_is_not_addressed_at_all(node, envelope):
    """Shape first, so a ref cannot name a path or an object outside the CAS."""
    for bad in ("cas/../../etc/passwd", "cas/" + "A" * 64, "cas/short", "artifacts/a-1.content",
                "", None, 7):
        assert cs.is_cas_ref(bad) is False
    assert cs.is_cas_ref("cas/" + "0123456789abcdef" * 4) is True


def test_content_cannot_be_overwritten_by_naming_its_address(node, envelope):
    """A content-addressed store has no overwrite: the address IS the hash, so bytes that are not
    the named content are refused rather than stored over it."""
    from mantle.db.content_cache import CacheCorrupt
    tier = cs.local_content_tier(bootstrap=True)
    ref = cs.cas_ref_for(b"the real content")
    tier.put(ref, b"the real content", collection=None)
    with pytest.raises(CacheCorrupt):
        tier.put(ref, b"attacker-chosen content", collection=None)


# ---------------------------------------------------------------------------
# Idempotence, and the mirror
# ---------------------------------------------------------------------------

def test_the_same_bytes_are_stored_once(node, envelope):
    """Content-addressed, so a repeat write of the same object is a no-op rather than a duplicate.
    (The route decides one step earlier, on `sha256` of the plaintext, because a fresh envelope
    nonce would otherwise make identical content look new.)"""
    tier = cs.local_content_tier(bootstrap=True)
    blob = b"identical bytes"
    ref = cs.cas_ref_for(blob)
    assert tier.put(ref, blob, collection=None) is True
    assert tier.put(ref, blob, collection=None) is False
    assert cs.local_content_has(ref) is True
    assert cs.local_content_has(cs.cas_ref_for(b"never written")) is False


def test_an_unreachable_mirror_does_not_fail_a_write_that_already_landed(node, envelope):
    """Ingest never blocks on the WAN, and neither does this. The bytes are stored and verified on
    this node; failing the request to report an unreachable mirror would discard a durable write."""
    fake = MagicMock()
    fake.put_object.side_effect = OSError("connection refused")
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        ref = cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")

    assert cs.is_cas_ref(ref)
    assert cs.get_bytes_decrypted("artifacts/a-1.content", "owner-a", cas_ref=ref) == b"body"


def test_a_local_copy_answers_without_touching_the_mirror(node, envelope):
    """A node with a local copy and an unreachable mirror reads normally — the remote tier is
    consulted only on a local miss."""
    ref = cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")
    fake = MagicMock()
    fake.get_object.side_effect = AssertionError("the mirror must not be touched on a local hit")
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        assert cs.get_bytes_decrypted("artifacts/a-1.content", "owner-a", cas_ref=ref) == b"body"
    fake.get_object.assert_not_called()


def test_a_mirror_failure_with_no_local_copy_still_raises(node, envelope):
    """Swallowing that would report a success for bytes nothing holds."""
    fake = MagicMock()
    fake.put_object.side_effect = OSError("connection refused")
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True), \
            patch.object(cs, "local_content_tier", return_value=None):
        with pytest.raises(OSError):
            cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")


# ---------------------------------------------------------------------------
# The object store leg still behaves as it did
# ---------------------------------------------------------------------------

def test_the_object_store_still_gets_the_write_and_still_answers_the_read(node, envelope):
    """This must not become a local-only route. An S3-configured node writes the same key with the
    same bytes, and serves them when the local copy is gone."""
    store_root, _keys = node
    fake = MagicMock()
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        ref = cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")
        assert fake.put_object.call_args.kwargs["Key"] == "artifacts/a-1.content"
        stored = fake.put_object.call_args.kwargs["Body"]

        _cas_path(store_root, ref).unlink()          # the local working set is evicted
        fake.get_object.return_value = {"Body": MagicMock(read=lambda: stored)}
        assert cs.get_bytes_decrypted("artifacts/a-1.content", "owner-a", cas_ref=ref) == b"body"
    fake.get_object.assert_called_once()


def test_a_legacy_plaintext_object_store_object_still_passes_through(node, envelope):
    """Objects predating envelope encryption genuinely carry no magic; that is compatibility on the
    object-store leg, and it is exactly what the CAS leg refuses."""
    fake = MagicMock()
    fake.get_object.return_value = {"Body": MagicMock(read=lambda: b"pre-envelope bytes")}
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        assert cs.get_bytes_decrypted("artifacts/a-1.content", "owner-a") == b"pre-envelope bytes"


def test_the_inline_content_mirror_writes_no_unreferenced_cas_object(node, envelope):
    """`put_text_direct` mirrors content whose authoritative copy is the artifact document itself.
    It records no ref, so a CAS object written for it would be one nothing points at."""
    store_root, _keys = node
    fake = MagicMock()
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        cs.put_text_direct("artifacts/a-1.content", "inline body", owner_id="owner-a")
    fake.put_object.assert_called_once()
    assert not (store_root / "cas").exists() or not any((store_root / "cas").rglob("*"))


# ---------------------------------------------------------------------------
# Neither tier
# ---------------------------------------------------------------------------

def test_no_tier_at_all_is_a_named_failure(node, envelope):
    """A node with neither a local content tier nor an object store cannot serve this route, and
    says which two things are missing — the `NoCredentialsError` behind a generic 500 could not."""
    with patch.object(cs, "local_content_tier", return_value=None), \
            patch.object(cs, "edge_store_configured", return_value=False):
        with pytest.raises(cs.ContentStoreUnavailable) as exc:
            cs.put_bytes_encrypted("artifacts/a-1.content", b"body", "text/plain", "owner-a")
    assert "object store" in str(exc.value) and "local content tier" in str(exc.value)


def test_a_read_with_nowhere_to_look_is_not_a_server_fault(node, envelope):
    """`ContentStoreUnavailable` rather than a boto parameter error, so the route can answer 404
    ("no copy here") instead of 500 ("something broke")."""
    with patch.object(cs, "edge_store_configured", return_value=False):
        with pytest.raises(cs.ContentStoreUnavailable):
            cs.get_bytes_decrypted("artifacts/a-1.content", "owner-a",
                                   cas_ref=cs.cas_ref_for(b"never stored"))


# ---------------------------------------------------------------------------
# The route itself, over HTTP, against a real store
# ---------------------------------------------------------------------------

@pytest.fixture()
def routed(node, envelope, tmp_path: Path):
    """The app's content routes wired to a real lattice on the same node the fixtures provisioned.

    A MagicMock store cannot hold this: the write records the content's address on the artifact,
    and the read has to find it there. Only a real store round-trips that.
    """
    from types import SimpleNamespace

    from mantle.db import lattice_api
    from mantle.entities.artifact import Artifact
    from mantle.main import app
    from mantle.services.dependencies import get_store_db

    store_root, _keys = node
    db = lattice_api.LatticeDatabase(str(tmp_path / "routed.db"), origin="node-a")
    lattice_api.create_artifact(db, Artifact(
        id="a-1", root_id="a-1", collection_id="", name="upload", content="",
        created_by="owner-a", content_type="application/octet-stream"))

    grant = SimpleNamespace(can_read=True, can_create=True, can_update=True, can_delete=True,
                            can_invoke=True, can_add=True, can_share=True, resource_id=None)
    app.dependency_overrides[get_store_db] = lambda: db
    with patch("mantle.routers.artifacts_router.check_access", return_value=grant):
        yield db, store_root
    app.dependency_overrides.pop(get_store_db, None)


@pytest.mark.asyncio
async def test_put_then_get_round_trips_binary_on_a_node_with_no_object_store(routed, client):
    """The whole defect, end to end: `PUT /artifacts/{id}/content` → `GET .../content`, binary,
    byte for byte, on a node that has no bucket. It used to 500 and then 404."""
    _db, store_root = routed
    payload = bytes(range(256)) * 8 + b"\x00\xff\x00"

    put = await client.put("/artifacts/a-1/content", content=payload,
                           headers={"content-type": "application/octet-stream"})
    assert put.status_code == 200, put.text
    body = put.json()
    assert body["stored"] is True and body["size"] == len(payload)
    assert cs.is_cas_ref(body["content_ref"])
    assert _cas_path(store_root, body["content_ref"]).is_file()

    got = await client.get("/artifacts/a-1/content")
    assert got.status_code == 200
    assert got.content == payload


@pytest.mark.asyncio
async def test_the_same_upload_twice_stores_one_object(routed, client):
    """Deterministic and content-addressed: the second upload writes nothing."""
    _db, store_root = routed
    payload = b"identical bytes, uploaded twice"

    first = await client.put("/artifacts/a-1/content", content=payload)
    second = await client.put("/artifacts/a-1/content", content=payload)

    assert first.json()["deduplicated"] is False
    assert second.json()["deduplicated"] is True
    assert second.json()["content_ref"] == first.json()["content_ref"]
    assert len(list((store_root / "cas").glob("*/*/*"))) == 1, "one object, not two"


@pytest.mark.asyncio
async def test_the_upload_records_both_addresses_on_the_artifact(routed, client):
    """The object store is addressed by `content_key` and the local CAS by the content's own
    address; only the artifact knows both, so an upload that recorded neither was unfindable."""
    db, _store_root = routed
    import json as _json

    put = await client.put("/artifacts/a-1/content", content=b"body")
    ctx = _json.loads(db.artifacts.get_artifact("a-1")["context"])
    assert ctx["content_key"] == "artifacts/a-1.content"
    assert ctx["content_cas_ref"] == put.json()["content_ref"]
    assert ctx["content_sha256"] == __import__("hashlib").sha256(b"body").hexdigest()


@pytest.mark.asyncio
async def test_an_artifact_with_no_stored_content_is_404_not_500(routed, client):
    """"Nothing here" and "something broke" are different answers, and only one is about the
    request."""
    got = await client.get("/artifacts/a-1/content")
    assert got.status_code == 404


def test_no_object_store_means_no_bucket_probe(node, envelope):
    """Quiet, not warned-about: a node that deliberately has no object store used to log
    'Unable to locate credentials' about a bucket at every boot."""
    cs._BUCKET_CHECKED = False
    fake = MagicMock()
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=False):
        cs.ensure_content_bucket()
    fake.head_bucket.assert_not_called()
    fake.create_bucket.assert_not_called()
    cs._BUCKET_CHECKED = False
