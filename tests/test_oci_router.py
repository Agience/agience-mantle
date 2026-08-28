"""`/v2` — the read-only OCI surface, over the store that already holds the bytes.

The claim this suite has to hold up is narrow and load-bearing: **a digest promoted through this
registry is the same number everywhere else**. `test_oci_store.py` proves that for the STORE (two
separate stores arrive at the same digest); this proves it for the WIRE — what a client asks for is
what it gets back, verified against its own name, and anything that is not a digest is refused with
a reason rather than served from a guess.

The auth dependency is overridden by `conftest.override_dependencies`, so these run as an
authenticated caller. That is deliberate and it is also the boundary worth naming: the router
requires authentication and the CAS read below it consults no grant, so these tests say nothing
about per-image authorization — there is none yet, and `oci_router`'s module docstring says so.
"""
from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from mantle.main import app
from mantle.routers import oci_router as mod

client = TestClient(app)

BODY = b'{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json"}'
DIGEST = "sha256:" + hashlib.sha256(BODY).hexdigest()


class _Store:
    """The narrowest stand-in that is still honest: content_handle's contract is an object the
    `mantle.shard.content` helpers read through, and this suite replaces `read_blob` itself, so the
    handle only has to be not-None."""


@pytest.fixture
def holds_the_blob(monkeypatch):
    """This node holds BODY under DIGEST and nothing else."""
    monkeypatch.setattr(mod, "content_handle", lambda: _Store())

    # `**kw` rather than a fixed signature: `read_blob` gained a `collection` keyword on
    # 2026-08-24 (repository == collection, and a node's store scopes reads by it). A double that
    # pins the old shape fails on the change instead of measuring what it exists to measure.
    def _read(content_store, keys_dir, digest, **kw):
        if digest != DIGEST:
            raise FileNotFoundError(digest)
        return BODY

    monkeypatch.setattr(mod, "read_blob", _read)


def test_the_version_endpoint_announces_the_api(holds_the_blob):
    """`GET /v2/` is what a client hits before anything else; the header is how it decides."""
    r = client.get("/v2/")
    assert r.status_code == 200
    assert r.json() == {}
    assert r.headers["Docker-Distribution-Api-Version"] == "registry/2.0"


def test_a_blob_comes_back_byte_identical_and_verifiable(holds_the_blob):
    """The whole point. The bytes hash to the name they were asked for.

    Re-hashing here rather than trusting the response is the same discipline the store applies on
    ingest: `oci.layout` refuses a blob that does not hash to its own name, and a registry that
    served one would turn a bad file into an unreadable image reported against the puller.
    """
    r = client.get("/v2/agience/agience-mantle/blobs/%s" % DIGEST)
    assert r.status_code == 200
    assert r.content == BODY
    assert "sha256:" + hashlib.sha256(r.content).hexdigest() == DIGEST
    assert r.headers["Docker-Content-Digest"] == DIGEST


def test_a_manifest_is_served_by_the_same_path_as_a_blob(holds_the_blob):
    """In the distribution spec a manifest IS a blob — `read_layout` stores it as blob zero. One
    read path, so a manifest cannot be served by code that verifies differently."""
    r = client.get("/v2/agience/agience-mantle/manifests/%s" % DIGEST)
    assert r.status_code == 200
    assert r.content == BODY
    assert r.headers["Docker-Content-Digest"] == DIGEST


def test_head_carries_the_real_length_and_no_body(holds_the_blob):
    """`Content-Length` on a HEAD is the number a client uses to decide whether to fetch, so zero
    would be worse than absent."""
    r = client.head("/v2/x/blobs/%s" % DIGEST)
    assert r.status_code == 200
    assert r.content == b""
    assert r.headers["Content-Length"] == str(len(BODY))


def test_a_tag_is_refused_and_the_refusal_says_why(holds_the_blob):
    """The design decision, pinned. A bare 404 would read as "this image does not exist here",
    which is a different and wrong statement — the registry has no opinion about tags at all.

    Both reasons are in the message because both are independently sufficient: the fleet's deploy
    path refuses tags, and resolving one would need the unindexed scan `mantle/oci/__init__.py`
    forbids by name.
    """
    r = client.get("/v2/agience/agience-mantle/manifests/latest")
    assert r.status_code == 404
    err = r.json()["errors"][0]
    assert err["code"] == "MANIFEST_UNKNOWN"
    assert "DIGEST only" in err["message"]
    assert "latest" in err["message"]


def test_a_digest_this_node_does_not_hold_is_a_clean_404(holds_the_blob):
    absent = "sha256:" + "0" * 64
    r = client.get("/v2/x/blobs/%s" % absent)
    assert r.status_code == 404
    assert r.json()["errors"][0]["code"] == "BLOB_UNKNOWN"


def test_an_unreadable_blob_is_NOT_reported_as_missing(monkeypatch):
    """The distinction this fleet keeps paying for. A wrong `content.key` makes every blob
    decrypt-fail, and `content_encryption` calls that a silent partition: the node is healthy,
    answers 200 elsewhere, and reads none of its own corpus.

    Flattened to 404, the investigation goes looking for a blob that is present. This asserts the
    two answers stay different, and that the message names the key.
    """
    monkeypatch.setattr(mod, "content_handle", lambda: _Store())

    def _boom(content_store, keys_dir, digest, **kw):
        raise ValueError("decryption failed")

    monkeypatch.setattr(mod, "read_blob", _boom)
    r = client.get("/v2/x/blobs/%s" % DIGEST)
    assert r.status_code == 500
    assert "content.key" in r.json()["errors"][0]["message"]


def test_an_index_only_node_says_so_rather_than_404(monkeypatch):
    """`content_handle()` returning None is a legal node shape, not a missing image. A client told
    "not found" would conclude the image does not exist anywhere; 503 says "ask another node"."""
    monkeypatch.setattr(mod, "content_handle", lambda: None)
    r = client.get("/v2/x/blobs/%s" % DIGEST)
    assert r.status_code == 503
    assert r.json()["errors"][0]["code"] == "UNAVAILABLE"


def test_the_router_offers_no_write_surface():
    """Read-only is a property of the mounted routes, not of intent.

    Push is a separate act with its own hazards — write auth, ref-update races, garbage collection —
    and it is where "authorization is the encryption" has to earn its keep, because a push is a
    grant. Asserted against the app's own route table so adding one is a deliberate, visible change.
    """
    writes = {(m, r.path) for r in app.routes
              for m in (getattr(r, "methods", None) or ())
              if str(getattr(r, "path", "")).startswith("/v2")
              and m in {"PUT", "POST", "PATCH", "DELETE"}}
    assert not writes, "the /v2 surface has acquired write routes: %s" % sorted(writes)
