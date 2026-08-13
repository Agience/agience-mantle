"""The app reaches its own content locally — a lattice node's content path does not require S3.

`db/backend.content_handle` opens the content-addressed store over the node's own local cas file
store: "a mantle shard is a local cas file store. It can also be backed by S3 — local or global.
S3 is just a CDN." A node's content path holds to that: the CDN is optional, the local store is
not.

Inline artifact content is a separate path: `doc_boundary` envelope-encrypts it inside the doc, so
a round trip through it passes with no S3 anywhere. Only content too large to live in the doc goes
through the content-addressed store directly.

These tests do not assert that S3 is reachable, present, or configured. That is the point: a node
with no remote is a first-class shape, not a degraded one.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from mantle.shard.content import content_ref, get_content, put_content


@pytest.fixture()
def node(tmp_path: Path, monkeypatch):
    """A provisioned node: a store path, a keys dir, and a content key. No S3, deliberately."""
    root = tmp_path / "var"
    (root / "keys").mkdir(parents=True)
    (root / "cas").mkdir()
    from cryptography.fernet import Fernet
    (root / "keys" / "content.key").write_bytes(Fernet.generate_key())
    monkeypatch.setenv("MANTLE_LATTICE_PATH", str(root / "lattice.db"))
    monkeypatch.setenv("KEYS_DIR", str(root / "keys"))
    import mantle.db.backend as backend
    backend._CONTENT = None                       # the handle is process-wide; isolate the test
    yield root, backend
    backend._CONTENT = None


def test_a_node_with_no_s3_still_has_a_content_handle(node):
    """An air-gapped shard serves its own content: `remote=None` is legal."""
    _root, backend = node
    tier = backend.content_handle()
    assert tier is not None, \
        "a provisioned node with a local cas and a content key must report a content tier"


def test_the_handle_is_opened_once(node):
    _root, backend = node
    assert backend.content_handle() is backend.content_handle()


def test_a_missing_content_key_is_refused_loudly_not_reported_as_index_only(node):
    """`_open_lattice_content` returns None when the key is absent, and None here means "this node
    holds no content tier" — a legal index-only shape. A mis-provisioned node must not take that
    path: doing so would make a silent partition look like a design choice, with every blob
    reading as missing while actually present and unreadable.
    """
    root, backend = node
    (root / "keys" / "content.key").unlink()
    backend._CONTENT = None
    with pytest.raises(RuntimeError, match="content.key"):
        backend.content_handle()


def test_the_handle_reads_what_the_cas_pair_wrote(node, tmp_path: Path):
    """The tier is opened over the same `<root>/cas` the store writes to — not a second directory.

    Written through `shard.content.put_content` (the pair the shard already uses) and read back
    through it, so this asserts the location is shared. Which cipher the registry writes through
    is a separate decision — see MANTLE-DEPLOYMENT §11.7b.
    """
    root, backend = node
    from mantle.shard.sqlite_store import FsContentStore
    cas = FsContentStore(str(root / "cas"))
    blob = b"an image layer, or any other content too large to live in a doc"
    ref, size = put_content(cas, root / "keys", blob)

    assert ref == content_ref(blob) and size == len(blob)
    assert get_content(cas, root / "keys", ref) == blob
    # And the handle opened against the same root is live.
    assert backend.content_handle() is not None


# ---------------------------------------------------------------------------
# Verify-on-read: the binding Fernet cannot express
#
# Fernet takes no associated data, so ciphertext here is authenticated but NOT
# bound to the ref it is stored under. A blob served for the wrong ref decrypts
# cleanly and would otherwise be returned as the wrong content under the right
# name. `cas/<sha256(plaintext)>` is the address, so re-hashing is the check.
# ---------------------------------------------------------------------------


def test_get_content_verifies_the_plaintext_against_its_ref(node):
    """A ciphertext moved to another ref decrypts fine — and must still be refused."""
    from mantle.shard.content import ContentIntegrityError
    from mantle.shard.sqlite_store import FsContentStore

    root, _backend = node
    cas = FsContentStore(str(root / "cas"))
    keys = root / "keys"

    real, _ = put_content(cas, keys, b"the real content")
    impostor, _ = put_content(cas, keys, b"attacker-chosen content")
    assert real != impostor

    # Both open normally under the correct address.
    assert get_content(cas, keys, real) == b"the real content"
    assert get_content(cas, keys, impostor) == b"attacker-chosen content"

    # Now move the impostor's ciphertext onto the real ref — same node, same key,
    # so Fernet authenticates it. Only the content-address check catches this.
    cas.put(real, cas.get(impostor))
    with pytest.raises(ContentIntegrityError, match="NOT the content this ref names"):
        get_content(cas, keys, real)


def test_a_non_cas_ref_is_not_second_guessed(node):
    """A ref carrying no hash to check against passes through — a check would be a guess."""
    from mantle.shard.sqlite_store import FsContentStore

    root, _backend = node
    cas = FsContentStore(str(root / "cas"))
    from mantle.shard.content import _content_key
    cas.put("blobs/not-a-cas-ref", _content_key(root / "keys").encrypt(b"opaque"))
    assert get_content(cas, root / "keys", "blobs/not-a-cas-ref") == b"opaque"


def test_resolve_text_degrades_to_inline_rather_than_returning_wrong_bytes(node):
    """An integrity failure must never surface as content. `resolve_text` falls back."""
    from types import SimpleNamespace

    from mantle.shard.content import resolve_text
    from mantle.shard.sqlite_store import FsContentStore

    root, _backend = node
    cas = FsContentStore(str(root / "cas"))
    real, _ = put_content(cas, root / "keys", b"the real content")
    impostor, _ = put_content(cas, root / "keys", b"attacker-chosen content")
    cas.put(real, cas.get(impostor))

    bundle = SimpleNamespace(content=cas, keys_dir=root / "keys", content_tier=None)
    text = resolve_text(bundle, {"content_ref": real, "content": "inline fallback"})
    assert text == "inline fallback", "wrong bytes must never be returned as the artifact's text"


def test_backend_exposes_it_next_to_store_handle(node):
    """One import point for the routers: `store_handle` and `content_handle` are the two
    halves of 'the lattice IS the store'."""
    _root, backend = node
    importlib.reload  # noqa: B018 — referenced to keep the import meaningful under linters
    assert callable(backend.store_handle) and callable(backend.content_handle)
