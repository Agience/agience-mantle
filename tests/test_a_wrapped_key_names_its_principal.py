"""A wrapped master key cannot be moved from one principal to another.

The KEK wrap protects the DEK's confidentiality and integrity and nothing else. `Fernet.encrypt`
takes no associated data; `AwsKmsKeyProvider.wrap` is called with no `EncryptionContext`;
`VaultTransitKeyProvider.wrap` passes no `context`. So the only thing tying a wrapped key to its
owner was the document id it happened to be filed under — `"master-key:" + principal_id` — which is
a plaintext, mutable database column, not something the ciphertext agrees with.

That made keys relocatable. With write access to the lattice (a compromised node process, a restored
backup, any bug that reaches `put_artifact` with a chosen id):

    copy   vertex.doc.token  FROM  "master-key:alice"  INTO  "master-key:mallory"
    then   Mallory asks for her own master key

`_authorize`'s base case lets a principal fetch its own key unconditionally, so the grant ledger is
never consulted. The unwrap succeeds — nothing in the ciphertext disagrees — and Mallory holds
Alice's master key, from which the content key, the SSE key and every cell key derive offline.

The fix frames the principal inside the authenticated plaintext, so every provider's existing
integrity check now covers it too, with no change to the `KeyProvider` interface.
"""
from __future__ import annotations

import pytest

from mantle.search.mantle import oracle as orc
from mantle.search.mantle.oracle import LatticeMasterKeyStore, MasterKeyUnavailable

ALICE_KEY = bytes(range(32))


class _FakeKek:
    """Wrap/unwrap with integrity but NO associated data — i.e. every provider in the tree."""

    def wrap(self, plaintext: bytes) -> str:
        return plaintext.hex()

    def unwrap(self, token: str) -> bytes:
        return bytes.fromhex(token)


class _FakeArtifacts:
    def __init__(self):
        self.docs = {}

    def put_artifact(self, doc):
        self.docs[doc["id"]] = dict(doc)

    def get_artifact(self, artifact_id):
        return self.docs.get(artifact_id)


class _FakeDb:
    def __init__(self):
        self.artifacts = _FakeArtifacts()


@pytest.fixture()
def store():
    db = _FakeDb()
    return LatticeMasterKeyStore(_FakeKek(), lambda: db), db


def test_a_key_round_trips_for_the_principal_it_was_minted_for(store):
    keys, _db = store
    keys.put("alice", ALICE_KEY)
    assert keys.get("alice") == ALICE_KEY


def test_a_key_moved_to_another_principal_is_refused(store):
    """The attack, verbatim: relocate the token, ask as yourself, get an error rather than a key."""
    keys, db = store
    keys.put("alice", ALICE_KEY)
    stolen = db.artifacts.docs["master-key:alice"]["token"]

    db.artifacts.put_artifact({
        "id": "master-key:mallory",
        "content_type": LatticeMasterKeyStore.CONTENT_TYPE,
        "token": stolen,
    })

    with pytest.raises(MasterKeyUnavailable) as exc:
        keys.get("mallory")
    assert "alice" in str(exc.value), "the error should name the principal the key really belongs to"


def test_the_refusal_does_not_overwrite_the_relocated_key(store):
    """Failing closed must not become failing destructive: a refusal that let the caller mint a
    replacement in place would turn this into a way to DESTROY another principal's key."""
    keys, db = store
    keys.put("alice", ALICE_KEY)
    db.artifacts.put_artifact({
        "id": "master-key:mallory",
        "content_type": LatticeMasterKeyStore.CONTENT_TYPE,
        "token": db.artifacts.docs["master-key:alice"]["token"],
    })

    with pytest.raises(MasterKeyUnavailable):
        keys.get("mallory")

    assert keys.get("alice") == ALICE_KEY, "Alice's key must be untouched by the refused read"


def test_two_principals_with_the_same_key_bytes_get_different_tokens(store):
    """The binding is in the wrapped payload, not alongside it — so the tokens differ even when
    the DEKs are identical. If they matched, the frame would not be inside the authenticated
    bytes and moving a token would still work."""
    keys, db = store
    keys.put("alice", ALICE_KEY)
    keys.put("bob", ALICE_KEY)

    assert (db.artifacts.docs["master-key:alice"]["token"]
            != db.artifacts.docs["master-key:bob"]["token"])
    assert keys.get("alice") == ALICE_KEY and keys.get("bob") == ALICE_KEY


def test_an_unframed_key_still_opens_and_is_counted(store):
    """Keys written before the binding existed carry no frame. Breaking every existing install to
    close a database-write-access attack would be the wrong trade — they open, and the counter
    says how many remain so the fallback has a measurable end."""
    keys, db = store
    db.artifacts.put_artifact({
        "id": "master-key:legacy",
        "content_type": LatticeMasterKeyStore.CONTENT_TYPE,
        "token": _FakeKek().wrap(ALICE_KEY),          # raw DEK, no frame
    })

    before = orc.legacy_unbound_master_keys
    assert keys.get("legacy") == ALICE_KEY
    assert orc.legacy_unbound_master_keys == before + 1


def test_a_rewrite_migrates_a_legacy_key_to_the_bound_form(store):
    """`put` is the migration: a key re-wrapped after the change is bound, and stops counting."""
    keys, db = store
    db.artifacts.put_artifact({
        "id": "master-key:legacy",
        "content_type": LatticeMasterKeyStore.CONTENT_TYPE,
        "token": _FakeKek().wrap(ALICE_KEY),
    })

    keys.put("legacy", ALICE_KEY)                     # re-wrapped through the framing path

    before = orc.legacy_unbound_master_keys
    assert keys.get("legacy") == ALICE_KEY
    assert orc.legacy_unbound_master_keys == before, "a re-wrapped key must no longer read as legacy"
