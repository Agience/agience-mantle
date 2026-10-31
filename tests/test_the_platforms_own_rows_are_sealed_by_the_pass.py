"""The platform's own rows are sealed by the back-fill like everyone else's: grants under the node's
key, a member's profile as its member, and the platform's collections as the platform's system
principal, which is granted them.

2026-10-30, on the release candidate: of 414 rows the pass left in plain text, 371 were grants
(sealed with a content key no actor holds, rather than the node's), 41 were member profiles in the
People directory (keyed to the directory, which members hold no grant on), and 2 were the People
and Authorities collections, created by the platform's id, which nothing acts as.

Keys are decided as in production: the light-cone verifier over the store's own grants.
"""
from __future__ import annotations

import json
import uuid

import pytest

from mantle.db import doc_boundary, node_seal
from mantle.db import lattice_api as la
from mantle.services.seed_provisioning import user_provisioning as up
from mantle.system.seal_titles import seal_stored_titles, unseal_stored_titles

from .test_there_is_no_promotion_state import USER, real_node

SYSTEM = str(uuid.uuid5(uuid.NAMESPACE_URL, "test://platform-system-principal"))
WORD = "marrowfinch"


@pytest.fixture
def db(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    from mantle.search.mantle import wiring
    from mantle.search.mantle.oracle import FernetMasterKeyStore, LightConeGrantVerifier, OracleService
    from mantle.services import content_crypto, peer_signing

    encrypt, decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content
    database = real_node(tmp_path, monkeypatch)
    monkeypatch.setattr(content_crypto, "encrypt_content", encrypt)
    monkeypatch.setattr(content_crypto, "decrypt_content", decrypt)
    oracle = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())),
                           grant_verifier=LightConeGrantVerifier(database, ttl_s=0))
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)
    monkeypatch.setattr(peer_signing, "get_system_principal_id", lambda: SYSTEM)
    return database


def _member(db):
    """A member's profile, as first login writes it, in the People directory."""
    from mantle.services.acting_principal import acting_as
    ctx = up.UserContext(id=USER, email=f"{WORD}@example.invalid", name=f"Ada {WORD}")
    with acting_as(USER, principal_type="user"):
        up._ensure_person_artifact(db, ctx)
    return up.person_artifact_id(USER)


def _raw(db, artifact_id):
    return db.artifacts.get_artifact(artifact_id)


def test_a_member_profile_and_the_directory_survive_the_rollback_round_trip(db):
    pid = _member(db)
    people = _raw(db, pid)["collection_id"]
    assert WORD not in json.dumps(_raw(db, pid)), "the control: written sealed"

    back = unseal_stored_titles(db)
    assert back["denied"] == 0, back
    assert WORD in json.dumps(_raw(db, pid)), "as an older Mantle reads it"

    counts = seal_stored_titles(db)
    assert counts["left_plain"] == 0, counts
    assert WORD not in json.dumps(_raw(db, pid))
    ctx = json.loads(_raw(db, people)["context"] or "{}")
    assert doc_boundary.SEALED in ctx and "name" not in _raw(db, people), "the directory's own name"


def test_a_grant_stored_plain_is_sealed_under_the_nodes_key_and_back(db):
    from mantle.entities.grant import Grant
    g = Grant(id=str(uuid.uuid4()), resource_id=str(uuid.uuid4()), grantee_type="user", grantee_id=USER,
              granted_by=USER, can_read=True, name=f"Shared {WORD}", notes=f"for {WORD}",
              target_entity=f"{WORD}@example.invalid")
    la.create_grant(db, g)
    raw = dict(_raw(db, g.id))
    assert WORD not in json.dumps(raw) and raw.get(node_seal.SEALED), "the control: written sealed"
    node_seal.unseal_fields(raw)
    db.artifacts.put_artifact(raw)                      # as a row written before grants were sealed
    assert WORD in json.dumps(_raw(db, g.id))

    assert seal_stored_titles(db)["left_plain"] == 0
    assert WORD not in json.dumps(_raw(db, g.id))
    read = la.get_grant_by_id(db, g.id)
    assert (read.name, read.notes, read.target_entity) == (f"Shared {WORD}", f"for {WORD}",
                                                           f"{WORD}@example.invalid")

    assert unseal_stored_titles(db)["denied"] == 0
    assert _raw(db, g.id).get("target_entity") == f"{WORD}@example.invalid", "for an older Mantle"


def test_the_platform_principal_holds_read_and_update_on_the_platforms_collections(db):
    pid = _member(db)
    people = _raw(db, pid)["collection_id"]
    grants = la.get_active_grants_for_principal_resource(db, grantee_id=SYSTEM, resource_id=people)
    assert [(g.grantee_type, g.can_read, g.can_update, g.can_admin) for g in grants] == \
        [("user", True, True, False)]
    assert not up.ensure_platform_principal_grant(db, people), "idempotent"


def test_a_directory_made_before_the_grant_is_granted_by_the_pass(db):
    """Production's People directory predates the grant: the pass makes it before it seals."""
    pid = _member(db)
    people = _raw(db, pid)["collection_id"]
    for g in la.get_active_grants_for_principal_resource(db, grantee_id=SYSTEM, resource_id=people):
        db.artifacts.delete_artifact(g.id)
    assert not la.get_active_grants_for_principal_resource(db, grantee_id=SYSTEM, resource_id=people)

    counts = seal_stored_titles(db)
    assert counts["platform_grants"] >= 1, counts
    assert unseal_stored_titles(db)["denied"] == 0, "the directory opens for the platform again"


def test_a_link_named_before_sealing_is_tokenized_by_the_pass_and_back(db):
    """Links named before this release keep their name in `edge.label` and in its counter until the
    pass rewrites them; a rollback puts the name back, because an older Mantle reads it there."""
    from mantle.system.seal_titles import seal_stored_labels, unseal_stored_labels
    name = f"step1.{WORD}"
    db.graph.add_edge("coll-a", "art-b", name, {"order_key": "a0", "is_origin": False})

    def raw():
        return [dict(r) for r in db.graph.db.read().execute("SELECT * FROM edge WHERE src = 'coll-a'")]

    def counters():
        return [r[0] for r in db.graph.db.read().execute("SELECT name FROM counter")]

    assert any(WORD in r["label"] for r in raw()), "the control: the name is plain"
    assert seal_stored_labels(db)["labels_sealed"] == 1
    (row,) = raw()
    assert WORD not in json.dumps(row, default=repr) and WORD not in json.dumps(counters())
    assert la._shown_label({"label": row["label"], "props": json.loads(row["props"])}) == name
    assert row["order_key"] == "a0"
    assert seal_stored_labels(db)["plain_labels"] == 0, "idempotent"

    assert unseal_stored_labels(db)["labels_unsealed"] == 1
    assert [r["label"] for r in raw()] == [name], "as an older Mantle reads it"


def test_a_grant_sealed_under_another_nodes_key_still_authorizes_and_survives_a_save(db):
    """A store copied to a node with its own key: the grant's fields do not open, and must neither
    fail every lookup for its grantee nor be erased by a save."""
    from cryptography.fernet import Fernet
    from mantle.entities.grant import Grant
    g = Grant(id=str(uuid.uuid4()), resource_id="res-x", grantee_type="user", grantee_id=USER,
              granted_by=USER, can_read=True, name=f"Shared {WORD}")
    la.create_grant(db, g)
    raw = dict(_raw(db, g.id))
    foreign = Fernet(Fernet.generate_key()).encrypt(b'{"name": "elsewhere"}').decode("ascii")
    raw[node_seal.SEALED] = foreign
    db.artifacts.put_artifact(raw)

    (got,) = la.get_active_grants_for_principal_resource(db, grantee_id=USER, resource_id="res-x")
    assert got.can_read and not got.name
    got.can_update = True
    la.update_grant(db, got)
    assert _raw(db, g.id)[node_seal.SEALED] == foreign, "the save kept what it could not open"
