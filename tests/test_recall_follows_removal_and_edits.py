"""Recall answers with what a workspace holds NOW: not a note removed from it, not deleted text.

Both are properties of the index rather than of the rows, so a test of the rows cannot see
either. The probes here run the production recall on a real store: a `LatticeDatabase` on disk,
the real light cone over it, the real lexical and vector indexers writing to local files through
the production wiring, and the accessor `POST /artifacts/recall` builds. Only key custody is
stood in for — every requester may hold every key — because custody is not the question: which
index entries exist is.

* **Remove.** Recall reads an artifact's index entry from its home collection. A note removed
  from its workspace that kept its home and its entry would still answer a recall scoped to that
  workspace.
* **Edit.** The vector arm upserts chunks by `(root, chunk_id)`. An edit that removes a paragraph
  must take that paragraph's chunk with it, or the deleted text stays matchable.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
from cryptography.fernet import Fernet

from mantle.db import lattice_api as store
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE
from mantle.search.ingest import pipeline_unified
from mantle.search.mantle import wiring
from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService
from mantle.search.types import SearchQuery
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext, get_auth, get_store_db

#: The dependency the routes declare, captured before the `db` fixture patches the module
#: attribute the index path reads — a route override must be keyed on this original.
_ROUTE_STORE_DB = get_store_db

USER = "u-live"
DIM = 16


class _EveryRequesterHoldsEveryKey:
    """Key custody stood in for; see the module note."""

    def authorized(self, *, requester_id, requester_type, principal_id, collection_id,
                   action) -> bool:
        return True


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A standalone node: one store file, local index directories, no object storage."""
    database = store.LatticeDatabase(str(tmp_path / "live.db"), origin="live-test")

    monkeypatch.delenv("MANTLE_SSE_STORE", raising=False)
    monkeypatch.delenv("MANTLE_CELL_STORE", raising=False)
    monkeypatch.setenv("MANTLE_SSE_DIR", str(tmp_path / "sse"))
    monkeypatch.setenv("MANTLE_CELL_DIR", str(tmp_path / "cells"))
    monkeypatch.setattr(wiring, "edge_object_storage_is_configured", lambda: False)
    monkeypatch.setattr(wiring, "edge_s3_if_reachable", lambda what: (None, None))
    oracle = OracleService(FernetMasterKeyStore(Fernet(Fernet.generate_key())),
                           grant_verifier=_EveryRequesterHoldsEveryKey())
    monkeypatch.setattr(wiring, "_build_oracle", lambda: oracle)

    # The index path reads the store through the dependency, as it does in production.
    import mantle.services.dependencies as deps

    def _store_db():
        yield database
    monkeypatch.setattr(deps, "get_store_db", _store_db)
    monkeypatch.setattr(pipeline_unified, "index_queue", None)

    # Bodies stay on the row, sealed by the real envelope under an injected master key; this node
    # has no CAS for the conftest stub's addresses to resolve against.
    monkeypatch.setattr(
        ws, "_store_content_in_s3",
        lambda artifact_id, content, context_str, **_: (f"artifacts/{artifact_id}.content",
                                                        content, None),
    )
    from mantle.services import content_crypto
    real_encrypt, real_decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content

    def _provider(principal_id: str) -> bytes:
        return hashlib.sha256(("master:" + principal_id).encode("utf-8")).digest()

    monkeypatch.setattr(content_crypto, "encrypt_content",
                        lambda pid, pt, **kw: real_encrypt(pid, pt, master_key_provider=_provider, **kw))
    monkeypatch.setattr(content_crypto, "decrypt_content",
                        lambda pid, blob, **kw: real_decrypt(pid, blob, master_key_provider=_provider, **kw))
    return database


def _auth() -> AuthContext:
    return AuthContext(principal_id=USER, principal_type="user", user_id=USER)


def _container(db, name: str) -> str:
    with acting_as(USER, principal_type="user"):
        return ws.create_container(db, USER, content_type=WORKSPACE_CONTENT_TYPE, name=name).id


def _note(db, workspace: str, text: str) -> str:
    with acting_as(USER, principal_type="user"):
        return ws.create_workspace_artifact(
            db, USER, workspace, context='{"content_type":"text/plain"}', content=text,
            content_type="text/plain",
        ).root_id


def _recall(db, text: str, scope=None) -> set:
    """The default recall (`live` segment), as the router runs it."""
    accessor = wiring.build_sse_search_accessor(db, segment="live")
    assert accessor is not None, "no recall stack on a standalone node"
    with acting_as(USER, principal_type="user"):
        result = accessor.search(SearchQuery(query_text=text, user_id=USER, scope=scope, size=20))
    return {hit.root_id for hit in result.hits}


# ── remove ────────────────────────────────────────────────────────────────────────────────────

def test_the_probe_can_see_a_note_at_all(db):
    """The positive control: without it, every "not returned" below could be an empty stack."""
    inbox = _container(db, "Inbox")
    note = _note(db, inbox, "periwinkle harbour schedule")
    assert note in _recall(db, "periwinkle", scope=[inbox])
    assert note in _recall(db, "periwinkle")


def test_a_removed_note_is_not_recalled_from_its_workspace(db):
    inbox = _container(db, "Inbox")
    note = _note(db, inbox, "periwinkle harbour schedule")
    assert note in _recall(db, "periwinkle", scope=[inbox])

    with acting_as(USER, principal_type="user"):
        ws.remove_artifact_from_container(db, USER, inbox, note, auth=_auth())

    assert note not in _recall(db, "periwinkle", scope=[inbox])
    assert note not in _recall(db, "periwinkle")
    # Nothing else holds it and its owner removed it, so the lineage is gone rather than orphaned.
    assert store.get_artifact(db, note) is None


def test_a_removed_note_another_collection_holds_stays_there(db):
    inbox = _container(db, "Inbox")
    shared = _container(db, "Shared")
    note = _note(db, inbox, "periwinkle harbour schedule")
    with acting_as(USER, principal_type="user"):
        store.add_artifact_to_collection(db, shared, note, origin=False)
        ws.remove_artifact_from_container(db, USER, inbox, note, auth=_auth())

    assert note not in _recall(db, "periwinkle", scope=[inbox]), (
        "the note still answers a recall scoped to the workspace it was removed from")
    listed = {row.get("root_id") or row.get("id")
              for row in store.list_collection_artifacts(db, shared)}
    assert note in listed, "removing it from one collection took it out of another"
    assert store.get_artifact(db, note) is not None


def test_a_removal_by_someone_who_may_not_delete_keeps_the_row_but_not_the_entry(db):
    """`evict` on the container unlinks; it does not authorize destroying the artifact."""
    inbox = _container(db, "Inbox")
    note = _note(db, inbox, "periwinkle harbour schedule")
    with acting_as(USER, principal_type="user"):
        ws.remove_artifact_from_container(db, USER, inbox, note, auth=None)

    assert store.get_artifact(db, note) is not None
    assert note not in _recall(db, "periwinkle", scope=[inbox])


@pytest.mark.anyio
async def test_its_owner_can_delete_a_workspace_note(db, client):
    """`DELETE /artifacts/{id}` on a live note, through the route, by its owner."""
    from mantle.main import app

    inbox = _container(db, "Inbox")
    note = _note(db, inbox, "periwinkle harbour schedule")
    assert store.get_artifact(db, note).state == "live"

    def _store_db():
        yield db
    app.dependency_overrides[_ROUTE_STORE_DB] = _store_db
    app.dependency_overrides[get_auth] = _auth
    with acting_as(USER, principal_type="user"):
        response = await client.delete(f"/artifacts/{note}")

    assert response.status_code == 200, response.text
    assert response.json()["deleted"] is True
    assert store.get_artifact(db, note) is None
    assert note not in _recall(db, "periwinkle", scope=[inbox])


# ── edit ──────────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def vector_arm(monkeypatch):
    """A provisioned AnchorSet of one anchor (so every chunk lands in one cell), paragraph
    chunking, and an embedder that gives each distinct text its own direction."""
    from mantle.search.anchors import store as anchors
    from mantle.search.anchors.anchorset import AnchorSet
    from mantle.search.anchors.repo import InMemoryAnchorRepo

    anchors.set_anchor_repo(InMemoryAnchorRepo())
    aset = AnchorSet("hf:test@1.0", DIM)
    aset.add_text("anchor-0", np.ones(DIM, dtype=np.float32))
    anchors.save_live_anchorset(aset)

    def _embed(texts):
        out = []
        for text in texts:
            v = np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[:DIM],
                              dtype=np.uint8).astype(np.float64) + 1.0
            out.append((v / np.linalg.norm(v)).tolist())
        return out

    monkeypatch.setattr(pipeline_unified, "_embeddings", _embed)
    monkeypatch.setattr(
        pipeline_unified, "_content_chunks",
        lambda content: [{"chunk_id": i, "text": p}
                         for i, p in enumerate(x for x in content.split("\n\n") if x)],
    )
    yield
    anchors.set_anchor_repo(None)


def _chunk_texts(db, workspace: str, root: str) -> list:
    with acting_as(USER, principal_type="user"):
        artifact = store.get_artifact(db, root)
        return [c.get("text") for c in pipeline_unified.get_artifact_embeddings(artifact, workspace)]


def test_an_edit_that_removes_a_paragraph_removes_its_chunk(db, vector_arm):
    inbox = _container(db, "Inbox")
    first, second = "the ferry leaves at noon", "the lighthouse code is 4471"
    note = _note(db, inbox, f"{first}\n\n{second}")
    assert sorted(_chunk_texts(db, inbox, note)) == sorted([first, second]), (
        "the vector arm did not index the note — the probe below would pass on an empty cell")

    with acting_as(USER, principal_type="user"):
        ws.update_artifact(db, USER, inbox, note, content=first)

    assert _chunk_texts(db, inbox, note) == [first], (
        "the deleted paragraph is still in the vector index and still matchable")
    assert note not in _recall(db, "lighthouse", scope=[inbox]), (
        "the deleted paragraph's words still find the note")
