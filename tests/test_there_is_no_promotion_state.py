"""An artifact is `live` or `archived`. Nothing is a draft, and nothing needs promoting.

Every write path writes `live`, and `recall` reads the live segment by default, so a saved artifact
is findable with no further step. A store that has not been migrated holds `committed` and
`draft` rows; every reader treats both as live (`db/constants.state_of`, `live_states`) until
`system/manage_state_collapse` rewrites them:

* a lineage resolves to its newest non-archived row, whichever legacy value it records;
* a lattice query for `state="live"` matches all three recorded values, and its count sums them;
* `recall` accepts `committed` and `draft` as deprecated aliases that read the live segment, and an
  alias reaches nothing the default does not;
* `PATCH` accepts `committed` as an alias of `live` and refuses `draft`.

The write-path and recall probes run on a real store with the real light cone and the real local
index (the `test_recall_follows_removal_and_edits` stack); only key custody is stood in for.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from starlette.requests import Request

from mantle.db import lattice_api as store
from mantle.entities.artifact import Artifact, WORKSPACE_CONTENT_TYPE
from mantle.routers import artifacts_router as artifacts
from mantle.search.ingest import pipeline_unified
from mantle.search.mantle import wiring
from mantle.search.mantle.oracle import FernetMasterKeyStore, OracleService
from mantle.search.types import SearchQuery
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as
from mantle.services.dependencies import AuthContext

USER = "u-live"
OTHER = "u-reader"

#: The segment `recall` reads when the caller names none, read from the request model.
DEFAULT_RECALL_STATE = artifacts.ArtifactRecallRequest.model_fields["state"].default


# ── the vocabulary ────────────────────────────────────────────────────────────────────────────

def test_a_constructed_artifact_is_live():
    assert Artifact().state == "live"
    assert Artifact.VALID_STATES == {"live", "archived"}
    assert "draft" not in Artifact.VALID_STATES
    assert "committed" not in Artifact.VALID_STATES


def test_a_stored_legacy_value_reads_as_live():
    for recorded in ("committed", "draft", "live", None, ""):
        assert Artifact.from_dict({"id": "a", "state": recorded}).state == "live", recorded
    assert Artifact.from_dict({"id": "a", "state": "archived"}).state == "archived"


async def test_patch_draft_answers_422(client):
    r = await client.patch("/artifacts/a-1", json={"state": "draft"})
    assert r.status_code == 422, r.text


def test_the_revert_route_is_absent_from_the_api():
    from mantle.main import app
    paths = app.openapi()["paths"]
    assert not [p for p in paths if p.endswith("/revert")], sorted(paths)


# ── lineage resolution on a legacy store ──────────────────────────────────────────────────────

@pytest.fixture
def bare_db(tmp_path):
    return store.LatticeDatabase(str(tmp_path / "bare.db"), origin="bare-test")


def _raw(db, doc_id, root, state, collection="col-1"):
    db.artifacts.put_artifact({"id": doc_id, "root_id": root, "collection_id": collection,
                               "state": state, "content": doc_id})


@pytest.mark.parametrize("older,newer", [("draft", "committed"), ("committed", "draft")])
def test_the_newest_row_wins_in_a_mixed_legacy_lineage(bare_db, older, newer):
    """An older draft beside a newer committed row in one collection, and the reverse: the newer
    row is current, whichever legacy value it records."""
    _raw(bare_db, "r-1", "r-1", older)
    _raw(bare_db, "r-1-v2", "r-1", newer)

    current = store.get_current_in_collection(bare_db, "col-1", "r-1")
    assert current is not None and current.id == "r-1-v2"
    assert current.state == "live"
    assert store.get_latest_live_artifact(bare_db, "r-1", "col-1").id == "r-1-v2"
    many = store.get_current_in_collection_many(bare_db, "col-1", ["r-1"])
    assert many["r-1"].id == "r-1-v2"


def test_resolve_roots_finds_a_draft_only_lineage(bare_db):
    """`lightcone._resolve_roots` fills in a lineage the id lookup missed. A lineage that only
    ever held a draft is live, so it is found."""
    from mantle.search.mantle.lightcone import _resolve_roots

    _raw(bare_db, "d-2", "root-d", "draft")
    out: dict = {}
    _resolve_roots(bare_db, ["root-d"], out)
    assert "root-d" in out and out["root-d"]["id"] == "d-2"


def test_resolve_roots_skips_an_archived_lineage(bare_db):
    """The guard beside it: an archived lineage is not live and is not filled in."""
    from mantle.search.mantle.lightcone import _resolve_roots

    _raw(bare_db, "a-2", "root-a", "archived")
    out: dict = {}
    _resolve_roots(bare_db, ["root-a"], out)
    assert out == {}


def test_a_live_query_returns_legacy_rows_and_counts_them(bare_db):
    for i, state in enumerate(("live", "committed", "draft", "archived")):
        _raw(bare_db, "q-%d" % i, "q-%d" % i, state)
    a = bare_db.artifacts

    live = sorted(d["id"] for d in a.list_artifacts(state="live"))
    assert live == ["q-0", "q-1", "q-2"]
    assert a.count(state="live") == 3
    # `committed` is read as `live` for callers that still send it.
    assert sorted(d["id"] for d in a.list_artifacts(state="committed")) == live
    assert a.count(state="committed") == 3
    # `draft` and the recorded counters are exact, for the migration that needs them.
    assert [d["id"] for d in a.list_artifacts(state="draft")] == ["q-2"]
    assert a.count(state="draft") == 1
    assert [a.count_recorded(s) for s in ("live", "committed", "draft", "archived")] == [1, 1, 1, 1]
    assert [d["id"] for d in a.list_artifacts(state="archived")] == ["q-3"]


# ── a real node: every write path, and recall ─────────────────────────────────────────────────

class _EveryRequesterHoldsEveryKey:
    def authorized(self, *, requester_id, requester_type, principal_id, collection_id,
                   action) -> bool:
        return True


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


def real_node(tmp_path, monkeypatch):
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

    import mantle.services.dependencies as deps

    def _store_db():
        yield database
    monkeypatch.setattr(deps, "get_store_db", _store_db)
    monkeypatch.setattr(pipeline_unified, "index_queue", None)
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


def _auth(user: str = USER) -> AuthContext:
    return AuthContext(principal_id=user, principal_type="user", user_id=user)


def _container(db, name: str) -> str:
    with acting_as(USER, principal_type="user"):
        return ws.create_container(db, USER, content_type=WORKSPACE_CONTENT_TYPE, name=name).id


def _recall(db, text: str, *, user: str = USER, state: str = DEFAULT_RECALL_STATE) -> set:
    accessor = wiring.build_sse_search_accessor(db, segment=state)
    assert accessor is not None, "no recall stack on a standalone node"
    with acting_as(user, principal_type="user"):
        result = accessor.search(SearchQuery(query_text=text, user_id=user, size=20))
    return {hit.root_id for hit in result.hits}


def _ctx() -> str:
    return json.dumps({"content_type": "text/plain"})


def _write_workspace_create(db, inbox, text, monkeypatch):
    return ws.create_workspace_artifact(db, USER, inbox, context=_ctx(), content=text,
                                        content_type="text/plain").root_id


def _write_bulk(db, inbox, text, monkeypatch):
    return ws.create_workspace_artifacts_bulk(
        db, USER, inbox, [{"context": {"content_type": "text/plain"}, "content": text}])[0].root_id


def _write_upload_complete(db, inbox, text, monkeypatch):
    import mantle.services.content_service as cs
    ref = "cas/" + "c" * 64
    monkeypatch.setattr(cs, "head_object", lambda key: None)
    monkeypatch.setattr(cs, "persist_object_to_durable", lambda key: False)
    monkeypatch.setattr(cs, "local_content_has", lambda r: r == ref)
    # The bytes are read back from storage for indexing and none were written here; what they
    # extract to is stood in for, which segment they land in is what is measured.
    monkeypatch.setattr(pipeline_unified, "_extract_artifact_fields",
                        lambda artifact: {"title": "notes.txt", "content": text})
    _descriptor, created = ws.initiate_upload_and_create_artifact(
        db, USER, inbox, filename="notes.txt", content_type="text/plain", size=len(text))
    ws.update_upload_status(db=db, user_id=USER, workspace_id=inbox, upload_id=created.id,
                            context_patch={"content_cas_ref": ref})
    done = ws.update_upload_status(db=db, user_id=USER, workspace_id=inbox,
                                   upload_id=created.id, status_value="complete")
    return done.root_id


def _write_inbound_webhook(db, inbox, text, monkeypatch):
    workspace = ws.get_workspace(db, USER, inbox)
    # The card-key check is a separate question; the writer behind it is what is measured.
    monkeypatch.setattr(ws, "resolve_card_key",
                        lambda *a, **k: ({"id": "card-1"}, workspace, USER))
    msg_id, _ws = ws.receive_card_inbound_message(db, "card-1", "tok", text, channel="sms")
    return store.get_artifact(db, msg_id).root_id


def _write_identity_upsert(db, inbox, text, monkeypatch):
    return ws.upsert_identity_member(db, USER, inbox, "ident-root-1", context=_ctx(),
                                     content=text, content_type="text/plain").root_id


def _write_seed_card(db, inbox, text, monkeypatch):
    """A seed card filed in the workspace. Seeding writes rows and indexes nothing; the reindex
    pass (`init_search.reindex_all_artifacts` → `index_artifact`) is what indexes a seed, so it
    runs here too."""
    from mantle.services.seed_provisioning import loader
    aid = str(uuid.uuid4())
    body = {"namespace": "test", "slug": "seed-note", "content_type": "text/plain",
            "content": text, "edges": [{"to": inbox}]}
    loader._apply_artifact_card(db, loader._RawArtifact(Path("seed.json"), body, "artifact"),
                                aid, loader.SeedReport())
    store.add_artifact_to_collection(db, inbox, aid)
    pipeline_unified.index_artifact(store.get_artifact(db, aid), inbox)
    return aid


async def _write_mcp_create(db, inbox, text, monkeypatch):
    from mantle.routers import mcp_router
    request = Request({"type": "http", "method": "POST", "headers": [], "path": "/mcp"})
    out = await mcp_router._call_tool(
        "create_artifact",
        {"container_id": inbox, "content": text, "content_type": "text/plain",
         "context": {"content_type": "text/plain"}},
        auth=_auth(), store_db=db, request=request)
    return out.get("root_id") or out["id"]


WRITE_PATHS = {
    "workspace_create": _write_workspace_create,
    "bulk": _write_bulk,
    "upload_complete": _write_upload_complete,
    "inbound_webhook": _write_inbound_webhook,
    "identity_upsert": _write_identity_upsert,
    "seed_card": _write_seed_card,
    "mcp_create_artifact": _write_mcp_create,
}


@pytest.mark.parametrize("path", sorted(WRITE_PATHS))
async def test_every_write_path_writes_live_and_is_found_by_default_recall(db, path, monkeypatch):
    inbox = _container(db, "Inbox")
    word = "zephyrine" + path.replace("_", "")
    with acting_as(USER, principal_type="user"):
        root = WRITE_PATHS[path](db, inbox, "%s harbour schedule" % word, monkeypatch)
        if hasattr(root, "__await__"):
            root = await root

    rows = db.artifacts.versions_of(root)
    assert rows, "the write left no row under %s" % root
    assert {r.get("state") for r in rows} == {"live"}, [r.get("state") for r in rows]
    assert store.get_current_in_collection(db, inbox, root).state == "live"
    assert root in _recall(db, word), "%s wrote an artifact the default recall does not find" % path


async def test_patch_committed_restores_an_archived_artifact(db, monkeypatch):
    """`committed` is what the deployed client sends as Restore; it restores to `live`."""
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "quillwort restore probe", monkeypatch)
        archived = await artifacts.update_artifact(
            artifact_id=root, body=artifacts.UpdateArtifactRequest(state="archived"),
            auth=_auth(), store_db=db)
        assert archived["state"] == "archived"
        restored = await artifacts.update_artifact(
            artifact_id=root, body=artifacts.UpdateArtifactRequest(state="committed"),
            auth=_auth(), store_db=db)

    assert restored["state"] == "live"
    assert db.artifacts.get_artifact(root)["state"] == "live"
    assert root in _recall(db, "quillwort")


async def test_archiving_twice_answers_the_archived_artifact_and_editing_it_is_refused(db, monkeypatch):
    """2026-10-13, a tester: archiving answered 404 when two copies of one card on screen archived
    the same artifact; the second request found it already archived."""
    from fastapi import HTTPException
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "sallowmere archive twice", monkeypatch)
        for _ in range(2):
            again = await artifacts.update_artifact(
                artifact_id=root, body=artifacts.UpdateArtifactRequest(state="archived"),
                auth=_auth(), store_db=db)
            assert again["state"] == "archived"
        with pytest.raises(HTTPException) as refused:
            await artifacts.update_artifact(
                artifact_id=root, body=artifacts.UpdateArtifactRequest(state="archived", content="x"),
                auth=_auth(), store_db=db)
    assert refused.value.status_code == 404, "an archived artifact is edited only after a restore"


# ── recall aliases ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("alias", ["draft", "committed"])
async def test_a_legacy_recall_state_builds_the_live_prefix(monkeypatch, alias):
    """The alias reaches the store builders as a segment whose prefixes are the live ones —
    the unsuffixed `mantle-cells` / `mantle-sse`."""
    asked: list = []

    def _no_store(segment="live"):
        asked.append(segment)
        return None
    monkeypatch.setattr(wiring, "_build_sse_store", _no_store)
    monkeypatch.setattr(wiring, "_build_cell_store", _no_store)

    with pytest.raises(Exception):
        await artifacts.recall_artifacts(
            body=artifacts.ArtifactRecallRequest(query_text="anything", state=alias),
            auth=_auth(), store_db=None)
    assert asked, "the recall never asked for a store"
    assert {wiring._segment_prefixes(s) for s in asked} == {("mantle-cells", "mantle-sse")}


async def test_a_second_reader_gets_identical_hits_for_an_alias_and_the_default(db):
    """Nothing widens: a principal with read on one collection gets the same hits whichever name
    it gives the live segment, and none from a collection it cannot read."""
    shared = _container(db, "Shared")
    private = _container(db, "Private")
    with acting_as(USER, principal_type="user"):
        visible = ws.create_workspace_artifact(db, USER, shared, context=_ctx(),
                                               content="marshwren shared note",
                                               content_type="text/plain").root_id
        hidden = ws.create_workspace_artifact(db, USER, private, context=_ctx(),
                                              content="marshwren private note",
                                              content_type="text/plain").root_id
        store.upsert_user_collection_grant(db, user_id=OTHER, collection_id=shared,
                                           granted_by=USER, can_read=True)

    owner = _recall(db, "marshwren")
    assert {visible, hidden} <= owner, "the positive control: the owner sees both"

    answers = {}
    for state in (DEFAULT_RECALL_STATE, "committed", "draft"):
        with acting_as(OTHER, principal_type="user"):
            result = await artifacts.recall_artifacts(
                body=artifacts.ArtifactRecallRequest(query_text="marshwren", state=state),
                auth=_auth(OTHER), store_db=db)
        hits = result["hits"] if isinstance(result, dict) else result.hits
        answers[state] = sorted((h["root_id"] if isinstance(h, dict) else h.root_id)
                                for h in hits)
    assert answers[DEFAULT_RECALL_STATE] == [visible], answers
    assert answers["committed"] == answers["draft"] == answers[DEFAULT_RECALL_STATE], answers


# ── the index segments and the mesh ───────────────────────────────────────────────────────────

def test_the_live_segment_keeps_the_unsuffixed_prefixes():
    """`live` is stored under the unsuffixed prefixes the legacy `committed` segment used, so
    no live index data moves. `committed` names it too; `draft` names the legacy draft prefixes, which only a
    removal addresses, and every READ of `draft` is a read of `live`."""
    assert wiring.VALID_SEGMENTS == ("live", "archived")
    assert wiring._segment_prefixes("live") == ("mantle-cells", "mantle-sse")
    assert wiring._segment_prefixes("committed") == ("mantle-cells", "mantle-sse")
    assert wiring._segment_prefixes("archived") == ("mantle-cells-archived", "mantle-sse-archived")
    assert wiring._segment_prefixes("draft") == wiring.LEGACY_DRAFT_PREFIXES
    assert {wiring.normalise_segment(s) for s in ("live", "committed", "draft")} == {"live"}
    for state in ("live", "committed", "draft", "", None):
        assert pipeline_unified._segment_for_state(state) == "live", state
    assert pipeline_unified._segment_for_state("archived") == "archived"


def test_a_replica_from_an_unmigrated_peer_is_stored_live():
    """A peer that has not migrated sends `committed` and `draft`; the replica is stored `live`,
    on a copy, keeping the incoming `(_origin, _seq)`."""
    from mantle.mesh.sync import _normalise_legacy_state

    for legacy in ("committed", "draft"):
        sent = {"id": "p", "state": legacy, "_origin": "peer", "_seq": 4}
        stored = _normalise_legacy_state(sent)
        assert stored["state"] == "live"
        assert (stored["_origin"], stored["_seq"]) == ("peer", 4)
        assert sent["state"] == legacy, "the doc as sent was changed in place"
    for other in ("live", "archived", "active"):
        sent = {"id": "p", "state": other}
        assert _normalise_legacy_state(sent) is sent


# ── restore applies to a retired lineage only ─────────────────────────────────────────────────

async def _patch(db, artifact_id, **body):
    return await artifacts.update_artifact(
        artifact_id=artifact_id, body=artifacts.UpdateArtifactRequest(**body),
        auth=_auth(), store_db=db)


def _live_rows(db, root):
    return [v["id"] for v in db.artifacts.versions_of(root) if v.get("state") != "archived"]


async def test_restoring_a_superseded_version_is_refused(db, monkeypatch):
    """A content snapshot (or an older row a migration archived) records `superseded_by`.
    Restoring it would put a second live row under the root and shadow the head."""
    from fastapi import HTTPException

    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "sandpiper head", monkeypatch)
    head = db.artifacts.get_artifact(root)
    db.artifacts.put_artifact(dict(
        {k: v for k, v in head.items() if k not in ("_origin", "_seq")},
        id=root + "@old", state="archived", superseded_by=root))

    for body in ({"state": "live"}, {"state": "committed"}):
        with acting_as(USER, principal_type="user"), pytest.raises(HTTPException) as exc:
            await _patch(db, root + "@old", **body)
        assert exc.value.status_code == 409, exc.value.detail
    assert _live_rows(db, root) == [root]


async def test_restoring_an_archived_row_beside_a_live_head_is_refused(db, monkeypatch):
    """The same refusal without `superseded_by`: the root still has a live version."""
    from fastapi import HTTPException

    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "sandpiper head", monkeypatch)
    head = db.artifacts.get_artifact(root)
    db.artifacts.put_artifact(dict(
        {k: v for k, v in head.items() if k not in ("_origin", "_seq")},
        id=root + "~stray", state="archived"))

    with acting_as(USER, principal_type="user"), pytest.raises(HTTPException) as exc:
        await _patch(db, root + "~stray", state="live")
    assert exc.value.status_code == 409
    assert _live_rows(db, root) == [root]


async def test_a_state_change_combined_with_an_edit_is_refused_and_writes_nothing(db, monkeypatch):
    """422 rather than applying half of it; a PATCH naming the current state still edits."""
    from fastapi import HTTPException

    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "curlew original", monkeypatch)
        before = db.artifacts.get_artifact(root)
        with pytest.raises(HTTPException) as exc:
            await _patch(db, root, state="archived", content="curlew edited")
        assert exc.value.status_code == 422
        assert db.artifacts.get_artifact(root)["_seq"] == before["_seq"], "a refused PATCH wrote"

        await _patch(db, root, state="archived")
        with pytest.raises(HTTPException) as exc:
            await _patch(db, root, state="live", name="renamed")
        assert exc.value.status_code == 422
        assert db.artifacts.get_artifact(root)["state"] == "archived"

        await _patch(db, root, state="live")
        edited = await _patch(db, root, state="live", content="curlew edited")
    assert edited["state"] == "live" and edited["content"] == "curlew edited"


def _retired_lineage(db):
    """A lineage a user retired twice: v1 archived with no `superseded_by` (archived before any
    newer version existed), then v2 written and archived."""
    inbox = _container(db, "Inbox")
    db.artifacts.put_artifact({"id": "v1", "root_id": "r", "collection_id": inbox,
                               "state": "archived", "content": "old", "created_by": USER})
    db.artifacts.put_artifact({"id": "v2", "root_id": "r", "collection_id": inbox,
                               "state": "live", "content": "new", "created_by": USER})
    with acting_as(USER, principal_type="user"):
        ws.update_artifact(db, USER, inbox, "v2", state="archived", reindex=False)
    return inbox


def test_only_the_newest_row_of_a_retired_lineage_can_be_restored(db):
    from fastapi import HTTPException

    inbox = _retired_lineage(db)
    with acting_as(USER, principal_type="user"):
        with pytest.raises(HTTPException) as exc:
            ws.update_artifact(db, USER, inbox, "v1", state="live", reindex=False)
        assert exc.value.status_code == 409, "an older version was restored as the head"
        out = ws.update_artifact(db, USER, inbox, "v2", state="live", reindex=False)
    assert (out.id, out.state) == ("v2", "live")
    assert db.artifacts.get_artifact("v1")["state"] == "archived"


def test_an_archived_head_restores_and_its_snapshot_does_not(db, monkeypatch):
    """A content snapshot (`vertex._snapshot_prior`) beside an archived head: the head restores,
    the snapshot is refused."""
    from fastapi import HTTPException
    from mantle.db import doc_boundary

    # The rows name CAS addresses this node holds no bytes for; what is decided here is the
    # state, not the body.
    monkeypatch.setattr(doc_boundary, "decrypt_artifact_content", lambda d, **k: d)

    inbox = _container(db, "Inbox")
    base = {"id": "m", "root_id": "m", "collection_id": inbox, "state": "live",
            "content": "one", "created_by": USER, "content_ref": "cas/" + "1" * 64}
    db.artifacts.put_artifact(dict(base))
    db.artifacts.put_artifact(dict(base, content="two", content_ref="cas/" + "2" * 64))
    (snap,) = [v["id"] for v in db.artifacts.versions_of("m") if v["id"] != "m"]
    with acting_as(USER, principal_type="user"):
        ws.update_artifact(db, USER, inbox, "m", state="archived", reindex=False)
        with pytest.raises(HTTPException) as exc:
            ws.update_artifact(db, USER, inbox, snap, state="live", reindex=False)
        assert exc.value.status_code == 409
        out = ws.update_artifact(db, USER, inbox, "m", state="live", reindex=False)
    assert (out.id, out.state) == ("m", "live")
    assert db.artifacts.get_artifact(snap)["state"] == "archived"



async def test_an_archived_artifact_is_listed_by_every_container_holding_it_when_asked_for(db, monkeypatch):
    """2026-10-13, a tester: after a refresh an archived artifact was gone from its workspace and
    its collection, with no way in the app to find it again. Default listings read past it; the
    `archived` listing finds it wherever it is held, homed there or linked in."""
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    monkeypatch.setattr(artifacts, "_readable_members", lambda chunk, auth, db: list(chunk))
    inbox = _container(db, "Inbox")
    coll = _container(db, "Collection B")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "tamarisk archived listing", monkeypatch)
        store.add_artifact_to_collection(db, coll, root, origin=False, propagate=[])
        await artifacts.update_artifact(artifact_id=root, body=artifacts.UpdateArtifactRequest(state="archived"),
                                        auth=_auth(), store_db=db)

        async def ids(container, state):
            page = await artifacts.list_children(artifact_id=container, request=None, content_type=None,
                                                 workspace_id=None, state=state, limit=100, offset=0,
                                                 store_db=db, auth=_auth())
            return [(c.get("root_id") or c.get("id"), c.get("state")) for c in page["items"]]

        for container in (inbox, coll):
            assert await ids(container, "live") == []
            assert await ids(container, "archived") == [(root, "archived")], container


async def test_an_archived_artifact_is_not_linked_into_another_container(db, monkeypatch):
    """2026-10-13, a tester: dragging an archived artifact into a container linked it where no
    default listing shows it; it vanished from both with no error. The link is refused, naming why."""
    from fastapi import HTTPException
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    coll = _container(db, "Collection B")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "tamarisk archived link", monkeypatch)
        await artifacts.update_artifact(artifact_id=root, body=artifacts.UpdateArtifactRequest(state="archived"),
                                        auth=_auth(), store_db=db)
        with pytest.raises(HTTPException) as refused:
            artifacts._link_source_artifact(
                db, artifacts.CreateArtifactRequest(container_id=coll, source_artifact_id=root), _auth())
    assert refused.value.status_code == 409 and "Restore it" in refused.value.detail
    assert store.get_edge(db, coll, root) is None


async def test_a_batch_fetch_returns_an_archived_artifact_only_when_asked(db, monkeypatch):
    """An archived recall hit is hydrated through `/artifacts/batch`; without the option the batch
    skipped it, so an archived search found hits and showed none."""
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "tamarisk batch archived", monkeypatch)
        await artifacts.update_artifact(artifact_id=root, body=artifacts.UpdateArtifactRequest(state="archived"),
                                        auth=_auth(), store_db=db)
    assert artifacts._fetch_authorized_docs(db, _auth(), [root]) == []
    got = artifacts._fetch_authorized_docs(db, _auth(), [root], include_archived=True)
    assert [(d.get("root_id") or d.get("id"), d.get("state")) for d in got] == [(root, "archived")]


async def test_an_archived_artifact_can_be_deleted(db, monkeypatch):
    """2026-10-13, a tester: deleting an archived artifact answered 404 from every control."""
    monkeypatch.setattr(artifacts, "check_access", lambda *a, **k: None)
    inbox = _container(db, "Inbox")
    with acting_as(USER, principal_type="user"):
        root = _write_workspace_create(db, inbox, "tamarisk delete archived", monkeypatch)
        await artifacts.update_artifact(artifact_id=root, body=artifacts.UpdateArtifactRequest(state="archived"),
                                        auth=_auth(), store_db=db)
        out = await artifacts.delete_artifact(artifact_id=root, cascade=False, auth=_auth(), store_db=db)
    assert out["deleted"] is True
    assert not db.artifacts.versions_of(root), "the archived lineage is still stored"
    assert store.list_collection_artifacts(db, inbox, include_archived=True) == []
