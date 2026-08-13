"""`PUT /artifacts/{id}/content` when the object store is CONFIGURED BUT UNREACHABLE.

`test_content_route_needs_no_object_store.py` holds the two ends of that route's tier: a node with
no object store (local is the store, complete and quiet) and a node whose mirror works. This file
holds the middle case, which used to fall through both.

The bytes were never at risk — they are local, verified, and the request has always, correctly,
returned 200. What was lost was PEER REACHABILITY: nothing recorded that the mirror leg was still
owed, so the object store never received the copy and no drain could ever discover that it had
not. `shard/content_tier.promote_local_content` was not that drain and could not become one: it
walks the `content_ref` COLUMN toward the OVH origin, while this route records `content_cas_ref` in
the artifact's CONTEXT and mirrors to the MinIO/S3 edge.

What is asserted here is the record, not the retry. The obligation is enqueued on the store's own
work pool (`db/schema.py`'s `task` sidecar) in the shape `ember/runtime/pool.py::enqueue` uses, and
a drain that claims and completes those tasks is a separate piece that does not exist yet. These
tests are written so they keep holding when it does.

The three configurations, and what each owes:

    no object store      -> nothing. An absent mirror is a configuration, not a failure.
    mirror reachable     -> nothing. It has the copy.
    mirror unreachable   -> exactly one task, naming what to redo.
    mirror REFUSING      -> nothing queued, loudly logged. A byte-identical retry of a request the
                            store answered gets the same answer; that is a person's problem.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from mantle.routers import artifacts_router as ar
from mantle.services import content_service as cs
from mantle.services import mirror_drain as md

# Bound HERE, at collection, and deliberately not inside the fixture. `conftest.py`'s `client`
# builds its ASGI transport over the `app` object it imported at ITS import time, and another test
# module in this suite calls `importlib.reload(mantle.main)` — which replaces `mantle.main.app`
# with a different FastAPI instance. A fixture importing it later would install its dependency
# override on an app the client never calls, and every request would fall through to the
# autouse `MagicMock` store, 404ing for reasons that have nothing to do with this file.
from mantle.main import app

# Imported rather than spelled, so the type the route WRITES and the type this file ASSERTS ON
# cannot drift apart. What the value has to be is asserted once, below, where the reason lives.
TASK_CT = md.MIRROR_TASK_CT
CONTENT_KEY = "artifacts/a-1.content"
TASK_ID = "task-op.content.mirror:" + hashlib.blake2b(
    CONTENT_KEY.encode("utf-8"), digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures — a provisioned node, the real envelope, the real routes, a real store
# ---------------------------------------------------------------------------

@pytest.fixture()
def envelope(monkeypatch):
    """The REAL `content_crypto` envelope with only the master-key SOURCE injected — the same
    fixture `test_content_route_needs_no_object_store.py` uses, for the same reason: the CAS ref
    these tests assert about is the address of a genuine `MEC1` envelope."""
    from mantle.services import content_crypto

    real_encrypt, real_decrypt = content_crypto.encrypt_content, content_crypto.decrypt_content

    def _provider(principal_id: str) -> bytes:
        return hashlib.sha256(("master:" + principal_id).encode("utf-8")).digest()

    monkeypatch.setattr(content_crypto, "encrypt_content",
                        lambda pid, pt, **kw: real_encrypt(pid, pt, master_key_provider=_provider, **kw))
    monkeypatch.setattr(content_crypto, "decrypt_content",
                        lambda pid, blob, **kw: real_decrypt(pid, blob, master_key_provider=_provider, **kw))
    return _provider


@pytest.fixture()
def routed(tmp_path: Path, monkeypatch, envelope):
    """A provisioned node (store path + keys dir) with its content routes wired to a real lattice.

    A real store, not a mock: the whole claim is that the pending state is DURABLE, and only a
    store that actually writes rows can be asked whether it holds one.
    """
    from types import SimpleNamespace

    from mantle.db import lattice_api
    from mantle.entities.artifact import Artifact
    from mantle.services.dependencies import get_store_db

    root = tmp_path / "var"
    (root / "keys").mkdir(parents=True)
    (root / ".data").mkdir()
    monkeypatch.setenv("MANTLE_LATTICE_PATH", str(root / ".data" / "lattice.db"))
    monkeypatch.setenv("KEYS_DIR", str(root / "keys"))

    import mantle.db.backend as backend
    backend._CONTENT = None
    cs._LOCAL_TIER_ABSENT_LOGGED = False

    db = lattice_api.LatticeDatabase(str(tmp_path / "routed.db"), origin="node-a")
    lattice_api.create_artifact(db, Artifact(
        id="a-1", root_id="a-1", collection_id="", name="upload", content="",
        created_by="owner-a", content_type="application/octet-stream"))

    grant = SimpleNamespace(can_read=True, can_create=True, can_update=True, can_delete=True,
                            can_invoke=True, can_add=True, can_share=True, resource_id=None)
    app.dependency_overrides[get_store_db] = lambda: db
    with patch("mantle.routers.artifacts_router.check_access", return_value=grant):
        yield db
    app.dependency_overrides.pop(get_store_db, None)
    backend._CONTENT = None


def _tasks(db):
    """The `task` sidecar's own rows — the indexed coordination state, read as a drain would."""
    return [dict(r) for r in db.conn.read().execute(
        "SELECT id, ct, status, priority, operator, task_key, claimed_by, claimed_at,"
        " next_retry_at, completed_at FROM task ORDER BY id").fetchall()]


def _unreachable():
    """A mirror this node HAS and cannot reach. `EndpointConnectionError` carries no `response`,
    because botocore never got one."""
    fake = MagicMock()
    fake.put_object.side_effect = EndpointConnectionError(endpoint_url="http://minio:9000")
    return fake


def _answered(status: int, code: str):
    """A mirror that ANSWERED. `ClientError` carries the store's own HTTP status, which is what
    decides whether repeating the request could tell us anything new."""
    fake = MagicMock()
    fake.put_object.side_effect = ClientError(
        {"Error": {"Code": code, "Message": code},
         "ResponseMetadata": {"HTTPStatusCode": status}}, "PutObject")
    return fake


# ---------------------------------------------------------------------------
# The three configurations
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_node_with_no_object_store_records_nothing(routed, client):
    """The air-gap invariant, restated as the thing that must NOT appear: no mirror means nothing
    is owed, so a queue that accumulated here would be a permanent, growing backlog of work no
    node ever intends to do. The early return in `put_bytes_encrypted` is the guard — the callback
    is structurally unreachable, not merely unlikely."""
    with patch.object(cs, "edge_store_configured", return_value=False):
        r = await client.put("/artifacts/a-1/content", content=b"air-gapped bytes")
    assert r.status_code == 200 and r.json()["stored"] is True
    assert _tasks(routed) == []


@pytest.mark.asyncio
async def test_a_mirror_that_took_the_bytes_records_nothing(routed, client):
    """A working mirror owes nothing, and the success path is untouched."""
    fake = MagicMock()
    with patch.object(cs, "_s3_edge_internal", fake), \
            patch.object(cs, "edge_store_configured", return_value=True):
        r = await client.put("/artifacts/a-1/content", content=b"mirrored bytes")
    assert r.status_code == 200
    assert fake.put_object.call_args.kwargs["Key"] == CONTENT_KEY
    assert _tasks(routed) == []


@pytest.mark.asyncio
async def test_an_unreachable_mirror_records_exactly_one_task(routed, client):
    """The gap, closed: one row, on the store's own work pool, naming what has to be redone."""
    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        r = await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    assert r.status_code == 200, "the bytes are local and verified; this request succeeded"
    body = r.json()
    assert body["stored"] is True and body["deduplicated"] is False

    rows = _tasks(routed)
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == TASK_ID
    assert row["ct"] == TASK_CT
    assert row["status"] == "pending"
    assert row["operator"] == "op.content.mirror"
    assert row["task_key"] == "op.content.mirror:" + CONTENT_KEY
    assert row["priority"] == 0
    assert row["claimed_by"] is None and row["claimed_at"] is None
    assert row["completed_at"] is None


@pytest.mark.asyncio
async def test_the_payload_is_enough_to_redo_the_mirror_leg_and_nothing_more(routed, client):
    """Which artifact, which object-store key, which local CAS ref — the three things the mirror
    leg needs and cannot re-derive. No bucket, no endpoint, no size, no URL: every one of those is
    configuration read at the time the retry runs, and a frozen copy would be the stale half of a
    disagreement."""
    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        r = await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    doc = routed.artifacts.get_artifact(TASK_ID)
    assert doc["arguments"] == {"artifact_id": "a-1", "content_key": CONTENT_KEY,
                                "content_ref": r.json()["content_ref"]}
    assert "EndpointConnectionError" in doc["last_error"]
    # The ref names the artifact's CURRENT content — the same address its context now points at.
    ctx = json.loads(routed.artifacts.get_artifact("a-1")["context"])
    assert ctx["content_cas_ref"] == doc["arguments"]["content_ref"]


@pytest.mark.asyncio
async def test_the_ref_is_not_projected_into_the_content_ref_column(routed, client):
    """`content_ref` stays inside `arguments`. At the top level it is a projected vertex column,
    and `shard/content_tier.promote_local_content` walks exactly that column — a task row would be
    picked up as content to promote to a DIFFERENT mirror entirely."""
    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    row = routed.conn.read().execute(
        "SELECT content_ref FROM vertex WHERE id = ?", (TASK_ID,)).fetchone()
    assert row["content_ref"] is None


# ---------------------------------------------------------------------------
# Transient, permanent, and the difference
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_store_that_refused_is_not_queued_for_retry(routed, client):
    """403 AccessDenied: the store answered. This request is byte-identical on every retry, so the
    answer will be too — queueing it would be a loop wearing a queue's clothes. The upload still
    succeeds and the failure is still loud; it is simply not a retry."""
    with patch.object(cs, "_s3_edge_internal", _answered(403, "AccessDenied")), \
            patch.object(cs, "edge_store_configured", return_value=True):
        r = await client.put("/artifacts/a-1/content", content=b"refused bytes")
    assert r.status_code == 200 and r.json()["stored"] is True
    assert _tasks(routed) == []


@pytest.mark.asyncio
async def test_a_store_that_faulted_is_queued(routed, client):
    """503: RFC 9110 §15.6 — the server failed to fulfil an apparently valid request. That is the
    definition of retryable, read out of the protocol rather than out of an error-string list."""
    with patch.object(cs, "_s3_edge_internal", _answered(503, "ServiceUnavailable")), \
            patch.object(cs, "edge_store_configured", return_value=True):
        r = await client.put("/artifacts/a-1/content", content=b"deferred bytes")
    assert r.status_code == 200
    assert len(_tasks(routed)) == 1


def test_the_classifier_reads_http_rather_than_error_strings():
    """The two 4xx codes whose own definitions license a repeat are read as such; the rest of 4xx
    is not. Anything that is not a `ClientError` never got an answer at all, and an undecidable
    case resolves to transient — a visible task carrying its reason costs less than content that
    is silently unreachable from every peer."""
    def answered(status):
        return ClientError({"Error": {"Code": "X"},
                            "ResponseMetadata": {"HTTPStatusCode": status}}, "PutObject")

    assert cs.mirror_failure_is_transient(OSError("connection refused")) is True
    assert cs.mirror_failure_is_transient(
        EndpointConnectionError(endpoint_url="http://minio:9000")) is True
    for permanent in (400, 401, 403, 404, 411, 413):
        assert cs.mirror_failure_is_transient(answered(permanent)) is False, permanent
    for transient in (408, 429, 500, 502, 503, 504):
        assert cs.mirror_failure_is_transient(answered(transient)) is True, transient
    # A ClientError whose status cannot be read is undecidable, not permanent.
    assert cs.mirror_failure_is_transient(
        ClientError({"Error": {"Code": "X"}}, "PutObject")) is True


# ---------------------------------------------------------------------------
# Idempotence — the natural key is the content key
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_second_failed_upload_does_not_stack_a_second_task(routed, client):
    """Two failures for one artifact are one obligation. Driven with the route's content-addressed
    no-op disabled for the second attempt, so the mirror leg genuinely runs twice — otherwise this
    would only be re-proving the dedup that lives one step earlier."""
    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        await client.put("/artifacts/a-1/content", content=b"stranded bytes")
        assert len(_tasks(routed)) == 1
        with patch.object(cs, "local_content_has", return_value=False):
            second = await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    assert second.json()["deduplicated"] is False, "the mirror leg really was attempted again"
    rows = _tasks(routed)
    assert len(rows) == 1 and rows[0]["id"] == TASK_ID


@pytest.mark.asyncio
async def test_new_bytes_refresh_the_one_task_rather_than_adding_another(routed, client):
    """The key is the object store's ADDRESS, not the bytes. A second upload supersedes the first:
    the earlier ref stopped being this artifact's content when its context was overwritten, so a
    task still naming it would oblige the mirror to hold a version nothing will ever ask for.

    This is also why the ref cannot be part of the key: it addresses the ENVELOPE, whose nonce is
    fresh on every encryption, so even identical plaintext yields a different ref each time."""
    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        first = await client.put("/artifacts/a-1/content", content=b"stranded bytes")
        second = await client.put("/artifacts/a-1/content", content=b"newer stranded bytes")

    assert first.json()["content_ref"] != second.json()["content_ref"]
    rows = _tasks(routed)
    assert len(rows) == 1 and rows[0]["id"] == TASK_ID
    doc = routed.artifacts.get_artifact(TASK_ID)
    assert doc["arguments"]["content_ref"] == second.json()["content_ref"]


def test_the_task_id_is_a_function_of_the_natural_key_and_survives_a_url():
    """Deterministic in the content key — that is what makes the upsert the whole of the
    idempotence — and free of the `/` that a content key carries, because an id with an extra path
    segment in it does not survive `GET /artifacts/{id}`."""
    assert ar._mirror_task_id(CONTENT_KEY) == TASK_ID
    assert ar._mirror_task_id(CONTENT_KEY) == ar._mirror_task_id(CONTENT_KEY)
    assert ar._mirror_task_id("artifacts/b-2.content") != TASK_ID
    assert "/" not in TASK_ID
    assert ar._mirror_task_key(CONTENT_KEY) == "op.content.mirror:" + CONTENT_KEY


# ---------------------------------------------------------------------------
# Durability, visibility, and the promise not to break a completed write
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_record_outlives_the_process(routed, client, tmp_path: Path):
    """Durable means on disk, not in a module-level dict. Re-opened from the same file, the row is
    still there — and still `pending`, with `next_retry_at` NULL, which `pending_window` reads as
    eligible now."""
    from mantle.db import lattice_api

    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    reopened = lattice_api.LatticeDatabase(str(tmp_path / "routed.db"), origin="node-a")
    window = reopened.artifacts.pending_window(TASK_CT, limit=8, now_iso="2999-01-01T00:00:00Z")
    assert [r["id"] for r in window] == [TASK_ID]
    assert reopened.artifacts.count_by_status(TASK_CT)["pending"] == 1


@pytest.mark.asyncio
async def test_the_record_is_readable_at_its_own_id(routed, client):
    """A task IS an artifact, which is the point of putting it on the work pool rather than in a
    private side-table: the existing read surface answers for it, so "is this content pending
    mirror" needs no new endpoint and no hand-written query."""
    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    got = await client.get("/artifacts/" + TASK_ID)
    assert got.status_code == 200
    seen = got.json()
    assert seen["operator"] == "op.content.mirror"
    assert seen["status"] == "pending"
    assert seen["arguments"]["content_key"] == CONTENT_KEY


@pytest.mark.asyncio
async def test_the_record_never_leaves_this_node(routed, client):
    """Per-box operational state, pinned outside every observer's proper time by `_put_op` — so it
    costs no `_seq`, churns no publish cursor, and cannot reach a peer. A node's list of things it
    has not managed to upload is not an observation about the universe."""
    from mantle.mesh.sync import _is_replicated

    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    assert _is_replicated(TASK_CT) is False
    doc = routed.artifacts.get_artifact(TASK_ID)
    assert doc["_origin"] == "_local:" + TASK_ID and doc["_seq"] == 1
    assert doc["_origin"] != routed.origin


@pytest.mark.asyncio
async def test_the_row_is_written_under_mantles_own_content_type(routed, client):
    """The value, pinned where the reason for it lives.

    `ember/runtime/pool.py::claim` selects on `(ct, status)` and cannot filter on an operator, so
    the content type is the ONLY thing deciding whether an ember worker sharing this lattice takes
    a row it does not implement. Under the shared type it took them, failed to invoke
    `op.content.mirror`, and dead-lettered a mirror this node genuinely owed. Under this one it
    never sees them.

    The shape stays `ember`'s (operator / arguments / status / task_key) — that is what the typed
    accessors read. Only the name is mantle's.
    """
    assert md.MIRROR_TASK_CT == "application/vnd.agience.mirror-task+json"
    assert md.MIRROR_TASK_CT != md.SHARED_TASK_CT

    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True):
        await client.put("/artifacts/a-1/content", content=b"stranded bytes")

    assert [r["ct"] for r in _tasks(routed)] == [md.MIRROR_TASK_CT]
    # And the shared pool is empty: nothing of ours is written there any more.
    assert routed.artifacts.count_by_status(md.SHARED_TASK_CT) == {
        "pending": 0, "claimed": 0, "done": 0, "failed": 0, "dead": 0}


@pytest.mark.asyncio
async def test_a_failing_enqueue_cannot_turn_a_stored_write_into_a_500(routed, client):
    """The upload succeeded: the bytes are encrypted, written, verified and readable. Losing the
    note that the mirror is owed is bad; discarding a durable write to report that we could not
    write the note is worse."""
    with patch.object(cs, "_s3_edge_internal", _unreachable()), \
            patch.object(cs, "edge_store_configured", return_value=True), \
            patch.object(ar, "_record_mirror_pending",
                         side_effect=RuntimeError("the task table is unavailable")):
        r = await client.put("/artifacts/a-1/content", content=b"bookkeeping fails")

    assert r.status_code == 200 and r.json()["stored"] is True
    assert _tasks(routed) == []
    got = await client.get("/artifacts/a-1/content")
    assert got.status_code == 200 and got.content == b"bookkeeping fails"
