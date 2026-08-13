"""The OTHER half of `test_an_unreachable_mirror_leaves_a_record.py`: something acts on the record.

That file asserts the obligation is written. This one asserts it is discharged — claimed, redone
through the same mirror-write path that failed, and settled — and, just as importantly, that the
three ways a drain goes wrong do not happen here:

    it must not run at all on a node with no object store   (the air-gap invariant)
    it must not retry what can never succeed                (dead-lettering)
    two of them must not do the same work                   (the claim is a compare-and-set)

Everything is driven against a real lattice and the real content tier. The only thing faked is the
object store itself, which is the one component the whole feature is about not being able to reach.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from mantle.services import content_service as cs
from mantle.services import mirror_drain as md
from mantle.services.dependencies import get_store_db

# Bound HERE, at collection, and deliberately not inside the fixture. `conftest.py`'s `client`
# builds its ASGI transport over the `app` object it imported at ITS import time, and another test
# module in this suite calls `importlib.reload(mantle.main)` — which replaces `mantle.main.app`
# with a different FastAPI instance. A fixture importing it later would install its dependency
# override on an app the client never calls, and every request would fall through to the
# autouse `MagicMock` store, 404ing for reasons that have nothing to do with this file.
from mantle.main import app

TASK_CT = md.MIRROR_TASK_CT
CONTENT_KEY = "artifacts/a-1.content"
TASK_ID = md.mirror_task_id(CONTENT_KEY)


# ---------------------------------------------------------------------------
# Fixtures — the same provisioned node the record test uses
# ---------------------------------------------------------------------------

@pytest.fixture()
def envelope(monkeypatch):
    """The REAL `content_crypto` envelope with only the master-key SOURCE injected: the bytes the
    drain reads out of the CAS and pushes to the object store are a genuine `MEC1` envelope, so
    "the object landed" is a claim about the actual content and not about a placeholder."""
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
    """A provisioned node (store path + keys dir) with its content routes wired to a real lattice."""
    from types import SimpleNamespace

    from mantle.db import lattice_api
    from mantle.entities.artifact import Artifact

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
    return [dict(r) for r in db.conn.read().execute(
        "SELECT id, ct, status, priority, operator, task_key, claimed_by, claimed_at,"
        " next_retry_at, completed_at FROM task ORDER BY id").fetchall()]


def _unreachable():
    fake = MagicMock()
    fake.put_object.side_effect = EndpointConnectionError(endpoint_url="http://minio:9000")
    return fake


def _answered(status: int, code: str, headers: dict | None = None):
    fake = MagicMock()
    fake.put_object.side_effect = ClientError(
        {"Error": {"Code": code, "Message": code},
         "ResponseMetadata": {"HTTPStatusCode": status, "HTTPHeaders": headers or {}}}, "PutObject")
    return fake


class _Bucket:
    """A mirror that works, and remembers. The object store, as a dict."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.types: dict[str, str] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None, **kw):
        self.objects[Key] = bytes(Body)
        self.types[Key] = ContentType
        return {}

    def __getattr__(self, name):        # head_bucket / create_bucket / put_bucket_cors / ...
        return MagicMock()


async def _upload(client, body: bytes, *, mirror) -> dict:
    """One `PUT /artifacts/a-1/content` against a given object store."""
    with patch.object(cs, "_s3_edge_internal", mirror), \
            patch.object(cs, "edge_store_configured", return_value=True):
        r = await client.put("/artifacts/a-1/content", content=body,
                             headers={"Content-Type": "application/octet-stream"})
    assert r.status_code == 200, r.text
    return r.json()


def _drain(db, mirror, *, worker_id=None, configured=True):
    """One drain pass against a given object store."""
    with patch.object(cs, "_s3_edge_internal", mirror), \
            patch.object(cs, "edge_store_configured", return_value=configured):
        return md.drain_mirror_pending(db, worker_id=worker_id)


def _task_doc(db):
    return db.artifacts.get_artifact(TASK_ID)


# ---------------------------------------------------------------------------
# The whole point: the obligation is discharged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_drain_puts_the_owed_object_and_completes_the_task(routed, client):
    """Unreachable mirror -> one task. Mirror returns -> the object lands and the task is done.

    The bytes put are asserted to be the SAME envelope the upload stored, byte for byte, because
    that is the property that makes a mirrored copy readable at all: a re-encryption here would
    produce an object whose envelope no owner's key opens.
    """
    up = await _upload(client, b"the payload", mirror=_unreachable())
    assert [t["status"] for t in _tasks(routed)] == ["pending"]

    bucket = _Bucket()
    report = _drain(routed, bucket)

    assert report["claimed"] == 1 and report["done"] == 1
    assert CONTENT_KEY in bucket.objects
    # The object is the artifact's own envelope, unchanged and unopened by the drain.
    tier = cs.local_content_tier()
    assert bucket.objects[CONTENT_KEY] == tier.get(up["content_ref"], collection="")
    assert bucket.objects[CONTENT_KEY][:4] == b"MEC1"

    row = _tasks(routed)[0]
    assert row["status"] == "done"
    assert row["claimed_by"] is None
    assert row["completed_at"]                      # `recent_terminal` orders on this
    doc = _task_doc(routed)
    assert doc["status"] == "done" and doc["attempts"] == 1 and doc["last_error"] is None


@pytest.mark.asyncio
async def test_a_drained_task_is_not_claimed_again(routed, client):
    """`done` leaves the pending bucket, so a second pass is a no-op rather than a second PUT."""
    await _upload(client, b"once", mirror=_unreachable())
    first = _Bucket()
    _drain(routed, first)
    second = _Bucket()
    report = _drain(routed, second)
    assert report["claimed"] == 0 and second.objects == {}


# ---------------------------------------------------------------------------
# Dead-lettering: a task that can never succeed stops consuming attempts
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_store_that_refuses_dead_letters_rather_than_looping(routed, client):
    """The mirror came back and said NO. `mirror_failure_is_transient` reads that off HTTP, and a
    request that is byte-identical on every retry gets the identical answer — so the task stops.

    This is the sequence that actually happens: the record exists because the store was
    UNREACHABLE (no answer, transient), and the permanence is only discovered on the retry.
    """
    await _upload(client, b"payload", mirror=_unreachable())
    report = _drain(routed, _answered(403, "AccessDenied"))

    assert report["claimed"] == 1 and report["dead"] == 1
    row = _tasks(routed)[0]
    assert row["status"] == "dead" and row["completed_at"] and row["next_retry_at"] is None
    doc = _task_doc(routed)
    assert doc["dead_reason"] == "permanent"
    assert "refused" in doc["last_error"] and "AccessDenied" in doc["last_error"]

    # And it is not looping: a dead row is outside the pending bucket forever.
    again = _drain(routed, _answered(403, "AccessDenied"))
    assert again["claimed"] == 0
    assert routed.artifacts.count_by_status(TASK_CT)["dead"] == 1


@pytest.mark.asyncio
async def test_a_dead_task_carries_the_operator_the_reason_and_the_stranded_artifact(routed, client):
    """What a dead task LOOKS like: the operator, why it died, and which artifact's content is
    now readable here and nowhere else.

    Read at its own id through the ordinary artifact route, which is what the derived, slash-free
    id is FOR — this fixture patches `check_access`, and that is not incidental. A row written by
    `_put_op` has no grant minted for it, so an unpatched read of it 404s for every principal on
    the node, exactly as it does for every mesh cursor and watermark written the same way. The
    fields asserted here are what `/status`'s `dead` count sends an operator to look at; the
    surface that serves them without a grant is the `task` sidecar, not this route.
    """
    await _upload(client, b"payload", mirror=_unreachable())
    _drain(routed, _answered(403, "AccessDenied"))

    r = await client.get("/artifacts/" + TASK_ID)
    assert r.status_code == 200, r.text
    ctx = r.json().get("context") or {}
    if isinstance(ctx, str):
        ctx = json.loads(ctx)
    body = {**r.json(), **(ctx if isinstance(ctx, dict) else {})}
    assert body["operator"] == "op.content.mirror"
    assert body["status"] == "dead"
    assert body["dead_reason"] == "permanent"
    assert body["arguments"]["artifact_id"] == "a-1"
    assert body["arguments"]["content_key"] == CONTENT_KEY
    assert body["last_error"]


@pytest.mark.asyncio
async def test_a_task_whose_artifact_is_gone_dead_letters(routed, client):
    """`never` also covers a premise that stopped being true. No artifact, no obligation — and no
    amount of waiting brings one back."""
    await _upload(client, b"payload", mirror=_unreachable())
    with routed.conn.write() as cur:
        cur.execute("UPDATE vertex SET doc = json_set(doc, '$.state', 'archived') WHERE id = 'a-1'")

    report = _drain(routed, _Bucket())
    assert report["dead"] == 1
    assert "no longer exists" in _task_doc(routed)["last_error"]


@pytest.mark.asyncio
async def test_the_bytes_being_gone_dead_letters_rather_than_retrying_forever(routed, client):
    """The local CAS no longer holds the ref. There is nothing on this node to mirror and no
    second copy to recover it from — that absent copy is what the mirror was FOR."""
    up = await _upload(client, b"payload", mirror=_unreachable())
    with patch.object(cs, "local_content_has", return_value=False):
        report = _drain(routed, _Bucket())
    assert report["dead"] == 1
    assert up["content_ref"] in _task_doc(routed)["last_error"]


# ---------------------------------------------------------------------------
# The security property: the artifact authorizes the work, the task only points at it
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_task_cannot_name_a_key_its_artifact_does_not_own(routed, client):
    """A row of this content type is something a caller can CREATE, so `arguments` is a pointer and
    never an instruction. A forged task asking for an artifact's bytes at somebody else's key is
    refused against the artifact's own recorded context — and refused permanently, because the
    answer cannot change."""
    await _upload(client, b"payload", mirror=_unreachable())
    doc = _task_doc(routed)
    doc["arguments"] = {**doc["arguments"], "content_key": "artifacts/somebody-elses.content"}
    routed.artifacts.put_artifact(doc, stamp_rev=False)

    bucket = _Bucket()
    report = _drain(routed, bucket)
    assert report["dead"] == 1
    assert bucket.objects == {}                     # nothing was written anywhere
    assert "does not record content_key" in _task_doc(routed)["last_error"]


@pytest.mark.asyncio
async def test_a_task_cannot_name_a_ref_that_is_not_a_cas_address(routed, client):
    """Shape is checked before anything is read: a ref cannot address a path."""
    await _upload(client, b"payload", mirror=_unreachable())
    doc = _task_doc(routed)
    doc["arguments"] = {**doc["arguments"], "content_ref": "../../etc/passwd"}
    routed.artifacts.put_artifact(doc, stamp_rev=False)

    report = _drain(routed, _Bucket())
    assert report["dead"] == 1
    assert "not a CAS address" in _task_doc(routed)["last_error"]


# ---------------------------------------------------------------------------
# The air-gap invariant
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_node_with_no_object_store_does_nothing(routed, client):
    """A node with no mirror never records a task, so its drain has nothing to do — but the drain
    must not even LOOK. The predicate is the same one `put_bytes_encrypted` uses to decide there is
    nothing to defer, so the two cannot disagree about what "no object store" means."""
    await _upload(client, b"payload", mirror=_unreachable())     # the row exists...
    before = _tasks(routed)

    with patch.object(cs, "edge_store_configured", return_value=False):
        with patch.object(routed.artifacts, "pending_window",
                          side_effect=AssertionError("the queue must not be read")):
            report = md.drain_mirror_pending(routed)

    assert report == {"skipped": "no object store is configured on this node"}
    assert _tasks(routed) == before                  # ...and is untouched


def test_the_runner_declines_to_start_without_an_object_store():
    """`start_mirror_drain` costs one predicate and creates no task: no coroutine, no timer, no
    wake-up, nothing to cancel. An air-gapped node is silent by construction, not by a branch
    inside a loop that is nevertheless running."""
    async def _go():
        with patch.object(cs, "edge_store_configured", return_value=False):
            started = md.start_mirror_drain(MagicMock())
        assert started is False
        assert md._task is None and md._wake is None
        md.notify_pending()                          # and waking a drain that isn't there is safe

    import asyncio
    asyncio.run(_go())


# ---------------------------------------------------------------------------
# The claim is a compare-and-set
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_two_drains_cannot_claim_the_same_task(routed, client):
    """`try_claim` is `UPDATE ... WHERE status='pending'` under SQLite's single-writer lock, so one
    of two concurrent drains wins and the other sees an empty pass. The object is therefore put
    once, not twice, and only one of them settles the row."""
    await _upload(client, b"payload", mirror=_unreachable())

    bucket = _Bucket()
    reports: list[dict] = []
    start = threading.Barrier(2)

    def _go(name):
        start.wait(timeout=10)
        reports.append(md.drain_mirror_pending(routed, worker_id=name))

    # Patched around the threads, not inside them: `patch.object` mutates a module global and
    # restores it, which two threads doing it at once would race — a rig defect, not the property
    # under test.
    with patch.object(cs, "_s3_edge_internal", bucket), \
            patch.object(cs, "edge_store_configured", return_value=True):
        threads = [threading.Thread(target=_go, args=("drain-%d" % i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

    assert sum(r["claimed"] for r in reports) == 1
    assert sum(r.get("done", 0) for r in reports) == 1
    assert [t["status"] for t in _tasks(routed)] == ["done"]


@pytest.mark.asyncio
async def test_a_drain_never_claims_an_operator_it_does_not_implement(routed, client):
    """A content type MAY carry more than one operator. This drain narrows its window to its own
    in SQL, so another operator's task is never claimed, never invoked, and never put back — and
    that holds whether or not anything else ever writes under this type."""
    await _upload(client, b"payload", mirror=_unreachable())
    from mantle.mesh.sync import _put_op
    _put_op(routed, {"id": "task-op.someone.else:1", "content_type": TASK_CT,
                     "operator": "op.someone.else", "arguments": {}, "status": "pending",
                     "priority": 9, "task_key": "op.someone.else:1", "content": "other"})

    report = _drain(routed, _Bucket())
    assert report["claimed"] == 1                    # ours, and only ours
    rows = {t["id"]: t for t in _tasks(routed)}
    assert rows["task-op.someone.else:1"]["status"] == "pending"
    assert rows["task-op.someone.else:1"]["claimed_by"] is None
    assert rows[TASK_ID]["status"] == "done"


# ---------------------------------------------------------------------------
# Claim vs refresh: the in-flight claim is for superseded bytes, and it must NOTICE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_refresh_beats_an_in_flight_claim_and_the_claim_finds_out(routed, client):
    """A new failed upload for the same `content_key` rewrites the row to `pending` and clears
    `claimed_by`. That races a drain that is mid-attempt on the previous ref.

    The refresh wins — it names the bytes the artifact now has — and `settle`, which compares on
    `claimed_by`, is what makes the loser find out instead of stamping `done` over the newer row.
    """
    first = await _upload(client, b"first", mirror=_unreachable())
    me = "drain-in-flight"
    assert routed.artifacts.try_claim(TASK_ID, worker_id=me, now_iso="2026-01-01T00:00:00+00:00")

    second = await _upload(client, b"second bytes", mirror=_unreachable())
    assert second["content_ref"] != first["content_ref"]

    row = _tasks(routed)[0]
    assert row["status"] == "pending" and row["claimed_by"] is None

    # The in-flight attempt tries to complete. It cannot, and says so.
    assert routed.artifacts.settle(TASK_ID, worker_id=me, to_status="done",
                                   completed_at="2026-01-01T00:00:01+00:00") is False
    row = _tasks(routed)[0]
    assert row["status"] == "pending"
    assert _task_doc(routed)["arguments"]["content_ref"] == second["content_ref"]

    # And the drain then mirrors the CURRENT bytes, not the superseded ones.
    bucket = _Bucket()
    _drain(routed, bucket)
    tier = cs.local_content_tier()
    assert bucket.objects[CONTENT_KEY] == tier.get(second["content_ref"], collection="")


@pytest.mark.asyncio
async def test_a_superseded_task_is_retried_once_and_then_dead_lettered(routed, client):
    """A task naming a ref the artifact no longer points at is momentarily indistinguishable from a
    live one — the refresh rewrites the row BEFORE `_record_content_ref` rewrites the artifact — so
    it is retried once. A second sighting is the artifact's settled state and it dies."""
    await _upload(client, b"payload", mirror=_unreachable())
    doc = _task_doc(routed)
    doc["arguments"] = {**doc["arguments"], "content_ref": "cas/" + "0" * 64}
    routed.artifacts.put_artifact(doc, stamp_rev=False)

    bucket = _Bucket()
    first = _drain(routed, bucket)
    assert first["pending"] == 1 and bucket.objects == {}
    row = _tasks(routed)[0]
    assert row["status"] == "pending" and row["next_retry_at"]
    assert _task_doc(routed)["attempts"] == 1

    # Eligible again -> seen twice -> dead.
    with routed.conn.write() as cur:
        cur.execute("UPDATE task SET next_retry_at = NULL WHERE id = ?", (TASK_ID,))
    second = _drain(routed, bucket)
    assert second["dead"] == 1
    assert _task_doc(routed)["dead_reason"] == "superseded"
    assert bucket.objects == {}


@pytest.mark.asyncio
async def test_a_process_that_died_mid_write_recovers_its_own_claim(routed, client):
    """Nothing else in mantle reclaims a stale lease, so a claim that outlives the attempt that
    made it would strand the obligation in `claimed` forever. The drain reclaims exactly the id it
    is about to use — the one claim it can prove is not live."""
    await _upload(client, b"payload", mirror=_unreachable())
    me = md._worker_id(routed)
    assert routed.artifacts.try_claim(TASK_ID, worker_id=me, now_iso="2026-01-01T00:00:00+00:00")

    bucket = _Bucket()
    report = _drain(routed, bucket)
    assert report["reclaimed"] == 1 and report["done"] == 1
    assert CONTENT_KEY in bucket.objects


# ---------------------------------------------------------------------------
# Backoff, derived
# ---------------------------------------------------------------------------

def test_a_retry_after_from_the_store_wins_outright():
    """RFC 9110 §10.2.3: the origin's own statement about when it can answer. Nothing computed
    here can improve on being told."""
    from datetime import datetime, timezone
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    exc = ClientError({"Error": {"Code": "SlowDown"},
                       "ResponseMetadata": {"HTTPStatusCode": 429,
                                            "HTTPHeaders": {"retry-after": "90"}}}, "PutObject")
    assert md._next_retry_at(when, attempts=3, elapsed_s=1.0, exc=exc) == "2026-01-01T00:01:30+00:00"


def test_the_delay_is_one_round_of_waiting_per_consecutive_failure():
    """With no `Retry-After` the delay is `max(what the attempt cost, what this node was already
    willing to wait) x the number of consecutive failures`. Both factors are measured or counted;
    neither is chosen."""
    from datetime import datetime, timezone
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with patch.object(md, "_client_patience_seconds", return_value=10.0):
        assert md._next_retry_at(when, attempts=1, elapsed_s=0.0) == "2026-01-01T00:00:10+00:00"
        assert md._next_retry_at(when, attempts=3, elapsed_s=0.0) == "2026-01-01T00:00:30+00:00"
        # A store that hung for longer than the floor sets the round itself.
        assert md._next_retry_at(when, attempts=2, elapsed_s=25.0) == "2026-01-01T00:00:50+00:00"


def test_the_floor_is_the_clients_own_configured_patience():
    """Not a constant in this module: the connect/read timeout the edge client is already built
    with. A connection REFUSED costs microseconds to discover, and without this floor the derived
    delay would be a hot loop rather than a backoff."""
    assert md._client_patience_seconds() == max(
        float(cs._s3_edge_internal.meta.config.connect_timeout),
        float(cs._s3_edge_internal.meta.config.read_timeout))
    assert md._client_patience_seconds() > 0


@pytest.mark.asyncio
async def test_backoff_grows_across_consecutive_failures_on_the_real_row(routed, client):
    """End to end on the row: `attempts` accumulates and `next_retry_at` moves further out."""
    await _upload(client, b"payload", mirror=_unreachable())
    with patch.object(md, "_client_patience_seconds", return_value=5.0):
        _drain(routed, _unreachable())
        first = _tasks(routed)[0]["next_retry_at"]
        assert _task_doc(routed)["attempts"] == 1
        with routed.conn.write() as cur:
            cur.execute("UPDATE task SET next_retry_at = NULL WHERE id = ?", (TASK_ID,))
        _drain(routed, _unreachable())
    second = _tasks(routed)[0]["next_retry_at"]
    assert _task_doc(routed)["attempts"] == 2
    assert _tasks(routed)[0]["status"] == "pending"          # transient: still owed, never dead
    assert second > first


@pytest.mark.asyncio
async def test_new_bytes_reset_the_backoff_rather_than_inheriting_it(routed, client):
    """`_record_mirror_pending` writes no `attempts`, so a refreshed row starts at one round again.
    The failures the previous ref accumulated say nothing about the new one."""
    await _upload(client, b"payload", mirror=_unreachable())
    _drain(routed, _unreachable())
    assert _task_doc(routed)["attempts"] == 1
    await _upload(client, b"different payload", mirror=_unreachable())
    assert _task_doc(routed).get("attempts") in (None, 0)
    assert _tasks(routed)[0]["next_retry_at"] is None         # eligible now


# ---------------------------------------------------------------------------
# The runner's cadence is the queue's, not a chosen interval
# ---------------------------------------------------------------------------

def test_an_empty_queue_schedules_nothing_at_all():
    """`None` means "there is nothing to be scheduled FOR" — the runner waits for work to appear
    instead of waking on a timer to look for it."""
    assert md._sleep_for(None) is None


def test_the_sleep_is_the_time_until_the_queues_own_next_eligible_row():
    from datetime import timedelta
    soon = md._now() + timedelta(seconds=30)
    assert 25.0 < md._sleep_for(md._iso(soon)) <= 30.0
    past = md._now() - timedelta(seconds=30)
    assert md._sleep_for(md._iso(past)) == 0.0


@pytest.mark.asyncio
async def test_the_drain_reports_when_its_own_queue_next_becomes_eligible(routed, client):
    """The pass hands the runner the schedule it just wrote, read back off the same bounded window
    the next pass will claim from — so the two cannot disagree."""
    await _upload(client, b"payload", mirror=_unreachable())
    report = _drain(routed, _unreachable())
    assert report["next_eligible"] == _tasks(routed)[0]["next_retry_at"]


# ---------------------------------------------------------------------------
# Ember: the conflict, and the name that ends it
# ---------------------------------------------------------------------------

def test_ember_selects_by_content_type_alone_and_this_is_not_the_one():
    """The load-bearing fact, asserted against ember's own source rather than assumed.

    `pool.claim` selects `pending_window(TASK_CT, ...)` with no operator predicate, and mantle
    cannot change that from here — the decision is made in ember's query. So the content type is
    the whole of what decides whether an ember worker sharing this lattice takes a mantle row: it
    used to, fail to invoke `op.content.mirror`, and dead-letter at its own `MAX_ATTEMPTS` an
    obligation it never looked at the bytes for.

    Both halves are checked here, because either alone is worthless: that ember still selects on
    the type alone (so the type is what matters), and that the type it selects is not ours (so it
    selects none of our rows). What used to be a deployment invariant — *run the drain where an
    ember pool worker is not* — is now a naming fact the query planner enforces.

    Read as SOURCE rather than imported: `ember` imports `mantle` (not the other way round), so
    importing it from mantle's own suite would be a dependency cycle in the test rig. Skipped when
    the sibling checkout is not present — this asserts a fact about a neighbour, and a neighbour
    that is not there cannot be asserted about.
    """
    here = Path(__file__).resolve()
    candidates = [p / "agience-ember" / "src" / "ember" / "runtime" / "pool.py"
                  for p in here.parents]
    src_path = next((c for c in candidates if c.exists()), None)
    if src_path is None:
        pytest.skip("no ember checkout alongside this one")
    src = src_path.read_text(encoding="utf-8")
    claim = src.split("def claim(", 1)[1].split("\ndef ", 1)[0]
    # The claim predicate is (ct, status) and nothing else — unchanged, and not ours to change.
    assert "pending_window(TASK_CT" in claim
    assert "operator" not in claim.split("pending_window", 1)[1].split(")", 1)[0]

    # ...so the type is the whole decision, and ember's is not the one mantle writes under. Read
    # off ember's own source too: a copy of the string here would agree with itself forever.
    ember_ct = src.split("TASK_CT", 1)[1].split("=", 1)[1].split("\n", 1)[0].strip().strip('"\'')
    assert ember_ct == md.SHARED_TASK_CT, "ember's pool moved; mantle's separation must be rechecked"
    assert md.MIRROR_TASK_CT != ember_ct


def test_a_row_under_the_old_shared_type_is_adopted_rather_than_stranded(routed):
    """MIGRATION. A node that ran the previous version has rows on disk under the shared type.

    They are the obligations this whole module exists to keep, and after the rename nothing else
    would ever look at them: the drain reads its own type, and the only worker that reads the
    shared one is the one that destroys them. So the pass adopts them — same id, same key, same
    status, same backoff, a different content type — and then discharges them in the ordinary way.
    """
    from mantle.mesh.sync import _put_op

    stale_id = md.mirror_task_id("artifacts/old.content")
    _put_op(routed, {"id": stale_id, "content_type": md.SHARED_TASK_CT,
                     "operator": md.MIRROR_OPERATOR, "status": "pending", "priority": 0,
                     "task_key": md.mirror_task_key("artifacts/old.content"),
                     "attempts": 3, "next_retry_at": None,
                     "arguments": {"artifact_id": "gone", "content_key": "artifacts/old.content",
                                   "content_ref": "cas/" + "0" * 64},
                     "content": "stale"})
    assert routed.artifacts.count_by_status(md.SHARED_TASK_CT)["pending"] == 1

    report = _drain(routed, _Bucket())

    assert report["adopted"] == 1
    row = {t["id"]: t for t in _tasks(routed)}[stale_id]
    assert row["ct"] == md.MIRROR_TASK_CT
    assert row["task_key"] == md.mirror_task_key("artifacts/old.content")
    # The counters followed the row: no double count, nothing left behind on the old type.
    assert routed.artifacts.count_by_status(md.SHARED_TASK_CT)["pending"] == 0
    # And it was not merely renamed — it was WORKED in the same pass. The artifact it names does
    # not exist, so the drain dead-letters it on determinate grounds of its own.
    assert row["status"] == "dead"
    assert routed.artifacts.get_artifact(stale_id)["attempts"] == 4

    # Idempotent: a second pass finds nothing left to adopt.
    assert _drain(routed, _Bucket())["adopted"] == 0


def test_adoption_leaves_another_operators_shared_row_alone(routed):
    """It moves what it owes and nothing else. A row under the shared type belonging to another
    operator stays exactly where that operator's worker will look for it — adopting it would be
    this drain stealing work in the very act of refusing to claim it."""
    from mantle.mesh.sync import _put_op

    _put_op(routed, {"id": "task-op.someone.else:1", "content_type": md.SHARED_TASK_CT,
                     "operator": "op.someone.else", "arguments": {}, "status": "pending",
                     "priority": 0, "task_key": "op.someone.else:1", "content": "other"})

    assert _drain(routed, _Bucket())["adopted"] == 0
    rows = {t["id"]: t for t in _tasks(routed)}
    assert rows["task-op.someone.else:1"]["ct"] == md.SHARED_TASK_CT
    assert rows["task-op.someone.else:1"]["status"] == "pending"
    assert routed.artifacts.count_by_status(md.SHARED_TASK_CT)["pending"] == 1


@pytest.mark.asyncio
async def test_this_drain_narrows_in_sql_and_not_after_the_claim(routed, client):
    """The one direction mantle owns, asserted where it matters: the narrowing is in the QUERY.

    `test_a_drain_never_claims_an_operator_it_does_not_implement` shows the outcome; this shows
    the mechanism, because the two are not the same guarantee. Filtering after `pending_window`
    would produce the same visible result while still burning the window on rows the drain cannot
    do — and on a pool shared with an ember worker, a window full of somebody else's tasks is a
    drain that never reaches its own.
    """
    await _upload(client, b"payload", mirror=_unreachable())
    seen: list[dict] = []
    real = routed.artifacts.pending_window

    def _spy(ct, **kw):
        seen.append(kw)
        return real(ct, **kw)

    with patch.object(routed.artifacts, "pending_window", _spy):
        _drain(routed, _Bucket())
    assert seen and all(kw.get("operator") == "op.content.mirror" for kw in seen)


# ---------------------------------------------------------------------------
# /status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_reports_a_backlog_that_goes_down(routed, client, monkeypatch):
    """The count is worth publishing now because it moves in both directions. Five counter reads —
    `count_by_status` — so the cost does not grow with the backlog it is reporting."""
    import mantle.db.backend as backend
    monkeypatch.setattr(backend, "store_handle", lambda: routed)

    await _upload(client, b"payload", mirror=_unreachable())
    r = await client.get("/status")
    assert r.status_code == 200
    pool = r.json()["work_pool"]
    assert pool["content_type"] == TASK_CT
    # Operator-exact, and free: this content type carries one operator's rows and no others, so
    # the counter IS the mirror backlog rather than a pool's depth that happens to include it.
    assert pool["operator"] == md.MIRROR_OPERATOR
    assert pool["pending"] == 1 and pool["done"] == 0

    _drain(routed, _Bucket())
    pool = (await client.get("/status")).json()["work_pool"]
    assert pool["pending"] == 0 and pool["done"] == 1
