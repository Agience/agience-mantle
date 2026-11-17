"""`POST /system/seed` answers 202 at once and the seed runs on; `GET /system/seed/{run_id}` follows it.

BUG-122 (2026-11-16): on a fresh store the first seed ran past 30 s with no answer at all.
"""
from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import HTTPException, Response

from mantle.routers import system_router as S


@pytest.fixture
def seed(monkeypatch, tmp_path):
    """A seed whose tree pass blocks until `release` is set, with nothing else provisioned."""
    (tmp_path / "platform").mkdir()
    monkeypatch.setenv("AGIENCE_SEEDS_ROOT", str(tmp_path))
    monkeypatch.setattr(S, "require_platform_admin", lambda auth, db: "op-1", raising=False)
    monkeypatch.setattr(S, "_SEED_RUNS", {})
    release, outcome = threading.Event(), {"fail": None}

    class Report:
        errors = []

        def summary(self):
            return "artifacts +41 ~0 =0 | errors 0"

    def seed_from_artifacts(db, root):
        assert release.wait(10), "the test never released the seed"
        if outcome["fail"]:
            raise outcome["fail"]
        return Report()
    import mantle.services.seed_provisioning as SP
    monkeypatch.setattr(SP, "seed_from_artifacts", seed_from_artifacts)
    from mantle.services.seed_provisioning import platform_operators as P
    monkeypatch.setattr(P, "ensure_platform_operators", lambda db: True)
    monkeypatch.setattr(P, "ids", lambda: {P._COLLECTION_SLUG: "coll-1"})
    import mantle.db.backend as B
    monkeypatch.setattr(B, "get_artifact", lambda db, i: object())
    import mantle.services.seed_provisioning.user_provisioning as UP
    monkeypatch.setattr(UP, "provision_existing_accounts", lambda db, base: {"accounts": 0})
    yield release, outcome
    release.set()


async def _post():
    response = Response()
    run = await S.seed_platform(response=response, auth=None, store_db=object())
    return run, response


async def _until_finished(run_id):
    for _ in range(200):
        run = await S.seed_run(run_id=run_id, auth=None, store_db=object())
        if run.state != "running":
            return run
        await asyncio.sleep(0.02)
    raise AssertionError("the run never finished")


async def test_the_seed_answers_at_once_and_is_followed_to_done(seed):
    release, _ = seed
    run, response = await asyncio.wait_for(_post(), timeout=2)
    assert run.state == "running" and not run.applied
    assert run.follow == f"/system/seed/{run.run_id}" == response.headers["Location"]

    again, _ = await _post()
    assert again.run_id == run.run_id, "a call while a run is going answers with that run"

    release.set()
    done = await _until_finished(run.run_id)
    assert done.state == "done" and done.applied and done.errors == []
    assert done.summary.startswith("artifacts +41") and done.finished_at


async def test_a_failed_run_says_so_and_the_next_call_starts_another(seed):
    release, outcome = seed
    outcome["fail"] = RuntimeError("store unreachable")
    run, _ = await _post()
    release.set()
    failed = await _until_finished(run.run_id)
    assert failed.state == "failed" and not failed.applied
    assert failed.errors == ["seed: RuntimeError: store unreachable"]

    outcome["fail"] = None
    second, _ = await _post()
    assert second.run_id != run.run_id
    assert (await _until_finished(second.run_id)).state == "done"


async def test_an_unknown_run_is_404(seed):
    with pytest.raises(HTTPException) as e:
        await S.seed_run(run_id="no-such-run", auth=None, store_db=object())
    assert e.value.status_code == 404


async def test_a_bare_node_still_refuses_at_once(seed, monkeypatch):
    monkeypatch.delenv("AGIENCE_SEEDS_ROOT")
    with pytest.raises(HTTPException) as e:
        await _post()
    assert e.value.status_code == 409 and S._SEED_RUNS == {}


def test_the_contract_is_202_and_a_follow_route():
    from mantle.main import app
    paths = app.openapi()["paths"]
    assert "202" in paths["/system/seed"]["post"]["responses"]
    assert "get" in paths["/system/seed/{run_id}"]
