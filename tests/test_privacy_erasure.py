"""The privacy surface: erasure has a front door, identity has a delete, audit has a bound.

Three separate mechanisms, one property — a person's data can leave the store by an act somebody
can actually perform. Each of these existed as a correct primitive with nothing calling it, which
is indistinguishable from not having it: a right nobody can exercise is not a right.

What is asserted here, and why each assertion has a refusal beside it:

* **Dry run is the default.** `POST /system/erasure/{id}` with no `apply` deletes nothing. This
  is tested by asserting on the *deletes that did not happen*, not on the response body — a report
  saying `applied: false` proves what the handler claims, not what it did.
* **Admin-only.** The same endpoint refuses a non-admin, using the one platform-admin predicate
  rather than a second implementation that could disagree with it.
* **`delete_person` reaches the identity plane.** The people plane holds email, provider subject
  and password hash; those are not artifacts, so the erasure primitive cannot see them.
* **Retention defaults to unlimited.** The prune is a no-op with nothing configured, and only then
  does the age bound matter — a retention feature whose default deleted anything would be a data
  loss shipped as a privacy improvement.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.db import lattice_identity  # noqa: E402
from mantle.db import audit as lattice_audit  # noqa: E402
from mantle.db.lattice_api import LatticeDatabase  # noqa: E402
from mantle.routers.system_router import router  # noqa: E402
from mantle.services.dependencies import AuthContext, get_auth, get_store_db  # noqa: E402
from mantle.shard import erasure  # noqa: E402

ADMIN = "user-admin"
PERSON = "user-erased"


# ---------------------------------------------------------------------------
# POST /system/erasure/{person_id}
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch):
    """The endpoint with the platform-admin gate satisfied and the primitive observable.

    `require_platform_admin` is left REAL and its predicate stubbed, so the gate under test is
    the one the other platform endpoints use. Stubbing the dependency itself would test a route
    that no longer has a gate."""
    from mantle.services import dependencies

    monkeypatch.setattr(dependencies, "is_platform_admin", lambda *a, **k: True)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_auth] = lambda: AuthContext(
        principal_id=ADMIN, principal_type="user", user_id=ADMIN, grants=[])
    app.dependency_overrides[get_store_db] = lambda: MagicMock()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def seen(monkeypatch):
    """Capture the arguments the route hands the primitive, without running it."""
    calls: list = []

    def _fake_erase(store, person, *, apply=False, include_identity=False):
        calls.append({"person": person, "apply": apply, "include_identity": include_identity})
        return {"person": person, "ids": [person], "scanned": 3, "found": {}, "counts": {},
                "total": 0, "not_yours": [], "unresolved": [], "applied": apply}

    monkeypatch.setattr(erasure, "erase", _fake_erase)
    return calls


def test_erasure_is_a_dry_run_unless_apply_is_asked_for(client, seen):
    resp = client.post(f"/system/erasure/{PERSON}")
    assert resp.status_code == 200
    assert seen == [{"person": PERSON, "apply": False, "include_identity": False}]
    assert resp.json()["dry_run"] is True
    assert resp.json()["applied"] is False


def test_erasure_applies_only_on_an_explicit_true(client, seen):
    assert client.post(f"/system/erasure/{PERSON}?apply=true").status_code == 200
    assert seen[-1]["apply"] is True
    # …and the falsy spellings are not a way in.
    client.post(f"/system/erasure/{PERSON}?apply=false")
    assert seen[-1]["apply"] is False


def test_erasure_reports_who_asked(client, seen):
    assert client.post(f"/system/erasure/{PERSON}").json()["requested_by"] == ADMIN


def test_erasure_refuses_a_non_admin(monkeypatch, seen):
    """The gate is the platform-admin predicate — not authentication alone."""
    from mantle.services import dependencies

    monkeypatch.setattr(dependencies, "is_platform_admin", lambda *a, **k: False)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_auth] = lambda: AuthContext(
        principal_id="user-nobody", principal_type="user", user_id="user-nobody", grants=[])
    app.dependency_overrides[get_store_db] = lambda: MagicMock()
    with TestClient(app) as c:
        assert c.post(f"/system/erasure/{PERSON}").status_code == 403
    assert seen == []          # the primitive was never reached


def test_full_erasure_also_drops_the_identity_record(client, seen, monkeypatch):
    """`include_identity` + `apply` has to reach the plane the primitive cannot see."""
    from mantle.db import identity_backend

    removed: list = []
    monkeypatch.setattr(identity_backend, "delete_person",
                        lambda db, pid: bool(removed.append(pid)) or True)

    body = client.post(
        f"/system/erasure/{PERSON}?apply=true&include_identity=true").json()

    assert removed == [PERSON]
    assert body["identity_record_removed"] is True


def test_a_dry_run_never_touches_the_identity_record(client, seen, monkeypatch):
    from mantle.db import identity_backend

    monkeypatch.setattr(identity_backend, "delete_person",
                        lambda db, pid: pytest.fail("dry run deleted an identity record"))
    body = client.post(f"/system/erasure/{PERSON}?include_identity=true").json()
    assert "identity_record_removed" not in body


# ---------------------------------------------------------------------------
# The primitive now has callers — the finding was that it had none
# ---------------------------------------------------------------------------

def test_erasure_has_a_front_door():
    """A guard against the primitive drifting back to zero callers.

    Named endpoint and named script, because "something imports it" would still pass with both
    front doors deleted and one incidental test import left behind."""
    import inspect

    from mantle.routers import system_router
    from mantle.scripts import manage_erasure

    assert "erasure" in inspect.getsource(system_router)
    assert "/erasure/{person_id}" in inspect.getsource(system_router)
    assert "erasure.erase" in inspect.getsource(manage_erasure)


def test_the_script_defaults_to_reporting():
    """`--apply` is the only thing that turns the CLI into a delete."""
    from mantle.scripts import manage_erasure

    parser = manage_erasure.build_parser()
    assert parser.parse_args(["--person", "p-1"]).apply is False
    assert parser.parse_args(["--person", "p-1", "--dry-run"]).apply is False
    assert parser.parse_args(["--person", "p-1", "--apply"]).apply is True
    assert parser.parse_args(["--person", "p-1"]).include_identity is False


def test_the_script_refuses_to_be_told_both(capsys):
    """`--apply --dry-run` is a contradiction, and an ambiguous erasure resolves to neither."""
    from mantle.scripts import manage_erasure

    with pytest.raises(SystemExit):
        manage_erasure.build_parser().parse_args(["--person", "p-1", "--apply", "--dry-run"])


@pytest.fixture()
def db(tmp_path):
    return LatticeDatabase(str(tmp_path / "identity.db"), origin="test-node")


# ---------------------------------------------------------------------------
# The primitive over a real store — positively defined, and it stops where it should
# ---------------------------------------------------------------------------

def test_erasure_takes_what_is_grounded_and_leaves_the_rest(db):
    """Over real rows: the private collection and authored work go; the commons and other
    people's work stay, and the registry artifact this person touched is reported rather than
    silently skipped. Erasure defined by exclusion would take one of the last three."""
    db.artifacts.put_artifact({"id": "mine-private", "content_type": "text/plain",
                               "collection_id": "private.%s" % PERSON})
    db.artifacts.put_artifact({"id": "mine-authored", "content_type": "text/plain",
                               "created_by": PERSON})
    db.artifacts.put_artifact({"id": "the-registry", "created_by": PERSON,
                               "content_type": "application/vnd.agience.operator+json"})
    db.artifacts.put_artifact({"id": "someone-elses", "content_type": "text/plain",
                               "created_by": "user-other"})

    inventory = erasure.attached(db, PERSON)
    assert inventory["counts"]["private"] == 1
    assert inventory["counts"]["authored"] == 1
    assert inventory["total"] == 2
    assert inventory["not_yours"] == ["the-registry"]

    assert erasure.erase(db, PERSON)["applied"] is False
    assert len(list(db.artifacts.list_artifacts())) == 4      # the dry run changed nothing

    report = erasure.erase(db, PERSON, apply=True)
    assert report["removed"] == 2 and report["complete"] is True
    assert sorted(r["id"] for r in db.artifacts.list_artifacts()) == ["someone-elses", "the-registry"]


# ---------------------------------------------------------------------------
# db.lattice_identity.delete_person — the people plane gets its siblings' delete
# ---------------------------------------------------------------------------


def test_delete_person_removes_the_record(db):
    pid = lattice_identity.create_person(db, {"id": "p-1", "email": "jane@x.com",
                                              "oidc_subject": "sub-1"})
    assert pid == "p-1"
    assert lattice_identity.get_person_by_email(db, "jane@x.com") is not None

    assert lattice_identity.delete_person(db, "p-1") is True

    assert lattice_identity.get_person_by_id(db, "p-1") is None
    assert lattice_identity.get_person_by_email(db, "jane@x.com") is None
    assert lattice_identity.count_people(db) == 0


def test_delete_person_reports_a_miss_like_its_siblings(db):
    """False, not an exception — the same contract as settings / passkeys / OTP."""
    assert lattice_identity.delete_person(db, "never-existed") is False


def test_delete_person_is_reachable_through_the_identity_backend():
    """Call sites import the backend module, never the implementation."""
    from mantle.db import identity_backend

    assert identity_backend.delete_person is lattice_identity.delete_person


def test_create_person_error_does_not_log_the_email(db, caplog, monkeypatch):
    """The failure path logs the id. A log line is the one copy that leaves the store."""
    import logging

    monkeypatch.setattr(lattice_identity._PEOPLE, "put",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.ERROR):
        assert lattice_identity.create_person(db, {"id": "p-9", "email": "jane@x.com"}) is None
    assert "jane@x.com" not in caplog.text
    assert "p-9" in caplog.text


# ---------------------------------------------------------------------------
# Audit retention — unlimited by default, bounded when asked
# ---------------------------------------------------------------------------

def _append(db, artifact_id: str, *, ts: str) -> None:
    lattice_audit.append_access_events(db.conn, [{
        "principal_id": "p-1", "artifact_id": artifact_id, "action": "read",
        "result": "allowed", "ts": ts, "ctx": {}}])


def _count(db) -> int:
    return db.conn.read().execute("SELECT COUNT(*) AS n FROM access_event").fetchone()["n"]


def test_retention_is_unlimited_by_default(db, monkeypatch):
    monkeypatch.delenv("MANTLE_AUDIT_RETENTION_DAYS", raising=False)
    old = (datetime.now(timezone.utc) - timedelta(days=4000)).isoformat()
    _append(db, "a-1", ts=old)

    assert lattice_audit.retention_days() == 0
    assert lattice_audit.retention_cutoff() is None
    assert lattice_audit.prune_access_events(db.conn) == 0
    assert _count(db) == 1                      # nothing was deleted


def test_retention_prunes_only_what_is_older_than_the_bound(db, monkeypatch):
    monkeypatch.setenv("MANTLE_AUDIT_RETENTION_DAYS", "30")
    now = datetime.now(timezone.utc)
    _append(db, "a-old", ts=(now - timedelta(days=90)).isoformat())
    _append(db, "a-new", ts=(now - timedelta(days=1)).isoformat())

    assert lattice_audit.prune_access_events(db.conn) == 1
    assert _count(db) == 1
    assert lattice_audit.access_log_of(db.conn, "a-new")      # the recent one survives
    assert lattice_audit.access_log_of(db.conn, "a-old") == []


def test_an_unparseable_retention_setting_retains_rather_than_guesses(db, monkeypatch):
    """Deleting an audit trail because a setting was mistyped is the worse failure."""
    monkeypatch.setenv("MANTLE_AUDIT_RETENTION_DAYS", "thirty")
    _append(db, "a-1", ts=(datetime.now(timezone.utc) - timedelta(days=9999)).isoformat())
    assert lattice_audit.retention_days() == 0
    assert lattice_audit.prune_access_events(db.conn) == 0
    assert _count(db) == 1


def test_prune_is_idempotent(db, monkeypatch):
    monkeypatch.setenv("MANTLE_AUDIT_RETENTION_DAYS", "1")
    _append(db, "a-1", ts=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat())
    assert lattice_audit.prune_access_events(db.conn) == 1
    assert lattice_audit.prune_access_events(db.conn) == 0
