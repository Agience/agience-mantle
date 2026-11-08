"""Review 32, at Mantle.

F3. A re-run of step 21 over a rotated charged root is `already`: the key keeps its kcv, which the
    operator recorded from the run that rotated it. A mark is taken for `already` only while no
    removed part's row opens under the current key. A split charged root is not `left`.
F4. The split refuses a root whose key has a removed part's rows under it (Mantle's own off-root
    test) and no rotation mark, though Origin charges none.
F2. A split whose activation Origin never heard counts as failed, and the split exits non-zero.
(v) Step 21 refuses while Mantle's server holds its lock, whatever `--mantle-stopped` says.
"""
from __future__ import annotations

import pytest

from mantle.search.mantle import oracle as o
from mantle.system import rekey_collections as rk
from mantle.system import split_master_keys as smk

from . import test_review30_a_charged_root_s_key_is_rotated as r30
from . import test_review31_rotation as r31

ROOT, OLD = r30.ROOT, r30.OLD
charged = r30.charged                     # the fixture review 30's own test uses


def _store(kek, db):
    return o.LatticeMasterKeyStore(kek, lambda: db)


def test_a_rerun_over_a_rotated_root_is_already(charged):
    db, kek, _ = charged
    first = rk.rotate_charged_roots(db, kek, [ROOT])[ROOT]
    k1 = _store(kek, db).get(ROOT)
    second = rk.rotate_charged_roots(db, kek, [ROOT])[ROOT]
    assert second.get("already") is True and second["kcv"] == first["kcv"]
    assert _store(kek, db).get(ROOT) == k1


def test_a_mark_over_a_key_that_still_opens_a_removed_part_is_no_rotation(charged):
    db, kek, _ = charged
    # A mark for the OLD key itself: the removed part's row still opens under it.
    db.artifacts.put_artifact(rk.rotated_key_row(kek, ROOT, OLD))
    report = rk.rotate_charged_roots(db, kek, [ROOT])[ROOT]
    assert "already" not in report and report["rotated"] == 1
    assert _store(kek, db).get(ROOT) != OLD


def test_a_split_charged_root_is_not_left_at_step_21(charged):
    db, kek, _ = charged
    doc = db.artifacts.get_artifact("master-key:" + ROOT)
    db.artifacts.put_artifact({"id": doc["id"], "content_type": o.SHARE_CONTENT_TYPE,
                               "state": "active", "share": "x", "kcv": "k"})
    out, failed = rk.step21(db, kek, r31._client({ROOT: "root"}, [ROOT]), mantle_stopped=True)
    assert not failed, out
    assert out["charged_roots"][ROOT]["left"].startswith("split")


@pytest.fixture()
def env(charged, tmp_path):
    """The charged store, with the split's stand-ins for Origin's share routes and the escrow."""
    from .test_a_split_master_key_needs_two_custodians import make_env

    db, kek, _ = charged
    (tmp_path / "e").mkdir()
    e = make_env(tmp_path / "e")
    e.db, e.kek = db, kek
    e.origin.charged_roots = lambda ids: []              # Origin charges no root here
    e.origin.principal_kinds = lambda ids: {i: "root" for i in ids}
    return e


def _split(e):
    from mantle.services.acting_principal import acting_as
    with acting_as("system", source="system", scope="platform.custody"):
        return smk.split_master_keys(e.db, e.kek, e.origin, e.recipient, e.escrow, e.lock)


def test_the_split_refuses_a_root_with_a_removed_part_and_no_mark(env):
    counts = _split(env)
    assert counts["split"] == 0 and counts["refused_unrotated"] == 1
    rk.rotate_charged_roots(env.db, env.kek, [ROOT])
    counts = _split(env)
    assert counts["split"] == 1 and counts["refused_unrotated"] == 0


def test_a_split_whose_activation_failed_is_failed(env):
    rk.rotate_charged_roots(env.db, env.kek, [ROOT])

    def refuse(*a, **k):
        raise RuntimeError("Origin did not take the activation")

    env.origin.activate_share = refuse
    counts = _split(env)
    assert counts["split"] == 0 and counts["activation_failed"] == 1 and counts["failed"] == 1
    assert smk.gate_exit({"whole_non_person_master_keys": 0}, counts) == 1


def test_step_21_refuses_while_the_server_holds_its_lock(charged, tmp_path, monkeypatch):
    from mantle.services import server_lock

    db, kek, _ = charged
    path = str(tmp_path / "server.lock")
    monkeypatch.setattr(server_lock, "default_path", lambda: path)
    held = server_lock.hold(path)
    try:
        assert server_lock.held_elsewhere(path) is False   # this process holds it itself
        monkeypatch.setattr(server_lock, "_HELD", {})          # as if another process held it

        def taken(p=None):
            raise RuntimeError("another Mantle server holds it")

        monkeypatch.setattr(server_lock, "hold", taken)
        out, failed = rk.step21(db, kek, r31._client({ROOT: "root"}, [ROOT]),
                                mantle_stopped=True)
        assert failed and "mantle_running" in out["refused"]
        assert _store(kek, db).get(ROOT) == OLD
    finally:
        server_lock.release(held)


def test_the_server_lock_is_seen_from_another_process(tmp_path):
    import subprocess
    import sys

    from mantle.services import server_lock

    path = str(tmp_path / "server.lock")
    assert server_lock.held_elsewhere(path) is False
    held = server_lock.hold(path)
    try:
        code = ("import sys; from mantle.services import server_lock as s; "
                "sys.exit(0 if s.held_elsewhere(sys.argv[1]) else 3)")
        r = subprocess.run([sys.executable, "-c", code, path], capture_output=True, timeout=60)
        assert r.returncode == 0, r.stderr
    finally:
        server_lock.release(held)
    assert server_lock.held_elsewhere(path) is False
