"""Review 33, at Mantle.

F2. `rekey_collections --rotate-again <root>` rotates a charged root whose key carries a valid
    mark, onto a fresh key and kcv: the operator's way past a split that stored Origin's share and
    failed its read-back.
F3. On the production plane the server refuses to start without its lock (elsewhere it logs), and
    step 21 holds the lock for the whole rotation, so a server cannot start beside it.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from mantle.search.mantle import oracle as o
from mantle.system import rekey_collections as rk

from . import test_review30_a_charged_root_s_key_is_rotated as r30
from . import test_review31_rotation as r31

ROOT = r30.ROOT
charged = r30.charged                     # the fixture review 30's own test uses


def test_rotate_again_ignores_the_mark(charged):
    db, kek, _ = charged
    first = rk.rotate_charged_roots(db, kek, [ROOT])[ROOT]
    again = rk.rotate_charged_roots(db, kek, [ROOT], again=[ROOT])[ROOT]
    assert "already" not in again and again["rotated"] == 1 and again["kcv"] != first["kcv"]
    assert rk.rotate_charged_roots(db, kek, [ROOT])[ROOT]["kcv"] == again["kcv"]   # `already`


def test_step_21_rotates_again_only_a_root_it_rotates(charged):
    db, kek, _ = charged
    first, _f = rk.step21(db, kek, r31._client({ROOT: "root"}, [ROOT]), mantle_stopped=True)
    out, failed = rk.step21(db, kek, r31._client({ROOT: "root"}, [ROOT]), mantle_stopped=True,
                            rotate_again=[ROOT, "not-a-charged-root"])
    assert failed and "rotate_again_unknown" in out["refused"]
    out, failed = rk.step21(db, kek, r31._client({ROOT: "root"}, [ROOT]), mantle_stopped=True,
                            rotate_again=[ROOT])
    assert not failed, out
    assert out["charged_roots"][ROOT]["kcv"] != first["charged_roots"][ROOT]["kcv"]


def _held_from_another_process(path) -> bool:
    code = ("import sys; from mantle.services import server_lock as s; "
            "sys.exit(0 if s.held_elsewhere(sys.argv[1]) else 3)")
    r = subprocess.run([sys.executable, "-c", code, path], capture_output=True, timeout=60)
    return r.returncode == 0


def test_step_21_holds_the_lock_for_the_whole_rotation(charged, tmp_path, monkeypatch):
    from mantle.services import server_lock

    db, kek, _ = charged
    path = str(tmp_path / "server.lock")
    monkeypatch.setattr(server_lock, "default_path", lambda: path)
    seen = {}
    real = rk.rotate_charged_roots

    def watching(*a, **k):
        seen["held"] = _held_from_another_process(path)
        return real(*a, **k)

    monkeypatch.setattr(rk, "rotate_charged_roots", watching)
    out, failed = rk.step21(db, kek, r31._client({ROOT: "root"}, [ROOT]), mantle_stopped=True)
    assert not failed, out
    assert seen["held"] is True                       # a server could not start meanwhile
    assert _held_from_another_process(path) is False  # and it is let go after


def test_the_server_refuses_to_start_unlocked_on_production(tmp_path, monkeypatch):
    from mantle.services import server_lock

    path = str(tmp_path / "server.lock")
    code = ("import sys, time; from mantle.services import server_lock as s\n"
            "s.hold(sys.argv[1]); print('held', flush=True); time.sleep(60)\n")
    other = subprocess.Popen([sys.executable, "-c", code, path], stdout=subprocess.PIPE, text=True)
    try:
        assert other.stdout.readline().strip() == "held"
        monkeypatch.setenv("AGIENCE_PLANE", "production")
        with pytest.raises(RuntimeError):
            server_lock.take_for_server(path)
        monkeypatch.setenv("AGIENCE_PLANE", "home")
        assert server_lock.take_for_server(path) is None           # logged, not refused
    finally:
        other.kill()
        other.wait()


def test_the_rotation_is_left_whole_when_the_lock_is_taken(charged, tmp_path, monkeypatch):
    from mantle.services import server_lock

    db, kek, _ = charged
    path = str(tmp_path / "server.lock")
    monkeypatch.setattr(server_lock, "default_path", lambda: path)
    code = ("import sys, time; from mantle.services import server_lock as s\n"
            "s.hold(sys.argv[1]); print('held', flush=True); time.sleep(60)\n")
    other = subprocess.Popen([sys.executable, "-c", code, path], stdout=subprocess.PIPE, text=True)
    try:
        assert other.stdout.readline().strip() == "held"
        out, failed = rk.step21(db, kek, r31._client({ROOT: "root"}, [ROOT]),
                                mantle_stopped=True)
        assert failed and "mantle_running" in out["refused"]
        assert o.LatticeMasterKeyStore(kek, lambda: db).get(ROOT) == r30.OLD
    finally:
        other.kill()
        other.wait()
