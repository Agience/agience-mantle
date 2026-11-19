"""A grant made at Origin is applied within one feed interval while the reach comparison with
Origin runs: the comparison digests every grant and edge, and an accepted invite must not wait on
it (BUG-119: the invitee's list read before the grant was applied)."""
from __future__ import annotations

import threading

from mantle.services import authority_ledger as al


def test_the_feed_applies_while_a_reach_check_is_still_running(monkeypatch):
    in_verify, release, applied = threading.Event(), threading.Event(), threading.Event()
    passes = []

    def slow_verify(db):
        in_verify.set()
        release.wait(5)

    def apply(db):
        passes.append(in_verify.is_set())
        if in_verify.is_set():
            applied.set()

    monkeypatch.setattr(al, "verify_with_origin", slow_verify)
    monkeypatch.setattr(al, "apply_remote", apply)
    stop = al.start_feed(lambda: None, interval_s=0.01, verify_interval_s=0.01)
    try:
        assert in_verify.wait(2)
        assert applied.wait(2), "no feed pass ran while the reach check was running"
    finally:
        release.set()
        stop.set()
