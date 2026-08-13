"""Tests for the access-audit 'force' (services/audit_service.py).

Lattice mode: flush writes through `db/audit.py` into the `access_event`
TABLE via `db.conn` — exercised against a REAL temp lattice store.
"""
from unittest.mock import MagicMock, patch

import mantle.services.audit_service as audit


def _open_store(tmp_path):
    from mantle.db import lattice_api as api
    return api.open_database(str(tmp_path / "t.db"), origin="test-audit")


def setup_function(_):
    audit._reset_for_tests()


def test_record_buffers_and_flush_inserts(tmp_path):
    for i in range(5):
        audit.record_access(principal_id=f"u{i}", artifact_id="art-1",
                            action="read", result="allowed", context={"via": "user"})
    assert audit.pending() == 5

    db = _open_store(tmp_path)
    n = audit.flush_once(db)

    assert n == 5
    assert audit.pending() == 0
    # The events landed in the access_event table with their fields intact
    # (the store-era `_from`/`_to` edge keys are gone; principal_id/artifact_id
    # are the persisted identities).
    rows = audit.get_artifact_access_log(db, "art-1", limit=10)
    assert len(rows) == 5
    assert {r["principal_id"] for r in rows} == {f"u{i}" for i in range(5)}
    e = rows[0]
    assert e["result"] == "allowed" and e["action"] == "read"
    assert "ts" in e
    assert e["ctx"] == {"via": "user"}


def test_flush_empty_is_noop():
    """An idle tick writes nothing — asserted at the real seam.

    The seam is `mantle.db.audit.append_access_events`, the one function `flush_once`
    reaches the store through. It is asserted to EXIST first: a bare `MagicMock` answers to any
    attribute name, so `mock.whatever.assert_not_called()` passes for a method that was renamed,
    or never existed — which is what this test used to do (`db.collection(...).insert_many`, a
    Mongo-era name absent from the whole of `src/mantle`).
    """
    import mantle.db.audit as _la

    assert callable(getattr(_la, "append_access_events", None)), (
        "the audit write seam was renamed or removed — this test is now asserting against a name "
        "nothing calls, which is a pass that means nothing")

    with patch.object(_la, "append_access_events", autospec=True) as append:
        written = audit.flush_once(MagicMock())
    append.assert_not_called()
    assert written == 0


def test_record_never_raises_on_bad_input():
    # Missing artifact_id is dropped, not raised.
    audit.record_access(principal_id=None, artifact_id="", action="read", result="allowed")
    assert audit.pending() == 0
    # None principal is allowed (anonymous witness).
    audit.record_access(principal_id=None, artifact_id="a", action="read", result="denied")
    assert audit.pending() == 1


def test_backpressure_drops_oldest(monkeypatch):
    monkeypatch.setattr(audit, "_MAX_BUFFER", 3)
    for i in range(5):
        audit.record_access(principal_id=f"u{i}", artifact_id="a", action="read", result="allowed")
    # Bounded to the cap; oldest dropped.
    assert audit.pending() == 3
    remaining = audit._drain(10)
    principals = [e["principal_id"] for e in remaining]
    assert principals == ["u2", "u3", "u4"]  # u0,u1 dropped


def test_denied_result_is_recorded_with_reason(tmp_path):
    audit.record_access(principal_id="u1", artifact_id="a", action="read",
                        result="denied", context={"reason": "no_grant"})
    db = _open_store(tmp_path)
    audit.flush_once(db)
    doc = audit.get_artifact_access_log(db, "a", limit=10)[0]
    assert doc["result"] == "denied"
    assert doc["ctx"]["reason"] == "no_grant"


def test_get_artifact_access_log_filters_by_artifact_and_result(tmp_path):
    # The read is scoped to ONE artifact and (optionally) one result value.
    db = _open_store(tmp_path)
    audit.record_access(principal_id="u1", artifact_id="art-9", action="read",
                        result="denied", context={})
    audit.record_access(principal_id="u2", artifact_id="art-9", action="read",
                        result="allowed", context={})
    audit.record_access(principal_id="u3", artifact_id="other", action="read",
                        result="denied", context={})
    audit.flush_once(db)

    out = audit.get_artifact_access_log(db, "art-9", limit=10, result="denied")
    assert [(e["principal_id"], e["result"]) for e in out] == [("u1", "denied")]
    # Unfiltered read still scopes to the artifact.
    assert {e["principal_id"] for e in audit.get_artifact_access_log(db, "art-9", limit=10)} \
        == {"u1", "u2"}


def test_a_failed_flush_is_counted_not_just_logged(tmp_path, monkeypatch):
    """`flush_once` drains the buffer, then inserts. On failure it still returns 0, the same
    value it returns when there was nothing to flush, so a caller cannot tell an idle tick from
    one that lost events by return value alone.

    At shutdown `drain_and_stop` stops on 0, so an unflagged loss would also cut the drain short.
    `stats()["lost_total"]` is what makes a loss visible when the return value cannot.
    """
    for i in range(3):
        audit.record_access(principal_id=f"u{i}", artifact_id="art-1",
                            action="read", result="allowed", context={"via": "user"})
    assert audit.pending() == 3
    assert audit.stats()["lost_total"] == 0

    db = _open_store(tmp_path)

    import mantle.db.audit as _la

    def _boom(*a, **kw):
        raise RuntimeError("disk gone")

    monkeypatch.setattr(_la, "append_access_events", _boom)

    assert audit.flush_once(db) == 0            # indistinguishable from idle, by design of the loop
    assert audit.pending() == 0                 # ...and the events are gone from the buffer
    assert audit.stats()["lost_total"] == 3     # which is exactly why the loss must be counted

    # an idle flush must not inflate the counter — otherwise the number means nothing
    assert audit.flush_once(db) == 0
    assert audit.stats()["lost_total"] == 3
