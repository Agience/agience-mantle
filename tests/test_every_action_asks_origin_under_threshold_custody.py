"""Under threshold custody every action, not only Read, needs Origin's verdict as well as this
store's grants (sixth review, M3): a store that drifted from what Origin co-signed decides
nothing on its own, including Invoke and the shared-credential use route, which authorizes through
`check_access`.

Origin is asked by its own walk (`GET /internal/authority/reach`), on the caller's own
credential. An allow is remembered per requester, resource and action for at most 30 seconds, the
canon's verdict memo; a refusal is not remembered. An Origin that cannot answer refuses.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from mantle.db import lattice_api as la
from mantle.db import reach_journal
from mantle.entities.grant import Grant
from mantle.services import authority_ledger
from mantle.services.acting_principal import ForwardedCredential, acting_as
from mantle.services.dependencies import AuthContext, check_access


class Origin:
    def __init__(self, allow=True, down=False):
        self.allow, self.down, self.asked = allow, down, []

    def submit_entry(self, entry, **_k):
        return {"seq": 1, "outcome": "appended", "entry_hash": "h" * 64, "prev_hash": "p" * 64,
                "signature": "s", "kid": "k"}

    def reach_verdict(self, resource_id, action, *, credential):
        self.asked.append((resource_id, action, credential))
        if self.down:
            raise RuntimeError("Origin is unreachable")
        return self.allow


@pytest.fixture
def db(tmp_path, monkeypatch):
    from .test_there_is_no_promotion_state import real_node
    database = real_node(tmp_path, monkeypatch)
    database.artifacts.put_artifact({"id": "C1", "root_id": "C1", "state": "live"})
    with reach_journal.suspended():
        la.create_grant(database, Grant(id="g-bob", resource_id="C1", grantee_type="user",
                                        grantee_id="bob", granted_by="alice", can_read=True,
                                        can_invoke=True))
        la.create_grant(database, Grant(id="g-al", resource_id="C1", grantee_type="user",
                                        grantee_id="al", granted_by="alice", can_invoke=True))
    authority_ledger.forget_verdicts()
    yield database
    authority_ledger.forget_verdicts()
    reach_journal.uninstall()


def _as(who, origin, db, action="invoke"):
    reach_journal.install(authority_ledger.Submitter(origin, "enforce"))
    with acting_as(who, principal_type="user", source="request",
                   credential=ForwardedCredential(f"{who}-token")):
        return check_access(AuthContext(principal_id=who, user_id=who), "C1", action, db)


def test_origin_refusing_refuses_an_action_the_store_allows(db, monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    origin = Origin(allow=False)
    with pytest.raises(HTTPException) as refused:
        _as("bob", origin, db)
    assert refused.value.status_code == 404
    assert origin.asked == [("C1", "invoke", "bob-token")]


def test_every_action_is_asked_not_only_read(db, monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    origin = Origin(allow=True)
    _as("bob", origin, db, "invoke")
    _as("bob", origin, db, "read")
    assert [a for _r, a, _c in origin.asked] == ["invoke", "read"]


def test_an_allow_is_remembered_per_requester_for_thirty_seconds(db, monkeypatch):
    import time

    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    origin = Origin(allow=True)
    _as("bob", origin, db)
    _as("bob", origin, db)
    assert len(origin.asked) == 1                              # memoized for bob
    _as("al", origin, db)
    assert len(origin.asked) == 2                              # per requester
    now = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: now + authority_ledger.VERDICT_TTL_S + 1)
    _as("bob", origin, db)
    assert len(origin.asked) == 3                              # asked again once it lapses
    assert authority_ledger.VERDICT_TTL_S <= 30


def test_a_refusal_is_not_remembered(db, monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    origin = Origin(allow=False)
    for _ in range(2):
        with pytest.raises(HTTPException):
            _as("bob", origin, db)
    assert len(origin.asked) == 2


def test_an_origin_that_cannot_answer_refuses(db, monkeypatch):
    monkeypatch.setenv("MANTLE_MASTER_KEY_CUSTODY", "threshold")
    with pytest.raises(HTTPException) as refused:
        _as("bob", Origin(down=True), db)
    assert refused.value.status_code == 404


def test_without_threshold_custody_origin_is_not_asked(db, monkeypatch):
    monkeypatch.delenv("MANTLE_MASTER_KEY_CUSTODY", raising=False)
    origin = Origin(allow=False)
    assert _as("bob", origin, db).id == "g-bob"
    assert origin.asked == []
