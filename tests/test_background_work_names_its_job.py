"""Background work names its job, and Origin mints a delegation only under a job grant for it
(John, 2026-11-02).

Mantle asks Origin for a background delegation for one principal and one named job: the scope the
acting block declared (`system_acting_context(scope=...)`, `acting_as(..., scope=...)`), sent in
the body and bound in Mantle's service token. A block that names no job is refused here, before
Origin is asked: what no one granted stays shut.
"""
from __future__ import annotations

import base64
import json
import time
import uuid

import httpx
import pytest

from mantle.services.acting_principal import acting_as


def _client(monkeypatch, seen):
    from mantle.clients import origin_key_share_client as oc

    monkeypatch.setattr(oc.peer_signing, "sign_service_jwt",
                        lambda **kw: "svc:" + json.dumps(kw["additional_claims"], sort_keys=True))

    def tok(ttl):
        seg = base64.urlsafe_b64encode(json.dumps({"exp": time.time() + ttl}).encode()).decode()
        return f"h.{seg.rstrip('=')}.s"

    def handler(req):
        seen.append((req.url.path, json.loads(req.content or b"{}"),
                     req.headers["authorization"]))
        return httpx.Response(200, json={"token": tok(300), "jti": uuid.uuid4().hex})

    return oc.OriginShareCustodian("http://origin.test",
                                   client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_the_job_is_sent_and_bound_and_each_job_has_its_own_delegation(monkeypatch):
    seen = []
    c = _client(monkeypatch, seen)
    a = c.background_delegation("p-1", "platform.reindex")
    assert c.background_delegation("p-1", "platform.reindex") == a       # kept, per job
    c.background_delegation("p-1", "platform.rekey")
    assert [(p, b["job"]) for p, b, _a in seen] == [
        ("/internal/custody-delegations/p-1", "platform.reindex"),
        ("/internal/custody-delegations/p-1", "platform.rekey")]
    assert all('"job": "' in auth for _p, _b, auth in seen)               # bound in the token


def test_a_block_that_names_no_job_is_refused_before_origin_is_asked(monkeypatch):
    from mantle.search.mantle import threshold_custody as tc

    seen = []
    c = _client(monkeypatch, seen)
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    with acting_as("sys-1", principal_type="user", source="system"):
        with pytest.raises(tc.MasterKeyUnavailable, match="names its job"):
            tc.acting_credential("p-1", c)
    assert seen == []
    with acting_as("sys-1", principal_type="user", source="system", scope="platform.seal"):
        tc.acting_credential("p-1", c)
    assert [b["job"] for _p, b, _a in seen] == ["platform.seal"]


def test_the_ledger_s_retried_withdrawal_names_its_job(monkeypatch):
    from mantle.db import reach_journal
    from mantle.services import authority_ledger

    jobs = []

    class Custodian:
        def background_delegation(self, principal_id, job):
            jobs.append(job)
            return f"background:{principal_id}"

        def submit_entry(self, entry, *, credential, proof=None, compensates=None, batch=None):
            return {"outcome": "appended", "seq": 9, "entry_hash": "h", "prev_hash": "p",
                    "signature": "s", "kid": "k"}

    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-1")
    sub = authority_ledger.Submitter(Custodian(), "enforce")
    reach_journal.install(sub)
    try:
        sub.retry({"kind": "edge.remove", "src": "a", "dst": "b"}, 3)
    finally:
        reach_journal.uninstall()
    assert jobs == ["ledger.retry"]
