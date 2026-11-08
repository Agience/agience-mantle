"""Review 29 at Mantle.

L1. Mantle's real Origin client passed `pending` on from no release, so a pending share was always
completed, never replaced. It now passes it, failing towards a fresh split when Origin says
nothing; the split is shown through the real client over a mock transport.

L3. The owner-call re-key acts as the credential's own subject, not as the key's principal, and
remembers refusals by that subject.

L4. An emptied master key is removed only when Origin answered `root` or `platform` for it; with
the explicit `member_keys` remedy (step 21's `--retire-member-keys`) one Origin answered `member`
or `unknown` for too (L5).

I3. `exposure_counts` counts rows under a master key that is not their root's.

I4. The planner's copy is zeroed with its index folders.

The invite outbox names its files in sending order, whatever the clock's resolution.
"""
from __future__ import annotations

import base64
import json
import uuid

import httpx
import pytest
from origin.services.authority_reach import MemoryReplica, Reach

from mantle.clients.origin_key_share_client import OriginShareCustodian
from mantle.db import doc_boundary as b
from mantle.db import lattice_api as la
from mantle.search.mantle import oracle as o
from mantle.search.mantle import threshold_custody as tc
from mantle.services import owner_rekey
from mantle.services.acting_principal import ForwardedCredential, acting_as
from mantle.system import split_master_keys as smk

from .origin_rule import replica_grant
from .test_a_split_master_key_needs_two_custodians import ALICE_KEY, make_env

ALICE = str(uuid.uuid4())


@pytest.fixture(autouse=True)
def _no_remembered_refusals():
    owner_rekey.forget_refusals()
    yield
    owner_rekey.forget_refusals()


def _as(who, token, **kw):
    return acting_as(who, principal_type="user", source="request",
                     credential=ForwardedCredential(token), **kw)


# -- L1: the real client, over a mock transport -------------------------------------------------

def test_the_real_client_passes_pending_on():
    def answer(request):
        return httpx.Response(200, json={"principal_id": "ck:C", "x": 2, "y": "AAAA",
                                         "kcv": "k", "pending": True})

    c = OriginShareCustodian("http://origin.invalid",
                             client=httpx.Client(transport=httpx.MockTransport(answer)))
    c._headers = lambda *a, **k: {}
    assert c.get_share("ck:C", credential="opaque")["pending"] is True


def test_a_release_that_says_nothing_of_pending_reads_as_pending():
    def answer(request):
        return httpx.Response(200, json={"principal_id": "ck:C", "x": 2, "y": "AAAA",
                                         "kcv": "k"})

    c = OriginShareCustodian("http://origin.invalid",
                             client=httpx.Client(transport=httpx.MockTransport(answer)))
    c._headers = lambda *a, **k: {}
    assert c.get_share("ck:C", credential="opaque")["pending"] is True


class _MockOrigin:
    """Origin's key-share routes behind a mock transport, deciding with Origin's own `Reach`."""

    def __init__(self, replica, tokens):
        self.replica, self.tokens = replica, tokens
        self.shares, self.pending, self.stored_by = {}, set(), {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        pid = request.url.path.rsplit("/", 1)[-1]
        if request.url.path.endswith("/activate"):
            pid = request.url.path.split("/")[-2]
            self.pending.discard(pid)
            return httpx.Response(200, json={"principal_id": pid, "rekey_pending": False})
        subject = self.tokens.get(request.headers.get("x-acting-credential", ""))
        if subject is None:
            return httpx.Response(403, json={"detail": "acting credential refused"})
        rc = Reach(self.replica)
        if request.method == "GET":
            ok, why = rc.may_release(subject=subject, principal_id=pid,
                                     collection_id=request.url.params.get("collection_id"))
            if not ok:
                return httpx.Response(403, json={"detail": why})
            if pid not in self.shares:
                return httpx.Response(404, json={"absent": True, "principal_id": pid})
            return httpx.Response(200, json={"principal_id": pid, **self.shares[pid],
                                             "pending": pid in self.pending})
        body = json.loads(request.content)
        share = {k: body[k] for k in ("x", "y", "kcv")}
        old = self.shares.get(pid)
        if old is None:
            self.stored_by[pid] = subject
            self.pending.add(pid)
        elif old != share and not (pid in self.pending and self.stored_by.get(pid) == subject):
            return httpx.Response(403, json={"detail": "not replaced"})
        self.shares[pid] = share
        return httpx.Response(200, json={"principal_id": pid,
                                         "outcome": "replaced" if old else "stored"})


def test_a_pending_share_is_replaced_through_the_real_client(tmp_path):
    e = make_env(tmp_path)
    kid = o.collection_key_id("c-team")
    o.LatticeMasterKeyStore(e.kek, lambda: e.db).put(kid, ALICE_KEY)
    mock = _MockOrigin(MemoryReplica(
        grants={"g-a": replica_grant("g-a", ALICE, "c-team", ("read",))},
        roots={"c-team": {"root_id": "c-team"}}, nodes={"c-team": {}},
        persons={ALICE: ("a@x", True)}), {"alice-token": ALICE})
    client = OriginShareCustodian("http://origin.invalid",
                                  client=httpx.Client(transport=httpx.MockTransport(mock)))
    client._headers = lambda *a, **k: {}
    client._token = lambda *a, **k: "t"
    e.store._custodian = client
    e.escrow.down = True
    with _as(ALICE, "alice-token"):
        with pytest.raises(tc.MasterKeyUnavailable):
            e.store.get_scoped(kid, collection_id="c-team")
    planted = dict(mock.shares[kid])
    e.escrow.down = False
    with _as(ALICE, "alice-token"):
        assert e.store.get_scoped(kid, collection_id="c-team") == ALICE_KEY
    assert mock.shares[kid]["y"] != planted["y"]


# -- L3: the re-key acts as the credential's own subject -----------------------------------------

def test_the_owner_re_key_acts_as_the_credential_s_subject(tmp_path, monkeypatch):
    from mantle.services.acting_principal import current_acting_principal

    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    seen = []
    monkeypatch.setattr(owner_rekey, "rekey_owner",
                        lambda *_a, **_k: seen.append(current_acting_principal().principal_id))
    monkeypatch.setattr(owner_rekey, "_start", lambda fn: fn())
    root = str(uuid.uuid4())
    with _as(ALICE, "alice-token"):
        owner_rekey.schedule(lambda: db, root, "alice-token")
    assert seen == [ALICE]


# -- L4 and L5: which emptied keys are removed ----------------------------------------------------

@pytest.mark.parametrize("kind, removed, with_remedy", [
    ("root", True, True), ("platform", True, True),
    ("member", False, True), ("unknown", False, True),
])
def test_an_emptied_key_is_removed_only_for_a_root_or_platform_answer(tmp_path, kind, removed,
                                                                      with_remedy):
    e = make_env(tmp_path)
    o.LatticeMasterKeyStore(e.kek, lambda: e.db).put("p-1", ALICE_KEY)
    assert bool(smk.retire_empty_master_keys(e.db, kinds={"p-1": kind})["removed"]) is removed
    if not removed and with_remedy:
        done = smk.retire_empty_master_keys(e.db, kinds={"p-1": kind}, member_keys=True)
        assert done["removed"] == ["p-1"]


# -- I3: rows under a master key that is not their root's ----------------------------------------

def test_rows_under_another_principal_s_key_are_counted(tmp_path):
    db = la.LatticeDatabase(str(tmp_path / "k.db"), origin="k")
    team, bob = str(uuid.uuid4()), str(uuid.uuid4())
    db.artifacts.put_artifact({"id": team, "content_type": "application/vnd.agience.collection+json",
                               "origin_root": team, "root_id": team})
    la.add_artifact_to_collection(db, team, "c-docs", origin=True)
    from .test_review28_custody import _mec1
    db.artifacts.put_artifact({
        "id": "b1", "collection_id": "c-docs", "origin_root": team, b.CONTENT_KEY_PRINCIPAL: bob,
        "created_by": bob, "state": "live", "content_encrypted": True,
        "content": base64.b64encode(_mec1(bob, bytes(32), "c-docs", b"x")).decode()})
    la.add_artifact_to_collection(db, "c-docs", "b1", origin=True)
    counts = smk.exposure_counts(db, kinds={})
    assert counts["rows_keyed_off_their_root"] == 1


# -- I4 -------------------------------------------------------------------------------------------

def test_the_planner_zeroes_its_index_folders_too(tmp_path):
    from mantle.system import seed_plan as sp

    (tmp_path / "sse").mkdir()
    (tmp_path / "cells" / "p").mkdir(parents=True)
    (tmp_path / "sse" / "postings.db").write_bytes(b"tokens" * 40)
    (tmp_path / "cells" / "p" / "c1").write_bytes(b"cell" * 40)
    seen = {}
    sp._scrub_copy(str(tmp_path), observe=lambda p, data: seen.__setitem__(p, data))
    assert len(seen) == 2 and all(set(d) <= {0} for d in seen.values())


# -- the invite outbox's order --------------------------------------------------------------------

def test_the_outbox_names_mail_in_sending_order(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from mantle.services import email_service as es

    monkeypatch.setenv("EMAIL_OUTBOX_DIR", str(tmp_path))
    fixed = datetime(2026, 11, 8, tzinfo=timezone.utc)

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed                              # a coarse clock: one tick for both

    hexes = iter(["ffffffff", "00000000"])           # the later mail draws the lower suffix
    monkeypatch.setattr(es, "datetime", _Clock)
    monkeypatch.setattr(es.secrets, "token_hex", lambda n: next(hexes))
    es._send_outbox("a@x", "first", "", "first")
    es._send_outbox("a@x", "second", "", "second")
    names = sorted(p.name for p in tmp_path.glob("*.json"))
    assert [json.loads((tmp_path / n).read_text())["subject"] for n in names] == \
        ["first", "second"]
