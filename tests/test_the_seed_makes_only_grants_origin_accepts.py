"""The seed route's operator provisioning submits only grants Origin accepts on a token Mantle holds:
a creator's own grant on what it creates, in the batch that creates it. Everything else a person
receives Origin makes itself. 202611.16.1 on the home test node: a standalone owner grant on the
operators' collection was refused ("a grant is made in the person's Origin session")."""
from __future__ import annotations

import pytest

from mantle.db import reach_journal
from mantle.services import authority_ledger
from mantle.services.acting_principal import ForwardedCredential, acting_as
from mantle.services.seed_provisioning import platform_operators as P

from .test_platform_operators_are_granted_artifacts import PLATFORM, db  # noqa: F401
from .test_reach_changes_go_to_origin import FakeOrigin


class HeldTokenOrigin(FakeOrigin):
    """Origin's rule for a grant on a token Mantle holds: only the actor's own grant on a resource
    whose root is in the same batch."""

    def __init__(self):
        super().__init__()
        self.roots, self.asked = {}, []

    def submit_entry(self, entry, *, credential, batch=None, **kw):
        from mantle.clients.origin_key_share_client import AuthorityRefused
        if batch is None and kw.get("compensates") is None:
            self.issued = getattr(self, "issued", 0) + 1
            batch = f"batch-{self.issued}"
        if entry.get("kind") == "root":
            self.roots[entry["root_id"]] = batch
        if entry.get("kind") == "grant" and kw.get("compensates") is None and not (
                entry["grantee_id"] == PLATFORM and self.roots.get(entry["resource_id"]) == batch):
            raise AuthorityRefused(403, "a grant is made in the person's Origin session")
        return super().submit_entry(entry, credential=credential, batch=batch, **kw)

    def new_person_grants(self, person_id):
        self.asked.append(person_id)
        return {"granted": 0, "seqs": []}


@pytest.fixture
def origin(monkeypatch):
    o = HeldTokenOrigin()
    reach_journal.install(authority_ledger.Submitter(o, "enforce"))
    monkeypatch.setattr(P, "apply_remote", lambda _db: 0, raising=False)
    yield o
    reach_journal.uninstall()


def test_the_operator_seeding_on_its_own_session_is_accepted(db, origin):  # noqa: F811
    with acting_as(PLATFORM, principal_type="user", source="request",
                   credential=ForwardedCredential("op-token")):
        assert P.ensure_platform_operators(db)
        assert P.ensure_platform_operators(db), "a second pass converges"
        P.grant_to_every_account(db)
    grants = [e for e in origin.entries if e.get("kind") == "grant"]
    assert grants and {g["grantee_id"] for g in grants} == {PLATFORM}
    assert {g["resource_id"] for g in grants} <= set(origin.roots)
