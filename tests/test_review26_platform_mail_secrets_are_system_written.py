"""Review 26, M1: the platform mail secrets are provisioned where Origin counts them as the system's.

`platform_email._provision` wrote the client secret and the refresh token as roots: no container,
no creation edge, owned by the operator, with the system principal holding Invoke and the operator
Read, Invoke and Admin but no Update. Origin's `use_verdict.system_written` then saw neither a
creation edge the system added nor a bootstrap-stated one under a collection the system
administers, so the seed's record of their seals was refused, and so was the operator's (no
Update). Cutover step 20 could not complete.

Now both credentials are members of a platform collection the system principal administers, by
their creation edge (added here before the cutover, so the bootstrap states it with no adder), and
the operator holds Update on them.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mantle.services.seed_provisioning import platform_email as pe

from .test_platform_email_provisioner import _OPERATOR, gmail_env, wired  # noqa: F401

SYSTEM = "sys-principal-1"
COLL = "id-platform-email-credentials"
SECRETS = ("id-platform-email-client-secret", "id-platform-email-refresh-token")


@pytest.fixture()
def provisioned(gmail_env, wired, monkeypatch):  # noqa: F811
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: SYSTEM)
    pe.ensure_platform_email_sender(db=MagicMock())
    return wired


def _grants(w, grantee, resource):
    return [c.kwargs for c in w.upsert_grant.call_args_list
            if c.kwargs["user_id"] == grantee and c.kwargs["collection_id"] == resource]


def test_the_secrets_are_members_of_a_platform_collection(provisioned):
    created = {c.args[1].id: c.args[1] for c in provisioned.create_artifact.call_args_list}
    assert created[COLL].content_type == pe.COLLECTION_CT
    edges = {(c.args[1], c.args[2]): c.kwargs for c in provisioned.add_edge.call_args_list}
    for secret in SECRETS:
        assert (COLL, secret) in edges, f"{secret} has no edge from the platform collection"
        # The creation edge: a root with none takes this one (`origin` left to the store).
        assert edges[(COLL, secret)].get("origin") is not False
        assert not edges[(COLL, secret)].get("relationship")


def test_the_system_principal_administers_the_collection_and_reads_no_secret(provisioned):
    on_coll = _grants(provisioned, SYSTEM, COLL)
    assert on_coll and on_coll[0]["can_admin"]
    for secret in SECRETS:
        g = _grants(provisioned, SYSTEM, secret)[0]
        assert not g["can_read"] and not g["can_admin"]


def test_the_operator_may_update_the_secrets(provisioned):
    for secret in SECRETS:
        assert _grants(provisioned, _OPERATOR, secret)[0]["can_update"]


def test_a_secret_already_provisioned_as_a_root_is_placed_in_the_collection(gmail_env, wired,  # noqa: F811
                                                                             monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: SYSTEM)
    wired.get_artifact.side_effect = lambda db, artifact_id: SimpleNamespace(_key=artifact_id)
    pe.ensure_platform_email_sender(db=MagicMock())
    edges = {(c.args[1], c.args[2]) for c in wired.add_edge.call_args_list}
    assert {(COLL, s) for s in SECRETS} <= edges
