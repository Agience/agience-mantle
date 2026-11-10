"""A timer's job delegation asks for a person's content key only to read an artifact its job grant
lists by id (John, 2026-11-10, Q9). Every other narrowed delegation still asks for none, and a
job's writes still go to the drop key; Origin decides the release on the same rule.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from mantle.services import content_crypto as cc
from mantle.services.acting_principal import KeyCustodyDenied, acting_from_auth


def _job(reach=("t-1", "target-1")):
    return NS(narrowed="lumen", job_reach=tuple(reach))


def test_it_reads_the_key_of_an_artifact_its_grant_lists():
    for artifact in ("t-1", "target-1"):
        with cc.content_key_artifact(artifact):
            cc._refuse_ember(_job())


@pytest.mark.parametrize("artifact", ["inbox-1", None])
def test_it_reads_no_other_key(artifact):
    with cc.content_key_artifact(artifact), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(_job())


def test_it_mints_and_writes_under_no_key():
    with cc.content_key_artifact("t-1"), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(_job(), may_create=True)


def test_a_narrowed_delegation_with_no_job_reads_nothing():
    with cc.content_key_artifact("t-1"), pytest.raises(KeyCustodyDenied):
        cc._refuse_ember(NS(narrowed="lumen", job_reach=()))


def test_the_acting_principal_carries_the_reach_origin_signed():
    auth = NS(principal_id="alice", principal_type="user", actor="agience-server-lumen",
              host_id=None, authority=None, scope="persona:lumen", narrowed_audience="lumen",
              job_reach=("t-1",))
    assert acting_from_auth(auth).job_reach == ("t-1",)
    assert acting_from_auth(NS(**{**vars(auth), "job_reach": None})).job_reach == ()
