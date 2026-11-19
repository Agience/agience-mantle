"""First-login provisioning runs for an account, never for an id the platform acts as.

A request that resolves to the system principal (the type publisher, platform automation) or the
platform user finds no reach of its own, which is what a first login looks like. Provisioned, it
writes an Inbox, a person and grants for an id Origin holds no person for: Origin refuses their
co-signature, and in record mode the grants stay in Mantle with no ledger entry (71/test,
2026-11-17: four grants, Mantle 72 against the replica's 68). The post-seed backfill already
leaves the system principal out; `provision_user` does the same for both ids."""
from __future__ import annotations

import pytest

from mantle import config
from mantle.services.seed_provisioning import user_provisioning as up

SYSTEM = "00000000-0000-5000-8000-00000000c0de"


@pytest.fixture
def calls(monkeypatch):
    seen: list = []
    monkeypatch.setattr(up, "_provision_user", lambda db, user_id, **kw: seen.append(user_id))
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: SYSTEM)
    return seen


@pytest.mark.parametrize("pid", [SYSTEM, config.AGIENCE_PLATFORM_USER_ID])
def test_a_platform_principal_writes_nothing(calls, pid):
    up.provision_user(None, user_id=pid)
    assert calls == []


def test_an_account_is_still_provisioned(calls):
    up.provision_user(None, user_id="a-person")
    assert calls == ["a-person"]


def test_without_an_instance_namespace_an_account_is_still_provisioned(calls, monkeypatch):
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "")
    up.provision_user(None, user_id="a-person")
    assert calls == ["a-person"]
