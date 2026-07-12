"""Tests for the platform outbound-email provisioner.

Covers the no-op skip branches and the happy path: it provisions the email
operator graph (operator + authorizer + 2 secret artifacts), wires the
operator→authorizer edge, injects the two secret materials, and grants the
operator invoke/read — all from GMAIL_OAUTH_* env, owned by the platform operator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.seed_provisioning import platform_email as pe  # noqa: E402

_OPERATOR = "operator-1"


@pytest.fixture
def gmail_env(monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "gmail")
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com")
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("GMAIL_OAUTH_REFRESH_TOKEN", "rtoken")
    monkeypatch.setenv("PLATFORM_EMAIL_ADDRESS", "connect@agience.ai")


@pytest.fixture
def wired(monkeypatch):
    """Stub db/secrets/topology helpers; return the mocks for assertions."""
    monkeypatch.setattr(pe, "derive_uuid", lambda ns, namespace, slug: f"id-{slug}")
    monkeypatch.setattr(pe, "get_instance_namespace", lambda: "ns")
    monkeypatch.setattr(pe, "register_id", lambda *a, **k: None)
    monkeypatch.setattr(pe, "_operator_id", lambda db: _OPERATOR)

    get_artifact = MagicMock(return_value=None)   # nothing exists yet
    create_artifact = MagicMock()
    update_artifact = MagicMock()
    add_edge = MagicMock()
    upsert_grant = MagicMock()
    monkeypatch.setattr(pe, "db_get_artifact", get_artifact)
    monkeypatch.setattr(pe, "db_create_artifact", create_artifact)
    monkeypatch.setattr(pe, "db_update_artifact", update_artifact)
    monkeypatch.setattr(pe, "db_add_edge", add_edge)
    monkeypatch.setattr(pe, "db_upsert_user_collection_grant", upsert_grant)

    secrets = MagicMock()
    monkeypatch.setattr(pe, "secrets_service", secrets)

    # System principal resolution is imported inside the function from agience_core;
    # default it to "" so grant-count assertions are deterministic (operator-only).
    # The dedicated test overrides it to exercise the system-principal branch.
    monkeypatch.setattr("services.peer_signing.get_system_principal_id", lambda: "")

    return SimpleNamespace(
        get_artifact=get_artifact, create_artifact=create_artifact,
        update_artifact=update_artifact, add_edge=add_edge,
        upsert_grant=upsert_grant, secrets=secrets,
    )


def test_skips_when_provider_not_gmail(monkeypatch, wired):
    monkeypatch.setenv("EMAIL_PROVIDER", "smtp")
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()
    wired.secrets.set_secret_material.assert_not_called()


def test_skips_when_creds_missing(monkeypatch, wired):
    monkeypatch.setenv("EMAIL_PROVIDER", "gmail")
    monkeypatch.delenv("GMAIL_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GMAIL_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GMAIL_OAUTH_REFRESH_TOKEN", raising=False)
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()


def test_skips_when_no_operator(gmail_env, wired, monkeypatch):
    monkeypatch.setattr(pe, "_operator_id", lambda db: "")
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()
    wired.secrets.set_secret_material.assert_not_called()


def test_happy_path_provisions_graph(gmail_env, wired):
    pe.ensure_platform_email_sender(db=MagicMock())

    # Four artifacts: operator + authorizer + 2 secrets.
    cts = {c.args[1].content_type for c in wired.create_artifact.call_args_list}
    assert cts == {pe._OPERATOR_CT, pe._AUTHORIZER_CT, pe._SECRET_CT}
    assert wired.create_artifact.call_count == 4
    for c in wired.create_artifact.call_args_list:
        assert c.args[1].created_by == _OPERATOR

    # Two secret materials injected, owned by the operator, keyed by artifact id.
    assert wired.secrets.set_secret_material.call_count == 2
    types = {c.kwargs["secret_type"] for c in wired.secrets.set_secret_material.call_args_list}
    assert types == {"oauth_client_secret", "oauth_refresh_token"}
    for c in wired.secrets.set_secret_material.call_args_list:
        assert c.args[1] == _OPERATOR

    # Operator→authorizer typed edge.
    wired.add_edge.assert_called_once()
    assert wired.add_edge.call_args.kwargs["relationship"] == "authorizer"

    # Grants: operator gets invoke on the operator artifact, read on all four.
    assert wired.upsert_grant.call_count == 4
    invoke_grants = [c for c in wired.upsert_grant.call_args_list if c.kwargs.get("can_invoke")]
    assert len(invoke_grants) == 1
    assert invoke_grants[0].kwargs["collection_id"] == "id-platform-email-sender"


def test_idempotent_when_graph_exists(gmail_env, wired):
    # db_get_artifact returns an Artifact ENTITY (not a dict); _ensure_artifact
    # reconciles its attributes in place. A SimpleNamespace stands in for the
    # existing entity so those attribute writes succeed.
    wired.get_artifact.return_value = SimpleNamespace(_key="exists")  # already there
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()      # no re-create
    wired.add_edge.assert_not_called()             # edge not re-added
    assert wired.secrets.set_secret_material.call_count == 2   # material still self-heals


def test_grants_system_principal_when_available(gmail_env, wired, monkeypatch):
    """When the system principal resolves, it's granted the same mail perms as
    the operator — granted_by the operator (the provenance that roots it to a
    person) — so webhook/background sends can act AS it."""
    monkeypatch.setattr("services.peer_signing.get_system_principal_id", lambda: "sys-principal-1")
    pe.ensure_platform_email_sender(db=MagicMock())

    # 8 grants: 4 (operator) + 4 (system principal).
    assert wired.upsert_grant.call_count == 8
    grantees = {c.kwargs["user_id"] for c in wired.upsert_grant.call_args_list}
    assert grantees == {_OPERATOR, "sys-principal-1"}
    # Every grant is issued BY the operator — that's the rooting to a person.
    for c in wired.upsert_grant.call_args_list:
        assert c.kwargs["granted_by"] == _OPERATOR
    # The system principal also gets invoke on the operator artifact.
    sys_invoke = [c for c in wired.upsert_grant.call_args_list
                  if c.kwargs["user_id"] == "sys-principal-1" and c.kwargs.get("can_invoke")]
    assert len(sys_invoke) == 1
    assert sys_invoke[0].kwargs["collection_id"] == "id-platform-email-sender"
