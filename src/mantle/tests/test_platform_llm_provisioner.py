"""Tests for the platform default LLM connection provisioner.

Covers the no-op skip branches and the happy path: it provisions the
llm-connection artifact + an API-key secret artifact, injects the key material
into the operator's vault, and grants read/invoke — all from LLM_* env, owned by
the platform operator (and the system principal when available). Mirrors the
platform-email provisioner tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from services.seed_provisioning import platform_llm as pl  # noqa: E402

_OPERATOR = "operator-1"


@pytest.fixture
def llm_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_API_KEY", "sk-platform-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_ENDPOINT", raising=False)


@pytest.fixture
def wired(monkeypatch):
    """Stub db/secrets/topology helpers; return the mocks for assertions."""
    monkeypatch.setattr(pl, "derive_uuid", lambda ns, namespace, slug: f"id-{slug}")
    monkeypatch.setattr(pl, "get_instance_namespace", lambda: "ns")
    monkeypatch.setattr(pl, "register_id", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_operator_id", lambda db: _OPERATOR)

    get_artifact = MagicMock(return_value=None)   # nothing exists yet
    create_artifact = MagicMock()
    update_artifact = MagicMock()
    upsert_grant = MagicMock()
    monkeypatch.setattr(pl, "db_get_artifact", get_artifact)
    monkeypatch.setattr(pl, "db_create_artifact", create_artifact)
    monkeypatch.setattr(pl, "db_update_artifact", update_artifact)
    monkeypatch.setattr(pl, "db_upsert_user_collection_grant", upsert_grant)

    secrets = MagicMock()
    monkeypatch.setattr(pl, "secrets_service", secrets)

    # Default the system principal to "" so grant-count assertions are
    # deterministic (operator-only). A dedicated test overrides it.
    monkeypatch.setattr("services.peer_signing.get_system_principal_id", lambda: "")

    return SimpleNamespace(
        get_artifact=get_artifact, create_artifact=create_artifact,
        update_artifact=update_artifact, upsert_grant=upsert_grant, secrets=secrets,
    )


def test_skips_when_no_api_key(monkeypatch, wired):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert pl.ensure_platform_llm_connection(db=MagicMock()) is True
    wired.create_artifact.assert_not_called()
    wired.secrets.set_secret_material.assert_not_called()


def test_transient_when_no_operator(llm_env, wired, monkeypatch):
    monkeypatch.setattr(pl, "_operator_id", lambda db: "")
    assert pl.ensure_platform_llm_connection(db=MagicMock()) is False
    wired.create_artifact.assert_not_called()
    wired.secrets.set_secret_material.assert_not_called()


def test_happy_path_provisions_connection_and_secret(llm_env, wired):
    assert pl.ensure_platform_llm_connection(db=MagicMock()) is True

    # Two artifacts: connection + secret, both owned by the operator.
    cts = {c.args[1].content_type for c in wired.create_artifact.call_args_list}
    assert cts == {pl._CONNECTION_CT, pl._SECRET_CT}
    assert wired.create_artifact.call_count == 2
    for c in wired.create_artifact.call_args_list:
        assert c.args[1].created_by == _OPERATOR

    # The connection points at the secret via credentials_ref with the
    # platform_secret resolution (vault-custodied, NOT config-read).
    conn = next(c.args[1] for c in wired.create_artifact.call_args_list
                if c.args[1].content_type == pl._CONNECTION_CT)
    ctx = json.loads(conn.context)
    assert ctx["is_platform_default"] is True
    assert ctx["provider"] == "anthropic"
    assert ctx["credentials_ref"]["resolution"] == "platform_secret"
    assert ctx["credentials_ref"]["secret_id"] == "id-platform-llm-api-key"

    # The API key material is injected into the operator's vault as an llm_key.
    wired.secrets.set_secret_material.assert_called_once()
    call = wired.secrets.set_secret_material.call_args
    assert call.args[1] == _OPERATOR
    assert call.kwargs["secret_type"] == "llm_key"
    assert call.kwargs["value"] == "sk-platform-key"

    # Grants: operator gets read on both + invoke on the connection.
    assert wired.upsert_grant.call_count == 2
    invoke_grants = [c for c in wired.upsert_grant.call_args_list if c.kwargs.get("can_invoke")]
    assert len(invoke_grants) == 1
    assert invoke_grants[0].kwargs["collection_id"] == "id-platform-llm-connection"


def test_default_model_applied(llm_env, wired):
    pl.ensure_platform_llm_connection(db=MagicMock())
    conn = next(c.args[1] for c in wired.create_artifact.call_args_list
                if c.args[1].content_type == pl._CONNECTION_CT)
    ctx = json.loads(conn.context)
    assert ctx["model"] == pl._DEFAULT_MODELS["anthropic"]


def test_explicit_model_and_endpoint(llm_env, wired, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    monkeypatch.setenv("LLM_ENDPOINT", "https://api.anthropic.com")
    pl.ensure_platform_llm_connection(db=MagicMock())
    conn = next(c.args[1] for c in wired.create_artifact.call_args_list
                if c.args[1].content_type == pl._CONNECTION_CT)
    ctx = json.loads(conn.context)
    assert ctx["model"] == "claude-haiku-4-5-20251001"
    assert ctx["endpoint"] == "https://api.anthropic.com"


def test_idempotent_when_already_provisioned(llm_env, wired):
    wired.get_artifact.return_value = SimpleNamespace(_key="exists")  # already there
    assert pl.ensure_platform_llm_connection(db=MagicMock()) is True
    wired.create_artifact.assert_not_called()      # no re-create
    wired.secrets.set_secret_material.assert_called_once()   # material still self-heals


def test_grants_system_principal_when_available(llm_env, wired, monkeypatch):
    """When the system principal resolves, it gets the same read/invoke perms as
    the operator — granted_by the operator — so Verso can resolve the platform
    key for any user's chat turn."""
    monkeypatch.setattr("services.peer_signing.get_system_principal_id", lambda: "sys-principal-1")
    pl.ensure_platform_llm_connection(db=MagicMock())

    # 4 grants: 2 (operator) + 2 (system principal).
    assert wired.upsert_grant.call_count == 4
    grantees = {c.kwargs["user_id"] for c in wired.upsert_grant.call_args_list}
    assert grantees == {_OPERATOR, "sys-principal-1"}
    for c in wired.upsert_grant.call_args_list:
        assert c.kwargs["granted_by"] == _OPERATOR
