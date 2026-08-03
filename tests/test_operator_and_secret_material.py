"""Sovereign operator resolution + secrets-as-artifacts (local material vault).

1. `services.operator.resolve_operator_id` — Mantle setting → env
   (AGIENCE_OPERATOR_ID, for standalone Origin-off) → Origin fallback.
2. `secrets_service.set_secret_material` / `fetch_secret_material` round-trip —
   the `secret+json` `fetch` op, now backed by Mantle's OWN encrypted store (no
   Origin round-trip).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.services import secrets_service  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Operator resolution — sovereign-capable precedence
# ---------------------------------------------------------------------------

def test_resolve_operator_id_precedence(monkeypatch):
    from mantle.services.operator import resolve_operator_id

    # (1) Mantle's own setting wins.
    ps = MagicMock()
    ps.get.return_value = "op-from-setting"
    monkeypatch.setattr("mantle.services.platform_settings_service.settings", ps)
    assert resolve_operator_id() == "op-from-setting"

    # (2) env config (AGIENCE_OPERATOR_ID) when the setting is empty — standalone.
    ps.get.return_value = None
    monkeypatch.setattr("origin.config.OPERATOR_ID", "op-from-env", raising=False)
    assert resolve_operator_id() == "op-from-env"

    # (3) Origin fallback when setting + env are both empty — full platform.
    monkeypatch.setattr("origin.config.OPERATOR_ID", "", raising=False)
    fake_client = MagicMock()
    fake_client.get_operator_id.return_value = "op-from-origin"
    monkeypatch.setattr("mantle.clients.origin_client.get_origin_client", lambda: fake_client)
    assert resolve_operator_id() == "op-from-origin"


def test_resolve_operator_id_empty_when_nothing_configured(monkeypatch):
    """Standalone node before setup, Origin absent → '' (non-fatal)."""
    ps = MagicMock()
    ps.get.return_value = None
    monkeypatch.setattr("mantle.services.platform_settings_service.settings", ps)
    monkeypatch.setattr("origin.config.OPERATOR_ID", "", raising=False)

    def _boom():
        raise RuntimeError("origin unreachable")

    monkeypatch.setattr("mantle.clients.origin_client.get_origin_client", _boom)
    from mantle.services.operator import resolve_operator_id
    assert resolve_operator_id() == ""


# ---------------------------------------------------------------------------
# 2. Secret material — local vault round-trip (no Origin)
# ---------------------------------------------------------------------------

def test_set_then_fetch_secret_material_round_trip(monkeypatch):
    """Material is stored + resolved in Mantle's OWN encrypted store, keyed by
    the secret artifact id — no Origin call."""
    store: dict = {}
    ps = MagicMock()
    ps.set_setting.side_effect = (
        lambda db, key, value, category, is_secret=False: store.__setitem__(key, value)
    )
    ps.get_secret.side_effect = lambda key: store.get(key)
    monkeypatch.setattr("mantle.services.platform_settings_service.settings", ps)

    secrets_service.set_secret_material(
        db=MagicMock(), owner_id="owner-1",
        secret_id="secret-artifact-1",
        value="refresh-token-xyz",
        secret_type="oauth_refresh_token",
        provider="google",
        authorizer_id="authz-1",
    )
    # Stored locally under the prefixed key (is_secret path).
    assert "secret.material.secret-artifact-1" in store

    artifact = {"root_id": "secret-artifact-1", "created_by": "owner-1"}
    result = secrets_service.fetch_secret_material(
        artifact, {}, SimpleNamespace(store_db=MagicMock()),
    )
    assert result == {"secret_id": "secret-artifact-1", "material": "refresh-token-xyz"}
    # The category is the secret type; stored as a secret setting.
    _, kwargs = ps.set_setting.call_args
    assert kwargs["is_secret"] is True
    assert kwargs["category"] == "oauth_refresh_token"


def test_fetch_secret_material_missing_raises():
    import pytest
    ps = MagicMock()
    ps.get_secret.return_value = None
    import mantle.services.platform_settings_service as pss
    orig = pss.settings
    pss.settings = ps
    try:
        artifact = {"root_id": "nope", "created_by": "owner-1"}
        with pytest.raises(RuntimeError, match="No material stored"):
            secrets_service.fetch_secret_material(artifact, {}, SimpleNamespace(store_db=MagicMock()))
    finally:
        pss.settings = orig
