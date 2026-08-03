"""server_registry is DYNAMIC — personas self-register; nothing is manifest-derived."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from mantle.services import server_registry as sr


def _fake_settings(monkeypatch, store: dict):
    """Mock platform_settings.set_setting (the persist path used by register())."""
    fake = MagicMock()
    fake.set_setting.side_effect = (
        lambda db, key, value, category: store.__setitem__(key, value)
    )
    monkeypatch.setattr("mantle.services.platform_settings_service.settings", fake)
    return fake


def _fake_lattice_rows(monkeypatch, store: dict):
    """Mock store get_all_platform_settings (the read path used by load_from_store())."""
    rows = [{"id": k, "value": v} for k, v in store.items()]
    monkeypatch.setattr("mantle.db.identity_backend.get_all_platform_settings", lambda db: rows)


def test_register_and_resolve(monkeypatch):
    sr._reset_for_tests()
    monkeypatch.setattr(sr, "_derive_server_uuid", lambda name: f"uuid-{name}")
    _fake_settings(monkeypatch, {})

    uid = sr.register(
        MagicMock(), name="aria", client_id="agience-server-aria",
        path="/aria/mcp", role="assistant", title="Aria",
    )
    assert uid == "uuid-aria"
    assert sr.resolve_name_to_id("aria") == "uuid-aria"
    assert sr.get_id("aria") == "uuid-aria"
    assert sr.get_name_by_id("uuid-aria") == "aria"
    assert sr.is_builtin_id("uuid-aria")
    assert sr.get_entry("aria").client_id == "agience-server-aria"
    assert "agience-server-aria" in sr.all_client_ids()
    assert sr.all_names() == ["aria"]


def test_register_is_idempotent(monkeypatch):
    sr._reset_for_tests()
    monkeypatch.setattr(sr, "_derive_server_uuid", lambda name: f"uuid-{name}")
    _fake_settings(monkeypatch, {})
    sr.register(MagicMock(), name="iris", client_id="agience-server-iris", path="/iris/mcp")
    sr.register(MagicMock(), name="iris", client_id="agience-server-iris", path="/iris/mcp", role="email")
    assert sr.all_names() == ["iris"]
    assert sr.get_entry("iris").role == "email"


def test_load_from_store_repopulates(monkeypatch):
    sr._reset_for_tests()
    monkeypatch.setattr(sr, "_derive_server_uuid", lambda name: f"uuid-{name}")
    store = {
        "server.iris": json.dumps({
            "name": "iris", "title": "Iris", "path": "/iris/mcp",
            "client_id": "agience-server-iris", "role": "email", "summary": "",
        }),
        # a non-server platform setting must be ignored:
        "platform.operator_id": "op-1",
    }
    _fake_lattice_rows(monkeypatch, store)

    sr.load_from_store(MagicMock())
    assert sr.resolve_name_to_id("iris") == "uuid-iris"
    assert "agience-server-iris" in sr.all_client_ids()
    assert sr.all_names() == ["iris"]


def test_resolve_unknown_raises():
    sr._reset_for_tests()
    with pytest.raises(ValueError):
        sr.resolve_name_to_id("nope")


def test_empty_by_default():
    """No manifest → registry is empty until something registers."""
    sr._reset_for_tests()
    assert sr.all_entries() == []
    assert sr.all_client_ids() == frozenset()
