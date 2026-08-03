"""gate_service capability feature (`beacon`): enforcement-off grants all;
when enforced, the feature must be present in the entitlement cache."""

from mantle.services import gate_service


def test_features_open_when_enforcement_disabled(monkeypatch):
    # Dev / self-host: enforcement off → every capability granted, no DB touch.
    monkeypatch.setattr(gate_service, "enforcement_enabled", lambda: False)
    assert gate_service.has_feature(None, "user-1", "beacon") is True


def test_feature_present_when_enforced(monkeypatch):
    monkeypatch.setattr(gate_service, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(gate_service, "get_limits", lambda db, pid: {"features": ["beacon"]})
    assert gate_service.has_feature(object(), "user-1", "beacon") is True


def test_feature_absent_when_enforced(monkeypatch):
    monkeypatch.setattr(gate_service, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(gate_service, "get_limits", lambda db, pid: {"features": []})
    assert gate_service.has_feature(object(), "user-1", "beacon") is False


def test_no_entitlement_row_when_enforced(monkeypatch):
    monkeypatch.setattr(gate_service, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(gate_service, "get_limits", lambda db, pid: None)
    assert gate_service.has_feature(object(), "user-1", "beacon") is False


# --- set_limits writes the capability set (the gate_router `features` path) ----
#
# Lattice plane: entitlement rows are typed docs in the ONE artifact store
# ("entitlement:<person>"), written via db.artifacts.put_artifact — run against
# a real temp lattice instead of an store collection mock.

def test_set_limits_persists_features_on_insert(tmp_path):
    from mantle.db import lattice_api as api

    db = api.open_database(str(tmp_path / "t.db"), origin="test-gate-features")
    gate_service.set_limits(db, "user-1", max_workspaces=2, features=["beacon"])
    doc = db.artifacts.get_artifact("entitlement:user-1")
    assert doc["features"] == ["beacon"]


def test_set_limits_without_features_does_not_clobber_on_update(tmp_path):
    from mantle.db import lattice_api as api

    db = api.open_database(str(tmp_path / "t.db"), origin="test-gate-features")
    # Existing row with a capability already granted.
    db.artifacts.put_artifact({
        "id": "entitlement:user-1",
        "content_type": gate_service._ENT_CT,
        "person_id": "user-1",
        "max_workspaces": 1,
        "features": ["beacon"],
    })
    gate_service.set_limits(db, "user-1", max_workspaces=2)  # features omitted
    doc = db.artifacts.get_artifact("entitlement:user-1")
    # Omitting features leaves the existing set untouched.
    assert doc["features"] == ["beacon"]
    assert doc["max_workspaces"] == 2
