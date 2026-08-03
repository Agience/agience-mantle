"""Unit tests for services.gate_service.

Covers entitlement cache CRUD, free-tier defaults, usage tally accumulation,
and the live count queries. After the lattice flip, entitlement rows and
tallies live as typed docs in the ONE artifact store ("entitlement:<person>" /
"tally:<key>") and counts stream `list_artifacts` — exercised here against a
REAL temp lattice store, not a mock.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mantle.db import lattice_api as api
from mantle.services import gate_service
from mantle.entities.collection import WORKSPACE_CONTENT_TYPE


@pytest.fixture
def db(tmp_path):
    return api.open_database(str(tmp_path / "t.db"), origin="test-gate")


def _seed(db, doc: dict) -> None:
    db.artifacts.put_artifact(doc)


# ---------------------------------------------------------------------------
# enforcement_enabled
# ---------------------------------------------------------------------------

class TestEnforcementEnabled:
    def test_true_when_config_flag_set(self):
        with patch("origin.config.BILLING_ENFORCEMENT_ENABLED", True):
            assert gate_service.enforcement_enabled() is True

    def test_false_when_config_flag_unset(self):
        with patch("origin.config.BILLING_ENFORCEMENT_ENABLED", False):
            assert gate_service.enforcement_enabled() is False


# ---------------------------------------------------------------------------
# get_limits / get_or_default_limits / set_limits
# ---------------------------------------------------------------------------

class TestLimits:
    def test_get_limits_returns_none_when_no_row(self, db):
        assert gate_service.get_limits(db, "user-1") is None

    def test_get_limits_returns_dict_when_row_exists(self, db):
        _seed(db, {
            "id": "entitlement:user-1",
            "content_type": gate_service._ENT_CT,
            "person_id": "user-1",
            "max_workspaces": 10, "max_artifacts": 500, "vu_limit": 1000,
        })
        out = gate_service.get_limits(db, "user-1")
        # `features` defaults to [] when the row carries no capability flags.
        assert out == {
            "max_workspaces": 10,
            "max_artifacts": 500,
            "vu_limit": 1000,
            "features": [],
        }

    def test_get_or_default_falls_back_to_free_tier(self, db):
        out = gate_service.get_or_default_limits(db, "user-1")
        # Free tier defaults from the module constant.
        assert out["max_workspaces"] == 1
        assert out["vu_limit"] == 100

    def test_set_limits_inserts_when_missing(self, db):
        gate_service.set_limits(db, "user-1", max_workspaces=10, vu_limit=500)
        doc = db.artifacts.get_artifact("entitlement:user-1")
        assert doc is not None
        assert doc["id"] == "entitlement:user-1"
        assert doc["max_workspaces"] == 10
        assert doc["vu_limit"] == 500
        # A fresh insert stamps the empty capability set.
        assert doc["features"] == []

    def test_set_limits_updates_when_present(self, db):
        _seed(db, {
            "id": "entitlement:user-1",
            "content_type": gate_service._ENT_CT,
            "person_id": "user-1",
            "max_workspaces": 1,
            "features": ["beacon"],
        })
        gate_service.set_limits(db, "user-1", max_workspaces=99)
        doc = db.artifacts.get_artifact("entitlement:user-1")
        assert doc["max_workspaces"] == 99
        # Updated in place — same plane doc, no duplicate row minted.
        ent_docs = [d for d in db.artifacts.list_artifacts(
            content_type=gate_service._ENT_CT)]
        assert [d["id"] for d in ent_docs] == ["entitlement:user-1"]
        # An update that omits `features` leaves the capability set untouched.
        assert doc["features"] == ["beacon"]


# ---------------------------------------------------------------------------
# Tallies
# ---------------------------------------------------------------------------

class TestTallies:
    def test_get_tally_zero_when_no_row(self, db):
        assert gate_service.get_tally(db, "user-1", "vu", "2026-04") == 0

    def test_get_tally_returns_total(self, db):
        _seed(db, {
            "id": "tally:user-1:vu:2026-04",
            "content_type": gate_service._TALLY_CT,
            "person_id": "user-1", "dimension": "vu", "period": "2026-04",
            "total": 42,
        })
        assert gate_service.get_tally(db, "user-1", "vu", "2026-04") == 42

    def test_add_tally_inserts_when_missing(self, db):
        result = gate_service.add_tally(db, "user-1", "vu", "2026-04", amount=5)
        assert result == 5
        doc = db.artifacts.get_artifact("tally:user-1:vu:2026-04")
        assert doc is not None
        assert doc["id"] == "tally:user-1:vu:2026-04"
        assert doc["total"] == 5
        assert doc["dimension"] == "vu"
        assert doc["period"] == "2026-04"

    def test_add_tally_accumulates_when_present(self, db):
        _seed(db, {
            "id": "tally:user-1:vu:2026-04",
            "content_type": gate_service._TALLY_CT,
            "person_id": "user-1", "dimension": "vu", "period": "2026-04",
            "total": 5,
        })
        result = gate_service.add_tally(db, "user-1", "vu", "2026-04", amount=7)
        assert result == 12
        assert db.artifacts.get_artifact("tally:user-1:vu:2026-04")["total"] == 12

    def test_get_all_tallies_groups_by_dimension(self, db):
        rows = [
            {"dimension": "vu", "period": "2026-04", "total": 100},
            {"dimension": "vu", "period": "2026-03", "total": 250},
            {"dimension": "tokens", "period": "2026-04", "total": 5000},
        ]
        for r in rows:
            _seed(db, {
                "id": f"tally:user-1:{r['dimension']}:{r['period']}",
                "content_type": gate_service._TALLY_CT,
                "person_id": "user-1", **r,
            })
        out = gate_service.get_all_tallies(db, "user-1")
        assert out == {
            "vu": {"2026-04": 100, "2026-03": 250},
            "tokens": {"2026-04": 5000},
        }


# ---------------------------------------------------------------------------
# Live counts (streamed off the unified store)
# ---------------------------------------------------------------------------

class TestLiveCounts:
    def test_count_workspaces_filters_by_workspace_content_type(self, db):
        """Workspaces are Collections in the unified store, discriminated by
        content_type=workspace and owned via created_by. The inbox collection is
        keyed by the person_id and is excluded from the count."""
        for i in range(7):
            _seed(db, {"id": f"ws-{i}", "content_type": WORKSPACE_CONTENT_TYPE,
                       "created_by": "user-1"})
        # Inbox: a workspace whose id IS the person id — excluded.
        _seed(db, {"id": "user-1", "content_type": WORKSPACE_CONTENT_TYPE,
                   "created_by": "user-1"})
        # Someone else's workspace — not counted.
        _seed(db, {"id": "ws-other", "content_type": WORKSPACE_CONTENT_TYPE,
                   "created_by": "user-2"})
        # A non-workspace artifact by this user — not counted.
        _seed(db, {"id": "a-1", "content_type": "text/plain", "created_by": "user-1"})
        assert gate_service.count_workspaces(db, "user-1") == 7

    def test_count_workspaces_zero_when_empty(self, db):
        assert gate_service.count_workspaces(db, "user-1") == 0

    def test_count_artifacts_counts_unified_store_by_creator(self, db):
        for i in range(42):
            _seed(db, {"id": f"a-{i}", "content_type": "text/plain",
                       "created_by": "user-1"})
        # Another creator's artifact must not be counted.
        _seed(db, {"id": "a-other", "content_type": "text/plain",
                   "created_by": "user-2"})
        assert gate_service.count_artifacts(db, "user-1") == 42

    def test_count_artifacts_excludes_archived(self, db):
        for i in range(5):
            _seed(db, {"id": f"a-{i}", "content_type": "text/plain",
                       "created_by": "user-1"})
        _seed(db, {"id": "a-archived", "content_type": "text/plain",
                   "created_by": "user-1", "state": "archived"})
        assert gate_service.count_artifacts(db, "user-1") == 5

    def test_count_artifacts_zero_when_empty(self, db):
        assert gate_service.count_artifacts(db, "user-1") == 0
