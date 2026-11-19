"""The roles a share or invite offers are Role artifacts in the platform's Roles collection, which
Mantle reads each time: an operator edits a role, or adds one, without a release."""
from __future__ import annotations

import json
import uuid

import pytest

from mantle.db import lattice_api as la
from mantle.entities.grant import Grant
from mantle.routers import grants_router
from mantle.services import role_presets as R
from mantle.services.seed_provisioning import platform_operators as P
from mantle.services.seed_provisioning.platform_email import _ensure_artifact

from .test_there_is_no_promotion_state import real_node

PLATFORM = "u-platform-operator"
NS = uuid.UUID("33333333-3333-4333-8333-333333333333")


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = real_node(tmp_path, monkeypatch)
    from mantle.services.seed_provisioning import loader, platform_email
    monkeypatch.setattr(loader, "get_instance_namespace", lambda: NS)
    monkeypatch.setattr(platform_email, "_operator_id", lambda db: PLATFORM)
    return d


def _role(db, rid, name, actions):
    _ensure_artifact(db, rid, content_type=R.ROLE_CT, name=name.capitalize(),
                     description=f"What the {name} role grants.",
                     context=R._context(name, actions), content="", owner=PLATFORM)


def test_the_seeded_roles_are_the_presets(db):
    assert P.ensure_platform_operators(db)
    assert R.presets(db) == {n: {k: v for k, v in b.items() if v} for n, b in Grant.ROLE_PRESETS.items()}
    names = [r["name"] for r in grants_router.roles(db)]
    assert names == list(Grant.ROLE_PRESETS)


def test_an_edited_role_stands_across_a_reseed(db):
    assert P.ensure_platform_operators(db)
    _role(db, R.role_id("viewer"), "viewer", ["read", "invoke"])
    assert P.ensure_platform_operators(db)
    assert R.permissions_for_role(db, "viewer") == {"can_invoke": True, "can_read": True}
    raw = db.artifacts.get_artifact(R.role_id("viewer"))
    assert json.loads(raw["context"])["actions"] == ["read", "invoke"]


def test_a_role_added_to_the_collection_is_offered(db):
    assert P.ensure_platform_operators(db)
    rid = str(uuid.uuid4())
    _role(db, rid, "reviewer", ["read", "update"])
    la.add_artifact_to_collection(db, R.roles_collection_id(), rid, origin=True)
    assert R.permissions_for_role(db, "reviewer") == {"can_read": True, "can_update": True}
    names = [r["name"] for r in grants_router.roles(db)]
    assert names.index("reviewer") > names.index("contributor")
    assert {"name": "reviewer", "actions": ["read", "update"]} in grants_router.roles(db)


def test_an_unknown_role_is_refused(db):
    assert P.ensure_platform_operators(db)
    with pytest.raises(ValueError):
        R.permissions_for_role(db, "owner")


def test_before_the_roles_are_seeded_the_presets_are_the_seed():
    assert R.presets(None) == Grant.ROLE_PRESETS
