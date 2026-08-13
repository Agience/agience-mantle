"""Per-user provisioning: the declarative `package/seeds/user` grant artifacts
applied through `provision_user`, with the inbox-workspace + materialization glue
stubbed (they loop over live DB state and are covered by their own paths).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mantle.services.bootstrap_types import ALL_PLATFORM_COLLECTION_SLUGS
from ._package_root import seeds_root  # noqa: E402 — single source of the package tree location
SEEDS_BASE = seeds_root()

# Platform collections a first-login user is granted on, derived from
# ALL_PLATFORM_COLLECTION_SLUGS rather than hand-typed: the platform constant is the one place
# that knows which collections exist (currently eleven, matching the eleven user grant files
# under `package/seeds/user/grants/*.yaml` and the eleven resources named in
# `seeds/admin/grants/platform-admin.yaml`). A duplicated roster here would be a seam — a slug
# omitted from a hand-maintained list resolves to nothing, so the loader just logs
# "unresolved grant resource" and drops the grant rather than raising.
# `test_the_seed_tree_and_the_platform_list_agree` below checks the seed files on disk against
# this constant independently, so the two cannot silently agree with each other by construction.
_USER_GRANT_SLUGS = list(ALL_PLATFORM_COLLECTION_SLUGS)


@pytest.fixture(autouse=True)
def _registry(monkeypatch, tmp_path):
    monkeypatch.setenv("KEYS_DIR", str(tmp_path / "keys"))
    from mantle.services import platform_topology

    platform_topology.clear_registry()
    for slug in _USER_GRANT_SLUGS:
        platform_topology.register_id(slug, f"id-{slug}")
    yield
    platform_topology.clear_registry()


def _provision(is_admin: bool = False):
    captured: list[dict] = []
    with (
        patch("mantle.services.seed_provisioning.loader.db_upsert_user_collection_grant",
              side_effect=lambda db, **kw: (captured.append(kw), (MagicMock(), True))[1]),
        patch("mantle.services.seed_provisioning.user_provisioning._ensure_inbox_workspace",
              return_value="ws-1") as mock_ws,
        patch("mantle.services.seed_provisioning.user_provisioning._materialize_inbox") as mock_mat,
        patch("mantle.services.seed_provisioning.user_provisioning._is_platform_admin",
              return_value=is_admin),
        # Identity-linkage (Person artifact + lead conversion) has its own test
        # (test_person_and_lead_conversion); stub here so this grants-focused test
        # stays fast and doesn't reach person_service over HTTP.
        patch("mantle.services.person_service.get_user_by_id", return_value=None),
        patch("mantle.services.seed_provisioning.user_provisioning._ensure_person_artifact", return_value=False),
        patch("mantle.services.seed_provisioning.user_provisioning._convert_leads_for_person"),
    ):
        from mantle.services.seed_provisioning import user_provisioning
        user_provisioning.provision_user(MagicMock(), "user-9", seeds_base=SEEDS_BASE)
    return captured, mock_ws, mock_mat


def _seeded_grant_slugs(subdir: str) -> set[str]:
    """Collection slugs the seed files themselves name — read off disk, so it is independent of
    both `ALL_PLATFORM_COLLECTION_SLUGS` and of anything `provision_user` does."""
    import yaml

    out: set[str] = set()
    for f in sorted((SEEDS_BASE / subdir / "grants").glob("*.yaml")):
        doc = yaml.safe_load(f.read_text(encoding="utf-8-sig")) or {}
        refs = doc.get("resources") or ([doc["resource"]] if doc.get("resource") else [])
        for ref in refs:
            out.add(str(ref).rsplit("/", 1)[-1])   # "agience/agience-hosts" -> "agience-hosts"
    return out


def test_the_seed_tree_and_the_platform_list_agree():
    """Cross-checks `ALL_PLATFORM_COLLECTION_SLUGS` against the grant seed files on disk — the
    one comparison independent of `_USER_GRANT_SLUGS` itself.

    Because `_USER_GRANT_SLUGS` is derived from `ALL_PLATFORM_COLLECTION_SLUGS`,
    `len(grants) == len(_USER_GRANT_SLUGS)` alone would pass even if the seed tree had drifted:
    a grant file that fails to resolve is logged and dropped rather than counted, so a missing
    slug would not show up as a length mismatch. Comparing against the files on disk catches
    that case. Drop `agience-anchorset` from either side and this fails."""
    assert _seeded_grant_slugs("user") == set(_USER_GRANT_SLUGS), (
        "user grant seeds and ALL_PLATFORM_COLLECTION_SLUGS disagree — a grant file naming a "
        "collection the platform does not list is silently dropped (logged as 'unresolved grant "
        "resource'), and a listed collection with no grant file is simply never granted")
    assert _seeded_grant_slugs("admin") == set(_USER_GRANT_SLUGS), (
        "the admin grant set does not cover exactly the platform collections")


def test_provision_user_issues_read_grants():
    grants, _, _ = _provision(is_admin=False)
    assert len(grants) == len(_USER_GRANT_SLUGS)
    assert all(g["user_id"] == "user-9" for g in grants)
    by_res = {g["collection_id"]: g for g in grants}
    # Read-only on the registry collections; no admin for a normal user.
    auth = by_res["id-agience-authorities"]
    assert auth["can_read"] and not auth["can_update"] and not auth["can_admin"]
    # Servers get read + invoke.
    servers = by_res["id-agience-seeds-all-servers"]
    assert servers["can_read"] and servers["can_invoke"] and not servers["can_update"]


def test_provision_designated_admin_adds_full_grants():
    """The designated platform admin gets the admin grant set (full) on every
    platform collection, on top of the base user reads — same grant format, just
    a fuller set. The admin grants are applied last, so they win."""
    grants, _, _ = _provision(is_admin=True)
    # one user read + one admin grant per platform collection.
    assert len(grants) == 2 * len(_USER_GRANT_SLUGS)
    by_res_last = {g["collection_id"]: g for g in grants}  # admin set applied last
    for slug in _USER_GRANT_SLUGS:
        g = by_res_last[f"id-{slug}"]
        assert g["can_read"] and g["can_update"] and g["can_invoke"] and g["can_admin"], slug


def test_provision_user_runs_inbox_glue():
    _, mock_ws, mock_mat = _provision()
    mock_ws.assert_called_once()
    mock_mat.assert_called_once()


def test_provision_user_empty_user_id_is_noop():
    with patch("mantle.services.seed_provisioning.user_provisioning._ensure_inbox_workspace") as mock_ws:
        from mantle.services.seed_provisioning import user_provisioning
        user_provisioning.provision_user(MagicMock(), "")
    mock_ws.assert_not_called()
