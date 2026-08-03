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

# Platform collections a first-login user is granted on.
#
# 🔴 THIS WAS A HAND-TYPED LIST OF 10 WHILE THE ASSERTIONS BELOW SAID 11 (fixed 2026-07-30). The
# tests read `assert len(grants) == len(_USER_GRANT_SLUGS) == 11`, which cannot hold for a 10-item
# list — so the file failed the moment the seed tree was actually supplied, and the number looked
# like the suspect. MEASURED, it was not: the 11 is RIGHT and the LIST was short.
#
#   · the seed tree carries ELEVEN user grant files (`package/seeds/user/grants/*.yaml`) and the
#     admin set names ELEVEN resources (`seeds/admin/grants/platform-admin.yaml`);
#   · `ALL_PLATFORM_COLLECTION_SLUGS` (bootstrap_types.py) holds ELEVEN;
#   · the one the list omitted was `agience-anchorset` — which is NOT retired. It is
#     `ANCHORSET_COLLECTION_SLUG`, it is in `ALL_PLATFORM_COLLECTION_SLUGS` and in
#     `USER_READABLE_SEED_SLUGS`, it has a platform collection seed (`anchorset-collection.yaml`),
#     and `test_anchor_repo.py` asserts the repo registers it.
#   · the mechanism of the loss: this list also drove the `_registry` fixture, so the omitted slug
#     had no registered id, the loader logged *"unresolved grant resource 'agience/agience-
#     anchorset'"* and DROPPED the grant. The list under-registered the platform and then measured
#     its own omission — 10 grants — while the count constant remembered the truth.
#
# So it is DERIVED now, not typed. A duplicated roster is a seam; the platform constant is the one
# place that knows which collections exist. `test_the_seed_tree_and_the_platform_list_agree` below
# supplies the INDEPENDENT oracle so this cannot become self-confirming.
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
        patch("mantle.services.seed_provisioning.user_provisioning._send_welcome_email"),
    ):
        from mantle.services.seed_provisioning import user_provisioning
        user_provisioning.provision_user(MagicMock(), "user-9", seeds_base=SEEDS_BASE)
    return captured, mock_ws, mock_mat


def _seeded_grant_slugs(subdir: str) -> set[str]:
    """Collection slugs the SEED FILES themselves name — read off disk, so it is independent of
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
    """THE ORACLE for the counts below, and the check that was missing.

    FAILURE MODE, stated first: with `_USER_GRANT_SLUGS` derived from
    `ALL_PLATFORM_COLLECTION_SLUGS`, `len(grants) == len(_USER_GRANT_SLUGS)` would pass even if the
    seed tree had drifted — every grant file that resolves is counted, and one that does NOT resolve
    is merely logged and dropped, so a missing slug shows up as agreement between two numbers that
    came from the same place. This test compares the platform's list against the FILES ON DISK, the
    one comparison that can see the drift. Drop `agience-anchorset` from either side and it fails."""
    assert _seeded_grant_slugs("user") == set(_USER_GRANT_SLUGS), (
        "user grant seeds and ALL_PLATFORM_COLLECTION_SLUGS disagree — a grant file naming a "
        "collection the platform does not list is silently DROPPED (logged as 'unresolved grant "
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
    # one user read + one admin grant per platform collection. Was a hard-coded 22 against a
    # 10-item roster; derived now for the reason recorded at `_USER_GRANT_SLUGS`.
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
