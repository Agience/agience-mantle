"""V4: first-login provisioning runs from a GET, so its writes race. The Inbox id is derived.

THE DEFECT. `_ensure_inbox_workspace` is check-then-act — it lists the user's workspaces and
creates one if the list is empty — and nothing makes those two steps atomic. There is no lock in
`user_provisioning`, and provisioning is triggered by a *read*, where parallel arrivals are the
normal case: clients retry, prefetch, and open tabs. Two requests on a first login therefore both
saw an empty list and both created an Inbox, each with a fresh UUID. The duplicate was then
papered over by `min(created_time)`, leaving the loser owned, orphaned and reported to nobody.

THE FIX, and the measurement it rests on. The id is now DERIVED from the user id, so both
racers address the same row and the duplicate cannot exist. Measured 2026-08-26:
`lattice_api.create_collection` is `put_artifact` — the identical call `update_collection` makes —
so the second write UPSERTS rather than raising. That is last-writer-wins. It is safe here only
because both racers write byte-identical content; it is NOT an arbiter, so nothing may build
create-then-catch on a duplicate-id failure that never comes.

Existing users are deliberately NOT migrated. The create path runs only for a user with no
workspace at all, so an Inbox already carrying a random UUID keeps it.
"""
from __future__ import annotations

import pytest

from mantle.db import lattice_api
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE
from mantle.services import workspace_service
from mantle.services.seed_provisioning import user_provisioning as up


@pytest.fixture
def db(tmp_path):
    return lattice_api.LatticeDatabase(str(tmp_path / "prov.db"), origin="prov-test")


def test_the_inbox_id_is_derived_and_per_user():
    a = up.inbox_workspace_id("user-a")
    assert a == up.inbox_workspace_id("user-a"), "not deterministic"
    assert a != up.inbox_workspace_id("user-b"), "two users share an Inbox id"


def test_two_racing_first_logins_converge_on_one_workspace(db, monkeypatch):
    """The race, played out exactly: both callers observe the empty list before either writes."""
    user = "racer"
    real_list = workspace_service.list_workspaces
    seen: list = []

    def both_see_empty(store_db, user_id):
        # The interleaving under test: the second caller's LIST happens before the first
        # caller's CREATE is visible, which is what check-then-act cannot prevent.
        seen.append(user_id)
        return [] if len(seen) <= 2 else real_list(store_db, user_id)

    monkeypatch.setattr(workspace_service, "list_workspaces", both_see_empty)
    first = up._ensure_inbox_workspace(db, user)
    second = up._ensure_inbox_workspace(db, user)
    monkeypatch.undo()

    assert first == second == up.inbox_workspace_id(user)
    surviving = workspace_service.list_workspaces(db, user)
    assert len(surviving) == 1, (
        "the race left %d workspaces: %s" % (len(surviving), [w.id for w in surviving]))


def test_the_old_minted_id_would_have_duplicated(db):
    """The vacuous-pass guard. A test that cannot fail proves nothing, so this shows the
    behaviour the fix removes: two unpinned creates DO leave two workspaces."""
    user = "unpinned"
    workspace_service.create_workspace(db, user, "Inbox")
    workspace_service.create_workspace(db, user, "Inbox")
    assert len(workspace_service.list_workspaces(db, user)) == 2


def test_an_existing_inbox_keeps_its_random_id(db):
    """No migration. A user provisioned before this change keeps the id they have."""
    user = "incumbent"
    legacy = workspace_service.create_workspace(db, user, "Inbox")
    assert legacy.id != up.inbox_workspace_id(user), "fixture did not build a legacy id"

    assert up._ensure_inbox_workspace(db, user) == legacy.id
    assert len(workspace_service.list_workspaces(db, user)) == 1


def test_the_derived_workspace_is_a_real_workspace(db):
    """Pinning the id must not change what gets built — same content type, same owner grant."""
    user = "shape"
    ws_id = up._ensure_inbox_workspace(db, user)
    entity = workspace_service.get_workspace_unsafe(db, ws_id)
    assert entity is not None and entity.content_type == WORKSPACE_CONTENT_TYPE
    # `get_workspace` is the CHECKED read — it resolves only if the owner grant was issued.
    assert workspace_service.get_workspace(db, user, ws_id, required="admin").id == ws_id


# ---------------------------------------------------------------------------
# The same defect, found while measuring the one above: the Observations container.
# ---------------------------------------------------------------------------

def test_the_observations_container_id_is_derived_and_per_principal():
    from mantle.events import observation

    a = observation.observations_container_id("p-a")
    assert a == observation.observations_container_id("p-a")
    assert a != observation.observations_container_id("p-b")


def test_two_racing_observation_provisions_converge(db, monkeypatch):
    """`ensure_observations_container` was check-then-act too, and its docstring argued FOR
    find-then-create using the very outcome find-then-create produces."""
    from mantle.events import observation

    principal = "obs-racer"
    monkeypatch.setattr(observation, "_lookup_container", lambda *_a, **_k: None)
    monkeypatch.setattr(observation, "_container_cache", {})
    first = observation.ensure_observations_container(db, principal)
    second = observation.ensure_observations_container(db, principal)
    monkeypatch.undo()

    assert first == second == observation.observations_container_id(principal)
    owned = lattice_api.get_collections_by_owner_and_type(
        db, principal, observation.OBSERVATIONS_CONTENT_TYPE)
    assert len(owned) == 1, (
        "the race left %d Observations containers: %s" % (len(owned), [c.id for c in owned]))


def test_a_caller_cannot_derive_its_way_onto_the_inbox():
    """The security question a pinned id raises: since a create is an UPSERT, anyone who can
    name the Inbox's id can overwrite it. Nobody can.

    `derive_artifact_id` is the only caller-facing derivation, and its `identity` half is
    deliberately opaque and caller-chosen. It derives in `ARTIFACT_IDENTITY_NS`, a different
    uuid5 namespace from the `NAMESPACE_URL` the provisioning ids use, so no `identity` string
    reaches them — not by guessing the path, and not by embedding it.
    """
    from mantle.services import artifact_identity as ai

    user = "11111111-1111-1111-1111-111111111111"
    target = up.inbox_workspace_id(user)
    for probe in (
        "agience://workspace/inbox/" + user,
        user,
        "Inbox",
    ):
        assert ai.derive_artifact_id(user, probe) != target, probe
        assert ai.derive_artifact_id("other-principal", probe) != target, probe

    # ...and the two provisioning derivations do not collide with each other.
    assert up.inbox_workspace_id(user) != up.person_artifact_id(user)
