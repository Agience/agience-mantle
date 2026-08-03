"""Commit lifecycle + first-observation (lazy materialization).

Commit: a child is born `draft`; `PATCH {state: committed}` promotes it. The
committed segment is what `state="committed"` search sees.

First-observation (only meaningful under `MANTLE_LAZY_INDEX=on`, gate with
`E2E_LAZY_INDEX=1`): a `lazy` child is latent — absent from search — until its
first authorized `GET` materializes it (valence 2 → 5), after which it is
findable. `/warm` bulk-materializes a collection.
"""
from __future__ import annotations

import time
import uuid

import pytest

import _config as cfg


def _find(api, token, scope, tries=6, delay=0.5) -> bool:
    """Search a few times (indexing can lag the write)."""
    for _ in range(tries):
        r = api.search(token, state="committed", scope=scope)
        if r.status_code == 503:
            pytest.skip("search backend unavailable")
        if r.status_code == 200:
            hits = r.json().get("hits", [])
            if hits:
                return True
        time.sleep(delay)
    return False


def test_state_transition_draft_to_committed(user):
    coll = user["api"].create_collection(f"commit-{uuid.uuid4().hex[:6]}")
    child = user["api"].create_child(coll["id"], name="doc", content="draft body")
    assert child.get("state") == "draft"

    r = user["api"].commit(child["id"])
    assert r.status_code == 200, r.text
    assert r.json().get("state") == "committed"

    # Read-back confirms the persisted state.
    got = user["api"].get_artifact(child["id"]).json()
    assert got.get("state") == "committed"


@pytest.mark.search
def test_commit_makes_it_searchable(user):
    tok = f"zqc{uuid.uuid4().hex[:8]}"
    coll = user["api"].create_collection(f"commit-{uuid.uuid4().hex[:6]}")
    child = user["api"].create_child(coll["id"], name="doc", content=f"commit me {tok}")

    # Draft: not in the committed segment.
    r = user["api"].search(tok, state="committed", scope=[coll["id"]])
    if r.status_code == 503:
        pytest.skip("search backend unavailable")
    assert not r.json().get("hits"), "draft leaked into committed search"

    user["api"].commit(child["id"])
    assert _find(user["api"], tok, [coll["id"]]), "committed artifact not searchable"


@pytest.mark.lazy
@pytest.mark.search
def test_first_observation_materializes(user):
    if not cfg.LAZY_INDEX:
        pytest.skip("requires MANTLE_LAZY_INDEX=on + E2E_LAZY_INDEX=1")

    tok = f"zql{uuid.uuid4().hex[:8]}"
    coll = user["api"].create_collection(f"lazy-{uuid.uuid4().hex[:6]}")
    # Latent from birth: lazy child, committed but not yet indexed.
    child = user["api"].create_child(coll["id"], name="doc", content=f"latent {tok}", index="lazy")
    user["api"].commit(child["id"])

    r = user["api"].search(tok, state="committed", scope=[coll["id"]])
    if r.status_code == 503:
        pytest.skip("search backend unavailable")
    assert not r.json().get("hits"), "latent (lazy) artifact should be invisible pre-observation"

    # First observation: an authorized GET materializes it.
    assert user["api"].get_artifact(child["id"]).status_code == 200
    assert _find(user["api"], tok, [coll["id"]]), "artifact not searchable after first observation"


@pytest.mark.lazy
@pytest.mark.search
def test_warm_bulk_materializes(user):
    if not cfg.LAZY_INDEX:
        pytest.skip("requires MANTLE_LAZY_INDEX=on + E2E_LAZY_INDEX=1")

    coll = user["api"].create_collection(f"warm-{uuid.uuid4().hex[:6]}")
    toks = []
    for _ in range(3):
        t = f"zqw{uuid.uuid4().hex[:8]}"
        c = user["api"].create_child(coll["id"], name="doc", content=f"warm {t}", index="lazy")
        user["api"].commit(c["id"])
        toks.append(t)

    r = user["api"].post(f"/artifacts/{coll['id']}/warm")
    if r.status_code == 503:
        pytest.skip("search backend unavailable")
    assert r.status_code == 200, r.text
    assert r.json().get("materialized", 0) >= 1, r.json()
    # At least one warmed artifact is now findable.
    assert _find(user["api"], toks[0], [coll["id"]])
