"""Born findable + first-observation (lazy materialization).

A child is born `live`, the segment a default search reads, with no promotion step between
saving it and finding it.

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
        r = api.search(token, scope=scope)
        if r.status_code == 503:
            pytest.skip("search backend unavailable")
        if r.status_code == 200:
            hits = r.json().get("hits", [])
            if hits:
                return True
        time.sleep(delay)
    return False


def test_a_child_is_born_live(user):
    coll = user["api"].create_collection(f"live-{uuid.uuid4().hex[:6]}")
    child = user["api"].create_child(coll["id"], name="doc", content="a body")
    assert child.get("state") == "live"

    # Read-back confirms the persisted state.
    got = user["api"].get_artifact(child["id"]).json()
    assert got.get("state") == "live"


def test_draft_is_refused(user):
    coll = user["api"].create_collection(f"live-{uuid.uuid4().hex[:6]}")
    child = user["api"].create_child(coll["id"], name="doc", content="a body")
    r = user["api"].patch(f"/artifacts/{child['id']}", json={"state": "draft"})
    assert r.status_code == 422, r.text


@pytest.mark.search
def test_a_new_child_is_searchable_with_no_further_step(user):
    tok = f"zqc{uuid.uuid4().hex[:8]}"
    coll = user["api"].create_collection(f"live-{uuid.uuid4().hex[:6]}")
    user["api"].create_child(coll["id"], name="doc", content=f"find me {tok}")
    assert _find(user["api"], tok, [coll["id"]]), "a new artifact is not searchable"


@pytest.mark.lazy
@pytest.mark.search
def test_first_observation_materializes(user):
    if not cfg.LAZY_INDEX:
        pytest.skip("requires MANTLE_LAZY_INDEX=on + E2E_LAZY_INDEX=1")

    tok = f"zql{uuid.uuid4().hex[:8]}"
    coll = user["api"].create_collection(f"lazy-{uuid.uuid4().hex[:6]}")
    # Latent from birth: lazy child, live but not yet indexed.
    child = user["api"].create_child(coll["id"], name="doc", content=f"latent {tok}", index="lazy")

    r = user["api"].search(tok, scope=[coll["id"]])
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
        user["api"].create_child(coll["id"], name="doc", content=f"warm {t}", index="lazy")
        toks.append(t)

    r = user["api"].post(f"/artifacts/{coll['id']}/warm")
    if r.status_code == 503:
        pytest.skip("search backend unavailable")
    assert r.status_code == 200, r.text
    assert r.json().get("materialized", 0) >= 1, r.json()
    # At least one warmed artifact is now findable.
    assert _find(user["api"], toks[0], [coll["id"]])
