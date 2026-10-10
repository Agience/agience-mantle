"""ONE meaning for a doc that carries no `state`, and every reader derives from it.

`state` partitions the search index into separately-keyed encrypted trees, one per state (`live`,
`archived`). So "what state is a doc with no `state` field in" is not a stylistic question: two
readers that answer it differently move an artifact between encrypted index trees on a write that
changed nothing.

The failure this file pins: a doc written without `state` (`lattice_api.mark_materialized` writes
one; so does any raw `put_artifact`) is returned by `list_artifacts()` and filed by the indexer
under **live**. Round-tripping that same doc through `Artifact.from_dict` → `to_dict` must file it
in the same tree; a round trip that materialised any other state would relocate the artifact
although nothing about it changed, only which key opened it.

The same reading covers the legacy values: a stored `committed` or `draft` reads, round-trips and
indexes as `live`.

The single source is `db.constants.STATE_WHEN_ABSENT` (with `state_of`). It lives at the store
layer because that is the layer that stands alone (`db/vertex.py` cannot import `entities`) and
because the fact is about a STORED DOC, not about an entity.
"""
from __future__ import annotations

import pytest

from mantle.db import constants as K
from mantle.entities.artifact import Artifact
from mantle.search.ingest.pipeline_unified import _segment_for_state


def _stateless_doc() -> dict:
    """A doc the store can hold and the indexer can file, carrying no `state`."""
    return {
        "id": "a-1",
        "root_id": "a-1",
        "collection_id": "c-1",
        "context": "",
        "content": "hello",
        "content_type": "text/markdown",
        "created_by": "user-1",
    }


def _segment(doc: dict) -> str:
    """The index tree this doc's entry belongs in, read the way the ingester reads it."""
    return _segment_for_state(doc.get("state") or "")


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------

def test_stateless_doc_round_trip_does_not_change_index_segment():
    """Read a stateless doc, write it back unchanged, and it must still index into the same tree.

    A segment change here is not a cosmetic difference: the segments are encrypted under separate
    keys, so an artifact that moves is an artifact whose postings were re-keyed by a write that
    carried no edit.
    """
    raw = _stateless_doc()
    before = _segment(raw)

    round_tripped = Artifact.from_dict(raw).to_dict()
    after = _segment(round_tripped)

    assert after == before, (
        "a read-modify-write of a stateless doc relocated it from the %r index segment to %r"
        % (before, after))


# ---------------------------------------------------------------------------
# The single source, and each derived site
# ---------------------------------------------------------------------------

def test_absent_state_resolves_to_live():
    """The chosen default, named once. Both null-safe SQL predicates keep a stateless row as live
    rather than archived, and every write path writes `live`."""
    assert K.STATE_WHEN_ABSENT == Artifact.STATE_LIVE == "live"


def test_state_of_reads_absence_and_the_legacy_values_as_live():
    assert K.state_of(_stateless_doc()) == K.STATE_WHEN_ABSENT
    assert K.state_of({"state": "draft"}) == "live"
    assert K.state_of({"state": "committed"}) == "live"
    assert K.state_of({"state": "archived"}) == "archived"
    # An empty string is not a state; it is absence spelled differently.
    assert K.state_of({"state": ""}) == K.STATE_WHEN_ABSENT
    # A value on another axis (a grant's lifecycle) is not an artifact state and passes through.
    assert K.state_of({"state": "active"}) == "active"


@pytest.mark.parametrize("state", ["live", "archived"])
def test_a_recorded_state_survives_the_round_trip(state):
    """The default applies to ABSENCE only — it never overrides a state the doc records."""
    doc = dict(_stateless_doc(), state=state)
    assert Artifact.from_dict(doc).to_dict()["state"] == state
    assert _segment(Artifact.from_dict(doc).to_dict()) == state


@pytest.mark.parametrize("legacy", ["committed", "draft"])
def test_a_legacy_state_round_trips_and_indexes_as_live(legacy):
    doc = dict(_stateless_doc(), state=legacy)
    assert _segment(doc) == "live"
    assert Artifact.from_dict(doc).to_dict()["state"] == "live"


def test_entity_layer_derives_from_the_single_source():
    assert Artifact.from_dict(_stateless_doc()).state == K.STATE_WHEN_ABSENT


def test_index_segment_derives_from_the_single_source():
    """An unknown or absent state files into the segment the single source names."""
    assert _segment_for_state("") == K.STATE_WHEN_ABSENT
    assert _segment_for_state("not-a-state") == K.STATE_WHEN_ABSENT


def test_a_new_artifact_is_live():
    """A newly constructed artifact is live — the same answer absence gets, so creating an
    artifact and reading a stored doc cannot file it into different trees."""
    assert Artifact().state == Artifact.STATE_LIVE == K.STATE_WHEN_ABSENT


# ---------------------------------------------------------------------------
# Through the real store
# ---------------------------------------------------------------------------

def test_stateless_doc_survives_a_store_round_trip_in_one_segment(tmp_path):
    """End to end: put a stateless doc, list it back, round-trip it through the entity, put it
    again — and the tree it indexes into never moves."""
    from mantle.db import open_lattice

    lattice = open_lattice(str(tmp_path / "lattice.db"), origin="node-71")
    lattice.artifacts.put_artifact(_stateless_doc())

    listed = [d for d in lattice.artifacts.list_artifacts(collection_id="c-1")]
    assert len(listed) == 1, "a stateless row must stay visible to the head-only list"
    first = _segment(listed[0])

    lattice.artifacts.put_artifact(Artifact.from_dict(listed[0]).to_dict())
    relisted = [d for d in lattice.artifacts.list_artifacts(collection_id="c-1")]
    assert len(relisted) == 1
    assert _segment(relisted[0]) == first
