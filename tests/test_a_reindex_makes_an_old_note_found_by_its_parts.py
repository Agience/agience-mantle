"""The reindex migration makes a note found by the parts of a hyphenated word, as a fresh save does.

2026-10-31, a tester: fresh hyphenated Notes matched space-separated terms, but the unchanged old
`quasar-lantern-731` Note still matched only the full token, after the migration had recorded its
generation. A note whose index entries predate the change stands in here as one with none.
"""
from __future__ import annotations

import pytest

from mantle.search.ingest.pipeline_unified import delete_artifact_from_index
from mantle.services.acting_principal import acting_as
from mantle.services import workspace_service as ws
from mantle.system import reindex

from .test_a_hyphenated_word_is_found_by_its_parts import WORKSPACE_CT, _found, _note
from .test_the_index_is_rebuilt_once_per_generation import Settings
from .test_there_is_no_promotion_state import USER, real_node


@pytest.fixture
def db(tmp_path, monkeypatch):
    return real_node(tmp_path, monkeypatch)


async def test_the_migration_makes_an_old_note_found_by_its_parts(db):
    with acting_as(USER, principal_type="user"):
        home = ws.create_container(db, USER, content_type=WORKSPACE_CT, name="Inbox").id
    note = await _note(db, home, "Search Gamma", "Regression marker quasar-lantern-731 for search.")
    with acting_as(USER, principal_type="user"):
        delete_artifact_from_index(note, collection_id=home)
    assert note not in await _found(db, "quasar lantern"), "the stand-in for an old entry"

    out = reindex.reindex_once(db, Settings())
    assert out["rebuilt"] and out["failed"] == 0, out
    assert note in await _found(db, "quasar lantern"), out
