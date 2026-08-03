"""Capability artifacts participate in per-state search (the importer writes via the
db layer, so it must enqueue indexing explicitly)."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from mantle.entities.artifact import Artifact
from mantle.services import mcp_resource_importer as imp

TOOL_CT = "application/vnd.agience.tool+json"


class TestCapabilityIndexing:
    def test_create_indexes_into_committed(self):
        with (
            patch("mantle.services.mcp_resource_importer.db_get_artifact", return_value=None),
            patch("mantle.services.mcp_resource_importer.db_create_artifact"),
            patch("mantle.services.mcp_resource_importer.db_add_artifact_to_collection"),
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as idx,
        ):
            imp._upsert_capability_artifact(
                MagicMock(), artifact_id="cap-1", collection_id="col-1",
                content_type=TOOL_CT, context={"name": "x"}, content="c", user_id="u")
        idx.assert_called_once()
        assert idx.call_args[0][0].state == "committed"
        assert idx.call_args.kwargs.get("vacate") is None   # fresh create

    def test_recommit_from_archived_vacates_archived(self):
        existing = Artifact(id="cap-1", collection_id="col-1", state="archived",
                            context=json.dumps({"name": "old"}), content_type=TOOL_CT)
        with (
            patch("mantle.services.mcp_resource_importer.db_get_artifact", return_value=existing),
            patch("mantle.services.mcp_resource_importer.db_update_artifact"),
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as idx,
        ):
            imp._upsert_capability_artifact(
                MagicMock(), artifact_id="cap-1", collection_id="col-1",
                content_type=TOOL_CT, context={"name": "new"}, content="c", user_id="u")
        idx.assert_called_once()
        assert idx.call_args[0][0].state == "committed"
        assert idx.call_args.kwargs["vacate"] == ["archived", "draft"]

    def test_archive_stale_vacates_committed(self):
        stale = {"id": "cap-1", "collection_id": "col-1", "state": "committed",
                 "context": json.dumps({"name": "x"}), "content_type": TOOL_CT}
        with (
            patch("mantle.services.mcp_resource_importer._db_list_collection_artifacts",
                  return_value=[stale]),
            patch("mantle.services.mcp_resource_importer.db_update_artifact"),
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as idx,
        ):
            n = imp._archive_stale_capabilities(MagicMock(), "parent", set(), "u")
        assert n == 1
        assert idx.call_args[0][0].state == "archived"
        assert idx.call_args.kwargs["vacate"] == ["committed", "draft"]
