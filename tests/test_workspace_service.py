"""Unit tests for services.workspace_service.

Covers the artifact-lifecycle spine:
  - _safe_parse_context tolerates None / bad JSON
  - Workspace CRUD: create, get/owner-mismatch 404, update, delete
  - Artifact create / list / get with collection-scope guard
  - update_artifact state transitions:
      draft  → edited (dirty in-place)
      committed → _ensure_draft promotes to new draft (no committed mutation)
      archive toggle
      no-op when nothing dirty
      409 on editing archived
  - delete_artifact removes edges only when no other versions remain
  - revert_artifact drops draft and returns latest committed
  - move_workspace_artifact picks a fractional mid_key between neighbours
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.entities.collection import Collection as CollectionEntity, WORKSPACE_CONTENT_TYPE
from mantle.entities.grant import Grant as GrantEntity
from mantle.services import workspace_service as ws_svc


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_s3_put():
    """Stub out content_service.put_text_direct so no real S3 calls are made.

    workspace_service._store_content_in_s3 uploads content to S3 on every
    create/update. For small text (< 128 KB) the content is also kept inline;
    for large content it is cleared from the artifact (stored in S3 only).
    """
    with patch("mantle.services.content_service.put_text_direct") as mock_put:
        mock_put.return_value = None
        yield mock_put


@pytest.fixture(autouse=True)
def _patch_grants():
    """Return a permissive full-CRUDEASIO grant for all workspace service tests.

    get_workspace now verifies the caller holds the lattice grant. Most tests
    don't care about the grant check itself — they just need get_workspace to
    return the workspace entity. This fixture makes the grant lookup succeed
    transparently for all tests that don't override it.

    Tests that specifically validate the no-grant 404 path must explicitly
    override this fixture or patch the same target with return_value=[].
    """
    allow_grant = GrantEntity(
        resource_id="ws-1",
        grantee_type="user",
        grantee_id="user-1",
        granted_by="user-1",
        effect="allow",
        can_read=True,
        can_create=True,
        can_update=True,
        can_delete=True,
        can_evict=True,
        can_invoke=True,
        can_add=True,
        can_share=True,
        can_admin=True,
    )
    with patch(
        "mantle.services.workspace_service.store.get_active_grants_for_principal_resource",
        return_value=[allow_grant],
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ws(owner: str = "user-1", wid: str = "ws-1") -> CollectionEntity:
    return CollectionEntity(
        id=wid,
        name="My WS",
        created_by=owner,
        content_type=WORKSPACE_CONTENT_TYPE,
        context="",
    )


def _full_grant() -> GrantEntity:
    """A full-CRUDEASIO grant — what a creator gets at creation time."""
    return GrantEntity(
        resource_id="ws-1",
        grantee_type="user",
        grantee_id="user-1",
        granted_by="user-1",
        can_create=True,
        can_read=True,
        can_update=True,
        can_delete=True,
        can_invoke=True,
        can_add=True,
        can_share=True,
        can_admin=True,
        state="active",
    )


def _artifact(
    aid: str = "a-1",
    root_id: str | None = None,
    state: str = ArtifactEntity.STATE_DRAFT,
    collection_id: str = "ws-1",
    context: str = '{"content_type":"text/plain"}',
    content: str = "hello",
) -> ArtifactEntity:
    return ArtifactEntity(
        id=aid,
        root_id=root_id or aid,
        collection_id=collection_id,
        context=context,
        content=content,
        state=state,
        created_by="user-1",
        modified_by="user-1",
        created_time="2026-04-07T00:00:00+00:00",
        modified_time="2026-04-07T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_safe_parse_context_handles_none(self):
        assert ws_svc._safe_parse_context(None) == {}

    def test_safe_parse_context_handles_bad_json(self):
        assert ws_svc._safe_parse_context("not-json") == {}

    def test_safe_parse_context_handles_non_object_json(self):
        assert ws_svc._safe_parse_context("[1,2,3]") == {}

    def test_safe_parse_context_returns_dict(self):
        assert ws_svc._safe_parse_context('{"k":1}') == {"k": 1}

    def test_ensure_json_str_default(self):
        assert ws_svc._ensure_json_str(None) == "{}"

    def test_now_iso_returns_string(self):
        now = ws_svc._now_iso()
        assert isinstance(now, str) and "T" in now


# ---------------------------------------------------------------------------
# Workspace CRUD
# ---------------------------------------------------------------------------

class TestWorkspaceCrud:
    def test_create_workspace_always_generates_uuid(self):
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.store.create_collection") as create,
            patch("mantle.services.collection_service.ensure_collection_descriptor") as ensure,
            patch("mantle.services.workspace_service.store.upsert_user_collection_grant"),
        ):
            ws = ws_svc.create_workspace(db, "user-1", "Inbox")
        assert ws.id != "user-1"
        assert ws.content_type == WORKSPACE_CONTENT_TYPE
        create.assert_called_once()
        ensure.assert_called_once_with(db, ws)

    def test_create_workspace_generates_uuid_for_normal(self):
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.store.create_collection"),
            patch("mantle.services.collection_service.ensure_collection_descriptor") as ensure,
            patch("mantle.services.workspace_service.store.upsert_user_collection_grant"),
        ):
            ws = ws_svc.create_workspace(db, "user-1", "Project")
        assert ws.id != "user-1"
        assert ws.created_by == "user-1"
        ensure.assert_called_once_with(db, ws)

    def test_get_workspace_missing_404(self):
        db = MagicMock()
        with patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=None):
            with pytest.raises(HTTPException) as ei:
                ws_svc.get_workspace(db, "user-1", "ws-1")
        assert ei.value.status_code == 404

    def test_create_workspace_artifact_grants_owner_directly(self):
        # A child artifact's creator gets a DIRECT owner grant (full CRUDEASIO) on the
        # artifact — ownership is a grant on the artifact, not just edge propagation.
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.get_workspace"),
            patch("mantle.services.workspace_service.store.create_artifact"),
            patch("mantle.services.workspace_service.store.add_artifact_to_collection"),
            patch("mantle.services.workspace_service._link_to_target_collections"),
            patch("mantle.services.workspace_service._emit_event"),
            patch("mantle.services.workspace_service.store.upsert_user_collection_grant") as grant,
        ):
            art = ws_svc.create_workspace_artifact(
                db, "user-1", "ws-1", context="{}", content="", order_key="a0",
                enqueue_index=False,
            )
        grant.assert_called_once()
        kw = grant.call_args.kwargs
        assert kw["user_id"] == "user-1"
        assert kw["collection_id"] == art.root_id
        assert kw["can_admin"] is True and kw["can_read"] is True

    def test_get_workspace_no_grant_404(self):
        """Workspace exists but user has no the lattice grant → 404.

        Overrides the autouse _patch_grants fixture with an empty list so
        get_workspace sees no grants and raises 404.
        """
        db = MagicMock()
        ws = _ws()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch(
                "mantle.services.workspace_service.store.get_active_grants_for_principal_resource",
                return_value=[],
            ),
        ):
            with pytest.raises(HTTPException) as ei:
                ws_svc.get_workspace(db, "user-1", "ws-1")
        assert ei.value.status_code == 404

    def test_update_workspace_renames_when_dirty(self):
        db = MagicMock()
        ws = _ws()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch("mantle.services.workspace_service.store.update_collection") as update,
            patch("mantle.services.collection_service.ensure_collection_descriptor") as ensure,
        ):
            out = ws_svc.update_workspace(db, "user-1", "ws-1", name="New", description=None)
        assert out.name == "New"
        update.assert_called_once()
        ensure.assert_called_once_with(db, ws)

    def test_update_workspace_noop_when_no_change(self):
        db = MagicMock()
        ws = _ws()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch("mantle.services.workspace_service.store.update_collection") as update,
            patch("mantle.services.collection_service.ensure_collection_descriptor") as ensure,
        ):
            ws_svc.update_workspace(db, "user-1", "ws-1", name="My WS", description=None)
        update.assert_not_called()
        ensure.assert_not_called()

    def test_update_workspace_replaces_content(self):
        """A top-level artifact's body can be rewritten, and the rewrite reindexes.

        Every artifact created without a `container_id` is a top-level one carrying content
        (`create_container`'s own `content` parameter), so this is the ordinary edit path for a
        note, a transcript or a captured file — not an exotic case. Without a `content` parameter
        on `update_workspace`, the router's top-level branch would have nothing to hand a new body
        to and would return 200 unchanged. A writer that cannot rewrite an artifact has to create a
        second one, which is how a store ends up holding several copies of the same file with
        nothing marking which is current.
        """
        db = MagicMock()
        ws = _ws()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch("mantle.services.workspace_service.store.update_collection") as update,
            patch("mantle.services.collection_service.ensure_collection_descriptor") as ensure,
        ):
            out = ws_svc.update_workspace(
                db, "user-1", "ws-1", name=None, description=None, content="rewritten body",
            )
        assert out.content == "rewritten body"
        update.assert_called_once()
        ensure.assert_called_once_with(db, ws)

    def test_update_workspace_archives_a_top_level_artifact(self):
        """`state: "archived"` on a top-level artifact retires it, and moves its index segment.

        This was a silent no-op: the router's top-level branch never passed `state`, and
        `update_workspace` had no parameter to receive it, so the call returned 200 with the
        row untouched. Archiving is the only remediation that retires a superseded copy
        WITHOUT destroying it — `delete_artifact` was the sole working alternative — so every
        artifact created outside a collection, which is every note, transcript and captured
        file, could only be forgotten by being erased.

        The segment move is half the fix and the half that matters for recall: each state is a
        separately keyed tree, so an archived artifact is not filtered out of a committed
        search, it is absent from the tree being searched.
        """
        db = MagicMock()
        ws = _ws()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch("mantle.services.workspace_service.store.update_collection") as update,
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as enqueue,
        ):
            out = ws_svc.update_workspace(
                db, "user-1", "ws-1", name=None, description=None, state="archived",
            )
        assert out.state == CollectionEntity.STATE_ARCHIVED
        update.assert_called_once()
        enqueue.assert_called_once()
        # The container id is the artifact's OWN id: a top-level artifact is its own
        # container, and passing `collection_id` (empty here) would move nothing.
        assert enqueue.call_args.args[1] == "ws-1"
        assert enqueue.call_args.kwargs["vacate"] == ["committed", "draft"]

    def test_update_workspace_unarchives_to_committed_not_draft(self):
        """Unarchiving a top-level artifact returns it to `committed`.

        Deliberately different from `update_artifact`, which unarchives a collection member to
        `draft`. That is right for a member because a commit path is waiting for it; a
        top-level artifact has none — `create_container` writes `committed` directly and
        nothing here promotes a draft — so unarchiving into draft would strand it in a segment
        the default recall does not search and no path could move it out of.
        """
        db = MagicMock()
        ws = _ws()
        ws.state = CollectionEntity.STATE_ARCHIVED
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch("mantle.services.workspace_service.store.update_collection"),
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as enqueue,
        ):
            out = ws_svc.update_workspace(
                db, "user-1", "ws-1", name=None, description=None, state="committed",
            )
        assert out.state == CollectionEntity.STATE_COMMITTED
        assert enqueue.call_args.kwargs["vacate"] == ["archived"]

    def test_update_workspace_refuses_to_edit_an_archived_artifact(self):
        """Editing an archived top-level artifact is a 409, not a silent revival.

        Matches `update_artifact`'s behaviour for a collection member, for the same reason: an
        archived artifact has been retired, and a write that quietly un-retired it would defeat
        the retirement.
        """
        db = MagicMock()
        ws = _ws()
        ws.state = CollectionEntity.STATE_ARCHIVED
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch("mantle.services.workspace_service.store.update_collection") as update,
        ):
            with pytest.raises(HTTPException) as exc:
                ws_svc.update_workspace(
                    db, "user-1", "ws-1", name=None, description=None, content="new body",
                )
        assert exc.value.status_code == 409
        update.assert_not_called()

    def test_update_workspace_noop_when_content_unchanged(self):
        """Re-storing an identical body is not an edit — no write, no reindex.

        The hook that captures every Write/Edit re-sends a file's whole content each time, and
        an editor that saves without changing anything is common, so this is the difference
        between a store that grows only when something actually changed and one that churns a
        version per keystroke.
        """
        db = MagicMock()
        ws = _ws()
        ws.content = "same body"
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws),
            patch("mantle.services.workspace_service.store.update_collection") as update,
            patch("mantle.services.collection_service.ensure_collection_descriptor") as ensure,
        ):
            ws_svc.update_workspace(
                db, "user-1", "ws-1", name=None, description=None, content="same body",
            )
        update.assert_not_called()
        ensure.assert_not_called()


# ---------------------------------------------------------------------------
# Artifact CRUD
# ---------------------------------------------------------------------------

class TestArtifactCrud:
    def test_get_workspace_artifact_returns_when_in_workspace(self):
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=_artifact()),
        ):
            out = ws_svc.get_workspace_artifact(db, "user-1", "ws-1", "a-1")
        assert out.id == "a-1"

    def test_get_workspace_artifact_404_when_collection_mismatch(self):
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch(
                "mantle.services.workspace_service.store.get_artifact",
                return_value=_artifact(collection_id="other-ws"),
            ),
        ):
            with pytest.raises(HTTPException) as ei:
                ws_svc.get_workspace_artifact(db, "user-1", "ws-1", "a-1")
        assert ei.value.status_code == 404


# ---------------------------------------------------------------------------
# update_artifact state machine
# ---------------------------------------------------------------------------

class TestUpdateArtifactStateMachine:
    @pytest.fixture(autouse=True)
    def _silence_side_effects(self):
        with (
            patch("mantle.services.workspace_service._emit_event"),
        ):
            yield

    def test_archive_toggle_marks_archived(self):
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_DRAFT)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1",
                state=ArtifactEntity.STATE_ARCHIVED, reindex=False,
            )
        assert out.state == ArtifactEntity.STATE_ARCHIVED
        upd.assert_called_once()

    def test_unarchive_back_to_draft(self):
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_ARCHIVED)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact"),
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1",
                state=ArtifactEntity.STATE_DRAFT, reindex=False,
            )
        assert out.state == ArtifactEntity.STATE_DRAFT

    def test_archive_reindexes_and_vacates_committed_and_draft(self):
        """Archiving (re)indexes into the archived segment and, in the same job,
        vacates the segments the root is leaving (committed + draft)."""
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_DRAFT)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact"),
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as idx,
        ):
            ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1", state=ArtifactEntity.STATE_ARCHIVED,
            )
        idx.assert_called_once()
        assert idx.call_args.kwargs["vacate"] == ["committed", "draft"]

    def test_editing_archived_without_unarchive_raises_409(self):
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_ARCHIVED)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
        ):
            with pytest.raises(HTTPException) as ei:
                ws_svc.update_artifact(
                    db, "user-1", "ws-1", "a-1", content="new"
                )
        assert ei.value.status_code == 409

    def test_editing_committed_promotes_to_new_draft_with_same_root(self):
        db = MagicMock()
        committed = _artifact(state=ArtifactEntity.STATE_COMMITTED)
        # No existing draft for this root.
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=committed),
            patch("mantle.services.workspace_service.store.get_draft_artifact", return_value=None),
            patch("mantle.services.workspace_service.store.create_artifact") as create_new,
            patch("mantle.services.workspace_service.store.update_artifact"),
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1", content="edited", reindex=False
            )
        # A new draft was created with the same root_id, distinct id.
        create_new.assert_called_once()
        new_draft = create_new.call_args[0][1]
        assert new_draft.root_id == committed.root_id
        assert new_draft.id != committed.id
        assert new_draft.state == ArtifactEntity.STATE_DRAFT
        assert out.content == "edited"

    @pytest.mark.real_content_store
    def test_store_content_in_s3_addresses_the_bytes(self):
        """Content goes to the CAS and the caller gets its address back.

        There is no inline/clear split by size: the row never carries a body at any size, so
        there is no threshold to test and no size at which the answer differs.
        """
        content = "x" * 1024
        with patch("mantle.services.content_service.put_bytes_encrypted",
                   return_value="cas/" + "d" * 64) as put_cas:
            key, echoed, ref = ws_svc._store_content_in_s3(
                "a-1", content, '{"content_type":"text/plain"}', owner_id="owner-1")

        assert key == "artifacts/a-1.content"
        assert ref == "cas/" + "d" * 64
        assert echoed == content, "the caller keeps the body for indexing and the API response"
        assert put_cas.call_args.kwargs["cas"] is True
    def test_editing_committed_reuses_existing_draft(self):
        db = MagicMock()
        committed = _artifact(aid="committed-id", state=ArtifactEntity.STATE_COMMITTED)
        existing_draft = _artifact(aid="draft-id", root_id=committed.root_id)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=committed),
            patch(
                "mantle.services.workspace_service.store.get_draft_artifact",
                return_value=existing_draft,
            ),
            patch("mantle.services.workspace_service.store.create_artifact") as create_new,
            patch("mantle.services.workspace_service.store.update_artifact"),
        ):
            ws_svc.update_artifact(
                db, "user-1", "ws-1", "committed-id", content="x", reindex=False
            )
        create_new.assert_not_called()

    def test_root_id_update_prefers_existing_draft_in_workspace(self):
        db = MagicMock()
        draft = _artifact(aid="draft-id", root_id="root-1", content="old")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=None),
            patch("mantle.services.workspace_service.store.get_draft_artifact", return_value=draft) as get_draft,
            patch("mantle.services.workspace_service.store.get_latest_committed_artifact") as get_committed,
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "root-1", content="new", reindex=False
            )
        get_draft.assert_called_once_with(db, "root-1", "ws-1")
        get_committed.assert_not_called()
        upd.assert_called_once()
        assert out.id == "draft-id"
        assert out.content == "new"

    def test_root_id_update_falls_back_to_latest_committed(self):
        db = MagicMock()
        committed = _artifact(aid="committed-id", root_id="root-1", state=ArtifactEntity.STATE_COMMITTED)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=None),
            patch("mantle.services.workspace_service.store.get_draft_artifact", return_value=None) as get_draft,
            patch(
                "mantle.services.workspace_service.store.get_latest_committed_artifact",
                return_value=committed,
            ) as get_committed,
            patch("mantle.services.workspace_service.store.create_artifact") as create_new,
            patch("mantle.services.workspace_service.store.update_artifact"),
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "root-1", content="edited", reindex=False
            )
        assert get_draft.call_args_list[0].args == (db, "root-1", "ws-1")
        get_committed.assert_called_once_with(db, "root-1", "ws-1")
        create_new.assert_called_once()
        assert out.root_id == "root-1"
        assert out.state == ArtifactEntity.STATE_DRAFT
        assert out.content == "edited"

    def test_noop_when_nothing_dirty(self):
        db = MagicMock()
        art = _artifact(content="same", context="same-ctx")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1", content="same", context="same-ctx", reindex=False
            )
        upd.assert_not_called()
        assert out is art

    def test_update_content_persists_new_value(self):
        """Editing content on a draft artifact stores the new value and marks dirty."""
        db = MagicMock()
        art = _artifact(content="old content")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1", content="new content", reindex=False
            )
        upd.assert_called_once()
        assert out.content == "new content"

    def test_update_context_persists_new_value(self):
        """Editing context on a draft artifact stores the new value and marks dirty."""
        db = MagicMock()
        art = _artifact(context='{"content_type":"text/plain","title":"old"}')
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
        ):
            new_ctx = '{"content_type":"text/plain","title":"new"}'
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1", context=new_ctx, reindex=False
            )
        upd.assert_called_once()
        assert out.context == new_ctx

    def test_update_content_and_context_together(self):
        """Both content and context can be updated in a single call."""
        db = MagicMock()
        art = _artifact(content="old", context='{"content_type":"text/plain"}')
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1",
                content="new", context='{"content_type":"text/markdown"}',
                reindex=False,
            )
        upd.assert_called_once()
        assert out.content == "new"

    def test_commit_draft_flips_state_and_reindexes(self):
        """PATCH state:committed on a draft flips it to committed and triggers indexing."""
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_DRAFT)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as idx,
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1", state=ArtifactEntity.STATE_COMMITTED
            )
        assert out.state == ArtifactEntity.STATE_COMMITTED
        upd.assert_called_once()
        idx.assert_called_once()

    def test_commit_already_committed_is_noop(self):
        """PATCH state:committed on an already-committed artifact is a no-op."""
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_COMMITTED)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.update_artifact") as upd,
        ):
            out = ws_svc.update_artifact(
                db, "user-1", "ws-1", "a-1", state=ArtifactEntity.STATE_COMMITTED
            )
        assert out.state == ArtifactEntity.STATE_COMMITTED
        upd.assert_not_called()


# ---------------------------------------------------------------------------
# delete_artifact / revert_artifact
# ---------------------------------------------------------------------------

class TestDeleteRevert:
    @pytest.fixture(autouse=True)
    def _silence(self):
        with (
            patch("mantle.services.workspace_service._emit_event"),
            patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact"),
        ):
            yield

    def test_delete_drops_edges_when_no_other_versions(self):
        db = MagicMock()
        art = _artifact()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.list_version_history", return_value=[art]),
            patch("mantle.services.workspace_service.store.get_draft_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.list_collection_artifacts", return_value=[]),
            patch("mantle.services.workspace_service.store.delete_artifact"),
            patch("mantle.services.workspace_service.store.remove_all_edges_for_root") as remove_edges,
        ):
            ws_svc.delete_artifact(db, "user-1", "ws-1", "a-1")
        remove_edges.assert_called_once_with(db, art.root_id)

    def test_delete_detaches_a_sub_collections_members_by_default(self):
        """`a-1` is itself a container here, so it has members. Unchecked, the delete removes the
        container's own row and leaves the members' edges pointing at an id that resolves to
        nothing; the default detaches them instead, the same rule `delete_workspace` applies to a
        top-level container."""
        db = MagicMock()
        art = _artifact()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.list_version_history", return_value=[art]),
            patch("mantle.services.workspace_service.store.get_draft_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.list_collection_artifacts",
                  return_value=[{"id": "child-1", "root_id": "child-root-1"}]),
            patch("mantle.services.workspace_service.store.delete_artifact"),
            patch("mantle.services.workspace_service.store.remove_all_edges_for_root"),
            patch("mantle.services.workspace_service.store.delete_artifacts_by_root") as destroy,
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection") as evict,
            patch("mantle.services.workspace_service.store.count_other_containers_for_root") as shared_check,
        ):
            ws_svc.delete_artifact(db, "user-1", "ws-1", "a-1")
        assert not destroy.called, "a sub-collection's member was destroyed by the default delete"
        assert not shared_check.called, "share-count only matters to the cascade branch"
        evict.assert_called_once_with(db, "a-1", "child-root-1")

    def test_delete_cascade_true_destroys_a_sub_collections_exclusive_members(self):
        db = MagicMock()
        art = _artifact()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.list_version_history", return_value=[art]),
            patch("mantle.services.workspace_service.store.get_draft_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.list_collection_artifacts",
                  return_value=[{"id": "child-1", "root_id": "child-root-1"}]),
            patch("mantle.services.workspace_service.store.delete_artifact"),
            patch("mantle.services.workspace_service.store.remove_all_edges_for_root") as nuke_edges,
            patch("mantle.services.workspace_service.store.delete_artifacts_by_root") as destroy,
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection") as evict,
            patch("mantle.services.workspace_service.store.count_other_containers_for_root", return_value=0),
            # A sub-collection's exclusive member is origin-parented in it, so the caller's grant
            # reaches it and the cascade may destroy it. Stated here because `db` is a MagicMock:
            # without it the origin lookup returns a mock that is not this container and the guard
            # fails closed, turning this into a refusal test.
            patch("mantle.services.workspace_service.store.get_origin_parent",
                  return_value=("a-1", 0)),
            patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
        ):
            ws_svc.delete_artifact(db, "user-1", "ws-1", "a-1", cascade=True)
        destroy.assert_any_call(db, "child-root-1")
        nuke_edges.assert_any_call(db, "child-root-1")
        evict.assert_not_called()

    def test_delete_refuses_without_the_containing_collection(self):
        """A delete that cannot name its container is refused, not degraded.

        Both cleanup arms — the object-storage content and the search index — are keyed on the
        collection id. Without it the vertex row goes and the artifact's plaintext chunks stay
        searchable, so the call reports a successful delete having left the readable copy in
        place. Raising is the only outcome that does not lie about what happened.
        """
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.get_workspace_artifact") as fetched,
            patch("mantle.services.workspace_service.store.delete_artifact") as deleted,
        ):
            with pytest.raises(HTTPException) as exc:
                ws_svc.delete_artifact(db, "user-1", "", "a-1")
        assert exc.value.status_code == 400
        fetched.assert_not_called()
        deleted.assert_not_called()

    def test_delete_keeps_edges_when_other_versions_exist(self):
        db = MagicMock()
        art = _artifact(aid="a-1")
        sibling = _artifact(aid="a-2", root_id=art.root_id)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch(
                "mantle.services.workspace_service.store.list_version_history",
                return_value=[art, sibling],
            ),
            patch("mantle.services.workspace_service.store.get_draft_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.delete_artifact"),
            patch("mantle.services.workspace_service.store.remove_all_edges_for_root") as remove_edges,
        ):
            ws_svc.delete_artifact(db, "user-1", "ws-1", "a-1")
        remove_edges.assert_not_called()

    def test_revert_drops_draft_returns_committed(self):
        db = MagicMock()
        draft = _artifact(state=ArtifactEntity.STATE_DRAFT)
        committed = _artifact(aid="committed-id", root_id=draft.root_id, state=ArtifactEntity.STATE_COMMITTED)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=draft),
            patch(
                "mantle.services.workspace_service.store.get_latest_committed_artifact",
                return_value=committed,
            ),
            patch("mantle.services.workspace_service.store.delete_artifact") as del_a,
        ):
            out = ws_svc.revert_artifact(db, db, "user-1", "ws-1", "a-1")
        assert out is committed
        del_a.assert_called_once_with(db, draft.id)

    def test_revert_returns_target_when_not_a_draft(self):
        db = MagicMock()
        committed = _artifact(state=ArtifactEntity.STATE_COMMITTED)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=committed),
        ):
            out = ws_svc.revert_artifact(db, db, "user-1", "ws-1", "a-1")
        assert out is committed

    def test_revert_returns_none_when_no_committed_anchor(self):
        db = MagicMock()
        draft = _artifact(state=ArtifactEntity.STATE_DRAFT)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=draft),
            patch(
                "mantle.services.workspace_service.store.get_latest_committed_artifact",
                return_value=None,
            ),
        ):
            assert ws_svc.revert_artifact(db, db, "user-1", "ws-1", "a-1") is None


# ---------------------------------------------------------------------------
# Move / order
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# remove_artifact_from_container
# ---------------------------------------------------------------------------

class TestRemoveArtifactFromWorkspace:
    @pytest.fixture(autouse=True)
    def _silence(self):
        with patch("mantle.services.workspace_service._emit_event"):
            yield

    def test_404_when_artifact_not_found(self):
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=None),
            patch("mantle.services.workspace_service.store.get_edge", return_value=None),
            patch("mantle.services.workspace_service.store.get_current_in_collection", return_value=None),
        ):
            with pytest.raises(HTTPException) as ei:
                ws_svc.remove_artifact_from_container(db, "user-1", "ws-1", "missing")
        assert ei.value.status_code == 404

    def test_removes_edge_for_committed_artifact(self):
        """Committed artifact: edge removed, artifact doc kept."""
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_COMMITTED)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.get_edge",
                  return_value={"root_id": art.root_id, "origin": True}),
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection") as rm_edge,
            patch("mantle.services.workspace_service.store.delete_artifact") as del_art,
        ):
            result = ws_svc.remove_artifact_from_container(db, "user-1", "ws-1", "a-1")

        rm_edge.assert_called_once_with(db, "ws-1", art.root_id)
        del_art.assert_not_called()
        assert result.id == art.id

    def test_removes_edge_and_deletes_draft_doc(self):
        """Draft artifact owned by this workspace: edge removed and draft doc deleted."""
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_DRAFT, collection_id="ws-1")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.get_edge", return_value={"_id": "ws-1/a-1"}),
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection") as rm_edge,
            patch("mantle.services.workspace_service.store.get_current_in_collection", return_value=art),
            patch("mantle.services.workspace_service.store.delete_artifact") as del_art,
            patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
            # Authorized caller: the draft cleanup is a deletion and needs `delete` on the
            # artifact itself — `evict` on the container authorizes the unlink, not the
            # destruction of a version. Passing no `auth` fails closed, which is what the
            # companion test below asserts.
            patch("mantle.services.dependencies.check_access", return_value=None),
        ):
            result = ws_svc.remove_artifact_from_container(
                db, "user-1", "ws-1", "a-1", auth=object())

        rm_edge.assert_called_once_with(db, "ws-1", art.root_id)
        del_art.assert_called_once_with(db, art.id)
        assert result.id == art.id

    def test_an_unauthorized_caller_unlinks_but_does_NOT_delete_the_draft(self):
        """The assertion that catches the gap `evict`-only authorization would leave open.

        `evict` on the container is not enough on its own: a caller who could evict could destroy
        a draft version homed there — another user's unfinished work — under a container
        permission, with the artifact's own grants never consulted.

        The edge must still come off (that is a container property); the draft must survive.
        """
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_DRAFT, collection_id="ws-1")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.get_edge", return_value={"_id": "ws-1/a-1"}),
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection") as rm_edge,
            patch("mantle.services.workspace_service.store.get_current_in_collection", return_value=art),
            patch("mantle.services.workspace_service.store.delete_artifact") as del_art,
            patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
            patch("mantle.services.dependencies.check_access",
                  side_effect=HTTPException(status_code=404, detail="Not found")),
        ):
            ws_svc.remove_artifact_from_container(db, "user-1", "ws-1", "a-1", auth=object())

        rm_edge.assert_called_once_with(db, "ws-1", art.root_id)
        del_art.assert_not_called()

    def test_no_auth_at_all_also_leaves_the_draft(self):
        """Fails CLOSED on the absence of a caller context, not open. "Could not tell" must never
        resolve to "delete it"."""
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_DRAFT, collection_id="ws-1")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.get_edge", return_value={"_id": "ws-1/a-1"}),
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection"),
            patch("mantle.services.workspace_service.store.get_current_in_collection", return_value=art),
            patch("mantle.services.workspace_service.store.delete_artifact") as del_art,
        ):
            ws_svc.remove_artifact_from_container(db, "user-1", "ws-1", "a-1")

        del_art.assert_not_called()

    def test_falls_back_to_get_current_when_not_in_workspace(self):
        """Artifact in different collection: fallback to get_current_in_collection."""
        db = MagicMock()
        art = _artifact(state=ArtifactEntity.STATE_COMMITTED, collection_id="other-ws")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch("mantle.services.workspace_service.store.get_edge",
                  return_value={"root_id": art.root_id, "origin": True}),
            patch(
                "mantle.services.workspace_service.store.get_current_in_collection",
                return_value=_artifact(state=ArtifactEntity.STATE_COMMITTED),
            ) as get_current,
            patch("mantle.services.workspace_service.store.remove_artifact_from_collection"),
            patch("mantle.services.workspace_service.store.delete_artifact") as del_art,
        ):
            ws_svc.remove_artifact_from_container(db, "user-1", "ws-1", "a-1")

        get_current.assert_called_once_with(db, "ws-1", "a-1")
        del_art.assert_not_called()


class TestMoveAndOrder:
    @pytest.fixture(autouse=True)
    def _silence(self):
        with patch("mantle.services.workspace_service._emit_event"):
            yield

    def test_move_artifact_404_when_not_found(self):
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=None),
        ):
            with pytest.raises(HTTPException) as ei:
                ws_svc.move_workspace_artifact(db, "user-1", "ws-1", "missing", None, None)
        assert ei.value.status_code == 404

    def test_move_artifact_picks_mid_key_between_neighbours(self):
        db = MagicMock()
        art = _artifact()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=_ws()),
            patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
            patch(
                "mantle.services.workspace_service.store.get_edge",
                side_effect=[{"order_key": "a"}, {"order_key": "z"}],
            ),
            patch("mantle.services.workspace_service.store.set_edge_order_key") as set_key,
        ):
            ws_svc.move_workspace_artifact(
                db, "user-1", "ws-1", "a-1", before_id="prev", after_id="next"
            )
        # Whatever mid_key('a','z') returns, it should land between them lexicographically.
        new_key = set_key.call_args[0][3]
        assert "a" < new_key < "z"


# ---------------------------------------------------------------------------
# Binding Resolution
# ---------------------------------------------------------------------------

def _col(owner: str = "user-1", cid: str = "col-1") -> CollectionEntity:
    return CollectionEntity(id=cid, name="Bound Col", created_by=owner)


def _grant(grantee: str = "user-1", resource_id: str = "col-1", can_read: bool = True) -> GrantEntity:
    return GrantEntity(
        resource_id=resource_id,
        grantee_type="user",
        grantee_id=grantee,
        granted_by="admin",
        can_read=can_read,
    )


def _ws_with_bindings(bindings: dict, owner: str = "user-1", wid: str = "ws-1") -> CollectionEntity:
    import json as _json
    ctx = _json.dumps({"collections": [], "bindings": bindings})
    return CollectionEntity(id=wid, name="My WS", created_by=owner, content_type=WORKSPACE_CONTENT_TYPE, context=ctx)


class TestBindingResolution:
    """Tests for resolve_binding() and resolve_all_bindings()."""

    @staticmethod
    def _grants_for_ws_only(ws_id: str = "ws-1"):
        """Return a side_effect that gives a full grant for the workspace
        and an empty list for any other resource (binding targets)."""
        full = GrantEntity(
            resource_id=ws_id,
            grantee_type="user", grantee_id="user-1", granted_by="user-1",
            can_create=True, can_read=True, can_update=True, can_delete=True,
            can_invoke=True, can_add=True, can_share=True, can_admin=True,
            state="active",
        )
        def _side_effect(_db, *, grantee_id, resource_id):
            if resource_id == ws_id:
                return [full]
            return []
        return _side_effect

    @staticmethod
    def _grants_for_all():
        """Return a side_effect that gives a full grant for any resource."""
        def _side_effect(_db, *, grantee_id, resource_id):
            return [GrantEntity(
                resource_id=resource_id,
                grantee_type="user", grantee_id=grantee_id, granted_by=grantee_id,
                can_create=True, can_read=True, can_update=True, can_delete=True,
                can_invoke=True, can_add=True, can_share=True, can_admin=True,
                state="active",
            )]
        return _side_effect

    def test_resolve_workspace_level(self):
        db = MagicMock()
        ws = _ws_with_bindings({"memory": {"artifact_id": "col-1"}})
        col = _col()
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=lambda _db, cid: ws if cid == "ws-1" else col),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=self._grants_for_all()),
        ):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "memory")
        assert result == "col-1"

    def test_resolve_missing_role_returns_none(self):
        db = MagicMock()
        ws = _ws_with_bindings({"memory": {"artifact_id": "col-1"}})
        with patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "tools")
        assert result is None

    def test_resolve_no_bindings_key_returns_none(self):
        db = MagicMock()
        ws = _ws(owner="user-1")
        with patch("mantle.services.workspace_service.store.get_collection_by_id", return_value=ws):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "memory")
        assert result is None

    def test_resolve_access_denied_returns_none(self):
        db = MagicMock()
        ws = _ws_with_bindings({"memory": {"artifact_id": "col-1"}})
        col = _col(owner="other-user")
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=lambda _db, cid: ws if cid == "ws-1" else col),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=self._grants_for_ws_only()),
        ):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "memory")
        assert result is None

    def test_resolve_collection_not_found_returns_none(self):
        db = MagicMock()
        ws = _ws_with_bindings({"memory": {"artifact_id": "col-gone"}})
        with patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=lambda _db, cid: ws if cid == "ws-1" else None):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "memory")
        assert result is None

    def test_transform_overrides_workspace(self):
        db = MagicMock()
        ws = _ws_with_bindings({"tools": {"artifact_id": "col-ws"}})
        col_t = _col(cid="col-transform")
        transform_ctx = {"bindings": {"tools": {"artifact_id": "col-transform"}}}
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=lambda _db, cid: ws if cid == "ws-1" else col_t),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=self._grants_for_all()),
        ):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "tools", transform_context=transform_ctx)
        assert result == "col-transform"

    def test_step_overrides_transform(self):
        db = MagicMock()
        ws = _ws_with_bindings({"tools": {"artifact_id": "col-ws"}})
        col_s = _col(cid="col-step")
        transform_ctx = {"bindings": {"tools": {"artifact_id": "col-transform"}}}
        step_ctx = {"bindings": {"tools": {"artifact_id": "col-step"}}}
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=lambda _db, cid: ws if cid == "ws-1" else col_s),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=self._grants_for_all()),
        ):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "tools", transform_context=transform_ctx, step_context=step_ctx)
        assert result == "col-step"

    def test_cascade_falls_through(self):
        db = MagicMock()
        ws = _ws_with_bindings({"memory": {"artifact_id": "col-1"}})
        col = _col()
        step_ctx = {"bindings": {}}
        transform_ctx = {"bindings": {}}
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=lambda _db, cid: ws if cid == "ws-1" else col),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=self._grants_for_all()),
        ):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "memory", transform_context=transform_ctx, step_context=step_ctx)
        assert result == "col-1"

    def test_resolve_all_returns_all_roles(self):
        db = MagicMock()
        ws = _ws_with_bindings({
            "memory": {"artifact_id": "col-m"},
            "tools": {"artifact_id": "col-t"},
            "data": {"artifact_id": "col-d"},
        })
        def _get_col(_db, cid):
            if cid == "ws-1":
                return ws
            return _col(cid=cid)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=_get_col),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=self._grants_for_all()),
        ):
            result = ws_svc.resolve_all_bindings(db, "user-1", "ws-1")
        assert result == {"memory": "col-m", "tools": "col-t", "data": "col-d"}

    def test_resolve_all_omits_inaccessible(self):
        db = MagicMock()
        ws = _ws_with_bindings({
            "memory": {"artifact_id": "col-m"},
            "tools": {"artifact_id": "col-t"},
        })
        col_m = _col(cid="col-m")
        col_t = _col(owner="other-user", cid="col-t")
        def _get_col(_db, cid):
            if cid == "ws-1":
                return ws
            if cid == "col-m":
                return col_m
            if cid == "col-t":
                return col_t
            return None
        # Grant for ws-1 and col-m, but NOT col-t
        def _grants(_db, *, grantee_id, resource_id):
            if resource_id in ("ws-1", "col-m"):
                return [GrantEntity(
                    resource_id=resource_id,
                    grantee_type="user", grantee_id=grantee_id, granted_by=grantee_id,
                    can_read=True, state="active",
                )]
            return []
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=_get_col),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=_grants),
        ):
            result = ws_svc.resolve_all_bindings(db, "user-1", "ws-1")
        assert result == {"memory": "col-m"}

    def test_resolve_all_merges_cascade(self):
        db = MagicMock()
        ws = _ws_with_bindings({"memory": {"artifact_id": "col-m"}})
        transform_ctx = {"bindings": {"tools": {"artifact_id": "col-t"}}}
        def _get_col(_db, cid):
            if cid == "ws-1":
                return ws
            return _col(cid=cid)
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=_get_col),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=self._grants_for_all()),
        ):
            result = ws_svc.resolve_all_bindings(db, "user-1", "ws-1", transform_context=transform_ctx)
        assert result == {"memory": "col-m", "tools": "col-t"}

    def test_resolve_grant_read_allows_access(self):
        db = MagicMock()
        ws = _ws_with_bindings({"memory": {"artifact_id": "col-1"}})
        col = _col(owner="other-user")
        grant = _grant(grantee="user-1", resource_id="col-1", can_read=True)
        # Grant for ws-1 (autouse provides it) and the specific binding target.
        def _grants(_db, *, grantee_id, resource_id):
            if resource_id == "col-1":
                return [grant]
            if resource_id == "ws-1":
                return [GrantEntity(
                    resource_id="ws-1",
                    grantee_type="user", grantee_id="user-1", granted_by="user-1",
                    can_read=True, state="active",
                )]
            return []
        with (
            patch("mantle.services.workspace_service.store.get_collection_by_id", side_effect=lambda _db, cid: ws if cid == "ws-1" else col),
            patch("mantle.services.workspace_service.store.get_active_grants_for_principal_resource", side_effect=_grants),
        ):
            result = ws_svc.resolve_binding(db, "user-1", "ws-1", "memory")
        assert result == "col-1"


class TestANamedLineageMaterialisesItsRoot:
    """A supplied `root_id` must become a real vertex, or every edge to it dangles.

    `Artifact` states the contract — "First version of an artifact: id == root_id. The root doc
    persists forever and is the stable target of `collection_artifacts` edges" — and
    `vertex.py::_place` writes the containment edge to `root_id or id`, "The member is the ROOT,
    not the version." `create_workspace_artifact` minted a fresh uuid4 regardless, so a caller
    passing `root_id` (the identity-addressed path, `upsert_identity_member`) created a version
    pointing at a root nothing ever wrote.

    Measured 2026-08-25 on 71/home: the `Claude Code` capture collection held 97 containment edges
    of which 0 pointed at an existing vertex, and 183 captures with 0 materialised roots. Every
    session transcript and file mirror the hooks had written was placed and unreachable, because
    authorization walks edges and every edge ended nowhere.
    """

    @staticmethod
    def _create(root_id, existing_root):
        db = MagicMock()
        with (
            patch("mantle.services.workspace_service.get_workspace"),
            patch("mantle.services.workspace_service.store.create_artifact"),
            patch("mantle.services.workspace_service.store.add_artifact_to_collection"),
            patch("mantle.services.workspace_service._link_to_target_collections"),
            patch("mantle.services.workspace_service._emit_event"),
            patch("mantle.services.workspace_service.store.upsert_user_collection_grant"),
            patch("mantle.services.workspace_service.store.get_raw_artifact",
                  return_value=existing_root),
        ):
            return ws_svc.create_workspace_artifact(
                db, "user-1", "ws-1", context="{}", content="", order_key="a0",
                root_id=root_id, enqueue_index=False,
            )

    def test_the_first_version_becomes_the_root(self):
        art = self._create("root-abc", existing_root=None)
        assert art.id == "root-abc", (
            "the containment edge is written to root_id; if no vertex answers to it, the "
            "artifact is placed nowhere"
        )
        assert art.root_id == "root-abc"

    def test_a_later_version_keeps_its_own_id(self):
        """A rewrite must not collide with the root it is a version of."""
        art = self._create("root-abc", existing_root={"id": "root-abc"})
        assert art.id != "root-abc"
        assert art.root_id == "root-abc"

    def test_no_root_id_is_unchanged(self):
        """The ordinary create still mints its own id and is its own root."""
        art = self._create(None, existing_root=None)
        assert art.id == art.root_id
        assert art.id != "root-abc"
