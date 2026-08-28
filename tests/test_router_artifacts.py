"""Tests for routers/artifacts_router.py — the unified artifact API.

Existing coverage in test_router_artifacts_op.py covers the
POST /artifacts/{id}/op/{op_name} dispatch path. This file covers everything
else: CRUD, container creation, invoke, add-to-container, batch, search,
upload, commit/preview, content-url, list-commits, reorder, revert, move,
upload-status, multipart-part-url.

The auth dependency is overridden by the autouse conftest fixture to a user
principal (`user-123`); we patch `routers.artifacts_router.check_access` to a
no-op so the grant check doesn't reach the DB.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from mantle.main import app
from mantle.services.dependencies import AuthContext, get_auth
from mantle.entities.collection import Collection as CollectionEntity, WORKSPACE_CONTENT_TYPE
from mantle.entities.artifact import Artifact as ArtifactEntity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_s3_put():
    """Prevent real S3 calls in workspace_service._store_content_in_s3.

    All router tests deal with the service layer, which now uploads content to
    S3 on create/update. Stub out put_text_direct so tests don't need MinIO.
    The returned content_key uses a predictable format so assertions still work.
    """
    with patch("mantle.services.content_service.put_text_direct") as mock_put:
        mock_put.return_value = None
        yield mock_put


@pytest.fixture(autouse=True)
def _patch_check_access():
    """`check_access` is the only side-effect path into the DB from this router.
    Replace it with a no-op grant for every test in this file."""
    grant = SimpleNamespace(
        can_read=True,
        can_create=True,
        can_update=True,
        can_delete=True,
        can_invoke=True,
        can_add=True,
        can_share=True,
        resource_id=None,
    )
    with patch("mantle.routers.artifacts_router.check_access", return_value=grant):
        yield grant


@pytest.fixture
def anon_client(client: AsyncClient):
    """Override auth to anonymous for negative tests."""
    app.dependency_overrides[get_auth] = lambda: AuthContext(
        user_id=None, principal_id=None, principal_type="anonymous"
    )
    yield client
    app.dependency_overrides.pop(get_auth, None)


def _coll_doc(content_type: str = WORKSPACE_CONTENT_TYPE) -> dict:
    return {
        "_key": "container-1",
        "name": "Container",
        "created_by": "user-123",
        "content_type": content_type,
    }


def _artifact_doc(state: str = "draft") -> dict:
    return {
        "_key": "art-1",
        "id": "art-1",
        "root_id": "art-1",
        "collection_id": "container-1",
        "context": '{"content_type":"text/plain"}',
        "content": "hello",
        "state": state,
        "created_by": "user-123",
    }


def _patch_db_collection(db_mock: MagicMock, *, container_doc=None, artifact_doc=None):
    """Wire up the lattice handle — container-as-artifact: all docs live in
    `db.artifacts` and are fetched via `get_artifact(id)`; root-id resolution
    falls back to `versions_of(root_id)`. Side effects are wired explicitly —
    never rely on MagicMock auto-attributes for asserted behavior."""
    docs: dict = {}
    for d in (container_doc, artifact_doc):
        if d:
            docs[d.get("id") or d.get("_key")] = d

    db_mock.artifacts.get_artifact.side_effect = lambda key: docs.get(key)
    db_mock.artifacts.versions_of.side_effect = lambda root: [
        d for d in docs.values() if d.get("root_id") == root
    ]
    db_mock.graph.edges_of.side_effect = lambda node, **kw: []


# ---------------------------------------------------------------------------
# Auth guards
# ---------------------------------------------------------------------------

class TestAuthGuards:
    @pytest.mark.asyncio
    async def test_create_requires_user(self, anon_client: AsyncClient):
        r = await anon_client.post("/artifacts", json={"container_id": "c-1"})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_update_requires_user(self, anon_client: AsyncClient):
        r = await anon_client.patch("/artifacts/a-1", json={"content": "x"})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_delete_requires_user(self, anon_client: AsyncClient):
        r = await anon_client.delete("/artifacts/a-1")
        assert r.status_code == 401




# ---------------------------------------------------------------------------
# POST /artifacts — create
# ---------------------------------------------------------------------------

class TestCreateArtifact:
    @pytest.mark.asyncio
    async def test_create_top_level_artifact(self, client: AsyncClient):
        # No container_id -> top-level artifact.
        ws = CollectionEntity(
            id="ws-1", name="My WS", created_by="user-123",
            content_type=WORKSPACE_CONTENT_TYPE, context="",
        )
        with patch("mantle.services.workspace_service.create_container", return_value=ws) as mk:
            r = await client.post(
                "/artifacts",
                json={"content_type": WORKSPACE_CONTENT_TYPE, "name": "My WS"},
            )
        assert r.status_code == 201
        assert r.json()["id"] == "ws-1"
        mk.assert_called_once()   # top-level create, not a child

    @pytest.mark.asyncio
    async def test_create_top_level_collection(self, client: AsyncClient):
        col = CollectionEntity(
            id="col-1", name="My Col", created_by="user-123",
            content_type="application/vnd.agience.collection+json", context="",
        )
        with patch("mantle.services.workspace_service.create_container", return_value=col):
            r = await client.post(
                "/artifacts",
                json={"content_type": "application/vnd.agience.collection+json", "name": "My Col"},
            )
        assert r.status_code == 201
        assert r.json()["id"] == "col-1"

    @pytest.mark.asyncio
    async def test_create_artifact_in_unknown_container_returns_404(
        self, client: AsyncClient
    ):
        store = MagicMock()
        _patch_db_collection(store, container_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.post(
                "/artifacts",
                json={"container_id": "missing", "content": "x"},
            )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404
        assert "Container not found" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_create_artifact_happy_path(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, container_doc=_coll_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        captured: dict = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return ArtifactEntity(
                id="art-1",
                root_id="art-1",
                collection_id="container-1",
                context='{"content_type":"text/plain"}',
                content="hello",
                state=ArtifactEntity.STATE_DRAFT,
            )

        try:
            with patch("mantle.services.workspace_service.create_workspace_artifact", side_effect=fake_create):
                r = await client.post(
                    "/artifacts",
                    json={
                        "container_id": "container-1",
                        "content": "hello",
                        "content_type": "text/plain",
                    },
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 201
        body = r.json()
        assert body["id"] == "art-1"
        assert body["collection_id"] == "container-1"
        assert "slug" not in captured

    @pytest.mark.asyncio
    async def test_create_artifact_merges_content_type_into_context(
        self, client: AsyncClient
    ):
        """When `content_type` is supplied alongside a JSON `context`, the
        router merges it into the context dict before persisting."""
        store = MagicMock()
        _patch_db_collection(store, container_doc=_coll_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        captured: dict = {}

        def fake_create(**kwargs):
            entity = ArtifactEntity(
                id="art-merge",
                root_id="art-merge",
                collection_id=kwargs.get("workspace_id", "container-1"),
                context=kwargs.get("context", "{}"),
                content=kwargs.get("content", ""),
                state=ArtifactEntity.STATE_DRAFT,
            )
            captured["context"] = kwargs.get("context", "")
            return entity

        try:
            with patch("mantle.services.workspace_service.create_workspace_artifact", side_effect=fake_create):
                await client.post(
                    "/artifacts",
                    json={
                        "container_id": "container-1",
                        "context": '{"existing":"field"}',
                        "content_type": "text/plain",
                        "content": "x",
                    },
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        ctx = captured["context"]
        assert '"content_type": "text/plain"' in ctx or "content_type" in ctx
        assert "existing" in ctx


# ---------------------------------------------------------------------------
# GET /artifacts/{id} — read
# ---------------------------------------------------------------------------

class TestReadArtifact:
    @pytest.mark.asyncio
    async def test_read_returns_normalized_doc(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc={
            "_id": "artifacts/a-1",
            "_key": "a-1",
            "_rev": "_abc",
            "id": "a-1",
            "context": "{}",
            "content": "x",
            "state": "draft",
        })
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.get("/artifacts/a-1")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        body = r.json()
        # Internal the lattice keys stripped.
        assert "_id" not in body
        assert "_rev" not in body
        assert "_key" not in body
        assert body["id"] == "a-1"

    @pytest.mark.asyncio
    async def test_read_404_when_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.get("/artifacts/missing")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_read_404_when_archived(self, client: AsyncClient):
        """Archived artifacts are filtered out by `_find_artifact`."""
        store = MagicMock()
        _patch_db_collection(store, artifact_doc={"_key": "a-1", "state": "archived"})
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.get("/artifacts/a-1")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_read_404_for_legacy_collection_prefixed_id(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, container_doc={
            "id": "container-1",
            "content_type": WORKSPACE_CONTENT_TYPE,
            "state": "draft",
        })
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.get("/artifacts/collection:container-1")
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /artifacts/{id} — update
# ---------------------------------------------------------------------------

class TestUpdateArtifact:
    @pytest.mark.asyncio
    async def test_update_routes_container_to_workspace_service(
        self, client: AsyncClient
    ):
        store = MagicMock()
        _patch_db_collection(store, container_doc=_coll_doc(WORKSPACE_CONTENT_TYPE))
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        updated = CollectionEntity(
            id="container-1",
            name="Renamed",
            created_by="user-123",
            content_type=WORKSPACE_CONTENT_TYPE,
            context="",
        )
        try:
            with patch(
                "mantle.services.workspace_service.update_workspace", return_value=updated
            ) as upd:
                r = await client.patch(
                    "/artifacts/container-1",
                    json={"name": "Renamed"},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        upd.assert_called_once()
        assert r.json()["name"] == "Renamed"

    @pytest.mark.asyncio
    async def test_update_top_level_artifact_forwards_content(self, client: AsyncClient):
        """A top-level artifact's new BODY reaches the service, rather than being dropped.

        The branch above it routes anything with no `collection_id` to `update_workspace`. Every
        artifact created without a `container_id` is top-level and content-bearing
        (`create_container` takes `content`), so a forward that carried only
        name/description/context/vector would leave `PATCH {"content": ...}` on a note, a
        transcript or a captured file returning 200 and changing nothing -- no error to notice, the
        old body still indexed and still what `recall` answers with.

        Asserting the forwarded argument is the point. The sibling test above patches the same
        function and asserts only that it was called, which a router dropping the one field
        under test passes just as happily.
        """
        store = MagicMock()
        _patch_db_collection(store, container_doc=_coll_doc(WORKSPACE_CONTENT_TYPE))
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        updated = CollectionEntity(
            id="container-1",
            name="Container",
            created_by="user-123",
            content_type=WORKSPACE_CONTENT_TYPE,
            context="",
            content="rewritten body",
        )
        try:
            with patch(
                "mantle.services.workspace_service.update_workspace", return_value=updated
            ) as upd:
                r = await client.patch(
                    "/artifacts/container-1",
                    json={"content": "rewritten body"},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        upd.assert_called_once()
        assert upd.call_args.kwargs["content"] == "rewritten body"
        assert r.json()["content"] == "rewritten body"

    @pytest.mark.asyncio
    async def test_update_artifact_routes_to_workspace_service(
        self, client: AsyncClient
    ):
        store = MagicMock()
        # `_is_collection` returns False (artifact has collection_id), `_find_artifact` returns the doc.
        _patch_db_collection(store, artifact_doc=_artifact_doc())

        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        updated = ArtifactEntity(
            id="art-1",
            root_id="art-1",
            collection_id="container-1",
            context="{}",
            content="new",
            state="draft",
        )
        try:
            with patch(
                "mantle.services.workspace_service.update_artifact", return_value=updated
            ):
                r = await client.patch(
                    "/artifacts/art-1",
                    json={"content": "new"},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        assert r.json()["content"] == "new"

    @pytest.mark.asyncio
    async def test_update_artifact_with_content_type_routes_to_artifact_update(
        self, client: AsyncClient
    ):
        """Artifacts created via the type picker have a top-level content_type
        (e.g. text/markdown). PATCH must still route to the artifact update
        path, not the container update path — `_is_collection` must not treat
        any non-None content_type as a container."""
        store = MagicMock()

        # Artifact with a content_type that is NOT a container type.
        typed_doc = {
            "_key": "typed-1",
            "id": "typed-1",
            "root_id": "typed-1",
            "collection_id": "container-1",
            "context": '{"content_type":"text/markdown"}',
            "content": "# old",
            "content_type": "text/markdown",
            "state": "draft",
            "created_by": "user-123",
        }

        _patch_db_collection(store, artifact_doc=typed_doc)

        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store

        updated = ArtifactEntity(
            id="typed-1",
            root_id="typed-1",
            collection_id="container-1",
            context='{"content_type":"text/markdown"}',
            content="# new content",
            state="draft",
            content_type="text/markdown",
        )
        try:
            with patch(
                "mantle.services.workspace_service.update_artifact", return_value=updated
            ) as upd:
                r = await client.patch(
                    "/artifacts/typed-1",
                    json={"content": "# new content"},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 200
        upd.assert_called_once()
        assert r.json()["content"] == "# new content"

    @pytest.mark.asyncio
    async def test_update_artifact_404_when_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store)

        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.patch(
                "/artifacts/missing",
                json={"content": "x"},
            )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /artifacts/{id}
# ---------------------------------------------------------------------------

class TestDeleteArtifact:
    @pytest.mark.asyncio
    async def test_delete_calls_service_and_returns_id(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=_artifact_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            with patch("mantle.services.workspace_service.delete_artifact") as deleted:
                r = await client.delete("/artifacts/art-1")
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 200
        # A delete reports what it did via `detached`/`destroyed`, because `{deleted: true}` alone
        # is the same answer for a note and for a cascade over a 1.85M-member collection. `deleted`
        # is unchanged, so this stays an exact-shape assertion.
        assert r.json() == {"id": "art-1", "deleted": True, "detached": 0, "destroyed": 0,
                            "refused": 0}
        deleted.assert_called_once()
        assert deleted.call_args.kwargs["cascade"] is False, (
            "cascade must default to False when the query param is omitted"
        )

    @pytest.mark.asyncio
    async def test_delete_forwards_cascade_true_from_the_query_string(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=_artifact_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            with patch("mantle.services.workspace_service.delete_artifact") as deleted:
                r = await client.delete("/artifacts/art-1?cascade=true")
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 200
        assert deleted.call_args.kwargs["cascade"] is True

    @pytest.mark.asyncio
    async def test_delete_404_when_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.delete("/artifacts/missing")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_a_top_level_artifact_deletes_and_then_reads_404(self, client: AsyncClient):
        """Omitting `container_id` on create is the documented way to make a top-level
        artifact, and such an artifact carries no `collection_id`. Deleting one takes the
        container primitive — `delete_workspace`, which cleans up members, edges and index
        docs — rather than `delete_artifact`, which refuses a blank containing collection.

        The read afterwards is the half that matters: a 200 that left the row in place
        would be a delete that reports success and erases nothing.
        """
        from mantle.services.dependencies import get_store_db

        top_level = {
            "_key": "top-1",
            "id": "top-1",
            "root_id": "top-1",
            "collection_id": "",
            "context": "{}",
            "content": "",
            "state": "draft",
            "created_by": "user-123",
        }
        docs = {"top-1": top_level}
        store = MagicMock()
        store.artifacts.get_artifact.side_effect = lambda key: docs.get(key)
        store.artifacts.versions_of.side_effect = lambda root: [
            d for d in docs.values() if d.get("root_id") == root
        ]
        store.graph.edges_of.side_effect = lambda node, **kw: []

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            with patch("mantle.services.workspace_service.delete_workspace") as dropped:
                dropped.side_effect = lambda db, user_id, workspace_id, **_kw: docs.pop(
                    workspace_id, None)
                r = await client.delete("/artifacts/top-1")
                assert r.status_code == 200, r.text
                # This case is the guard: the mock above returns the popped artifact document, not
                # counts. The response is built from named keys, so a callee returning something
                # else — like a whole document — contributes nothing to it instead of leaking it.
                assert r.json() == {"id": "top-1", "deleted": True,
                                    "detached": 0, "destroyed": 0, "refused": 0}
                dropped.assert_called_once()
                # The container primitive, not the one that needs a parent collection id.
                assert dropped.call_args.args[2] == "top-1"

            after = await client.get("/artifacts/top-1")
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert after.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_does_not_pass_a_blank_collection_id_to_the_service(
        self, client: AsyncClient
    ):
        """`workspace_service.delete_artifact` is keyed on the containing collection for both
        its S3 arm and its index arm and raises without one, so the top-level branch must not
        reach it at all."""
        from mantle.services.dependencies import get_store_db

        store = MagicMock()
        _patch_db_collection(store, artifact_doc={
            "_key": "top-2", "id": "top-2", "root_id": "top-2",
            "collection_id": "", "context": "{}", "content": "", "state": "draft",
        })

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            with (
                patch("mantle.services.workspace_service.delete_workspace"),
                patch("mantle.services.workspace_service.delete_artifact") as per_member,
            ):
                r = await client.delete("/artifacts/top-2")
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 200
        per_member.assert_not_called()


# ---------------------------------------------------------------------------
# POST /artifacts/{artifact_id}/remove — remove from workspace (soft)
# ---------------------------------------------------------------------------

class TestRemoveArtifactFromWorkspace:
    """DELETE /artifacts/{container_id}/children/{artifact_id} — remove from workspace (soft).

    The container is a path segment, not a request-body field: DELETE with a request body has no
    defined semantics in HTTP and intermediaries may drop it, so the field naming what to detach
    from must not depend on a carrier that may not arrive.
    """
    @pytest.mark.asyncio
    async def test_requires_auth(self, anon_client: AsyncClient):
        r = await anon_client.delete("/artifacts/ws-1/children/art-1")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_happy_path_returns_removed_response(self, client: AsyncClient):
        from mantle.entities.artifact import Artifact as ArtifactEntity
        removed = ArtifactEntity(
            id="art-1",
            root_id="art-1",
            collection_id="ws-1",
            context='{"content_type":"text/plain"}',
            content="hi",
            state=ArtifactEntity.STATE_DRAFT,
            created_by="user-123",
        )
        with patch(
            "mantle.services.workspace_service.remove_artifact_from_container",
            return_value=removed,
        ) as svc:
            r = await client.delete("/artifacts/ws-1/children/art-1")

        assert r.status_code == 200
        body = r.json()
        assert body == {"id": "art-1", "removed": True, "container_id": "ws-1"}
        svc.assert_called_once()

    @pytest.mark.asyncio
    async def test_404_propagates(self, client: AsyncClient):
        from fastapi import HTTPException
        with patch(
            "mantle.services.workspace_service.remove_artifact_from_container",
            side_effect=HTTPException(status_code=404, detail="Artifact not found"),
        ):
            r = await client.delete("/artifacts/ws-1/children/missing")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /artifacts/{id}/op/invoke
# ---------------------------------------------------------------------------

class TestInvokeArtifact:
    @pytest.mark.asyncio
    async def test_404_when_artifact_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.post("/artifacts/missing/op/invoke", json={"input": "x"})
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404




# ---------------------------------------------------------------------------
# POST /artifacts/batch
# ---------------------------------------------------------------------------

class TestBatchFetch:
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, anon_client: AsyncClient):
        r = await anon_client.post("/artifacts/batch", json={"artifact_ids": ["a-1"]})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_only_accessible_artifacts(self, client: AsyncClient):
        from fastapi import HTTPException

        store = MagicMock()
        # Two distinct artifacts; the second one's check_access raises 403.
        docs = {
            "a-1": {"id": "a-1", "context": "{}", "state": "draft"},
            "a-2": {"id": "a-2", "context": "{}", "state": "draft"},
        }
        store.artifacts.get_artifact.side_effect = lambda k: docs.get(k)
        store.artifacts.versions_of.side_effect = lambda root: []
        store.graph.edges_of.side_effect = lambda node, **kw: []

        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store

        # Override the autouse check_access fixture: a-2 is forbidden.
        def fake_check(auth, aid, action, db):
            if aid == "a-2":
                raise HTTPException(status_code=403)
            return SimpleNamespace(can_read=True)

        try:
            with patch(
                "mantle.routers.artifacts_router.check_access", side_effect=fake_check
            ):
                r = await client.post(
                    "/artifacts/batch",
                    json={"artifact_ids": ["a-1", "a-2"]},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 200
        out = r.json()["items"]
        assert [a["id"] for a in out] == ["a-1"]

    @pytest.mark.asyncio
    async def test_container_row_is_normalized_to_artifact_shape(self, client: AsyncClient):
        store = MagicMock()
        container_doc = {
            "id": "ws-1",
            "name": "Inbox",
            "description": "Seed inbox workspace",
            "content_type": WORKSPACE_CONTENT_TYPE,
            "state": "draft",
        }
        _patch_db_collection(store, container_doc=container_doc)

        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.post("/artifacts/batch", json={"artifact_ids": ["ws-1"]})
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 200
        out = r.json()["items"]
        assert len(out) == 1
        assert out[0]["id"] == "ws-1"
        assert out[0]["root_id"] == "ws-1"
        # Content defaults to "" for containers (normalization does not
        # synthesize content from description).
        assert out[0]["content"] == ""
        # Context defaults to "" when not set (no type-specific synthesis).
        assert out[0]["context"] == ""


# ---------------------------------------------------------------------------
# Reorder
# ---------------------------------------------------------------------------

class TestReorder:
    @pytest.mark.asyncio
    async def test_404_when_artifact_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, container_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.patch(
                "/artifacts/container-1/children/order",
                json={"ordered_ids": ["a-1", "a-2"]},
            )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_works_on_any_container(self, client: AsyncClient):
        # Any artifact with edges can be reordered, not just workspaces.
        store = MagicMock()
        _patch_db_collection(store, container_doc=_coll_doc("application/json"))
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            with patch(
                "mantle.db.backend.get_artifact",
                side_effect=lambda _db, aid: SimpleNamespace(root_id=aid),
            ), patch(
                # The count matters: this function returns the number of edges it updated, and the
                # route refuses when that is short of the ids asked for — a reorder is one intent,
                # and applying some of it produces an arrangement nobody chose. Two ids are sent
                # below, so two is the honest stub.
                "mantle.db.backend.reorder_collection_artifacts",
                return_value=2,
            ) as svc:
                r = await client.patch(
                    "/artifacts/container-1/children/order",
                    json={"ordered_ids": ["a-1", "a-2"]},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        # Asserting a literal `order_version` here would be wrong: the route accepts an
        # `order_version` field but never reads it, and the value is derived from the membership
        # edges (`lattice_api.order_fingerprint`) — a real token, not a constant. Pinning it to a
        # specific value would silently permit a regression to a hardcoded answer, which is how two
        # clients reordering the same container could both succeed while the second silently
        # discarded the first's arrangement. What matters here is the shape and that the route
        # still calls the service exactly once; the token's own properties are covered by
        # `test_child_order_version_is_real.py`.
        body = r.json()
        assert set(body) == {"order_version"}
        assert isinstance(body["order_version"], int)
        svc.assert_called_once()


# ---------------------------------------------------------------------------
# Content URL
# ---------------------------------------------------------------------------

class TestContentUrl:
    @pytest.mark.asyncio
    async def test_404_when_no_content_key(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(
            store,
            artifact_doc={
                "_key": "a-1",
                "context": '{"content_type":"text/plain"}',
                "state": "draft",
            },
        )
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.get("/artifacts/a-1/content-url")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404
        assert "No downloadable content" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_happy_path_returns_proxied_content_url(self, client: AsyncClient):
        # content-url points at Mantle's proxied content endpoint, which decrypts on
        # the byte path — storage is invisible to callers and holds only ciphertext.
        store = MagicMock()
        _patch_db_collection(
            store,
            artifact_doc={
                "_key": "a-1",
                "context": '{"content_key":"u-1/a-1.content","filename":"f.pdf","content_type":"application/pdf"}',
                "state": "draft",
            },
        )
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.get("/artifacts/a-1/content-url")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        assert r.json() == {"url": "/artifacts/a-1/content"}


# ---------------------------------------------------------------------------
# Upload initiate
# ---------------------------------------------------------------------------

class TestUploadInitiate:
    @pytest.mark.asyncio
    async def test_delegates_to_workspace_service(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store

        artifact = ArtifactEntity(
            id="a-1",
            root_id="a-1",
            collection_id="container-1",
            context="{}",
            content="",
            state=ArtifactEntity.STATE_DRAFT,
        )
        out = {"upload_id": "a-1", "mode": "single", "url": "https://s3/put", "key": "u/a-1.content"}
        try:
            with patch(
                "mantle.services.workspace_service.initiate_upload_and_create_artifact",
                return_value=(out, artifact),
            ) as initiate:
                r = await client.post(
                    "/artifacts/container-1/upload-initiate",
                    json={"filename": "f.pdf", "content_type": "application/pdf", "size": 1234},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        body = r.json()
        assert body["upload_id"] == "a-1"
        assert body["mode"] == "single"
        initiate.assert_called_once()


# ---------------------------------------------------------------------------
# List commits
# ---------------------------------------------------------------------------

class TestListCommits:
    @pytest.mark.asyncio
    async def test_a_missing_artifact_yields_an_empty_page_not_a_type_error(self, client: AsyncClient):
        """With authorization stubbed by this module's autouse fixture, there is nothing left to
        refuse a missing id: `artifacts_router.check_access` is patched to a granting stub, so
        `_artifact_exists` is the only remaining gate. In production `check_access` runs first,
        calls the same `get_raw_artifact`, and answers 404 for missing-or-denied, so a branch that
        reports a type restriction on a missing artifact is unreachable there.

        The honest answer under a granting stub is the one the endpoint has always given for an
        artifact with no commits: an empty page. That is what this asserts.
        """
        store = MagicMock()
        _patch_db_collection(store, container_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.get("/artifacts/missing/commits")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200, r.json()
        assert r.json() == {"items": [], "total": 0, "has_more": False}

    @pytest.mark.asyncio
    async def test_returns_commit_list(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, container_doc=_coll_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        commits = [
            SimpleNamespace(
                id="c-1",
                collection_id="container-1",
                message="initial",
                author_id="user-123",
                created_time="t0",
                adds=["a-1"],
                removes=[],
            ),
        ]
        try:
            with patch(
                "mantle.services.collection_service.get_commits_for_collection",
                return_value=commits,
            ):
                r = await client.get("/artifacts/container-1/commits")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        body = r.json()
        assert len(body["items"]) == 1
        assert body["items"][0]["id"] == "c-1"
        assert body["items"][0]["message"] == "initial"


# ---------------------------------------------------------------------------
# Revert / Move / Upload-status / Multipart-part-url
#
# These four endpoints all share the same "find the artifact, look up its
# container" pattern, and read the unified-store `collection_id` field to do
# it.
# ---------------------------------------------------------------------------


class TestRevertArtifact:
    """Revert is a dedicated `POST /artifacts/{id}/revert` route, the canonical
    path; the `op/revert` dispatch route also still works.
    """

    @pytest.mark.asyncio
    async def test_404_when_artifact_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.post("/artifacts/missing/op/revert", json={})
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404


    @pytest.mark.asyncio
    async def test_dedicated_revert_route_returns_committed_version(
        self, client: AsyncClient
    ):
        """`POST /artifacts/{id}/revert` calls workspace_service.revert_artifact and
        returns the restored committed version — through `_artifact_body`.

        `_artifact_body` guarantees every declared key is present, `null` when unset — a client can
        program against a fixed shape, unlike a raw `to_dict()`, which omits unset fields and so
        varies with the data.

        The assertion checks the two values that matter and the guarantee, rather than equality
        with a literal: an equality test here would fail the next time a field is added to the
        response, which is the kind of churn the guarantee exists to prevent."""
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=_artifact_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            restored = MagicMock()
            restored.to_dict.return_value = {"id": "art-1", "state": "committed"}
            with patch(
                "mantle.services.workspace_service.revert_artifact",
                return_value=restored,
            ):
                r = await client.post("/artifacts/art-1/revert")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "art-1"
        assert body["state"] == "committed"
        from mantle.routers.artifacts_router import _ARTIFACT_KEYS
        missing = [k for k in _ARTIFACT_KEYS if k not in body]
        assert not missing, (
            "revert declares `ok=ArtifactResponse`, which promises every key is present "
            "(`null` when unset), but the body omits %r" % missing)

    @pytest.mark.asyncio
    async def test_dedicated_revert_route_204_when_no_committed_version(
        self, client: AsyncClient
    ):
        """If there's no committed version to revert to, return 204 No Content."""
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=_artifact_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            with patch(
                "mantle.services.workspace_service.revert_artifact",
                return_value=None,
            ):
                r = await client.post("/artifacts/art-1/revert")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 204
        assert r.content == b""

    @pytest.mark.asyncio
    async def test_dedicated_revert_route_404_when_artifact_missing(
        self, client: AsyncClient
    ):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.post("/artifacts/missing/revert")
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404


class TestMoveArtifact:
    @pytest.mark.asyncio
    async def test_404_when_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.post(
                "/artifacts/missing/op/move",
                json={"target_container_id": "ws-2"},
            )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404



class TestUploadStatus:
    @pytest.mark.asyncio
    async def test_404_when_missing(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=None)
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store
        try:
            r = await client.patch(
                "/artifacts/missing/upload-status",
                json={"status": "uploading", "progress": 0.5},
            )
        finally:
            app.dependency_overrides.pop(get_store_db, None)
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_happy_path_uses_collection_id(self, client: AsyncClient):
        store = MagicMock()
        _patch_db_collection(store, artifact_doc=_artifact_doc())
        from mantle.services.dependencies import get_store_db

        app.dependency_overrides[get_store_db] = lambda: store

        updated = ArtifactEntity(
            id="art-1",
            root_id="art-1",
            collection_id="container-1",
            context='{"upload":{"status":"complete"}}',
            content="",
            state=ArtifactEntity.STATE_DRAFT,
        )
        try:
            with patch(
                "mantle.services.workspace_service.update_upload_status",
                return_value=updated,
            ) as svc:
                r = await client.patch(
                    "/artifacts/art-1/upload-status",
                    json={"status": "complete", "progress": 1.0},
                )
        finally:
            app.dependency_overrides.pop(get_store_db, None)

        assert r.status_code == 200
        assert svc.call_args.kwargs["workspace_id"] == "container-1"
        assert svc.call_args.kwargs["upload_id"] == "art-1"
        assert svc.call_args.kwargs["status_value"] == "complete"


class TestMultipartPartUrl:
    """The route is retired. This class pins its absence.

    Presigned multipart parts would upload straight to the object store, bypassing Mantle, which
    cannot envelope-encrypt them on that path — an operation that never worked, so it is removed
    rather than kept to explain itself.

    Asserting only the status would be a weak guard — every absent route answers 404. The schema
    assertion is the real one: the operation must not be advertised, because a client generates
    from the spec, not from a probe.
    """

    @pytest.mark.asyncio
    async def test_the_operation_is_no_longer_advertised(self, client: AsyncClient):
        from mantle.main import app

        paths = app.openapi()["paths"]
        assert not [p for p in paths if "multipart-part-url" in p], (
            "the retired route is back in the schema: %r"
            % [p for p in paths if "multipart-part-url" in p])

    @pytest.mark.asyncio
    async def test_calling_it_is_an_ordinary_404(self, client: AsyncClient):
        r = await client.get("/artifacts/any/multipart-part-url", params={"part_number": 1})
        assert r.status_code == 404
        assert "proxied" not in (r.text or "").lower(), (
            "a 409-era explanation is still being served — the handler is still wired")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class TestSearchArtifacts:
    @pytest.mark.asyncio
    async def test_400_on_empty_query(self, client: AsyncClient):
        r = await client.post("/artifacts/recall", json={"query_text": "   "})
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_unauthenticated_search_returns_401(self, anon_client: AsyncClient):
        r = await anon_client.post("/artifacts/recall", json={"query_text": "x"})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_happy_path_returns_hits(self, client: AsyncClient):
        """Search endpoint goes through MantleSseSearchAccessor. Patch the wiring
        builder to return a stub accessor with a canned result."""
        accessor_result = SimpleNamespace(
            hits=[
                SimpleNamespace(
                    doc_id="a-1",
                    score=1.5,
                    root_id="a-1",
                    version_id="v-1",
                    workspace_id="ws-1",
                    collection_id=None,
                    title="Test Artifact",
                    description="A test artifact",
                    content="Some content here",
                    tags=["test"],
                    highlights=None,
                ),
            ],
            total=1,
            parsed_query="x",
            applied_filters=[],
            corrections=[],
            ordering="semantic",
        )
        fake_accessor = SimpleNamespace(search=lambda query: accessor_result)
        with patch(
            "mantle.search.mantle.wiring.build_sse_search_accessor",
            return_value=fake_accessor,
        ):
            r = await client.post(
                "/artifacts/recall",
                json={"query_text": "x", "size": 10},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["hits"][0]["id"] == "a-1"
        assert body["ordering"] == "semantic"
        assert body["hits"][0]["score"] == 1.5
        assert body["from"] == 0

    @pytest.mark.asyncio
    async def test_503_when_sse_prereqs_missing(self, client: AsyncClient):
        """When SSE prerequisites aren't met (no S3, no Oracle), the search
        endpoint returns 503 — there's no plaintext fallback."""
        with patch(
            "mantle.search.mantle.wiring.build_sse_search_accessor",
            return_value=None,
        ):
            r = await client.post(
                "/artifacts/recall",
                json={"query_text": "x", "size": 10},
            )
        assert r.status_code == 503

    @pytest.mark.asyncio
    async def test_caller_supplied_embedding_is_rejected(self, client: AsyncClient):
        """`embedding` is not the ingress and never was — `vector` + `space_id` is.

        A bare list of floats under a name nothing reads is a request with nothing to
        rank on, so it is an ordinary 400. The distinction is not decorative: the arm
        needs the space name to know what the numbers are comparable to, and a field
        that carried numbers without one would be exactly the unusable vector
        `api/vectors.py` refuses.
        """
        r = await client.post(
            "/artifacts/recall",
            json={"embedding": [0.1, 0.2, 0.3], "size": 5},
        )
        assert r.status_code == 400

    @pytest.fixture()
    def seeded_space(self):
        """A node that HAS an AnchorSet, in `space-a`, three-dimensional.

        The two tests below are about the vector reaching the arm, and a vector only reaches
        the arm on a node that ranks in some space: `api/vectors.project_to_anchor_space`
        refuses a query vector when no AnchorSet is seeded, because there is nowhere to place
        it and the alternative is a recency-ordered dump the caller cannot tell from a ranking.
        Seeding here keeps these tests about their own subject.
        `tests/test_recall_vector_without_anchorset.py` covers the unseeded node.
        """
        import numpy as np
        from mantle.search.anchors import store as _anchor_store
        from mantle.search.anchors.anchorset import Anchor
        from mantle.search.anchors.repo import InMemoryAnchorRepo

        repo = InMemoryAnchorRepo()
        rng = np.random.default_rng(11)
        repo.bulk_add([
            Anchor.make(f"a{i}", rng.standard_normal(3).astype("float32"), "space-a")
            for i in range(3)
        ])
        _anchor_store.set_anchor_repo(repo)
        try:
            yield repo
        finally:
            _anchor_store.set_anchor_repo(None)

    @pytest.mark.asyncio
    async def test_a_query_vector_reaches_the_accessor(self, client: AsyncClient, seeded_space):
        """The reader's vector arrives at the vector arm — the whole point of the field."""
        captured = {}

        def _search(query):
            captured["embedding"] = query.query_embedding
            return SimpleNamespace(hits=[], total=0, parsed_query="", applied_filters=[],
                                   corrections=[], ordering="recency")

        with patch(
            "mantle.search.mantle.wiring.build_sse_search_accessor",
            return_value=SimpleNamespace(search=_search),
        ):
            r = await client.post(
                "/artifacts/recall",
                json={"query_text": "x", "vector": [0.1, 0.2, 0.3],
                      "space_id": "space-a", "size": 5},
            )
        assert r.status_code == 200
        assert captured["embedding"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_a_vector_alone_is_a_complete_query(self, client: AsyncClient, seeded_space):
        """kNN with no text is a real request, not a missing-query_text 400."""
        captured = {}

        def _search(query):
            captured["embedding"] = query.query_embedding
            captured["text"] = query.query_text
            return SimpleNamespace(hits=[], total=0, parsed_query="", applied_filters=[],
                                   corrections=[], ordering="recency")

        with patch(
            "mantle.search.mantle.wiring.build_sse_search_accessor",
            return_value=SimpleNamespace(search=_search),
        ):
            r = await client.post(
                "/artifacts/recall",
                json={"vector": [0.1, 0.2, 0.3], "space_id": "space-a", "size": 5},
            )
        assert r.status_code == 200
        assert captured["embedding"] == [0.1, 0.2, 0.3]
        assert captured["text"] == ""

    @pytest.mark.asyncio
    async def test_a_query_vector_without_a_space_id_is_a_400(self, client: AsyncClient):
        """Same rule as the write: numbers with no space name are not comparable to anything."""
        r = await client.post(
            "/artifacts/recall",
            json={"query_text": "x", "vector": [0.1, 0.2, 0.3], "size": 5},
        )
        assert r.status_code == 400
        assert "space_id" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_a_space_id_without_a_vector_is_a_400(self, client: AsyncClient):
        r = await client.post(
            "/artifacts/recall",
            json={"query_text": "x", "space_id": "space-a", "size": 5},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad", [[], [0.0, 0.0]])
    async def test_a_malformed_query_vector_fails_the_writer_s_way(
        self, client: AsyncClient, bad,
    ):
        """No components and no direction are the writer's two shape refusals, verbatim."""
        r = await client.post(
            "/artifacts/recall",
            json={"query_text": "x", "vector": bad, "space_id": "space-a", "size": 5},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_embedding_field_is_ignored_when_query_text_present(self, client: AsyncClient):
        """An `embedding` alongside valid text must never reach the accessor."""
        captured = {}

        def _search(query):
            captured["embedding"] = query.query_embedding
            return SimpleNamespace(hits=[], total=0, parsed_query="", applied_filters=[],
                                   corrections=[], ordering="semantic")

        with patch(
            "mantle.search.mantle.wiring.build_sse_search_accessor",
            return_value=SimpleNamespace(search=_search),
        ):
            r = await client.post(
                "/artifacts/recall",
                json={"query_text": "x", "embedding": [0.1, 0.2], "size": 5},
            )
        assert r.status_code == 200
        assert captured["embedding"] is None

    @pytest.mark.asyncio
    async def test_400_when_neither_query_text_nor_embedding(self, client: AsyncClient):
        r = await client.post("/artifacts/recall", json={"size": 5})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /artifacts/visible — action-scoped visibility
# ---------------------------------------------------------------------------


class _FakeResolver:
    """Records the actions it's resolved against and returns a fixed set per
    action. 'read' is non-empty so first-login provisioning never fires."""

    seen_actions: list = []
    by_action = {
        "read": {"col-readonly", "col-mine"},
        "create": {"col-mine"},
    }

    def __init__(self, db, **kwargs):
        pass

    def resolve(self, principal_id, action="read", **kwargs):
        _FakeResolver.seen_actions.append(action)
        return set(self.by_action.get(action, set()))


class TestListVisibleActionFilter:
    """The assign-to-collection picker requests `?action=create` so read-only
    platform collections (read-granted to every user) are not offered as
    assignable. The endpoint must filter by the requested CRUDEASIO action,
    not by 'read'."""

    @pytest.mark.asyncio
    async def test_action_create_filters_to_addable_collections(self, client: AsyncClient):
        _FakeResolver.seen_actions = []
        with patch("mantle.search.mantle.lightcone.LightConeResolver", _FakeResolver), patch(
            "mantle.routers.artifacts_router._hydrate_batch",
            side_effect=lambda db, ids: {aid: {"id": aid} for aid in ids},
        ):
            r = await client.get("/artifacts/visible?action=create")
        assert r.status_code == 200
        ids = {a["id"] for a in r.json()["items"]}
        # Only the create-capable collection — the read-only one is excluded.
        assert ids == {"col-mine"}
        # Provisioning gate is always resolved against "read"; the visible set
        # is resolved against the requested action.
        assert "read" in _FakeResolver.seen_actions
        assert "create" in _FakeResolver.seen_actions

    @pytest.mark.asyncio
    async def test_default_action_is_read(self, client: AsyncClient):
        _FakeResolver.seen_actions = []
        with patch("mantle.search.mantle.lightcone.LightConeResolver", _FakeResolver), patch(
            "mantle.routers.artifacts_router._hydrate_batch",
            side_effect=lambda db, ids: {aid: {"id": aid} for aid in ids},
        ):
            r = await client.get("/artifacts/visible")
        assert r.status_code == 200
        ids = {a["id"] for a in r.json()["items"]}
        assert ids == {"col-readonly", "col-mine"}
        # No extra resolve for the default path — 'read' covers both the
        # provisioning gate and the visible set.
        assert _FakeResolver.seen_actions == ["read"]

    @pytest.mark.asyncio
    async def test_unknown_action_rejected(self, client: AsyncClient):
        r = await client.get("/artifacts/visible?action=bogus")
        assert r.status_code == 400


class TestEmbeddingEndpoint:
    """Neither endpoint exists, under the no-models rule.

    `GET /artifacts/{id}/embedding` would serve stored bge-m3 vectors ("raw vectors out, no text")
    and `POST /artifacts/activate` would accept a caller-supplied carrier vector and echo it back.
    Both are embed/score-as-a-service, which the standing ruling on `/coherence` and `/embed` says
    to remove entirely rather than answer 501 — "an observer does not offer 'embed this' or 'score
    this' as a service."
    """

    @pytest.mark.asyncio
    async def test_get_embedding_route_is_gone(self, client: AsyncClient):
        r = await client.get("/artifacts/a-1/embedding")
        assert r.status_code in (404, 405), (
            "the embedding-serving route must not exist"
        )

    @pytest.mark.asyncio
    async def test_activate_route_is_gone(self, client: AsyncClient):
        r = await client.post("/artifacts/activate", json={"embedding": [0.1, 0.2, 0.3]})
        assert r.status_code in (404, 405), (
            "the vector-carrier route must not exist"
        )
