"""A write that names what it is OF lands on one artifact, however many times it runs.

`services/artifact_identity` derives an artifact id from the acting principal and a
caller-chosen natural key, so a client re-storing the same file, session or note updates one
artifact instead of accumulating copies. These tests pin the three properties the derivation is
relied on for — determinism, per-principal separation, and refusal on an empty key — plus the
router behaviour that makes it useful: an `identity` that already exists is an UPDATE.

The failure being prevented is measured, not hypothetical. Before this landed, the dogfooding
node held `agience-mantle/README.md` as two artifacts created three minutes apart and one
Claude Code session as five, because the only way to make a second write land on the first
artifact was a client-side map from path to id — and a write whose reply is lost still succeeds
here, leaving that map short one entry and the artifact orphaned for good.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mantle.services.artifact_identity import (
    ARTIFACT_IDENTITY_NS,
    derive_artifact_id,
    derived_id_for,
)


class TestDerivation:
    def test_same_principal_and_identity_derive_the_same_id(self):
        """The property the whole mechanism rests on: no state, no clock, no counter."""
        a = derive_artifact_id("user-1", "file:/repo/README.md")
        b = derive_artifact_id("user-1", "file:/repo/README.md")
        assert a == b

    def test_it_is_a_uuid5_in_the_declared_namespace(self):
        """Checkable from outside, the way an anchor's id is checkable against its content.

        A derivation nobody can reproduce is just an opaque id with extra steps; being able to
        recompute it is what lets a client confirm which artifact a write will land on before
        making it.
        """
        got = derive_artifact_id("user-1", "session:abc")
        assert got == str(uuid.uuid5(ARTIFACT_IDENTITY_NS, "user-1\nsession:abc"))

    def test_different_identities_derive_different_ids(self):
        assert derive_artifact_id("user-1", "file:a") != derive_artifact_id("user-1", "file:b")

    def test_two_principals_never_collide_on_one_identity(self):
        """Two people capturing their own `README.md` must not name one artifact.

        This is why the principal is inside the derivation rather than beside it. A global
        derivation would make the second writer's create target the first writer's artifact —
        a collision neither can see, on a store where the correct answer is that they simply
        have different artifacts. Convergence is what a grant is for.
        """
        mine = derive_artifact_id("user-1", "file:/repo/README.md")
        theirs = derive_artifact_id("user-2", "file:/repo/README.md")
        assert mine != theirs

    def test_the_separator_cannot_be_forged_from_the_left(self):
        """`(ab, c)` and `(a, bc)` must not derive one id.

        A principal id is a UUID and contains no newline, so splitting on it is unambiguous —
        the same reasoning `services/oidc.external_user_id` joins issuer and subject on.
        """
        assert derive_artifact_id("user-1", "x") != derive_artifact_id("user", "1\nx")

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_an_empty_identity_is_refused(self, bad):
        """Deriving from `""` would give every such write from one principal a single id —
        the loudest possible version of the bug this module exists to prevent."""
        if bad is None:
            # `None` means "no identity supplied", which is the unchanged uuid4 path.
            assert derived_id_for("user-1", None) is None
            return
        with pytest.raises(ValueError):
            derive_artifact_id("user-1", bad)

    def test_an_empty_principal_is_refused(self):
        with pytest.raises(ValueError):
            derive_artifact_id("", "file:/repo/README.md")


class TestRouterUpsert:
    """`identity` on the create path: first call creates at the derived id, second updates it."""

    @pytest.mark.asyncio
    async def test_first_write_creates_at_the_derived_id(self):
        from mantle.routers import artifacts_router as ar

        auth = MagicMock(user_id="user-1")
        store_db = MagicMock()
        created = MagicMock()
        created.to_dict.return_value = {"id": "derived"}

        with (
            patch.object(ar, "_artifact_exists", return_value=False),
            patch("mantle.services.workspace_service.create_container",
                  return_value=created) as create,
            patch("mantle.services.workspace_service.update_workspace") as update,
        ):
            await ar._default_create_artifact(
                {"identity": "file:/repo/README.md", "content": "first body",
                 "content_type": "text/markdown"},
                auth, store_db,
            )

        update.assert_not_called()
        create.assert_called_once()
        assert create.call_args.kwargs["artifact_id"] == derive_artifact_id(
            "user-1", "file:/repo/README.md",
        )

    @pytest.mark.asyncio
    async def test_second_write_updates_instead_of_creating_a_duplicate(self):
        """The whole point. Same identity, artifact already there → one artifact, new body.

        Without this branch the second call mints a fresh `uuid4` and the store holds two
        roots for one file, which is what was measured on the live node.
        """
        from mantle.routers import artifacts_router as ar

        auth = MagicMock(user_id="user-1")
        store_db = MagicMock()
        updated = MagicMock()
        updated.to_dict.return_value = {"id": "derived"}

        with (
            patch.object(ar, "_artifact_exists", return_value=True),
            patch("mantle.services.workspace_service.create_container") as create,
            patch("mantle.services.workspace_service.update_workspace",
                  return_value=updated) as update,
        ):
            await ar._default_create_artifact(
                {"identity": "file:/repo/README.md", "content": "second body",
                 "content_type": "text/markdown"},
                auth, store_db,
            )

        create.assert_not_called()
        update.assert_called_once()
        # Positional: (db, user_id, artifact_id)
        assert update.call_args.args[2] == derive_artifact_id(
            "user-1", "file:/repo/README.md",
        )
        assert update.call_args.kwargs["content"] == "second body"

    @pytest.mark.asyncio
    async def test_no_identity_keeps_the_uuid4_path(self):
        """Omitting `identity` changes nothing — every existing caller is unaffected."""
        from mantle.routers import artifacts_router as ar

        auth = MagicMock(user_id="user-1")
        created = MagicMock()
        created.to_dict.return_value = {"id": "fresh"}

        with (
            patch("mantle.services.workspace_service.create_container",
                  return_value=created) as create,
        ):
            await ar._default_create_artifact(
                {"content": "body", "content_type": "text/markdown"}, auth, MagicMock(),
            )

        assert create.call_args.kwargs["artifact_id"] is None

    @pytest.mark.asyncio
    async def test_identity_with_container_id_is_refused_by_name(self):
        """Refused rather than ignored.

        A collection member is born a draft and grows a second live version the first time it
        is edited after commit, so "the artifact for this identity" stops being one row.
        Accepting the parameter and dropping it would report idempotency that is not there.
        """
        from mantle.routers import artifacts_router as ar

        auth = MagicMock(user_id="user-1")
        with pytest.raises(HTTPException) as exc:
            await ar._default_create_artifact(
                {"identity": "file:x", "container_id": "col-1", "content": "b"},
                auth, MagicMock(),
            )
        assert exc.value.status_code == 400
        assert "identity" in exc.value.detail

    @pytest.mark.asyncio
    async def test_an_empty_identity_is_a_400_not_a_shared_id(self):
        from mantle.routers import artifacts_router as ar

        auth = MagicMock(user_id="user-1")
        with pytest.raises(HTTPException) as exc:
            await ar._default_create_artifact(
                {"identity": "   ", "content": "b"}, auth, MagicMock(),
            )
        assert exc.value.status_code == 400
