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
                ar.CreateArtifactRequest(**{"identity": "file:/repo/README.md", "content": "first body",
                 "content_type": "text/markdown"}),
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
                ar.CreateArtifactRequest(**{"identity": "file:/repo/README.md", "content": "second body",
                 "content_type": "text/markdown"}),
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
                ar.CreateArtifactRequest(**{"content": "body", "content_type": "text/markdown"}), auth, MagicMock(),
            )

        assert create.call_args.kwargs["artifact_id"] is None

    @pytest.mark.asyncio
    async def test_identity_inside_a_container_upserts_on_the_ROOT(self):
        """`identity` + `container_id` is supported, and it names the root rather than the version.

        A collection member is born a draft and grows a second live version on edit-after-commit,
        so "the artifact for this identity" would not be one row. That is a property of the draft
        lifecycle rather than of identity: an
        identity names something outside the store (a file, a session, a commit) which has one
        true version. Naming the root resolves it, and the root is what edges, grants and both
        search arms are already keyed on.

        It buys more than convenience. Without it every hook artifact is top-level, and a
        top-level artifact is its own origin root, which is the SSE principal: 115 self-rooted
        artifacts are 115 separately keyed owners for a recall to read.
        """
        from mantle.routers import artifacts_router as ar
        from mantle.services.artifact_identity import derived_id_for

        auth = MagicMock(user_id="user-1")
        member = MagicMock()
        member.to_dict.return_value = {"id": "member-1"}

        with (
            patch("mantle.routers.artifacts_router.check_access"),
            patch("mantle.routers.artifacts_router._artifact_exists", return_value=True),
            patch("mantle.services.workspace_service.upsert_identity_member",
                  return_value=member) as upsert,
        ):
            out = await ar._default_create_artifact(
                ar.CreateArtifactRequest(**{"identity": "file:x", "container_id": "col-1", "content": "b"}),
                auth, MagicMock(),
            )

        assert out == {"id": "member-1"}
        args = upsert.call_args.args
        assert args[2] == "col-1", "the member must be filed in the container it named"
        assert args[3] == derived_id_for("user-1", "file:x"), (
            "the root must be the DERIVED id — that is what makes the write idempotent"
        )

    @pytest.mark.asyncio
    async def test_a_container_member_without_identity_is_untouched(self):
        """The ordinary draft lifecycle must not have moved. Only identity writes are mirrors."""
        from mantle.routers import artifacts_router as ar

        auth = MagicMock(user_id="user-1")
        member = MagicMock()
        member.to_dict.return_value = {"id": "member-2"}

        with (
            patch("mantle.routers.artifacts_router.check_access"),
            patch("mantle.routers.artifacts_router._artifact_exists", return_value=True),
            patch("mantle.services.workspace_service.create_workspace_artifact",
                  return_value=member) as create,
            patch("mantle.services.workspace_service.upsert_identity_member") as upsert,
        ):
            await ar._default_create_artifact(
                ar.CreateArtifactRequest(**{"container_id": "col-1", "content": "b"}), auth, MagicMock(),
            )

        assert create.called, "a member with no identity still takes the ordinary create path"
        assert not upsert.called, "no identity was supplied; nothing should be upserted"

    @pytest.mark.asyncio
    async def test_an_empty_identity_is_a_400_not_a_shared_id(self):
        from mantle.routers import artifacts_router as ar

        auth = MagicMock(user_id="user-1")
        with pytest.raises(HTTPException) as exc:
            await ar._default_create_artifact(
                ar.CreateArtifactRequest(**{"identity": "   ", "content": "b"}), auth, MagicMock(),
            )
        assert exc.value.status_code == 400


class TestALexiconEntrysNamesSurviveTheRoundTrip:
    """`lemmas` is modelled on the entity, so it must also come back out of it.

    A synset's lemmas are the words that mean it, and only the first becomes the title. The OEWN
    oxygen entry is titled `O`; `oxygen` and `atomic number 8` are the other two, and its gloss
    never says any of them. `from_dict` dropped the field, so the indexing pipeline never saw the
    names and `recall("what is oxygen")` could not narrow to the concept at all.

    Reading a field that `to_dict` does not write would be WORSE than not modelling it: the first
    save of a lexicon entry would delete the words that name it. That is the failure `content_ref`
    and `origin_root` carry the same warning about, and this states it as a test rather than a
    comment.
    """

    @staticmethod
    def _doc(**over):
        doc = {
            "id": "wn-oewn-14672278-n",
            "collection_id": "stage.0.lexicon",
            "state": "committed",
            "title": "O",
            "content": "a nonmetallic bivalent element",
            "content_type": "text/x-wordnet",
            "lemmas": ["o", "atomic number 8", "oxygen"],
        }
        doc.update(over)
        return doc

    def test_lemmas_survive_dict_to_entity_to_dict(self):
        from mantle.entities.artifact import Artifact

        once = Artifact.from_dict(self._doc())
        assert once.lemmas == ["o", "atomic number 8", "oxygen"]

        stored = once.to_dict()
        assert stored.get("lemmas") == ["o", "atomic number 8", "oxygen"], (
            "a save must not drop the names — %r" % (stored,)
        )
        # The second hop is the one that matters: a field that survives one direction and not the
        # other destroys data on the FIRST save, not on the read that follows it.
        twice = Artifact.from_dict(stored)
        assert twice.lemmas == once.lemmas

    def test_an_artifact_with_no_lemmas_emits_no_key(self):
        """Absent and empty are different claims, and only a lexicon entry has names here."""
        from mantle.entities.artifact import Artifact

        prose = Artifact.from_dict(self._doc(content_type="text/markdown", lemmas=None))
        assert prose.lemmas is None
        assert "lemmas" not in prose.to_dict()


class TestAMergeSaysThatItIsOne:
    """`colimit_of` names the synsets a merge absorbed, and it must survive the round trip.

    Same contract as `lemmas` above and the same failure if it is dropped, one layer along:
    `pipeline_unified._lemmas_are_names` reads this field to decide whether a record's `lemmas`
    are names it goes by or words taken out of it. Measured 2026-08-24, that decision cannot be
    made from `content_type` — `application/x-concept` carries 5,484 colimits AND 1,165,110
    ConceptNet terms, and the two put opposite things in `lemmas`. A save that dropped
    `colimit_of` would turn every merge into an ordinary term with no visible change.
    """

    @staticmethod
    def _doc(**over):
        doc = {
            "id": "concept-be603d8ac3dd0185fcf653e18a8dc72d",
            "collection_id": "collection:concepts-consolidated",
            "content_type": "application/x-concept",
            "state": "committed",
            "lemmas": ["decade", "decennary", "decennium"],
            "colimit_of": ["wn-decade.n.01", "wn-oewn-15174893-n"],
        }
        doc.update(over)
        return doc

    def test_colimit_of_survives_dict_to_entity_to_dict(self):
        from mantle.entities.artifact import Artifact

        once = Artifact.from_dict(self._doc())
        assert once.colimit_of == ["wn-decade.n.01", "wn-oewn-15174893-n"]
        stored = once.to_dict()
        assert stored.get("colimit_of") == ["wn-decade.n.01", "wn-oewn-15174893-n"], (
            "a merge that cannot say what it merged stops being recognisable as one — %r"
            % (stored,)
        )
        assert Artifact.from_dict(stored).colimit_of == once.colimit_of

    def test_an_ordinary_concept_emits_no_colimit_key(self):
        """A ConceptNet term shares the content type and is not a merge."""
        from mantle.entities.artifact import Artifact

        term = Artifact.from_dict(self._doc(id="cn-decade", colimit_of=None))
        assert term.colimit_of is None
        assert "colimit_of" not in term.to_dict()
