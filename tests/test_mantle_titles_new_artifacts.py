"""Mantle names a new empty artifact, keeps a name when a save blanks it, and answers `name` as the
title a document is shown by."""
import json

from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE
from mantle.routers import artifacts_router as router
from mantle.services import new_titles


def test_the_first_free_number_is_taken():
    assert new_titles.next_free("Untitled Markdown", []) == "Untitled Markdown"
    assert new_titles.next_free("Untitled Markdown", ["untitled markdown"]) == "Untitled Markdown 2"
    taken = ["Untitled Markdown", "Untitled Markdown 3", "Other"]
    assert new_titles.next_free("Untitled Markdown", taken) == "Untitled Markdown 2"


def test_only_an_empty_unnamed_create_is_titled():
    assert new_titles.needs_title(None, None, None)
    assert new_titles.needs_title(json.dumps({"x": 1}), "", "")
    assert not new_titles.needs_title(None, "Mine", None)
    assert not new_titles.needs_title(None, None, "# body")
    assert not new_titles.needs_title(json.dumps({"title": "Kept"}), None, None)
    assert not new_titles.needs_title(json.dumps({"filename": "a.md"}), None, None)
    assert not new_titles.needs_title("not json", None, None)


def test_a_workspace_is_new_workspace_and_an_unknown_type_is_artifact():
    assert new_titles.base_title(WORKSPACE_CONTENT_TYPE) == new_titles.DEFAULT_WORKSPACE_NAME
    assert new_titles.base_title("application/x-no-such-type") == "Untitled Artifact"


def test_a_blank_title_on_save_comes_from_the_content():
    stored = {"context": json.dumps({"title": "Old"}), "content": "stored body"}
    out = json.loads(new_titles.title_on_save(json.dumps({"title": " "}), "\n## Plan\nmore", stored))
    assert out["title"] == "Plan"
    out = json.loads(new_titles.title_on_save(json.dumps({"title": ""}), None, stored))
    assert out["title"] == "stored body"
    out = json.loads(new_titles.title_on_save(json.dumps({"title": ""}), "", stored))
    assert out["title"] == "Old"
    out = json.loads(new_titles.title_on_save(json.dumps({"title": ""}), "", {}))
    assert out["title"] == "Untitled"


def test_a_save_that_names_a_title_or_none_is_left_alone():
    ctx = json.dumps({"title": "Typed"})
    assert new_titles.title_on_save(ctx, "body", {}) == ctx
    ctx = json.dumps({"tags": []})
    assert new_titles.title_on_save(ctx, "body", {}) == ctx
    assert new_titles.title_on_save(None, "body", {}) is None


def test_a_title_from_content_is_cut():
    assert len(new_titles.title_from_content("x" * 200, "")) == new_titles.TITLE_FROM_CONTENT_MAX


def test_name_is_answered_as_the_shown_title():
    doc = {"name": "", "context": json.dumps({"title": "Shown"})}
    assert new_titles.name_as_shown(doc)["name"] == "Shown"
    doc = {"name": "Older row", "context": ""}
    assert new_titles.name_as_shown(doc)["name"] == "Older row"
    body = router._artifact_body({"id": "a", "context": json.dumps({"title": "T"})})
    assert body["name"] == "T"
    page = router._page([{"id": "b", "context": json.dumps({"title": "P"})}], total=1, has_more=False)
    assert page["items"][0]["name"] == "P"


def test_a_new_member_is_numbered_against_its_container(monkeypatch):
    import mantle.db.lattice_api as lattice_api
    members = [{"context": json.dumps({"title": "Untitled Artifact"})},
               {"context": "", "name": "Untitled Artifact 2"}]
    monkeypatch.setattr(lattice_api, "list_collection_artifacts", lambda db, cid: members)
    assert router._title_a_new_one(None, "u1", "application/x-no-such-type", "c1") == "Untitled Artifact 3"


def test_a_new_workspace_is_numbered_against_the_persons_workspaces(monkeypatch):
    from mantle.services import workspace_service

    class _Ws:
        def __init__(self, name):
            self._name = name

        def to_dict(self):
            return {"name": self._name, "context": ""}

    monkeypatch.setattr(workspace_service, "list_workspaces",
                        lambda db, uid: [_Ws("New workspace"), _Ws("Plans")])
    assert router._title_a_new_one(None, "u1", WORKSPACE_CONTENT_TYPE, None) == "New workspace 2"
    assert router._title_a_new_one(None, None, WORKSPACE_CONTENT_TYPE, None) == "New workspace"
