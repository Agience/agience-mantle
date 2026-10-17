"""A content type says which fields give its artifacts their title, description and tags.

The `offer` block of a `type.json`, published by Crystal and stored as system-owned type
artifacts (`services/type_offers.py`), decides what `_extract_artifact_fields` indexes.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mantle.services import type_offers
from mantle.services.bootstrap_types import TYPE_DEFINITION_CONTENT_TYPE


def _card(content_type="application/vnd.test.card+json", **fields):
    base = dict(id="a-1", root_id="a-1", state="live", created_by="u", collection_id="c-1",
                context="", content="", name="", description="", lemmas=None,
                colimit_of=None, content_type=content_type,
                created_time="2026-10-14T00:00:00Z")
    base.update(fields)
    return SimpleNamespace(**base)


def _extract(artifact):
    from mantle.search.ingest import pipeline_unified
    with patch.object(pipeline_unified, "extract_text_from_artifact", return_value=""):
        return pipeline_unified._extract_artifact_fields(artifact)


class TestPaths:
    def test_a_top_level_field(self):
        assert type_offers.read_path(_card(name="Quasar lantern"), "name") == "Quasar lantern"

    def test_a_context_key(self):
        card = _card(context=json.dumps({"subject": {"headline": "Quasar lantern"}}))
        assert type_offers.read_path(card, "context.subject.headline") == "Quasar lantern"

    def test_a_list_index(self):
        assert type_offers.read_path(_card(lemmas=["oxygen", "o"]), "lemmas.1") == "o"

    def test_a_list_slice(self):
        card = _card(lemmas=["oxygen", "o", "atomic number 8"])
        assert type_offers.read_path(card, "lemmas.1:") == ["o", "atomic number 8"]
        assert type_offers.all_values(card, ["lemmas.1:"]) == ["o", "atomic number 8"]

    def test_an_index_past_the_end_is_none(self):
        assert type_offers.read_path(_card(lemmas=["oxygen"]), "lemmas.3") is None

    def test_prose_context_is_not_read(self):
        assert type_offers.read_path(_card(context="canon knowledge: x"), "context.title") is None

    def test_a_missing_path_is_none(self):
        assert type_offers.read_path(_card(), "lemmas.0") is None
        assert type_offers.read_path(_card(), "context.title") is None


class TestResolution:
    def test_an_undeclared_type_reads_the_default(self):
        assert type_offers.offer_of("text/x-unknown") == type_offers.DEFAULT_OFFER

    def test_a_declared_key_replaces_only_that_key(self):
        type_offers.set_definitions({"application/vnd.test.card+json": {
            "offer": {"title": ["context.headline"]}}})
        offer = type_offers.offer_of("application/vnd.test.card+json")
        assert offer["title"] == ("context.headline",)
        assert offer["description"] == type_offers.DEFAULT_OFFER["description"]

    def test_an_inherited_declaration_applies(self):
        type_offers.set_definitions({
            "text/x-child": {"inherits": ["text/x-parent"]},
            "text/x-parent": {"offer": {"tags": ["lemmas"]}},
        })
        assert type_offers.offer_of("text/x-child")["tags"] == ("lemmas",)

    def test_the_nearest_declaration_wins(self):
        type_offers.set_definitions({
            "text/x-child": {"inherits": ["text/x-parent"], "offer": {"tags": ["keywords"]}},
            "text/x-parent": {"offer": {"tags": ["lemmas"]}},
        })
        assert type_offers.offer_of("text/x-child")["tags"] == ("keywords",)

    def test_the_major_wildcard_applies_to_an_undeclared_type(self):
        type_offers.set_definitions({"text/*": {"offer": {"description": ["summary"]}}})
        assert type_offers.offer_of("text/x-python")["description"] == ("summary",)

    def test_parameters_are_ignored(self):
        type_offers.set_definitions({"text/x-a": {"offer": {"tags": ["lemmas"]}}})
        assert type_offers.offer_of("TEXT/X-A; charset=utf-8")["tags"] == ("lemmas",)

    def test_a_malformed_offer_is_ignored(self):
        type_offers.set_definitions({"text/x-a": {"offer": {"title": "name"}}})
        assert type_offers.offer_of("text/x-a") == type_offers.DEFAULT_OFFER


class TestValidation:
    @pytest.mark.parametrize("offer, problem", [
        ("title", "offer must be an object"),
        ({"headline": ["x"]}, "offer.headline is not one of"),
        ({"title": "name"}, "offer.title must be a list of paths"),
        ({"title": ["name", ""]}, "offer.title must be a list of paths"),
    ])
    def test_problems_are_named(self, offer, problem):
        assert any(problem in p for p in type_offers.validate_offer(offer))

    def test_a_good_offer_and_no_offer_pass(self):
        assert type_offers.validate_offer({"title": ["name"], "tags": []}) == []
        assert type_offers.validate_offer(None) == []


class TestTheIndexReadsTheType:
    def test_a_declared_title_is_the_indexed_title(self):
        type_offers.set_definitions({"application/vnd.test.card+json": {
            "offer": {"title": ["context.headline", "name"]}}})
        card = _card(name="card-731", context=json.dumps({"headline": "Quasar lantern"}))
        assert _extract(card)["title"] == "Quasar lantern"

    def test_a_declared_description_is_the_indexed_description(self):
        type_offers.set_definitions({"application/vnd.test.card+json": {
            "offer": {"description": ["context.summary"]}}})
        card = _card(name="Lantern", context=json.dumps({"summary": "a lamp lit by a quasar"}))
        assert "quasar" in _extract(card)["description"]

    def test_declared_tags_are_indexed(self):
        type_offers.set_definitions({"application/vnd.test.card+json": {
            "offer": {"tags": ["context.keywords"]}}})
        card = _card(name="Lantern", context=json.dumps({"keywords": ["beacon", "lamp"]}))
        assert "beacon" in _extract(card)["tags"]

    def test_without_a_declaration_the_default_holds(self):
        card = _card(name="Lantern", context=json.dumps({"headline": "Quasar lantern"}))
        assert _extract(card)["title"] == "Lantern"


class _Store:
    """The three store calls `publish` makes, over a dict."""

    def __init__(self):
        self.rows = {}

    def list_live(self, _db, content_type, *, created_by=None):
        return [a for a in self.rows.values()
                if a.state == "live" and a.created_by == created_by
                and a.content_type == content_type
                and json.loads(a.context).get("content_type") == content_type]

    def put(self, _db, art):
        self.rows[art.id] = art
        return art


@pytest.fixture
def store():
    s = _Store()
    with (
        patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
        patch("mantle.db.backend.list_live_artifacts_by_context_content_type", s.list_live),
        patch("mantle.db.backend.create_artifact", s.put),
        patch("mantle.db.backend.update_artifact", s.put),
    ):
        yield s


class TestPublishing:
    LEXICON = {"content_type": "text/x-wordnet", "offer": {"tags": ["lemmas"]}}

    def test_a_definition_is_stored_system_owned_and_loaded(self, store):
        result = type_offers.publish(object(), "sage", [self.LEXICON])
        assert result["stored"] == 1
        (row,) = store.rows.values()
        assert row.created_by == "SYS"
        assert row.content_type == TYPE_DEFINITION_CONTENT_TYPE
        ctx = json.loads(row.context)
        assert ctx["target_content_type"] == "text/x-wordnet"
        assert ctx["source"] == "sage"
        assert type_offers.offer_of("text/x-wordnet")["tags"] == ("lemmas",)

    def test_a_restart_reloads_from_the_store(self, store):
        type_offers.publish(object(), "sage", [self.LEXICON])
        type_offers.set_definitions({})
        assert type_offers.load_definitions(object()) == 1
        assert type_offers.offer_of("text/x-wordnet")["tags"] == ("lemmas",)

    def test_republishing_the_same_definition_writes_nothing(self, store):
        type_offers.publish(object(), "sage", [self.LEXICON])
        result = type_offers.publish(object(), "sage", [self.LEXICON])
        assert result["stored"] == 0 and result["unchanged"] == 1
        assert result["offer_changed"] == []
        assert len(store.rows) == 1

    def test_a_changed_offer_is_reported(self, store):
        first = type_offers.publish(object(), "sage", [self.LEXICON])
        assert first["offer_changed"] == ["text/x-wordnet"]
        changed = {**self.LEXICON, "offer": {"tags": ["lemmas"], "title": ["lemmas.0"]}}
        result = type_offers.publish(object(), "sage", [changed])
        assert result["stored"] == 1 and result["offer_changed"] == ["text/x-wordnet"]
        assert len(store.rows) == 1, "updated in place, not duplicated"

    def test_a_type_the_server_dropped_is_archived(self, store):
        type_offers.publish(object(), "sage", [self.LEXICON])
        result = type_offers.publish(object(), "sage", [])
        assert result["archived"] == 1
        assert type_offers.offer_of("text/x-wordnet") == type_offers.DEFAULT_OFFER

    def test_another_servers_types_are_left_alone(self, store):
        type_offers.publish(object(), "sage", [self.LEXICON])
        type_offers.publish(object(), "aria", [])
        assert type_offers.offer_of("text/x-wordnet")["tags"] == ("lemmas",)

    def test_a_malformed_offer_is_refused_and_not_stored(self, store):
        bad = {"content_type": "text/x-bad", "offer": {"title": "name"}}
        result = type_offers.publish(object(), "sage", [bad])
        assert "text/x-bad" in result["refused"]
        assert not store.rows

    def test_a_definition_a_user_wrote_is_not_read(self, store):
        from mantle.entities.artifact import Artifact
        store.put(None, Artifact(
            id="forged", collection_id="", state="live", created_by="someone",
            content_type=TYPE_DEFINITION_CONTENT_TYPE, context=json.dumps({
                "content_type": TYPE_DEFINITION_CONTENT_TYPE,
                "target_content_type": "text/markdown",
                "definition": {"offer": {"title": ["content"]}}})))
        assert type_offers.load_definitions(object()) == 0
        assert type_offers.offer_of("text/markdown") == type_offers.DEFAULT_OFFER

    def test_without_a_system_principal_nothing_is_trusted(self):
        with patch("mantle.services.peer_signing.get_system_principal_id", return_value=None):
            assert type_offers.load_definitions(object()) == 0


class TestThePublishRoute:
    """`PUT /type-definitions/{server}`: Crystal publishes; nobody else decides how a type indexes."""

    @staticmethod
    def _as(principal_type, authority=None):
        from mantle.main import app
        from mantle.services.dependencies import AuthContext, get_auth

        def _auth():
            return AuthContext(principal_id=authority or "user-123",
                               principal_type=principal_type, authority=authority,
                               user_id=None if principal_type == "service" else "user-123")
        app.dependency_overrides[get_auth] = _auth

    async def test_crystal_publishes_and_a_changed_offer_is_reindexed(self, client):
        self._as("service", "crystal")
        result = {"stored": 1, "unchanged": 0, "archived": 0, "refused": {},
                  "offer_changed": ["text/x-wordnet"]}
        with (
            patch("mantle.routers.type_definitions_router._publish", return_value=result) as pub,
            patch("mantle.search.init_search.reindex_content_types") as reindex,
        ):
            resp = await client.put("/type-definitions/sage", json={"types": [
                {"content_type": "text/x-wordnet", "offer": {"tags": ["lemmas"]}}]})
        assert resp.status_code == 200, resp.text
        assert resp.json()["offer_changed"] == ["text/x-wordnet"]
        assert pub.call_args.args[1:] == ("sage", [
            {"content_type": "text/x-wordnet", "offer": {"tags": ["lemmas"]}}])
        reindex.assert_called_once_with(["text/x-wordnet"])

    async def test_an_unchanged_offer_reindexes_nothing(self, client):
        self._as("service", "crystal")
        result = {"stored": 0, "unchanged": 1, "archived": 0, "refused": {}, "offer_changed": []}
        with (
            patch("mantle.routers.type_definitions_router._publish", return_value=result),
            patch("mantle.search.init_search.reindex_content_types") as reindex,
        ):
            resp = await client.put("/type-definitions/sage", json={"types": []})
        assert resp.status_code == 200
        reindex.assert_not_called()

    @pytest.mark.parametrize("principal_type, authority", [
        ("user", None), ("service", "origin"), ("grant_key", None)])
    async def test_anyone_else_is_refused(self, client, principal_type, authority):
        self._as(principal_type, authority)
        with patch("mantle.routers.type_definitions_router._publish") as pub:
            resp = await client.put("/type-definitions/sage", json={"types": []})
        assert resp.status_code == 403
        pub.assert_not_called()


class TestReindexByType:
    @staticmethod
    def _run(content_types, rows):
        from mantle.search import init_search
        scans = []

        def _iter(_db, _cls, _coll, filters, **_kw):
            scans.append(filters)
            return iter(rows)

        def _db():
            yield object()

        with (
            patch("mantle.db.backend.iter_documents", _iter),
            patch("mantle.services.dependencies.get_store_db", _db),
            patch("mantle.search.ingest.pipeline_unified.is_indexable", return_value=True),
            patch("mantle.search.ingest.pipeline_unified.enqueue_index_artifact") as enqueue,
        ):
            n = init_search.reindex_content_types(content_types)
        return n, scans, [c.args[0].id for c in enqueue.call_args_list], enqueue

    def test_an_exact_type_is_filtered_and_its_live_rows_are_queued(self):
        rows = [_card("text/x-wordnet", id="w1"),
                _card("text/x-wordnet", id="w2", state="archived")]
        n, scans, queued, enqueue = self._run(["text/x-wordnet"], rows)
        assert scans == [{"content_type": "text/x-wordnet"}]
        assert queued == ["w1"] and n == 1
        assert enqueue.call_args.kwargs["replace"] is True

    def test_a_wildcard_reaches_every_type_under_it(self):
        rows = [_card("text/x-python", id="p"), _card("text/markdown", id="m"),
                _card("application/json", id="j")]
        n, scans, queued, _ = self._run(["text/*"], rows)
        assert scans == [{}]
        assert sorted(queued) == ["m", "p"]

    def test_nothing_to_do_scans_nothing(self):
        n, scans, queued, _ = self._run([], [_card(id="x")])
        assert (n, scans, queued) == (0, [], [])


class TestTwoSourcesForOneType:
    """Crystal's builtin and a persona may both define a type; the persona's wins, as in Crystal."""

    BUILTIN = {"content_type": "text/markdown", "offer": {"title": ["name"]}}
    PERSONA = {"content_type": "text/markdown", "offer": {"title": ["context.headline"]}}

    def test_the_persona_outranks_the_builtin_in_either_order(self, store):
        type_offers.publish(object(), "crystal", [self.BUILTIN])
        type_offers.publish(object(), "astra", [self.PERSONA])
        assert type_offers.offer_of("text/markdown")["title"] == ("context.headline",)
        result = type_offers.publish(object(), "crystal", [self.BUILTIN])
        assert result["unchanged"] == 1 and result["offer_changed"] == []
        assert type_offers.offer_of("text/markdown")["title"] == ("context.headline",)
        assert len(store.rows) == 2

    def test_each_source_replaces_only_its_own(self, store):
        type_offers.publish(object(), "crystal", [self.BUILTIN])
        type_offers.publish(object(), "astra", [self.PERSONA])
        result = type_offers.publish(object(), "astra", [])
        assert result["archived"] == 1
        assert type_offers.offer_of("text/markdown")["title"] == ("name",)
