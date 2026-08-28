r"""A tag, a collection, a group and an attribute are the same thing: an edge (§112).

[John, 2026-08-22] — so there is no `tags` field to read. Membership IS the tag set, and
`collection_id` / `collections` are the field mirror of the `contains` edges that record it.

This replaced `parse_tags_from_context`, which read a `tags` key out of the context blob: a second,
parallel answer to a question the graph already answers, and the two could disagree with nothing to
notice — an artifact moved between collections kept whatever tag strings its context carried.
"""
from __future__ import annotations

from mantle.search.field_filters import FILTERABLE_FIELDS, parse_context
from mantle.search.ingest.pipeline_unified import _group_terms


class _Artifact:
    def __init__(self, collections=None, collection_id=""):
        self.collections = collections or []
        self.collection_id = collection_id


class TestMembershipIsTheTagSet:

    def test_both_names_for_membership_are_read(self):
        """`curate._memberships` maintains `collection_id` AND `collections`; reading one loses
        artifacts written by the path that files the other."""
        assert _group_terms(_Artifact(["collection:a/b"], "")) == ["a", "b"]
        assert _group_terms(_Artifact([], "collection:a/b")) == ["a", "b"]

    def test_a_group_id_is_split_into_the_words_a_person_would_type(self):
        assert _group_terms(_Artifact(["collection:agience-pharos/features"])) == [
            "agience", "pharos", "features"]

    def test_terms_are_deduped_across_groups(self):
        assert _group_terms(_Artifact(["collection:a/b", "collection:a/c"])) == ["a", "b", "c"]

    def test_no_membership_is_no_tags_rather_than_an_error(self):
        assert _group_terms(_Artifact()) == []


class TestTheSchemeIsNotATag:
    """The third time this defect has appeared today, from a third direction."""

    def test_the_scheme_prefix_never_becomes_a_tag(self):
        """`collection:` is a type marker. Kept, it tags every artifact that is in a collection —
        which is all of them — and a term carried by every member of a corpus cannot distinguish
        between them. Same shape as `canon knowledge` in the offer (§111) and the cardinal numbers
        in the position set (§110)."""
        terms = _group_terms(_Artifact(["collection:agience-pharos/features"]))
        assert "collection" not in terms, terms

    def test_a_bare_id_with_no_scheme_is_still_split(self):
        assert _group_terms(_Artifact(["stage.0.lexicon"])) == ["stage", "lexicon"]

    def test_a_numeric_path_segment_is_not_a_tag(self):
        assert "0" not in _group_terms(_Artifact(["stage.0.lexicon"]))


class TestTheQuerySideAgrees:
    """`tag:x` must ask the same question the indexer answered, or the filter selects hits whose
    echoed tags do not contain `x`."""

    def _tags(self, doc):
        return FILTERABLE_FIELDS["tags"].read(doc, parse_context(doc))

    def test_tags_are_read_from_membership(self):
        doc = {"collections": ["collection:a/b"]}
        assert self._tags(doc) == ["collection:a/b"]

    def test_a_legacy_context_tags_key_is_UNIONED_not_replaced(self):
        """Every artifact belongs to a group, so membership is never empty. Preferring it would
        make a pre-flattening row's stated tags unreachable for as long as the row exists —
        `tag:budget` would answer nothing for the whole corpus written before the flattening."""
        legacy = {"context": {"tags": ["old"]}}
        assert self._tags(legacy) == ["old"]
        both = {"collections": ["collection:a/b"], "context": {"tags": ["old"]}}
        assert self._tags(both) == ["collection:a/b", "old"]

    def test_tags_are_deduped_on_the_query_side_too(self):
        doc = {"collections": ["collection:a/b"], "collection_id": "collection:a/b"}
        assert self._tags(doc) == ["collection:a/b"]


class TestTheOfferIsReadTopLevel:

    def test_description_prefers_the_top_level_offer(self):
        doc = {"description": "the real offer", "context": {"description": "a stale carrier"}}
        assert FILTERABLE_FIELDS["description"].read(doc, parse_context(doc)) == "the real offer"

    def test_a_structured_context_still_answers_for_older_rows(self):
        doc = {"context": {"description": "a stale carrier"}}
        assert FILTERABLE_FIELDS["description"].read(doc, parse_context(doc)) == "a stale carrier"
