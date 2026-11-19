"""`type:text/*` names the text family, as the type definitions do; `*/*` names every type."""

from __future__ import annotations

from mantle.search.field_filters import compile_filters
from mantle.search.query_parser import parse_query


def _keep(query: str, content_type: str) -> bool:
    pred = compile_filters(parse_query(query).filters)
    assert pred is not None
    return pred({"id": "a", "content_type": content_type})


def test_a_family_matches_every_type_under_it():
    assert _keep("type:text/*", "text/markdown")
    assert _keep("type:text/*", "text/plain")
    assert not _keep("type:text/*", "application/pdf")


def test_a_family_sits_beside_exact_types():
    assert _keep("type:image/*,application/pdf", "application/pdf")
    assert _keep("type:image/*,application/pdf", "image/png")
    assert not _keep("type:image/*,application/pdf", "application/json")


def test_every_type_is_the_family_of_all():
    assert _keep("type:*/*", "application/vnd.agience.workspace+json")


def test_a_family_is_not_a_prefix_of_a_longer_major_type():
    assert not _keep("type:text/*", "textual/x")


def test_another_field_takes_the_value_as_written():
    pred = compile_filters(parse_query("tag:text/*").filters)
    assert pred is not None
    assert not pred({"id": "a", "context": '{"tags": ["text/markdown"]}'})
