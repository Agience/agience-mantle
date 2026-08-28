"""Recall's retired fields leave a trace, so the compatibility window can be closed on evidence
.

The item asks "when does the window close?" and the code could not answer. `ArtifactRecallRequest`
ignores unknown fields by documented decision — a client still sending a retired name gets a normal
search, not a 422. The retired set was measured inert on every path: `embedding`, `aperture`,
`use_hybrid`, `content_types`. But an ignored field left no trace anywhere, so *"is anyone still
sending `use_hybrid`?"* had no answer and closing the window was a guess about callers in the field.

Same instrument as `CreateArtifactRequest`, for the same reason: **characterise the
population before migrating it.** The behaviour is unchanged — only the silence is.

`populate_by_name=True` means an alias is also a legal key, so both spellings must count as
known. `from` is the alias of `from_`; reporting it as a typo would make the log useless on the
one field most likely to appear.
"""
from __future__ import annotations

import logging

import pytest

from mantle.routers.artifacts_router import ArtifactRecallRequest

RETIRED = ["embedding", "aperture", "use_hybrid", "content_types"]


@pytest.mark.parametrize("field", RETIRED)
def test_each_retired_field_is_named_when_it_arrives(field, caplog):
    with caplog.at_level(logging.WARNING, logger="mantle.routers.artifacts_router"):
        ArtifactRecallRequest.model_validate({"query_text": "x", field: "whatever"})
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert field in joined, "%s arrived and left no trace: %r" % (field, joined)


def test_a_retired_field_is_still_accepted(caplog):
    """The window is still OPEN. If this starts failing someone closed it — which is a decision
    about callers in the field, not a tidy-up."""
    parsed = ArtifactRecallRequest.model_validate({"query_text": "x", "use_hybrid": True})
    assert parsed.query_text == "x"
    assert not hasattr(parsed, "use_hybrid")


def test_a_clean_request_warns_about_nothing(caplog):
    """A warning that always fires is noise, and noise is ignored."""
    with caplog.at_level(logging.WARNING, logger="mantle.routers.artifacts_router"):
        ArtifactRecallRequest.model_validate({"query_text": "x", "size": 5, "sort": "recency"})
    assert not [r for r in caplog.records if "unknown field" in r.getMessage()], (
        [r.getMessage() for r in caplog.records])


def test_an_alias_is_not_reported_as_unknown(caplog):
    """`from` is the alias of `from_`. Reporting it would make the log useless on the field most
    likely to show up in a real request."""
    with caplog.at_level(logging.WARNING, logger="mantle.routers.artifacts_router"):
        ArtifactRecallRequest.model_validate({"query_text": "x", "from": 0})
    assert not [r for r in caplog.records if "unknown field" in r.getMessage()], (
        "the `from` alias was reported as unknown: %r"
        % [r.getMessage() for r in caplog.records])
