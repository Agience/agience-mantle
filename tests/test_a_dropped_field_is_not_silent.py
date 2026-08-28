"""A field this API drops must say so.

The state this replaces. `CreateArtifactRequest` carries no `model_config`, so pydantic v2's
`extra="ignore"` applied: a body with `contentType` (camelCase typo of `content_type`) was accepted
with `201` and the field silently discarded. The caller's write "succeeded" and its data was gone.

`extra="forbid"` is the right end state and was deliberately NOT taken. `crystal/dispatcher.py`
posts a caller-supplied body VERBATIM to this route, so the population of fields arriving here is
whatever arbitrary content-type operations send; forbidding would convert an unknown number of
`201`s into `422`s with no way to know whose. **Characterise the population before migrating it** —
this warning is what makes it countable.

The two surfaces onto the SAME handler already disagree: all seven MCP tools declare
`"additionalProperties": False`, so the identical typo is refused over MCP and dropped over REST.
That is the argument FOR forbidding; it is not an argument for doing it blind.
"""
from __future__ import annotations

import logging

from mantle.routers.artifacts_router import CreateArtifactRequest


def test_an_unknown_field_is_logged_by_name(caplog):
    with caplog.at_level(logging.WARNING, logger="mantle.routers.artifacts_router"):
        CreateArtifactRequest.model_validate({"name": "x", "contentType": "oops"})
    assert any("contentType" in r.getMessage() for r in caplog.records), (
        "an ignored field was not named in any warning: %r" % [r.getMessage() for r in caplog.records])


def test_the_warning_names_every_unknown_field(caplog):
    with caplog.at_level(logging.WARNING, logger="mantle.routers.artifacts_router"):
        CreateArtifactRequest.model_validate(
            {"name": "x", "contentType": "a", "collectionId": "b", "descrption": "c"})
    joined = " ".join(r.getMessage() for r in caplog.records)
    for f in ("contentType", "collectionId", "descrption"):
        assert f in joined, "%s missing from %r" % (f, joined)


def test_a_clean_body_warns_about_nothing(caplog):
    """Guard against a warning that always fires — it would be noise, and noise is ignored."""
    with caplog.at_level(logging.WARNING, logger="mantle.routers.artifacts_router"):
        CreateArtifactRequest.model_validate({"name": "x", "content_type": "text/plain"})
    assert not [r for r in caplog.records if "unknown field" in r.getMessage()], (
        "a well-formed body produced an unknown-field warning: %r"
        % [r.getMessage() for r in caplog.records])


def test_the_unknown_field_is_still_accepted_and_dropped(caplog):
    """The behaviour is unchanged — only the silence is. If this starts failing, someone moved to
    `extra="forbid"`, which is a contract change that needs the population measured first."""
    m = CreateArtifactRequest.model_validate({"name": "x", "contentType": "oops"})
    assert m.name == "x"
    assert not hasattr(m, "contentType")
