"""Recall runs `query_text` as the caller sends it. A caller that offers filter choices (a phrase,
a tag, content types) writes them into the text as query filters (`="..."`, `tag:`, `type:a,b`)
before sending; the request names no field for them."""
from __future__ import annotations

from mantle.main import app


def test_the_request_names_no_filter_field():
    props = app.openapi()["components"]["schemas"]["ArtifactRecallRequest"]["properties"]
    assert not {"exact", "tag", "tag_exact", "types"} & set(props)
