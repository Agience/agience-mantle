"""An artifact id reaches the server whole, whatever characters it contains.

## What went wrong without this

An artifact id is an opaque string, and this corpus's ids carry characters that mean something in a
URL. Interpolated into an f-string, `canon:best-practices#intro` yields:

    httpx.Request('GET', f"{base}/artifacts/{artifact_id}")
        path     = /artifacts/canon:best-practices
        fragment = 'intro'

A fragment is never sent to a server. Every canon id has a `#`, so every fetch of that documentation
asked for `canon:best-practices` instead. Measured on the live store, none of the 276 truncated
forms is an artifact, so the request 404s — a loud failure covering all 6,480 canon artifacts, in 35
places across seven persona servers. Ids without a `#` (`wn-*`, `wiki-*`) were never affected.

## What is pinned

That the id survives as ONE segment — `#` cannot start a fragment, `?` cannot start a query, `/`
cannot introduce a segment — and that route structure appended after it stays structure rather than
becoming part of the id.
"""
from __future__ import annotations

import httpx
import pytest

from mantle.clients.artifact_helpers import artifact_url

BASE = "https://mantle.example"


@pytest.mark.parametrize("artifact_id", [
    "canon:best-practices#intro",          # every canon id carries a '#'
    "canon:manifesto#1-2",
    "wn-glacier.n.01",                     # dots are ordinary and must not be touched oddly
    "wn-oewn-13558632-n",
    "wiki-simple-3518",
    "a?b=c",                               # would otherwise start a query string
    "a/b",                                 # would otherwise introduce a segment
    "a%b",                                 # a stray percent must not be read as an escape
    "a b",                                 # a space is not a separator
])
def test_the_whole_id_reaches_the_path(artifact_id):
    url = httpx.Request("GET", artifact_url(BASE, artifact_id)).url

    assert url.fragment == "", (
        "%r put %r in a fragment, which is never sent to a server" % (artifact_id, url.fragment))
    assert not url.query, "%r leaked into the query string: %r" % (artifact_id, url.query)
    # Starlette decodes a path parameter, so what the route receives is the id again.
    from urllib.parse import unquote
    assert unquote(url.path) == "/artifacts/" + artifact_id


def test_route_structure_after_the_id_stays_structure():
    """`children` and `content-url` are parts of the ROUTE. They are appended unencoded, so the
    server still matches `/artifacts/{id}/children` rather than reading them as part of the id."""
    url = artifact_url(BASE, "canon:a#b", "children")
    assert url == BASE + "/artifacts/canon%3Aa%23b/children"

    url = artifact_url(BASE, "x", "op", "invoke")
    assert url == BASE + "/artifacts/x/op/invoke"


def test_a_trailing_slash_on_the_base_does_not_double():
    assert artifact_url("https://m/", "x") == "https://m/artifacts/x"


def test_an_id_that_looks_like_a_path_is_still_one_segment():
    """`a/b` is one id, not two segments. Encoding is what makes the route see it that way."""
    assert artifact_url(BASE, "a/b") == BASE + "/artifacts/a%2Fb"


def test_the_naive_form_is_what_this_replaces():
    """The control: interpolating the id directly is what loses the tail. If this ever stops being
    true the helper is no longer earning its place — but it is true of every URL library, because
    it is what a fragment means."""
    naive = httpx.Request("GET", "%s/artifacts/%s" % (BASE, "canon:best-practices#intro")).url
    assert naive.fragment == "intro"
    assert naive.path == "/artifacts/canon:best-practices"
