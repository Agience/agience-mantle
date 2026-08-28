"""`GET /content-url` and `GET /content` answer the same question the same way.

THE DEFECT. `content-url` gated on `content_key` alone; `content` accepts `content_key` OR
`content_cas_ref`. An artifact whose bytes are in the local CAS but which never got a
`content_key` therefore answered **404 here and 200 there** — and the 404's message said "No
downloadable content for this artifact" about content that downloads.

Why that is wrong by construction rather than by taste. The value `content-url` returns is the
PATH TO `content`. A pointer that refuses on a condition its target does not use is not a stricter
pointer, it is a broken one — and a URL-discovery endpoint is exactly what a client uses to ask
"is there content?", so every CAS-only artifact answered "no".

These tests assert the two AGREE, rather than pinning either one's condition. If the rule changes
again, it must change in both places or this fails — which is the property that was missing.
"""
from __future__ import annotations

import ast
import inspect

from mantle.routers import artifacts_router as ar

#: The context keys that say WHERE an artifact's bytes live. Both routes must agree on
#: these and only these; anything else either route reads is its own business.
_LOCATION_KEYS = {"content_key", "content_cas_ref"}


def _context_keys_tested(fn_name: str) -> set:
    """Which `ctx.get(...)` keys a handler tests before deciding there is no content.

    Read from the source rather than exercised through the app: the two routes have different
    dependencies and different happy paths, and the thing under test is the CONDITION, not the
    plumbing around it."""
    src = inspect.getsource(getattr(ar, fn_name))
    tree = ast.parse(src.lstrip())
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "get"):
            continue
        if getattr(f.value, "id", "") != "ctx":
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            k = node.args[0].value
            # ONLY the keys that say WHERE the bytes are. `get_artifact_content` also reads
            # `content_type` — to set the response media type, which a URL-returning route has no
            # use for — so comparing every `content_*` key would fail on a legitimate difference.
            # The first version of this test did exactly that, and the code was right.
            if k in _LOCATION_KEYS:
                keys.add(k)
    return keys


def test_both_routes_test_the_same_content_locations():
    """The assertion the defect would have failed."""
    url_keys = _context_keys_tested("content_url")
    content_keys = _context_keys_tested("get_artifact_content")

    assert url_keys, "content_url tests no content_* context key — the scan is looking in the wrong place"
    assert content_keys, "get_artifact_content tests none either — same"
    assert url_keys == content_keys, (
        "content-url tests %r while content tests %r. One route POINTS AT the other, so an "
        "artifact the target can serve must not be refused by the pointer."
        % (sorted(url_keys), sorted(content_keys)))


def test_the_cas_only_case_is_the_one_that_regressed():
    """Named explicitly, because it is the case the old condition dropped: bytes in the local CAS,
    no `content_key`. If `content_cas_ref` disappears from either route's test, this says why it
    mattered."""
    for fn in ("content_url", "get_artifact_content"):
        keys = _context_keys_tested(fn)
        assert "content_cas_ref" in keys, (
            "%s no longer considers `content_cas_ref`; a CAS-only artifact will be treated as "
            "having no content" % fn)


def test_content_url_still_points_at_the_content_route():
    """The premise the agreement rests on. If this ever returns a presigned object-store URL
    instead, the two conditions are free to differ again — and this test should be revisited rather
    than deleted."""
    src = inspect.getsource(ar.content_url)
    assert "/content" in src and "artifact_id" in src, src[-400:]
