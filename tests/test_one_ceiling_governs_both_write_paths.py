"""One configurable ceiling, enforced on BOTH write paths.

The state this replaces. `_MAX_CONTENT_BYTES = 2**31 - 1` is 2 GiB and is the CIPHER's
per-message limit, not a policy number — its own comment says so. It was enforced at the two
`PUT .../content` routes and **nowhere else**: neither `CreateArtifactRequest.content` nor
`UpdateArtifactRequest.content` carried a `max_length`, and no length check existed on the inline
create path. **The same bytes were capped on one write path and uncapped on the other**, and an
operator had no ceiling below two gigabytes on either.

`MANTLE_MAX_CONTENT_BYTES` defaults to the cipher bound, so nothing changes until it is set.

It is CLAMPED, not trusted. A value above the cipher bound cannot be honoured — AES-GCM will not
encrypt it — and a nonsense value falls back rather than crashing the node on a config typo. **A
ceiling that refuses to start is a ceiling nobody sets.**
"""
from __future__ import annotations

import logging

import pytest

from mantle.main import app
from mantle.routers.artifacts_router import (
    _MAX_CONTENT_BYTES,
    _refuse_oversize_inline,
    max_content_bytes,
)


@pytest.fixture
def ceiling(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("MANTLE_MAX_CONTENT_BYTES", raising=False)
        else:
            monkeypatch.setenv("MANTLE_MAX_CONTENT_BYTES", str(value))
    return _set


def test_the_default_is_todays_bound_so_nothing_changes_until_it_is_set(ceiling):
    ceiling(None)
    assert max_content_bytes() == _MAX_CONTENT_BYTES


def test_an_operator_can_set_a_smaller_ceiling(ceiling):
    ceiling(1024)
    assert max_content_bytes() == 1024


def test_a_ceiling_above_the_cipher_bound_is_clamped(ceiling):
    """AES-GCM will not encrypt more than 2**31-1 in one message, so a larger setting is a
    promise the node cannot keep."""
    ceiling(_MAX_CONTENT_BYTES * 4)
    assert max_content_bytes() == _MAX_CONTENT_BYTES


@pytest.mark.parametrize("bad", ["banana", "-5", "0", ""])
def test_a_nonsense_ceiling_falls_back_rather_than_crashing(bad, ceiling, caplog):
    ceiling(bad)
    with caplog.at_level(logging.WARNING, logger="mantle.routers.artifacts_router"):
        assert max_content_bytes() == _MAX_CONTENT_BYTES
    if bad:
        assert any("MANTLE_MAX_CONTENT_BYTES" in r.getMessage() for r in caplog.records), (
            "a rejected setting was ignored silently: %r" % [r.getMessage() for r in caplog.records])


def test_inline_content_over_the_ceiling_is_refused(ceiling):
    from fastapi import HTTPException
    ceiling(16)
    with pytest.raises(HTTPException) as exc:
        _refuse_oversize_inline("x" * 100, "test")
    assert exc.value.status_code == 413
    assert "MANTLE_MAX_CONTENT_BYTES" in exc.value.detail, exc.value.detail


@pytest.mark.parametrize("value", [None, "", "short"])
def test_content_within_the_ceiling_is_accepted(value, ceiling):
    ceiling(16)
    _refuse_oversize_inline(value, "test")


def test_the_refusal_names_which_ceiling_was_hit(ceiling):
    """An operator ceiling and the cipher bound are different facts and a caller can act on only
    one of them: the first means "ask this node's operator", the second means "split the upload".

    The cipher branch is reached with a string that REPORTS a huge length rather than having one --
    2 GiB cannot be allocated in a test, and the guard only ever asks for the length."""
    from fastapi import HTTPException

    class _Huge(str):
        def __len__(self):
            return _MAX_CONTENT_BYTES + 1

    ceiling(None)
    with pytest.raises(HTTPException) as exc:
        _refuse_oversize_inline(_Huge("x"), "test")
    assert exc.value.status_code == 413
    assert "AES-GCM" in exc.value.detail, exc.value.detail
    assert "MANTLE_MAX_CONTENT_BYTES" not in exc.value.detail, (
        "the cipher bound was reported as an operator setting: %s" % exc.value.detail)

    ceiling(64)
    with pytest.raises(HTTPException) as exc:
        _refuse_oversize_inline("x" * 128, "test")
    assert "MANTLE_MAX_CONTENT_BYTES" in exc.value.detail, exc.value.detail
    assert "AES-GCM" not in exc.value.detail, (
        "an operator ceiling was reported as the cipher's: %s" % exc.value.detail)


def test_a_long_string_is_refused_without_being_encoded(ceiling):
    """UTF-8 is at least one byte per character, so a string longer than the limit in CHARACTERS is
    over it in bytes. Encoding a body to measure it is the cost the ceiling exists to avoid."""
    from fastapi import HTTPException
    ceiling(8)

    class _Tripwire(str):
        def encode(self, *a, **kw):  # pragma: no cover - must never run
            raise AssertionError("the oversize string was encoded to measure it")

    with pytest.raises(HTTPException) as exc:
        _refuse_oversize_inline(_Tripwire("y" * 64), "test")
    assert exc.value.status_code == 413


def test_both_write_paths_declare_the_413():
    """The point of the ruling: the two paths must agree, in the spec as well as in the code."""
    paths = app.openapi()["paths"]
    for path, method in (("/artifacts", "post"),
                         ("/artifacts/{artifact_id}", "patch"),
                         ("/artifacts/{artifact_id}/content", "put")):
        codes = paths[path][method]["responses"]
        assert "413" in codes, "%s %s does not declare 413" % (method.upper(), path)
