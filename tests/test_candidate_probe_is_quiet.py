"""A candidate miss is not an error; the final failure still is.

⛔ WHAT THIS FIXES. `doc_boundary` hydrates a body by trying ORDERED CANDIDATES — the stamped
principal, then the collection scope, then `created_by`. Missing on the earlier ones is the whole
design: the comment there records the 310,003-artifact case where only the collection scope
opened the blob. Every one of those expected misses went through `_decrypt_envelope` and was
logged at ERROR with a traceback.

Measured on srv, 2026-09-10: **744 "failed to decrypt stored content" errors in a day, against
ZERO actual failures** — "content hydration failed" never appeared once. Every read succeeded on
a later candidate. An error emitted by correct behaviour is how a log stops being read, and it
very nearly had me report a healthy content tier as 100% unreadable.

The two halves are tested together on purpose. Silencing the probe is only safe if a genuine
failure is still loud, so the second test is what makes the first one safe to keep.
"""
from __future__ import annotations

import logging

import pytest

from mantle.services import content_service


class _Boom(Exception):
    pass


@pytest.fixture()
def always_fails(monkeypatch):
    """Make the underlying decrypt raise, so only the LOGGING differs between the cases."""
    class _FakeCrypto:
        @staticmethod
        def decrypt_content(*a, **k):
            raise _Boom("nope")

    import sys
    monkeypatch.setitem(sys.modules, "mantle.services.content_crypto", _FakeCrypto)
    return _FakeCrypto


def _attempt(probing: bool):
    return content_service._decrypt_envelope(
        "cas/deadbeef", b"MEC1xxxx", "some-principal",
        require_encrypted=True, collection_id="col-1", probing=probing,
    )


def test_a_probe_miss_does_not_log_an_error(always_fails, caplog):
    """⛔ THE REGRESSION. 744 of these a day, on a node where every read succeeded."""
    with caplog.at_level(logging.DEBUG, logger=content_service.logger.name):
        with pytest.raises(Exception):
            _attempt(probing=True)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not errors, f"a probe miss logged at ERROR: {[r.getMessage() for r in errors]}"


def test_a_real_failure_still_logs_an_error(always_fails, caplog):
    """⛔ THE GUARD ON THE GUARD. Without this, `probing=True` everywhere would silence genuine
    content loss and every other test here would still pass."""
    with caplog.at_level(logging.DEBUG, logger=content_service.logger.name):
        with pytest.raises(Exception):
            _attempt(probing=False)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a non-probe decrypt failure must still be an ERROR"
    assert "could not be decrypted" in str(errors[0].getMessage()) or \
           "failed to decrypt" in str(errors[0].getMessage())


def test_both_paths_still_raise(always_fails):
    """Quieter logging must not become a quieter RESULT. Both still refuse to return ciphertext."""
    for probing in (True, False):
        with pytest.raises(Exception):
            _attempt(probing=probing)


def test_the_default_is_loud():
    """A caller that says nothing gets the old behaviour. Silence must be opt-in, so a new call
    site cannot inherit it by accident."""
    import inspect
    sig = inspect.signature(content_service._decrypt_envelope)
    assert sig.parameters["probing"].default is False
