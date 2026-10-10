"""Federation's peer calls carry the credential a peer's `/v1/invoke` requires.

Ember's `/v1/invoke` refuses a request whose `X-Ember-Token` does not match its `EMBER_INVOKE_TOKEN`,
so a mesh pull that omits the header is refused by every peer. The request is captured at
`urlopen`, the last point before the network, so what is asserted is what would be sent.
"""

from __future__ import annotations

import io
import json

import pytest

from mantle.mesh import federation


@pytest.fixture
def sent(monkeypatch):
    captured = []

    def _urlopen(req, timeout=None):
        captured.append(req)
        return io.BytesIO(json.dumps({"result": {"artifacts": []}}).encode())

    monkeypatch.setattr(federation.urllib.request, "urlopen", _urlopen)
    return captured


def test_the_invoke_token_is_sent_when_set(monkeypatch, sent):
    monkeypatch.setenv("EMBER_INVOKE_TOKEN", "shared-secret")
    federation._post("http://peer:8091", "op.mesh.export", {"offset": 0})
    (req,) = sent
    assert req.full_url == "http://peer:8091/v1/invoke"
    assert req.get_header("X-ember-token") == "shared-secret"


def test_no_header_is_invented_when_unset(monkeypatch, sent):
    """Unset, nothing is sent rather than an empty value a peer might compare against."""
    monkeypatch.delenv("EMBER_INVOKE_TOKEN", raising=False)
    federation._post("http://peer:8091", "op.mesh.export", {})
    (req,) = sent
    assert req.get_header("X-ember-token") is None
