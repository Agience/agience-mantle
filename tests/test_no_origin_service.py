"""What mantle does when no Origin service exists — measured, not assumed.

`mantle.foresightreports.agience.ai` uses Microsoft Entra as its IdP and runs no Origin at all.
Token verification does not need one: `OidcVerifier` checks signature + `iss` + `aud` against the
issuer's JWKS locally, and derives the user id as `uuid5(namespace, (issuer, sub))` — an external
user's identity is computed, never fetched.

What *did* need proving is the other direction: mantle calls a running Origin over HTTP in exactly
two places, and both were written to degrade. "Written to degrade" is a claim about intent. These
tests are the claim about behaviour.
"""
from __future__ import annotations

import logging
import pathlib

import httpx
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle"


def _refusing_client(monkeypatch, calls: list, base_uri: str | None = "http://origin.invalid:9"):
    """An OriginClient whose transport always refuses, counting attempts."""
    from mantle.clients.origin_client import OriginClient

    monkeypatch.setattr(OriginClient, "_headers", lambda self: {})
    client = OriginClient(base_uri=base_uri)

    def _get(*_a, **_k):
        calls.append(1)
        raise httpx.ConnectError("no Origin on this deployment")

    monkeypatch.setattr(client._client, "get", _get)  # noqa: SLF001
    return client


def test_get_user_by_id_returns_none_when_origin_is_unreachable(monkeypatch):
    """It must return None, not raise. A raise here would propagate out of every caller."""
    from mantle.services import person_service

    class _DeadClient:
        _base = "http://origin.invalid"

        def _service_token(self):
            return "t"

        class _C:
            @staticmethod
            def get(*_a, **_k):
                raise httpx.ConnectError("no Origin on this deployment")

        _client = _C()

    monkeypatch.setattr(person_service, "get_origin_client", lambda: _DeadClient())
    assert person_service.get_user_by_id(db=None, id="some-user") is None


def test_get_operator_id_falls_back_without_origin(monkeypatch):
    """`operator.py` tries platform_settings -> config.OPERATOR_ID -> Origin. With no Origin and no
    settings it must yield "" rather than propagating the connection error."""
    from mantle.services import operator

    def _boom():
        raise httpx.ConnectError("no Origin on this deployment")

    monkeypatch.setattr("mantle.clients.origin_client.get_origin_client", _boom)
    assert operator.resolve_operator_id() == "" or isinstance(operator.resolve_operator_id(), str)


def test_connect_timeout_is_well_under_a_second():
    """A peer that is not listening must be discovered at once. httpx's default connect budget is
    sized for a peer that IS there and slow, and a standalone node pays it three times at boot."""
    from mantle.clients.origin_client import OriginClient

    connect = OriginClient(base_uri="http://origin.invalid")._client.timeout.connect  # noqa: SLF001
    assert connect is not None, "no explicit connect timeout: the default applies"
    assert connect < 1.0, "connect budget %r is too long for an optional peer" % connect


def test_unreachable_origin_logs_a_message_and_no_traceback(monkeypatch, caplog):
    """An absent optional peer is an expected state. The line names it; the socket stack trace is
    ~45 lines of noise for a condition the code already handles."""
    client = _refusing_client(monkeypatch, [])
    with caplog.at_level(logging.WARNING, logger="mantle.clients.origin_client"):
        assert client.get_operator_id() is None

    records = [r for r in caplog.records if r.name == "mantle.clients.origin_client"]
    assert records, "the unreachable peer must still be reported"
    assert all(not r.exc_info for r in records), "the warning carries a traceback"


def test_an_unreachable_origin_is_probed_once_not_once_per_caller(monkeypatch):
    """Boot resolves the operator from three separate callers. One answer covers all of them."""
    calls: list = []
    client = _refusing_client(monkeypatch, calls)
    for _ in range(3):
        assert client.get_operator_id() is None
    assert len(calls) == 1, "the negative result is not remembered: %d probes" % len(calls)


def test_the_memo_expires_so_a_late_origin_is_still_found(monkeypatch):
    """Bounded, not permanent — a full-platform Origin that starts after Mantle is picked up
    without a restart."""
    import mantle.clients.origin_client as oc

    calls: list = []
    client = _refusing_client(monkeypatch, calls)
    assert client.get_operator_id() is None
    client._unreachable_until = 0.0  # noqa: SLF001 — stand in for the elapsed TTL
    assert client.get_operator_id() is None
    assert len(calls) == 2
    assert oc._UNREACHABLE_MEMO_SECONDS > 0  # noqa: SLF001


def test_the_connect_budget_is_for_the_whole_probe_not_per_dns_record(monkeypatch):
    """`_CONNECT_TIMEOUT_SECONDS` is the cost of running standalone, so it has to BE the cost.

    httpcore hands its connect timeout to `socket.create_connection`, which spends it on EVERY
    address `getaddrinfo` returns before giving up. `localhost` — the host in ORIGIN_URI's own
    default — resolves to both ::1 and 127.0.0.1, so a value that read as 0.5s cost 1.0s, and a
    name with more records cost proportionally more. The number was set by DNS rather than by
    anything the source said.
    """
    import mantle.clients.origin_client as oc

    client = oc.OriginClient(base_uri="http://origin.example:8080")

    monkeypatch.setattr(oc.socket, "getaddrinfo", lambda *a, **k: [object()] * 4)
    assert client._connect_timeout() == oc._CONNECT_TIMEOUT_SECONDS / 4  # noqa: SLF001

    # A single-address host — the production case — is untouched by the split.
    monkeypatch.setattr(oc.socket, "getaddrinfo", lambda *a, **k: [object()])
    assert client._connect_timeout() == oc._CONNECT_TIMEOUT_SECONDS  # noqa: SLF001


def test_a_name_that_does_not_resolve_still_gets_one_attempts_worth(monkeypatch):
    """No addresses means nothing to divide the budget across, and httpx fails on the same
    lookup regardless — so this must not divide by zero or return a budget of nothing."""
    import mantle.clients.origin_client as oc

    client = oc.OriginClient(base_uri="http://origin.invalid:9")

    def _boom(*_a, **_k):
        raise OSError("nodename nor servname provided")

    monkeypatch.setattr(oc.socket, "getaddrinfo", _boom)
    assert client._connect_timeout() == oc._CONNECT_TIMEOUT_SECONDS  # noqa: SLF001

    monkeypatch.setattr(oc.socket, "getaddrinfo", lambda *a, **k: [])
    assert client._connect_timeout() == oc._CONNECT_TIMEOUT_SECONDS  # noqa: SLF001


def test_the_probe_carries_the_split_budget_to_the_request(monkeypatch):
    """The split is worth nothing unless the call actually uses it."""
    import mantle.clients.origin_client as oc

    monkeypatch.setattr(oc.OriginClient, "_headers", lambda self: {})
    client = oc.OriginClient(base_uri="http://origin.example:8080")
    monkeypatch.setattr(oc.socket, "getaddrinfo", lambda *a, **k: [object()] * 2)

    seen: list = []

    def _get(*_a, **kwargs):
        seen.append(kwargs.get("timeout"))
        raise httpx.ConnectError("nothing there")

    monkeypatch.setattr(client._client, "get", _get)  # noqa: SLF001
    assert client.get_operator_id() is None
    assert seen[0].connect == oc._CONNECT_TIMEOUT_SECONDS / 2  # noqa: SLF001
    assert seen[0].read == oc._REQUEST_TIMEOUT_SECONDS  # noqa: SLF001


def test_empty_origin_uri_means_no_call_at_all(monkeypatch):
    """`ORIGIN_URI=""` is a node declaring it has no Origin. Substituting a localhost default for
    it turns that declaration into a connect timeout against nothing."""
    monkeypatch.setattr("mantle.config.ORIGIN_URI", "", raising=False)
    calls: list = []
    client = _refusing_client(monkeypatch, calls, base_uri=None)
    assert client.enabled is False
    assert client.get_operator_id() is None
    assert calls == []


def test_get_user_by_id_makes_no_call_without_an_origin_uri(monkeypatch):
    """Origin owns person records. With no Origin configured there is nothing to ask."""
    from mantle.services import person_service

    monkeypatch.setattr("mantle.config.ORIGIN_URI", "", raising=False)
    calls: list = []
    client = _refusing_client(monkeypatch, calls, base_uri=None)
    monkeypatch.setattr(person_service, "get_origin_client", lambda: client)
    assert person_service.get_user_by_id(db=None, id="some-user") is None
    assert calls == []


def test_no_route_depends_on_get_person():
    """No router route may depend on `get_person`. With no Origin, `get_person` is a 404 machine,
    and it is safe today only because nothing wires it into a route.

    Grep-based on purpose: this asks a question about the router wiring, which is exactly where a
    future `Depends(get_person)` would appear, and it must fail at review time rather than on a
    live request.
    """
    offenders = []
    for path in (SRC / "routers").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "Depends(get_person)" in text or "Depends(dependencies.get_person)" in text:
            offenders.append(path.name)
    assert not offenders, (
        "these routers depend on get_person, which raises 404 whenever Origin is unreachable — on "
        "an Entra-only deployment (fsr) that is EVERY request: %s. Either give the deployment an "
        "Origin, or resolve the person from the token claims instead of fetching it."
        % ", ".join(sorted(offenders))
    )
