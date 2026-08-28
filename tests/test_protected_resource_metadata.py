"""RFC 9728 discovery — how a client learns where to authenticate to this node.

Without it, `POST /mcp` with no token returns a bare `401` — no `WWW-Authenticate` — and
`/.well-known/oauth-protected-resource` is `404`. A standards-compliant MCP client is told "no"
with no way to learn how to say yes, and can only authenticate if configured out of band.

It is also the missing half of delegation: an application tier acts for a user by exchanging that
user's token for an RFC 8693 delegation, and an organon acting on a caller-supplied resource id
carries no authority without one. The token still has to arrive from somewhere, and discovery is
where a client finds out from whom to get it.

The negative controls are the substance here: metadata that always answers, and a challenge that is
always attached, would both pass a happy-path test while being wrong.
"""
from __future__ import annotations

import importlib

import pytest


PATH = "/.well-known/oauth-protected-resource"


def _reload(monkeypatch, **env):
    """Rebuild config with a controlled environment. Returns the config module."""
    for k in ("AGIENCE_TRUSTED_ISSUERS", "ORIGIN_URI", "AUTHORITY_ISSUER",
              "MANTLE_URI", "MANTLE_OIDC_SCOPE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AGIENCE_NO_DOTENV", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import mantle.config as cfg
    importlib.reload(cfg)
    return cfg


# ---------------------------------------------------------------------------
# authorization_servers() — the one definition
# ---------------------------------------------------------------------------
def test_platform_authority_is_the_server_when_no_external_issuer(monkeypatch):
    cfg = _reload(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai")
    assert cfg.authorization_servers() == ["https://origin.home.agience.ai"]


def test_an_external_issuer_wins_and_index_zero_is_the_browsers(monkeypatch):
    """Order is load-bearing: `browse_page._boot` sends the sign-in button to index 0, and a
    browser can be sent to exactly one provider. If this list reordered, humans would be sent to a
    different issuer than the metadata advertises."""
    import json
    cfg = _reload(
        monkeypatch,
        ORIGIN_URI="https://origin.home.agience.ai",
        AGIENCE_TRUSTED_ISSUERS=json.dumps([
            {"issuer": "https://login.microsoftonline.com/t/v2.0", "audience": "a"},
            {"issuer": "https://accounts.foresightreports.com/x/v2.0/", "audience": "b"},
        ]),
    )
    assert cfg.authorization_servers() == [
        "https://login.microsoftonline.com/t/v2.0",
        "https://accounts.foresightreports.com/x/v2.0",   # trailing slash normalised
    ]


def test_the_signin_button_and_the_metadata_cannot_disagree(monkeypatch):
    """The reason the helper exists: two readings of "which issuer" would send a human and an MCP
    client to different authorization servers, and only the human's failure gets noticed."""
    import json
    cfg = _reload(
        monkeypatch,
        ORIGIN_URI="https://origin.home.agience.ai",
        AGIENCE_TRUSTED_ISSUERS=json.dumps([{"issuer": "https://idp.example/v2.0", "audience": "a"}]),
    )
    import mantle.ui.browse_page as bp
    importlib.reload(bp)
    assert bp._boot()["issuer"] == cfg.authorization_servers()[0]


def test_no_issuer_configured_yields_an_empty_list(monkeypatch):
    cfg = _reload(monkeypatch, ORIGIN_URI="", AUTHORITY_ISSUER="")
    assert cfg.authorization_servers() == []


# ---------------------------------------------------------------------------
# Declared, or merely defaulted
# ---------------------------------------------------------------------------
# A standalone node — no `ORIGIN_URI`, no `AUTHORITY_ISSUER`, no stored row — publishes no
# `authorization_servers` key. `ORIGIN_URI`'s Phase-1 default is non-empty, so a fallback that read
# non-empty as declared would publish `["http://localhost:8080"]`, and an MCP client acting on that
# dials `http://localhost:8080/.well-known/oauth-authorization-server`, finds nothing, and dies at
# the one step whose purpose is to tell it where to go. An absent key is actionable.

def test_a_standalone_node_advertises_NO_authorization_server(monkeypatch):
    """The default install: nothing set. The Phase-1 default is a developer convenience, not a
    statement that an authorization server answers there."""
    cfg = _reload(monkeypatch)                       # no environment at all
    assert cfg.ORIGIN_URI == "http://localhost:8080", "the Phase-1 default moved"
    assert cfg.AUTHORITY_ISSUER == "http://localhost:8080"
    assert cfg.authorization_servers() == []


def test_declaring_the_SAME_value_the_default_holds_still_counts(monkeypatch):
    """Provenance, not value. An operator who writes `ORIGIN_URI=http://localhost:8080` is running
    an Origin there and has said so; a node that never mentioned it has not. Keying off the string
    would silence the one deployment that most needs the metadata — a local platform stack."""
    cfg = _reload(monkeypatch, ORIGIN_URI="http://localhost:8080")
    assert cfg.authorization_servers() == ["http://localhost:8080"]


def test_an_empty_env_var_declares_nothing(monkeypatch):
    """The same reading `load_settings_from_db` already applies to an empty env var: a stock `.env`
    template line is not an operator suppressing a key."""
    cfg = _reload(monkeypatch, ORIGIN_URI="   ")
    assert cfg.authorization_servers() == []


def test_AUTHORITY_ISSUER_alone_is_a_declaration(monkeypatch):
    cfg = _reload(monkeypatch, AUTHORITY_ISSUER="https://idp.example")
    assert cfg.authorization_servers() == ["https://idp.example"]


def _with_store(cfg, monkeypatch, **rows):
    """Run config's Phase 2 against a fake settings store holding exactly `rows`."""
    for k in ("AGIENCE_TRUSTED_ISSUERS", "ORIGIN_URI", "AUTHORITY_ISSUER"):
        monkeypatch.delenv(k, raising=False)
    cfg.set_settings_provider(lambda key: rows.get(key))
    cfg.load_settings_from_db()
    return cfg


def test_a_stored_row_is_a_declaration(monkeypatch):
    """The third way to say it. An operator who set `branding.origin_uri` in the settings store
    named an authority as surely as an env var does."""
    cfg = _reload(monkeypatch)
    _with_store(cfg, monkeypatch, **{"branding.origin_uri": "https://origin.stored.example"})
    assert cfg.authorization_servers() == ["https://origin.stored.example"]


def test_the_DEFAULTS_mirror_coming_back_from_the_store_is_not_a_declaration(monkeypatch):
    """`settings.get()` falls back to `platform_settings_service.DEFAULTS`, which REFLECTS config's
    own Phase-1 attribute, so it never returns None and this branch runs on a node holding no row at
    all. The mirror is exactly what an absent row produces, so a value equal to the Phase-1 default
    says nothing the default did not already say — otherwise every node in existence would read as
    having declared an authorization server."""
    cfg = _reload(monkeypatch)
    _with_store(cfg, monkeypatch, **{"branding.origin_uri": "http://localhost:8080"})
    assert cfg.ORIGIN_URI == "http://localhost:8080"
    assert cfg.authorization_servers() == []


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def _client(monkeypatch, **env):
    _reload(monkeypatch, **env)
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    import mantle.config as cfg
    import mantle.main as main
    importlib.reload(main)
    return TestClient(main.app), cfg


def test_metadata_names_the_resource_and_its_authorization_server(monkeypatch):
    client, _ = _client(
        monkeypatch,
        ORIGIN_URI="https://origin.home.agience.ai",
        MANTLE_URI="https://mantle.home.agience.ai",
        MANTLE_OIDC_SCOPE="openid profile email",
    )
    r = client.get(PATH)
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["resource"] == "https://mantle.home.agience.ai"
    assert doc["authorization_servers"] == ["https://origin.home.agience.ai"]
    assert doc["bearer_methods_supported"] == ["header"]
    assert doc["scopes_supported"] == ["openid", "profile", "email"]


def test_metadata_is_public_because_it_must_be(monkeypatch):
    """A document you must already be authenticated to read cannot tell you how to authenticate.
    It carries no secret — issuer, resource and scope names are all things the redirect is about
    to disclose anyway."""
    client, _ = _client(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai",
                        MANTLE_URI="https://mantle.home.agience.ai")
    assert client.get(PATH).status_code == 200          # no Authorization header at all


def test_an_unconfigured_node_OMITS_the_key_rather_than_publishing_an_empty_one(monkeypatch):
    """`"authorization_servers": []` is a positive claim that there is nowhere to authenticate. The
    key's absence says the node has not been configured to say — a client can act on the second and
    can only despair at the first."""
    client, _ = _client(monkeypatch, ORIGIN_URI="", AUTHORITY_ISSUER="",
                        MANTLE_URI="https://mantle.home.agience.ai")
    doc = client.get(PATH).json()
    assert "authorization_servers" not in doc
    assert doc["resource"] == "https://mantle.home.agience.ai"


def test_a_DEFAULT_standalone_node_omits_the_key_too(monkeypatch):
    """The branch that actually ships. `test_..._OMITS_the_key...` above blanks both variables by
    hand, which nobody does; this is `pip install` followed by `mantle-serve`, and until now it
    published `["http://localhost:8080"]` at a machine that reads the document and acts on it."""
    client, _ = _client(monkeypatch, MANTLE_URI="https://mantle.home.agience.ai")
    doc = client.get(PATH).json()
    assert "authorization_servers" not in doc
    assert doc["resource"] == "https://mantle.home.agience.ai"


def test_this_document_IS_the_whole_oauth_surface_of_a_standalone_node(monkeypatch):
    """Said plainly, and measured, because it is the reason the key is omitted rather than filled in
    with something plausible. Mantle serves no authorization-server metadata, no authorization
    endpoint, no token endpoint and no dynamic client registration, so a standards-compliant MCP
    OAuth flow cannot complete against it however the document is written. A static `Authorization`
    header holding a `mantle-token` credential is the supported path."""
    client, _ = _client(monkeypatch, MANTLE_URI="https://mantle.home.agience.ai")
    assert client.get(PATH).status_code == 200
    for absent in ("/.well-known/oauth-authorization-server",
                   "/.well-known/openid-configuration",
                   "/authorize", "/token", "/register"):
        assert client.get(absent).status_code == 404, (
            f"{absent} answered — this node now serves part of an OAuth flow, and the metadata "
            f"and the README both say it does not")


def test_the_resource_falls_back_to_the_request_when_MANTLE_URI_is_unset(monkeypatch):
    """Not every deployment sets `MANTLE_URI`, so the endpoint derives the resource identifier from
    the request when it is unset rather than having nothing to describe itself with."""
    client, _ = _client(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai")
    doc = client.get(PATH).json()
    assert doc["resource"].startswith("http")
    assert not doc["resource"].endswith("/")


# ---------------------------------------------------------------------------
# The challenge
# ---------------------------------------------------------------------------
def test_an_unauthenticated_mcp_post_challenges_and_points_at_the_metadata(monkeypatch):
    """The whole point: a 401 that tells the client where to go."""
    client, _ = _client(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai",
                        MANTLE_URI="https://mantle.home.agience.ai")
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    headers={"accept": "application/json, text/event-stream"})
    assert r.status_code == 401
    challenge = r.headers.get("www-authenticate", "")
    assert challenge.startswith("Bearer ")
    assert f'resource_metadata="https://mantle.home.agience.ai{PATH}"' in challenge


def test_the_challenge_is_attached_to_EVERY_401_not_one_raise_site(monkeypatch):
    """`services/dependencies.py` alone raises 401 from thirteen places. A resource that is
    discoverable down some paths and not others is worse than one that is not discoverable at all:
    the first client to take the wrong path concludes the server is broken.

    The 401 is ASSERTED, not tested for with an `if`. Under `if r.status_code == 401:` the header
    check evaporates the moment an unauthenticated response stops being a 401 — a route moved
    behind a 403, or a redirect — and the test keeps reporting that every 401 carries a challenge
    while checking no 401 at all.
    """
    client, _ = _client(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai",
                        MANTLE_URI="https://mantle.home.agience.ai")
    for path in ("/artifacts/visible", "/mcp"):
        r = client.get(path) if path == "/artifacts/visible" else client.post(
            path, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            headers={"accept": "application/json, text/event-stream"})
        assert r.status_code == 401, (
            f"{path} answered {r.status_code} to an unauthenticated request, not 401 — either the "
            f"route stopped requiring auth, or it now refuses in a way this test no longer covers")
        assert "www-authenticate" in {k.lower() for k in r.headers}, f"{path} 401 with no challenge"


def test_a_bare_framework_Bearer_is_UPGRADED_not_treated_as_already_set(monkeypatch):
    """FastAPI's `HTTPBearer` raises 401 with a bare `WWW-Authenticate: Bearer`. A guard that
    skipped whenever the header merely existed would suppress the pointer on exactly the routes
    using the security scheme — `/mcp` among them — while hand-raised 401s still got it. The
    challenge is upgraded whether it originates in the framework or in mantle's own code.
    """
    client, _ = _client(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai",
                        MANTLE_URI="https://mantle.home.agience.ai")
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    headers={"accept": "application/json, text/event-stream"})
    assert r.status_code == 401
    assert r.headers["www-authenticate"] != "Bearer", "the bare framework challenge survived"
    assert "resource_metadata=" in r.headers["www-authenticate"]


def test_existing_challenge_parameters_are_preserved_beside_the_pointer(monkeypatch):
    """A raise site that said `error="invalid_token"` meant it; the pointer joins it, not replaces it."""
    _reload(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai",
            MANTLE_URI="https://mantle.home.agience.ai")
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    import mantle.main as main
    importlib.reload(main)

    @main.app.get("/_test_challenge")
    def _raise():
        raise HTTPException(status_code=401, detail="nope",
                            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})

    v = TestClient(main.app).get("/_test_challenge").headers["www-authenticate"]
    assert 'error="invalid_token"' in v
    assert "resource_metadata=" in v


def test_a_garbage_token_is_also_challenged(monkeypatch):
    client, _ = _client(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai",
                        MANTLE_URI="https://mantle.home.agience.ai")
    r = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    headers={"authorization": "Bearer not.a.token",
                             "accept": "application/json, text/event-stream"})
    assert r.status_code == 401
    assert "www-authenticate" in {k.lower() for k in r.headers}


def test_a_NON_401_carries_no_challenge(monkeypatch):
    """The control for the handler edit: it must key on the status, not attach to everything."""
    client, _ = _client(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai",
                        MANTLE_URI="https://mantle.home.agience.ai")
    r = client.get("/version")
    assert r.status_code == 200
    assert "www-authenticate" not in {k.lower() for k in r.headers}
    r404 = client.get("/no-such-route-here")
    assert r404.status_code == 404
    assert "www-authenticate" not in {k.lower() for k in r404.headers}


def test_explicit_headers_on_an_HTTPException_are_no_longer_discarded(monkeypatch):
    """The handler rebuilds the response but must preserve `exc.headers` — otherwise a
    `Retry-After`, or a hand-set `WWW-Authenticate`, vanishes while the status code still looks
    correct."""
    _reload(monkeypatch, ORIGIN_URI="https://origin.home.agience.ai")
    from fastapi import HTTPException
    from fastapi.testclient import TestClient
    import mantle.main as main
    importlib.reload(main)

    @main.app.get("/_test_headers")
    def _raise():
        raise HTTPException(status_code=429, detail="slow down", headers={"Retry-After": "42"})

    r = TestClient(main.app).get("/_test_headers")
    assert r.status_code == 429
    assert r.headers.get("retry-after") == "42"
