"""The human door: a browser at `/`, signed in by Authorization Code + PKCE.

The tests below cover three security properties, each verified both by the presence of the current
shape and the absence of the alternative it would otherwise take:
  · `redirect_uri` is fixed to this origin, not caller-supplied — an unvalidated `redirect_uri`
    lets an IdP send a token wherever the caller asks (open redirector).
  · the authorization code is exchanged for a token server-side over PKCE — a token never travels
    in a URL fragment, where it would land in history, in Referer, and be readable by every script
    on the page.
  · the token lives in `sessionStorage`, not `localStorage`, so it does not outlive the tab.

Asserting the absence of each alternative shape matters as much as asserting the presence of the
current one: an absence is what regresses silently.
"""
from __future__ import annotations

import importlib
import json
import re

import pytest


def _page(monkeypatch, **env) -> str:
    for k in ("AGIENCE_TRUSTED_ISSUERS", "ORIGIN_URI", "AUTHORITY_ISSUER",
              "MANTLE_OIDC_CLIENT_ID", "MANTLE_OIDC_SCOPE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AGIENCE_NO_DOTENV", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import mantle.config as cfg
    import mantle.ui.browse_page as bp
    importlib.reload(cfg)
    importlib.reload(bp)
    return bp.render()


ENTRA = json.dumps([{"issuer": "https://login.microsoftonline.com/t/v2.0",
                     "audience": "aud-123", "role": "external"}])


# ── the flow is the standards-track one ─────────────────────────────────────────────────────────

def test_it_uses_authorization_code_with_pkce(monkeypatch):
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert 'response_type: "code"' in p
    assert "code_challenge_method: \"S256\"" in p
    assert "code_verifier" in p
    assert "SHA-256" in p, "the challenge must be the S256 hash, not the plain verifier"


def test_the_verifier_never_travels_with_the_authorization_request(monkeypatch):
    """PKCE's whole guarantee: the redirect carries the HASH; the verifier stays in this tab until
    the token exchange. Sending both would make the code redeemable by whoever intercepts it."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    authorize = p[p.index("async function signIn"):p.index("async function finish")]
    assert "code_challenge" in authorize
    assert "code_verifier" not in authorize, "the verifier leaked into the authorization request"


def test_state_is_verified_before_the_code_is_redeemed(monkeypatch):
    """State is verified before the code is redeemed, so a response belonging to a flow this tab
    did not start cannot be exchanged (CSRF)."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    finish = p[p.index("async function finish"):p.index("async function api")]
    assert finish.index("state mismatch") < finish.index("token_endpoint"), \
        "the code is exchanged before state is checked"


def test_endpoints_come_from_discovery_not_from_a_guess(monkeypatch):
    """A tenant that moves its endpoints keeps working, and no URL is constructed by pattern."""
    p = _page(monkeypatch, AGIENCE_TRUSTED_ISSUERS=ENTRA, MANTLE_OIDC_CLIENT_ID="c1")
    assert "/.well-known/openid-configuration" in p
    assert "d.authorization_endpoint" in p and "d.token_endpoint" in p


# ── insecure token-handling shapes must not appear ──────────────────────────────────────────────

def test_no_implicit_flow_and_no_token_in_a_fragment(monkeypatch):
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert "response_type: \"token\"" not in p and "response_type=token" not in p
    # An implicit-flow page reads `h.get("access_token")` off `location.hash`; this page must not.
    assert not re.search(r"location\.hash", p), "the page still reads the URL fragment"


def test_the_token_is_not_put_in_localstorage(monkeypatch):
    """localStorage outlives the tab and is readable by any script that ever runs on the origin."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert "localStorage" not in p
    assert "sessionStorage" in p


def test_the_redirect_uri_is_this_origin_not_a_caller_supplied_value(monkeypatch):
    """The redirect target is fixed to this page; a caller cannot choose where the token is sent."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1",
              MANTLE_URI="https://mantle.agience.ai")
    assert p.count("redirect_uri") >= 2
    assert "origin.agience.ai/login?redirect_uri=" not in p

    # The redirect_uri is declared server-side from `MANTLE_URI` and injected into BOOT, not
    # computed in the browser from `location.origin` — which would also drop any path prefix, so a
    # mantle served under `https://example.com/mantle/` would compute a URL it does not serve.
    assert '"redirect_uri": "https://mantle.agience.ai/auth/callback"' in p
    assert "location.origin" not in p, "the callback is being derived in the browser again"


def test_an_UNDECLARED_public_url_REFUSES_instead_of_guessing(monkeypatch):
    """No fallback: with no `MANTLE_URI`, the page must say which setting is missing — not invent
    a callback that fails at the IdP after the person has typed a password, which reads as "the
    login is broken" rather than as "this is unset". Same shape as the missing-client refusal.

    `config.py`'s `http://localhost:8081` default is a developer default, not a declaration, and
    is rejected as firmly as an empty value — otherwise a production node hands the IdP a redirect
    to localhost.
    """
    for env in ({}, {"MANTLE_URI": ""}, {"MANTLE_URI": "   "}):
        p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai",
                  MANTLE_OIDC_CLIENT_ID="c1", **env)
        assert '"redirect_uri": ""' in p, f"a callback was invented for {env!r}"
        assert "has not been told its own public URL" in p
        assert "MANTLE_URI" in p


def test_a_DECLARED_public_url_is_honoured_even_at_the_default_VALUE(monkeypatch):
    """The distinction is DECLARED vs DEFAULTED, not the string the value happens to be.

    Matching the value against `http://localhost` refuses the one sentence a node on a laptop can
    truthfully say about itself, so `MANTLE_URI`'s own default was rejected BY VALUE and setting it
    to that same value changed nothing: no configuration opened the page on a local node. This is
    the reading `config.authority_is_declared()` already applies to `ORIGIN_URI`, whose Phase-1
    default is `http://localhost:8080` — saying it counts, arriving at it by silence does not.

    Both halves are asserted together because either one alone passes under the wrong rule: a
    value-matching implementation passes the silent half, and an accept-everything implementation
    passes the declared half.
    """
    silent = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai",
                   MANTLE_OIDC_CLIENT_ID="c1")
    assert '"redirect_uri": ""' in silent, "silence was read as a declaration"

    for declared in ("http://localhost:8081", "http://127.0.0.1:8081", "http://localhost:9999/"):
        p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai",
                  MANTLE_OIDC_CLIENT_ID="c1", MANTLE_URI=declared)
        want = declared.rstrip("/") + "/auth/callback"
        assert '"redirect_uri": "%s"' % want in p, f"{declared!r} was declared and still refused"

    # The refusal TEXT is in every copy of the page — it is the body of a branch, not a rendering
    # decision — so `redirect_uri` above is what says whether the branch is reachable. This asserts
    # the branch is the only thing guarding it, so the check above cannot be passing vacuously.
    assert 'if (!REDIRECT_URI){' in p


def test_the_declaration_is_read_live_so_a_dotenv_still_counts(monkeypatch):
    """`main.py`'s lifespan calls `config.load_env()` AFTER this module has been imported, so a
    `MANTLE_URI` that arrives from `.env` never reaches the module attribute. Reading it at import
    would make the setting work from the shell and silently not from the file that documents it —
    which is why `authority_is_declared()` reads the environment live, and why this does."""
    import mantle.config as cfg
    importlib.reload(cfg)                       # attribute bound with MANTLE_URI absent
    monkeypatch.setenv("MANTLE_URI", "https://mantle.agience.ai")
    assert cfg.declared_public_uri() == "https://mantle.agience.ai"
    assert cfg.MANTLE_URI != "https://mantle.agience.ai", \
        "the attribute updated itself — this test no longer proves the live read"


def test_the_authorization_code_is_removed_from_the_url(monkeypatch):
    """A single-use code left in the address bar gets bookmarked, shared and re-submitted."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert "history.replaceState" in p


# ── it says what is wrong instead of failing silently ───────────────────────────────────────────

def test_without_a_client_id_it_names_the_setting_rather_than_offering_a_broken_button(monkeypatch):
    """A guessed client_id produces a button that always fails at the IdP and looks
    authoritative. The page states which setting is missing and for which issuer."""
    p = _page(monkeypatch, AGIENCE_TRUSTED_ISSUERS=ENTRA)
    assert "MANTLE_OIDC_CLIENT_ID" in p
    import mantle.ui.browse_page as bp
    assert "MANTLE_OIDC_CLIENT_ID" in bp.unconfigured_reason()


def test_a_configured_node_reports_no_reason(monkeypatch):
    _page(monkeypatch, AGIENCE_TRUSTED_ISSUERS=ENTRA, MANTLE_OIDC_CLIENT_ID="c1")
    import mantle.ui.browse_page as bp
    assert bp.unconfigured_reason() == ""


def test_each_node_advertises_its_own_issuer(monkeypatch):
    """Each node advertises the issuer it is actually configured to accept, not a fixed platform
    default — a node running no Origin at all must not point at origin.agience.ai."""
    entra = _page(monkeypatch, AGIENCE_TRUSTED_ISSUERS=ENTRA, MANTLE_OIDC_CLIENT_ID="c1")
    assert "login.microsoftonline.com/t/v2.0" in entra
    platform = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert "origin.agience.ai" in platform


# ── it is one self-contained document ───────────────────────────────────────────────────────────

def test_the_page_makes_no_external_request_to_render(monkeypatch):
    """The one page that must work when everything else is broken cannot depend on a CDN."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert not re.search(r'<(script|link)[^>]+(src|href)="https?://', p), \
        "the page loads an external asset"


def test_the_boot_blob_is_valid_json(monkeypatch):
    """It is injected into a <script>; a broken substitution would be a syntax error at load,
    which renders a blank page and no error anyone can see."""
    p = _page(monkeypatch, AGIENCE_TRUSTED_ISSUERS=ENTRA, MANTLE_OIDC_CLIENT_ID="c1")
    blob = re.search(r"const BOOT = (\{.*?\});", p, re.S).group(1)
    parsed = json.loads(blob)
    assert parsed["client_id"] == "c1" and parsed["external"] is True
    assert "__BOOT__" not in p


def test_it_calls_mantles_own_grant_filtered_api(monkeypatch):
    """Not a separate app's /api/* routes over a different data layer."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert "/artifacts/visible" in p and "/artifacts/recall" in p
    assert "/api/collections" not in p and "/api/artifacts" not in p


# ── a missing session redirects, and does not loop ──────────────────────────────────────────────

def test_a_missing_session_redirects_rather_than_offering_a_button(monkeypatch):
    """A person opening this hostname wants the corpus, not a page about how to ask for it: a
    missing session starts the sign-in flow immediately rather than waiting for a button."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    tail = p[p.index("async function paint"):]
    assert "return signIn()" in tail, "an unauthenticated load does not start the flow itself"
    assert "Signing you in" in tail


def test_the_redirect_is_guarded_against_a_loop(monkeypatch):
    """If the IdP returns without a usable token — consent declined, an `aud` this node rejects, a
    provider that is not configured — redirecting again would bounce the user between two servers
    forever, with a flickering browser and no error.

    The guard is set before leaving and cleared only when a token actually arrives, so the second
    unauthenticated load in a session stops and explains instead."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert "mantle_tried" in p
    paint = p[p.index("async function paint"):]
    assert paint.index('setItem("mantle_tried"') < paint.index("return signIn()"), \
        "the guard is set AFTER the redirect — it would never be recorded"
    assert "did not produce a session" in paint


def test_a_real_session_clears_the_guard(monkeypatch):
    """Otherwise the first token would leave the flag set, and the next expiry in the same session
    would not redirect — a working sign-in that breaks on its second use."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    finish = p[p.index("async function finish"):p.index("async function api")]
    assert 'removeItem("mantle_tried")' in finish


def test_a_node_with_no_issuer_names_the_setting_rather_than_404ing_on_itself(monkeypatch):
    """`config.authorization_servers()` returns nothing on a standalone node BY DESIGN, so an empty
    issuer is an ordinary state and not a broken one. Unguarded, `disco()` derives
    `"" + "/.well-known/openid-configuration"`, fetches THIS node's own root and reports a 404 on a
    relative URL — a message about a path nobody set, on a node whose actual problem is that it
    names no authorization server.

    Reachable only now that a declared localhost `MANTLE_URI` gets past `_public_base()`: the
    redirect_uri refusal used to fire first on every local node and hide this one.
    """
    p = _page(monkeypatch, MANTLE_URI="http://localhost:8081", MANTLE_OIDC_CLIENT_ID="c1")
    assert '"issuer": ""' in p, "the premise moved — this node now names an issuer"
    paint = p[p.index("async function paint"):]
    assert "No authorization server is configured" in paint
    assert "AGIENCE_TRUSTED_ISSUERS" in paint and "AUTHORITY_ISSUER" in paint
    assert paint.index("No authorization server is configured") < paint.index("return signIn()"), \
        "the flow starts before the issuer is checked — it would 404 on this node's own root"


def test_the_page_refuses_in_the_order_unconfigured_reason_reports(monkeypatch):
    """Two surfaces answer "why can no human use this node" — the page a person sees and the
    `/status` payload a machine reads. Refusing in different orders makes them name different
    settings on a node missing both, and only one of the two is ever looked at."""
    p = _page(monkeypatch, MANTLE_URI="http://localhost:8081")
    import mantle.ui.browse_page as bp
    assert "no issuer is configured" in bp.unconfigured_reason(), \
        "unconfigured_reason() reports the client first — the page reports the issuer first"
    paint = p[p.index("async function paint"):]
    assert paint.index("No authorization server is configured") < paint.index("MANTLE_OIDC_CLIENT_ID")


def test_an_unconfigured_node_does_not_redirect_anywhere(monkeypatch):
    """No client means no flow to start. Redirecting would send someone to an IdP that cannot
    possibly complete, which is worse than saying what is missing."""
    p = _page(monkeypatch, AGIENCE_TRUSTED_ISSUERS=ENTRA)
    paint = p[p.index("async function paint"):]
    assert paint.index("MANTLE_OIDC_CLIENT_ID") < paint.index("return signIn()")


# ── the callback path ───────────────────────────────────────────────────────────────────────────

def test_the_redirect_uri_is_a_dedicated_callback_path(monkeypatch):
    """`/auth/callback` matches Origin's own provider callbacks (`auth_router`, prefix `/auth`,
    route `/callback`) — this system's own convention. A named path keeps the auth handler off the
    busiest route, keeps a single-use code out of the address bar of the page people bookmark, and
    grants one path as a redirect target instead of the origin root.
    """
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1",
              MANTLE_URI="https://mantle.agience.ai")
    assert "/auth/callback" in p
    assert 'const REDIRECT_URI = BOOT.redirect_uri;' in p, "the page invents the callback again"


def test_both_legs_send_the_same_redirect_uri(monkeypatch):
    """RFC 6749 requires the token request to match the authorization request, and Entra and
    Origin both enforce it. Building the string twice is how a trailing slash goes missing in one
    place, and the failure then lands at the token exchange, after the user has already logged in."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert p.count("redirect_uri: REDIRECT_URI") == 2
    assert 'redirect_uri: location.origin + "/"' not in p, "a leg still builds the URI inline"


def test_the_callback_lands_the_user_back_on_the_root(monkeypatch):
    """`/auth/callback` is a leg of a flow, not a page to sit on — and the code must not survive
    in history."""
    p = _page(monkeypatch, ORIGIN_URI="https://origin.agience.ai", MANTLE_OIDC_CLIENT_ID="c1")
    assert 'history.replaceState(null, "", "/")' in p


def test_the_unconfigured_message_names_the_callback_not_the_root(monkeypatch):
    """The message tells an operator what to register. Naming the wrong URI sends them to configure
    something that will then be rejected as a redirect mismatch."""
    p = _page(monkeypatch, AGIENCE_TRUSTED_ISSUERS=ENTRA)
    assert "esc(REDIRECT_URI)" in p
