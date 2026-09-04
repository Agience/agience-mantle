"""The human door — an artifact browser that signs you in and brings you back.

One document, no build step, no external request. CSS and JS are inline because the one page that
must render when everything else is broken cannot depend on a second request succeeding. It is a
Python constant rather than a package data file so nothing has to be declared in `pyproject`, added
to the wheel, or copied by a packaging step to keep working.

Authorization code + PKCE (RFC 7636). A login flow that instead does
`location.replace("https://issuer/login?redirect_uri=" + location.origin + "/")` and then reads
`#access_token=…` out of the URL fragment into `localStorage` carries three problems, none cosmetic:

  1. **An unvalidated `redirect_uri` is an open redirector.** The IdP was asked to send a token to
     whatever URL the caller named. That is the vulnerability class the redirect-URI registration
     requirement exists to close.
  2. **A token in a URL fragment leaks.** It lands in browser history, in `Referer` on any
     subsequent navigation, and is readable by every script on the page. The implicit flow was
     deprecated for exactly this (OAuth 2.0 Security Best Current Practice).
  3. **`localStorage` survives the tab.** A token there outlives the session and is readable by any
     script that ever runs on the origin. `sessionStorage` is narrower and is what this uses.

The code flow avoids all three: the IdP redirects to a registered URI, the authorization code in the
query string is useless without the verifier, and the verifier never leaves this tab.

It requires a registered public client, which is the property that makes the flow safe rather than a
limitation to engineer around. Without `MANTLE_OIDC_CLIENT_ID` the page renders and says which
setting is missing, rather than showing a button that fails at the IdP.
"""
from __future__ import annotations

import html as _html
import json

from mantle import config

#: Everything the page needs to know about this node, computed server-side and injected as one JSON
#: blob. The browser never guesses an endpoint: `authorization_endpoint` and `token_endpoint` come
#: from the issuer's own discovery document at runtime, so a tenant that moves them keeps working.
def _public_base() -> str:
    """This node's declared public base URL, or "" when it has not been told.

    `config.py`'s default is `http://localhost:8081`, a developer default rather than a statement
    about where this node answers. Treating it as declared would hand the IdP a redirect to
    localhost from a production host, so it is rejected here as firmly as an empty value.

    The test is whether it was declared, not what it says — `config.declared_public_uri()`, the
    same distinction `config.authority_is_declared()` draws for the authority and deliberately not
    a second reading of it. Matching the VALUE against `http://localhost` instead refuses the one
    sentence a node on a laptop can truthfully say about itself: `MANTLE_URI=http://localhost:8081`
    set on purpose is a statement, and there was no way to make it, so the page short-circuited on
    every local node and no setting could open it.
    """
    return config.declared_public_uri()


def _boot() -> dict:
    external = [i for i in (config.TRUSTED_ISSUERS or []) if i.get("issuer")]
    # The same list the protected-resource metadata publishes, and deliberately not a second
    # reading of it: the sign-in button and the RFC 9728 document must name the same issuer, or a
    # human and an MCP client are sent to different authorization servers and only the human's
    # failure is ever noticed. `config.authorization_servers()` is that one definition.
    _servers = config.authorization_servers()
    issuer = _servers[0] if _servers else ""
    return {
        "issuer": issuer or "",
        # An issuer URL is not always a discovery base, so a node may pin the document — the same
        # permission it already has for `jwks_uri`, and for the same reason: the derived path is a
        # guess about another system's URL layout, correct for Entra and wrong for B2C. Empty means
        # "derive it", which is what every issuer verified so far needs.
        "discovery": (external[0].get("discovery_uri") if external else "") or "",
        "client_id": config.OIDC_CLIENT_ID,
        # The callback is declared by the node, not derived by the browser. `location.origin` is
        # scheme://host:port and drops any path prefix, so a mantle served under
        # `https://example.com/mantle/` cannot reconstruct `https://example.com/auth/callback` from
        # the browser alone. RFC 6749 also requires the token exchange to send a redirect_uri
        # byte-identical to the authorization request, which a browser-invented value cannot be
        # checked against.
        #
        # `MANTLE_URI` is the node's own declaration of where it answers; the compose requires it
        # with `:?`. One source, server-side.
        #
        # There is no fallback: an empty value here makes the page refuse and name the setting,
        # exactly as a missing `MANTLE_OIDC_CLIENT_ID` already does — rather than guessing a URL that
        # will fail at the IdP after a successful login, which reads as "the login is broken" instead
        # of "this is unset".
        "redirect_uri": (_public_base() + "/auth/callback") if _public_base() else "",
        "scope": config.OIDC_SCOPE,
        # The audience this node verifies. Surfaced so the page can explain an `aud` mismatch —
        # the single most common misconfiguration, and one that otherwise reads as "login broken".
        "audience": (external[0].get("audience") if external else "") or "",
        "external": bool(external),
    }


_PAGE = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agience · Mantle</title>
<style>
:root{color-scheme:light dark;--bg:#0b0d12;--panel:#151924;--line:#222839;--txt:#e8eaf0;
--mut:#9aa3b8;--acc:#6ea8fe}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}
header{display:flex;gap:1rem;align-items:center;padding:.75rem 1rem;border-bottom:1px solid var(--line)}
header h1{font-size:1rem;margin:0;font-weight:600}
header .sp{flex:1}
#who{color:var(--mut);font-size:.85rem}
button{background:var(--panel);color:var(--txt);border:1px solid var(--line);border-radius:6px;
padding:.4rem .8rem;cursor:pointer;font:inherit}
button:hover{border-color:var(--acc)}
main{display:grid;grid-template-columns:260px 1fr;height:calc(100vh - 57px)}
#side{border-right:1px solid var(--line);overflow:auto;padding:.5rem}
#body{overflow:auto;padding:1rem}
#q{width:100%;padding:.5rem .7rem;background:var(--panel);color:var(--txt);
border:1px solid var(--line);border-radius:6px;font:inherit}
.row{padding:.5rem .6rem;border:1px solid var(--line);border-radius:6px;margin-bottom:.4rem;
background:var(--panel);cursor:pointer}
.row:hover{border-color:var(--acc)}
.row .id{font-family:ui-monospace,monospace;font-size:.78rem;color:var(--mut)}
.row .ct{font-size:.75rem;color:var(--acc)}
.mut{color:var(--mut)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:1rem;
margin:0 auto;max-width:46rem}
pre{white-space:pre-wrap;word-break:break-word;background:#0e1119;border:1px solid var(--line);
border-radius:6px;padding:.75rem;overflow:auto;max-height:60vh}
a{color:var(--acc)}
#iface{font-size:.78rem;margin-top:1rem;border-top:1px solid var(--line);padding-top:.75rem}
#iface p{margin:.35rem 0}
code{font-family:ui-monospace,monospace;font-size:.85em}
</style></head><body>
<header>
  <h1>Agience · Mantle</h1><span class="mut" id="site"></span>
  <span class="sp"></span><span id="who"></span>
  <button id="auth" hidden>Sign in</button>
</header>
<main>
  <div id="side">
    <input id="q" placeholder="Search artifacts…" autocomplete="off">
    <div id="cols"></div>
    <!-- The machine door stays visible on the human one: someone who opens this hostname to point
         an MCP client at it should not have to read the source or guess the path.
         `tests/test_mcp_router.py::test_root_serves_html_to_a_browser` asserts `/mcp` appears
         here. -->
    <div id="iface" class="mut">
      <p><code>/mcp</code> — Model Context Protocol over Streamable HTTP. Point any MCP client here.</p>
      <p><code>/artifacts</code> — the REST API. Every read is filtered by your grants.</p>
      <p><code>/status</code> — store health.</p>
    </div>
  </div>
  <div id="body"></div>
</main>
<script>
const BOOT = __BOOT__;

/* One definition, used by both legs. The authorization request and the token exchange must send a
   byte-identical redirect_uri — RFC 6749 requires the second to match the first, and Entra and
   Origin both enforce it. Building the string twice is how a trailing slash goes missing in one
   place and the exchange fails with "redirect_uri mismatch" long after the user has logged in.

   The path is `/auth/callback`, matching Origin's own provider callbacks (auth_router, prefix
   `/auth`). A dedicated path keeps the auth handler off the busiest route, keeps a single-use code
   out of the address bar of the page people bookmark, and grants only that path as a redirect
   target rather than the whole origin root. */
const REDIRECT_URI = BOOT.redirect_uri;
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
$("#site").textContent = location.host;

/* ── PKCE (RFC 7636) ──────────────────────────────────────────────────────────────────────────
   The verifier is generated here, kept in sessionStorage, and sent to the IdP only at the token
   exchange. What travels in the redirect is its S256 hash alone, so an intercepted authorization
   code cannot be redeemed by anyone who did not start this flow. */
const b64u = b => btoa(String.fromCharCode(...new Uint8Array(b))).replace(/\\+/g,"-").replace(/\\//g,"_").replace(/=+$/,"");
const rand = n => b64u(crypto.getRandomValues(new Uint8Array(n)));
async function challenge(v){ return b64u(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(v))); }

let DISCO = null;
async function disco(){
  if (DISCO) return DISCO;
  /* The derived path is a guess about another system's URL layout. Azure B2C publishes its document
     at `<host>/<tenant>/v2.0/.well-known/openid-configuration?p=<policy>` and answers 404 without
     the policy, and token verification does not reveal that because `jwks_uri` is pinned — the
     failure lands here, in the browser, on a node whose tokens verify perfectly. A node may
     therefore pin the document with `discovery_uri`; deriving is the default. */
  const url = BOOT.discovery || BOOT.issuer.replace(/\\/+$/,"") + "/.well-known/openid-configuration";
  const r = await fetch(url);
  if (!r.ok) throw new Error("no discovery document at " + url + " (" + r.status + ")");
  return DISCO = await r.json();
}

const tok = () => sessionStorage.getItem("mantle_at");
function authed(){
  const t = tok(); if (!t) return false;
  try { const c = JSON.parse(atob(t.split(".")[1].replace(/-/g,"+").replace(/_/g,"/")));
        return !c.exp || c.exp*1000 > Date.now(); } catch(e){ return false; }
}

async function signIn(){
  const d = await disco();
  const verifier = rand(64), state = rand(16), nonce = rand(16);
  sessionStorage.setItem("mantle_v", verifier);
  sessionStorage.setItem("mantle_s", state);
  /* Each parameter is set individually, leaving the endpoint's own query intact. Assigning
     `u.search` would discard whatever the discovery document already put there, and B2C's
     `authorization_endpoint` carries `?p=<policy>` — the one parameter that selects the user flow.
     Dropping it fails at the IdP after the redirect, where it presents as a broken login rather
     than as a lost query string. */
  const u = new URL(d.authorization_endpoint);
  const q = {
    response_type: "code", client_id: BOOT.client_id,
    redirect_uri: REDIRECT_URI, scope: BOOT.scope,
    state, nonce, code_challenge: await challenge(verifier), code_challenge_method: "S256"
  };
  for (const k in q) u.searchParams.set(k, q[k]);
  location.assign(u);
}

/* The IdP comes back to this page with ?code=&state=. State is compared before anything else:
   a mismatch means this response belongs to a flow this tab did not start (CSRF), and the only
   safe action is to discard it. */
async function finish(){
  const p = new URLSearchParams(location.search);
  if (p.get("error")) throw new Error(p.get("error_description") || p.get("error"));
  const code = p.get("code"); if (!code) return false;
  if (p.get("state") !== sessionStorage.getItem("mantle_s")) throw new Error("state mismatch — this response did not come from a sign-in started in this tab");
  const d = await disco();
  const r = await fetch(d.token_endpoint, {
    method: "POST", headers: {"content-type":"application/x-www-form-urlencoded"},
    body: new URLSearchParams({ grant_type:"authorization_code", code,
      redirect_uri: REDIRECT_URI, client_id: BOOT.client_id,
      code_verifier: sessionStorage.getItem("mantle_v") })
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error_description || j.error || "token exchange failed");
  sessionStorage.setItem("mantle_at", j.access_token);
  sessionStorage.removeItem("mantle_tried");   /* a session arrived: the guard has done its job */
  sessionStorage.removeItem("mantle_v"); sessionStorage.removeItem("mantle_s");
  /* Back to the root: the code is single-use and must not be bookmarked, and /auth/callback is
     a leg of a flow rather than a page anyone should end up sitting on. */
  history.replaceState(null, "", "/");
  return true;
}

async function api(path, opts){
  const o = Object.assign({headers:{}}, opts||{});
  o.headers["Authorization"] = "Bearer " + tok();
  const r = await fetch(path, o);
  if (r.status === 401){ sessionStorage.removeItem("mantle_at"); paint(); throw new Error("401"); }
  if (!r.ok) throw new Error(path + " → " + r.status);
  return r.json();
}

function rows(list){
  if (!list.length) return '<p class="mut">Nothing here. An empty result means "nothing you may see", not "nothing exists".</p>';
  return list.map(a => `<div class="row" data-id="${esc(a.id)}">
    <div>${esc(a.name || a.id)}</div>
    <div class="id">${esc(a.id)}</div>
    <div class="ct">${esc(a.content_type || "")}</div></div>`).join("");
}

async function load(){
  const d = await api("/artifacts/visible");
  $("#body").innerHTML = rows(d.artifacts || d.items || d || []);
}
async function detail(id){
  const a = await api("/artifacts/" + encodeURIComponent(id));
  $("#body").innerHTML = `<div class="card"><h2>${esc(a.name||a.id)}</h2>
    <p class="id mut">${esc(a.id)}</p><p class="ct">${esc(a.content_type||"")}</p>
    <pre>${esc(typeof a.content === "string" ? a.content : JSON.stringify(a, null, 2))}</pre>
    <p><button onclick="load()">← back</button></p></div>`;
}
$("#body").addEventListener("click", e => {
  const r = e.target.closest(".row"); if (r) detail(r.dataset.id);
});
let t; $("#q").oninput = e => {
  clearTimeout(t); const q = e.target.value.trim();
  t = setTimeout(async () => {
    if (q.length < 2) return load();
    const d = await api("/artifacts/recall", {method:"POST",
      headers:{"content-type":"application/json"}, body: JSON.stringify({query_text: q})});
    $("#body").innerHTML = rows(d.artifacts || d.results || []);
  }, 250);
};

function problem(msg, detail){
  $("#body").innerHTML = `<div class="card"><h2>${esc(msg)}</h2><p class="mut">${detail}</p></div>`;
}

async function paint(){
  if (!REDIRECT_URI){
    /* Same shape as the missing-client refusal below: name the setting, do not offer a button that
       cannot work. A sign-in started with no redirect_uri fails at the IdP, after the person has
       already typed a password. */
    return problem("This node has not been told its own public URL",
      'Sign-in needs a callback address, and it is derived from <code>MANTLE_URI</code>, which is ' +
      'unset. Set it to the URL this node answers on — <code>https://mantle.agience.ai</code>, or ' +
      '<code>http://localhost:8081</code> if that is genuinely where it answers. Setting it is ' +
      'the statement, whatever the value; leaving it to default is not.');
  }
  if (!BOOT.issuer){
    /* Third refusal, in the order `unconfigured_reason()` reports them — issuer, then client.
       Without this the flow reaches `disco()`, which derives `"" + "/.well-known/openid-
       configuration"`, fetches THIS node's own root, and shows a 404 on a relative URL. That is a
       message about a path nobody set, on a node whose actual problem is that it names no
       authorization server. `config.authorization_servers()` returns nothing here BY DESIGN — a
       standalone node advertising an issuer that is not there is worse than one advertising none —
       so this state is ordinary, not broken, and the page says which of the two settings ends it. */
    $("#auth").hidden = true;
    return problem("No authorization server is configured for this node",
      'Sign-in needs an issuer to send you to, and this node names none. Set ' +
      '<code>AGIENCE_TRUSTED_ISSUERS</code> to an external IdP, or <code>AUTHORITY_ISSUER</code> / ' +
      '<code>ORIGIN_URI</code> to a platform authority. The API still serves any caller holding a ' +
      'valid token, and <code>mantle-token</code> mints one this node accepts — the sign-in flow ' +
      'is the only part that needs an issuer.');
  }
  if (!BOOT.client_id){
    $("#auth").hidden = true;
    return problem("No browser client is configured for this node",
      'Set <code>MANTLE_OIDC_CLIENT_ID</code> to a public OAuth client registered with ' +
      '<code>' + esc(BOOT.issuer) + '</code>, whose redirect URI is <code>' + esc(REDIRECT_URI) + '</code>. ' +
      'The API still serves any caller holding a valid token — only this page needs the client.');
  }
  if (authed()){
    $("#auth").hidden = false; $("#auth").textContent = "Sign out";
    $("#auth").onclick = () => { sessionStorage.removeItem("mantle_at"); paint(); };
    $("#who").textContent = "signed in";
    return load();
  }
  /* ── No session: go and get one ───────────────────────────────────────────────────────────────
     A person opening this hostname wants the corpus, not a page about how to ask for it, so an
     unauthenticated load starts the flow itself rather than offering a button.

     The attempt is guarded, because an auto-redirect can loop. If the IdP sends us back without a
     usable token — consent declined, an `aud` this node does not accept, a provider that is not
     configured — redirecting again bounces the browser between two servers with no error shown.
     `mantle_tried` is set before leaving and cleared only when a token actually arrives, so the
     second unauthenticated load in one session stops and shows what happened. One automatic
     attempt, then a human decides. */
  $("#who").textContent = "";
  const tried = sessionStorage.getItem("mantle_tried");
  $("#auth").hidden = false; $("#auth").textContent = "Sign in";
  $("#auth").onclick = () => { sessionStorage.removeItem("mantle_tried");
                               signIn().catch(e => problem("Sign-in could not start", esc(e.message))); };
  if (!tried){
    sessionStorage.setItem("mantle_tried", "1");
    problem("Signing you in…", "Redirecting to <code>" + esc(BOOT.issuer) + "</code>.");
    return signIn().catch(e => problem("Sign-in could not start", esc(e.message)));
  }
  problem("Sign-in did not produce a session",
    "The redirect to <code>" + esc(BOOT.issuer) + "</code> came back without a usable token, so " +
    "this page stopped rather than bouncing you between two servers. Use Sign in to try again.");
}

(async () => {
  try { if (await finish()) { /* fall through to paint, now authed */ } }
  catch (e) { return problem("Sign-in did not complete", esc(e.message)); }
  paint();
})();
</script></body></html>"""


def render() -> str:
    """The page, with this node's configuration injected. No template engine, one substitution."""
    return _PAGE.replace("__BOOT__", json.dumps(_boot()))


def unconfigured_reason() -> str:
    """Why the human door is unavailable, or "" when it is available.

    Separate from the page so the status payload can carry the same fact. A machine asking this
    node whether a human can use it should not have to scrape HTML for the answer.
    """
    b = _boot()
    if not b["issuer"]:
        return "no issuer is configured, so no token can be verified"
    if not b["client_id"]:
        return "MANTLE_OIDC_CLIENT_ID is unset — no public OAuth client is registered for the browser"
    return ""
