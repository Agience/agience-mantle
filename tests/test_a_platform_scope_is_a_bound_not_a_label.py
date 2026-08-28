"""A token that carries a `scope` is held to it.

Origin issues system delegations under a named purpose and stamps the matching scope, and its
purpose table says why:

    "platform-mail": {"scope": "platform.email.send", "audience": "mantle"}
    "platform-stripe": {"scope": "platform.stripe.resolve", "audience": "mantle"}
    # "Scoped to one capability, like every entry here, so it cannot be widened into a
    # general secret-reader."

This closes a real gap. Measured 2026-08-26: `payload.get("scope")` appeared nowhere in
`agience-mantle/src`. The delegation branch of `get_auth` returns `principal_type="user"`, so a
scoped platform token became an ordinary user principal carrying `platform-system`'s full grants —
a `platform-mail` token authenticated identically to a `platform-stripe` one, and both reached
whatever that principal reached.

`aud` and `act.sub` sit inside the same Core-signed token, so a stolen delegation carries both
unchanged — a presenter-replay check on those fields cannot distinguish platform automation from a
genuine user delegation. What actually constrains a platform-scoped token is its shape: the only
tokens whose `aud` and `act.sub` differ are system delegations, by construction, and the rule this
enforces is that platform automation may not read a credential; only a user-delegated call may.
Enforcing the scope Mantle is already sent does this at the layer that authorizes, covers both
purposes, and does not care which service presents the token.

Safe to make strict: `grep -c "system-delegation" origin.log` is 0 on
71/home — the exchange has never run, so there is no live traffic to break. Both allowed
destinations were derived from the only caller (`chorus/src/ophan/server.py:269` and `:351`), not
guessed from the scope names.

The narrow prefix is load-bearing. Ordinary OIDC tokens carry `scope` claims full of things
Mantle has never interpreted (`openid`, `profile`, an IdP's own vocabulary). Governing only
`platform.*` is what makes this enforceable today instead of a migration, and
`test_an_ordinary_oidc_scope_is_not_governed` is that boundary asserted.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from mantle.services.dependencies import AuthContext, _enforce_token_scope, _scope_allows


def _req(method: str, path: str):
    """The two attributes `_enforce_token_scope` reads, and nothing else."""
    return SimpleNamespace(method=method, url=SimpleNamespace(path=path))


def _auth(scope):
    return AuthContext(principal_id="platform-system", principal_type="user",
                       user_id="platform-system", actor="ophan", scope=scope)


# ── the destinations each scope legitimately needs ───────────────────────────────────────────────

def test_the_mail_scope_may_invoke_an_operator():
    """`ophan/server.py:269` — POST {MANTLE}/artifacts/{op_id}/op/invoke."""
    _enforce_token_scope(_auth("platform.email.send"),
                         _req("POST", "/artifacts/8f1c-op-email/op/invoke"))


def test_the_stripe_scope_may_reach_the_secret_route():
    """`ophan/server.py:351` — POST {MANTLE}/secrets/reveal. That route does not currently exist;
    this table says what a scope MAY reach, which is a different question from whether it resolves."""
    _enforce_token_scope(_auth("platform.stripe.resolve"), _req("POST", "/secrets/reveal"))


# ── the bound itself: each scope refused on the OTHER one's destination ──────────────────────────

def test_the_mail_scope_may_not_read_a_secret():
    """Asserted directly: this is exactly the call that must be refused — platform automation
    reading a credential."""
    with pytest.raises(HTTPException) as e:
        _enforce_token_scope(_auth("platform.email.send"), _req("POST", "/secrets/reveal"))
    assert e.value.status_code == 403, e.value.status_code


def test_the_stripe_scope_may_not_invoke_an_operator():
    with pytest.raises(HTTPException) as e:
        _enforce_token_scope(_auth("platform.stripe.resolve"),
                             _req("POST", "/artifacts/8f1c-op-email/op/invoke"))
    assert e.value.status_code == 403


@pytest.mark.parametrize("method,path", [
    ("GET", "/artifacts/anything"),
    ("DELETE", "/artifacts/anything"),
    ("POST", "/artifacts/batch"),
    ("GET", "/grants"),
    ("POST", "/mcp"),
    ("GET", "/system"),
])
def test_a_platform_scope_reaches_nothing_else(method, path):
    """The general form. Before this, a `platform-mail` token reached all 66 mounted routes as an
    ordinary user; now it reaches one shape."""
    with pytest.raises(HTTPException):
        _enforce_token_scope(_auth("platform.email.send"), _req(method, path))


# ── the fail-closed defaults ─────────────────────────────────────────────────────────────────────

def test_an_unknown_platform_scope_is_refused_everywhere():
    """Default-deny: adding a purpose to Origin's table without adding it here must fail closed
    and loudly. Treating an unrecognised bound as "unrestricted" is how the claim became decorative
    in the first place."""
    for path in ("/artifacts/x/op/invoke", "/secrets/reveal", "/system"):
        with pytest.raises(HTTPException):
            _enforce_token_scope(_auth("platform.something.new"), _req("POST", path))


def test_a_scoped_token_with_no_request_is_refused():
    """A scoped token whose target cannot be determined is exactly the case this exists to refuse."""
    with pytest.raises(HTTPException):
        _enforce_token_scope(_auth("platform.email.send"), None)


# ── the boundary: what is NOT governed ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("scope", ["openid profile email", "profile", "read:user", ""])
def test_an_ordinary_oidc_scope_is_not_governed(scope):
    """Governing every `scope` claim would refuse every external-IdP login on arrival. Only the
    `platform.` prefix — the vocabulary Origin's purpose table actually mints — is enforced."""
    _enforce_token_scope(_auth(scope), _req("GET", "/artifacts/anything"))


def test_a_token_with_no_scope_is_untouched():
    """The overwhelming majority of tokens. A user delegation carries no `scope`."""
    _enforce_token_scope(_auth(None), _req("DELETE", "/artifacts/anything"))


# ── the claim is actually carried off the token ──────────────────────────────────────────────────

def test_the_delegation_branch_reads_the_scope_claim():
    """The half that was missing: the claim is always present on the token; nothing
    read it. Asserted at the source so a refactor cannot drop the one line and leave every test
    above passing against a scope that is never populated."""
    import inspect

    from mantle.services import dependencies

    src = inspect.getsource(dependencies.resolve_auth)
    assert 'payload.get("scope")' in src, (
        "the delegation branch no longer reads the `scope` claim, so AuthContext.scope is always "
        "None and every scope check above silently passes")


def test_enforcement_precedes_the_acting_principal():
    """Order matters: `set_acting_principal` is what lets the key oracle issue material to this
    caller, so a token refused for scope must not first be installed as the identity that may ask
    for keys — the whole point of the bound is that platform automation does not read credentials."""
    import inspect

    from mantle.services import dependencies

    src = inspect.getsource(dependencies.get_auth)
    assert "_enforce_token_scope" in src, "get_auth no longer enforces token scope at all"
    assert src.index("_enforce_token_scope") < src.index("set_acting_principal("), (
        "the scope check runs AFTER the acting principal is published, so a refused token has "
        "already been granted key-custody standing")


def test_the_allowlist_is_a_function_of_scope_and_not_a_wildcard():
    """A guard against the check being softened into a no-op: at least one real path must be
    refused for each known scope, or `_scope_allows` has become `return True`."""
    assert _scope_allows("platform.email.send", "POST", "/artifacts/x/op/invoke")
    assert not _scope_allows("platform.email.send", "POST", "/secrets/reveal")
    assert _scope_allows("platform.stripe.resolve", "POST", "/secrets/reveal")
    assert not _scope_allows("platform.stripe.resolve", "GET", "/secrets/reveal")
