"""`mantle-token` — the credential a standalone node had no way to produce.

`mantle-init-keys` already writes `mantle.private.pem` AND an `authority.manifest.json` whose
`trust_anchors.mantle` is the matching public half, so a node trusts its own key from the first
boot. Nothing signed a *user* token against it: `services/peer_signing` signs one service JWT and
must keep signing only that. An installed, booted, standalone node therefore answered 401 to
everything, and that was the entire gap between `pip install` and a working MCP round trip.

The substance here is the round trip and the negative controls. A test that only checked the token's
SHAPE would pass on a token this node rejects; a test that only checked acceptance would pass on a
verifier that accepts anything. So: the node's own verifier accepts what the command mints, refuses
the same token with one byte changed, and the subject is asserted to be the SAME across runs —
because a fresh subject per run is the failure that looks like success, minting a new principal with
no grants and stranding everything the previous token stored.
"""
from __future__ import annotations

import importlib
import json
import uuid

import pytest

from mantle.scripts import dev_init_keys, dev_mint_token


@pytest.fixture
def keyset(tmp_path, monkeypatch):
    """A real keyset, written by the real command, with KEYS_DIR pointed at it.

    `dev_init_keys.main` rather than a hand-built directory: the pair this command depends on — a
    private key and a manifest anchor holding its public half — is exactly what that command
    guarantees, and a fixture that fabricated its own pair could not fail when the guarantee did.
    """
    d = tmp_path / "keys"
    assert dev_init_keys.main(["--keys-dir", str(d)]) == 0
    monkeypatch.setenv("AGIENCE_NO_DOTENV", "1")
    monkeypatch.setenv("KEYS_DIR", str(d))
    for k in ("AUTHORITY_ISSUER", "ORIGIN_URI", "AGIENCE_TRUSTED_ISSUERS"):
        monkeypatch.delenv(k, raising=False)
    yield d


@pytest.fixture
def verifier(keyset):
    """The node's own verifier, reloaded against `keyset`, and put back afterwards.

    The singleton is process-wide and the suite's session fixture points it at a different KEYS_DIR,
    so leaving it built against a `tmp_path` that pytest then deletes would break every later test
    that verifies a token.
    """
    from mantle.services.oidc import reset_oidc_verifier

    reset_oidc_verifier()
    yield
    reset_oidc_verifier()


@pytest.fixture(autouse=True)
def restore_config():
    """`dev_mint_token.audience()` reloads `mantle.config`; put it back on the real environment."""
    yield
    import mantle.config as cfg
    importlib.reload(cfg)


# ---------------------------------------------------------------------------
# The subject: stable for a keyset, distinct across keysets
# ---------------------------------------------------------------------------

def test_the_same_keyset_always_mints_for_the_same_person(keyset):
    """THE reason the subject is derived rather than random. `sub` IS the user id — `resolve_auth`
    returns `user_id = payload["sub"]` — and every grant and owner record hangs off that string. A
    random subject per run mints a new principal holding no grants, so yesterday's artifacts become
    unreachable while every request still answers 200."""
    assert dev_mint_token.subject_id(keyset) == dev_mint_token.subject_id(keyset)


def test_a_different_keyset_is_a_different_person(tmp_path, keyset):
    """`instance.uuid` is the namespace and is regenerated per keyset, so two nodes cannot name the
    same principal by accident, and `mantle-init-keys --force` is a new instance."""
    other = tmp_path / "other-keys"
    assert dev_init_keys.main(["--keys-dir", str(other)]) == 0
    assert dev_mint_token.subject_id(other) != dev_mint_token.subject_id(keyset)


def test_the_subject_is_derived_from_instance_uuid_and_nothing_else(keyset):
    """Named explicitly so the derivation cannot be changed without this failing: it is a uuid5 over
    the keyset's own namespace, computed and never stored."""
    ns = uuid.UUID((keyset / "instance.uuid").read_text(encoding="utf-8").strip())
    assert dev_mint_token.subject_id(keyset) == str(uuid.uuid5(ns, "mantle/local-user"))


def test_a_label_names_a_second_identity_on_one_keyset(keyset):
    assert dev_mint_token.subject_id(keyset, "bot") != dev_mint_token.subject_id(keyset)
    assert dev_mint_token.subject_id(keyset, "bot") == dev_mint_token.subject_id(keyset, "bot")


def test_the_derived_subject_collides_with_neither_id_peer_signing_derives(keyset):
    """`get_host_id` and `get_system_principal_id` hang off the SAME namespace. A label collision
    would silently make a pasted header act as the platform system principal."""
    ns = uuid.UUID((keyset / "instance.uuid").read_text(encoding="utf-8").strip())
    reserved = {
        str(uuid.uuid5(ns, "agience/agience-host-current-instance")),
        str(uuid.uuid5(ns, "platform/platform-system-principal")),
    }
    assert dev_mint_token.subject_id(keyset) not in reserved


# ---------------------------------------------------------------------------
# The audience and the lifetime: read, not restated
# ---------------------------------------------------------------------------

def test_the_audience_is_the_one_the_verifier_compares_against(keyset, monkeypatch):
    """`_validate_aud_for_principal` rejects a user token whose `aud != config.AUTHORITY_ISSUER`.
    The command reads that same attribute, so the two cannot drift into a 401 that says only
    'Invalid token audience'."""
    monkeypatch.setenv("AUTHORITY_ISSUER", "https://authority.example")
    assert dev_mint_token.audience() == "https://authority.example"


def test_the_audience_follows_ORIGIN_URI_exactly_as_config_derives_it(keyset, monkeypatch):
    """Including the path strip: a path prefix must never appear in `iss`/`aud`, and `config`
    already applies `_origin_only`. Restating the rule here would be a second derivation."""
    monkeypatch.setenv("ORIGIN_URI", "https://origin.example/auth/")
    assert dev_mint_token.audience() == "https://origin.example"


def test_the_lifetime_is_the_packages_own_declared_user_token_lifetime():
    """Not a number chosen in the script. `ACCESS_TOKEN_EXPIRE_HOURS` is what this distribution
    already says an end-user access token lives for, and this is one."""
    from mantle.services.auth_service import ACCESS_TOKEN_EXPIRE_HOURS

    assert dev_mint_token.default_ttl_seconds() == ACCESS_TOKEN_EXPIRE_HOURS * 3600


def test_the_service_jwt_lifetime_is_NOT_the_one_borrowed():
    """`peer_signing._TTL` is sized for one outbound peer call. A header a human pastes into a
    client config and expects to keep working is not that, and copying it would produce a token that
    expires between minting and the first tool call."""
    from mantle.services import peer_signing

    assert dev_mint_token.default_ttl_seconds() != peer_signing._TTL


# ---------------------------------------------------------------------------
# The round trip: this node accepts it, and refuses it tampered
# ---------------------------------------------------------------------------

def _mint(keyset, capsys, *args) -> str:
    assert dev_mint_token.main(["--keys-dir", str(keyset), "--token-only", *args]) == 0
    return capsys.readouterr().out.strip()


def test_the_node_accepts_a_token_this_command_minted(keyset, verifier, capsys):
    """The whole claim, end to end: the command's own output, through the node's own verifier, with
    no trust path added — `_load_manifest_anchors` registered the `mantle` anchor at construction."""
    from mantle.services.auth_service import verify_token

    token = _mint(keyset, capsys)
    claims = verify_token(token)
    assert claims is not None, "the node rejected a token signed by its own trusted key"
    assert claims["iss"] == "mantle"
    assert claims["sub"] == dev_mint_token.subject_id(keyset)
    assert claims["aud"] == dev_mint_token.audience()


def test_a_tampered_token_is_refused(keyset, verifier, capsys):
    """The negative control. Without it, a verifier that accepted anything would pass the test
    above and prove nothing at all about the signature."""
    from mantle.services.auth_service import verify_token

    token = _mint(keyset, capsys)
    head, payload, sig = token.split(".")
    flipped = "B" if sig[0] != "B" else "C"
    assert verify_token("%s.%s.%s%s" % (head, payload, flipped, sig[1:])) is None


def test_a_token_signed_by_a_DIFFERENT_keyset_is_refused(tmp_path, keyset, verifier, capsys):
    """A second negative control against the first one passing for the wrong reason: the signature
    is well-formed and the claims are identical in shape, and it is still refused."""
    from mantle.services.auth_service import verify_token

    stranger = tmp_path / "stranger"
    assert dev_init_keys.main(["--keys-dir", str(stranger)]) == 0
    token, _exp = dev_mint_token.mint(
        stranger, subject=dev_mint_token.subject_id(stranger),
        aud=dev_mint_token.audience(), ttl_seconds=600)
    assert verify_token(token) is None


def test_the_token_resolves_to_a_USER_with_a_user_id(keyset, verifier, capsys):
    """The point of the exercise. `sign_service_jwt` hard-codes `principal_type: service`, which
    resolves to a principal with no `user_id` and 404s on every route that needs one. The absence of
    the claim is what selects the user branch (`payload.get("principal_type", "user")`)."""
    from mantle.services.dependencies import resolve_auth

    token = _mint(keyset, capsys)
    auth = resolve_auth(token, store_db=None)
    assert auth.principal_type == "user"
    assert auth.user_id == dev_mint_token.subject_id(keyset)


def test_sign_service_jwt_still_signs_only_a_service(keyset):
    """The constraint, asserted rather than assumed: nothing here relaxed the one outbound signer."""
    from mantle.services import peer_signing

    peer_signing._priv_pem = None
    token = peer_signing.sign_service_jwt()
    body = token.split(".")[1]
    claims = json.loads(__import__("base64").urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    assert claims["principal_type"] == "service"
    assert claims["sub"] == "mantle"
    peer_signing._priv_pem = None


# ---------------------------------------------------------------------------
# Refusals, and the conventions `mantle-init-keys` set
# ---------------------------------------------------------------------------

def test_no_keys_directory_is_the_same_refusal_mantle_init_keys_gives(monkeypatch, capsys):
    monkeypatch.delenv("KEYS_DIR", raising=False)
    with pytest.raises(SystemExit) as exc:
        dev_mint_token.main([])
    assert exc.value.code == 2
    assert "no keys directory" in capsys.readouterr().err


def test_KEYS_DIR_is_the_fallback(keyset, capsys):
    """Same fallback as `mantle-init-keys`, so the two halves of one operation take the same
    environment."""
    assert dev_mint_token.main(["--token-only"]) == 0
    assert capsys.readouterr().out.strip().count(".") == 2


def test_an_empty_directory_names_the_command_that_fills_it(tmp_path, capsys):
    assert dev_mint_token.main(["--keys-dir", str(tmp_path / "nothing")]) == 1
    err = capsys.readouterr().err
    assert "mantle-init-keys" in err
    assert "mantle.private.pem" in err


def test_a_manifest_that_holds_a_DIFFERENT_key_is_refused_before_minting(tmp_path, keyset, capsys):
    """The one failure this command can catch before the user does. A signing key and a manifest
    anchor that disagree produce a bare `401 Invalid token`, which is indistinguishable from an
    expired token or a broken build — and it is reachable through a shared directory or a restored
    key file."""
    stranger = tmp_path / "stranger"
    assert dev_init_keys.main(["--keys-dir", str(stranger)]) == 0
    (keyset / "authority.manifest.json").write_text(
        (stranger / "authority.manifest.json").read_text(encoding="utf-8"), encoding="utf-8")

    assert dev_mint_token.main(["--keys-dir", str(keyset), "--token-only"]) == 1
    assert "DIFFERENT public key" in capsys.readouterr().err


def test_help_survives_a_legacy_windows_code_page(capsys):
    """A `` in an argparse epilog raises `UnicodeEncodeError` on a cp1252 console and takes
    `--help` down with it — on the platform a newcomer is most likely to be reading it. This has
    bitten once already, so it is measured rather than reviewed for."""
    with pytest.raises(SystemExit):
        dev_mint_token.main(["--help"])
    out = capsys.readouterr().out
    out.encode("cp1252")            # the assertion: raises if any character is unencodable
    assert "KEYS_DIR is the root credential" in out


def test_the_help_names_the_security_consequence(capsys):
    """Saying it out loud IS the deliverable: the command does not create this exposure, it names
    it. Anyone who can read KEYS_DIR could already sign this token by hand."""
    with pytest.raises(SystemExit):
        dev_mint_token.main(["--help"])
    out = capsys.readouterr().out
    assert "can already mint this token by hand" in out


def test_the_printed_client_line_is_the_documented_claude_code_spelling(keyset, capsys):
    """Confirmed against https://code.claude.com/docs/en/mcp.md: `claude mcp add --transport http
    <name> <url>`, with `--header "Key: Value"`. A near-miss spelling is worse than printing the
    header alone, because it fails inside a tool the reader does not yet know how to debug."""
    assert dev_mint_token.main(["--keys-dir", str(keyset), "--url", "http://localhost:8081"]) == 0
    out = capsys.readouterr().out
    assert 'claude mcp add --transport http mantle http://localhost:8081/mcp --header ' \
           '"Authorization: Bearer ' in out
    assert "Authorization: Bearer " in out


def test_the_output_states_that_no_oauth_flow_can_complete_here(keyset, capsys):
    """The limit, where the reader who needs it is standing. A standalone node serves one document
    of the OAuth surface and none of the endpoints."""
    assert dev_mint_token.main(["--keys-dir", str(keyset)]) == 0
    out = capsys.readouterr().out
    assert "no /authorize" in out and "dynamic client registration" in out


def test_everything_printed_survives_a_legacy_windows_code_page(keyset, capsys):
    assert dev_mint_token.main(["--keys-dir", str(keyset)]) == 0
    capsys.readouterr().out.encode("cp1252")
