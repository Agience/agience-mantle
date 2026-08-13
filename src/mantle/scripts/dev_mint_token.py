#!/usr/bin/env python3
"""Mint a user token this node already trusts, from the keyset in `KEYS_DIR`.

    mantle-token --keys-dir ./.data/keys
    python -m mantle.scripts.dev_mint_token --keys-dir ./.data/keys
    python src/mantle/scripts/dev_mint_token.py --keys-dir ./.data/keys   # no install needed

⛔ KEYS_DIR IS THE ROOT CREDENTIAL OF A STANDALONE NODE. Anyone who can read that directory can
already mint this token by hand: `mantle.private.pem` is the signing key, and
`authority.manifest.json` is the node telling itself to trust the matching public half. This
command does not create that exposure — it *names* it. Read access to `KEYS_DIR` is full access to
every artifact in the store, and no grant, revocation or expiry limits it. Back it up separately
and under different custody, exactly as README's "Backing a node up" says.

## Why this exists

`mantle-init-keys` writes `mantle.private.pem` and an `authority.manifest.json` whose
`trust_anchors.mantle.jwks` is the matching public half, and boot seeds that anchor as a trusted
issuer (`services/oidc._load_manifest_anchors`). So **the node already trusts its own key** — it
just never told anyone, and nothing here signed a *user* token. `services/peer_signing` signs one
thing and one thing only, a service JWT with `principal_type: service`, which
`services/dependencies.resolve_auth` resolves to a principal with no `user_id`; that principal
404s on every route that needs one. A standalone node was therefore installable, bootable and
unusable: no Origin to sign a user token, no external IdP configured, and no way in.

This adds no trust path. The anchor is already seeded and already verified; this command signs
against it.

## The token

    header  {"kid": "mantle-1", "alg": "RS256"}
    claims  {"iss": "mantle", "sub": <uuid>, "aud": <AUTHORITY_ISSUER>, "iat": …, "exp": …}

`iss` is the anchor NAME, not a URL: `_load_manifest_anchors` registers each `trust_anchors` key as
an issuer whose JWKS is that anchor's, and `mantle` is the only anchor `mantle-init-keys` writes.
There is deliberately no `principal_type` claim — `resolve_auth` reads
`payload.get("principal_type", "user")`, so its ABSENCE is what selects the user branch, and that
is the shape measured against a live node. Nothing here weakens `sign_service_jwt`: a service token
is still a service token.

## What this is not

It is not an OAuth flow, and a standalone node has none. The complete OAuth surface here is one
document, `/.well-known/oauth-protected-resource`; there is no
`/.well-known/oauth-authorization-server`, no `/authorize`, no `/token`, and no dynamic client
registration. A standards-compliant MCP OAuth flow cannot complete against a standalone Mantle. A
static `Authorization` header carrying this token is the supported path, which is why this command
prints the header.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

#: The anchor name `mantle-init-keys` writes into `authority.manifest.json`, and the `iss` the
#: verifier selects a JWKS by. Same string as `services/peer_signing._NAME`.
_ISSUER = "mantle"

#: The `kid` on the manifest's JWK. Same string as `services/peer_signing._KID` and
#: `dev_init_keys._MANTLE_KID`; a verifier that cannot select the key by kid rejects the token.
_KID = "mantle-1"

#: Files this command reads. All three, because a token minted from a private key whose public half
#: is not the one in the manifest verifies nowhere, and the resulting 401 says only "Invalid token".
_REQUIRED = ("mantle.private.pem", "instance.uuid", "authority.manifest.json")

#: The label the default subject id is derived under. A path-shaped name, like the two ids
#: `services/peer_signing` already derives from the same namespace
#: (`agience/agience-host-current-instance`, `platform/platform-system-principal`), so the three
#: cannot collide.
_SUBJECT_LABEL = "mantle/local-user"


# ---------------------------------------------------------------------------
#  Derivations
# ---------------------------------------------------------------------------

def instance_namespace(keys_dir: Path) -> Optional[uuid.UUID]:
    """The keyset's `instance.uuid`, as `services/peer_signing.get_instance_namespace` reads it.

    Read here rather than imported because that module imports `jose`, which arrives with the
    `[service]` extra; this command runs on the base install, beside `mantle-init-keys`, which is
    where a reader who has just written a keyset actually is.
    """
    try:
        return uuid.UUID((keys_dir / "instance.uuid").read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def subject_id(keys_dir: Path, label: Optional[str] = None) -> Optional[str]:
    """The stable `sub` for this keyset — `uuid5(instance.uuid, "mantle/local-user"[/label])`.

    STABILITY IS THE POINT. `sub` is the user id: `resolve_auth`'s user branch returns
    `user_id = payload["sub"]`, and every grant, every owner record and every light cone hangs off
    that string. A random subject per run would mint a *new* principal each time, holding no grants,
    and would make the artifacts written by the previous run unreachable — a node that silently
    forgets everything on every token refresh.

    Derived the same way `services/oidc.external_user_id` derives an external user's id — a uuid5
    over a namespace and a name — so the id is computed, never fetched, and never stored. The
    namespace is the keyset's own `instance.uuid`, so the answer is reproducible for one keyset and
    different for any other: two nodes cannot accidentally name the same principal, and re-running
    `mantle-init-keys --force` is a new instance and therefore a new person, which it should be.

    `label` names a second (third, …) local identity on the same keyset, for separating one client's
    artifacts from another's.
    """
    ns = instance_namespace(keys_dir)
    if ns is None:
        return None
    name = "%s/%s" % (_SUBJECT_LABEL, label) if label else _SUBJECT_LABEL
    return str(uuid.uuid5(ns, name))


def audience() -> str:
    """The `aud` this node's verifier requires: `config.AUTHORITY_ISSUER`.

    READ FROM `config`, NOT RESTATED. `services/dependencies._validate_aud_for_principal` rejects a
    user token whose `aud != config.AUTHORITY_ISSUER`, so the minting side and the verifying side
    have to be the same expression or they drift into a 401 that says only "Invalid token
    audience". This calls the same module attribute the verifier compares against.

    `load_env` then reload, in that order and for the same reason `main.py` does it: `config` binds
    its Phase-1 values at import, `.env` is loaded afterwards and only fills what the shell left
    unset, so a value that came from `.env` reaches the module only on a re-bind. `main.py` gets its
    re-bind from Phase 2 (`load_settings_from_db` recomputes `AUTHORITY_ISSUER` at the end); a
    one-shot command has no Phase 2, so it reloads.

    THE ONE CASE THIS CANNOT SEE is a stored `branding.origin_uri` row: that lives in the lattice,
    the running node reads it at Phase 2, and this command opens no store. An operator who set the
    authority through the settings API rather than the environment must set `AUTHORITY_ISSUER` in
    this shell too, or the node answers `401 Invalid token audience` — which names the mismatch. Not
    papered over by opening the lattice here: a token minter that could open the store would be a
    second reader of a database whose whole point is that one process owns it.
    """
    import importlib

    from mantle import config

    config.load_env()
    importlib.reload(config)
    return config.AUTHORITY_ISSUER


def default_ttl_seconds() -> int:
    """The lifetime, taken from the package's own declared user-token lifetime.

    NOT A NUMBER CHOSEN HERE. `services/auth_service.ACCESS_TOKEN_EXPIRE_HOURS` is what this
    distribution already says an end-user access token lives for, and this token is exactly that —
    a user token, on the user branch of `resolve_auth`. `peer_signing._TTL` (300s) is the other
    lifetime in the tree and is the wrong one to copy: it is sized for one outbound peer call, and a
    human pasting a header into a client config is not that.
    """
    from mantle.services.auth_service import ACCESS_TOKEN_EXPIRE_HOURS

    return int(ACCESS_TOKEN_EXPIRE_HOURS) * 3600


# ---------------------------------------------------------------------------
#  Signing
# ---------------------------------------------------------------------------

def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _load_private_key(keys_dir: Path) -> rsa.RSAPrivateKey:
    pem = (keys_dir / "mantle.private.pem").read_bytes()
    return serialization.load_pem_private_key(pem, password=None)


def manifest_anchor_matches(keys_dir: Path, key: rsa.RSAPrivateKey) -> Optional[str]:
    """None when the manifest's `mantle` anchor IS this private key's public half; else why not.

    The one failure this command can detect before the user finds it. A signing key and a manifest
    that disagree produce a token the node refuses with a bare `401 Invalid token`, and nothing in
    that answer distinguishes "your keyset is mismatched" from "your token expired" or "this build
    is broken". It is a reachable state: `mantle-init-keys` writes both files together, but a
    directory two installs share, a restored `mantle.private.pem` from a different backup, or a
    hand-edited manifest all break the pair.
    """
    try:
        manifest = json.loads((keys_dir / "authority.manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return "authority.manifest.json is unreadable (%s)" % exc
    anchor = ((manifest.get("trust_anchors") or {}).get(_ISSUER) or {})
    keys = ((anchor.get("jwks") or {}).get("keys") or [])
    jwk = next((k for k in keys if isinstance(k, dict) and k.get("kid") == _KID), None)
    if jwk is None:
        return ("authority.manifest.json has no trust_anchors.%s JWK with kid=%r, so this node "
                "verifies nothing signed by mantle.private.pem" % (_ISSUER, _KID))

    nums = key.public_key().public_numbers()
    expected_n = _b64url(nums.n.to_bytes((nums.n.bit_length() + 7) // 8, "big"))
    expected_e = _b64url(nums.e.to_bytes((nums.e.bit_length() + 7) // 8, "big"))
    if jwk.get("n") != expected_n or jwk.get("e") != expected_e:
        return ("authority.manifest.json's %s anchor is a DIFFERENT public key from "
                "mantle.private.pem, so a token signed here is rejected by this node" % _ISSUER)
    return None


def mint(keys_dir: Path, *, subject: str, aud: str, ttl_seconds: int,
         now: Optional[int] = None) -> Tuple[str, int]:
    """Sign the JWT. Returns `(token, exp)`.

    RS256 spelled out against `cryptography` rather than through `jose`, so this command has the
    same install requirement as `mantle-init-keys` (the base install, `cryptography` alone) — the
    two halves of one operation should not need different extras. RS256 IS RSASSA-PKCS1-v1_5 over
    SHA-256 across `b64url(header) + "." + b64url(payload)`; there is no room here for the two
    spellings to differ, and the round trip through the node's own verifier is what proves it.
    """
    issued = int(time.time()) if now is None else int(now)
    exp = issued + int(ttl_seconds)
    header = {"alg": "RS256", "kid": _KID, "typ": "JWT"}
    claims = {"iss": _ISSUER, "sub": subject, "aud": aud, "iat": issued, "exp": exp}

    def _seg(obj) -> str:
        return _b64url(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8"))

    signing_input = ("%s.%s" % (_seg(header), _seg(claims))).encode("ascii")
    signature = _load_private_key(keys_dir).sign(
        signing_input, padding.PKCS1v15(), hashes.SHA256())
    return "%s.%s" % (signing_input.decode("ascii"), _b64url(signature)), exp


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def _iso(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def main(argv: list | None = None) -> int:
    # ASCII only in everything argparse may print. A Windows console defaults to a legacy code page
    # and `argparse` writes help through it; a single non-cp1252 character in a description or an
    # epilog raises `UnicodeEncodeError` and takes `--help` down with it. The module docstring above
    # is not printed and may use whatever it likes.
    parser = argparse.ArgumentParser(
        prog="mantle-token",
        description="Mint a user token this node already trusts, from the keyset in KEYS_DIR. "
                    "Development only.",
        epilog="SECURITY: KEYS_DIR is the root credential of a standalone node. Anyone who can "
               "read it can already mint this token by hand - mantle.private.pem signs it and "
               "authority.manifest.json is the node trusting the matching public key. This command "
               "names that exposure, it does not create it. Read access to KEYS_DIR is full access "
               "to the store, bounded by no grant and no revocation.",
    )
    parser.add_argument(
        "--keys-dir",
        default=os.getenv("KEYS_DIR"),
        help="Where the keyset is. Defaults to $KEYS_DIR; required if that is unset.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Which local identity to mint for. A UUID is used verbatim; any other value names a "
             "second identity on this keyset. Omitted, the subject is derived from instance.uuid, "
             "so the same keyset always mints for the same person and last run's artifacts stay "
             "reachable.",
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=None,
        help="Token lifetime. Defaults to services/auth_service.ACCESS_TOKEN_EXPIRE_HOURS, this "
             "package's own declared lifetime for an end-user access token.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8081",
        help="This node's base URL, for the printed client line (default: %(default)s).",
    )
    parser.add_argument(
        "--name",
        default="mantle",
        help="Name for the MCP server in the printed client line (default: %(default)s).",
    )
    parser.add_argument(
        "--token-only",
        action="store_true",
        help="Print the token and nothing else, for TOKEN=$(mantle-token --token-only).",
    )
    args = parser.parse_args(argv)

    if not args.keys_dir:
        parser.error("no keys directory: pass --keys-dir or set KEYS_DIR")

    keys_dir = Path(args.keys_dir).expanduser().resolve()

    missing = [n for n in _REQUIRED if not (keys_dir / n).is_file()]
    if missing:
        print("No usable keyset in %s (missing: %s)." % (keys_dir, ", ".join(missing)),
              file=sys.stderr)
        print("\nWrite one first:\n    mantle-init-keys --keys-dir %s" % keys_dir, file=sys.stderr)
        return 1

    try:
        key = _load_private_key(keys_dir)
    except (OSError, ValueError, TypeError) as exc:
        print("mantle.private.pem in %s could not be read as a private key: %s" % (keys_dir, exc),
              file=sys.stderr)
        return 1

    complaint = manifest_anchor_matches(keys_dir, key)
    if complaint:
        print("Refusing to mint a token this node would reject.", file=sys.stderr)
        print("    %s" % complaint, file=sys.stderr)
        print("\nThe signing key and the trust anchor are written together and must stay a set.\n"
              "    mantle-init-keys --keys-dir %s --verify\n"
              "and, if this directory is really this install's own, re-write it with --force -\n"
              "expecting every secret sealed under the old encryption key to become unreadable."
              % keys_dir, file=sys.stderr)
        return 1

    if args.subject:
        try:
            subject = str(uuid.UUID(args.subject))
        except ValueError:
            subject = subject_id(keys_dir, args.subject) or ""
    else:
        subject = subject_id(keys_dir) or ""
    if not subject:
        print("instance.uuid in %s is not a UUID, so no stable subject can be derived from this "
              "keyset." % keys_dir, file=sys.stderr)
        return 1

    ttl = int(args.ttl_hours * 3600) if args.ttl_hours is not None else default_ttl_seconds()
    if ttl <= 0:
        parser.error("--ttl-hours must be positive")

    aud = audience()
    token, exp = mint(keys_dir, subject=subject, aud=aud, ttl_seconds=ttl)

    if args.token_only:
        print(token)
        return 0

    base = args.url.rstrip("/")
    # `keys_dir / name`, not `"%s/mantle.private.pem"`: a printed path a reader may paste has to be
    # the platform's own, and string-joining produces `C:\...\keys/mantle.private.pem` on Windows.
    print("Minted a user token, signed by %s and trusted by this node's own\n"
          "authority.manifest.json anchor. DEVELOPMENT ONLY - these keys have no custody.\n"
          % (keys_dir / "mantle.private.pem"))
    print("    subject   %s" % subject)
    print("    audience  %s   (config.AUTHORITY_ISSUER - what the verifier requires)" % aud)
    print("    expires   %s   (%.4g hours)" % (_iso(exp), ttl / 3600.0))
    print("\nAdd it to Claude Code:\n")
    print('    claude mcp add --transport http %s %s/mcp --header "Authorization: Bearer %s"'
          % (args.name, base, token))
    print("\nOr send the header yourself:\n")
    print("    Authorization: Bearer %s" % token)
    print(
        "\nThe subject is derived from this keyset's instance.uuid, so re-running mints for the\n"
        "SAME person and everything the last token stored stays reachable. There is no OAuth flow\n"
        "to complete here: a standalone node serves /.well-known/oauth-protected-resource and\n"
        "nothing else of the protocol - no authorization server, no /authorize, no /token, no\n"
        "dynamic client registration - so a static Authorization header is the supported path.\n"
        "\nSECURITY: anyone who can read %s can mint this token by hand. That directory is the\n"
        "root credential of this node, bounded by no grant and no revocation." % keys_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
