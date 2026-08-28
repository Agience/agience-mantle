#!/usr/bin/env python3
"""Write a throwaway development keyset into `KEYS_DIR`, so a fresh checkout boots.

    python -m mantle.scripts.dev_init_keys --keys-dir ./.data/keys
    python src/mantle/scripts/dev_init_keys.py --keys-dir ./.data/keys   # no install needed
    python -m mantle.scripts.dev_init_keys --keys-dir ./.data/keys --force   # replace an existing set
    python -m mantle.scripts.dev_init_keys --keys-dir ./.data/keys --dry-run # report, write nothing

Development only. These keys are generated on the spot by a script anyone can read, on a box
with no key custody, and this file's whole job is to make them easy to produce. A deployment's
keys come from its own custody process — the platform installer, a KMS, an init container — and
`--force` exists so that replacing a dev keyset is a decision rather than an accident.

## Why this exists

`main.py`'s lifespan loads key material off disk before it will serve anything, and every loader
in `prism.trust.key_manager` raises rather than inventing a key it cannot have written. That is
the correct behaviour for a store — a service that silently generates its own encryption key at
boot has re-keyed every secret it holds — but it means an empty `KEYS_DIR` is a hard stop, and
nothing else in this distribution writes one. `tests/conftest.py` fabricates an equivalent set
in memory for the suite; this is the same set, on disk, for a human.

## What it writes

| File | Read by | Why boot needs it |
|---|---|---|
| `encryption.key` | `key_manager.init_encryption_key` | Fernet key for secrets and platform settings at rest |
| `inbound_nonce.secret` | `key_manager.init_nonce_secret` | HMAC secret behind the inbound-nonce gate |
| `mantle.private.pem` | `services/peer_signing.init` | signs Mantle's one outbound token, the service JWT |
| `mantle.public.pem` | — | the public half, so the JWK below can be re-derived by hand |
| `instance.uuid` | `services/peer_signing.get_instance_namespace` | the namespace host and system-principal ids are derived from |
| `authority.manifest.json` | `services/oidc._read_authority_manifest` | inline JWKS for the trust anchors this node verifies against |

Only the anchor this node can actually speak for — `mantle` — goes into the manifest. An entry for
a peer would mean asserting a public key for a service whose private key is elsewhere, which is a
trust statement, not a convenience. A node that must verify Origin-signed tokens gets Origin's
anchor from the platform installer; a standalone node names an external IdP through
`AGIENCE_TRUSTED_ISSUERS` instead and needs no peer anchor at all.

`setup.token`, `minio.pass` and the `origin.*` / `licensing_*` keys are deliberately absent, and
nothing here needs them. Mantle has no setup flow of its own — `key_manager.init_setup_token`
only *loads* a token file if one is already there, and boot deletes it either way, so an absent
`setup.token` is the completed state rather than a missing file. `content_service` treats an
uninitialized MinIO password as "no S3 edge configured" and carries on. Writing key files nothing
here reads would suggest they matter.

## Then

    KEYS_DIR=./.data/keys MANTLE_LATTICE_PATH=./mantle.db mantle-serve --port 8081
    mantle-token --keys-dir ./.data/keys

The second command is not optional on a standalone node. The keyset above is ALREADY the whole
credential — `mantle.private.pem` signs, and the `mantle` anchor in `authority.manifest.json` is
this node trusting the matching public half — but nothing in this distribution signed a *user*
token against it, so a node that booted perfectly answered 401 to every request.
`scripts/dev_mint_token.py` is that missing half, and it adds no trust path.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import stat
import sys
import uuid
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

#: The `kid` `services/peer_signing.py` stamps on every JWT it signs. The manifest's JWK must
#: carry the same one or a verifier cannot select the key it needs.
_MANTLE_KID = "mantle-1"

#: 2048 is what `tests/conftest.py` generates and what the platform's own keysets use. Larger is
#: slower to generate for no gain on a keyset whose lifetime is one developer's afternoon.
_RSA_BITS = 2048

#: Every file this script owns. Presence of ANY of them means a keyset is already here — the check
#: is over the set rather than per-file, because a half-written keyset is the state that produces
#: the most confusing boot failure: some material loads, some does not, and the error names only
#: the first missing file.
_KEYSET = (
    "encryption.key",
    "inbound_nonce.secret",
    "mantle.private.pem",
    "mantle.public.pem",
    "instance.uuid",
    "authority.manifest.json",
)

#: This component's name in the ownership marker. Four of the six files above are NOT
#: service-namespaced — `encryption.key`, `inbound_nonce.secret`, `instance.uuid` and
#: `authority.manifest.json` are the names every Agience component's init writes. Pointed at one
#: shared directory, two installs silently overwrite each other, and the damage is asymmetric:
#: losing a signing key costs a restart, but losing `encryption.key` makes every secret and
#: platform setting sealed under it unreadable. There is no recovery, only a re-key of the store.
_COMPONENT = "mantle"

#: Names the install that owns this directory, and fingerprints the files that carry no component
#: in their name. Fingerprints rather than the material: this file is written beside the keys, so
#: it must be useless to anyone who reads it.
_OWNER_FILE = "keyset.owner"

#: The unnamespaced members — the ones a second install collides with.
_SHARED_NAMES = (
    "encryption.key",
    "inbound_nonce.secret",
    "instance.uuid",
    "authority.manifest.json",
)


def _fingerprint(path: Path) -> str:
    """A truncated SHA-256 of a key file — enough to detect replacement, useless for recovery."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _read_owner(keys_dir: Path) -> Optional[dict]:
    """The ownership marker, or None when absent or unreadable.

    Unreadable is treated as absent on purpose: a corrupt marker must not be able to block an
    operator from re-initialising their own development directory.
    """
    try:
        return json.loads((keys_dir / _OWNER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _owner_conflict(keys_dir: Path) -> Optional[tuple]:
    """`(message, forceable)` describing why this directory is not ours, or None when it is.

    Three situations, kept apart because the operator's next step differs and because only one of
    them justifies ignoring `--force`:

    * a marker naming another component -- KNOWN foreign. `--force` must not apply: overwriting a
      component that is not running is not a decision this command line can consent to.
    * key files with no marker at all -- UNKNOWN writer. Also refused, but `--force` may proceed:
      the operator can see the directory and this script cannot, so a blanket refusal here would
      make a stray file an unrecoverable state.
    * a marker that exists but does not parse -- damaged, and evidence this scheme wrote here.
      Treated as ours: a corrupt byte must not lock an operator out of their own directory.
    """
    marker_path = keys_dir / _OWNER_FILE
    marker = _read_owner(keys_dir)

    if marker is not None:
        owner = str(marker.get("component") or "")
        if owner and owner != _COMPONENT:
            return ("%s belongs to the %r install (see %s)." % (keys_dir, owner, _OWNER_FILE),
                    False)
        return None

    if marker_path.exists():
        return None                     # unparseable, but ours by the fact it is here at all

    if any((keys_dir / name).exists() for name in _SHARED_NAMES):
        return ("%s already holds key files written by something that left no %s marker. Those "
                "names are shared across Agience components, so this directory may belong to "
                "another install." % (keys_dir, _OWNER_FILE),
                True)
    return None


def _owner_record(keys_dir: Path) -> dict:
    """The marker's contents: who owns the directory, and what the shared files looked like."""
    return {
        "component": _COMPONENT,
        "shared_fingerprints": {
            name: _fingerprint(keys_dir / name)
            for name in _SHARED_NAMES
            if (keys_dir / name).exists()
        },
    }


def verify_keyset(keys_dir: Path) -> list:
    """Return a list of complaints about the keyset in *keys_dir*; empty means it is intact.

    The check that matters is the last one: a shared file whose fingerprint does not match the
    marker was replaced by something other than this script. That is what a second install
    writing into the same directory looks like, and it is otherwise silent until a secret fails to
    decrypt — long after the run that caused it.
    """
    problems: list = []
    if not keys_dir.is_dir():
        return ["%s does not exist" % keys_dir]

    missing = [name for name in _KEYSET if not (keys_dir / name).exists()]
    if missing:
        problems.append("missing: %s" % ", ".join(missing))

    marker = _read_owner(keys_dir)
    if marker is None:
        problems.append("no %s marker; the owning install is unknown" % _OWNER_FILE)
        return problems

    owner = str(marker.get("component") or "")
    if owner != _COMPONENT:
        problems.append("owned by %r, not %r" % (owner, _COMPONENT))

    for name, expected in (marker.get("shared_fingerprints") or {}).items():
        path = keys_dir / name
        if not path.exists():
            problems.append("%s is gone since it was written" % name)
        elif _fingerprint(path) != expected:
            problems.append(
                "%s was REPLACED after this keyset was written; another install probably shares "
                "this directory" % name)
    return problems


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _public_jwk(public_key: rsa.RSAPublicKey, kid: str) -> dict:
    """The RSA public key as a JWK, in the shape `services/oidc.py` reads out of the manifest."""
    nums = public_key.public_numbers()
    return {
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "kid": kid,
        "n": _b64url(nums.n.to_bytes((nums.n.bit_length() + 7) // 8, "big")),
        "e": _b64url(nums.e.to_bytes((nums.e.bit_length() + 7) // 8, "big")),
    }


def _fernet_key() -> str:
    """A Fernet key without importing Fernet.

    Fernet's key IS 32 random bytes, urlsafe-base64 encoded — `Fernet.generate_key()` does exactly
    this. Spelling it out keeps the script's import surface to `cryptography.hazmat`, so it runs
    from a bare checkout with nothing installed but `cryptography`.
    """
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii")


def _write(path: Path, text: str, *, private: bool) -> None:
    """Write one key file, owner-readable where the platform enforces file modes.

    The mode is set before the content is written, so the secret is never briefly world-readable.
    Windows ignores the group/other bits; that is a property of the platform, not a failure here,
    and a dev keyset is not the place to emulate POSIX permissions.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = stat.S_IRUSR | stat.S_IWUSR if private else 0o644
    fd = os.open(path, flags, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def build_keyset(*, issuer: str, node_uri: str) -> dict:
    """Generate the keyset in memory and return `{filename: (text, is_private)}`.

    Separated from the writing so the dry run reports exactly what a real run would write, rather
    than a description of it that could drift.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=_RSA_BITS)
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")

    manifest = {
        "artifact_id": str(uuid.uuid4()),
        "content_type": "application/vnd.agience.authority+json",
        "schema_version": 1,
        "issuer": issuer,
        "trust_anchors": {
            "mantle": {"uri": node_uri, "jwks": {"keys": [_public_jwk(public_key, _MANTLE_KID)]}},
        },
        "bootstrap_token_hash": None,
    }

    return {
        "encryption.key": (_fernet_key() + "\n", True),
        # 32 bytes of entropy, hex-encoded: the gate HMACs with it, so any high-entropy string
        # works and hex keeps the file safe to `cat` into a shell.
        "inbound_nonce.secret": (secrets.token_hex(32) + "\n", True),
        "mantle.private.pem": (private_pem, True),
        "mantle.public.pem": (public_pem, False),
        # The namespace every derived platform id hangs off. Random, because a dev node is its own
        # instance — reusing another node's namespace would make the two claim the same host id.
        "instance.uuid": (str(uuid.uuid4()) + "\n", False),
        "authority.manifest.json": (json.dumps(manifest, indent=2) + "\n", False),
    }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mantle-init-keys",
        description="Write a throwaway development keyset into KEYS_DIR. Development only.",
        epilog="Then: KEYS_DIR=<dir> MANTLE_LATTICE_PATH=./mantle.db mantle-serve --port 8081, "
               "and `mantle-token --keys-dir <dir>` for a credential the node accepts.",
    )
    parser.add_argument(
        "--keys-dir",
        default=os.getenv("KEYS_DIR"),
        help="Where to write the keyset. Defaults to $KEYS_DIR; required if that is unset.",
    )
    parser.add_argument(
        "--issuer",
        default="http://localhost:8081",
        help="The `issuer` field recorded in authority.manifest.json (default: %(default)s).",
    )
    parser.add_argument(
        "--node-uri",
        default=None,
        help="The mantle anchor's `uri` in the manifest (default: the --issuer value).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing keyset. Without this, an occupied KEYS_DIR is left untouched.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written and exit. Touches nothing.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Check an existing keyset is intact and still owned by this install. Writes nothing.",
    )
    args = parser.parse_args(argv)

    if not args.keys_dir:
        parser.error("no keys directory: pass --keys-dir or set KEYS_DIR")

    keys_dir = Path(args.keys_dir).expanduser().resolve()

    if args.verify:
        problems = verify_keyset(keys_dir)
        if not problems:
            print("Keyset at %s is intact and owned by %r." % (keys_dir, _COMPONENT))
            return 0
        print("Keyset at %s has problems:" % keys_dir, file=sys.stderr)
        for problem in problems:
            print("    %s" % problem, file=sys.stderr)
        return 1

    # Ownership is checked before the keyset check, and --force does NOT override it. Replacing
    # this install's own keys is a decision an operator can make about their own data; replacing
    # another install's `encryption.key` destroys secrets belonging to a component that is not
    # even running, and no flag on this command line is informed consent for that.
    conflict = _owner_conflict(keys_dir)
    if conflict and not (conflict[1] and args.force):
        message, forceable = conflict
        print(message, file=sys.stderr)
        print(
            "\nRefusing to write%s: these file names are shared across Agience components, so\n"
            "overwriting them would take another install's secrets with them. Give each install\n"
            "its own directory, for example:\n"
            "\n"
            "    KEYS_DIR=~/.agience/_secrets/%s\n"
            "\n"
            "%s"
            % ("" if forceable else ", and --force does not apply",
               _COMPONENT,
               "Re-run with --force if this directory really is unused."
               if forceable else
               "If that install is gone, remove its directory and re-run."),
            file=sys.stderr,
        )
        return 1

    # Refuse before generating anything. A keyset is only useful as a set — rotating the signing
    # key while leaving the manifest's JWK behind produces a node that signs tokens it then
    # refuses to verify, and that failure surfaces far from here.
    existing = [name for name in _KEYSET if (keys_dir / name).exists()]
    if existing and not args.force:
        print("A keyset is already present in %s:" % keys_dir, file=sys.stderr)
        for name in existing:
            print("    %s" % name, file=sys.stderr)
        # ASCII only in everything this script prints: a Windows console defaults to a legacy
        # codepage, and a message that mangles itself on the platform a newcomer is most likely to
        # be reading it on is a message that failed at its one job.
        print(
            "\nRefusing to overwrite. Re-run with --force to replace it, and expect every secret\n"
            "sealed under the old encryption key to become unreadable: a Fernet key change is a\n"
            "re-key of the whole store, not a rotation.",
            file=sys.stderr,
        )
        return 1

    node_uri = args.node_uri or args.issuer

    if args.dry_run:
        print("Would write %d file(s) into %s:" % (len(_KEYSET), keys_dir))
        for name in _KEYSET:
            marker = "  (replacing)" if (keys_dir / name).exists() else ""
            print("    %s%s" % (name, marker))
        return 0

    keys_dir.mkdir(parents=True, exist_ok=True)
    files = build_keyset(issuer=args.issuer, node_uri=node_uri)

    # The set is generated before the first write, so a failure part-way through generation cannot
    # leave a half-keyset on disk.
    for name, (text, private) in files.items():
        _write(keys_dir / name, text, private=private)

    # Written last, so it can fingerprint what actually landed rather than what was intended, and
    # so a run that dies mid-write leaves no marker claiming a keyset that is not there.
    _write(keys_dir / _OWNER_FILE,
           json.dumps(_owner_record(keys_dir), indent=2) + "\n",
           private=False)

    print("Wrote a development keyset to %s:" % keys_dir)
    for name in files:
        print("    %s" % name)
    # The next TWO commands, not one. A booted node with no credential answers 401 to everything and
    # is indistinguishable from a broken one, and nothing else in this distribution announces that
    # `mantle-token` exists — the keyset written above is already both halves of the credential
    # (the signing key, and the manifest anchor that trusts it), so the reader is one command away
    # and has no way to know it.
    print(
        "\nDEVELOPMENT ONLY - these keys have no custody. Start the node with:\n"
        "    KEYS_DIR=%s MANTLE_LATTICE_PATH=./mantle.db mantle-serve --port 8081\n"
        "\nThen mint a token it accepts (this keyset is already the credential):\n"
        "    mantle-token --keys-dir %s" % (keys_dir, keys_dir)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
