"""`agience` — the whole of a standalone node, in one executable.

    agience init     [--dir D]          seed key material into an EMPTY directory
    agience serve    [--dir D] [--port] run the node
    agience token    [--dir D]          mint a credential this node trusts
    agience version                     what this binary is, and what it was built from

Why a binary at all. `pip install -e '.[service]'` cannot resolve on a machine that is not a
developer's: `agience-prism-py` is not on PyPI, so a source delivery assumes a sibling checkout at
`../agience-prism/py`. This build vendors prism into the artifact instead, so what is handed to
someone else carries everything it needs and resolves nothing at install time.

A front door rather than a second implementation. Every subcommand delegates to the module that
already does the job — `mantle.scripts.dev_init_keys`, `uvicorn`, `mantle.scripts.dev_mint_token` —
and nothing here decides anything those modules decide. A packaging layer that reimplemented the
seed or the server is how an installed node comes to differ from a developer's checkout.

The four paths are set together. `MANTLE_SSE_DIR` and `MANTLE_CELL_DIR` default to a path derived
from the install root rather than from the lattice, so a node that set only the store path would
share its indexes with anything else running from the same root, and both would come up healthy
while answering searches from each other's postings. `_env_for` is the one place that resolves them,
so a subcommand cannot set three of the four.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Written by the build. A binary that cannot say what it is cannot be promoted by digest, and
# "which build is this box running" is the first question of every deploy.
# Imported by its own name rather than as `packaging.…`, even though this file lives in
# `packaging/`. `packaging` is a real PyPI distribution that pip, setuptools and much of the build
# ecosystem import, and shadowing it from a directory that lands on `sys.path[0]` breaks tooling far
# from here in a way that reads as a pip bug. This directory has no `__init__.py` and the module
# below is uniquely named for the same reason.
try:
    from agience_build_info import BUILD_INFO  # type: ignore
except Exception:  # a source checkout, run directly — not a build
    BUILD_INFO = {"version": "dev", "built": "not-a-build", "commit": "unknown"}


def _default_dir() -> Path:
    """Where a node lives when nobody says.

    An absolute, per-user path rather than the working directory or the executable's own folder.
    A relative default resolves against wherever the node was started, which is how
    `MANTLE_LATTICE_PATH`'s own default (`mantle-lattice.db`) has a node mint a 4 KB store and
    report "Found 5 artifacts to reindex" while the real store sits elsewhere untouched.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "Agience" / "node"
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "agience" / "node"


def _env_for(d: Path, issuer: str | None = None) -> None:
    """Point every path this node reads at `d`. The one place the four are resolved together."""
    d = d.resolve()
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ["AGIENCE_BASE_DIR"] = str(d)
    os.environ["KEYS_DIR"] = str(d / "keys")
    os.environ["MANTLE_LATTICE_PATH"] = str(d / "lattice.db")
    os.environ["MANTLE_SSE_DIR"] = str(d / ".data" / "mantle-sse")
    os.environ["MANTLE_CELL_DIR"] = str(d / ".data" / "mantle-cells")
    if issuer:
        os.environ["AUTHORITY_ISSUER"] = issuer
        os.environ["MANTLE_URI"] = issuer
    # Pin the BLAS thread pool. `numpy.linalg.eigh` faults when two Python threads call it at once
    # on this build (3/3 exit 139 with no agience code in the picture), and an ASGI host dispatches
    # sync endpoints on a threadpool, so two concurrent requests that both reach the instrument are
    # the fatal pattern: the whole process dies with no traceback.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def cmd_init(args: argparse.Namespace) -> int:
    d = Path(args.dir)
    _env_for(d)
    d.mkdir(parents=True, exist_ok=True)
    keys = d / "keys"
    # Refuse a seeded directory rather than no-op over it. `dev_init_keys` skips what exists, so
    # re-running is harmless — but "harmless" is not "intended", and an operator who runs init at a
    # live node expecting a fresh start should be told, not handed a silent success.
    if (keys / "mantle.private.pem").exists() and not args.force:
        print(f"already seeded — {keys / 'mantle.private.pem'} exists.", file=sys.stderr)
        print("Use --force to run over it (it skips existing files and deletes nothing),", file=sys.stderr)
        print("or point --dir at an empty directory.", file=sys.stderr)
        return 1
    from mantle.scripts import dev_init_keys
    rc = dev_init_keys.main(["--keys-dir", str(keys)])
    if rc in (0, None):
        print(f"\nseeded {d}")
        print(f"next:  agience serve --dir {d}")
    return rc or 0


def cmd_serve(args: argparse.Namespace) -> int:
    d = Path(args.dir)
    issuer = args.issuer or f"http://127.0.0.1:{args.port}"
    _env_for(d, issuer)
    if not (d / "keys" / "mantle.private.pem").exists():
        print(f"not seeded — no keys at {d / 'keys'}. Run:  agience init --dir {d}", file=sys.stderr)
        return 1
    import uvicorn
    print(f"agience {BUILD_INFO['version']} — serving {d} on http://{args.host}:{args.port}")
    uvicorn.run("mantle.main:app", host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    d = Path(args.dir)
    _env_for(d, args.issuer)
    from mantle.scripts import dev_mint_token
    argv = ["--keys-dir", str(d / "keys")]
    if args.token_only:
        argv.append("--token-only")
    if args.ttl_hours:
        argv += ["--ttl-hours", str(args.ttl_hours)]
    return dev_mint_token.main(argv) or 0


def cmd_version(args: argparse.Namespace) -> int:
    print(f"agience {BUILD_INFO['version']}")
    print(f"  built   {BUILD_INFO['built']}")
    print(f"  commit  {BUILD_INFO['commit']}")
    print(f"  python  {sys.version.split()[0]}")
    print(f"  frozen  {getattr(sys, 'frozen', False)}")
    try:
        import mantle, prism  # noqa: F401
        print(f"  mantle  {Path(mantle.__file__).parent}")
        print(f"  prism   bundled")
    except Exception as exc:
        print(f"  ⛔ bundle is incomplete: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agience", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="store_true", help="print the build and exit")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--dir", default=str(_default_dir()), help="the node's directory (keys, store, indexes)")
        return sp

    s = common(sub.add_parser("init", help="seed key material into an empty directory"))
    s.add_argument("--force", action="store_true", help="run over an already-seeded directory (deletes nothing)")
    s.set_defaults(fn=cmd_init)

    s = common(sub.add_parser("serve", help="run the node"))
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8081)
    s.add_argument("--issuer", default=None, help="the authority this node's tokens name (default: its own address)")
    s.add_argument("--log-level", default="warning")
    s.set_defaults(fn=cmd_serve)

    s = common(sub.add_parser("token", help="mint a credential this node trusts"))
    s.add_argument("--issuer", default=None)
    s.add_argument("--ttl-hours", type=int, default=None)
    s.add_argument("--token-only", action="store_true")
    s.set_defaults(fn=cmd_token)

    sub.add_parser("version", help="what this binary is").set_defaults(fn=cmd_version)

    args = p.parse_args(argv)
    if args.version:
        return cmd_version(args)
    if not getattr(args, "fn", None):
        p.print_help()
        return 0
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
