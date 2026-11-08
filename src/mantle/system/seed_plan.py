"""Plan a seed pass: print the ledger entries it will make, before its credential is minted.

A seed credential makes only the entries in its plan (Origin, review 25). This runs the pass in
planning mode (`services/seed_plan.py`) against a copy of the store, with a fresh salt for its new
grant ids, and prints each entry's body and digest for the operator to read. Then:

    python -m mantle.system.seed_plan --pass startup --out /root/seed-plan.json
    python -m mantle.system.seed_plan --pass manage-seed --action migrate --out /root/plan.json

  1. read every entry it printed;
  2. mint the seed credential with the printed digests (`POST /oracle/seed-credential`
     `{"minutes": 10, "plan": [...]}`, at Origin's console, with a step-up);
  3. run the pass with `MANTLE_SEED_CREDENTIAL_FILE` and `MANTLE_SEED_PLAN_FILE` set: the startup
     pass is the restart of Mantle (cutover step 20), `manage-seed` is
     `python -m mantle.system.manage_seed --action ...` (step 26).

The pass derives the same grant ids from the plan's salt, refuses before Origin is asked any entry
outside the plan, and reports any planned entry it did not make. Plan again if the store changed
between planning and the pass: Origin then refuses what differs, and nothing outside the plan is
made.

Planning never reaches the real store, Origin or the object store. The process is isolated before
any `mantle` module is imported (Origin's address and every object-store, AWS and mesh setting
unset), and again after the copy's own settings are loaded (`_seal`: no Origin, no bucket,
object-store clients that fail, a content tier with no remote), so any call outward fails the
planning run. It runs on a copy of the store, named with `--store` or `MANTLE_LATTICE_PATH` and
read as its owner, in a temporary folder, with the index switched off and lattice key custody on
the copy: a pass that must open a key held as threshold shares fails in planning and
plans nothing. The copy is removed after.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import shutil
import sqlite3
import sys
import tempfile

logger = logging.getLogger("mantle.system.seed_plan")

PASSES = ("startup", "manage-seed")

#: Settings that would let a planning run reach anything but its own copy, unset before any
#: `mantle` module reads them (review 27): each named one, and every setting under each prefix
#: (the object stores' buckets, endpoints and credentials, root-user and AWS settings included,
#: and the mesh).
_OUTSIDE = ("ORIGIN_URI", "MANTLE_SSE_STORE", "MANTLE_CELL_STORE", "MANTLE_SEED_CREDENTIAL_FILE",
            "MANTLE_SEED_PLAN_FILE", "ORIGIN_INTERNAL_URI")
_OUTSIDE_PREFIXES = ("CONTENT_", "AWS_", "MESH_", "OVH_", "S3_", "BACKUP_")


def _copy_store(src: str, folder: str) -> str:
    dst = os.path.join(folder, "plan-copy.db")
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as a, sqlite3.connect(dst) as b:
        a.backup(b)
    return dst


def _scrub_copy(folder: str, *, observe=None) -> None:
    """Overwrite the planning copy of the store (and its log and index files) with zeros, then
    remove the folder (review 28, I4): the copy holds what the store holds, and a removed file's
    bytes stay on the disk until written over. `observe(path, data)` sees each file as overwritten
    (for a test)."""
    paths = []
    for dirpath, _dirs, files in os.walk(folder):       # the store, its log, and `sse/`, `cells/`
        paths.extend(os.path.join(dirpath, f) for f in files)
    for path in sorted(paths):
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            fh.write(b"\0" * size)
            fh.flush()
            os.fsync(fh.fileno())
        if observe is not None:
            with open(path, "rb") as fh:
                observe(path, fh.read())
    shutil.rmtree(folder, ignore_errors=True)


def _isolate(folder: str, store: str) -> None:
    """The process's settings for a planning run: before any `mantle` module is imported."""
    for name in list(os.environ):
        if name in _OUTSIDE or name.startswith(_OUTSIDE_PREFIXES):
            os.environ.pop(name, None)
    os.environ["MANTLE_LATTICE_PATH"] = store
    os.environ["MANTLE_SSE_DIR"] = os.path.join(folder, "sse")
    os.environ["MANTLE_CELL_DIR"] = os.path.join(folder, "cells")
    os.environ["MANTLE_MASTER_KEY_CUSTODY"] = "lattice"
    os.environ["MANTLE_AUTHORITY_LEDGER"] = "off"


class _Unreachable:
    """An object-store client for a planning run: it has no credentials, and every call on it
    fails rather than reaching a bucket."""
    _request_signer = type("Signer", (), {"_credentials": None})()

    def __getattr__(self, name):
        raise RuntimeError("a planning run reaches no object store")


def _refuse(*_a, **_k):
    raise RuntimeError("a planning run reaches no object store and no Origin")


def _seal() -> None:
    """After the copy's settings are loaded (`config.load_settings_from_db`, which rebinds
    Origin's address and the bucket from the copied store), take them away again (review 27):
    no Origin, no bucket, object-store clients that fail, and a content tier with no remote. Any
    call the pass makes outward then fails, and the planning run fails with it, rather than
    reaching production."""
    from mantle import config
    from mantle.services import content_service as cs
    from mantle.shard import content_tier

    config.ORIGIN_URI = ""
    config.CONTENT_BUCKET = ""
    for name in ("_EDGE_BUCKET", "_DURABLE_BUCKET"):
        setattr(cs, name, "")
    for name in ("_s3_edge_internal", "_s3_edge_public", "_s3_edge_server", "_s3_durable"):
        setattr(cs, name, _Unreachable())
    cs._make_s3_client = _refuse
    content_tier.open_ovh_store = _refuse


def run_pass(name: str, db, args) -> None:
    """The pass, as the server or its command runs it."""
    from mantle.services.system_identity import system_acting_context

    if name == "startup":
        from mantle.services.issuers import seed_platform_issuer_artifacts
        from mantle.services.seed_provisioning.platform_email import ensure_platform_email_sender
        from mantle.services.seed_provisioning.platform_operators import ensure_platform_operators
        with system_acting_context(scope="platform.issuer-seed"):
            seed_platform_issuer_artifacts(db)
        with system_acting_context(scope="platform.email-seed"):
            ensure_platform_email_sender(db)
        with system_acting_context(scope="platform.operators-seed"):
            ensure_platform_operators(db)
        return
    from mantle.system import manage_seed as ms
    action = {"seed": lambda: ms.action_seed(db, False),
              "populate": lambda: ms.action_populate(db, False),
              "grant-write": lambda: ms.action_grant_write(db, args.user, False),
              "revoke": lambda: ms.action_revoke(db, args.user, False),
              "migrate": lambda: ms.action_migrate(db, False)}[args.action]
    with system_acting_context(scope="platform.manage-seed"):
        action()


def plan(name: str, db, args, *, salt: str) -> list:
    """The entries the pass makes over `db` (a copy), in order."""
    from mantle.services import seed_plan
    with seed_plan.planning(salt) as made:
        run_pass(name, db, args)
    return made


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="pass_name", choices=PASSES, required=True)
    ap.add_argument("--action", choices=("seed", "populate", "grant-write", "revoke", "migrate"))
    ap.add_argument("--user")
    ap.add_argument("--out", required=True, help="the plan file to write (never replaced)")
    ap.add_argument("--store", default=os.getenv("MANTLE_LATTICE_PATH", ""),
                    help="the store to plan against (default MANTLE_LATTICE_PATH); run as its "
                         "owner, which may read it")
    args = ap.parse_args(argv)
    if args.pass_name == "manage-seed" and not args.action:
        ap.error("--action is required for --pass manage-seed")
    if os.path.exists(args.out):
        ap.error(f"{args.out} exists")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    # The store is named, not looked up through `mantle.config`: no `mantle` module is imported
    # until the process is isolated (review 27).
    if not args.store:
        ap.error("name the store: --store or MANTLE_LATTICE_PATH")
    src = os.path.abspath(os.path.expanduser(args.store))
    if not os.path.isfile(src):
        ap.error(f"{src} is no store file")
    folder = tempfile.mkdtemp(prefix="seed-plan-")
    try:
        _isolate(folder, _copy_store(src, folder))
        from prism.trust.key_manager import init_encryption_key, init_nonce_secret
        init_encryption_key()
        init_nonce_secret()
        from mantle.services import peer_signing
        peer_signing.init()
        from mantle import config
        from mantle.services.dependencies import get_store_db
        from mantle.services.platform_settings_service import settings as platform_settings
        db = next(get_store_db())
        platform_settings.load_all(db)
        config.set_settings_provider(platform_settings.get)
        config.load_settings_from_db()
        _seal()

        salt = secrets.token_hex(16)
        entries = plan(args.pass_name, db, args, salt=salt)
        from mantle.services import seed_plan
        written = seed_plan.write_plan(args.out, args.pass_name, salt, entries)
    finally:
        try:
            from mantle.db import backend
            if backend._HANDLE is not None:          # the copy's connections, before its bytes
                backend._HANDLE.artifacts.db.conn().close()
        except Exception:  # noqa: BLE001 — closing is best effort; the overwrite still runs
            pass
        _scrub_copy(folder)

    for i, e in enumerate(written["entries"], 1):
        print(f"[{i}] {e['digest']}\n    {json.dumps(e['body'], sort_keys=True)}")
    print("\nplan for the seed credential:")
    print(json.dumps([e["digest"] for e in written["entries"]]))
    logger.info("%d planned entr(ies) written to %s", len(written["entries"]), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
