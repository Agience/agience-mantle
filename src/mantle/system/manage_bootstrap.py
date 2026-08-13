#!/usr/bin/env python3
"""
manage_bootstrap.py -- bring a new, empty store to the point where it can be used.

    python -m mantle.system.manage_bootstrap --principal <uuid> [--name "..."] [--dry-run]

Three steps, and nothing else:

    1. the authority collection      the resource platform-admin roots to
    2. the first user                person card + inbox workspace + owner grant
    3. the admin grant on (1)        which closes the bootstrap window

!! THIS IS NOT REQUIRED TO STORE AND FIND THINGS. An ordinary write reaches a virgin store on its
own: `POST /artifacts` (and the `create_artifact` MCP tool) roots the content key at the new doc's
own `origin_root` and the write path issues the creator's owner grant in the same call, so
`oracle.get_or_create_master_key` finds the `read` grant it requires and the artifact is created,
indexed and recallable with nothing bootstrapped. MEASURED against a fresh lattice: 201, then a
`recall` that returns it. This docstring previously claimed the opposite — that the first write
fails with `ContentEncryptionError: content encryption unavailable; refusing to persist plaintext`
— and that is not what happens.

What DOES require this tool is the platform-admin surface: `/system/*` roots admin at the authority
collection, and `grants_router._require_admin` checks `grant_service.can_admin(db, user,
resource_id)` — a grant on the resource, which the operator fast-path does not satisfy — so there
is no API by which the first such grant can be created.

!! IT DOES NOT INITIALIZE KEY MATERIAL, so what it writes is not searchable. `main.py`'s lifespan
calls `init_encryption_key()` at Phase 0; this script does not, so the key oracle has no KEK and
every artifact below is logged `sse=skipped vector=skipped` and stays that way until a reindex.
The exit code is 0 regardless.

The two admin paths disagree by design. `/system/*` honours the bootstrap window
(`dependencies._authority_bootstrap_complete`); grant management does not. This script closes the
gap with a deliberate, one-time action rather than by widening an authorization path permanently.

This is not seeding. `manage_seed.py` and `POST /system/seed` load the install package's corpus
from `AGIENCE_SEEDS_ROOT` — which, in `agience-bundle`, is the Agience persona roster
(`agent-aria`, `agent-astra`, `agent-sage`, `agent-seraph`, `agent-verso`) plus grants making every
new user able to read and invoke it. An enterprise store wants none of that: installing it would
seed someone else's tenant with Agience agents. Nothing here reads `AGIENCE_SEEDS_ROOT`.

It closes itself. `dependencies._authority_bootstrap_complete` reports True once an active
`can_admin`/`can_update` grant exists on the authority collection, and step 3 creates exactly that.
The moment this finishes, the window shuts, `AGIENCE_OPERATOR_ID` stops being honoured, and every
later request goes through ordinary grant checks.

Idempotent: every underlying call is create-if-missing or upsert, so re-running is safe. It reports
"already bootstrapped" rather than pretending to have done work.

Run it where the store is. This opens the lattice directly (`MANTLE_LATTICE_PATH`), so run it on
the node — in the container, or anywhere that env and `KEYS_DIR` resolve to the real ones. Writing
to the wrong store is silent; the path it opened is logged before anything is written.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger("mantle.bootstrap")


def connect():
    """Open THE store — the standalone lattice (MANTLE_LATTICE_PATH / MANTLE_ORIGIN env)."""
    from mantle.db import backend
    return backend.store_handle()


def _admin_grant_exists(db, authority_id: str) -> bool:
    """True when the authority collection already carries an active admin grant.

    Same question `dependencies._authority_bootstrap_complete` asks, asked the same way — if these
    two ever disagree, the bootstrap window's state and this tool's idea of it diverge, and one of
    them is lying about whether the store is bootstrapped.
    """
    from mantle.db import backend as db_store
    from mantle.entities.grant import grant_is_allow
    try:
        grants = db_store.get_grants_for_collection(db, authority_id)
    except Exception:
        return False
    return any(
        getattr(g, "state", "active") == "active"
        and grant_is_allow(g)
        and (getattr(g, "can_admin", False) or getattr(g, "can_update", False))
        for g in grants
    )


def bootstrap(db, principal: str, *, name: str, dry_run: bool) -> int:
    from mantle.services.seed_provisioning.user_provisioning import (
        ensure_authority_collection, provision_user,
    )
    from mantle.db import backend as db_store

    # ── 1. the authority collection ──────────────────────────────────────────────────────────
    # Deterministic id from the instance namespace, create-if-missing. This is the resource a
    # platform admin roots to; it must exist whether or not any corpus was ever loaded.
    authority_id = ensure_authority_collection(db)
    print(f"  authority collection : {authority_id}")

    if _admin_grant_exists(db, authority_id):
        print(f"  ALREADY BOOTSTRAPPED — an active admin grant exists on {authority_id}.")
        print("  Nothing to do. Manage further admins through POST /system/users/{id}/grant-admin.")
        return 0

    if dry_run:
        print("\n  --dry-run: would provision the user and issue the admin grant, then stop.")
        print(f"  principal : {principal}")
        return 0

    # ── 2. the first user ────────────────────────────────────────────────────────────────────
    # Person card + inbox workspace; `create_workspace` issues the owner grant, which is what gives
    # this principal any reachable resource at all.
    provision_user(db, user_id=principal, name=name)
    print(f"  provisioned user     : {principal}")

    # ── 3. the admin grant — this closes the window ──────────────────────────────────────────
    # `granted_by` is the principal itself: at this instant there is no other identity that could
    # have granted it, and recording a platform placeholder would assert an authority that never
    # acted. The grant is self-granted, once, and the record says so.
    db_store.upsert_user_collection_grant(
        db,
        user_id=principal,
        collection_id=authority_id,
        granted_by=principal,
        can_read=True,
        can_update=True,
        can_admin=True,
        name="Platform admin (bootstrap)",
    )
    print(f"  admin grant issued   : {principal} -> {authority_id}")

    # Assert the thing this tool exists to achieve, rather than assuming the writes landed.
    if not _admin_grant_exists(db, authority_id):
        print("\n  FAILED: the admin grant is not readable back — the store is NOT bootstrapped.",
              file=sys.stderr)
        return 1

    print("\n  Bootstrap window is now CLOSED.")
    print("  AGIENCE_OPERATOR_ID is no longer honoured; remove it from the deployment.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap a new, empty mantle store. Not seeding — see the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--principal", required=True,
        help="The first admin's principal id. For an external IdP this is "
             "uuid5(namespace, (issuer, sub)) — the id mantle derives from the token, which is "
             "deterministic, so it can be computed ahead of time rather than captured at runtime.",
    )
    parser.add_argument("--name", default="Platform administrator",
                        help="Display name for the person card.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would happen; write nothing.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Say which store, before touching it. An unset MANTLE_LATTICE_PATH resolves to a relative
    # default and would bootstrap a store created on the spot in the working directory — which then
    # looks perfectly healthy and is not the one anybody meant.
    path = os.getenv("MANTLE_LATTICE_PATH")
    print(f"\n  store: {os.path.abspath(path) if path else '<MANTLE_LATTICE_PATH UNSET — refusing>'}")
    if not path:
        print("\n  Set MANTLE_LATTICE_PATH. Unset, mantle opens (or CREATES) a relative default,\n"
              "  and this tool would bootstrap a store nobody asked for.", file=sys.stderr)
        return 2

    db = connect()

    from mantle.services.platform_settings_service import settings as platform_settings
    platform_settings.load_all(db)

    # No request context, so the acting identity is declared explicitly — every write below goes
    # through the grant ledger, and content encryption needs a principal it can check.
    from mantle.services.system_identity import system_acting_context
    with system_acting_context(scope="platform.bootstrap"):
        return bootstrap(db, args.principal, name=args.name, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
