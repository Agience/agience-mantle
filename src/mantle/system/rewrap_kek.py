"""Rotate the KEK (the node's `encryption.key`): re-wrap and re-seal what it holds, then scrub.

WHY. John, 2026-11-02 (review 10): every key from the era when services ran as root rotates at
cutover, the KEK among them, since a root process could read it. The KEK wraps every stored key and
seals the node's own fields, so replacing the file alone would leave all of them unreadable.

What it re-wraps or re-seals, each opened under the old KEK and written under the new:
  * every stored key: a whole master or collection key (`master-key:*`, `token`), and this node's
    share of each split key (`share`);
  * every row carrying node-sealed fields (`node_sealed`: a grant's name, notes and invite address);
  * every link name: its sealed name, and the keyed tokens the store finds it by (`edge.label`
    and one per dotted prefix), recomputed under the new key;
  * every secret platform setting.
Each new wrap is opened again under the new KEK and compared before the row is written. A row
already under the new KEK is counted and left; one that opens under neither is left and reported.
Then the store file is scrubbed (`db.scrub`), so no old wrap survives in it.

The release step that runs this stops Mantle first, then puts the new key file in place only after
this reports nothing failed (`agience-cloud`'s `rotate-keys`).

    python -m mantle.system.rewrap_kek --new-kek-file PATH --dry-run
    python -m mantle.system.rewrap_kek --new-kek-file PATH
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from typing import Iterator, Optional

logger = logging.getLogger("rewrap_kek")

_KEY_FIELDS = ("token", "share")


@contextlib.contextmanager
def _under(key: str) -> Iterator[None]:
    """Run a block with the node's data key set to `key` (`prism.trust.key_manager`)."""
    import prism.trust.key_manager as km
    prior = km._encryption_key
    km._encryption_key = key
    try:
        yield
    finally:
        km._encryption_key = prior


def _clean(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def _rewrap_value(old, new, value: str) -> tuple:
    """`(outcome, rewrapped)`: outcome is `rewrapped`, `already` or `failed`."""
    try:
        plain = old.decrypt(value.encode("ascii"))
    except Exception:  # noqa: BLE001 — not under the old KEK
        try:
            new.decrypt(value.encode("ascii"))
            return "already", value
        except Exception:  # noqa: BLE001
            return "failed", value
    wrapped = new.encrypt(plain).decode("ascii")
    if new.decrypt(wrapped.encode("ascii")) != plain:
        return "failed", value
    return "rewrapped", wrapped


def rewrap_kek(db, *, new_key: str, dry_run: bool = False) -> dict:
    """Re-wrap and re-seal everything the current KEK holds under `new_key`. Returns the counts."""
    from cryptography.fernet import Fernet

    from mantle.db import lattice_api as la
    from mantle.db import node_seal
    from mantle.db.lattice_identity import _SETTINGS
    from mantle.search.mantle.oracle import SHARE_CONTENT_TYPE, LatticeMasterKeyStore
    from mantle.system.seal_titles import _named_edges, _relabel
    from prism.trust.key_manager import get_encryption_key

    old_key = get_encryption_key()
    new_key = (new_key or "").strip()
    if not new_key or new_key == old_key:
        raise ValueError("the new KEK must be a Fernet key other than the current one")
    old, new = Fernet(old_key.encode()), Fernet(new_key.encode())
    counts = {"keys": 0, "already": 0, "sealed_rows": 0, "settings": 0, "labels": 0,
              "failed": 0}

    def count(outcome: str, field: str) -> None:
        if outcome == "rewrapped":
            counts[field] += 1
        elif outcome == "already":
            counts["already"] += 1
        else:
            counts["failed"] += 1

    # 1. Every stored key.
    for ct in (LatticeMasterKeyStore.CONTENT_TYPE, SHARE_CONTENT_TYPE):
        for doc in db.artifacts.list_artifacts(content_type=ct):
            if not str(doc.get("id", "")).startswith("master-key:"):
                continue
            field = next((f for f in _KEY_FIELDS if doc.get(f)), None)
            if field is None:
                continue
            outcome, wrapped = _rewrap_value(old, new, doc[field])
            count(outcome, "keys")
            if outcome == "rewrapped" and not dry_run:
                db.artifacts.put_artifact({**_clean(doc), field: wrapped})

    # 2. Every row with node-sealed fields, and 3. every secret platform setting.
    for doc in list(db.artifacts.list_artifacts()):
        blob = doc.get(node_seal.SEALED)
        if blob:
            outcome, sealed = _rewrap_value(old, new, blob)
            count(outcome, "sealed_rows")
            if outcome == "rewrapped" and not dry_run:
                db.artifacts.put_artifact({**_clean(doc), node_seal.SEALED: sealed})
    for s in list(_SETTINGS.all(db)):
        if not s.get("is_secret") or not s.get("value"):
            continue
        outcome, value = _rewrap_value(old, new, str(s["value"]))
        count(outcome, "settings")
        if outcome == "rewrapped" and not dry_run:
            _SETTINGS.update(db, s.get("key") or s.get("_key"), {"value": value})

    # 4. Every link name, and the keyed tokens it is found by, under the new key.
    from mantle.db import reach_journal
    for src, dst, label, props in _named_edges(db, tokens=True):
        sealed = props.get("label_sealed")
        if not sealed:
            continue
        try:
            name = old.decrypt(sealed.encode("ascii")).decode("utf-8")
        except Exception:  # noqa: BLE001
            try:
                new.decrypt(sealed.encode("ascii"))
                counts["already"] += 1
            except Exception:  # noqa: BLE001
                counts["failed"] += 1
            continue
        counts["labels"] += 1
        if dry_run:
            continue
        with _under(new_key):
            new_label = la._stored_label(name)
            fresh = {k: v for k, v in props.items() if k not in ("label_sealed", "label_prefixes")}
            fresh.update(la._label_props(name))
        with reach_journal.suspended():          # a link's name is not reach
            _relabel(db, src, dst, label, new_label, fresh)
    if counts["labels"] and not dry_run:
        with db.graph.db.write() as cur:
            cur.execute("DELETE FROM counter WHERE name LIKE 'edge:label:%' AND n <= 0")

    if not dry_run and any(counts[k] for k in ("keys", "sealed_rows", "settings", "labels")):
        from mantle.db.scrub import scrub_freed
        scrub_freed(db)
    return counts


def main(argv: Optional[list] = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")             # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-kek-file", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    from prism.trust.key_manager import init_encryption_key
    init_encryption_key()
    from mantle.services.dependencies import get_store_db

    db = next(get_store_db())
    with open(args.new_kek_file, encoding="ascii") as fh:
        counts = rewrap_kek(db, new_key=fh.read(), dry_run=args.dry_run)
    print(json.dumps(counts))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
