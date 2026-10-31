"""Fields sealed under the node's own key: what the store must read for a caller who holds no grant
yet, or writes from a path that has no person's key.

The rest of the store seals under per-principal content keys (`db/doc_boundary`), opened only for a
grant that reaches them. A few fields cannot be: a grant's invite address is matched at acceptance
by a claimant who holds no grant yet, and grants are written by background paths (provisioning, the
platform's operators) that hold no person's key. These are sealed under the node's data key
(`prism.trust.key_manager.get_encryption_key`, the key platform settings already seal secrets
under). That protects them in the store's files and backups; it does not keep them from the node,
which is the open item "keys are held by the principals, not the node" (GAPS.md). Who may read them
through the API is decided by authorization, as before.

`keyed_token` is the equality form of the same key: a value that can be matched without being
stored, for a field the store looks rows up by (a relationship label).
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Iterable

SEALED = "node_sealed"
_TOKEN_PREFIX = "kt1:"


def _key() -> bytes:
    try:
        from prism.trust.key_manager import get_encryption_key
    except ImportError as e:
        raise ImportError(
            "sealing a grant's fields or a link's name needs the node's data key, from "
            "prism.trust.key_manager: install agience-prism. A store embedded without it keeps "
            "grants and links out of the store, or holds no named links.") from e
    return get_encryption_key().encode("utf-8")


def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(_key())


def seal_fields(doc: dict, fields: Iterable[str]) -> None:
    """Move `fields` of `doc` into `doc[SEALED]`, encrypted under the node's key, in place. A field
    absent or empty is left out; nothing to seal leaves `doc` as it is."""
    plain = {k: doc.pop(k) for k in list(fields) if doc.get(k) not in (None, "")}
    if not plain:
        return
    prior = doc.get(SEALED)
    if prior:
        plain = {**_open(prior), **plain}
    doc[SEALED] = _fernet().encrypt(json.dumps(plain).encode("utf-8")).decode("ascii")


def _open(blob: str) -> dict:
    value = json.loads(_fernet().decrypt(blob.encode("ascii")).decode("utf-8"))
    return value if isinstance(value, dict) else {}


def unseal_fields(doc: dict) -> None:
    """Restore the fields `seal_fields` moved, in place. A field already present is kept."""
    blob = doc.pop(SEALED, None)
    if not blob:
        return
    try:
        opened = _open(blob)
    except Exception:  # noqa: BLE001 — sealed under another node's key (a copied or restored store)
        # Its fields stay absent; the grant's structure, which authorization reads, is untouched.
        # Raising here failed every lookup for the grantee, not only this row.
        doc[SEALED] = blob
        return
    for k, v in opened.items():
        if doc.get(k) in (None, ""):
            doc[k] = v


def keyed_token(purpose: str, value: str) -> str:
    """A deterministic token for `value` under the node's key, separated by `purpose`: equal values
    give equal tokens, and the value cannot be read back from one."""
    sub = hmac.new(_key(), ("agience.keyed-token." + purpose).encode("utf-8"), hashlib.sha256).digest()
    return _TOKEN_PREFIX + hmac.new(sub, value.encode("utf-8"), hashlib.sha256).hexdigest()


def is_token(value: str) -> bool:
    return isinstance(value, str) and value.startswith(_TOKEN_PREFIX)


def seal_value(value: str) -> str:
    """One string sealed under the node's key."""
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def open_value(blob: str) -> str:
    """The string `seal_value` sealed."""
    return _fernet().decrypt(blob.encode("ascii")).decode("utf-8")
