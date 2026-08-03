"""`mantle_cas_rekey` must work for a consumer that names its OWN collection content type.

⛔ THE DEFECT THIS PINS, AND WHY NO IN-HOUSE FIXTURE WOULD HAVE FOUND IT.

`_legacy_roots()` harvested origin_roots from ONE hardcoded content type
(`application/vnd.agience.collection+json`). Every Mantle-authored store uses that type, so every
test we would naturally write passed. **A consumer that names its own collection type got an empty
key map** — and an empty map means every object fails both the shared key and the (empty) legacy
set, so all of them were classified `unreadable` and reported as "needs a re-fetch from the durable
tier". On a `remote=None` node that reads as total data loss.

MEASURED by EREA (2026-07-29), the first external consumer, on a reconstructed corpus whose
collections are `application/vnd.erea.project+json`: **487 of 487 objects unreadable**, 0 keys
derived, while the same blobs decrypt 487/487 under `collection_key(secret, origin_root)`. The
crypto was right; discovery was wrong. A false alarm that could have provoked a destructive recovery.

The fix reads what the derivation actually CONSUMES — `origin_root`, which is a real indexed column
— so the content type stops mattering. These tests use a deliberately foreign content type
throughout, because that is the case the in-house fixture cannot produce.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

from mantle.db.lattice.content_cache import (collection_key,          # noqa: E402
                                             shared_content_key)
from mantle.scripts.mantle_cas_rekey import _legacy_roots, main       # noqa: E402

# Deliberately NOT `application/vnd.agience.collection+json`.
FOREIGN_CT = "application/vnd.erea.project+json"


def _store(tmp_path, *, roots, with_column=True, ct=FOREIGN_CT):
    """A minimal lattice-shaped store carrying `roots` under a FOREIGN collection type."""
    db = str(tmp_path / "l.db")
    con = sqlite3.connect(db)
    cols = "id TEXT PRIMARY KEY, ct TEXT, doc TEXT" + (", origin_root TEXT" if with_column else "")
    con.execute("CREATE TABLE vertex (%s)" % cols)
    for i, r in enumerate(roots):
        doc = json.dumps({"origin_root": r})
        if with_column:
            con.execute("INSERT INTO vertex VALUES (?,?,?,?)", ("c%d" % i, ct, doc, r))
        else:
            con.execute("INSERT INTO vertex VALUES (?,?,?)", ("c%d" % i, ct, doc))
    con.commit()
    con.close()
    return db


def _write_legacy(cas, ref, plain, origin_root, root_secret):
    """Write one object exactly as the pre-§1 cache did."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    h = ref[len("cas/"):]
    p = os.path.join(cas, h[:2], h[2:4], h)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    nonce = os.urandom(12)
    key = collection_key(root_secret, origin_root)
    with open(p, "wb") as fh:
        fh.write(nonce + AESGCM(key).encrypt(nonce, plain, ref.encode("utf-8")))
    return p


def _keys_dir(tmp_path):
    """`(keys_dir, root_secret)` — the secret derived EXACTLY as the script derives it, so the test
    exercises the real key path instead of monkeypatching around it."""
    from cryptography.fernet import Fernet
    d = tmp_path / "keys"
    d.mkdir(exist_ok=True)
    kb = Fernet.generate_key()
    (d / "content.key").write_bytes(kb)
    return str(d), hashlib.blake2b(kb.strip(), digest_size=32).digest()


# ── discovery ────────────────────────────────────────────────────────────────
def test_roots_are_found_under_a_foreign_collection_content_type(tmp_path):
    """The regression itself. Content type must not gate discovery."""
    db = _store(tmp_path, roots=["root-a", "root-b", "root-c"])
    assert _legacy_roots(db) == {"root-a", "root-b", "root-c"}


def test_roots_are_found_when_the_column_is_absent(tmp_path):
    """`origin_root` is a MIGRATED column — `schema._VERTEX_ADDED_COLUMNS` adds it to stores that
    predate it, and node 45 measured stores lacking it entirely. Falling back to the doc keeps
    those readable instead of failing with `no such column`."""
    db = _store(tmp_path, roots=["root-a", "root-b"], with_column=False,
                ct="application/vnd.agience.collection+json")
    assert _legacy_roots(db) == {"root-a", "root-b"}


def test_a_root_only_in_the_column_is_still_found(tmp_path):
    """The column is the primary source: it covers roots whose collection artifact was archived,
    is of an unknown type, or never existed at all."""
    db = _store(tmp_path, roots=["root-a"])
    con = sqlite3.connect(db)
    con.execute("INSERT INTO vertex VALUES (?,?,?,?)",
                ("orphan", "text/markdown", json.dumps({}), "root-orphan"))
    con.commit()
    con.close()
    assert _legacy_roots(db) == {"root-a", "root-orphan"}


def test_explicit_escape_hatches_are_honoured(tmp_path):
    db = _store(tmp_path, roots=["root-a"])
    assert "root-manual" in _legacy_roots(db, extra_roots=("root-manual",))


# ── the end-to-end case EREA hit ─────────────────────────────────────────────
def test_a_foreign_typed_corpus_migrates_instead_of_reporting_total_loss(tmp_path, capsys):
    cas = str(tmp_path / "cas")
    os.makedirs(cas)
    db = _store(tmp_path, roots=["root-a", "root-b"])
    keys, secret = _keys_dir(tmp_path)

    for i, root in enumerate(["root-a", "root-a", "root-b"]):
        plain = b"legacy object %d" % i
        ref = "cas/" + hashlib.sha256(plain).hexdigest()
        _write_legacy(cas, ref, plain, root, secret)

    rc = main(["--cas", cas, "--keys-dir", keys, "--db", db, "--dry-run"])
    out = capsys.readouterr().out
    assert "2 legacy origin_root key(s)" in out, out
    assert "rekeyed         3" in out, out
    assert "unreadable      0" in out, out
    assert rc == 0


def test_no_discovered_roots_refuses_instead_of_crying_data_loss(tmp_path, capsys):
    """⛔ THE WORSE HALF OF THE DEFECT. With zero keys, every object fails — and saying "needs a
    re-fetch from the durable tier" asserts a cause the run cannot distinguish. A key you never had
    fails byte-identically to a destroyed object. Refusing is the difference between "I could not
    look" and "it is gone"."""
    cas = str(tmp_path / "cas")
    os.makedirs(cas)
    plain = b"an object whose root is undiscoverable"
    ref = "cas/" + hashlib.sha256(plain).hexdigest()
    keys, secret = _keys_dir(tmp_path)
    _write_legacy(cas, ref, plain, "root-hidden", secret)

    db = str(tmp_path / "empty.db")                     # a store with NO origin_roots at all
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE vertex (id TEXT PRIMARY KEY, ct TEXT, doc TEXT, origin_root TEXT)")
    con.commit()
    con.close()

    rc = main(["--cas", cas, "--keys-dir", keys, "--db", db, "--dry-run"])
    cap = capsys.readouterr()
    combined = cap.out + cap.err
    assert rc == 2
    assert "NO legacy origin_root was discovered" in combined, combined
    assert "REFUSING to classify" in combined, combined
    assert "unreadable" not in cap.out.lower() or "REFUSING" in combined
    assert os.path.exists(os.path.join(cas, ref[4:6], ref[6:8], ref[4:])), "the object was touched"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
