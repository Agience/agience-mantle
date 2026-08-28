"""`oci_ingest.py` — the tool that makes the sovereign copy real, run the way the deploy runs it.

Why this exists: `oci.store.ingest_image` is built and tested with nothing to call it across the
gap between a workstation holding a layout and a box holding the CAS. `oci_ingest.py` is that
crossing, and reviewing or syntax-checking it alone would leave the crossing itself never run.

Run as a subprocess, with a real store, which is the only version that means anything: on the box
this is `docker run --rm … python /ingest/oci_ingest.py`, a fresh process whose store comes
entirely from `MANTLE_LATTICE_PATH` and `KEYS_DIR`. Monkeypatching `content_handle` in-process
would test the half that is never in doubt and skip the wiring that is — whether the tool can open
a store it is only told about through the environment.

Skipped when `agience-cloud` is not beside this repo, and that skip is named rather than silent:
mantle is legitimately checked out alone, while the sovereign plane materialises both, so this runs
where it matters.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mantle.shard.content import put_content
from mantle.shard.sqlite_store import FsContentStore

from ._oci_layout import make_layout

WORKSPACE = Path(__file__).resolve().parents[2]
TOOL = WORKSPACE / "agience-cloud" / "build" / "mantle" / "oci_ingest.py"

pytestmark = pytest.mark.skipif(
    not TOOL.is_file(),
    reason="agience-cloud is not checked out beside this repo, so its ingest tool is not on disk")


def _node(tmp: Path):
    """A store the way a node has one: a lattice path, a keys dir, and a minted content key.

    The key is minted by writing one byte through `put_content`, which is how the first key on any
    node comes into existence. Without it `content_handle` refuses — correctly, and this fixture
    exists partly to prove that refusal is reachable (see the last test).
    """
    store = tmp / "store"
    keys = store / "keys"
    keys.mkdir(parents=True, exist_ok=True)
    put_content(FsContentStore(str(store / "cas")), keys, b"seed")
    return store, keys


def _run(layout: Path, expect: str, store: Path, keys: Path):
    env = dict(os.environ)
    env["MANTLE_LATTICE_PATH"] = str(store / "lattice.db")
    env["KEYS_DIR"] = str(keys)
    return subprocess.run(
        [sys.executable, str(TOOL), str(layout), "--expect", expect],
        capture_output=True, text=True, env=env, timeout=300)


def test_a_layout_lands_in_a_real_nodes_store_and_verifies(tmp_path: Path):
    """The sovereign copy, end to end, against the store a node actually has.

    `put_content` must be able to write through a `TieredContentStore`, not only the
    caller-encrypts `FsContentStore` shape — a store that requires `collection` and gets none from
    `ingest_image` would make every OCI test pass while the real path stayed impossible.

    `verified: true` is printed only after the manifest and every blob it names have been re-read
    through the same path a puller uses and re-hashed. `mantle_ingest_layout` greps for that string
    rather than trusting the exit code, because a store that accepts a write and cannot return it
    would otherwise be recorded as a sovereign copy.
    """
    layout, manifest = make_layout(tmp_path / "layout")
    store, keys = _node(tmp_path)

    proc = _run(layout, manifest, store, keys)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)

    assert out["ok"] is True and out["verified"] is True
    assert out["digest"] == manifest
    assert out["blobs"] == 4 and out["written"] == 4 and out["already_held"] == 0


def test_a_second_run_writes_nothing_and_still_verifies(tmp_path: Path):
    """"0 written" is correct for an unchanged image and alarming for a changed one.
    `IngestedBlob.stored` is the only thing that tells those apart, so both counts are reported
    rather than a total — and a re-run must still verify, because idempotence that skipped the check
    would report a copy nobody looked at."""
    layout, manifest = make_layout(tmp_path / "layout")
    store, keys = _node(tmp_path)

    assert _run(layout, manifest, store, keys).returncode == 0
    out = json.loads(_run(layout, manifest, store, keys).stdout)

    assert out["written"] == 0 and out["already_held"] == 4
    assert out["verified"] is True


def test_the_blobs_are_readable_back_out_of_the_store(tmp_path: Path):
    """The claim `/v2` depends on: after an ingest, this node can serve the image. Read through
    `read_blob` — the same call `routers/oci_router` makes — rather than by looking at files, so it
    exercises decryption and the ref check, not just presence on disk."""
    layout, manifest = make_layout(tmp_path / "layout")
    store, keys = _node(tmp_path)
    assert _run(layout, manifest, store, keys).returncode == 0

    import os as _os
    _os.environ["MANTLE_LATTICE_PATH"] = str(store / "lattice.db")
    _os.environ["KEYS_DIR"] = str(keys)
    import importlib
    from mantle.db import backend as _backend
    importlib.reload(_backend)
    from mantle.oci.store import read_blob

    # The repository IS the collection, so the read names the scope the ingest wrote into.
    body = read_blob(_backend.content_handle(), str(keys), manifest,
                     collection="agience-mantle")
    import hashlib as _h
    assert "sha256:" + _h.sha256(body).hexdigest() == manifest


def test_a_digest_the_layout_does_not_contain_is_refused(tmp_path: Path):
    """`--expect` is required for this reason. A layout can hold several images, and ingesting
    "whichever one was first" would put a different artifact in the store than the deploy promotes —
    the two would then disagree about what this node holds, with nothing saying so."""
    layout, manifest = make_layout(tmp_path / "layout")
    store, keys = _node(tmp_path)

    proc = _run(layout, "sha256:" + "0" * 64, store, keys)
    assert proc.returncode == 2
    assert "does not contain" in proc.stderr
    # Both sides named: what was asked for and what is actually there.
    assert "0" * 64 in proc.stderr and manifest in proc.stderr


def test_the_verified_marker_is_absent_from_every_refusal(tmp_path: Path):
    """The shell greps stdout for `"verified": true`. If a refusal could print that string anywhere,
    `mantle_ingest_layout` would record a sovereign copy for a run that made none."""
    layout, _ = make_layout(tmp_path / "layout")
    store, keys = _node(tmp_path)

    proc = _run(layout, "sha256:" + "0" * 64, store, keys)
    assert '"verified": true' not in (proc.stdout + proc.stderr)


def test_a_node_with_no_content_key_refuses_instead_of_tracebacking(tmp_path: Path):
    """The provisioning fault, reported rather than raised.

    `content_handle` refuses loudly on a missing `content.key` because that is a silent partition:
    every blob would read as absent while being present and unreadable. The tool catches it so a
    deploy step reports a stated refusal instead of a stack trace — the cause is the same either way,
    the difference is whether an operator has to read a traceback to find it.
    """
    layout, manifest = make_layout(tmp_path / "layout")
    store = tmp_path / "bare-store"
    keys = store / "keys"
    keys.mkdir(parents=True, exist_ok=True)          # a keys dir with NO content.key in it

    proc = _run(layout, manifest, store, keys)
    assert proc.returncode == 2
    assert "content tier could not be opened" in proc.stderr
    assert "PROVISIONING fault" in proc.stderr
    assert "Traceback" not in proc.stderr
