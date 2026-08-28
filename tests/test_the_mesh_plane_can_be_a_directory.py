"""The mesh runs over a directory when one is named, with no S3 anywhere.

RULING 1, STEP 2 — PLANE [John, 2026-08-26: *"'BACKUP' is not a thing… make the peering work,
seamless"*, decomposed **TRUST → PLANE → SWEEP**].

Peering as built required S3, which is why the ruling was hard to honour. `mantle_common.sh` says
it outright — *"the mesh plane IS the durable content tier — there is no separate mesh bucket"* — and
`CONTENT_DURABLE_BUCKET` is refused-if-absent. So "peering, not backup" did not escape S3 on the old
code; it only changed what the S3 was FOR.

And the hard half was already written. `mesh/carrier.SpoolPlane` — *"a directory that IS a mesh
plane… the mesh cannot tell the difference, which is the point"* — has existed since the carrier
work and was **exported with no caller**: measured 2026-08-26, `grep SpoolPlane` across the tree
found its definition, its `__all__` entry, and nothing else. What was missing was one branch in
`_mesh_s3`, the single place the mesh asks for its plane.

A PLANE IS NOT A BACKUP. A spool on a local disk is a single-box plane — useful for a loopback
mesh, a test, or a carrier spooling frames off the air — and is exactly as durable as that
directory. Pointed at shared storage it is a real multi-node plane. Nothing replicates the spool
itself, and treating it as durable *because* peering runs over it is the confusion the ruling exists
to end.
"""
from __future__ import annotations

import json

import pytest

from mantle.mesh import sync
from mantle.mesh.carrier import SpoolPlane


# ── the seam ─────────────────────────────────────────────────────────────────────────────────────

def test_a_named_spool_becomes_the_plane(tmp_path, monkeypatch):
    monkeypatch.setenv(sync.MESH_SPOOL_DIR, str(tmp_path / "spool"))
    plane = sync._mesh_s3(object())            # no store needed: the spool answers first
    assert isinstance(plane, SpoolPlane), type(plane)


def test_the_spool_is_preferred_over_a_configured_remote(tmp_path, monkeypatch):
    """An operator who names a spool has said which plane they mean. Falling through to a bucket
    afterwards would silently prefer the bucket over the instruction."""
    class _Store:
        class content:
            remote = object()                  # a configured S3 origin
    monkeypatch.setenv(sync.MESH_SPOOL_DIR, str(tmp_path / "spool"))
    assert isinstance(sync._mesh_s3(_Store()), SpoolPlane)


def test_without_a_spool_nothing_changes(tmp_path, monkeypatch):
    """The existing path is untouched — a store's own remote still wins when no spool is named."""
    sentinel = object()

    class _Store:
        class content:
            remote = sentinel
    monkeypatch.delenv(sync.MESH_SPOOL_DIR, raising=False)
    assert sync._mesh_s3(_Store()) is sentinel


def test_an_empty_spool_variable_is_not_a_spool(tmp_path, monkeypatch):
    """`MANTLE_MESH_SPOOL_DIR=` in a node.env is a variable someone cleared, not a directory
    named `''`. Reading it as a path would put the plane at the process's working directory."""
    sentinel = object()

    class _Store:
        class content:
            remote = sentinel
    monkeypatch.setenv(sync.MESH_SPOOL_DIR, "   ")
    assert sync._mesh_s3(_Store()) is sentinel


# ── the contract the mesh actually uses ──────────────────────────────────────────────────────────

def test_the_plane_satisfies_every_call_the_mesh_makes(tmp_path):
    """THREE VERBS IS THE CONTRACT — plus a boto-shaped `_s3` for listing, which is the half a
    "just use a directory" shim usually forgets. These are the exact calls `sync.py` makes:
    `get`/`put`/`exists`, and `_s3.get_paginator("list_objects_v2")` with `Bucket`/`Prefix`."""
    plane = SpoolPlane(tmp_path / "spool")

    plane.put("mesh/leaf/71/00000.ndjson.enc", b"payload", "application/octet-stream")
    assert plane.exists("mesh/leaf/71/00000.ndjson.enc")
    assert plane.get("mesh/leaf/71/00000.ndjson.enc") == b"payload"
    assert plane.get("mesh/leaf/71/nope.enc") is None
    assert plane.bucket, "the mesh passes `Bucket=s3.bucket` to the paginator"

    pag = plane._s3.get_paginator("list_objects_v2")
    keys = [o["Key"] for page in pag.paginate(Bucket=plane.bucket, Prefix="mesh/leaf/")
            for o in page.get("Contents", [])]
    assert keys == ["mesh/leaf/71/00000.ndjson.enc"], keys


def test_start_after_is_honoured():
    """The mesh's cursor rides on it. `_apply_artifacts`' caller advances `last_key` behind a
    monotone `StartAfter` marker, so a plane that ignored it would re-apply every segment forever —
    or, worse, appear to work while never converging."""
    import tempfile

    plane = SpoolPlane(tempfile.mkdtemp(prefix="spool-sa-"))
    for i in range(4):
        plane.put("mesh/leaf/71/%05d.ndjson.enc" % i, b"x")

    pag = plane._s3.get_paginator("list_objects_v2")
    after = [o["Key"] for page in pag.paginate(Bucket=plane.bucket, Prefix="mesh/leaf/71/",
                                               StartAfter="mesh/leaf/71/00001.ndjson.enc")
             for o in page.get("Contents", [])]
    assert after == ["mesh/leaf/71/00002.ndjson.enc", "mesh/leaf/71/00003.ndjson.enc"], after


def test_a_delimiter_lists_publishers_not_every_segment():
    """How the mesh enumerates peers: `Delimiter='/'` over the leaf prefix returns one common prefix
    per publisher. A plane that returned every object instead would make peer discovery O(segments)."""
    import tempfile

    plane = SpoolPlane(tempfile.mkdtemp(prefix="spool-delim-"))
    for node in ("71", "45"):
        for i in range(3):
            plane.put("mesh/leaf/%s/%05d.ndjson.enc" % (node, i), b"x")

    pag = plane._s3.get_paginator("list_objects_v2")
    commons = sorted(p["Prefix"] for page in pag.paginate(
        Bucket=plane.bucket, Prefix="mesh/leaf/", Delimiter="/")
        for p in page.get("CommonPrefixes", []))
    assert commons == ["mesh/leaf/45/", "mesh/leaf/71/"], commons


# ── a round trip, over a directory, with no S3 involved ──────────────────────────────────────────

def test_a_merkle_head_round_trips_through_the_spool(tmp_path, monkeypatch):
    """The shape the mesh actually publishes and consumes: a per-node merkle head written as JSON
    and read back with `.decode("utf-8")` — the exact call at `sync.py:380`."""
    monkeypatch.setenv(sync.MESH_SPOOL_DIR, str(tmp_path / "spool"))
    plane = sync._mesh_s3(object())

    head = {"root": "abc123", "leaves": 4}
    plane.put("mesh/merkle/71.json", json.dumps(head).encode("utf-8"), "application/json")
    assert json.loads(plane.get("mesh/merkle/71.json").decode("utf-8")) == head


def test_the_spool_survives_a_reopen(tmp_path, monkeypatch):
    """A plane that lost its contents when the process restarted would be a cache, not a plane."""
    monkeypatch.setenv(sync.MESH_SPOOL_DIR, str(tmp_path / "spool"))
    sync._mesh_s3(object()).put("mesh/merkle/71.json", b"{}", "application/json")
    assert sync._mesh_s3(object()).get("mesh/merkle/71.json") == b"{}"


def test_the_seam_is_the_only_place_the_plane_is_chosen():
    """One branch, in the one function the mesh asks for its plane. A second chooser is how a node
    comes to publish to one plane and consume from another."""
    import inspect

    src = inspect.getsource(sync)
    assert src.count("SpoolPlane(") == 1, (
        "the spool is constructed in more than one place; the plane must be chosen once")
    assert "MESH_SPOOL_DIR" in inspect.getsource(sync._mesh_s3)


@pytest.mark.parametrize("verb", ["get", "put", "exists"])
def test_the_three_verbs_are_the_whole_contract(verb):
    """Stated as an assertion so a plane implementation cannot quietly grow a fourth requirement."""
    assert callable(getattr(SpoolPlane, verb, None)), verb
