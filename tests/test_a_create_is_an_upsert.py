"""In this store a "create" is an UPSERT, and the code says so in one line.

`lattice_api.create_collection`, `lattice_api.update_collection` and
`lattice_api.create_artifact` all reduce to the same call — `db.artifacts.put_artifact(...)`.
There is no create-if-absent anywhere on this path. Writing an id that already exists REPLACES
the row and returns normally.

Measured 2026-08-26, and it is not what the schema suggests: `vertex.id` is `TEXT PRIMARY KEY`
and the insert in `db/vertex.py` carries no `OR REPLACE`, so reading the DDL predicts an
`IntegrityError` that never arrives.

**This is defensible for a replicated store** — the same artifact arriving twice from two peers
must converge, not fail — so this file PINS the behaviour rather than reporting it as a bug. What
it exists to prevent is the two ways the assumption bites:

1. **Nothing may build create-then-catch on a duplicate-id failure.** It does not come. The fix
   for the first-login race (`test_first_login_provisioning_is_concurrency_safe.py`) relies on
   convergence-by-upsert instead, and that reasoning is only sound while this holds.
2. **Any caller treating `create_*` as "create if absent" silently clobbers.** Safe today only
   because every id is either minted (`uuid4`) or derived from the ACTING principal — no route
   lets a caller name an arbitrary id.

If someone later makes these raise on an existing id, that is a legitimate design change — and
this file is where it announces itself, next to the two call sites that assume otherwise.
"""
from __future__ import annotations

import io

import pytest

from mantle.db import lattice_api
from mantle.entities.artifact import Artifact as ArtifactEntity
from mantle.entities.artifact import WORKSPACE_CONTENT_TYPE
from mantle.services import workspace_service


@pytest.fixture
def db(tmp_path):
    return lattice_api.LatticeDatabase(str(tmp_path / "upsert.db"), origin="upsert-test")


PINNED = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def test_a_second_container_create_on_the_same_id_replaces_rather_than_raises(db):
    workspace_service.create_container(
        db, "owner", content_type=WORKSPACE_CONTENT_TYPE, name="first", artifact_id=PINNED)
    workspace_service.create_container(
        db, "owner", content_type=WORKSPACE_CONTENT_TYPE, name="second", artifact_id=PINNED)

    rows = lattice_api.get_collections_by_owner_and_type(db, "owner", WORKSPACE_CONTENT_TYPE)
    assert len(rows) == 1, "a duplicate id produced %d rows" % len(rows)
    assert rows[0].name == "second", "the upsert is not last-writer-wins"


def test_a_second_artifact_create_on_the_same_id_replaces_rather_than_raises(db):
    container = workspace_service.create_container(
        db, "owner", content_type=WORKSPACE_CONTENT_TYPE, name="home").id

    def put(name):
        lattice_api.create_artifact(db, ArtifactEntity(
            id=PINNED, root_id=PINNED, collection_id=container,
            state=ArtifactEntity.STATE_COMMITTED, created_by="owner", name=name))

    put("first")
    put("second")           # would be an IntegrityError if the PK were the arbiter
    assert lattice_api.get_artifact(db, PINNED).name == "second"


def test_create_and_update_are_the_same_call():
    """The source-level statement of it, so the prose above cannot drift from the code.

    Parses the FILE, not the live attributes. `inspect.getsource` on
    `lattice_api.create_collection` reads whatever is bound to that name at call time, and other
    tests in this suite patch these functions — which is how the same shortcut passed alone and
    failed in the full suite earlier. The file is the fact; the attribute is a binding.
    """
    import ast

    tree = ast.parse(io.open(lattice_api.__file__, encoding="utf-8").read())
    calls = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
                "create_collection", "update_collection", "create_artifact"):
            calls[node.name] = sorted(
                sub.func.attr for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute))

    assert len(calls) == 3, "expected all three definitions, found %s" % sorted(calls)
    for name, attrs in calls.items():
        assert "put_artifact" in attrs, "%s no longer calls put_artifact: %s" % (name, attrs)
