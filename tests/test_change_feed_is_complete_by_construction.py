"""Every write to the artifact substrate is accounted for on the change feed, or named as a gap.

"CRUD coverage is complete by construction" is a claim about the *shape* of the write path: every
create and update funnels through one store adapter, that adapter calls
`db.doc_boundary.emit_artifact_change`, and therefore no create or update can be silent without
someone deleting a line from a function they had to edit anyway. Prose cannot hold that; a scan of
the tree can.

Two detectors, one table (Section 1). Section 2 fixes the enumeration against the tree, so a new
write site must be classified before the suite is green again. Section 3 checks the classifications
that claim an emit actually have one. Section 4 pins the set of *known-silent* artifact-plane
writes, so the gap set can shrink by deleting an entry but cannot grow without adding one — the
property a coverage claim actually needs, since a claim that only fails when everything is broken
never fails. Section 5 exercises the feed end to end: a real store, a real subscriber, and one
event per write — one case per write site, because the scan proves a call exists and only a
subscriber holding the event proves the event arrives.

Guard style follows `tests/test_rounding_law_is_single_sourced.py` — AST rather than grep (this
file and several docstrings discuss `put_artifact` in prose), an annotated allow-table rather than
a bare count, and a control that shows the scanner can speak.
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

BACKEND = pathlib.Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SRC = BACKEND / "src" / "mantle"

from mantle.events import event_bus                        # noqa: E402
from mantle.db import doc_boundary                  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 1 · The detectors and the table
# ═════════════════════════════════════════════════════════════════════════════════════════════

#: Substrate write methods — the actual chokepoint. `put_many` is included because a bulk write is
#: still a write; a replication path that used it to bypass the feed would be invisible otherwise.
_SUBSTRATE_WRITES = {"put_artifact", "delete_artifact", "put_many", "add_edge", "delete_edge"}

#: Store-module deletes. A delete cannot emit at the boundary (see `emit_artifact_change`'s
#: docstring: the doc is gone and the container is no longer readable from an id), so the emit is
#: the caller's. These are the call sites where that caller is visible.
_API_DELETES = {"delete_artifact", "delete_collection", "delete_artifacts_by_root"}
_STORE_MODULE_ALIASES = {"store", "db_store", "backend", "lattice_api"}

#: The emit helpers. Any of them in the same function counts as that function announcing its write.
_EMITTERS = {"emit_artifact_change", "emit_artifact_event_sync", "publish_event_sync",
             "publish_event", "_emit", "_emit_event"}

# -- verdicts ---------------------------------------------------------------------------------

#: Emits at the write itself, through `doc_boundary.emit_artifact_change`. This is the "by
#: construction" half — the write and the announcement are the same few lines. Usually
#: `artifact.created` / `artifact.updated`; a site that still holds the container while it writes
#: (a soft delete, a curation move) announces its own delete here too, because the reason deletes
#: normally move up a layer — the container is unreadable from a bare id — does not apply to it.
BOUNDARY = "boundary"
#: Silent here on purpose; the emit belongs to the caller, which holds context this layer lost.
SERVICE = "service"
#: Deliberately off the artifact change feed. A typed side-plane, a replication apply, or index
#: bookkeeping — none of them is a user-visible artifact change, and putting them on the feed would
#: make every subscriber filter them back out.
OFF_FEED = "off_feed"
#: An artifact-plane write that announces nothing. A real hole in the feed, enumerated so it is a
#: known quantity rather than a discovery.
GAP = "gap"

#: Every site that writes the artifact substrate, and what it does about the change feed.
#:
#: The annotation is the point, not the membership. A new entry means deciding which of the four
#: this write is, which is the decision that was being skipped when the coverage claim lived only
#: in a docstring.
WRITE_SITES: dict[tuple[str, str], tuple[str, str]] = {
    # ── the artifact plane: creates and updates, announced where they happen ──────────────
    ("db/lattice_api.py", "create_artifact"): (
        BOUNDARY, "the artifact create. Emits artifact.created."),
    ("db/lattice_api.py", "update_artifact"): (
        BOUNDARY, "the artifact update. Emits artifact.updated."),
    ("db/lattice_api.py", "create_collection"): (
        BOUNDARY, "a container create IS an artifact create — same chokepoint, same event."),
    ("db/lattice_api.py", "update_collection"): (
        BOUNDARY, "a container update IS an artifact update — the rename/description edit that a "
                  "live tree is likeliest to be showing stale."),
    ("db/lattice_api.py", "batch_commit_drafts"): (
        BOUNDARY, "writes each doc directly rather than through update_artifact, so it emits one "
                  "artifact.updated per document — a subscriber filters per artifact, and a "
                  "batch-shaped event would reach nobody watching one artifact in the batch."),
    ("db/lattice_api.py", "archive_artifact"): (
        BOUNDARY, "soft delete. Emits artifact.deleted, not updated: every container read filters "
                  "archived versions out, so from a subscriber's side the artifact has left. The "
                  "container is on the doc being archived, so unlike a hard delete this one can "
                  "be announced at the write."),

    # ── the artifact plane: deletes, announced by the caller that still has the container ──
    ("db/lattice_api.py", "delete_artifact"): (
        SERVICE, "hard-deletes one version doc. The container the event must be addressed to is "
                 "gone with the doc, so workspace_service.delete_artifact emits it."),
    ("db/lattice_api.py", "delete_collection"): (
        SERVICE, "same shape as delete_artifact, for a container."),
    ("db/lattice_api.py", "delete_artifacts_by_root"): (
        SERVICE, "whole-lineage delete behind revert / remove; the service emits one "
                 "artifact.deleted for the root rather than one per version."),
    ("services/workspace_service.py", "delete_artifact"): (
        SERVICE, "emits artifact.deleted with the container the db layer no longer has."),
    ("services/workspace_service.py", "remove_artifact_from_container"): (
        SERVICE, "detach; emits artifact.deleted against the container it was removed from."),
    ("services/workspace_service.py", "revert_artifact"): (
        SERVICE, "drops the draft and emits artifact.updated for the version now current."),
    ("services/workspace_service.py", "delete_workspace"): (
        SERVICE, "emits one artifact.deleted per root and then one for the workspace itself. Per "
                 "root and not just for the container, because a subscriber told only that the "
                 "container went cannot name the artifact ids it must drop."),
    ("services/workspace_service.py", "_delete_or_detach_members"): (
        SERVICE, "shared by delete_workspace and delete_artifact for what happens to a "
                 "container's members: the cascade=True branch's hard-delete of a member with "
                 "nowhere else to be reached from. Emits artifact.deleted per member either way "
                 "-- destroyed under cascade, merely detached by default -- because a watcher of "
                 "the container sees the same thing: the artifact is no longer in it."),

    # ── typed side planes: their own vocabulary, not the artifact feed ─────────────────────
    ("db/lattice_api.py", "create_grant"): (
        OFF_FEED, "the grant plane. grant_service emits the grant.* vocabulary; a grant surfacing "
                  "as artifact.created would put authorization changes on a feed subscribers "
                  "filter for content changes."),
    ("db/lattice_api.py", "update_grant"): (OFF_FEED, "the grant plane — see create_grant."),
    ("db/lattice_api.py", "_put_typed"): (
        OFF_FEED, "commits and commit items. A provenance plane, discriminated by content_type; "
                  "neither is an artifact change."),
    ("db/lattice_api.py", "mark_materialized"): (
        OFF_FEED, "index bookkeeping — a marker recording that a version was sent for indexing. "
                  "Feeding it back into the feed that triggered the indexing would loop."),
    ("db/lattice_identity.py", "put"): (
        OFF_FEED, "the identity planes (person records, platform settings, passkeys, OTP codes). "
                  "Personal and credential data; a change feed carrying them would widen exposure "
                  "to anyone holding a wildcard subscription."),
    ("db/lattice_identity.py", "update"): (OFF_FEED, "identity planes — see put."),
    ("db/lattice_identity.py", "delete"): (OFF_FEED, "identity planes — see put."),
    ("search/mantle/oracle.py", "put"): (
        OFF_FEED, "encrypted-search cell writes. Derived from artifacts already on the feed; "
                  "announcing the derivation as well would double every indexed change."),
    ("search/anchors/repo.py", "clear"): (
        OFF_FEED, "removes the anchors of a superseded AnchorSet. Anchors are the COORDINATE "
                  "SYSTEM, not content — an anchor's id is a cluster id naming a cell path, an "
                  "HKDF info and a mesh region, so replacing a set is one geometry change, not N "
                  "artifact deletions. It is already announced where it means something: the "
                  "AnchorSet fingerprint changes, `/status` reports `matches_cells: false`, and "
                  "every cell must be rewritten before search answers again. A subscriber given "
                  "16 delete rows would learn less than the fingerprint tells it, and would have "
                  "to reconstruct the one fact that matters from the pieces."),

    # ── membership edges: the artifact did not change, its placement did ───────────────────
    ("db/lattice_api.py", "add_artifact_to_collection"): (
        OFF_FEED, "a membership edge. Containment changes are announced by the services that "
                  "make them, against the container, because only they know whether the write was "
                  "an add, a move or a reorder."),
    ("db/lattice_api.py", "remove_artifact_from_collection"): (
        OFF_FEED, "a membership edge — see add_artifact_to_collection."),
    ("db/lattice_api.py", "set_edge_order_key"): (
        OFF_FEED, "reordering. No artifact version changes."),
    ("db/lattice_api.py", "remove_all_edges_for_root"): (
        OFF_FEED, "edge cleanup behind a delete already announced by the service."),

    # ── replication: foreign-authored writes, announced by their own origin ────────────────
    ("mesh/sync.py", "_put_op"): (
        OFF_FEED, "mesh reconciliation applying a peer's row. The authoring node emitted it "
                  "against its own origin; re-emitting here would attribute a peer's write to "
                  "this node's proper time."),
    ("mesh/sync.py", "_apply_artifacts"): (OFF_FEED, "mesh apply — see _put_op."),
    ("mesh/sync.py", "publish_manifest"): (OFF_FEED, "a mesh manifest, not an artifact change."),
    ("mesh/federation.py", "pull_from_peers"): (OFF_FEED, "federation apply — see _put_op."),
    ("mesh/export.py", "import_shard"): (
        OFF_FEED, "bulk shard import into a store handed in by the caller. An offline load, not a "
                  "live change; a feed event per row would be an unbounded burst nobody asked "
                  "for."),
    ("shard/content_tier.py", "promote_local_content"): (
        OFF_FEED, "rewrites content refs after promotion to the durable tier. The bytes behind "
                  "the ref are unchanged, so nothing a subscriber models has changed."),

    # ── out-of-band tooling on the live store: announced anyway ────────────────────────────
    #
    # Curation has no caller inside the service layer, so it is genuinely out-of-band. It still
    # emits, because "out-of-band" describes who invokes it, not what it writes: the rows are
    # ordinary artifacts in the live store, and a subscriber cannot tell which layer moved one.
    ("shard/curate.py", "file_into"): (
        BOUNDARY, "operator curation writing artifact docs directly. Emits artifact.created "
                  "against the collection the artifact arrived in — to a watcher of that "
                  "container, it is new there."),
    ("shard/curate.py", "unfile"): (
        BOUNDARY, "curation detach. Emits artifact.deleted against the collection it left: the "
                  "artifact survives, but leaving is the whole of what that container models."),
    ("shard/curate.py", "move"): (
        BOUNDARY, "curation move — two events for one write, artifact.deleted against the old "
                  "container and artifact.created against the new, because a watcher of either "
                  "one learns only about its own side."),
    ("shard/erasure.py", "erase"): (
        BOUNDARY, "the erasure path. Emits artifact.deleted per erased id, carrying an id and a "
                  "verb and nothing off the erased doc — a subscriber holding derived state is "
                  "the one place erased data survives an erasure, and an announcement naming "
                  "what it erased would put that data back on a feed."),
}



def _owner_map(tree: ast.AST) -> dict[int, str]:
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    return owner


def _receiver_is_artifact_store(node: ast.expr) -> bool:
    """Is this call's receiver the artifact/graph substrate rather than some other object?

    The distinction matters: `workspace_service.delete_artifact(...)` and
    `db.artifacts.delete_artifact(...)` share a method name and are different layers. Matching on
    the receiver — `.artifacts` / `.graph`, or a name that says it is an artifact store — keeps the
    substrate detector on the substrate. The looser "any call named put_artifact" version of this
    scan reports service calls as store writes and the table stops meaning anything.
    """
    if isinstance(node, ast.Attribute):
        return node.attr in ("artifacts", "graph")
    if isinstance(node, ast.Name):
        return node.id.endswith("artifact_store")
    return False


#: The store modules themselves — they *are* the substrate, so their writes are the implementation
#: of the chokepoint rather than uses of it. Named one by one rather than matched by directory:
#: `db/` also holds `lattice_api.py` and `lattice_identity.py`, which ARE users of the substrate and
#: whose write sites the table below pins. A directory-wide skip would silently exempt them, and an
#: exempted write site is indistinguishable from a covered one when you read the table.
_SUBSTRATE_MODULES = frozenset({
    "db/__init__.py", "db/access.py", "db/audit.py", "db/constants.py", "db/content_cache.py",
    "db/content_tier.py", "db/edge.py", "db/plane.py", "db/s3_content.py", "db/schema.py",
    "db/seq.py", "db/typed_fetch.py", "db/vertex.py",
})


def _scan(root: pathlib.Path) -> dict[tuple[str, str], set[str]]:
    """Every artifact-substrate write in the tree, keyed by (file, enclosing function).

    Skips the store modules themselves (see `_SUBSTRATE_MODULES`) and any test module: a test's
    writes are fixture setup, not a production write site the feed must account for. The store keeps
    its tests in-tree beside the modules they cover, so both skips land in the same directory.
    """
    found: dict[tuple[str, str], set[str]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if rel in _SUBSTRATE_MODULES or path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _owner_map(tree)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            attr, receiver = node.func.attr, node.func.value
            hit = (attr in _SUBSTRATE_WRITES and _receiver_is_artifact_store(receiver)) or (
                attr in _API_DELETES and isinstance(receiver, ast.Name)
                and receiver.id in _STORE_MODULE_ALIASES)
            if hit:
                found.setdefault((rel, owner.get(node.lineno, "<module>")), set()).add(attr)
    return found


def _emitters_in(path: pathlib.Path, function: str) -> set[str]:
    """The emit helpers reachable from *function*, following calls to helpers in the same module.

    Reachability rather than a literal match, because announcing through a small named helper —
    `curate._announce`, `erasure._announce_erased`, `workspace_service._emit_event` — is the shape
    a site with several emit points naturally takes. A checker that only saw the literal call would
    read those as silent and push every one of them toward an inlined copy of the same four lines,
    which is how the emit and the write drift apart in the first place.

    Bounded to this one file: a call into another module is not followed, so the evidence stays
    something a reader of this file can check by eye.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bodies = {n.name: n for n in ast.walk(tree)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    if function not in bodies:
        return set()
    found: set[str] = set()
    seen: set[str] = set()
    pending = [function]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for sub in ast.walk(bodies[name]):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            called = func.attr if isinstance(func, ast.Attribute) else \
                (func.id if isinstance(func, ast.Name) else None)
            if called in _EMITTERS:
                found.add(called)
            elif called in bodies:
                pending.append(called)
    return found


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 2 · The enumeration matches the tree
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_every_artifact_write_site_is_classified() -> None:
    """A new write path must state what it does about the feed before the suite goes green.

    This is the whole enforcement mechanism. "Complete by construction" is only true while every
    write is one of the four kinds below; a fifth kind that nobody classified is exactly the silent
    write the claim denies can exist.
    """
    sites = _scan(SRC)
    unclassified = sorted(set(sites) - set(WRITE_SITES))
    assert not unclassified, (
        "artifact-substrate writes appeared with no entry in WRITE_SITES:\n"
        + "\n".join(f"  {f} :: {fn}()  writes {sorted(sites[(f, fn)])}" for f, fn in unclassified)
        + "\n\nClassify each one: BOUNDARY (emits here), SERVICE (the caller emits), OFF_FEED "
          "(deliberately not an artifact change — say why), or GAP (silent, and that is a hole).")


def test_the_substrate_skip_still_matches_the_tree() -> None:
    """Vacuous-pass guard for `_scan`'s exemption list.

    A skip entry that matches no file is a phantom exemption; worse, an entry that STOPS matching
    silently readmits the substrate as if it were a caller, which floods the roster instead of
    hiding from it. Either way the list has stopped describing the tree, so it is checked against
    the tree rather than trusted.
    """
    missing = sorted(m for m in _SUBSTRATE_MODULES if not (SRC / m).is_file())
    assert not missing, (
        f"_SUBSTRATE_MODULES names files that are gone: {missing} — update the list to the store's "
        f"actual modules; do not delete the check.")
    scanned = {str(p.relative_to(SRC)).replace("\\", "/") for p in SRC.rglob("*.py")}
    assert _SUBSTRATE_MODULES <= scanned, sorted(_SUBSTRATE_MODULES - scanned)
    assert "db/lattice_api.py" in scanned and "db/lattice_api.py" not in _SUBSTRATE_MODULES, (
        "lattice_api.py is a USER of the substrate and its write sites are pinned in WRITE_SITES — "
        "it must stay in the scan")


def test_the_table_has_not_drifted_from_the_tree() -> None:
    """An allow-table naming sites that are not in the tree stops being evidence about it."""
    sites = _scan(SRC)
    stale = sorted(set(WRITE_SITES) - set(sites))
    assert not stale, (
        f"WRITE_SITES names write sites that are gone: {stale}. Remove them — a stale entry is "
        f"indistinguishable from a covered one when you read the table.")


def test_the_scanner_fires_on_a_seeded_write(tmp_path: pathlib.Path) -> None:
    """The control. This suite concludes from an absence, so the scanner must be seen to speak.

    Also pins the receiver rule: a same-named call on a service module is NOT a substrate write,
    and a scanner that could not tell them apart would classify half the router layer as store
    writes and hide the real ones in the noise.
    """
    (tmp_path / "sneaky.py").write_text(
        "def quietly_writes(db, doc):\n"
        "    db.artifacts.put_artifact(doc)\n"
        "\n"
        "def merely_calls_a_service(store_db, uid, cid, aid):\n"
        "    workspace_service.delete_artifact(store_db, uid, cid, aid)\n",
        encoding="utf-8")
    sites = _scan(tmp_path)
    assert ("sneaky.py", "quietly_writes") in sites, \
        f"the scanner missed a plain substrate write; it saw {sorted(sites)}"
    assert ("sneaky.py", "merely_calls_a_service") not in sites, \
        "the scanner counted a service call as a substrate write, so its findings are noise"


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 3 · Sites that claim to emit, do
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_boundary_sites_actually_call_the_chokepoint() -> None:
    """BOUNDARY means the emit is right there. Deleting it must fail this, not go unnoticed."""
    missing = []
    for (rel, function), (verdict, _note) in sorted(WRITE_SITES.items()):
        if verdict != BOUNDARY:
            continue
        if "emit_artifact_change" not in _emitters_in(SRC / rel, function):
            missing.append(f"{rel} :: {function}()")
    assert not missing, (
        "these are classified BOUNDARY but no longer call doc_boundary.emit_artifact_change:\n"
        + "\n".join("  " + m for m in missing)
        + "\n\nEither restore the emit or reclassify the site as GAP with prose saying what the "
          "feed now loses.")


def test_service_sites_emit_somewhere_they_can_see_the_container() -> None:
    """SERVICE means the emit moved up a layer, not that it vanished.

    Checked at the service functions themselves: the db-layer half of a SERVICE pair is silent by
    design, so what has to be true is that the service half is not.
    """
    silent = []
    for (rel, function), (verdict, _note) in sorted(WRITE_SITES.items()):
        if verdict != SERVICE or not rel.startswith("services/"):
            continue
        if not _emitters_in(SRC / rel, function):
            silent.append(f"{rel} :: {function}()")
    assert not silent, (
        "these carry the delete emit for the db layer but emit nothing:\n"
        + "\n".join("  " + s for s in silent)
        + "\n\nA delete that reaches no subscriber leaves every derived view holding a dead id.")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 4 · The gap set is a ratchet
# ═════════════════════════════════════════════════════════════════════════════════════════════

def test_no_new_silent_artifact_write_appears() -> None:
    """The gap set may shrink; it may not grow. It is currently empty.

    A coverage test that only fails when coverage is zero never fails. This one fails the moment a
    write is added without an emit, because adding it means adding a GAP entry with prose saying
    what the feed loses — and writing that sentence is usually enough to make someone add the emit
    instead.

    At zero the ratchet stops being a budget and becomes the claim itself: every artifact-plane
    write in the tree announces itself. Lower it, never raise it — a GAP added here is a change
    some subscriber can never learn about, and the number is the record of how many such changes
    the feed is allowed to lose.
    """
    gaps = {k for k, (verdict, _n) in WRITE_SITES.items() if verdict == GAP}
    assert len(gaps) <= 0, (
        f"{len(gaps)} artifact-plane writes now emit nothing (was 0 — every write announced). "
        f"Each one is a change a subscriber can never learn about:\n"
        + "\n".join(f"  {f} :: {fn}()" for f, fn in sorted(gaps)))


def test_no_gap_is_undocumented() -> None:
    """A GAP entry with no prose is a TODO wearing a test's clothes."""
    thin = [k for k, (verdict, note) in WRITE_SITES.items()
            if verdict == GAP and len(note) < 60]
    assert not thin, (
        f"these gaps are enumerated but not explained: {sorted(thin)}. Say what a subscriber "
        f"fails to learn, or the entry only records that someone noticed.")


# ═════════════════════════════════════════════════════════════════════════════════════════════
# 5 · End to end: one write, one event
# ═════════════════════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def store(tmp_path):
    from mantle.db import lattice_api
    return lattice_api.LatticeDatabase(str(tmp_path / "feed.db"), origin="feed-test")


@pytest.fixture
async def bus():
    """A clean bus bound to the running loop, restored afterwards.

    Async so it binds the loop the test actually runs on: `publish_event_sync` schedules onto
    whatever loop was registered, and a sync fixture would register a different one — the events
    would be published to a loop nobody is awaiting and the test would fail as a timeout rather
    than as a wrong answer.

    Module state, so it is reset both ways: a test that left a subscriber attached would make the
    next one's assertions about counts depend on ordering.
    """
    event_bus._filtered_subscribers.clear()
    event_bus.set_event_loop(asyncio.get_running_loop())
    yield event_bus
    event_bus._filtered_subscribers.clear()
    event_bus.set_event_log(None)
    event_bus.set_container_resolver(None)


async def _drain(queue, *, expect: int, timeout: float = 2.0):
    """Collect exactly *expect* events, or fail saying how many arrived.

    Bounded wait rather than a sleep: `publish_event_sync` schedules onto the loop, so the events
    arrive on a later tick and a fixed sleep would either be flaky or slow.
    """
    out = []
    for _ in range(expect):
        out.append(await asyncio.wait_for(queue.get(), timeout))
    return out


@pytest.mark.asyncio
async def test_a_create_produces_exactly_one_artifact_created_event(store, bus):
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    artifact = Artifact(collection_id="", name="c", content="", created_by="u-1")
    lattice_api.create_artifact(store, artifact)

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.created"
    assert event.artifact_id == artifact.id
    assert event.actor_id == "u-1"
    assert queue.empty(), "a single create produced more than one event"


@pytest.mark.asyncio
async def test_an_update_produces_exactly_one_artifact_updated_event(store, bus):
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api

    # Subscribe first and drain the setup create, rather than subscribing afterwards: the sync
    # publish path schedules onto the loop, so an event emitted "before" the subscription still
    # arrives after it. Draining is deterministic where a sleep would be a guess.
    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    artifact = Artifact(collection_id="", name="c", content="", created_by="u-1")
    lattice_api.create_artifact(store, artifact)
    await _drain(queue, expect=1)

    artifact.name = "renamed"
    artifact.modified_by = "u-2"
    lattice_api.update_artifact(store, artifact)

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.updated"
    assert event.artifact_id == artifact.id
    assert event.actor_id == "u-2", "the update's actor, not the creator, authored the change"
    assert queue.empty()


@pytest.mark.asyncio
async def test_a_delete_produces_exactly_one_artifact_deleted_event(store, bus):
    """Exercised at the seam the services use, because that is where a delete is announced.

    `emit_artifact_change` cannot carry a delete — by the time the db layer has one, the doc that
    named the container is gone. The static half of this suite (Section 3) proves the three service
    delete paths call this seam; this half proves the seam produces the event they promise.
    """
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    artifact = Artifact(collection_id="ws-1", name="c", content="", created_by="u-1")
    lattice_api.create_artifact(store, artifact)
    await _drain(queue, expect=1)

    assert lattice_api.delete_artifact(store, artifact.id) is True
    bus.emit_artifact_event_sync("ws-1", doc_boundary.DELETED,
                                 {"artifact_id": artifact.id}, actor_id="u-1")

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.deleted"
    assert event.artifact_id == artifact.id
    assert event.container_id == "ws-1"
    assert queue.empty()
    assert lattice_api.get_artifact(store, artifact.id) is None, \
        "the delete event was emitted for a row that is still there"


@pytest.mark.asyncio
async def test_the_three_change_events_are_the_declared_vocabulary(store, bus):
    """A subscriber filters on names, so the names are contract. Drift breaks every filter."""
    assert doc_boundary.CHANGE_EVENTS == (
        "artifact.created", "artifact.updated", "artifact.deleted")


@pytest.fixture
def content_key(monkeypatch):
    """A deterministic content master key, so the envelope boundary runs for real.

    Only needed by the cases that write non-empty content: without it the encrypt path has no key
    oracle and refuses the write, which would test the wrong failure.
    """
    from mantle.services import content_crypto
    monkeypatch.setattr(
        content_crypto, "_default_master_key",
        lambda principal_id, collection_id=None, *, may_create=False, creator_id=None: b"\x01" * 32,
    )


def _artifact_of(event) -> dict:
    return event.payload.get("artifact") or {}


# ── the container update ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_container_rename_reaches_a_subscriber(store, bus):
    """The rename is the change a live tree is likeliest to be showing stale."""
    from mantle.entities.collection import Collection
    from mantle.db import lattice_api

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    container = Collection(id="ws-1", collection_id="", name="before", content="", created_by="u-1")
    lattice_api.create_collection(store, container)
    await _drain(queue, expect=1)

    container.name = "after"
    container.modified_by = "u-2"
    lattice_api.update_collection(store, container)

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.updated"
    assert event.artifact_id == "ws-1"
    assert _artifact_of(event).get("name") == "after", \
        "the event carried the container but not the new name, so a subscriber still shows the old"
    assert event.actor_id == "u-2"
    assert queue.empty()


# ── the soft delete ───────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_archiving_tells_the_container_the_artifact_is_gone(store, bus):
    """Archived versions are filtered out of every container read, so the verb is `deleted`."""
    from mantle.entities.artifact import Artifact
    from mantle.entities.grant import Grant
    from mantle.db import lattice_api

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    artifact = Artifact(id="a-1", collection_id="col-1", name="doomed", content="",
                        created_by="u-1", state="committed")
    lattice_api.create_artifact(store, artifact)
    await _drain(queue, expect=1)
    lattice_api.create_grant(store, Grant(
        id="g-1", resource_id="col-1", grantee_type="user", grantee_id="u-9",
        granted_by="admin", can_read=True, can_delete=True))

    assert lattice_api.archive_artifact(store, "u-9", "a-1") is True

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.deleted", \
        "an archived artifact a subscriber is never told about keeps appearing in its container"
    assert event.artifact_id == "a-1"
    assert event.container_id == "col-1"
    assert event.actor_id == "u-9", "the archiving user is the actor, not the doc's last writer"
    assert queue.empty()


@pytest.mark.asyncio
async def test_a_refused_archive_announces_nothing(store, bus):
    """The control. Without the `can_delete` grant nothing is written, so nothing is announced."""
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    lattice_api.create_artifact(store, Artifact(
        id="a-2", collection_id="col-1", name="safe", content="", created_by="u-1"))
    await _drain(queue, expect=1)

    assert lattice_api.archive_artifact(store, "u-9", "a-2") is False
    await asyncio.sleep(0)
    assert queue.empty(), "a write that was refused still announced itself"


# ── the batch commit ──────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_batch_commit_emits_one_event_per_document(store, bus):
    """Per document, because a subscriber filters per artifact.

    One batch-shaped event would carry ids nobody's filter selects on, so a subscriber watching a
    single artifact in the batch would learn nothing about the commit of the artifact it watches.
    """
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    for aid in ("d-1", "d-2"):
        lattice_api.create_artifact(store, Artifact(
            id=aid, root_id=aid, collection_id="col-1", name=aid, content="",
            created_by="u-1", state="draft"))
    lattice_api.create_artifact(store, Artifact(
        id="d-3", root_id="d-3", collection_id="col-other", name="d-3", content="",
        created_by="u-1", state="draft"))
    await _drain(queue, expect=3)

    n = lattice_api.batch_commit_drafts(
        store, "col-1", ["d-1", "d-2", "d-3", "ghost"], "u-2", "2026-07-22T12:00:00+00:00")
    assert n == 2

    events = await _drain(queue, expect=2)
    assert [e.name for e in events] == ["artifact.updated", "artifact.updated"]
    assert sorted(e.artifact_id for e in events) == ["d-1", "d-2"], \
        "the out-of-collection draft was not committed, so it must not be announced"
    assert all(_artifact_of(e).get("state") == "committed" for e in events)
    assert all(e.actor_id == "u-2" for e in events)
    assert queue.empty()


@pytest.mark.asyncio
async def test_a_batch_commit_puts_no_ciphertext_on_the_feed(store, bus, content_key):
    """The commit writes the stored doc back, and a stored doc holds `content` encrypted.

    Forwarding it verbatim would hand every subscriber a base64 blob it has no key for — worse
    than an absent field, because it reads as content rather than as something to go and fetch.
    """
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    lattice_api.create_artifact(store, Artifact(
        id="d-9", root_id="d-9", collection_id="col-1", name="d-9",
        content="the-plaintext-marker", created_by="u-1", state="draft"))
    await _drain(queue, expect=1)

    stored = store.artifacts.get_artifact("d-9")
    assert stored.get("content_encrypted") is True, "the fixture did not exercise the envelope"

    lattice_api.batch_commit_drafts(store, "col-1", ["d-9"], "u-2", "2026-07-22T12:00:00+00:00")
    (event,) = await _drain(queue, expect=1)

    artifact = _artifact_of(event)
    assert "content" not in artifact and "content_encrypted" not in artifact
    assert stored["content"] not in json.dumps(event.payload), \
        "the commit event carried the stored ciphertext"


# ── the workspace delete ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deleting_a_workspace_announces_every_root_and_then_itself(bus):
    """The largest hole: everything under a container disappearing in silence.

    Store calls are stubbed because what is under test is the announcement, not the deletion —
    `tests/test_scoped_deletion_and_urls.py` owns which of the two delete arms is taken. Both arms
    are exercised here, because both end the same way for a watcher of this container: one shared
    root evicted, one exclusive root destroyed, and each has to produce an event.
    """
    from mantle.services import workspace_service

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    rows = [{"id": "a-1", "root_id": "r-shared"}, {"id": "a-2", "root_id": "r-mine"}]
    with (
        patch.object(workspace_service, "get_workspace"),
        patch("mantle.search.ingest.pipeline_unified.delete_artifact_from_index"),
        patch("mantle.db.backend.list_collection_artifacts", return_value=rows),
        patch("mantle.db.backend.count_other_containers_for_root",
              side_effect=lambda _db, root, _ws: 1 if root == "r-shared" else 0),
        patch("mantle.db.backend.delete_artifacts_by_root"),
        patch("mantle.db.backend.remove_all_edges_for_root"),
        patch("mantle.db.backend.remove_artifact_from_collection"),
        patch("mantle.db.backend.delete_collection"),
    ):
        workspace_service.delete_workspace(MagicMock(), "u-1", "ws-1")

    events = await _drain(queue, expect=3)
    assert all(e.name == "artifact.deleted" for e in events)
    assert all(e.container_id == "ws-1" for e in events)
    assert all(e.actor_id == "u-1" for e in events)
    assert [e.artifact_id for e in events] == ["r-shared", "r-mine", "ws-1"], \
        ("each root must be named — a subscriber told only that the container went cannot know "
         "which ids to drop — and the container itself must come last")
    assert queue.empty()


# ── curation ──────────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def curated(store, bus):
    """One artifact whose owner may curate it: authored by `u-1`, not cited, not a registry type."""
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api
    lattice_api.create_artifact(store, Artifact(
        id="c-1", root_id="c-1", collection_id="home", name="mine", content="",
        content_type="text/markdown", created_by="u-1"))
    return "c-1"


@pytest.mark.asyncio
async def test_filing_into_a_collection_announces_it_there(store, bus, curated):
    from mantle.shard import curate

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    await _drain(queue, expect=1)                     # the fixture's create

    result = curate.file_into(store, curated, "col-x", person="u-1", apply=True)
    assert result["ok"] and result["applied"]

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.created"
    assert event.artifact_id == curated
    assert event.container_id == "col-x", \
        "addressed to the artifact's home, not the collection whose watcher this concerns"
    assert queue.empty()


@pytest.mark.asyncio
async def test_a_curation_dry_run_announces_nothing(store, bus, curated):
    """The control. A dry run writes nothing, so it must announce nothing — an event for a change
    that did not happen is worse than a missing one, because a subscriber cannot detect it."""
    from mantle.shard import curate

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    await _drain(queue, expect=1)

    curate.file_into(store, curated, "col-x", person="u-1")
    curate.move(store, curated, "col-y", person="u-1")
    await asyncio.sleep(0)
    assert queue.empty()


@pytest.mark.asyncio
async def test_unfiling_announces_the_artifact_left(store, bus, curated):
    from mantle.shard import curate

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    await _drain(queue, expect=1)
    curate.file_into(store, curated, "col-x", person="u-1", apply=True)
    await _drain(queue, expect=1)

    assert curate.unfile(store, curated, "col-x", person="u-1", apply=True)["applied"]

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.deleted"
    assert event.artifact_id == curated
    assert event.container_id == "col-x"
    assert queue.empty()


@pytest.mark.asyncio
async def test_a_curation_move_announces_both_containers(store, bus, curated):
    """One write, two events — a watcher of either container learns only about its own side."""
    from mantle.shard import curate

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    await _drain(queue, expect=1)

    assert curate.move(store, curated, "col-new", person="u-1", apply=True)["applied"]

    left, arrived = await _drain(queue, expect=2)
    assert (left.name, left.container_id) == ("artifact.deleted", "home")
    assert (arrived.name, arrived.container_id) == ("artifact.created", "col-new")
    assert left.artifact_id == arrived.artifact_id == curated
    assert queue.empty()


# ── erasure ───────────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_erasure_announces_each_id_and_nothing_off_the_erased_doc(store, bus, content_key):
    """A subscriber holding derived state is the one place erased data survives an erasure.

    So it has to be told — and told with an id and a verb only. An event that described what it
    erased would copy the erased data onto a feed every wildcard subscriber reads and into every
    subscriber's log, where it would outlive the thing the erasure was exercised against.
    """
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api
    from mantle.shard import erasure

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    lattice_api.create_artifact(store, Artifact(
        id="e-1", root_id="e-1", collection_id="private.u-1", name="Jane Doe, 12 Elm St",
        content="her diary", content_type="text/markdown", created_by="u-1"))
    await _drain(queue, expect=1)

    report = erasure.erase(store, "u-1", apply=True)
    assert report["removed"] == 1 and report["complete"]

    (event,) = await _drain(queue, expect=1)
    assert event.name == "artifact.deleted"
    assert event.artifact_id == "e-1"
    assert event.container_id == "private.u-1", \
        "addressed to the artifact's own id, so nobody watching the collection is told"
    assert _artifact_of(event) == {"id": "e-1", "collection_id": "private.u-1"}, \
        "the erasure event carries fields off the erased doc"
    body = json.dumps(event.payload)
    assert "Jane Doe" not in body and "diary" not in body
    assert queue.empty()
    assert lattice_api.get_artifact(store, "e-1") is None


@pytest.mark.asyncio
async def test_an_erasure_dry_run_announces_nothing(store, bus, content_key):
    """The control. The inventory is a read; announcing it would tell subscribers to drop state
    for artifacts that are still there."""
    from mantle.entities.artifact import Artifact
    from mantle.db import lattice_api
    from mantle.shard import erasure

    queue = await bus.subscribe_filtered(bus.EventFilter(event_names=["artifact.*"]))
    lattice_api.create_artifact(store, Artifact(
        id="e-2", root_id="e-2", collection_id="private.u-1", name="kept", content="x",
        content_type="text/markdown", created_by="u-1"))
    await _drain(queue, expect=1)

    assert erasure.erase(store, "u-1")["applied"] is False
    await asyncio.sleep(0)
    assert queue.empty()
    assert lattice_api.get_artifact(store, "e-2") is not None
