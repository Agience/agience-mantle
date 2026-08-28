"""P-10 and P-3 from the `/artifacts` audit.

**P-10 — `context` is an OBJECT.** It was `Optional[str]` (a JSON string) on create/update and
`Optional[Dict]` on upload-initiate, so the same field had two wire types. Measured across the
2,658,783-artifact store before the ruling:

    context._opaque (the string form's only real advantage)          0
    context == ""   (INGEST-ONE-DOOR's finding, in the data)  1,165,110
    context is a real object (the colimits)                      5,484
    context is a human description ("adds two numbers")              48

It is landed as ACCEPT-BOTH rather than a type flip, and that is the point of these tests: a
flip is a breaking change. Measured the same day, **39 call sites across 31 files in 7 repos** send
`json.dumps(...)` — mantle's own `seed_provisioning` and `issuers`, every chorus persona, prism's
CLI, crystal's push, cloud's `ci_runner`, and the Claude Code memory hooks. The object is legal and
preferred; the string is deprecated and retires when those callers move. **Both must work until
then, and both must land on the same stored shape** — otherwise the two forms disagree and the
field means different things depending on how it was sent.

**P-3 — one path parameter name.** `warm` and `commits` named theirs `{container_id}` while
thirteen routes used `{artifact_id}`. Renaming does not change the URL, only the spec and the
handler signature — so the test that matters is that the URL did NOT move.
"""
from __future__ import annotations

import json

import pytest

from mantle.main import app
from mantle.routers.artifacts_router import (
    CreateArtifactRequest,
    UpdateArtifactRequest,
    _context_as_stored,
)

OBJ = {"title": "Canon", "tags": ["a", "b"], "n": 1}


# ── P-10 ──────────────────────────────────────────────────────────────────────────────────────
def test_the_object_form_is_accepted_on_create():
    assert CreateArtifactRequest(context=OBJ).context == OBJ


def test_the_object_form_is_accepted_on_update():
    assert UpdateArtifactRequest(context=OBJ).context == OBJ


def test_the_json_string_form_still_works():
    """39 call sites in 7 repos send this today. Breaking them is the thing being avoided."""
    raw = json.dumps(OBJ)
    assert CreateArtifactRequest(context=raw).context == raw
    assert UpdateArtifactRequest(context=raw).context == raw


def test_both_forms_reach_the_same_stored_shape():
    """The whole point of accept-both: the field cannot mean two things."""
    from_object = _context_as_stored(CreateArtifactRequest(context=OBJ).context)
    from_string = _context_as_stored(CreateArtifactRequest(context=json.dumps(OBJ)).context)
    assert json.loads(from_object) == json.loads(from_string) == OBJ


def test_absent_context_stays_absent():
    """`None` and `{}` are different answers — an omitted context must not become one."""
    assert _context_as_stored(None) is None
    assert CreateArtifactRequest().context is None


def test_a_non_json_string_is_not_mangled():
    """Measured-dead (0 of 2,658,783 carry `context._opaque`) but still reachable while the string
    form is accepted. It must pass through rather than raise or be silently dropped."""
    assert _context_as_stored("adds two numbers") == "adds two numbers"


def test_an_empty_object_is_preserved_and_is_not_none():
    assert json.loads(_context_as_stored({})) == {}


# ── P-3 ───────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def spec():
    return app.openapi()


@pytest.mark.parametrize("suffix", ["warm", "commits"])
def test_the_url_did_not_move(spec, suffix):
    """A path-parameter rename must be invisible on the wire. If this fails, callers broke."""
    paths = [p for p in spec["paths"] if p.endswith("/" + suffix)]
    assert paths == ["/artifacts/{artifact_id}/" + suffix], (
        "the %s URL changed shape: %s" % (suffix, paths))


def test_a_second_id_is_added_without_respelling_the_first(spec):
    """P-3 RE-AIMED 2026-08-26, twice — and the second time the tree was ahead of me.

    WHAT P-3 FORBIDS is ONE POSITION spelled two ways. Thirteen routes said `artifact_id` and
    two said `container_id` for the SAME slot, and a generator read that as two unrelated
    resources. "The string `container_id` is banned" was a true summary while every path had one
    id, and it stopped being the rule the moment a path needed two.

    The resolution in the tree is better than the one this test first asserted. The two-id path
    is `/artifacts/{artifact_id}/children/{child_id}`: the CONTAINER position keeps the universal
    spelling every other route uses, and only the genuinely new position takes a new name. So P-3's
    uniformity is not traded away at all — it is extended.

    This asserts that shape rather than merely permitting a second name: a two-id path that
    respelled its FIRST slot would be exactly P-3's original defect wearing a longer path."""
    for p in spec["paths"]:
        if not p.startswith("/artifacts/") or p.count("{") < 2:
            continue
        slots = [seg[1:-1] for seg in p.split("/") if seg.startswith("{")]
        assert slots[0] == "artifact_id", (
            "%s respells its first path parameter as %r; the container position keeps the one "
            "spelling every single-id route uses" % (p, slots[0]))


def test_a_single_id_route_always_spells_it_the_same_way(spec):
    """The uniformity the rename buys, asserted where it still applies.

    Scoped to SINGLE-id paths since 2026-08-26. On those the name is the only thing telling a
    generator that two routes address the same resource, so one spelling is the whole point. A
    two-id path carries that information in its shape instead, and needs distinct names to be
    readable at all."""
    names = set()
    for p in spec["paths"]:
        if not p.startswith("/artifacts/") or p.count("{") != 1:
            continue
        names.update(seg[1:-1] for seg in p.split("/") if seg.startswith("{"))
    assert names <= {"artifact_id"}, (
        "more than one spelling for the single-id position on /artifacts: %s" % names)


def test_the_two_id_route_names_its_positions_distinctly(spec):
    """The other half, and the reason the rule had to be re-aimed rather than relaxed: a two-id
    path whose slots shared a name would be unreadable, which is P-3's own complaint."""
    two_id = [p for p in spec["paths"] if p.startswith("/artifacts/") and p.count("{") >= 2]
    assert two_id, "no two-id path found — this test is looking in the wrong place"
    for p in two_id:
        slots = [seg[1:-1] for seg in p.split("/") if seg.startswith("{")]
        assert len(set(slots)) == len(slots), (
            "%s repeats a path-parameter name, so neither slot says which id it wants" % p)
