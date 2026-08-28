"""`PATCH` publishes the artifact states and refuses anything else.

The state this replaces: `state` was a free `string` here, while `recall` answers
`400 state must be one of committed, draft, archived` for the identical mistake. One API, two
answers to one question — the same shape as the `additionalProperties` split between MCP and REST.

And the silence was not inert. `update_workspace` assigns `ws.state` directly, and its only test
is *is it archived*:

    if state == STATE_ARCHIVED and ws.state != STATE_ARCHIVED:   -> archive
    if ws.state == STATE_ARCHIVED:
        if state and state != STATE_ARCHIVED:                    -> un-archive to COMMITTED

So on an archived artifact a misspelling such as `"drft"` is not ignored — it falls into the
un-archive branch and moves the artifact to committed, a state the caller never named, answered
`200`. A caller who meant `draft` gets `committed` and is told nothing.

Both the enum and the message are derived from `Artifact.VALID_STATES`. That vocabulary has
four other homes, each spelling the same three strings as its own literal. They agree today; the day
they stop, this route and the index disagree about what a state is. Recorded as a finding and
tripwired below rather than refactored, because reconciling them touches the search arm.

Measured 2026-08-26: grepping for the triple rather than for the two names already known found
two more homes, one of them a published API surface:

    entities/artifact.py:32          VALID_STATES      set     the entity truth
    routers/artifacts_router.py      _ARTIFACT_STATES  list    derived — pinned above
    search/mantle/wiring.py:273      VALID_SEGMENTS    tuple   object-storage prefixes
    search/ingest/pipeline_unified.py:78  _SEGMENTS    tuple   which tree a doc is indexed into
    routers/mcp_router.py TOOLS[recall]   enum         list    the published MCP contract

The last one is why the widened tripwire is worth more than the narrow one. `wiring` and
`pipeline_unified` drifting apart breaks an internal write path and something eventually crashes.
The MCP enum drifting is worse and quieter: it is what a caller reads to learn the vocabulary, so a
state added to the entity and not here is simply unreachable over MCP, with no error anywhere.
"""
from __future__ import annotations

import pytest

from mantle.entities.artifact import Artifact
from mantle.main import app
from mantle.routers.artifacts_router import _ARTIFACT_STATES, UpdateArtifactRequest


def test_the_enum_is_derived_from_the_entity():
    assert _ARTIFACT_STATES == sorted(Artifact.VALID_STATES)


def test_the_published_enum_matches_what_is_enforced():
    sch = app.openapi()["components"]["schemas"]["UpdateArtifactRequest"]["properties"]["state"]
    assert sch.get("enum") == _ARTIFACT_STATES, sch


@pytest.mark.parametrize("state", sorted(Artifact.VALID_STATES))
def test_every_published_state_is_accepted(state):
    assert UpdateArtifactRequest.model_validate({"state": state}).state == state


@pytest.mark.parametrize("bad", ["commited", "drft", "COMMITTED", "", "deleted"])
def test_anything_else_is_refused(bad):
    with pytest.raises(Exception) as exc:
        UpdateArtifactRequest.model_validate({"state": bad})
    assert "must be one of" in str(exc.value), str(exc.value)


def test_omitting_state_is_still_legal():
    """The field is optional and PATCH is partial — refusing an absent state would break every
    update that changes something else."""
    assert UpdateArtifactRequest.model_validate({"name": "x"}).state is None


@pytest.mark.asyncio
async def test_the_wire_answers_422_and_names_the_field(client):
    r = await client.patch("/artifacts/a-1", json={"state": "commited"})
    assert r.status_code == 422, r.text
    fields = {str(loc) for item in r.json()["detail"] for loc in (item.get("loc") or ())}
    assert "state" in fields, r.json()


def _other_homes():
    """`{where: the three strings it spells}` for every home that is NOT derived from the entity.

    Imported inside the function, not at module scope: `pipeline_unified` pulls in the embeddings
    stack, and making the whole of this file—including the route tests—depend on that would trade a
    fast contract test for a slow one."""
    from mantle.routers.mcp_router import _TOOLS_BY_NAME
    from mantle.search.ingest.pipeline_unified import _SEGMENTS
    from mantle.search.mantle.wiring import VALID_SEGMENTS
    recall = _TOOLS_BY_NAME["recall"]["inputSchema"]["properties"]["state"]
    return {
        "search.mantle.wiring.VALID_SEGMENTS": VALID_SEGMENTS,
        "search.ingest.pipeline_unified._SEGMENTS": _SEGMENTS,
        "mcp_router.TOOLS[recall].state.enum": recall["enum"],
    }


@pytest.mark.parametrize("where", sorted(_other_homes()))
def test_every_other_home_of_the_vocabulary_still_agrees(where):
    """Not a fix — a tripwire, one case per home so a failure NAMES the one that drifted.

    A single assert over all of them would report "the vocabulary diverged" and leave the reader to
    find which of four literals moved."""
    assert sorted(_other_homes()[where]) == _ARTIFACT_STATES, (
        "%s has diverged from Artifact.VALID_STATES: %r vs %r"
        % (where, sorted(_other_homes()[where]), _ARTIFACT_STATES))


def test_the_tripwire_covers_every_home_that_exists():
    """The vacuous-pass guard, and the reason this file was widened at all.

    The tripwire is only worth its line count if `_other_homes()` is complete — a home nobody
    listed is exactly the one that drifts. So rather than trust the list, grep the source for the
    triple and require that every file holding it is either accounted for here or derived from the
    entity. This is what turned a second home into five.

    This test reads source off disk, the shape recorded as flaky in this workspace when
    several sessions edit the tree at once. Accepted deliberately: the alternative is a
    hand-maintained list of everywhere the vocabulary lives, and that list going stale is what made
    this finding in the first place. The failure mode is also benign — a half-written file
    yields a filename to check, never a wrong pass."""
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle"
    known = {"entities/artifact.py", "routers/artifacts_router.py", "search/mantle/wiring.py",
             "search/ingest/pipeline_unified.py", "routers/mcp_router.py"}
    # A dot does not cross a newline, so this needs no negated-newline class and carries no
    # backslash at all. The first draft used one and it did not survive the shell that wrote
    # the file, twice — an escape that only has to be right in transit is an escape to delete.
    pat = re.compile(r"""['"]draft['"].*['"]archived['"]|['"]archived['"].*['"]draft['"]""")
    found = {p.relative_to(src).as_posix() for p in src.rglob("*.py")
             if "test" not in p.name and pat.search(p.read_text(encoding="utf-8-sig",
                                                                errors="replace"))}
    assert found <= known, (
        "a NEW home for the artifact-state vocabulary appeared and nothing tripwires it: %r. "
        "Add it to _other_homes() above, or derive it from Artifact.VALID_STATES." % sorted(found - known))


def test_the_mcp_default_is_the_absent_state_not_a_typed_literal():
    """The one place the vocabulary carries a DEFAULT as well as a set. `recall` defaults to
    `committed`, and that has a single home in `db/constants.STATE_WHEN_ABSENT` — if the two ever
    disagree, a caller who omits `state` searches a different tree than a stored doc with no
    `state` was filed into, and both sides think they are right."""
    from mantle.routers.mcp_router import _TOOLS_BY_NAME
    default = _TOOLS_BY_NAME["recall"]["inputSchema"]["properties"]["state"].get("default")
    assert default == Artifact.STATE_WHEN_ABSENT, (
        "recall defaults to %r but a doc with no state is in %r"
        % (default, Artifact.STATE_WHEN_ABSENT))
