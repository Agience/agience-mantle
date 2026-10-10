"""`PATCH` publishes the artifact states and refuses anything else.

An artifact is `live` or `archived` (`Artifact.VALID_STATES`). `PATCH` also accepts `committed`,
as a deprecated alias of `live`, which clients still send to restore an archived artifact. `draft` is refused, since nothing is a draft.

The refusal is load-bearing. `update_workspace` assigns `ws.state` directly, and its only test is
*is it archived*:

    if state == STATE_ARCHIVED and ws.state != STATE_ARCHIVED:   -> archive
    if ws.state == STATE_ARCHIVED:
        if state and state != STATE_ARCHIVED:                    -> un-archive to LIVE

So on an archived artifact an accepted misspelling such as `"arcived"` would fall into the
un-archive branch and move the artifact to live, a state the caller never named, answered `200`.

Both the enum and the message are derived from `Artifact.VALID_STATES`. The vocabulary has other
homes, each spelling the strings as its own literal; they are tripwired below:

    entities/artifact.py              VALID_STATES      set     the entity truth
    routers/artifacts_router.py       _ARTIFACT_STATES  list    derived — pinned above
    search/mantle/wiring.py           VALID_SEGMENTS    tuple   object-storage prefixes
    search/ingest/pipeline_unified.py _SEGMENTS         tuple   which tree a doc is indexed into
    routers/mcp_router.py             TOOLS[recall] and TOOLS[update_artifact] enums, the
                                      published MCP contract

The MCP enums are the ones worth the tripwire: they are what a caller reads to learn the
vocabulary, so a state added to the entity and not there is unreachable over MCP with no error.
"""
from __future__ import annotations

import pytest

from mantle.entities.artifact import Artifact
from mantle.main import app
from mantle.routers.artifacts_router import (_ARTIFACT_STATES, _PATCH_STATES,
                                             UpdateArtifactRequest)

#: What PATCH accepts beyond the states, and what each means.
PATCH_ALIASES = {"committed": "live"}
#: What recall accepts beyond the states: deprecated aliases that read the live segment.
RECALL_ALIASES = ("committed", "draft")


def test_the_enum_is_derived_from_the_entity():
    assert _ARTIFACT_STATES == sorted(Artifact.VALID_STATES) == ["archived", "live"]


def test_the_published_enum_matches_what_is_enforced():
    sch = app.openapi()["components"]["schemas"]["UpdateArtifactRequest"]["properties"]["state"]
    assert sch.get("enum") == _PATCH_STATES == sorted(set(_ARTIFACT_STATES) | set(PATCH_ALIASES))
    assert sch.get("x-deprecated-values") == sorted(PATCH_ALIASES)
    recall = app.openapi()["components"]["schemas"]["ArtifactRecallRequest"]["properties"]["state"]
    assert recall.get("x-deprecated-values") == sorted(RECALL_ALIASES)


@pytest.mark.parametrize("state", sorted(Artifact.VALID_STATES))
def test_every_state_is_accepted_as_itself(state):
    assert UpdateArtifactRequest.model_validate({"state": state}).state == state


@pytest.mark.parametrize("alias", sorted(PATCH_ALIASES))
def test_a_deprecated_alias_is_accepted_as_the_state_it_names(alias):
    assert UpdateArtifactRequest.model_validate({"state": alias}).state == PATCH_ALIASES[alias]


@pytest.mark.parametrize("bad", ["draft", "commited", "arcived", "LIVE", "", "deleted"])
def test_anything_else_is_refused(bad):
    with pytest.raises(Exception) as exc:
        UpdateArtifactRequest.model_validate({"state": bad})
    assert "must be one of" in str(exc.value), str(exc.value)


def test_omitting_state_is_still_legal():
    """The field is optional and PATCH is partial — refusing an absent state would break every
    update that changes something else."""
    assert UpdateArtifactRequest.model_validate({"name": "x"}).state is None


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["commited", "draft"])
async def test_the_wire_answers_422_and_names_the_field(client, bad):
    r = await client.patch("/artifacts/a-1", json={"state": bad})
    assert r.status_code == 422, r.text
    fields = {str(loc) for item in r.json()["detail"] for loc in (item.get("loc") or ())}
    assert "state" in fields, r.json()


def _other_homes():
    """`{where: (the strings it spells, the aliases it may add)}` for every home that is NOT
    derived from the entity.

    Imported inside the function, not at module scope: `pipeline_unified` pulls in the embeddings
    stack, and making the whole of this file—including the route tests—depend on that would trade a
    fast contract test for a slow one."""
    from mantle.routers.mcp_router import _TOOLS_BY_NAME
    from mantle.search.ingest.pipeline_unified import _SEGMENTS
    from mantle.search.mantle.wiring import VALID_SEGMENTS
    recall = _TOOLS_BY_NAME["recall"]["inputSchema"]["properties"]["state"]
    update = _TOOLS_BY_NAME["update_artifact"]["inputSchema"]["properties"]["state"]
    return {
        "search.mantle.wiring.VALID_SEGMENTS": (VALID_SEGMENTS, ()),
        "search.ingest.pipeline_unified._SEGMENTS": (_SEGMENTS, ()),
        "mcp_router.TOOLS[recall].state.enum": (recall["enum"], RECALL_ALIASES),
        "mcp_router.TOOLS[update_artifact].state.enum": (update["enum"], tuple(PATCH_ALIASES)),
    }


@pytest.mark.parametrize("where", sorted(_other_homes()))
def test_every_other_home_of_the_vocabulary_still_agrees(where):
    """Not a fix — a tripwire, one case per home so a failure NAMES the one that drifted."""
    spelled, aliases = _other_homes()[where]
    assert sorted(spelled) == sorted(set(_ARTIFACT_STATES) | set(aliases)), (
        "%s has diverged from Artifact.VALID_STATES: %r vs %r (+ aliases %r)"
        % (where, sorted(spelled), _ARTIFACT_STATES, aliases))


def test_the_tripwire_covers_every_home_that_exists():
    """The vacuous-pass guard: `_other_homes()` is only worth its line count if it is complete.

    So rather than trust the list, grep the source for the pair and require that every file
    holding it is accounted for here or derived from the entity.

    This test reads source off disk. The failure mode is benign — a half-written file yields a
    filename to check, never a wrong pass."""
    import pathlib
    import re
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "mantle"
    known = {"entities/artifact.py", "routers/artifacts_router.py", "search/mantle/wiring.py",
             "search/ingest/pipeline_unified.py", "routers/mcp_router.py"}
    pat = re.compile(r"""['"]live['"].*['"]archived['"]|['"]archived['"].*['"]live['"]""")
    found = {p.relative_to(src).as_posix() for p in src.rglob("*.py")
             if "test" not in p.name and pat.search(p.read_text(encoding="utf-8-sig",
                                                                errors="replace"))}
    assert found <= known, (
        "a NEW home for the artifact-state vocabulary appeared and nothing tripwires it: %r. "
        "Add it to _other_homes() above, or derive it from Artifact.VALID_STATES." % sorted(found - known))


def test_the_mcp_default_is_the_absent_state_not_a_typed_literal():
    """The one place the vocabulary carries a DEFAULT as well as a set. `recall` defaults to
    `live`, and that has a single home in `db/constants.STATE_WHEN_ABSENT` — if the two ever
    disagree, a caller who omits `state` searches a different tree than a stored doc with no
    `state` was filed into."""
    from mantle.routers.mcp_router import _TOOLS_BY_NAME
    from mantle.routers.artifacts_router import ArtifactRecallRequest
    default = _TOOLS_BY_NAME["recall"]["inputSchema"]["properties"]["state"].get("default")
    assert default == Artifact.STATE_WHEN_ABSENT == "live", (
        "recall defaults to %r but a doc with no state is in %r"
        % (default, Artifact.STATE_WHEN_ABSENT))
    assert ArtifactRecallRequest.model_fields["state"].default == default
