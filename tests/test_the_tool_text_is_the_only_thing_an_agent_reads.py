"""The MCP surface describes itself accurately, and does not break on a malformed message.

A human reads the code when a description misleads them; a model reads the description and nothing
else. A wrong tool description is therefore not a documentation bug — it is the tool being wrong for
its only caller.

The checks here pin both halves of that, because a description can only be accurate about behaviour
that works:

- `update_artifact` changes `name`/`description` on a collection member, which is what "only the
  fields you supply change" promises.
- `create_artifact` keeps `description` on the ordinary member-create path — the field its own
  schema recommends supplying so the artifact can be found again.
- `list_artifacts` says it returns id order, and points at the tool that does match.
- `additionalProperties: false` is enforced rather than merely declared.
- Malformed JSON-RPC answers a JSON-RPC error, not a 500.
"""
from __future__ import annotations

import pytest

from mantle.routers.mcp_router import TOOLS, _schema_violation


BY_NAME = {t["name"]: t for t in TOOLS}


# ── 3. the descriptions say what is true ─────────────────────────────────────────────────────


def test_list_artifacts_does_not_claim_an_order_it_does_not_have():
    """It said "newest first" and returns `id` order — `db/vertex.list_artifacts` is documented as
    "Filtered stream in `id` order".

    Newest-first is not a small fix and is not made here: no index supports it (`ix_v_origin` is
    `(_origin, _seq)`, so a global recency sort is a full sort of the table, the query shape
    `db/schema.py`'s header bans) and that listing deliberately refuses OFFSET paging in favour of
    keyset. So the honest correction is the description, and the audit's instruction is satisfied by
    it: an agent that believed "newest first" was choosing this tool for a job it cannot do.
    """
    description = BY_NAME["list_artifacts"]["description"]
    assert "newest first" not in description.lower().replace("not newest first", "")
    assert "id order" in description.lower()


def test_list_artifacts_says_it_matches_on_nothing():
    """The server's own `initialize` instructions already say `list_artifacts` "matches on nothing";
    the tool description is where an agent choosing between tools actually looks."""
    description = BY_NAME["list_artifacts"]["description"].lower()
    assert "recall" in description, "it should point at the tool that does match"


@pytest.mark.parametrize("name", sorted(BY_NAME))
def test_every_tool_says_what_it_is_for(name):
    """A tool with no description is a tool an agent has to guess at."""
    assert (BY_NAME[name].get("description") or "").strip()


# ── 4. the schema is enforced, not decoration ────────────────────────────────────────────────


def test_an_unknown_argument_is_refused_and_named():
    """`additionalProperties: false` now decides something.

    Silently dropping the key meant the tool ran without it, and the error that eventually came back
    described the request that ARRIVED rather than the mistake — "query_text or vector is required"
    to a caller who thought they had sent a query.
    """
    violation = _schema_violation("recall", {"query": "budget"})
    assert violation is not None
    assert "query" in violation, "the key that was wrong must be named"
    assert "query_text" in violation, "and the keys that exist, so one retry is enough"


def test_a_missing_required_argument_is_refused_and_named():
    violation = _schema_violation("get_artifact", {})
    assert violation is not None and "artifact_id" in violation


def test_a_correct_call_is_not_refused():
    """The enforcement must not be a new way to fail. Defaults are absent, not wrong."""
    assert _schema_violation("get_artifact", {"artifact_id": "art-1"}) is None
    assert _schema_violation("list_artifacts", {}) is None
    assert _schema_violation("list_artifacts", {"limit": 10, "offset": 5}) is None
    assert _schema_violation("recall", {"query_text": "budget"}) is None


def test_every_tool_declares_the_arguments_its_schema_requires():
    """A `required` key absent from `properties` would refuse every call to that tool while
    reporting the caller's fault. Cheap to assert, impossible to notice by reading."""
    for name, tool in BY_NAME.items():
        schema = tool["inputSchema"]
        declared = set(schema.get("properties") or {})
        for key in schema.get("required") or []:
            assert key in declared, f"{name} requires {key!r} but does not declare it"


def test_the_enforcement_reads_the_declaration_rather_than_a_second_list():
    """The check must be driven by the schema the client was handed, or the two can disagree — and
    the client would be right, because the schema is the contract it was given."""
    for name, tool in BY_NAME.items():
        if tool["inputSchema"].get("additionalProperties") is not False:
            continue
        made_up = "definitely_not_a_real_argument"
        assert _schema_violation(name, {
            **{k: "x" for k in tool["inputSchema"].get("required") or []},
            made_up: "x",
        }), f"{name} declares additionalProperties:false and accepted {made_up}"


# ── 5. a malformed message is a rejection, not a crash ───────────────────────────────────────


@pytest.mark.anyio
@pytest.mark.parametrize("payload,expect", [
    ([1, 2], "each message must be an object"),
    (["hello"], "each message must be an object"),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "nope"},
     "params must be an object"),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": [1]},
     "params must be an object"),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
      "params": {"name": "list_artifacts", "arguments": "nope"}}, "arguments must be an object"),
    ({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
      "params": {"name": "list_artifacts", "arguments": [1]}}, "arguments must be an object"),
])
async def test_a_malformed_message_is_answered_not_crashed(payload, expect):
    """FOUR OF THESE RETURNED 500, reproduced 2026-08-17.

    `params` and `arguments` are caller-supplied JSON and need not be objects; both were assumed to
    be, so `.get` on a string raised `AttributeError` out of the handler. A batch element need not be
    an object either — `[1, 2]` is well-formed JSON and reached `msg.get`.

    A 500 tells a client the SERVER broke. Every one of these is a message the server understood well
    enough to reject, and the difference matters to a caller deciding whether to retry.
    """
    from mantle.routers import mcp_router as m

    class _Req:
        async def json(self):
            return payload

    resp = await m.mcp_post(_Req(), auth=None, store_db=None)          # type: ignore[arg-type]
    assert resp.status_code in (200, 400), resp.status_code
    text = resp.body.decode("utf-8")
    assert expect in text, text
    assert "AttributeError" not in text, (
        "an internal type error reached the caller — it names a Python type, not a fix"
    )


# ── 1 & 2. the fields the tools promised and dropped ─────────────────────────────────────────
#
# These are the behaviour half, and they are why "correct the descriptions" was not enough on its
# own: both tools described what they were supposed to do, and the description was the accurate part.

from unittest.mock import MagicMock, patch                             # noqa: E402

from mantle.entities.artifact import Artifact as ArtifactEntity        # noqa: E402
from mantle.services import workspace_service as ws_svc                # noqa: E402


def _member(**kw):
    d = dict(id="a-1", root_id="a-1", collection_id="ws-1", name="before",
             description="before", content="", content_type="text/plain",
             state=ArtifactEntity.STATE_DRAFT, created_by="user-1", modified_by="user-1")
    d.update(kw)
    return ArtifactEntity(**d)


def test_updating_a_member_changes_its_name_and_description():
    """200 AND NO CHANGE, for a collection member only.

    `artifacts_router.update_artifact` forks on whether the artifact has a parent. The top-level
    branch passed `name` and `description` to `update_workspace`; the member branch did not pass them
    to `update_artifact`, which did not accept them. So `PATCH /artifacts/{id}` on a member returned
    200 and moved neither field, while the identical call on a top-level artifact worked.

    The MCP tool promises "only the fields you supply change". For a member it changed less.

    Findability rides on it as well as the field: `name` is what the lexical arm indexes as `title`
    (`pipeline_unified._STATED_OFFER_FIELDS`), so a member whose rename is dropped stays findable
    only under its old name.
    """
    db = MagicMock()
    art = _member()
    with (
        patch("mantle.services.workspace_service.store.get_collection_by_id"),
        patch("mantle.services.workspace_service.get_workspace"),
        patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
        patch("mantle.services.workspace_service.store.update_artifact", return_value=art),
        patch("mantle.services.workspace_service._emit_event"),
    ):
        out = ws_svc.update_artifact(
            db, "user-1", "ws-1", "a-1",
            name="after", description="after too", reindex=False,
        )
    assert out.name == "after", "a member update dropped `name` and returned success"
    assert out.description == "after too", "a member update dropped `description`"


def test_updating_a_member_without_naming_them_leaves_them_alone():
    """"Only the fields you supply change" cuts both ways — the fix must not clear what was omitted."""
    db = MagicMock()
    art = _member()
    with (
        patch("mantle.services.workspace_service.store.get_collection_by_id"),
        patch("mantle.services.workspace_service.get_workspace"),
        patch("mantle.services.workspace_service.store.get_artifact", return_value=art),
        patch("mantle.services.workspace_service.store.update_artifact", return_value=art),
        patch("mantle.services.workspace_service._emit_event"),
    ):
        out = ws_svc.update_artifact(db, "user-1", "ws-1", "a-1", content="new", reindex=False)
    assert out.name == "before" and out.description == "before"


def test_creating_a_member_keeps_its_description():
    """Dropped on the common path. `POST /artifacts` with a `container_id` goes through
    `create_workspace_artifact`, which took `name` and not `description` — so a writer that supplied
    one got a 200 and an artifact without it.

    Worse than losing a field: `description` is one of the three stated offer fields the lexical arm
    indexes, and the `create_artifact` tool description recommends supplying it precisely so the
    artifact can be found again. The tool asked for the one thing this path threw away.
    """
    db = MagicMock()
    with (
        patch("mantle.services.workspace_service.get_workspace"),
        patch("mantle.services.workspace_service.store.create_artifact"),
        patch("mantle.services.workspace_service.store.add_artifact_to_collection"),
        patch("mantle.services.workspace_service._link_to_target_collections"),
        patch("mantle.services.workspace_service._emit_event"),
        patch("mantle.services.workspace_service.store.upsert_user_collection_grant"),
    ):
        art = ws_svc.create_workspace_artifact(
            db, "user-1", "ws-1", context="{}", content="",
            name="a name", description="what it is for",
            order_key="a0", enqueue_index=False,
        )
    assert art.name == "a name"
    assert art.description == "what it is for", (
        "the create path dropped `description` — the field its own tool schema recommends for "
        "retrieval, and one the lexical arm indexes"
    )


def test_the_create_tool_still_recommends_the_field_it_now_keeps():
    """The description and the behaviour have to agree in BOTH directions. This one was accurate all
    along and the code was wrong; asserting it keeps the pair together if either moves."""
    described = (BY_NAME["create_artifact"]["inputSchema"]["properties"]
                 .get("description", {}).get("description", "")).lower()
    assert described, "the `description` argument is undocumented"
    assert "recall" in described or "find" in described or "retriev" in described, (
        f"the schema no longer says why `description` is worth supplying: {described!r}"
    )
