"""`/mcp` can write and can find — and reaches nothing the API would not.

A read-only tool surface is not a store. `create_artifact` and `recall` close the loop, and the
whole risk in adding them is that a WRITE door and a SEARCH door are exactly the two places a
second entry point stops being a second entry point and becomes a second implementation. So the
claims here are about identity, not about features:

* the write is the service call `POST /artifacts` makes, reached through that handler, so the
  self-grant, the index enqueue, the event and the state default are not restated anywhere;
* the search is the accessor `POST /artifacts/recall` builds, so the field filters, the
  `applied_filters` echo and the coverage `ordering` are not restated either;
* a refusal reaches the model as the API's own words, because those words name a remedy;
* and a tool naming an artifact the caller may not read is indistinguishable from one matching
  nothing — measured on the production lexical arm over a real encrypted index, against the same
  corpus and the same withheld artifact as `test_field_filters_narrow_recall.py`, through the
  other door. That is what "MCP does not widen the light cone" has to mean to be worth saying.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mantle.main import app
from mantle.services.dependencies import AuthContext, get_auth

# The corpus, the light cone and the withheld artifact are imported rather than rebuilt: the
# claim is that the OTHER DOOR answers the same way, and a second corpus would only prove that a
# second corpus behaves. `_live_anchorset` is autouse in its own module and stays autouse here.
from tests.test_field_filters_narrow_recall import (  # noqa: F401 - `stack`/`_live_anchorset` are fixtures
    ALICE,
    AUTHORIZED,
    PDF,
    SECRET,
    TERM,
    _accessor,
    _act,
    _live_anchorset,
    _NoVector,
    stack,
)


def _rpc(method: str, params=None, req_id=1):
    msg = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        msg["params"] = params
    return msg


def _call(name: str, arguments: dict, req_id=1):
    return _rpc("tools/call", {"name": name, "arguments": arguments}, req_id=req_id)


def _tool(client_response) -> dict:
    """The `result` of a `tools/call`, asserting it was not a PROTOCOL error."""
    body = client_response.json()
    assert "error" not in body, body
    return body["result"]


def _as_user(user_id: str | None) -> None:
    """Rebind `get_auth` for this test. Cleared by conftest's `override_dependencies`."""
    app.dependency_overrides[get_auth] = lambda: AuthContext(
        principal_id=user_id or "", principal_type="user", user_id=user_id,
    )


#: The OAuth client in the `mcp_client` cases below. A string no grant is keyed by, on purpose:
#: if the light cone were ever resolved from it the recalls would answer emptily for the wrong
#: reason, so `test_the_client_id_is_not_what_carries_the_reach` pins that separately.
MCP_CLIENT_ID = "mcp-client-9f2a"


def _as_mcp_client(user_id: str | None, client_id: str = MCP_CLIENT_ID) -> None:
    """The shape `resolve_auth` builds from Origin's scoped token: the USER is the subject,
    the OAuth client is the actor. See `test_mcp_client_acts_for_its_user.py` for why."""
    app.dependency_overrides[get_auth] = lambda: AuthContext(
        principal_id=user_id or "", principal_type="mcp_client", user_id=user_id,
        actor=client_id,
    )


# ── the surface ───────────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestTheToolSurfaceCanWriteAndFind:
    async def _tools(self, client) -> dict:
        r = await client.post("/mcp", json=_rpc("tools/list"))
        return {t["name"]: t for t in r.json()["result"]["tools"]}

    async def test_both_halves_of_the_loop_are_advertised(self, client):
        tools = await self._tools(client)
        assert "create_artifact" in tools, "a client that cannot write has no use for a store"
        assert "recall" in tools, "and one that cannot search cannot get back what it wrote"

    async def test_the_schemas_survive_the_wire(self, client):
        """`tools/list` is the only description of these tools a model ever sees."""
        for name in ("create_artifact", "recall"):
            schema = (await self._tools(client))[name]["inputSchema"]
            assert schema["type"] == "object"
            assert schema["additionalProperties"] is False
            assert schema["properties"], "a schema with no properties describes nothing"

    async def test_recalls_text_argument_is_named_query_text_and_only_that(self, client):
        """The common mistake, closed in the two places a model can read.

        `query` is refused by the request model — it is ignored, which leaves the recall with
        nothing to search on and produces the 400. So the schema must not permit it and the
        description must say the name, or a model guesses and spends a turn finding out."""
        recall = (await self._tools(client))["recall"]
        props = recall["inputSchema"]["properties"]
        assert "query_text" in props
        assert "query" not in props
        assert "`query_text`" in recall["description"]
        assert "no `query` argument" in recall["description"]

    async def test_the_ten_filterable_fields_are_named_where_a_model_reads_them(self, client):
        """Anything else before a colon is a search term, silently. A model that has not been
        told the roster invents `author:` and `date:` and gets an empty result with no error."""
        from mantle.search.field_filters import FILTERABLE_FIELDS

        text = (await self._tools(client))["recall"]["inputSchema"]["properties"]["query_text"][
            "description"
        ]
        for field in FILTERABLE_FIELDS:
            assert f"`{field}`" in text, f"{field} is filterable and unnamed in the schema"
        assert "ORDINARY SEARCH TERM" in text, (
            "the roster is only useful beside the rule that makes it load-bearing"
        )


# ── one write path, one search path ───────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestTheToolsUseTheApisOwnPaths:
    """Not "a tool exists" but "the tool is the same code". A tool that reimplemented either
    verb would pass every protocol test in `test_mcp_router.py` and still be a bypass."""

    async def test_create_artifact_calls_the_service_post_artifacts_calls(self, client):
        """`workspace_service.create_container` is where the owner grant, the grant-cache
        invalidation and the indexing live. Reaching it is what makes an MCP write and an HTTP
        write the same write."""
        entity = MagicMock()
        entity.to_dict.return_value = {"id": "art-new", "state": "committed"}
        with patch("mantle.services.workspace_service.create_container",
                   return_value=entity) as create:
            r = await client.post("/mcp", json=_call(
                "create_artifact", {"content": "a conversation", "name": "chat"}))

        assert create.called, "create_artifact did not reach the service POST /artifacts uses"
        assert create.call_args.kwargs["user_id"] == "user-123", "with the caller's own principal"
        assert create.call_args.kwargs["content"] == "a conversation"
        assert _tool(r)["structuredContent"] == {"id": "art-new", "state": "committed"}

    async def test_a_container_create_is_authorized_before_anything_is_written(self, client):
        """Filing into a collection needs `create` on it, checked with the caller's own
        AuthContext — the same call the REST handler makes, in the same place."""
        with patch("mantle.routers.artifacts_router.check_access") as check:
            check.side_effect = AssertionError("stop here")
            await client.post("/mcp", json=_call(
                "create_artifact", {"content": "x", "container_id": "col-1"}))

        assert check.called
        auth, container_id, action, _db = check.call_args.args
        assert (container_id, action) == ("col-1", "create")
        assert auth.user_id == "user-123"

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_recall_runs_the_accessor_post_artifacts_recall_builds(self, builder, client):
        """The accessor is where narrowing, the filters and the ordering are. The tool must be
        handed the one the router wires, carrying the caller's own `user_id` — `_key_request`
        reads that to ask the oracle for keys, so a search with the wrong one gets no keys."""
        accessor = MagicMock()
        accessor.search.return_value = _empty_result()
        builder.return_value = accessor

        r = await client.post("/mcp", json=_call("recall", {"query_text": "budget type:pdf"}))

        assert accessor.search.called, "recall did not reach the router's accessor"
        query = accessor.search.call_args.args[0]
        assert query.user_id == "user-123"
        assert query.query_text == "budget type:pdf"
        assert _tool(r)["structuredContent"]["applied_filters"] == ["content_type:pdf"]

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_the_tool_result_is_the_rest_body(self, builder, client):
        """Field for field, including `from` — which exists only because the response model's
        own serializer renames `from_`. A tool result assembled by hand would not have it."""
        def _fresh(*_a, **_k):
            accessor = MagicMock()
            accessor.search.return_value = _empty_result()
            return accessor
        builder.side_effect = _fresh

        body = {"query_text": "budget", "size": 5, "from": 10}
        rest = await client.post("/artifacts/recall", json=body)
        mcp = await client.post("/mcp", json=_call("recall", body))

        assert rest.status_code == 200
        assert _tool(mcp)["structuredContent"] == rest.json()
        assert "from" in rest.json()


def _empty_result():
    """A `SearchResult` the router can map, with the real parse behind `applied_filters`."""
    from mantle.search.field_filters import describe
    from mantle.search.query_parser import parse_query
    from mantle.search.types import SearchResult

    parsed = parse_query("budget type:pdf")
    return SearchResult(hits=[], total=0, parsed_query=parsed,
                        applied_filters=describe(parsed.filters), corrections=[],
                        ordering="recency")


# ── a refusal must name a remedy ──────────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestARefusalReachesTheModelIntact:
    """An MCP client shows a tool error to the model, and the model is the only reader that can
    act on it. The API's 400s were deliberately written to name the way out; a tool error that
    said "the call failed" would throw that away and cost a turn to learn nothing."""

    async def _both(self, client, builder, body):
        accessor = MagicMock()
        accessor.search.side_effect = lambda q: _plan_only(q)
        builder.return_value = accessor
        rest = await client.post("/artifacts/recall", json=body)
        mcp = await client.post("/mcp", json=_call("recall", body))
        return rest, _tool(mcp)

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_refused_field_carries_the_apis_own_words(self, builder, client):
        rest, tool = await self._both(client, builder, {"query_text": "budget state:draft"})
        assert rest.status_code == 400
        assert tool["isError"] is True
        text = tool["content"][0]["text"]
        assert rest.json()["detail"] in text, "the tool must carry the API's message, not its own"
        assert "request field" in text, "and that message is what names the way out"
        assert text.startswith("400: "), "a 400 says 'fix the request'; a 503 says something else"

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_an_unsupported_operator_carries_it_too(self, builder, client):
        rest, tool = await self._both(client, builder, {"query_text": "budget tag:~ai"})
        assert rest.status_code == 400
        assert rest.json()["detail"] in tool["content"][0]["text"]

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_filter_only_query_carries_it_too(self, builder, client):
        rest, tool = await self._both(client, builder, {"query_text": "type:pdf"})
        assert rest.status_code == 400
        assert "not a recall by itself" in tool["content"][0]["text"]

    @patch("mantle.search.mantle.wiring.build_sse_search_accessor")
    async def test_a_caller_sending_query_instead_of_query_text_is_told_the_name(
            self, builder, client):
        """The mistake the schema exists to prevent, and what it costs when a client makes it
        anyway: unknown fields are ignored, so the recall has nothing to search on. The message
        names the field that would have worked."""
        builder.return_value = MagicMock()
        r = await client.post("/mcp", json=_call("recall", {"query": "budget"}))
        tool = _tool(r)
        assert tool["isError"] is True
        assert "query_text" in tool["content"][0]["text"]

    async def test_an_unconfigured_node_says_so_rather_than_failing_blankly(self, client):
        """503, not 400: nothing about the request is wrong, and the message points at the
        prerequisite an operator has to supply."""
        with patch("mantle.search.mantle.wiring.build_sse_search_accessor", return_value=None):
            rest = await client.post("/artifacts/recall", json={"query_text": "budget"})
            mcp = await client.post("/mcp", json=_call("recall", {"query_text": "budget"}))

        assert rest.status_code == 503
        tool = _tool(mcp)
        assert tool["isError"] is True
        assert tool["content"][0]["text"] == f"503: {rest.json()['detail']}"

    async def test_a_failed_tool_is_still_a_result_and_not_a_protocol_error(self, client):
        """A refusal the model can read and adapt to. A JSON-RPC error would read as a broken
        server and hide the reason."""
        with patch("mantle.search.mantle.wiring.build_sse_search_accessor", return_value=None):
            body = (await client.post("/mcp", json=_call("recall", {"query_text": "x"}))).json()
        assert "error" not in body and body["result"]["isError"] is True


def _plan_only(query):
    """Run the REAL query plan and answer emptily — so the 400s above are the production
    validation raising, not a mock configured to raise."""
    from mantle.search.field_filters import describe
    from mantle.search.mantle.sse.router_accessor import plan_recall
    from mantle.search.types import SearchResult

    plan = plan_recall(query.query_text, has_vector=query.query_embedding is not None)
    return SearchResult(hits=[], total=0, parsed_query=plan.parsed,
                        applied_filters=describe(plan.parsed.filters), corrections=[],
                        ordering="recency")


# ── THE SECURITY PROPERTY ─────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestAToolDoesNotWidenTheLightCone:
    """The MCP counterpart of `TestAFilterOnlyEverNarrows`, on the same corpus.

    `art-secret` is real, indexed, a PDF, and carries the query term — and is not in Alice's
    light cone. If a recall naming it differed in ANY observable from one naming a value that
    exists nowhere, the tool would be an oracle for the existence of artifacts outside the light
    cone, and the second door would reach further than the first.
    """

    async def _recall(self, client, stack, query_text: str) -> dict:
        _act(ALICE, "user")
        with patch("mantle.search.mantle.wiring.build_sse_search_accessor",
                   return_value=_accessor(stack, embeddings=_NoVector())):
            r = await client.post("/mcp", json=_call("recall", {"query_text": query_text}))
        tool = _tool(r)
        assert "isError" not in tool, tool["content"][0]["text"]
        return tool["structuredContent"]

    async def test_the_corpus_answers_at_all(self, client, stack):
        """Without this every assertion below passes on a tool that returns nothing, ever."""
        _as_user(ALICE)
        out = await self._recall(client, stack, f"{TERM} type:application/pdf")
        assert [h["id"] for h in out["hits"]] == [PDF]
        assert out["ordering"] == "coverage"

    async def test_naming_an_unreadable_artifact_returns_nothing(self, client, stack):
        _as_user(ALICE)
        out = await self._recall(client, stack, f"{TERM} id:{SECRET}")
        assert out["hits"] == [] and out["total"] == 0

    async def test_that_answer_is_indistinguishable_from_one_matching_nothing(self, client, stack):
        """Stated as an indistinguishability, which is the only form that is worth anything:
        every field of the two responses agrees except the caller's own echoed input."""
        _as_user(ALICE)
        unreadable = await self._recall(client, stack, f"{TERM} id:{SECRET}")
        nonexistent = await self._recall(client, stack, f"{TERM} id:art-no-such-thing")

        echoes = {"query_text", "parsed_query", "applied_filters"}
        assert ({k: v for k, v in unreadable.items() if k not in echoes}
                == {k: v for k, v in nonexistent.items() if k not in echoes})
        assert unreadable["applied_filters"] == [f"id:{SECRET}"]
        assert nonexistent["applied_filters"] == ["id:art-no-such-thing"]

    async def test_a_tool_cannot_search_outside_the_callers_light_cone(self, client, stack):
        """The same tool, the same corpus, a different principal. Bob holds nothing, so the
        recall that returns a hit for Alice returns none for him — the light cone is resolved
        from the AuthContext the tool was handed, never from the arguments."""
        _as_user("user-bob")
        _act("user-bob", "user")
        with patch("mantle.search.mantle.wiring.build_sse_search_accessor",
                   return_value=_accessor(stack, embeddings=_NoVector())):
            r = await client.post("/mcp", json=_call(
                "recall", {"query_text": f"{TERM} type:application/pdf"}))
        assert _tool(r)["structuredContent"]["hits"] == []

    async def test_the_withheld_artifact_is_genuinely_in_the_index(self, stack):
        """The control for all three: `art-secret` is a document in the same encrypted index,
        not an absent one, so the empty answers above are custody and not vacuity."""
        assert SECRET not in AUTHORIZED
        from mantle.search.mantle.oracle import KeyPurpose, KeyRequest
        from tests.test_field_filters_narrow_recall import CELL_PRINCIPAL, COLLECTION

        _act(CELL_PRINCIPAL, "principal")
        lookup = stack["narrower"].lookup_for(
            TERM, KeyRequest(requester_id=CELL_PRINCIPAL, purpose=KeyPurpose.SELF,
                             requester_type="principal", action="read"))
        assert SECRET in set(lookup([(CELL_PRINCIPAL, COLLECTION)]))


# ── THE SAME PROPERTY, FOR THE PRINCIPAL THAT ACTUALLY ARRIVES OVER OAUTH ─────────────────────
@pytest.mark.anyio
class TestAnMcpClientDoesNotWidenTheLightCone:
    """`TestAToolDoesNotWidenTheLightCone` again, for an `mcp_client` rather than a `user`.

    That class proves the tool surface does not widen a USER's cone. But the principal that
    actually reaches `/mcp` over an OAuth connection is an `mcp_client` — Origin's scoped token
    — and it resolves to the user it acts for (`sub`), with the OAuth client kept only as
    `actor`. That resolution is what decides the cone, so the property has to be re-measured
    under it rather than inherited: the same corpus, the same withheld `art-secret`, the same
    indistinguishability, through the same door as the principal that really uses it.

    The claim is exactly "an `mcp_client` reaches the cone its resolution implies, and nothing
    outside it" — no wider than the user it acts for, and no narrower.
    """

    async def _recall(self, client, stack, query_text: str, subject: str = ALICE) -> dict:
        _act(subject, "mcp_client")
        with patch("mantle.search.mantle.wiring.build_sse_search_accessor",
                   return_value=_accessor(stack, embeddings=_NoVector())):
            r = await client.post("/mcp", json=_call("recall", {"query_text": query_text}))
        tool = _tool(r)
        assert "isError" not in tool, tool["content"][0]["text"]
        return tool["structuredContent"]

    async def test_it_reaches_exactly_what_the_user_it_acts_for_reaches(self, client, stack):
        """The upper bound AND the lower one in a single assertion. Alice's own recall returns
        `art-pdf`; a client acting for Alice returns `art-pdf`. Anything more would be a
        widening; anything less is the `GrantDenied` this file's sibling was written for."""
        _as_mcp_client(ALICE)
        out = await self._recall(client, stack, f"{TERM} type:application/pdf")

        assert [h["id"] for h in out["hits"]] == [PDF]
        assert out["ordering"] == "coverage"

    async def test_naming_the_withheld_artifact_returns_nothing(self, client, stack):
        _as_mcp_client(ALICE)
        out = await self._recall(client, stack, f"{TERM} id:{SECRET}")
        assert out["hits"] == [] and out["total"] == 0

    async def test_that_answer_is_indistinguishable_from_one_matching_nothing(self, client, stack):
        """The property that makes the tool unusable as an existence oracle, restated for this
        principal: every field of the two responses agrees except the caller's own echoed input.
        A client that could tell "exists but withheld" from "does not exist" would be reading
        outside the cone even while returning no hits."""
        _as_mcp_client(ALICE)
        unreadable = await self._recall(client, stack, f"{TERM} id:{SECRET}")
        nonexistent = await self._recall(client, stack, f"{TERM} id:art-no-such-thing")

        echoes = {"query_text", "parsed_query", "applied_filters"}
        assert ({k: v for k, v in unreadable.items() if k not in echoes}
                == {k: v for k, v in nonexistent.items() if k not in echoes})
        assert unreadable["applied_filters"] == [f"id:{SECRET}"]
        assert nonexistent["applied_filters"] == ["id:art-no-such-thing"]

    async def test_the_subject_is_what_carries_the_reach_not_the_client(self, client, stack):
        """The same OAuth client, a different subject. It holds nothing for Bob, so the recall
        that answers for Alice answers emptily for him — the cone follows `sub`, and the client
        id is along for provenance only. If `actor` were ever consulted, these two would agree."""
        _as_mcp_client("user-bob")
        assert (await self._recall(client, stack, f"{TERM} type:application/pdf",
                                   subject="user-bob"))["hits"] == []

    async def test_the_client_id_is_not_what_carries_the_reach(self, client, stack):
        """The converse, and the actual defect this guards. Resolving the CLIENT ID as the
        principal is what the tree did before: it names no grant, so the cone is empty and every
        content key is refused. Measured directly so the empty answer above is known to be Bob's
        missing grants rather than this."""
        _as_mcp_client(MCP_CLIENT_ID)
        assert (await self._recall(client, stack, f"{TERM} type:application/pdf",
                                   subject=MCP_CLIENT_ID))["hits"] == []

    async def test_the_two_doors_agree_for_this_principal_too(self, client, stack):
        """`_call_tool` passes `auth` through, so MCP and REST see one principal. Asserted for
        `mcp_client` specifically: the tool must not be the place where a client type acquires
        reach the API would not have given it."""
        _as_mcp_client(ALICE)
        _act(ALICE, "mcp_client")
        with patch("mantle.search.mantle.wiring.build_sse_search_accessor",
                   return_value=_accessor(stack, embeddings=_NoVector())):
            rest = await client.post("/artifacts/recall",
                                     json={"query_text": f"{TERM} type:application/pdf"})
        assert rest.status_code == 200
        assert [h["id"] for h in rest.json()["hits"]] == [PDF]


# ── the same refusal at the same door ─────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestAnUnauthenticatedCallIsRefusedTheSameWay:
    """`_call_tool` passes `auth` straight through, so an anonymous caller meets the handler's
    own 401 rather than a second one written for MCP. Asserted against the REST response for the
    same principal — one refusal, two doors."""

    async def test_recall_gives_the_apis_own_401(self, client):
        _as_user(None)
        rest = await client.post("/artifacts/recall", json={"query_text": "budget"})
        tool = _tool(await client.post("/mcp", json=_call("recall", {"query_text": "budget"})))

        assert rest.status_code == 401
        assert tool["isError"] is True
        assert tool["content"][0]["text"] == f"401: {rest.json()['detail']}"
        assert rest.json()["detail"] == "Missing authorization"

    async def test_create_gives_the_apis_own_401(self, client):
        _as_user(None)
        rest = await client.post("/artifacts", json={"content": "x"})
        tool = _tool(await client.post("/mcp", json=_call("create_artifact", {"content": "x"})))

        assert rest.status_code == 401
        assert tool["isError"] is True
        assert tool["content"][0]["text"] == f"401: {rest.json()['detail']}"
        assert rest.json()["detail"] == "User identification required"

    async def test_nothing_was_written_on_the_way_to_that_refusal(self, client):
        """A 401 that had already created the artifact would be a write with no principal."""
        _as_user(None)
        with patch("mantle.services.workspace_service.create_container") as create:
            await client.post("/mcp", json=_call("create_artifact", {"content": "x"}))
        assert not create.called
