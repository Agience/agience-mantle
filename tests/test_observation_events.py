"""An observation is recorded for every MCP tool call, and it cannot carry a body or a leak.

Three claims, each of which fails silently if it is only intended:

1. The payload carries no artifact body. `event_bus.redact_content` enforces this at publish, but
   only for the key names it reaches — so the test is that the descriptors this module builds are
   already bodiless AND that they sit under the key redaction reads.
2. `artifact_id` is None, which is what keeps one principal's query out of the event feed of
   everyone holding a grant on whatever it matched.
3. The emit covers all seven tools, and covers them because there is ONE site rather than seven.
"""
from __future__ import annotations

import re
import pathlib

import pytest

from mantle.events import event_bus, observation
from mantle.routers import mcp_router


_SRC = pathlib.Path(mcp_router.__file__).read_text(encoding="utf-8")


@pytest.fixture()
def captured(monkeypatch):
    """Every event `record_observation` publishes, with the store stubbed out.

    `_resolve_container` is stubbed rather than exercised: what these tests are about is the shape
    of the announcement, and standing up a store to obtain a container id would make a failure in
    provisioning read as a failure in redaction.
    """
    events = []
    monkeypatch.setattr(observation, "_lookup_container", lambda db, pid: "obs-container")
    monkeypatch.setattr(event_bus, "publish_event_sync", events.append)
    return events


def test_the_recording_path_never_creates_a_container(monkeypatch):
    """A read does not write. `_lookup_container` looks and gives up; provisioning is
    `ensure_observations_container`, called from `user_provisioning`. Creation on the read path
    would put a ~11s container create inside a user-facing tool call, announce it on
    the change feed as a side effect of looking, and make an observation's `create_container` call
    the one that `test_mcp_can_write_and_find` sees instead of the tool's own."""
    from mantle.services import workspace_service

    def fail(*a, **k):
        raise AssertionError("the observation path created a container")

    monkeypatch.setattr(workspace_service, "create_container", fail)
    monkeypatch.setattr(observation, "_container_cache", {})
    # No store at all: the lookup fails, and the only correct response is to record nothing.
    observation.record_observation(
        store_db=None, principal_id="unprovisioned", tool="recall", query_text="q", hits=[])


def test_an_unprovisioned_principal_records_nothing(monkeypatch):
    """The audit gap is the honest failure. Silently provisioning here is the alternative this
    module rejects."""
    published = []
    monkeypatch.setattr(observation, "_lookup_container", lambda db, pid: None)
    monkeypatch.setattr(event_bus, "publish_event_sync", published.append)
    observation.record_observation(
        store_db=None, principal_id="p1", tool="recall", query_text="q", hits=[{"id": "a"}])
    assert published == []


# ── 1. no body, ever ────────────────────────────────────────────────────────────────────

def test_a_descriptor_drops_the_body_it_was_handed():
    """`recall` hits carry `content`. The descriptor built from one must not."""
    out = observation.descriptors([
        {"id": "a1", "score": 3.0, "title": "T", "content": "PLAINTEXT",
         "content_encrypted": "CIPHERTEXT"},
    ])
    assert out == [{"id": "a1", "score": 3.0, "title": "T"}]


def test_the_published_payload_carries_no_body_anywhere(captured):
    observation.record_observation(
        store_db=None, principal_id="p1", tool="recall", query_text="secrets",
        hits=[{"id": "a1", "score": 1.0, "content": "PLAINTEXT"}],
    )
    (event,) = captured
    blob = repr(event.payload)
    assert "PLAINTEXT" not in blob
    for field in event_bus.FEED_BODY_FIELDS:
        assert field not in event.payload
        for descriptor in event.payload["artifacts"]:
            assert field not in descriptor


def test_the_result_set_sits_under_the_key_redaction_reaches(captured):
    """The key name is the defence. `redact_content` strips bodies from `artifact`/`artifacts` and
    nowhere else, so a rename to `hits`/`results` would take raw hits past it into the durable log.
    This asserts the contract between the two modules rather than one side of it."""
    observation.record_observation(
        store_db=None, principal_id="p1", tool="recall", query_text="q", hits=[{"id": "a1"}])
    (event,) = captured
    assert "artifacts" in event.payload

    # The other half: prove redaction actually reaches that key, so the assertion above is worth
    # making. A payload smuggling a body under it comes back clean.
    smuggled = event_bus.redact_content({"artifacts": [{"id": "a1", "content": "PLAINTEXT"}]})
    assert smuggled["artifacts"] == [{"id": "a1"}]


# ── 2. the query does not leak to the observed ──────────────────────────────────────────

def test_an_observation_names_no_artifact(captured):
    """`events_router._event_visible_to` asks `may_read(artifact_id)` when the event names one, and
    falls back to `container_id` only when it does not. Naming a matched artifact here would make
    this observer's query visible to every principal that can read that artifact."""
    observation.record_observation(
        store_db=None, principal_id="p1", tool="recall", query_text="my private search",
        hits=[{"id": "someone-elses-artifact"}])
    (event,) = captured
    assert event.artifact_id is None
    assert event.container_id == "obs-container"
    assert event.containers == ("obs-container",)


def test_the_observer_is_recorded_separately_from_the_machine(captured):
    """`principal_id` is whose authority looked; `actor` is which machine did. Collapsing them is
    what makes "which agent" unanswerable — the state this whole event exists to fix."""
    observation.record_observation(
        store_db=None, principal_id="p1", tool="recall", query_text="q",
        hits=[], actor="dcr_abc", via="mcp_client")
    (event,) = captured
    assert event.payload["principal_id"] == "p1"
    assert event.payload["actor"] == "dcr_abc"
    assert event.payload["via"] == "mcp_client"
    assert event.actor_id == "dcr_abc"


def test_an_unauthenticated_read_records_nothing(captured):
    """No observer, no observation — and specifically no event addressed to a container that would
    then have to be invented for a principal that does not exist."""
    observation.record_observation(store_db=None, principal_id=None, tool="recall", hits=[])
    assert captured == []


def test_recording_never_raises(captured, monkeypatch):
    """The read is the fact; this is the announcement of it. An announcement that cannot be made
    must not undo what it was announcing."""
    def boom(_event):
        raise RuntimeError("bus down")
    monkeypatch.setattr(event_bus, "publish_event_sync", boom)
    observation.record_observation(
        store_db=None, principal_id="p1", tool="recall", query_text="q", hits=[])


# ── 3. all seven tools, from one site ───────────────────────────────────────────────────

def test_every_tool_is_covered_because_there_is_one_emit_site():
    """Coverage by construction, asserted as construction.

    Seven per-tool emits would be seven places to forget one. The dispatch has a single success
    path, so a single `_observe` on it covers every tool that will ever be added — and this test
    fails if someone adds a second success return that bypasses it."""
    body = _SRC.split("if method == \"tools/call\":", 1)[1]
    # `structuredContent` is the SUCCESS shape specifically. The branch also returns via
    # `_result(...)` twice for refusals (`isError: True`), and those deliberately record nothing:
    # a refused call produced no answer to describe, and the refusal is already audited on the
    # access path.
    successes = re.findall(r'"structuredContent"', body)
    assert len(successes) == 1, (
        f"the tools/call branch has {len(successes)} success returns; an observation emitted on "
        "one of them is not coverage of the others")
    assert body.index("_observe(") < body.index('"structuredContent"'), \
        "_observe must run before the result is returned, or a crash loses the record"


def test_the_emit_names_every_declared_tool_by_covering_the_dispatch():
    """The funnel `_observe` sits on is the one every declared tool leaves through."""
    assert len(mcp_router._TOOLS_BY_NAME) == 7, "tool count changed — re-check the funnel claim"
    for tool in mcp_router._TOOLS_BY_NAME:
        assert f'name == "{tool}"' in _SRC or f"name == '{tool}'" in _SRC, \
            f"{tool} is declared but not dispatched in _call_tool"


@pytest.mark.parametrize("shape,expected", [
    ({"hits": [{"id": "a"}]}, [{"id": "a"}]),          # recall
    ({"result": [{"id": "b"}]}, [{"id": "b"}]),        # list envelope
    ([{"id": "c"}], [{"id": "c"}]),                    # bare list
    ({"id": "d"}, [{"id": "d"}]),                      # a single artifact
    ("not an artifact", []),                           # anything else contributes nothing
])
def test_results_are_read_out_of_each_tools_own_shape(shape, expected):
    assert list(mcp_router._results_of(shape)) == expected


def test_only_recall_carries_a_question():
    """The other six are addressed by id. An entry here that named one of them would put an
    argument in the `query` field that was never a query."""
    assert mcp_router._QUERY_ARG == {"recall": "query_text"}
