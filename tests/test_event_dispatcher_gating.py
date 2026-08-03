"""`services/event_dispatcher` — what decides that a handler FIRES. Previously untested.

Added to the security audit at John's request (2026-07-29). `event_dispatcher` ranked near-zero on
the keyword-density survey — it has no `grant`/`token`/`decrypt` vocabulary — which is exactly why
it was worth a second look: **its security surface is not authorization, it is ACTIVATION.** A
handler artifact in a workspace causes `chorus_client.call_tool` to run with a signed delegation
JWT. Whatever decides "is this a handler, and does it match?" decides what executes.

⛔ THE DEFAULT IS ENABLED, AND THE ENABLED CHECK IS `is not False`:

    ctx.get("type") == "workspace-event-handler" and ctx.get("enabled", True) is not False

Two distinct behaviours ride on that line, and both are pinned below because both are easy to
"tidy" into something different:

  1. **Absent `enabled` means ON.** Writing a handler artifact is enough to make it fire.
  2. **Only the literal `False` turns it off.** `"false"` (a string — what a form post, a YAML
     round-trip, or a hand-edited JSON field most naturally produces), `0`, and `None` ALL leave the
     handler live. Someone disabling a handler by setting `"enabled": "false"` gets no error and no
     effect.

These are recorded as the behaviour that exists, not endorsed. (2) in particular is the shape that
bites: it fails OPEN, silently, on the most plausible way to get it wrong.
"""
from __future__ import annotations

import pytest

from mantle.services.event_dispatcher import (_is_handler, _matches_event_types,
                                       _matches_source, _replace_templates,
                                       _resolve_json_path)

HANDLER = "workspace-event-handler"


# ── is this a handler at all ─────────────────────────────────────────────────
def test_a_non_handler_artifact_never_fires():
    assert _is_handler({"type": "something-else", "enabled": True}) is False
    assert _is_handler({}) is False


def test_absent_enabled_means_ENABLED():
    """Writing the artifact is enough. Pinned so nobody flips the default to off-by-default without
    noticing that every existing handler stops firing."""
    assert _is_handler({"type": HANDLER}) is True


def test_only_the_literal_False_disables_a_handler():
    """⚠ `is not False` is an IDENTITY check. The falsy-looking values below all leave the handler
    LIVE — including the string "false", which is what a form post or a YAML round-trip produces.
    Disabling a handler that way silently does nothing."""
    assert _is_handler({"type": HANDLER, "enabled": False}) is False      # the only off switch
    for still_on in ("false", "False", 0, "", None, [], "no"):
        assert _is_handler({"type": HANDLER, "enabled": still_on}) is True, (
            "%r now disables a handler — if that was deliberate, this test should be rewritten; "
            "if not, a handler someone believed was off is firing" % (still_on,))


# ── matching: an absent filter matches EVERYTHING ────────────────────────────
def test_no_event_type_filter_matches_every_event():
    """Another open default: a handler with no `event_types` fires on all of them."""
    assert _matches_event_types({}, "anything") is True
    assert _matches_event_types({"on": {}}, "anything") is True


def test_an_event_type_filter_is_exact():
    h = {"on": {"event_types": ["upload_complete"]}}
    assert _matches_event_types(h, "upload_complete") is True
    assert _matches_event_types(h, "upload_completed") is False      # no prefix matching
    assert _matches_event_types(h, "UPLOAD_COMPLETE") is False       # case-sensitive


def test_a_non_list_event_type_filter_matches_NOTHING():
    """Fail-closed on a malformed filter — a bare string would otherwise `in`-match substrings
    (`"complete" in "upload_complete"`), firing handlers on events they never named."""
    assert _matches_event_types({"on": {"event_types": "upload_complete"}}, "upload_complete") is False


def test_no_source_filter_matches_everything_but_a_filter_needs_a_source():
    assert _matches_source({}, None) is True                          # no filter at all
    assert _matches_source({"on": {"source": {"content_type": "text/markdown"}}}, None) is False


def test_every_declared_source_facet_must_match():
    f = {"on": {"source": {"context_type": "recording", "content_type": "audio/wav",
                           "require_transcript_status": "done"}}}
    ok = {"type": "recording", "content_type": "audio/wav", "transcript": {"status": "done"}}
    assert _matches_source(f, ok) is True
    for key, bad in (("type", "note"), ("content_type", "audio/mp3")):
        assert _matches_source(f, {**ok, key: bad}) is False
    assert _matches_source(f, {**ok, "transcript": {"status": "pending"}}) is False


# ── template substitution: the value that reaches call_tool ──────────────────
def test_an_unknown_template_is_left_literal_not_blanked():
    """`{{missing}}` stays as-is rather than becoming "". Blanking would silently turn a typo'd
    variable into an empty argument, which a downstream tool may read as "all" or "default"."""
    assert _replace_templates("{{nope}}", {}) == "{{nope}}"
    assert _replace_templates("{{a}}-{{nope}}", {"a": "x"}) == "x-{{nope}}"


def test_substitution_reaches_nested_structures_but_does_not_evaluate():
    out = _replace_templates({"a": ["{{v}}", {"b": "{{v}}"}], "c": 7}, {"v": "V"})
    assert out == {"a": ["V", {"b": "V"}], "c": 7}


def test_a_substituted_value_is_inserted_verbatim():
    """No re-parsing of the result: a value containing template syntax is NOT expanded again, so a
    hostile event field cannot inject a second round of substitution."""
    assert _replace_templates("{{v}}", {"v": "{{other}}"}) == "{{other}}"


# ── json-path resolution ─────────────────────────────────────────────────────
def test_json_path_requires_the_dollar_prefix():
    assert _resolve_json_path("plain", {"plain": "x"}) == "plain"      # returned unchanged


def test_json_path_walks_dicts_and_indexes_lists():
    v = {"a": {"b": [{"c": 1}, {"c": 2}]}}
    assert _resolve_json_path("$.a.b[1]", v) == {"c": 2}


def test_a_malformed_or_missing_json_path_yields_None_not_an_exception():
    """This runs on the dispatch path; raising would take down event handling for every handler in
    the workspace, not just the one with the bad expression."""
    for expr in ("$.a..b", "$.a[x]", "$.nope.deeper", "$.a[9]"):
        assert _resolve_json_path(expr, {"a": {"b": [1]}}) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
