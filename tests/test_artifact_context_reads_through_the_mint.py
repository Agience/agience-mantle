"""`parse_artifact_context` finds a caller's context keys wherever a stored context holds them.

Mantle stores a caller's keys at the top level, beside the store's own facets (`minted_by`,
`minted`, `addressing`, `provenance`, …). A row that no write through `mint_context` has touched
can also hold them under `caller`, alone or beside some at the top level, and a top-level read of
such a row finds nothing for a caller-supplied key, with no error and no log line. Measured
against production, 58 of 67 lead artifacts held their keys under `caller` and 9 at the top level.

The readers that depend on finding them:

    iris `notify_inbound`   prints `source`, company, role, interest, email_domain and lead_id;
                            a missed key prints "unknown"
    iris `_resolve_field`   resolves the recipient of a templated email out of context
    astra `_find_child`     matches `ctx["title"]` for idempotent ingest; a miss does not raise,
                            it forks the series, "splitting one symbol's history across two
                            collections" as its own docstring states

Both shapes are live in one corpus, so the reader merges rather than switching: the top level wins,
and `caller` supplies a key the top level does not hold (`mint_context.client_view`).
"""
from __future__ import annotations

import json

import pytest

from mantle.clients.artifact_helpers import (
    get_artifact_content_type,
    parse_artifact_context,
)


class TestParseArtifactContextReadsThroughTheMint:
    def test_the_minted_shape_a_create_produces(self):
        art = {"context": json.dumps({
            "addressing": {}, "minted_by": "mantle.mint_context", "provenance": "unknown",
            "caller": {"source": "website-contact", "lead_id": "abc"}})}
        ctx = parse_artifact_context(art)
        assert ctx["source"] == "website-contact"
        assert ctx["lead_id"] == "abc"

    def test_the_flat_shape_reads_exactly_as_before(self):
        art = {"context": json.dumps({"source": "website-contact"})}
        assert parse_artifact_context(art) == {"source": "website-contact"}

    def test_a_deliberate_patch_overrides_the_mint(self):
        art = {"context": {"source": "patched", "caller": {"source": "minted"}}}
        assert parse_artifact_context(art)["source"] == "patched"

    def test_the_stores_own_facets_stay_reachable(self):
        """Merging must not hide what the store observed; provenance is read elsewhere."""
        ctx = parse_artifact_context(
            {"context": {"provenance": "unknown", "caller": {"title": "AAPL"}}})
        assert ctx["provenance"] == "unknown"
        assert ctx["title"] == "AAPL"

    def test_nothing_is_invented_without_a_caller_block(self):
        """The negative case, or the lookup would appear to work by always finding something."""
        assert parse_artifact_context({"context": {"addressing": {}}}) == {"addressing": {}}
        assert parse_artifact_context({"context": {}}) == {}
        assert parse_artifact_context({}) == {}

    def test_a_non_dict_caller_does_not_raise(self):
        """`caller` is caller-influenced, so a string or list there must not explode a reader."""
        assert parse_artifact_context(
            {"context": {"caller": "oops", "source": "flat"}})["source"] == "flat"
        assert parse_artifact_context({"context": {"caller": []}}) == {"caller": []}

    def test_a_non_json_string_context_still_raises(self):
        """Not swallowed. A non-JSON context is the caller's own statement about its artifact,
        which the store keeps under `_opaque`; a reader that returned {} here would turn it into
        silence."""
        with pytest.raises(json.JSONDecodeError):
            parse_artifact_context({"context": "not json at all"})

    def test_content_type_resolves_through_the_mint(self):
        """`get_artifact_content_type` reads `context.content_type`, one under `caller` included."""
        assert get_artifact_content_type(
            {"context": json.dumps({"caller": {"content_type": "text/csv; charset=utf-8"}})}
        ) == "text/csv"


class TestTheRealProductionShape:
    """A real lead context with its keys under `caller`.

    Taken from `/opt/agience/brain/lattice.db` on 2026-09-11, with person details redacted; only
    the shape is asserted. Two of its properties do not show up in an invented fixture.
    """

    #: A real minted lead context, redacted. `minted.source` is the MINT's own provenance
    #: ("request"), and it sits beside the caller's `source` ("website-contact").
    REAL = {
        "minted_by": "mantle.mint_context",
        "minted": {"at": "2026-09-10T17:13:15.563340+00:00",
                   "principal_id": "d6558e32-9cc7-45f0-b0cb-67f107ad7513",
                   "principal_type": "user", "source": "request"},
        "placement": {"collection_id": "b1fefeae-491a-4cc4-a807-311ee8a04261"},
        "addressing": {"content_type": "application/vnd.agience.lead+json", "bytes": 205},
        "caller": {"source": "website-contact", "type": "lead", "status": "new",
                   "role": "<redacted>", "company": "<redacted>", "interest": "<redacted>"},
    }

    def test_the_callers_source_is_what_comes_back(self):
        ctx = parse_artifact_context({"context": self.REAL})
        assert ctx["source"] == "website-contact"

    def test_the_mints_own_source_never_shadows_the_callers(self):
        """`minted.source` is "request": the mint's provenance, not the lead's.

        A flatten that merged every nested block rather than only `caller` collides on this key,
        and the collision resolves by dict order, so the same data answers differently depending
        on which block the producer wrote first:

            full-flatten, minted first -> 'website-contact'
            full-flatten, caller first -> 'request'        <- same artifact, different answer

        Not reliably wrong, which is worse than wrong: it would read correctly in a test and drift
        in production the day the mint changed its key order. Merging only `caller` has no such
        dependency, and the mint's facets stay where the store put them.
        """
        ctx = parse_artifact_context({"context": self.REAL})
        assert ctx["source"] == "website-contact"
        assert ctx["minted"]["source"] == "request"

    def test_every_lead_field_the_notification_prints_is_reachable(self):
        ctx = parse_artifact_context({"context": self.REAL})
        for field in ("source", "type", "status", "role", "company", "interest"):
            assert field in ctx, f"{field} is not reachable — the notification prints it"

    def test_the_stores_own_facets_survive_the_merge(self):
        ctx = parse_artifact_context({"context": self.REAL})
        assert ctx["addressing"]["content_type"] == "application/vnd.agience.lead+json"
        assert ctx["placement"]["collection_id"] == "b1fefeae-491a-4cc4-a807-311ee8a04261"
        assert ctx["minted_by"] == "mantle.mint_context"
