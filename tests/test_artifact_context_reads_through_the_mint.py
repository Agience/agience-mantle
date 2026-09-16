"""The caller's own context keys survive a create one level down, and readers must find them.

⛔ WHAT WAS WRONG, MEASURED AGAINST PRODUCTION 2026-09-11. `artifacts_router._mint_context` stamps
a context onto EVERY create and keeps the caller's own "verbatim under `caller`" — its stated
contract. `parse_artifact_context` read only the top level, so every consumer asking for a
caller-supplied key by name silently began answering nothing the day minting went live. No error
and no log line: the key is simply absent.

Three consequences were already live, and none of them looked like a failure:

    iris `notify_inbound`   ctx["source"] -> None for 58 of 67 lead artifacts, so the operator's
                            notification called every one of them "unknown", and company, role,
                            interest, email_domain and lead_id went with it
    iris `_resolve_field`   resolves the RECIPIENT of a templated email out of context
    astra `_find_child`     matches `ctx["title"]` for IDEMPOTENT ingest — a miss does not raise,
                            it forks the series, "splitting one symbol's history across two
                            collections" exactly as its own docstring warns

⚠ BOTH SHAPES ARE LIVE IN ONE CORPUS. `PATCH` does not mint, so patched and pre-mint artifacts
keep their keys flat — 9 of those 67 leads do. That is why this merges rather than switching, and
why the top level has to win: a PATCH is a deliberate statement, the mint is a record of a create.
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
        """Deliberately NOT swallowed. `_mint_context` keeps an opaque context under
        `caller._opaque` rather than dropping it, so a reader that quietly returned {} here would
        turn the caller's own statement about its artifact into silence."""
        with pytest.raises(json.JSONDecodeError):
            parse_artifact_context({"context": "not json at all"})

    def test_content_type_resolves_through_the_mint(self):
        """`get_artifact_content_type` reads `context.content_type`, which the mint nests too."""
        assert get_artifact_content_type(
            {"context": json.dumps({"caller": {"content_type": "text/csv; charset=utf-8"}})}
        ) == "text/csv"


class TestTheRealProductionShape:
    """The exact context a live lead carries, so the fix is pinned to real data not a fixture.

    Taken from `/opt/agience/brain/lattice.db` on 2026-09-11 (person details redacted; only the
    SHAPE is asserted). Two things here are easy to get wrong and neither shows up in an invented
    fixture.
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
        """⛔ THE TRAP. `minted.source` is "request" — the mint's provenance, not the lead's.

        A flatten that merged every nested block rather than only `caller` collides on this key,
        and the collision resolves by DICT ORDER — so the same data answers differently depending
        on which block the producer happened to write first:

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
