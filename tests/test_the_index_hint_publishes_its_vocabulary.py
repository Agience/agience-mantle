"""`index` publishes the two values it accepts, and says where it applies, C4).

C5. `index` was a free `string` in the spec. `resolve_lazy` compares against `"eager"` and
`"lazy"` and returns the deployment default for anything else, so `"eger"` was not an error — it
silently produced the default. The parameter's own description warned about that, which makes the
fall-through documented rather than silent; what was NOT knowable from the spec was the vocabulary
itself. Publishing it removes the typo where it originates, in the client.

The runtime acceptance was deliberately left alone. The fall-through is a stated decision that
predates this pass; reversing it is a contract change and is John's call, not a tidy-up. This test
pins the published vocabulary, not the leniency — so a later decision to refuse unknown values does
not have to fight it.

C4. The description said only that top-level containers are always eager. `identity` members are
too, for a different reason: `upsert_identity_member` calls
`create_workspace_artifact(enqueue_index=False)` and then indexes synchronously, because an
identity artifact is a mirror whose purpose is to be found. `parsed.index` is therefore read at
exactly ONE call site and ignored on both other branches — by design, and now stated.
"""
from __future__ import annotations

import pytest

from mantle.main import app
from mantle.search.lazy import INDEX_HINTS, resolve_lazy


@pytest.fixture(scope="module")
def index_schema():
    sch = app.openapi()["components"]["schemas"]["CreateArtifactRequest"]
    return sch["properties"]["index"]


def test_the_enum_is_published_and_derived(index_schema):
    assert index_schema.get("enum") == list(INDEX_HINTS), (
        "index publishes enum=%r but resolve_lazy accepts %r"
        % (index_schema.get("enum"), list(INDEX_HINTS)))


def test_the_description_says_where_the_hint_applies(index_schema):
    """C4: it is ignored on two of three branches, and both must be named."""
    d = index_schema.get("description") or ""
    assert "identity" in d, "the description does not say identity members ignore the hint: %r" % d
    assert "top-level" in d.lower(), "the description does not say top-level artifacts do: %r" % d


def test_resolve_lazy_honours_exactly_the_published_values():
    """The published enum must be the set that actually changes behaviour, not a superset."""
    outcomes = {h: resolve_lazy(h) for h in INDEX_HINTS}
    assert len(set(outcomes.values())) == len(INDEX_HINTS), (
        "two published hints produce the same result, so one of them is decorative: %r" % outcomes)


def test_an_unknown_hint_still_falls_through(index_schema):
    """Pins the DOCUMENTED leniency, so tightening it is a deliberate act that fails here first."""
    from mantle.search.lazy import lazy_index_default
    assert resolve_lazy("eger") == lazy_index_default()
    assert "not refused" in (index_schema.get("description") or ""), (
        "the fall-through is no longer documented on the parameter")
