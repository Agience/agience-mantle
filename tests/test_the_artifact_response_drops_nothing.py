"""The declared keys are always present, and nothing else is dropped.

C9's ruling was "a response DTO with a fixed key set". Built literally, that meant enumerating
twelve keys into a `return {...}` and dropping everything else — a hand-written `response_model`,
doing exactly the filtering both the `/artifacts` and `/grants` audits refused, on the grounds that
"a model which falls behind its handler would silently stop returning a field."

What it dropped, measured: `created_by` on every create, read and update response, plus
`modified_by`, `content_ref`, `content_encrypted`, `lemmas`, `colimit_of` — and on a task artifact,
seventeen store-level fields at once (`_seq`, `_origin`, `status`, `attempts`, `dead_reason`, …).

The "twelve keys" came from sweeping combinations of `Artifact(...)` constructor arguments.
`created_by` is set as an attribute, so it never appeared in the sweep, while `to_dict` is what
actually decides a response's contents: measure the serialiser, not the constructor.

The router already says the rest, two screens above the code that broke it: "an artifact
document is open — a content type may add fields, measured on a real lattice. Typing them would be
the same drop-what-you-did-not-anticipate bug one level down."

So the guarantee is one-directional: every declared key is present (`null` when the document has
none), so a caller can rely on it without testing for existence — and every other key survives
untouched.

The cost is recorded rather than hidden: these routes cannot carry a documented envelope, because
`test_response_envelopes_match_the_handlers` reads `return {...}` literals and an open document has
none. An undeclared shape is a documentation gap; a truncated one is data loss.
"""
from __future__ import annotations

import itertools

import pytest

from mantle.entities.artifact import Artifact
from mantle.routers.artifacts_router import (
    _ARTIFACT_KEYS,
    ArtifactResponse,
    _artifact_body,
)


def test_the_key_list_is_derived_from_the_model():
    """A hand-kept second list is what let the shape drift in the first place."""
    assert tuple(ArtifactResponse.model_fields) == _ARTIFACT_KEYS


def test_the_declared_keys_are_ones_the_serialiser_actually_emits():
    """Measured against `to_dict` — the thing that decides a response — not the constructor.

    That distinction is the whole correction: sweeping constructor arguments is what missed
    `created_by` and produced a "fixed key set" that dropped it."""
    a = Artifact(id="x", state="committed", name="n", description="d", content="c",
                 content_type="t", collection_id="col", context="{}")
    a.created_by = "u"
    a.modified_by = "u"
    emitted = set(a.to_dict())
    unknown = set(_ARTIFACT_KEYS) - emitted - {"origin_root"}
    assert not unknown, (
        "declared keys the serialiser never emits for a populated artifact: %s" % sorted(unknown))


@pytest.mark.parametrize("doc", [
    {"id": "a"},
    {"id": "a", "state": "committed"},
    {"id": "a", "name": "n", "description": "d", "content_type": "t"},
    {},
])
def test_every_declared_key_is_always_present(doc):
    body = _artifact_body(doc)
    missing = set(_ARTIFACT_KEYS) - set(body)
    assert not missing, sorted(missing)


def test_an_unset_declared_field_is_null_rather_than_missing():
    body = _artifact_body({"id": "a"})
    for k in ("name", "description", "content_type", "origin_root"):
        assert k in body, "%s is absent; a caller cannot tell 'no value' from 'not sent'" % k
        assert body[k] is None, (k, body[k])


@pytest.mark.parametrize("extra", [
    {"created_by": "u-1"},
    {"modified_by": "u-1", "content_ref": "cas://x"},
    {"lemmas": ["a", "b"], "colimit_of": ["c"]},
    {"_seq": 7, "_origin": "71", "status": "dead", "attempts": 3},
])
def test_a_key_the_model_does_not_declare_SURVIVES(extra):
    """The defect this pins: these fields must survive. `created_by` is on every artifact; the last
    row is a real task artifact, which carries seventeen store-level fields."""
    body = _artifact_body(dict({"id": "a"}, **extra))
    for k, v in extra.items():
        assert k in body, "%s was dropped from the response" % k
        assert body[k] == v, (k, body[k], v)


def test_the_computed_read_fields_survive_too():
    """`read_artifact` adds these before serialising; they are not on any write's model."""
    body = _artifact_body({"id": "a", "has_children": True, "child_count": 4})
    assert body["has_children"] is True and body["child_count"] == 4


def test_nothing_is_lost_for_any_combination_the_entity_can_produce():
    """The general form: whatever `to_dict` emits, the response still carries."""
    opts = dict(name="n", description="d", content="c", content_type="t",
                collection_id="col", context="{}")
    for r in range(len(opts) + 1):
        for combo in itertools.combinations(opts, r):
            a = Artifact(id="x", state="committed", **{k: opts[k] for k in combo})
            a.created_by = "u"
            d = a.to_dict()
            body = _artifact_body(d)
            lost = {k for k in d if k not in body}
            assert not lost, "lost %s for combination %s" % (sorted(lost), combo)
