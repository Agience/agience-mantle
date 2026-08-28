"""Caller-supplied counts on the artifacts surface are bounded.

What was unbounded, and why it mattered. `_fetch_authorized_docs` states its own cost: *"the body
is two store operations per id"*. `BatchFetchRequest.artifact_ids` had no length bound, so ONE
request bought an unbounded number of store reads — and `main.py`'s per-client rate limiter counts
that request as one against its 600/min, so it bounds how many calls arrive and not what a single
call costs. `recall.size` and `candidate_budget` reached the search path unclamped the same way,
and `from_` accepted a negative offset.

The ceilings are GENEROUS on purpose — 10x this router's own `le=1000` query idiom. The job is to
remove the unbounded case, not to tune a page size, so the tests below assert BOTH directions: the
pathological value is refused AND the plausible one still passes. A bound that refused ordinary
traffic would be a worse defect than the one it replaced.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from mantle.routers import artifacts_router as ar


# ── refused ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model, kwargs, why", [
    (ar.ArtifactRecallRequest, {"size": 10 ** 9}, "recall.size"),
    (ar.ArtifactRecallRequest, {"candidate_budget": 10 ** 9}, "recall.candidate_budget"),
    (ar.ArtifactRecallRequest, {"from_": -1}, "a negative offset"),
    (ar.BatchFetchRequest, {"artifact_ids": ["a"] * (ar._MAX_PAGE + 1)}, "batch length"),
    (ar.BatchFetchRequest, {"artifact_ids": []}, "an empty batch"),
])
def test_a_pathological_value_is_refused(model, kwargs, why):
    with pytest.raises(ValidationError):
        model(**kwargs)


def test_an_upload_larger_than_the_cipher_can_encrypt_is_refused():
    """The ceiling is the cipher's, not a tuned number: `PUT content` already answers 413 above
    2**31 - 1 because AES-GCM accepts no more per message and that route encrypts the body whole.
    `upload-initiate` declares the same bound rather than inventing a second one."""
    with pytest.raises(ValidationError):
        ar.UploadInitiateRequest(size=2 ** 31, filename="x", content_type="text/plain")
    assert ar._MAX_CONTENT_BYTES == 2 ** 31 - 1


# ── still accepted ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("model, kwargs, why", [
    (ar.ArtifactRecallRequest, {}, "the defaults"),
    (ar.ArtifactRecallRequest, {"size": 1000, "from_": 0}, "a 1000-hit page (the old idiom's cap)"),
    (ar.ArtifactRecallRequest, {"candidate_budget": 1000}, "a 1000-candidate budget"),
    (ar.BatchFetchRequest, {"artifact_ids": ["a"] * 5000}, "a 5000-id batch"),
])
def test_a_plausible_request_is_unchanged(model, kwargs, why):
    """The inverted guard. A ceiling that refused ordinary traffic would be worse than no ceiling,
    and every test above would still pass."""
    model(**kwargs)


def test_a_ten_megabyte_upload_is_unchanged():
    ar.UploadInitiateRequest(size=10 * 1024 * 1024, filename="x", content_type="text/plain")


# ── the trap ─────────────────────────────────────────────────────────────────────────────────

def test_the_RESPONSE_model_echoing_from_and_size_is_NOT_bounded():
    """`ArtifactRecallResponse` declares `from_: int = 0` and `size: int` — textually identical to
    the request fields, in the same file.

    A string-based edit bounds these too, and a bound on a RESPONSE field does not refuse a bad
    request: it makes the server fail to serialise a legitimate echo, turning a caller's oversized
    request into a 500 on the way back out. Pinned because the next person editing these will meet
    the same two-matches problem."""
    fields = ar.ArtifactRecallResponse.model_fields
    for name in ("from_", "size"):
        assert not fields[name].metadata, (
            "%s on the RESPONSE model has acquired constraints (%r) — it echoes what was asked "
            "for and must serialise whatever the request carried"
            % (name, fields[name].metadata))


def test_the_request_fields_really_do_carry_the_bounds():
    """The other half of the pair above: if a refactor moved the constraints off the request
    models, every 'refused' test would still pass only if the model rejected for some other
    reason. This asserts the mechanism, not the symptom."""
    req = ar.ArtifactRecallRequest.model_fields
    assert req["size"].metadata, "recall.size lost its bounds"
    assert req["from_"].metadata, "recall.from_ lost its floor"
    assert ar.BatchFetchRequest.model_fields["artifact_ids"].metadata, "batch lost its length bound"
