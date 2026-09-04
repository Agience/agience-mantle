"""–P5: `PATCH .../upload-status` invented a status, clamped a fraction, and shipped a
`parts` list with no item schema.

The sharpest one (P4). The router sent `body.status or "uploading"`, so an empty `PATCH {}`
wrote `uploading` — and the service assigns whatever it is given, so on an upload that had already
COMPLETED that silently reverted it. **A PATCH that names no field must change no field**; that is
what PATCH means, and the `or` turned an omission into an instruction.

P2 is the same shape one level down: `progress` was unbounded and the service clamps with
`max(0.0, min(1.0, progress))`, so a caller working in PERCENT sent `50` and recorded `1.0` — a
completed-looking upload on its first progress report, with no error. A clamp cannot tell a caller
they used the wrong scale; a `422` naming the range can.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from fastapi import HTTPException

from mantle.db import lattice_api as store
from mantle.entities.artifact import Artifact as ArtifactEntity, WORKSPACE_CONTENT_TYPE
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as


# `acting_as` is not decoration. `update_artifact` issues key material and refuses without an
# acting principal — on a request path the auth dependency installs one, and a test that calls the
# service directly is NOT the router. Skipping it is the harness error that produced a uniform
# failure across an entire corpus earlier this week and was very nearly filed as a store defect.

@pytest.fixture
def db():
    return store.LatticeDatabase(os.path.join(tempfile.mkdtemp(), "up.db"), origin="upload-test")


@pytest.fixture
def upload(db):
    """An artifact inside a container, whose upload has already COMPLETED."""
    import json

    cid = ws.create_container(db, "u", content_type=WORKSPACE_CONTENT_TYPE, name="c").id
    aid = "upload-1"
    ctx = {"upload": {"status": "complete", "s3_key": "k", "mode": "single", "progress": 1.0}}
    store.create_artifact(db, ArtifactEntity(
        id=aid, root_id=aid, collection_id=cid, created_by="u",
        state=ArtifactEntity.STATE_COMMITTED, name=aid,
        context=json.dumps(ctx)))
    return cid, aid


def _upload_of(result):
    """The upload block of the entity the service RETURNED.

    Not a re-read by id: this store is versioned, so `update_artifact` writes a NEW version and
    the original id still resolves to the context as it was. The router serialises exactly this
    returned entity back to the caller, so it is also what the caller sees."""
    import json

    raw = getattr(result, "context", None)
    ctx = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return ctx.get("upload") or {}


def test_a_patch_with_no_status_leaves_the_status_alone(db, upload):
    """A patch that omits `status` leaves a completed upload completed."""
    cid, aid = upload
    with acting_as("u", principal_type="user"):
        result = ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=aid,
                                         status_value=None, progress=0.5)

    after = _upload_of(result)
    assert after["status"] == "complete", "an omitted status reverted a completed upload"
    assert after["progress"] == 0.5, "the fields that WERE named must still be applied"


def test_a_named_status_is_still_applied(db, upload):
    """The inverted guard — if nothing were written, the test above would pass on a no-op."""
    cid, aid = upload
    with acting_as("u", principal_type="user"):
        result = ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=aid,
                                         status_value="failed")
    assert _upload_of(result)["status"] == "failed"


def test_an_unknown_status_is_still_refused(db, upload):
    """`None` means 'leave it alone'; it must not have become a hole in the vocabulary check."""
    cid, aid = upload
    with pytest.raises(HTTPException) as exc, acting_as("u", principal_type="user"):
        ws.update_upload_status(db=db, user_id="u", workspace_id=cid, upload_id=aid,
                                status_value="finished")
    assert exc.value.status_code == 400
    assert "finished" in str(exc.value.detail)


def test_progress_is_bounded_at_the_boundary_not_clamped_in_the_service():
    """P2. The bound belongs where a caller can be TOLD about it."""
    from pydantic import ValidationError

    from mantle.routers.artifacts_router import UploadStatusRequest

    assert UploadStatusRequest(progress=0.5).progress == 0.5
    for bad in (50, -5, 1.5):
        with pytest.raises(ValidationError):
            UploadStatusRequest(progress=bad)


def test_a_part_needs_both_halves_and_tolerates_extras():
    """P3. The shape is not ours to invent — it is what the object store hands back."""
    from pydantic import ValidationError

    from mantle.routers.artifacts_router import UploadPart

    p = UploadPart(PartNumber=1, ETag="abc", ChecksumSHA256="whatever")
    assert p.PartNumber == 1 and p.ETag == "abc"
    with pytest.raises(ValidationError):
        UploadPart(PartNumber=1)          # no ETag: completion cannot work
    with pytest.raises(ValidationError):
        UploadPart(PartNumber=0, ETag="a")  # parts are 1-based


def test_the_router_passes_the_omission_through_rather_than_defaulting_it():
    """P4 lives in the ROUTER, and the service tests above cannot see it.

    Written after noticing exactly that: reverting the router to `body.status or "uploading"`
    left every test in this file green, because they call the service directly. A fix needs a
    guard on the layer that had the defect.

    Asserted on the CALL NODE, not on the file's text: the question is whether the argument is
    `body.status` itself or an expression that substitutes something when it is falsy, and that is
    a property of the syntax tree rather than of how the line happens to be spelled."""
    import ast
    import io as _io

    from mantle.routers import artifacts_router

    tree = ast.parse(_io.open(artifacts_router.__file__, encoding="utf-8").read())
    passed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == "status_value":
                passed.append(kw.value)

    assert passed, "no call passes `status_value` — the scan is looking in the wrong place"
    for value in passed:
        assert not isinstance(value, ast.BoolOp), (
            "`status_value` is passed through an `or`, so an omitted status becomes an "
            "instruction — an empty PATCH would write a status the caller never sent")
        assert isinstance(value, ast.Attribute) and value.attr == "status", (
            "`status_value` should be the request's own `status`, unmodified; got %s"
            % ast.dump(value)[:80])
