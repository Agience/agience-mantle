"""A marker written into every field a person supplies is found nowhere the node stores or logs.

The ruling (John, 2026-10-29): everything related to content is encrypted, and a grant is the ability
to decrypt. Structure is the named exception (`doc_boundary.STRUCTURAL_CONTEXT_KEYS`). So a word the
writer chose may reach the node's files, the event log and the service log only as ciphertext or as a
keyed token. It is the canary's assertion run as a test: write the marker everywhere, then read every
byte the node wrote.
"""
from __future__ import annotations

import json
import logging

import pytest

from fastapi.responses import Response
from starlette.requests import Request

from mantle.db import lattice_api
from mantle.routers import artifacts_router as artifacts
from mantle.services import grant_service
from mantle.services import workspace_service as ws
from mantle.services.acting_principal import acting_as

from .test_a_title_is_content_and_sealed_at_rest import _auth
from .test_there_is_no_promotion_state import USER, _recall, real_node

MARK = "zq7marker"


async def _write_everywhere(db, *, links_and_grants: bool) -> None:
    req = Request({"type": "http", "method": "POST", "headers": [], "path": "/artifacts", "query_string": b""})
    if True:  # one scope for every write, as one person
        with acting_as(USER, principal_type="user"):
            shelf = ws.create_container(db, USER, content_type="application/vnd.agience.collection+json",
                                        name=f"shelf {MARK}", description=f"about {MARK}").id
            note = await artifacts.create_artifact(
                request=req, response=Response(), auth=_auth(), store_db=db,
                body=artifacts.CreateArtifactRequest(
                    container_id=shelf, content=f"body {MARK} text",
                    context=json.dumps({"title": f"title {MARK}", "description": f"desc {MARK}",
                                        "tags": [f"tag{MARK}"], "filename": f"{MARK}.md",
                                        "custom_key": f"custom {MARK}", "kind": "note"})))
            other = await artifacts.create_artifact(
                request=req, response=Response(), auth=_auth(), store_db=db,
                body=artifacts.CreateArtifactRequest(container_id=shelf, content="second",
                                                     context=json.dumps({"title": "second"})))
            if not links_and_grants:
                return
            lattice_api.add_artifact_to_collection(db, note.get("root_id") or note["id"],
                                                   other.get("root_id") or other["id"],
                                                   relationship=f"rel{MARK}")
            grant_service.create_invite(db, user_id=USER, resource_id=shelf,
                                        target_email=f"{MARK}@example.invalid",
                                        name=f"invite {MARK}", notes=f"notes {MARK}")


def _files_holding(root, marker: str) -> list:
    hits = []
    needle = marker.encode("utf-8")
    for path in root.rglob("*"):
        if path.is_file():
            try:
                if needle in path.read_bytes():
                    hits.append(str(path.relative_to(root)))
            except OSError:
                continue
    return sorted(hits)


async def _leaks(tmp_path, monkeypatch, caplog, *, links_and_grants: bool) -> dict:
    db = real_node(tmp_path, monkeypatch)
    caplog.set_level(logging.DEBUG)
    await _write_everywhere(db, links_and_grants=links_and_grants)
    _recall(db, f"{MARK} body")
    return {"files": _files_holding(tmp_path, MARK),
            "loggers": sorted({r.name for r in caplog.records if MARK in r.getMessage()})}


async def test_every_artifact_field_is_found_nowhere_at_rest_nor_in_the_log(tmp_path, monkeypatch, caplog):
    """Title, name, description, tags, filename, a custom key, the body and a search for them."""
    leaks = await _leaks(tmp_path, monkeypatch, caplog, links_and_grants=False)
    assert not leaks["files"] and not leaks["loggers"], leaks


@pytest.mark.xfail(strict=True, reason=(
    "open, 2026-10-30: a relationship label is plaintext in `edge.label` and `counter.name`, and an "
    "invite's email, name and notes are plaintext on the grant (MANTLE-SEAL-EVERYTHING.md). Strict, "
    "so this fails the suite the day they are sealed and the marker has to come off."))
async def test_a_link_label_and_an_invite_are_found_nowhere_at_rest(tmp_path, monkeypatch, caplog):
    leaks = await _leaks(tmp_path, monkeypatch, caplog, links_and_grants=True)
    assert not leaks["files"] and not leaks["loggers"], leaks

