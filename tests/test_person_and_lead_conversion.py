"""Step 3 identity linkage: Person artifact at first-login + lead conversion.

Covers `_ensure_person_artifact` (deterministic id, agience_root_id in context,
idempotent) and `_convert_leads_for_person` (email match, skips non-matching and
already-converted leads).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import mantle.services.seed_provisioning.user_provisioning as up  # noqa: E402
from mantle.services.seed_provisioning.loader import UserContext  # noqa: E402


# ---------------------------------------------------------------------------
# Person artifact
# ---------------------------------------------------------------------------

def test_ensure_person_artifact_creates(monkeypatch):
    created: list = []
    edges: list = []
    grants: list = []
    # People collection is ensured at runtime; the card is homed there, NOT the inbox.
    monkeypatch.setattr(up, "_ensure_people_collection", lambda db: "people-col")
    monkeypatch.setattr(up, "ensure_authority_collection", lambda db: "authority-col")
    monkeypatch.setattr(up, "db_get_artifact", lambda db, pid: None)
    monkeypatch.setattr(up, "db_create_artifact", lambda db, e: created.append(e))
    monkeypatch.setattr(up, "db_get_edge", lambda db, c, ch: None)
    monkeypatch.setattr(up, "db_add_artifact_to_collection",
                        lambda db, c, ch, **kw: edges.append((c, ch)))
    monkeypatch.setattr(up, "db_get_grants_for_principal_resource", lambda db, g, r: [])
    monkeypatch.setattr(up, "db_create_grant", lambda db, g: grants.append(g))

    ctx = UserContext(id="u-1", email="jane@x.com", name="Jane", inbox_id="inbox-1")
    up._ensure_person_artifact(MagicMock(), ctx)

    assert len(created) == 1
    e = created[0]
    assert e.content_type == up.PERSON_CONTENT_TYPE
    assert e.id == up.person_artifact_id("u-1")
    assert e.id != "u-1"  # not == user_id (avoids personal-collection collision)
    ctxd = json.loads(e.context)
    assert ctxd["identity"]["agience_root_id"] == "u-1"
    assert ctxd["display_name"] == "Jane"
    assert ctxd["email"] == "jane@x.com"
    # Homed in People, not the inbox.
    assert edges == [("people-col", e.id)]
    assert e.collection_id == "people-col"
    # Owner gets read + update on their own card.
    assert len(grants) == 1
    assert grants[0].resource_id == e.id
    assert grants[0].grantee_id == "u-1"
    assert grants[0].can_update is True


def test_ensure_person_artifact_idempotent(monkeypatch):
    created: list = []
    monkeypatch.setattr(up, "_ensure_people_collection", lambda db: "people-col")
    monkeypatch.setattr(up, "ensure_authority_collection", lambda db: "authority-col")
    monkeypatch.setattr(up, "db_get_artifact", lambda db, pid: object())  # already exists
    monkeypatch.setattr(up, "db_create_artifact", lambda db, e: created.append(e))
    # Already homed in People, no legacy inbox edge → migrate is a no-op.
    monkeypatch.setattr(up, "db_get_edge",
                        lambda db, c, ch: object() if c == "people-col" else None)
    monkeypatch.setattr(up, "db_add_artifact_to_collection", lambda db, c, ch, **kw: None)
    # Owner grant already present → no duplicate.
    monkeypatch.setattr(up, "db_get_grants_for_principal_resource",
                        lambda db, g, r: [SimpleNamespace(can_update=True)])
    monkeypatch.setattr(up, "db_create_grant",
                        lambda db, g: (_ for _ in ()).throw(AssertionError("should not create")))
    up._ensure_person_artifact(MagicMock(), UserContext(id="u-1", inbox_id="inbox-1"))
    assert created == []


def test_person_display_name_falls_back_to_email_local_part(monkeypatch):
    created: list = []
    monkeypatch.setattr(up, "_ensure_people_collection", lambda db: "people-col")
    monkeypatch.setattr(up, "ensure_authority_collection", lambda db: "authority-col")
    monkeypatch.setattr(up, "db_get_artifact", lambda db, pid: None)
    monkeypatch.setattr(up, "db_create_artifact", lambda db, e: created.append(e))
    monkeypatch.setattr(up, "db_get_edge", lambda db, c, ch: None)
    monkeypatch.setattr(up, "db_add_artifact_to_collection", lambda db, c, ch, **kw: None)
    monkeypatch.setattr(up, "db_get_grants_for_principal_resource", lambda db, g, r: [])
    monkeypatch.setattr(up, "db_create_grant", lambda db, g: None)
    up._ensure_person_artifact(MagicMock(), UserContext(id="u-2", email="bob@acme.com", inbox_id="i"))
    assert json.loads(created[0].context)["display_name"] == "bob"


# ---------------------------------------------------------------------------
# People collection (created at runtime, NOT seeded; admin-managed, user-private)
# ---------------------------------------------------------------------------

def _zero_ns():
    import uuid as _uuid
    return _uuid.UUID(int=0)


def test_ensure_people_collection_creates_and_grants_operator(monkeypatch):
    created: list = []
    grants: list = []
    monkeypatch.setattr(up, "get_instance_namespace", _zero_ns)
    monkeypatch.setattr(up, "derive_uuid", lambda ns, n, s: "people-col")
    monkeypatch.setattr(up, "register_id", lambda *a, **k: None)
    monkeypatch.setattr(up, "db_get_collection_by_id", lambda db, cid: None)  # missing
    monkeypatch.setattr(up, "db_create_collection", lambda db, c: created.append(c))
    monkeypatch.setattr(up, "_persist_seed_ids", lambda db, m: None)
    monkeypatch.setattr("mantle.services.platform_settings_service.settings.get",
                        lambda key, *a, **k: "op-1" if key == "platform.operator_id" else None)
    monkeypatch.setattr(up, "db_upsert_user_collection_grant",
                        lambda db, **kw: (grants.append(kw), (None, True))[1])

    pid = up._ensure_people_collection(MagicMock())

    assert pid == "people-col"
    assert len(created) == 1
    assert created[0].id == "people-col"
    assert created[0].content_type == up.COLLECTION_CONTENT_TYPE
    # The operator (not regular users) gets management of the directory.
    assert len(grants) == 1
    assert grants[0]["user_id"] == "op-1"
    assert grants[0]["can_admin"] is True


def test_ensure_people_collection_idempotent(monkeypatch):
    created: list = []
    monkeypatch.setattr(up, "get_instance_namespace", _zero_ns)
    monkeypatch.setattr(up, "derive_uuid", lambda ns, n, s: "people-col")
    monkeypatch.setattr(up, "register_id", lambda *a, **k: None)
    monkeypatch.setattr(up, "db_get_collection_by_id", lambda db, cid: object())  # exists
    monkeypatch.setattr(up, "db_create_collection", lambda db, c: created.append(c))
    monkeypatch.setattr(up, "_persist_seed_ids", lambda db, m: None)
    monkeypatch.setattr("mantle.services.platform_settings_service.settings.get", lambda key, *a, **k: None)

    pid = up._ensure_people_collection(MagicMock())
    assert pid == "people-col"
    assert created == []  # not re-created


# ---------------------------------------------------------------------------
# Lead conversion
# ---------------------------------------------------------------------------

def _lead(ctx: dict, content: dict) -> SimpleNamespace:
    return SimpleNamespace(context=json.dumps(ctx), content=json.dumps(content))


def test_convert_leads_matches_by_email(monkeypatch):
    monkeypatch.setattr(up, "get_id_optional", lambda slug: "leads-col")
    monkeypatch.setattr(up, "db_list_collection_artifacts",
                        lambda db, c: [{"root_id": "lead-1"}, {"root_id": "lead-2"}, {"root_id": "lead-3"}])
    leads = {
        "lead-1": _lead({"type": "lead", "source": "website-contact"}, {"email": "Jane@X.com", "name": "Jane"}),
        "lead-2": _lead({"type": "lead"}, {"email": "other@x.com"}),                       # email mismatch
        "lead-3": _lead({"type": "lead", "person_id": "someone"}, {"email": "jane@x.com"}),  # already converted
    }
    monkeypatch.setattr(up, "db_get_artifact", lambda db, rid: leads.get(rid))
    updated: list = []
    monkeypatch.setattr(up, "db_update_artifact", lambda db, e: updated.append(e))

    n = up._convert_leads_for_person(MagicMock(), "u-1", "jane@x.com")

    assert n == 1
    assert len(updated) == 1
    ctxd = json.loads(updated[0].context)
    assert ctxd["person_id"] == "u-1"
    assert ctxd["status"] == "converted"
    assert "converted_at" in ctxd


def test_convert_leads_noop_without_email_or_collection(monkeypatch):
    monkeypatch.setattr(up, "get_id_optional", lambda slug: None)
    assert up._convert_leads_for_person(MagicMock(), "u-1", "jane@x.com") == 0
    monkeypatch.setattr(up, "get_id_optional", lambda slug: "leads-col")
    assert up._convert_leads_for_person(MagicMock(), "u-1", "") == 0


# ---------------------------------------------------------------------------
# Welcome email (sent as the platform operator so the email secret resolves)
# ---------------------------------------------------------------------------

def test_send_welcome_email_sends_as_operator(monkeypatch):
    calls: list = []
    monkeypatch.setattr("mantle.services.platform_settings_service.settings.get",
                        lambda key, *a, **k: "op-1" if key == "platform.operator_id" else None)
    monkeypatch.setattr("mantle.services.server_registry.resolve_name_to_id", lambda name: "iris-uuid")
    monkeypatch.setattr("mantle.services.chorus_client.call_tool",
                        lambda sid, tool, args, *, user_id: calls.append((sid, tool, args, user_id)))

    up._send_welcome_email(MagicMock(), UserContext(id="u-1", email="jane@x.com", name="Jane", inbox_id="i"))

    assert len(calls) == 1
    sid, tool, args, user_id = calls[0]
    assert sid == "iris-uuid"
    assert tool == "send_email"
    assert user_id == "op-1"           # sent as the operator (holds secret read)
    assert args["to"] == "jane@x.com"  # delivered to the new user


def test_send_welcome_email_noop_without_email(monkeypatch):
    calls: list = []
    monkeypatch.setattr("mantle.services.chorus_client.call_tool", lambda *a, **k: calls.append(1))
    up._send_welcome_email(MagicMock(), UserContext(id="u-1", email=None, inbox_id="i"))
    assert calls == []
