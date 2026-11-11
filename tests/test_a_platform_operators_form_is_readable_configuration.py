"""A platform operator's input form and wiring are readable configuration; its title stays sealed.

John, 2026-11-10 (BUG-057): on staging every platform operator read "Operator", with no input
fields, because the copies are sealed under production's keys and staging holds none of them. The
operators the platform seeds are its public definition, so they join the readable types: input
form, wiring and description in the open, the title alone sealed. A node that cannot open the title
shows the title of the operator's definition in the catalog. On the release candidate the seeder
converges a row it cannot seal with nothing sealed at all, rather than failing.

A person's own operator is not platform configuration: its wiring may name a private server.
"""
from __future__ import annotations

import json

from mantle.db import doc_boundary as B
from mantle.services.acting_principal import acting_as
from mantle.services.seed_provisioning import platform_operators as P

from .test_platform_operators_are_granted_artifacts import PLATFORM, db  # noqa: F401  (fixture)


def _stored_ctx(db, artifact_id):  # noqa: F811
    raw = db.artifacts.get_artifact(artifact_id)
    return dict(raw), json.loads(raw["context"])


def test_the_form_and_wiring_are_in_the_open_and_the_title_is_sealed(db):  # noqa: F811
    assert P.ensure_platform_operators(db)
    op_id = P.ids()["op-answer-from-my-notes"]
    raw, ctx = _stored_ctx(db, op_id)
    assert "title" not in ctx and B.SEALED in ctx, "the title must stay sealed"
    assert ctx["input_schema"]["required"] == ["question", "evidence"]
    assert ctx["operator"]["server"] == "sage" and ctx["operator"]["tool"] == "ask"
    assert ctx["description"]
    assert raw.get("name") == "Answer from my notes"


def test_a_reader_that_cannot_open_the_title_gets_the_catalogs(db, monkeypatch):  # noqa: F811
    P.ensure_platform_operators(db)
    op_id = P.ids()["op-answer-from-my-notes"]
    raw, _ = _stored_ctx(db, op_id)

    def shut(*_a, **_k):   # as on staging: the key this was sealed under does not unwrap here
        raise RuntimeError("Failed to unwrap master key")
    monkeypatch.setattr(B, "_open", shut)
    B.unseal_content_fields(raw)
    ctx = json.loads(raw["context"])
    assert ctx["title"] == "Answer from my notes"
    assert ctx["input_schema"]["properties"]["question"]


def test_the_owner_still_opens_the_sealed_title(db):  # noqa: F811
    P.ensure_platform_operators(db)
    op_id = P.ids()["op-read-an-item"]
    raw, _ = _stored_ctx(db, op_id)
    with acting_as(PLATFORM, principal_type="user"):
        B.unseal_content_fields(raw)
    assert json.loads(raw["context"])["title"] == "Read an item"


def test_a_persons_own_operator_stays_sealed():
    doc = {"content_type": B.PLATFORM_OPERATOR_TYPE}
    own = {"content_type": B.PLATFORM_OPERATOR_TYPE, "title": "Mine",
           "operator": {"server": "a-private-server", "tool": "t"}, "input_schema": {}}
    assert set(B.sealable_keys(doc, own)) >= {"title", "operator", "input_schema"}
    platform = {**own, "platform": True}
    assert B.sealable_keys(doc, platform) == ["title"]
    assert B.sealable_keys(doc, {**own, "platform": "true"}) != ["title"], "only the seeder's flag"


def test_the_rc_converges_a_row_it_cannot_seal_with_nothing_sealed(db, monkeypatch):  # noqa: F811
    """Staging: every rewrite of a copied operator must seal its title under a key production
    wrapped. The RC writes the definition in the open without the title instead."""
    P.ensure_platform_operators(db)
    known = P.ids()
    from mantle.services.seed_provisioning import platform_email

    real = platform_email._ensure_artifact

    def unsealable(_db, artifact_id, **kw):
        try:
            raise RuntimeError("Failed to unwrap master key")
        except RuntimeError as inner:
            raise B.ContentEncryptionError("context field encryption unavailable") from inner
    monkeypatch.setattr(platform_email, "_ensure_artifact", unsealable)

    monkeypatch.setenv("AGIENCE_PLANE", "production")
    import pytest
    with pytest.raises(B.ContentEncryptionError):
        P.ensure_platform_operators(db)

    monkeypatch.setenv("AGIENCE_PLANE", "rc")
    assert P.ensure_platform_operators(db)
    raw, ctx = _stored_ctx(db, known["op-answer-from-my-notes"])
    assert B.SEALED not in ctx and "title" not in ctx
    assert ctx["input_schema"]["required"] == ["question", "evidence"]
    B.unseal_content_fields(raw)
    assert json.loads(raw["context"])["title"] == "Answer from my notes"
    _, coll = _stored_ctx(db, known["platform-operators"] if "platform-operators" in known
                          else next(v for k, v in known.items() if not k.startswith("op-")))
    assert B.SEALED not in coll
    monkeypatch.setattr(platform_email, "_ensure_artifact", real)


def test_a_step_can_answer_from_or_read_a_single_note():
    """BUG-052, reopened 2026-11-10: the evidence picker offered collections alone, because the
    field named no type and the picker then lists collections."""
    by_slug = {o["slug"]: o for o in P.OPERATORS}
    for slug, field in (("op-answer-from-my-notes", "evidence"), ("op-read-an-item", "artifact_id")):
        types = by_slug[slug]["input_schema"]["properties"][field]["x-content-types"]
        assert "text/markdown" in types and P.COLLECTION_CT in types, (slug, types)
