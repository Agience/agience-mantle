"""Review 16, John's decision (2026-11-03): "It's not a drop key. It's a grant with Add rights and
nothing else." Writing without reading is an Add grant, decided at Origin like any action. The
public half a writer seals to is Mantle's own key material: derived from the collection key when a
reader opens the collection, cached in this store, re-checked on every Read open (an ALARM and a
repair on a mismatch). Origin never stores it, co-signs it, or serves it."""
from __future__ import annotations

import base64
import logging

from mantle.db import reach_journal
from mantle.search.mantle import oracle as o
from mantle.services import content_crypto as cc
from mantle.services.acting_principal import acting_as

from .test_add_does_not_carry_read import OWNER, _inbox, node  # noqa: F401


class _Origin:
    """An authority that records every entry and every read it is asked for."""

    def __init__(self):
        self.asked = []

    def drop_key(self, cid):
        self.asked.append(("read", cid))
        return None


def _install(monkeypatch, origin):
    class _Auth:
        custodian = origin

        def __call__(self, entry, **kw):
            origin.asked.append(("submit", entry.get("kind")))
            return {"seq": 1, "outcome": "appended"}
    monkeypatch.setattr(reach_journal, "_authority", _Auth())
    monkeypatch.setattr(reach_journal, "installed", lambda: True)


def _open(oracle, inbox):
    oracle.evict()
    with acting_as(OWNER, principal_type="user"):
        return oracle.get_or_create_collection_key(inbox, inbox, o.KeyRequest(
            requester_id=OWNER, purpose=o.KeyPurpose.GRANT, action="read"))


def test_a_reader_s_open_caches_the_public_half_and_tells_origin_nothing(node, monkeypatch,  # noqa: F811
                                                                         caplog):
    db, oracle = node
    inbox = _inbox(db)
    origin = _Origin()
    _install(monkeypatch, origin)
    caplog.set_level(logging.WARNING)
    ck = _open(oracle, inbox)
    assert oracle.add_seal_public(inbox) == cc.drop_public_key(ck)     # cached here
    assert not [a for a in origin.asked if a[1] == "drop_key" or a[0] == "read"], origin.asked
    assert not any("co-signed" in r.getMessage() for r in caplog.records)


def test_a_writer_seals_to_this_store_s_copy_never_to_origin_s(node, monkeypatch):  # noqa: F811
    db, oracle = node
    inbox = _inbox(db)
    origin = _Origin()
    _install(monkeypatch, origin)
    ck = _open(oracle, inbox)
    origin.asked.clear()
    assert oracle.add_seal_public(inbox) == cc.drop_public_key(ck)
    assert origin.asked == []
    assert not hasattr(oracle, "drop_key_check")
    assert not hasattr(oracle, "_compare_with_origin")


def test_no_route_offers_the_web_app_a_drop_key_to_send_back():
    """Review 15, L3: a repair the web app sent from Mantle's answer would carry Mantle's word."""
    from mantle.routers import artifacts_router
    assert not [r for r in artifacts_router.router.routes
                if "drop-key" in getattr(r, "path", "")]


def test_no_alarm_text_names_a_route_that_does_not_exist():
    import inspect
    assert "/auth/me/authority/drop-key" not in inspect.getsource(o)
    assert base64  # the module's own import, kept for the helpers above
