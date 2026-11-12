"""A platform collection's `created_by` is the platform user, which no caller is. The seed names
the acting principal as the creator of that collection's first key, so the admin's seed can make
it; the name applies to that one collection only (71/test, 2026-11-11: every platform collection
failed with AddSealPending)."""
from __future__ import annotations

from mantle.services import content_crypto as cc


def test_the_named_creator_reaches_the_key_request_for_that_collection_only(monkeypatch):
    asked = []

    def key(principal_id, collection_id, *, may_create=False, creator_id=None):
        asked.append((collection_id, creator_id))
        return b"k" * 32
    monkeypatch.setattr(cc, "_default_collection_key", key)

    with cc.collection_key_minted_by("admin-1", "plat-1"):
        cc.encrypt_content("plat-1", b"x", collection_id="plat-1", creator_id="platform-user")
        cc.encrypt_content("other-1", b"x", collection_id="other-1", creator_id="platform-user")
    cc.encrypt_content("plat-1", b"x", collection_id="plat-1", creator_id="platform-user")

    assert asked == [("plat-1", "admin-1"), ("other-1", "platform-user"),
                     ("plat-1", "platform-user")]
