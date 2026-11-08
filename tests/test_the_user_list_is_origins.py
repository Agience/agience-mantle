"""The platform's user list is Origin's person records (John, 2026-11-02): no card in Mantle is
its source, and a node with no Origin says so rather than listing nothing."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from mantle.routers import system_router


def test_no_origin_configured_is_503(monkeypatch):
    from mantle import config
    monkeypatch.setattr(config, "ORIGIN_URI", "", raising=False)
    with pytest.raises(HTTPException) as refused:
        system_router._platform_user_page(object(), 10, 0, credential="t")
    assert refused.value.status_code == 503


def test_origins_refusal_is_a_refusal(monkeypatch):
    from mantle.clients.origin_key_share_client import AuthorityRefused

    class Refusing:
        def platform_users(self, **_k):
            raise AuthorityRefused(403, "Platform admin access required")

    with pytest.raises(HTTPException) as refused:
        system_router._platform_user_page(object(), 10, 0, credential="t", custodian=Refusing())
    assert refused.value.status_code == 403


def test_the_people_directory_is_not_the_user_list():
    assert not hasattr(system_router, "PEOPLE_COLLECTION_SLUG")
