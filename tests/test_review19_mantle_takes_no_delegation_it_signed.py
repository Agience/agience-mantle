"""Review 19, H2: a delegation Mantle's own key signed names a person nobody handed it. Mantle
mints none, so its auth dependency refuses one, whatever else it carries."""
from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_an_iss_mantle_delegation_is_refused(monkeypatch):
    from mantle.services import dependencies as deps
    payload = {"iss": "mantle", "sub": "victim", "aud": "agience-server-iris",
               "principal_type": "delegation", "act": {"sub": "agience-server-iris",
                                                       "iss": "iris"}, "host_id": "h"}
    with pytest.raises(HTTPException) as e:
        deps._refuse_a_delegation_mantle_signed(payload)
    assert e.value.status_code == 401
    deps._refuse_a_delegation_mantle_signed({**payload, "iss": "https://origin.test"})
