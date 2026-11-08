"""Review 23, H1: Mantle takes a person's token only when Origin addressed it to Mantle.

The web app called Mantle on the person's Origin-audience token, so Mantle held exactly what
Origin's token exchange took, and could mint a delegation narrowed to any persona. The web app now
calls Mantle on a token Origin issues in the person's session addressed to Mantle (`aud` `mantle`),
which mints nothing; Mantle refuses a person's token addressed to Origin.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mantle.config import AUTHORITY_ISSUER
from mantle.services.dependencies import resolve_auth


def _resolve(aud):
    import time
    payload = {"iss": AUTHORITY_ISSUER, "sub": "user-123", "aud": aud, "iat": int(time.time())}
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        return resolve_auth(token="jwt-token", store_db=MagicMock())


def test_a_person_s_token_addressed_to_origin_is_refused():
    with pytest.raises(HTTPException) as refused:
        _resolve(AUTHORITY_ISSUER)
    assert refused.value.status_code == 401


def test_a_person_s_token_addressed_to_mantle_is_theirs():
    ctx = _resolve("mantle")
    assert ctx.principal_type == "user" and ctx.user_id == "user-123"
