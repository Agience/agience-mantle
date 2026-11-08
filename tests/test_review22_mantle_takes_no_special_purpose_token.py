"""Review 22, F6: Mantle takes no special-purpose token as a bearer.

Origin signs its refresh, email-verification and password-reset tokens, and a use verdict, with
the same key as its access tokens, and marks each with `token_type`. Mantle resolved one as an
ordinary bearer for its subject. Any `token_type` claim is now refused.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mantle.config import AUTHORITY_ISSUER
from mantle.services.dependencies import resolve_auth


@pytest.mark.parametrize("token_type", ["refresh", "email_verify", "password_reset", "use_verdict"])
def test_a_token_with_a_purpose_is_no_bearer(token_type):
    payload = {"sub": "user-123", "aud": AUTHORITY_ISSUER, "token_type": token_type}
    with patch("mantle.services.dependencies.verify_token", return_value=payload):
        with pytest.raises(HTTPException) as refused:
            resolve_auth(token="jwt-token", store_db=MagicMock())
    assert refused.value.status_code == 401
