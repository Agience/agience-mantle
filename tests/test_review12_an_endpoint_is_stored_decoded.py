"""Review 12, info: a credential's `allowed_endpoints` are stored decoded until stable, so an
encoded `..` (once or twice over) or an encoded separator never names a path other than the one
the persona will compare against."""
from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from mantle.services.bootstrap_types import CREDENTIAL_CONTENT_TYPE
from mantle.services.workspace_service import credential_endpoints


def _ends(*urls):
    return credential_endpoints(CREDENTIAL_CONTENT_TYPE,
                                json.dumps({"allowed_endpoints": list(urls)}))


@pytest.mark.parametrize("url", ["https://api.example.test/v1/%2e%2e/admin",
                                 "https://api.example.test/v1/%252e%252e/admin",
                                 "https://api.example.test/v1/../admin",
                                 "https://api.example.test/v1%5cadmin",
                                 "https://api.example.test/v1/%00"])
def test_a_traversal_or_an_encoded_separator_is_refused(url):
    with pytest.raises(HTTPException):
        _ends(url)


def test_an_encoded_path_is_stored_decoded():
    assert _ends("https://API.example.test/v1/%7Euser/") == ["https://api.example.test/v1/~user/"]
