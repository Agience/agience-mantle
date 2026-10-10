"""A refusal costs the node nothing it had not already spent: the error handlers never read the body.

`main.http_exception_logger` answers every `HTTPException`, the unauthenticated 401 included, and
`main.unhandled_exception_logger` every crash. When they logged the request body, each refusal
first buffered and parsed whatever the caller sent — a 40 MB chunked PUT with no credential was
read to the end to be told 401. These drive both handlers with a request whose body stream counts
what is pulled from it, and require that nothing is.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from fastapi import HTTPException
from starlette.requests import Request

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle import main  # noqa: E402


def _request_with_counted_body():
    """A PUT whose body is 1000 chunks, and the list of chunks the app pulled."""
    pulled = []

    async def receive():
        pulled.append(1)
        return {"type": "http.request", "body": b"x" * 65536, "more_body": len(pulled) < 1000}

    scope = {
        "type": "http", "method": "PUT", "path": "/artifacts/a-1/content", "raw_path": b"",
        "query_string": b"", "headers": [(b"transfer-encoding", b"chunked")],
        "client": ("203.0.113.7", 40000), "server": ("mantle", 8081), "scheme": "http",
        "path_params": {"artifact_id": "a-1"},
    }
    return Request(scope, receive), pulled


def test_a_401_reads_none_of_the_body():
    request, pulled = _request_with_counted_body()
    resp = asyncio.run(main.http_exception_logger(request, HTTPException(401, "Missing Bearer token")))
    assert resp.status_code == 401
    assert pulled == [], f"the refusal pulled {len(pulled)} body chunks"


def test_a_500_reads_none_of_the_body():
    request, pulled = _request_with_counted_body()
    resp = asyncio.run(main.unhandled_exception_logger(request, RuntimeError("boom")))
    assert resp.status_code == 500
    assert pulled == [], f"the crash handler pulled {len(pulled)} body chunks"
