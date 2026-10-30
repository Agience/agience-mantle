"""What a person searched for is theirs: recall logs its counts and timings, never the query.

2026-10-29, the reader inventory: both recall paths logged the query text at INFO
(`search/mantle/sse/router_accessor.py`), so every search was in the service log in plain text.
"""
from __future__ import annotations

import logging

from .test_a_title_is_content_and_sealed_at_rest import _note, db  # noqa: F401
from .test_there_is_no_promotion_state import _recall

WORD = "quillfeather"


async def test_no_log_record_carries_the_query(db, caplog):  # noqa: F811
    await _note(db)
    caplog.set_level(logging.DEBUG)
    _recall(db, f"{WORD} figures")
    leaked = [r.name + ": " + r.getMessage() for r in caplog.records if WORD in r.getMessage()]
    assert not leaked, leaked
