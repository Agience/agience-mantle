"""Smoke tests for the MANTLE encrypted-search package skeleton.

Each MANTLE substep has its own dedicated test file. This file only
verifies the package surface imports cleanly.
"""

from __future__ import annotations


def test_public_surface_importable():
    from mantle.search.mantle import (
        MantleIndexer,
        MantleQueryEngine,
        LightConeResolver,
        OracleService,
    )
    # Just touching the names is enough — Python resolves them on import.
    assert all(c is not None for c in [
        MantleIndexer, MantleQueryEngine, LightConeResolver, OracleService,
    ])


def test_sse_surface_importable():
    from mantle.search.mantle.sse import (
        MantleSseSearchAccessor,
        SseIndexer,
        TokenNarrower,
    )
    assert all(c is not None for c in [
        MantleSseSearchAccessor, SseIndexer, TokenNarrower,
    ])
