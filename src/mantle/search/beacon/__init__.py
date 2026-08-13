# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc.  Licensed under the Apache License, Version 2.0.
#
# Beacon is the permissive half of the two-tier model, deliberately: mantle ships
# Apache so a store can be taken, built on and shipped by anyone, and beacon is the
# reduced instrument that makes such a store genuinely useful on its own.
#
# This file is Apache-2.0 with a public remote. The downstream consumer's
# `beacon_engine.py` carries its own proprietary-restriction notice — that is the
# Foresight white-label pilot, a different tree and a different arrangement.
# ---------------------------------------------------------------------------

"""The Beacon spectral engine — how many independent directions a set of vectors has.
Mantle's semantic line needs to decide where signal stops.
"""
from mantle.search.beacon.engine import (  # noqa: F401
    DEFAULT_FAR,
    ENGINE_ID,
    ENGINE_ID_PERM,
    BeaconEngineError,
    RankResult,
    signal_rank,
)

__all__ = ["DEFAULT_FAR", "ENGINE_ID", "ENGINE_ID_PERM", "BeaconEngineError", "RankResult",
           "signal_rank"]
