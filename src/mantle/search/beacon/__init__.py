# ---------------------------------------------------------------------------
# Copyright (c) 2026 Ikailo Inc. All rights reserved.
#
# CONFIDENTIAL — CONTAINS TRADE SECRETS OF IKAILO INC.
#
# This package and the BEACON algorithms it implements are the confidential and
# proprietary trade-secret property of Ikailo Inc., disclosed only to authorized
# parties under obligation of confidentiality. No part may be used, reproduced,
# modified, distributed, or disclosed without the prior written consent of
# Ikailo Inc. Unauthorized use or disclosure is prohibited and may violate
# trade-secret, copyright, and contract law.
# ---------------------------------------------------------------------------

"""The BEACON spectral engine — how many independent directions a set of vectors has.

Mantle's semantic line needs to decide where signal stops. Every such decision here
used to be a constant somebody picked; this answers it from the spectrum instead, at a
stated false-alarm level. See `engine` for the derivation.
"""
from mantle.search.beacon.engine import (  # noqa: F401
    DEFAULT_FAR,
    ENGINE_ID,
    BeaconEngineError,
    RankResult,
    signal_rank,
)

# ⛔ THE EXPORTED SURFACE IS THE BEACON CUT AND NOTHING ELSE.
# This package is CONFIDENTIAL (header above), so an export is DISCLOSED SURFACE, not just
# public API. Measured 2026-07-30: `spectrum_stats`, `noise_floor`, `component_significance`,
# `occupancy_fraction`, `whiten`, `shannon_bits`, `johnstone`, `tw1_quantile`, `tw1_sf` and
# `ComponentSignificance` were all exported with ZERO callers outside this package. They remain
# reachable by path for the package's own tests; they are not contract.
__all__ = ["DEFAULT_FAR", "ENGINE_ID", "BeaconEngineError", "RankResult", "signal_rank"]
