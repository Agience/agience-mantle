"""E2E harness configuration — all knobs come from the environment so the same
suite runs against a locally-run stack or a remote deployment.

Nothing here imports mantle/origin/core: this is a true blackbox client.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- endpoints --------------------------------------------------------------
ORIGIN_URL = os.getenv("E2E_ORIGIN_URL", "http://localhost:8080").rstrip("/")
MANTLE_URL = os.getenv("E2E_MANTLE_URL", "http://localhost:8081").rstrip("/")

# The `aud` Origin stamps on user tokens (AUTHORITY_ISSUER). Mantle verifies
# Origin-signed tokens against this; a few flows echo it back.
AUTHORITY_ISSUER = os.getenv("E2E_AUTHORITY_ISSUER", "http://origin:8080")

# --- bootstrap --------------------------------------------------------------
# Provisioning writes a single-use bootstrap token to <data>/keys/bootstrap.token.
# A suite running on the same host as the stack reads it directly; anywhere else,
# pass E2E_BOOTSTRAP_TOKEN.
#
# E2E_DATA_DIR names that data root. The default below assumes a sibling
# `agience-observe` checkout, so the walk goes up past the mantle repo root:
# parents[0] = tests/e2e, [1] = tests, [2] = the repo, [3] = the workspace. Set
# E2E_DATA_DIR when the stack keeps its state anywhere else.
_MANTLE_REPO = Path(__file__).resolve().parents[2]
assert (_MANTLE_REPO / "src" / "mantle").is_dir(), (
    "path depth is wrong: parents[2] should be the agience-mantle repo root, got %s"
    % _MANTLE_REPO
)
_DEFAULT_DATA = _MANTLE_REPO.parent / "agience-observe" / ".data-local"
DATA_DIR = Path(os.getenv("E2E_DATA_DIR", str(_DEFAULT_DATA)))
BOOTSTRAP_TOKEN_ENV = os.getenv("E2E_BOOTSTRAP_TOKEN", "")

# --- regime flags -----------------------------------------------------------
# Set to "1" when the stack is running with MANTLE_LAZY_INDEX=on so the
# first-observation (lazy materialization) assertions are meaningful.
LAZY_INDEX = os.getenv("E2E_LAZY_INDEX", "") not in ("", "0", "false", "False")

# The remote embeddings provider (prism) is absent in a local run; semantic
# search degrades to lexical. Skip semantic-only assertions unless told.
HAS_EMBEDDINGS = os.getenv("E2E_HAS_EMBEDDINGS", "") not in ("", "0", "false", "False")

HTTP_TIMEOUT = float(os.getenv("E2E_HTTP_TIMEOUT", "15"))


def bootstrap_token() -> str | None:
    """Resolve the single-use bootstrap token: env override first, else the
    on-disk token the init container wrote. Returns None if neither is present
    (already claimed, or running against a remote stack without the override)."""
    if BOOTSTRAP_TOKEN_ENV:
        return BOOTSTRAP_TOKEN_ENV.strip()
    tok = DATA_DIR / "keys" / "bootstrap.token"
    if tok.is_file():
        return tok.read_text(encoding="utf-8").strip()
    return None
