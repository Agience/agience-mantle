"""The identity plane — people / platform settings / passkeys / OTP codes.

Delegates to `db.lattice_identity` (typed planes in the one store). Call sites import this
module (`from db import identity_backend as identity_store` — the historical alias) and never
reach a store directly. The store identity module was deleted with the 2026-07-22 flip.
"""
from __future__ import annotations


from mantle.db import lattice_identity as _impl
# ⛔ AN `assert MANTLE_DB == "lattice"` STOOD HERE, guarding against a backend selector that no
# longer exists. There is one store, so the assertion could not fail [John, 2026-07-23], and an
# assertion that cannot fail asserts nothing.


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
