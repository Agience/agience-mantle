"""The identity plane — people / platform settings / passkeys / OTP codes.

Delegates to `db.lattice_identity` (typed planes in the one store). Call sites import this
module (`from db import identity_backend as identity_store`) and never reach a store directly.
"""
from __future__ import annotations


from mantle.db import lattice_identity as _impl


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
