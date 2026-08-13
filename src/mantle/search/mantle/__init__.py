"""MANTLE encrypted search — vector + lexical.

Public surface:

- :class:`OracleService` — per-owner master keys, HKDF-derived cell + SSE keys
- :class:`MantleIndexer` — FAISS clustering, AES-256-GCM cell encryption, S3 upload
- :class:`MantleQueryEngine` — centroid routing, cell decrypt, ANN over decrypted vectors
- :class:`LightConeResolver` — AQL BFS over origin edges with `propagate` masks

The router-shape adapter — :class:`MantleSseSearchAccessor` — and
production wiring builders live under :mod:`sse` and :mod:`wiring`.

Names are resolved lazily, through module ``__getattr__``, so each name costs only
what the caller asking for it needs. ``from search.mantle import OracleService``
works exactly as it would with eager imports, and ``__all__`` lists every resolvable
name.
"""

from __future__ import annotations

from importlib import import_module

#: name -> the submodule that defines it. One source for both ``__getattr__`` and ``__all__``, so a
#: name can never be exported without a resolvable home — the "declared but missing resolves to a
#: quiet pass" defect class this codebase keeps closing.
_EXPORTS = {
    "MantleQueryEngine": ".engine",
    "MantleIndexer": ".indexer",
    "LightConeResolver": ".lightcone",
    "GrantDenied": ".oracle",
    "GrantVerifier": ".oracle",
    "KeyPurpose": ".oracle",
    "KeyRequest": ".oracle",
    "LightConeGrantVerifier": ".oracle",
    "OracleService": ".oracle",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    return getattr(import_module(target, __name__), name)


def __dir__():
    return sorted(list(globals()) + __all__)
