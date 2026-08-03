"""MANTLE encrypted search — vector + lexical (Steps 2.0 → 2.6.9).

Public surface:

- :class:`OracleService` — per-owner master keys, HKDF-derived cell + SSE keys
- :class:`MantleIndexer` — FAISS clustering, AES-256-GCM cell encryption, S3 upload
- :class:`MantleQueryEngine` — centroid routing, cell decrypt, ANN over decrypted vectors
- :class:`LightConeResolver` — AQL BFS over origin edges with `propagate` masks
- :mod:`sse` — encrypted lexical (BM25) index per
  ``.dev/features/mantle-sse-lexical-index.md``. Replaces the legacy lexical index.

The router-shape adapter — :class:`MantleSseSearchAccessor` — and
production wiring builders live under :mod:`sse` and :mod:`wiring`.
The legacy the legacy lexical index-arm ``MantleSearchAccessor`` was retired with
the legacy lexical index in Step 2.6.9 part 2.

⛔ EVERY NAME BELOW IS RESOLVED LAZILY (PEP 562), AND THAT IS LOAD-BEARING.

This package used to import ``.engine``, ``.indexer``, ``.lightcone`` and ``.oracle`` at module
scope. The consequence was not a slow import — it was that **the encrypted LEXICAL arm could not be
imported, or shipped, without the vector arm and the whole key-custody hierarchy.** ``import
…search.mantle.sse.tokenizer`` executes this file first, so a pure-stdlib tokenizer dragged in
``numpy``, ``embeddings`` and ``services.acting_principal``.

That is the coupling EREA §5 hit from the outside ("it currently sits service-side, entangled with
the vector arm and the ``OracleService`` custody hierarchy"). The lexical modules themselves were
almost clean all along — 10 of 11 needed nothing but stdlib + ``cryptography``. The entanglement
lived in the two ``__init__.py`` files, which is where nobody reads for it.

Lazy resolution keeps the public API EXACTLY as it was — ``from search.mantle import OracleService``
still works and ``__all__`` is unchanged — while making each name cost only what the caller asking
for it needs. Nothing is removed; what changed is WHEN.

⚠ It also breaks a real CYCLE: ``oracle`` imports the key contract from ``.sse.keys``
(implementation → interface), and eager imports here made that
``oracle → sse/__init__ → router_accessor → oracle``, partially initialized.
"""

from __future__ import annotations

from importlib import import_module

#: name -> the submodule that defines it. ONE source for both ``__getattr__`` and ``__all__``, so a
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
