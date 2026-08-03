"""Transducer MEASUREMENT read + shared state — the ember (instrument) side of the transducer.

A transducer is a surface↔concept conversion stored as an artifact (`op.transducer.<name>`). The CONVERSION
CLASSES (`Transducer`/`LanguageTransducer`/`StubTransducer` + `get_transducer` — the entry/render facet
binding) MOVED to `lumen/transducer.py` (the persona facet, F2/P3, ember→chorus migration 2026-07-29). What
stays here is the INSTRUMENT side, mirroring the P7 split:

  · `persisted_xi` — the keyed MEASUREMENT read: ξ measured once at build time and stored on the language
    transducer artifact, read off `spec.xi` DIRECTLY (not through the conversion classes), so the chat path
    never re-derives it AND this measurement stays ember with NO persona import. `ember.match.xi` calls it.
  · `TRANSDUCER_CT` / `TRANSDUCER_OP` — the stored content-type + op-id prefix (option B), shared constants.
  · `_REG` — the get_transducer INSTANCE cache (a plain dict). It lives here because `ember/seed_lattice.py`
    `build()` clears it after (re)writing the artifact, and ember cannot reach lumen; lumen's `get_transducer`
    reads/writes this same dict (lumen→ember allowed). Same shape as `match._OFFER_CACHE`/`invalidate`.

`build()` moved to `ember/seed_lattice.py` (P2). [[ember-is-a-runner]] [[ember-tekton-facet-consolidation]]

⚠ MOVED HERE FROM `ember/signal/transducer.py` — 2026-08-02, the chorus→ember DAG work. Measured at the
move it imported `mantle.ontology.driver` and `prism.grounding` and nothing from ember at all.

⭐ `_REG` IS THE REASON THIS MATTERS, not the constants. It is ONE dict, shared: lumen POPULATES it
(`lumen/transducer.py` builds the conversion classes into it) and a completed `seed_lattice.build`
CLEARS it, so a rebuilt substrate cannot be served from a stale instance. A shared cache reachable from
only one side of a boundary is the shape where the writer fills one dict and the invalidator clears
another — the same defect `_OFFER_CACHE` was nearly split into. Here, below both, there is exactly one.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from mantle.ontology import driver as _wn
from prism.grounding import TRANSDUCER_OP       # the op-id prefix — flipped WITH the migration, 2026-07-30

# ⭐ FLIPPED 2026-07-30 in the same step as `grounding.TRANSDUCER_OP` and the stored artifact rename
# (`node/transducer_rename.py --apply`). Both halves land together or the node sits on the slow
# unkeyed path in silence — see the note in `grounding.py`.
TRANSDUCER_CT = "application/x-transducer"

# The get_transducer INSTANCE cache — cleared by `seed_lattice.build()` after a (re)write, populated by
# lumen's `get_transducer`. Holds lumen Transducer instances; typed `Any` because ember does not import them.
_REG: Dict[str, Any] = {}


def persisted_xi(lang: str = "en") -> Optional[float]:
    """ξ as measured once at build time and stored on the language transducer artifact — read KEYED
    (`spec.xi` off the stored doc), so the chat path never triggers the whole-corpus derivation. Reads the
    artifact DIRECTLY (not via the lumen conversion classes), so this measurement stays ember with no persona
    import — `ember.match.xi` calls it."""
    try:
        doc = _wn._arts().get_artifact(TRANSDUCER_OP + "language." + lang)
    except Exception:
        doc = None
    v = (doc.get("spec") or {}).get("xi") if doc else None
    return float(v) if isinstance(v, (int, float)) else None


__all__ = ["TRANSDUCER_CT", "TRANSDUCER_OP", "persisted_xi", "_REG"]
