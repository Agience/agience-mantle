"""The coupling a semantic relation carries — the SIGN the field reads and the NAMES the readout matches.

This is a stable grounding surface (Apache, ember-side): the conversation tekton reads it, the op table
does not define it. When the tekton migrates to a persona it REACHES this surface, rather than importing a
genesis internal (see `_scratch/EMBER-CONSOLIDATION-PLAN.md` Phase 0).

`sign`: +1 attraction (broader/narrower/same pulls the concept onto the screen), −1 detraction (the
OPPOSITE lands at negative sign). `names`: how a need's operator names the relation.

SEEDED SEAM — two halves with DIFFERENT verdicts (measured on the 71 substrate, 2026-07-29):

  • `sign` — **NOT recoverable from endpoint geometry** (empirically settled, do not force). The seam
    once read "measure the sign from the geometry of the relation's endpoints (aligned/anti-aligned in the
    coordinate)." Measured over real antonym edges: antonym endpoints are POSITIVELY aligned, never
    anti-aligned — `love.n.01↔hate.n.01` cos=+0.78 (jc 0.27), `good↔evil` +0.56, `increase↔decrease`
    +0.72 — sitting INSIDE the hypernym cos range (mean +0.84). Worse, the prototypical antonyms are
    ADJECTIVES (hot/cold, big/small, strong/weak), which have NO JC coordinate at all (adjectives aren't in
    the IS-A taxonomy). So `Screen.couple` over endpoint coordinates would read antonyms as ATTRACTING and
    BREAK opposites. The `sign` is a VALENCE/opposition property orthogonal to taxonomic position — it is
    the DECLARED SEMANTICS of the relation type (antonym = opposite), not an imposed arbitrary constant.
    It stays as data on the relation type; deriving it would need a valence transducer, not this coordinate.

  • `names` — **IS derivable from the lexicon** (the real remaining seam). "opposite"/"contrary"/"reverse"
    are the synonyms of the relation-label word, reachable via the keyed `lex:en` lookup ([[self-contained-wordnet]]).
    That half can be READ OFF the corpus; the sign cannot.

[[never-impose-knowledge-derive-it]] (the prescribed "sign from geometry" derivation is provably
infeasible here — measurement beats the guess, and a forced −1 is worse than this honest default)
[[gauge-is-an-artifact-coupling-is-measured]] [[one-resolution-not-thresholds]] [[no-arbitrary-caps]]
"""
from __future__ import annotations

from typing import Any, Dict

SEED_ETYPE_COUPLING: Dict[str, Dict[str, Any]] = {
    "hypernym":          {"sign": 1.0, "names": ["kind", "type", "sort"]},
    "instance_hypernym": {"sign": 1.0, "names": []},
    "instance_of":       {"sign": 1.0, "names": []},
    "hyponym":           {"sign": 1.0, "names": []},
    "synonym":           {"sign": 1.0, "names": ["same", "synonym", "like"]},
    "antonym":           {"sign": -1.0, "names": ["opposite", "antonym", "contrary", "reverse"]},
}
