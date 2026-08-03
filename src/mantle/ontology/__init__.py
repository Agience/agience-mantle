"""mantle.ontology — the STORE'S ONTOLOGY SURFACE.

What a corpus contains, read from the corpus: concepts, their lemmas, the is-a graph, information
content, morphology, term statistics, freshness. It reads artifacts, so it lives with the store.

⚠ MOVED HERE FROM `ember/ontology/` — 2026-08-02, the chorus→ember DAG work. This is the KEYSTONE of
that move: nearly every remaining `chorus → ember` import bottomed out here, because the personas
needed to ask the corpus what it knows and the only door was the runner's package.

⭐ **THERE IS NO WORDNET MODULE HERE, AND THAT IS THE POINT.** [John, 2026-08-02: *"There shouldn't be
a separate module for wordnet I don't think."*] What moved is `driver.py` — the ontology DRIVER — and
WordNet is one dataset it serves, not a module. The code already half-agreed: the driver reads its id
prefix, content type, POS alphabet and is-a edge labels from the stored `language:*` transducer spec,
with the WordNet values as DEFAULTS. Its own header has claimed for a while that "a second ontology
drops in as DATA"; renaming the module to what it actually is stops that claim being contradicted by
the filename. (What still makes it only half-true: `seed_lattice.build` phase 5 does not emit those
vocabulary keys yet, so every corpus currently takes the defaults. That is a feature, not a move — see
`_scratch/ONTOLOGY-TO-MANTLE.md` §5.)

WHY MANTLE, and not beam — decided by measurement, not by label:
  * These modules REACH THE STORE. `driver` opens it 9 times, `corpus_stats` reads
    `mantle.db.lattice.fts`, `freshness` writes `edge_mark`, `embed` fits against
    `mantle.search.anchors`. A thing that reads the store belongs with the store.
  * beam could not take them. beam is entroptics + signal and imports neither mantle nor ember;
    mantle is beam's SIBLING in the target DAG. Putting a store-backed driver in beam would create
    `beam → mantle` and destroy the "beam is signal only" target the dependency audit records as
    achieved.
  * mantle sits below BOTH the runner and the personas — ember and chorus already import it — so this
    is the one home from which both can reach recognition without an upward edge.

WHAT DID NOT COME, and why the split falls where it does. `geometry` (1303 lines) reaches this driver
30 times and reaches the store ZERO times. That is the real seam: **driver vs computation.** The driver
answers *what does this corpus contain*; the computation answers *how far apart are these two things*
over an abstract IC + is-a graph, and reads nothing. The computation is a MEASUREMENT and belongs in
beam — but only once it RECEIVES the driver instead of importing it, the same dependency inversion
`beam.reach` already uses for `Keyring`/`Lightcone`. Until then it stays put; moving it first would
create `beam → ember`, which is strictly worse than the edge being removed.

Modules:
  * `driver`       — the ontology driver: concepts, lemmas, hypernyms, IC, morphology (was `wn_store`)
  * `corpus_stats` — df/IDF term measurement over the store's FTS index
  * `freshness`    — per-concept staleness marks on the lattice
  * `embed`        — fitting concept embeddings against the anchor set
"""
from __future__ import annotations

__all__ = ["driver", "corpus_stats", "freshness", "embed"]
