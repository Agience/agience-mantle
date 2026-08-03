#!/usr/bin/env python3
"""manage_anchors.py — inspect the live MANTLE AnchorSet.

The AnchorSet is the shared coordinate system / routing centroids / grounding
(see `.dev/features/mantle-canonical-architecture.md` §3). It is **provisioned**: the canonical set
arrives as an artifact and every node admits the same one. It grows from there as the manifold grows.

Run from the `mantle/` directory.

Actions
-------
seed-corpus  Report the platform seed corpus this deployment ships (what gets INDEXED once the
             AnchorSet exists). It is NOT the source of the anchors.
inspect      Show the current live AnchorSet (count, model, sample anchors).
grow         Propose a candidate anchor from novel text.

⛔ THE `bootstrap` ACTION IS REMOVED (2026-07-31). It clustered the seed corpus with k-means and
admitted the medoids. Anchors derived locally mint region ids no peer computes, so it produced an
index that looked healthy and shared with nobody. There is no replacement action: provision the
canonical AnchorSet artifact into the anchor repo.

⛔ NO EMBEDDINGS PROVIDER EXISTS (2026-07-22, universal no-models rule — see embeddings.py), so
`grow` fails with an explicit error: it needs vectors and nothing produces them. `inspect` and
`seed-corpus` work without any.
"""

import argparse
import logging

from mantle.search.anchors import (
    gather_seed_corpus,
    get_anchor_repo,
    get_live_anchorset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s  %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("manage_anchors")


def action_seed_corpus(_dry_run: bool) -> None:
    """Report the seed corpus. It is what gets INDEXED; it is not where anchors come from."""
    corpus = gather_seed_corpus()
    if not corpus:
        raise SystemExit("No seed corpus under package/seeds/platform/artifacts.")
    logger.info("Platform seed corpus: %d items.", len(corpus))
    for label, _ in corpus[:12]:
        logger.info("  - %s", label)
    if len(corpus) > 12:
        logger.info("  ... and %d more", len(corpus) - 12)
    logger.info("Anchors are NOT derived from these — the AnchorSet is provisioned.")


def action_inspect(_dry_run: bool) -> None:
    aset = get_live_anchorset()
    if aset is None:
        logger.info("No live AnchorSet. It is provisioned, not generated here — load the canonical "
                    "AnchorSet artifact into the anchor repo.")
        return
    logger.info("Live AnchorSet: %d anchors, model=%s, dim=%d", len(aset), aset.model_id, aset.dim)
    logger.info("  (manifold analysis is available via the Beacon add-on)")
    for a in aset.anchors[: min(len(aset), 25)]:
        logger.info("  [%s] %s  (%s)", a.tier, a.label, a.anchor_id[:12])


def action_grow(text: str, dry_run: bool) -> None:
    """Admit a novel signal as a new CANDIDATE anchor (RG-flow growth)."""
    if not text:
        raise SystemExit("--text is required for grow")
    from mantle.embeddings import Embeddings
    vectors = Embeddings()([text])
    if not vectors or not vectors[0]:
        raise SystemExit(
            "No embeddings provider exists (removed 2026-07-22 under the universal "
            "no-models rule — see embeddings.py). grow requires vectors."
        )
    # Both branches report the CONTINUOUS density, not just the layer. An operator told
    # "layer L1" cannot tell a signal sitting a thousandth from the threshold — where the
    # answer will flip at the next refit — from one sitting well inside it, and the layer
    # thresholds themselves are refitted from anchor spacing on every growth step. The
    # admit/reject stays binary; the evidence for it no longer has to be guessed at.
    if dry_run:
        from mantle.search.anchors import get_density_zoom
        dz = get_density_zoom()
        if dz is None:
            logger.info("[DRY-RUN] %r → no density zoom available", text)
            return
        layer, density = dz.layer(vectors[0])
        logger.info(
            "[DRY-RUN] %r → density layer %s (density=%.4f, t_low=%s, t_high=%s); "
            "admitted only if L0/novel",
            text, layer, float(density),
            getattr(dz, "_t_low", None), getattr(dz, "_t_high", None),
        )
        return
    from mantle.search.anchors.grow import propose_anchor_decided
    decision = propose_anchor_decided(text[:80], vectors[0])
    if decision.anchor is None:
        # `reason` separates the three cases the old single `None` merged: no AnchorSet,
        # dimension mismatch, and a genuine geometric rejection.
        logger.info("Not admitted (%s): %s", decision.reason, decision.as_read())
    else:
        logger.info("Admitted candidate anchor: %s (%s) %s",
                    decision.anchor.label, decision.anchor.anchor_id[:12],
                    decision.as_read())


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage the MANTLE AnchorSet.")
    parser.add_argument("--action", choices=["seed-corpus", "inspect", "grow"], required=True)
    parser.add_argument("--text", help="Text to grow a candidate anchor from (grow).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # CLI: no request context. Anchor admission writes anchor artifacts through the
    # ordinary artifact path.
    from mantle.services.system_identity import system_acting_context

    with system_acting_context(scope="platform.manage-anchors"):
        if args.action == "seed-corpus":
            action_seed_corpus(args.dry_run)
        elif args.action == "inspect":
            action_inspect(args.dry_run)
        elif args.action == "grow":
            action_grow(args.text, args.dry_run)


if __name__ == "__main__":
    main()
