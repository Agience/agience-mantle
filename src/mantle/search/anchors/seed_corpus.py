"""The platform seed corpus — the common grounded knowledge every deployment ships with.

This is the text that gets INDEXED once the AnchorSet exists. It is not the source of the anchors:
this module used to cluster it with k-means and admit the medoids, which is removed (2026-07-31).
An anchor id is content-addressed over ``(label, model_id, embedding)``, so a locally-derived set
mints region ids no peer computes — the deployments only discover the split when a sync returns no
overlap. The AnchorSet is PROVISIONED; see :func:`store.require_live_anchorset`.

Renamed from ``bootstrap.py``, which named a mechanism that no longer exists here.

INVARIANT (§1): public, non-authorizing geometry. No keys, no light-cone, no ledger — just public
seed text.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .anchorset import AnchorSet

logger = logging.getLogger(__name__)


# Generic grounded-knowledge corpus shipped inside the Mantle package, so a
# standalone deploy (no platform seeds) can still bootstrap the coordinate
# system. Unioned with any platform seeds present.
_BUNDLED_CORPUS = Path(__file__).resolve().parent / "corpus.yaml"


def _entry_text(body: dict, fallback_label: str) -> tuple[str, str] | None:
    """Fold a seed doc's name/description/content into one ``(label, text)``
    pair, or ``None`` when it carries no usable text."""
    label = str(body.get("slug") or body.get("name") or fallback_label)
    parts = [
        str(body.get("name", "")),
        str(body.get("description", "")),
        str(body.get("content", "")),
    ]
    text = " ".join(p for p in parts if p and p != "None").strip()
    return (label, text) if text else None


def _gather_platform_seed_corpus() -> list[tuple[str, str]]:
    """Platform seed artifacts (present only when Origin/app has provisioned
    them under ``package/seeds/platform/artifacts``). Empty in a pure DB."""
    from origin import config

    root = config.BASE_DIR / "package" / "seeds" / "platform" / "artifacts"
    corpus: list[tuple[str, str]] = []
    if not root.is_dir():
        return corpus
    for path in sorted(root.glob("*.yaml")):
        try:
            body = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(body, dict):
            entry = _entry_text(body, path.stem)
            if entry:
                corpus.append(entry)
    return corpus


def _gather_bundled_corpus() -> list[tuple[str, str]]:
    """The generic grounded-knowledge corpus that ships with Mantle (a YAML
    list). Always present in the image — makes vector search work out of the
    box without any platform seeds."""
    if not _BUNDLED_CORPUS.is_file():
        return []
    try:
        items = yaml.safe_load(_BUNDLED_CORPUS.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read bundled anchor corpus %s", _BUNDLED_CORPUS, exc_info=True)
        return []
    corpus: list[tuple[str, str]] = []
    for i, body in enumerate(items or []):
        if isinstance(body, dict):
            entry = _entry_text(body, f"corpus-{i}")
            if entry:
                corpus.append(entry)
    return corpus


def gather_seed_corpus() -> list[tuple[str, str]]:
    """Return ``(label, text)`` for the common grounded knowledge that seeds the
    universal coordinate system — the Mantle-shipped generic corpus unioned with
    any platform seed artifacts an application has provisioned."""
    return _gather_platform_seed_corpus() + _gather_bundled_corpus()
