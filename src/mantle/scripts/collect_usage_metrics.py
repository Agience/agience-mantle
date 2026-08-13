"""Collect local deployment usage metrics from the store (the standalone lattice).

The metric key set is stable — downstream usage snapshots/licensing read these names — while the
values are read from the lattice: maintained counters where published (`count`, `count_edges`),
typed-plane streams for the small per-type tallies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


backend_root = _backend_root()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

# Injectable for tests: a callable returning the store handle.
get_store_handle = None


def _load_backend_dependencies() -> None:
    global get_store_handle
    if get_store_handle is None:
        from mantle.db import backend
        get_store_handle = backend.store_handle


def _count_ct(db, content_type: str) -> int:
    """Stream-count one typed plane. Small planes only (grants/commits/containers)."""
    return sum(1 for _ in db.artifacts.list_artifacts(content_type=content_type))


def collect_usage_metrics() -> dict[str, int]:
    _load_backend_dependencies()
    db = get_store_handle()

    from mantle.db.lattice_api import _GRANT_CT, _COMMIT_CT               # the typed planes
    from mantle.db.lattice_identity import _PEOPLE
    from mantle.entities.collection import WORKSPACE_CONTENT_TYPE, COLLECTION_CONTENT_TYPE

    return {
        "users_total": sum(1 for _ in _PEOPLE.all(db)),
        "workspaces_total": _count_ct(db, WORKSPACE_CONTENT_TYPE),
        "workspace_artifacts_total": int(db.graph.count_edges()),   # membership edges
        "collections_total": _count_ct(db, COLLECTION_CONTENT_TYPE),
        "committed_artifact_versions_total": int(db.artifacts.count(state="committed")),
        "grants_total": _count_ct(db, _GRANT_CT),
        "commits_total": _count_ct(db, _COMMIT_CT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect local deployment usage metrics from the lattice store."
    )
    parser.add_argument(
        "--output-file",
        default="/app/keys/usage-metrics.json",
        help="Path to write the collected usage metrics JSON.",
    )
    args = parser.parse_args()

    metrics = collect_usage_metrics()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
