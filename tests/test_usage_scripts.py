from __future__ import annotations

import json

from mantle.scripts import collect_usage_metrics, generate_usage_snapshot


def test_collect_usage_metrics_returns_stable_metric_set(tmp_path, monkeypatch):
    """The metric key set is the contract — seeded against a real temp lattice."""
    from mantle.db import lattice_api as api
    from mantle.entities.artifact import Artifact
    from mantle.entities.grant import Grant
    from mantle.entities.commit import Commit
    from mantle.entities.collection import WORKSPACE_CONTENT_TYPE

    db = api.open_database(str(tmp_path / "usage.db"), origin="usage-test")
    monkeypatch.setattr(collect_usage_metrics, "get_store_handle", lambda: db)
    monkeypatch.setattr(
        "mantle.services.content_crypto._default_master_key",
        # Mirrors the real signature, `may_create` included — see the note on the same stub in
        # `db/test_lattice_api.py`.
        lambda principal_id, collection_id=None, *, may_create=False, creator_id=None: b"" * 32)

    api.create_collection(db, Artifact(id="ws-1", collection_id="", content="",
                                       content_type=WORKSPACE_CONTENT_TYPE,
                                       state="committed", created_by="u1"))
    api.create_artifact(db, Artifact(id="a-1", collection_id="ws-1", content="x",
                                     content_type="text/markdown", state="committed",
                                     created_by="u1", origin_root="ws-1"))
    api.add_artifact_to_collection(db, "ws-1", "a-1")
    api.create_grant(db, Grant(id="g-1", resource_id="ws-1", grantee_type="user",
                               grantee_id="u1", granted_by="u1"))
    api.create_commit(db, Commit(id="c-1", message="m", timestamp="2026-01-01T00:00:00Z",
                                 item_ids=[]))
    from mantle.db import lattice_identity as idl
    idl.create_person(db, {"id": "u1", "email": "u1@x"})

    metrics = collect_usage_metrics.collect_usage_metrics()

    assert set(metrics) == {
        "users_total", "workspaces_total", "workspace_artifacts_total",
        "collections_total", "committed_artifact_versions_total",
        "grants_total", "commits_total",
    }
    assert metrics["users_total"] == 1
    assert metrics["workspaces_total"] == 1
    assert metrics["workspace_artifacts_total"] == 1     # the membership edge
    assert metrics["grants_total"] == 1
    assert metrics["commits_total"] == 1
    assert metrics["committed_artifact_versions_total"] >= 2   # ws + artifact (+ commit plane rows are committed-by-absence? no: typed docs carry no state)


def test_generate_usage_snapshot_computes_overages_and_infers_ids(tmp_path):
    usage_file = tmp_path / "usage.json"
    allowances_file = tmp_path / "allowances.json"
    license_file = tmp_path / "license.json"
    state_file = tmp_path / "state.json"
    output_file = tmp_path / "snapshot.json"

    usage_file.write_text(json.dumps({"users_total": 12, "workspace_artifacts_total": 1200}), encoding="utf-8")
    allowances_file.write_text(json.dumps({"users_total": 10, "workspace_artifacts_total": 2000}), encoding="utf-8")
    license_file.write_text(
        json.dumps(
            {
                "license_id": "lic-123",
                "account_id": "acct-123",
                "product": {
                    "distribution_profiles": ["standard"],
                    "runtime_roles": ["standard"],
                },
            }
        ),
        encoding="utf-8",
    )
    state_file.write_text(json.dumps({"install_id": "inst-123"}), encoding="utf-8")

    # Drive the script through its argparse entrypoint semantics by patching argv.
    argv = [
        "generate_usage_snapshot.py",
        "--usage-file",
        str(usage_file),
        "--allowances-file",
        str(allowances_file),
        "--license-file",
        str(license_file),
        "--state-file",
        str(state_file),
        "--output-file",
        str(output_file),
        "--captured-at",
        "2026-03-07T12:00:00Z",
        "--period-label",
        "2026-03",
        "--usage-id",
        "usage-123",
    ]

    original_argv = generate_usage_snapshot.sys.argv
    generate_usage_snapshot.sys.argv = argv
    try:
        result = generate_usage_snapshot.main()
    finally:
        generate_usage_snapshot.sys.argv = original_argv

    assert result == 0
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["license_id"] == "lic-123"
    assert payload["account_id"] == "acct-123"
    assert payload["install_id"] == "inst-123"
    assert payload["profile"] == "standard"
    assert payload["runtime_role"] == "standard"
    assert payload["reporting_period"] == {"label": "2026-03"}
    assert payload["overages"] == {
        "users_total": {
            "used": 12.0,
            "allowed": 10.0,
            "over": 2.0,
        }
    }