"""Review 17: the mesh plane is Mantle's own content bucket, checked as Ember checks its mirror
(`ember.mantle_api.refuse_mantle_bucket`). A plane on another service's bucket (Ember's, or any
`MANTLE_FORBIDDEN_BUCKETS` names), or on one that is not `CONTENT_DURABLE_BUCKET`, is no plane."""
from __future__ import annotations

from types import SimpleNamespace

from mantle.mesh import sync


def _store(bucket):
    return SimpleNamespace(content=SimpleNamespace(remote=SimpleNamespace(bucket=bucket)),
                           keys_dir="")


def test_the_plane_must_be_mantle_s_own_bucket(monkeypatch):
    monkeypatch.delenv(sync.MESH_SPOOL_DIR, raising=False)
    monkeypatch.setenv("CONTENT_DURABLE_BUCKET", "mantle-content")
    monkeypatch.setenv("EMBER_OVH_BUCKET", "ember-mirror")
    monkeypatch.setenv("MANTLE_FORBIDDEN_BUCKETS", "shared-one")
    assert sync._mesh_s3(_store("mantle-content")) is not None
    for other in ("ember-mirror", "shared-one", "someone-else"):
        assert sync._mesh_s3(_store(other)) is None, other


def test_a_plane_with_no_bucket_named_is_not_checked(monkeypatch):
    monkeypatch.delenv(sync.MESH_SPOOL_DIR, raising=False)
    monkeypatch.delenv("CONTENT_DURABLE_BUCKET", raising=False)
    monkeypatch.setenv("EMBER_OVH_BUCKET", "ember-mirror")
    assert sync._mesh_s3(_store("ember-mirror")) is None        # Ember's is never Mantle's
    assert sync._mesh_s3(_store("any")) is not None
