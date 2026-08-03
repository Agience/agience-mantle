"""Trusted-issuer artifacts -> verifier trust set."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from mantle.entities.artifact import Artifact
from mantle.services.issuers import ISSUER_CONTENT_TYPE, load_issuer_configs
from mantle.services.oidc import OidcVerifier


def _issuer_artifact(iss="https://idp.test", *, created_by="SYS", role="external", **extra):
    ctx = {"content_type": ISSUER_CONTENT_TYPE, "issuer": iss,
           "audience": "client-1", "jwks": {"keys": [{"kid": "k1"}]}, "role": role}
    ctx.update(extra)
    return Artifact(id="i-1", collection_id="c", state="committed",
                    created_by=created_by, context=json.dumps(ctx),
                    content_type=ISSUER_CONTENT_TYPE)


class TestLoadIssuerConfigs:
    def test_fail_closed_when_system_principal_unknown(self):
        # No system principal -> trust NO issuer artifacts (a label alone must not
        # confer trust, since create is label-blind).
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value=None),
            patch("mantle.db.backend.list_committed_artifacts_by_context_content_type") as q,
        ):
            assert load_issuer_configs(object()) == []
            q.assert_not_called()

    def test_maps_system_owned_committed_artifacts(self):
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
            patch("mantle.db.backend.list_committed_artifacts_by_context_content_type",
                  return_value=[_issuer_artifact()]) as q,
        ):
            cfgs = load_issuer_configs(object())
        # Only system-owned artifacts are queried (created_by passed through).
        assert q.call_args.kwargs["created_by"] == "SYS"
        assert cfgs == [{
            "issuer": "https://idp.test", "audience": "client-1",
            "jwks": {"keys": [{"kid": "k1"}]}, "role": "external",
        }]

    def test_skips_artifact_without_issuer(self):
        art = _issuer_artifact()
        art.context = json.dumps({"content_type": ISSUER_CONTENT_TYPE})  # no issuer
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
            patch("mantle.db.backend.list_committed_artifacts_by_context_content_type",
                  return_value=[art]),
        ):
            assert load_issuer_configs(object()) == []


class TestVerifierRefresh:
    def test_refresh_merges_external_issuer(self):
        v = OidcVerifier(trusted_issuers=[])
        assert "https://idp.test" not in v._by_iss
        with patch("mantle.services.issuers.load_issuer_configs",
                   return_value=[{"issuer": "https://idp.test", "role": "external",
                                  "jwks": {"keys": []}}]):
            v.refresh_from_db(object())
        # External issuer is verifiable AND tenant-namespaced.
        assert "https://idp.test" in v._by_iss
        assert v.is_trusted("https://idp.test")

    def test_platform_role_not_namespaced(self):
        v = OidcVerifier(trusted_issuers=[])
        with patch("mantle.services.issuers.load_issuer_configs",
                   return_value=[{"issuer": "svc", "role": "platform",
                                  "jwks": {"keys": []}}]):
            v.refresh_from_db(object())
        assert "svc" in v._by_iss          # verifiable
        assert not v.is_trusted("svc")     # but native tenant, not external

    def test_handle_issuer_event_refreshes(self):
        from mantle.services import issuers
        fake_db = object()
        with patch("mantle.services.oidc.get_oidc_verifier") as gv:
            issuers._handle_issuer_event(
                type("E", (), {"name": "artifact.updated"})(),
                get_db=lambda: iter([fake_db]),
            )
        gv.return_value.refresh_from_db.assert_called_once_with(fake_db)

    def test_filter_matches_issuer_events_only(self):
        from mantle import event_bus
        from mantle.services import issuers
        flt = event_bus.EventFilter(
            content_type=issuers.ISSUER_CONTENT_TYPE,
            event_names=issuers._ISSUER_EVENT_NAMES,
        )

        def ev(name, ct):
            return event_bus.Event(name=name, payload={}, content_type=ct)

        assert flt.matches(ev("artifact.created", issuers.ISSUER_CONTENT_TYPE))
        assert flt.matches(ev("artifact.updated", issuers.ISSUER_CONTENT_TYPE))
        assert flt.matches(ev("issuer.deleted", issuers.ISSUER_CONTENT_TYPE))
        # Wrong content_type (a normal artifact write) is ignored.
        assert not flt.matches(ev("artifact.created", "text/plain"))
        # Right content_type, irrelevant event name is ignored.
        assert not flt.matches(ev("artifact.viewed", issuers.ISSUER_CONTENT_TYPE))

    async def test_watch_refreshes_on_published_issuer_event(self):
        import asyncio
        from mantle import event_bus
        from mantle.services import issuers

        event_bus.set_event_loop(asyncio.get_event_loop())
        refreshed = []
        with (
            patch("mantle.services.dependencies.get_store_db",
                  side_effect=lambda: iter([object()])),
            patch("mantle.services.oidc.get_oidc_verifier") as gv,
        ):
            gv.return_value.refresh_from_db.side_effect = lambda db: refreshed.append(db)
            task = asyncio.create_task(issuers.watch_issuer_changes())
            await asyncio.sleep(0.05)  # let it subscribe
            await event_bus.publish_event(event_bus.Event(
                name="artifact.created", payload={},
                content_type=issuers.ISSUER_CONTENT_TYPE))
            await asyncio.sleep(0.05)  # let the handler run
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert len(refreshed) == 1

class TestAdminCreateRevoke:
    def test_create_issuer_artifact_is_system_owned(self):
        from mantle.services import issuers
        captured = {}
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
            patch("mantle.db.backend.create_artifact",
                  side_effect=lambda db, art: captured.setdefault("art", art) or art),
        ):
            issuers.create_issuer_artifact(
                object(), issuer="https://idp.test", authorized_by="admin-1",
                jwks={"keys": []}, audience="aud-1")
        a = captured["art"]
        assert a.created_by == "SYS"               # trusted by the loader
        assert a.state == "committed"
        assert a.content_type == ISSUER_CONTENT_TYPE
        ctx = json.loads(a.context)
        assert ctx["issuer"] == "https://idp.test"
        assert ctx["audience"] == "aud-1"
        assert ctx["authorized_by"] == "admin-1"   # provenance roots to the admin
        assert ctx["content_type"] == ISSUER_CONTENT_TYPE

    def test_create_requires_keys(self):
        from mantle.services import issuers
        with patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"):
            with pytest.raises(ValueError):
                issuers.create_issuer_artifact(object(), issuer="x", authorized_by="a")

    def test_create_fails_without_system_principal(self):
        from mantle.services import issuers
        with patch("mantle.services.peer_signing.get_system_principal_id", return_value=None):
            with pytest.raises(RuntimeError):
                issuers.create_issuer_artifact(
                    object(), issuer="x", authorized_by="a", jwks={"keys": []},
                    audience="a")  # valid input → reaches the system-principal check

    def test_create_external_requires_audience(self):
        """Fail fast: an external tenant IdP must bind an audience (else its tokens
        are rejected at verify time — confused-deputy across tenants)."""
        from mantle.services import issuers
        with pytest.raises(ValueError, match="audience"):
            issuers.create_issuer_artifact(
                object(), issuer="x", authorized_by="a", jwks={"keys": []})

    def test_revoke_archives_issuer(self):
        from mantle.services import issuers
        art = Artifact(id="i-1", state="committed", created_by="SYS",
                       context=json.dumps({"content_type": ISSUER_CONTENT_TYPE, "issuer": "x"}))
        updated = {}
        with (
            patch("mantle.db.backend.get_artifact", return_value=art),
            patch("mantle.db.backend.update_artifact",
                  side_effect=lambda db, a: updated.setdefault("a", a)),
        ):
            assert issuers.revoke_issuer_artifact(object(), "i-1", by="admin-1") is True
        assert updated["a"].state == "archived"

    def test_revoke_rejects_non_issuer(self):
        from mantle.services import issuers
        art = Artifact(id="x", state="committed",
                       context=json.dumps({"content_type": "text/plain"}))
        with patch("mantle.db.backend.get_artifact", return_value=art):
            assert issuers.revoke_issuer_artifact(object(), "x", by="a") is False


class TestIssuerRouter:
    async def test_create_requires_admin(self, client):
        with patch("mantle.routers.issuers_router.require_platform_admin",
                   side_effect=HTTPException(status_code=403, detail="nope")):
            r = await client.post("/issuers", json={"issuer": "https://idp.test",
                                                    "jwks": {"keys": []}})
        assert r.status_code == 403

    async def test_create_dispatches_to_service_as_admin(self, client):
        with (
            patch("mantle.routers.issuers_router.require_platform_admin", return_value="admin-1"),
            patch("mantle.services.issuers.create_issuer_artifact",
                  return_value=SimpleNamespace(id="new-id")) as svc,
        ):
            r = await client.post("/issuers", json={"issuer": "https://idp.test",
                                                    "audience": "aud", "jwks": {"keys": []}})
        assert r.status_code == 201
        assert r.json()["id"] == "new-id"
        # The artifact is authorized by the admin (provenance), owned by system.
        assert svc.call_args.kwargs["authorized_by"] == "admin-1"


class TestSeedPlatformIssuers:
    def test_seeds_manifest_anchors_and_env(self):
        from mantle.services import issuers
        manifest = {"trust_anchors": {
            "origin": {"jwks": {"keys": [{"kid": "o"}]}},
            "mantle": {"jwks": {"keys": [{"kid": "m"}]}},
        }}
        created = []
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
            patch("mantle.services.issuers.list_issuer_artifacts", return_value=[]),
            patch("mantle.services.oidc._read_authority_manifest", return_value=manifest),
            patch("origin.config.AUTHORITY_ISSUER", "https://auth.test"),
            patch("origin.config.TRUSTED_ISSUERS",
                  [{"issuer": "https://ext.test", "jwks": {"keys": []}}], create=True),
            patch("mantle.services.issuers.create_issuer_artifact",
                  side_effect=lambda db, **kw: created.append(kw)),
        ):
            n = issuers.seed_platform_issuer_artifacts(object())
        seeded = {c["issuer"]: c for c in created}
        assert n == 4
        assert seeded["origin"]["role"] == "platform"
        assert seeded["mantle"]["role"] == "platform"
        # AUTHORITY_ISSUER seeded as platform, against ORIGIN's jwks (user tokens).
        assert seeded["https://auth.test"]["role"] == "platform"
        assert seeded["https://auth.test"]["jwks"] == {"keys": [{"kid": "o"}]}
        assert seeded["https://ext.test"]["role"] == "external"

    def test_seed_is_idempotent(self):
        from mantle.services import issuers
        existing = SimpleNamespace(context=json.dumps(
            {"content_type": ISSUER_CONTENT_TYPE, "issuer": "origin"}))
        manifest = {"trust_anchors": {"origin": {"jwks": {"keys": [{"kid": "o"}]}}}}
        created = []
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
            patch("mantle.services.issuers.list_issuer_artifacts", return_value=[existing]),
            patch("mantle.services.oidc._read_authority_manifest", return_value=manifest),
            patch("origin.config.AUTHORITY_ISSUER", None),
            patch("origin.config.TRUSTED_ISSUERS", [], create=True),
            patch("mantle.services.issuers.create_issuer_artifact",
                  side_effect=lambda db, **kw: created.append(kw)),
        ):
            n = issuers.seed_platform_issuer_artifacts(object())
        assert n == 0 and created == []   # 'origin' already present -> skipped

    def test_seed_fail_closed_without_system_principal(self):
        from mantle.services import issuers
        with patch("mantle.services.peer_signing.get_system_principal_id", return_value=None):
            assert issuers.seed_platform_issuer_artifacts(object()) == 0


class TestContainerEmitsChangeEvent:
    def test_create_collection_emits_artifact_created(self):
        # Containers are artifacts; create_collection must fire the db chokepoint
        # event too (so container creates drive the change-feed + issuer watcher).
        from mantle.db import backend as dba
        entity = MagicMock()
        db = MagicMock()
        with patch("mantle.db.lattice_api._boundary.emit_artifact_change") as emit:
            dba.create_collection(db, entity)
        db.artifacts.put_artifact.assert_called_once()   # the write chokepoint
        emit.assert_called_once_with(entity, "artifact.created")


class TestArtifactsAuthoritative:
    def test_artifact_issuer_overrides_manifest_anchor(self):
        # conftest writes a manifest with origin/mantle/chorus anchors; the verifier
        # loads them. A platform issuer ARTIFACT for the same iss must WIN (artifacts
        # authoritative), while the manifest still fills the gaps (fallback).
        v = OidcVerifier(trusted_issuers=[])
        assert "origin" in v._by_iss          # seeded from the test manifest
        rotated = {"keys": [{"kid": "rotated"}]}
        with patch("mantle.services.issuers.load_issuer_configs",
                   return_value=[{"issuer": "origin", "role": "platform", "jwks": rotated}]):
            v.refresh_from_db(object())
        assert v._by_iss["origin"]["jwks"] == rotated   # artifact wins
        assert "mantle" in v._by_iss                      # manifest still fills the gap


class TestVerifierRefreshUnknownIss:
    def test_refresh_if_unknown_iss_throttled(self):
        v = OidcVerifier(trusted_issuers=[])
        with (
            patch("mantle.services.oidc.jwt.get_unverified_claims",
                  return_value={"iss": "https://new.test"}),
            patch("mantle.services.issuers.load_issuer_configs",
                  return_value=[{"issuer": "https://new.test", "jwks": {"keys": []}}]) as load,
        ):
            assert v.refresh_if_unknown_iss(object(), "tok") is True
            assert "https://new.test" in v._by_iss
            # Second call within the throttle window does NOT re-hit the store.
            load.reset_mock()
            assert v.refresh_if_unknown_iss(object(), "tok") is False  # now known anyway
            # A different unknown iss is throttled (no reload until interval passes).
            with patch("mantle.services.oidc.jwt.get_unverified_claims",
                       return_value={"iss": "https://other.test"}):
                assert v.refresh_if_unknown_iss(object(), "tok2") is False
                load.assert_not_called()
