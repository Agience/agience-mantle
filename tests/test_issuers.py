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
    return Artifact(id="i-1", collection_id="c", state="live",
                    created_by=created_by, context=json.dumps(ctx),
                    content_type=ISSUER_CONTENT_TYPE)


class TestLoadIssuerConfigs:
    def test_fail_closed_when_system_principal_unknown(self):
        # No system principal -> trust NO issuer artifacts (a label alone must not
        # confer trust, since create is label-blind).
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value=None),
            patch("mantle.db.backend.list_live_artifacts_by_context_content_type") as q,
        ):
            assert load_issuer_configs(object()) == []
            q.assert_not_called()

    def test_maps_system_owned_committed_artifacts(self):
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
            patch("mantle.db.backend.list_live_artifacts_by_context_content_type",
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
            patch("mantle.db.backend.list_live_artifacts_by_context_content_type",
                  return_value=[art]),
        ):
            assert load_issuer_configs(object()) == []


class TestVerifierRefresh:
    def test_refresh_keeps_a_configured_external_issuer_live(self):
        """A stored artifact keeps a configured issuer live; it never adds one (review 12)."""
        cfg = {"issuer": "https://idp.test", "audience": "a", "jwks": {"keys": []}}
        v = OidcVerifier(trusted_issuers=[cfg])
        with patch("mantle.services.issuers.load_issuer_configs",
                   return_value=[{"issuer": "https://idp.test", "role": "external"}]):
            v.refresh_from_db(object())
        assert "https://idp.test" in v._by_iss and v.is_trusted("https://idp.test")
        with patch("mantle.services.issuers.load_issuer_configs", return_value=[]):
            v.refresh_from_db(object())
        assert not v.is_trusted("https://idp.test")        # revoked: no live artifact

    def test_an_artifact_alone_adds_no_issuer(self):
        v = OidcVerifier(trusted_issuers=[])
        with patch("mantle.services.issuers.load_issuer_configs",
                   return_value=[{"issuer": "svc", "role": "platform", "jwks": {"keys": []}},
                                 {"issuer": "https://idp.test", "role": "external",
                                  "jwks": {"keys": []}}]):
            v.refresh_from_db(object())
        assert "svc" not in v._by_iss and "https://idp.test" not in v._by_iss

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
        from mantle.events import event_bus
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
        from mantle.events import event_bus
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
        assert a.state == "live"
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
        """A missing system principal must fail closed rather than fabricate an owner.

        Asserts the specific type now, not `RuntimeError` [P-9, 2026-08-26].

        `create_issuer_artifact` used to raise a bare `RuntimeError` here, and `system_router`
        caught `RuntimeError` and returned `str(exc)` as a 503 — correct for THIS raise, whose
        message is curated, and wrong for every other `RuntimeError` the call stack can produce,
        whose text would have gone straight into a response body.

        `SystemPrincipalUnavailable` already existed for exactly this condition, and its own
        docstring names this module: *"a fabricated system identity would acquire whatever grants
        happened to match it, which is the failure mode `issuers.py` already fails closed against."*
        The bare `RuntimeError` was the drift; this is not a new contract, it is the intended one.
        """
        from mantle.services import issuers
        from mantle.services.acting_principal import SystemPrincipalUnavailable
        with patch("mantle.services.peer_signing.get_system_principal_id", return_value=None):
            with pytest.raises(SystemPrincipalUnavailable):
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
        art = Artifact(id="i-1", state="live", created_by="SYS",
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
        art = Artifact(id="x", state="live",
                       context=json.dumps({"content_type": "text/plain"}))
        with patch("mantle.db.backend.get_artifact", return_value=art):
            assert issuers.revoke_issuer_artifact(object(), "x", by="a") is False


class TestIssuerRouter:
    async def test_no_route_adds_an_issuer(self, client):
        r = await client.post("/system/issuers", json={"issuer": "https://idp.test",
                                                      "jwks": {"keys": []}})
        assert r.status_code in (404, 405)


class TestSeedPlatformIssuers:
    def test_seeds_only_configured_external_issuers(self):
        """The manifest's anchors are never copied into the store (review 12, H2)."""
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
            patch("mantle.config.AUTHORITY_ISSUER", "https://auth.test"),
            patch("mantle.config.TRUSTED_ISSUERS",
                  [{"issuer": "https://ext.test", "jwks": {"keys": []}}]),
            patch("mantle.services.issuers.create_issuer_artifact",
                  side_effect=lambda db, **kw: created.append(kw)),
        ):
            n = issuers.seed_platform_issuer_artifacts(object())
        assert n == 1 and [c["issuer"] for c in created] == ["https://ext.test"]
        assert created[0]["role"] == "external"

    def test_seed_is_idempotent(self):
        from mantle.services import issuers
        existing = SimpleNamespace(context=json.dumps(
            {"content_type": ISSUER_CONTENT_TYPE, "issuer": "https://ext.test"}))
        created = []
        with (
            patch("mantle.services.peer_signing.get_system_principal_id", return_value="SYS"),
            patch("mantle.services.issuers.list_issuer_artifacts", return_value=[existing]),
            patch("mantle.config.AUTHORITY_ISSUER", None),
            patch("mantle.config.TRUSTED_ISSUERS",
                  [{"issuer": "https://ext.test", "jwks": {"keys": []}}]),
            patch("mantle.services.issuers.create_issuer_artifact",
                  side_effect=lambda db, **kw: created.append(kw)),
        ):
            n = issuers.seed_platform_issuer_artifacts(object())
        assert n == 0 and created == []   # already present -> skipped

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


class TestTheManifestWins:
    def test_an_artifact_never_replaces_a_manifest_anchor(self):
        # conftest writes a manifest with origin/mantle/peer anchors; the verifier loads them, and
        # an artifact naming one of them changes nothing (review 12, H2).
        v = OidcVerifier(trusted_issuers=[])
        before = v._by_iss["origin"]["jwks"]
        with patch("mantle.services.issuers.load_issuer_configs",
                   return_value=[{"issuer": "origin", "role": "platform",
                                  "jwks": {"keys": [{"kid": "rotated"}]}}]):
            v.refresh_from_db(object())
        assert v._by_iss["origin"]["jwks"] == before
        assert "mantle" in v._by_iss


class TestVerifierRefreshUnknownIss:
    def test_refresh_if_unknown_iss_throttled(self):
        cfg = {"issuer": "https://new.test", "audience": "a", "jwks": {"keys": []}}
        v = OidcVerifier(trusted_issuers=[cfg])
        with patch("mantle.services.issuers.load_issuer_configs", return_value=[]):
            v.refresh_from_db(object())             # configured, not yet live
        v._last_refresh = 0.0
        with (
            patch("mantle.services.oidc.jwt.get_unverified_claims",
                  return_value={"iss": "https://new.test"}),
            patch("mantle.services.issuers.load_issuer_configs",
                  return_value=[{"issuer": "https://new.test"}]) as load,
        ):
            assert v.refresh_if_unknown_iss(object(), "tok") is True
            assert "https://new.test" in v._by_iss
            load.reset_mock()
            assert v.refresh_if_unknown_iss(object(), "tok") is False  # now known anyway
            with patch("mantle.services.oidc.jwt.get_unverified_claims",
                       return_value={"iss": "https://other.test"}):
                assert v.refresh_if_unknown_iss(object(), "tok2") is False
                load.assert_not_called()
