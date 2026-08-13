"""Tests for the platform outbound-email provisioner.

Covers the no-op skip branches and the happy path: it provisions the email
operator graph (operator + authorizer + 2 credential artifacts), wires the
operator→authorizer edge, writes the two credential values into artifact
content, and grants the operator invoke/read — all from GMAIL_OAUTH_* env,
owned by the platform operator.

The last class runs the provisioner against a REAL lattice with a REAL oracle,
because the claim being made — "a credential is an ordinary artifact, encrypted
by the envelope and authorized by the light cone" — is only worth anything if
the value round-trips through the ordinary artifact path and comes back out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from mantle.services.seed_provisioning import platform_email as pe  # noqa: E402

_OPERATOR = "operator-1"


@pytest.fixture
def gmail_env(monkeypatch):
    monkeypatch.setenv("EMAIL_PROVIDER", "gmail")
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_ID", "cid.apps.googleusercontent.com")
    monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("GMAIL_OAUTH_REFRESH_TOKEN", "rtoken")
    monkeypatch.setenv("PLATFORM_EMAIL_ADDRESS", "connect@agience.ai")


@pytest.fixture
def wired(monkeypatch):
    """Stub db/topology helpers; return the mocks for assertions."""
    monkeypatch.setattr(pe, "derive_uuid", lambda ns, namespace, slug: f"id-{slug}")
    monkeypatch.setattr(pe, "get_instance_namespace", lambda: "ns")
    monkeypatch.setattr(pe, "register_id", lambda *a, **k: None)
    monkeypatch.setattr(pe, "_operator_id", lambda db: _OPERATOR)

    get_artifact = MagicMock(return_value=None)   # nothing exists yet
    create_artifact = MagicMock()
    update_artifact = MagicMock()
    add_edge = MagicMock()
    upsert_grant = MagicMock()
    monkeypatch.setattr(pe, "db_get_artifact", get_artifact)
    monkeypatch.setattr(pe, "db_create_artifact", create_artifact)
    monkeypatch.setattr(pe, "db_update_artifact", update_artifact)
    monkeypatch.setattr(pe, "db_add_edge", add_edge)
    monkeypatch.setattr(pe, "db_upsert_user_collection_grant", upsert_grant)

    # System principal resolution is imported inside the function from core;
    # default it to "" so grant-count assertions are deterministic (operator-only).
    # The dedicated test overrides it to exercise the system-principal branch.
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "")

    return SimpleNamespace(
        get_artifact=get_artifact, create_artifact=create_artifact,
        update_artifact=update_artifact, add_edge=add_edge,
        upsert_grant=upsert_grant,
    )


def _created(wired, content_type):
    """The artifact entities created with *content_type*."""
    return [c.args[1] for c in wired.create_artifact.call_args_list
            if c.args[1].content_type == content_type]


def test_skips_when_provider_not_gmail(monkeypatch, wired):
    monkeypatch.setenv("EMAIL_PROVIDER", "smtp")
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()


def test_skips_when_creds_missing(monkeypatch, wired):
    monkeypatch.setenv("EMAIL_PROVIDER", "gmail")
    monkeypatch.delenv("GMAIL_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GMAIL_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GMAIL_OAUTH_REFRESH_TOKEN", raising=False)
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()


def test_skips_when_no_operator(gmail_env, wired, monkeypatch):
    monkeypatch.setattr(pe, "_operator_id", lambda db: "")
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()


def test_happy_path_provisions_graph(gmail_env, wired):
    pe.ensure_platform_email_sender(db=MagicMock())

    # Four artifacts: operator + authorizer + 2 credentials.
    cts = {c.args[1].content_type for c in wired.create_artifact.call_args_list}
    assert cts == {pe._OPERATOR_CT, pe._AUTHORIZER_CT, pe._CREDENTIAL_CT}
    assert wired.create_artifact.call_count == 4
    for c in wired.create_artifact.call_args_list:
        assert c.args[1].created_by == _OPERATOR

    # Operator→authorizer typed edge.
    wired.add_edge.assert_called_once()
    assert wired.add_edge.call_args.kwargs["relationship"] == "authorizer"

    # Grants: operator gets invoke on the operator artifact, read on all four.
    assert wired.upsert_grant.call_count == 4
    invoke_grants = [c for c in wired.upsert_grant.call_args_list if c.kwargs.get("can_invoke")]
    assert len(invoke_grants) == 1
    assert invoke_grants[0].kwargs["collection_id"] == "id-platform-email-sender"


def test_the_credential_value_is_the_artifact_content(gmail_env, wired):
    """The value goes in `content` — the one field the envelope encrypts."""
    pe.ensure_platform_email_sender(db=MagicMock())
    creds = _created(wired, pe._CREDENTIAL_CT)

    assert len(creds) == 2
    values = {json.loads(a.content)["value"] for a in creds}
    assert values == {"csecret", "rtoken"}


def test_no_credential_value_reaches_plaintext_context(gmail_env, wired):
    """`context` is stored in the clear, so a value there would be a value in the open.

    Asserted over EVERY artifact in the graph, not just the credentials: the authorizer
    holds the client_id and the sender address, and a future field added to any of these
    contexts must not be a place a secret can land.
    """
    pe.ensure_platform_email_sender(db=MagicMock())

    for c in wired.create_artifact.call_args_list:
        art = c.args[1]
        assert "csecret" not in (art.context or "")
        assert "rtoken" not in (art.context or "")
        assert "csecret" not in (art.name or "") + (art.description or "")

    creds = _created(wired, pe._CREDENTIAL_CT)
    kinds = {json.loads(a.context)["kind"] for a in creds}
    assert kinds == {"oauth_client_secret", "oauth_refresh_token"}
    for a in creds:
        ctx = json.loads(a.context)
        assert ctx["provider"] == "google"
        assert ctx["label"]                     # the label lives in the clear, by design
        assert ctx["is_default"] is True        # so does which one is default


def test_the_writes_are_made_as_the_operator(gmail_env, wired, monkeypatch):
    """Content encryption needs a key, and the creator is who holds it for the artifact
    it is creating — so the acting principal during the writes must be the operator."""
    from mantle.services.acting_principal import current_acting_principal

    seen = []
    original = pe.db_create_artifact

    def _record(db, entity):
        principal = current_acting_principal()
        seen.append(principal.principal_id if principal else None)
        return original(db, entity)

    monkeypatch.setattr(pe, "db_create_artifact", _record)
    pe.ensure_platform_email_sender(db=MagicMock())

    assert seen and set(seen) == {_OPERATOR}


def test_idempotent_when_graph_exists(gmail_env, wired):
    # db_get_artifact returns an Artifact ENTITY (not a dict); _ensure_artifact
    # reconciles its attributes in place. A SimpleNamespace stands in for the
    # existing entity so those attribute writes succeed.
    # One entity per id, so the four reconciled artifacts stay distinguishable.
    wired.get_artifact.side_effect = lambda db, artifact_id: SimpleNamespace(_key=artifact_id)
    pe.ensure_platform_email_sender(db=MagicMock())
    wired.create_artifact.assert_not_called()      # no re-create
    wired.add_edge.assert_not_called()             # edge not re-added
    assert wired.update_artifact.call_count == 4   # values re-written, so rotation self-heals
    contents = [c.args[1].content for c in wired.update_artifact.call_args_list]
    assert {"csecret", "rtoken"} <= {
        json.loads(c).get("value") for c in contents if c.startswith('{"value"')
    }


def test_grants_system_principal_when_available(gmail_env, wired, monkeypatch):
    """When the system principal resolves, it's granted the same mail perms as
    the operator — granted_by the operator (the provenance that roots it to a
    person) — so webhook/background sends can act AS it."""
    monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "sys-principal-1")
    pe.ensure_platform_email_sender(db=MagicMock())

    # 8 grants: 4 (operator) + 4 (system principal).
    assert wired.upsert_grant.call_count == 8
    grantees = {c.kwargs["user_id"] for c in wired.upsert_grant.call_args_list}
    assert grantees == {_OPERATOR, "sys-principal-1"}
    # Every grant is issued BY the operator — that's the rooting to a person.
    for c in wired.upsert_grant.call_args_list:
        assert c.kwargs["granted_by"] == _OPERATOR
    # The system principal also gets invoke on the operator artifact.
    sys_invoke = [c for c in wired.upsert_grant.call_args_list
                  if c.kwargs["user_id"] == "sys-principal-1" and c.kwargs.get("can_invoke")]
    assert len(sys_invoke) == 1
    assert sys_invoke[0].kwargs["collection_id"] == "id-platform-email-sender"


# ---------------------------------------------------------------------------
# The round trip, against the real store and the real oracle
# ---------------------------------------------------------------------------

class TestTheCredentialIsAnOrdinaryArtifact:
    """No stubbed crypto and no stubbed grant check: a real lattice, a real
    `OracleService` with the real `LightConeGrantVerifier`, and the ordinary
    artifact read. What is asserted is that the value comes back through that
    path — and that the bytes on disk are not the value."""

    @pytest.fixture
    def stack(self, tmp_path, monkeypatch, gmail_env):
        from cryptography.fernet import Fernet

        from mantle.db.lattice_api import LatticeDatabase
        from mantle.search.mantle import wiring
        from mantle.search.mantle.key_provider import LocalKeyProvider
        from mantle.search.mantle.oracle import (
            LatticeMasterKeyStore,
            LightConeGrantVerifier,
            OracleService,
        )

        db = LatticeDatabase(str(tmp_path / "email.db"), origin="test-platform-email")

        # ttl_s=0 — no memoized authorization decision, so every check below really
        # re-reads the grant ledger rather than a cache warmed by an earlier one.
        oracle = OracleService(
            LatticeMasterKeyStore(LocalKeyProvider(Fernet(Fernet.generate_key())), lambda: db),
            grant_verifier=LightConeGrantVerifier(db, ttl_s=0),
        )
        monkeypatch.setattr(wiring, "_oracle_singleton", oracle)

        monkeypatch.setattr(pe, "derive_uuid", lambda ns, namespace, slug: f"id-{slug}")
        monkeypatch.setattr(pe, "get_instance_namespace", lambda: "ns")
        monkeypatch.setattr(pe, "register_id", lambda *a, **k: None)
        monkeypatch.setattr(pe, "_operator_id", lambda _db: _OPERATOR)
        monkeypatch.setattr("mantle.services.peer_signing.get_system_principal_id", lambda: "")

        assert pe.ensure_platform_email_sender(db) is True
        return db

    def test_the_value_reads_back_through_the_ordinary_artifact_path(self, stack):
        from mantle.db.backend import get_artifact
        from mantle.services.acting_principal import acting_as

        with acting_as(_OPERATOR, principal_type="user"):
            art = get_artifact(stack, "id-platform-email-client-secret")
            rt = get_artifact(stack, "id-platform-email-refresh-token")

        assert art.content_type == pe._CREDENTIAL_CT
        assert json.loads(art.content)["value"] == "csecret"
        assert json.loads(rt.content)["value"] == "rtoken"

    def test_the_stored_bytes_are_ciphertext(self, stack):
        """Read UNDER the artifact API — what the store itself is holding."""
        doc = stack.artifacts.get_artifact("id-platform-email-client-secret")

        assert doc.get("content_encrypted") is True
        assert "csecret" not in (doc.get("content") or "")

    def test_a_principal_without_the_grant_does_not_get_the_value(self, stack):
        """The light cone is the whole authorization story, so someone outside it
        must fail — otherwise the round trip above proves only that decryption works."""
        from mantle.db.backend import get_artifact
        from mantle.services.acting_principal import KeyCustodyDenied, acting_as

        with acting_as("stranger-1", principal_type="user"):
            with pytest.raises(KeyCustodyDenied):
                get_artifact(stack, "id-platform-email-client-secret")

    def test_a_rerun_rewrites_the_value_and_it_still_reads_back(self, stack, monkeypatch):
        """Rotation: the env changes, the provisioner re-runs, the artifact converges."""
        from mantle.db.backend import get_artifact
        from mantle.services.acting_principal import acting_as

        monkeypatch.setenv("GMAIL_OAUTH_CLIENT_SECRET", "csecret-rotated")
        assert pe.ensure_platform_email_sender(stack) is True

        with acting_as(_OPERATOR, principal_type="user"):
            art = get_artifact(stack, "id-platform-email-client-secret")
        assert json.loads(art.content)["value"] == "csecret-rotated"
