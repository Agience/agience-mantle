"""`search/mantle/key_provider.py` — KEK custody selection, previously untested.

Surfaced by the same coverage survey as `grant_service` and `grant_store` (`NEXT.md §G.3`): modules
no test references, ranked by security surface. This one wraps the platform KEK that protects every
per-principal master DEK (`oracle.LatticeMasterKeyStore`), and it had **no tests**.

⛔ THE PROPERTY THAT MATTERS IS THE REFUSAL, NOT THE ROUND-TRIP.

`build_key_provider` reads `MANTLE_KEK_PROVIDER` and its own docstring states the contract: it
"raises if the selected provider can't be constructed — the oracle wiring treats that as 'no search'
(503), **never a silent plaintext fallback**."

That is the whole security value of the indirection. A managed deployment sets
`MANTLE_KEK_PROVIDER=kms` so the KEK never enters the process; if a typo (`kmss`, `KMS `), a dropped
env var, or a missing `MANTLE_KMS_KEY_ID` quietly fell back to `local`, the platform would keep
working — wrapping DEKs under an on-box key — and nothing would look wrong. Custody would have
silently descended the maturity ladder the module exists to climb. Every test below asserts a
REFUSAL; a round-trip-only suite would pass against a version that defaulted to local on error.

⚠ NO NETWORK, NO CREDENTIALS. The KMS and Vault providers are exercised through injected fakes, so
these run anywhere. What is asserted about them is what can be asserted without a real backend: that
they refuse incomplete wiring, and that **only the DEK crosses the boundary — never the KEK**.
"""
from __future__ import annotations

import base64

import pytest

from mantle.search.mantle.key_provider import (AwsKmsKeyProvider, LocalKeyProvider,
                                        VaultTransitKeyProvider, build_key_provider)

DEK = b"\x01" * 32          # a master DEK: 32 bytes, well under every KMS direct-encrypt limit


# ── the selector must refuse, never downgrade ────────────────────────────────
@pytest.mark.parametrize("value", ["kmss", "aws", "hsm", "plaintext", "none", "  ", "loca1"])
def test_an_unknown_provider_raises_instead_of_falling_back(monkeypatch, value):
    """⛔ THE CORE REFUSAL. A silent fallback to `local` would mean a managed deployment kept
    running with the KEK on-box after a typo — working, and wrong, with nothing to see."""
    monkeypatch.setenv("MANTLE_KEK_PROVIDER", value)
    with pytest.raises(ValueError) as ei:
        build_key_provider()
    assert "MANTLE_KEK_PROVIDER" in str(ei.value)


def test_kms_without_a_key_id_raises_at_wiring_not_at_first_wrap(monkeypatch):
    """Fail where the operator can see it. A provider constructed without its key id would raise on
    the first DEK wrap instead — during a request, long after the misconfiguration."""
    monkeypatch.setenv("MANTLE_KEK_PROVIDER", "kms")
    monkeypatch.delenv("MANTLE_KMS_KEY_ID", raising=False)
    with pytest.raises(ValueError):
        build_key_provider()


@pytest.mark.parametrize("missing", ["VAULT_ADDR", "VAULT_TOKEN", "MANTLE_VAULT_TRANSIT_KEY"])
def test_vault_refuses_any_incomplete_wiring(monkeypatch, missing):
    """Each of the three is individually required — a provider missing one cannot wrap, and
    discovering that at first use is discovering it in production."""
    monkeypatch.setenv("MANTLE_KEK_PROVIDER", "vault")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.example")
    monkeypatch.setenv("VAULT_TOKEN", "tok")
    monkeypatch.setenv("MANTLE_VAULT_TRANSIT_KEY", "mantle-kek")
    monkeypatch.delenv(missing, raising=False)
    with pytest.raises(ValueError):
        build_key_provider()


def test_the_selector_tolerates_case_and_whitespace(monkeypatch):
    """`.strip().lower()` is deliberate — an env var with a trailing space must select the provider
    the operator meant, not fall into the refusal path. The refusal is for genuinely unknown values."""
    from cryptography.fernet import Fernet
    monkeypatch.setenv("MANTLE_KEK_PROVIDER", "  KMS  ")
    monkeypatch.setenv("MANTLE_KMS_KEY_ID", "arn:aws:kms:::key/abc")
    assert isinstance(build_key_provider(), AwsKmsKeyProvider)
    del Fernet


# ── the local provider actually protects the DEK ─────────────────────────────
def _local():
    from cryptography.fernet import Fernet
    return LocalKeyProvider(Fernet(Fernet.generate_key()))


def test_local_wrap_unwrap_round_trips():
    p = _local()
    assert p.unwrap(p.wrap(DEK)) == DEK


def test_the_wrapped_token_does_not_contain_the_dek():
    """A "wrap" that stored the DEK recoverably would round-trip perfectly and protect nothing."""
    p = _local()
    token = p.wrap(DEK)
    assert DEK not in token.encode()
    assert base64.b64encode(DEK).decode().rstrip("=") not in token


def test_a_foreign_token_does_not_unwrap():
    """Tokens are KEK-bound: another platform's key must not open ours."""
    a, b = _local(), _local()
    from cryptography.fernet import InvalidToken
    with pytest.raises(InvalidToken):
        b.unwrap(a.wrap(DEK))


# ── non-exportable providers: only the DEK crosses the boundary ──────────────
class _FakeKms:
    """Records what actually goes over the wire."""

    def __init__(self):
        self.encrypt_calls, self.decrypt_calls = [], []

    def encrypt(self, KeyId, Plaintext):
        self.encrypt_calls.append((KeyId, Plaintext))
        return {"CiphertextBlob": b"KMSWRAPPED:" + Plaintext}

    def decrypt(self, KeyId, CiphertextBlob):
        self.decrypt_calls.append((KeyId, CiphertextBlob))
        return {"Plaintext": CiphertextBlob.split(b"KMSWRAPPED:", 1)[1]}


def test_kms_sends_only_the_dek_and_round_trips():
    """The custody claim for a non-exportable KEK: "only the 32-byte DEK plaintext transits the wire
    — never the KEK." Pinned by inspecting the call, because the claim is about what is SENT."""
    fake = _FakeKms()
    p = AwsKmsKeyProvider("arn:aws:kms:::key/abc", client=fake)
    token = p.wrap(DEK)
    assert p.unwrap(token) == DEK
    (key_id, sent), = fake.encrypt_calls
    assert sent == DEK and len(sent) == 32          # the DEK, nothing more
    assert key_id == "arn:aws:kms:::key/abc"        # the KEK is named, never carried
    assert isinstance(token, str)                   # storable


def test_kms_token_is_base64_not_raw_bytes():
    """The token is persisted as text in the lattice; raw bytes would corrupt on the JSON boundary."""
    p = AwsKmsKeyProvider("k", client=_FakeKms())
    base64.b64decode(p.wrap(DEK))                   # raises if not valid base64


def test_vault_round_trips_without_a_network(monkeypatch):
    p = VaultTransitKeyProvider("https://vault.example", "tok", "mantle-kek")
    seen = {}

    def fake_post(op, payload):
        seen[op] = payload
        if op == "encrypt":
            return {"ciphertext": "vault:v1:" + payload["plaintext"]}
        return {"plaintext": payload["ciphertext"].split("vault:v1:", 1)[1]}

    monkeypatch.setattr(p, "_post", fake_post)
    token = p.wrap(DEK)
    assert token.startswith("vault:v1:")
    assert p.unwrap(token) == DEK
    assert base64.b64decode(seen["encrypt"]["plaintext"]) == DEK   # the DEK, base64'd — not the KEK


def test_vault_addr_trailing_slash_does_not_double_up():
    """`addr.rstrip("/")` — a trailing slash in VAULT_ADDR would otherwise build `//v1/transit/...`,
    which some proxies 404 and others silently redirect."""
    p = VaultTransitKeyProvider("https://vault.example/", "tok", "k")
    assert p._addr == "https://vault.example"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
