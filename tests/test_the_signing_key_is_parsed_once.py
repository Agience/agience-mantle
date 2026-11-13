"""2026-11-13: Mantle's private key was parsed from PEM on every Add-seal check, which runs on every
collection-key open, and a collection of ninety items took longer to list than the app waits."""
from types import SimpleNamespace

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from mantle.services import peer_signing as ps


def _pem():
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                           serialization.NoEncryption()).decode("ascii")


def test_the_key_is_parsed_once_for_any_number_of_signatures(monkeypatch):
    monkeypatch.setattr(ps, "_priv_pem", _pem())
    monkeypatch.setattr(ps, "_loaded_key", None)
    calls = []
    real = serialization.load_pem_private_key
    monkeypatch.setattr(serialization, "load_pem_private_key",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    sigs = [ps.sign_bytes(b"seal %d" % i) for i in range(20)]
    assert all(ps.verify_bytes(b"seal %d" % i, s) for i, s in enumerate(sigs))
    assert not ps.verify_bytes(b"other", sigs[0])
    assert len(calls) == 1


def test_a_new_key_is_parsed_again(monkeypatch):
    monkeypatch.setattr(ps, "_priv_pem", _pem())
    monkeypatch.setattr(ps, "_loaded_key", None)
    first = ps.sign_bytes(b"x")
    monkeypatch.setattr(ps, "_priv_pem", _pem())               # the key rotated
    assert not ps.verify_bytes(b"x", first)
    assert ps.verify_bytes(b"x", ps.sign_bytes(b"x"))


def test_a_verified_seal_is_not_verified_again(monkeypatch):
    from mantle.search.mantle.oracle import OracleService
    seen = []
    monkeypatch.setattr(ps, "verify_bytes", lambda data, sig: seen.append(sig) or sig == "good")
    o = OracleService.__new__(OracleService)
    for _ in range(5):
        assert o._seal_signed("c1", b"pub", {"sig": "good"})
    assert not o._seal_signed("c1", b"pub", {"sig": "bad"})
    assert not o._seal_signed("c1", b"pub", {})
    assert seen == ["good", "bad"]
