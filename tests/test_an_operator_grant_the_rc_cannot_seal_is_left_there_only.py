"""The release candidate leaves an operator grant it cannot seal; any other plane fails on it.

2026-11-09, release 2026.11.2: the RC runs on a copy of production's store with a key encryption
key of its own, so a grant written under a collection key production wrapped (the accounts made
since 2026.11.0's promote) cannot be sealed there, and `platform_operators` failed the RC.
"""
from __future__ import annotations

import pytest


def _db_with(accounts):
    class Db:
        pass
    return Db(), [{"grantee_type": "user", "grantee_id": a} for a in accounts]


@pytest.mark.parametrize("plane, fails", [("rc", False), ("production", True), ("", True)])
def test_an_unsealable_grant_is_left_only_on_the_rc(monkeypatch, plane, fails):
    from mantle.db import lattice_api as la
    from mantle.db.doc_boundary import ContentEncryptionError
    from mantle.services import peer_signing
    from mantle.services.seed_provisioning import platform_operators as po

    monkeypatch.setenv("AGIENCE_PLANE", plane)
    db, docs = _db_with(["a", "b", "c"])
    monkeypatch.setattr(la, "_grant_docs", lambda _db, state=None: docs)
    monkeypatch.setattr(peer_signing, "get_system_principal_id", lambda: "")

    def grant(_db, user_id):
        if user_id == "b":
            try:
                raise RuntimeError("unwrap")
            except RuntimeError as inner:
                raise ContentEncryptionError("context field encryption unavailable") from inner
        return 2
    monkeypatch.setattr(po, "grant_to_account", grant)

    if fails:
        with pytest.raises(ContentEncryptionError):
            po.grant_to_every_account(db)
    else:
        assert po.grant_to_every_account(db) == {"accounts": 3, "granted": 2, "left_on_rc": 1}


def test_any_other_fault_fails_on_the_rc_too(monkeypatch):
    from mantle.db import lattice_api as la
    from mantle.services import peer_signing
    from mantle.services.seed_provisioning import platform_operators as po

    monkeypatch.setenv("AGIENCE_PLANE", "rc")
    db, docs = _db_with(["a"])
    monkeypatch.setattr(la, "_grant_docs", lambda _db, state=None: docs)
    monkeypatch.setattr(peer_signing, "get_system_principal_id", lambda: "")
    monkeypatch.setattr(po, "grant_to_account", lambda _db, user_id: (_ for _ in ()).throw(RuntimeError("disk")))
    with pytest.raises(RuntimeError):
        po.grant_to_every_account(db)


@pytest.mark.parametrize("plane, ok", [("rc", True), ("production", False)])
def test_the_rc_leaves_the_operators_collection_it_cannot_seal(monkeypatch, plane, ok):
    """The operators' collection itself is converged under production's collection key; the RC
    leaves it as production has it and goes on to the grants."""
    from mantle.db.doc_boundary import ContentEncryptionError
    from mantle.services.seed_provisioning import platform_operators as po

    monkeypatch.setenv("AGIENCE_PLANE", plane)

    def ensure(_db):
        raise ContentEncryptionError("context field encryption unavailable")
    monkeypatch.setattr(po, "ensure_platform_operators", ensure)
    monkeypatch.setattr(po, "grant_to_every_account", lambda _db: {"accounts": 0, "granted": 0})
    import mantle.services.dependencies as deps
    import mantle.services.peer_signing as ps
    import prism.trust.key_manager as km
    from mantle.services import platform_settings_service as pss
    from mantle import config
    monkeypatch.setattr(km, "init_encryption_key", lambda: None)
    monkeypatch.setattr(km, "init_nonce_secret", lambda: None)
    monkeypatch.setattr(ps, "init", lambda: None)
    monkeypatch.setattr(deps, "get_store_db", lambda: iter([object()]))
    monkeypatch.setattr(pss.settings, "load_all", lambda _db: None)
    monkeypatch.setattr(config, "set_settings_provider", lambda _f: None)
    monkeypatch.setattr(config, "load_settings_from_db", lambda: None)
    if ok:
        assert po.main() == 0
    else:
        with pytest.raises(ContentEncryptionError):
            po.main()
