"""The admin seed route also provisions the platform operators, on the calling admin's credential
(John, 2026-11-11: the home test node is seeded through the admin route, with no job grant).

Mantle's startup provisions them too, but on a fresh node it starts before Origin answers and finds
no operator. A bare principal of the operator's id holds no credential, and under custody it is given
no key, so the provisioning keeps the caller's own principal when the caller is the operator.
"""
from __future__ import annotations

from mantle.services import acting_principal as ap
from mantle.services.seed_provisioning import platform_operators as P


def test_the_operator_acting_on_their_own_request_is_kept(monkeypatch):
    replaced = []
    monkeypatch.setattr(ap, "acting_as", lambda *a, **k: replaced.append(a) or __import__("contextlib").nullcontext())
    with ap.acting_as("op-1", principal_type="user"):
        pass
    replaced.clear()
    token = ap.set_acting_principal(ap.ActingPrincipal(principal_id="op-1", principal_type="user"))
    try:
        with P._as_operator("op-1"):
            assert ap.current_acting_principal().principal_id == "op-1"
        assert replaced == [], "the caller's own principal, with its credential, is kept"
        with P._as_operator("someone-else"):
            pass
        assert replaced == [("someone-else",)], "anyone else acts as the operator, as before"
    finally:
        ap.reset_acting_principal(token)


async def test_an_operators_failure_is_reported_and_the_seed_still_answers(monkeypatch, tmp_path):
    from mantle.routers import system_router as S

    (tmp_path / "platform").mkdir()
    monkeypatch.setenv("AGIENCE_SEEDS_ROOT", str(tmp_path))
    monkeypatch.setattr(S, "require_platform_admin", lambda auth, db: "op-1", raising=False)

    class Report:
        errors = []

        def summary(self):
            return "artifacts +0 ~0 =41 | errors 0"
    import mantle.services.seed_provisioning as SP
    monkeypatch.setattr(SP, "seed_from_artifacts", lambda db, root: Report())

    def boom(db):
        raise RuntimeError("no key")
    monkeypatch.setattr(P, "ensure_platform_operators", boom)
    import mantle.db.backend as B0
    monkeypatch.setattr(B0, "get_artifact", lambda db, i: None)
    out = await S.seed_platform(auth=None, store_db=object())
    assert out.applied and out.errors == ["platform operators: RuntimeError: no key"]
    assert out.summary.endswith("| operators: not provisioned | errors 1 in all"), out.summary

    # The call that creates the operators grants them to every account, once.
    import mantle.db.backend as B
    import mantle.services.seed_provisioning.platform_email as E
    monkeypatch.setattr(P, "ids", lambda: {P._COLLECTION_SLUG: "coll-1"})
    monkeypatch.setattr(E, "_operator_id", lambda db: "op-1")
    monkeypatch.setattr(P, "ensure_platform_operators", lambda db: True)
    granted = []
    monkeypatch.setattr(P, "grant_to_every_account",
                        lambda db: granted.append(1) or {"accounts": 2, "granted": 2})
    monkeypatch.setattr(B, "get_artifact", lambda db, i: None)
    out = await S.seed_platform(auth=None, store_db=object())
    assert out.errors == [] and out.summary.endswith("| operators granted 2 of 2")
    # A later call converges them and grants nothing: a withheld grant stays withheld (review).
    monkeypatch.setattr(B, "get_artifact", lambda db, i: object())
    out = await S.seed_platform(auth=None, store_db=object())
    assert out.errors == [] and out.summary.endswith("| operators converged") and granted == [1]


async def test_one_call_repeats_the_seed_while_each_pass_clears_errors(monkeypatch, tmp_path):
    # 71/test, 2026-11-12: pass 1 left 12 Add-seal ordering errors that pass 2 cleared.
    from mantle.routers import system_router as S
    import mantle.services.seed_provisioning as SP

    (tmp_path / "platform").mkdir()
    monkeypatch.setenv("AGIENCE_SEEDS_ROOT", str(tmp_path))
    monkeypatch.setattr(S, "require_platform_admin", lambda auth, db: "op-1", raising=False)

    def report(n):
        class R:
            errors = [f"e{i}" for i in range(n)]

            def summary(self):
                return f"errors {n}"
        return R()
    runs = iter([report(12), report(0)])
    monkeypatch.setattr(SP, "seed_from_artifacts", lambda db, root: next(runs))
    monkeypatch.setattr(P, "ensure_platform_operators", lambda db: (_ for _ in ()).throw(RuntimeError("x")))
    out = await S.seed_platform(auth=None, store_db=object())
    assert out.summary.startswith("errors 0 | passes 2 (first: errors 12)"), out.summary

    stuck = iter([report(3), report(3), report(0)])
    monkeypatch.setattr(SP, "seed_from_artifacts", lambda db, root: next(stuck))
    out = await S.seed_platform(auth=None, store_db=object())
    assert out.summary.startswith("errors 3 | passes 2"), "a pass that clears nothing ends it"
