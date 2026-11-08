"""Review 26, open item (c): a seed pass's plan can be written ahead, and the pass makes exactly it.

A seed credential makes only the entries in its plan (Origin, review 25), but a new grant's id was
a random UUID, so a pass's entries could not be known before it ran. Now a pass planned with a
salt (`services/seed_plan.planning`) derives its new grant ids from that salt; the planning run
prints each entry's body and digest; and the pass run on that plan (`enforcing`) derives the same
ids, submits only planned entries and reports any planned entry it did not make.
"""
from __future__ import annotations

import pytest

from mantle.db import lattice_api as la
from mantle.db import reach_journal
from mantle.services import seed_plan

from .test_there_is_no_promotion_state import real_node

SALT = "0123456789abcdef0123456789abcdef"


def _the_pass(db, who="u-1"):
    la.upsert_user_collection_grant(db, user_id=who, collection_id="c-plan", granted_by="op",
                                    can_read=True, can_invoke=True)


def _grant_entries(entries):
    return [e for e in entries if e.get("kind") == "grant"]


@pytest.fixture()
def two_stores(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = real_node(tmp_path / "a", monkeypatch)
    b = real_node(tmp_path / "b", monkeypatch)
    return a, b


@pytest.fixture()
def origin():
    """An authority that takes every entry, as Origin takes a seed's planned ones."""
    taken = []

    def take(entry, *, proof=None, compensates=None, batch=None):
        taken.append(entry)
        return None

    reach_journal.install(take)
    try:
        yield taken
    finally:
        reach_journal.uninstall()


def test_a_planned_pass_makes_exactly_its_plan(two_stores, origin, tmp_path):
    plan_db, real_db = two_stores
    with seed_plan.planning(SALT) as planned:
        _the_pass(plan_db)
    grants = _grant_entries(planned)
    assert grants, "planning recorded no grant entry"
    plan = seed_plan.write_plan(str(tmp_path / "plan.json"), "test", SALT, planned)
    assert seed_plan.read_plan(str(tmp_path / "plan.json")) == plan

    with seed_plan.enforcing(plan) as report:
        _the_pass(real_db)
    assert [seed_plan.digest(e) for e in _grant_entries(origin)] == \
        [seed_plan.digest(e) for e in grants]
    assert not report["missing"]


def test_an_entry_outside_the_plan_is_refused_before_origin(two_stores, origin, tmp_path):
    plan_db, real_db = two_stores
    with seed_plan.planning(SALT) as planned:
        _the_pass(plan_db, "u-1")
    plan = seed_plan.write_plan(str(tmp_path / "plan.json"), "test", SALT, planned)
    with seed_plan.enforcing(plan) as report:
        with pytest.raises(reach_journal.ReachChangeRefused):
            _the_pass(real_db, "u-2")
    assert not _grant_entries(origin)
    assert report["missing"]                              # what it planned and did not make
    assert not la.get_active_grants_for_principal_resource(real_db, "u-2", "c-plan")


def test_outside_a_plan_a_grant_id_is_random(two_stores):
    a, b = two_stores
    _the_pass(a)
    _the_pass(b)
    ga = la.get_active_grants_for_principal_resource(a, "u-1", "c-plan")[0]
    gb = la.get_active_grants_for_principal_resource(b, "u-1", "c-plan")[0]
    assert ga.id != gb.id
    with seed_plan.planned_ids(SALT):
        assert seed_plan.grant_id("c-plan", "user", "u-1") == \
            seed_plan.grant_id("c-plan", "user", "u-1")
    assert seed_plan.grant_id("c-plan", "user", "u-1") is None


def test_the_seed_pass_reads_the_plan_file(tmp_path, monkeypatch):
    """`MANTLE_SEED_PLAN_FILE` puts a seed pass on its plan; a plan whose digest does not match its
    body stops the pass before anything is made."""
    import json

    from mantle.services import system_identity as si

    cred = tmp_path / "seed.cred"
    cred.write_text("seed-token")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": seed_plan.PLAN_FORMAT, "salt": SALT, "entries": [
        {"body": {"kind": "grant"}, "digest": "0" * 64}]}))
    monkeypatch.setenv(si.SEED_CREDENTIAL_SETTING, str(cred))
    monkeypatch.setenv(seed_plan.PLAN_FILE_SETTING, str(bad))

    class Custodian:
        def seed_done(self, token):
            pass

    with pytest.raises(ValueError):
        with si.seed_pass(custodian=Custodian()):
            pass
    assert not cred.exists()                           # the credential is spent either way
