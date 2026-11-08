"""Origin's port of the reach rules answers every question exactly as Mantle's own rules do.

Origin decides key release and who may change a grant by its own walk over its replica of the
authority ledger (`origin.services.authority_reach`), a line-for-line port of Mantle's
`LightConeGrantVerifier`, `origin_chain`, `get_origin_root`, `grant_service.user_has_any_action`
and `effective_flags`. Two implementations of one rule are two chances to disagree, so this file
holds them together:

1. A shared fixture is written through Mantle's real store (`lattice_api.create_grant`,
   `add_artifact_to_collection`): containment with attenuating, empty, compact-string and
   unrestricted masks, a non-conducting link, deny nearer an allow, expired, revoked and
   non-person grants, an admin who cannot read.
2. Every change is journaled (`db.reach_journal`) as it is written, and Origin builds its replica
   from those entries alone, through its own `normalise`: the same path production takes.
3. Every question either side can answer is asked of both, for every subject, node and action,
   and the answers must be identical.

`test_the_parity_check_sees_a_divergence` breaks the port on purpose and requires this file to
notice, so a comparison that silently compares nothing cannot pass.
"""
from __future__ import annotations

import itertools

import pytest

from mantle.attenuation import ACTIONS
from mantle.db import lattice_api as la
from mantle.db import reach_journal
from mantle.entities.grant import Grant

origin_reach = pytest.importorskip("origin.services.authority_reach")
origin_ledger = pytest.importorskip("origin.services.authority_ledger")

SUBJECTS = ("alice", "bob", "carol", "erin", "frank", "gina", "hank", "ivan", "jill", "nobody")
NODES = ("W1", "C1", "C2", "C3", "card", "L", "X", "E", "W2", "D1", "alice", "ghost")
PRINCIPALS = ("W1", "W2", "alice", "bob")

#: (src, dst, origin, propagate as stored) — every shape the `propagate` column holds.
EDGES = [
    ("W1", "C1", True, None),                    # unrestricted
    ("C1", "C2", True, ["update"]),              # attenuates read
    ("C1", "C3", True, ["read", "update"]),
    ("C3", "card", True, None),
    ("W1", "L", False, None),                    # a link: conducts nothing upward
    ("C1", "X", True, "r"),                      # the compact substrate form: carries nothing
    ("C1", "E", True, []),                       # explicitly nothing
    ("W2", "D1", True, ["read"]),
]


def _g(gid, grantee, resource, actions, *, effect="allow", state="active", grantee_type="user",
       expires_at=None):
    return Grant(id=gid, resource_id=resource, grantee_type=grantee_type, grantee_id=grantee,
                 granted_by="fixture", effect=effect, state=state, expires_at=expires_at,
                 **{f"can_{a}": a in actions for a in ACTIONS})


GRANTS = [
    _g("g1", "alice", "W1", ACTIONS),
    _g("g2", "bob", "C1", ("read", "share")),
    _g("g3", "bob", "C3", ("read",), effect="deny"),           # deny nearer the allow
    _g("g4", "carol", "W1", ("read",), expires_at="2000-01-01T00:00:00Z"),
    _g("g5", "erin", "card", ("read",)),                       # the artifact alone
    _g("g6", "frank", "W1", ("read", "share"), state="revoked"),
    _g("g7", "gina", "W1", ("admin", "share")),                # manages, cannot read
    _g("g8", "hank", "D1", ("read", "create", "add")),
    _g("g9", "hank", "W2", ("update",)),
    _g("g10", "ivan", "C1", ("read",), grantee_type="invite"),  # not a person's grant
    _g("g11", "ivan", "C2", ("read", "update")),
    _g("g12", "erin", "card", ("update",), effect="deny"),
    _g("g13", "bob", "L", ("read",)),
    _g("g14", "alice", "W2", ("read",), expires_at="2999-01-01T00:00:00Z"),
    _g("g15", "jill", "C1", ("read", "create")),
    _g("g16", "jill", "W1", ("read", "create"), effect="deny"),   # a deny farther up the allow
]


@pytest.fixture
def shared(tmp_path):
    """Mantle's store with the fixture written, and Origin's replica built from its journal."""
    entries = []

    def authority(entry, **_kw):
        entries.append(entry)
        return None

    db = la.LatticeDatabase(str(tmp_path / "parity.db"), origin="parity")
    reach_journal.install(authority)
    try:
        for src, dst, origin, prop in EDGES:
            assert la.add_artifact_to_collection(db, src, dst, origin=origin, propagate=prop)
        for g in GRANTS:
            la.create_grant(db, g)
    finally:
        reach_journal.uninstall()

    replica = origin_reach.MemoryReplica()
    for raw in entries:
        b = origin_ledger.normalise(raw)
        if b["kind"] == "grant":
            replica.grants[b["grant_id"]] = origin_ledger._grant_from_body(b)
        elif b["kind"] == "edge.add":
            replica.edges[(b["src"], b["dst"])] = {"origin": b["origin"],
                                                   "propagate": b["propagate"], "added_by": ""}
    return db, origin_reach.Reach(replica), entries


def _mantle_verifier(db):
    from mantle.search.mantle.oracle import LightConeGrantVerifier

    return LightConeGrantVerifier(db, ttl_s=0)


def mismatches(db, reach) -> list:
    """Every question on which Mantle and Origin's port disagree."""
    from mantle.services import grant_service

    out = []
    verifier = _mantle_verifier(db)
    for node, action in itertools.product(NODES, ACTIONS):
        m = list(la.origin_chain(db, node, action))
        o = list(reach.chain(node, action))
        if m != o:
            out.append(("chain", node, action, m, o))
    for node in NODES:
        if str(la.get_origin_root(db, node)) != reach.root_of(node):
            out.append(("root", node))
    for subject, node, action in itertools.product(SUBJECTS, NODES, ACTIONS):
        if verifier._reaches(subject, "user", node, action) != reach.reaches(subject, node, action):
            out.append(("reaches", subject, node, action))
    # The deny walk (review 17, M2): a deny anywhere up the creation chain, whatever each edge
    # on the way conducts, on both sides.
    for subject, node, action in itertools.product(SUBJECTS, NODES, ACTIONS):
        if verifier._denied(subject, "user", node, action) != reach.denies(subject, node, action):
            out.append(("denies", subject, node, action))
    for subject, action in itertools.product(SUBJECTS, ACTIONS):
        if verifier._granted_roots(subject, "user", action) != reach.granted_roots(subject,
                                                                                   action):
            out.append(("granted_roots", subject, action))
    for subject, principal, coll, art, action in itertools.product(
            SUBJECTS, PRINCIPALS, (None,) + NODES, (None, "card"), ("read", "update", "admin")):
        m = verifier.authorized(requester_id=subject, requester_type="user",
                                principal_id=principal, collection_id=coll, action=action,
                                artifact_id=art)
        o = reach.authorized(subject=subject, principal_id=principal, collection_id=coll,
                             action=action, artifact_id=art)
        if m != o:
            out.append(("authorized", subject, principal, coll, art, action))
    for subject, node in itertools.product(SUBJECTS, NODES):
        for actions in (("share", "admin"), ("admin",), ("read",)):
            if grant_service.user_has_any_action(db, subject, node, *actions) != \
                    reach.has_any_action(subject, node, *actions):
                out.append(("has_any_action", subject, node, actions))
        held = frozenset(a for a, f in ((a, "can_" + a) for a in ACTIONS)
                         if grant_service.effective_flags(db, subject, node)[f])
        if held != reach.effective_actions(subject, node):
            out.append(("effective", subject, node))
    return out


def test_every_answer_is_the_same_on_both_sides(shared):
    db, reach, entries = shared
    assert len(entries) == len(EDGES) - 0 + len(GRANTS)          # every write was journaled
    assert mismatches(db, reach) == []


def test_the_fixture_exercises_both_answers(shared):
    """Guard against a vacuous pass: the fixture must produce both yes and no on the questions
    that matter, so agreement is not agreement on all-False."""
    db, reach, _ = shared
    assert reach.reaches("bob", "C1", "read") and not reach.reaches("bob", "C3", "read")
    assert not reach.reaches("alice", "C2", "read") and reach.reaches("alice", "C2", "update")
    assert reach.reaches("erin", "card", "read") and not reach.reaches("erin", "card", "update")
    assert not reach.reaches("carol", "W1", "read") and not reach.reaches("frank", "W1", "read")
    assert reach.has_any_action("gina", "W1", "share", "admin")
    assert not reach.reaches("gina", "W1", "read")
    assert reach.authorized(subject="hank", principal_id="W2", collection_id=None, action="read")
    assert not reach.reaches("ivan", "C1", "read") and reach.reaches("ivan", "C2", "read")
    assert reach.reaches("jill", "C3", "read") and not reach.reaches("jill", "W1", "read")


def test_the_parity_check_sees_a_divergence(shared, monkeypatch):
    """Break one rule in the port (attenuation ignored) and the comparison must report it."""
    db, reach, _ = shared
    monkeypatch.setattr(origin_reach, "propagates", lambda value, action: True)
    found = mismatches(db, reach)
    assert any(m[0] == "chain" for m in found) and any(m[0] == "reaches" for m in found)


def test_both_sides_hash_an_entry_the_same_way(shared):
    """Mantle's service token names the SHA-256 of the entry it submits; Origin recomputes it from
    the body it receives. One byte of disagreement and every submission is refused."""
    from mantle.clients.origin_key_share_client import entry_digest

    _, _, entries = shared
    entries = entries + [reach_journal.unedge_entry("W1", "C1"), reach_journal.root_entry("W1")]
    for e in entries:
        assert entry_digest(e) == origin_ledger.body_digest(e)


def test_an_entry_carries_no_name_or_note(shared):
    db, _, entries = shared
    named = _g("g-named", "bob", "W1", ("read",))
    named.name, named.notes = "Bob's view", "secret note"
    assert set(reach_journal.grant_entry(named)) == {
        "kind", "grant_id", "grantee_id", "grantee_type", "resource_id", "actions", "effect",
        "state", "expires_at", "target", "max_claims"}
    assert all("name" not in e and "notes" not in e and "label" not in e for e in entries)
