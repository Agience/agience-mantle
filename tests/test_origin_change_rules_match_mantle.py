"""Origin's change rules and release decision agree with Mantle's own, on one shared fixture.

`test_origin_reach_matches_mantle.py` holds the reach walk. This file holds what Origin decides
with it, against the Mantle code that decides the same thing:

    Origin (`authority_ledger.decide`, `Reach.may_release`)   Mantle
    a grant created: rights and clamp                          grants_router: can_share for an
                                                               invite, can_admin for a direct grant;
                                                               grant_service.clamp_grant_to_issuer
    a grant revoked                                            grants_router.revoke_grant
    an invite claim                                            grant_service.claim_invite checks
    an edge added (secondary containment)                      check_access(create | add)
    an edge removed                                            check_access(evict | delete)
    a key share released                                       OracleService._authorize (read)

Every id in the fixture is known to Origin as bootstrapped (no creator), so Origin's own additions
(the creator's base case, ids coming into being) do not enter, and each side answers by its rules
alone. For revocation Origin also lets the grantee give up a grant and an issuer narrow their own;
those only narrow, so the check is one-way: everything Mantle lets through, Origin does.
"""
from __future__ import annotations

import itertools
import types

import pytest
from fastapi import HTTPException

from mantle.attenuation import ACTIONS
from mantle.services.dependencies import AuthContext, check_access

from .test_origin_reach_matches_mantle import EDGES, GRANTS, NODES, SUBJECTS, origin_ledger, \
    origin_reach, shared  # noqa: F401 — `shared` is the fixture

EMAILS = {s: f"{s}@team.test" for s in SUBJECTS}


@pytest.fixture
def world(shared):
    db, reach, entries = shared
    for node in NODES:
        db.artifacts.put_artifact({"id": node, "state": "live"})
        reach.r.nodes[node] = {"created_by": ""}
    for s in SUBJECTS:
        reach.r.persons[s] = (EMAILS[s], True)
    return db, reach


def _auth(s):
    return AuthContext(principal_id=s, principal_type="user", user_id=s)


def _mantle_may(db, s, node, *actions) -> bool:
    for a in actions:
        try:
            check_access(_auth(s), node, a, db)
            return True
        except HTTPException:
            continue
    return False


def _origin_decides(reach, actor, entry, **ctx) -> bool:
    b = origin_ledger.normalise(entry)
    # A grant is made at Origin, in the person's Origin session (review 12, John 2026-11-03): its
    # rules are decided on that credential, which is where Mantle's grant rules now apply.
    if b["kind"] == "grant":
        ctx.setdefault("kind", getattr(origin_ledger, "ORIGIN_PAGE", "access"))
    c = origin_ledger.Context(actor=actor, **ctx)
    try:
        return origin_ledger._DECIDE[b["kind"]](reach, c, b) is not None
    except origin_ledger.EntryRefused:
        return False


def _grant_entry(gid, grantee, resource, actions, gtype="user", **kw):
    return {"kind": "grant", "grant_id": gid, "grantee_id": grantee, "grantee_type": gtype,
            "resource_id": resource, "actions": list(actions), "effect": "allow",
            "state": "active", "expires_at": None, "target": kw.get("target"),
            "max_claims": kw.get("max_claims")}


ACTION_SETS = (("read",), ("read", "update"), ("share",), ("admin",), ACTIONS)


def test_creating_a_grant_needs_the_same_rights_and_the_same_clamp(world):
    from mantle.services import grant_service

    db, reach = world
    mismatches = []
    for actor, node, gtype, acts in itertools.product(SUBJECTS, NODES, ("user", "invite"),
                                                      ACTION_SETS):
        may = (grant_service.can_share if gtype == "invite" else grant_service.can_admin)(
            db, actor, node)
        held = {a for a, f in grant_service.effective_flags(db, actor, node).items() if f}
        mantle = may and {"can_" + a for a in acts} <= held
        # A direct grant names a grantee who exists (Origin co-signs no grant to an id that does
        # not, review 12 n1b); an invite names its token's hash.
        grantee = "x" if gtype == "invite" else next(p for p in SUBJECTS if p != actor)
        origin = _origin_decides(reach, actor, _grant_entry("new", grantee, node, acts, gtype))
        if mantle != origin:
            mismatches.append((actor, node, gtype, acts, mantle, origin))
    assert mismatches == []


def test_whatever_mantle_lets_revoke_origin_does(world):
    """Mantle: the granter revokes their own invite; anything else needs admin."""
    from mantle.services import grant_service

    db, reach = world
    for g in GRANTS:
        if g.state != "active":
            continue
        for actor in SUBJECTS:
            prior = reach.r.grants[g.id]
            reach.r.grants[g.id] = origin_reach.Grant(**{**prior.__dict__, "created_by": "fixture"})
            mantle = ((g.grantee_type == "invite" and g.granted_by == actor)
                      or grant_service.can_admin(db, actor, g.resource_id))
            entry = {**_grant_entry(g.id, g.grantee_id, g.resource_id,
                                    [a for a in ACTIONS if getattr(g, "can_" + a)], g.grantee_type),
                     "effect": "allow" if g.effect == "allow" else "deny",
                     "expires_at": g.expires_at, "state": "revoked"}
            origin = _origin_decides(reach, actor, entry)
            reach.r.grants[g.id] = prior
            assert not (mantle and not origin), (g.id, actor)


@pytest.mark.parametrize("target,max_claims", [(None, None), (("email", "bob@team.test"), None),
                                               (("domain", "team.test"), 1),
                                               (("user_id", "carol"), None)])
def test_an_invite_claim_is_accepted_for_the_same_claimants(world, target, max_claims):
    from mantle.services import grant_service

    db, reach = world
    digest = (origin_ledger.target_digest(*target) if target else None)
    invite = types.SimpleNamespace(target_entity=target[1] if target else None,
                                   target_entity_type=target[0] if target else None)
    reach.r.grants["inv"] = origin_reach.Grant(
        id="inv", grantee_id=__import__("hashlib").sha256(b"tok").hexdigest(),
        grantee_type="invite", resource_id="W1", actions=frozenset({"read"}),
        target_kind=target[0] if target else None, target_digest=digest, max_claims=max_claims)
    for s in SUBJECTS:
        try:
            if invite.target_entity:
                grant_service._verify_target_match(db, s, invite, EMAILS[s])
            mantle = True
        except grant_service.InviteIdentityMismatch:
            mantle = False
        new = origin_ledger._grant_from_body(origin_ledger.normalise(
            _grant_entry("claim", s, "W1", ["read"])))
        try:
            origin = bool(origin_ledger._claim(reach, origin_ledger.Context(
                actor=s, proof={"invite_id": "inv", "token": "tok"}), new))
        except origin_ledger.EntryRefused:
            origin = False
        assert mantle == origin, s
        # And the whole decision lets every claimant Mantle lets claim, whatever else they hold.
        if mantle:
            assert _origin_decides(reach, s, _grant_entry("claim", s, "W1", ["read"]),
                                   proof={"invite_id": "inv", "token": "tok"}), s


def test_adding_and_removing_an_edge_need_the_same_rights(world):
    db, reach = world
    mismatches = []
    for actor, src, dst in itertools.product(SUBJECTS, NODES, ("card", "L")):
        if src == dst:
            continue
        mantle = _mantle_may(db, actor, src, "create", "add")
        origin = _origin_decides(reach, actor, {"kind": "edge.add", "src": src, "dst": dst,
                                                "origin": False, "propagate": []})
        if (src, dst) not in reach.r.edges and mantle != origin:
            mismatches.append(("add", actor, src, dst, mantle, origin))
    for (src, dst), _e in list(reach.r.edges.items()):
        for actor in SUBJECTS:
            from mantle.services import grant_service
            # Evict (or Admin) on the collection removes a secondary edge; deleting the member
            # takes it. A creation edge is a deletion (review 15): Delete on the member, or Admin.
            if _e.get("origin"):
                mantle = (grant_service.can_admin(db, actor, src)
                          or _mantle_may(db, actor, dst, "delete"))
            else:
                mantle = (_mantle_may(db, actor, src, "evict")
                          or grant_service.can_admin(db, actor, src)
                          or _mantle_may(db, actor, dst, "delete"))
            origin = _origin_decides(reach, actor, {"kind": "edge.remove", "src": src,
                                                    "dst": dst})
            if mantle != origin:
                mismatches.append(("remove", actor, src, dst, mantle, origin))
    assert mismatches == []


def test_a_release_matches_the_oracle_s_authorization(world):
    """A collection key opens for exactly whom Mantle's `_authorize` opens that collection's
    content; a master key opens for its own principal alone (John, 2026-10-31), which is narrower
    than Mantle's principal-scoped check by design."""
    from mantle.search.mantle.oracle import (
        GrantDenied, KeyPurpose, KeyRequest, LightConeGrantVerifier, OracleService,
    )
    from mantle.services.acting_principal import acting_as

    db, reach = world
    oracle = OracleService(store=None, grant_verifier=LightConeGrantVerifier(db, ttl_s=0))
    mismatches = []
    for subject, coll in itertools.product(SUBJECTS, NODES):
        principal = reach.root_of(coll)
        reach.r.roots.setdefault(principal, {"root_id": principal, "registered_by": ""})
        with acting_as(subject, principal_type="user"):
            try:
                oracle._authorize(principal, coll, KeyRequest(
                    requester_id=subject, purpose=KeyPurpose.GRANT, action="read"))
                mantle = True
            except GrantDenied:
                mantle = False
        reach.r.nodes.setdefault(coll, {"created_by": ""})
        origin, _why = reach.may_release(subject=subject, principal_id=f"ck:{coll}",
                                         collection_id=coll)
        if mantle != origin:
            mismatches.append((subject, coll, principal, mantle, origin))
        master, _ = reach.may_release(subject=subject, principal_id=principal, collection_id=coll)
        assert master == (subject == principal), (subject, principal)
    assert mismatches == []


def _mantle_reaches(db, s, node) -> set:
    """Mantle's light-cone walk (`LightConeGrantVerifier._reaches`), the one
    `test_origin_reach_matches_mantle.py` holds Origin's port to."""
    from .test_origin_reach_matches_mantle import _mantle_verifier

    verifier = _mantle_verifier(db)
    return {a for a in ACTIONS if verifier._reaches(s, "user", node, a)}


def test_created_by_grants_nothing_on_either_side(world):
    """Architecture §51: all access, the creator's included, is by explicit grant; `created_by`
    is provenance only. With every node recorded as the actor's creation and no creation batch,
    Origin admits a self-grant exactly when Mantle's own grant rules do (`can_admin` and the
    clamp)."""
    from mantle.services import grant_service

    db, reach = world
    mismatches = []
    for actor, node in itertools.product(SUBJECTS, NODES):
        if node in SUBJECTS:
            continue
        reach.r.nodes[node] = {"created_by": actor}
        held = {a for a, f in grant_service.effective_flags(db, actor, node).items() if f}
        for acts in ACTION_SETS:
            mantle = grant_service.can_admin(db, actor, node) and {
                "can_" + a for a in acts} <= held
            origin = _origin_decides(reach, actor, _grant_entry("own", actor, node, acts))
            if mantle != origin:
                mismatches.append((actor, node, acts, mantle, origin))
        reach.r.nodes[node] = {"created_by": ""}
    assert mismatches == [], mismatches


def test_a_creation_edge_needs_the_same_rights_as_mantle_s_create(world):
    """Creating into a collection: Mantle checks `create` (or `add`) reaching the container
    (`check_access`); Origin the same, for an id it has never seen."""
    db, reach = world
    mismatches = []
    for n, (actor, src) in enumerate(itertools.product(SUBJECTS, NODES)):
        if src in SUBJECTS:
            continue
        mantle = _mantle_may(db, actor, src, "create", "add")
        origin = _origin_decides(reach, actor, {"kind": "edge.add", "src": src,
                                                "dst": f"fresh-{n}", "origin": True,
                                                "propagate": None})
        if mantle != origin:
            mismatches.append((actor, src, mantle, origin))
    assert mismatches == []


def test_the_check_sees_a_creator_given_rights_by_created_by(world, monkeypatch):
    """Let `created_by` grant the creator their own grant in Origin and the comparison must
    report it."""
    monkeypatch.setattr(origin_ledger, "_creation_grant",
                        lambda rc, ctx, new: "created" if rc.r.node(new.resource_id) and
                        rc.r.node(new.resource_id).get("created_by") == ctx.actor else None)
    with pytest.raises(AssertionError):
        test_created_by_grants_nothing_on_either_side(world)


def test_the_change_rule_check_sees_a_divergence(world, monkeypatch):
    """Break the clamp in Origin and the comparison must report it."""
    db, reach = world
    monkeypatch.setattr(origin_reach.Reach, "effective_actions",
                        lambda self, s, r: frozenset(ACTIONS))
    with pytest.raises(AssertionError):
        test_creating_a_grant_needs_the_same_rights_and_the_same_clamp(world)


def test_an_invite_whose_grantee_id_is_a_person_s_is_no_grant_on_either_side(world):
    """`ivan` holds an invite on `C1` whose grantee id is his own id. Neither `check_access` nor
    the light-cone verifier counts it as his grant."""
    from .test_origin_reach_matches_mantle import _mantle_verifier

    db, reach = world
    verifier = _mantle_verifier(db)
    for node in ("C1", "C3", "card"):
        assert _mantle_may(db, "ivan", node, "read") == verifier._reaches(
            "ivan", "user", node, "read"), node
    assert not _mantle_may(db, "ivan", "C3", "read")
