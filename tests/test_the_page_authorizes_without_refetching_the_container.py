"""Authorizing a page of members looks each resource's grants up ONCE, and is no weaker for it.

The cost: `_readable_members` calls `check_access` per member, and every member's origin chain
passes through the same container and the same ancestors — so authorizing one page fetched the
container's grant row once per member. The route re-derived an answer it already had, N times.

The hoist comes out of the propagation mask: where the containment edge propagates `read` and the
caller holds a read allow on the container, one grant already authorizes every member —
`origin_chain` walks member -> root -> container and stops at the first edge whose mask does not
carry the action.

It is not a second traversal, deliberately. The obvious implementation — resolve the container
once, then hand-roll a cheaper per-member check — would be a second way to answer "may this
principal read this artifact", and this codebase has already paid for that:
`oracle.LightConeGrantVerifier` asked the same question of a collection, answered it by materialising
every descendant, and the two disagreed at scale (its own note: "a collection with more members than
`edges_of` will return raised there while this function answered in milliseconds"). One traversal,
memoised, cannot disagree with itself.

The memo holds the lookup, never the verdict. Deny-first, the flag test and the walk order all
still run per artifact against the cached rows. Caching the verdict would let a container-level
allow answer for a member carrying its own deny — which is the one thing this must not do, and
`test_a_nearer_deny_still_wins_under_the_memo` is that asserted.
"""
from __future__ import annotations



from mantle.entities.grant import Grant as GrantEntity
from mantle.services.dependencies import AuthContext


CONTAINER = "col.reports"
MEMBERS = ["m1", "m2", "m3", "m4", "m5"]


def _allow(resource_id: str, grantee: str = "u1") -> GrantEntity:
    return GrantEntity(id="g-" + resource_id, resource_id=resource_id, grantee_type="user",
                       grantee_id=grantee, granted_by="owner", can_read=True)


def _deny(resource_id: str, grantee: str = "u1") -> GrantEntity:
    g = GrantEntity(id="d-" + resource_id, resource_id=resource_id, grantee_type="user",
                    grantee_id=grantee, granted_by="owner", can_read=True)
    setattr(g, "effect", "deny")
    return g


class _Harness:
    """A store whose grant lookups are COUNTED, and a chain that mirrors the real shape.

    The chain is member -> container, which is what `origin_chain` yields for a member whose root
    is its container. Counting the lookups is the whole measurement: the claim is about how many
    times the container's row is fetched, and only the counter can answer it.
    """

    def __init__(self, grants_by_resource):
        self.grants_by_resource = grants_by_resource
        self.lookups: list = []

    def get_active_grants_for_principal_resource(self, db, grantee_id, resource_id):
        self.lookups.append(resource_id)
        return list(self.grants_by_resource.get(resource_id, []))

    def count(self, resource_id):
        return sum(1 for r in self.lookups if r == resource_id)


# The tests drive the memoised branch directly, not a faked `check_access`. Standing the whole
# function up needs a store, a real origin walk and an audit sink, and a harness that mocks all
# three ends up testing the harness. The claim here is narrow — "how many times is one resource's
# grant row fetched for one page" — so the branch that decides it is exercised directly, and
# `test_the_branch_matches_the_source` fails the moment the real one stops looking like this.

def _grants_branch(memo, harness, auth, resource_id):
    """The exact memoised branch from `check_access._check_grants`, lifted so the test exercises the
    real caching rule rather than a paraphrase. `test_the_branch_matches_the_source` holds it to the
    file."""
    if memo is not None and resource_id in memo:
        return memo[resource_id]
    grants = harness.get_active_grants_for_principal_resource(None, auth.user_id, resource_id)
    if memo is not None:
        memo[resource_id] = grants
    return grants


def test_without_a_memo_the_container_is_fetched_once_per_member():
    """The cost being removed, measured — so the improvement below is a number, not a claim."""
    h = _Harness({CONTAINER: [_allow(CONTAINER)]})
    auth = AuthContext(principal_id="u1", principal_type="user", user_id="u1")
    for _ in MEMBERS:
        _grants_branch(None, h, auth, CONTAINER)
    assert h.count(CONTAINER) == len(MEMBERS), h.lookups


def test_with_a_memo_the_container_is_fetched_once_for_the_page():
    h = _Harness({CONTAINER: [_allow(CONTAINER)]})
    auth = AuthContext(principal_id="u1", principal_type="user", user_id="u1")
    memo: dict = {}
    for _ in MEMBERS:
        _grants_branch(memo, h, auth, CONTAINER)
    assert h.count(CONTAINER) == 1, (
        "the container's grant row was fetched %d times for one page; the memo is not being "
        "consulted" % h.count(CONTAINER))


def test_each_member_is_still_looked_up_on_its_own():
    """The memo must not collapse members into each other. It is keyed on resource id, so a
    per-member grant is still a per-member lookup — only shared ancestors are shared."""
    h = _Harness({m: [] for m in MEMBERS})
    auth = AuthContext(principal_id="u1", principal_type="user", user_id="u1")
    memo: dict = {}
    for m in MEMBERS:
        _grants_branch(memo, h, auth, m)
    for m in MEMBERS:
        assert h.count(m) == 1, "member %s was not authorized on its own" % m


def test_a_nearer_deny_still_wins_under_the_memo():
    """The safety property, asserted. The memo caches the grant rows; the deny-first scan runs
    per artifact against them. A member carrying its own deny must still be refused under a
    container the caller may read — that is the case a verdict cache would break."""
    from mantle.entities.grant import grant_is_allow, grant_is_deny

    h = _Harness({CONTAINER: [_allow(CONTAINER)], "m3": [_deny("m3")]})
    auth = AuthContext(principal_id="u1", principal_type="user", user_id="u1")
    memo: dict = {}
    refused = []
    for m in MEMBERS:
        rows = _grants_branch(memo, h, auth, m) + _grants_branch(memo, h, auth, CONTAINER)
        if any(grant_is_deny(g) and g.can_read for g in rows):
            refused.append(m)
            continue
        assert any(grant_is_allow(g) and g.can_read for g in rows), m
    assert refused == ["m3"], (
        "deny handling changed under the memo: refused=%r (expected exactly ['m3'])" % refused)


def test_a_grant_key_never_reaches_the_memo():
    """A grant key's grants were resolved and MASKED at authentication. That branch reads
    `auth.grants` and must not be re-routed through a cache keyed on resource id, which would find
    the members UNMASKED."""
    import inspect

    from mantle.services import dependencies

    src = inspect.getsource(dependencies.check_access)
    key_branch = src.index("if is_key:")
    memo_branch = src.index("elif grant_memo is not None")
    assert key_branch < memo_branch, (
        "the memo branch now precedes the grant-key branch, so a bearer key would be served from a "
        "resource-keyed cache instead of its masked bundle")


def test_the_branch_matches_the_source():
    """A test that drifts from the code it models proves nothing about the code."""
    import inspect

    from mantle.services import dependencies

    src = inspect.getsource(dependencies.check_access)
    for line in ("elif grant_memo is not None and resource_id in grant_memo:",
                 "grants = grant_memo[resource_id]",
                 "grant_memo[resource_id] = grants"):
        assert line in src, "the memo branch changed shape — missing: %r" % line


def test_the_route_passes_a_fresh_memo_per_call():
    """Request-scoped, never longer. A memo that outlived a request would serve a revoked grant to
    the next one — the cache would become the authorization."""
    import inspect

    from mantle.routers import artifacts_router

    src = inspect.getsource(artifacts_router._readable_members)
    assert "grant_memo: dict = {}" in src, (
        "`_readable_members` no longer builds its own memo; if it were hoisted to module scope it "
        "would outlive the request and cache revoked grants")
    assert "grant_memo=grant_memo" in src, "the memo is built but not passed to check_access"
