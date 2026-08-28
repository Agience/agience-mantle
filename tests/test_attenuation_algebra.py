"""The attenuation algebra, proved by exhaustion over its entire domain.

`mantle.attenuation` claims to be a bounded meet-semilattice over CRUDEASIO: composition
along a path is monotone and non-amplifying, deny is the absorbing zero, full authority is
the identity. Those are not decorations — the light cone, the bundle ceiling and the
`edge.propagate` column all rest on them, so each one is asserted here rather than
described.

**The domain is small enough to enumerate, so it is enumerated.** Nine actions plus the
allow bit is a 10-bit code: 1,024 masks, 1,048,576 ordered pairs. Every pair law below
sweeps all of them — no sampling, no property-based search, no new dependency. Where a law
needs triples (associativity, path composition) 1024³ is 1.07·10⁹ and does not run, so it
is factored instead, and Section 3 says exactly how the factoring is closed.

Four sections:

1. The domain itself — that it is what this file thinks it is, and that the type is a value.
2. Codec fidelity, against a frozen copy of the way the live lattice reads
   ``edge.propagate``. This is the "no migration" claim, measured: an encoding that drifted
   from the column would silently re-authorize every edge on disk.
3. The laws, exhaustively.
4. The two consumers, at the level they are consumed: the bundle ceiling and the light cone.
"""
from __future__ import annotations

import itertools
import json

import pytest

from mantle.attenuation import (
    ACTIONS,
    DENY,
    FLAG_OF,
    NOTHING,
    TOP,
    Mask,
    meet,
    compose,
    propagates,
)
from mantle.entities.grant import Grant, mask_of

#: Every mask there is. 2^9 action subsets x {allow, not-allow}.
ALL = tuple(Mask(code) for code in range(1 << (len(ACTIONS) + 1)))

#: The masks an `edge.propagate` value can decode to — an edge has no effect axis, so its
#: sub-domain is the 512 allow masks.
ALL_PROPAGATE = tuple(m for m in ALL if m.is_allow)


# ═════════════════════════════════════════════════════════════════════════════════════════
# 1 · The domain
# ═════════════════════════════════════════════════════════════════════════════════════════

def test_the_domain_is_exactly_the_size_the_proofs_below_assume() -> None:
    """Every law in this file is a sweep, and a sweep proves nothing about a domain larger
    than the one it walked. Pinning the size here means adding a tenth action makes THIS
    fail loudly rather than making every proof below quietly partial."""
    assert len(ACTIONS) == 9, f"CRUDEASIO grew or shrank: {ACTIONS}"
    assert len(set(ACTIONS)) == 9, "a duplicated action name would collapse two bits into one"
    assert len(ALL) == 1024
    assert len({m.code for m in ALL}) == 1024, "two masks share a code"
    assert len(ALL_PROPAGATE) == 512


def test_masks_are_values_so_equality_and_identity_cannot_disagree() -> None:
    """Interned. An authorization decision that turned on which of two equal masks you were
    holding would be a decision on object identity, which is not a security property."""
    for m in ALL:
        assert Mask(m.code) is m
        assert m == Mask(m.code) and hash(m) == hash(Mask(m.code))
    assert Mask.of(["read", "create"]) is Mask.of(["create", "read"]), "order changed the value"
    with pytest.raises(AttributeError):
        object.__getattribute__(TOP, "__class__").__setattr__(TOP, "_code", 0)
    for bad in (-1, 1024, 99999):
        with pytest.raises(ValueError):
            Mask(bad)


def test_the_named_elements_are_the_ones_the_laws_name() -> None:
    assert DENY.code == 0 and not DENY.is_allow and DENY.actions == frozenset()
    assert TOP.is_allow and TOP.actions == frozenset(ACTIONS)
    assert NOTHING.is_allow and NOTHING.actions == frozenset()
    assert NOTHING is not DENY, (
        "a permitted edge that transmits nothing and a refusal are different statements; "
        "collapsing them would make `propagate=[]` indistinguishable from a deny")
    assert not bool(DENY) and not bool(NOTHING) and bool(TOP)


def test_an_unknown_action_name_is_denied_rather_than_erroring() -> None:
    """An unmapped verb must be a denial, not a hole opened by a typo and not a 500."""
    for name in ("publish", "", "READ", "can_read", "Read "):
        assert TOP.allows(name) is False and TOP.carries(name) is False
    assert Mask.of(["read", "publish"]) is Mask.of(["read"]), "an unknown name became a bit"


# ═════════════════════════════════════════════════════════════════════════════════════════
# 2 · Codec fidelity — the "no migration" claim, measured
# ═════════════════════════════════════════════════════════════════════════════════════════
#
# The oracle below is a verbatim copy of how the lattice reads the column TODAY. It is kept
# frozen and is never wired to `mantle.attenuation`: an oracle that called the thing it
# checks would prove only that a function equals itself. If the column semantics change
# deliberately, this body changes in the same commit, so every moved answer is explained.


def _lattice_prop_mask(v):
    """Verbatim `mantle.db.lattice_api._prop_mask`. Do not refactor; do not import."""
    if isinstance(v, str) and v.startswith("["):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _lattice_allows(column_value, action: str) -> bool:
    """Verbatim membership test from `lattice_api.list_origin_descendants` (and, identically,
    `services.dependencies.check_access`): ``mask is not None and action not in mask`` prunes,
    so anything else propagates."""
    mask = _lattice_prop_mask(column_value)
    return not (mask is not None and action not in mask)


def _column_corpus():
    """Every shape the `propagate` column is known to hold, plus the awkward ones.

    Three writers exist: `lattice_api._ser_propagate` json-dumps a list, `artifacts_router`
    writes `[]`, and the substrate stamps a bare compact string on creation edges
    (`schema.py`: ``propagate="r"``). `_prop_mask` also returns raw containers when the value
    landed in the edge's `props` blob rather than the promoted column."""
    yield None                                     # NULL — unrestricted
    for r in range(len(ACTIONS) + 1):              # every JSON array a writer can produce
        for combo in itertools.combinations(ACTIONS, r):
            yield json.dumps(list(combo))
    yield "[]"
    yield from ("r", "ru", "crudeasio", "c", "")   # the compact/legacy bare-string form
    yield from ("read", "read,update", "null", '"read"')
    yield from ("[bad json", '["read"', "[1,2]", "[null]", '["read", 1]')
    yield from ([], ["read"], ("read", "invoke"), {"read", "add"}, frozenset({"delete"}))
    yield {"read": True, "invoke": True}           # a props-blob dict


def test_the_decoder_reproduces_the_lattice_column_on_every_known_shape() -> None:
    """Bit-for-bit against the frozen oracle, over every column shape x every action.

    This is the whole on-disk-compatibility argument. `edge.propagate` is authorization data
    that already exists in live stores and there is no migration, so a decoder that answered
    differently anywhere would re-authorize — or de-authorize — edges nobody touched."""
    checked, bad = 0, []
    for value in _column_corpus():
        decoded = Mask.from_propagate(value)
        for action in ACTIONS:
            checked += 1
            want = _lattice_allows(value, action)
            got = decoded.allows(action)
            if got != want:
                bad.append((value, action, got, want))
            if propagates(value, action) != want:
                bad.append((value, action, "propagates()", want))
    assert checked >= 4500, f"the sweep collapsed to {checked} checks and proves almost nothing"
    assert not bad, f"{len(bad)} disagreements with the live column semantics; first five: {bad[:5]}"


def test_the_compact_bare_string_form_stays_as_closed_as_the_lattice_has_it() -> None:
    """The documented substrate form (`propagate="r"`) reaches the membership test as a
    SUBSTRING check against full action names, so it matches nothing and the edge propagates
    nothing. That is fail-closed, and it is preserved rather than quietly "corrected" —
    reading `"r"` as Read here would widen every such edge on disk in a commit that claims
    to change no behavior."""
    assert Mask.from_propagate("r") is NOTHING
    assert Mask.from_propagate("crudeasio") is NOTHING
    assert _lattice_allows("r", "read") is False, "the oracle disagrees; re-derive this claim"
    # ...while a string that literally contains an action name still matches, as it does today.
    assert Mask.from_propagate("read").allows("read") is True


def test_every_propagate_mask_round_trips_through_the_column_encoding() -> None:
    """All 512 of them. NULL stays the way unrestricted is spelled, and the array form is
    byte-identical to what `lattice_api._ser_propagate` already writes."""
    assert TOP.to_propagate() is None, "unrestricted must still be spelled NULL"
    for m in ALL_PROPAGATE:
        encoded = m.to_propagate()
        assert Mask.from_propagate(encoded) is m, f"{m!r} did not survive the column"
        if encoded is not None:
            assert encoded == json.dumps(sorted(m.actions, key=ACTIONS.index))
            assert json.loads(encoded) == [a for a in ACTIONS if a in m.actions]
    assert NOTHING.to_propagate() == "[]", "the non-lineage link edge changed shape"
    assert Mask.of(["read", "invoke"]).to_propagate() == json.dumps(["read", "invoke"])


def test_a_deny_mask_has_no_column_representation_and_says_so() -> None:
    """There is no such thing as a deny edge. Inventing a spelling for one would put an
    effect into a column nothing reads an effect out of."""
    for m in ALL:
        if m.is_allow:
            continue
        with pytest.raises(ValueError):
            m.to_propagate()


def test_every_mask_round_trips_through_the_grant_flag_encoding() -> None:
    """All 1,024, including the deny half. The flag codec is effect-blind in both
    directions — a deny grant's bits say which actions it denies — so the allow half has to
    be carried alongside, and this pins that it is."""
    for m in ALL:
        flags = m.to_flags()
        assert set(flags) == set(FLAG_OF.values()), (
            "a partial flag map would inherit the Grant constructor's can_read=True default "
            "and silently widen")
        assert Mask.from_flags(flags, allow=m.is_allow) is m
        assert Mask.from_flags(Grant("r", "user", "u", "o", **flags),
                               allow=m.is_allow) is m, "the object form disagreed with the mapping form"


# ═════════════════════════════════════════════════════════════════════════════════════════
# 3 · The laws
# ═════════════════════════════════════════════════════════════════════════════════════════

def test_every_unary_law_on_every_mask() -> None:
    """Idempotence, the identity, the absorbing zero, and the bounds — all 1,024 masks."""
    for a in ALL:
        assert a & a is a, f"{a!r} is not idempotent"
        assert TOP & a is a and a & TOP is a, f"TOP is not neutral for {a!r}"
        assert DENY & a is DENY and a & DENY is DENY, f"DENY is not absorbing for {a!r}"
        assert a <= TOP and DENY <= a, f"{a!r} escapes the bounds"
        assert meet(a, a) is a and a.intersect(a) is a
        # `actions` is the code, read out — which is what lets the pair sweep prove the
        # subset law once, in bit terms, and have it hold in action-name terms as well.
        assert a.actions == frozenset(
            name for i, name in enumerate(ACTIONS) if a.code & (1 << i))
        assert all(a.carries(name) for name in a.actions)
        assert all(a.allows(name) == a.is_allow for name in a.actions)


def test_every_binary_law_on_every_one_of_the_1048576_pairs() -> None:
    """The full pair sweep — commutativity, closure, the canonical encoding, and the
    security-critical one.

    ``a & b <= a`` **and** ``a & b <= b`` is the property everything else in this repo leans
    on when it says a credential "narrows and never widens": there is no pair anywhere in
    the domain whose composition reaches outside either operand. Asserting it on examples
    would leave 1,048,575 unexamined; asserting it here leaves none.

    Deny-absorption is asserted in its general form too, not only against the canonical
    zero: if either operand is not an allow, the result is not an allow. That is the law the
    light-cone path used to break."""
    n, widened, other = 0, [], []
    for a in ALL:
        for b in ALL:
            n += 1
            m = a & b
            if not (m <= a and m <= b):
                widened.append((a, b, m))
            if m is not (b & a):
                other.append(("commutativity", a, b))
            if m.code != a.code & b.code:
                other.append(("canonical encoding", a, b))
            if (a.is_allow and b.is_allow) != m.is_allow:
                other.append(("deny absorption", a, b))
    assert n == 1024 * 1024, f"the sweep walked {n} pairs, not the whole domain"
    assert not widened, (
        f"{len(widened)} pairs composed to something OUTSIDE an operand — attenuation "
        f"widened. First five: {widened[:5]}")
    assert not other, f"{len(other)} law violations; first five: {other[:5]}"


def test_the_meet_is_associative_on_every_triple_of_every_factor() -> None:
    """Associativity, closed by factoring rather than by sampling.

    A 1024³ loop is 1.07·10⁹ iterations and will not run. It does not have to: the previous
    test pins, over **every** pair in the domain, that ``(a & b).code == a.code & b.code`` —
    i.e. that the meet acts independently in each of the 10 bit positions. A componentwise
    operator on a product is associative exactly when it is associative in each component,
    and each component here is the two-element lattice {0, 1}. So the remaining obligation is
    10 factors x 2³ triples = 80 triples, and all 80 are enumerated below with nothing left
    over. Together the two tests are a complete proof, with no sampled step."""
    factors = [1 << i for i in range(len(ACTIONS) + 1)]
    assert len(factors) == 10 and sum(factors) == max(m.code for m in ALL), (
        "the factorisation does not cover the code space, so the argument above has a gap")
    for bit in factors:
        for x, y, z in itertools.product((0, bit), repeat=3):
            assert (x & y) & z == x & (y & z)


def test_the_meet_is_associative_on_every_triple_of_a_complete_subalgebra() -> None:
    """The same law again, observed end-to-end on real `Mask` objects rather than on ints.

    The carrier is the sub-algebra spanned by the allow bit and the first six actions: 128
    elements, closed under the meet, containing DENY and its own top. All 2,097,152 triples
    of it are walked. This does not replace the factored argument above — it is the control
    that the factored argument is about the object actually shipped."""
    sub = tuple(m for m in ALL if not (m.code & ~0b1000111111))
    assert len(sub) == 128 and DENY in sub
    assert all((a & b) in sub for a in sub for b in sub), "the carrier is not closed"
    n = 0
    for a in sub:
        for b in sub:
            ab = a & b
            for c in sub:
                n += 1
                if ab & c is not a & (b & c):
                    pytest.fail(f"associativity failed at {a!r}, {b!r}, {c!r}")
    assert n == 128 ** 3


def test_a_composed_path_is_below_every_edge_on_it() -> None:
    """The light-cone statement: however long the chain, the authority that survives it is
    below every mask it passed through.

    Exhaustive twice over. Every ordered path of length 0..4 drawn from a complete 16-element
    sub-algebra (69,905 paths) is walked directly; and the inductive step for arbitrary length
    over the FULL domain is exactly ``acc & next <= acc`` and ``<= next``, which the pair
    sweep asserts on all 1,048,576 pairs. Length is therefore not a loophole.

    The empty path is TOP: a zero-hop walk must not narrow anything, which is also what makes
    ``compose(p + q) == compose(p) & compose(q)`` hold at the edges."""
    assert compose([]) is TOP
    sub = tuple(m for m in ALL if not (m.code & ~0b1000000111))
    assert len(sub) == 16
    n = 0
    for length in range(5):
        for path in itertools.product(sub, repeat=length):
            n += 1
            folded = compose(path)
            for edge in path:
                assert folded <= edge, f"{folded!r} escaped {edge!r} on path {path}"
            # split-anywhere: composition of a path is the meet of its pieces
            for cut in range(length + 1):
                assert compose(path[:cut]) & compose(path[cut:]) is folded
    assert n == sum(16 ** k for k in range(5)) == 69905


def test_a_deny_anywhere_on_a_path_survives_to_the_end() -> None:
    """Absorption, stated the way the bug it forecloses would have been reported: a deny
    edge at ANY position, at any depth, and the path still ends in a deny. Every position of
    every path up to length 4 over a complete 16-element carrier — 69,904 paths, and the
    pair sweep closes arbitrary length by induction."""
    sub = tuple(m for m in ALL if not (m.code & ~0b1000000111))
    for length in range(1, 5):
        for path in itertools.product(sub, repeat=length):
            if any(not m.is_allow for m in path):
                assert not compose(path).is_allow, f"a deny vanished along {path}"
            else:
                assert compose(path).is_allow


def test_the_order_is_the_one_the_meet_is_the_greatest_lower_bound_of() -> None:
    """`a <= b` iff `a & b == a` — over every pair. Without this the subset assertions above
    would be about some other relation than the one "never widens" is stated in.

    The greatest-lower-bound property itself is checked over a complete 64-element
    sub-algebra: for every pair, no element of the carrier is both below the two and strictly
    above their meet."""
    for a in ALL:
        for b in ALL:
            assert (a <= b) == (a & b is a)
    sub = tuple(m for m in ALL if not (m.code & ~0b1000011111))
    assert len(sub) == 64
    for a in sub:
        for b in sub:
            m = a & b
            for c in sub:
                if c <= a and c <= b:
                    assert c <= m, f"{c!r} is a lower bound of {a!r},{b!r} below their meet"


def test_the_meet_refuses_operands_from_outside_the_algebra() -> None:
    """A mask met with a bool or an int must not silently answer. Python would happily hand
    `Mask & 7` to `int.__and__` if this returned NotImplemented in only one direction."""
    for other in (7, True, "read", None, ["read"], frozenset({"read"})):
        with pytest.raises(TypeError):
            TOP & other
        with pytest.raises(TypeError):
            other & TOP


# ═════════════════════════════════════════════════════════════════════════════════════════
# 4 · The consumers, at the level they are consumed
# ═════════════════════════════════════════════════════════════════════════════════════════

def _grant(**kw):
    base = dict(resource_id="res-1", grantee_type="user", grantee_id="u", granted_by="o")
    base.update(kw)
    return Grant(**base)


def test_the_grant_adapter_answers_the_whole_authorization_question() -> None:
    """`mask_of(g).allows(action)` must be the bit AND the effect. The bare
    `getattr(g, flag, False)` it replaces answered only the bit, which is audit finding S1."""
    allow = _grant(can_read=True)
    deny = _grant(effect="deny", can_read=True)
    weird = _grant(effect="permit", can_read=True)

    assert mask_of(allow).allows("read") is True
    assert mask_of(deny).allows("read") is False, "a deny grant authorized"
    assert mask_of(weird).allows("read") is False, "an unrecognized effect authorized"
    # ...but the bits survive, because a deny grant's bits say WHICH actions it denies.
    assert mask_of(deny).carries("read") is True
    assert mask_of(deny).to_flags()["can_read"] is True


def _assert_ceiling_is_the_meet(member_masks, ceiling_masks) -> int:
    checked = 0
    effects = ("allow", "deny")
    for m_bits in member_masks:
        member_flags = m_bits.to_flags()
        for c_bits in ceiling_masks:
            ceiling_flags = c_bits.to_flags()
            for m_eff, c_eff in itertools.product(effects, effects):
                checked += 1
                member = _grant(effect=m_eff, **member_flags)
                ceiling = _grant(resource_id="", effect=c_eff, **ceiling_flags)
                want = mask_of(member) & mask_of(ceiling)
                assert member.masked_by(ceiling).to_mask() is want, (
                    f"masked_by drifted from the meet at {m_eff}/{c_eff} "
                    f"{m_bits!r} under {c_bits!r}")
                assert want <= mask_of(member) and want <= mask_of(ceiling)
    return checked


def test_the_bundle_ceiling_is_the_meet_for_every_bit_pattern() -> None:
    """`Grant.masked_by` against the operator — the ceiling IS the meet, not something that
    agrees with it on the cases someone thought to write down.

    Two exhaustive carriers rather than one, because building a `Grant` pair per point costs
    ~70 µs and 512x512x4 would put 75 s in the suite for a law the mask sweep already proves
    over the whole domain. What is left to show is that `masked_by` DELEGATES to that law,
    which is a per-bit-independent claim:

    * every member x ceiling pair from the complete 7-action sub-carrier (128x128), x all
      four effect pairings — 65,536 points;
    * every one of the 512 full-width member patterns against the generators of the whole
      lattice (TOP, the empty allow, and each single-action mask), x all four effect
      pairings — so all nine bits are exercised at full width.
    """
    seven = tuple(m for m in ALL_PROPAGATE if not (m.code & ~0b1001111111))
    assert len(seven) == 128
    generators = (TOP, NOTHING) + tuple(Mask.of([a]) for a in ACTIONS)
    assert len(generators) == 11
    n = _assert_ceiling_is_the_meet(seven, seven)
    n += _assert_ceiling_is_the_meet(ALL_PROPAGATE, generators)
    assert n == 128 * 128 * 4 + 512 * 11 * 4


def test_the_bundle_ceiling_never_widens_and_never_mutates_its_member() -> None:
    member = _grant(can_read=True, can_update=True, can_delete=True)
    read_only = _grant(resource_id="", can_read=True)
    masked = member.masked_by(read_only)
    assert masked.can_read is True and masked.can_update is False and masked.can_delete is False
    assert member.can_update is True, "masking wrote its narrowing back into the member"
    open_bundle = _grant(resource_id="", **TOP.to_flags())
    assert _grant(can_read=True).masked_by(open_bundle).to_mask() is Mask.of(["read"])


def test_a_deny_ceiling_makes_its_member_a_deny_however_the_member_was_spelled() -> None:
    """Deny is absorbing at the entity level too, and the effect STRING is only ever
    escalated toward deny — a member already carrying an unrecognized effect keeps it,
    because rewriting it would lose information without changing any decision."""
    deny_bundle = _grant(resource_id="", effect="deny", can_read=True)
    assert _grant(can_read=True).masked_by(deny_bundle).is_deny() is True
    assert _grant(effect="permit", can_read=True).masked_by(deny_bundle).effect == "permit"
    assert mask_of(_grant(effect="permit", can_read=True).masked_by(deny_bundle)).is_allow is False
    # An unrecognized CEILING effect narrows to deny: `grant_is_allow` is positive matching,
    # so a ceiling nobody can read as an allow must not pass authority through.
    weird_bundle = _grant(resource_id="", effect="permit", **TOP.to_flags())
    assert _grant(can_read=True).masked_by(weird_bundle).is_deny() is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
