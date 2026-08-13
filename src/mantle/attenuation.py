"""Attenuation — the one authorization meet, and the one type both encodings are views of.

> **Composition along a path is monotone and non-amplifying. Authority can only be lost,
> never gained.**

That sentence is an algebra, not a slogan: a bounded **meet-semilattice** over the
CRUDEASIO action set, with a deny/zero element that is absorbing and a full-authority
identity. Capability security calls it *attenuation* (SPKI/SDSI chains intersect
authorization tags; macaroon caveats only narrow); this codebase's own metaphor calls it
the **light cone** — causal composition is transitive but never amplifying.

Before this module the algebra existed twice, in two encodings that disagreed about the
zero element:

===========================  =================================================
`edge.propagate` (`TEXT`)    a mask column, intersected during the origin BFS
`Grant`'s nine `can_*` bools intersected by :meth:`Grant.masked_by`
===========================  =================================================

`masked_by` knew deny was absorbing; the light-cone path did not, and would read a
`deny`-effect grant as authorizing (audit finding S1). One operator makes that
unrepresentable rather than repaired. **Neither storage format changes** — both are
codecs onto :class:`Mask`, and both round-trip.

Scope boundary
--------------
Security **Invariant #1** ("geometry never authorizes") is the same *principle* but not
this *operator*: it is an ordering and import-boundary discipline (the routing path
receives no key material and runs strictly before any key request), not a mask
intersection. It stays enforced where it already is.

Fan-out is **not** attenuation either. Event *visibility* narrows and may reuse this
operator; event *delivery* amplifies (one write notifies many subscribers). Reaching for
`&` on a delivery path would turn fan-out into a way to widen authority.

Dependency floor
----------------
stdlib only, and no import from anywhere else in ``mantle``. Both consumers sit below the
service layer — :mod:`mantle.entities.grant` is an entity and :mod:`mantle.db.access`
is inside the embeddable surface — so anything this module imported, they would import too.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

__all__ = [
    "ACTIONS",
    "FLAG_OF",
    "Mask",
    "DENY",
    "TOP",
    "NOTHING",
    "attenuate",
    "compose",
    "propagates",
]


#: The CRUDEASIO action vocabulary, in the order the ledger's flag tuple has always had
#: (C-R-U-D-E, then I-A-S-O as the boolean columns were declared). The order is part of the
#: contract — :data:`Grant.PERMISSION_FLAGS` is built from it and routers iterate it — so it
#: is a tuple, not a set, and it changes only deliberately.
ACTIONS: Tuple[str, ...] = (
    "create",
    "read",
    "update",
    "delete",
    "evict",
    "invoke",
    "add",
    "share",
    "admin",
)

#: action name -> the `Grant` boolean attribute that carries it. The naming rule is total
#: and mechanical (`can_<action>`), which is why the map can be derived rather than typed
#: out a second time: a hand-written map that disagreed about which flag gates which action
#: would be a silent authorization bug.
FLAG_OF: Dict[str, str] = {action: "can_" + action for action in ACTIONS}

_INDEX: Dict[str, int] = {action: i for i, action in enumerate(ACTIONS)}

#: Bit layout. Actions occupy bits 0..N-1; one bit above them carries the ALLOW half of
#: `Grant.effect`. Encoding the effect as a bit — rather than as a separate field ORed the
#: other way round — is what makes the meet plain integer AND: allow∧allow is the only way
#: to stay an allow, so deny is absorbing *by construction* rather than by a branch someone
#: has to remember to write.
_ACTION_BITS: int = (1 << len(ACTIONS)) - 1
_ALLOW_BIT: int = 1 << len(ACTIONS)
_CODE_SPACE: int = _ALLOW_BIT << 1

#: Every distinct mask is created once and shared. Masks are values (511-plus of them in
#: total), so identity and equality coincide and the operator can be a dict lookup rather
#: than an allocation — which is what makes the exhaustive proofs cheap enough to run.
_INTERN: Dict[int, "Mask"] = {}


class Mask:
    """A CRUDEASIO authority: which actions, and whether it is an allow at all.

    Immutable, interned and totally ordered by nothing — the order is the *subset*
    order (:meth:`__le__`), which is partial, and the meet of that order is :meth:`__and__`.

    Two codecs, neither of which changes a storage format:

    * :meth:`from_propagate` / :meth:`to_propagate` — the ``edge.propagate`` TEXT column.
    * :meth:`from_flags` / :meth:`to_flags` — ``Grant``'s nine ``can_*`` booleans.
    """

    __slots__ = ("_code",)

    # -- construction -------------------------------------------------------------

    def __new__(cls, code: int) -> "Mask":
        code = int(code)
        if not 0 <= code < _CODE_SPACE:
            raise ValueError(
                f"mask code {code} is outside the {len(ACTIONS)}-action domain "
                f"[0, {_CODE_SPACE})"
            )
        got = _INTERN.get(code)
        if got is None:
            got = object.__new__(cls)
            object.__setattr__(got, "_code", code)
            _INTERN[code] = got
        return got

    def __setattr__(self, name: str, value: Any) -> None:      # pragma: no cover - guard
        raise AttributeError("Mask is immutable; compose a new one with `&`")

    def __delattr__(self, name: str) -> None:                  # pragma: no cover - guard
        raise AttributeError("Mask is immutable")

    @classmethod
    def of(cls, actions: Iterable[str] = (), *, allow: bool = True) -> "Mask":
        """A mask carrying exactly *actions*.

        Unknown action names are dropped rather than rejected: the callers are decoders
        for stored data, and a mask that named a verb this build does not know must confer
        nothing for it, not fail the read. Dropping is the fail-closed direction.
        """
        code = _ALLOW_BIT if allow else 0
        for name in actions:
            bit = _INDEX.get(str(name))
            if bit is not None:
                code |= 1 << bit
        return cls(code)

    # -- codec: `Grant`'s boolean flags -------------------------------------------

    @classmethod
    def from_flags(cls, source: Any, *, allow: bool = True) -> "Mask":
        """Read the nine ``can_*`` booleans off *source* (an object or a mapping).

        Duck-typed for the same reason :func:`entities.grant.grant_is_allow` is: grant-like
        objects reach the enforcement path from several producers (entities, AQL row shims,
        test doubles), and authorization must not depend on which one built the object.

        *allow* is a parameter rather than something read from ``source.effect`` so this
        module never has to hold a second copy of the effect normalization — the caller
        that owns the effect vocabulary supplies the verdict. See
        :func:`entities.grant.mask_of`.
        """
        get = source.get if isinstance(source, Mapping) else (
            lambda key, default=False: getattr(source, key, default))
        code = _ALLOW_BIT if allow else 0
        for i, action in enumerate(ACTIONS):
            if get(FLAG_OF[action], False):
                code |= 1 << i
        return cls(code)

    def to_flags(self) -> Dict[str, bool]:
        """The nine ``can_*`` booleans, all of them present.

        Every flag is emitted, including the False ones: a partial map fed back into a
        ``Grant`` would inherit the constructor's ``can_read=True`` default and silently
        widen — the same hazard :func:`grant_key_service._flags_from` documents.

        Effect-blind on purpose. A deny grant's bits say *which actions it denies*
        (`check_access` tests ``grant_is_deny(g) and getattr(g, flag)``), so folding the
        allow bit in here would turn every deny into a deny of nothing.
        """
        return {FLAG_OF[a]: bool(self._code & (1 << i)) for i, a in enumerate(ACTIONS)}

    # -- codec: the `edge.propagate` column ----------------------------------------

    @classmethod
    def from_propagate(cls, value: Any) -> "Mask":
        """Decode an ``edge.propagate`` value exactly as the lattice has always read it.

        The column is TEXT and holds three shapes, all of which stay legal — this is a
        decoder for data already on disk, not a new format:

        ``NULL`` / ``None``
            unrestricted. ``list_origin_descendants`` prunes only when
            ``mask is not None and action not in mask``, so a null column propagates every
            action → :data:`TOP`.
        a JSON array of full action names, e.g. ``'["read", "invoke"]'``
            what every writer in the tree produces (`lattice_api._ser_propagate` json-dumps
            the list it is handed). ``'[]'`` is the legal "nothing propagates" that
            `artifacts_router` writes on a non-lineage link.
        a bare string, e.g. the substrate's compact ``"r"`` on a creation edge
            `_prop_mask` passes it through untouched and the membership test degrades to a
            **substring** test (``action in mask``). Reproduced here literally: the mask
            carries an action iff the action's full name occurs in the string. For every
            compact letter form that means it carries nothing, which is what the lattice
            already does with it — fail-closed, and preserved rather than quietly
            "corrected", because correcting it here would silently widen live edges.

        A list/tuple/set/dict is accepted too, since `_prop_mask` returns one whenever the
        value landed in the edge's ``props`` blob rather than the promoted column.
        """
        if isinstance(value, Mask):
            return value
        if value is None:
            return TOP
        if isinstance(value, Mapping):
            return cls.of(value.keys())
        if isinstance(value, (list, tuple, set, frozenset)):
            return cls.of(str(v) for v in value)
        text = value if isinstance(value, str) else str(value)
        if text.startswith("["):
            try:
                decoded = json.loads(text)
            except ValueError:
                decoded = None                     # `_prop_mask` hands the raw string back
            if isinstance(decoded, list):
                return cls.of(str(v) for v in decoded)
        # The bare-string form: membership was, and stays, substring containment.
        return cls.of(a for a in ACTIONS if a in text)

    def to_propagate(self) -> Optional[str]:
        """Encode back to the column: ``None`` for :data:`TOP`, else a JSON array of names.

        `None` rather than the nine-element array because `None` is what the column means by
        unrestricted, and it is what every writer that does not narrow already stores. The
        array form is byte-identical to `lattice_api._ser_propagate(sorted_action_list)`.

        An edge has no effect axis — there is no such thing as a deny edge — so encoding a
        non-allow mask is a programming error rather than a value this can invent a
        representation for.
        """
        if not self.is_allow:
            raise ValueError(
                "a deny mask has no `edge.propagate` representation: edges carry "
                "propagation, not effect"
            )
        if self is TOP:
            return None
        return json.dumps([a for a in ACTIONS if self._code & (1 << _INDEX[a])])

    # -- the meet -------------------------------------------------------------------

    def __and__(self, other: "Mask") -> "Mask":
        """The meet: `a & b` is the strongest authority both `a` and `b` allow.

        Integer AND over the canonical code, so every law comes out of the bit layout
        rather than a branch: idempotent, commutative, associative, :data:`TOP` neutral,
        :data:`DENY` absorbing, and `a & b` below both operands. Composition can only
        remove.
        """
        if not isinstance(other, Mask):
            return NotImplemented
        return Mask(self._code & other._code)

    __rand__ = __and__

    def intersect(self, other: "Mask") -> "Mask":
        """Spelled-out alias for :meth:`__and__`, for call sites that read better as prose."""
        return self & other

    # -- the order ------------------------------------------------------------------

    def __le__(self, other: "Mask") -> bool:
        """`self ⊆ other` — every action self carries, other carries, and self is an allow
        only if other is. This is the ordering the meet is the greatest lower bound of, and
        the one "attenuation never widens" is stated in."""
        if not isinstance(other, Mask):
            return NotImplemented
        return self._code & other._code == self._code

    def __lt__(self, other: "Mask") -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        return self != other and self <= other

    def __ge__(self, other: "Mask") -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        return other <= self

    def __gt__(self, other: "Mask") -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        return other < self

    def issubset(self, other: "Mask") -> bool:
        """Prose alias for ``self <= other``."""
        return self <= other

    # -- reading it -----------------------------------------------------------------

    @property
    def code(self) -> int:
        """The canonical integer encoding — action bits, plus the allow bit above them."""
        return self._code

    @property
    def is_allow(self) -> bool:
        """True only for an explicit allow. A mask that is not an allow authorizes nothing,
        whatever bits it carries."""
        return bool(self._code & _ALLOW_BIT)

    @property
    def is_deny(self) -> bool:
        """The complement of :attr:`is_allow` *within this type*.

        Narrower than it looks, and deliberately not the same question as
        :func:`entities.grant.grant_is_deny`: an unrecognized ``effect`` string is neither
        allow nor deny to the grant predicates, but it lands here as not-an-allow, because
        a mask has only the two states and the fail-closed one is the right home for
        "unrecognized"."""
        return not self.is_allow

    @property
    def actions(self) -> frozenset:
        """The action names this mask carries, effect-blind."""
        return frozenset(a for i, a in enumerate(ACTIONS) if self._code & (1 << i))

    def carries(self, action: str) -> bool:
        """Is *action*'s bit set? Effect-blind — see :meth:`to_flags` for why that matters."""
        bit = _INDEX.get(action)
        return bit is not None and bool(self._code & (1 << bit))

    def allows(self, action: str) -> bool:
        """Does this authority actually authorize *action*?

        The whole question, in one call: an allow effect **and** the bit. This is the
        replacement for `getattr(grant, flag_attr, False)` at every enforcement point — that
        expression is the S1 bug, because it answers only the second half.

        An action name outside :data:`ACTIONS` is False, never an error: an unmapped verb
        must be a denial, not a hole opened by a typo.
        """
        return self.is_allow and self.carries(action)

    # -- dunder plumbing --------------------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mask):
            return self._code == other._code
        return NotImplemented

    def __hash__(self) -> int:
        return hash((Mask, self._code))

    def __bool__(self) -> bool:
        """Truthy iff this authority authorizes *something*. `bool(DENY) is False`."""
        return self.is_allow and bool(self._code & _ACTION_BITS)

    def __repr__(self) -> str:
        listed = ",".join(sorted(self.actions)) or "-"
        return f"Mask({'allow' if self.is_allow else 'deny'}:{listed})"


#: The zero. Absorbing under the meet: ``DENY & a is DENY`` for every `a`, which is the law
#: `masked_by` always honored ("a deny member inside an allow bundle must stay a deny") and
#: the light-cone path did not.
DENY: Mask = Mask(0)

#: The identity. Full authority over every action: ``TOP & a is a`` for every `a`. It is
#: what a null ``edge.propagate`` column decodes to, so an unrestricted edge composes to a
#: no-op — which is exactly what unrestricted should mean.
TOP: Mask = Mask(_ACTION_BITS | _ALLOW_BIT)

#: An allow that carries no action — the `propagate=[]` edge `artifacts_router` writes on a
#: non-lineage link. Distinct from :data:`DENY`: it is a permitted path that transmits
#: nothing, not a refusal. They meet the same way, but they say different things and the
#: encodings keep them apart (`'[]'` versus a deny effect on a grant).
NOTHING: Mask = Mask(_ALLOW_BIT)


def attenuate(a: Mask, b: Mask) -> Mask:
    """Function form of the meet, for call sites that pass it around as a value."""
    return a & b


def compose(masks: Iterable[Mask]) -> Mask:
    """Fold the meet along a path — the composed authority of a chain of edges.

    The empty path is :data:`TOP`, which is the identity and therefore the only answer that
    keeps `compose(p + q) == compose(p) & compose(q)` true when either side is empty: a
    zero-hop walk must not narrow anything.

    The result is below **every** mask on the path, at any length. That is the whole
    security content of "the light cone is a light cone".
    """
    out = TOP
    for m in masks:
        out = out & m
    return out


def propagates(column_value: Any, action: str) -> bool:
    """Does an ``edge.propagate`` column value let *action* through?

    One call replacing the inline `mask is not None and action not in mask` that the origin
    BFS (`lattice_api.list_origin_descendants`) and the upward walk
    (`services.dependencies.check_access`) each spell out for themselves.
    """
    return Mask.from_propagate(column_value).allows(action)
