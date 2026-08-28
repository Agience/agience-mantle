"""The one access decision over the lattice — the light-cone, not flags.

Module-level imports are `__future__` and `typing` and nothing else, so the embeddable-surface claim
is untouched. Ember reaches it as `mantle.db.access` — ember → mantle is legal; the reverse
is not.

The model:
  · Public is the un-keyed top — the shared corpus everyone reads with no grant. "No explicit grant"
    means public, not denied.
  · Grants only add private/owned reach on top. An owner collection is made private by minting a grant
    on it (the owner's Read grant); its members are then reachable only through a light-cone that
    includes that grant.
  · So `is_public(art)` ⇔ art's grounding collection is gated by no grant; and
    `can_read(art, person)` = `is_public(art)` OR person's read light-cone reaches art's grounding.
  · Authorship is provenance (`created_by`), which grants no access — there is no owner fast-path.

Nothing here reads `visibility`/`no_share`/`no_promote`/`owner` or the `private.` prefix; the grant
light-cone replaces them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

# The action a plain read requires, in CRUDEASIO terms.
_READ = "read"

# The COMMONS entity. Public is either "born public" (grounded in the un-keyed TOP, no grant) OR "made
# public" by granting THIS reserved principal Read on the resource — no copy, no re-key: the same grant
# mechanism, with `PUBLIC_PRINCIPAL` as grantee, is what turns a private thing public. Anyone's read
# includes what the public entity can reach.
PUBLIC_PRINCIPAL = "public"


def _api():
    """The lattice grant/light-cone API. Lazy so a non-lattice caller never imports it."""
    from mantle.db import lattice_api as _la   # sibling now, not a cross-repo reach
    return _la


def _grant_mask(d: Dict[str, Any]):
    """A raw grant DOCUMENT's authority as the one attenuation `Mask` — its CRUDEASIO bits and
    its effect, joined by `entities.grant.mask_of`.

    Everything on this side of the lattice reads grants as raw dicts, never as entities, which
    is why `d.get("can_invoke")` was the natural spelling and why it was only ever half the
    question: the bit says *which* action, the effect says whether the grant authorizes at all.
    A deny grant carrying `can_invoke` answers True to the bare `.get` and would authorise
    actuation's sharpest sibling).

    The doc is viewed through a `SimpleNamespace` rather than passed as a mapping because
    `grant_is_allow` is `getattr`-duck-typed: a dict would read as "no effect attribute" and
    therefore as a non-allow, which is fail-closed but wrong (it would deny every grant). The
    namespace makes the predicates see the doc exactly as they see an entity, so neither the
    effect vocabulary nor the flag set is restated here. Missing flags default False, as
    `.get` did — unlike `Grant.from_dict`, whose `can_read=True` default would WIDEN a doc that
    is missing the column.

    Both imports are lazy: this module's module-level imports are `__future__` and `typing`
    only, and the embeddable-surface claim above depends on that staying true.
    """
    from types import SimpleNamespace
    from mantle.entities.grant import mask_of
    return mask_of(SimpleNamespace(**d))


class _DB:
    """Adapts whatever store shape ember hands us to the `(.artifacts, .graph)` object `lattice_api`
    wants. Ember passes either a full store/bundle (`.artifacts`, `.graph`) or the bare artifacts store
    (e.g. from `keyed`). `.graph` may be `None` in the latter case; grant containment-propagation is
    then skipped, which is exact while no collection nests (the measured reality) and the honest
    partial otherwise."""
    __slots__ = ("artifacts", "graph")

    def __init__(self, store):
        a = getattr(store, "artifacts", None)
        if a is not None:                      # a full store/bundle
            self.artifacts = a
            self.graph = getattr(store, "graph", None)
        else:                                  # `store` IS the artifacts store
            self.artifacts = store
            self.graph = getattr(store, "graph", None)


def _descendants(db, la, rid):
    """The COLLECTIONS gated by a grant on `rid` — never its members.

    `gated_collections` puts collection ids in a set, so expanding a grant to its member artifacts
    adds nothing it can use and costs everything. Measured 2026-08-25 on 71/home: a grant on
    `stage.0.lexicon` sent `list_origin_descendants` through **1,846,819** containment edges to
    discover which of **70** collections were gated, and raised `EdgesTruncated` above the
    1,000,000-edge cap before returning any of them. That failure reached
    `consolidate_colimit` -> `is_public`, so consolidation could not run at all on a store whose
    lexicon had grown past the cap.

    So this asks the COLLECTIONS instead. They are few, they each name their own parent, and the
    walk is up rather than down: `O(collections x depth)` instead of `O(members)`. The answer is
    identical — a member is not a collection and never belonged in the set.

    Nesting is supported here even though this store does not use it: measured, ZERO collections
    are contained by another, so the loop below finds nothing to add today. It is written for the
    tree rather than for the current shape, because a flat store is a fact about now.
    """
    if db.graph is None:
        return []
    out, seen = [], {str(rid)}
    frontier = [str(rid)]
    while frontier:
        parent = frontier.pop()
        for cid in _collection_children(db, parent):
            if cid in seen:
                continue                      # a cycle is a lie about the tree; do not walk it twice
            seen.add(cid)
            out.append(cid)
            frontier.append(cid)
    return out


def _collection_children(db, parent: str):
    """Collection ids whose origin parent is `parent`, read off the vertex table.

    One indexed statement over the collection rows rather than a graph walk over the parent's
    membership — see `_descendants` for the measurement that made this necessary.
    """
    try:
        conn = db.artifacts.db.read()
    except Exception:                          # noqa: BLE001 — no SQL handle: nothing to add
        return []
    try:
        rows = conn.execute(
            "SELECT v.id FROM vertex v WHERE v.ct = ? AND EXISTS("
            "  SELECT 1 FROM edge e WHERE e.dst = v.id AND e.label = 'contains' "
            "  AND e.is_origin = 1 AND e.src = ?)",
            ("application/vnd.agience.collection+json", parent))
        return [str(r[0]) for r in rows]
    except Exception:                          # noqa: BLE001
        return []


def grounding_of(art: Dict[str, Any]) -> Optional[str]:
    """The collection an artifact is grounded in — its containment root. `collection_id` is the
    author's declared home (== `origin_root`, flat today). `None` means ungrounded: it belongs to no
    collection, so it is part of the public TOP and gated by nothing."""
    if not art:
        return None
    cid = art.get("collection_id")
    if cid:
        return str(cid)
    root = art.get("origin_root")
    return str(root) if root else None


def gated_collections(store) -> Set[str]:
    """Every collection id that any active grant gates — the grant's `resource_id` plus everything
    reachable from it through the read light-cone (containment). A collection in this set is owned
    (private); a collection not in it is public top. Public corpus roots carry no grants, so they are
    never here. Grants are few (a node's owners' collections), so this is cheap.

    Read through the one operator, but **effect-blind on purpose** (`carries`, not `allows`).
    Gating is not an authorization decision: it asks "is this collection administered by
    somebody", and a *deny* grant naming its Read column is administration. Reading it as
    `allows` would drop such a collection out of this set, `is_public` would then answer True
    for its members, and a grant that says "no" would have made the collection world-readable.
    That is the fail-OPEN direction, so the deny-blindness here is preserved deliberately
    rather than "fixed" — it is `Mask.carries`'s documented meaning, not a missing check."""
    la, db = _api(), _DB(store)
    gated: Set[str] = set()
    try:
        for d in la._grant_docs(db, state="active"):
            rid = d.get("resource_id")
            if not rid or not _grant_mask(d).carries(_READ) or not la._unexpired(d):
                continue
            gated.add(str(rid))
            for did in _descendants(db, la, rid):
                gated.add(str(did))
    except Exception:
        # Unknown gating must fail closed for the mesh (treat all as gated is too strict for reads, so
        # callers decide) — here we surface the empty set only when there are genuinely no grants; a
        # fault re-raises so no caller silently treats private data as public.
        raise
    return gated


def reachable_collections(store, principal: Optional[str]) -> Set[str]:
    """A principal's read light-cone as collection ids: every collection its active readable grants
    reach, directly and by containment. An artifact is reachable by the principal iff its grounding is
    in this set. An anonymous caller (`principal` falsy) reaches nothing private — only the public top,
    which needs no grant and is handled by `is_public`."""
    if not principal:
        return set()
    la, db = _api(), _DB(store)
    out: Set[str] = set()
    for rid in la.get_active_collection_ids_for_user(db, str(principal)):
        out.add(str(rid))
        for did in _descendants(db, la, rid):
            out.add(str(did))
    return out


def gated_owner_map(store) -> Dict[str, str]:
    """`{gated_collection_id -> owner principal}`. The owner is the grantee of the collection's Read
    grant (the raw principal — e.g. `private.<p>` → `<p>`), derived from grants alone: never a stored
    `owner` field and never the `private.` name. Descendants inherit the root's owner. This is what
    `curate` reports as who holds a private collection; its keys are exactly `gated_collections`.

    Two passes, allow-effect first. The key set has to stay exactly `gated_collections`' — a
    gated collection with no owner would report as unheld — and that set is deliberately
    effect-blind (see there), so a collection gated only by a deny grant must still get a name.
    But a deny grantee is the wrong answer whenever a real holder exists, and `setdefault` makes
    the first writer win, so the allow pass runs first and the deny pass only fills gaps. This
    map answers "who holds this" for reporting; nothing authorizes from it."""
    la, db = _api(), _DB(store)
    owners: Dict[str, str] = {}
    docs = [d for d in la._grant_docs(db, state="active") if la._unexpired(d)]
    for allow_pass in (True, False):
        for d in docs:
            rid, who = d.get("resource_id"), d.get("grantee_id")
            if not rid or not who:
                continue
            mask = _grant_mask(d)
            if not mask.carries(_READ) or mask.is_allow is not allow_pass:
                continue
            if str(who) == PUBLIC_PRINCIPAL:           # the commons is not an owner of anything
                continue
            owners.setdefault(str(rid), str(who))
            for did in _descendants(db, la, rid):
                owners.setdefault(str(did), str(who))
    return owners


def is_made_public(store, art: Dict[str, Any]) -> bool:
    """Made public ⇔ the `PUBLIC_PRINCIPAL` holds a Read grant reaching this artifact.

    The grant half of :func:`is_public`, and only that half. The two halves answer to different
    callers, and this one is what the API takes.

    `is_public` also returns True for **born public**: ungrounded, or grounded in a collection no
    grant gates. That reading belongs to the ember/shard model this module was written for, where an
    ungated collection IS the un-keyed TOP and carries nothing secret. It is the wrong reading for
    Mantle's API, where "no grant" is the ordinary state of a private thing: a collection before its
    grants are attached, and — because `lattice_api._stamp_origin_root` deliberately leaves the
    field unset rather than guess a principal — any artifact whose key root could not be
    established. Adopting born-public into `services.dependencies.check_access` would have made
    every one of those world-readable, which is the fail-OPEN direction and the opposite of the
    README's binding invariant that authorization is decided only by the light cone and grants.

    This half is safe there because it is a positive act by somebody holding authority: a Read grant
    to the commons, subject to the same clamp (`grant_service.clamp_to_issuer`) and the same
    attenuation as any other grant. It is also **exactly** what `mesh/sync._withheld_lattice`
    consults (`reachable_collections(store, PUBLIC_PRINCIPAL)`, "keep made-public members"), so the
    API and the mesh now answer the same question from the same function rather than from two
    readings that happened to disagree.
    """
    reach = reachable_collections(store, PUBLIC_PRINCIPAL)
    g = grounding_of(art)
    return (g is not None and g in reach) or str(art.get("id") or "") in reach


def is_public(store, art: Dict[str, Any]) -> bool:
    """Public ⇔ born public (ungrounded, or its grounding collection is gated by no grant — the un-keyed
    top) or made public (the `PUBLIC_PRINCIPAL` holds a Read grant reaching it, on the collection or the
    artifact). Making a private thing public is a grant to the public entity — no copy, no re-key.

    Both halves. See :func:`is_made_public` for the half the API takes, and why.
    """
    g = grounding_of(art)
    if g is None or g not in gated_collections(store):
        return True                                   # born public
    return is_made_public(store, art)                 # made public


def can_read(store, art: Dict[str, Any], principal: Optional[str]) -> bool:
    """The read decision: public, or the principal's light-cone reaches this artifact — either through
    a grant on its grounding collection or a grant on the artifact itself (a one-artifact share).
    No flags, no owner fast-path, no prefix."""
    if not art:
        return False
    if is_public(store, art):                         # born or made public
        return True
    reach = reachable_collections(store, principal)   # else: this principal's own light-cone
    g = grounding_of(art)
    return (g is not None and g in reach) or str(art.get("id") or "") in reach


def _flag_private(d: Dict[str, Any]) -> bool:
    """The flag-based privacy markers, read only by the one-time backfill (never by a live access
    decision). A collection carries them in `context`; a memory top-level."""
    def _f(x) -> bool:
        if not isinstance(x, dict):
            return False
        return bool(x.get("no_share")) or str(x.get("visibility") or "").strip().lower() == "private" \
            or str(x.get("kind") or "").strip().lower() == "private"
    return _f(d) or _f(d.get("context"))


def backfill_grants_from_flags(store, *, cap: int = 200000) -> Dict[str, int]:
    """One-time migration: mint the owner's Read grant for every collection that the flag-based scheme
    marks private but that has no grant yet. This is what makes the flag scheme safe to remove on a
    store that still holds flag-private data — once run, `gated_collections` covers exactly what the
    flags mark. Idempotent (grant upsert). Runs at bootstrap, before any read is served.

    Collection-scoped by design: owner data is written into an owner collection (`_ensure_private`),
    whose collection artifact carries the flags and whose members ground to it. So minting one grant per
    flagged collection gates all its members. (A stray flag-private member in an unflagged collection
    would not be covered — not a shape the writers produce; logged via the returned counts.)"""
    from mantle.db.constants import COLLECTION_CONTENT_TYPE
    minted = 0
    already = gated_collections(store)
    cols, _exh = store.artifacts.list_by_content_type(COLLECTION_CONTENT_TYPE, cap=cap)
    for c in cols:
        if not (c and _flag_private(c)):
            continue
        cid = c.get("id")
        ctx = c.get("context") if isinstance(c.get("context"), dict) else {}
        owner = c.get("owner") or ctx.get("owner")
        if not cid or not owner or cid in already:
            continue
        mint_owner_read_grant(store, cid, owner)
        minted += 1
    return {"minted": minted, "collections_seen": len(cols)}


# ── invoke — the discharge verb (CRUDEASIO "I"), closed by default ────────────────────────────────
# Discharge can occur only if the energy is granted access. Discharge actuates the world (a sensor,
# an actuator, fs.write, net.request), so its polarity is the opposite of Read's: Read is open by
# default (no grant = the public top), Invoke is closed by default (no grant = nothing fires).
# Reusing the read light-cone would make every organon world-firable — same mechanism, opposite default.
#
# No new schema was needed: `lattice_api.upsert_user_collection_grant` already carries `can_invoke`
# (default False), so CRUDEASIO's I is Invoke and the closed default is already the stored shape.
_INVOKE = "invoke"


def invokable_resources(store, principal: Optional[str]) -> Set[str]:
    """The principal's invoke light-cone: every resource an active, unexpired `can_invoke` grant
    reaches, directly and by containment — so granting Invoke on a crystal/bundle propagates to the
    organons it contains (grants propagate; a per-organon ACL would be the pipeline shape).

    Returns the empty set for an anonymous principal, and that is the honest answer: unlike Read there
    is no public top to fall back on.

    Effect is decided by the one operator, and this one is `allows` — not `carries`. Unlike
    `gated_collections`, where deny-blindness errs closed, this set IS the authorization: it is
    what `may_invoke` and `DischargeAuthority.may_discharge` answer from, and actuation fires
    the world. `d.get("can_invoke")` asked only whether the bit was set, so a `deny`-effect
    grant carrying `can_invoke` — the exact shape written to say "this principal must never
    fire this" — authorised discharge. That was 's sharpest sibling, and it failed OPEN.

    Deny is then subtracted rather than merely skipped, so this agrees with `check_access`,
    where deny is tested first and wins over any allow at any level of the walk. Subtraction
    only ever removes, so it cannot itself widen: an allow grant on a crystal plus a deny grant
    on one organon inside it fires everything but that organon."""
    if not principal:
        return set()
    la, db = _api(), _DB(store)
    out: Set[str] = set()
    denied: Set[str] = set()
    for d in la._grant_docs(db, state="active"):
        rid = d.get("resource_id")
        if (d.get("grantee_id") != str(principal)
                or d.get("grantee_type") != "user"
                or not rid
                or not la._unexpired(d)):
            continue
        mask = _grant_mask(d)
        if not mask.carries(_INVOKE):
            continue
        # `carries` selected the ones that SPEAK about invoke; `is_allow` sorts them into the
        # two piles. A deny grant's bits name the actions it denies, which is why the bit test
        # comes first and the effect decides the pile rather than the membership.
        sink = out if mask.is_allow else denied
        sink.add(str(rid))
        for did in _descendants(db, la, rid):
            sink.add(str(did))
    return out - denied


def may_invoke(store, resource_id: Optional[str], principal: Optional[str]) -> bool:
    """Closed by default — no grant means no, never "public". The one decision for actuation."""
    if not principal or not resource_id:
        return False
    return str(resource_id) in invokable_resources(store, principal)


def grant_invoke(store, resource_id: str, grantee: str, granted_by: str) -> None:
    """Mint (upsert) an invoke grant — `grantee` may actuate `resource_id` and, by containment,
    what it contains. Deliberately separate from `grant_read`: being able to read a crystal must not
    imply being able to fire it."""
    _api().upsert_user_collection_grant(
        _DB(store), user_id=str(grantee), collection_id=str(resource_id),
        granted_by=str(granted_by), can_read=False, can_invoke=True)


class DischargeAuthority:
    """The duck-typed authority `crystal.Crystal.discharge(..., authority=)` asks.

    This is the ember-side bridge: crystal is Apache and downstream, so it can never import the grant
    machinery — it only asks `may_discharge(principal, organon, crystal_sha)` and this answers from the
    Invoke light-cone. The grant is looked up against the crystal (its content address) first, since
    that is what a person grants; the organon name is accepted too so a single organon can be granted
    narrowly."""

    __slots__ = ("_store",)

    def __init__(self, store) -> None:
        self._store = store

    def may_discharge(self, principal: Optional[str], organon: str,
                      crystal_sha: Optional[str] = None) -> bool:
        reach = invokable_resources(self._store, principal)
        if not reach:
            return False
        return (crystal_sha is not None and str(crystal_sha) in reach) or str(organon) in reach


def grant_read(store, resource_id: str, grantee: str, granted_by: str) -> None:
    """Mint (upsert) a Read grant giving `grantee` reach to `resource_id` — a collection or a single
    artifact. Idempotent. This is CRUDEASIO Share (#1): sharing with a person without making anything
    public — the resource stays gated, but the grantee's light-cone now reaches it, and revoking is one
    edit. A grant on an artifact id reaches exactly that artifact (see `can_read`), so one memory can be
    shared without exposing the owner's whole collection."""
    _api().upsert_user_collection_grant(
        _DB(store), user_id=str(grantee), collection_id=str(resource_id),
        granted_by=str(granted_by), can_read=True)


def mint_owner_read_grant(store, collection_id: str, owner: str) -> None:
    """Make a collection private by minting its owner's Read grant — the write-path primitive that
    replaces stamping `visibility`/`no_share`/`owner`. After this, the collection is gated (so
    `is_public` is False for its members) and reachable only by a light-cone that includes `owner`.
    Idempotent. Authorship stays provenance (`created_by`); the grant governs access."""
    grant_read(store, collection_id, owner, owner)


# ── visibility: the same decision, at the point of retrieval ─────────────────────────────────────
def visible_to(art: Dict[str, Any], person: str, *, store: Any = None) -> bool:
    """May a delegate acting for `person` see this artifact?

    Pass `store` to use the grant light-cone (the real mechanism, `ember.access.can_read`). Without it,
    only the flag-based gate applies.

    The rule is the one the writers already stamp, now read on the way out:
      · not private              -> visible to everyone
      · private and owned by me  -> visible
      · private and owned by someone else, or private with no owner -> not visible

    This is design decision D4 (AGENT-HOST-DESIGN.md) at the point of retrieval: authorization is
    the extent of the field, not a gate at a door. A delegate simply cannot activate outside what
    its person can reach.
    """
    if not art:
        return False
    # Access is CRUDEASIO grants that propagate (see `ember.access`): visible iff the artifact is
    # public (its grounding collection is gated by no grant) or the person's read light-cone reaches
    # it. This is the one mechanism — the same light-cone the mantle service uses. No flags.
    if store is None:
        # The decision requires the store (grants resolve against it). A caller that cannot supply one
        # cannot authorize, so this fails closed rather than guessing — there is no flag fallback.
        return False
    return can_read(store, art, person)


def filter_visible(rows, person: str, *, store: Any = None) -> List[Dict[str, Any]]:
    """`visible_to` over a sequence, preserving order."""
    return [a for a in (rows or []) if visible_to(a, person, store=store)]
