"""The ONE access decision over the lattice — the light-cone, not flags.

⚠ MOVED FROM `ember/identity/access.py` ON 2026-07-31. It always belonged here: its own first
paragraph said the mechanism "already exists on the lattice", named `mantle.db.lattice_api` as the
place grants are stored and light-cones computed, and described itself as ember's LOCAL path to the
same answer without an HTTP round trip. A second implementation of the store's own access model,
living in the runner, is the shape this whole cleanup removes — and this one composed mantle's
primitives (`get_active_collection_ids_for_user`, `list_origin_descendants`) into the 15 functions
callers actually want, which is exactly the composition mantle should have been offering.

It qualifies for this subtree on the store's own terms: module-level imports are `__future__` and
`typing` and nothing else, so the embeddable-surface claim is untouched.

Ember reaches it as `mantle.db.lattice.access` — ember → mantle is legal; the reverse was not.

Access in Agience is CRUDEASIO grants that PROPAGATE by graph reachability, and that mechanism already
exists on the lattice: `mantle.db.lattice_api` stores grants (artifacts, `content_type=_GRANT_CT`) and
computes read light-cones (`get_active_collection_ids_for_user` + `list_origin_descendants` over the
`contains` containment edges, honoring each edge's `propagate` mask). The mantle service already
enforces it; ember's chat/serve path had drifted to a PARALLEL flag scheme (`no_share`/`visibility`/
`owner` + the `private.<principal>` prefix) — the exact "second mechanism for sharing" canon forbids.
This module is the single mechanism ember's LOCAL path uses instead, driven straight against its own
lattice store (no HTTP round trip — ember holds `.artifacts` and `.graph`, which is all the light-cone
needs).

THE MODEL (confirmed by John, 2026-07-28):
  · Public is the UN-KEYED TOP — the shared corpus everyone reads with NO grant. "No explicit grant"
    means public, not denied.
  · Grants only ADD private/owned reach on top. An owner collection is made private by MINTING a grant
    on it (the owner's Read grant); its members are then reachable only through a light-cone that
    includes that grant.
  · So `is_public(art)` ⇔ art's grounding collection is gated by NO grant; and
    `can_read(art, person)` = `is_public(art)` OR person's read light-cone reaches art's grounding.
  · Authorship is PROVENANCE (`created_by`), which grants no access — there is no owner fast-path.

Nothing here reads `visibility`/`no_share`/`no_promote`/`owner` or the `private.` prefix. Those are the
drift; this replaces them.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Set

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


class _DB:
    """Adapts whatever store shape ember hands us to the `(.artifacts, .graph)` object `lattice_api`
    wants. Ember passes EITHER a full store/bundle (`.artifacts`, `.graph`) OR the bare artifacts store
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
    """Read-reachable containment descendants of `rid`, or empty when there is no graph to walk."""
    if db.graph is None:
        return []
    return la.list_origin_descendants(db, [rid], _READ)


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
    """Every collection id that ANY active grant gates — the grant's `resource_id` plus everything
    reachable from it through the read light-cone (containment). A collection in this set is OWNED
    (private); a collection NOT in it is public TOP. Public corpus roots carry no grants, so they are
    never here. Grants are few (a node's owners' collections), so this is cheap.

    ⛔ This is the whole definition of "non-public," and it is computed from grants ALONE — never a flag
    and never the `private.` name."""
    la, db = _api(), _DB(store)
    gated: Set[str] = set()
    try:
        for d in la._grant_docs(db, state="active"):
            rid = d.get("resource_id")
            if not rid or not d.get("can_read") or not la._unexpired(d):
                continue
            gated.add(str(rid))
            for did in _descendants(db, la, rid):
                gated.add(str(did))
    except Exception:
        # Unknown gating must FAIL CLOSED for the mesh (treat all as gated is too strict for reads, so
        # callers decide) — here we surface the empty set only when there are genuinely no grants; a
        # fault re-raises so no caller silently treats private data as public.
        raise
    return gated


def reachable_collections(store, principal: Optional[str]) -> Set[str]:
    """A principal's READ light-cone as collection ids: every collection its active readable grants
    reach, directly and by containment. An artifact is reachable by the principal iff its grounding is
    in this set. An anonymous caller (`principal` falsy) reaches nothing private — only the public TOP,
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
    """`{gated_collection_id -> owner principal}`. The owner is the GRANTEE of the collection's Read
    grant (the raw principal — e.g. `private.<p>` → `<p>`), derived from grants alone: never a stored
    `owner` field and never the `private.` name. Descendants inherit the root's owner. This is what
    `curate` reports as WHO holds a private collection; its keys are exactly `gated_collections`."""
    la, db = _api(), _DB(store)
    owners: Dict[str, str] = {}
    for d in la._grant_docs(db, state="active"):
        rid, who = d.get("resource_id"), d.get("grantee_id")
        if not rid or not who or not d.get("can_read") or not la._unexpired(d):
            continue
        if str(who) == PUBLIC_PRINCIPAL:               # the commons is not an OWNER of anything
            continue
        owners.setdefault(str(rid), str(who))
        for did in _descendants(db, la, rid):
            owners.setdefault(str(did), str(who))
    return owners


def is_public(store, art: Dict[str, Any]) -> bool:
    """Public ⇔ BORN public (ungrounded, or its grounding collection is gated by no grant — the un-keyed
    TOP) OR MADE public (the `PUBLIC_PRINCIPAL` holds a Read grant reaching it, on the collection or the
    artifact). Making a private thing public is a grant to the public entity — no copy, no re-key."""
    g = grounding_of(art)
    if g is None or g not in gated_collections(store):
        return True                                   # born public
    reach = reachable_collections(store, PUBLIC_PRINCIPAL)   # made public
    return g in reach or str(art.get("id") or "") in reach


def can_read(store, art: Dict[str, Any], principal: Optional[str]) -> bool:
    """The read decision: public, OR the principal's light-cone reaches this artifact — either through
    a grant on its grounding COLLECTION or a grant on the ARTIFACT itself (a one-artifact share).
    No flags, no owner fast-path, no prefix."""
    if not art:
        return False
    if is_public(store, art):                         # born or made public
        return True
    reach = reachable_collections(store, principal)   # else: this principal's own light-cone
    g = grounding_of(art)
    return (g is not None and g in reach) or str(art.get("id") or "") in reach


def _flag_private(d: Dict[str, Any]) -> bool:
    """The LEGACY private markers, read ONLY by the one-time backfill (never by a live access
    decision). A collection carries them in `context`; a memory top-level."""
    def _f(x) -> bool:
        if not isinstance(x, dict):
            return False
        return bool(x.get("no_share")) or str(x.get("visibility") or "").strip().lower() == "private" \
            or str(x.get("kind") or "").strip().lower() == "private"
    return _f(d) or _f(d.get("context"))


def backfill_grants_from_flags(store, *, cap: int = 200000) -> Dict[str, int]:
    """ONE-TIME migration: mint the owner's Read grant for every collection that the LEGACY flags
    marked private but that has no grant yet. This is what makes deleting the flag scheme safe on a
    store that already holds flag-private data — afterwards `gated_collections` covers exactly what the
    flags used to. Idempotent (grant upsert). Runs at bootstrap, before any read is served.

    Collection-scoped by design: owner data is written into an owner collection (`_ensure_private`),
    whose collection artifact carries the flags and whose members ground to it. So minting one grant per
    flagged collection gates all its members. (A stray flag-private member in an UNflagged collection
    would not be covered — not a shape the writers produce; logged via the returned counts.)"""
    from mantle.db.lattice.constants import COLLECTION_CONTENT_TYPE
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


# ── INVOKE — the discharge verb (CRUDEASIO "I"), CLOSED by default ────────────────────────────────
# John, 2026-07-29: "can discharge only if the energy is granted access." Discharge ACTUATES the world
# (a sensor, an actuator, fs.write, net.request), so its polarity is the OPPOSITE of Read's: Read is
# open by default (no grant = the public TOP), Invoke is CLOSED by default (no grant = nothing fires).
# Reusing the read light-cone would make every organon world-firable — same MECHANISM, opposite DEFAULT.
#
# No new schema was needed: `lattice_api.upsert_user_collection_grant` already carries `can_invoke`
# (default False), so CRUDEASIO's I is Invoke and the closed default is already the stored shape.
_INVOKE = "invoke"


def invokable_resources(store, principal: Optional[str]) -> Set[str]:
    """The principal's INVOKE light-cone: every resource an active, unexpired `can_invoke` grant
    reaches, directly and by CONTAINMENT — so granting Invoke on a crystal/bundle propagates to the
    organons it contains (grants propagate; a per-organon ACL would be the pipeline shape).

    Returns the EMPTY set for an anonymous principal, and that is the honest answer: unlike Read there
    is no public TOP to fall back on."""
    if not principal:
        return set()
    la, db = _api(), _DB(store)
    out: Set[str] = set()
    for d in la._grant_docs(db, state="active"):
        if (d.get("grantee_id") == str(principal)
                and d.get("grantee_type") == "user"
                and d.get("can_invoke")
                and d.get("resource_id")
                and la._unexpired(d)):
            rid = str(d["resource_id"])
            out.add(rid)
            for did in _descendants(db, la, rid):
                out.add(str(did))
    return out


def may_invoke(store, resource_id: Optional[str], principal: Optional[str]) -> bool:
    """CLOSED BY DEFAULT — no grant means NO, never "public". The one decision for actuation."""
    if not principal or not resource_id:
        return False
    return str(resource_id) in invokable_resources(store, principal)


def grant_invoke(store, resource_id: str, grantee: str, granted_by: str) -> None:
    """Mint (upsert) an INVOKE grant — `grantee` may actuate `resource_id` and, by containment,
    what it contains. Deliberately separate from `grant_read`: being able to READ a crystal must not
    imply being able to FIRE it."""
    _api().upsert_user_collection_grant(
        _DB(store), user_id=str(grantee), collection_id=str(resource_id),
        granted_by=str(granted_by), can_read=False, can_invoke=True)


class DischargeAuthority:
    """The duck-typed authority `crystal.Crystal.discharge(..., authority=)` asks.

    This is the ember-side bridge: crystal is Apache and downstream, so it can never import the grant
    machinery — it only asks `may_discharge(principal, organon, crystal_sha)` and this answers from the
    Invoke light-cone. The grant is looked up against the CRYSTAL (its content address) first, since
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
    """Mint (upsert) a Read grant giving `grantee` reach to `resource_id` — a COLLECTION or a single
    ARTIFACT. Idempotent. This is CRUDEASIO Share (#1): sharing WITH a person WITHOUT making anything
    public — the resource stays gated, but the grantee's light-cone now reaches it, and revoking is one
    edit. A grant on an artifact id reaches exactly that artifact (see `can_read`), so one memory can be
    shared without exposing the owner's whole collection."""
    _api().upsert_user_collection_grant(
        _DB(store), user_id=str(grantee), collection_id=str(resource_id),
        granted_by=str(granted_by), can_read=True)


def mint_owner_read_grant(store, collection_id: str, owner: str) -> None:
    """Make a collection PRIVATE by minting its owner's Read grant — the write-path primitive that
    replaces stamping `visibility`/`no_share`/`owner`. After this, the collection is gated (so
    `is_public` is False for its members) and reachable only by a light-cone that includes `owner`.
    Idempotent. Authorship stays PROVENANCE (`created_by`); the grant governs access."""
    grant_read(store, collection_id, owner, owner)


# ── VISIBILITY: the same decision, at the point of RETRIEVAL ─────────────────────────────────────
# ⚠ MOVED FROM `ember/runtime/delegate.py` ON 2026-07-31. `visible_to` had already been reduced to a
# fail-closed guard around `access.can_read` — the delegate module was holding a two-line wrapper
# over a decision that lives here, and `store/keyed.py` reached across a package boundary to get it.
# Authorization is the EXTENT OF THE FIELD, not a gate at a door, so it belongs with the light-cone.
def visible_to(art: Dict[str, Any], person: str, *, store: Any = None) -> bool:
    """May a delegate acting for `person` see this artifact?

    Pass `store` to use the GRANT light-cone (the real mechanism, `ember.access.can_read`). Without it,
    only the legacy flag gate applies — kept for the transition and removed at the grant cutover.

    ⛔ THE LEAK THIS CLOSES. Retrieval was entirely unscoped: `keyed.lookup_by_lemma`,
    `_aligned_snippet`, `router.route` and `ember.ask` took no principal at all. But `learn()`
    writes a PRIVATE triple carrying `"lemmas": [subject, object]`, and `remember()` writes private
    rows with keyed lemmas by design — so a lemma lookup by ANY delegate could return ANOTHER
    person's private memory. Nothing bounded what an activation could reach.

    The rule is the one the writers already stamp, now read on the way out:
      · not private              -> visible to everyone
      · private and owned by me  -> visible
      · private and owned by someone else, OR private with NO owner -> NOT visible

    ⚠ FAILS CLOSED ON A MALFORMED ROW. A row marked private with no `owner` is unattributable, so
    it is refused rather than shown — an unowned private row is a bug, and showing it to everybody
    is the worst available reading of one.

    This is design decision D4 (AGENT-HOST-DESIGN.md) at the point of retrieval: authorization is
    the EXTENT OF THE FIELD, not a gate at a door. A delegate simply cannot activate outside what
    its person can reach.
    """
    if not art:
        return False
    # Access is CRUDEASIO grants that propagate (see `ember.access`): visible iff the artifact is
    # public (its grounding collection is gated by no grant) OR the person's read light-cone reaches
    # it. This is the ONE mechanism — the same light-cone the mantle service uses. No flags.
    if store is None:
        # The decision REQUIRES the store (grants resolve against it). A caller that cannot supply one
        # cannot authorize, so FAIL CLOSED rather than guess — there is no flag fallback any more.
        return False
    return can_read(store, art, person)


def filter_visible(rows, person: str, *, store: Any = None) -> List[Dict[str, Any]]:
    """`visible_to` over a sequence, preserving order.

    ⚠ THE LIMIT IS SPENT BEFORE THIS RUNS. Like the `typed=False` post-filter in `keyed.py`, this
    cannot recover rows a page already spent on artifacts the caller may not see — a page that is
    entirely someone else's private rows filters to `[]`. That is the correct answer (they are not
    visible), but it is a THIN answer, not an authoritative "not found". Callers that care should
    over-fetch, exactly as the typed path does."""
    return [a for a in (rows or []) if visible_to(a, person, store=store)]
