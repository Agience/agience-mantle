"""The light cone — BFS over the containment lattice, confined by the context one, and the
translation of what it reaches into the two granularities key custody and recall each need.

:class:`LightConeResolver` answers "which artifact ids may this principal reach", and
:func:`resolve_authorized_scope` turns that answer into the shapes its consumers can act on:
the ``(cell_principal, collection_id)`` pairs a key can be derived for, and the artifact ids
themselves. Both consumers live outside this module — `oracle.LightConeGrantVerifier` gates
master-key issuance on the pairs, and the search accessors meet the ids against retrieval —
so the translation belongs with the resolver rather than with either caller. There is one
implementation of "what may this principal reach, and under which key", and a second opinion
about it is the S1 class of defect this module's sections below keep closing.

The resolver
------------
Resolves the set of artifact IDs reachable from a principal's grants by
walking ``origin: true, relationship: null`` edges outbound through
the lattice, intersecting each edge's ``propagate`` mask with the requested
action, and then walking **context** edges — `db.edge.CONTEXT_LABEL` — out of
everything that reached, **confined to what containment already authorized**. Context is an
artifact with edges to its own context (D16), so authorization and recall traverse one
structure rather than computing an authorized set and filtering results against it.

The context half is :mod:`mantle.services.context_service`, which owns the composition, the
cycle guard and the bound; this resolver decides only what to seed it with, what action
ceiling to hold it under, and — the load-bearing one — what identity universe it may not
leave.

The context walk may not widen this set
---------------------------------------
An unconfined context walk unioned into the result would widen it: a grant on `org` would come
back holding `{"org", "project", "doc-1"}` — ids no grant reaches and `check_access` would
refuse. This path feeds `oracle.LightConeGrantVerifier` (content-key issuance) and
`sse/router_accessor` (result decryption), so reaching more than the read gate allows is not a
recall nicety; it is a key handed out for an artifact the gate would 404. The walk is confined
to the grant-derived set — `context_service.reach(..., within=...)` — which makes
`resolve(p, a) ⊆ grants-alone(p, a)` a property of the call rather than a claim about it.

Where this stands against `check_access`
----------------------------------------
`services.dependencies.check_access` is the gate in front of every artifact read and is the
authority; this resolver must not exceed it. They agree by construction on the two things
they both do — the same grant ledger, and the same `attenuation.propagates` prune on each
`propagate` column — and they compute the same containment closure from opposite ends:
`check_access` walks origin edges UPWARD from one artifact looking for a grant,
`list_origin_descendants` walks them DOWNWARD from every grant. Neither traverses a context
edge, and that is the honest statement of the current position:

    **A context edge confers no authority today.** `check_access` does not honour one, so
    this resolver does not either, and the walk below is confined to the point where it can
    admit nothing.

That is a deliberate under-reach, not an oversight: two implementations of "what may this
principal reach" that disagree is the S1 class of defect, and when they disagree the narrower
one is the only safe answer. Making context a real unit of sharing means teaching
`check_access` to walk the context lattice — one change, in `services/dependencies.py`, after
which this resolver's `within` widens to match and the walk starts doing work. The machinery
below is complete and tested for that day; what it is not is a second, wider opinion in the
meantime.

The authority a grant carries is one value, not two — a :class:`~mantle.attenuation.Mask`,
composed with the single attenuation meet rather than with a local copy of it. That is what
makes a ``deny``-effect grant unable to seed a light cone: deny is the absorbing element of
that operator. Filtering on the CRUDEASIO flag alone (``getattr(g, flag_attr, False)``)
answered only half the question and read a deny as authorizing — audit finding S1.

The per-edge composition below the seeds still lives in
`db.lattice_api.list_origin_descendants`, which spells the same prune out for itself
(`mask is not None and action not in mask`). :func:`mantle.attenuation.propagates` is that
expression, proved bit-for-bit equivalent to it, and the guard test
`test_attenuation_is_single_sourced.py` carries the site as an outstanding re-point.

CRUDEASIO lives in Mantle (the lattice grants collection). Grants are read
directly from `db_store.get_active_grants_for_grantee` — no Origin HTTP
calls, so the resolver reads the same ledger `check_access` reads. Same data
source is one of the two things the section above says they share; it is not on
its own an agreement about what either reaches.

This is the only ACL path — both MANTLE-SSE lexical and MANTLE vector search
consume the resolver's authorized artifact set.

See `.dev/features/mantle-mvp.md` § Layer 1.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from mantle.attenuation import ACTIONS, Mask
from mantle.db import backend as db_store
from mantle.db.constants import state_of
from mantle.entities.grant import mask_of
from mantle.services import context_service

logger = logging.getLogger(__name__)


#: The ledger's ``grantee_type`` and the acting context's ``principal_type`` are
#: different axes, and this is the one place that says so.
#:
#: ``grantee_type`` is a credential kind — a grant is held either by a principal id
#: (``"user"``) or by a hashed bearer token (``"grant_key"``). That is the
#: whole vocabulary; see ``lattice_api.upsert_user_collection_grant``, which hard-codes
#: ``grantee_type="user"`` for every principal-held grant it mints.
#:
#: ``ActingPrincipal.principal_type`` is a wider entity vocabulary —
#: ``user | server | mcp_client | grant_key | service | delegation``.
#:
PRINCIPAL_GRANT_TYPE = "user"
_LEDGER_KEY_TYPES = frozenset({"grant_key"})


def ledger_grantee_type(principal_type: Optional[str]) -> str:
    """Map an acting-context ``principal_type`` to the ledger's ``grantee_type``.

    Key-shaped principals keep their own kind (they hold grants as a hashed
    credential); every other entity kind acts as a principal id and therefore holds
    :data:`PRINCIPAL_GRANT_TYPE` grants. See the comment above for why this is not
    the identity function.

    NOTE for ``grant_key``: this returns the right *grantee_type*, but a grant key's
    grants are NOT found by looking its principal id up under it. The acting principal
    for a key is the ROOT GRANT's id, while the stored ``grantee_id`` on that root is
    the token hash — so the lookup would come back empty. Resolution for a key goes
    through :meth:`LightConeResolver._grants_for` instead, which walks the bundle.
    """
    return principal_type if principal_type in _LEDGER_KEY_TYPES else PRINCIPAL_GRANT_TYPE


class LightConeResolver:
    """BFS over origin containment and context edges, with `propagate` masks."""

    def __init__(self, db, *, max_depth: Optional[int] = None,
                 max_nodes: Optional[int] = None) -> None:
        """*max_depth* / *max_nodes* bound the CONTEXT walk — see
        :func:`services.context_service.reach`. They are constructor arguments rather than
        per-call ones because the bound is a property of the node's resource envelope, not of
        one query; a caller that could raise it per request would have a way to make any
        single query as expensive as it liked.
        """
        self._db = db
        self._max_depth = (context_service.DEFAULT_MAX_DEPTH if max_depth is None
                           else int(max_depth))
        self._max_nodes = (context_service.DEFAULT_MAX_NODES if max_nodes is None
                           else int(max_nodes))

    def resolve(
        self,
        principal_id: str,
        action: str = "read",
        *,
        principal_type: str = "user",
    ) -> Set[str]:
        """Return artifact IDs the principal can reach for ``action``.

        Three-step traversal:

        1. Fetch the principal's grants from the lattice (grants collection). Keep the
           ones whose authority — CRUDEASIO bit **and** allow effect, read as one
           :class:`~mantle.attenuation.Mask` — permits ``action`` and that name a
           ``resource_id``.
        2. For each granted resource, BFS outbound through `origin: true,
           relationship: null`` edges in the lattice, pruning when an edge's
           ``propagate`` mask doesn't include the action. `seen` is the termination guard.
        3. Walk the CONTEXT lattice out of everything step 2 reached, **confined to what
           steps 1 and 2 already authorized**, alternating context hops with containment
           expansion so nesting composes. Each hop meets the authority held so far with the
           edge's mask through the one attenuation operator; non-origin context edges are
           refused; and no node outside the step-1-plus-2 set is admitted at all — naming
           someone else's artifact as sitting in your context must not be a way to acquire
           authority over it, and refusing only the *edge* left the *destination* unguarded.

        The returned set is therefore the union of directly-granted IDs and all descendants
        reachable through an unbroken chain of action-permitted origin edges — exactly what
        grants alone authorize. Step 3 can narrow within that and can never add to it, which
        is the invariant `tests/test_context_lattice.py` sweeps over grant/edge
        configurations rather than asserting on one case.

        The context walk is **bounded** (`max_depth` / `max_nodes` on the constructor) and
        says so when a bound bites, because the context lattice is a deeper structure than
        containment and an unbounded walk over it is a cost with no ceiling. Truncation only
        removes nodes, so it under-reaches — fail-closed for authority, and logged rather
        than silent for recall. The containment BFS inside `list_origin_descendants` still
        runs to exhaustion; bounding it is a change to `db/lattice_api.py`.

        ``principal_type`` is the caller's ACTING-CONTEXT entity kind, not a ledger
        ``grantee_type``; :func:`ledger_grantee_type` maps between the two axes and
        its comment explains why they are not the same vocabulary.

        Returns an empty set when the principal has no relevant grants or
        the action name is unknown.
        """
        if action not in ACTIONS:
            return set()

        grants = self._grants_for(principal_id, principal_type)
        # `mask_of(g).allows(action)` is the whole authorization question in one call: the
        # bit AND the effect. The bare flag check this replaced let a `deny`-effect grant
        # seed the cone — for key issuance (`oracle.LightConeGrantVerifier`) and for
        # search-result decryption alike.
        masked = [(g.resource_id, mask_of(g)) for g in grants if g.resource_id]
        granted_ids = [rid for rid, m in masked if m.allows(action)]

        # Absorbing a deny is not the same as subtracting one. `allows()` keeps a deny grant from
        # seeding the cone, which covers a deny standing alone; a deny sitting beside an allow on
        # the same resource needs what the allow seeded removed. With allow-read and deny-read both
        # on `col-1` and no subtraction:
        #       resolve('u1', 'read') -> {'col-1', 'art-a'}
        #       check_access('col-1') -> 404
        # `recall` returns the denied artifact and hydration decrypts its content inline.
        # Worse, this set feeds `LightConeGrantVerifier` -> `OracleService`, and keys are
        # derived per (principal, collection) while `contexts` is deliberately un-narrowed
        # below — so the denied principal was issued a key for the WHOLE COLLECTION.
        #
        # The module docstring's own contract is "this resolver must not exceed
        # `check_access`", which tests deny first and 404s. Both sibling implementations
        # already subtract — `lattice_api.get_active_collection_ids_for_user` and
        # `db/access.invokable_resources` — and this one is the path that decides key custody.
        #
        # `carries`, not `allows`, for the deny pile: a deny grant's bits name the actions it
        # denies, so `allows()` is false for every one of them by construction. `db/access.py`
        # states the same rule where it sorts the two piles.
        denied_ids = [rid for rid, m in masked if m.is_deny and m.carries(action)]

        if not granted_ids:
            return set()

        # What grants alone authorize. Everything after this line may narrow it; nothing may
        # add to it.
        granted: Set[str] = set(granted_ids)
        granted.update(db_store.list_origin_descendants(self._db, granted_ids, action))

        # Deny expands the same way the allow does. A deny on a collection has to reach that
        # collection's members, or denying a container leaves every artifact inside it reachable.
        # `db/access.invokable_resources` expands both piles for exactly this reason.
        if denied_ids:
            denied: Set[str] = set(denied_ids)
            denied.update(db_store.list_origin_descendants(self._db, denied_ids, action))
            granted -= denied

        # The context lattice, seeded from everything containment already authorized — the
        # two are one structure, so a context edge hanging off a descendant is as real as one
        # hanging off the grant itself.
        #
        # `within=granted` is the whole fix. A context edge shapes authority a grant already
        # conferred; it does not manufacture any, so the walk may not name an id outside the
        # set above. See the module docstring: `check_access` does not walk context edges, so
        # `granted` is the widest universe this resolver may honestly claim, and the walk
        # consequently contributes nothing today. Union rather than assignment because the
        # confinement is what makes it safe — if `within` ever widens (when `check_access`
        # learns the context lattice) this line starts carrying the extra reach without
        # needing to be rewritten, and until then it is provably a no-op.
        #
        # The action ceiling is `Mask.of((action,))` rather than any single grant's mask: the
        # seeds come from several grants with different bits, and every one of them has
        # already passed `allows(action)`. Stating the ceiling as exactly what is true of all
        # of them is honest, and the meet at the first hop makes it a real ceiling rather
        # than a comment.
        context_reach = context_service.reach(
            self._db, granted, action,
            within=granted,
            authority=Mask.of((action,)),
            max_depth=self._max_depth,
            max_nodes=self._max_nodes,
            expand_containment=db_store.list_origin_descendants,
        )
        result: Set[str] = set(granted)
        # Intersected, not trusted: `reach` guarantees `ids ⊆ within`, and this is the one
        # line that would have to be wrong as well for the guarantee to fail open.
        result.update(context_reach.ids & granted)
        return result

    def _grants_for(self, principal_id: str, principal_type: str) -> list:
        """The grants this principal holds, by principal kind.

        A grant key is its own principal (see `services.acting_principal`), and
        `principal_id` is the id of its ROOT grant. Its authority is the bundle that
        root resolves to — the root's own resource plus every member, each already
        narrowed by the root's bits — which is the same function the auth path uses,
        so search and `check_access` cannot disagree about a key's reach.
        """
        if principal_type == "grant_key":
            from mantle.services import grant_key_service

            root = db_store.get_grant_by_id(self._db, principal_id)
            if root is None or not root.is_active():
                return []
            return grant_key_service.resolve(self._db, root)

        return db_store.get_active_grants_for_grantee(
            self._db,
            grantee_id=principal_id,
            grantee_type=ledger_grantee_type(principal_type),
        )


# ---------------------------------------------------------------------------
# From "what may this principal reach" to "under which key"
# ---------------------------------------------------------------------------


def _raw_artifact(store_db, artifact_id: str):
    """Raw artifact doc by id, through `db.backend.get_raw_artifact` — the one raw-doc read
    shape every call site uses. Never shape-sniff the handle: a mock has every attribute, so
    a `hasattr` branch here would take whichever path the test happened to build."""
    from mantle.db.backend import get_raw_artifact
    return get_raw_artifact(store_db, artifact_id)


def _raw_artifacts(store_db, artifact_ids):
    """Raw artifact docs for MANY ids, in as few round trips as the store allows.

    The narrowed set for an ordinary question reaches thousands — a description narrows to 4,013
    and a common phrase to 8,517 — and this loop read one document per candidate. Measured on the
    4,013 case: 1.58s of a 5.58s recall, 389us per candidate, for rows the lattice can hand over in
    one statement. The same argument `ranking._reach_rank` already makes about its own reads:
    `id` is the primary key, so `IN (...)` is that index walk done once.

    Falls back to the per-id read when the batched shape is unavailable, so a store that does not
    answer SQL — a test double, a non-lattice backend — behaves exactly as before.

    A missing id simply does not come back, which is the same absence `_raw_artifact` returned as
    `None`; the caller already skips those, and skipping is what keeps an id the narrowing named
    but the store cannot show out of an authorized set.

    Root ids resolve too, and they have to. `ingest/pipeline_unified._sse_index_artifact` indexes
    under `artifact.root_id or artifact.id`, so for a versioned artifact every posting names the
    ROOT -- and an identity-addressed member has a root that is deliberately never materialised
    ("the root is the identity and the id is the version",
    `workspace_service.upsert_identity_member`). A root is therefore an id the narrowing legitimately
    names and `SELECT ... WHERE id IN (...)` can never answer.

    Measured 2026-08-25 on 71/home: the `Claude Code` capture collection held 97 containment edges,
    NONE of whose targets existed as a vertex, and 183 captures whose roots were all unmaterialised.
    Every hook capture -- session transcripts, file mirrors, commits -- was narrowed to and then
    dropped one line later by `if not doc: continue`. `recall("Session transcript agience-build")`
    answered with canon documents and never the transcript. The lane wrote for a month and returned
    nothing it wrote.

    So a second hop resolves whatever the first could not, through `versions_of_many` -- the store's
    own batched lineage read over `ix_v_root_id`, one statement per `IN_CHUNK` roots rather than one
    per miss. It runs ONLY on the misses, so the ordinary path (every id a real vertex) pays one
    extra `dict` comparison and no query at all.
    """
    import json as _json

    wanted = [str(a) for a in artifact_ids]
    out: Dict[str, Any] = {}
    if not wanted:
        return out
    try:
        conn = store_db.artifacts.db.read()
    except Exception:  # noqa: BLE001 — no SQL handle: fall back to the point reads
        conn = None
    if conn is not None:
        try:
            for start in range(0, len(wanted), 400):
                chunk = wanted[start:start + 400]
                rows = conn.execute(
                    "SELECT id, doc FROM vertex WHERE id IN (%s)" % ",".join("?" * len(chunk)),
                    chunk,
                )
                for artifact_id, blob in rows:
                    try:
                        doc = _json.loads(blob) if isinstance(blob, (str, bytes)) else blob
                    except Exception:  # noqa: BLE001 — a malformed row is an absent row
                        continue
                    if isinstance(doc, dict):
                        out[str(artifact_id)] = doc
            _resolve_roots(store_db, wanted, out)
            return out
        except Exception:  # noqa: BLE001 — the batched shape did not work here
            out = {}
    for artifact_id in wanted:
        try:
            doc = _raw_artifact(store_db, artifact_id)
        except Exception:  # noqa: BLE001 — store reads raise broadly
            continue
        if doc:
            out[artifact_id] = doc
    _resolve_roots(store_db, wanted, out)
    return out


def _resolve_roots(store_db, wanted, out) -> None:
    """Fill in, by ROOT, whatever `wanted` the id lookup could not answer. Mutates `out`.

    The newest COMMITTED version wins, which is the same choice `get_latest_committed_artifact`
    makes and for the same reason: `versions_of_many` orders each lineage oldest-first by
    `(_origin, _seq)` -- the gap-free proper-time identity -- so the last committed doc in the list
    is the current one. A draft is not substituted for a missing commit: `recall` reads the
    committed segment, and answering a committed query with a draft would return content that was
    never published.

    Never raises. A store with no `versions_of_many` (a test double, a non-lattice backend) leaves
    `out` exactly as the id lookup left it, which is the behaviour every caller had before this.
    """
    missing = [a for a in wanted if a not in out]
    if not missing:
        return
    try:
        lineages = store_db.artifacts.versions_of_many(missing) or {}
    except Exception:  # noqa: BLE001 — no lineage read here; the misses stay missing
        return
    for root_id, docs in lineages.items():
        for doc in reversed(list(docs or ())):
            if not isinstance(doc, dict):
                continue
            # `state_of` is the store layer's one answer for what a doc carrying no `state`
            # is in; a second answer here would file a stateless doc differently from the way
            # the index segments already file it.
            if state_of(doc) != "committed":
                continue
            out[str(root_id)] = doc
            break


#: The no-timestamps answer, for a scope that read no docs. Read-only so the shared default
#: cannot be written through by one caller and observed by the next.
_NO_TIMESTAMPS: Mapping[str, str] = MappingProxyType({})


class AuthorizedScope(NamedTuple):
    """What a principal may search, at BOTH granularities the query path needs.

    ``contexts`` is the coarse ``(cell_principal, collection_id)`` set. It decides
    which encrypted cells / posting lists are opened at all — it is a *key custody*
    question, and it is necessarily collection-shaped because cell and SSE keys are
    derived per ``(principal, collection)``.

    ``artifact_ids`` is the fine set the light cone actually resolved: direct grants
    plus everything reachable through action-permitted containment and context edges.
    An artifact-scoped grant names one id here while widening ``contexts`` to that
    id's whole collection, because there is no narrower key. Keeping both is what
    stops the second fact from being thrown away — see
    :meth:`~mantle.search.mantle.sse.router_accessor.MantleSseSearchAccessor.search`
    for why re-applying it downstream is a meet and not a bolted-on ACL filter.

    ``updated_at`` is each surviving artifact's ``modified_time``, read in the loop below
    from the doc that loop is already holding for its ``collection_id`` — the same argument
    that puts ``artifact_predicate`` there rather than after the fact: it costs no store read.
    It is what lets a recall that narrowed to a set and has nothing to RANK that set with
    return it most-recently-updated first instead of in whatever order a set iterates. It is
    keyed by exactly the ids in ``artifact_ids`` and no others, so a consumer that walks it
    instead of them cannot widen; an id whose doc could not be read is simply absent, which
    sorts it last rather than dropping it.
    """

    contexts: List[Tuple[str, str]]
    artifact_ids: FrozenSet[str]
    updated_at: Mapping[str, str] = _NO_TIMESTAMPS


#: A blind-token narrowing, expressed as a function of the CONTEXTS rather than of a doc.
#:
#: This is why token narrowing cannot be a second ``artifact_predicate``. A field filter is a
#: function of one doc, and the loop below is already holding that doc — so it composes inside
#: the loop for free. A token set is a function of the owner SSE key, which is a function of the
#: ``(cell_principal, collection_id)`` pairs, which are what the loop PRODUCES. The dependency
#: runs the other way, so the meet has to happen after the loop, and it happens HERE rather than
#: at a call site so that ``ids ⊆ authorized`` stays true by construction: there is one place the
#: two sets meet and it is an intersection.
#:
#: :class:`~mantle.search.mantle.sse.narrowing.TokenNarrower` compiles a query into one.
TokenLookup = Callable[[Sequence[Tuple[str, str]]], AbstractSet[str]]


def _token_narrowing(
    token_lookup: TokenLookup, contexts: Sequence[Tuple[str, str]],
) -> FrozenSet[str]:
    """Run the lookup over the resolved contexts. Fails CLOSED, to the empty set.

    A lookup that raises has proved nothing about any artifact, and a narrowing reports what it
    could prove — the same reading ``artifact_predicate`` gets for a doc it cannot evaluate. The
    result is intersected by the caller, so an empty answer removes everything and can never add
    anything; a swallowed failure therefore costs recall and never authority.
    """
    if not contexts:
        return frozenset()
    try:
        found = token_lookup(contexts)
    except Exception:  # noqa: BLE001 — a token lookup reads a store; those raise broadly
        logger.warning("token narrowing failed; narrowing to nothing", exc_info=True)
        return frozenset()
    return frozenset(str(a) for a in (found or ()))


def authorized_page(store_db, lightcone, principal_id: str, *, action: str = "read",
                    offset: int = 0, limit: int = 100,
                    principal_type: str = "user") -> List[str]:
    """Ids this principal may `action`, in id order, WITHOUT materialising the whole light cone.

    The enumerating route asks "what may this principal reach", builds the answer, sorts it and
    takes a slice. Measured 2026-08-25 on 71/home, `list_origin_descendants` walking DOWN:

        collection:concepts-consolidated       5,484 descendants     0.2s
        stage.2.world                        121,682 descendants     4.1s
        stage.1.grammar                      188,321 descendants     7.0s
        stage.0.lexicon                      RAISES EdgesTruncated (>1,000,000)   34-51s

    ~27 microseconds per edge, so 1.85M would be ~50s even with the cap lifted. And it does not
    fail locally: the raise happens inside `resolve_authorized_scope`, so a principal granted on
    the lexicon loses the enumerating route for EVERY collection it holds. Consequence measured:
    only 12,494 vertices (0.57%) were reachable by any principal whose cone could be computed.

    This inverts it the way the NARROWED route already works — walk UP per candidate with
    `origin_chain`, which is `O(candidates x depth)` and never builds the set. Each id is checked
    by exactly the walk `_reaches` uses, so this can only return what the enumerating route would
    have returned; it just never has to hold it.

    The honest cost, stated because it is the opposite trade: this SCANS ids in order and checks
    each, so it is fast when the caller's reach is dense (the first page fills almost immediately)
    and slow when it is sparse (a principal who may see ten artifacts in a 2.17M-row store walks a
    long way to find them). That is why it is a FALLBACK and not the default — the enumerating
    route is better for a small cone, and a small cone is the one it can still compute.

    Ordering is by id, which is what the caller sliced a sorted set by, so `offset` indexes into
    the same sequence it always did.
    """
    allow, deny = _grant_sets(lightcone, principal_id, principal_type, action)
    if not allow:
        return []
    try:
        conn = store_db.artifacts.db.read()
    except Exception:  # noqa: BLE001 — no SQL handle: this route cannot run
        return []
    out: List[str] = []
    passed = 0
    try:
        rows = conn.execute(
            "SELECT id, json_extract(doc,'$.root_id') FROM vertex ORDER BY id")
    except Exception:  # noqa: BLE001
        return []
    for artifact_id, root_id in rows:
        aid = str(artifact_id)
        if not _reaches(store_db, aid, action, allow, deny, root_id=str(root_id or aid)):
            continue
        passed += 1
        if passed <= offset:
            continue
        out.append(aid)
        if len(out) >= limit:
            break
    return out


def _grant_sets(lightcone, principal_id: str, principal_type: str, action: str):
    """`(allow, deny)` — the resources this principal's grants allow or deny for `action`."""
    from mantle.entities.grant import mask_of

    allow, deny = set(), set()
    for g in (lightcone._grants_for(principal_id, principal_type) or []):   # noqa: SLF001
        rid = getattr(g, "resource_id", None)
        if not rid:
            continue
        m = mask_of(g)
        if m.is_deny and m.carries(action):
            deny.add(str(rid))
        elif m.allows(action):
            allow.add(str(rid))
    return allow, deny


def _reaches(store_db, artifact_id: str, action: str, allow: set, deny: set,
             root_id: Optional[str] = None) -> bool:
    """Does a grant sit on this artifact, or above it on an edge that still conducts `action`?

    The same walk `check_access` and `oracle.LightConeGrantVerifier` read — `origin_chain` yields
    the artifact, its root, then each origin ancestor, stopping at the first edge whose propagate
    mask does not carry the action. Deny is checked at every level before allow, so nothing further
    up re-allows what a nearer deny refused.
    """
    from mantle.db.backend import OriginChainUnterminated, origin_chain

    try:
        for resource in origin_chain(store_db, artifact_id, action, root_id=root_id):
            if resource in deny:
                return False
            if resource in allow:
                return True
    except OriginChainUnterminated:
        return False
    except Exception:  # noqa: BLE001 — an unreadable chain under-reaches, which is fail-closed
        return False
    return False


def _custody_contexts(store_db, lightcone, principal_id: str, action: str,
                      principal_type: str) -> list:
    """The `(cell_principal, collection)` pairs this principal may hold keys for.

    Derived from the COLLECTIONS, not from every artifact the principal can reach. Those are two
    routes to one answer and only one of them scales: a collection with more members than
    `edges_of` will return makes the artifact route raise `EdgesTruncated`, and this store has
    1,841,335 members in `stage.0.lexicon` against 68 collections in total. `vertex(ct)` is
    indexed, so finding the 68 is a keyed read.

    Two sources, because a grant can confer custody two ways:

      * a collection the principal REACHES — by a grant on it or on an ancestor that still
        conducts the action;
      * the collection of an artifact granted DIRECTLY. An artifact-scoped grant widens custody to
        that artifact's whole collection, because keys are derived per collection and there is no
        narrower one. That is the existing contract, not a new allowance.
    """
    from mantle.db.constants import COLLECTION_CONTENT_TYPE

    from .principal import resolve_cell_principal

    allow, deny = _grant_sets(lightcone, principal_id, principal_type, action)
    if not allow:
        return []

    pairs, seen = set(), set()

    def _add(collection_id: str) -> None:
        if not collection_id or collection_id in seen:
            return
        seen.add(collection_id)
        try:
            cell = resolve_cell_principal(store_db, collection_id)
        except Exception:  # noqa: BLE001 — a collection whose root cannot be resolved confers no
            return         # custody; substituting the collection id would derive the wrong key
        if cell:
            pairs.add((cell, collection_id))

    try:
        collections = [str(a.get("id")) for a in
                       store_db.artifacts.list_artifacts(content_type=COLLECTION_CONTENT_TYPE)
                       if a.get("id")]
    except Exception:  # noqa: BLE001 — no collection listing is an empty custody answer
        collections = []
    for collection_id in collections:
        if _reaches(store_db, collection_id, action, allow, deny):
            _add(collection_id)

    for resource in allow:
        if resource in deny:
            continue
        try:
            doc = _raw_artifact(store_db, resource)
        except Exception:  # noqa: BLE001
            doc = None
        if doc:
            _add(str(doc.get("collection_id") or resource))

    return sorted(pairs)


def resolve_authorized_scope(
    store_db,
    principal_id: str,
    *,
    lightcone: LightConeResolver,
    action: str = "read",
    principal_type: str = "user",
    artifact_predicate: Optional[Callable[[dict], bool]] = None,
    token_lookup: Optional[TokenLookup] = None,
) -> AuthorizedScope:
    """Resolve the light cone into both search granularities.

    The light-cone resolver returns a flat set of artifact ids the
    requesting principal can ``read``. Each authorized artifact's
    ``collection_id`` is the MANTLE / SSE search scope; its **cell-key
    principal** is the collection's immutable origin root, not the
    artifact's ``created_by`` — the exact same value the index path used,
    so the derived keys match. We dedupe ``(cell_principal, collection)``.

    The artifact ids are returned ALONGSIDE the pairs rather than being consumed
    and discarded. Collapsing ``{art-1}`` into ``(bob, col-bob)`` and keeping only
    the pair is an escalation: a grant on one artifact becomes recall over its
    entire collection, because ``col-bob`` is all either engine is ever told. The
    two facts are not redundant and only one of them is expressible as a key.

    ``artifact_ids`` is the resolver's set verbatim, not just the ids that mapped
    to a pair. A lattice read that fails for one id must not silently shrink the
    authorized set — it already costs that id its collection pair, which is the
    fail-closed half; narrowing the meet on the same evidence would drop results
    the principal is genuinely entitled to.

    ``artifact_predicate`` is the caller's ``field:value`` filter, compiled by
    ``search.field_filters.compile_filters``. It narrows ``artifact_ids`` ONLY, and only ever
    downward: it is evaluated against docs this function was already reading — every authorized
    id's doc is fetched here anyway, for its ``collection_id`` — so it adds no store read, and it
    is only ever shown docs of artifacts the light cone already authorized. The filtered set is
    therefore a SUBSET by construction rather than by discipline: there is no path by which a
    filter admits an id ``resolve`` did not return. ``None`` leaves the result byte-identical to
    an unfiltered resolve, which is what every non-query caller gets.

    ``contexts`` is deliberately NOT narrowed by the predicate. Those pairs are a key-custody
    answer — which collections this principal may hold keys for — and a caller's filter is not
    evidence about custody. The query path short-circuits on an empty filtered set before it
    opens anything, so nothing is decrypted for a filter that matched nothing.

    A doc that cannot be read is EXCLUDED when a predicate is supplied, where it is kept when
    one is not. The two are not in tension: keeping it is an authorization decision (the
    principal is entitled to it and a store hiccup must not revoke that), while dropping it is
    a filter decision (an unread doc cannot be shown to satisfy a predicate). A filter reports
    what it could prove, and both readings only ever narrow.

    ``token_lookup`` is the blind-token narrowing — see :data:`TokenLookup` for why it is a
    callback over the CONTEXTS and not a second doc predicate. It runs in a SECOND PHASE, after
    the loop below has produced the pairs it needs, and its answer is INTERSECTED into
    ``artifact_ids``. Intersection is the whole security argument: a token match can only remove
    ids, so a token naming an artifact outside the light cone lands in the same place as a token
    matching nothing — the empty set, with no observable between them. That makes the narrowing
    unusable as an existence oracle for content the requester cannot read. ``None`` leaves the
    result byte-identical to a resolve with no narrowing at all.

    ``contexts`` is NOT narrowed by it, for the same reason the field filter does not narrow
    them: the pairs are a key-custody answer, and what a caller searched for is not evidence
    about custody. The lookup is in fact handed those pairs, so it reads only indexes this
    principal already holds keys for.

    ``principal_type`` is the requester's acting-context entity kind. It defaults to
    ``"user"`` because the query call sites are user searches; the SYSTEM principal
    reaches this through the oracle's grant verifier with ``"service"``, and
    :func:`ledger_grantee_type` maps both onto the ledger's grantee vocabulary.

    Returns an empty scope when the principal has no authorized artifacts
    or when the lattice lookups fail. Empty result is safe — both engines
    return no hits for empty contexts.
    """
    from .principal import resolve_cell_principal

    # ── the narrowed route, when there is a narrowing to run ─────────────────────────────────────
    # Enumerating asks "what may this principal reach", materialises it, and MEETS the token
    # lookup's answer into it. On a corpus that does not survive: `stage.0.lexicon` holds 1,841,335
    # members, so `list_origin_descendants` raises at the 1,000,000-edge cap and a recall by anyone
    # granted there fails — including for every OTHER collection they hold.
    #
    # The same set is `matched` filtered by authorized rather than `authorized` filtered by
    # matched, and the second form needs no set: the lookup already produced the candidates, and
    # each is checked with the walk `check_access` uses. `O(candidates x depth)` instead of
    # `O(everything authorized)`, and the meet's contract is unchanged — a token naming an artifact
    # outside the light cone is dropped by the per-candidate check exactly as intersection dropped
    # it, so the narrowing is still no existence oracle.
    #
    # `contexts` comes from the COLLECTIONS rather than from the artifacts, because it is needed
    # BEFORE any candidate exists — the lookup is keyed per `(principal, collection)`. See
    # `_custody_contexts`.
    #
    # Without a `token_lookup` there is nothing to filter and the enumerating route below still
    # runs. That path answers "everything I may see", which has no smaller form.
    if token_lookup is not None:
        contexts = _custody_contexts(store_db, lightcone, principal_id, action, principal_type)
        if not contexts:
            return AuthorizedScope([], frozenset())
        allow, deny = _grant_sets(lightcone, principal_id, principal_type, action)
        matched = _token_narrowing(token_lookup, contexts)

        kept: set = set()
        stamps: Dict[str, str] = {}
        # ── the chain above a collection is walked once, not once per artifact ──────────────
        # `origin_chain` yields `artifact_id`, then `root_id`, then each origin ancestor, stopping
        # at the first edge whose propagate mask does not carry the action. Two measurements make
        # the tail shareable:
        #
        #   * `root_id` is the artifact's version root rather than its collection, and it equals
        #     the artifact's own id for 5,000 of 5,000 sampled synsets and 3,000 of 3,000 canon
        #     rows. Keying a cache on it would give one entry per artifact and never hit.
        #   * The first true ancestor is the collection, and a narrowed set spans very few:
        #     600 sampled artifacts had 2 distinct origin parents.
        #
        # So the walk is consumed lazily and handed off at the first resource that is neither the
        # artifact nor its version root. Everything above that point is a property of the ancestor,
        # not of the artifact, and is memoised on it. The generator keeps its own mask logic — this
        # does not re-implement `_propagates`, it just stops reading.
        #
        # Scoped to this call, so a cached authorization answer cannot outlive the `allow`/`deny`
        # sets it is an answer for.
        _above: Dict[str, bool] = {}

        def _authorized(artifact_id: str, root_id: str) -> bool:
            """`_reaches`, with the shared tail memoised on the first ancestor."""
            from mantle.db.backend import OriginChainUnterminated, origin_chain

            try:
                for resource in origin_chain(store_db, artifact_id, action, root_id=root_id):
                    if resource in deny:
                        return False
                    if resource in allow:
                        return True
                    if resource != artifact_id and resource != root_id:
                        if resource not in _above:
                            _above[resource] = _reaches(
                                store_db, resource, action, allow, deny, root_id=resource)
                        return _above[resource]
            except OriginChainUnterminated:
                return False
            except Exception:  # noqa: BLE001 — an unreadable chain under-reaches, fail-closed
                return False
            return False

        # One batched read for the whole narrowed set rather than one per candidate — see
        # `_raw_artifacts`. The loop below is unchanged in what it decides; only where the doc
        # comes from has moved.
        docs = _raw_artifacts(store_db, matched)
        for artifact_id in matched:
            doc = docs.get(str(artifact_id))
            if not doc:
                # An id the lookup named and the store cannot show. It is not authorized here:
                # the walk needs the doc's root, and admitting it unchecked would let the
                # narrowing add an id the light cone never authorized.
                continue
            if not _authorized(str(artifact_id),
                                str(doc.get("root_id") or artifact_id)):
                continue
            if artifact_predicate is not None:
                try:
                    if not artifact_predicate(doc):
                        continue
                except Exception:  # noqa: BLE001 — a malformed doc must not fail the search
                    continue
            kept.add(str(artifact_id))
            modified = doc.get("modified_time")
            if modified:
                stamps[str(artifact_id)] = str(modified)
        return AuthorizedScope(contexts, frozenset(kept), stamps)

    authorized = lightcone.resolve(
        principal_id, action=action, principal_type=principal_type
    )
    if not authorized:
        return AuthorizedScope([], frozenset())

    pairs: set[Tuple[str, str]] = set()
    principal_by_collection: dict[str, str] = {}
    # `None` means "no filter" and keeps `artifact_ids` the resolver's set verbatim. A set means
    # a filter is running, and it is filled only from ids `authorized` already contains.
    matched: Optional[set[str]] = None if artifact_predicate is None else set()
    # Read off the same doc the pair below is read from. `field_filters` resolves the
    # `updated_at` filter from `modified_time` on this doc, so recall ORDERS by the field it
    # already FILTERS by, rather than by a second notion of when something changed.
    timestamps: Dict[str, str] = {}
    for artifact_id in authorized:
        try:
            doc = _raw_artifact(store_db, artifact_id)
        except Exception:  # noqa: BLE001 — store reads can raise broadly
            continue
        if not doc:
            continue
        modified = doc.get("modified_time")
        if modified:
            timestamps[str(artifact_id)] = str(modified)
        if matched is not None:
            try:
                keep = artifact_predicate(doc)   # type: ignore[misc]
            except Exception:  # noqa: BLE001 — a malformed doc must not fail the search
                keep = False
            if keep:
                matched.add(str(artifact_id))
        # A root artifact (no parent collection) self-references its own id.
        # `_key` is the legacy doc-id shape, `id` the lattice shape — read both.
        collection_id = doc.get("collection_id") or doc.get("_key") or doc.get("id")
        if not collection_id:
            continue
        collection_id = str(collection_id)
        cell_principal = principal_by_collection.get(collection_id)
        if cell_principal is None:
            cell_principal = resolve_cell_principal(store_db, collection_id)
            principal_by_collection[collection_id] = cell_principal
        if not cell_principal:
            continue
        pairs.add((cell_principal, collection_id))

    ids = (
        frozenset(str(a) for a in authorized) if matched is None else frozenset(matched)
    )
    contexts = sorted(pairs)

    # ---- Second phase: the blind-token meet ---------------------------------------------
    #
    # It waits for the loop because it consumes the loop's output. `contexts` is what a
    # narrower needs to derive owner SSE keys, and it does not exist until every authorized
    # doc has been read for its collection — so this is not a third predicate that could have
    # ridden along above, it is a stage that could not have.
    #
    # `&=` is the entire contract. Whatever the lookup returns — ids from another principal's
    # collection, ids the light cone never authorized, ids that do not exist — meets a set that
    # is already ⊆ `authorized`, so the result is too. The narrowing chooses which authorized
    # ids survive; it has no vocabulary for adding one.
    if token_lookup is not None:
        ids &= _token_narrowing(token_lookup, contexts)

    # Trimmed to the surviving ids LAST, after every narrowing has run. A timestamp for an id
    # a filter or a token removed would be a second, wider set travelling beside the answer,
    # and the first consumer to iterate the wrong one would undo the narrowing.
    return AuthorizedScope(
        contexts, ids, {a: t for a, t in timestamps.items() if a in ids},
    )


def resolve_authorized_contexts(
    store_db,
    principal_id: str,
    *,
    lightcone: LightConeResolver,
    action: str = "read",
    principal_type: str = "user",
) -> List[Tuple[str, str]]:
    """The coarse half of :func:`resolve_authorized_scope`, for key-custody callers.

    ``oracle.LightConeGrantVerifier`` answers "may this requester hold the key for
    ``(principal, collection)``", a question that has no artifact-granular form —
    keys are derived per collection. It therefore wants exactly these pairs. The
    query path must NOT use this function: dropping ``artifact_ids`` there is the
    escalation, not a simplification.
    """
    return resolve_authorized_scope(
        store_db, principal_id, lightcone=lightcone,
        action=action, principal_type=principal_type,
    ).contexts
