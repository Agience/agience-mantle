"""The LATTICE'S BACKING for the communication plane's two pluggable contracts.

`beam/plane.py` declares `Lightcone` (who reaches which artifact) and `Keyring` (the per-group key) as
CONTRACTS, and ships in-memory models of both for tests. In production those contracts are backed by the
lattice's own machinery, which is what lives here:

  - `LatticeLightcone(store)` — `reaches(principal)` = the principal's READ light-cone (active readable
                               CRUDEASIO grants + containment descendants, computed straight off the
                               lattice store via `access.reachable_collections` — no HTTP), PLUS the
                               principal's own address (a direct reach to itself), PLUS any GROUND
                               collections a reactor has `join`ed for this session.
                               **Comms delivery IS read-access** — one mechanism, not a second sharing path.
  - `LatticeKeyring(root_secret)` — per-GROUP Fernet keys from `content_cache.collection_key(root, group)
                               = HKDF(root, origin_root)`. Deterministic: every member derives the same
                               key; a non-member never reaches the group, so never derives it.
                               `principal_keys` gates by the light-cone.

⚠ MOVED HERE FROM `ember/runtime/reach.py` (where they were `EmberLightcone`/`EmberKeyring`) — 2026-08-02,
the chorus→ember DAG work. **Nothing about the behaviour changed; only the address did.** The move is not a
reshuffle, it is a correction of a misfiling that was forcing a DAG violation:

  * These classes NEVER depended on ember. Measured at the move, their only imports were
    `access.reachable_collections` and `content_cache.collection_key` — both of them mantle's, both of them
    right here in this package. They were mantle code that happened to sit in the runner's tree.
  * Because they sat in ember, every chorus caller that needed the REAL grant-backed plane
    (`iris/comms/wiring.py`, `lumen/reach_provider.py`, `sage/reach_provider.py`) had to import ember to get
    them — and `chorus → ember` is the edge the target DAG forbids. The personas were importing the RUNNER to
    reach the STORE's own grant model.
  * `content_cache.collection_key`'s own docstring (:191) already cited `EmberKeyring.group_key` as the
    caller keeping it alive. That citation now points into this module, one package away from the function
    it names, instead of across two repos.

⛔ THIS MODULE MUST NOT IMPORT BEAM, and that is why the ASSEMBLER did not come with it. `beam.reach.Reactor`
is what wires these two backings into a working reactor, but mantle is beam's SIBLING in the DAG (mantle
reaches only origin), so a `mantle → beam` import would trade one violation for another. Assembly therefore
stays with each caller, which is what the persona providers already did by hand:

    Reactor(principal, keyring=LatticeKeyring(root_secret), lightcone=LatticeLightcone(store), ...)

The classes are duck-typed against the beam contracts rather than subclassing them — deliberately, since
subclassing is what would have required the import this module refuses.

⛔ THE COMMS PLANE'S PER-GROUP KEYING MUST NOT FOLLOW CONTENT-AT-REST. Content at rest moved to ONE
node-wide key (`shared_content_key`) because per-collection keying contradicted global content addressing —
one immutable object at one address had N keys, and storing a shared ref destroyed another root's copy
(mantle §1 / EREA, 2026-07-28). The comms plane is NOT affected: its per-group keying is real isolation — a
principal derives a group key only if its light-cone reaches that group — and no content-addressing
constraint forces one key here. `collection_key` is retained partly FOR this caller; do not "clean it up" on
the assumption that §1 removed its last user.
"""
from __future__ import annotations

import base64
from typing import Any, Callable, Iterable, List, Optional, Set

__all__ = ["LatticeLightcone", "LatticeKeyring"]


class LatticeLightcone:
    """The `Lightcone` contract, backed by `access.reachable_collections`.

    `reaches(principal)` = the principal's READ light-cone (active readable grants + containment
    descendants — computed straight off the lattice store, no HTTP), PLUS the principal's own address (a
    direct reach to itself), PLUS any GROUND collections a `Reactor` has `join`ed for this session.
    Grounding a persona onto a plane is a CONNECTION property (see `beam/reach.py`), so `join` is an
    in-memory session overlay, never a write into the durable store — durable grounding is a grant, minted
    by the write path, not a side effect of constructing a reactor.

    The `reach` fn is injectable so the adapter is exercisable without a live store; when omitted it lazily
    calls `access.reachable_collections` — the one, real, grant light-cone."""

    def __init__(self, store: Any, *, reach: Optional[Callable[[Any, str], Iterable[str]]] = None) -> None:
        self._store = store
        self._reach = reach
        self._joined: dict = {}                          # principal -> {session ground connections}

    def _reachable(self, principal: str) -> Iterable[str]:
        if self._reach is not None:
            return self._reach(self._store, principal)
        from mantle.db.lattice.access import reachable_collections   # lazy: a store-less caller stays valid
        return reachable_collections(self._store, principal)

    def reaches(self, principal: str) -> Set[str]:
        return (set(self._reachable(principal)) | {principal}       # grants/containment + own address
                | self._joined.get(principal, set()))               # + session ground connections

    def join(self, principal: str, group: str) -> "LatticeLightcone":
        """CONNECT `principal` to a ground plane for this session (how a reactor grounds — reaching the
        ground collection is the right to its key). An in-memory overlay, not a durable grant."""
        self._joined.setdefault(principal, set()).add(group)
        return self


class LatticeKeyring:
    """The `Keyring` contract, backed by the fleet's `collection_key` derivation.

    Per-GROUP Fernet keys from `content_cache.collection_key(root_secret, group) = HKDF(root, origin_root)`.
    Deterministic: every member derives the same key; a non-member never reaches the group, so never derives
    it. `principal_keys` gates by the light-cone (a principal holds keys only for the groups it reaches).

    ⚠ NOT "the same derivation that keys content at rest" — content moved to one node-wide key
    (`shared_content_key`); see this module's header for why the comms plane keeps per-group keying."""

    def __init__(self, root_secret: bytes) -> None:
        self._root = root_secret
        self._cache: dict = {}

    def group_key(self, group: str) -> bytes:
        if group not in self._cache:
            from mantle.db.lattice.content_cache import collection_key   # the fleet's own derivation
            self._cache[group] = base64.urlsafe_b64encode(collection_key(self._root, group))
        return self._cache[group]

    def principal_keys(self, principal: str, lightcone: Any) -> List[bytes]:
        return [self.group_key(g) for g in lightcone.reaches(principal)]
