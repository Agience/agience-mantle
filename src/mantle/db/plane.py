"""The lattice's backing for the communication plane's two pluggable contracts.

`beam/plane.py` declares `Lightcone` (who reaches which artifact) and `Keyring` (the per-group key) as
contracts, and ships in-memory models of both for tests. In production those contracts are backed by the
lattice's own machinery, which is what lives here:

  - `LatticeLightcone(store)` — `reaches(principal)` = the principal's read light-cone (active readable
                               CRUDEASIO grants + containment descendants, computed straight off the
                               lattice store via `access.reachable_collections` — no HTTP), plus the
                               principal's own address (a direct reach to itself), plus any ground
                               collections a reactor has `join`ed for this session.
                               Comms delivery is read-access — one mechanism, not a second sharing path.
  - `LatticeKeyring(root_secret)` — per-group AES-256 keys from `content_cache.collection_key(root, group)
                               = HKDF(root, origin_root)`. Deterministic: every member derives the same
                               key; a non-member never reaches the group, so never derives it.
                               `principal_keys` gates by the light-cone.

    Reactor(principal, keyring=LatticeKeyring(root_secret), lightcone=LatticeLightcone(store), ...)

The classes are duck-typed against the beam contracts rather than subclassing them, since subclassing
would require the import this module avoids.
"""
from __future__ import annotations

import base64
from typing import Any, Callable, Iterable, List, Optional, Set

__all__ = ["LatticeLightcone", "LatticeKeyring"]


class LatticeLightcone:
    """The `Lightcone` contract, backed by `access.reachable_collections`.

    `reaches(principal)` = the principal's read light-cone (active readable grants + containment
    descendants — computed straight off the lattice store, no HTTP), plus the principal's own address (a
    direct reach to itself), plus any ground collections a `Reactor` has `join`ed for this session.
    Grounding a persona onto a plane is a connection property (see `beam/reach.py`), so `join` is an
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
        from mantle.db.access import reachable_collections   # lazy: a store-less caller stays valid
        return reachable_collections(self._store, principal)

    def reaches(self, principal: str) -> Set[str]:
        return (set(self._reachable(principal)) | {principal}       # grants/containment + own address
                | self._joined.get(principal, set()))               # + session ground connections

    def join(self, principal: str, group: str) -> "LatticeLightcone":
        """Connect `principal` to a ground plane for this session (how a reactor grounds — reaching the
        ground collection is the right to its key). An in-memory overlay, not a durable grant."""
        self._joined.setdefault(principal, set()).add(group)
        return self


class LatticeKeyring:
    """The `Keyring` contract, backed by the fleet's `collection_key` derivation.

    Per-group AES-256 keys from `content_cache.collection_key(root_secret, group) = HKDF(root, origin_root)`.
    Deterministic: every member derives the same key; a non-member never reaches the group, so never derives
    it. `principal_keys` gates by the light-cone (a principal holds keys only for the groups it reaches);
    see this module's header for why the comms plane keeps per-group keying."""

    def __init__(self, root_secret: bytes) -> None:
        self._root = root_secret
        self._cache: dict = {}

    def group_key(self, group: str) -> bytes:
        """The raw 32 bytes of the derivation — an AES-256 key.

        The plane seals with AES-256-GCM (`beam.plane`), which takes the key material directly; a
        base64-encoded string is not a valid AES key length, so this returns the raw bytes.
        """
        if group not in self._cache:
            from mantle.db.content_cache import collection_key   # the fleet's own derivation
            self._cache[group] = collection_key(self._root, group)
        return self._cache[group]

    def principal_keys(self, principal: str, lightcone: Any) -> List[bytes]:
        return [self.group_key(g) for g in lightcone.reaches(principal)]
