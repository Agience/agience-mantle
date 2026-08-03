"""BROADCAST CARRIERS — the mesh over a medium that cannot be listed (John, 2026-07-23).

The mesh consumes from a plane it can `list`/`get` (S3, MinIO, a dict). A radio cannot be listed:
you cannot ask the air what was said before you tuned in. So RF is not a plane — **it is a CARRIER
that FILLS a plane**:

    emitter node                    the medium                 receiver node
    publish_merkle   ──frames──▶   (air / wire / drive)  ──▶   spool into the LOCAL plane
      (tree + leaves)                                                   │
                                                          reconcile_via_s3 diffs the tree
                                                          and applies the differing leaves

Two halves, and the split is the whole point:
  • `emit(frames)`  — hand bytes to the medium. Fire and forget. No address, no ACK, no reply.
  • `spool(frames)` — whatever arrived goes into the local plane. The receiver then runs its NORMAL
    reconcile; the mesh above never learns how the bytes got there.

**Nothing here may request, address, or await.** A carrier that asks a peer for something would be a
pipeline (forbidden — [[signals-propagate]]); a carrier only radiates and receives. That is why a
broadcast medium is the PUREST form of this system's communication: the signal goes out, and nearby
fitting contexts (peers holding the fleet key) activate. Peers without the key hear noise — the tree
and leaf objects are already fleet-key encrypted, so an open medium is exactly as safe as an
untrusted S3 shelf.

Stage 1 (this file): the carrier contract + a `SpoolPlane` (a directory that IS a mesh plane) + a
`LoopbackCarrier` (in-process medium) — pure software, no hardware, so the shim is proven before a
radio is attached. Stage 2 attaches a real SDR by implementing `emit`/`receive` over the air; the
mesh above changes NOTHING.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

#: One unit handed to the medium: (key, bytes). The key is the segment's mesh path, carried with the
#: payload so a receiver can spool it back to the same place — the medium stays dumb.
Frame = Tuple[str, bytes]


class SpoolPlane:
    """A directory that IS a mesh plane. Presents exactly the surface the mesh S3 client uses
    (`put`/`get`/`exists` + a boto-shaped `_s3` for listing), backed by files under `root`.

    This is what a radio fills: frames received off the air are spooled here, and the node's normal
    `reconcile_via_s3` reads them as if they had come from S3. The mesh cannot tell the difference,
    which is the point — the transport is swappable because the plane contract is three verbs."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.bucket = "spool"
        self._s3 = self._Boto(self.root)

    # ── the plane surface (Garage face) ───────────────────────────────────────────────────────
    def _path(self, key: str) -> Path:
        return self.root / key.replace("/", os.sep)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes(data))

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        return p.read_bytes() if p.is_file() else None

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def keys(self) -> List[str]:
        out: List[str] = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(self.root)).replace(os.sep, "/"))
        return out

    # ── the listing surface (boto face) — faithful Prefix/Delimiter/StartAfter semantics ───────
    class _Boto:
        def __init__(self, root: Path) -> None:
            self._root = root

        def _all(self) -> List[str]:
            out = []
            for p in sorted(self._root.rglob("*")):
                if p.is_file():
                    out.append(str(p.relative_to(self._root)).replace(os.sep, "/"))
            return out

        def get_paginator(self, _op):
            outer = self

            class _P:
                def paginate(self, Bucket=None, Prefix="", Delimiter=None, StartAfter=None, **_k):
                    matched = [k for k in outer._all() if k.startswith(Prefix)]
                    if StartAfter:
                        matched = [k for k in matched if k > StartAfter]
                    if not Delimiter:
                        yield {"Contents": [{"Key": k} for k in matched]}
                        return
                    commons, contents = set(), []
                    for k in matched:
                        rest = k[len(Prefix):]
                        i = rest.find(Delimiter)
                        if i >= 0:
                            commons.add(Prefix + rest[: i + len(Delimiter)])
                        else:
                            contents.append({"Key": k})
                    yield {"Contents": contents,
                           "CommonPrefixes": [{"Prefix": p} for p in sorted(commons)]}
            return _P()

        def get_object(self, Bucket=None, Key=None, **_k):
            data = (self._root / Key.replace("/", os.sep)).read_bytes()

            class _B:
                def read(self_inner):
                    return data
            return {"Body": _B()}


class Carrier:
    """The carrier contract. TWO verbs, and neither may address a peer or await a reply.

    `emit(frames)`   — hand bytes to the medium (broadcast). Returns how many frames left.
    `receive()`      — yield whatever the medium delivered since last call. May be empty; never blocks
                       on a specific peer, never requests.

    A subclass implementing these over an SDR turns the mesh into an RF mesh with no change above."""

    def emit(self, frames: Iterable[Frame]) -> int:                      # pragma: no cover - contract
        raise NotImplementedError

    def receive(self) -> List[Frame]:                                    # pragma: no cover - contract
        raise NotImplementedError


class LoopbackCarrier(Carrier):
    """An in-process medium: what is emitted is later received, by everyone listening. Models a
    broadcast channel with no addressing — the same shape a radio has, without hardware. Frames are
    delivered ONCE per receiver (each receiver has its own read offset), so two listeners both hear
    the same transmission — which is what broadcast means."""

    def __init__(self) -> None:
        self._air: List[Frame] = []
        self._offsets: Dict[int, int] = {}

    def emit(self, frames: Iterable[Frame]) -> int:
        n = 0
        for key, data in frames:
            self._air.append((key, bytes(data)))
            n += 1
        return n

    def receive(self, listener: int = 0) -> List[Frame]:
        off = self._offsets.get(listener, 0)
        out = self._air[off:]
        self._offsets[listener] = len(self._air)
        return list(out)


def frames_from_plane(plane, prefix: str = "mesh/", since: Optional[set] = None) -> List[Frame]:
    """Read out what a node has published, as frames ready to hand to a carrier. `since` is the set
    of keys already emitted (so a repeat transmission carries only what is new) — the emitter's own
    bookkeeping, never a peer's."""
    seen = since if since is not None else set()
    frames: List[Frame] = []
    for k in (plane.keys() if hasattr(plane, "keys") else []):
        if k.startswith(prefix) and k not in seen:
            data = plane.get(k)
            if data is not None:
                frames.append((k, data))
    return frames


def spool_frames(plane, frames: Iterable[Frame]) -> int:
    """Whatever arrived off the medium goes into the local plane, at the key it carried. The node's
    normal consume then applies it. A frame that arrives twice simply overwrites itself — the mesh's
    cursor discipline (never advance past a segment that did not apply) makes redelivery harmless."""
    n = 0
    for key, data in frames:
        plane.put(key, data)
        n += 1
    return n


def broadcast(plane, carrier: Carrier, *, prefix: str = "mesh/",
              emitted: Optional[set] = None) -> Dict[str, int]:
    """One emit pass: everything new this node has published goes to the medium. Fire and forget."""
    seen = emitted if emitted is not None else set()
    frames = frames_from_plane(plane, prefix=prefix, since=seen)
    sent = carrier.emit(frames)
    for k, _ in frames:
        seen.add(k)
    return {"frames": sent, "keys": len(seen)}


def absorb(plane, carrier: Carrier, *, listener: int = 0) -> Dict[str, int]:
    """One receive pass: whatever the medium delivered lands in the local plane. The caller then runs
    its ordinary `reconcile_via_s3(store)` — the mesh never learns a radio was involved."""
    try:
        frames = carrier.receive(listener=listener)      # type: ignore[call-arg]
    except TypeError:
        frames = carrier.receive()
    return {"frames": spool_frames(plane, frames)}


__all__ = ["Frame", "SpoolPlane", "Carrier", "LoopbackCarrier",
           "frames_from_plane", "spool_frames", "broadcast", "absorb"]
