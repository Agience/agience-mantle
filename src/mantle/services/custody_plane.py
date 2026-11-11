"""Only isolated production claims threshold custody (John, 2026-11-02).

Custody splits each key between this node, Origin and an offline escrow so that no single host
holds it. That holds only where the services are isolated from one another: on the home peer or a
developer's machine one user runs everything and reads every folder, so custody there would claim
a separation that does not exist. So it is refused there, and wherever isolation is absent. The home node, while it is the test
gate, holds test custody instead (`home_test_problems`): its own keys, an issuer production never
trusts, and none once it peers.

The signal is two things together, because either alone is a claim anyone can write:

* the plane the release tool renders into each production service's env,
  ``AGIENCE_PLANE=production``, which the home peer and a developer's machine never carry;
* evidence read at start that this process runs isolated as the release sets it up: it is not
  root, the kernel reports ``NoNewPrivs`` set (systemd's ``NoNewPrivileges=yes``; where the
  kernel reports nothing it is not evidence), and its keys folder and the folder holding its store
  are owned by its own user with no access for anyone else (0700), each read as itself, never
  through a link.

``MANTLE_CUSTODY_PLANE_OVERRIDE=1`` passes the check only inside a test run (pytest imported), so
tests can exercise custody; a deployment cannot set its way past it.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Tuple


class CustodyRefused(RuntimeError):
    pass


def _euid() -> Optional[int]:
    return os.geteuid() if hasattr(os, "geteuid") else None


def _no_new_privs() -> Optional[bool]:
    """True/False as the kernel reports `NoNewPrivs`, or None where it reports nothing."""
    try:
        with open("/proc/self/status", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("NoNewPrivs:"):
                    return line.split(":", 1)[1].strip() == "1"
    except OSError:
        return None
    return None


def _owner_and_mode(path: str) -> Tuple[Optional[int], int]:
    """The folder's own owner and mode: a link is read as the link, never its target (review
    19)."""
    st = os.stat(path, follow_symlinks=False)
    return (getattr(st, "st_uid", None), st.st_mode & 0o777)


def _overridden() -> bool:
    return (os.getenv("MANTLE_CUSTODY_PLANE_OVERRIDE") == "1"
            and sys.modules.get("pytest") is not None)


def problems(keys_dir: Optional[str]) -> List[str]:
    """Why this node may not claim custody; empty when it may."""
    out: List[str] = []
    plane = (os.getenv("AGIENCE_PLANE") or "").strip().lower()
    if plane == "home":
        return home_test_problems(keys_dir)
    if plane != "production":
        out.append(f"the plane is {plane or 'unset'!r}, not 'production' (AGIENCE_PLANE): the "
                   f"home and development planes never claim custody")
    uid = _euid()
    if uid is None:
        out.append("this platform reports no user id, so isolation cannot be shown")
    elif uid == 0:
        out.append("the process runs as root, so it is not isolated")
    nnp = _no_new_privs()
    if nnp is not True:
        out.append("the kernel does not report NoNewPrivileges set for this process"
                   if nnp is False else "NoNewPrivileges cannot be shown for this process")
    if not keys_dir:
        out.append("no keys folder is configured (KEYS_DIR)")
    else:
        try:
            owner, mode = _owner_and_mode(keys_dir)
        except OSError as exc:
            out.append(f"the keys folder {keys_dir} cannot be read ({exc})")
        else:
            if uid is not None and owner != uid:
                out.append(f"the keys folder {keys_dir} is not owned by this process's user")
            if mode & 0o077:
                out.append(f"the keys folder {keys_dir} is open to others (mode {mode:o}, "
                           f"expected 0700)")
    store_dir = _store_dir()
    if store_dir:
        try:
            owner, mode = _owner_and_mode(store_dir)
        except OSError as exc:
            out.append(f"the store folder {store_dir} cannot be read ({exc})")
        else:
            if uid is not None and owner != uid:
                out.append(f"the store folder {store_dir} is not owned by this process's user")
            if mode & 0o077:
                out.append(f"the store folder {store_dir} is open to others (mode {mode:o}, "
                           f"expected 0700)")
    return out


#: The home node's own domain. Test custody is held only by a node whose issuer is under it.
HOME_DOMAIN = "home.agience.ai"


def home_test_problems(keys_dir: Optional[str]) -> List[str]:
    """Why the home node may not hold test custody; empty when it may (John, 2026-11-11).

    While the home node is the test gate it holds test custody over its own keys, so a run, a
    timer and their approvals are accepted there before promote. It claims no separation (one user
    runs every service there). What keeps it apart from production is an issuer under
    `home.agience.ai`, which production never trusts, and having no peers: it is given up once
    peering starts (`EMBER_PEERS`). Origin decides by the same rule (`home_test_refusal`)."""
    from urllib.parse import urlsplit

    out: List[str] = []
    # Every source of the issuer must name a home host, as Origin decides (review, 2026-11-11):
    # passing on one while another names production would let a node hold custody and sign as it.
    try:
        from mantle import config
        configured = getattr(config, "AUTHORITY_ISSUER", None)
    except Exception:  # noqa: BLE001
        configured = None
    sources = [os.getenv("AUTHORITY_ISSUER"), os.getenv("ORIGIN_URI"), configured]
    hosts = [(urlsplit(s.strip()).hostname or "").lower() for s in sources if s and s.strip()]
    if not hosts or not all(h == HOME_DOMAIN or h.endswith("." + HOME_DOMAIN) for h in hosts):
        out.append(f"test custody is held only by a home node, whose every issuer is under "
                   f"{HOME_DOMAIN}; this one's are {sorted(set(hosts)) or 'unset'}")
    if (os.getenv("EMBER_PEERS") or "").strip():
        out.append("test custody is given up once the home node peers (EMBER_PEERS is set)")
    # The keys folder the node is given, as Origin reads it: `config.KEYS_DIR` always has a
    # default, so it could never show that one was set.
    if not (os.getenv("KEYS_DIR") or "").strip():
        out.append("test custody needs this node's own keys folder (KEYS_DIR)")
    return out


def _store_dir() -> Optional[str]:
    """The folder holding this node's store (`MANTLE_LATTICE_PATH`, else the default), as
    Origin checks the folder holding its database (review 19)."""
    path = (os.getenv("MANTLE_LATTICE_PATH") or "").strip()
    if not path:
        try:
            from mantle import config
            path = str(getattr(config, "DEFAULT_LATTICE_PATH", "") or "")
        except Exception:  # noqa: BLE001
            path = ""
    return os.path.dirname(path) or None if path else None


def custody_refusal(keys_dir: Optional[str]) -> Optional[str]:
    """Why this node may not hold custody, or None when it may: the same shape as Origin's
    `custody_plane.custody_refusal` (review 19)."""
    if _overridden():
        return None
    found = problems(keys_dir)
    return ("threshold custody is claimed only by isolated production, or as test custody by the "
            "home node: " + "; ".join(found) if found else None)


def require_isolated_production(keys_dir: Optional[str]) -> None:
    """Refuse threshold custody unless this node is isolated production."""
    why = custody_refusal(keys_dir)
    if why:
        raise CustodyRefused(why)


def keys_dir() -> Optional[str]:
    from mantle import config
    return str(getattr(config, "KEYS_DIR", "") or os.getenv("KEYS_DIR") or "") or None
