"""`/git` — a node serves its own source, read-only.

The idea (`NODE-DEPLOYMENT-AND-FLEET.md` §4): each endpoint points to an install, and adding the
git to it — `mantle.home.agience.ai/git` — makes it a git of this node's own code: anyone can fork
it locally and run it.

## What is served, and what that scope means

A node serves the instruments, not the platform — `prism`, `crystal` and this repository. The
Apache-2.0 layer.

The set is narrowed to Agience components: a separately-branded repository is a branding decision
this surface should not be making on anyone's behalf. A project that wants its source served
peer-to-peer serves it from its own node — that is what this router is for.

That is a real boundary and it changes what this surface is for. It is not fork-and-run: the
instruments alone cannot stand a node up, so someone who clones from here still obtains mantle and
origin elsewhere. What it is instead is cleaner, and is what a dependency should be — the
permissive, shareable layer travelling peer-to-peer with no copyleft obligation following it.

The allowlist is the licence boundary, which is why it is named and not discovered. Serving
"every bare repo under the root" would make the set depend on what somebody put in a directory, and
the day an AGPL repo lands there the node starts publishing it from a box that never decided to.
`SERVED` below is written out. A repo not in it is 404, whether or not it is on disk.

## Read-only, and push is refused by name

`git-upload-pack` only — clone and fetch. `git-receive-pack` is answered with a stated refusal
rather than a 404, because a client that tried to push should learn that this is a mirror rather
than that the repository does not exist. Push means write auth, ref-update races and garbage
collection, and it is the phase where "authorization is the encryption" has to earn its keep,
because a push is a grant.

## What this is NOT yet: lattice-native

`routers/oci_router` serves images out of the CAS, where `oci.store.ingest_image` genuinely put
them — the registry IS the lattice, and the address is the same number. **This surface is not that
yet.** It serves `git upload-pack` from bare repositories on disk.

The lattice-native version is the target and the mapping is already written down
(`NODE-DEPLOYMENT-AND-FLEET.md` §4): object ≡ content addressed `cas/<sha256>`, tree/commit ≡ an
artifact, repository ≡ a collection, ref ≡ a name on an artifact, `git fetch` ≡ Merkle anti-entropy.
Reaching it needs an INGEST side — something that puts git objects in the CAS — which does not
exist, exactly as the OCI ingest did not until it was written. What this file establishes is the
URL and the boundary; the substrate underneath can be swapped without the address changing, which
is the same discipline `/v2` follows.

Stated rather than implied, because a `/git` that quietly serves a directory while the docs describe
a lattice is the kind of drift this workspace audits for.

## Off unless configured

`MANTLE_GIT_ROOT` names the directory holding the bare repositories. Unset, every route answers 404
with the reason — a node does not serve source by default, and turning it on is a deliberate act on
a node whose reachability its operator has thought about (`NODE-DEPLOYMENT-AND-FLEET.md` §4, "the
apex question").
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from mantle.services.dependencies import AuthContext, get_auth, offload_sync

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/git", tags=["Git"])

#: The instruments, written out. See the module docstring: this list is the licence boundary, and a
#: discovered set would make it depend on directory contents rather than on a decision.
#:
#: The keys are the names a client clones (`<name>.git`); the values are the bare directory expected
#: under `MANTLE_GIT_ROOT`. Spelled separately because the NAS bares already use the flat, suffixed
#: names the workspace settled on (`agience-prism-py`, not `prism`).
#: Each entry carries three facts, and each is needed by a different reader:
#:   bare    — the directory under MANTLE_GIT_ROOT (the NAS spelling, flat and suffixed)
#:   repo    — the workspace/component name a BUILD stamps (`AGIENCE_COMPONENTS`), which is what
#:             joins a served repository to the revision this node is actually running
#:   license — so the source offer can state it without a second table to keep in step
#: The full Apache-2.0 set: `mantle` joins the three instruments because its licences are measured
#: rather than assumed — it is "the platform" by role and Apache-2.0 by licence, so the objection
#: that produced the original instruments-only scope does not apply to it. Including it is what
#: moves this surface from "a dependency mirror" toward fork-and-run, at no new licence exposure.
#:
#: The line this list draws is still the licence one. `origin`, `chorus`, `ember` and
#: `observe` are AGPL-3.0-only and are deliberately absent — not because serving them would be
#: wrong, but because it is a separate decision with a different consequence (AGPL §13 obligations
#: land on whoever modifies and serves, so publishing them is a service to downstream operators
#: rather than a risk to this one). `/.well-known/agience-source` reports any unserved component
#: honestly rather than omitting it, so the gap is visible from the node itself.
SERVED = {
    "prism":      {"bare": "agience-prism-py.git", "repo": "agience-prism-py",  "license": "Apache-2.0"},
    "crystal":    {"bare": "agience-crystal.git",  "repo": "agience-crystal",   "license": "Apache-2.0"},
    "mantle":     {"bare": "agience-mantle.git",   "repo": "agience-mantle",    "license": "Apache-2.0"},
}


def served_name_for_repo(repo: str) -> str | None:
    """`agience-prism-py` -> `prism`, the name it is cloned under here. None if not served.

    The join between "what this node RUNS" (a build stamp, in repo names) and "what this node
    SERVES" (an allowlist, in clone names). Returning None is a real answer and the source offer
    reports it as such — a component whose source this node does not carry must say so rather than
    be omitted, because an omission reads as "there is nothing else in the image".
    """
    for name, spec in SERVED.items():
        if spec["repo"] == repo:
            return name
    return None

#: Belt and braces with the allowlist above. A name is matched against this BEFORE it is looked up,
#: so a traversal attempt is rejected as a malformed name rather than as a missing repo — the two
#: deserve different answers and only one of them is worth logging.
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")

_UPLOAD_PACK = "git-upload-pack"


def _root() -> Path | None:
    raw = (os.getenv("MANTLE_GIT_ROOT") or "").strip()
    return Path(raw) if raw else None


def _off() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": "git_surface_not_configured",
                 "message": "this node does not serve source. Set MANTLE_GIT_ROOT to the directory "
                            "holding the bare repositories to turn it on."})


def _resolve(name: str) -> Path | None:
    """`<name>` -> the bare repo directory, or None. Never raises, never guesses.

    Three gates, and each rejects a different thing: the pattern rejects a malformed or traversing
    name, the allowlist rejects a repo this node has not decided to publish, and the containment
    check rejects anything that escaped the first two — a defence that costs one `resolve()` and
    covers the case where `SERVED` is ever edited to something clever.
    """
    root = _root()
    if root is None:
        return None
    # A trailing `.git` is how git spells it on the wire; both forms resolve to the same entry.
    key = name[:-4] if name.endswith(".git") else name
    if not _NAME.match(key) or key not in SERVED:
        return None
    path = (root / SERVED[key]["bare"]).resolve()
    try:
        # `is_relative_to` rather than a string prefix: `/srv/git-secrets` starts with `/srv/git`.
        if not path.is_relative_to(root.resolve()):
            return None
    except (OSError, ValueError):
        return None
    # A bare repo, verified rather than assumed. Pointed at a non-repo, `git upload-pack` writes a
    # confusing error to stderr and exits non-zero, which would surface as a 500 for what is really
    # a configuration fault on this node.
    if not (path / "HEAD").is_file() or not (path / "objects").is_dir():
        return None
    return path


def _git() -> str | None:
    return shutil.which("git")


def _pkt(line: str) -> bytes:
    """One pkt-line: a 4-hex length covering itself plus the payload.

    Computed, never hardcoded. The advertisement's service header is the one place a wrong constant
    produces a response that looks right in a terminal and that no git client will accept.
    """
    payload = line.encode()
    return ("%04x" % (len(payload) + 4)).encode() + payload


@router.get("/{name}/info/refs", include_in_schema=False)
async def info_refs(name: str, request: Request,
                    auth: AuthContext = Depends(get_auth)) -> Response:
    """The smart-HTTP discovery request: `GET /git/<name>/info/refs?service=git-upload-pack`."""
    service = request.query_params.get("service", "")
    if service == "git-receive-pack":
        # A stated refusal, not a 404. A client that tried to push should learn this is a mirror,
        # not that the repository is absent — those lead to different next actions.
        return PlainTextResponse(
            "this node serves source read-only; git-receive-pack (push) is not offered here.\n",
            status_code=403)
    if service != _UPLOAD_PACK:
        return PlainTextResponse(
            "only the smart protocol is served: ask for ?service=git-upload-pack\n", status_code=400)

    if _root() is None:
        return _off()
    repo = _resolve(name)
    if repo is None:
        return PlainTextResponse("no such repository on this node: %s\n" % name, status_code=404)
    git = _git()
    if git is None:
        return PlainTextResponse("git is not installed on this node\n", status_code=503)

    proc = await offload_sync(
        subprocess.run,
        [git, "upload-pack", "--stateless-rpc", "--advertise-refs", str(repo)],
        capture_output=True, timeout=60)
    if proc.returncode != 0:
        logger.warning("git advertise-refs failed for %s: %s", name, proc.stderr[:400])
        return PlainTextResponse("could not read %s on this node\n" % name, status_code=500)

    body = _pkt("# service=%s\n" % _UPLOAD_PACK) + b"0000" + proc.stdout
    return Response(
        content=body,
        media_type="application/x-%s-advertisement" % _UPLOAD_PACK,
        # git caches aggressively otherwise and a client can be handed a stale ref list.
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "Expires": "Fri, 01 Jan 1980 00:00:00 GMT"},
    )


@router.post("/{name}/git-upload-pack", include_in_schema=False)
async def upload_pack(name: str, request: Request,
                      auth: AuthContext = Depends(get_auth)) -> Response:
    """The fetch itself: the client's wants/haves in, a packfile out."""
    if _root() is None:
        return _off()
    repo = _resolve(name)
    if repo is None:
        return PlainTextResponse("no such repository on this node: %s\n" % name, status_code=404)
    git = _git()
    if git is None:
        return PlainTextResponse("git is not installed on this node\n", status_code=503)

    payload = await request.body()

    # Buffered, not streamed, and that is a known bound rather than an oversight. The whole
    # packfile is built in memory before the response starts. For the instruments — the largest is
    # a few tens of MB — that is fine and it keeps the negotiation in one obvious place. It would
    # not be fine for a large repository, and the fix when one appears is a streaming
    # `Popen`/`StreamingResponse` pair rather than a bigger timeout.
    proc = await offload_sync(
        subprocess.run,
        [git, "upload-pack", "--stateless-rpc", str(repo)],
        input=payload, capture_output=True, timeout=300)
    if proc.returncode != 0:
        logger.warning("git upload-pack failed for %s: %s", name, proc.stderr[:400])
        return PlainTextResponse("upload-pack failed for %s\n" % name, status_code=500)

    return Response(
        content=proc.stdout,
        media_type="application/x-%s-result" % _UPLOAD_PACK,
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/{name}/git-receive-pack", include_in_schema=False)
async def receive_pack(name: str, auth: AuthContext = Depends(get_auth)) -> Response:
    """Push, refused by name.

    Declared rather than left to fall through to a 404, so the refusal is a property of the route
    table that a test can assert — `test_git_router.py::test_push_is_refused_by_name` — instead of
    an accident of what happens not to be mounted.
    """
    return PlainTextResponse(
        "this node serves source read-only; git-receive-pack (push) is not offered here.\n",
        status_code=403)
