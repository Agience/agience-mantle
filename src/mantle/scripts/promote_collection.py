#!/usr/bin/env python3
"""Promote a collection from one Mantle node to another, over each node's own authorized API.

    python -m mantle.scripts.promote_collection --collection <id> \
        --from-url http://localhost:8182 --from-token-file ~/.claude/dev.token \
        --to-url   https://mantle.home.agience.ai --to-token-file ~/.claude/home.token \
        --dry-run

⭐ IT HAS TO GO THROUGH THE API, AND THAT IS THE PRODUCT WORKING RATHER THAN A DETOUR AROUND IT.
The first cut of this moved seed files between the two lattices directly and carried NO CONTENT:
every body arrived empty, the promotion looked complete, and `recall` on the target matched nothing.
The cause is the invariant the whole store is built on — content keys come from
`content_crypto._default_master_key`, "checked against the grant ledger", so a process holding the
FILE holds ciphertext and nothing else. Measured on one artifact, both ways:

    direct store access   -> ContentDecryptionError
    authenticated MCP     -> "Cross-authority promotion needs a grant seed. token-grantalpha."

So the source must DECRYPT as a principal that may read, and the target must ENCRYPT as a principal
that may write. Those are two authorized sessions, which means HTTP, which means a token per side.

⭐ AND GOING THROUGH THE API MAKES THIS SMALLER, NOT BIGGER. The direct-file version had to
hand-build three things the write path already does: the creator's owner grant (without which a
promotion lands invisible — measured: `list_artifacts` showed 4 unrelated artifacts and recall found
nothing), the index enqueue (without which it lands unfindable — the seed loader contains no index
call at all), and a grant seed card to repair the first. None of that is here, because
`POST /artifacts` does all three for every artifact it writes.

⭐ IDENTITY IS SERVER-SIDE, SO RE-PROMOTING IS AN UPDATE. Each artifact is written with
`identity = promoted:<source>:<root_id>` and the target derives its id from that
(`services/artifact_identity`), so the second promotion of a thing lands on the first artifact
without this script remembering anything. There is no promotion ledger, and losing this machine
changes nothing.

⚠ MEMBERSHIP IS A SECOND CALL, BECAUSE `identity` AND `container_id` ARE MUTUALLY EXCLUSIVE — "a
collection member is born a draft and grows a second live version on first edit-after-commit, so
there is no single row to aim at". Each member is therefore created TOP-LEVEL with its identity, and
then linked into the promoted collection with `source_artifact_id`, which the API documents as
"link an existing artifact in (edge only), no new artifact". Edges are keyed by
`blake2b(src ‖ dst ‖ label)`, so re-linking is an upsert and re-running promotes nothing twice.

⚠ ONE WAY. Two-way promotion is a conflict-resolution problem — both sides edited the same artifact
since the last run — and this has no answer for it. One direction has no conflicts by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ⚠ THE CONSOLE HERE IS cp1252 and every box character below raises UnicodeEncodeError on it — the
# previous cut of this tool died while printing its own progress, after the work was done.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

_TIMEOUT = 180          # a write encrypts and indexes before it answers; seconds, not milliseconds
_CONTAINER_TYPES = ("collection+json", "workspace+json")


class Node:
    """One authorized session against one Mantle node."""

    def __init__(self, base_url: str, token: str, label: str):
        self.base = base_url.rstrip("/")
        self.token = token
        self.label = label

    # -- transport -------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            f"{self.base}{path}", data=data, method=method,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json",
                     "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise SystemExit(f"{self.label}: {method} {path} -> {e.code}\n   {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise SystemExit(f"{self.label}: {method} {path} unreachable — {e}")

    # -- reads -----------------------------------------------------------------------------
    def artifact(self, artifact_id: str) -> dict:
        """One artifact WITH its decrypted body. This is the call that makes promotion possible:
        the same row read off disk is ciphertext."""
        return self._request("GET", f"/artifacts/{artifact_id}") or {}

    def children(self, artifact_id: str) -> List[dict]:
        out = self._request("GET", f"/artifacts/{artifact_id}/children?limit=1000") or []
        return out if isinstance(out, list) else (out.get("result") or out.get("items") or [])

    # -- writes ----------------------------------------------------------------------------
    def create(self, body: dict) -> dict:
        return self._request("POST", "/artifacts", body) or {}


def _is_container(content_type: Optional[str]) -> bool:
    return any(marker in (content_type or "") for marker in _CONTAINER_TYPES)


def _read_token(inline: Optional[str], file_path: Optional[Path], env_var: Optional[str],
                what: str) -> str:
    """A bearer, from a file or an env var in preference to the command line.

    argv is visible to every process on the box and lands in shell history; a token there is a
    credential published by accident. Inline is accepted because a one-off in a scratch shell is a
    real use, and refused silently would be worse than allowed loudly.
    """
    if file_path:
        try:
            return file_path.read_text(encoding="utf-8").strip()
        except OSError as e:
            raise SystemExit(f"{what}: cannot read token file {file_path}: {e}")
    if env_var and os.environ.get(env_var):
        return os.environ[env_var].strip()
    if inline:
        return inline.strip()
    raise SystemExit(f"{what}: no token — pass --{what}-token-file, set its env var, or --{what}-token")


def _collect(node: Node, root_id: str) -> List[Tuple[dict, Optional[str]]]:
    """The collection and everything under it, as (artifact, parent_id) in creation order.

    Breadth-first with a seen-set: a containment graph is not guaranteed to be a tree, and a cycle
    would otherwise walk until the recursion ceiling. Parents precede children, which is what lets
    the link pass below run without a second ordering step.
    """
    root = node.artifact(root_id)
    if not root:
        raise SystemExit(f"{node.label}: collection {root_id} not found, or not visible to this token")

    out: List[Tuple[dict, Optional[str]]] = [(root, None)]
    seen = {root_id}
    queue = [root_id]
    while queue:
        parent = queue.pop(0)
        for member in node.children(parent):
            mid = str(member.get("root_id") or member.get("id") or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            # The child listing carries no body — list paths never do — so each member is fetched
            # individually. That is one request per artifact and it is the only way the content
            # arrives decrypted.
            full = node.artifact(mid) or member
            out.append((full, parent))
            if _is_container(full.get("content_type")):
                queue.append(mid)
    return out


def _identity_for(source_label: str, artifact: dict) -> str:
    """`promoted:<source>:<root_id>` — stable for a given source artifact, forever.

    The ROOT id, not the version id: promoting a thing twice after editing it should update one
    artifact rather than accumulate one per version. The source label is in the key so the same
    collection promoted from two different nodes stays two things on the target.
    """
    rid = str(artifact.get("root_id") or artifact.get("id") or "").strip()
    return f"promoted:{source_label}:{rid}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collection", required=True, help="Collection artifact id on the SOURCE.")
    ap.add_argument("--from-url", required=True, help="Source node base URL (no /artifacts).")
    ap.add_argument("--to-url", required=True, help="Target node base URL.")
    ap.add_argument("--from-token", default=None, help="Source bearer (prefer --from-token-file).")
    ap.add_argument("--to-token", default=None, help="Target bearer (prefer --to-token-file).")
    ap.add_argument("--from-token-file", type=Path, default=None)
    ap.add_argument("--to-token-file", type=Path, default=None)
    ap.add_argument("--source-label", default=None,
                    help="Name for the source inside each identity key. Default: the source host.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Read the source and report; write nothing to the target.")
    args = ap.parse_args()

    src = Node(args.from_url, _read_token(args.from_token, args.from_token_file,
                                          "MANTLE_FROM_TOKEN", "from"), "source")
    dst = Node(args.to_url, _read_token(args.to_token, args.to_token_file,
                                        "MANTLE_TO_TOKEN", "to"), "target")
    if src.base == dst.base:
        print("⛔ source and target are the same node — refusing.", file=sys.stderr)
        return 2

    label = args.source_label or urllib.parse.urlparse(args.from_url).hostname or "source"

    print(f"── promote {args.collection}")
    print(f"   from : {src.base}")
    print(f"   to   : {dst.base}")
    print(f"   ident: promoted:{label}:<root_id>")
    print()

    items = _collect(src, args.collection)
    bodies = sum(1 for a, _ in items if a.get("content"))
    print(f"   read : {len(items)} artifact(s), {bodies} with a body")
    if bodies == 0:
        print("     ⚠ nothing carries content — check the source token can READ these artifacts")

    if args.dry_run:
        print("\nDRY RUN — the target was not written to. Would promote:")
        for artifact, parent in items:
            kind = "collection" if _is_container(artifact.get("content_type")) else "artifact"
            size = len(artifact.get("content") or "")
            print(f"     {kind:10} {str(artifact.get('name') or '(unnamed)')[:34]:34} "
                  f"{size:>7} chars  {'(root)' if parent is None else ''}")
        return 0

    # Pass 1 — every artifact TOP-LEVEL with its identity, so the target derives a stable id and
    # the write path issues the owner grant and enqueues the index.
    id_map: Dict[str, str] = {}
    created = failed = 0
    for artifact, _parent in items:
        source_id = str(artifact.get("root_id") or artifact.get("id") or "").strip()
        body = {
            "name": artifact.get("name") or "",
            "content": artifact.get("content") or "",
            "content_type": artifact.get("content_type") or "text/plain",
            "description": artifact.get("description") or "",
            "identity": _identity_for(label, artifact),
        }
        context = artifact.get("context")
        if context:
            body["context"] = context if isinstance(context, str) else json.dumps(context)
        try:
            written = dst.create(body)
        except SystemExit as e:
            print(f"     ⚠ {str(artifact.get('name'))[:30]}: {e}")
            failed += 1
            continue
        new_id = str(written.get("id") or "").strip()
        if new_id:
            id_map[source_id] = new_id
            created += 1
    print(f"   wrote: {created} artifact(s)" + (f", {failed} failed" if failed else ""))

    # Pass 2 — membership. `source_artifact_id` adds the edge and creates nothing, and the edge is
    # keyed by (src, dst, label), so running this again changes nothing.
    linked = 0
    for artifact, parent in items:
        if parent is None:
            continue
        source_id = str(artifact.get("root_id") or artifact.get("id") or "").strip()
        child, container = id_map.get(source_id), id_map.get(parent)
        if not child or not container:
            continue
        try:
            dst.create({"container_id": container, "source_artifact_id": child})
            linked += 1
        except SystemExit as e:
            print(f"     ⚠ link {str(artifact.get('name'))[:26]}: {e}")
    print(f"   linked: {linked} member(s) into their collection")

    root_new = id_map.get(str(items[0][0].get("root_id") or items[0][0].get("id")))
    if root_new:
        print(f"\n✅ promoted collection on target: {root_new}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
