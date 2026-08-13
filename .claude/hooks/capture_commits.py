#!/usr/bin/env python3
"""Store substantial git commits in the lattice, one artifact per commit. Re-runnable.

NOT A HOOK — nothing fires this. Run it by hand, or on a timer.

⭐ WHY COMMITS ARE THE BEST THING TO CAPTURE, and better than what the hooks capture today.

A commit message in these repos is a decision record: what changed, what it replaced, and why
the alternative was rejected. It is also the only capture source that is **already verified** —
someone chose to commit it — so admitting it costs no judgment call about whether it is worth
keeping, which is the judgment that makes every "should I remember this?" heuristic drift.

Measured 2026-08-13, against a store holding six session transcripts and a handful of files:
three real engineering questions ("why GHCR over DockerHub", "how does the light cone authorize
a recall", "what is the attenuation operator") returned 13 transcripts in 15 hits. The answers
existed — in commit messages and in the README — and lost on surface area, because `recall`
orders by coverage, which counts distinct query stems, and a 118 KB transcript carries more of
them than a precise 3 KB answer does. The corpus was made of the wrong thing.

RE-RUNNING IS FREE AND IS THE POINT. Identity is `commit:<repo>:<sha>`, so a commit already
stored is updated in place rather than duplicated, and a second run picks up only what is new.
There is no cursor to keep and nothing to lose.

⚠ THE SUBSTANCE GATE IS A REAL FILTER, AND IT IS REPORTED, NOT SILENT. A commit whose message is
a bare subject line — `checkpoint`, `wip`, `fix typo` — carries no reasoning, and admitting it
would add stems without adding an answer. `--min-body` is the threshold on the message body
below the subject; the run prints how many commits it skipped and why, because a capture tool
that quietly drops most of its input reads as "covered everything" when it did not.

    python capture_commits.py                     # dry run: what would be stored
    python capture_commits.py --apply
    python capture_commits.py --apply --min-body 0    # everything, checkpoints included
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mantle_common import store_artifact  # noqa: E402

#: ⛔ A COMMIT IS `text/markdown`. An earlier cut minted `text/vnd.agience.commit` so that
#: `type:` could select commits, and that was the same category error made for transcripts: a
#: content type says what the BYTES ARE, and a rendered commit message is markdown. What makes
#: this artifact a commit is what PRODUCED it, which is the operator below — and `tags:` on that
#: operator selects them without teaching the format field to carry provenance.
#:
#: The general rule, worth stating once: a new SOURCE of prose needs a new operator, never a new
#: MIME type. Otherwise every writer that shows up mints a type, and the field stops describing
#: content at all.
COMMIT_CONTENT_TYPE = "text/markdown"

#: What made these artifacts. `op.<domain>.<name>`, the convention `op.content.mirror` and
#: `op.source.wordnet` already use. The hyphenated form goes in `tags` because `normalize_tags`
#: keeps only `[a-z0-9-_ ]` and a colon in a filter value splits the query parser.
OPERATOR = "op.git.commit"

DEFAULT_REPOS = [
    "c:/Users/john/Workspace/Ikailo/Repos/agience-genesis/agience-mantle",
    "c:/Users/john/Workspace/Ikailo/Repos/agience-genesis/agience-origin",
    "c:/Users/john/Workspace/Ikailo/Repos/agience-genesis/agience-cloud",
    "c:/Users/john/Workspace/Ikailo/Repos/agience-genesis/agience-pharos",
]

#: Record and field separators for `git log --format`. ASCII 0x1e/0x1f rather than the more
#: usual NUL: these travel as ARGUMENTS to CreateProcess on Windows, which rejects an embedded
#: null outright (`ValueError: embedded null character`). Both are control characters a commit
#: message will not contain.
_REC = "\x1e"
_FLD = "\x1f"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout


def _commits(repo: Path, limit: int) -> list[dict]:
    """Every commit as `{sha, subject, body, author, date, files}`, newest first."""
    raw = _git(repo, "log", f"-{limit}",
               f"--format=%H{_FLD}%an{_FLD}%aI{_FLD}%s{_FLD}%b{_REC}")
    out: list[dict] = []
    for rec in raw.split(_REC):
        rec = rec.strip("\n")
        if not rec or rec.count(_FLD) < 4:
            continue
        sha, author, date, subject, body = rec.split(_FLD, 4)
        out.append({"sha": sha, "author": author, "date": date,
                    "subject": subject, "body": body.strip()})
    return out


def _files_of(repo: Path, sha: str) -> list[str]:
    """Paths touched, so 'why did ci.yml change' can match on the filename too."""
    txt = _git(repo, "show", "--name-only", "--format=", sha)
    return [ln.strip() for ln in txt.splitlines() if ln.strip()][:60]


def _render(repo_name: str, c: dict, files: list[str]) -> str:
    """The artifact body: the reasoning first, the mechanics after.

    The message leads because it is what answers a question; the file list follows because it is
    what makes the artifact findable from the other direction ('what changed ci.yml').
    """
    parts = [f"# {c['subject']}", "",
             f"**{repo_name}** · `{c['sha'][:12]}` · {c['author']} · {c['date'][:10]}", ""]
    if c["body"]:
        parts += [c["body"], ""]
    if files:
        parts += ["## Files", ""] + [f"- `{f}`" for f in files]
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-body", type=int, default=200,
                    help="minimum chars of message body below the subject (default 200)")
    ap.add_argument("--limit", type=int, default=400, help="commits to scan per repo")
    ap.add_argument("--repo", action="append", help="override the default repo list")
    args = ap.parse_args()

    repos = [Path(r) for r in (args.repo or DEFAULT_REPOS)]
    planned: list[tuple[Path, str, dict]] = []
    skipped_thin = 0
    skipped_nogit = []

    for repo in repos:
        if not (repo / ".git").exists():
            skipped_nogit.append(repo.name)
            continue
        for c in _commits(repo, args.limit):
            if len(c["body"]) < args.min_body:
                skipped_thin += 1
                continue
            planned.append((repo, repo.name, c))

    print(f"{len(planned)} commit(s) to store across {len(repos) - len(skipped_nogit)} repo(s).")
    if skipped_thin:
        print(f"  skipped {skipped_thin} with a body under {args.min_body} chars "
              f"(checkpoints and bare subjects — use --min-body 0 to include them)")
    for name in skipped_nogit:
        print(f"  ⚠ {name}: not a git checkout, nothing scanned")
    print()

    if not args.apply:
        for repo, name, c in planned[:15]:
            print(f"  {name:16s} {c['sha'][:10]}  {c['subject'][:64]}")
        if len(planned) > 15:
            print(f"  … and {len(planned) - 15} more")
        print("\nDry run. Re-run with --apply to store them.")
        return 0

    stored = unconfirmed = 0
    for i, (repo, name, c) in enumerate(planned, 1):
        files = _files_of(repo, c["sha"])
        body = _render(name, c, files)
        identity = f"commit:{name}:{c['sha']}"
        title = f"{name}@{c['sha'][:8]}: {c['subject']}"[:160]
        got = store_artifact(
            identity=identity,
            content=body,
            name=title,
            content_type=COMMIT_CONTENT_TYPE,
            description=f"Commit in {name} by {c['author']} on {c['date'][:10]}",
            context={
                "title": title,
                "tags": ["commit", name, "decision", OPERATOR.replace(".", "-")],
                "operator": OPERATOR,
                "repo": name,
                "sha": c["sha"],
                "author": c["author"],
                "date": c["date"],
            },
        )
        if got:
            stored += 1
        else:
            unconfirmed += 1
        if i % 10 == 0 or i == len(planned):
            print(f"  {i}/{len(planned)}  stored={stored} unconfirmed={unconfirmed}")

    print(f"\nStored {stored}; {unconfirmed} unconfirmed "
          f"(a reply that did not arrive — re-running is safe and lands on the same artifacts).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
