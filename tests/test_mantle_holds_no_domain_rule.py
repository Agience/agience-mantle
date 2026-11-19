"""Mantle is a generic store: it holds no rule about what an artifact is called, how a list is
ranked, what the Inbox is, or what a create costs. Those rules live in the artifacts that declare
them (types, views and operators, run by Crystal).

Each row names the identifiers one such rule was written with. A strict row fails when any of its
names appears in `src/` as a whole word; a row still moving out reports its hits and does not
fail. A row turns strict when its rule has moved.
"""
from __future__ import annotations

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

#: rule -> (strict, identifiers)
ROWS = {
    "naming": (False, (
        "DEFAULT_WORKSPACE_NAME", "TITLE_FROM_CONTENT_MAX", "base_title", "next_free",
        "needs_title", "with_title", "title_on_save", "title_from_content", "_title_a_new_one",
    )),
    "inbox_refusal": (True, ("_refuse_the_callers_inbox",)),
    "told_apart": (True, ("label_marks", "place_marks", "collection_marks", "label_at",
                          "name_shared", "place_yours")),
    "recall_query": (True, ("_compose_query_text", "tag_exact")),
    "charge": (True, ("_charge", "charge", "charged_to")),
    "ranking": (True, (
        "titles_lead", "_title_key", "_title_text", "_apply_search", "_by_rank", "ORDER_RANK",
        "RANK_KEYS", "_SEARCH_CT",
    )),
}


def hits(names, root=SRC):
    """`{name: [file:line, ...]}` for each of `names` found in `root` as a whole word."""
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, names)) + r")\b")
    found: dict = {}
    for path in sorted(root.rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in pattern.finditer(line):
                found.setdefault(m.group(1), []).append(f"{path.relative_to(root)}:{n}")
    return found


def test_a_strict_row_has_no_name_in_src():
    for rule, (strict, names) in ROWS.items():
        found = hits(names)
        if strict:
            assert not found, f"{rule}: Mantle names a rule that belongs to an artifact: {found}"
        elif found:
            print(f"[report] {rule}: {sum(map(len, found.values()))} hits in {sorted(found)}")


def test_a_seeded_name_is_caught(tmp_path):
    (tmp_path / "seeded.py").write_text("DEFAULT_WORKSPACE_NAME = 'x'\nnext_freedom = 1\n")
    assert hits(ROWS["naming"][1], tmp_path) == {"DEFAULT_WORKSPACE_NAME": ["seeded.py:1"]}
