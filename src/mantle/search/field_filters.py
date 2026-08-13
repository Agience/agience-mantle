"""`field:value`, resolved — the parsed filter turned into a membership predicate.

A FILTER IS THE SAME KIND OF OBJECT AS THE LIGHT CONE. `resolve_authorized_scope` produces a
set of artifact ids and both retrieval arms already apply it as set membership — SSE at
`sse/narrowing._ids_for_term`, the vector arm at `mantle/engine._score_chunks`, through the one
`authorized_artifacts` parameter they share. A field filter resolves to a set of the same shape,
so it is applied by narrowing that set BEFORE retrieval rather than by cutting hydrated hits
afterwards. Narrowing first is what keeps `total`, ranking and pagination honest: a post-hydration
cut removes rows a page was already built from, so `total` counts documents the caller cannot
reach and refilling the page means running retrieval again.

WHERE THE PREDICATE RUNS IS WHY IT IS FREE. `resolve_authorized_scope` already reads the raw doc
of every authorized artifact — it needs each one's `collection_id` to derive the collection pairs.
The filter is evaluated in that existing loop, on a doc already in hand, so it adds no store read
and no query. That is also why `db.vertex.list_by_doc_field` is NOT the primitive here: it wants a
`content_type` bucket up front, answers equality only (no range, no negation), and ranges over the
whole store rather than over the authorized set — which would make the intersection a separate
step that could be written wrong. Here the narrowing is structural: the predicate only ever sees
docs of already-authorized artifacts, so its result is a SUBSET by construction. There is no union
to get wrong and no bypass to forget.

WHAT IS FILTERABLE IS WHAT A DOC PLAINLY CARRIES. `content` is AEAD-encrypted at rest and the SSE
postings are blind tokens, so nothing inside the index or the cell can be filtered without
decrypting it. Everything else on the row is plaintext (`db/doc_boundary` encrypts `content` and
nothing else), and that is the filterable surface. Each reader below returns the SAME value the
recall response echoes for that field — `title` here reads what `_hydrate` puts in `SearchHit.title`
— so `title:report` selects the hits whose echoed title is `report`, rather than some near-miss
the caller cannot see.

THIS MODULE IS ALSO THE PARSER'S GRAMMAR. `word:value` is a filter only when `word` is a field
named here, and `is_filter_field` is the one place that question is answered — `query_parser`
asks it rather than carrying a roster of its own. Everything else is an ordinary search term, so
`https://example.com`, `3:30`, `C:\\Users\\john` and `16:9` search rather than fail. One roster
for both readings is the whole point: a list the parser kept in step by hand would eventually
call something a field that nothing here can resolve, which is the shape of bug this replaces.

THE COST IS A SILENT TYPO, TAKEN ON PURPOSE. `titel:foo` is not a field, so it becomes a search
term and returns nothing rather than a 400. That is recoverable — no results, look again — where
the alternative fails an ordinary URL search outright. There is no warning field and no
did-you-mean: guessing at intent from a colon is how the parser got into this in the first place.

`state` IS NOT FILTERABLE, and that is a statement about the store rather than a gap. It selects
the index SEGMENT: `committed`, `draft` and `archived` are separately-keyed encrypted trees under
distinct object-storage prefixes (`mantle/wiring._segment_prefixes`), chosen when the accessor is
BUILT and before any query runs. A `state:draft` filter could not narrow a committed-segment
recall to drafts, because no draft is in the tree being read. `ArtifactRecallRequest.state` is the
one way to say it, so a `state:` filter is refused with a message naming that field.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Iterable, List, Optional

from .query_parser import FieldFilter, FieldOperator


class QueryFilterError(ValueError):
    """A filter the recall path will not silently ignore.

    Raised for a field that cannot be resolved, an operator that is not supported on the field
    it was written against, and a query that is nothing but filters. The router maps it to a
    400 naming the offending field or operator; it is deliberately NOT a subclass anything else
    catches, because being swallowed is the whole failure mode this replaces.
    """


def parse_context(doc: Optional[dict]) -> dict:
    """An artifact doc's `context` as a dict — `{}` for absent, malformed or non-object.

    `context` arrives as a dict on some write paths and a JSON string on others (see
    `db/vertex._columns`), so both shapes are read. Single-sourced here because the filter
    predicate and `router_accessor._hydrate` must agree on what a doc's title IS; two readers
    that disagreed would make `title:x` select hits whose echoed title is not `x`.
    """
    if not doc:
        return {}
    ctx = doc.get("context")
    if not ctx:
        return {}
    if isinstance(ctx, dict):
        return ctx
    if isinstance(ctx, str):
        try:
            parsed = json.loads(ctx)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ── readers: query field name -> the value on the doc ────────────────────────────────────────
#
# Each returns a scalar or a list. A list field matches when ANY of its members matches, which
# is what makes `tags:budget` mean "carries the budget tag" rather than "its only tag is budget".


def _title(doc: dict, ctx: dict) -> Any:
    return ctx.get("title") or ctx.get("name") or doc.get("name")


def _description(doc: dict, ctx: dict) -> Any:
    return ctx.get("description") or doc.get("description")


def _tags(doc: dict, ctx: dict) -> Any:
    raw = ctx.get("tags") or ctx.get("tags_canonical") or []
    if isinstance(raw, str):
        return [p.strip() for p in raw.split(",") if p.strip()]
    return list(raw) if isinstance(raw, list) else []


class _Field:
    """One filterable field: how to read it off a doc, and whether it is ORDERED.

    `ordered` gates `>` and `<`. It is True only for the ISO-8601 timestamps, whose lexicographic
    order IS their chronological order. Everything else is an opaque identifier or free text, where
    a string comparison would answer confidently and wrongly — so `>` on one is refused rather
    than served.

    `needs_context` marks the readers that look inside the `context` blob. It is what lets the
    compiled predicate parse that JSON at most ONCE per doc no matter how many context-backed
    filters a query carries, and not at all when none does. Every reader takes `(doc, ctx)` —
    including the ones that ignore `ctx` — so there is one reader signature rather than two
    kinds of reader for a caller to keep straight.
    """

    __slots__ = ("read", "ordered", "needs_context")

    def __init__(self, read: Callable[[dict, dict], Any], *,
                 ordered: bool = False, needs_context: bool = False) -> None:
        self.read = read
        self.ordered = ordered
        self.needs_context = needs_context


FILTERABLE_FIELDS: Dict[str, _Field] = {
    "id":            _Field(lambda d, c: d.get("id") or d.get("_key")),
    "root_id":       _Field(lambda d, c: d.get("root_id")),
    "collection_id": _Field(lambda d, c: d.get("collection_id")),
    "content_type":  _Field(lambda d, c: d.get("content_type")),
    "owner_id":      _Field(lambda d, c: d.get("created_by")),
    "title":         _Field(_title, needs_context=True),
    "description":   _Field(_description, needs_context=True),
    "tags":          _Field(_tags, needs_context=True),
    "created_at":    _Field(lambda d, c: d.get("created_time"), ordered=True),
    "updated_at":    _Field(lambda d, c: d.get("modified_time"), ordered=True),
}

#: Spellings the parser normalizes onto a canonical field. `tag` -> `tags` is applied by the
#: parser itself; `type` -> `content_type` is applied here so the parser stays a parser.
FIELD_ALIASES: Dict[str, str] = {
    "tag": "tags",
    "type": "content_type",
}

#: Fields a caller can reasonably expect and that this path refuses ON PURPOSE, each with the
#: reason. Refusing with an explanation is the difference between "not built yet" and "cannot be
#: built here"; a bare "unknown field" would invite a bug report for `content:` forever.
REFUSED_FIELDS: Dict[str, str] = {
    "state": (
        "`state` selects the index segment (committed | draft | archived), which is a separately "
        "keyed encrypted tree chosen before the query runs — a filter cannot narrow one segment "
        "to another. Send the `state` request field instead"
    ),
    "content": (
        "`content` is AEAD-encrypted at rest and the lexical postings are blind tokens, so it "
        "cannot be filtered without decrypting the corpus. Search for the words instead — that "
        "is what the query terms do"
    ),
    "size": "artifact rows carry no size; nothing in the store can answer it",
    "filename": (
        "artifact rows carry no filename; put it in `context.title` or a tag if you need to "
        "select on it"
    ),
}


#: Shared empty mapping for the no-context-filter case, so the common path allocates
#: nothing. Never mutated — every reader only reads.
_NO_CONTEXT: Dict[str, Any] = {}


def filterable_field_names() -> List[str]:
    """The canonical filterable fields plus their accepted aliases, sorted — for error text."""
    return sorted(set(FILTERABLE_FIELDS) | set(FIELD_ALIASES))


def _canonical(field: str) -> str:
    return FIELD_ALIASES.get(field, field)


def is_filter_field(word: str) -> bool:
    """Is `word` a field this module has an opinion about?

    THE ONE ANSWER TO "IS `word:value` A FILTER". `query_parser` calls this and consults
    nothing else, so the parser's notion of a field and this module's notion of a resolvable
    field are the same three mappings read twice rather than two lists kept in step. Add a
    field to `FILTERABLE_FIELDS` and the parser recognizes it in the same commit; there is
    nowhere else to remember.

    `REFUSED_FIELDS` counts as a field, and that is the load-bearing half. `state:draft` is a
    filter the caller meant as one, so it must still reach `_validate` and be refused BY NAME.
    Narrowing which words are fields must not soften the error for a word that genuinely is
    one — a refusal that quietly became a search term would answer a different question.

    Aliases resolve first, so `type` is a field because `content_type` is.
    """
    field = _canonical(word.lower())
    return field in FILTERABLE_FIELDS or field in REFUSED_FIELDS


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _matches(spec: _Field, filt: FieldFilter, doc: dict, ctx: dict) -> bool:
    """Does this doc satisfy one filter, ignoring negation?

    EQUALS is case-insensitive and splits its value on commas, so `type:pdf,docx` is an
    any-of — the shape the parser's own grammar examples use. EXACT (`field:="v"`) is
    case-sensitive and takes its value whole, which is the only way to select a value that
    contains a comma or differs from another only by case.
    """
    present = _as_list(spec.read(doc, ctx))
    if not present:
        return False

    if filt.operator is FieldOperator.EXACT:
        return filt.value in present

    if filt.operator in (FieldOperator.GT, FieldOperator.LT):
        want = filt.value
        if filt.operator is FieldOperator.GT:
            return any(v > want for v in present)
        return any(v < want for v in present)

    wanted = {p.strip().lower() for p in filt.value.split(",") if p.strip()}
    if not wanted:
        return False
    return any(v.lower() in wanted for v in present)


def _validate(filt: FieldFilter) -> _Field:
    """Resolve one filter to its field spec, or raise the 400 that names what is wrong."""
    field = _canonical(filt.field)

    if field in REFUSED_FIELDS:
        raise QueryFilterError(
            "cannot filter on `%s`: %s. Filterable fields: %s"
            % (filt.field, REFUSED_FIELDS[field], ", ".join(filterable_field_names()))
        )

    spec = FILTERABLE_FIELDS.get(field)
    if spec is None:
        # Unreachable from a query string — `is_filter_field` is what the parser tests, so a
        # word that is not a field never becomes a `FieldFilter` at all. It stands for the
        # programmatic caller that builds one directly, and it stays loud for the same reason
        # every refusal here does: a filter that is dropped answers a different question.
        raise QueryFilterError(
            "unknown filter field `%s`. Filterable fields: %s. In a query string `%s:%s` is "
            "not a filter at all — only a known field makes one — so it searches as an "
            "ordinary term."
            % (filt.field, ", ".join(filterable_field_names()), filt.field, filt.value)
        )

    if filt.operator is FieldOperator.SEMANTIC:
        raise QueryFilterError(
            "the `%s:~value` operator (tag expansion) is not supported: expanding a value to its "
            "neighbours needs an embedding of that value, and Mantle runs no models. Use "
            "`%s:value`, or `~value` to steer the semantic arm with a term."
            % (filt.field, filt.field)
        )

    if filt.operator in (FieldOperator.GT, FieldOperator.LT) and not spec.ordered:
        ordered = sorted(n for n, s in FILTERABLE_FIELDS.items() if s.ordered)
        raise QueryFilterError(
            "the `%s` operator is not supported on `%s`: it is not an ordered field, and "
            "comparing it as text would answer confidently and wrongly. Range filters apply "
            "to: %s"
            % (">" if filt.operator is FieldOperator.GT else "<",
               filt.field, ", ".join(ordered))
        )

    return spec


def compile_filters(
    filters: Iterable[FieldFilter],
) -> Optional[Callable[[dict], bool]]:
    """Validate every filter and return one predicate over a raw artifact doc.

    `None` when there are no filters, which is the signal to leave the authorized set exactly
    as the light cone resolved it — a query with no filter must be byte-identical to one that
    never went near this module.

    Filters CONJOIN: every one must hold. Conjunction is the reading that can only narrow
    further as a caller adds terms, and it is the only reading under which each filter means
    the same thing alone as it does beside another.

    Raises `QueryFilterError` on the first filter that cannot be resolved. Failing the whole
    request is the point: a filter that is dropped because it was not understood returns a
    result set that looks like an answer and is not one.

    COST. Validation happens once here, per request, not per doc — the returned predicate does
    no field lookup, no alias resolution and no error checking. Per doc it is O(number of
    filters) dict reads plus at most one `json.loads` of `context`, taken only when some filter
    actually reads a context-backed field. The predicate itself allocates nothing per doc beyond
    the value lists it compares.
    """
    specs = [(_validate(f), f) for f in filters]
    if not specs:
        return None
    # Decided once, so the per-doc path carries no branch for it.
    wants_context = any(spec.needs_context for spec, _ in specs)

    def predicate(doc: dict) -> bool:
        ctx = parse_context(doc) if wants_context else _NO_CONTEXT
        for spec, filt in specs:
            hit = _matches(spec, filt, doc, ctx)
            if hit == filt.negated:      # negated -> a match disqualifies
                return False
        return True

    return predicate



def describe(filters: Iterable[FieldFilter]) -> List[str]:
    """The canonical spelling of each filter, for the response's `applied_filters` echo.

    Everything `compile_filters` accepted is applied, so this list is what narrowed the recall
    rather than what merely parsed — anything it could not apply raised instead of arriving here.

    Aliases are resolved: `type:pdf` echoes as `content_type:pdf`. The echo is an account of
    what ran, and what ran was the `content_type` reader — a caller comparing the echo against
    the field list in a 400 should find the same names on both sides.
    """
    out: List[str] = []
    for f in filters:
        canonical = _canonical(f.field)
        if canonical == f.field:
            out.append(str(f))
        else:
            out.append(str(FieldFilter(
                field=canonical, value=f.value,
                operator=f.operator, negated=f.negated,
            )))
    return out


__all__ = [
    "FIELD_ALIASES",
    "FILTERABLE_FIELDS",
    "REFUSED_FIELDS",
    "QueryFilterError",
    "compile_filters",
    "describe",
    "filterable_field_names",
    "is_filter_field",
    "parse_context",
]
