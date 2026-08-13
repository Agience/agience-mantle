"""
Query Parser v3.1 - Natural Language First

Key Features:
- Natural language default (no operators required)
- Auto-quoting for multi-word terms with +/~
- Soft parsing with graceful degradation
- `~term` marks a term semantic, which selects which text the query path embeds
- Exact phrase support (="text" standalone, field:="text" fielded)

The `@name:value` namespace is a LEXER feature and nothing more. `_extract_controls` lifts
those tokens out of the term stream — so `@lang:en` does not become a search term — and records
them on `ParsedQuery.controls`, which is parse output that no retrieval decision reads. There is
no control here that turns an arm on or off: the semantic arm runs when the request carries a
query vector and this node has a provisioned AnchorSet, both of which are facts about the
request and the node rather than about the query string.

`field:value` is NOT the same kind of output, and that is the one asymmetry in this module.
`_extract_filters` lifts a filter out of the term list and records it on `ParsedQuery.filters`,
and that list is READ: `search.field_filters.compile_filters` turns it into a predicate over
artifact docs, and `sse.router_accessor` hands that predicate to
`lightcone.resolve_authorized_scope`, which applies it to the authorized artifact set
before either retrieval arm runs. So `type:pdf` narrows a recall, on both arms, and the terms
that reach the index are the terms only.

A `word:value` token is a filter ONLY when `word` is a known field, and this module holds no
list of its own to decide that: it asks `field_filters.is_filter_field`, which reads the same
`FILTERABLE_FIELDS` / `FIELD_ALIASES` / `REFUSED_FIELDS` mappings the resolver resolves against.
One roster, two readings. A list here that had to be kept in step by hand would eventually admit
a word the resolver cannot resolve, and that gap is exactly the class of bug being closed —
`STANDARD_FIELDS` was such a list, read by nothing, naming `state`, a field the retrieval path
refuses.

Anything else keeps its colon and stays a term, so `https://example.com`, `3:30`, `C:\\Users\\john`
and `16:9` are ordinary searches rather than 400s. The cost is that a typo — `titel:foo` — is a
search term that finds nothing instead of an error naming it; see `field_filters` for why that
trade is taken and why there is no did-you-mean here.

This module still decides nothing beyond grammar. What a resolvable field IS, and what happens
when one carries an operator it cannot take, belongs to `field_filters`.

Both lists reach a caller through `__str__` — the recall response echoes it as `parsed_query`.
The response's `applied_filters` is the narrower claim: what was applied, not merely understood.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class TermModifier(Enum):
    """Term modifiers for query terms"""
    NONE = "none"           # Default OR behavior
    REQUIRED = "required"   # + prefix (AND)
    EXCLUDED = "excluded"   # ! prefix (NOT)
    SEMANTIC = "semantic"   # ~ prefix (kNN)
    EXACT = "exact"         # = prefix (no stemming)


class FieldOperator(Enum):
    """Field filter operators"""
    EQUALS = "equals"       # field:value
    GT = "gt"              # field:>value
    LT = "lt"              # field:<value
    SEMANTIC = "semantic"   # field:~value (tag expansion)
    EXACT = "exact"        # field:="value"


@dataclass
class Term:
    """A single search term or phrase"""
    text: str
    modifier: TermModifier = TermModifier.NONE
    is_phrase: bool = False
    
    def __str__(self):
        prefix = {
            TermModifier.REQUIRED: "+",
            TermModifier.EXCLUDED: "!",
            TermModifier.SEMANTIC: "~",
            TermModifier.EXACT: "=",
        }.get(self.modifier, "")
        
        if self.is_phrase:
            return f'{prefix}"{self.text}"'
        return f"{prefix}{self.text}"


@dataclass
class FieldFilter:
    """A field-based filter (e.g., type:pdf, size:>1000)"""
    field: str
    value: str
    operator: FieldOperator = FieldOperator.EQUALS
    negated: bool = False
    
    def __str__(self):
        op_str = {
            FieldOperator.GT: ">",
            FieldOperator.LT: "<",
            FieldOperator.SEMANTIC: "~",
            FieldOperator.EXACT: "=",
        }.get(self.operator, "")
        
        neg = "!" if self.negated else ""
        
        if self.operator == FieldOperator.EXACT:
            return f'{neg}{self.field}:="{self.value}"'
        elif op_str and self.operator != FieldOperator.EQUALS:
            return f"{neg}{self.field}:{op_str}{self.value}"
        return f"{neg}{self.field}:{self.value}"


@dataclass
class ControlParam:
    """@ namespace control parameter"""
    name: str
    value: str
    
    def __str__(self):
        return f"@{self.name}:{self.value}"


@dataclass
class ParsedQuery:
    """Fully parsed query with all components"""
    # Core search terms (topics)
    terms: List[Term] = field(default_factory=list)
    
    # Field filters (`type:`, `tag:`, `created_at:>`, …) — the one parse output that a
    # retrieval decision READS. `search.field_filters.compile_filters` validates every entry
    # and compiles it into a predicate over artifact docs; `lightcone.resolve_authorized_scope`
    # narrows the authorized artifact set with it before either arm retrieves, so a filter
    # here is a filter in the result, on both arms, and inside `total`.
    #
    # Every entry is therefore load-bearing, which is why a filter this parser accepts and
    # the retrieval path cannot resolve FAILS the request instead of being dropped. That
    # asymmetry with `controls` below is deliberate: an inert control the caller can see is
    # harmless, an inert FILTER returns a wrong result set that looks like a right one.
    filters: List[FieldFilter] = field(default_factory=list)

    # Control parameters (`@name:value`) — see the module docstring. Lifted out of the
    # term stream and recorded; no retrieval decision reads them.
    controls: Dict[str, str] = field(default_factory=dict)
    
    # Auto-corrections applied
    corrections: List[str] = field(default_factory=list)
    
    # Original raw query
    raw_query: str = ""
    
    def has_topics(self) -> bool:
        """Check if query has any search terms"""
        return len(self.terms) > 0
    
    def has_filters(self) -> bool:
        """Check if query has any field filters"""
        return len(self.filters) > 0
    
    def is_empty(self) -> bool:
        """Check if query is completely empty"""
        return not self.has_topics() and not self.has_filters()
    
    # NO `should_use_hybrid`. There is no query-side switch for the semantic arm, so there is
    # nothing here to derive one from. The arm runs when the request carries a query vector and
    # this node has a provisioned AnchorSet; a predicate over the query string could not move
    # either of those, and one that named itself as if it could is the reason `@hybrid:on` read
    # as a working control while gating nothing. What the ~ modifier DOES do is select which
    # text gets embedded, and `router_accessor._embed_or_none` reads `terms` for that directly.

    def __str__(self):
        """Canonical string representation"""
        parts = []
        
        # Terms
        if self.terms:
            parts.append(" ".join(str(t) for t in self.terms))
        
        # Filters
        if self.filters:
            parts.append(" ".join(str(f) for f in self.filters))
        
        # Controls
        if self.controls:
            parts.extend(f"@{k}:{v}" for k, v in self.controls.items())
        
        return " ".join(parts)


class QueryParser:
    """
    Query parser v3.1 with natural language support and soft error recovery.
    """
    
    # Regex patterns for tokenization
    CONTROL_PATTERN = re.compile(r'@(\w+):(\S+)')
    FIELD_FILTER_PATTERN = re.compile(r'(!?)(\w+):(~|=|>|<)?"?([^"\s]+)"?')
    QUOTED_PATTERN = re.compile(r'([+!~=])?"([^"]+)"')
    TERM_PATTERN = re.compile(r'([+!~=])?(\S+)')
    
    # NO field registry here. Which words are fields is a question about what an artifact doc
    # carries and what the store can answer, not about grammar, and it is answered in one
    # place: `search.field_filters.is_filter_field`, over `FILTERABLE_FIELDS` (plus
    # `REFUSED_FIELDS` for the ones refused with a reason). The parser recognizes the SHAPE
    # `field:value`, ASKS whether the word is a field, and hands on only what is; the resolver
    # still decides whether the operator and value mean anything.


    def __init__(self):
        self.corrections = []
    
    def parse(self, query: str) -> ParsedQuery:
        """
        Parse query string into structured ParsedQuery object.
        
        Args:
            query: Raw query string
            
        Returns:
            ParsedQuery with terms, filters, controls, and corrections
        """
        self.corrections = []
        
        if not query or not query.strip():
            return ParsedQuery(raw_query=query)
        
        parsed = ParsedQuery(raw_query=query.strip())
        
        # Extract controls first (@ namespace)
        query, controls = self._extract_controls(query)
        parsed.controls = controls
        
        # Extract field filters
        query, filters = self._extract_filters(query)
        parsed.filters = filters
        
        # Extract terms (remaining text)
        terms = self._extract_terms(query)
        parsed.terms = terms
        
        # Apply auto-corrections
        parsed.corrections = self.corrections.copy()
        
        return parsed
    
    def _extract_controls(self, query: str) -> Tuple[str, Dict[str, str]]:
        """Extract @control:value parameters"""
        controls = {}
        remaining = query
        
        for match in self.CONTROL_PATTERN.finditer(query):
            name = match.group(1).lower()
            value = match.group(2).lower()
            controls[name] = value
            remaining = remaining.replace(match.group(0), "", 1)
        
        return remaining.strip(), controls
    
    def _extract_filters(self, query: str) -> Tuple[str, List[FieldFilter]]:
        """
        Extract field:value filters with operators, for KNOWN fields only.

        Handles:
        - field:value
        - field:=  "quoted value"
        - !field:value
        - field:>value, field:<value
        - field:~value

        A colon-bearing token whose word is not a known field is left in the term stream
        untouched and unquoted — it is a search term, not a malformed filter.
        """
        filters = []
        remaining_parts = []
        
        # Split into tokens, preserving quoted strings
        tokens = self._tokenize_preserving_quotes(query)
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Check if this looks like a field filter
            if ":" in token and not token.startswith('"'):
                # Try to parse as field filter
                filter_obj, consumed = self._parse_field_filter_advanced(tokens, i)
                if filter_obj:
                    filters.append(filter_obj)
                    i += consumed
                    continue
            
            # Not a filter, keep as regular token
            remaining_parts.append(token)
            i += 1
        
        return " ".join(remaining_parts), filters
    
    def _tokenize_preserving_quotes(self, text: str) -> List[str]:
        """Split text into tokens, keeping quoted strings intact"""
        tokens = []
        current = []
        in_quotes = False
        
        for char in text:
            if char == '"':
                if in_quotes:
                    # End quote
                    current.append(char)
                    tokens.append("".join(current))
                    current = []
                    in_quotes = False
                else:
                    # Start quote
                    if current:
                        tokens.append("".join(current))
                        current = []
                    current.append(char)
                    in_quotes = True
            elif char.isspace() and not in_quotes:
                if current:
                    tokens.append("".join(current))
                    current = []
            else:
                current.append(char)
        
        if current:
            tokens.append("".join(current))
        
        return tokens
    
    def _parse_field_filter_advanced(self, tokens: List[str], start_idx: int) -> Tuple[Optional[FieldFilter], int]:
        """
        Parse field filter from token list, handling quoted values.
        
        Returns:
            (FieldFilter or None, number of tokens consumed)
        """
        token = tokens[start_idx]
        
        # Handle negation
        negated = token.startswith("!")
        if negated:
            token = token[1:]
        
        # Must contain colon
        if ":" not in token:
            return None, 0
        
        # Split on first colon
        parts = token.split(":", 1)
        if len(parts) != 2:
            return None, 0
        
        field, rest = parts

        # Validate field name
        if not field or not field.replace("_", "").isalnum():
            return None, 0

        # And it must be a field the resolver knows. Asked, never listed here — see the
        # module docstring. The import is deferred because `field_filters` imports
        # `FieldFilter` from this module; deferring it here rather than there keeps the
        # dependency pointing the way it reads, from grammar to meaning, and costs one
        # `sys.modules` lookup per colon-bearing token.
        #
        # Returning `(None, 0)` is what makes `https://example.com` a search term: the caller
        # puts the token back into the term stream verbatim, colon and all, unquoted.
        from .field_filters import is_filter_field

        if not is_filter_field(field):
            return None, 0

        # Determine operator
        operator = FieldOperator.EQUALS
        value = rest
        consumed = 1  # Number of tokens consumed
        
        if rest.startswith("~"):
            operator = FieldOperator.SEMANTIC
            value = rest[1:]
        elif rest.startswith("="):
            operator = FieldOperator.EXACT
            value = rest[1:]
            
            # For exact operator, next token might be quoted value
            if not value and start_idx + 1 < len(tokens):
                next_token = tokens[start_idx + 1]
                if next_token.startswith('"') and next_token.endswith('"'):
                    value = next_token[1:-1]  # Strip quotes
                    consumed = 2
        elif rest.startswith(">"):
            operator = FieldOperator.GT
            value = rest[1:]
        elif rest.startswith("<"):
            operator = FieldOperator.LT
            value = rest[1:]
        
        # Handle quoted values (for any operator)
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        
        # Normalize field name
        normalized_field = field.lower()
        
        # Map 'tag' to 'tags' (canonical)
        if normalized_field == "tag":
            normalized_field = "tags"
        
        if not value:
            # Invalid filter
            return None, 0
        
        return FieldFilter(
            field=normalized_field,
            value=value,
            operator=operator,
            negated=negated
        ), consumed
    
    def _extract_terms(self, query: str) -> List[Term]:
        """Extract search terms with modifiers"""
        terms = []
        remaining = query
        
        # First pass: extract quoted phrases
        for match in self.QUOTED_PATTERN.finditer(query):
            modifier_str = match.group(1) or ""
            text = match.group(2)
            
            modifier = self._parse_modifier(modifier_str)
            
            terms.append(Term(
                text=text,
                modifier=modifier,
                is_phrase=True
            ))
            
            # Remove from remaining
            remaining = remaining.replace(match.group(0), "", 1)
        
        # Second pass: extract unquoted terms
        for token in remaining.split():
            if not token:
                continue
            
            # Check for modifier prefix
            modifier = TermModifier.NONE
            text = token
            
            if token[0] in ["+", "!", "~", "="]:
                modifier = self._parse_modifier(token[0])
                text = token[1:]
            
            if not text:
                continue
            
            # Auto-quote multi-word terms with + or ~
            if " " in text and modifier in [TermModifier.REQUIRED, TermModifier.SEMANTIC]:
                self.corrections.append(f'{token} -> {modifier.value[0]}"{text}"')
                terms.append(Term(
                    text=text,
                    modifier=modifier,
                    is_phrase=True
                ))
            else:
                terms.append(Term(
                    text=text,
                    modifier=modifier,
                    is_phrase=False
                ))
        
        return terms
    
    def _parse_modifier(self, prefix: str) -> TermModifier:
        """Convert modifier prefix to TermModifier enum"""
        mapping = {
            "+": TermModifier.REQUIRED,
            "!": TermModifier.EXCLUDED,
            "~": TermModifier.SEMANTIC,
            "=": TermModifier.EXACT,
        }
        return mapping.get(prefix, TermModifier.NONE)


# Convenience function
def parse_query(query: str) -> ParsedQuery:
    """Parse a query string using QueryParser"""
    parser = QueryParser()
    return parser.parse(query)


if __name__ == "__main__":
    # Test cases
    parser = QueryParser()
    
    test_queries = [
        "machine learning",
        "+machine learning",
        '+"machine learning"',
        "~artificial intelligence",
        '="Q1 2025"',
        'title:="Q1 2025"',
        "budget !draft",
        'budget !"internal only"',
        "type:pdf tag:budget",
        "type:pdf,docx size:>1000",
        "+innovation +strategy type:pdf @lang:en",
        "tag:~ai !tag:draft,archive",
    ]

    for q in test_queries:
        parsed = parser.parse(q)
        print(f"\nQuery: {q}")
        print(f"Parsed: {parsed}")
        if parsed.corrections:
            print(f"Corrections: {parsed.corrections}")
