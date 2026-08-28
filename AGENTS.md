# Working in this repo

This is the Agience Mantle source repo (see README.md for the architecture: an
encrypted artifact lattice — authorization is the encryption). The `mantle`
MCP server connected in this session is a live dev instance ("71dev"), so this
project is both the thing being built and, deliberately, a dogfooding target
for its own memory/search capabilities.

## Mantle MCP vs. local memory — pick one lane per fact

Claude Code already has a free, automatic, local cross-session memory
(`~/.claude/.../memory/`) for user preferences, feedback on how to work in
this repo, and project facts. Prefer that for anything Claude-Code-specific —
it costs nothing and is read in every session automatically.

Use the `mantle` MCP tools (`create_artifact`, `recall`, `update_artifact`,
`get_artifact`, `list_artifacts`, `delete_artifact`) instead for:

- Substantive work products worth recalling later, especially across tools
  (Copilot Chat is wired to the same server)
- Structured artifacts that benefit from tags/search/versioning
- Anything meant to exercise Mantle itself as a product

**Don't write the same fact to both stores** — that's the one place real
duplication happens.

## Cost model

- The 7 Mantle tool schemas are deferred and load into context only on first
  use, then *append* to the tool list rather than replacing it — this
  preserves Claude Code's existing prompt cache prefix, so loading them
  mid-session doesn't blow the cache. If a session never touches Mantle, it
  costs nothing.
- Prompt caching (5 min / ~1hr TTL) is entirely intra-conversation and has
  nothing to do with Mantle — it can't survive between sessions either way.
  Mantle's value is substituting one cheap `recall` call for re-deriving
  context from scratch in a new session, not interacting with the cache
  mechanism itself.
- `create_artifact`/`update_artifact` cost roughly what writing the same
  content to a file would (content is tool-call input, billed once).
  `recall` is cheap — query text in, entropy-cut previews (the densest spans
  of each hit, not a truncated prefix — see `search/beacon/density.py`;
  length isn't fixed, it's whatever the cut finds signal for) + a score out.
  Pull full content via `get_artifact` only when actually needed, not by
  default.

## Practical usage

- Always set `content_type` explicitly on `create_artifact`.
- Prefer `recall` over `list_artifacts` for finding things — `recall` scores
  relevance/coverage; `list_artifacts` only pages in id/recency order and
  matches nothing.
- `recall` supports `field:value` filters (`title:`, `tags:`, `content_type:`,
  `created_at:`, etc.) — use them before falling back to free text.
