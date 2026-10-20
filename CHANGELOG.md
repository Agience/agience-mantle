# Changelog

All notable changes to `agience-mantle` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.4] - 2026-10-17

### Added
- A title is part of an artifact's content: the store seals it under the content's key (`context.sealed`) on every write and restores it on every read the key oracle serves; a reader it will not serve gets the context without the title, never the ciphertext. The change feed carries no title in either form. `python -m mantle.system.seal_titles` seals titles stored before, as each row's creator, idempotently. `--unseal` is its reverse, which the release runs before rolling back to a Mantle that does not read sealed titles; a title the key oracle will not open stays sealed rather than emptied. Titles were plain text in the artifact row and the event log.
- The platform system principal uses the platform mail secrets and never reads them: `invoke` on the authorizer and its two credentials, `read` only on the operator and the authorizer. A `platform.email.send` token reaches exactly the platform authorizer (`GET`) and the release of its two credentials (`POST .../use`), by their derived ids. It allowed only `POST .../op/invoke`, a route Mantle does not serve, so it reached nothing.
- `POST /artifacts/{id}/use` releases a credential's secret for one use: only to a persona that proved who it is (a delegation whose `act.iss` is the persona itself), only `for` a card that names the secret and is served by that persona (an LLM connection for Lumen; an authorizer or a person's MCP server for Iris), only to someone holding `invoke` on the card and the secret (not `read`), and only sealed to the persona's enrolled key, bound to the secret and the card. The answer, the edge and any log carry ciphertext. The content key is demanded under `invoke`; each release is in the card's and the secret's access logs.
- `GET /artifacts/{id}/children?state=archived` lists a container's archived members, homed there or linked in, behind the same per-member authorization as the live listing. `state=live` is the default.
- `POST /artifacts/batch` takes `include_archived` (default `false`) to return archived artifacts too, for a caller showing them on purpose such as a recall of the `archived` segment.
- An artifact on the children listing and on `POST /artifacts/batch` carries `container_ids`: every container holding it, workspaces included, since a workspace is a collection is an artifact. `committed_collection_ids` still leaves workspaces out, for the sharing checks. `POST /artifacts/batch` carried no memberships at all, so a card opened from search said it was in no collection.
- A type's `offer` lineage ends at the root type `*/*`, and an Agience or `+json` data type skips `application/*`, as Crystal resolves it.
- `mantle.services.email_service` sends Mantle's mail: invitations to a workspace or collection, through the Gmail API (`EMAIL_PROVIDER=gmail`, `PLATFORM_EMAIL_ADDRESS`, `GMAIL_OAUTH_*`, display name `PLATFORM_EMAIL_NAME`) or written to `EMAIL_OUTBOX_DIR` (`EMAIL_PROVIDER=outbox`). Invitations used to send nothing: the invite path imported a mail service Mantle did not have. The invitation names the inviter and the resource's title.
- A content type's `type.json` may declare an `offer`: lists of paths (`title`, `context.headline`, `lemmas.0`, `lemmas.1:`) for `title`, `description` and `tags`. The indexer reads the nearest declaration for each through the type's `inherits` and its major wildcard (`text/*`), then the default it used before (top-level `title`, a structured context's `title`, `name`; `description`, a structured context's `description`; no tags).
- `PUT /type-definitions/{server}` stores every type definition one server owns as a system-owned `application/vnd.agience.type+json` artifact, replacing what it published before and archiving a type it no longer sends. Only Crystal's service identity may call it. A persona's definition outranks Crystal's builtin for the same type, as in Crystal. The artifacts of a type whose resolved offer changed are indexed again. Definitions a user wrote are never read.
- Published type definitions are loaded at startup, before the index worker starts.
- Trash. `POST /artifacts/{id}/trash` (needs `delete`) archives the artifact and marks it `trashed_by` and `trashed_time`; `GET /artifacts/trash` lists the caller's Trash, most recently trashed first, each with `purge_after`. Restoring (`PATCH {"state": "live"}`) clears both marks. Every hour, an artifact trashed more than 30 days ago is deleted through the ordinary delete, as the person who trashed it; if they may no longer delete it, it is kept and the refusal logged. `DELETE /artifacts/{id}` still deletes at once.
- `trashed_by` is indexed in `listkey`, so a Trash listing reads the index instead of scanning archived versions; a store whose index is not built is scanned instead, with a warning. `LatticeArtifactStore.lookup_with_list_field` pages every artifact carrying an indexed field.

### Changed
- A grant key also updates what its grants let it update, and the version is its issuer's. A timer records each run on its own card with its key.
- A grant key creates inside a container its grants let it add to; what it creates belongs to the person who issued the key, and the mint context names the key. A key still creates nothing top-level and reaches nothing its grants do not. Every create by a key answered 401 whatever its permissions, so an agent holding `can_create` on a collection could not record anything in it.
- A container's archived listing (`children?state=archived`) leaves out artifacts in Trash; they are listed in Trash instead.
- Archiving an artifact that is already archived answers 200 with the artifact instead of 404. Any other change to an archived artifact is still refused until it is restored.
- `DELETE /artifacts/{id}` deletes an archived artifact; it answered 404.
- Linking an archived artifact into a container (`POST /artifacts` with `source_artifact_id`) is refused with 409 and a message to restore it first. The link used to succeed into a listing that does not show archived members.
- With archived members included, `list_collection_artifacts` returns an archived artifact linked in from another container, not only one homed in this container.
- Removing an artifact from a container never destroys it. Held elsewhere, it loses only its link here. Held nowhere else, it is archived in place and keeps its link, so the container's archived listing finds it; that needs `update` on the artifact, and the removal is refused (403) without it. The response's new `archived` field says which happened. An artifact held nowhere else used to be destroyed.
- A WordNet synset's lemmas are indexed as its names because its type declares them so (Sage's `text/x-wordnet` type), not by a rule in the indexer naming that type. Until Crystal publishes that declaration, a synset is indexed under the default offer.

## [0.1.3] - 2026-10-09

### Breaking
- Artifact `state` is now `live` or `archived`. Every write path records `live`, and responses report `"state": "live"` where they reported `committed`. Stored `committed` and `draft` rows, and those sent by mesh peers, read as `live` until `system/manage_state_collapse` relabels them.
- The draft and commit workflow is gone from `mantle.db.lattice_api`. `get_draft_artifact`, `list_draft_artifacts` and `batch_commit_drafts` are removed. `get_latest_committed_artifact` is now `get_latest_live_artifact`, and `list_committed_artifacts_by_context_content_type` is now `list_live_artifacts_by_context_content_type`. `db.backend.find_version_in_state` is also removed.
- `list_collection_artifacts` no longer returns `has_committed_version`. A collection member's current version is its newest non-archived version.
- The `state:*` counter and the usage metric formerly reported as `committed_artifact_versions_total` are now `live_artifact_versions_total`.
- The trusted-proxy setting in `.env.example` now defaults to loopback only (`127.0.0.0/8` and `::1`) when unset, instead of trusting no one. `none` trusts no one. A value replaces the default, so list loopback as well if a same-host edge still fronts the node. An edge on another host, or one reaching a container over a bridge network, must still be listed.

### Added
- `identity` can now be combined with `container_id` on artifact create, provided the caller has create permission on that collection. The identity names the member's root, derived as for top-level artifacts. A repeat write overwrites that member's live version in place, and an overwrite does not answer `201 Created`.
- Federation sends `EMBER_INVOKE_TOKEN` to peers as `X-Ember-Token`. It is the secret every shard's `/v1/invoke` requires. Peer base URLs are listed in `EMBER_PEERS`. Both are documented in `.env.example`.
- `db.constants` exposes `STATE_LIVE`, `STATE_ARCHIVED`, `LIVE_STATES`, `LEGACY_LIVE_STATES` and `live_states()`. A query for state `live` matches every recorded value that reads as live. A query for `draft` matches only legacy drafts, which is how the migration tool finds them.

### Changed
- `parse_artifact_context` merges a context's `caller` block under its top-level keys, with the top level winning. It no longer lifts store-owned keys (the rung, the mint and the content address) out of `caller`, so a client's claim is never read as the store's own derivation.
