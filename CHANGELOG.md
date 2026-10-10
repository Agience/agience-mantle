# Changelog

All notable changes to `agience-mantle` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
