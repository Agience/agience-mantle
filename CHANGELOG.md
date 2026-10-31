# Changelog

All notable changes to `agience-mantle` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.10] - 2026-10-31

### Added
- Threshold custody of master keys, selected by `MANTLE_MASTER_KEY_CUSTODY=threshold` (the default stays `lattice`). Each key is held as 2-of-3 Shamir shares: one in the `master-key:<principal>` row under the KEK, one at Origin (`/internal/key-shares`), and one sealed with pyrage to the escrow age recipient named by `MANTLE_ESCROW_RECIPIENT_FILE`, written to the durable object store at `escrow/<sha256(principal)>.<kcv>.<escrow_id>.age` and never to the lattice. Each split writes a new object only if none has that name, the row names its object, and recovery reads that one; no escrow object is overwritten. Threshold custody is refused at startup when no durable object store is configured. A key is used by combining this node's share with Origin's and checking the result against a stored key-check value; the escrow share is combined only in recovery mode (`MANTLE_MASTER_KEY_RECOVERY=escrow` with `MANTLE_ESCROW_IDENTITY_FILE`). A share that cannot be read, a missing Origin share, an unreachable Origin, or an Origin share while this node holds none all refuse; a key is minted only when nothing holds a share and Origin answers with an explicit "absent" body.
- Every write of a key row under threshold custody takes the custody lock, a file beside the lattice (`<lattice>.custody.lock`), and is conditional on the stored form read under it: a put writes the row pending, then the escrow share (a failed write refuses the put), then Origin's share, and marks the row active only if it is still the row it wrote.
- Each call to Origin for a share signs its own token: 30 seconds of life, a fresh `jti`, and claims naming the method, the route path and the principal.
- Calls made while holding the custody lock are bounded inside its 30-second wait: Origin at 3 seconds to connect and 8 to read, the durable store at 3 and 8 with one attempt (`content_service._make_s3_client` takes optional timeouts and attempts). A hung peer refuses the put.
- `mantle.system.split_master_keys` splits every whole master key into shares, sealing the escrow shares into one export file (`--escrow-out`) before any row is replaced, and replaces a row only after its shares, including Origin's stored copy, recombine to the key. Each row is re-read under the custody lock and left alone if it changed since the pass read it. `--dry-run` counts, `--combine` writes split keys back whole under the KEK for a rollback (dropping a pending row only if it is still pending), and `--export-escrow` copies every sealed escrow share from the durable store into one file for the operator's offline copy. A re-run splits only what is still whole. The keys do not change, so nothing they encrypt is rewritten.

### Changed
- `LatticeMasterKeyStore` refuses a row that holds a share of a split key instead of reading it as absent, so a node returned to whole-key custody without `--combine` cannot mint a replacement over a split key.
- The `service` and `dev` extras require `agience-prism>=0.1.8` (for `prism.trust.shamir`) and `pyrage>=1.2`.

## [0.1.9] - 2026-10-30

### Changed
- A grant's `name`, `notes` and invite address (`target_entity`) are sealed under the node's data key (`db/node_seal`), and the `grant.invite.created` event carries `has_target` instead of the address. A grant stored in plaintext is sealed by `mantle.system.seal_titles`, and `--unseal` restores it for a rollback.
- The name a person gives a link is stored as a keyed token, with the name sealed in the edge for display and a token per dotted prefix, so `remove_links` and the `links` filter still match. The store's own labels (`contains`, `context`, `authorizer`, `link`) stay readable. `mantle.system.seal_titles` rewrites a link named before this the same way and drops the emptied label counter that spelled its name; `--unseal` puts each name back for a rollback.
- The platform system principal holds read and update on the platform's own collections (People, Authorities, and every collection the platform created), granted at provisioning and by `mantle.system.seal_titles`, which then seals the platform's rows. People and Authorities are written as the platform's own id.
- A row's sealed fields open for a grant on the row alone, on every path that reads them, as its body does.
- `mantle.system.drop_plaintext_index` drops the plaintext lexical index an earlier Ember wrote into the store (`fts`, `fts_map`, `fts_vocab` and their shadow tables) and its `fts:total` counter.
- A person's provenance record (`services.principal.person_artifact`) carries no `lemmas`: they were the words of its `sub`, often an email address, stored in the clear and indexed.

### Removed
- The embeddings cache (`search/embeddings_cache.py`, `EMBEDDINGS_CACHE`, `EMBEDDINGS_CACHE_PATH`). Mantle makes no vectors, so it held none, and a cache keyed by a hash of the text would hold a confirmable fingerprint of each text outside every key.

### Fixed
- A grant whose fields were sealed under another node's key (a copied or restored store) failed every lookup for its grantee. Its fields stay absent, its structure authorizes as before, and a save writes the sealed blob back unchanged.
- `platform_operators.grant_to_every_account` does not count the platform system principal as an account.
- A link name that does not open on this node shows as its token, never as a plain membership.
- On a fresh node, the first sign-in was refused the key for the People directory's name, because provisioning acts as the member.
- `mantle.config` imports `python-dotenv` only when `load_env` runs, so the embeddable store surface loads nothing outside the standard library.

## [0.1.8] - 2026-10-30

### Changed
- Every context key an artifact carries, and its top-level `name` and `description`, are sealed under the body's key, as the title was. Structure is the named exception (`doc_boundary.STRUCTURAL_CONTEXT_KEYS`): `content_type`, `content_key`, `content_cas_ref`, the card-key bindings `stream` and `inbound`, and the upload bookkeeping `upload`, `processing` and `storage`. Issuers, type definitions, anchors, crosswalks and subscriptions keep their context readable, because it is read before anyone signs in, but their title is still sealed. A sealed blob written before this (title only) still opens.
- The change feed carries the structure alone: no sealed field, in either form, and no `name` or `description`. A subscriber reads them back through the authorized path.
- Recall logs its counts, never the query text.
- A new description or tag set makes a version, as a new title does. What the server records about the bytes (digest, size, `addressing`, the mint) is sealed too, but a completed upload does not fork its artifact.
- `mantle.system.seal_titles` seals every sealable field of a row stored in plaintext, and `--unseal` restores every field for a rollback.
- `provision_user` acts as the user it provisions, so an admin script that provisions writes under that user's key.

### Fixed
- `mantle.system.seal_titles` acts for each row as its creator or the owner of its root, as a person and then as a service, and then as the platform, as `mantle.system.reindex` does; a row with no owner goes to the platform. It writes a seal only if the same actor opens it again, and leaves a row it cannot verify in plain text, reported as `left_plain`. On the release candidate the pass had sealed 43 of the platform's rows that no actor could open.
- A reader the key service would not serve, which then saved what it read, erased the sealed fields for good: the blob was dropped before the decrypt. It is kept unless the decrypt succeeds, and a response never carries it.
- `title:`, `description:` and `tags:` filters read the stored row, where the fields are sealed, and matched nothing. They read the row opened for the searcher.
- `mantle.system.type_uploads` opens a sealed row before deciding whether it is an untyped upload.


## [0.1.7] - 2026-10-29

### Added

- `mantle.system.reindex` is a migration, run with `python -m mantle.system.reindex`, that rebuilds the search index once for each index generation recorded in the `search.reindex.generation` platform setting. Artifacts saved before a change to indexing, such as hyphenated names being found by their parts, are re-indexed on the next release. Each artifact is indexed as a principal whose grants reach it (its creator, the owner of its root, or the platform for its own artifacts), because the platform's principal holds no grant to a person's content. `--check` reports whether a rebuild is due and changes nothing.
- A pass that reports failures records no generation and exits non-zero, so the next release runs it again. Artifacts that cannot be read on the node are counted as unreadable, and index writes the key service refuses as denied; neither is a failure.
- `reindex_all_artifacts` in `mantle.search.init_search` accepts `force=True`, which rebuilds even when an earlier full pass has set the completion marker that makes later boots skip.
- The release workflow (`.github/workflows/publish.yml`) also runs on pushes to `release/**` branches. It builds and checks the package and keeps the files as the `dist` artifact for the platform release to deploy.

### Changed

- The README states what is encrypted and what is stored readable. Artifact content and `context.title` are encrypted per principal. Every other field is governed by grants, including `name`, `description`, other `context` keys, relationship names, grant details, ids, graph edges, the access log and the change feed. Recall logs its query text at `INFO`.
- The README states that the search index holds no plaintext terms: every artifact is indexed by blind token, with no plaintext lexical index to fall back to.

### Fixed

- `mantle.system.seal_titles` (and `--unseal`) handles each row as its creator, else as the owner of its root. A note left in a shared workspace by a contributor whose access was revoked could not be unsealed as them, and stopped the release's rollback check.
- `/version` names the installed version when no `build_info.json` was stamped, as Origin's does. A release installs Mantle from its wheel, with no `build_info.json`, and production Mantle answered with an empty version after 2026.10.14.

## [0.1.6] - 2026-10-28

### Added
- `POST /artifacts/{id}/untrash` restores an artifact from Trash to the state it had before it was deleted, live or archived. It needs `update` and answers 409 when the artifact is not in Trash. Trashed artifacts now carry `trashed_from`.
- `GET /artifacts/{id}/versions` lists an artifact's versions, newest first. `GET /artifacts/{id}/versions/{version_id}` reads one earlier version, with its content and title.
- `POST /artifacts/{id}/versions/{version_id}/restore` writes an earlier version back as the artifact's current content. The replaced version is kept. The route needs `update` on the artifact and `read` on the version. It answers 409 when the restore would show the version to someone it is denied to.
- Each version is shown only to principals who can read that version's own id. A grant on the artifact reaches every version, and a grant or deny on one version reaches that version alone.
- `POST /grants/{grant_id}/resend` sends a pending invite again with a new claim link, and the earlier link stops working. The invite's granter or an admin of its resource may resend it, and the response carries the new `claim_token`, `claim_url` and `email_sent`.
- The response to creating an invite grant in `grants_router` includes `email_sent`, which says whether the invitation was mailed.
- `python -m mantle.system.type_uploads` gives stored uploads the content type recorded in their context, in place and without keeping a version. It supports `--dry-run` and is safe to run more than once.

### Changed
- An invite grant created with an email address is now mailed to that address, with the optional message included. `email_sent` is false when the invite has no address, mail is not configured or the send did not go through. The invite still stands in those cases, and `claim_url` can be passed on.
- A member who claims an invite appears in the access list under the invite's name, or under the address it was sent to when it has no name.
- Uploaded files are typed as their file from creation, for example `text/markdown`. They render as that type and match `type:` filters. The type is kept when the upload completes.
- Deleting an already-archived artifact and restoring it from Trash returns it archived. `PATCH {"state": "live"}` still restores an artifact to live and takes it out of Trash.

## [0.1.5] - 2026-10-22

### Fixed
- The test that Trash is declared before `/artifacts/{artifact_id}` reads the order from the artifacts router itself: newer FastAPI keeps an included router as one entry in `app.routes`, and the test failed on GitHub's runners.
- The platform-operators migration loads the node's settings as Mantle's startup does, so it resolves the platform operator from the store rather than only from Origin; a release that redeployed Mantle alone could not run it.

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
- The platform provides its core operators as granted artifacts: reconcile accounts, answer from my notes, raise an Inbox item, list my collections and ask a model, in a "Platform operators" collection owned by the platform operator, which holds an owner's grant on each. Each is granted to every account one by one (read and invoke) at first sign-in, so each can be withheld or metered on its own; the release migration `mantle.services.seed_provisioning.platform_operators` grants every existing account. Each declares its inputs, and an input naming an artifact says so (`"format": "artifact-id"`, with the content types it accepts).
- `GET /artifacts/platform-operators` names the platform's own operators the caller may read, by id and slug, derived on the node, so an operator someone shares cannot pass for the platform's by what it says about itself.
- A typed link: `POST /artifacts` with `source_artifact_id` takes a `relationship`, which says what the linked artifact is to the one it hangs from and confers nothing. `GET /artifacts/{id}/children?links=*` (or a relationship's name, matching its `name.` sub-names) lists an artifact's links, each shown only to a reader who may read it; without `links`, links are not listed as members.
- `DELETE /artifacts/{id}/links/{relationship}` removes an artifact's typed links of that name and the names under it, and nothing else: no member is detached and nothing linked is archived, since a link is not a home. It needs `evict` on the artifact. Removing a link through the member route could archive what it named.
- A `contributor` role: read and add, and change nothing anyone else made. The only role that could add, `editor`, also rewrote and deleted another member's card, so a shared collection could not keep each contribution's author.

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
- A grant key carries what its owner holds now: each grant it carries is narrowed, at every use, to what its issuer holds on that resource, so a right the issuer loses leaves every key they issued too. It was clamped once, when minted.
- A key's owner may put on it any resource they hold something on, clamped to what they hold; it needed admin there. A timer's key reaches the operators its workflow runs, which an account holds as read and invoke.
- `GET /artifacts/platform-operators` names every operator the platform provides, with the persona tool it runs and whether the caller may invoke it, and answers a grant key too, so Crystal can refuse a withheld operator's tool however it is reached.
- Each version names the version it was made from (`parent_id`), so a lineage is a DAG a reconcile operator can join; no row records `superseded_by` any more, since a version may have more than one successor. A change of title keeps the version it replaced, as a change of body does: the title is part of the content (declared on the root type and inherited), and was overwritten in place with no trace. `revise()` treats the same content revised again as the version that already holds it.

### Fixed
- A card shared on its own, short of its collection, opens for the person it was shared with. Content keys are checked per collection, so the key oracle refused a reader who reached the card but not its collection: `/grants/my-access` said allowed and `GET /artifacts/{id}` answered 404. A key request now names the artifact being decrypted, and a grant reaching that artifact is enough for its key, by the same walk `check_access` makes; the key stays in Mantle, and the card's siblings stay closed.
- The maker of a card in someone else's collection edits and deletes it, as the owner's grant minted at its creation says. The update path asked for `update` on the container instead, by direct grant only: a contributor could not fix a typo in their own idea, and an editor of the collection could rewrite anyone's card. Authorization is now the routes' `check_access` on the artifact alone.

### Removed
- Commits: `GET /artifacts/{id}/commits`, the Commit and CommitItem entities and the two unreferenced service paths that wrote them. An artifact's changes are its versions. Stores that hold commit rows keep them hidden from artifact queries.

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
