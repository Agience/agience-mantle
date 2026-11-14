# Changelog

All notable changes to `agience-mantle` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.27] - 2026-11-13

### Changed
- `authority_ledger` now reads Origin's feed every 5 seconds instead of every 15, so a person who accepts an invite at Origin can open the shared workspace within seconds.
- Grants and revocations applied from Origin's feed take effect immediately: `authority_ledger` drops the grant memo as soon as a pass applies changes, rather than waiting for the memo's TTL.
- The `authority ledger: applied N change(s) from Origin` log line now counts every change applied in the pass.

## [0.1.26] - 2026-11-13

### Fixed
- A collection with many items lists again within the app's wait. Mantle checked a collection's Add seal on every read of its key, and since 0.1.25 each check parsed Mantle's signing key afresh, which takes tens to hundreds of milliseconds; a ninety-item collection took longer than ten seconds and showed as empty. The key is parsed once per process, and a seal already verified is not verified again.

## [0.1.25] - 2026-11-12

### Fixed
- The admin seed route (`POST /system/seed`) also provisions the platform operators on the calling admin's own credential, and grants them to every account, as the operator, only when this call creates them. A later call converges them and grants nothing, so a withheld grant stays withheld. On a fresh node, Mantle's startup ran before Origin answered and found no operator, so no account had operators. Its command-line migration holds no credential, and under custody gets no key. A failure here is listed in the response's `errors` and never fails the seed (John, 2026-11-11: the home node is seeded through the admin route, with no job grant).
- The seed command (`mantle.system.manage_seed`) loads the platform encryption key first, as the server does at startup. Without it every sealed field was refused, and a fresh store's seeded servers and operators were left without their forms and wiring.

## [0.1.24] - 2026-11-11

### Fixed
- A request refused by the rate limiter reaches the web app as a 429, not a CORS failure: the limiter answers outside the CORS middleware, so its answer now carries the CORS headers for an allowed origin itself, as the catch-all error handler does (BUG-056). On staging, a fast acceptance run read it as "blocked by CORS policy".
- A run that links a collection answers from the notes it holds (BUG-052). Sage lists a linked collection's children, and a narrowed Sage delegation was refused that route, so the collection read as empty and the run had no evidence. Sage may now list children. Each child is still decided by the caller's access and, on a run, by the job's reach.

### Changed
- The home node holds test custody while it is the test gate (John, 2026-11-11): with `AGIENCE_PLANE=home`, an issuer under `home.agience.ai`, no peers (`EMBER_PEERS`) and its own keys folder, `custody_plane` lets it hold custody over its own keys, so runs, timers and their approvals can be accepted there before promote. Every other plane but production is still refused, and production's own rule is unchanged.

## [0.1.23] - 2026-11-11

### Fixed
- A list of artifacts answers when a body in it can't be read (BUG-059): `/artifacts/visible` dropped the unreadable body and then decrypted the same row again, strictly, in `_normalize_artifact_doc`, so the page failed with a 500 after all. A list page now stays non-strict throughout, and a single read still refuses. On staging, which holds none of production's bodies, every typed listing failed, so the pickers read "No results".
- A note filed in a collection links as a workflow step's evidence (BUG-062). The link loaded the artifact strictly, so a body the caller holds no key to (Lumen's Add step runs on the persona's delegation) failed the call with a 500. A link writes an edge and needs no body, so it reads the artifact without refusing over it. A container's children list is non-strict, as every list is.
- A run that links a collection reads the notes it holds (John, 2026-11-11, BUG-052). A job delegation opens the key of an artifact its grant lists, or of one held under a listed artifact along edges that conduct a read, the walk access already decides a job's reach by. Writes still go to the drop key.

## [0.1.22] - 2026-11-10

### Changed
- The operators the platform seeds are readable platform configuration (John, 2026-11-10, BUG-057): their input form, wiring and description are kept in the open, and their title alone is sealed. A reader that can't open the title is shown the title of the operator's definition in the platform's catalog. A row sealed before this moves into the open on its next write. A person's own operator is unchanged and stays sealed, since its wiring may name a private server. On the release candidate, which holds none of production's keys, the seeder converges a copied operator it can't seal with nothing sealed, instead of failing, so staging shows every operator's form.
- "Answer from my notes" and "Read an item" take a note, a text file, a PDF, a collection or a workspace (BUG-052): the field names those types, so the picker offers single items as well as collections.

### Fixed
- A server error reaches the web app as itself, not as a CORS failure (BUG-056): the catch-all handler answers outside the CORS middleware, so it now adds the CORS headers for an allowed origin itself. A refusal to seal (Mantle never writes plaintext) is a 503 that says the collection's key isn't available on this server, not a bare 500.

## [0.1.21] - 2026-11-10

### Changed
- "Answer from my notes" is served by Sage's grounded `ask` and answers from the item linked to it (John, 2026-11-10, BUG-052): inside a workflow a run's grant reaches what it lists and never a search across everything, and Lumen could not reach Sage for its retrieval (one hop). It takes `question` and `evidence`; a step made earlier with `input` still runs. An operator may now declare its own argument mapping.

## [0.1.20] - 2026-11-10

### Fixed
- A list of artifacts answers when one of its bodies cannot be opened by the reader (BUG-051): `GET /artifacts/visible` failed whole (500) for a persona's delegation, which opens no person's content key, when one listed artifact's stored body could not be decrypted for it. A page drops that body from its row, as the boundary does for every list; a single read still refuses.

### Added
- A platform operator, "Read an item" (Sage, `get_artifact`): one item, named by a link. A scheduled workflow can read with Sage, since a timer's grant reaches the items it lists and no listing.

## [0.1.19] - 2026-11-10

### Fixed
- A timer's firing runs and is recorded (BUG-041). Crystal's gate asks `GET /grants/my-access` before running a timer's target, and a timer's job delegation could not reach that route, so every firing failed "upstream 403 from /grants/my-access"; it now may, and `check_access` still answers only within the timer's reach. Writing the run back to the timer was refused too: the seal asked for a key without naming the row, so a writer given no key could not fall back to the collection's drop key. It now names the row, as a read does.

## [0.1.18] - 2026-11-10

### Changed
- Beacon's head count (`derive_heads`) is capped by the geometry, `n_features // 2`, as context's is, in place of a stated ceiling of 64. A narrower head block has a cosine carrying only a sign. The `lo` and `hi` arguments are gone. No caller passed them.

## [0.1.17] - 2026-11-10

### Changed
- A timer's job delegation asks Origin for its owner's content key to read an artifact its job grant lists by id, and for nothing else (John, 2026-11-10): the timer's settings and its target are read whole on it, where before every sealed field was missing. Its writes still go to the drop key, and every other narrowed delegation still asks for no key.

## [0.1.16] - 2026-11-10

### Added
- A delegation that carries Origin's `job_reach` (a timer's, BUG-024 and BUG-035) reaches only those ids and what they contain, along edges that conduct the action, for create, read, update, invoke and add, and never past what the person holds; and only on routes that name the artifact they touch (`JOB_REACH_ROUTES`). It creates nothing at the top level and archives or restores nothing. A timer's own Lumen delegation may update what it reaches: the timer's state.

## [0.1.15] - 2026-11-10

### Fixed
- An edit never puts an artifact back in a container it was removed from (BUG-034). Every write re-placed the row's home membership when its edge was missing, so an artifact removed from its home and still held elsewhere came back there on its next edit (through the edit itself, and through the prior version's snapshot). A row now places a membership only when it is new to its lineage's collection, or names a new one; the home (`collection_id`, its creation edge at Origin) is kept for authority (John, 2026-11-10).

### Changed
- The personas the app now calls directly (FR-006) are narrow audiences here too (`DEFAULT_NARROW_AUDIENCES`): each `persona:<name>` scope reaches the platform operators' verdict alone. A persona's own server delegation reaches what its code calls: Aria, Astra and Ophan the operators' verdict, Seraph an artifact and a container's children. Seraph's archive is the person's, never its own delegation's: no write (John, 2026-11-10).

## [0.1.14] - 2026-11-10

### Added
- `python -m mantle.system.drop_grants_to --grantee <id> [--dry-run]` drops every grant to one account, archived ones included: for an account deactivated and retired. A plain store delete, for grants that never reached Origin's replica. The release tool runs it only for an account Origin shows deactivated.

## [0.1.13] - 2026-11-10

### Fixed
- Persona tool calls through Crystal work again (BUG-030). Crystal's operator gate asks `GET /artifacts/platform-operators` before every persona tool call, with the delegation the app sends the persona (narrowed to that persona, or to Crystal), and Mantle refused that delegation on every route ("this delegation's scope does not reach this route"); the gate failed closed and every call answered 503, a Timer's arm among them. Each `persona:*` scope now reaches that one read: the verdict of which operators the caller may invoke, never content and no write, as `ember:read` reaches its own verdict.

## [0.1.12] - 2026-11-09

### Fixed
- A person granted create on a shared workspace creates in the collections it holds (BUG-015). `POST /artifacts` with a `container_id` authorized the caller against a grant on that container alone, so a Collaborator on a shared workspace was refused with a 404 "Workspace not found" creating in its Backlog collection. The create now asks the light cone (`dependencies.check_access`, Create or else Add), as every read and update does; an in-process writer with no caller keeps asking the container's own grant. A reader of the workspace is still refused.
- Four lint issues in packaging and tests (an unused-placeholder f-string, a joined import, a lambda, an ambiguous name).

### Added
- `python -m mantle.system.drop_ungranteed_grants` drops stored grants that name no grantee (archived ones included); `--dry-run` counts them. Such a grant reaches no one, and Origin refuses it as malformed. It runs as a release migration.
- `cosign_authority` reports, for the grants it left out as touching ids Origin has seen, how many belong to each grantee (`left out: <n> grant(s) to <id>`), so the operator can tell throwaway accounts from people.

### Fixed
- Every grant names its grantee. The store refuses to write one without (`lattice_api.GrantWithoutGrantee`), and `POST /grants` refuses a direct grant whose request names no `grantee_id` (400). Two such grants were found in production when Origin's ledger was seeded, written by that route.

## [0.1.11] - 2026-11-09

### Fixed
- GitHub CI is green. The full suite installs agience-origin beside agience-prism, as an editable checkout: two review tests check Mantle's reach against Origin's own replica (`origin.services.authority_reach`), and the first-sign-in tests run against a real Origin with its own test helpers (`origin.tests`), which a built wheel leaves out; none could be collected without it. In the base-install job, the store tests that write a grant's sealed fields or a named link are marked as needing the node's data key (agience-prism) and get one where prism is installed; without prism, a grant with a field to seal is refused with the message naming what to install, and a test holds that.

## [0.1.11] - 2026-11-09

### Fixed
- Every call to Origin reads its answer once decoded. The bounded read (`origin_key_share_client._Bounded`) decoded the body and kept the answer's `Content-Encoding`, so an answer the edge had gzipped was decoded a second time and failed as `DecodingError`. Production reaches Origin through the edge: the authority-ledger seed of release 2026.11.4 stopped at its first entry, and live journaling lost entries the same way. On loopback, uncompressed, it never showed.

## [0.1.11] - 2026-11-09

### Changed
- `python -m mantle.system.cosign_authority` states only what Origin has not co-signed: an edge or grant whose receipt the store keeps is left out. `--ids` prints the ids the pending entries touch, and `--ids-stdin` states only entries whose ids are all on stdin. Accounts made since their authority is journaled put ids in Origin's replica, and Origin refuses a bootstrap that restates one; the seed now asks Origin which ids it has not seen and states those. The dry run also counts what is already co-signed.
- `cosign_authority` counts a row Origin refuses as malformed (400: a grant with no grantee, an id longer than Origin keeps) by kind and by reason, and goes on; such a row reaches no one. Any other refusal still stops the run.
- `cosign_authority` names by kind what it left out as touching ids Origin has seen (`left out, touching ids Origin has seen: <e> edge(s), <r> root(s), <g> grant(s)`).

## [0.1.11] - 2026-11-09

### Fixed (release candidate)
- `platform_operators` on the release candidate (`AGIENCE_PLANE=rc`) leaves, and counts, an account whose operator grant cannot be sealed there under a collection key production wrapped, and leaves the operators' collection as production has it when it cannot be converged there; on any other plane that still fails the migration.

### Security
- An invite mail carries one link, its claim link. The personal message, the inviter's display name and the resource's name are defanged in the subject, text and HTML (`services/defang.py`, the same rules as Origin's and the website's lead mail): URLs, markdown links, HTML anchors, bare domains, Unicode and punycode hosts and IPv4 addresses arrive as readable text no mail client links, and the HTML escapes everything. On 2026-11-09 an account sent a real Agience invite with phishing text in its message.
- An invite's personal message is at most 500 characters; a longer one is refused 400 naming the limit, before anything is recorded or mailed.
- The mail names its sender and their verified address ("Name (address) invited you to …"). Creating or resending an invite without an address the issuer verified (`email_verified`) is refused 403 before anything is recorded or mailed.

## [0.1.11] - 2026-11-08

### Fixed
- The search reindex counts an artifact whose index write failed only because its body could not be read here (`ContentStoreUnavailable`, a body it cannot sign an object-store URL for, an undecryptable seal), or was refused by a grant or key, as `unreadable` or `denied` rather than `failed`, whether the indexer raised that exception or logged it. On the release candidate, which has no object store, three type definitions had failed the `reindex` migration on exactly this. Any other fault still fails the pass.
- On the release candidate (`AGIENCE_PLANE=rc`), `upsert_user_collection_grant` leaves a grant name that will not open as it is and converges only the rights. The candidate runs on a copy of production's store under a key of its own, so every name production sealed reads as none there: the upsert saw a difference that was not there, set the name, and the write was refused, failing the `platform_operators` migration. On any other plane a seal that will not open is a fault, and the write still fails on it.
- `seal_titles` on the release candidate (`AGIENCE_PLANE=rc`) reports the titles it could not open and succeeds. The candidate holds keys of its own and no custody, so it can open none of production's sealed titles, and its rehearsal of `--unseal` failed on 612 of them. On any other plane a title that will not open still fails the pass.
- The test guard over the install-root store installs one audit hook per process; a later install adds its recorder to it. Under pytest-xdist, each worker's `sitecustomize` installed the guard before `conftest.py`, and the first hook refused before the second could record, so the detector test failed under `-n 4`.
- The detector test probes the store at its exact path everywhere, and spelled in another case only where the filesystem folds case. On Linux the upper-cased path names a different file, which the guard rightly lets through, so the test failed there.

### Fixed (local check, 2026-11-08)
- Mantle no longer loads or deletes `KEYS_DIR/setup.token` at startup. The token is Origin's, and a single-host install shares one `KEYS_DIR`, so Origin keeps it until setup completes.

### Changed (review 33)
- Offline recovery (`MANTLE_MASTER_KEY_RECOVERY=escrow` with the offline escrow identity) combines this node's share with the escrow share and asks Origin nothing, so it runs when Origin is down, refuses, or has lost its shares. A running node still refuses to start with the identity configured, and a refusal from Origin on a running node never falls through to escrow.
- `rekey_collections --rotate-again <root>` rotates a charged root onto a fresh key and kcv whatever its mark; it refuses a root the run does not rotate.
- On the production plane the server does not start without its lock; elsewhere it logs. Step 21 holds the lock for the whole rotation.

### Changed (review 32)
- A re-run of step 21 leaves a rotated charged root `already`, with its key and kcv, while none of its removed part's rows opens under that key; a split root is reported `split`, not `left`.
- `split_master_keys` refuses a root Mantle's own off-root test finds a removed part under and whose key carries no rotation mark, and counts a split whose activation Origin did not hear as failed (`activation_failed`), exiting non-zero.
- The server holds `<lattice>.server.lock` while it runs (`services/server_lock.py`); step 21 rotates nothing while another process holds it.

### Changed (review 31)
- Step 21 (`rekey_collections`) rotates charged roots' keys only with `--mantle-stopped`, with the ledger feed at Origin's head, and with Origin's `principal-kinds` answer; it exits non-zero without any of them, or when any charged root is left. It rotates every root Mantle's own keep/shred test finds a removed part under, as well as those Origin charges.
- The rotation re-checks each row's `_seq`/`_origin`, the rows under the old key and the key row inside its transaction, and leaves the root on any change; each root fails alone; the new key row carries the rotation mark (`rotated.kcv`), which the report gives for the operator to record at Origin.
- `split_master_keys` refuses a charged root whose key carries no rotation mark for that key, and every master key when Origin does not say which roots are charged (`refused_unrotated`).
- A drop (MED1) is under no master key: `rows_under_master_key` and the exposure counts leave it out.
- The `agience-prism` floor is 0.1.9.

### Changed (review 30)
- Step 21 (`rekey_collections`) rotates each charged root's master key: its own leftovers move under a new key, the removed part's are shredded with the old key; a root with nothing left has its key removed. `OriginShareCustodian.charged_roots` reads the charged roots from Origin.

### Changed (review 29)
- The Origin client passes a release's `pending` on, reading a missing one as pending.
- The owner-call re-key acts as the credential's own subject; an emptied key is removed only for Origin's `root` or `platform` answer, or with `rekey_collections --retire-member-keys`; the rekey report gives `left_per_root`.
- `exposure_counts` reports `rows_keyed_off_their_root`; the seed planner zeroes its index folders; the mail outbox names messages in sending order.

### Changed (review 28)
- The owner-call re-key finds rows by their content key principal (index `ix_v_ckp`), writes its done record only when none is left, counts refusals toward the batch and remembers them for ten minutes.
- A pending Origin share is replaced by a fresh split; only an activated one is completed.
- Only whole non-person master keys are removed; the split pass's exit code is the non-person gate alone; the seed planner overwrites its copy of the store before removing it.

### Changed (review 27)
- Key principals are classified by Origin's replica, not by an id's shape; a root's master key is non-person, split on the call of a person Origin admits for it, and what is under it is re-keyed on that call; the rekey pass removes each non-person master key that keys nothing any more.
- A split interrupted after Origin stored its share is finished with that share kept.
- Origin is told a split key is active, and the store file settled, only after the replacing row commits (`LatticeConn.after_commit`).
- The seed planner names its store and is isolated from Origin and the object stores before any `mantle` import and after the copy's settings load.
- The owner-call re-key scans by key principal, counts refusals as left, and reads its done record by kind and origin; the mesh applies no inbound `custody:` row.

### Changed (review 26)
- A whole collection key is split on the call of any person whose own credential Origin admits for it; only a person's own access token or an unscoped person delegation splits a key, never a persona's or Ember's.
- The split asks Origin's release rule first, stores Origin's share before writing the escrow share, and reads Origin's share back and recombines it before replacing the whole row; a refusal is a `ShareRefused`.
- The store's connections set `secure_delete`, and a split checkpoints the WAL after replacing a whole key row, so the old row leaves no bytes in the store file; a blocked checkpoint is recorded and counted.
- `split_master_keys.exposure_counts` counts by kind (whole person, non-person and collection keys, content under master keys, unsettled replacements); the pass exits non-zero while a non-person master key is still whole.
- The platform mail secrets are members of a platform collection the system principal administers, and the operator holds Update on them.
- Master-key and key-share rows never cross the mesh.
- Under threshold custody a person's content still under their master key is re-keyed onto its collection keys in the background of their own call, bounded and resumable; `exposure_counts` reports what remains and for how many people.
- A seed pass can be planned ahead: `python -m mantle.system.seed_plan` prints a pass's entries and digests, and a pass run with `MANTLE_SEED_PLAN_FILE` derives the same grant ids and submits only planned entries.

### Changed (review 25)
- A whole master key is split in place on its owner's own credential the first time they ask for it; `split_master_keys.exposure_counts` counts what remains whole.
- Telling Origin a key is active carries the acting credential.

### Changed (review 24)
- Comments state the seal rule as Origin now applies it (current Update authority).

### Changed (review 23)
- Under pytest the store and its content tier refuse to open the default `.data` location unless a store is named.
- A person's Origin token is taken only addressed to Mantle.
- A new credential's seals go to Origin after its creation edge; an update that changes a credential's secret reseals it and records the new seals.

### Changed (tester, 2026-10-31)
- `mantle.system.reindex` names each artifact a pass does not index, with its reason (`skipped`, `unreadable`, `denied`, `failed` or `ownerless`), up to 500, and its result carries them as `not_indexed`. Its last line is the summary.
- The index generation is 2, so the next release runs the pass once more. Generation 1 had recorded itself while an old note stayed findable only by its whole hyphenated name, and nothing named what that pass left out.

### Changed (review 22)
- A persona's delegation is taken only addressed to Mantle, and its narrowed bounds are the persona's.
- A credential write records its sealed-copy and endpoint digests at Origin on the writer's credential (`credential_sealing.record_at_origin`).
- `/use` requires the persona's `nonce` and forwards it to Origin.
- A token carrying `token_type` is refused as a bearer.

### Changed (review 21)
- `/artifacts/{id}/use` returns Origin's signed verdict (`verdict`) beside the sealed copy, asked on the persona's own delegation (`OriginShareCustodian.use_verdict`); with no Origin, or a refusal, nothing is returned.
- Invite mail and first-sign-in provisioning read a person record only on that person's own access token (`person_service.own_subject_token`), and ask Origin nothing without one.
- Ember's delegation addressed to Mantle (`aud` `mantle`, actor `ember`) is bounded as an Ember delegation (`dependencies.narrowed_for`).
- `EMBER_INVOKE_TOKEN` is noted as Ember's setting, read by the federation client in Ember's process.

### Changed (review 19)
- `/use` releases a sealed credential only on a delegation Origin signed. A delegation with `iss` `mantle` is refused at authentication.
- The reach digest is `prism.trust.ledger.canonical_reach` (expiry, invite target, claim limit). An applied invite keeps its target digest. A deny on a node deleted at Origin (noted when this store removes its creation edge) is left out, as at Origin.
- The catch-up reads to the head and writes each touched row's latest change over the whole ledger, under `ledger:catchup-review19`. A failure in it never stalls the feed.
- Custody checks the store folder too, reading each folder as itself, and offers `custody_refusal`.
- Every answer from Origin is capped at 4 MiB.

### Changed (review 18)
- The catch-up reads every page of Origin's feed, verified in one chain, before it works out the latest change to each row. It writes that latest change for every row a session touched, and runs again under a new marker for stores the one-page run marked. A short page leaves it undone.
- An expiry that does not parse is past.

### Changed (review 17)
- The feed applies `origin_session` entries, so changes people make in Facet reach this store. The kinds applied are `prism.trust.ledger.MANTLE_APPLIES`. A store whose cursor had moved past such entries catches up once (`catch_up_skipped`), applying each only while it is still the latest change to its row. The operator's `root.retire` drops the id's Add seal.
- The Add seal is `add-seal:<collection>`, bound to the collection key by a MAC. An Add-only write seals only to a seal it verifies against the collection key this node holds; otherwise it is refused with 503 "try again later" (`AddSealPending`) until a reader opens the collection. A legacy `drop-key:` row is never trusted.
- The deny walk follows the whole creation chain, as Origin's does, and a cached cell asks the store's verdict before it is served.
- The mesh plane must be Mantle's own content bucket.

### Changed (review 16)
- The Add seal is Mantle's own key material. It is cached when a reader opens the collection key and re-checked on every Read open, with an ALARM and a repair on a mismatch. Nothing is submitted to or read from Origin, and `drop_key_check` and the Origin comparison are removed.
- `DELETE /artifacts/{container}/children/{member}` resolves both ids to root ids before checking for a creation edge.

### Changed (review 15)
- `DELETE /artifacts/{container}/children/{member}` removing the member's creation edge needs Delete on the member, or Admin on the container, as Origin requires: it deletes the member at Origin. Evict still removes a secondary edge.

### Changed (review 14)

### Changed (round 13)
- `GET /grants/invite-context` and `GET /grants/invite-details` are removed: an invite is described and claimed at Origin, and no Mantle route takes an invite's raw token in its query.

### Changed (review 12)
- Issuer trust roots in configuration: the authority manifest's anchors always win, and Origin's own issuer and every pinned name are taken from the manifest alone; an issuer artifact never adds an issuer or a key, and only keeps a configured external issuer (`AGIENCE_TRUSTED_ISSUERS`) live until it is revoked. `POST /system/issuers` is removed, and no copy of the manifest's anchors is seeded into the store. A JWKS or discovery document is fetched only over https from an origin a configured issuer names or `MANTLE_JWKS_FETCH_ALLOWLIST` lists.
- With Origin's ledger installed, the grant routes (`POST /grants`, claims, accepts, revocations, grant keys and their members) answer 409: authority changes are made at Origin, in the person's Origin session. Every change made there (`origin_page`) is applied here from Origin's feed: grants, denies and invites, and containment edges with their masks.
- Every feed entry is checked before it is applied: its hash recomputed from its body, Origin's co-signature by the kid's key from the authority manifest's Origin anchor, and the chain from the last entry applied. One that fails is not applied and stops the pass, with an alarm.
- A persona's own server delegation exchanged from one narrowed to it (`narrowed` claim) is narrowed: it reaches only the routes that persona calls (`NARROWED_SERVER_ROUTES`), never a grant, an update or a delete.
- A cached key is served only on its requester's own verdict, the double-check under the oracle's lock included. Key-release verdicts are remembered per credential, and forgetting a person's verdicts also forgets the key-release verdicts and the key cache.
- Mantle's pages (/docs, the browser) carry a CSP forbidding framing, base tags and plugins, and `X-Frame-Options: DENY`.
- A credential's `allowed_endpoints` are stored decoded until stable; a path that decodes to a traversal, a backslash or a control character is refused.

### Changed (review 11)
- Ember makes no grant: an Ember delegation, whatever its scope, is refused at `POST /grants`, and `ember:share` reaches no route. Sharing happens only in the person's own session (John, 2026-11-02).
- A narrowed delegation is recognised from Origin's claims, not from its actor's name: a delegation addressed to itself as a narrow audience (`MANTLE_NARROW_AUDIENCES`, default Ember and the personas `iris`, `lumen`, `sage`, `crystal`) or to a page's own origin, or one carrying any scope that is not a platform purpose. A scope counts only for its own audience; an audience or scope Mantle does not know reaches nothing; a malformed `MANTLE_NARROW_AUDIENCES` treats every delegation as narrowed (fails closed). A persona's narrowed delegation reaches no Mantle route, and no narrowed delegation opens a key.
- An Ember delegation creates only plain memories (`text/markdown`, `text/plain`) in the caller's own memory collection: never a collection, a credential or another typed artifact.
- Tokens issued before Origin's published sign-in bound for their person, or everyone's, are refused (`prism.trust.not_before`), and that person's Origin verdicts are forgotten when the bound moves.
- A person's memory collection is made at any sign-in that finds it missing, not only the first, and never on a narrowed delegation.
- A deny on a collection, or above it, holds against every path to its key, a member's Read included (`OracleService._authorize`), matching Origin's release rule.

### Changed (review 10)
- Ember opens no person's key: on an Ember delegation no content key, master or collection, is asked for (`content_crypto`), whatever the person's grants. A memory Ember writes is sealed to its collection's drop key, written without Read, and indexed by no one until the person reads it.
- A person's memory collection for Ember is made at their sign-in, on their own token (`user_provisioning._ensure_ember_memory`), which gives it its key and its drop key; an Ember delegation never creates it, and writes only into it.
- An Ember delegation's read check is a verdict (`GET /grants/my-access`), never content; its share is one Read grant on one artifact (never a container or collection) to one named person (never the commons), expiring within 7 days (`grants_router.EMBER_SHARE_MAX_DAYS`). A persona page's narrowed delegation reaches no Mantle route; none needs one.
- A credential names the endpoints its secret may be sent to (`allowed_endpoints`, https URLs, fixed when it is written by its writer, who holds Admin on it by creation): they are sealed to each persona with the secret, stored beside it (`credential_endpoints`) and returned by `POST /artifacts/{id}/use`; no later edit moves them. `/use` no longer carries the unused per-card readers of secret names.
- The acting principal carries a delegation's scope.

### Added (review 10)
- `python -m mantle.system.rewrap_kek --new-kek-file PATH [--dry-run]`: re-wraps every stored key (whole master and collection keys, this node's share of each split key), re-seals every node-sealed field, every link name with its keyed tokens and every secret platform setting under a new KEK, verifies each by opening it again, and scrubs the old wraps from the store file.

### Changed (review 8)
- A delegation for Ember (`act.sub` `ember`) reaches only Ember's routes: the caller's memory collection and memories in it (`POST /artifacts`), `GET /artifacts/{id}` and Read grants (`POST /grants`); anything else is 403.
- No delegation is a platform admin.
- Threshold custody is claimed only by isolated production (`services.custody_plane`): Mantle refuses to start, and builds no threshold store, unless `AGIENCE_PLANE=production` and the process shows isolation (not root, `NoNewPrivs` set, its keys folder its own and 0700).

### Added
- Under threshold custody every action, not only Read, needs Origin's verdict as well as the store's grants (`authority_ledger.origin_allows`, through `GET /internal/authority/reach` on the caller's credential); an allow is remembered 30 seconds per requester, resource and action, a refusal not at all.
- Every 300 seconds the feed compares a digest of the store's active grants and containment edges with Origin's replica (`POST /internal/authority/verify`); a divergence is an ERROR alarm.
- Search is keyed per collection.
  - A search cell key is derived from its collection's key, with the owner bound in (`OracleService.derive_cell_key`).
  - Each collection has its own SSE key, derived from its collection key (`derive_sse_key(..., collection_id=)`). An artifact's blind tokens, posting entries and manifest (kept per collection) are under it.
  - A query spanning several collections opens each collection's key on its own (`TokenNarrower`). A collection whose key the caller cannot open (not written, not granted, or refused by Origin) contributes nothing. The master key is not used to search.
  - A key provider without `collection_id` (an embedding consumer's, one key per principal) is used as before (`sse.keys.collection_sse_key`).
  - `SseIndexer.remove_artifact` takes the artifact's `collection_id`, which the ingest pipeline passes.
- A collection's key is minted as it is created, after its creator's grant (`collection_service.ensure_collection_key`, for collections and workspaces). With no key oracle or no acting principal it is minted on the first write instead; under threshold custody that is refused.
- `mantle.system.rekey_collections` also moves the search index (`rekey_index`). Each cell is re-encrypted in place, after reading it back, from its master-derived key to its collection's. Each artifact's postings are written again under its collection's SSE key, then removed from under the principal-wide key, and a rerun finishes an interrupted move.
- An artifact's content is keyed under its creation parent's collection key, stamped on the row when it is created (`content_key_scope`, carried by the entity), as reach follows the creation edge. A move is secondary containment and does not re-key. A row written before the stamp keys to `collection_id`.
- Every collection's content is keyed under its own random collection key (`oracle.get_or_create_collection_key`), minted as the collection is created (below) and held in the key store as `ck:<collection>`. Under threshold custody it is split two-of-three like a master key. Content keyed under a collection is `MEC3` (`content_crypto`), its key derived from the collection key under the same scoped AAD. A `MEC1` blob under the master key still opens. Each collection key is cached under its own id for at most 300 seconds.
- `mantle.system.rekey_collections`, run at the cutover, moves each existing row onto its collection's key: the inline body, the sealed fields, and a content-tier copy. Prior versions are rows of their own and move the same way. The rewrite is in place (`put_artifact(..., snapshot=False)`), so the pass adds no version, and an old content-tier copy is removed once no row names it. The collection key is stored before anything is written under it. Each new blob is read back before the row is written, and the row is replaced in one write only if it is still the row the pass read. What is already collection-keyed is skipped, so the pass is resumable and a second run writes nothing. It reports progress (`--every`), counts (`--dry-run`), and every row it leaves under the master key with the reason (`left`).
- A seed scope (`*-seed`) runs on the operator's seed credential from `MANTLE_SEED_CREDENTIAL_FILE`, minted at Origin for that pass. Without one, a seed pass is background work, which can only narrow reach. One credential serves one pass (`system_identity.seed_pass`; startup's three seeds are one pass): when the pass ends, done or failed, Origin is told (`seed-done`) and the file is deleted.
- Under threshold custody, a `contains` edge or grant from a peer (mesh consume) that widens reach is held in `held_reach_change`, unwritten. A narrowing from a peer is written and journaled. `GET /held-reach` lists the held changes on resources the caller manages; `POST /held-reach/{id}/accept` writes one on the caller's own credential, and that write is the co-signed change (a refused acceptance stays held); `POST /held-reach/{id}/reject` drops it.
- Every change that affects reach is journaled at the store layer every writer goes through (`db.reach_journal`, stdlib only): a grant document written or deleted (`vertex._write_row`, `vertex.delete_artifact`), and a `contains` edge added, changed or retired (`edge._add_one`, `edge._retire_edge_row`). That covers the creation edge the vertex store writes for an artifact placed in a collection, mesh consume, the CLI passes, and the edges a delete takes with it. A new master key's principal and origin root are submitted by the oracle before it stores the key. Entries hold structure only: ids, the CRUDEASIO actions set, effect (`allow`, or `deny` for anything else, as the reach rules read it), state, expiry, an invite's target as a kind and digest (`target_digest`, taken before the address is sealed) and its claim limit, and the edge's decoded mask. They never hold a name, note, label, address or content.
- Recorded changes are submitted to Origin (`services.authority_ledger`, `POST /internal/authority/entries`) on the acting person's own credential, just before the write transaction commits (`seq.LatticeConn.write`). A refusal rolls the whole transaction back (`ReachChangeRefused`). An entry co-signed for a transaction that then fails is withdrawn by a compensating entry. Origin's receipt (`seq`, `entry_hash`, `signature`, `kid`) is kept in the store's `authority_receipt` table in the same transaction.
- A withdrawal Origin does not take is recorded (`authority_compensation_due`), retried before the next change to reach and at startup as the system principal's narrowing, and under threshold custody no change to reach is written while one is outstanding.
- Each receipt's ledger head (`head_seq`) is kept (`authority_receipt`, `ledger:head`, and in memory); a receipt naming a lower head is refused, since Origin's ledger then went backwards.
- A new master key's `root` entry carries the creation the store records for it (who made it, and when), which Origin requires for an id it has not seen.
- A grant's expiry is compared as time when deciding whether a change widens reach, and shortening a deny's expiry counts as widening.
- An entry Origin appended whose receipt is then refused (its head went backwards) is withdrawn.
- Changes to reach are co-signed outside the write lock (`seq.cosigned` on `put_artifact`, `put_many`, `delete_artifact` and the edge writes). The call runs once to learn its changes and is rolled back, Origin is asked with no lock held, and the call runs again and commits only if it makes the same changes against the same co-signed rows. Otherwise the entries are withdrawn and it runs again; after five tries it raises `ReachChangeConflict` with nothing written. Any other write block that changes reach is recorded and replayed only if no other connection committed meanwhile. Ordinary writes no longer wait for Origin.
- A retried withdrawal names the entry it withdraws, so Origin takes it only while that entry is the latest change to its row; a 409 means it was superseded, and it is cleared.
- A creator who does not already hold the owner's rights on the container gets no creator grant on what they add, and none is asked of Origin: the item is the collection's owner's. An inbox sender's add succeeds without reading the inbox.
- An invite is no principal's grant until claimed: `get_active_grants_for_principal_resource`, and so `check_access`, no longer counts an invite whose grantee id equals a principal's id.
- The cutover pass leaves a row to do when the content tier takes no copy, and removes the new copy it made for a row changed underfoot.
- Each co-signed entry is confirmed to Origin once its write commits, outside the write lock; one Origin does not hear is recorded (`authority_confirm_due`) and told again.
- A withdrawal answered 409 is cleared only when the row's latest entry at Origin is one this store committed; otherwise it stays due.
- A replayed write block is checked against the store's write mark, which writes on this connection move too, and no transaction is open while Origin is asked.
- A first key share is stored only on a credential Origin accepts for it; without one the store refuses before writing anything.
- CRUDEASIO (architecture §51): no action but Read releases a content key. The oracle checks Read for every content key, the create base case holds only while no grant names the collection, and the use route no longer widens the key demand to Invoke.
- Writing without reading is an Add grant. Each collection has a public Add seal (X25519, its private half derived from the collection key, cached as `drop-key:<collection>`). A writer holding Add or Create and not Read seals what they write to it (`MED1`, principal, collection and artifact bound); the first reader re-keys it under the collection key and indexes it as themselves.
- `create_workspace_artifact` accepts Add (intake) as well as Create.
- Each operation's changes to reach go to Origin as one batch (`reach_journal.batch`), and a container's root is registered in the batch with its creation grant; a seeded platform collection likewise.
- An `unchanged` receipt is never withdrawn or confirmed, and a withdrawal Origin answers 410 (resolved by the operator) clears.
- The combined-key cache holds a key at most 30 seconds, the canon's verdict memo TTL.
- The cutover pass gives each rewrite its own object-store key, removes object-store copies no version names, and re-files postings for heads only.
- `add_artifact_to_collection` never makes a second creation edge nor demotes one; target-collection links are secondary containment; an identity upsert into another container is refused.
- The held-reach listing asks once per resource (`held_reach_change.resource_id`).
- A person's card is their own root: first sign-in creates it as the person, on their own credential, with their own creation grant, in one Origin-issued batch (`workspace_service.create_container`), placed in no collection and granted to no one else. The People directory is no longer created or written by provisioning, a card is no longer moved into it, and no owner grant on an old card is widened. A member whose old card holds `person_artifact_id` gets their own card at an id of its own (`own_card_id`). A sign-in no longer creates the Authorities collection.
- `GET /system/users` lists Origin's own person records (`OriginShareCustodian.platform_users`, on the caller's own credential), each with this node's admin verdict; Origin decides who may list. A node with no Origin configured answers 503. The People directory is not read.
- `mantle.system.people_to_signups [--dry-run]` makes the old People directory the platform's sign-up collection: it renames it and revokes each member's grants on their old card, as the platform's system principal. It deletes nothing and is resumable and idempotent.
- A container's root and a key's root, submitted to Origin outside a store transaction, are confirmed once their write has landed (`reach_journal.confirm_submitted`).
- `split_master_keys` activates each key's share at Origin once the split key reads back, so it is no longer replaceable.
- The root `conftest.py` keeps every test off the checkout's `.data/`: an open there is refused where it happens (so it fails under xdist too), compared case-insensitively.
- Batches are Origin's: the first receipt of an operation names the batch the rest of it carries.
- A shared credential is sealed to its personas when written (`persona_sealed`); the use route hands the persona its sealed copy and reads the card as the persona, so no key is opened as the caller.
- Under threshold custody a cached key is used for a requester only after Origin's verdict for them, memoized 30 seconds.
- The ledger feed is read every 15 seconds and the operator's revocations and edge removals are applied here.
- Removing a member from a container takes Evict or Admin.
- The cutover pass counts object-store references from parsed contexts and re-files postings for archived heads.
- Origin's share of a key is told when this node activates it.
- Under threshold custody the ledger is enforced. A change Origin refuses, or cannot be asked about, is not written. A change to reach with no ledger installed is refused at the store, whatever wrote it. A ledger that cannot be installed is fatal at startup. Under lattice custody the ledger is recorded: the change is written either way and a refusal is logged. `MANTLE_AUTHORITY_LEDGER=off` sends nothing, and is refused under threshold custody. `manage_seed`, `manage_bootstrap` and `seal_titles` install the ledger for their pass.
- An invite claim sends its claim token to Origin as proof (`reach_journal.proof`), which Origin checks and does not store.
- `mantle.system.cosign_authority` co-signs the authority this node already holds at Origin, once, on the operator's bootstrap credential (`--bootstrap-token-file`), and ends that credential at Origin when the batch finishes or fails: every containment edge, then every key principal's root, then every grant, as bootstrap entries, keeping each receipt. A member with more than one creation edge is submitted with the one `get_origin_parent` reads, and the others as secondary containment; the pass reports how many. A re-run appends nothing, and `--dry-run` counts and sends nothing. It runs before threshold custody is turned on and before `mantle.system.split_master_keys`.
- Under threshold custody each share request to Origin carries the acting caller's own credential (`X-Acting-Credential`) and names the collection, and the artifact's root, the key is wanted for (`ThresholdMasterKeyStore.get_scoped`, which the oracle calls). Origin releases the share only when that credential's subject reaches the key by its own replica. A share is replaced only with such a credential.
- The request boundary (`get_auth`, the events socket) carries the caller's bearer in the acting principal when it is an Origin-issued person token or delegation (`acting_principal.ForwardedCredential`, shown only as a hash in any repr). A block run as the principal already acting keeps that credential (`acting_as`), so provisioning at a person's first sign-in acts on the person's own credential.
- Background work, and work acting as the platform system principal, shows a delegation Origin mints for the one principal being opened (`OriginShareCustodian.background_delegation`, reused until a minute before it expires). Origin mints one only while the operator's background authority is in force. Nothing long-lived is kept on this node.
- Under threshold custody a key is refused, warm or cold, to a caller with no credential Origin issued. That covers a grant key (`GrantKeyCannotOpenShare`), an external sign-in, an mcp_client or server token, and background work when Origin mints no delegation. Each refusal names why. A refusal from Origin never falls through to escrow recovery (`ShareRefused`); only an Origin that cannot answer does.
- A combined master key stays in the oracle's cache for at most five minutes (`MANTLE_MASTER_KEY_CACHE_TTL`, 300 seconds by default and at most), after which the store, and under threshold custody Origin, is asked again.

### Changed
- A shared credential's use (`POST /artifacts/{id}/use`) takes two ordinary grants: the caller's Invoke on the credential (and on the card), and the persona's own Read on its credential, granted by the collection's Admin when the credential is shared with it. Mantle checks both by grant, reads the card and the credential as stored for their type and id only, acts as no persona, mints no delegation for one and opens no key; it hands the persona the copy sealed to its key, which the persona opens itself. There is no `persona.use` job.
- A narrowed delegation (Origin's token exchange: Ember, or a persona page) reaches on every route only what its scopes list: `ember:memory` `POST /artifacts`, `ember:read` `GET /artifacts/{id}`, `ember:share` `POST /grants`, `persona` reads (`GET /artifacts/{id}`, its children and content, `POST /artifacts/recall`). A narrowed delegation with no or an unknown scope reaches nothing.
- A container's root is co-signed before its row is written, so a root Origin refuses leaves no row; Origin's verdict on a read is remembered per credential, so a narrowed delegation never rides on a verdict its subject's own token earned.
- Every background delegation names its job (the acting block's scope), and a block naming none is refused under threshold custody: the ledger's retries run as `ledger.retry`, the rekey pass as `platform.rekey`, the content-tier tool as `platform.content-tier`.
- Mantle answers cross-origin requests only from the origins `MANTLE_CORS_ORIGINS` names (comma-separated), defaulting to the web app's address `FACET_URI`; never `*`.
- Under threshold custody the server refuses to start while `MANTLE_ESCROW_IDENTITY_FILE` is set; recovery runs offline (`split_master_keys --combine` with `MANTLE_MASTER_KEY_RECOVERY=escrow`).
- `split_master_keys` and `rekey_collections` leave no replaced bytes in the store file: once they have rewritten anything they turn on `secure_delete`, `VACUUM` the file and truncate the WAL (`db.scrub`).
- The BLAS negative control runs its child with every thread cap removed, so it passes in a suite run under `OMP_NUM_THREADS=1`.
- The feed's cursor stops before an entry it fails to apply.
- A failed confirmation of an entry submitted outside a store transaction is recorded in `authority_confirm_due`.
- `people_to_signups` names the member by the card's derived id, and also revokes the grants the member issued on their old card and closes its invites.
- The rekey pass checks an artifact's own versions before it removes an object-store copy a stale count says no row names.
- The test guard also refuses sqlite `file:` URIs, paths through junctions and symlinks, subprocesses aimed at the store, and opens in child Python processes.
- The vertex store writes one creation edge per member, ever. An artifact first placed in a collection gets `is_origin`. An existing artifact that names a new collection (a move) is placed there as secondary containment, so a move never re-roots it. An existing membership is left as it is, so a later put no longer resets its `propagate` mask to unrestricted.
- `lattice_api` wrappers that read a failed write as `False` raise `ReachChangeRefused` instead of absorbing it.

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
