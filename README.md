# Agience Mantle

[![PyPI](https://img.shields.io/pypi/v/agience-mantle)](https://pypi.org/project/agience-mantle/)
[![Python](https://img.shields.io/pypi/pyversions/agience-mantle)](https://pypi.org/project/agience-mantle/)
[![License](https://img.shields.io/pypi/l/agience-mantle)](LICENSE)
[![CI](https://github.com/Agience/agience-mantle/actions/workflows/ci.yml/badge.svg)](https://github.com/Agience/agience-mantle/actions/workflows/ci.yml)
[![Sponsor](https://img.shields.io/badge/Sponsor-Agience-EA4AAA?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/Agience)

**The lattice — an encrypted artifact store and search engine where authorization is the
encryption.**

Every artifact carries its identity, version history and provenance inside itself, so audit is the
data structure for everything written through the system. Access control is enforced
cryptographically: each cell of the index is encrypted under a per-cell key derived from the owner's
master key, and authorization is computed as reachability in a typed graph — the *light cone*.

Reachability decides **which keys are issued**. The cell key is derived on demand and never
persisted, so there is nothing at rest to take, and one piece of code decides both what a search may
touch and what key is issued. Revocation is a single grant edit — no re-encryption, no key rotation
— effective within the authorization cache's window, 30 seconds by default and disableable outright.

*Grants are keys, not metadata.*

**What is encrypted, and what is not.** Content is encrypted per principal and bound
cryptographically to the collection it was written for. What the store keeps in the clear is what it
must read to *find* things: the offer text the lexical index reads, and the identifiers in the
posting store. The blind-token index closes exactly that, and covers **5.9%** of our reference
corpus today, with the remainder served by a plaintext lexical index.

Durability is an operator responsibility — [Backing a node up](#backing-a-node-up) is the procedure.

## Run it

**Mantle is the database.** The store is one SQLite file (`MANTLE_LATTICE_PATH`, schema created on
open) plus a filesystem CAS, opened in-process — zero external database processes to provision.

```bash
pip install 'agience-mantle[service]'

# A fresh KEYS_DIR does not boot: the lifespan loads key material. This writes a
# throwaway development keyset.
mantle-init-keys --keys-dir ./.data/keys

AGIENCE_BASE_DIR=$PWD KEYS_DIR=./.data/keys MANTLE_LATTICE_PATH=./.data/mantle.db \
  mantle-serve --port 8081
```

An install puts four commands on the path:

| command | what it does |
|---|---|
| `mantle-serve` | wraps uvicorn, so the app object, the default port and the log config travel with the package. Needs `[service]` |
| `mantle-init-keys` | writes a development keyset — the signing key *and* the trust anchor that verifies it, so a node trusts its own key from first boot |
| `mantle-token` | mints a user token against that keyset, offline, over `KEYS_DIR` |
| `mantle-wal-checkpoint` | reclaims the write-ahead log without a restart, for a node that has been up for weeks and cannot be bounced. It needs the readers quiet and exits non-zero when they were not |

**Set `AGIENCE_BASE_DIR` on a pip-installed node.** It is the root every derived default hangs off —
the SSE index, the encrypted cells, the embeddings cache, `KEYS_DIR` and `MANTLE_LATTICE_PATH`.
Unset, an installed node derives it from **the directory it was started in**, so starting the same
node from elsewhere brings it up healthy on an empty universe. A checkout keeps deriving the repo
root, so `.data/` stays beside `src/` when developing here.

Mantle boots as a pure database layer with an empty type registry; an application on top provisions
data through the API.

### Connect a client

A standalone node is a complete node — one keyset, one file, no bootstrap step. In a second shell:

```bash
mantle-token --keys-dir ./.data/keys
```

```text
Minted a user token, signed by ./.data/keys/mantle.private.pem and trusted by this node's own
authority.manifest.json anchor. DEVELOPMENT ONLY - these keys have no custody.

    subject   d99d859b-876e-57f6-b196-7b22fa54335c
    audience  http://localhost:8080   (config.AUTHORITY_ISSUER - what the verifier requires)
    expires   2026-08-12T13:55:49Z   (12 hours)

Add it to Claude Code:

    claude mcp add --transport http mantle http://localhost:8081/mcp \
      --header "Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6Im1hbnRsZS0xIiwidHlwIjoiSldUIn0..."
```

Three values in that block are read from the code rather than chosen by the command. **`audience`**
is `config.AUTHORITY_ISSUER`, the same attribute the verifier compares against — a *name* a token
must carry, not a host anything dials. **`expires`** is
`services/auth_service.ACCESS_TOKEN_EXPIRE_HOURS`. **`subject`** is
`uuid5(instance.uuid, "mantle/local-user")`, derived from the keyset, so re-running mints for the
same person and everything the last token stored stays reachable. `--subject <label>` names a second
identity on the same keyset.

Paste the `claude mcp add` line and the client is connected. The same thing over `curl`:

```bash
TOKEN=$(mantle-token --keys-dir ./.data/keys --token-only)

curl -s http://localhost:8081/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"create_artifact",
       "arguments":{"name":"Kickoff notes","content_type":"text/markdown",
                    "content":"We agreed to ship the encrypted lexical arm first."}}}'
```

```json
{"jsonrpc":"2.0","id":1,"result":{
 "structuredContent":{"id":"df65a79f-1a57-4a5b-b10d-62f4032557fc","name":"Kickoff notes",
   "content_type":"text/markdown","state":"committed","collection_id":"",
   "created_by":"d99d859b-876e-57f6-b196-7b22fa54335c",
   "created_time":"2026-08-12T01:56:01.828633+00:00"}}}
```

`created_by` is the token's `subject`, so the credential really is the principal — the creator gets
an owner grant, which is why the next call can find it.

**Send `content_type`.** Omitted, it defaults to `application/vnd.agience.collection+json`, the
label for a *container*, so a stored conversation comes back as a collection and every `type:`
filter that would have found it misses.

**Send `identity` for anything you will store more than once.** It names the *thing* the artifact is
of — `file:/repo/README.md`, `session:7c7bcb7b` — and the artifact's id is derived from it
(`services/artifact_identity`), so the write is idempotent: storing the same thing again updates that
one artifact.

```bash
curl -s http://localhost:8081/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"create_artifact",
       "arguments":{"identity":"file:/repo/NOTES.md","name":"Notes",
                    "content_type":"text/markdown","content":"Second revision."}}}'
```

Run that twice and there is one artifact holding `Second revision.` Without `identity` the id is a
fresh `uuid4` per call, so a client must remember the id it was given, and a write whose reply is
lost still succeeds — leaving the next write to create a second root.

The derivation includes the calling principal, so the same name from a different principal is a
different artifact: two people capturing their own `README.md` cannot collide, and converging on one
artifact stays a deliberate act — a grant. `identity` is top-level only; combining it with
`container_id` is a 400 naming the reason, because a member of a collection has a draft/committed
lifecycle with more than one live version.

```bash
curl -s http://localhost:8081/mcp \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"recall",
       "arguments":{"query_text":"encrypted lexical arm"}}}'
```

```json
{"jsonrpc":"2.0","id":2,"result":{"structuredContent":{
  "hits":[{"id":"df65a79f-1a57-4a5b-b10d-62f4032557fc","score":3.0,
           "root_id":"df65a79f-1a57-4a5b-b10d-62f4032557fc","collection_id":"",
           "title":null,"description":null,
           "content":"We agreed to ship the encrypted lexical arm first."}],
  "total":1,"ordering":"coverage","applied_filters":[],"corrections":[],
  "query_text":"encrypted lexical arm","size":20,"from":0}}}
```

Results are under **`hits`**. `ordering: "coverage"` is the lexical arm answering, and `score` is the
count of distinct query stems that hit carries. `title` is null because `name` and `title` are
different fields — `title` comes from the `context` JSON.

**`KEYS_DIR` is the root credential of a standalone node.** Read access to it is full access to the
store, bounded by no grant, no revocation and no expiry. `mantle-token` names that exposure rather
than creating it. Back the directory up separately and under different custody.

**Authenticate with a static `Authorization` header.** A standalone node serves one document of the
OAuth surface, `/.well-known/oauth-protected-resource`, and `config.authorization_servers()` names
an authority once one has been declared — an `AUTHORITY_ISSUER` or `ORIGIN_URI` in the environment,
a `branding.origin_uri` row that differs from the default, or a configured
`AGIENCE_TRUSTED_ISSUERS`. Point a node at a real issuer and the document names it.

### Semantic recall, and seeding an AnchorSet

A node you just installed answers `POST /artifacts/recall` from the lexical arm. Both arms are wired
and both have somewhere to write — the encrypted vector cells work on local disk with no bucket —
but the semantic arm routes every vector against an **AnchorSet**, the shared coordinate system, and
a fresh node has none.

The contract is three steps:

1. **Seed a set.** `python -m mantle.system.manage_anchors --action load --path anchors.json`
2. **Send query vectors in that set's space** — `space_id` must equal the set's `model_id`, and the
   width must equal its `dim`.
3. **Read ranked results.** `POST /artifacts/recall` returns them with `ordering: "semantic"` and a
   cosine on every hit.

The client owns the coordinate system end to end: it authors the set, it names the space, and one
node serves exactly one space. A query in any other space is refused with a 400 naming both.

An anchor id is content-addressed over `(label, model_id, embedding)`, so anchors fitted to whatever
corpus a node happens to hold would mint region ids no other node computes — two deployments each
routing confidently into disjoint cells. Vectors therefore arrive from a caller; this process runs
no model and fits no projection between spaces.

Until a set is seeded:

| path | what happens |
|---|---|
| an artifact write | succeeds and is indexed **lexically**; the vector arm returns `skipped` and logs a `WARNING` naming the missing AnchorSet (`search/ingest/pipeline_unified.py`) |
| `POST /artifacts/recall` | narrows on the query's terms, then answers most-recently-updated first: `ordering` is `"recency"`, `score` is `null` |
| `POST /artifacts/recall` with `vector` + `space_id` | **400.** This node ranks in no space, so the vector names one that does not exist here. The message names both ways out: seed the set, or send the recall without `vector` |
| a full reindex | runs, and reports `"vector_arm": "off (no AnchorSet)"` |

**Check which state a node is in** — `KEYS_DIR` must already hold a keyset, since the command derives
the platform system principal from it:

```bash
python -m mantle.system.manage_anchors --action inspect
```

It names the live set's anchor count, model, dimension and **fingerprint** — a hash over the anchor
ids, so two operators can establish that their nodes route into the same regions without either
exporting an anchor, a label or a vector. `/status` carries the same value.

**Seed it.** An anchor **is an artifact** (`application/vnd.agience.anchor+json`) and the AnchorSet
**is the collection** of them (slug `agience-anchorset`, created on first use):

```bash
python -m mantle.system.manage_anchors --action load --path anchors.json
```

`anchors.json` is the single-file form `AnchorSet.save`/`load` emits. `--dry-run` verifies the file
and reports its fingerprint without writing. Reindex afterwards so already-stored artifacts reach
the vector cells.

**Load anchors with that command rather than `POST /artifacts`.** An anchor's id is
`uuid5(sha256(label ‖ model_id ‖ embedding))`, and that id **is** the cluster id — it names the cell
storage path, the HKDF key `info`, the AEAD associated data and the mesh region. The write path
assigns a fresh `uuid4`, so posting anchors one at a time replaces exactly the value that makes two
nodes' cells comparable: queries route to regions the writer never produced, cells miss, the request
still answers `200` on lexical results, and mesh sync transfers nothing and reports success. The load
command preserves every id and verifies each against its own content, refusing a file whole if any
anchor disagrees.

| symptom | the error names |
|---|---|
| recall comes back `ordering: "recency"` with null scores | no set is seeded — `--action inspect` says so and gives the load command |
| `400` on a query with `vector` | the width is not the set's `dim`, or `space_id` is not its `model_id`. The message names the expected value and both spaces |
| `REFUSED:` from `--action load` | an anchor's stated id disagrees with its own content. Re-export with `AnchorSet.save`; the ids cannot be repaired by hand, because the id **is** the content hash |
| the arm refuses after it worked | the seeded set is a different space from the one this store's cells were written under. Restore that set, or drop the cells and reindex |

## The surfaces

Everything is an artifact, so most of what follows is a `content_type` rather than a new plane.

**HTTP.** Five routers: `/artifacts` (CRUD, children, commits, content, and `POST /artifacts/recall`
for ranked and candidate-set retrieval), `/grants` (grants, invites, grant keys and key bundles),
`/events`, `/system` (the admin namespace behind one predicate — issuers, users, seed, admin grants,
and `POST /system/erasure/{person_id}`), and `/mcp`.

`/mcp` is Model Context Protocol over Streamable HTTP, and the surface is round: `create_artifact`
stores and `recall` finds, beside `update_artifact`, `delete_artifact`, `list_artifacts`,
`get_artifact` and `get_children` — seven tools. Every tool dispatches into the REST handler that
owns the verb with the caller's own principal, so the write is the one `POST /artifacts` makes and
the search is the one `POST /artifacts/recall` makes, including the field filters, the coverage
ordering, and every 400 and 401 verbatim.

`/docs` and `/openapi.json` are always served: the schema is the API's contract, and every route
behind it enforces its own authorization. A secret is a `vnd.agience.credential+json` artifact whose
value is its content, so the envelope encrypts it at rest and the light cone decides who may read it
— one cipher and one authorization path, the same two every other artifact gets.

**The change feed.** Every artifact write emits an event at the persistence chokepoint
(`db/doc_boundary.py`), so coverage is complete by construction. `event_bus.py` fans out in-process
and appends to a durable log a consumer resumes from by cursor; a subscription is an artifact
(`vnd.agience.subscription+json`) carrying a filter and that cursor, created and shared through
`/artifacts` like anything else. `event_backplane.py` adds optional Redis or MQTT distribution across
processes — unconfigured is a supported configuration, and the app refuses to boot multi-worker
without one rather than dropping events quietly. Live delivery re-runs the ACL filter in every
receiving process, so a back-plane carries signal and never authority.

**The context lattice.** A context is an artifact too (`entities/context.py`), composed over context
edges by one bounded, attenuating walk (`services/context_service.py`). The walk takes a **required**
authority ceiling — the id universe it may not leave — so a context edge only ever narrows;
`UNCONFINED` is a named opt-out for callers wanting the lattice's shape rather than an authorization
answer, and there is no default. It is additive: cell-key derivation stays
`HKDF(master, collection_id ‖ cluster_id)` and no ciphertext moves. `services/dependencies.check_access`
— the gate in front of every read — walks containment only, so the resolver holds the context walk to
the grant-derived set. Two answers to "what may this principal reach" that disagree is a key issued
for an artifact the gate would then 404.

**Vector ingress.** An artifact create or update may carry an optional `vector` plus the `space_id`
it belongs to; `space_id` is required whenever `vector` is present, because two vectors are
comparable only within a named space. `api/vectors.py` validates shape — finite numbers, a bounded
positive dimension, a non-zero norm. `POST /artifacts/recall` takes the same pair as the reader's
half of that seam.

**Query syntax.** `query_text` carries terms and filters together. Terms: `+term` (required),
`!term` (excluded), `~term` (selects what gets embedded), `="phrase"` (exact).

| | |
|---|---|
| Filterable | `id`, `root_id`, `collection_id`, `content_type` (alias `type`), `owner_id`, `title`, `description`, `tags` (alias `tag`), `created_at`, `updated_at` |
| Operators | `field:value` (case-insensitive; `a,b` is any-of) · `field:="Exact Value"` (case-sensitive, whole) · `!field:value` · `field:>value` / `field:<value` on `created_at` / `updated_at` only |
| Combining | filters conjoin; each additional one narrows further |
| Refused, with a 400 naming it and why | `state:`, `content:`, `size:`, `filename:` · `field:~value` · a range on an unordered field · a query of nothing but filters |
| Not a filter at all | any other word — it searches as an ordinary term |

**`word:value` is a filter only when `word` is a field on one of those two rows**, so
`https://example.com`, `meeting at 3:30`, `C:\Users\example` and `ratio 16:9` are ordinary searches.
The parser holds no field list of its own — it asks `search/field_filters.is_filter_field`, the same
roster the resolver resolves against. The cost is that a **misspelled field is a search term**:
`titel:foo` searches for the literal text and finds nothing, so check the Filterable row when a
`field:value` query returns nothing.

A filter resolves to a set of artifact ids and is intersected with the light cone **before**
retrieval, so both arms honour it identically and `total` and pagination count filtered matches. It
can only narrow: the predicate is shown docs of authorized artifacts only, so a filter naming an
unreadable artifact is indistinguishable from one matching nothing. Filterable is everything a doc
plainly carries; `content` is encrypted at rest, and `state` selects the index segment — a
separately keyed tree chosen before the query runs. Filter tokens never reach the index, and
`applied_filters` on the response lists what actually narrowed the result. Quoting forces a term
either way, which is how to search for a field's name literally (`"type:pdf"`).

## Security invariants

Four properties hold across the codebase, are asserted by tests, and must survive every change:

1. **Geometry never authorizes.** Embeddings and routing rank *within* an already-authorized
   candidate set. The routing path receives no key material and runs strictly before any key request.
2. **Authorization is decided only by the light cone and grants, and the light cone is bounded above
   by them.** Access is default-deny, and even the creator holds an explicit, revocable grant. Two
   tighter statements hold inside this one, both structural:
   - *The resolve cannot exceed the read gate.* The context walk is confined to the grant-derived id
     set, so `resolve(principal, action) ⊆ grants-alone(principal, action)` is a property of how the
     call is made.
   - *A grant on one artifact means one artifact.* Recall cuts twice: a posting entry must clear the
     collection cut and the artifact-granular cut from the same resolve. An empty authorized set and
     an absent one are distinct — empty returns nothing, never the whole scope.
3. **Ciphertext is bound to its identity.** Every cell is AEAD-encrypted with associated data bound
   to its context and cluster, so a blob presented under the wrong key or moved to the wrong slot
   fails authentication before deserialization.
4. **Revocation requires no re-encryption.** Removing a grant alone prevents routing to, deriving
   keys for, and decrypting the affected cells.

**Composition along a path is monotone and non-amplifying**, and one operator makes it so.
[`src/mantle/attenuation.py`](src/mantle/attenuation.py) holds the meet: a bounded meet-semilattice
over the CRUDEASIO action set, with an absorbing deny and a full-authority identity. Both storage
encodings — the `edge.propagate` TEXT column and `Grant`'s nine `can_*` booleans — are codecs onto
the same `Mask` type and round-trip through it, so the light-cone walk and a grant's own mask cannot
disagree about the zero element. `tests/test_attenuation_algebra.py` proves the laws exhaustively and
`tests/test_attenuation_is_single_sourced.py` sweeps `src/mantle` by AST for a second implementation.

On the data plane Mantle serves and verifies rather than asserts: it never fabricates provenance and
never embeds on its own behalf. It signs exactly one thing
([`services/peer_signing.py`](src/mantle/services/peer_signing.py)): a short-lived, audience-scoped
service JWT saying "Mantle is calling", used on its one outbound peer call. RFC 8693 delegation is
inbound-only — `services/dependencies.py` accepts one, resolves it to the subject with the acting
server recorded in `actor`, and the peer that issues it is the authority issuer.

## Layout

| path | what it is |
|---|---|
| [`db/`](src/mantle/db/) | the lattice store. `backend.py` is the one import point → `lattice_api.py` → `vertex.py` / `edge.py` / `seq.py` / `schema.py` (SQLite + filesystem CAS), plus the object-storage content adapter. `doc_boundary.py` is the write chokepoint: content envelope crypto and the change event, in one place |
| [`routers/`](src/mantle/routers/) | the five FastAPI routers `main.py` mounts — `artifacts`, `grants`, `events`, `system`, `mcp`. Thin and type-agnostic: validate, delegate, return |
| [`services/`](src/mantle/services/) | orchestration — workspaces, collections, grants, content, contexts, OIDC, seed provisioning, and `peer_signing.py`. `content_crypto.py` is the per-principal content envelope; `acting_principal.py` answers who is acting, `principal.py` what artifact a principal *is* |
| [`api/`](src/mantle/api/) | Pydantic request/response models by domain, including `vectors.py` |
| [`entities/`](src/mantle/entities/) | entity models and serialization. A collection *is* an artifact; `collection.py` says so literally, and `context.py` and `subscription.py` are the same move |
| [`search/`](src/mantle/search/) | retrieval. `embeddings.py` and `embeddings_cache.py` are the vector arm's provider facade and cache; `mantle/sse/` is the encrypted lexical arm; `mantle/lightcone.py` is authorization as reachability; `anchors/` and `beacon/` are the semantic arm and its result cut; `ingest/` is the indexing queue |
| [`attenuation.py`](src/mantle/attenuation.py) | the one authorization meet — CRUDEASIO masks, deny absorbing, composed along every path |
| [`events/`](src/mantle/events/) | the change feed: in-process fan-out with a durable log, plus the optional back-plane |
| [`system/`](src/mantle/system/) | boot and operations — logging, `runner_hooks.py`, and the `manage_*.py` bootstrap, seed, addon and anchor operations |
| [`mesh/`](src/mantle/mesh/) | the peering plane: content-addressed Ed25519-signed shards, anchor-keyed regions, incremental Merkle sync |
| [`oci/`](src/mantle/oci/) | an OCI registry over the lattice — an image is a collection, a blob is content |
| [`shard/`](src/mantle/shard/) | persistence beneath the store: local cache and regions, content tiering, curation, erasure |
| [`clients/`](src/mantle/clients/) | the wire outward. `origin_client.py` is the one outbound peer client; `artifact_helpers.py` is the consumer's side, mapping `content_type` ⇄ `mimeType` |
| [`ui/`](src/mantle/ui/) | server-rendered browse pages |
| [`scripts/`](src/mantle/scripts/) | the console-script implementations plus `manage_erasure.py`, CAS rekey and usage snapshots |
| [`tools/`](src/mantle/tools/) | one-shot migrations |
| [`tests/e2e/`](tests/e2e/) | the blackbox HTTP suite, driving a live stack over the wire — see [tests/e2e/README.md](tests/e2e/README.md) |
| [`.env.example`](.env.example) | config template. Only `MANTLE_LATTICE_PATH` and `KEYS_DIR` are set outright, so an untouched copy runs on defaults |

## Backing a node up

This is a runbook. Nothing here runs on a schedule, verifies a copy, or notices if you never make
one — the store is a file and a directory, so the procedure is short and getting it slightly wrong is
silent.

A node is **four** things, and a backup missing any one of them does not restore:

| part | where | notes |
|---|---|---|
| the lattice | `MANTLE_LATTICE_PATH` | one SQLite file in WAL mode — so really three (`.db`, `-wal`, `-shm`) |
| key material | `KEYS_DIR` | **without this the rest is unreadable ciphertext** |
| content | the local CAS under `AGIENCE_BASE_DIR/.data`, and/or the content bucket | whichever tiers this node uses — see `CONTENT_*` in `.env.example` |
| the indexes | `MANTLE_SSE_DIR`, `MANTLE_CELL_DIR` | derived from the lattice, and a full rebuild is measured in days-to-weeks under write contention, so treat them as data |

**Copy the lattice with `VACUUM INTO`.** A plain file copy of a WAL-mode database while the service
is running captures the `.db` without the committed pages still in the `-wal`, and the result opens
without complaint and is missing recent writes. `VACUUM INTO` runs inside a read transaction and
writes one consistent, already-compacted file, with no downtime and without blocking writers:

```bash
sqlite3 "$MANTLE_LATTICE_PATH" "VACUUM INTO '/backups/mantle-lattice.db'"
```

Copy `KEYS_DIR` and the content tiers with an ordinary file copy, and take the key material
**separately and under different custody**: a copy of the lattice without `encryption.key` decrypts
to nothing, and a copy of both in one place is a single object that surrenders the whole store.

**Restoring** is placing those parts back where the environment points and starting the service.
Restore into a node whose `KEYS_DIR` holds the *same* keyset the backup was taken under; a different
one leaves every secret and platform setting permanently unreadable, which `.env.example` warns about
at length. A backup you have not restored is a hypothesis: restore into a scratch node and read an
artifact back. Scheduling, retention, verification and off-site replication belong in your platform.

## Contributing

Bug reports, tests and hardening contributions are welcome. Read
[CONTRIBUTING.md](CONTRIBUTING.md) first — Mantle has a security-invariant test discipline that
contributions must follow.

## Star history

<a href="https://www.star-history.com/?repos=Agience%2Fagience-mantle&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Agience/agience-mantle&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Agience/agience-mantle&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Agience/agience-mantle&type=date&legend=top-left" />
 </picture>
</a>

Security issues: email **connect@agience.ai** rather than opening a public issue.

Licensed under Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Declaration of generative AI use

The author used Anthropic's Claude Opus (versions 4.8 and 5) in the preparation of this work. Its
contribution was to write code, and to generate and validate content. The ideas, the construction
and the claims are the author's. No other generative AI tool was used. The author reviewed and
edited all output and takes full responsibility for the content of this publication.
